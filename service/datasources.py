"""
Data sources for the forecast engine.

Wraps the base atmospheric data source (GFS, or CDS/ERA5 once credentials are
configured) and, where a Met Office sea-surface-temperature feed is
available, overlays real observed SST onto the fields passed to CBottleInfill
/ CBottleSR. Those models can infer a plausible SST when it's missing from
their inputs, but a measured value is far better for marine forecasting:
SST controls marine fog risk, boundary-layer stability over water, and -
critically for North Atlantic/North Sea storms - how much energy a low
pressure system can draw from the sea surface as it deepens (explosive
cyclogenesis is very sensitive to the air-sea temperature contrast).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr

import config

logger = logging.getLogger(__name__)

# GHRSST L4 files (which OSTIA is) use these variable names by convention;
# support a couple of fallbacks in case the Marine Data Service's delivery
# differs slightly from the public CMEMS OSTIA product.
_SST_VARIABLE_CANDIDATES = ("analysed_sst", "sst", "sea_surface_temperature")


def get_base_data_source(source_type: Optional[str] = None):
    """Return the configured base atmospheric data source (GFS or CDS)."""
    source_type = (source_type or config.DATA_SOURCE).lower()

    if source_type == "cds":
        from earth2studio.data import CDS
        logger.info("Using CDS data source (ECMWF ERA5 reanalysis via Copernicus, requires ~/.cdsapirc)")
        return CDS()

    if source_type == "ifs":
        from earth2studio.data import IFS
        logger.info("Using IFS data source (ECMWF open-data operational forecast)")
        return IFS()

    if source_type == "gfs":
        from earth2studio.data import GFS
        logger.info("Using GFS data source (NOAA, no credentials required)")
        return GFS()

    raise ValueError(f"Unknown data source type: {source_type}")


def get_wave_forecast_source(ensemble_member: Optional[int] = None):
    """ECMWF Open Data forecast source for real wave fields (swh, mwd, mwp)
    and the real 10m wind gust field (fg10m) - free, global, no credentials,
    already-computed by ECMWF (not run locally). Only IFS exposes these in
    earth2studio's lexicon; AIFS's Open Data wrapper here doesn't carry wave
    or gust variables, only the core atmospheric/surface fields.

    ensemble_member: None for the deterministic IFS_FX; 0 for the ENS
    control member, >0 for a perturbed ENS member (IFS_ENS_FX).
    """
    if ensemble_member is None:
        from earth2studio.data import IFS_FX
        return IFS_FX()

    from earth2studio.data import IFS_ENS_FX
    return IFS_ENS_FX(member=ensemble_member)


# ECMWF Open Data's IFS_FX product stops publishing the plain "10fg"
# (instantaneous max 10m wind gust) field for lead times roughly 93-147h -
# only the 3-hour-windowed "10fg3" field exists in that range (reverts to
# plain "10fg" again beyond it). Confirmed by inspecting ECMWF's own .index
# files directly (data.ecmwf.int/forecasts/<date>/00z/ifs/0p25/oper/*.index
# for steps 90/96/120/144/168 - 10fg at 90 and 168, 10fg3 at 96/120/144),
# not documented anywhere we could find. earth2studio's IFSLexicon hardcodes
# a single "10fg" mapping with no lead-time awareness, so a fetch spanning
# this gap fails outright for those steps. The bounds below are a
# conservative margin around the observed gap, not exact ECMWF boundaries
# (which aren't published and could drift) - a step just outside them that
# turns out to still be in the gap just falls back to the normal per-field
# NaN handling already in _fetch_real_waves_and_gusts, same as any other
# fetch failure.
GUST_10FG3_GAP_START_H = 93
GUST_10FG3_GAP_END_H = 147


def get_gust_forecast_source_10fg3():
    """Like get_wave_forecast_source(), but requests "fg10m" as ECMWF's
    "10fg3" param instead of "10fg" - for lead times inside the gap above.
    A local subclass, not a mutation of earth2studio's shared IFSLexicon/
    IFS_FX classes (which are also used elsewhere in this process), so this
    has no effect outside calls that explicitly use it."""
    from earth2studio.data import IFS_FX
    from earth2studio.lexicon.ecmwf import IFSLexicon

    class _IFSLexicon10FG3(IFSLexicon):
        VOCAB = {**IFSLexicon.VOCAB, "fg10m": "10fg3::sfc::"}

    class _IFS_FX_10FG3(IFS_FX):
        LEXICON = _IFSLexicon10FG3

    return _IFS_FX_10FG3()


def get_ensemble_forecast_source(model: str = "aifs", member: int = 0):
    """ECMWF Open Data ensemble forecast source (member=0 is the control
    forecast, member>0 a perturbed member) - AIFS_ENS_FX or IFS_ENS_FX. Both
    fetch ECMWF's own pre-computed ensemble output directly; no local model
    inference is involved.

    Note: IFS_ENS_FX's control member (member=0) is unreliable on Open Data
    as of writing - ECMWF appears to have stopped publishing its index
    consistently (earth2studio logs its own warning about this). forecast_
    engine.generate_ensemble_risk only requests members 1.. for this reason.
    """
    model = model.lower()
    if model == "aifs":
        from earth2studio.data import AIFS_ENS_FX
        return AIFS_ENS_FX(member=member)
    if model == "ifs":
        from earth2studio.data import IFS_ENS_FX
        return IFS_ENS_FX(member=member)
    raise ValueError(f"Unknown ensemble model: {model!r} (expected 'aifs' or 'ifs')")


class MetOfficeSST:
    """
    Reads observed sea-surface temperature (OSTIA, via the Met Office Marine
    Data Service) from a local cache directory.

    The Marine Data Service isn't a REST API - it delivers GHRSST L4 NetCDF
    files over FTP/SFTP on a fixed daily schedule. metoffice_sync.py mirrors
    new files into config.METOFFICE_SST_LOCAL_DIR; this class just picks the
    file whose date best matches the requested time and reads it. Run
    metoffice_sync.py (e.g. from a systemd timer) to keep that cache current
    - this class does not fetch over the network itself.
    """

    VARIABLE = "sst"

    def __init__(self, local_dir: Optional[str] = None):
        self.local_dir = Path(local_dir or config.METOFFICE_SST_LOCAL_DIR)
        if not config.METOFFICE_SST_HOST:
            raise RuntimeError(
                "Met Office SST sync isn't configured - set METOFFICE_SST_HOST/"
                "_USERNAME/_PASSWORD (or _KEY_PATH) once the Marine Data "
                "Service account is set up, and run metoffice_sync.py. "
                "See README.md."
            )

    def _find_file_for(self, time: datetime) -> Path:
        """Pick the cached file whose date best matches `time`. OSTIA is a
        once-daily analysis, so this looks for the closest date, not an
        exact timestamp match - falls back to the most recent file available
        if nothing matches within a few days (better than failing the whole
        forecast over a short sync gap)."""
        candidates = sorted(self.local_dir.glob(config.METOFFICE_SST_FILE_PATTERN))
        if not candidates:
            raise FileNotFoundError(
                f"No Met Office SST files found in {self.local_dir} matching "
                f"{config.METOFFICE_SST_FILE_PATTERN!r} - run metoffice_sync.py first."
            )

        date_str = time.strftime("%Y%m%d")
        exact = [p for p in candidates if date_str in p.name]
        if exact:
            return exact[-1]

        logger.warning(
            f"No Met Office SST file dated {date_str} in {self.local_dir}; "
            f"using the most recent available file instead ({candidates[-1].name})"
        )
        return candidates[-1]

    def _fetch_raw(self, time: datetime) -> xr.DataArray:
        path = self._find_file_for(time)
        ds = xr.open_dataset(path)

        var_name = next((v for v in _SST_VARIABLE_CANDIDATES if v in ds.variables), None)
        if var_name is None:
            raise KeyError(
                f"None of {_SST_VARIABLE_CANDIDATES} found in {path.name} "
                f"(variables present: {list(ds.data_vars)})"
            )

        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=0)

        # OSTIA's native grid is -180..180 longitude; earth2studio's model
        # grids use 0-360. Normalize here so regrid_to's target-grid lookup
        # (which may ask for lon values like 358.7) doesn't just return NaN
        # for anything east of the prime meridian.
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
        return da.load()

    def __call__(self, time, variable) -> xr.DataArray:
        times = time if isinstance(time, list) else [time]
        frames = []
        for t in times:
            raw = self._fetch_raw(t)
            # OSTIA ships analysed_sst in Kelvin already; guard against a
            # Celsius file (some GHRSST mirrors differ) rather than assume.
            if float(raw.max()) < 100:
                raw = raw + 273.15
            frames.append(raw)
        da = xr.concat(frames, dim="time")
        da = da.expand_dims(variable=["sst"])
        da = da.assign_coords(time=("time", times))
        return da


def regrid_to(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Bilinear-regrid an (lat, lon) DataArray onto the target grid used by
    the rest of the pipeline (e.g. CBottleInfill's 721x1440 0.25 deg grid)."""
    return da.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": "extrapolate"})


def overlay_metoffice_sst(
    data: "xr.DataArray | tuple",
    coords: Optional[dict] = None,
    time: Optional[Union[datetime, List[datetime]]] = None,
) -> tuple:
    """
    Given a (tensor, coords) pair already fetched for the model's required
    variables (as produced by earth2studio.data.utils.fetch_data), replace
    the "sst" slice with an observed value read from the local Met Office
    SST cache (see MetOfficeSST / metoffice_sync.py) when configured. If
    METOFFICE_SST_HOST isn't set, or "sst" isn't one of the requested
    variables, this is a no-op - the model's own inferred SST is used,
    matching current behaviour.
    """
    import torch

    if coords is None or "variable" not in coords:
        return data, coords

    variables = list(coords["variable"])
    if "sst" not in variables or not config.METOFFICE_SST_HOST:
        return data, coords

    try:
        sst_source = MetOfficeSST()
        sst_da = sst_source(time, ["sst"])
        sst_regridded = regrid_to(sst_da, coords["lat"], coords["lon"])
        sst_idx = variables.index("sst")
        data[..., sst_idx, :, :] = torch.as_tensor(
            sst_regridded.values, device=data.device, dtype=data.dtype
        )
        logger.info("Overlaid observed Met Office SST onto model input")
    except Exception as e:
        logger.warning(f"Met Office SST overlay failed, using model-inferred SST instead: {e}")

    return data, coords
