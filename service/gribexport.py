"""
GRIB2 export for OpenCPN (and any other GRIB-reading chartplotter/navigation
software).

Writes one GRIB2 message per (timestep, variable) into a single file, using
eccodes to resolve the correct WMO parameter/grid/time-range encoding from a
shortName rather than hand-building parameter tables. The output is a plain
file readable via OpenCPN's built-in grib_pi "Open File" dialog - no plugin
changes needed.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# forecast_data["variables"] key -> eccodes shortName. Chosen so eccodes
# resolves the correct discipline/category/parameterNumber *and* the level
# type (heightAboveGround=10, surface, meanSea) automatically - see
# grib_pi/GRIB2 shortName tests done during development, e.g. "swh" resolves
# to discipline 10 (oceanographic) category 0 number 3, which is what
# grib_pi expects for a wave-height layer. "i10fg" (not "10fg") is used for
# gust because our gust field is instantaneous per lead time, not a
# statistically-processed max-since-previous-step quantity - "10fg" would
# require a different (time-range) product template we don't have the
# underlying statistics for.
VARIABLE_SHORTNAMES = {
    "wind_u": "10u",
    "wind_v": "10v",
    "wind_gust": "i10fg",
    "waves": "swh",
    "mslp_hpa": "prmsl",
}

# mslp_hpa is stored in hPa (see forecast_engine._run_inference); prmsl's
# native GRIB2 unit is Pa.
VARIABLE_SCALE = {
    "mslp_hpa": 100.0,
}

# Precipitation is deliberately not exported: FCN (this project's only fully
# wired prognostic model) doesn't output tp, so forecast_data's
# "precipitation" array is always zero (see README "Known gaps") - shipping
# an always-zero precip layer would misrepresent it as a real forecast.
EXCLUDED_VARIABLES = {"precipitation"}


def export_grib2(forecast_result: Dict, output_path: Path) -> Optional[Path]:
    """Write forecast_result (as returned by ForecastEngine.generate_forecast)
    to a GRIB2 file at output_path. Returns output_path, or None if there was
    no grid to export (e.g. an empty/failed forecast) - never raises, since a
    GRIB export failure shouldn't take down the rest of forecast generation.
    """
    try:
        import eccodes
    except ImportError:
        logger.warning("eccodes not installed - skipping GRIB2 export")
        return None

    forecast_data = forecast_result.get("data") or {}
    lat_array = forecast_data.get("lat_array")
    lon_array = forecast_data.get("lon_array")
    timesteps = forecast_data.get("timesteps")
    variables = forecast_data.get("variables") or {}

    if lat_array is None or lon_array is None or not timesteps:
        logger.warning("No grid/timesteps in forecast_result - skipping GRIB2 export")
        return None

    lat_array = np.asarray(lat_array, dtype=float)
    lon_array = np.asarray(lon_array, dtype=float)

    # GRIB2 stores longitude as an unsigned 0-360 value. Our cropped grid
    # can sit in a negative branch (e.g. -2.25..3.75 for a window near the
    # Solent, see forecast_engine._crop_to_region) - shift the whole,
    # already-contiguous window by +360 rather than wrapping each value
    # independently, which would break monotonicity.
    if lon_array.min() < 0:
        lon_array = lon_array + 360.0

    lat_ascending = bool(lat_array[-1] > lat_array[0]) if len(lat_array) > 1 else True
    lat_step = abs(float(lat_array[1] - lat_array[0])) if len(lat_array) > 1 else 0.25
    lon_step = abs(float(lon_array[1] - lon_array[0])) if len(lon_array) > 1 else 0.25

    initial_time = timesteps[0]
    if not isinstance(initial_time, datetime):
        raise TypeError(f"Expected timesteps[0] to be a datetime, got {type(initial_time)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_messages = 0
    with open(output_path, "wb") as f:
        for var_name, short_name in VARIABLE_SHORTNAMES.items():
            if var_name in EXCLUDED_VARIABLES:
                continue
            arr = variables.get(var_name)
            if arr is None:
                continue
            scale = VARIABLE_SCALE.get(var_name, 1.0)

            for t_idx, ts in enumerate(timesteps):
                if t_idx >= arr.shape[0]:
                    break
                step_hours = round((ts - initial_time).total_seconds() / 3600.0)
                values = np.asarray(arr[t_idx], dtype=float) * scale
                if values.shape != (len(lat_array), len(lon_array)):
                    logger.warning(
                        f"Skipping {var_name}@t{t_idx}: grid shape {values.shape} != "
                        f"({len(lat_array)}, {len(lon_array)})"
                    )
                    continue

                gid = eccodes.codes_grib_new_from_samples("GRIB2")
                try:
                    eccodes.codes_set(gid, "shortName", short_name)

                    eccodes.codes_set(gid, "gridType", "regular_ll")
                    eccodes.codes_set(gid, "Ni", len(lon_array))
                    eccodes.codes_set(gid, "Nj", len(lat_array))
                    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", float(lat_array[0]))
                    eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", float(lon_array[0]))
                    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", float(lat_array[-1]))
                    eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", float(lon_array[-1]))
                    eccodes.codes_set(gid, "iDirectionIncrementInDegrees", lon_step)
                    eccodes.codes_set(gid, "jDirectionIncrementInDegrees", lat_step)
                    eccodes.codes_set(gid, "iScansNegatively", 0)
                    eccodes.codes_set(gid, "jScansPositively", 1 if lat_ascending else 0)

                    eccodes.codes_set(gid, "dataDate", int(initial_time.strftime("%Y%m%d")))
                    eccodes.codes_set(gid, "dataTime", int(initial_time.strftime("%H%M")))
                    eccodes.codes_set(gid, "indicatorOfUnitOfTimeRange", 1)  # hours
                    eccodes.codes_set(gid, "forecastTime", int(step_hours))

                    flat = values.flatten()
                    nan_mask = np.isnan(flat)
                    if nan_mask.any():
                        # GRIB2's simple packing can't encode NaN directly
                        # (eccodes raises EncodingError) - real case, not
                        # hypothetical: swh (wave height) comes back
                        # entirely NaN over narrow/enclosed water the
                        # global wave model doesn't resolve, e.g. the
                        # Solent, while the surrounding open-water cells in
                        # the same message are valid. Use GRIB2's own
                        # missing-value mechanism (bitmapPresent + a
                        # sentinel) rather than silently substituting a
                        # fake number - any spec-compliant reader,
                        # including OpenCPN's grib_pi, treats bitmapped
                        # cells as "no data".
                        eccodes.codes_set(gid, "bitmapPresent", 1)
                        missing = eccodes.codes_get(gid, "missingValue")
                        flat = np.where(nan_mask, missing, flat)

                    eccodes.codes_set_values(gid, flat)
                    eccodes.codes_write(gid, f)
                    n_messages += 1
                finally:
                    eccodes.codes_release(gid)

    if n_messages == 0:
        logger.warning("GRIB2 export produced no messages - removing empty file")
        output_path.unlink(missing_ok=True)
        return None

    logger.info(f"Wrote {n_messages} GRIB2 messages to {output_path}")
    return output_path
