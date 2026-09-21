"""
Earth-2 Forecast Engine

Runs a prognostic model over the requested horizon, optionally hands the
result to the CBottle harbor-scale super-resolution stage, derives maritime
variables (wind, waves, wind gusts where possible), and packages everything
for visualize.py and gale_warnings.py.
"""

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import torch

import config
import datasources
import gale_warnings
import regions
import superres

logger = logging.getLogger(__name__)

# Maps a config.py "model" name to how to load it. Every entry here must be a
# prognostic (px) model exposing the standard earth2studio (tensor, coords)
# __call__ interface.
MODEL_LOADERS = {
    "fcn": lambda: _load_px("earth2studio.models.px", "FCN"),
    "graphcast": lambda: _load_px("earth2studio.models.px", "GraphCastOperational"),
    "graphcast_small": lambda: _load_px("earth2studio.models.px", "GraphCastSmall"),
    # European-domain satellite nowcasting (Meteosat-driven) - the only
    # StormScope variant with UK/North Sea/Baltic coverage. StormScopeGOES
    # and StormScopeMRMS are both US-only (GOES satellite / NOAA MRMS radar).
    "stormscope": lambda: _load_px("earth2studio.models.px", "StormScopeMeteosatEU"),
    "dlwp": lambda: _load_px("earth2studio.models.px", "DLWP"),
}

# Native output resolution per model, used to correct config.py's
# resolution_km in the response metadata after a model fallback (e.g.
# nowcast's stormscope -> fcn: the config still says "4" for StormScope's
# 4km output, which would misreport the resolution of what was actually
# returned once FCN's 25km output is substituted in).
MODEL_RESOLUTION_KM = {
    "fcn": 25,
    "graphcast": 25,
    "graphcast_small": 25,
    "stormscope": 4,
    "dlwp": 100,
}

# Variables WindgustAFNO needs. Only models with a full 13-level state
# (GraphCast) provide all of these; FCN does not (no 925hPa, no humidity),
# so gust warnings fall back to a wind-speed heuristic for FCN-driven runs.
WINDGUST_REQUIRED_VARS = {
    "u100m", "v100m", "msl", "u300", "u850", "u925", "v300", "v850", "v925",
    "z300", "z850", "z925", "t850", "t925", "q500", "q850", "q925",
}


def _load_px(module_path: str, class_name: str):
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls.load_model(cls.load_default_package())
    return model


class ForecastEngine:
    """Manages Earth-2 model inference with on-demand loading."""

    def __init__(self):
        self.current_model = None
        self.current_model_type = None

    def generate_forecast(
        self,
        forecast_id: str,
        location: Dict,
        forecast_type: str,
        radius_miles: int = 100,
        variables: Optional[List[str]] = None,
        super_resolution: Optional[bool] = None,
        real_wave_overlay: Optional[bool] = None,
        ensemble: Optional[bool] = None,
        ensemble_source: Optional[str] = None,
    ) -> Dict:
        """Generate a maritime forecast. Returns {forecast_id, data, metadata}."""
        location_name = location.get("name", "Unknown Location")
        logger.info(f"Starting forecast {forecast_id} for {location_name}")

        try:
            cfg = config.FORECAST_CONFIGS[forecast_type]
            self._load_model(forecast_type)

            initial_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            hour = (initial_time.hour // 6) * 6
            initial_time = initial_time.replace(hour=hour)

            num_steps = self._num_steps(cfg)
            logger.info(f"Running {cfg['model']} for {num_steps} steps from {initial_time}")

            try:
                forecast_data = self._run_inference(initial_time, cfg, num_steps)
            except Exception as e:
                # _load_model's fallback only guards model *loading* - some
                # models load fine but are structurally incompatible with
                # the generic GFS -> run.deterministic pipeline (e.g.
                # StormScopeMeteosatEU expects consecutive Meteosat satellite
                # frames at 10-min resolution, not GFS atmospheric fields on
                # our config's hourly cadence, and fails as soon as
                # run.deterministic tries to fetch GFS at a non-6-hourly
                # time). Fall back the same way here, at the inference
                # stage, rather than letting the whole forecast fail.
                fallback = cfg.get("fallback_model")
                if not fallback or cfg["model"] == fallback:
                    raise
                logger.warning(f"{cfg['model']} inference failed ({e}); falling back to {fallback}")
                self._release_memory()
                cfg["model"] = fallback
                cfg["resolution_km"] = MODEL_RESOLUTION_KM.get(fallback, cfg["resolution_km"])
                self.current_model_type = None
                self._load_model(forecast_type)
                num_steps = self._num_steps(cfg)
                forecast_data = self._run_inference(initial_time, cfg, num_steps)

            # Crop from the global grid down to a window around the requested
            # location before anything else. Without this, warnings and
            # storm tracking below would scan the *entire planet's* worst
            # weather each timestep - a Solent forecast could report a
            # Force 12 warning caused by a storm off Japan. The window is
            # generous (radius x3, min ~400 miles) so an approaching storm
            # currently just outside the requested radius, but forecast to
            # arrive within the validity window, isn't cropped out.
            forecast_data = self._crop_to_region(forecast_data, location, radius_miles)

            # Recompute waves fetch-limited now that the grid is small
            # enough for a coastline-distance lookup to be cheap (see
            # _compute_fetch_limited_waves - _run_inference's own estimate
            # assumes open-ocean fetch everywhere, which is wrong for
            # enclosed/narrow waters like the Solent). Real wave data below
            # overrides this again where it's available.
            forecast_data["variables"]["waves"] = self._compute_fetch_limited_waves(
                forecast_data["variables"]["wind_u"], forecast_data["variables"]["wind_v"],
                forecast_data["lat_array"], forecast_data["lon_array"],
            )

            use_sr = super_resolution if super_resolution is not None else config.SUPER_RESOLUTION["enabled_by_default"]
            if use_sr:
                forecast_data = self._apply_super_resolution(forecast_data, location, initial_time)

            self._release_memory()

            use_real_waves = (
                real_wave_overlay if real_wave_overlay is not None
                else config.REAL_WAVE_OVERLAY["enabled_by_default"]
            )
            wave_source = "fetch-limited estimate (SMB/CEM, coastline-distance fetch)"
            if use_real_waves:
                overlaid = self._fetch_real_waves_and_gusts(
                    initial_time, forecast_data["timesteps"], cfg,
                    forecast_data["lat_array"], forecast_data["lon_array"],
                )
                if overlaid is not None:
                    if "waves" in overlaid:
                        forecast_data["variables"]["waves"] = overlaid["waves"]
                        wave_source = "IFS Open Data (ECMWF, real wave model)"
                    if "wind_gust" in overlaid:
                        forecast_data["variables"]["wind_gust"] = overlaid["wind_gust"]

            warnings_bundle = None
            daily_breakdown = []
            if config.WARNINGS["enabled"]:
                local_radius = self._local_warning_radius_miles(location, radius_miles)
                local_data = self._crop_to_region(forecast_data, location, local_radius, pad=False)
                warnings_bundle = gale_warnings.build_warnings_bundle(
                    forecast_data,
                    area_name=location_name,
                    min_force=config.WARNINGS["min_beaufort_force"],
                    wave_threshold_m=config.WARNINGS["wave_warning_m"],
                    storm_tracking=config.WARNINGS["storm_tracking"],
                    local_forecast_data=local_data,
                )
                # Same tight window as the warnings above - daily_breakdown
                # is a per-day rollup of the same local conditions, not the
                # wide chart/storm-tracking grid.
                daily_breakdown = gale_warnings.daily_breakdown(local_data)

            use_ensemble = ensemble if ensemble is not None else config.ENSEMBLE["enabled_by_default"]
            if use_ensemble:
                try:
                    probabilistic = self.generate_ensemble_risk(
                        location, radius_miles, cfg,
                        forecast_data["timesteps"], initial_time,
                        source=ensemble_source or config.ENSEMBLE["source"],
                    )
                    if warnings_bundle is None:
                        warnings_bundle = {"wind": [], "sea_state": [], "storms": [], "probabilistic": []}
                    warnings_bundle["probabilistic"] = probabilistic
                except Exception as e:
                    logger.warning(f"Ensemble risk generation failed, continuing without it: {e}")

            logger.info(f"Forecast {forecast_id} complete")

            return {
                "forecast_id": forecast_id,
                "data": forecast_data,
                "warnings": warnings_bundle,
                "daily_breakdown": daily_breakdown,
                "metadata": {
                    "location": location,
                    "forecast_type": forecast_type,
                    "model": cfg["model"],
                    "initial_time": initial_time.isoformat(),
                    "resolution_km": cfg["resolution_km"],
                    "super_resolution": use_sr,
                    "wave_source": wave_source,
                    "ensemble": use_ensemble,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Forecast generation failed: {e}", exc_info=True)
            self._release_memory()
            raise

    def _num_steps(self, cfg: Dict) -> int:
        if "duration_days" in cfg:
            return cfg["duration_days"] * (24 // cfg["timestep_hours"])
        return cfg["duration_hours"] // cfg["timestep_hours"]

    def _crop_to_region(self, forecast_data: Dict, location: Dict, radius_miles: int, pad: bool = True) -> Dict:
        """Crop the global forecast grid to a window around the requested
        location.

        pad=True (default - used for the main forecast/chart pipeline):
        inflates radius_miles (x3, floor 400mi) so the window stays generous
        enough to still catch an approaching storm.
        pad=False: use radius_miles exactly as given, no inflation - for
        building a tight window for gale/sea-state warning scanning (see
        generate_forecast's use of _local_warning_radius_miles), where the
        wide padded window is exactly the wrong thing: a "gale warning for
        Solent" should mean wind within the Solent, not anywhere in an
        800-mile-wide box around it.

        Model grids use 0-360 longitude; location lon is -180..180. A naive
        boolean mask on longitude breaks for any location near the 0/360
        seam (e.g. the Solent, at lon ~358.7 in 0-360 terms): the matched
        indices land in two separate clusters (near 0 and near 360) that
        aren't contiguous, so slicing them produces a non-monotonic
        longitude axis with a spurious jump when contoured. Instead, crop as
        a contiguous circular window: find the nearest grid index to the
        target, take a symmetric run of indices around it modulo the grid
        width, and rebuild the longitude *values* analytically from the
        uniform grid step (rather than by reading wrapped-around array
        values) so they're continuous even where the window crosses 0/360.
        """
        lat_array = forecast_data.get("lat_array")
        lon_array = forecast_data.get("lon_array")
        if lat_array is None or lon_array is None:
            return forecast_data

        lat_array = np.asarray(lat_array)
        lon_array = np.asarray(lon_array)

        radius_deg = (max(radius_miles * 3, 400) if pad else radius_miles) / 69.0
        lat, lon = location["lat"], location["lon"]

        lat_mask = np.abs(lat_array - lat) <= radius_deg
        if not lat_mask.any():
            logger.warning(f"Region crop around ({lat}, {lon}) matched no grid points, keeping global grid")
            return forecast_data
        lat_idx = np.where(lat_mask)[0]

        n_lon = lon_array.shape[0]
        lon_step = 360.0 / n_lon
        lon_norm = lon % 360
        center_idx = int(round(lon_norm / lon_step)) % n_lon
        n_points = max(1, int(round(radius_deg / lon_step)))
        offsets = np.arange(-n_points, n_points + 1)
        lon_idx = (center_idx + offsets) % n_lon
        lon_values = lon_array[center_idx] + offsets * lon_step
        # Re-express in the branch nearest the original (possibly negative,
        # e.g. -180..180) target longitude, not wherever the 0-360 grid
        # happened to put it - visualize.py's map extent is built from the
        # raw location lon, so the two need to agree on which "copy" of the
        # periodic coordinate they're using. A single shift by a multiple of
        # 360 keeps the window's internal continuity intact.
        lon_values = lon_values - 360.0 * round((lon_array[center_idx] - lon) / 360.0)

        cropped = dict(forecast_data)
        cropped["lat_array"] = lat_array[lat_idx]
        cropped["lon_array"] = lon_values
        cropped["variables"] = {
            name: arr[:, lat_idx][:, :, lon_idx] if arr.ndim == 3 else arr
            for name, arr in forecast_data["variables"].items()
        }
        return cropped

    def _local_warning_radius_miles(self, location: Dict, fallback_radius_miles: int) -> int:
        """Radius (miles) for the tight window gale/sea-state warnings scan
        - deliberately much smaller than the wide, padded crop used for the
        rest of the pipeline (chart extent, storm tracking).

        Uses the resolved location's own Met Office sea-area bounding box
        when it falls inside one (regions.area_for_location) - derived from
        the box's real half-extents, so the window is sized to the actual
        named area rather than a guessed constant. Falls back to
        fallback_radius_miles (the tier's own configured radius, i.e.
        before _crop_to_region's own x3/400mi-floor padding) for a location
        that doesn't resolve to a known area.
        """
        area = regions.area_for_location(location["lat"], location["lon"])
        if area is None:
            return fallback_radius_miles

        lat_s, lon_w, lat_n, lon_e = area.bounds
        lat_half_deg = (lat_n - lat_s) / 2.0
        lon_half_deg = (lon_e - lon_w) / 2.0 * np.cos(np.radians(location["lat"]))
        half_deg = max(lat_half_deg, lon_half_deg)
        return max(10, int(round(half_deg * 69.0)))

    def _compute_fetch_limited_waves(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
    ) -> np.ndarray:
        """Significant wave height via the fetch-limited SMB/CEM growth
        curve, using distance-to-coastline *in the upwind direction* as the
        fetch.

        A plain `0.025 * wind_speed_ms**2` (the fully-developed open-ocean
        formula) assumes unlimited fetch everywhere, which is badly wrong
        for enclosed or narrow waters: it predicts 5m+ waves in the Solent
        at a Force 8 gale, when 1-1.5m is realistic. This uses the standard
        fetch-limited relation instead:

            g*Hs/U^2 = 0.283 * tanh(0.0125 * (g*F/U^2)^0.42)

        (F = fetch in metres, U = wind speed in m/s, g = 9.81 m/s^2).

        F is direction-aware: wind (with direction) is already computed
        before waves are, so it's used here rather than a plain nearest-
        coast-in-any-direction distance, which would give a point just
        offshore of a north-facing coast the same short fetch whether the
        wind is blowing onshore (correct - short fetch) or straight off the
        land out to sea (wrong - that wind has an open-water fetch, the
        coast behind the point is irrelevant to it). F is built by blending
        the two cardinal (N/E/S/W) coastline distances that flank the
        actual upwind bearing, weighted by how much of the wind vector
        points along each - not full 360-degree ray-casting (which would
        need re-marching per grid point per timestep against real
        coastline geometry - too slow to redo every request), but a
        real, cheap step up from ignoring direction entirely.
        """
        wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)  # m/s, shape (time, lat, lon)

        try:
            land_mask, lat_step_km, lon_step_km = self._land_mask(lat_array, lon_array)

            if land_mask is None:
                fetch_km = np.full(wind_speed.shape[1:], 500.0)  # no coastline anywhere nearby - open ocean
            elif land_mask.all():
                fetch_km = np.zeros(wind_speed.shape[1:])
            else:
                cardinal = self._cardinal_fetch_km(land_mask, lat_array, lat_step_km, lon_step_km)

                # Upwind bearing = where the wind is blowing FROM, i.e. the
                # reverse of wind_u/wind_v (which give the direction air is
                # moving TO, standard meteorological vector convention).
                # That's the direction fetch needs to extend in.
                upwind_u, upwind_v = -wind_u, -wind_v
                fetch_ew = np.where(upwind_u >= 0, cardinal["E"], cardinal["W"])
                fetch_ns = np.where(upwind_v >= 0, cardinal["N"], cardinal["S"])
                e_w_weight = np.abs(upwind_u) / (np.abs(upwind_u) + np.abs(upwind_v) + 1e-6)
                fetch_km = e_w_weight * fetch_ew + (1 - e_w_weight) * fetch_ns
        except Exception as e:
            logger.warning(f"Fetch-limited wave calc failed ({e}), falling back to open-ocean formula")
            return 0.025 * wind_speed ** 2

        fetch_m = np.clip(fetch_km, 0.5, 500.0) * 1000.0
        wind_speed_safe = np.maximum(wind_speed, 0.5)

        g = 9.81
        dimless_fetch = g * fetch_m / wind_speed_safe ** 2
        dimless_hs = 0.283 * np.tanh(0.0125 * dimless_fetch ** 0.42)
        return dimless_hs * wind_speed_safe ** 2 / g

    def _land_mask(self, lat_array: np.ndarray, lon_array: np.ndarray):
        """Boolean land/sea raster on our own (small, already-cropped) grid,
        via point-in-polygon tests against a bbox-filtered, prepared
        Natural Earth 10m coastline geometry (fast: ~0.1s for a 100x100
        grid). Returns (land_mask, lat_step_km, lon_step_km); land_mask is
        None if there's no coastline anywhere near the window (open ocean).
        """
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point, box
        from shapely.ops import unary_union
        from shapely.prepared import prep

        lat_array = np.asarray(lat_array)
        lon_array = np.asarray(lon_array)
        lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)

        # Natural Earth land polygons use -180..180 longitude; our cropped
        # grid may sit in a different continuous branch (e.g. ~355..365 for
        # a window straddling 0 degrees the other way) - normalize before
        # testing against them.
        lon_grid_180 = ((lon_grid + 180) % 360) - 180

        shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
        geoms = list(shpreader.Reader(shp).geometries())

        pad = 1.0
        bbox = box(
            float(lon_grid_180.min()) - pad, float(lat_grid.min()) - pad,
            float(lon_grid_180.max()) + pad, float(lat_grid.max()) + pad,
        )
        nearby = [g for g in geoms if prep(bbox).intersects(g)]

        lat_step_km = abs(lat_array[1] - lat_array[0]) * 111.0 if len(lat_array) > 1 else 25.0
        mean_lat = float(np.mean(lat_array))
        lon_step_km = (
            abs(lon_array[1] - lon_array[0]) * 111.0 * np.cos(np.radians(mean_lat))
            if len(lon_array) > 1 else 25.0
        )

        if not nearby:
            return None, lat_step_km, lon_step_km

        prepared = prep(unary_union(nearby))
        land_mask = np.array([
            [prepared.contains(Point(lon_grid_180[i, j], lat_grid[i, j])) for j in range(lat_grid.shape[1])]
            for i in range(lat_grid.shape[0])
        ])
        return land_mask, lat_step_km, lon_step_km

    def _cardinal_fetch_km(
        self,
        land_mask: np.ndarray,
        lat_array: np.ndarray,
        lat_step_km: float,
        lon_step_km: float,
    ) -> Dict[str, np.ndarray]:
        """Distance to nearest land (km) looking in each of the 4 cardinal
        directions (N/E/S/W), via a fully vectorized running-minimum scan
        over the land/sea raster (`np.minimum.accumulate` - no per-cell
        Python loop, no new polygon tests beyond the mask already built).

        Longitude is always stored ascending in this codebase (both
        regions.build_regional_grid and forecast_engine._crop_to_region
        construct it that way), so increasing column index is always East.
        Latitude is NOT always ascending (model grids are typically
        north-to-south descending; regions.build_regional_grid's is
        ascending) - handled by checking lat_array's own direction rather
        than assuming index order.
        """
        n_lat, n_lon = land_mask.shape
        BIG = n_lat + n_lon + 10_000

        lat_ascending = bool(lat_array[-1] > lat_array[0]) if n_lat > 1 else True
        mask_asc = land_mask if lat_ascending else land_mask[::-1, :]

        row_idx = np.arange(n_lat)
        col_idx = np.arange(n_lon)

        # North: nearest land at row-index >= this row, working in
        # ascending-latitude space (flipped back below if needed).
        nearest_n = np.minimum.accumulate(
            np.where(mask_asc, row_idx[:, None], BIG)[::-1, :], axis=0
        )[::-1, :]
        dist_n_cells = nearest_n - row_idx[:, None]

        # South: nearest land at row-index <= this row.
        nearest_s = np.maximum.accumulate(np.where(mask_asc, row_idx[:, None], -BIG), axis=0)
        dist_s_cells = row_idx[:, None] - nearest_s

        if not lat_ascending:
            dist_n_cells, dist_s_cells = dist_s_cells[::-1, :], dist_n_cells[::-1, :]

        # East: nearest land at column-index >= this column.
        nearest_e = np.minimum.accumulate(
            np.where(land_mask, col_idx[None, :], BIG)[:, ::-1], axis=1
        )[:, ::-1]
        dist_e_cells = nearest_e - col_idx[None, :]

        # West: nearest land at column-index <= this column.
        nearest_w = np.maximum.accumulate(np.where(land_mask, col_idx[None, :], -BIG), axis=1)
        dist_w_cells = col_idx[None, :] - nearest_w

        return {
            "N": np.clip(dist_n_cells * lat_step_km, 0.5, 500.0),
            "S": np.clip(dist_s_cells * lat_step_km, 0.5, 500.0),
            "E": np.clip(dist_e_cells * lon_step_km, 0.5, 500.0),
            "W": np.clip(dist_w_cells * lon_step_km, 0.5, 500.0),
        }

    def _load_model(self, forecast_type: str):
        cfg = config.FORECAST_CONFIGS[forecast_type]
        model_name = cfg["model"]

        if self.current_model_type == forecast_type:
            logger.info(f"Model {model_name} already loaded")
            return

        if self.current_model is not None:
            self._release_memory()

        loader = MODEL_LOADERS.get(model_name)
        if loader is None:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            logger.info(f"Loading {model_name}...")
            model = loader()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.current_model = model.to(device)
            self.current_model_type = forecast_type
            logger.info(f"Loaded {model_name} on {device}")
        except Exception as e:
            fallback = cfg.get("fallback_model")
            if fallback and fallback != model_name:
                logger.warning(f"{model_name} unavailable ({e}); falling back to {fallback}")
                loader = MODEL_LOADERS[fallback]
                model = loader()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.current_model = model.to(device)
                self.current_model_type = forecast_type
                cfg["model"] = fallback
                cfg["resolution_km"] = MODEL_RESOLUTION_KM.get(fallback, cfg["resolution_km"])
            else:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def _run_inference(self, initial_time: datetime, cfg: Dict, num_steps: int) -> Dict:
        import earth2studio.run as run
        from earth2studio.data.utils import fetch_data
        from earth2studio.io import ZarrBackend

        data_source = datasources.get_base_data_source()
        model = self.current_model

        with tempfile.TemporaryDirectory() as tmpdir:
            io = ZarrBackend(file_name=f"{tmpdir}/forecast.zarr")
            io = run.deterministic([initial_time], num_steps, model, data_source, io)

            available_vars = list(io.root.array_keys())
            logger.info(f"Model output variables: {available_vars}")

            def get_var(name):
                if name in io.root:
                    return np.array(io[name][:])
                return None

            wind_u = get_var("u10m")
            wind_v = get_var("v10m")
            precip = get_var("tp")
            msl = get_var("msl")
            lat_array = get_var("lat")
            lon_array = get_var("lon")

            gust = self._maybe_compute_gust(io, initial_time, num_steps, cfg)

        lat_dim = wind_u.shape[-2] if wind_u is not None else 720
        lon_dim = wind_u.shape[-1] if wind_u is not None else 1440

        all_times = [initial_time + timedelta(hours=i * cfg["timestep_hours"]) for i in range(num_steps + 1)]

        def squeeze_leading(arr):
            if arr is not None and arr.ndim == 4:
                return arr[0]
            return arr

        wind_u = squeeze_leading(wind_u)
        wind_v = squeeze_leading(wind_v)
        precip = squeeze_leading(precip)
        msl = squeeze_leading(msl)

        forecast_data = {
            "timesteps": all_times,
            "variables": {},
            "lat": None,
            "lon": None,
            "lat_array": lat_array,
            "lon_array": lon_array,
        }

        forecast_data["variables"]["wind_u"] = wind_u if wind_u is not None else np.zeros((len(all_times), lat_dim, lon_dim))
        forecast_data["variables"]["wind_v"] = wind_v if wind_v is not None else np.zeros((len(all_times), lat_dim, lon_dim))
        forecast_data["variables"]["precipitation"] = np.abs(precip) * 1000 if precip is not None else np.zeros((len(all_times), lat_dim, lon_dim))

        if msl is not None:
            # earth2studio's "msl" is Pa; warnings/plots want hPa.
            forecast_data["variables"]["mslp_hpa"] = msl / 100.0

        wind_speed = np.sqrt(forecast_data["variables"]["wind_u"] ** 2 + forecast_data["variables"]["wind_v"] ** 2)
        forecast_data["variables"]["waves"] = 0.025 * wind_speed ** 2

        # Unlike wind_u/wind_v/waves above, wind_gust had no guaranteed
        # baseline here - if WindgustAFNO returned None (always true for FCN,
        # see WINDGUST_REQUIRED_VARS/_maybe_compute_gust above) and the real
        # IFS overlay in generate_forecast() also misses (best-effort; can
        # fail on rate-limiting/timeout/lead-time gaps - see README "Known
        # gaps"), "wind_gust" was simply absent from forecast_data entirely.
        # gribexport.py silently skips any variable it can't find
        # (`variables.get(var_name) is None: continue`), so the GRIB2 file
        # sent to OpenCPN had wind and wave messages but zero gust messages -
        # this was the actual cause of gusts missing from the chart plotter,
        # separate from the GRIB2 template bug fixed previously. Same
        # gust-factor heuristic gale_warnings.py already uses as its own
        # fallback (see generate_wind_warnings's gust_factor=1.3) - real IFS
        # gust data, when the overlay succeeds, still overrides this later.
        if gust is not None:
            forecast_data["variables"]["wind_gust"] = gust
        else:
            forecast_data["variables"]["wind_gust"] = wind_speed * 1.3

        return forecast_data

    def _maybe_compute_gust(self, io, initial_time, num_steps, cfg) -> Optional[np.ndarray]:
        """Run the WindgustAFNO diagnostic per lead time if the prognostic
        model produced the full variable set it needs; otherwise None (the
        warnings module falls back to a wind-speed-based gust estimate)."""
        available_vars = set(io.root.array_keys())
        if not WINDGUST_REQUIRED_VARS.issubset(available_vars):
            logger.info("Prognostic model output doesn't cover WindgustAFNO's required variables - skipping real gust diagnostic")
            return None

        try:
            from earth2studio.models.dx import WindgustAFNO

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            gust_model = WindgustAFNO.load_model(WindgustAFNO.load_default_package()).to(device)

            # Use the model's own declared variable order - do not assume it
            # matches WINDGUST_REQUIRED_VARS's (arbitrary set) iteration order.
            var_order = [str(v) for v in gust_model.input_coords()["variable"]]
            lat = np.array(io["lat"][:])
            lon = np.array(io["lon"][:])

            gusts = []
            for lead_idx in range(num_steps + 1):
                stacked = np.stack([np.array(io[v][0, lead_idx]) for v in var_order])
                tensor = torch.as_tensor(stacked, device=device, dtype=torch.float32)[None, None, None, ...]
                coords = {
                    "batch": np.array([0]),
                    "time": np.array([initial_time]),
                    "lead_time": np.array([timedelta(hours=lead_idx * cfg["timestep_hours"])]),
                    "variable": np.array(var_order),
                    "lat": lat,
                    "lon": lon,
                }
                out, _ = gust_model(tensor, coords)
                gusts.append(np.asarray(out.detach().cpu()).squeeze())

            del gust_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return np.stack(gusts)
        except Exception as e:
            logger.warning(f"WindgustAFNO diagnostic failed, falling back to heuristic gusts: {e}")
            return None

    def _apply_super_resolution(self, forecast_data: Dict, location: Dict, initial_time: datetime) -> Dict:
        """Run the CBottleInfill -> CBottleSR stage on the final timestep only
        (super-resolution is expensive; it's meant for a "what does landfall
        look like at the harbor" snapshot, not the whole trajectory)."""
        try:
            pipeline = superres.get_pipeline()
            # NOTE: full wiring from forecast_data's numpy arrays back into an
            # earth2studio (tensor, coords) pair for the SR pass is the next
            # increment here; until then this is a documented no-op so a
            # super_resolution=True request degrades gracefully instead of
            # crashing the whole forecast.
            logger.info("Super-resolution requested but not yet wired into the main trajectory path - skipping for this run")
            return forecast_data
        except Exception as e:
            logger.warning(f"Super-resolution stage failed, returning base-resolution forecast: {e}")
            return forecast_data

    def _resolve_available_cycle(
        self,
        source_factory,
        initial_time: datetime,
        max_attempts: int = 4,
    ) -> datetime:
        """ECMWF Open Data cycles (00/06/12/18Z) aren't disseminated
        instantly - the most recent one can be an hour or more from ready.
        Probe with a minimal single-variable, single-lead-time request,
        stepping back one cycle (6h) at a time, and return the first init
        time that actually has data published. Raises the last error if
        nothing in range works.
        """
        candidate = initial_time
        last_exc: Optional[Exception] = None
        for _ in range(max_attempts):
            try:
                probe = source_factory()
                probe(
                    np.array([candidate], dtype="datetime64[ns]"),
                    np.array([np.timedelta64(0, "h")], dtype="timedelta64[ns]"),
                    ["u10m"],
                )
                return candidate
            except Exception as e:
                last_exc = e
                logger.info(f"ECMWF Open Data cycle {candidate} not available yet ({e}); trying the previous cycle")
                candidate = candidate - timedelta(hours=6)
        raise last_exc

    def _fetch_real_waves_and_gusts(
        self,
        initial_time: datetime,
        timesteps: List[datetime],
        cfg: Dict,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Overlay real wave height (swh) and wind gust (fg10m) from ECMWF
        Open Data (IFS_FX) onto the already-cropped regional grid, in place
        of the parametrized wave estimate / gust heuristic. Best-effort:
        returns None (caller keeps the existing parametrization) if the
        tier's timestep doesn't align with Open Data's 3-hour lead-time
        granularity, or the fetch fails for any reason (network, data not
        yet disseminated for this cycle, etc.) - this must never be the
        reason a forecast fails outright.

        Wall-clock bounded (config.REAL_WAVE_OVERLAY["timeout_seconds"]):
        ECMWF Open Data can return HTTP 503 for a given param/step, and the
        underlying ecmwf-opendata client's own retry/backoff on that is slow
        (observed: 120s per failing step) - without a hard cap here, a run
        with several failing steps can blow past a calling agent's own
        tool-call timeout before this function's try/except ever gets a
        chance to fall back, which defeats the "best-effort, never blocks"
        point of this whole method. Runs in a worker thread so it can
        actually be abandoned on timeout instead of just given up on while
        it keeps running - the fetch itself isn't cancellable, but we stop
        waiting on it and fall back immediately either way.
        """
        if cfg["timestep_hours"] % 3 != 0:
            logger.info(
                f"Timestep {cfg['timestep_hours']}h doesn't align with ECMWF Open "
                "Data's 3h lead-time granularity - keeping parametrized waves"
            )
            return None

        def _fetch_one(source, variable: str, lead_times: np.ndarray, interp_to: Dict, cycle_time) -> np.ndarray:
            from earth2studio.data.utils import fetch_data

            data, coords = fetch_data(
                source,
                np.array([cycle_time], dtype="datetime64[ns]"),
                [variable],
                lead_time=lead_times,
                interp_to=interp_to,
                interp_method="linear",
            )
            arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            # Shape is (time=1, lead_time, variable, lat, lon); drop the
            # size-1 time axis so this matches the rest of forecast_data's
            # per-variable (lead_time, lat, lon) convention.
            return arr[0][:, list(coords["variable"]).index(variable)]

        def _fetch_gust(all_lead_times: np.ndarray, interp_to: Dict, cycle_time) -> Optional[np.ndarray]:
            # See datasources.GUST_10FG3_GAP_START_H/END_H - ECMWF stops
            # publishing plain "10fg" in this lead-time range, only "10fg3"
            # exists there. Fetch each sub-range with the param that
            # actually exists for it, then reassemble in original order.
            hours = all_lead_times.astype("timedelta64[h]").astype(int)
            in_gap = (hours >= datasources.GUST_10FG3_GAP_START_H) & (hours <= datasources.GUST_10FG3_GAP_END_H)

            out = None
            for mask, source_factory in (
                (~in_gap, datasources.get_wave_forecast_source),
                (in_gap, datasources.get_gust_forecast_source_10fg3),
            ):
                if not mask.any():
                    continue
                sub = _fetch_one(source_factory(), "fg10m", all_lead_times[mask], interp_to, cycle_time)
                if out is None:
                    out = np.full((len(all_lead_times),) + sub.shape[1:], np.nan)
                out[mask] = sub
            return out

        def _do_fetch() -> Dict[str, np.ndarray]:
            cycle_time = self._resolve_available_cycle(datasources.get_wave_forecast_source, initial_time)
            lead_times = np.array([ts - cycle_time for ts in timesteps], dtype="timedelta64[ns]")
            # IFS Open Data's native longitude is 0-360, but lon_array here
            # is the already-cropped regional grid, which _crop_to_region
            # rebuilds analytically and can land in the negative branch
            # (e.g. -2.25..3.75 for a window near the Solent - see its
            # docstring). xarray's .interp() (used under fetch_data's
            # interp_to) does NOT wrap longitude, so any negative target
            # value falls outside the source's [0, 360) coordinate range and
            # interpolates to NaN. Only the sliver of the window that
            # happened to already be >=0 (the *eastern* edge) came back with
            # real data - live-confirmed: for the Solent this NaN'd out
            # almost the whole window and left a narrow strip of real IFS
            # data actually covering the North Sea/Dover Strait side of the
            # crop, not the Solent itself. Normalize to 0-360 for the
            # interpolation target only; the returned array still aligns
            # positionally with our original (possibly negative-branch)
            # lat_array/lon_array, so nothing downstream needs to change.
            interp_to = {"_lat": np.asarray(lat_array), "_lon": np.asarray(lon_array) % 360.0}

            swh = _fetch_one(datasources.get_wave_forecast_source(), "swh", lead_times, interp_to, cycle_time)
            gust = _fetch_gust(lead_times, interp_to, cycle_time)

            # ECMWF's global wave model doesn't resolve narrow/enclosed
            # waters (the Solent, for instance, comes back entirely NaN -
            # masked as land-adjacent by the wave grid) even though the
            # atmospheric gust field is fine there. Keep each field
            # independently rather than discarding a good gust fetch just
            # because the wave grid has no signal for this window.
            result = {}
            if not np.isnan(swh).all():
                result["waves"] = swh
            else:
                logger.info("IFS wave data is all-NaN for this region (too enclosed/coastal for the global wave model) - keeping parametrized waves")
            if gust is not None and not np.isnan(gust).all():
                result["wind_gust"] = gust

            if not result:
                raise ValueError("IFS wave/gust fetch returned no usable data for this region")
            return result

        timeout_s = config.REAL_WAVE_OVERLAY.get("timeout_seconds", 45)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            result = pool.submit(_do_fetch).result(timeout=timeout_s)
            logger.info(f"Overlaid real IFS data: {list(result.keys())}")
            return result
        except FuturesTimeoutError:
            logger.warning(
                f"Real wave/gust overlay took longer than {timeout_s}s (likely ECMWF "
                "rate-limiting), abandoning it and keeping parametrized estimate"
            )
            return None
        except Exception as e:
            logger.warning(f"Real wave/gust overlay failed, keeping parametrized estimate: {e}")
            return None
        finally:
            # wait=False: on a timeout, the fetch thread is still running
            # (stuck in ECMWF's retry loop) - don't block here waiting for
            # it to finish, that would silently reintroduce the exact delay
            # this timeout exists to avoid. It'll finish and die on its own;
            # its result is simply discarded.
            pool.shutdown(wait=False)

    def generate_ensemble_risk(
        self,
        location: Dict,
        radius_miles: int,
        cfg: Dict,
        timesteps: List[datetime],
        initial_time: datetime,
        source: str = "aifs",
    ) -> List[Dict]:
        """Probabilistic storm risk from an ECMWF Open Data ensemble
        (AIFS_ENS_FX or IFS_ENS_FX), fetched directly - no local model
        inference. Builds its own small regional grid (rather than reusing
        the deterministic run's, so this works even when called on its own)
        and fetches config.ENSEMBLE['members'] members' u10m/v10m for the
        requested timesteps, then hands them to
        gale_warnings.generate_probabilistic_wind_warnings.
        """
        if cfg["timestep_hours"] % 3 != 0:
            logger.info(f"Timestep {cfg['timestep_hours']}h doesn't align with ensemble data - skipping")
            return []

        from earth2studio.data.utils import fetch_data

        lat_array, lon_array = regions.build_regional_grid(location["lat"], location["lon"], radius_miles)
        interp_to = {"_lat": lat_array, "_lon": lon_array}

        cycle_time = self._resolve_available_cycle(
            lambda: datasources.get_ensemble_forecast_source(source, 1), initial_time
        )
        lead_times = np.array([ts - cycle_time for ts in timesteps], dtype="timedelta64[ns]")

        n_members = config.ENSEMBLE["members"]
        member_u, member_v = [], []
        # Perturbed members only (1..n), not the control (member=0): IFS ENS
        # Open Data no longer publishes its control-member index reliably
        # (earth2studio itself warns about this), and excluding it doesn't
        # meaningfully bias a probability estimate from n>=5 members anyway.
        for member in range(1, n_members + 1):
            try:
                src = datasources.get_ensemble_forecast_source(source, member)
                data, coords = fetch_data(
                    src,
                    np.array([cycle_time], dtype="datetime64[ns]"),
                    ["u10m", "v10m"],
                    lead_time=lead_times,
                    interp_to=interp_to,
                    interp_method="linear",
                )
                variables = list(coords["variable"])
                arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                arr = arr[0]  # drop size-1 time axis
                member_u.append(arr[:, variables.index("u10m")])
                member_v.append(arr[:, variables.index("v10m")])
            except Exception as e:
                logger.warning(f"Ensemble member {member} ({source}) fetch failed, skipping it: {e}")

        if len(member_u) < 2:
            logger.warning(f"Only {len(member_u)} ensemble member(s) fetched successfully - skipping probabilistic warnings")
            return []

        member_u = np.stack(member_u)
        member_v = np.stack(member_v)
        source_label = f"{source.upper()} ensemble ({len(member_u)} of {config.ENSEMBLE['total_members']} members)"

        return gale_warnings.generate_probabilistic_wind_warnings(
            member_u, member_v, timesteps, location.get("name", "Unknown Location"),
            source_label=source_label,
            min_force=config.WARNINGS["min_beaufort_force"],
            probability_threshold=config.ENSEMBLE["gale_probability_threshold"],
        )

    def _release_memory(self):
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU memory released")


_engine = None


def get_engine() -> ForecastEngine:
    global _engine
    if _engine is None:
        _engine = ForecastEngine()
    return _engine
