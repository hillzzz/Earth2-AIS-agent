"""
Gale/storm warning generation and low-pressure storm tracking.

Turns raw forecast fields (wind, gusts, waves, mean sea level pressure) into
UK Shipping Forecast-style warnings:
  - Beaufort-scale gale/storm/violent-storm/hurricane-force warnings, timed
    as Imminent (<6h) / Soon (6-12h) / Later (12-24h), per Met Office
    convention.
  - Rough/high seas advisories from significant wave height.
  - Low-pressure storm tracking from MSLP minima, including explosive
    cyclogenesis ("weather bomb") detection - central pressure falling
    >= 24 hPa in 24h, common in North Atlantic / North Sea windstorms.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MS_TO_KTS = 1.94384

# (lower_kt, force, name). The traditional Beaufort table is usually printed
# as inclusive integer bands (e.g. 28-33 kt, 34-40 kt), which is fine for
# whole-knot values but leaves gaps for real-valued speeds (33.4 kt falls
# between those two bands). Using just the lower threshold of each force and
# taking the highest one the speed clears avoids that gap entirely.
BEAUFORT_SCALE = [
    (0, 0, "Calm"),
    (1, 1, "Light Air"),
    (4, 2, "Light Breeze"),
    (7, 3, "Gentle Breeze"),
    (11, 4, "Moderate Breeze"),
    (17, 5, "Fresh Breeze"),
    (22, 6, "Strong Breeze"),
    (28, 7, "Near Gale"),
    (34, 8, "Gale"),
    (41, 9, "Severe Gale"),
    (48, 10, "Storm"),
    (56, 11, "Violent Storm"),
    (64, 12, "Hurricane Force"),
]


def beaufort_force(speed_kts: float) -> int:
    force = 0
    for lo, f, _name in BEAUFORT_SCALE:
        if speed_kts >= lo:
            force = f
        else:
            break
    return force


def beaufort_name(force: int) -> str:
    for _lo, f, name in BEAUFORT_SCALE:
        if f == force:
            return name
    return "Hurricane Force"


def _timing_bucket(hours_ahead: float) -> str:
    if hours_ahead < 6:
        return "Imminent"
    if hours_ahead < 12:
        return "Soon"
    if hours_ahead < 24:
        return "Later"
    return "Outlook"


def _format_valid_time(ts: datetime) -> str:
    """e.g. 'Thu 17 Sep 18:00 UTC' - every warning message includes this so
    multiple entries (which often share the same Imminent/Soon/Later label)
    are still distinguishable, and so the message is still meaningful if
    read well after the forecast was generated."""
    return ts.strftime("%a %d %b %H:%M UTC")


def generate_wind_warnings(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    timesteps: List[datetime],
    area_name: str,
    min_force: int = 8,
    gust_factor: float = 1.3,
    base_time: Optional[datetime] = None,
) -> List[Dict]:
    """
    Scan each forecast timestep for the strongest wind anywhere in the domain
    and emit a warning whenever the Beaufort force reaches min_force.

    wind_u/wind_v: arrays shaped [time, lat, lon] (m/s).
    gust_factor: applied to mean wind to approximate gusts when a dedicated
    gust field isn't available (see generate_wind_warnings_with_gusts).
    base_time: the forecast's own issue/analysis time (timesteps[0]) - not
    wall-clock "now". Imminent/Soon/Later/Outlook is timed from when the
    forecast was issued, same as a real Shipping Forecast bulletin, not
    from whenever this function happens to run; using wall-clock time here
    would drift depending on how long the run/request pipeline took, and
    since the forecast's own base time is itself rounded back to the most
    recent 6-hourly analysis cycle (which can already be a few hours behind
    actual now), timing off wall-clock "now" could make several of the
    earliest timesteps all read as "Imminent" at once.
    """
    base_time = base_time or (timesteps[0] if timesteps else datetime.now(timezone.utc))
    warnings = []

    for i, ts in enumerate(timesteps):
        if i >= wind_u.shape[0]:
            break
        speed_ms = np.sqrt(wind_u[i] ** 2 + wind_v[i] ** 2)
        max_speed_kts = float(np.nanmax(speed_ms)) * MS_TO_KTS
        gust_kts = max_speed_kts * gust_factor
        force = beaufort_force(gust_kts)

        if force >= min_force:
            hours_ahead = max(0.0, (ts - base_time).total_seconds() / 3600.0)
            timing = _timing_bucket(hours_ahead)
            warnings.append({
                "type": "wind",
                "area": area_name,
                "valid_time": ts.isoformat(),
                "timing": timing,
                "beaufort_force": force,
                "beaufort_name": beaufort_name(force),
                "max_mean_wind_kts": round(max_speed_kts, 1),
                "max_gust_kts": round(gust_kts, 1),
                "message": (
                    f"{beaufort_name(force)} warning (Force {force}) for {area_name}: "
                    f"gusts to {gust_kts:.0f} kt, {_format_valid_time(ts)} ({timing.lower()})"
                ),
            })

    return warnings


def generate_wave_warnings(
    wave_height_m: np.ndarray,
    timesteps: List[datetime],
    area_name: str,
    threshold_m: float = 4.0,
    base_time: Optional[datetime] = None,
) -> List[Dict]:
    """Rough/high seas advisories from significant wave height (Douglas sea
    scale). base_time: see generate_wind_warnings - the forecast's own issue
    time, not wall-clock "now"."""
    base_time = base_time or (timesteps[0] if timesteps else datetime.now(timezone.utc))
    warnings = []

    douglas = [
        (4.0, 6.0, "Rough Seas"),
        (6.0, 9.0, "Very Rough Seas"),
        (9.0, 14.0, "High Seas"),
        (14.0, float("inf"), "Phenomenal Seas"),
    ]

    for i, ts in enumerate(timesteps):
        if i >= wave_height_m.shape[0]:
            break
        max_wave = float(np.nanmax(wave_height_m[i]))
        # np.nanmax of an all-NaN slice returns nan with a warning, and
        # `nan < threshold_m` is always False (IEEE754) - without this
        # explicit check that silently falls through instead of skipping,
        # producing a "significant wave height up to nanm" advisory. A real
        # case, not hypothetical: the global wave model doesn't resolve
        # narrow/enclosed waters like the Solent at all, so its swh comes
        # back entirely NaN there even when wind/gust data is fine.
        if np.isnan(max_wave) or max_wave < threshold_m:
            continue

        label = next((name for lo, hi, name in douglas if lo <= max_wave < hi), "High Seas")
        hours_ahead = max(0.0, (ts - base_time).total_seconds() / 3600.0)
        timing = _timing_bucket(hours_ahead)
        warnings.append({
            "type": "sea_state",
            "area": area_name,
            "valid_time": ts.isoformat(),
            "timing": timing,
            "max_wave_height_m": round(max_wave, 1),
            "message": (
                f"{label} advisory for {area_name}: significant wave height "
                f"up to {max_wave:.1f}m, {_format_valid_time(ts)} ({timing.lower()})"
            ),
        })

    return warnings


def track_storm_centers(
    mslp_hpa: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    timesteps: List[datetime],
    pressure_threshold_hpa: float = 995.0,
    explosive_hpa_24h: float = 24.0,
) -> List[Dict]:
    """
    Track low-pressure storm centers across the forecast from MSLP minima.

    Finds local minima below pressure_threshold_hpa in each timestep's MSLP
    field, then greedily links minima across consecutive timesteps by nearest
    distance (< ~500 km) to build tracks. Flags explosive cyclogenesis
    ("weather bomb") when a tracked low deepens by >= explosive_hpa_24h in
    24 hours - the classic signature of North Atlantic / North Sea windstorms.

    mslp_hpa: array shaped [time, lat, lon].
    Returns a list of track dicts, each with a time-ordered list of
    {time, lat, lon, pressure_hpa} points and an "explosive_deepening" flag.
    """
    try:
        from scipy.ndimage import minimum_filter
    except ImportError:
        logger.warning("scipy not available, skipping storm-center tracking")
        return []

    if lat_array.ndim == 1 and lon_array.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    else:
        lat_grid, lon_grid = lat_array, lon_array

    per_timestep_minima = []
    for i in range(mslp_hpa.shape[0]):
        field = mslp_hpa[i]
        local_min = minimum_filter(field, size=5, mode="nearest") == field
        candidates = np.argwhere(local_min & (field < pressure_threshold_hpa))
        points = [
            {"lat": float(lat_grid[r, c]), "lon": float(lon_grid[r, c]), "pressure_hpa": float(field[r, c])}
            for r, c in candidates
        ]
        per_timestep_minima.append(points)

    # Greedy nearest-neighbour tracking between consecutive timesteps.
    tracks: List[List[Dict]] = []
    active: Dict[int, List[Dict]] = {}  # track index -> list of points so far

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    for i, points in enumerate(per_timestep_minima):
        ts = timesteps[i] if i < len(timesteps) else None
        unmatched = list(points)
        for idx, track_points in list(active.items()):
            last = track_points[-1]
            best_j, best_dist = None, 500.0  # max 500 km jump per step
            for j, p in enumerate(unmatched):
                d = haversine_km(last["lat"], last["lon"], p["lat"], p["lon"])
                if d < best_dist:
                    best_dist, best_j = d, j
            if best_j is not None:
                p = unmatched.pop(best_j)
                p["time"] = ts.isoformat() if ts else None
                track_points.append(p)
            else:
                tracks.append(track_points)
                del active[idx]

        for p in unmatched:
            p["time"] = ts.isoformat() if ts else None
            new_idx = max(active.keys(), default=-1) + 1
            active[new_idx] = [p]

    tracks.extend(active.values())

    results = []
    for track_points in tracks:
        if len(track_points) < 2:
            continue

        explosive = False
        for a in range(len(track_points)):
            for b in range(a + 1, len(track_points)):
                if track_points[a]["time"] is None or track_points[b]["time"] is None:
                    continue
                t_a = datetime.fromisoformat(track_points[a]["time"])
                t_b = datetime.fromisoformat(track_points[b]["time"])
                hours = (t_b - t_a).total_seconds() / 3600.0
                if 20 <= hours <= 28:
                    drop = track_points[a]["pressure_hpa"] - track_points[b]["pressure_hpa"]
                    if drop >= explosive_hpa_24h:
                        explosive = True

        min_pressure = min(p["pressure_hpa"] for p in track_points)
        times = [datetime.fromisoformat(p["time"]) for p in track_points if p["time"]]
        time_range = (
            f"{_format_valid_time(min(times))} - {_format_valid_time(max(times))}"
            if times else "time unknown"
        )
        results.append({
            "points": track_points,
            "min_pressure_hpa": round(min_pressure, 1),
            "explosive_deepening": explosive,
            "message": (
                f"Tracked low, minimum central pressure {min_pressure:.0f} hPa, "
                f"{time_range}"
                + (" - explosive cyclogenesis (\"weather bomb\")" if explosive else "")
            ),
        })

    return results


def daily_breakdown(forecast_data: Dict, base_time: Optional[datetime] = None) -> List[Dict]:
    """Lightweight per-calendar-day rollup: max Beaufort force, max wave
    height, and the "worst" (most urgent) timing bucket seen that day.

    For an agent consuming forecast_generate's result, the aggregate
    warning counts alone (e.g. "13 gale warnings") don't say which days
    those fall on, and the full per-timestep warning list is too much to
    put in every tool result - this sits between the two: one row per
    day, independent of whether it crossed the Force-8 warning threshold
    (so a calm day still shows its actual max force, e.g. Force 3, rather
    than being silently omitted like it would be from the warnings list).

    Groups by UTC calendar date of each timestep. "Worst timing" is the
    _timing_bucket of whichever timestep in that day is soonest after
    base_time - independent of which timestep had the max force/wave,
    since a day can validly have its worst-case wind and worst-case
    timing at different hours.
    """
    variables = forecast_data.get("variables", {})
    timesteps = forecast_data.get("timesteps", [])
    wind_u, wind_v = variables.get("wind_u"), variables.get("wind_v")
    if not timesteps or wind_u is None or wind_v is None:
        return []
    base_time = base_time or timesteps[0]

    gust = variables.get("wind_gust")
    waves = variables.get("waves")

    days: Dict[str, Dict] = {}
    min_hours: Dict[str, float] = {}

    for i, ts in enumerate(timesteps):
        if i >= wind_u.shape[0]:
            break
        if gust is not None and i < gust.shape[0] and not np.all(np.isnan(gust[i])):
            gust_kts = float(np.nanmax(gust[i])) * MS_TO_KTS
        else:
            speed_ms = np.sqrt(wind_u[i] ** 2 + wind_v[i] ** 2)
            gust_kts = float(np.nanmax(speed_ms)) * MS_TO_KTS * 1.3
        force = beaufort_force(gust_kts)
        # np.nanmax of an all-NaN slice is nan, not an error - guard
        # against it explicitly (max_wave_height_m: None) rather than
        # letting nan propagate into the comparisons below, where it would
        # silently compare False against everything. A real case: the
        # global wave model doesn't resolve narrow/enclosed waters like the
        # Solent at all, so swh comes back entirely NaN there.
        if waves is not None and i < waves.shape[0] and not np.all(np.isnan(waves[i])):
            wave_m = float(np.nanmax(waves[i]))
        else:
            wave_m = None
        hours_ahead = max(0.0, (ts - base_time).total_seconds() / 3600.0)

        key = ts.strftime("%Y-%m-%d")
        if key not in days:
            days[key] = {
                "date": key,
                "max_beaufort_force": force,
                "max_beaufort_name": beaufort_name(force),
                "max_wave_height_m": round(wave_m, 1) if wave_m is not None else None,
                "worst_timing": _timing_bucket(hours_ahead),
            }
            min_hours[key] = hours_ahead
            continue

        day = days[key]
        if force > day["max_beaufort_force"]:
            day["max_beaufort_force"] = force
            day["max_beaufort_name"] = beaufort_name(force)
        if wave_m is not None and (day["max_wave_height_m"] is None or wave_m > day["max_wave_height_m"]):
            day["max_wave_height_m"] = round(wave_m, 1)
        if hours_ahead < min_hours[key]:
            min_hours[key] = hours_ahead
            day["worst_timing"] = _timing_bucket(hours_ahead)

    return list(days.values())


def build_warnings_bundle(
    forecast_data: Dict,
    area_name: str,
    min_force: int = 8,
    wave_threshold_m: float = 4.0,
    storm_tracking: bool = True,
    local_forecast_data: Optional[Dict] = None,
) -> Dict:
    """Convenience wrapper: run all warning generators against a forecast_engine
    result's `variables` dict and return one bundle for the API response.

    local_forecast_data: if given, wind/sea-state warnings scan THIS grid
    instead of forecast_data - pass a tight crop around the named area
    (see forecast_engine._local_warning_radius_miles /
    _crop_to_region(..., pad=False)). Without it, a wind/wave warning
    reports the worst value *anywhere in forecast_data's window*, which for
    the wide, padded crop kept for chart context/storm tracking can be
    100-400+ miles from the named area - a real bug found live: a "gale
    warning for Solent" whose 36kt+ trigger was actually off Scotland, near
    Ireland, or in the Bay of Biscay.

    Storm tracking always uses forecast_data (the wide grid), regardless -
    spotting an approaching low needs to see beyond the immediate area, and
    each track's points already carry their own lat/lon so there's no
    mislabeling risk the way a single domain-max scan has.
    """
    wind_wave_data = local_forecast_data if local_forecast_data is not None else forecast_data
    variables = wind_wave_data.get("variables", {})
    timesteps = wind_wave_data.get("timesteps", [])
    # The forecast's own issue/analysis time, not wall-clock "now" - see
    # generate_wind_warnings for why that distinction matters.
    base_time = timesteps[0] if timesteps else None

    bundle = {"wind": [], "sea_state": [], "storms": [], "probabilistic": []}

    wind_u, wind_v = variables.get("wind_u"), variables.get("wind_v")
    if wind_u is not None and wind_v is not None:
        gust = variables.get("wind_gust")
        if gust is not None:
            bundle["wind"] = generate_wind_warnings_with_gusts(wind_u, gust, timesteps, area_name, min_force, base_time=base_time)
        else:
            bundle["wind"] = generate_wind_warnings(wind_u, wind_v, timesteps, area_name, min_force, base_time=base_time)

    waves = variables.get("waves")
    if waves is not None:
        bundle["sea_state"] = generate_wave_warnings(waves, timesteps, area_name, wave_threshold_m, base_time=base_time)

    mslp = forecast_data.get("variables", {}).get("mslp_hpa")
    lat_array, lon_array = forecast_data.get("lat_array"), forecast_data.get("lon_array")
    if storm_tracking and mslp is not None and lat_array is not None and lon_array is not None:
        bundle["storms"] = track_storm_centers(mslp, np.array(lat_array), np.array(lon_array), timesteps)

    return bundle


def generate_wind_warnings_with_gusts(
    wind_u: np.ndarray,
    wind_gust_ms: np.ndarray,
    timesteps: List[datetime],
    area_name: str,
    min_force: int = 8,
    base_time: Optional[datetime] = None,
) -> List[Dict]:
    """Same as generate_wind_warnings but uses a real gust field (e.g. from
    the earth2studio windgust-afno diagnostic) instead of a fixed multiplier.
    base_time: see generate_wind_warnings."""
    base_time = base_time or (timesteps[0] if timesteps else datetime.now(timezone.utc))
    warnings = []

    for i, ts in enumerate(timesteps):
        if i >= wind_gust_ms.shape[0]:
            break
        gust_kts = float(np.nanmax(wind_gust_ms[i])) * MS_TO_KTS
        mean_kts = float(np.nanmax(np.abs(wind_u[i]))) * MS_TO_KTS
        force = beaufort_force(gust_kts)

        if force >= min_force:
            hours_ahead = max(0.0, (ts - base_time).total_seconds() / 3600.0)
            timing = _timing_bucket(hours_ahead)
            warnings.append({
                "type": "wind",
                "area": area_name,
                "valid_time": ts.isoformat(),
                "timing": timing,
                "beaufort_force": force,
                "beaufort_name": beaufort_name(force),
                "max_mean_wind_kts": round(mean_kts, 1),
                "max_gust_kts": round(gust_kts, 1),
                "message": (
                    f"{beaufort_name(force)} warning (Force {force}) for {area_name}: "
                    f"gusts to {gust_kts:.0f} kt, {_format_valid_time(ts)} ({timing.lower()})"
                ),
            })

    return warnings


def generate_probabilistic_wind_warnings(
    member_wind_u: np.ndarray,
    member_wind_v: np.ndarray,
    timesteps: List[datetime],
    area_name: str,
    source_label: str,
    min_force: int = 8,
    gust_factor: float = 1.3,
    probability_threshold: float = 0.3,
    base_time: Optional[datetime] = None,
) -> List[Dict]:
    """
    Ensemble-based probabilistic gale/storm warnings: for each timestep, the
    fraction of ensemble members whose domain-max wind reaches min_force
    (approximated via gust_factor, same as the deterministic heuristic path -
    ECMWF Open Data's ensemble products don't carry a gust field). Emits a
    warning whenever that fraction clears probability_threshold, so a
    forecast can say "60% of members show gale force winds" instead of a
    single yes/no from one model run.

    member_wind_u/v: arrays shaped [member, time, lat, lon] (m/s).
    source_label: e.g. "AIFS ensemble (10 of 50 members)", surfaced in the
    warning message so it's clear this is a different kind of signal than
    the deterministic warnings.
    base_time: see generate_wind_warnings - the forecast's own issue time,
    not wall-clock "now".
    """
    base_time = base_time or (timesteps[0] if timesteps else datetime.now(timezone.utc))
    n_members = member_wind_u.shape[0]
    warnings = []

    for i, ts in enumerate(timesteps):
        if i >= member_wind_u.shape[1]:
            break

        member_forces = []
        for m in range(n_members):
            speed_ms = np.sqrt(member_wind_u[m, i] ** 2 + member_wind_v[m, i] ** 2)
            gust_kts = float(np.nanmax(speed_ms)) * MS_TO_KTS * gust_factor
            member_forces.append(beaufort_force(gust_kts))

        member_forces = np.array(member_forces)
        probability = float((member_forces >= min_force).mean())

        if probability >= probability_threshold:
            median_force = int(np.median(member_forces))
            hours_ahead = max(0.0, (ts - base_time).total_seconds() / 3600.0)
            timing = _timing_bucket(hours_ahead)
            warnings.append({
                "type": "probabilistic_wind",
                "area": area_name,
                "source": source_label,
                "valid_time": ts.isoformat(),
                "timing": timing,
                "probability": round(probability, 2),
                "median_beaufort_force": median_force,
                "message": (
                    f"{probability:.0%} of {source_label} members show Gale Force {min_force}+ "
                    f"winds for {area_name} at {_format_valid_time(ts)} ({timing.lower()}, "
                    f"median Force {median_force})"
                ),
            })

    return warnings
