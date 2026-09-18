"""
MCP server exposing the live AIS feed (ais_ingest.py / ais_db.py) as tools
for Hermes - vessel lookup, area search, storm-risk correlation with the
forecast pipeline, and AIS-gap ("went dark near X") detection.

A separate server from mcp_server.py (earth2-forecast) - different domain,
independently enable/disable-able. See config.py's AIS_* comment block for
why the underlying feed is a separate aisstream.io connection from the
Thor's existing AIS stack.

Register with Hermes:
    hermes mcp add ais-tracker \\
        --command /path/to/earth2-maritime-agent/.venv/bin/python \\
        --args /path/to/earth2-maritime-agent/service/ais_mcp_server.py

Run standalone for testing:
    python3 ais_mcp_server.py
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.mcpserver import MCPServer

import ais_db
import agent_tools
import config
import regions

CABLE_GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "baltic_cables.geojson"

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

server = MCPServer(
    name="ais-tracker",
    description="Live AIS vessel tracking (UK/North Atlantic/North Sea/Baltic waters): vessel lookup, area search, storm-risk correlation, AIS-gap detection.",
)


def _resolve_location(location: str) -> Optional[Dict]:
    """A location string is either "lat,lon" or a name regions.py knows
    (Shipping Forecast area, port, Baltic area) - same two paths app.py's
    own parse_location supports, reimplemented here since this server
    doesn't share that module."""
    location = location.strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", location)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return {"name": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon}
    return regions.resolve_location(location)


def _bbox_around(lat: float, lon: float, radius_nm: float) -> tuple:
    """(lat_s, lon_w, lat_n, lon_e) around a center point. Longitude
    degrees shrink with latitude (cos(lat)) - the min(...,~0.1) floor
    just guards against a division blowing up exactly at the pole,
    nowhere this project's coverage ever reaches."""
    dlat = radius_nm / 60.0
    dlon = radius_nm / 60.0 / max(0.1, math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


@server.tool()
def vessel_lookup(name_or_mmsi: str) -> dict:
    """Look up a specific vessel's last known position and details by name
    (case-insensitive substring match) or exact 9-digit MMSI. Only finds
    vessels seen within this feed's coverage (UK/North Atlantic/North Sea/
    Baltic waters) and while the ingestion
    service has been running - a vessel never seen in these waters, or
    that hasn't transmitted since ingestion started, won't be found.

    Args:
        name_or_mmsi: Vessel name (partial match ok, e.g. "TANKER ONE") or MMSI (e.g. "235000001").
    """
    result = ais_db.find_vessel(name_or_mmsi)
    if result is None:
        return {"found": False, "message": f"No vessel matching {name_or_mmsi!r} in the tracked coverage area."}
    return {"found": True, "vessel": result}


@server.tool()
def vessels_near(location: str, radius_nm: float = 20.0, ship_type: Optional[str] = None,
                  min_length_m: Optional[float] = None) -> dict:
    """List vessels currently tracked near a location, optionally filtered
    by type and minimum length. Only returns vessels with a position
    update in the last 6 hours (older than that isn't "currently near").

    For "supertankers": ship_type="Tanker", min_length_m=250 is a
    reasonable approximation - AIS ship-type codes distinguish Tanker
    broadly, not VLCC/ULCC specifically, so this is a size-based proxy,
    not an exact classification.

    Args:
        location: A Met Office shipping forecast area, UK port, Baltic
            area name, or "lat,lon" for anywhere else within this feed's
            coverage (UK/North Atlantic/North Sea/Baltic waters).
        radius_nm: Search radius in nautical miles (default 20).
        ship_type: Optional filter - one of "Tanker", "Cargo", "Fishing",
            "Passenger", "Tug", "Pleasure craft", "Military", etc.
        min_length_m: Optional minimum vessel length in metres (from AIS
            hull dimensions).
    """
    loc = _resolve_location(location)
    if loc is None:
        return {"error": f"Unknown location '{location}'. Try a Met Office shipping forecast area, UK port, Baltic area name, or 'lat,lon'."}

    lat_s, lon_w, lat_n, lon_e = _bbox_around(loc["lat"], loc["lon"], radius_nm)
    vessels = ais_db.vessels_in_bbox(lat_s, lon_w, lat_n, lon_e, ship_type_prefix=ship_type, min_length_m=min_length_m)
    return {
        "location": loc["name"],
        "radius_nm": radius_nm,
        "ship_type_filter": ship_type,
        "min_length_m_filter": min_length_m,
        "count": len(vessels),
        "vessels": vessels,
    }


@server.tool()
def vessels_at_storm_risk(location: str, forecast_type: str = "navigation") -> dict:
    """Run the maritime forecast for a location (same pipeline as the
    earth2-forecast MCP server's forecast_generate) and, if it carries a
    gale-force (Beaufort 8+) warning, list the vessels currently tracked
    within that sea area - the storm-risk correlation use case. If there's
    no gale warning, returns at_risk=False without an AIS query (nothing
    to correlate against).

    Args:
        location: A Met Office shipping forecast area, UK port, or Baltic area name.
        forecast_type: "nowcast" (0-6h), "navigation" (5-day, default), or "passage" (15-day).
    """
    forecast = agent_tools.generate_forecast(location, forecast_type)
    if "error" in forecast:
        return {"error": forecast["error"]}

    highest_force = forecast.get("highest_force")
    if highest_force is None or highest_force < 8:
        return {
            "location": forecast["location"],
            "forecast_type": forecast_type,
            "highest_force": highest_force,
            "at_risk": False,
            "message": "No gale-force warning for this forecast - no vessels flagged.",
        }

    loc = _resolve_location(location)
    area = regions.area_for_location(loc["lat"], loc["lon"]) if loc else None
    if area is not None:
        lat_s, lon_w, lat_n, lon_e = area.bounds
    else:
        lat_s, lon_w, lat_n, lon_e = _bbox_around(loc["lat"], loc["lon"], 30.0)

    vessels = ais_db.vessels_in_bbox(lat_s, lon_w, lat_n, lon_e)
    return {
        "location": forecast["location"],
        "forecast_type": forecast_type,
        "highest_force": highest_force,
        "gale_warnings": forecast.get("gale_warnings"),
        "sea_state_warnings": forecast.get("sea_state_warnings"),
        "daily_breakdown": forecast.get("daily_breakdown"),
        "at_risk": True,
        "vessels_in_area": len(vessels),
        "vessels": vessels,
    }


def _load_cable_lines():
    """Baltic submarine cable routes (service/fetch_cable_data.py's
    output) as shapely LineStrings, or None if that script hasn't been
    run yet. Loaded fresh each call - the file is small (~44KB) and each
    MCP tool call is a fresh subprocess anyway, so there's no persistent
    process to usefully cache this in."""
    if not CABLE_GEOJSON_PATH.exists():
        return None
    from shapely.geometry import shape

    with open(CABLE_GEOJSON_PATH) as f:
        data = json.load(f)
    lines = []
    for feat in data.get("features", []):
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        lines.append((feat.get("properties", {}).get("name", "Unknown cable"), geom))
    return lines


def _nm_to_nearest_cable(lat: float, lon: float, cable_lines) -> Optional[Dict]:
    """Nearest cable to a point, in nautical miles - a flat-earth degrees-
    to-nm approximation (1 deg latitude = 60 nm, longitude scaled by
    cos(lat)), which is fine at Baltic latitudes for a proximity heuristic
    over the short distances this flags."""
    from shapely.geometry import Point

    if not cable_lines:
        return None
    lat_scale = 60.0
    lon_scale = 60.0 * max(0.1, math.cos(math.radians(lat)))
    pt = Point(lon * lon_scale, lat * lat_scale)
    best_name, best_nm = None, float("inf")
    for name, geom in cable_lines:
        scaled = _scale_geom(geom, lat_scale, lon_scale)
        d = pt.distance(scaled)
        if d < best_nm:
            best_nm, best_name = d, name
    if best_name is None:
        return None
    return {"name": best_name, "distance_nm": round(best_nm, 2)}


def _scale_geom(geom, lat_scale: float, lon_scale: float):
    """Apply the same lon/lat -> nm scaling used for the query point, so
    shapely's planar .distance() is in nautical miles."""
    from shapely.ops import transform

    return transform(lambda x, y, z=None: (x * lon_scale, y * lat_scale), geom)


@server.tool()
def check_ais_gaps(location: Optional[str] = None, min_gap_hours: Optional[float] = None,
                    max_distance_from_cable_nm: Optional[float] = 5.0) -> dict:
    """Find vessels with an unusually long AIS silence whose last position
    before it was in the given area - the "went dark near X" case. Each
    result says whether the vessel is STILL dark right now (the more
    urgent case) or reappeared elsewhere (reappeared_at shows where/when -
    reappearing far from where it went dark, given the elapsed time, is
    itself often the notable pattern).

    When max_distance_from_cable_nm is set (default 5.0) and cable route
    data is available (currently only the Baltic - run
    fetch_cable_data.py once to enable this), results are further
    filtered to vessels that went dark within that distance of a real
    submarine cable route - the undersea-cable-sabotage-concern use case.
    Set it to null/None to skip cable filtering and get general dark-
    vessel detection for any area instead.

    Args:
        location: A Met Office shipping forecast area, UK port, Baltic
            area name, or "lat,lon". Omit (the default) to scan the whole
            Baltic basin - the primary use case this tool was built for.
        min_gap_hours: Minimum silence duration to flag (default: config.AIS_GAP_THRESHOLD_HOURS, 2h).
        max_distance_from_cable_nm: Max distance from a known cable route to flag, or null to disable cable filtering.
    """
    if location is None:
        location_name = "Baltic"
        lat_s, lon_w, lat_n, lon_e = 90.0, 180.0, -90.0, -180.0
        for area in regions.ALL_AREAS.values():
            if area.basin != "baltic":
                continue
            s, w, n, e = area.bounds
            lat_s, lon_w = min(lat_s, s), min(lon_w, w)
            lat_n, lon_e = max(lat_n, n), max(lon_e, e)
    else:
        loc = _resolve_location(location)
        if loc is None:
            return {"error": f"Unknown location '{location}'. Try a Met Office shipping forecast area, UK port, Baltic area name, or 'lat,lon'."}
        location_name = loc["name"]
        area = regions.area_for_location(loc["lat"], loc["lon"])
        if area is not None:
            lat_s, lon_w, lat_n, lon_e = area.bounds
        else:
            lat_s, lon_w, lat_n, lon_e = _bbox_around(loc["lat"], loc["lon"], 50.0)

    gaps = ais_db.find_ais_gaps(lat_s, lon_w, lat_n, lon_e, min_gap_hours=min_gap_hours)

    cable_note = None
    if max_distance_from_cable_nm is not None:
        cable_lines = _load_cable_lines()
        if cable_lines is None:
            cable_note = "No cable route data loaded (run fetch_cable_data.py) - returning all AIS gaps in the area, unfiltered by cable proximity."
        else:
            filtered = []
            for v in gaps:
                nearest = _nm_to_nearest_cable(v["lat"], v["lon"], cable_lines)
                if nearest and nearest["distance_nm"] <= max_distance_from_cable_nm:
                    v["nearest_cable"] = nearest
                    filtered.append(v)
            gaps = filtered

    return {
        "location": location_name,
        "min_gap_hours": min_gap_hours or config.AIS_GAP_THRESHOLD_HOURS,
        "cable_filter_nm": max_distance_from_cable_nm,
        "cable_note": cable_note,
        "count": len(gaps),
        "vessels": gaps,
    }


if __name__ == "__main__":
    server.run(transport="stdio")
