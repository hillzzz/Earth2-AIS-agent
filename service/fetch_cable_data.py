"""
Fetches submarine cable route geometry and clips it to the Baltic, for
ais_mcp_server.py's check_ais_gaps (the "went dark near a cable" use
case). Not committed to git (data/ is gitignored, and this is fetched
third-party data, not something this project authors) - run this once to
produce data/baltic_cables.geojson, re-run any time to refresh it.

Data source: https://www.submarinecablemap.com/api/v3/cable/cable-geo.json
it's a reasonable approximation good enough for a proximity heuristic, not for anything safety-critical
or commercial.

Usage:
    python3 fetch_cable_data.py
"""

import json
import logging
from pathlib import Path

import requests
from shapely.geometry import box, shape

import config
import regions

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

CABLE_SOURCE_URL = "https://www.submarinecablemap.com/api/v3/cable/cable-geo.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "baltic_cables.geojson"


def main() -> None:
    logger.info(f"Fetching {CABLE_SOURCE_URL} ...")
    resp = requests.get(CABLE_SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"{len(data['features'])} cable systems in source data")

    # Baltic basin extent - same as config.py's _ais_bounding_boxes() Baltic
    # box, so "near a cable" and "in the Baltic AIS coverage" agree.
    baltic_area = next(a for a in regions.ALL_AREAS.values() if a.basin == "baltic")
    lat_s, lon_w, lat_n, lon_e = baltic_area.bounds
    for area in regions.ALL_AREAS.values():
        if area.basin != "baltic":
            continue
        s, w, n, e = area.bounds
        lat_s, lon_w = min(lat_s, s), min(lon_w, w)
        lat_n, lon_e = max(lat_n, n), max(lon_e, e)
    baltic_box = box(lon_w, lat_s, lon_e, lat_n)

    clipped = []
    for feat in data["features"]:
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        if geom.intersects(baltic_box):
            clipped.append(feat)
    logger.info(f"{len(clipped)} cables intersect the Baltic ({lat_s},{lon_w} to {lat_n},{lon_e})")

    out = {
        "type": "FeatureCollection",
        "name": "baltic_submarine_cables",
        "source": (
            "https://www.submarinecablemap.com (public API the interactive map "
            "itself uses) - an approximation, not TeleGeography's own licensed "
            "authoritative dataset. Fine for a proximity heuristic, not for "
            "anything safety-critical or commercial."
        ),
        "features": clipped,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f)
    logger.info(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
