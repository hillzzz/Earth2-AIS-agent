"""
Plain functions wrapping the forecast pipeline (forecast_engine, gribexport, opencpn_bridge),
exposed to agents as MCP tools by mcp_server.py.
Kept separate from the MCP wiring so the logic is testable without an MCP client/server round-trip 
the agent doesn't get a "generate arbitrary code" tool, only these specific, bounded actions, and every
function returns a small JSON-serializable summary rather than raw forecast
arrays, to keep an agent's context bounded over a long session.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import config
import gribexport
import regions
from forecast_engine import get_engine
from opencpn_bridge import OpenCPNBridge, OpenCPNBridgeError

logger = logging.getLogger(__name__)

AGENT_FORECAST_DIR = config.FORECAST_DIR / "agent"


def _summarize_warnings(warnings_bundle) -> Dict:
    if not warnings_bundle:
        return {"gale_warnings": 0, "sea_state_warnings": 0, "storms_tracked": 0, "highest_force": None}
    wind = warnings_bundle.get("wind", [])
    sea = warnings_bundle.get("sea_state", [])
    storms = warnings_bundle.get("storms", [])
    forces = [w.get("beaufort_force") for w in wind if isinstance(w, dict) and w.get("beaufort_force") is not None]
    return {
        "gale_warnings": len(wind),
        "sea_state_warnings": len(sea),
        "storms_tracked": len(storms),
        "highest_force": max(forces) if forces else None,
    }


def generate_forecast(location: str, forecast_type: str = "navigation") -> Dict:
    """Run the forecast pipeline for a location and export it to GRIB2.
    Includes daily_breakdown (one entry per day: max_beaufort_force,
    max_wave_height_m, worst_timing) this is hopefully already? everything needed to
    answer a "what's the forecast for the next few days" question; there is
    no need to read the GRIB2 file or this project's source code to get
    more detail. Does NOT push to OpenCPN - call push_grib_to_opencpn
    separately."""
    resolved = regions.resolve_location(location)
    if resolved is None:
        return {"error": f"Unknown location '{location}'. Try a Met Office shipping forecast area or UK port name."}

    if forecast_type not in config.FORECAST_CONFIGS:
        return {"error": f"Unknown forecast_type '{forecast_type}'. Valid: {list(config.FORECAST_CONFIGS)}"}

    forecast_id = f"agent_{forecast_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cfg = config.FORECAST_CONFIGS[forecast_type]

    try:
        engine = get_engine()
        result = engine.generate_forecast(
            forecast_id=forecast_id,
            location=resolved,
            forecast_type=forecast_type,
            radius_miles=cfg["radius_miles"],
        )
    except Exception as e:
        logger.error(f"Agent forecast generation failed: {e}", exc_info=True)
        return {"error": f"Forecast generation failed: {e}"}

    grib_path = gribexport.export_grib2(result, AGENT_FORECAST_DIR / f"{forecast_id}.grb2")
    if grib_path is None:
        return {"error": "Forecast generated but GRIB2 export produced no messages"}

    return {
        "forecast_id": forecast_id,
        "grib_path": str(grib_path),
        "location": resolved["name"],
        "forecast_type": forecast_type,
        "model": result["metadata"]["model"],
        "initial_time": result["metadata"]["initial_time"],
        "wave_source": result["metadata"]["wave_source"],
        "daily_breakdown": result.get("daily_breakdown", []),
        **_summarize_warnings(result.get("warnings")),
    }


def push_grib_to_opencpn(grib_path: str) -> Dict:
    """Push a previously-generated GRIB2 file to the running OpenCPN
    instance's grib_pi, so it opens automatically."""
    try:
        bridge = OpenCPNBridge()
        result = bridge.push_grib(Path(grib_path))
        return {"pushed": True, "grib_path": grib_path, "response": result}
    except OpenCPNBridgeError as e:
        return {"pushed": False, "error": str(e)}
    except Exception as e:
        logger.error(f"push_grib_to_opencpn failed: {e}", exc_info=True)
        return {"pushed": False, "error": f"Unexpected error: {e}"}


def check_opencpn_status() -> Dict:
    """Check whether OpenCPN's REST server is reachable and this client is
    paired. Use this before generating a forecast if you're unsure whether
    a push will succeed."""
    bridge = OpenCPNBridge()
    try:
        version = bridge.get_version()
    except Exception as e:
        return {"reachable": False, "error": f"OpenCPN REST server unreachable at {bridge.base_url}: {e}"}

    if not bridge.api_key:
        return {"reachable": True, "paired": False, "version": version, "error": "No OPENCPN_API_KEY configured"}

    try:
        bridge.ping()
        return {"reachable": True, "paired": True, "version": version}
    except Exception as e:
        return {"reachable": True, "paired": False, "version": version, "error": str(e)}
