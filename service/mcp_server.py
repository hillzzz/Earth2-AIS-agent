"""
MCP server exposing the maritime forecast pipeline as tools for Hermes (or
any other MCP-speaking agent) - generate a forecast, push it to OpenCPN,
check OpenCPN's REST bridge status. Thin wrapper around agent_tools.py;
see that module for the actual implementation and config.py for the
OpenCPN/Thor connection settings.

Register with Hermes:
    hermes mcp add earth2-forecast \\
        --command /path/to/earth2-maritime-agent/.venv/bin/python \\
        --args /path/to/earth2-maritime-agent/service/mcp_server.py

Run standalone for testing:
    python3 mcp_server.py
"""

import logging
from typing import Optional

from mcp.server.mcpserver import MCPServer

import config
from agent_tools import check_opencpn_status, generate_forecast, push_grib_to_opencpn

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

server = MCPServer(
    name="earth2-forecast",
    description="Maritime weather forecasts (UK/North Atlantic/North Sea/Baltic) and pushing them to OpenCPN as a GRIB2 overlay.",
)


@server.tool()
def forecast_generate(location: str, forecast_type: str = "navigation") -> dict:
    """Run the Earth-2 maritime forecast pipeline for a location and export it as
    a GRIB2 file. Returns a summary: model used, gale/sea-state warning counts,
    highest Beaufort force, tracked storms, and daily_breakdown - one entry per
    day (date, max_beaufort_force, max_wave_height_m, worst_timing). This
    already answers "what's the forecast/what's it looking like for the next
    few days" directly - no need to read the GRIB2 file, this project's source
    code, or any other tool to get day-by-day detail. Also returns grib_path,
    to pass to opencpn_push if the forecast should be pushed to OpenCPN -
    forecast_generate never pushes by itself.

    Args:
        location: A Met Office shipping forecast area (e.g. "Dogger", "Irish Sea"),
            a UK port, or a Baltic area name.
        forecast_type: "nowcast" (0-6h), "navigation" (5-day, default), or
            "passage" (15-day).
    """
    return generate_forecast(location, forecast_type)


@server.tool()
def opencpn_push(grib_path: str) -> dict:
    """Push a GRIB2 file (from forecast_generate's grib_path) to the running
    OpenCPN instance so it loads automatically in grib_pi.

    Args:
        grib_path: Absolute path to the .grb2 file, from forecast_generate's output.
    """
    return push_grib_to_opencpn(grib_path)


@server.tool()
def opencpn_status() -> dict:
    """Check whether OpenCPN's REST server is reachable and this agent is paired
    with it. Call this if a push fails or at the start of a session."""
    return check_opencpn_status()


if __name__ == "__main__":
    server.run(transport="stdio")
