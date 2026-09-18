"""
Configuration for the Earth-2 maritime agent layer (forecast pipeline,
OpenCPN bridge, AIS tracking) - the MCP tools Hermes calls directly.
"""
import os
from pathlib import Path

import regions

# Forecast storage
FORECAST_DIR = Path(__file__).resolve().parent.parent / "forecasts"

# Default location: Solent (the service's original focus area). Any of the
# named ports/sea-areas in regions.py can be requested instead.
DEFAULT_LOCATION = regions.resolve_location(regions.DEFAULT_AREA)

# ---------------------------------------------------------------------------
# Forecast tiers. "model" names are resolved by forecast_engine.MODEL_REGISTRY.
# ---------------------------------------------------------------------------
FORECAST_CONFIGS = {
    "nowcast": {
        # StormScopeMeteosatEU currently ALWAYS falls back to fcn: it's a
        # 10-min-cadence satellite image-frame model (consecutive Meteosat
        # MTG frames in, frames out), fundamentally incompatible with our
        # generic GFS -> earth2studio.run.deterministic pipeline - it fails
        # as soon as that pipeline tries to fetch GFS at a non-6-hourly
        # time, before it would even hit the deeper variable-name mismatch.
        # Wiring it up for real needs earth2studio.data.MeteosatFCI (a
        # EUMETSAT account) plus a dedicated image-frame orchestration path,
        # not the generic one - left as documented future work (see
        # README's Known gaps). Kept as the primary model rather than
        # simplified to fcn directly so that work benefits immediately once
        # done: forecast_engine.generate_forecast catches the inference
        # failure and falls back per-process (one wasted attempt at
        # startup, not per-request - see forecast_engine.MODEL_RESOLUTION_KM
        # for how the reported resolution is corrected after a fallback).
        "model": "stormscope",
        "fallback_model": "fcn",
        "duration_hours": 6,
        "resolution_km": 4,
        "timestep_hours": 1,
        "radius_miles": 40,
        "description": "0-6h nowcast for imminent gale/squall risk (currently FCN at 25km - see comment above)",
    },
    "navigation": {
        "model": "fcn",
        "duration_days": 5,
        "resolution_km": 25,
        "timestep_hours": 6,
        "radius_miles": 100,
        "description": "5-day navigation forecast",
    },
    "passage": {
        "model": "fcn",
        "duration_days": 15,
        "resolution_km": 25,
        "timestep_hours": 12,
        "radius_miles": 300,
        "description": "15-day passage planning forecast",
    },
}

# Harbor-scale super-resolution (CBottleInfill -> CBottleSR). Off by default
# (adds ~1-2 min/timestep); enable per-request with "super_resolution": true.
SUPER_RESOLUTION = {
    "enabled_by_default": False,
    "resolution": "10km",       # "10km" or "5km"
    "sampler_steps": 18,        # 18 quality / 10 speed
}

# Real wave height (swh) and wind gust (fg10m) overlay from ECMWF Open Data
# (IFS_FX/IFS_ENS_FX) - free, global, no credentials. Replaces the crude
# 0.025 * wind_kts**2 wave parametrization and the gust-factor heuristic
# with ECMWF's own wave model and gust forecast wherever the forecast
# tier's timestep aligns with Open Data's 3-hour lead-time granularity
# (navigation/passage; not nowcast's hourly steps). Best-effort: on any
# failure (network, data not yet disseminated, tier misaligned) this is a
# silent no-op and the existing parametrization/heuristic is used instead.
REAL_WAVE_OVERLAY = {
    "enabled_by_default": True,
    # ECMWF Open Data occasionally rate-limits (HTTP 503) or is missing a
    # requested param at some lead times, and the underlying ecmwf-opendata
    # client's own retry/backoff on that can take minutes per failing
    # step before this function's own try/except ever gets a chance to
    # fall back - defeating the "best-effort, never blocks the forecast"
    # design (seen live: a 5-day navigation forecast blew past a calling
    # agent's 300s tool-call timeout entirely on this fetch). Bounding it
    # with a hard wall-clock timeout, after which we abandon the fetch and
    # fall back to the parametrized estimate immediately, restores that.
    "timeout_seconds": 45,
}

# Ensemble-based probabilistic storm risk (e.g. "60% of members show gale
# force winds"), from ECMWF Open Data's ensemble products - AIFS_ENS_FX
# (ECMWF's AI model) or IFS_ENS_FX (physics-based HRES ensemble), fetched
# directly rather than run locally. Off by default (adds N member fetches
# per request); enable per-request with "ensemble": true.
ENSEMBLE = {
    "enabled_by_default": False,
    "source": "aifs",             # "aifs" or "ifs"
    "members": 10,                 # subset of the full 50-member ensemble
    "total_members": 50,
    "gale_probability_threshold": 0.3,  # flag when >=30% of members show Force 8+
}

# Data source used for initial conditions.
#   "gfs" - NOAA GFS, no credentials required (default).
#   "ifs" - ECMWF open-data operational forecast/analysis, no credentials
#           required either, and generally better skill over European/UK
#           waters than GFS - worth switching to.
#   "cds" - ECMWF ERA5 *reanalysis* via Copernicus (needs ~/.cdsapirc).
#           Only useful for past dates (reanalysis, not a forecast) -
#           for backtesting/validation, not for a live forecast's initial
#           conditions.
# See datasources.py for how the Met Office SST feed is layered on top.
DATA_SOURCE = "gfs"

# ---------------------------------------------------------------------------
# Met Office Marine Data Service: OSTIA sea-surface temperature, layered onto
# the base atmospheric data source so CBottleInfill/CBottleSR get observed
# SST rather than an inferred value.
#
# This is NOT a REST API - the Marine Data Service delivers NetCDF files over
# FTP/SFTP on a fixed daily schedule (OSTIA lands ~0640 UTC). metoffice_sync.py
# mirrors new files into METOFFICE_SST_LOCAL_DIR; datasources.MetOfficeSST
# reads the most recent file from that local cache rather than making a
# live network call per forecast request. Run metoffice_sync.py from a
# systemd timer shortly after the daily delivery window (see
# service/metoffice-sst-sync.timer).
#
# All of these are unset until the Met Office account is provisioned; until
# then MetOfficeSST raises a clear "not configured" error and the rest of
# the pipeline falls back to CBottleInfill's own SST climatology (today's
# behaviour) rather than failing the forecast.
# ---------------------------------------------------------------------------
METOFFICE_SST_PROTOCOL = os.environ.get("METOFFICE_SST_PROTOCOL", "sftp")  # "sftp" or "ftp"
METOFFICE_SST_HOST = os.environ.get("METOFFICE_SST_HOST")
METOFFICE_SST_PORT = int(os.environ.get("METOFFICE_SST_PORT", "22"))
METOFFICE_SST_USERNAME = os.environ.get("METOFFICE_SST_USERNAME")
METOFFICE_SST_PASSWORD = os.environ.get("METOFFICE_SST_PASSWORD")
METOFFICE_SST_KEY_PATH = os.environ.get("METOFFICE_SST_KEY_PATH")  # SSH private key, if using key auth instead of a password
METOFFICE_SST_FTP_TLS = os.environ.get("METOFFICE_SST_FTP_TLS", "true").lower() == "true"  # only used when PROTOCOL=ftp
METOFFICE_SST_REMOTE_DIR = os.environ.get("METOFFICE_SST_REMOTE_DIR", "/")
METOFFICE_SST_FILE_PATTERN = os.environ.get("METOFFICE_SST_FILE_PATTERN", "*OSTIA*.nc")
METOFFICE_SST_LOCAL_DIR = os.environ.get(
    "METOFFICE_SST_LOCAL_DIR", str(Path(__file__).resolve().parent.parent / "data" / "metoffice_sst")
)

# ---------------------------------------------------------------------------
# Storm warnings (Beaufort-scale gale/storm thresholds, Douglas sea-state)
# ---------------------------------------------------------------------------
WARNINGS = {
    "enabled": True,
    "min_beaufort_force": 8,   # Force 8 (Gale) is the UK Shipping Forecast floor
    "wave_warning_m": 4.0,     # "Rough/High seas" advisory threshold
    "storm_tracking": True,    # MSLP-minima low-pressure tracking
    "explosive_cyclogenesis_hpa_24h": 24,  # "weather bomb" threshold
}

# GRIB2 export (wind, gusts, waves, MSLP) alongside the PNG/webp charts, for
# OpenCPN's built-in grib_pi "Open File" dialog or any other GRIB-reading
# navigation software. See gribexport.py. Best-effort: a failure here never
# fails the forecast, it just omits the .grb2 file.
GRIB_EXPORT = {
    "enabled": True,
}

# ---------------------------------------------------------------------------
# OpenCPN REST bridge - pushes gribexport.py's output to OpenCPN running on
# another machine (this project's setup: OpenCPN 5.14.1 on an AGX Thor,
# reached over Tailscale from this GB10) via its built-in REST server and
# grib_pi's GRIB_APPLY_JSON_CONFIG plugin message. Two things had to be
# confirmed against the actual OpenCPN source (1:5.14.1+dfsg, from the
# opencpn/opencpn PPA) rather than its docs/doxygen, which are wrong on both
# counts:
#   - The query parameter is "apikey" (no underscore), not "api_key" as
#     rest_server.h's own doc comment and the public doxygen page say -
#     src/model/src/rest_server.cpp's HandlePluginMsg (and every other
#     handler) reads HttpVarToString(hm->query, "apikey").
#   - The endpoint (opencpn -r --get_rest_endpoint) reports "http://", but
#     it's actually HTTPS with a self-signed cert - verified with curl -k.
#     Default port in this build is 8443, not the docs' example 8000.
#
# GRIB_APPLY_JSON_CONFIG's payload is {"grib_file": "<path>"} - confirmed
# from GRIBUICtrlBar::OpenFileFromJSON in grib_pi/src/grib_ui_dlg.cpp - and
# that path is read via wxFileExists() on OPENCPN'S OWN filesystem, so the
# GRIB2 file has to be copied to the Thor (see THOR_* below) before pushing
# a message that references it - a path on the Spark means nothing there.
#
# Pairing: OpenCPN generates a random 4-digit pincode and shows it in a
# dialog on its own screen (RestServer::CheckApiKey, src/model/src/
# rest_server.cpp) the first time a given "source" string is seen with an
# unrecognized api_key - clicking OK/Cancel makes no difference, the key is
# already stored in OpenCPN's config the moment the dialog appears
# (PINCreateDialog::OnOKClick / OnCancelClick both just Close()). The actual
# api_key isn't the displayed digits - it's a hash of them:
#   - api_key.size() < 10 in the *failing* request that triggered the
#     pincode ("old-style" client) -> CompatHash(): a linear-congruential
#     step (a=48271, m=2**64-1) seeded with the pincode's integer value,
#     formatted as uppercase hex, no padding (pincode.cpp: CompatHash()).
#   - api_key.size() >= 10 ("new-style") -> Hash(): first 12 hex chars of
#     sha256(zero-padded 4-digit pincode string) (pincode.cpp: Hash()).
# opencpn_bridge.compat_hash()/sha256_hash() implement both; pair_thor.py
# automates working out which one a given OpenCPN build wants and confirming
# it, so this dance only has to happen once per Thor/source pairing - the
# resulting api_key is config below (persists in OpenCPN's own config too,
# survives its restarts).
# ---------------------------------------------------------------------------
OPENCPN_REST_URL = os.environ.get("OPENCPN_REST_URL", "https://opencpn-host.example:8443")
OPENCPN_API_KEY = os.environ.get("OPENCPN_API_KEY")
OPENCPN_SOURCE = os.environ.get("OPENCPN_SOURCE", "earth2-forecast-agent")
OPENCPN_VERIFY_SSL = os.environ.get("OPENCPN_VERIFY_SSL", "false").lower() == "true"  # self-signed cert

# SSH access to the machine actually running OpenCPN, for copying the GRIB2
# file there before the REST push (see above - the REST API can't take file
# content directly for this endpoint, only a path OpenCPN itself can read).
# Set these three for your own setup - example values below assume a
# Tailscale (or any VPN mesh) hostname; a plain LAN IP/hostname works too.
THOR_SSH_HOST = os.environ.get("THOR_SSH_HOST", "opencpn-host.example")
THOR_SSH_USER = os.environ.get("THOR_SSH_USER", "opencpn")
THOR_SSH_KEY_PATH = os.environ.get("THOR_SSH_KEY_PATH", str(Path.home() / ".ssh" / "opencpn_bridge"))
THOR_GRIB_INBOX_DIR = os.environ.get("THOR_GRIB_INBOX_DIR", "/home/opencpn/grib_inbox")

# ---------------------------------------------------------------------------
# Live AIS feed (aisstream.io) - ais_ingest.py + ais_mcp_server.py.
#
# Deliberately a SEPARATE aisstream.io API key/connection from the Thor's
# existing aisstream_to_opencpn.py -> cpa_service.py stack (own-vessel
# collision avoidance in the Solent only, in-memory targets with a 5-minute
# TTL, no history). That stack's own commit history documents a real
# lesson: two consumers sharing one API key fought over its single
# connection slot and both starved during an outage. This solves a
# different problem - multi-region historical vessel intelligence for
# Hermes - not an extension of that one, so it gets its own key.
# ---------------------------------------------------------------------------
AIS_API_KEY = os.environ.get("AIS_API_KEY")
AIS_WS_URL = "wss://stream.aisstream.io/v0/stream"
AIS_DB_PATH = os.environ.get("AIS_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "ais.db"))

# Store at most one position row per vessel per this many seconds, even if
# aisstream delivers updates far more often - keeps the database's growth
# manageable without losing meaningful track resolution.
AIS_POSITION_THROTTLE_SECONDS = int(os.environ.get("AIS_POSITION_THROTTLE_SECONDS", "120"))
# Prune position rows older than this many days.
AIS_RETENTION_DAYS = int(os.environ.get("AIS_RETENTION_DAYS", "60"))
# A vessel is flagged as having "gone dark" when the gap between two
# consecutive stored positions exceeds this.
AIS_GAP_THRESHOLD_HOURS = float(os.environ.get("AIS_GAP_THRESHOLD_HOURS", "2.0"))

# A Strait of Hormuz box was tried here and dropped after real testing:
# aisstream.io's free tier showed zero vessel coverage there after 20,500+
# messages processed (dense real coverage everywhere else) - almost
# certainly no community AIS receivers on the Iranian side, and sparse
# coverage on the UAE/Oman side, unlike Europe's dense hobbyist network.
# Worth retrying if this project ever moves to a paid tier with satellite
# AIS. Any future arbitrary watch area (Bab-el-Mandeb, Bosphorus, Gulf of
# Aden) that isn't part of the UK Shipping Forecast domain regions.py
# models belongs as its own small constant here, same convention as
# SeaArea.bounds - not forced into that UK-specific SeaArea registry.


def _ais_bounding_boxes():
    """aisstream.io subscription boxes: one merged box per regions.py basin
    (north_atlantic/north_sea/english_channel/baltic - so storm-risk
    correlation has vessels to find wherever we can actually issue a
    warning). Merged per-basin rather than one box per sea area (43 of
    them, many adjacent/overlapping) to keep the subscription small;
    aisstream.io takes the union of whatever boxes it's given, so a larger
    merged box just means broader coverage, not an error - the extra
    "empty ocean" included is fine.
    """
    import regions

    basins = {}
    for area in regions.ALL_AREAS.values():
        lat_s, lon_w, lat_n, lon_e = area.bounds
        b = basins.setdefault(area.basin, [lat_s, lon_w, lat_n, lon_e])
        b[0] = min(b[0], lat_s)
        b[1] = min(b[1], lon_w)
        b[2] = max(b[2], lat_n)
        b[3] = max(b[3], lon_e)

    return [[[b[0], b[1]], [b[2], b[3]]] for b in basins.values()]


AIS_BOUNDING_BOXES = _ais_bounding_boxes()

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
