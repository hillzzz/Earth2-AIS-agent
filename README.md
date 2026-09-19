# earth2-maritime-agent

MCP tools for a long-running maritime intelligence agent: real storm/gale
forecasts from [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio),
a live global AIS feed with historical vessel-gap detection, and a bridge
that pushes forecasts straight into a running [OpenCPN](https://opencpn.org)
instance as a GRIB2 overlay - all exposed as tools for
[Hermes](https://github.com/NousResearch) or any other MCP-speaking
tool-calling agent.

This is the agent-facing layer only (see "What's not here" below) - a
focused set of MCP tools an agent can actually reason with, not a general
weather/AIS dashboard.

## What it does

**Forecasts** (`mcp_server.py`, server name `earth2-forecast`)
- `forecast_generate(location, forecast_type)` - a real Earth2Studio
  forecast (FCN, with StormScopeMeteosatEU attempted first for the 0-6h
  nowcast tier) for any of the 31 official Met Office Shipping Forecast sea
  areas, a UK port, or a Baltic sea area. Returns gale/storm warnings and a
  day-by-day breakdown (max Beaufort force, max wave height, worst timing),
  scoped to the actual named area - not a wide chart-context box that could
  attribute a storm 400 miles away to the wrong sea area (a real bug this
  project hit and fixed; see `forecast_engine._local_warning_radius_miles`).
- `opencpn_push(grib_path)` / `opencpn_status()` - push that forecast into a
  running OpenCPN instance's chart display automatically, via OpenCPN's own
  REST server and `grib_pi`'s plugin-message API (`opencpn_bridge.py`) - no
  GUI automation, no custom OpenCPN plugin.

**AIS** (`ais_mcp_server.py`, server name `ais-tracker`), backed by
`ais_ingest.py` - a standalone, continuously-running process that persists a
throttled position history + latest static data per vessel to SQLite:
- `vessel_lookup(name_or_mmsi)` - find a specific vessel.
- `vessels_near(location, radius_nm, ship_type, min_length_m)` - area search,
  with a size-based filter approximating "supertanker" (AIS ship-type codes
  distinguish Tanker broadly, not VLCC/ULCC specifically).
- `vessels_at_storm_risk(location, forecast_type)` - runs a real forecast
  and, if it carries a gale warning, lists vessels currently in that sea
  area.
- `check_ais_gaps(location, min_gap_hours, max_distance_from_cable_nm)` -
  vessels with an unusually long AIS silence, optionally filtered to gaps
  that began near a real submarine cable route (`fetch_cable_data.py`,
  Baltic by default) - flags both still-dark vessels and ones that
  reappeared elsewhere.

## What's not here

The original project this was extracted from also has a FastAPI HTTP
service, chart/animation rendering, and an Open WebUI tool plugin - all
deliberately left out. This repo is the part an *agent* calls directly; a
human-facing dashboard is a different, separable concern.

## Setup

```bash
uv sync
```

Pulls `earth2studio` pinned to `0.18.0` with the model/diagnostic extras
this project uses (`fcn`, `fcn3`, `graphcast`, `aurora`, `dlwp`, `dlesym`,
`cbottle`, `windgust-afno`, `stormscope`, `data`, `perturbation`). `natten`
and `torch-harmonics` build CUDA kernels from source on first sync - expect
10-20+ minutes. See the note at the bottom of `pyproject.toml` if you hit a
torch/CUDA version mismatch (common on non-x86_64 or unusual CUDA setups).

No credentials are needed for weather forecasts themselves (GFS, NOAA's
public data, no API key). Two things do need setup:

**AIS** - a free key from [aisstream.io](https://aisstream.io). Copy
`service/.env.example` to `service/.env` and set `AIS_API_KEY`. Use your own
key - don't share a connection with another AIS consumer you might already
run; aisstream.io allows one connection per key, and two consumers sharing
one will both starve, silently, during any hiccup (this happened for real
during development of the project this was extracted from - see
`config.py`'s `AIS_*` comment block).

**OpenCPN push** - needs OpenCPN 5.9+ (confirmed working against 5.14.1)
running with its REST server enabled, reachable from wherever you run
`ais_ingest.py`/the MCP servers (same machine, LAN, or a VPN mesh like
Tailscale - see `config.py`'s `THOR_SSH_*`/`OPENCPN_*` settings, named after
the original dev setup's second machine). Pairing is a one-time interactive
step: OpenCPN shows a pincode dialog the first time an unrecognized client
connects, and the *displayed digits are not the actual key* - it's a hash of
them, computed one of two ways depending on a subtlety in OpenCPN's own
pairing protocol. `pair_thor.py` automates this:

```bash
python3 service/pair_thor.py
```

follow the prompts, then put the resulting key in `service/.env` as
`OPENCPN_API_KEY`. See `config.py`'s `OPENCPN_*` comment block for exactly
what this does and how it was reverse-engineered from OpenCPN's actual
source (its public API docs are wrong about both the endpoint scheme and
the auth query parameter name).

**Cable data for `check_ais_gaps`** (optional - it works without this, just
without cable-proximity filtering):

```bash
python3 service/fetch_cable_data.py
```

Pulls real submarine cable routes from submarinecablemap.com's public API
(the same data the interactive map itself serves any visitor - not
TeleGeography's own paid-licensed dataset, so treat it as an approximation
good enough for a proximity heuristic, not anything safety-critical or
commercial) and clips it to the Baltic. Not committed to git; re-run to
refresh.

### Running the AIS ingestion service

`ais_ingest.py` needs to run continuously (it's what actually populates the
database the AIS tools query) - it's not something an MCP tool spawns
on-demand. A systemd user unit is the straightforward way to keep it running
across reboots/logouts:

```ini
# ~/.config/systemd/user/ais-ingest.service
[Unit]
Description=AIS ingestion for the maritime agent
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart=/path/to/earth2-maritime-agent/.venv/bin/python /path/to/earth2-maritime-agent/service/ais_ingest.py
WorkingDirectory=/path/to/earth2-maritime-agent/service
EnvironmentFile=/path/to/earth2-maritime-agent/service/.env
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ais-ingest.service
loginctl enable-linger $USER   # so it keeps running after you log out
```

### Registering with Hermes

```bash
hermes mcp add earth2-forecast \
  --command /path/to/earth2-maritime-agent/.venv/bin/python \
  --env OPENCPN_API_KEY=<your key> \
  --args /path/to/earth2-maritime-agent/service/mcp_server.py

hermes mcp add ais-tracker \
  --command /path/to/earth2-maritime-agent/.venv/bin/python \
  --args /path/to/earth2-maritime-agent/service/ais_mcp_server.py
```

`--env` must come *before* `--args` - `--args` greedily consumes every
token after it (it's documented as needing to be the last option), so
anything placed after it silently becomes an extra script argument instead
of an actual environment variable. A real, previously-hit bug, not a
hypothetical one - the tools quietly ran with no credentials at all until
this was caught by checking the actual subprocess environment
(`/proc/<pid>/environ`), not just the CLI's own success message.

## Known gaps

- **Nowcast tier** always falls back from StormScopeMeteosatEU to FCN: that
  model needs consecutive Meteosat satellite frames at 10-minute resolution
  (a EUMETSAT account, and a dedicated image-frame orchestration path), not
  the generic GFS-driven pipeline the other tiers use. Kept as the primary
  model so that work benefits immediately once done.
- **Precipitation** is always 0 - FCN doesn't output `tp`.
- **Real wave/gust overlay** (ECMWF Open Data) silently falls back to a
  parametrized estimate on rate-limiting, missing params at some lead times,
  or a tier/timestep misalignment - by design (never block a forecast on
  this), bounded to 45s so a flaky fetch can't blow past an agent's own
  tool-call timeout either.
- **AIS coverage** is free-tier aisstream.io: dense in European/UK waters,
  essentially zero in the Strait of Hormuz as of testing (almost certainly
  no community receivers on the Iranian side) - a paid/satellite-AIS tier
  would likely fix this, not evaluated here.
- **Cable proximity data** is a community-sourced approximation (see
  above), Baltic-only by default.
- `superres.py` (CBottleInfill -> CBottleSR harbor-scale super-resolution)
  exists but isn't wired into the live forecast path yet.

## License

All rights reserved - see `LICENSE`. This is not open-source software;
contact the copyright holder for permission to use it.
