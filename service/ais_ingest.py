"""
ais_ingest.py - live AIS ingestion into SQLite (ais_db.py), via a
dedicated aisstream.io WebSocket subscription (config.AIS_API_KEY /
AIS_BOUNDING_BOXES - see config.py's AIS_* comment block for why this is
a separate key/connection from the Thor's existing AIS stack, not an
extension of it).

Connection handling (idle-timeout detection, exponential backoff,
reconnect) mirrors the AGX Thor's (Chart Plotter) aisstream_to_opencpn.py,
Problem a real aisstream.io outage (zombie connections the WebSocket ping/pong
couldn't detect because the server kept answering pings while delivering zero data frames, and a
reconnect storm on instant retry earning extended rate-limiting errors reusing
that proven code the same way.

Usage:
    AIS_API_KEY=... python3 ais_ingest.py
"""

import asyncio
import json
import logging
import sys
import time
from typing import Optional

import websockets

import ais_db
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

IDLE_TIMEOUT_S = 180.0
BACKOFF_MIN_S = 5.0
BACKOFF_MAX_S = 300.0
PRUNE_INTERVAL_S = 6 * 3600

# mmsi -> last-stored epoch time, for config.AIS_POSITION_THROTTLE_SECONDS
# throttling. In-memory and reset on restart - worst case a slightly
# denser-than-usual burst of stored positions right after a restart, not
# worth persisting just to avoid that.
_last_stored: dict = {}


def _clean(raw) -> Optional[str]:
    """AIS text fields pad with '@'; strip padding + whitespace. Empty
    after stripping -> None, so it doesn't overwrite a previously known
    good value with an empty string."""
    s = str(raw or "").replace("@", " ").strip()
    return s or None


def _handle_position(conn, msg_type: str, message: dict) -> None:
    if msg_type == "PositionReport":
        pr = message.get("PositionReport", {})
        nav_status = pr.get("NavigationalStatus")
    elif msg_type == "StandardClassBPositionReport":
        pr = message.get("StandardClassBPositionReport", {})
        nav_status = None
    else:
        return

    mmsi = pr.get("UserID")
    lat, lon = pr.get("Latitude"), pr.get("Longitude")
    if mmsi is None or lat is None or lon is None:
        return
    if lat == 0.0 and lon == 0.0:
        return
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return

    now = time.time()
    if now - _last_stored.get(mmsi, 0.0) < config.AIS_POSITION_THROTTLE_SECONDS:
        return
    _last_stored[mmsi] = now

    ais_db.insert_position(conn, mmsi, now, lat, lon, pr.get("Cog"), pr.get("Sog"), nav_status)


def _handle_static(conn, msg_type: str, message: dict) -> None:
    now = time.time()

    if msg_type == "ShipStaticData":
        r = message.get("ShipStaticData", {})
        mmsi = r.get("UserID")
        if mmsi is None:
            return
        dim = r.get("Dimension", {})
        candidate = {
            "name": _clean(r.get("Name")),
            "callsign": _clean(r.get("CallSign")),
            "ship_type": r.get("Type"),
            "to_bow": dim.get("A"), "to_stern": dim.get("B"),
            "to_port": dim.get("C"), "to_starboard": dim.get("D"),
            "destination": _clean(r.get("Destination")),
        }
    elif msg_type == "StaticDataReport":
        r = message.get("StaticDataReport", {})
        mmsi = r.get("UserID")
        if mmsi is None:
            return
        part_a = r.get("ReportA", {}) or {}
        part_b = r.get("ReportB", {}) or {}
        dim = part_b.get("Dimension", {}) or {}
        candidate = {
            "name": _clean(part_a.get("Name")) if part_a.get("Valid") else None,
            "callsign": _clean(part_b.get("CallSign")) if part_b.get("Valid") else None,
            "ship_type": part_b.get("ShipType") if part_b.get("Valid") else None,
            "to_bow": dim.get("A") if part_b.get("Valid") else None,
            "to_stern": dim.get("B") if part_b.get("Valid") else None,
            "to_port": dim.get("C") if part_b.get("Valid") else None,
            "to_starboard": dim.get("D") if part_b.get("Valid") else None,
        }
    else:
        return

    # Only pass fields we actually have - upsert_static sets whatever key
    # it's given, so a None here would silently clobber a previously
    # known-good value (e.g. a StaticDataReport ReportA-only fragment
    # carries a name but no dimensions).
    fields = {k: v for k, v in candidate.items() if v is not None}
    if fields:
        ais_db.upsert_static(conn, mmsi, now, **fields)


def _handle_message(conn, msg_type: str, message: dict) -> None:
    try:
        if msg_type in ("PositionReport", "StandardClassBPositionReport"):
            _handle_position(conn, msg_type, message)
        elif msg_type in ("ShipStaticData", "StaticDataReport"):
            _handle_static(conn, msg_type, message)
    except Exception as e:
        logger.warning(f"Failed to handle {msg_type}: {e}")


async def _prune_loop(conn) -> None:
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_S)
        try:
            n = ais_db.prune_old_positions(conn)
            conn.commit()
            if n:
                logger.info(f"Pruned {n} position rows older than {config.AIS_RETENTION_DAYS} days")
        except Exception as e:
            logger.warning(f"Prune failed: {e}")


async def _run() -> None:
    if not config.AIS_API_KEY:
        logger.error("AIS_API_KEY not set - get a free key at https://aisstream.io")
        sys.exit(1)

    conn = ais_db.get_connection()
    ais_db.init_db(conn)
    logger.info(f"AIS database: {config.AIS_DB_PATH}")
    logger.info(f"Bounding boxes: {config.AIS_BOUNDING_BOXES}")

    asyncio.create_task(_prune_loop(conn))

    subscribe_msg = {"APIKey": config.AIS_API_KEY, "BoundingBoxes": config.AIS_BOUNDING_BOXES}
    count = 0
    commit_counter = 0
    backoff = BACKOFF_MIN_S

    while True:
        got_frames = False
        try:
            logger.info(f"Connecting to {config.AIS_WS_URL} ...")
            async with websockets.connect(config.AIS_WS_URL) as ws:
                await ws.send(json.dumps(subscribe_msg))
                logger.info("Subscribed - receiving vessel data")
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        logger.warning(f"No data for {IDLE_TIMEOUT_S:.0f}s - dropping stale connection")
                        break
                    got_frames = True
                    try:
                        msg = json.loads(raw)
                        _handle_message(conn, msg.get("MessageType", ""), msg.get("Message", {}))
                        count += 1
                        commit_counter += 1
                        if commit_counter >= 20:
                            conn.commit()
                            commit_counter = 0
                        if count % 500 == 0:
                            logger.info(f"{count} messages processed")
                    except Exception as e:
                        logger.warning(f"Error processing message: {e}")
        except Exception as e:
            logger.warning(f"Connection lost: {e}")
        finally:
            conn.commit()

        if got_frames:
            backoff = BACKOFF_MIN_S
        logger.info(f"Reconnecting in {backoff:.0f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, BACKOFF_MAX_S)


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
