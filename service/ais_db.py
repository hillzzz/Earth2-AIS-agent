"""
SQLite persistence for the live AIS feed (see config.py's AIS_* settings
for why this is a separate feed/key from the Thor's existing AIS stack).

ais_ingest.py writes; ais_mcp_server.py (and this module's own query
helpers) read. Schema is deliberately small: a throttled position history
per vessel, and the latest known static (name/type/dimensions) data per
vessel - everything else (gaps, area queries, "supertanker" filtering) is
computed on read rather than maintained as derived tables, so it's always
correct against whatever's actually stored.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import config

# ITU-R M.1371 AIS ship type codes, broad categories only (the ones this
# project's tools actually filter on) - not exhaustive.
SHIP_TYPE_LABELS = {
    30: "Fishing", 31: "Towing", 32: "Towing (large)", 33: "Dredging",
    34: "Diving ops", 35: "Military", 36: "Sailing", 37: "Pleasure craft",
    40: "High-speed craft", 50: "Pilot vessel", 51: "Search and rescue",
    52: "Tug", 53: "Port tender", 55: "Law enforcement", 58: "Medical",
    60: "Passenger", 61: "Passenger", 62: "Passenger", 63: "Passenger",
    64: "Passenger", 65: "Passenger", 66: "Passenger", 67: "Passenger",
    68: "Passenger", 69: "Passenger",
    70: "Cargo", 71: "Cargo", 72: "Cargo", 73: "Cargo", 74: "Cargo",
    75: "Cargo", 76: "Cargo", 77: "Cargo", 78: "Cargo", 79: "Cargo",
    80: "Tanker", 81: "Tanker", 82: "Tanker", 83: "Tanker", 84: "Tanker",
    85: "Tanker", 86: "Tanker", 87: "Tanker", 88: "Tanker", 89: "Tanker",
}

NAV_STATUS_LABELS = {
    0: "Underway (engine)", 1: "At anchor", 2: "Not under command",
    3: "Restricted manoeuvrability", 4: "Constrained by draught",
    5: "Moored", 6: "Aground", 7: "Engaged in fishing",
    8: "Underway (sailing)", 15: "Not defined",
}


def get_connection() -> sqlite3.Connection:
    Path(config.AIS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.AIS_DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL: ais_ingest.py (writer) and ais_mcp_server.py (reader, a fresh
    # subprocess per MCP call) run concurrently against the same file.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    owns_conn = conn is None
    conn = conn or get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS vessel_positions (
            mmsi INTEGER NOT NULL,
            ts REAL NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            cog REAL,
            sog REAL,
            nav_status INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_positions_mmsi_ts ON vessel_positions(mmsi, ts);
        CREATE INDEX IF NOT EXISTS idx_positions_ts ON vessel_positions(ts);

        CREATE TABLE IF NOT EXISTS vessel_static (
            mmsi INTEGER PRIMARY KEY,
            name TEXT,
            callsign TEXT,
            ship_type INTEGER,
            to_bow REAL,
            to_stern REAL,
            to_port REAL,
            to_starboard REAL,
            destination TEXT,
            updated_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_static_name ON vessel_static(name);
        """
    )
    conn.commit()
    if owns_conn:
        conn.close()


def insert_position(conn: sqlite3.Connection, mmsi: int, ts: float, lat: float, lon: float,
                     cog: Optional[float], sog: Optional[float], nav_status: Optional[int]) -> None:
    conn.execute(
        "INSERT INTO vessel_positions (mmsi, ts, lat, lon, cog, sog, nav_status) VALUES (?,?,?,?,?,?,?)",
        (mmsi, ts, lat, lon, cog, sog, nav_status),
    )


def upsert_static(conn: sqlite3.Connection, mmsi: int, ts: float, **fields) -> None:
    """fields: any of name/callsign/ship_type/to_bow/to_stern/to_port/
    to_starboard/destination. Only provided fields are updated - a static
    report often carries just a subset (e.g. type 24 part A is name-only)."""
    existing = conn.execute("SELECT mmsi FROM vessel_static WHERE mmsi = ?", (mmsi,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO vessel_static (mmsi, updated_at) VALUES (?, ?)", (mmsi, ts)
        )
    if fields:
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE vessel_static SET {set_clause}, updated_at = ? WHERE mmsi = ?",
            (*fields.values(), ts, mmsi),
        )


def prune_old_positions(conn: sqlite3.Connection, retention_days: Optional[int] = None) -> int:
    cutoff = time.time() - (retention_days or config.AIS_RETENTION_DAYS) * 86400
    cur = conn.execute("DELETE FROM vessel_positions WHERE ts < ?", (cutoff,))
    return cur.rowcount


# ---------------------------------------------------------------------------
# Query helpers (used by ais_mcp_server.py)
# ---------------------------------------------------------------------------

def _row_to_vessel(static_row: Optional[sqlite3.Row], pos_row: Optional[sqlite3.Row]) -> Dict:
    now = time.time()
    out: Dict = {
        "mmsi": (static_row["mmsi"] if static_row else pos_row["mmsi"]),
        "name": (static_row["name"] if static_row and static_row["name"] else None),
        "callsign": (static_row["callsign"] if static_row else None),
        "ship_type": SHIP_TYPE_LABELS.get((static_row["ship_type"] if static_row else None), "Unknown"),
        "destination": (static_row["destination"] if static_row else None),
    }
    if static_row and static_row["to_bow"] is not None and static_row["to_stern"] is not None:
        out["length_m"] = round(static_row["to_bow"] + static_row["to_stern"], 1)
    else:
        out["length_m"] = None
    if pos_row is not None:
        out.update({
            "lat": round(pos_row["lat"], 5),
            "lon": round(pos_row["lon"], 5),
            "cog_deg": pos_row["cog"],
            "sog_kts": pos_row["sog"],
            "nav_status": NAV_STATUS_LABELS.get(pos_row["nav_status"], "Unknown"),
            "position_age_minutes": round((now - pos_row["ts"]) / 60.0, 1),
        })
    return out


def find_vessel(identifier: str) -> Optional[Dict]:
    """Look up by MMSI (exact) or name (case-insensitive substring, most
    recently updated match wins)."""
    conn = get_connection()
    try:
        identifier = str(identifier).strip()
        if identifier.isdigit():
            static_row = conn.execute(
                "SELECT * FROM vessel_static WHERE mmsi = ?", (int(identifier),)
            ).fetchone()
            mmsi = int(identifier)
        else:
            static_row = conn.execute(
                "SELECT * FROM vessel_static WHERE name LIKE ? ORDER BY updated_at DESC LIMIT 1",
                (f"%{identifier}%",),
            ).fetchone()
            if static_row is None:
                return None
            mmsi = static_row["mmsi"]

        pos_row = conn.execute(
            "SELECT * FROM vessel_positions WHERE mmsi = ? ORDER BY ts DESC LIMIT 1", (mmsi,)
        ).fetchone()
        if static_row is None and pos_row is None:
            return None
        return _row_to_vessel(static_row, pos_row)
    finally:
        conn.close()


def vessels_in_bbox(lat_s: float, lon_w: float, lat_n: float, lon_e: float,
                     ship_type_prefix: Optional[str] = None, min_length_m: Optional[float] = None,
                     max_age_hours: float = 6.0) -> List[Dict]:
    """Vessels whose LATEST known position falls in the box and is fresher
    than max_age_hours. ship_type_prefix: e.g. "Tanker"/"Cargo"/"Fishing"
    (matches SHIP_TYPE_LABELS values). min_length_m: from AIS dimension
    fields (to_bow + to_stern) - the "supertanker" filter (Hormuz use
    case) is ship_type_prefix="Tanker", min_length_m~250, which is an
    approximation: AIS ship-type codes distinguish Tanker broadly, not
    VLCC/ULCC specifically."""
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_hours * 3600
        rows = conn.execute(
            """
            SELECT p.* FROM vessel_positions p
            JOIN (SELECT mmsi, MAX(ts) AS max_ts FROM vessel_positions GROUP BY mmsi) latest
              ON p.mmsi = latest.mmsi AND p.ts = latest.max_ts
            WHERE p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ? AND p.ts >= ?
            """,
            (lat_s, lat_n, lon_w, lon_e, cutoff),
        ).fetchall()

        results = []
        for pos_row in rows:
            static_row = conn.execute(
                "SELECT * FROM vessel_static WHERE mmsi = ?", (pos_row["mmsi"],)
            ).fetchone()
            vessel = _row_to_vessel(static_row, pos_row)
            if ship_type_prefix and vessel["ship_type"] != ship_type_prefix:
                continue
            if min_length_m and (vessel["length_m"] is None or vessel["length_m"] < min_length_m):
                continue
            results.append(vessel)
        results.sort(key=lambda v: v.get("position_age_minutes", 0))
        return results
    finally:
        conn.close()


def find_ais_gaps(lat_s: float, lon_w: float, lat_n: float, lon_e: float,
                   min_gap_hours: Optional[float] = None) -> List[Dict]:
    """Vessels with an unusually long silence whose last position before it
    fell inside the box - the "went dark near X" case. Covers two distinct
    situations, both worth surfacing separately:
      - still_dark=True: no position since (the more urgent case - a
        vessel that may currently be dark near the box, right now).
      - still_dark=False: the vessel reappeared; reappeared_at shows
        where/when, which by itself can be a signal (reappearing far from
        where it went dark, given the elapsed time, is the "suspicious"
        pattern the Baltic-cables use case is actually after).
    Computed on read (LAG/LEAD window functions) rather than a maintained
    table, so it's always correct against whatever's actually stored.
    """
    conn = get_connection()
    try:
        threshold_s = (min_gap_hours or config.AIS_GAP_THRESHOLD_HOURS) * 3600
        now = time.time()
        rows = conn.execute(
            """
            SELECT mmsi, ts, lat, lon, cog, sog, nav_status,
                   LEAD(ts) OVER (PARTITION BY mmsi ORDER BY ts) AS next_ts,
                   LEAD(lat) OVER (PARTITION BY mmsi ORDER BY ts) AS next_lat,
                   LEAD(lon) OVER (PARTITION BY mmsi ORDER BY ts) AS next_lon
            FROM vessel_positions
            """
        ).fetchall()

        results = []
        for row in rows:
            gap_end = row["next_ts"] if row["next_ts"] is not None else now
            if (gap_end - row["ts"]) < threshold_s:
                continue
            if not (lat_s <= row["lat"] <= lat_n and lon_w <= row["lon"] <= lon_e):
                continue

            static_row = conn.execute(
                "SELECT * FROM vessel_static WHERE mmsi = ?", (row["mmsi"],)
            ).fetchone()
            vessel = _row_to_vessel(static_row, row)
            vessel["last_seen_before_gap"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(row["ts"]))
            vessel["gap_hours"] = round((gap_end - row["ts"]) / 3600.0, 1)
            vessel["still_dark"] = row["next_ts"] is None
            if row["next_ts"] is not None:
                vessel["reappeared_at"] = {
                    "lat": round(row["next_lat"], 5),
                    "lon": round(row["next_lon"], 5),
                    "time": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(row["next_ts"])),
                }
            results.append(vessel)

        results.sort(key=lambda v: (not v["still_dark"], -v["gap_hours"]))
        return results
    finally:
        conn.close()
