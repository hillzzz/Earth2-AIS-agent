"""
Maritime region registry.

Generalizes location handling beyond the original Solent-only defaults to
cover the UK Shipping Forecast sea areas across the North Atlantic approaches
and North Sea, plus the Baltic Sea basins. Used by app.py (location parsing),
forecast_engine.py (default forecast windows) and visualize.py (map extents
and harbor/port labels).

Sea area boundaries are approximate (a bounding box, not the official
Met Office polygon) - good enough for picking a forecast window and map
extent, not for legal/regulatory boundary purposes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Port:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class SeaArea:
    name: str
    basin: str  # "north_atlantic" | "north_sea" | "english_channel" | "baltic"
    # (lat_south, lon_west, lat_north, lon_east)
    bounds: Tuple[float, float, float, float]
    center: Tuple[float, float]
    default_radius_miles: int = 120
    ports: Tuple[Port, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# North Atlantic approaches to the UK (Met Office Shipping Forecast areas)
# ---------------------------------------------------------------------------
NORTH_ATLANTIC_AREAS: Dict[str, SeaArea] = {
    "rockall": SeaArea(
        "Rockall", "north_atlantic", (54, -18, 58, -10), (56.0, -14.0), 150,
        (Port("Rockall", 57.60, -13.69),),
    ),
    "malin": SeaArea(
        "Malin", "north_atlantic", (54, -11, 57, -6), (55.5, -8.5), 120,
        (Port("Malin Head", 55.37, -7.34), Port("Tory Island", 55.27, -8.23)),
    ),
    "hebrides": SeaArea(
        "Hebrides", "north_atlantic", (56, -10, 59, -5), (57.5, -7.5), 120,
        (Port("Stornoway", 58.21, -6.39), Port("Castlebay", 56.97, -7.49)),
    ),
    "bailey": SeaArea(
        "Bailey", "north_atlantic", (57, -15, 61, -8), (59.0, -11.5), 150,
        (Port("Bailey Bank", 59.0, -11.5),),
    ),
    "shannon": SeaArea(
        "Shannon", "north_atlantic", (51, -12, 54, -7), (52.5, -9.5), 120,
        (Port("Shannon Estuary", 52.61, -9.44), Port("Foynes", 52.61, -9.11)),
    ),
    "sole": SeaArea(
        "Sole", "north_atlantic", (48, -12, 51, -6), (49.5, -9.0), 150,
        (Port("Scilly", 49.92, -6.32),),
    ),
    "fastnet": SeaArea(
        "Fastnet", "north_atlantic", (50, -11, 52, -7), (51.4, -9.6), 100,
        (Port("Fastnet Rock", 51.38, -9.60), Port("Cork", 51.90, -8.47)),
    ),
    "lundy": SeaArea(
        "Lundy", "north_atlantic", (50, -6, 52, -3), (51.2, -4.7), 80,
        (Port("Bristol", 51.45, -2.60), Port("Swansea", 51.62, -3.94)),
    ),
    "irish_sea": SeaArea(
        "Irish Sea", "north_atlantic", (52, -6.5, 55, -3), (53.5, -4.8), 100,
        (Port("Liverpool", 53.41, -3.00), Port("Holyhead", 53.31, -4.63), Port("Douglas", 54.15, -4.48)),
    ),
    "biscay": SeaArea(
        "Biscay", "north_atlantic", (43.5, -10, 48.5, -1), (46.0, -6.0), 180,
        (Port("Bordeaux", 44.84, -0.58), Port("La Rochelle", 46.16, -1.15)),
    ),
    "trafalgar": SeaArea(
        "Trafalgar", "north_atlantic", (35, -9, 37.5, -5), (36.0, -7.5), 100,
        (Port("Cadiz", 36.53, -6.30), Port("Gibraltar", 36.14, -5.35)),
    ),
    "fitzroy": SeaArea(
        "FitzRoy", "north_atlantic", (43.5, -15, 48.5, -9), (45.5, -10.0), 180,
        (Port("A Coruna", 43.37, -8.40), Port("Vigo", 42.24, -8.72)),
    ),
    "fair_isle": SeaArea(
        "Fair Isle", "north_atlantic", (58.5, -4, 61, 1), (59.5, -1.5), 90,
        (Port("Fair Isle", 59.53, -1.63), Port("Lerwick", 60.15, -1.15)),
    ),
    "faeroes": SeaArea(
        "Faeroes", "north_atlantic", (60, -10, 64, -3), (62.0, -6.5), 150,
        (Port("Torshavn", 62.01, -6.77),),
    ),
    "south_east_iceland": SeaArea(
        "South East Iceland", "north_atlantic", (62, -20, 66, -10), (63.5, -15.0), 200,
        (Port("Hofn", 64.25, -15.21), Port("Vestmannaeyjar", 63.44, -20.27)),
    ),
}

# ---------------------------------------------------------------------------
# North Sea (Met Office Shipping Forecast areas)
# ---------------------------------------------------------------------------
NORTH_SEA_AREAS: Dict[str, SeaArea] = {
    "viking": SeaArea(
        "Viking", "north_sea", (59, -1, 62, 4), (60.5, 1.5), 150,
        (Port("Lerwick", 60.15, -1.15),),
    ),
    "forties": SeaArea(
        "Forties", "north_sea", (57, -2, 60, 2), (58.5, 0.0), 120,
        (Port("Aberdeen", 57.15, -2.10), Port("Peterhead", 57.51, -1.78)),
    ),
    "cromarty": SeaArea(
        "Cromarty", "north_sea", (57, -4, 59, -1), (58.0, -2.5), 90,
        (Port("Invergordon", 57.69, -4.17), Port("Wick", 58.44, -3.09)),
    ),
    "forth": SeaArea(
        "Forth", "north_sea", (55.5, -3.5, 57, 0), (56.2, -2.0), 90,
        (Port("Edinburgh / Leith", 55.98, -3.17), Port("Dundee", 56.46, -2.97)),
    ),
    "tyne": SeaArea(
        "Tyne", "north_sea", (54, -2, 56, 1), (55.2, -0.9), 90,
        (Port("Newcastle", 54.97, -1.61), Port("Sunderland", 54.91, -1.38)),
    ),
    "dogger": SeaArea(
        "Dogger", "north_sea", (54, 1, 56, 4), (55.0, 2.5), 120,
        (Port("Dogger Bank", 54.75, 2.75),),
    ),
    "german_bight": SeaArea(
        "German Bight", "north_sea", (53.5, 3, 56, 9), (54.7, 6.5), 130,
        (Port("Bremerhaven", 53.54, 8.58), Port("Cuxhaven", 53.87, 8.70), Port("Esbjerg", 55.47, 8.45)),
    ),
    "humber": SeaArea(
        "Humber", "north_sea", (52.5, -1, 54, 2), (53.4, 0.5), 90,
        (Port("Hull", 53.74, -0.34), Port("Grimsby", 53.57, -0.08)),
    ),
    "thames": SeaArea(
        "Thames", "north_sea", (51, 0.5, 53, 3), (52.0, 1.8), 80,
        (Port("Felixstowe", 51.96, 1.35), Port("Harwich", 51.95, 1.29)),
    ),
    "north_utsire": SeaArea(
        "North Utsire", "north_sea", (59.5, 2, 61.5, 6), (60.3, 3.5), 100,
        (Port("Utsira", 59.31, 4.88),),
    ),
    "south_utsire": SeaArea(
        "South Utsire", "north_sea", (57.5, 2, 59.5, 6), (59.0, 3.5), 100,
        (Port("Haugesund", 59.41, 5.27),),
    ),
    "fisher": SeaArea(
        "Fisher", "north_sea", (56.5, -1, 59, 4), (58.0, 1.5), 100,
        (Port("Fisher Bank", 58.0, 1.5),),
    ),
}

# ---------------------------------------------------------------------------
# English Channel / Solent (original focus area, kept intact)
# ---------------------------------------------------------------------------
ENGLISH_CHANNEL_AREAS: Dict[str, SeaArea] = {
    "solent": SeaArea(
        "Solent", "english_channel", (50.5, -2.0, 51.0, -0.8), (50.7633, -1.2985), 50,
        (
            Port("Southampton", 50.9097, -1.4044),
            Port("Portsmouth", 50.8198, -1.0880),
            Port("Cowes", 50.7633, -1.2985),
            Port("Lymington", 50.7594, -1.5441),
            Port("Yarmouth", 50.7070, -1.4990),
            Port("Hamble", 50.8536, -1.3187),
        ),
    ),
    "wight": SeaArea(
        "Wight", "english_channel", (49.5, -2.5, 51, -0.5), (50.4, -1.3), 60,
        (Port("Isle of Wight", 50.6938, -1.3047),),
    ),
    "dover": SeaArea(
        "Dover", "english_channel", (50.5, 0, 51.5, 2.5), (51.0, 1.4), 60,
        (Port("Dover", 51.13, 1.31), Port("Calais", 50.95, 1.85)),
    ),
    "portland": SeaArea(
        "Portland", "english_channel", (49.5, -3.5, 51, -1.5), (50.3, -2.5), 60,
        (Port("Weymouth", 50.61, -2.46), Port("Portland", 50.57, -2.43)),
    ),
    "plymouth": SeaArea(
        "Plymouth", "english_channel", (49.5, -5, 51, -3), (50.2, -4.1), 70,
        (Port("Plymouth", 50.37, -4.14), Port("Falmouth", 50.15, -5.06)),
    ),
    "english_channel": SeaArea(
        "English Channel", "english_channel", (49, -5.5, 51, 2), (50.2, -1.7), 150,
        (Port("Portsmouth", 50.8198, -1.0880), Port("Cherbourg", 49.64, -1.62)),
    ),
}

# ---------------------------------------------------------------------------
# Baltic Sea basins (not part of the UK Shipping Forecast, added per project
# scope: Baltic storm warnings)
# ---------------------------------------------------------------------------
BALTIC_AREAS: Dict[str, SeaArea] = {
    "skagerrak": SeaArea(
        "Skagerrak", "baltic", (57.5, 7, 59.5, 11), (58.3, 9.5), 100,
        (Port("Skagen", 57.72, 10.59), Port("Kristiansand", 58.15, 7.99)),
    ),
    "kattegat": SeaArea(
        "Kattegat", "baltic", (55.5, 10, 58, 12.5), (56.8, 11.2), 90,
        (Port("Gothenburg", 57.71, 11.97), Port("Aarhus", 56.15, 10.21)),
    ),
    "the_belts": SeaArea(
        "The Belts & Sound", "baltic", (54.5, 9.5, 56.5, 13), (55.5, 11.5), 80,
        (Port("Copenhagen", 55.68, 12.57), Port("Kiel", 54.32, 10.14), Port("Malmo", 55.61, 13.00)),
    ),
    "western_baltic": SeaArea(
        "Western Baltic", "baltic", (53.5, 10, 55.5, 15), (54.5, 12.5), 100,
        (Port("Rostock", 54.09, 12.13), Port("Bornholm", 55.10, 14.90)),
    ),
    "southern_baltic": SeaArea(
        "Southern Baltic", "baltic", (53.5, 14, 55.5, 20), (54.4, 17.5), 120,
        (Port("Gdansk", 54.35, 18.65), Port("Swinoujscie", 53.91, 14.25)),
    ),
    "gulf_of_riga": SeaArea(
        "Gulf of Riga", "baltic", (56.5, 21, 58.5, 24.5), (57.5, 23.0), 80,
        (Port("Riga", 56.95, 24.11), Port("Klaipeda", 55.71, 21.14)),
    ),
    "gulf_of_finland": SeaArea(
        "Gulf of Finland", "baltic", (59, 22, 61, 30), (60.0, 26.0), 100,
        (Port("Helsinki", 60.17, 24.94), Port("Tallinn", 59.44, 24.75), Port("St Petersburg", 59.93, 30.34)),
    ),
    "northern_baltic": SeaArea(
        "Northern Baltic & Bothnian Sea", "baltic", (58.5, 17, 63, 22), (60.5, 19.5), 130,
        (Port("Stockholm", 59.33, 18.07), Port("Turku", 60.45, 22.27)),
    ),
    "bothnian_bay": SeaArea(
        "Bothnian Bay", "baltic", (63, 19, 66, 25), (64.5, 22.0), 130,
        (Port("Lulea", 65.58, 22.15), Port("Oulu", 65.01, 25.47)),
    ),
}

ALL_AREAS: Dict[str, SeaArea] = {
    **NORTH_ATLANTIC_AREAS,
    **NORTH_SEA_AREAS,
    **ENGLISH_CHANNEL_AREAS,
    **BALTIC_AREAS,
}
# "Finisterre" was FitzRoy's name until 2002 and is still in common use -
# same area, an extra lookup key rather than a second SeaArea.
ALL_AREAS["finisterre"] = ALL_AREAS["fitzroy"]

# Flat port lookup (name.lower() -> Port), used for map harbor labels and for
# resolving a location string like "Cowes" straight to coordinates.
ALL_PORTS: Dict[str, Port] = {
    port.name.lower(): port
    for area in ALL_AREAS.values()
    for port in area.ports
}

DEFAULT_AREA = "solent"


def resolve_location(name: str) -> Optional[dict]:
    """Resolve a free-text name to {name, lat, lon}. Tries port names first,
    then sea area names. Returns None if nothing matches."""
    key = name.strip().lower()

    if key in ALL_PORTS:
        p = ALL_PORTS[key]
        return {"name": p.name, "lat": p.lat, "lon": p.lon}

    # ALL_AREAS keys are underscored internal identifiers ("irish_sea",
    # "german_bight"), but real-world/LLM input is the natural-language
    # name with spaces ("Irish Sea") - a real bug found live, not
    # hypothetical: 14 of the 43 registered areas have multi-word names
    # and every one of them failed to resolve on its own natural name
    # before this fallback. Try the exact key first (keeps working for
    # any caller that already uses the underscored form directly), then
    # the space-to-underscore form.
    area = ALL_AREAS.get(key) or ALL_AREAS.get(key.replace(" ", "_"))
    if area is not None:
        return {"name": area.name, "lat": area.center[0], "lon": area.center[1]}

    return None


def area_for_location(lat: float, lon: float) -> Optional[SeaArea]:
    """Find which registered sea area (if any) a lat/lon point falls inside."""
    for area in ALL_AREAS.values():
        lat_s, lon_w, lat_n, lon_e = area.bounds
        if lat_s <= lat <= lat_n and lon_w <= lon <= lon_e:
            return area
    return None


def nearby_ports(lat: float, lon: float, radius_deg: float = 3.0) -> Tuple[Port, ...]:
    """Ports within radius_deg of a point, for harbor labels on a chart."""
    return tuple(
        p for p in ALL_PORTS.values()
        if abs(p.lat - lat) <= radius_deg and abs(p.lon - lon) <= radius_deg
    )


def build_regional_grid(
    lat: float, lon: float, radius_miles: int, resolution_deg: float = 0.25
):
    """Build a small lat/lon grid around a point, for requesting a regional
    window directly from a global data source (e.g. via earth2studio's
    fetch_data(..., interp_to=...)) without first running/holding a full
    global grid. Longitude is expressed in the 0-360 convention the model
    grids use, as a single continuous run (no seam) even when the window
    straddles 0 degrees - see forecast_engine._crop_to_region for the same
    reasoning applied to an existing grid instead of a new one.
    """
    import numpy as np

    radius_deg = max(radius_miles, 50) / 69.0
    lat_array = np.arange(lat - radius_deg, lat + radius_deg + resolution_deg, resolution_deg)
    lat_array = np.clip(lat_array, -90.0, 90.0)

    lon_norm = lon % 360
    n_points = max(1, int(round(radius_deg / resolution_deg)))
    offsets = np.arange(-n_points, n_points + 1)
    lon_array = lon_norm + offsets * resolution_deg

    return lat_array, lon_array
