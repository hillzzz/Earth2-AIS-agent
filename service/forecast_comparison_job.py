#!/usr/bin/env python
"""Nightly Earth-2 vs Met Office shipping-forecast comparison.

Picks 3 Met Office shipping-forecast areas at random, runs the local
Earth-2 navigation forecast for each, and fetches the official Met Office
shipping forecast (server-side rendered HTML, no browser needed). Writes a
marked-up comparison for the overlapping 24h window (day 1) and APPENDS
the result to ../forecast_comparison.md. Designed to run unattended from
cron: no LLM, no MCP server, no browser - only stdlib + this project's
agent_tools + the project venv.

Usage:
    python forecast_comparison_job.py          # normal run (appends report)
    python forecast_comparison_job.py --print  # print report to stdout, do NOT append
    python forecast_comparison_job.py --dry-run "Thames,German Bight,Sole"
                                            # compare these areas, do NOT append

Exit codes: 0 = ok, 1 = hard failure (nothing appended), 2 = partial
(>=1 area failed, partial report appended).
"""
import argparse
import datetime as _dt
import html as _html
import json
import random
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))

REPORT_PATH = SERVICE_DIR.parent / "forecast_comparison.html"
MO_SHP_FORECAST_URL = (
    "https://weather.metoffice.gov.uk/specialist-forecasts/coast-and-sea/shipping-forecast"
)
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125 Safari/537.36")

# The 31 official Met Office shipping-forecast sea areas (exactly as they
# appear in the page's "Select a sea area" dropdown / area headers).
METOFFICE_AREAS = [
    "Viking", "North Utsire", "South Utsire", "Forties", "Cromarty",
    "Forth", "Tyne", "Dogger", "Fisher", "German Bight", "Humber",
    "Thames", "Dover", "Wight", "Portland", "Plymouth", "Biscay",
    "Trafalgar", "FitzRoy", "Sole", "Lundy", "Fastnet", "Irish Sea",
    "Shannon", "Rockall", "Malin", "Hebrides", "Bailey", "Fair Isle",
    "Faeroes", "Southeast Iceland",
]

# Sea-state word -> representative Beaufort-equivalent scale (for comparison
# only; the Mo words are not a formal scale, so we keep them qualitative).
SEA_STATE_TO_NUM = {
    "smooth": 0, "calm": 0, "slight": 2, "moderate": 3, "rough": 4,
    "very rough": 6, "high": 7, "phenomenal": 8,
}


# --------------------------------------------------------------------------- #
# Met Office fetch + parse (stdlib only, server-side rendered HTML)
# --------------------------------------------------------------------------- #

def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def fetch_mo_forecast(timeout=30, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(MO_SHP_FORECAST_URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
                return r.read().decode(errors="replace")
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Met Office fetch failed after {retries} tries: {last}")


def parse_mo_area(raw_html, area):
    """Extract WIND / SEA STATE / WEATHER / VISIBILITY + gale status for one area.

    The page renders each area as a
        <section ... data-value="German Bight" class="marine-card">
            <h2 ...>German Bight</h2>
            <dl><dt>Wind</dt><dd>...</dd><dt>Sea state</dt><dd>...</dd>
                <dt>Weather</dt><dd>...</dd><dt>Visibility</dt><dd>...</dd></dl>
        </section>
    so we anchor on that section and read its dt/dd pairs - never the raw
    'area name' string (which also appears in a dropdown near the top of the
    page, where a naive window would pick up a different area's text).

    Returns a dict, or None if the area's card isn't in the page.
    """
    # Anchor: the section opening tag carrying this area's data-value.
    m = re.search(r'data-value="' + re.escape(area) + r'"[^>]*class="marine-card"',
                  raw_html)
    if m is None:
        # Fall back: an <h2 class="card-name">AREA</h2>, then bound to next </section>
        m2 = re.search(r'<h2[^>]*class="card-name">' + re.escape(area) + r'</h2>', raw_html)
        if m2 is None:
            return None
        start = m2.start()
    else:
        start = m.start()
    end = raw_html.find("</section>", start)
    section = raw_html[start:end if end >= 0 else start + 4000]

    # dt/dd pairs
    pairs = re.findall(r"<dt>\s*([^<]+?)\s*</dt>\s*<dd>\s*(.*?)\s*</dd>", section,
                       re.S | re.I)
    fields = {}
    for k, v in pairs:
        fields[k.strip().lower()] = re.sub(r"\s+", " ", v).strip()

    wind = fields.get("wind", "")
    sea = fields.get("sea state")
    weather = fields.get("weather")
    visibility = fields.get("visibility")

    # Gale warning: banner text may sit between the <h2> and the <dl>.
    gale = bool(re.search(r"\bGALE WARNING\b", section, re.I))
    gale_note = None
    gm = re.search(r"GALE WARNING\b(.*?)(?:WIND\b|<[^>]+>)", section, re.I | re.S)
    if gm:
        gale_note = re.sub(r"<[^>]+>", " ", gm.group(1))
        gale_note = re.sub(r"\s+", " ", gale_note).strip()

    # Peak Beaufort force from the wind line: plain numbers and any "gale N".
    forces = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", wind)]
    gale_hits = [int(x) for x in re.findall(r"\bgale\s+(\d{1,2})\b", wind, re.I)]
    peak = max(forces + gale_hits) if (forces or gale_hits) else None

    # Sea state: strongest word mentioned (strongest = highest number).
    sea_words = re.findall(r"very rough|phenomenal|high|rough|moderate|slight|smooth|calm",
                           (sea or "").lower())
    sea_max = max((SEA_STATE_TO_NUM.get(w, 0) for w in sea_words), default=None)
    sea_max_word = (sorted((w for w in sea_words if SEA_STATE_TO_NUM.get(w) == sea_max),
                           key=len, reverse=True)[0] if sea_words else None)

    if wind == "" and sea is None and weather is None and visibility is None and not gale:
        return None  # no card body at all - treat as not found

    return {
        "found": True,
        "wind": wind,
        "sea_state": sea,
        "weather": weather,
        "visibility": visibility,
        "gale_warning": gale,
        "gale_note": gale_note,
        "peak_force": peak,
        "sea_max_num": sea_max,
        "sea_max_word": sea_max_word,
    }


# --------------------------------------------------------------------------- #
# Per-area comparison
# --------------------------------------------------------------------------- #

def compare_area(area, e2, mo):
    """Merge the two sources for one area and produce findings + a row dict."""
    findings = []

    d1 = None
    for d in e2.get("daily_breakdown", []):
        if d.get("worst_timing", "").lower() == "imminent":
            d1 = d
            break
    if d1 is None and e2.get("daily_breakdown"):
        d1 = e2["daily_breakdown"][0]

    e2_peak = d1.get("max_beaufort_force") if d1 else None
    e2_name = (d1 or {}).get("max_beaufort_name", "")
    e2_wave = (d1 or {}).get("max_wave_height_m")
    e2_gale = e2_peak is not None and e2_peak >= 8

    mo_peak = mo["peak_force"]
    mo_gale = mo["gale_warning"]

    if d1 is None:
        findings.append("Earth-2 produced no day-1 breakdown for this area.")
    if mo_peak is None:
        findings.append("Could not parse a wind force from the Met Office entry.")
    if e2_peak is not None and mo_peak is not None:
        if e2_peak == mo_peak:
            pass  # agree
        elif abs(e2_peak - mo_peak) == 1:
            findings.append(
                f"Peak force differs by one: Earth-2 {e2_peak} vs Met Office {mo_peak} "
                f"(Earth-2 {'stronger' if e2_peak > mo_peak else 'weaker'})."
            )
        else:
            findings.append(
                f"Peak force differs by {abs(e2_peak - mo_peak)}: Earth-2 {e2_peak} vs "
                f"Met Office {mo_peak} (Earth-2 {'stronger' if e2_peak > mo_peak else 'weaker'})."
            )

    # Gale agreement
    if e2_peak is not None and mo_peak is not None:
        if e2_gale and not mo_gale:
            findings.append("Earth-2 flags a Gale (force>=8) here; Met Office has no gale warning.")
        elif mo_gale and not e2_gale:
            findings.append("Met Office issues a Gale warning here; Earth-2 is below Gale.")

    # Sea state vs wave height (qualitative)
    if e2_wave is not None and mo["sea_max_num"] is not None:
        # rough: height 0.5-1.0m, moderate 1-2, rough 2-4, very rough 4-6
        def band(h):
            if h <= 1.0: return "slight/mild"
            if h <= 2.0: return "moderate"
            if h <= 4.0: return "rough"
            return "very rough/high"
        if band(e2_wave) != mo["sea_max_word"]:
            findings.append(
                f"Sea state: Earth-2 ~{e2_wave}m ({band(e2_wave)}) vs Met Office "
                f"'{mo['sea_max_word']}' - different band."
            )

    match = (e2_peak is not None and mo_peak is not None and e2_peak == mo_peak
             and e2_gale == mo_gale)
    return {
        "area": area,
        "e2_peak": e2_peak, "e2_name": e2_name, "e2_wave": e2_wave,
        "e2_gale": e2_gale,
        "e2_gale_warnings": e2.get("gale_warnings"),
        "e2_storms": e2.get("storms_tracked"),
        "e2_wave_source": e2.get("wave_source"),
        "mo_peak": mo_peak, "mo_wind": mo.get("wind"),
        "mo_sea": mo.get("sea_state"), "mo_weather": mo.get("weather"),
        "mo_vis": mo.get("visibility"), "mo_gale": mo_gale,
        "mo_sea_word": mo.get("sea_max_word"),
        "match": match, "findings": findings,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def _fmt_force(v):
    if v is None:
        return "n/a"
    return f"{v} ({BEAUFORT_NAMES.get(v, '')})"


BEAUFORT_NAMES = {
    1: "Light air", 2: "Light breeze", 3: "Gentle breeze", 4: "Moderate breeze",
    5: "Fresh breeze", 6: "Strong breeze", 7: "Near gale", 8: "Gale",
    9: "Severe gale", 10: "Storm", 11: "Violent storm", 12: "Hurricane",
}


APPEND_MARKER = "<!-- reports-append-here -->"

# Standalone-document shell for forecast_comparison.html. New <section> reports
# are spliced in immediately before APPEND_MARKER (inserted at the bottom, so
# the file reads chronologically) - the first run creates the shell, later runs
# splice into the existing file, and the result is always a valid HTML doc.
_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Earth-2 vs Met Office — daily forecast comparison</title>
<style>
  :root { --bg:#0f1115; --panel:#161a22; --ink:#e7ecf3; --muted:#9aa4b2;
          --line:#262c37; --accent:#5aa9ff; --ok:#3fb950; --warn:#d4a72c; --bad:#f0766a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:28px 20px 64px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .subtitle { color:var(--muted); margin:0 0 24px; }
  .report { background:var(--panel); border:1px solid var(--line); border-radius:10px;
            padding:18px 20px; margin:0 0 22px; }
  .report h2 { font-size:17px; margin:0 0 6px; }
  .meta { color:var(--muted); font-size:13px; margin:0 0 14px; }
  .note { background:rgba(212,167,44,.12); border-left:3px solid var(--warn);
          padding:8px 12px; border-radius:6px; margin:10px 0; font-size:13.5px; }
  table { border-collapse:collapse; width:100%; margin:6px 0 4px; font-size:13.5px; }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; background:rgba(255,255,255,.02); }
  .verdict { font-weight:600; } .match { color:var(--ok); } .disc { color:var(--bad); }
  .area { background:rgba(255,255,255,.03); border:1px solid var(--line); border-radius:8px;
          padding:12px 14px; margin:12px 0; }
  .area h3 { font-size:15px; margin:0 0 8px; color:var(--accent); }
  .area ul { margin:6px 0 0; padding-left:18px; }
  .area li { margin:3px 0; }
  .fail { color:var(--bad); font-weight:600; }
  .foot { color:var(--muted); font-size:12.5px; margin-top:10px; padding-top:10px;
          border-top:1px dashed var(--line); }
</style>
</head>
<body>
<main class="wrap">
  <h1>Earth-2 vs Met Office — daily forecast comparison</h1>
  <p class="subtitle">Random 3 shipping-forecast areas per day. Newest run is at the bottom.</p>
  <div class="reports">
""" + APPEND_MARKER + """
  </div>
</main>
</body>
</html>
"""


def _esc(s):
    return _html.escape(str(s))


def _fmt(m):
    return "n/a" if m is None or m == "" else str(m)


def render_report(picked, rows, mo_issue):
    """Render one run as a self-contained HTML <section> block."""
    now = _dt.datetime.now(_dt.timezone.utc)
    L = []
    L.append('<section class="report">')
    L.append(f"<h2>{now:%Y-%m-%d %H:%M} UTC — Earth-2 vs Met Office "
             f"(random {len(picked)} areas)</h2>")
    L.append(f'<p class="meta">Areas picked: {_esc(", ".join(picked))}</p>')
    if mo_issue:
        L.append(f'<div class="note">{_esc(mo_issue)}</div>')

    # Comparison table
    L.append("<table>")
    L.append("<thead><tr><th>Area</th><th>Earth-2 peak (day 1)</th><th>Earth-2 wave</th>"
             "<th>Met Office wind</th><th>Met Office sea</th><th>Gale (E2 / MO)</th>"
             "<th>Verdict</th></tr></thead>")
    L.append("<tbody>")
    for r in rows:
        cls = "match" if r["match"] else "disc"
        verdict = "MATCH" if r["match"] else "DISCREPANCY"
        wave = "n/a" if r["e2_wave"] in (None, "") else f"{r['e2_wave']}m"
        L.append("<tr>"
                 f"<td>{_esc(r['area'])}</td>"
                 f"<td>{_fmt(r['e2_peak'])}</td>"
                 f"<td>{wave}</td>"
                 f"<td>{_fmt(r['mo_peak'])}</td>"
                 f"<td>{_esc(r['mo_sea'])}</td>"
                 f"<td>{'Y' if r['e2_gale'] else '–'} / {'Y' if r['mo_gale'] else '–'}</td>"
                 f'<td class="verdict {cls}">{verdict}</td>'
                 "</tr>")
    L.append("</tbody></table>")

    # Per-area detail
    for r in rows:
        L.append('<div class="area">')
        L.append(f"<h3>{_esc(r['area'])}</h3>")
        L.append("<ul>")
        if r.get("_e2_error"):
            L.append(f'<li class="fail">Earth-2: <b>FAILED</b> — {_esc(r["_e2_error"])}</li>')
        else:
            wave = "n/a" if r["e2_wave"] is None else f"{r['e2_wave']}m"
            L.append(
                f"<li>Earth-2 (day 1): peak {_fmt(r['e2_peak'])}, wave {wave}, "
                f"gale warnings {_fmt(r['e2_gale_warnings'])}, "
                f"storm systems {_fmt(r['e2_storms'])}, "
                f"wave source: {_esc(r['e2_wave_source'] or 'n/a')}</li>")
            days = ", ".join(f"{_esc(d.get('date'))}:F{_fmt(d.get('max_beaufort_force'))}"
                             for d in r.get('_e2_days', []))
            L.append(f"<li>Earth-2 full window: {_esc(days)}</li>")
        if r.get("_mo_error"):
            L.append(f'<li class="fail">Met Office: <b>FAILED</b> — {_esc(r["_mo_error"])}</li>')
        else:
            L.append(f"<li>Met Office wind: {_esc(r['mo_wind'] or 'n/a')}</li>")
            L.append(f"<li>Met Office sea state: {_esc(r['mo_sea'] or 'n/a')}</li>")
            L.append(f"<li>Met Office weather: {_esc(r['mo_weather'] or 'n/a')}"
                     f" &nbsp;|&nbsp; visibility: {_esc(r['mo_vis'] or 'n/a')}</li>")
        L.append("<li><b>Findings:</b></li>")
        if r["findings"]:
            for fd in r["findings"]:
                L.append(f"    <li>{_esc(fd)}</li>")
        else:
            L.append("    <li>No discrepancies flagged on the overlapping 24h window.</li>")
        L.append("</ul>")
        L.append("</div>")

    L.append('<p class="foot">Scope: the Met Office shipping forecast covers a 24h period; '
             'Earth-2 is a 5-6 day navigation forecast, so the comparable window is day 1. '
             'Earth-2 = ECMWF GFS/IFS; Met Office = UKV/IFS blend + forecaster. A 1-2 force '
             'peak split between them is normal.</p>')
    L.append("</section>")
    return "\n".join(L)


def _save_html_report(section):
    """Insert a rendered <section> before APPEND_MARKER; create the shell if new."""
    shell = _HTML_SHELL
    if REPORT_PATH.exists():
        existing = REPORT_PATH.read_text(encoding="utf-8")
        if APPEND_MARKER in existing:
            shell = existing
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(shell.replace(APPEND_MARKER, section + "\n\n" + APPEND_MARKER, 1),
                           encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(picked, do_append=True, show_report=False):
    import agent_tools  # imported inside so --help/parse paths don't need it

    mo_issue = None
    raw_html = None
    try:
        raw_html = fetch_mo_forecast()
    except Exception as e:
        mo_issue = f"Met Office fetch failed — Mo side of comparison unavailable ({e})."

    rows = []
    hard_fail = 0
    _INIT_ROW = {
        "e2_peak": None, "e2_name": None, "e2_wave": None, "e2_gale": False,
        "e2_gale_warnings": None, "e2_storms": None, "e2_wave_source": None,
        "mo_peak": None, "mo_wind": None, "mo_sea": None, "mo_weather": None,
        "mo_vis": None, "mo_gale": False, "mo_sea_word": None,
    }
    for area in picked:
        row = {"area": area, "findings": [], "match": False,
               "_e2_days": [], "_e2_error": None, "_mo_error": None}
        row.update(_INIT_ROW)
        # Earth-2
        e2 = None
        try:
            e2 = agent_tools.generate_forecast(area, "navigation")
        except Exception as e:
            row["_e2_error"] = f"{type(e).__name__}: {e}"
        if e2 is not None and not (e2 or {}).get("error"):
            row["_e2_days"] = e2.get("daily_breakdown", [])
        elif e2 is not None:
            row["_e2_error"] = str(e2.get("error"))
        # Met Office
        mo = None
        if raw_html is None:
            row["_mo_error"] = "Met Office fetch failed"
        else:
            try:
                mo = parse_mo_area(raw_html, area)
            except Exception as e:
                row["_mo_error"] = f"{type(e).__name__}: {e}"
        if mo is not None and not mo.get("found"):
            row["_mo_error"] = "area not found in Met Office page"
            mo = None

        if e2 and mo:
            merged = compare_area(area, e2, mo)
            row.update(merged)
        elif row["_e2_error"] and row["_mo_error"]:
            hard_fail += 1
        else:
            # partial: keep whatever we have
            if e2 and not mo:
                d1 = next((d for d in (e2.get("daily_breakdown") or [])
                           if d.get("worst_timing","").lower()=="imminent"),
                          (e2.get("daily_breakdown") or [None])[0])
                row["e2_peak"] = (d1 or {}).get("max_beaufort_force")
                row["e2_name"] = (d1 or {}).get("max_beaufort_name")
                row["e2_wave"] = (d1 or {}).get("max_wave_height_m")
                row["e2_gale"] = bool(row["e2_peak"] and row["e2_peak"] >= 8)
                row["e2_gale_warnings"] = e2.get("gale_warnings")
                row["e2_storms"] = e2.get("storms_tracked")
                row["e2_wave_source"] = e2.get("wave_source")
        rows.append(row)

    report = render_report(picked, rows, mo_issue)

    if do_append:
        _save_html_report(report)

    n_ok = sum(1 for r in rows if not (r["_e2_error"] or r["_mo_error"]))
    n_disc = sum(1 for r in rows if not r["match"] and not (r["_e2_error"] or r["_mo_error"]))
    if show_report:
        print(report)
    else:
        print("=" * 60)
        print(f"Wrote: {REPORT_PATH}" if do_append else "(dry run / print-only)")
    print(f"Areas OK: {n_ok}/{len(picked)}  Discrepancies on 24h window: {n_disc}")
    if mo_issue:
        print("WARN:", mo_issue)
    for r in rows:
        tag = "ok" if not (r["_e2_error"] or r["_mo_error"]) else "PARTIAL/FAIL"
        v = "MATCH" if r["match"] else ("DISC" if tag == "ok" else "--")
        print(f"  - {r['area']:<16} {tag:<10} {v}")
    return (0 if n_ok == len(picked) else (2 if n_ok else 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="print only, do not append")
    ap.add_argument("--dry-run", metavar="AREAS", help="compare these comma-separated areas, no append")
    ap.add_argument("--areas", metavar="AREAS", help=argparse.SUPPRESS)  # alias
    a = ap.parse_args()

    if a.dry_run or a.areas:
        vals = (a.dry_run or a.areas).split(",")
        vals = [v.strip() for v in vals if v.strip()]
        run(vals, do_append=False, show_report=True)
    else:
        picked = random.sample(METOFFICE_AREAS, 3)
        rc = run(picked, do_append=not a.print)
        sys.exit(rc)


if __name__ == "__main__":
    main()
