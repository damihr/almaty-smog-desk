"""Builds the multi-page demo site in docs/ from site/ + results JSON files.  Run: python build_site.py"""
import json, pathlib, re, shutil
ROOT = pathlib.Path(__file__).parent
SITE, OUT = ROOT / "site", ROOT / "docs"
data = {"cams": json.load(open(ROOT / "results.json")), "cams_fc": json.load(open(ROOT / "forecast.json")),
        "station": json.load(open(ROOT / "station_results.json")), "school_fc": json.load(open(ROOT / "forecast_school.json")),
        "compare": json.load(open(ROOT / "compare_results.json")),
        "map": json.load(open(ROOT / "map_results.json"))}
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "assets").mkdir(parents=True)
for f in (SITE / "assets").iterdir(): shutil.copy(f, OUT / "assets" / f.name)
import datetime as _dt
data["snapshot_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
(OUT / "assets/data.js").write_text("window.DATA = " + json.dumps(data) + ";\n")
HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Almaty Smog Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="assets/data.js"></script><script src="assets/app.js"></script>
</head><body><div class="wrap">
"""
FOOT = """<footer>Almaty Smog Desk · Veritas AI International High School Hackathon 2026 · AI + Climate.<br>This project is a prototype and is not medical advice.</footer></div></body></html>"""
for p in sorted((SITE / "pages").glob("*.html")):
    body = p.read_text()
    title = re.search(r"<!--title:(.*?)-->", body).group(1)
    (OUT / p.name).write_text(HEAD.format(title=title) + body + FOOT)
    print("wrote docs/" + p.name)
