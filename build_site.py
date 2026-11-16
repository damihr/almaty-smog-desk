"""Builds docs/index.html (the demo page) from results.json + forecast.json.  Run: python build_site.py"""
import json, pathlib
ROOT = pathlib.Path(__file__).parent
res = json.load(open(ROOT / "results.json"))
fc = json.load(open(ROOT / "forecast.json"))
html = (ROOT / "site_template.html").read_text()
html = html.replace("/*__RESULTS__*/null", json.dumps(res)).replace("/*__FORECAST__*/null", json.dumps(fc))
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/index.html").write_text(html)
print("wrote docs/index.html")
