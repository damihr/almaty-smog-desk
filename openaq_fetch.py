"""Download daily PM2.5 from OpenAQ v3 (needs env var OPENAQ_API_KEY). Run: python openaq_fetch.py"""
import os as _os, pathlib as _pl
# Load secrets from a local .env file (never committed, see .gitignore)
_env = _pl.Path(__file__).parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1); _os.environ.setdefault(_k.strip(), _v.strip())

import os, time, pathlib, requests, pandas as pd
KEY = os.environ["OPENAQ_API_KEY"]
SENSORS = {"embassy": 25903,      # US Embassy Almaty, AirNow reference monitor, 2020-04 -> 2025-11
           "school182": 14840476} # No.182 school-gymnasium, AirGradient low-cost sensor, 2025-12 -> now
OUT = pathlib.Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)

def fetch(sensor, start="2020-01-01", end="2026-11-17"):
    rows, page = [], 1
    while True:
        r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor}/days", headers={"X-API-Key": KEY},
                         params=dict(datetime_from=start, datetime_to=end, limit=1000, page=page), timeout=60)
        r.raise_for_status(); res = r.json()["results"]
        rows += [{"date": x["period"]["datetimeFrom"]["local"][:10], "pm25": x["value"],
                  "hours": x["coverage"]["observedCount"]} for x in res]
        if len(res) < 1000: break
        page += 1; time.sleep(1)
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()

if __name__ == "__main__":
    for name, sid in SENSORS.items():
        df = fetch(sid); df.to_csv(OUT / f"openaq_{name}.csv")
        print(name, len(df), df.index[0], df.index[-1], "median hours/day", df.hours.median())
