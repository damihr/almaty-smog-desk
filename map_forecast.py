"""
City map: forecast tomorrow's PM2.5 at EVERY live OpenAQ sensor in Almaty.

For each sensor that reported in the last 2 days: fetch its last 10 days of hourly PM2.5,
then predict tomorrow with Ridge + Gradient boosting (trained on the embassy monitor) and
Chronos-Bolt (zero-shot), and average them. (The LSTM is left out of the map to keep the
live run fast.) The site draws Voronoi zones around sensors, coloured by US EPA category.

Run (needs OPENAQ_API_KEY):  python map_forecast.py  -> map_results.json
"""
import json, os, time, pathlib
import numpy as np, pandas as pd, requests
from train import TZ, LAT, LON
from train_station import load_station, weather, features, models
from compare_models import chronos_predict
from http_util import get_json

ROOT = pathlib.Path(__file__).parent
H = {"X-API-Key": os.environ["OPENAQ_API_KEY"]}
CATS = [(9.0, "Good"), (35.4, "Moderate"), (55.4, "Unhealthy for sensitive groups"),
        (125.4, "Unhealthy"), (225.4, "Very unhealthy"), (1e9, "Hazardous")]
cat = lambda v: next(n for t, n in CATS if v <= t)


def main():
    now = pd.Timestamp.now(tz="UTC")
    locs = get_json("https://api.openaq.org/v3/locations", headers=H, key="results", params=dict(
        coordinates=f"{LAT},{LON}", radius=25000, parameters_id=2, limit=100))["results"]
    live = []
    for l in locs:
        last = (l.get("datetimeLast") or {}).get("utc")
        s = [x for x in l["sensors"] if x["parameter"]["name"] == "pm25"]
        if last and s and now - pd.Timestamp(last) < pd.Timedelta(days=2):
            live.append({"id": l["id"], "name": l["name"].strip(), "provider": l["provider"]["name"],
                         "lat": l["coordinates"]["latitude"], "lon": l["coordinates"]["longitude"], "sensor": s[0]["id"]})
    print(len(live), "live sensors")

    series = {}
    for st in live:
        try:
            res = get_json(f"https://api.openaq.org/v3/sensors/{st['sensor']}/hours", headers=H, key="results", tries=3,
                           params=dict(datetime_from=str((now - pd.Timedelta(days=11)).date()), limit=1000))["results"]
        except RuntimeError as e:
            print("  skip sensor", st["name"], e); continue
        hr = pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["local"][:19]): x["value"] for x in res
                        if x["value"] is not None and 0 <= x["value"] < 1000}).sort_index()
        if len(hr) > 48: series[st["id"]] = hr
        time.sleep(0.4)

    emb = load_station("embassy"); wx = weather("2020-03-25", "2026-11-14")
    X, y = features(emb, wx); fs = list(X.columns); data = X.join(y).dropna()
    fitted = {n: m.fit(data[fs], data.target) for n, m in models().items()}
    wl = weather(None, None, url="https://api.open-meteo.com/v1/forecast", past_days=11, forecast_days=2)
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)

    out = []
    for st in live:
        hr = series.get(st["id"])
        if hr is None: continue
        daily = hr.resample("D").mean()
        cnt = hr.resample("D").count()
        base = today if cnt.get(today, 0) >= 18 else today - pd.Timedelta(days=1)
        if base not in daily.index or np.isnan(daily.loc[base]): continue
        Xl, _ = features(daily, wl)
        row = Xl.loc[[base], fs]
        if row.isna().any(axis=1).iloc[0]:     # gaps: fill 7-day mean / yesterday from available days
            row = row.fillna({"pm25_yday": daily.loc[base], "pm25_7d": daily.loc[:base].tail(7).mean(), "trend": 0})
        if row.isna().any(axis=1).iloc[0]: continue
        p = {n: float(m.predict(row)[0]) for n, m in fitted.items()}
        st.update(base=base, daily=daily, p=p, now=float(hr.tail(3).mean()), now_time=str(hr.index[-1]))
        out.append(st)
    # Chronos in one batch: each sensor's own daily history
    import compare_models as cm, torch
    if cm._chronos is None:
        from chronos import BaseChronosPipeline
        cm._chronos = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    ctx = [torch.tensor(s["daily"].loc[:s["base"]].interpolate().bfill().values, dtype=torch.float32) for s in out]
    ctx = torch.nn.utils.rnn.pad_sequence([c.flip(0) for c in ctx], batch_first=True, padding_value=float("nan")).flip(1)
    q, _ = cm._chronos.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])

    feats = []
    for i, s in enumerate(out):
        s["p"]["Chronos-Bolt"] = float(q[i, 0, 1])
        ens = float(np.mean(list(s["p"].values())))
        feats.append({"id": s["id"], "name": s["name"], "provider": s["provider"], "lat": s["lat"], "lon": s["lon"],
                      "now": round(s["now"], 1), "now_cat": cat(s["now"]), "last_reading": s["now_time"],
                      "today": round(float(s["daily"].loc[s["base"]]), 1), "based_on": str(s["base"].date()),
                      "forecast": round(ens, 1), "forecast_cat": cat(ens),
                      "models": {k: round(v, 1) for k, v in s["p"].items()},
                      "band": [round(float(q[i, 0, 0]), 1), round(float(q[i, 0, 2]), 1)],
                      "last_days": [round(float(v), 1) for v in s["daily"].tail(7).values]})
    R = {"generated": str(pd.Timestamp.now(tz=TZ))[:16], "for": str((today + pd.Timedelta(days=1)).date()),
         "sensors": feats, "categories": [[t, n] for t, n in CATS]}
    json.dump(R, open(ROOT / "map_results.json", "w"), indent=1, default=str)
    for f in sorted(feats, key=lambda f: -f["forecast"]):
        print(f"{f['name'][:28]:28} now {f['now']:6.1f}  today {f['today']:6.1f}  tmrw {f['forecast']:6.1f}  {f['forecast_cat']}")


if __name__ == "__main__":
    main()
