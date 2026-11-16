"""
Live forecast: what will tomorrow's PM2.5 be in Almaty, and should PE/recess move indoors?

Uses the last 8 days of PM2.5 + weather and tomorrow's weather forecast from Open-Meteo,
then the model saved by train.py.   Run:  python predict.py
"""
import json, pickle, pathlib
import pandas as pd, requests
from train import LAT, LON, TZ, WEATHER, LIMIT, daily_table, make_features

OUT = pathlib.Path(__file__).parent


def live_forecast():
    aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(
        latitude=LAT, longitude=LON, hourly="pm2_5", past_days=8, forecast_days=1, timezone=TZ), timeout=60).json()
    wx = requests.get("https://api.open-meteo.com/v1/forecast", params=dict(
        latitude=LAT, longitude=LON, hourly=",".join(WEATHER), past_days=8, forecast_days=2, timezone=TZ), timeout=60).json()
    pm = pd.Series(aq["hourly"]["pm2_5"], index=pd.to_datetime(aq["hourly"]["time"]))
    w = pd.DataFrame({k: wx["hourly"][k] for k in WEATHER}, index=pd.to_datetime(wx["hourly"]["time"]))
    pm = pm.reindex(pm.index.union(w.index))   # add tomorrow's (empty) hours so tomorrow's weather row exists
    d = daily_table(pm, w)
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    X = make_features(d)
    bundle = pickle.load(open(OUT / "model.pkl", "rb"))
    row = X.loc[[today], bundle["features"]]
    if row.isna().any(axis=1).iloc[0]:
        raise SystemExit(f"Missing inputs for {today.date()}: {row.columns[row.isna().iloc[0]].tolist()}")
    pred = float(bundle["model"].predict(row)[0])
    cut = bundle["cut"]
    if pred > cut:
        verdict, advice = "INDOORS", "Move PE and recess indoors. Keep windows closed during the evening heating peak."
    elif pred > 15:
        verdict, advice = "OUTDOORS, LIGHT", "Outdoor recess is fine. Keep long or intense PE indoors for children with asthma."
    else:
        verdict, advice = "OUTDOORS", "Air is expected to be clean. Normal outdoor activities."
    out = {
        "issued": str(today.date()), "for": str((today + pd.Timedelta(days=1)).date()),
        "pm25_today": round(float(d.loc[today, "pm25"]), 1),
        "pm25_tomorrow_forecast": round(pred, 1), "warning_cut": cut, "limit": LIMIT,
        "verdict": verdict, "advice": advice, "model": bundle["name"],
        "tomorrow_weather": {"temp_min": round(float(d["temp_min"].shift(-1).loc[today]), 1),
                             "wind": round(float(d["wind"].shift(-1).loc[today]), 1),
                             "precip": round(float(d["precip"].shift(-1).loc[today]), 1)},
        "last_days": {str(k.date()): round(float(v), 1) for k, v in d["pm25"].loc[:today].tail(7).items()},
    }
    json.dump(out, open(OUT / "forecast.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    f = live_forecast()
    print(f"\nAlmaty, forecast for {f['for']}")
    print(f"  Today's PM2.5:      {f['pm25_today']} ug/m3")
    print(f"  Tomorrow forecast:  {f['pm25_tomorrow_forecast']} ug/m3  (warn above {f['warning_cut']})")
    print(f"  Decision:           {f['verdict']}  - {f['advice']}\n")
