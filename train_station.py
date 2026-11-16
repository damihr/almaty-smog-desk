"""
Part 2: train and test on REAL measurements (OpenAQ), not model estimates.

  - US Embassy Almaty reference monitor (AirNow), Apr 2020 - Nov 2025: train + test (split by date)
  - School No.182 low-cost sensor (AirGradient), Dec 2025 - Nov 2026: second test on a NEW source
  - CAMS model estimate vs embassy measurement: how wrong is the free model data?
  - Live forecast for School No.182 for tomorrow

Run after openaq_fetch.py:  python train_station.py  -> station_results.json, forecast_school.json
"""
import json, pathlib
import numpy as np, pandas as pd, requests
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from train import LAT, LON, TZ, LIMIT, decision_stats
from http_util import get_json

ROOT = pathlib.Path(__file__).parent
TEST_START, VAL_START = "2024-10-01", "2023-10-01"
WX = ["temperature_2m_mean", "temperature_2m_min", "relative_humidity_2m_mean", "wind_speed_10m_mean",
      "precipitation_sum", "pressure_msl_mean", "cloud_cover_mean"]
SHORT = ["temp", "temp_min", "rh", "wind", "precip", "pressure", "cloud"]


def weather(start, end, url="https://archive-api.open-meteo.com/v1/archive", **extra):
    """Daily weather. Historical archive is cached in data/ (it never changes); the live forecast
    is cached for 45 min so all scripts in one refresh share a single API call."""
    import time as _t
    live = start is None
    cache = ROOT / ("data/weather_live.csv" if live else f"data/weather_archive_{start}_{end}.csv")
    if cache.exists() and (not live or _t.time() - cache.stat().st_mtime < 45 * 60):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    p = dict(latitude=LAT, longitude=LON, daily=",".join(WX), timezone=TZ)
    if live: p.update(past_days=30, forecast_days=2)        # one superset request for every caller
    else: p.update(start_date=start, end_date=end, **extra)
    d = get_json(url, p, key="daily")["daily"]
    df = pd.DataFrame({s: d[w] for s, w in zip(SHORT, WX)}, index=pd.to_datetime(d["time"]))
    df.to_csv(cache)
    return df


def load_station(name, min_hours=18):
    s = pd.read_csv(ROOT / f"data/openaq_{name}.csv", index_col=0, parse_dates=True)
    return s.loc[s.hours >= min_hours, "pm25"]


def features(pm, wx):
    pm = pm.reindex(wx.index)                    # daily calendar; missing station days stay NaN
    f = pd.DataFrame(index=wx.index)
    f["pm25_today"], f["pm25_yday"] = pm, pm.shift(1)
    f["pm25_7d"] = pm.rolling(7, min_periods=5).mean()
    f["trend"] = pm - pm.shift(1)
    for c in SHORT:
        f[c + "_today"], f[c + "_tmrw"] = wx[c], wx[c].shift(-1)
    doy = wx.index.day_of_year
    f["season_sin"], f["season_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    f["heating"] = wx.index.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    f["weekend_tmrw"] = ((wx.index + pd.Timedelta(days=1)).day_of_week >= 5).astype(int)
    return f, pm.shift(-1).rename("target")


def models():
    return {"Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                               min_samples_leaf=20, l2_regularization=1.0, random_state=0)}


def main():
    emb = load_station("embassy"); sch = load_station("school182")
    wx = weather("2020-03-25", "2026-11-14")
    X, y = features(emb, wx)
    data = X.join(y).dropna()
    train, test = data[data.index < TEST_START], data[data.index >= TEST_START]
    yte = test.target.values
    print(f"embassy usable days: train {len(train)}  test {len(test)}")

    R = {"train_period": [str(train.index[0].date()), str(train.index[-1].date())],
         "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
         "n_train": len(train), "n_test": len(test), "mae": {}, "decision": {}}
    preds = {"Persistence (tomorrow = today)": test.pm25_today.values, "7-day average": test.pm25_7d.values}
    fs = list(X.columns)
    for n, m in models().items():
        preds[n] = m.fit(train[fs], train.target).predict(test[fs])
    heat = test.heating.values == 1
    R["mae"] = {k: round(mean_absolute_error(yte, p), 2) for k, p in preds.items()}
    R["mae_heating"] = {k: round(mean_absolute_error(yte[heat], p[heat]), 2) for k, p in preds.items()}
    best = min(models(), key=lambda k: R["mae"][k]); R["best_model"] = best

    # warning cut-off chosen on validation season (miss = 3x worse than false alarm)
    tr2, val = train[train.index < VAL_START], train[train.index >= VAL_START]
    pv = models()[best].fit(tr2[fs], tr2.target).predict(val[fs]); yv = val.target.values
    costs = {c: 3*((yv > LIMIT) & (pv <= c)).sum() + ((yv <= LIMIT) & (pv > c)).sum() for c in np.arange(20, 45.5, 0.5)}
    cut = float(min(costs, key=costs.get)); R["warning_cut"] = cut
    R["decision"] = {"Persistence (tomorrow = today)": decision_stats(yte, preds["Persistence (tomorrow = today)"], LIMIT),
                     f"{best}, cut {LIMIT}": decision_stats(yte, preds[best], LIMIT),
                     f"{best}, tuned cut {cut}": decision_stats(yte, preds[best], cut)}
    R["series"] = {"date": [str((t + pd.Timedelta(days=1)).date()) for t in test.index],
                   "actual": yte.round(1).tolist(), "model": preds[best].round(1).tolist(),
                   "persistence": test.pm25_today.values.round(1).tolist()}
    err = pd.DataFrame({"date": R["series"]["date"], "actual": yte.round(1), "model": preds[best].round(1),
                        "persistence": test.pm25_today.values.round(1), "wind_tmrw": test.wind_tmrw.values.round(1),
                        "precip_tmrw": test.precip_tmrw.values.round(1), "temp_tmrw": test.temp_tmrw.values.round(1)})
    err["abs_err"] = (err.actual - err.model).abs().round(1)
    R["worst_days"] = err.sort_values("abs_err", ascending=False).head(6).to_dict("records")

    # ---- final model on ALL embassy data -> test on School 182 (new source, never seen) ----
    final = models()[best].fit(data[fs], data.target)
    Xs, ys = features(sch, wx)
    sd = Xs.join(ys).dropna()
    ps = final.predict(sd[fs]); yss = sd.target.values
    R["school"] = {"period": [str(sd.index[0].date()), str(sd.index[-1].date())], "n": len(sd),
                   "mae": {"Persistence (tomorrow = today)": round(mean_absolute_error(yss, sd.pm25_today), 2),
                           best: round(mean_absolute_error(yss, ps), 2)},
                   "decision": {"Persistence (tomorrow = today)": decision_stats(yss, sd.pm25_today.values, LIMIT),
                                f"{best}, cut {cut}": decision_stats(yss, ps, cut)},
                   "series": {"date": [str((t + pd.Timedelta(days=1)).date()) for t in sd.index],
                              "actual": yss.round(1).tolist(), "model": ps.round(1).tolist()}}

    # ---- CAMS model estimate vs real measurement --------------------------------------------
    cams = pd.read_csv(ROOT / "data/daily.csv", index_col=0, parse_dates=True)["pm25"]
    both = pd.DataFrame({"station": emb, "cams": cams}).dropna()
    both["month"] = both.index.month
    R["cams_vs_station"] = {
        "n": len(both), "mae": round(float((both.station - both.cams).abs().mean()), 2),
        "bias": round(float((both.cams - both.station).mean()), 2),
        "corr": round(float(both.station.corr(both.cams)), 3),
        "smog_days_station": int((both.station > LIMIT).sum()), "smog_days_cams": int((both.cams > LIMIT).sum()),
        "monthly": both.groupby("month")[["station", "cams"]].mean().round(1).reset_index().to_dict("list"),
    }

    # ---- importance (Ridge coefficients on standardised features) ------------------------
    if best == "Ridge":
        coef = final[-1].coef_
        R["coef"] = dict(sorted({f: round(float(c), 2) for f, c in zip(fs, coef)}.items(), key=lambda kv: -abs(kv[1]))[:8])

    json.dump(R, open(ROOT / "station_results.json", "w"), indent=1, default=str)   # save evaluation first
    fc = None
    try:
        # ---- live forecast for School 182 ------------------------------------------------------
        import os
        live = get_json("https://api.openaq.org/v3/sensors/14840476/hours",
                        headers={"X-API-Key": os.environ["OPENAQ_API_KEY"]}, key="results",
                        params=dict(datetime_from=str((pd.Timestamp.now() - pd.Timedelta(days=9)).date()), limit=1000))["results"]
        hr = pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["local"][:19]): x["value"] for x in live}).sort_index()
        daily = hr.resample("D").mean()
        wl = weather(None, None, url="https://api.open-meteo.com/v1/forecast", past_days=9, forecast_days=2)
        today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
        # Before noon, today's measurements are incomplete: use the latest full day as "today"
        base = today if hr.index.max().hour >= 18 and hr.index.max().normalize() == today else today - pd.Timedelta(days=1)
        Xl, _ = features(daily, wl)
        row = Xl.loc[[base], fs]
        pred = float(final.predict(row)[0])
        verdict = "INDOORS" if pred > cut else ("OUTDOORS, LIGHT" if pred > 15 else "OUTDOORS")
        fc = {"station": "School No.182 (AirGradient sensor)", "based_on": str(base.date()),
              "for": str((base + pd.Timedelta(days=1)).date()), "pm25_today": round(float(daily.loc[base]), 1),
              "pm25_tomorrow_forecast": round(pred, 1), "warning_cut": cut, "limit": LIMIT, "verdict": verdict,
              "last_days": {str(k.date()): round(float(v), 1) for k, v in daily.dropna().tail(8).items()},
              "tomorrow_weather": {"temp_min": round(float(wl.temp_min.get(base + pd.Timedelta(days=1), np.nan)), 1),
                                   "wind": round(float(wl.wind.get(base + pd.Timedelta(days=1), np.nan)), 1),
                                   "precip": round(float(wl.precip.get(base + pd.Timedelta(days=1), np.nan)), 1)},
              "hourly_recent": {"time": [t.strftime("%m-%d %H:00") for t in hr.index[-72:]],
                                "pm25": [round(float(v), 1) for v in hr.values[-72:]]}}
        json.dump(fc, open(ROOT / "forecast_school.json", "w"), indent=1)
    except Exception as e:   # live APIs can rate-limit CI runners; keep the previous school forecast
        print(f"WARNING: live School 182 forecast skipped ({type(e).__name__}: {e}); keeping previous forecast_school.json", flush=True)
    print(json.dumps({k: R[k] for k in ["mae", "mae_heating", "warning_cut", "decision"]}, indent=1))
    print("SCHOOL", json.dumps({k: R["school"][k] for k in ["period", "n", "mae", "decision"]}, indent=1))
    print("CAMS", {k: v for k, v in R["cams_vs_station"].items() if k != "monthly"})
    if fc: print("LIVE", {k: v for k, v in fc.items() if k != "hourly_recent"})


if __name__ == "__main__":
    main()
