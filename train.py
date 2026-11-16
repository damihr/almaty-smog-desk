"""
Almaty Smog Desk: next-day PM2.5 forecast for school outdoor-activity decisions.

Pipeline:
  1. Download hourly PM2.5 (Open-Meteo Air Quality, CAMS model estimates) and hourly
     weather (Open-Meteo Historical Forecast API) for central Almaty.
  2. Aggregate to daily values and build features from TODAY's data
     (+ optionally TOMORROW's weather forecast).
  3. Split by time: train on Aug 2022 - Sep 2025, test on Oct 2025 - Nov 2026.
  4. Compare against the persistence baseline ("tomorrow = today").
  5. Evaluate the school decision: move PE/recess indoors when tomorrow's
     PM2.5 is above 35.4 ug/m3 (US EPA "Unhealthy for Sensitive Groups"; children are a sensitive group).

Run:  python train.py      -> writes data/, results.json, model.pkl
"""
import json, pickle, pathlib
import numpy as np, pandas as pd, requests
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

LAT, LON, TZ = 43.24, 76.89, "Asia/Almaty"      # central Almaty
START, END = "2022-08-01", "2026-11-15"
TEST_START = "2025-10-01"                        # covers one full heating season + current one
VAL_START = "2024-10-01"                         # used only to choose the warning threshold
LIMIT = 35.4                                     # US EPA PM2.5 24h breakpoint (USG)
WEATHER = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "precipitation", "pressure_msl", "cloud_cover"]   # boundary_layer_height dropped: archive only has it from Sep 2024
OUT = pathlib.Path(__file__).parent
(OUT / "data").mkdir(exist_ok=True)


def fetch_pm25(start, end):
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(
        latitude=LAT, longitude=LON, hourly="pm2_5", start_date=start, end_date=end, timezone=TZ), timeout=60)
    h = r.json()["hourly"]
    return pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"]), name="pm25")


def fetch_weather(start, end, url="https://historical-forecast-api.open-meteo.com/v1/forecast"):
    r = requests.get(url, params=dict(latitude=LAT, longitude=LON, hourly=",".join(WEATHER),
                                      start_date=start, end_date=end, timezone=TZ), timeout=60)
    h = r.json()["hourly"]
    return pd.DataFrame({k: h[k] for k in WEATHER}, index=pd.to_datetime(h["time"]))


def daily_table(pm, wx):
    d = pd.DataFrame({
        "pm25": pm.resample("D").mean(),
        "pm25_max": pm.resample("D").max(),
        "pm25_night": pm.between_time("18:00", "23:59").resample("D").mean(),  # evening heating peak
    })
    agg = wx.resample("D").agg({"temperature_2m": ["mean", "min"], "relative_humidity_2m": "mean",
                                "wind_speed_10m": "mean", "precipitation": "sum", "pressure_msl": "mean",
                                "cloud_cover": "mean"})
    agg.columns = ["temp", "temp_min", "rh", "wind", "precip", "pressure", "cloud"]
    return d.join(agg)


def make_features(d):
    """Row t uses only information available on day t (plus tomorrow's weather forecast)."""
    f = pd.DataFrame(index=d.index)
    f["pm25_today"] = d["pm25"]
    f["pm25_max_today"] = d["pm25_max"]
    f["pm25_evening"] = d["pm25_night"]
    f["pm25_yday"] = d["pm25"].shift(1)
    f["pm25_7d"] = d["pm25"].rolling(7).mean()
    f["trend"] = d["pm25"] - d["pm25"].shift(1)
    for c in ["temp", "temp_min", "rh", "wind", "precip", "pressure", "cloud"]:
        f[c + "_today"] = d[c]
        f[c + "_tmrw"] = d[c].shift(-1)          # tomorrow's weather FORECAST (see README: caveat)
    doy = d.index.day_of_year
    f["season_sin"], f["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["heating"] = d.index.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    f["weekend_tmrw"] = ((d.index + pd.Timedelta(days=1)).day_of_week >= 5).astype(int)
    return f


TODAY_ONLY = lambda cols: [c for c in cols if not c.endswith("_tmrw")]


def models():
    return {
        "Linear (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           max_leaf_nodes=15, min_samples_leaf=20,
                                                           l2_regularization=1.0, random_state=0),
    }


SEQ_COLS = ["pm25", "pm25_max", "pm25_night", "temp", "temp_min", "rh", "wind", "precip", "pressure", "cloud"]
TMRW_COLS = ["temp", "temp_min", "rh", "wind", "precip", "pressure", "cloud"]
WINDOW = 14


def lstm_forecast(d, train_idx, test_idx, seeds=(0, 1, 2), epochs=150):
    """LSTM over the last 14 days of PM2.5 + weather, plus tomorrow's weather forecast.
    Scaling stats come from training days only. Early stopping uses the last 15% of
    the training period (chronological), never the test period. Ensemble of 3 seeds."""
    import torch, torch.nn as nn
    torch.set_num_threads(4)
    doy = d.index.day_of_year
    base = d[SEQ_COLS].copy()
    base["season_sin"], base["season_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    tmrw = d[TMRW_COLS].shift(-1)
    mu, sd = base.loc[train_idx].mean(), base.loc[train_idx].std() + 1e-6
    tmu, tsd = tmrw.loc[train_idx].mean(), tmrw.loc[train_idx].std() + 1e-6
    B, T = ((base - mu) / sd).values, ((tmrw - tmu) / tsd).values
    ymu, ysd = d.loc[train_idx, "pm25"].mean(), d.loc[train_idx, "pm25"].std()
    pos = {t: i for i, t in enumerate(d.index)}

    def build(idx):
        xs, xt, ok = [], [], []
        for t in idx:
            i = pos[t]
            if i >= WINDOW - 1:
                w = B[i - WINDOW + 1:i + 1]
                if not np.isnan(w).any() and not np.isnan(T[i]).any():
                    xs.append(w); xt.append(T[i]); ok.append(t)
        return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(xt), dtype=torch.float32), ok

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(B.shape[1], 32, batch_first=True)
            s.head = nn.Sequential(nn.Linear(32 + T.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(s, xs, xt):
            _, (h, _) = s.lstm(xs)
            return s.head(torch.cat([h[-1], xt], 1)).squeeze(1)

    Xs, Xt, ok_tr = build(train_idx)
    y = torch.tensor(((d["pm25"].shift(-1).loc[ok_tr] - ymu) / ysd).values, dtype=torch.float32)
    cut = int(len(ok_tr) * 0.85)
    Ts, Tt, ok_te = build(test_idx)
    outs = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = Net(); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(cut)
            for b in range(0, cut, 64):
                j = perm[b:b+64]
                opt.zero_grad(); loss = nn.functional.l1_loss(net(Xs[j], Xt[j]), y[j]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = nn.functional.l1_loss(net(Xs[cut:], Xt[cut:]), y[cut:]).item()
            if v < best - 1e-4:
                best, best_state, patience = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience > 15: break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            outs.append(net(Ts, Tt).numpy() * ysd + ymu)
    return pd.Series(np.mean(outs, 0), index=ok_te)


def decision_stats(y_true, y_pred, cut):
    bad = y_true > LIMIT
    warn = y_pred > cut
    return dict(bad_days=int(bad.sum()), warned=int(warn.sum()),
                caught=int((bad & warn).sum()), missed=int((bad & ~warn).sum()),
                false_alarms=int((~bad & warn).sum()),
                recall=round(float((bad & warn).sum() / max(bad.sum(), 1)), 3),
                precision=round(float((bad & warn).sum() / max(warn.sum(), 1)), 3))


def main():
    print("Downloading data ...")
    pm, wx = fetch_pm25(START, END), fetch_weather(START, END)
    pm.to_csv(OUT / "data/pm25_hourly.csv"); wx.to_csv(OUT / "data/weather_hourly.csv")
    d = daily_table(pm, wx)
    d.to_csv(OUT / "data/daily.csv")
    X = make_features(d)
    y = d["pm25"].shift(-1).rename("target")              # tomorrow's daily mean PM2.5
    data = X.join(y).dropna()
    print(f"{len(data)} usable days, {data.index[0].date()} -> {data.index[-1].date()}")

    train, test = data[data.index < TEST_START], data[data.index >= TEST_START]
    feats_all = [c for c in X.columns]
    feats_today = TODAY_ONLY(feats_all)
    yte = test["target"].values

    res = {"n_train": len(train), "n_test": len(test),
           "train_period": [str(train.index[0].date()), str(train.index[-1].date())],
           "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
           "limit": LIMIT, "mae": {}, "mae_heating": {}, "mae_by_level": {}}
    heat = test["heating"].values == 1

    preds = {"Persistence (tomorrow = today)": test["pm25_today"].values,
             "7-day average": test["pm25_7d"].values}
    for fs_name, fs in [("today only", feats_today), ("+ tomorrow's weather forecast", feats_all)]:
        for name, m in models().items():
            m.fit(train[fs], train["target"])
            preds[f"{name}, {fs_name}"] = m.predict(test[fs])
    print("Training LSTM ...")
    lstm = lstm_forecast(d, train.index, test.index)
    preds["LSTM (14-day sequence), + tomorrow's weather forecast"] = lstm.reindex(test.index).values
    assert not np.isnan(preds["LSTM (14-day sequence), + tomorrow's weather forecast"]).any()

    level = pd.cut(yte, [0, 15, LIMIT, 1e9], labels=["clean (<15)", "moderate (15-35)", "unhealthy (>35)"])
    for k, p in preds.items():
        res["mae"][k] = round(mean_absolute_error(yte, p), 2)
        res["mae_heating"][k] = round(mean_absolute_error(yte[heat], p[heat]), 2)
        res["mae_by_level"][k] = {str(l): round(float(np.abs(yte - p)[level == l].mean()), 2)
                                  for l in level.categories}
    # The live tool uses the best sklearn model (LSTM is reported for comparison).
    best = min((k for k in preds if "Persistence" not in k and "7-day" not in k and "LSTM" not in k),
               key=lambda k: res["mae"][k])
    res["best_model"] = best
    print(json.dumps(res["mae"], indent=1))

    # ---- Choose the warning cut-off on a VALIDATION season (never on test) --------------
    # A missed smog day exposes children; a false alarm only costs one outdoor PE lesson.
    # We weight a miss as 3x worse than a false alarm and pick the cut that minimises cost.
    fs = feats_all if "forecast" in best else feats_today
    mname = best.split(",")[0]
    tr2, val = train[train.index < VAL_START], train[train.index >= VAL_START]
    mv = models()[mname].fit(tr2[fs], tr2["target"])
    pv, yv = mv.predict(val[fs]), val["target"].values
    costs = {c: 3 * ((yv > LIMIT) & (pv <= c)).sum() + ((yv <= LIMIT) & (pv > c)).sum()
             for c in np.arange(20, 40.5, 0.5)}
    cut = float(min(costs, key=costs.get))
    res["warning_cut"] = cut

    final = models()[mname].fit(train[fs], train["target"])
    p_best = preds[best]
    res["decision"] = {
        "Persistence (tomorrow = today)": decision_stats(yte, preds["Persistence (tomorrow = today)"], LIMIT),
        f"Model, naive cut {LIMIT}": decision_stats(yte, p_best, LIMIT),
        f"Model, tuned cut {cut}": decision_stats(yte, p_best, cut),
        f"LSTM, cut {cut}": decision_stats(yte, preds["LSTM (14-day sequence), + tomorrow's weather forecast"], cut),
    }

    # Trade-off table on TEST (for reporting only; the cut above was fixed on validation)
    res["tradeoff"] = [dict(cut=c, **decision_stats(yte, p_best, c)) for c in [25, 27.5, 30, 32.5, 35.4]]

    # ---- Where does it fail? ------------------------------------------------------------
    err = pd.DataFrame({"date": (test.index + pd.Timedelta(days=1)).date.astype(str),  # the day being forecast
                        "actual": yte.round(1), "model": p_best.round(1),
                        "persistence": test["pm25_today"].values.round(1),
                        "wind_tmrw": test["wind_tmrw"].values.round(1),
                        "precip_tmrw": test["precip_tmrw"].values.round(1),
                                                "temp_tmrw": test["temp_tmrw"].values.round(1)})
    err["abs_err"] = (err["actual"] - err["model"]).abs().round(1)
    res["worst_days"] = err.sort_values("abs_err", ascending=False).head(8).to_dict("records")
    err["lstm"] = preds["LSTM (14-day sequence), + tomorrow's weather forecast"].round(1)
    res["series"] = err[["date", "actual", "model", "lstm", "persistence"]].to_dict("list")

    # Feature importance (permutation, on test) for the explanation section
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(final, test[fs], yte, n_repeats=10, random_state=0,
                                scoring="neg_mean_absolute_error")
    res["importance"] = dict(sorted({f: round(float(v), 2) for f, v in zip(fs, pi.importances_mean)}.items(),
                                    key=lambda kv: -kv[1])[:10])

    # Refit on ALL data for live use
    live = models()[mname].fit(data[fs], data["target"])
    pickle.dump({"model": live, "features": fs, "cut": cut, "name": best}, open(OUT / "model.pkl", "wb"))
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
    print("best:", best, "| cut:", cut)
    print(json.dumps(res["decision"], indent=1))
    print(json.dumps(res["importance"], indent=1))


if __name__ == "__main__":
    main()
