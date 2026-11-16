"""
Model showdown on REAL station data: every model sees the same data, the same date split
and is scored on the same test days. Then every model makes a live forecast from the same
freshly fetched School No.182 data.

Models
  1. Persistence            "tomorrow = today" (baseline)
  2. Ridge                  linear regression on today's PM2.5 + weather + tomorrow's forecast
  3. Gradient boosting      histogram gradient-boosted trees (same family as XGBoost/LightGBM)
  4. LSTM                   PyTorch, reads a 14-day sequence of PM2.5 + weather
  5. Chronos-Bolt           Amazon's pretrained time-series foundation model (2024), ZERO-SHOT:
                            never trained on our data, sees only the PM2.5 history
  6. Ensemble               plain average of models 2-5 (fixed in advance, not tuned on test)

Run after train_station.py (needs OPENAQ_API_KEY):  python compare_models.py -> compare_results.json
"""
import json, os, pathlib
import numpy as np, pandas as pd, requests, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error
from train import TZ, LIMIT, decision_stats
from train_station import load_station, weather, features, models, TEST_START, SHORT

ROOT = pathlib.Path(__file__).parent
WINDOW, CTX = 14, 256
torch.manual_seed(0)


# ---------------- LSTM on daily station data ----------------
def lstm_fit_predict(pm, wx, train_end, predict_idx, seeds=(0, 1, 2)):
    doy = wx.index.day_of_year
    seq = wx[SHORT].copy(); seq.insert(0, "pm25", pm.reindex(wx.index))
    seq["s"], seq["c"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    tm = wx[SHORT].shift(-1)
    tr = seq.index < train_end
    mu, sd = seq[tr].mean(), seq[tr].std() + 1e-6; tmu, tsd = tm[tr].mean(), tm[tr].std() + 1e-6
    S, T = ((seq - mu) / sd).values, ((tm - tmu) / tsd).values
    y = ((seq.pm25.shift(-1) - mu.pm25) / sd.pm25).values
    pos = {t: i for i, t in enumerate(seq.index)}

    def win(t, need_y):
        i = pos[t]
        if i < WINDOW - 1: return None
        w = S[i-WINDOW+1:i+1]
        if np.isnan(w).any() or np.isnan(T[i]).any() or (need_y and np.isnan(y[i])): return None
        return w, T[i], y[i]
    tr_rows = [(t, r) for t in seq.index[tr] if (r := win(t, True))]
    Xs = torch.tensor(np.array([r[0] for _, r in tr_rows]), dtype=torch.float32)
    Xt = torch.tensor(np.array([r[1] for _, r in tr_rows]), dtype=torch.float32)
    Y = torch.tensor(np.array([r[2] for _, r in tr_rows]), dtype=torch.float32)
    pr_rows = [(t, r) for t in predict_idx if (r := win(t, False))]
    Ps = torch.tensor(np.array([r[0] for _, r in pr_rows]), dtype=torch.float32)
    Pt = torch.tensor(np.array([r[1] for _, r in pr_rows]), dtype=torch.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.l = nn.LSTM(S.shape[1], 32, batch_first=True)
            s.h = nn.Sequential(nn.Linear(32 + T.shape[1], 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 1))
        def forward(s, a, b):
            _, (h, _) = s.l(a); return s.h(torch.cat([h[-1], b], 1)).squeeze(1)
    cut, outs = int(len(Y) * .85), []
    for seed in seeds:
        torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
        best, state, pat = 1e9, None, 0
        for ep in range(200):
            net.train(); perm = torch.randperm(cut)
            for b in range(0, cut, 64):
                j = perm[b:b+64]; opt.zero_grad(); nn.functional.l1_loss(net(Xs[j], Xt[j]), Y[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): v = nn.functional.l1_loss(net(Xs[cut:], Xt[cut:]), Y[cut:]).item()
            if v < best - 1e-4: best, state, pat = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat > 20: break
        net.load_state_dict(state); net.eval()
        with torch.no_grad(): outs.append(net(Ps, Pt).numpy() * sd.pm25 + mu.pm25)
    return pd.Series(np.mean(outs, 0), index=[t for t, _ in pr_rows])


# ---------------- Chronos-Bolt zero-shot ----------------
_chronos = None
def chronos_predict(pm, idx):
    """For each day t, forecast t+1 from the PM2.5 history up to and including t. Returns median, p10, p90."""
    global _chronos
    from chronos import BaseChronosPipeline
    if _chronos is None:
        _chronos = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    s = pm.asfreq("D").interpolate(limit=7).ffill()
    ctx = [torch.tensor(s.loc[:t].tail(CTX).values, dtype=torch.float32) for t in idx]
    ctx = torch.nn.utils.rnn.pad_sequence([c.flip(0) for c in ctx], batch_first=True, padding_value=float("nan")).flip(1)
    q, m = _chronos.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])
    return pd.DataFrame({"med": q[:, 0, 1].numpy(), "p10": q[:, 0, 0].numpy(), "p90": q[:, 0, 2].numpy()}, index=idx)


def main():
    emb = load_station("embassy")
    wx = weather("2020-03-25", "2026-11-14")
    X, y = features(emb, wx); fs = list(X.columns)
    data = X.join(y).dropna()
    train, test = data[data.index < TEST_START], data[data.index >= TEST_START]
    cut = json.load(open(ROOT / "station_results.json"))["warning_cut"]

    P = {"Persistence": test.pm25_today}
    fitted = {}
    for name, m in models().items():
        fitted[name] = m.fit(train[fs], train.target); P[name] = pd.Series(m.predict(test[fs]), index=test.index)
    print("LSTM ..."); P["LSTM"] = lstm_fit_predict(emb, wx, TEST_START, test.index)
    print("Chronos ..."); ch = chronos_predict(emb.loc[:"2025-11-14"], test.index); P["Chronos-Bolt (zero-shot)"] = ch.med
    common = test.index
    for k in P: common = common.intersection(P[k].dropna().index)
    MEMBERS = ["Ridge", "Gradient boosting", "LSTM", "Chronos-Bolt (zero-shot)"]
    P["Ensemble (average)"] = pd.concat([P[k].loc[common] for k in MEMBERS], axis=1).mean(axis=1)

    yt = test.target.loc[common].values; heat = test.heating.loc[common].values == 1
    smog = yt > LIMIT
    board = []
    for k, p in P.items():
        p = p.loc[common].values
        board.append({"model": k, "mae": round(mean_absolute_error(yt, p), 2),
                      "mae_heating": round(mean_absolute_error(yt[heat], p[heat]), 2),
                      "mae_smog": round(float(np.abs(yt - p)[smog].mean()), 2),
                      **decision_stats(yt, p, LIMIT if k == "Persistence" else cut)})
    inside = ((yt >= ch.p10.loc[common].values) & (yt <= ch.p90.loc[common].values)).mean()
    # Trust score: rank on the three things a school cares about (lower is better), averaged
    df = pd.DataFrame(board)
    df["trust_rank"] = (df.mae_heating.rank() + df.mae_smog.rank() + (-df.recall).rank() + df.false_alarms.rank()) / 4
    df = df.sort_values("trust_rank")
    R = {"n_test": len(common), "test_period": [str(common[0].date()), str(common[-1].date())], "cut": cut,
         "board": df.to_dict("records"), "most_trusted": df.iloc[0].model, "chronos_80_coverage": round(float(inside), 3),
         "series": {"date": [str((t + pd.Timedelta(days=1)).date()) for t in common], "actual": yt.round(1).tolist(),
                    **{k: P[k].loc[common].round(1).tolist() for k in P},
                    "chronos_p10": ch.p10.loc[common].round(1).tolist(), "chronos_p90": ch.p90.loc[common].round(1).tolist()}}
    # agreement: how often do models agree on INDOORS/not
    votes = np.array([(P[k].loc[common].values > cut) for k in MEMBERS])
    agree = (votes.all(0) | (~votes).all(0))
    R["agreement"] = {"all_agree_share": round(float(agree.mean()), 3),
                      "mae_when_agree": round(float(np.abs(yt - P["Ensemble (average)"].loc[common].values)[agree].mean()), 2),
                      "mae_when_disagree": round(float(np.abs(yt - P["Ensemble (average)"].loc[common].values)[~agree].mean()), 2)}

    # ---------------- live: all models on the same fetched School No.182 data ----------------
    live = requests.get("https://api.openaq.org/v3/sensors/14840476/hours", headers={"X-API-Key": os.environ["OPENAQ_API_KEY"]},
                        params=dict(datetime_from=str((pd.Timestamp.now() - pd.Timedelta(days=30)).date()), limit=1000), timeout=60).json()["results"]
    hr = pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["local"][:19]): x["value"] for x in live}).sort_index()
    daily = hr.resample("D").mean()
    wl = weather(None, None, url="https://api.open-meteo.com/v1/forecast", past_days=30, forecast_days=2)
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    base = today if hr.index.max().normalize() == today and hr.index.max().hour >= 18 else today - pd.Timedelta(days=1)
    Xl, _ = features(daily, wl)
    allpm = pd.concat([emb, load_station("school182"), daily]).groupby(level=0).last().sort_index()
    allwx = pd.concat([wx, wl]).groupby(level=0).last().sort_index()
    finals = {n: models()[n].fit(data[fs], data.target) for n in models()}
    lp = {"Persistence": float(daily.loc[base])}
    for n, m in finals.items(): lp[n] = float(m.predict(Xl.loc[[base], fs])[0])
    lp["LSTM"] = float(lstm_fit_predict(allpm.loc[:"2025-11-14"].combine_first(daily), allwx, "2025-11-15", [base]).iloc[0])
    chl = chronos_predict(daily.loc[:base].combine_first(allpm.loc[:base]), [base])
    lp["Chronos-Bolt (zero-shot)"] = float(chl.med.iloc[0])
    lp["Ensemble (average)"] = float(np.mean([lp[k] for k in MEMBERS]))
    R["live"] = {"based_on": str(base.date()), "for": str((base + pd.Timedelta(days=1)).date()), "cut": cut,
                 "today": round(lp["Persistence"], 1), "pred": {k: round(v, 1) for k, v in lp.items()},
                 "chronos_band": [round(float(chl.p10.iloc[0]), 1), round(float(chl.p90.iloc[0]), 1)],
                 "votes_indoors": int(sum(lp[k] > cut for k in MEMBERS)), "n_models": len(MEMBERS)}
    json.dump(R, open(ROOT / "compare_results.json", "w"), indent=1, default=str)
    print(df[["model", "mae", "mae_heating", "mae_smog", "caught", "bad_days", "false_alarms", "trust_rank"]].to_string(index=False))
    print("chronos 80% band coverage:", R["chronos_80_coverage"], "| agreement:", R["agreement"])
    print("LIVE", R["live"])


if __name__ == "__main__":
    main()
