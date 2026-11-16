# Almaty Smog Desk: should recess be outside tomorrow?

**Track:** Climate
**Team:** _[names]_

**Demo page:** _[GitHub Pages link]_ · **Video:** _[link]_

> This project is a prototype and is not medical advice.

## The problem

Every heating season (October–April), Almaty's mountain-bowl geography traps coal smoke from heating under cold-air inversions, and PM2.5 climbs to levels that are unhealthy for children. The person who decides whether tomorrow's PE lesson and recess happen outdoors is a school duty officer or PE teacher. They make that call the afternoon before, usually by guessing from today's sky. Children are a *sensitive group*, so a wrong "outdoors" call means 25 kids exercising in bad air.

## What we built

A next-day PM2.5 forecast for central Almaty that turns into a three-level school decision.

- **In:** today's hourly PM2.5, the last 7 days of PM2.5, today's weather and **tomorrow's weather forecast** (all fetched automatically from Open-Meteo).
- **Out:** tomorrow's daily-mean PM2.5 and a verdict: **INDOORS** (forecast above 34.0 µg/m³), **OUTDOORS, LIGHT** (15–34) or **OUTDOORS** (under 15).

```
$ python predict.py
Almaty, forecast for 2026-11-17
  Today's PM2.5:      24.4 ug/m3
  Tomorrow forecast:  30.2 ug/m3  (warn above 34.0)
  Decision:           OUTDOORS, LIGHT  - Outdoor recess is fine. Keep long or intense PE indoors for children with asthma.
```

The demo page (`docs/index.html`) shows the live forecast card, all results, the forecast-vs-reality chart and the failure cases.

## What we added

We started from the suggested project **4.3.2 Next-day air quality forecast** and added the following.

1. **Our own setting and user:** Almaty, where we live, and a specific decision: a school moving PE and recess indoors.
2. **A decision-based evaluation.** We don't stop at error scores. We count smog days caught, missed and falsely alarmed, and we picked the warning cut-off on a *separate validation season*. We weighted a miss as 3× worse than a false alarm, because a missed smog day harms children and a false alarm only costs one outdoor lesson.
3. **Our own question: does a deep sequence model (LSTM) help where it matters?** It wins on average but **loses on smog days**, so we deploy the simpler Ridge model.
4. **A live tool** (`predict.py`) that issues tomorrow's verdict from today's data.

## How it works

**Data**
- PM2.5: Open-Meteo Air Quality API (Copernicus CAMS model, hourly) for 43.24°N, 76.89°E, Aug 2022 – Nov 2026.
- Weather: Open-Meteo Historical Forecast API, hourly: temperature, humidity, wind, precipitation, pressure and cloud cover.
- Aggregated to **1,558 daily rows**. Two rows with missing PM2.5 were dropped. We first included boundary-layer height, but it only exists from Sep 2024, which silently cut our dataset in half, so we removed it.

**Features (all known by the afternoon of day *t*)**
- Today's mean, peak and evening (18–24h) PM2.5; yesterday's PM2.5; the 7-day mean; today's trend.
- Today's weather and tomorrow's weather forecast.
- Season (sine/cosine of day of year), a heating-season flag and a weekend flag.

**Split**

Split by date, never shuffled:

| | Period | Days |
|---|---|---|
| Train | 2022-08-10 → 2025-09-30 | 1,148 |
| Validation (cut-off choice only) | 2024-10-01 → 2025-09-30 (inside train) | |
| **Test** | **2025-10-01 → 2026-11-14** | **410** |

The test period covers one full heating season and the start of the current one.

**Models compared**

- Persistence baseline ("tomorrow = today") and a 7-day-average baseline.
- Ridge regression: chosen for deployment because it is simple, explainable and best on smog days.
- Gradient boosting (scikit-learn HistGradientBoosting).
- **LSTM** (PyTorch): reads a 14-day sequence of PM2.5 and weather, combined with tomorrow's weather forecast. It has one LSTM layer (32 units), is trained with L1 loss and early stopping on the last 15% of the training period, and averages 3 random seeds.

## Results

All numbers are on the 410 test days the models never saw.

**Forecast error** (mean absolute error, µg/m³; lower is better)

| Model | All test days | Heating season |
|---|---|---|
| Persistence (tomorrow = today) · *baseline* | 4.39 | 5.24 |
| 7-day average · *baseline* | 4.77 | 5.53 |
| Ridge, today's data only | 3.73 | 4.43 |
| Gradient boosting, today's data only | 3.96 | 4.72 |
| **Ridge + tomorrow's weather forecast (deployed)** | **3.63** | **4.24** |
| Gradient boosting + tomorrow's weather forecast | 3.88 | 4.60 |
| **LSTM + tomorrow's weather forecast** | **3.48** | **4.00** |

The deployed model beats the persistence baseline by **17%** and the LSTM by **21%**. Adding tomorrow's weather forecast helps every model. Gradient boosting does *worse* than plain linear regression, probably because of the small dataset (about 1,100 training days).

**The school decision** (45 smog days in the test period, where the daily mean was above 35.4 µg/m³)

| Rule | Smog days caught | Missed | False alarms |
|---|---|---|---|
| Persistence: warn if today > 35.4 | 21 / 45 | 24 | 24 |
| Ridge, warn if forecast > 35.4 | 28 / 45 | 17 | 16 |
| **Ridge, warn if forecast > 34.0 (cut-off chosen on validation)** | **35 / 45** | **10** | **25** |
| LSTM, warn if forecast > 34.0 | 28 / 45 | 17 | 15 |

With the same number of false alarms as the baseline (25 vs 24), our tool catches **14 more smog days** and cuts missed smog days from 24 to 10.

**LSTM vs Ridge: error by how polluted the air actually was**

| Model | Clean (< 15) | Moderate (15–35) | Smog (> 35) |
|---|---|---|---|
| Persistence | 2.80 | 4.46 | 7.22 |
| Ridge (deployed) | 2.32 | 3.84 | **4.97** |
| LSTM | **2.26** | **3.36** | 6.69 |

The LSTM smooths out spikes. That makes it great on ordinary days but worse on exactly the days a school cares about, so the "cool" model isn't the right one here.

**What drives the forecast** (permutation importance): today's evening PM2.5 (the heating peak), then tomorrow's minimum and mean temperature (cold nights mean more coal burning and stronger inversions), the 7-day average and air pressure.

**Where it fails**

| Day | Actual | Model | Persistence | Why |
|---|---|---|---|---|
| 2026-01-01 | 15.2 | 37.5 | 50.6 | 7.5 mm of snow/rain and wind washed the air clean overnight. The model still trusted the high evening PM2.5. |
| 2025-12-31 | 50.6 | 31.0 | 33.4 | New Year's Eve: fireworks and holiday heating. Nothing in the weather predicts it. |
| 2026-05-02 | 39.6 | 23.0 | 22.2 | A spring spike outside the heating season, probably dust carried by 6 km/h wind. The model has learned that "May = clean". |
| 2025-12-07 | 26.8 | 42.9 | 48.2 | A sudden clean-up that persistence also missed. |

In short, the model reacts a day late to sudden breaks in the weather and can't know about one-off events like fireworks.

**Honest limits**
- PM2.5 here is a **model estimate (CAMS, about 11 km grid), not station measurements**. Almaty's ground stations often read much higher on winter evenings, especially in the lower districts. Our absolute levels are likely too low, and the next step is retraining on real station data.
- In training, "tomorrow's weather" is the archived forecast/analysis, which is a bit more accurate than a real day-ahead forecast, so live errors will be somewhat larger.
- We tested one location and one heating season plus the current one.

## What we'd do next

1. Train and test on **real ground-station measurements** (Kazhydromet or the OpenAQ stations in Almaty) and compare with CAMS's own next-day forecast as a stronger baseline.
2. Add **inversion strength** (the temperature difference between the city and the mountain stations) and run a separate forecast per district, since the lower, northern districts suffer most.

## How to run

```bash
pip install -r requirements.txt
python train.py        # downloads data, trains, writes results.json + model.pkl (about 40 s on a laptop CPU)
python predict.py      # tomorrow's forecast + school verdict
python build_site.py   # rebuilds docs/index.html demo page
```

## Sources

- Open-Meteo Air Quality API (Copernicus CAMS data): https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Forecast API (live mode): https://open-meteo.com/en/docs
- US EPA PM2.5 AQI breakpoints (35.4 µg/m³ = top of "Moderate"): https://www.epa.gov/aqi
- Libraries: scikit-learn, PyTorch, pandas, NumPy, requests; Chart.js for the demo page.
- AI tools: **Claude (Anthropic)** was used as a coding assistant to write and debug code and structure the write-up. We reviewed and can explain every part.
- Starting point: Veritas AI Hackathon 2026 Project Briefs, suggested project 4.3.2.
