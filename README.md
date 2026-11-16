# Almaty Smog Desk: should recess be outside tomorrow?

**Track:** Climate
**Team:** Damir Kamalov (11th grade, Almaty, Kazakhstan)

**Demo site:** _[GitHub Pages link]_ · **Video:** _[link]_

> This project is a prototype and is not medical advice.

## The problem

Every heating season (October–April), Almaty's mountain-bowl geography traps coal smoke under cold-air inversions, and PM2.5 regularly goes above 100 µg/m³. The person who decides whether tomorrow's PE lesson and recess happen outdoors is a school duty officer or PE teacher, and they decide the afternoon before. Children are a *sensitive group*, and the free air-quality data most apps show (CAMS model estimates) **misses 43% of Almaty's real smog days**.

## What we built

A next-day PM2.5 forecast trained on **real measurements**, which becomes a three-level school decision.

- **In:** a school's recent PM2.5 sensor readings (OpenAQ), today's weather and tomorrow's weather forecast (Open-Meteo).
- **Out:** tomorrow's daily-mean PM2.5 and a verdict: **INDOORS** (above 37.5 µg/m³), **OUTDOORS, LIGHT** (15–37.5) or **OUTDOORS** (under 15).
- **Live example:** from School No.182's sensor data up to 15 Nov 2026, the ensemble forecast for 16 Nov is **43.3 µg/m³, INDOORS**. Only 2 of 4 models voted INDOORS, so the tool flags low confidence.
- **City map:** the same models forecast tomorrow at **all 56 live OpenAQ sensors in Almaty**. Each sensor's Voronoi zone is coloured by US EPA category, so parents and schools can see which districts will be safe for outdoor play.

The demo site (6 pages, in `docs/`) has these pages:
- **Today:** the live verdict, the last 72 hours of the school sensor and the last 8 days.
- **City map:** live hazard zones across Almaty, switchable between now, yesterday and tomorrow, with every model's prediction for each sensor.
- **Model showdown:** a leaderboard, all models' live forecasts side by side and a test-period chart you can toggle.
- **Real-station results:** the test results, explained.
- **Model data vs reality:** how far the CAMS estimates are from real measurements.
- **How it works:** the pipeline, the model weights and the limits.

## What we added

We started from suggested project **4.3.2 Next-day air quality forecast** and added the following.

1. **Our own city and user:** Almaty, where we live, and a school deciding whether PE and recess move indoors.
2. **Our own data:** real PM2.5 from the **US Embassy reference monitor** (2020–2025) and a test on a **different sensor at School No.182** that the model never saw (Dec 2025 – Nov 2026).
3. **A decision-based evaluation.** We count smog days caught, missed and falsely alarmed. A miss counts as 3× worse than a false alarm, because children breathe bad air on a miss, while a false alarm costs one outdoor lesson. The warning cut-off was chosen on a validation season, never on the test data.
4. **Our own questions:**
   - How wrong is the free model data compared with reality? (Very: it misses 43% of smog days.)
   - Does a deep LSTM help where it matters? (It wins on average but loses on smog days.)

## How it works

**Data**
- **OpenAQ v3, US Embassy Almaty (AirNow reference monitor):** 1,752 daily values, Apr 2020 – Nov 2025. We kept the 1,501 days with at least 18 of 24 hours measured.
- **OpenAQ v3, School No.182 (AirGradient low-cost sensor):** Dec 2025 – Nov 2026.
- **Open-Meteo ERA5 archive (training) and forecast API (live):** daily temperature, minimum temperature, humidity, wind, rain/snow, pressure and cloud.
- **Open-Meteo Air Quality API (CAMS model):** hourly, Aug 2022 – Nov 2026, used for the reality check and the LSTM experiment.

**Features (all known on day *t*)**
- PM2.5 today and yesterday, the 7-day mean and the change since yesterday.
- Today's weather and tomorrow's forecast.
- Season, a heating-season flag and a weekend flag.

**Split by date, never shuffled**

| | Period | Days |
|---|---|---|
| Train | Apr 2020 – Sep 2024 | 804 |
| Validation (cut-off choice only) | Oct 2023 – Sep 2024 (inside train) | |
| **Test** | **Oct 2024 – Nov 2025** | **391** |
| **Second test (new sensor)** | **School No.182, Dec 2025 – Nov 2026** | **129** |

**Models**
- Persistence baseline ("tomorrow = today") and a 7-day-average baseline.
- **Ridge regression (deployed).**
- Gradient boosting.
- In the CAMS experiment, a **PyTorch LSTM** that reads 14-day sequences.

## Results

**1. Embassy reference monitor, 391 held-out days**

Mean absolute error in µg/m³; lower is better.

| Model | All test days | Heating season |
|---|---|---|
| Persistence (tomorrow = today) · *baseline* | 9.88 | 13.93 |
| 7-day average · *baseline* | 10.32 | 14.52 |
| Gradient boosting | 9.32 | 13.33 |
| **Ridge (deployed)** | **9.24** | **11.92 (14% lower)** |

The school decision (88 real smog days, daily mean above 35.4 µg/m³):

| Rule | Caught | Missed | False alarms |
|---|---|---|---|
| Persistence: warn if today > 35.4 | 61 / 88 | 27 | 27 |
| Ridge, warn if forecast > 35.4 | 75 / 88 | 13 | 28 |
| **Ridge, warn if forecast > 37.5 (cut-off chosen on validation)** | **72 / 88** | **16** | **25** |

With fewer false alarms than the baseline, the model catches **11 more smog days per year**.

**2. School No.182: a new sensor, area and time period** (model frozen, never trained on it)

| | Error | Smog days caught | False alarms |
|---|---|---|---|
| Persistence | 15.37 | 59 / 68 | 10 |
| **Our model** | **13.63** | **60 / 68** | **8** |

The model still beats the baseline on a cheaper sensor in a different neighbourhood. Here the gain in the decision itself is small, because winter 2025–26 at this school had long, steady smog runs that persistence also handles well.

**3. Free model data vs reality** (1,057 days with both the embassy reading and CAMS)

- The embassy monitor recorded **208 smog days**; CAMS shows only **119**, so it **misses 43%**.
- CAMS reads 4.2 µg/m³ lower on average, with a typical daily error of 11.2 µg/m³ and a correlation of 0.65.
- It's close in summer and far too low in winter, which is exactly when it matters.

**4. Model showdown on real station data** (embassy monitor, 391 shared test days, cut-off 37.5)

| Rank | Model | Error, all | Heating season | Smog days (error) | Caught / 88 | False alarms |
|---|---|---|---|---|---|---|
| 1 | **Ensemble (average of the 4 below)** | **8.36** | **11.91** | **15.28** | 73 | 25 |
| 2 | LSTM (PyTorch, 14-day sequence) | 8.82 | 12.59 | 18.25 | **75** | **19** |
| 3 | Ridge | 9.44 | 12.43 | 16.12 | 72 | 25 |
| 4 | Gradient boosting (histogram GBM, XGBoost-style) | 9.63 | 14.08 | 17.08 | 75 | 31 |
| 5 | Chronos-Bolt (Amazon foundation model, **zero-shot**) | 9.80 | 14.22 | 19.31 | 59 | 30 |
| – | Persistence · *baseline* | 10.17 | 14.66 | 20.49 | 61 | 27 |

- **Ranking:** we rank by the average of four ranks: heating-season error, smog-day error, smog days caught and false alarms.
- **The ensemble cuts error by 18%** compared with the baseline. The ensemble was fixed in advance and not tuned on the test data.
- **Disagreement is a warning sign.** All 4 models agreed on the INDOORS/OUTDOORS call on 86% of days, and then the error was 7.69. When they disagreed, it was 12.43. The tool shows how many models vote INDOORS.
- **Chronos's single-number forecast is weak** without training, but its 80% range contained the real value **80.5%** of the time, so its uncertainty is well calibrated.

**5. LSTM experiment (CAMS data, 410 test days)**

| Model | All days | Smog days (> 35) | Smog days caught |
|---|---|---|---|
| Persistence | 4.39 | 7.22 | 21 / 45 |
| Ridge | 3.63 | **4.97** | **35 / 45** |
| LSTM (14-day sequence) | **3.48** | 6.69 | 28 / 45 |

The LSTM has the best average but smooths out spikes, so it does worse on the days a school cares about. We choose the model by the decision it supports.

**What drives the forecast** (Ridge weights on standardised inputs):
- **Tomorrow's temperature is the strongest input, at −17.4.** Colder means more coal burning and stronger inversions.
- Today's temperature and air pressure come next. High pressure means stagnant air.
- Today's and yesterday's PM2.5 follow.

**Where it fails**

| Day | Measured | Forecast | Why |
|---|---|---|---|
| 2024-12-25 | 127.0 | 76.9 | An extreme spike on a still, freezing day (−8.8 °C, 2.8 km/h wind). A linear model pulls extremes toward the average. The verdict was still INDOORS. |
| 2025-02-03 | 125.0 | 75.1 | Same pattern: −9.2 °C, a multi-day inversion. |
| 2025-01-15 | 23.3 | 74.9 | A sudden clean-up (89.6 → 23.3) with no rain. The inversion broke, which daily-average weather doesn't capture. This was a false alarm. |

**Honest limits**
- The embassy monitor stopped reporting in Nov 2025, so the main test period ends there.
- The school sensor is low-cost and less accurate.
- Training uses actual (reanalysis) weather for "tomorrow", but live use relies on a forecast, so live errors will be somewhat larger.

## What we'd do next

1. Add **inversion strength** (the temperature difference between the city and a mountain station such as Shymbulak) and hourly wind, so the model can catch the breaks and the extreme spikes.
2. Forecast **per district** using the dozens of school sensors in OpenAQ, and send each school its own verdict at 3 PM by Telegram.

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env            # then paste your free OpenAQ key (openaq.org) into .env
python train.py                 # CAMS data: baselines, Ridge, gradient boosting, LSTM -> results.json
python predict.py               # live CAMS-based forecast -> forecast.json
python openaq_fetch.py          # real station data -> data/openaq_*.csv
python train_station.py         # real-station models, school test, CAMS vs reality, live school forecast
python compare_models.py        # 5-model showdown + ensemble + live forecasts (downloads Chronos-Bolt, ~200 MB)
python map_forecast.py          # forecast at every live Almaty sensor for the city map
python build_site.py            # builds the 6-page site in docs/
```

## Sources

- OpenAQ API v3 (US Embassy Almaty via AirNow; AirGradient sensor at School No.182): https://openaq.org
- Open-Meteo: Air Quality API (Copernicus CAMS), Historical Forecast API, ERA5 Archive API and Forecast API: https://open-meteo.com
- US EPA PM2.5 AQI breakpoints: https://www.epa.gov/aqi
- Chronos-Bolt (Amazon, pretrained time-series model): https://huggingface.co/amazon/chronos-bolt-small
- Map: MapLibre GL JS, OpenFreeMap tiles (© OpenMapTiles, data © OpenStreetMap contributors), d3-delaunay.
- Libraries: scikit-learn, PyTorch, chronos-forecasting, pandas, NumPy, requests, Chart.js.
- AI tools: **Claude (Anthropic)** was used as a coding assistant to write and debug code and structure this write-up. We reviewed and can explain every part.
- Starting point: Veritas AI Hackathon 2026 Project Briefs, suggested project 4.3.2.
