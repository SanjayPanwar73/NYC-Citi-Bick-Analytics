# NYC Citi Bike: demand and trip-duration analysis

> **STATUS: PARTIAL.** Section 3 holds preliminary descriptive results from the user's real run (June 2024). Cleaning waterfall and model
> results are still **pending** and contain no numbers. Fill them only from `reports/real/<run>/INTERPRETATION.md`, `EDA_insights.md`, `data/exclusion_report.csv`,
> and `data/model_results.json`. Never copy anything from `reports/sample/` (synthetic).

## 1. Data and provenance  *(fill from `run_manifest.json`)*
- Source: Citi Bike official NYC trip data · range: `<START>`–`<END>` · files: `<n>` · weather: Open-Meteo archive, NYC centre point
- Rows parsed → kept: `<…>` → `<…>` (`<x>`% excluded; top reasons: `<…>`) · trips without weather match: `<…>`%
- Package versions: from the manifest.

## 2. Cleaning decisions
Rules and handling of missing coordinates are listed in README ("Cleaning rules"). Report the exclusion waterfall and
the share of kept trips with missing end coordinates (they are excluded from the duration model only).

## 3. Descriptive findings - REAL run `2024-06_2024-06` (from the run's printed EDA notes; descriptive only)
- **Coverage:** 4,775,401 trips over 30 days (2024-06-01 to 2024-06-30) from 2,272 start stations; every trip has a weather match (0.000% missing).
- **Daily volume:** busiest day 2024-06-01 (172,686 trips); quietest 2024-06-30 (107,221).
- **Typical trip:** median 9.9 min and 1.58 km straight-line; 99.8% of sampled trips have a distance.
- **Hourly shape:** weekday peak at 17:00 (~15,957 trips/hour on an average weekday).
- **Riders:** members make 77% of trips; median duration 8.9 min (members) vs 14.1 min (casual).
- **Stations:** busiest start station West St & Chambers St (18,152 starts). Largest imbalance ratio among above-median-activity stations:
  St Nicholas Ave & W 134 St (+0.77). Trip counts only - not bike inventory, and rebalancing trucks are invisible.
- *Still to add:* interpretation of figures 01-11 after viewing them (seasonality cannot be assessed from one month).

## 4. Weather - associations only
Preliminary: daily log-trips regression (weekend, holiday, trend; HAC errors; n = 30 days) gives +5 C associated with -2.5% daily trips,
95% CI -6.1% to +1.3%. The interval includes zero and the window is a single month, so this is **exploratory only** and shows no
detectable association; temperature cannot be separated from trend/season. Add the hourly relative-demand table (rain/temperature bins)
from `rain_ratio.csv` / `temp_ratio.csv`. Use "associated with", never "caused".

## 5. Models
- Design: rolling-origin backtest (4 folds), horizon 1 h and 24 h, gap = horizon, train-only baselines and station selection.
- Tables: mean ± sd across folds for MAE/RMSE/R²/WAPE (A) and MAE/RMSE/WAPE/sMAPE/Poisson deviance/bias (B).
- Skill vs the **best baseline** (not only the mean predictor); interval coverage for Model A; residual and calibration plots.
- State assumptions verbatim from `model_results.json` (known destination, observed weather as forecast stand-in, censored demand).

## 6. Limitations and next steps
Copy README "Known limitations". Next: borough polygons, routed distance, gradient boosting/Poisson models, more months/seasons,
historical station capacity, weather forecasts instead of observations.
