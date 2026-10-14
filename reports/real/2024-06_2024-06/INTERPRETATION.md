# Interpretation (auto-generated from saved results)
**Run:** `[REAL] 2024-06..2024-06`

## Data and cleaning
- 4,783,576 rows parsed from 5 CSV file(s); 4,775,401 kept (0.2% excluded). See `exclusion_report.csv` for the per-rule waterfall.

## Model A - trip duration (minutes)
Backtest: 4 rolling-origin folds on a 1,000,000-trip random sample (mean ± sd across folds).

| model | MAE | RMSE | R2 | WAPE_% |
|---|---|---|---|---|
| Baseline: distance x median pace | 5.026 ± 0.406 | 13.012 ± 0.772 | 0.208 ± 0.021 | 36.452 ± 1.312 |
| Baseline: global median | 8.127 ± 0.447 | 15.110 ± 0.818 | -0.068 ± 0.019 | 58.996 ± 0.637 |
| Baseline: median by user x bike | 7.911 ± 0.406 | 14.709 ± 0.744 | -0.012 ± 0.011 | 57.436 ± 0.459 |
| Lasso | 5.026 ± 0.420 | 12.625 ± 0.731 | 0.255 ± 0.017 | 36.448 ± 1.514 |
| Linear (OLS) | 5.025 ± 0.421 | 12.625 ± 0.730 | 0.254 ± 0.017 | 36.445 ± 1.520 |
| Poly2+Ridge | 5.137 ± 0.425 | 12.759 ± 0.747 | 0.239 ± 0.019 | 37.257 ± 1.457 |
| Ridge | 5.025 ± 0.421 | 12.625 ± 0.730 | 0.254 ± 0.017 | 36.445 ± 1.520 |

- OLS does not meaningfully improve on the best baseline (+0.0% MAE) ('Baseline: distance x median pace').
- Nominal 80% prediction interval covered 78% of trips in the last test block (reasonably calibrated).
- Low R2 is expected: duration is driven by behaviour (commute vs leisure) that these features do not observe.

## Model B - hourly station demand
### Horizon 1 h (150 stations)

| model | MAE | RMSE | WAPE_% | PoissonDeviance | bias |
|---|---|---|---|---|---|
| Baseline: persistence (lag h) | 5.291 ± 0.746 | 7.834 ± 1.204 | 42.238 ± 1.430 | 5.803 ± 0.537 | 0.052 ± 0.077 |
| Baseline: same hour last week | 5.240 ± 0.411 | 7.899 ± 0.680 | 42.216 ± 4.956 | 6.401 ± 1.155 | 0.261 ± 0.563 |
| Baseline: same hour yesterday | 5.590 ± 0.532 | 8.397 ± 0.943 | 45.406 ± 8.473 | 7.005 ± 0.682 | 0.452 ± 1.239 |
| Baseline: station mean | 8.620 ± 0.520 | 11.176 ± 1.204 | 69.317 ± 5.267 | 9.862 ± 0.928 | 0.052 ± 1.534 |
| Baseline: station x hour-of-week mean | 4.780 ± 0.531 | 7.192 ± 0.901 | 38.629 ± 6.451 | 4.401 ± 1.528 | 0.453 ± 0.612 |
| Linear (OLS) | 4.130 ± 0.291 | 5.995 ± 0.535 | 33.192 ± 2.415 | 3.003 ± 0.279 | -0.035 ± 0.411 |
| Ridge | 4.130 ± 0.291 | 5.995 ± 0.535 | 33.191 ± 2.415 | 3.002 ± 0.279 | -0.035 ± 0.412 |

- OLS improves MAE by 13.6% over the best baseline ('Baseline: station x hour-of-week mean').
- Share of zero-demand station-hours in test blocks: 12% (why MAPE is not used).

### Horizon 24 h (150 stations)

| model | MAE | RMSE | WAPE_% | PoissonDeviance | bias |
|---|---|---|---|---|---|
| Baseline: persistence (lag h) | 5.590 ± 0.532 | 8.397 ± 0.943 | 45.406 ± 8.473 | 7.005 ± 0.682 | 0.452 ± 1.239 |
| Baseline: same hour last week | 5.240 ± 0.411 | 7.899 ± 0.680 | 42.216 ± 4.956 | 6.401 ± 1.155 | 0.261 ± 0.563 |
| Baseline: station mean | 8.619 ± 0.519 | 11.181 ± 1.210 | 69.306 ± 5.151 | 9.875 ± 0.932 | 0.015 ± 1.521 |
| Baseline: station x hour-of-week mean | 4.780 ± 0.531 | 7.192 ± 0.901 | 38.629 ± 6.451 | 4.401 ± 1.528 | 0.453 ± 0.612 |
| Linear (OLS) | 4.550 ± 0.322 | 6.636 ± 0.636 | 36.605 ± 3.341 | 3.713 ± 0.501 | 0.138 ± 0.323 |
| Ridge | 4.550 ± 0.322 | 6.635 ± 0.636 | 36.603 ± 3.342 | 3.710 ± 0.501 | 0.139 ± 0.323 |

- OLS improves MAE by 4.8% over the best baseline ('Baseline: station x hour-of-week mean').
- Share of zero-demand station-hours in test blocks: 12% (why MAPE is not used).

## Weather (associations only)
Daily log-trips regressed on weather, weekend, holiday and trend (n = 30 days, HAC errors). Status: **exploratory only (<60 days)**.

```
              term  pct_effect  pct_ci95_low  pct_ci95_high
             const  19334752.5    15846445.2     23590942.4
       temp_per_5C        -2.5          -6.1            1.3
precip_per_5mm_day        -2.7          -4.7           -0.6
    wind_per_10kmh        -0.4          -8.2            8.1
        is_weekend        -9.3         -15.6           -2.5
        is_holiday         4.2          -0.3            9.0
     trend_per_30d        -5.5         -16.1            6.3
```

These are associations in this sample. Temperature is entangled with season, ridership growth and calendar effects; no causal effect of weather is claimed.

## Limitations
- See the Assumptions listed in `model_results.json` and the README 'Known limitations' section.
