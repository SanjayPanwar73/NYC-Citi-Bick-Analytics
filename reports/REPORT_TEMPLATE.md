# NYC Citi Bike: What Drives Demand and Trip Duration?

## 1. Summary (3-4 sentences)
Data: Citi Bike NYC trips, <period>, <N> rides after cleaning, joined with hourly Open-Meteo weather.
Key finding 1: ...  Key finding 2: ...  Best model: ... (MAE ..., R2 ...).

## 2. Data & cleaning
- Source / scope / row counts: `<raw> -> <final>` (<x>% removed)
- Rules: duration 60 s-6 h, speed <= 40 km/h, NYC bounding box, de-dup on ride_id, schema mapping
- Issues found: ...

## 3. EDA highlights (embed 4-5 best figures)
1. Seasonality: ...
2. Commute peaks vs weekend leisure: ...
3. Members vs casual: ...
4. Station imbalance / rebalancing implications: ...
5. Weather: rain effect ..., temperature effect ...

## 4. Modeling
| Model | Target | MAE | RMSE | R2 | MAPE |
|---|---|---|---|---|---|
| Baseline | | | | | |
| Linear | | | | | |
| Ridge | | | | | |
| Naive last-week (demand) | | | | | |
Residual diagnostics: ...   Strongest coefficients: ...   Error analysis: ...

## 5. Business implications
- Rebalancing: ...   - Pricing / promotions: ...   - Capacity planning in bad weather: ...

## 6. Limitations & next steps
Straight-line distance; single-point weather; linear assumptions; next: gradient boosting, station capacity, routed distance, longer history.
