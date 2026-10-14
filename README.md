# NYC Citi Bike - trips, weather, and linear-regression forecasting
# Live Linl :- https://nyc-citi-bick-analytics.streamlit.app/
> **Status:** the code is complete and unit-tested at the Python level. **No real-data run has been completed yet**
> (the first real download failed with an invalid archive). Any figures/metrics produced by `python run_all.py sample`
> are **synthetic** and are written to separate, watermarked folders. See `AUDIT.md` for what was fixed and what is unverified.

## Quick start (Windows cmd; PowerShell: use `$env:START_MONTH="2024-06"` instead of `set`)
```bat
cd path\to\citibike-project                  :: the folder that contains run_all.py and src\
python -m venv .venv
.venv\Scripts\activate                        :: prompt must show (.venv) - redo this in every new terminal
python -m pip install -r requirements.txt     :: always `python -m pip` (bare pip.exe can be blocked by policy)
python run_all.py check                       :: unit tests, no data needed
set START_MONTH=2024-06
set END_MONTH=2024-06                         :: set these in EVERY new terminal, or the default range is used
python run_all.py status                      :: which stages are done / what to run next
python run_all.py all                         :: REAL data; resumable - finished stages are skipped (--force to redo)
python -m streamlit run app/dashboard.py
python run_all.py sample                      :: SYNTHETIC end-to-end test (separate folders, watermarked)
```
Each stage can also be run alone, e.g. `python src/modeling.py`. A first real month is ~1 GB to download (936 MB for 2024-06) and
~4.8 M trips; allow several GB of disk. For seasonality use several months (e.g. 2024-05..2024-08), one download per month.
If legacy output from before the real/sample split exists: `python src/quarantine_legacy.py` (dry run) then `--apply`.
It **moves** (never deletes) unlabeled derived files and synthetic-looking raw CSVs into `data/_legacy_unlabeled_<ts>/`.

## Dashboard
`python -m streamlit run app/dashboard.py` - dark, card-based UI (theme in `.streamlit/config.toml`; Altair + pydeck ship with Streamlit).
Tabs: **Overview** (KPI cards, daily trips, rider mix, key facts) · **Rhythm** (hourly curves, weekly heatmap, member vs casual) ·
**Stations** (interactive map with dots or 3D density, rankings, search, routes) · **Weather** (relative-demand bars, adjusted-association forest plot,
association warnings) · **Models** (backtest comparison with error bars, fold stability, skill vs baseline) · **Predict** (all inputs exposed, 80% range,
what-if curves) · **Data quality** (exclusion waterfall, coverage, pipeline log, exact package versions) · **Live** (GBFS map) · **About**.
The run selector keeps real and synthetic runs apart; synthetic runs get a red banner. Charts are error-isolated: if one fails, a notice appears and the rest still render.
For the interactive member/casual charts rebuild EDA once: `python run_all.py phase2 --force`.

**Deploying the dashboard online:** see `DEPLOY.md` (Streamlit Community Cloud, free). Data attribution: Citi Bike / NYC bike share trip data.

## How real and synthetic data are kept apart
| | real | sample (synthetic) |
|---|---|---|
| raw | `data/real/raw/` | `data/sample/raw/` + `SYNTHETIC_DATA_MARKER.txt` |
| derived | `data/real/processed/<START>_<END>/` | `data/sample/processed/<START>_<END>/` |
| reports | `reports/real/<START>_<END>/` | `reports/sample/<START>_<END>/` (figures watermarked) |
`DATA_MODE` selects the tree; outputs are keyed by the month range, so changing the range cannot reuse stale Parquet or weather.
Every run writes `run_manifest.json` (source, range, row counts, coverage, package versions, `is_synthetic`). The dashboard shows it first.

## Pipeline and gates
| Phase | Script | Fails loudly if |
|---|---|---|
| 1 | `download_data.py` | archive size ≠ S3 listing, ZIP CRC fails, a requested month has no CSV (resumes via `.part`, retries, atomic rename, `.ok` marker; bad files kept as `.corrupt`) |
| 1 | `cleaning.py` | required columns missing, month missing, waterfall does not add up, cleaned data outside range / any month < 1,000 trips |
| 1 | `weather.py` | cache or fetch covers < 99.5% of requested hours (cache is revalidated every run) |
| 1 | `features.py` | row count changes through the join, > 0.5% trips without weather |
| 2 | `eda.py` | required stage not completed |
| 3 | `modeling.py`, `interpret.py` | required stage not completed |

### Cleaning rules (every row gets exactly one outcome; see `exclusion_report.csv`)
unparseable timestamp → outside requested range → negative duration → < 60 s → > 6 h → missing start coordinates →
start outside NYC box → end outside NYC box → speed > 40 km/h → duplicate `ride_id`.
**Missing END coordinates** are kept and flagged (they are real trips), have no distance/speed, and are excluded from the duration model only.
`is_staff_trip` is a name-keyword flag, not an exclusion. Malformed lines skipped by the CSV reader are estimated from line counts and reported.

## Analysis methodology
* **EDA** – hourly curves are per-day averages; station imbalance is `(starts-ends)/(starts+ends)` among above-median stations;
  weather is analysed three ways (unadjusted, relative to same month×weekday×hour norm, daily log-trips regression with weekend/holiday/trend and
  Newey-West errors). All weather results are **associations**; a single season cannot separate temperature from trend.
* **Model A (duration)** – linear/ridge/lasso/poly on log-duration. Assumes the destination (distance) is known at prediction time and uses observed weather.
  Baselines: global median, median by rider×bike, distance×median pace. 80% interval from empirical residual quantiles; coverage is reported.
* **Model B (hourly station demand)** – horizons **1 h and 24 h**; lags/rolling means use only data available at the forecast origin;
  station universe chosen from the first half of the period; gap = horizon between train and test.
  Baselines: station mean, station×hour-of-week mean, persistence, same hour yesterday/last week.
  Metrics handle zeros: MAE, RMSE, WAPE, sMAPE (0/0=0), Poisson deviance, bias. MAPE is intentionally not used.
* **Validation** – rolling-origin expanding window, 4 test blocks, mean ± sd across folds. Scalers/encoders are fitted inside each fold.

## Windows notes
* **`DLL load failed ... Application Control policy`** (DuckDB): Windows Smart App Control or an organisational policy blocks the file.
  Settings → Accounts → *Access work or school* (remove stale accounts), Windows Security → *App & browser control* → *Smart App Control*
  (turning it off is permanent without a Windows reset - your decision), or ask IT to allow the venv. Colab is a no-install alternative.
* The `ConnectionResetError [WinError 10054] ... _ProactorBasePipeTransport` message printed by Streamlit is harmless asyncio noise when a browser tab closes.
* Station-map tiles use OpenStreetMap (CartoDB tiles now need an API key).

## Known limitations
Straight-line distance; single-point weather and observed (not forecast) weather inputs; NYC bounding box also contains Hoboken/Jersey City;
station capacity (if used) is a *current* snapshot; demand = started trips (censored by empty stations); linear models only;
one short period cannot establish seasonal generalisation; `ignore_errors` may drop malformed lines (estimated in the report).

## Reproducibility
`requirements.txt` has version ranges; the manifest records exact installed versions. After a first successful run create
`requirements-lock.txt` with `python -m pip freeze`. Tested with Python 3.12, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1,
matplotlib 3.10.8, seaborn 0.13.2, requests 2.33.1 (unit tests only). DuckDB, pyarrow, folium, streamlit versions were **not** exercised by the author.

## Layout
`src/` pipeline · `app/dashboard.py` · `notebooks/` · `tests/` · `reports/REPORT.md` (write-up) · `AUDIT.md` · `TROUBLESHOOTING.md`
