# Audit - Citi Bike project (before → after)

Line numbers refer to the **pre-change** files (no git history in the workspace). The author's workspace contains only code;
the user's processed data (291,155 trips, 250 stations, Jan-Jun 2024) was **not visible** to the author. It is *consistent with*
`make_sample_data.py` (`n_st = 250`, default Jan-Jun range) but that was **not verified** against the file itself.
Treat all existing reports/metrics as synthetic until `run_manifest.json` says otherwise.

## Findings (priority order)
| # | Sev | Finding | Evidence (pre-change) | Fix |
|---|---|---|---|---|
| 1 | P0 | Synthetic and real data share the same folders; the generator overwrote the real weather cache | `config.py:6-8,32` single RAW/PROCESSED/EXTERNAL; `make_sample_data.py:9,37` writes CSVs into RAW and `weather_hourly.parquet`; `weather.py:12` trusts any existing file | `config.py` mode trees + range-keyed run dirs; generator refuses real mode and writes `SYNTHETIC_DATA_MARKER.txt`; cleaning refuses a marker in real raw; `provenance.py` manifest; figure watermark; dashboard banner; `quarantine_legacy.py` (moves, never deletes) |
| 2 | P0 | Downloads can silently be "complete" while truncated (explains the invalid archive) | `download_data.py:81-85` writes straight to the final name; `if not dest.exists()` treats any existing file as done; no size/CRC check, no retry | `.part` + HTTP Range resume, 4 retries with backoff, size vs S3 listing, full ZIP CRC test, atomic rename, `.ok` marker, `.corrupt` quarantine, missing-month failure |
| 3 | P0 | Cleaning reads every CSV in `raw/` regardless of range; stale per-file Parquet are globbed back in | `cleaning.py:100` `RAW.glob("*.csv")`; `:119` `read_parquet(CLEAN_DIR/*.parquet)` | range-filtered CSV list, explicit file list, run-tag directories, row-level `outside_requested_range` rule |
| 4 | P0 | NYC coordinate rule hidden: out-of-box coordinates only blanked the distance and the trip stayed; missing coordinates uncounted; `ignore_errors` silent | `cleaning.py:71-75`, `:59`; only a raw/kept print existed | explicit waterfall reasons, deliberate missing-end handling, unparsed-line estimate, `exclusion_report.csv` + assertion that it adds up |
| 5 | P0 | Weather cache ignores the requested range | `weather.py:12-13`; `config.py:32` fixed filename | range-keyed cache, coverage validation every run, validate-before-persist, DST-duplicate guard |
| 6 | P1 | Weather "effect" stated from unadjusted dry-vs-rainy day means (season/weekday confounded); Pearson on skewed data; imbalance by raw counts | `eda.py:159`, `:165`, `:116` | within-norm hourly ratios, HAC regression with weekend/holiday/trend + reliability label, Spearman, ratio ranking among above-median stations, associational wording |
| 7 | P1 | Weak evaluation: one 80/20 row-quantile split; station universe chosen on the full period; MAPE masks zeros; horizon unstated; mean-only baseline; no uncertainty | `modeling.py:175`, `:138-140`, `:31-35`, `:154-156` | rolling-origin backtest, gap = horizon, h∈{1,24}, train-only station selection, 4-7 baselines, WAPE/sMAPE/Poisson deviance, empirical 80% interval with coverage, explicit assumptions in results JSON |
| 8 | P1 | Dashboard hid fixed inputs and showed no provenance/uncertainty | `dashboard.py:120-121` (humidity 60, holiday 0) | provenance tab, synthetic banner, all inputs exposed, interval + coverage, limitations tab, run selector |
| 9 | P2 | No tests, no `.gitignore`, loose requirements, placeholder notebooks/report | repo root | `tests/` (34 tests), `.gitignore`, ranged requirements + manifest versions, notebooks and `reports/REPORT.md` rewritten without invented findings |

## Validation results
**Author's sandbox (Linux, Python 3.12, no DuckDB/pyarrow):** 41 unit tests pass, 1 skipped (incl. a fake-Streamlit smoke test of the whole dashboard: every tab, empty states, chart-failure isolation, NaN-free map data; it does NOT validate Altair/pydeck specs or visual rendering); modeling logic on synthetic frames with DuckDB
stubbed; synthetic generator checks; all modules compile. These did not exercise the SQL.

**User-reported runs on a real Windows machine (Python 3.14, DuckDB 1.5.6) - reported by the user, not independently verified by the author:**
- Real download of `202406-citibike-tripdata.zip` (936 MB) passed the size + ZIP-CRC gate; 5 CSV files extracted.
- `python run_all.py check`: 33/34 passed, including `test_cleaning_sql` on real DuckDB (every exclusion rule fires; missing end coordinates kept and flagged).
  The one failure was a test-side path-separator bug; a real Windows bug (rename onto an existing `.corrupt` file) was also found and fixed (`quarantine()`).
- Stages `download`, `cleaning`, `weather` completed (manifest); `features`: 4,775,401 rows, 0.000% without weather; `eda`: completed, 11 figures.
- The first EDA numbers are in `reports/REPORT.md` (labelled preliminary descriptive results).

**Still unverified:** exclusion-report contents for the real run (not yet inspected by the author); `modeling.py` and `interpret.py` on real data
(Model B SQL has never run on DuckDB); the Streamlit dashboard; `station_info.py`; models trained on a single month have very short test blocks.

## Remaining limitations / next steps
1. Download a real month, run `python run_all.py check`, then the 1-month real pipeline; review `exclusion_report.csv` and manifest.
2. Run the skipped SQL test; paste any traceback for fixes.
3. NYC box includes Hoboken/Jersey City - use borough polygons for exact boundaries.
4. Capacity (if enabled) is a current snapshot; weather inputs are observed, not forecast; demand is censored.
5. One period cannot show seasonal generalisation - extend months and re-run the backtest.
6. Consider Poisson/negative-binomial or gradient-boosted comparators for Model B.
