# Troubleshooting

| Problem | Fix |
|---|---|
| `pip.exe blocked by Device Guard` | `python -m pip install -r requirements.txt` (inside a venv), or use Google Colab |
| `Manifest ... belongs to a different run. Refusing to mix runs.` | You changed mode/months but reuse a processed folder; outputs are keyed by range, so this means the manifest was edited or copied. Use the right `START_MONTH/END_MONTH` or remove that run folder yourself |
| `VALIDATION FAILED: months missing from raw data` | A requested month has no CSV in `data/<mode>/raw`. Rerun `download_data.py` or place validated CSVs named `YYYYMM-citibike-tripdata*.csv` |
| `VALIDATION FAILED: weather covers only ...%` | Open-Meteo archive lags ~6 days; choose months that ended earlier, or delete the stale `weather_<range>.parquet` and rerun |
| Download keeps failing | Files stay as `.part` and resume. If a `.corrupt` file appears it was kept for inspection. Check your network/proxy; the S3 listing sizes are authoritative |
| `Refusing to run: make_sample_data.py only works with DATA_MODE=sample` | Use `python run_all.py sample` |
| Legacy outputs from before the real/sample split | `python src/quarantine_legacy.py` (dry run) then `--apply` |
| Out of memory | Narrow the range; lower `n_sample` in `modeling.model_duration`; set `PRAGMA memory_limit` in the scripts |
| Very low R2 for duration | Expected; judge by skill vs the best baseline and report it honestly |
| Dashboard shows nothing | It reads finished runs only: `reports/<mode>/<range>/data/run_manifest.json` |
| `DLL load failed while importing _duckdb: An Application Control policy has blocked this file` | See README "Windows notes": Smart App Control / work-or-school account / IT allowlist, or run on Colab. `Unblock-File` does not help for publisher/hash policies |
| `'streamlit' is not recognized` / `No such file ... run_manifest.json` | Wrong folder or venv not active. `cd` into the folder that contains `run_all.py`, run `.venv\Scripts\activate`, use `python -m streamlit run app/dashboard.py` |
| Dashboard says "No finished run found" | `python run_all.py status`; the dashboard needs at least the `eda` stage |
| `ConnectionResetError: [WinError 10054]` printed by Streamlit | Harmless browser-disconnect noise; ignore if the page works |
| Silent for a long time during download | It prints a line every ~50 MB now; also watch `data\real\raw\*.part` grow. Rerun to resume |
| `CartoDB tiles now require an API key` | Fixed: the map uses OpenStreetMap tiles. Rerun `python src/eda.py` to rebuild the map |
| Pipeline stopped midway | Fix the error, rerun `python run_all.py all` - completed stages are skipped; `--force` redoes everything |
| Dashboard shows "This chart is unavailable (…)" | One chart failed; the rest still work. Send me the exception name shown. Update Streamlit (`python -m pip install -U streamlit`) if it mentions `width` or `use_container_width` |
| Map is blank / grey | The basemap tiles may be blocked by your network; dots still render. Try another network or refresh |
| Rider charts on the Rhythm tab are static images | Run `python run_all.py phase2 --force` once to create `user_hourly.csv` |
