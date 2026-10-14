"""Central settings.

DATA_MODE separates REAL Citi Bike data from SYNTHETIC test data at the path level:
    DATA_MODE=real   (default)  -> data/real/...    reports/real/<run_tag>/
    DATA_MODE=sample            -> data/sample/...  reports/sample/<run_tag>/
Derived artifacts live under a run tag (START_END months), so changing the date range can
never silently reuse stale Parquet files. Raw downloads are shared per mode and never deleted.
"""
import os
import re
from datetime import date
from pathlib import Path

ROOT = Path(os.environ.get("CITIBIKE_ROOT") or Path(__file__).resolve().parents[1])   # CITIBIKE_ROOT: used by tests

# ---------------- mode ----------------
MODE = os.environ.get("DATA_MODE", "real").strip().lower()
if MODE not in ("real", "sample"):
    raise SystemExit(f"DATA_MODE must be 'real' or 'sample', got {MODE!r}")
IS_SYNTHETIC = MODE == "sample"

# ---------------- scope ----------------
_DEFAULT = ("2024-01", "2024-03") if IS_SYNTHETIC else ("2024-01", "2024-06")
START_MONTH = os.environ.get("START_MONTH", _DEFAULT[0])
END_MONTH = os.environ.get("END_MONTH", _DEFAULT[1])
for _m in (START_MONTH, END_MONTH):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", _m):
        raise SystemExit(f"Month must look like YYYY-MM, got {_m!r}")
if START_MONTH > END_MONTH:
    raise SystemExit(f"START_MONTH {START_MONTH} is after END_MONTH {END_MONTH}")

RUN_TAG = f"{START_MONTH}_{END_MONTH}"
RANGE_START = date(int(START_MONTH[:4]), int(START_MONTH[5:]), 1)
_ey, _em = int(END_MONTH[:4]), int(END_MONTH[5:])
RANGE_END_EXCL = date(_ey + (_em == 12), 1 if _em == 12 else _em + 1, 1)   # first day AFTER the range


def months_in_range():
    out, y, m = [], RANGE_START.year, RANGE_START.month
    while date(y, m, 1) < RANGE_END_EXCL:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ---------------- paths ----------------
DATA = ROOT / "data" / MODE
RAW = DATA / "raw"                         # original downloads - never deleted by the pipeline
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed" / RUN_TAG
CLEAN_DIR = PROCESSED / "clean_by_file"
REPORT_DIR = ROOT / "reports" / MODE / RUN_TAG
FIG = REPORT_DIR / "figures"
REP_DATA = REPORT_DIR / "data"
LIVE = ROOT / "data" / "live"              # live GBFS snapshots, kept apart from historical data
for _p in (RAW, EXTERNAL, PROCESSED, CLEAN_DIR, FIG, REP_DATA):
    _p.mkdir(parents=True, exist_ok=True)

MANIFEST = PROCESSED / "run_manifest.json"
TRIPS_CLEAN = PROCESSED / "trips_clean.parquet"
STATION_LOOKUP = PROCESSED / "station_lookup.parquet"
TRIPS_FEAT = PROCESSED / "trips_features.parquet"
EXCLUSION_REPORT = PROCESSED / "exclusion_report.csv"
WEATHER_PQ = EXTERNAL / f"weather_{RUN_TAG}.parquet"     # keyed by range: no stale cache reuse
STATION_INFO = EXTERNAL / "station_info.parquet"         # CURRENT capacity snapshot (not historical)
SYNTHETIC_MARKER = RAW / "SYNTHETIC_DATA_MARKER.txt"

# ---------------- cleaning rules (all counted in the exclusion report) ----------------
MIN_DURATION_S = 60
MAX_DURATION_S = 6 * 3600
MAX_SPEED_KMH = 40
# Rough NYC bounding box. NOTE: a box also contains Hoboken/Jersey City; use a borough polygon for exact boundaries.
NYC_LAT = (40.45, 41.00)
NYC_LON = (-74.30, -73.60)

# ---------------- weather ----------------
WEATHER_LAT, WEATHER_LON = 40.71, -74.00
ARCHIVE_LAG_DAYS = 6                      # Open-Meteo archive trails real time by ~5 days

# ---------------- validation thresholds ----------------
MAX_MISSING_WEATHER_SHARE = 0.005
MIN_ROWS_PER_MONTH = 1_000


def sql_path(p):
    """Path -> SQL string literal content (forward slashes, quotes escaped). Works on Windows."""
    return Path(p).as_posix().replace("'", "''")
