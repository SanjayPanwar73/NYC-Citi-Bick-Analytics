"""Phase gates: every phase validates its inputs and output coverage and FAILS LOUDLY instead of continuing."""
import re
from datetime import date, timedelta

import pandas as pd

import config


class ValidationError(SystemExit):
    pass


def _fail(msg):
    raise ValidationError(f"VALIDATION FAILED: {msg}")


def csv_month(name):
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", name)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def select_csvs(raw_dir=None):
    """CSVs for the requested range only (labelled out-of-range files are ignored, unlabelled ones kept)."""
    raw_dir = raw_dir or config.RAW
    keep, ignored = [], []
    for p in sorted(raw_dir.glob("*.csv")):
        m = csv_month(p.name)
        (keep if (m is None or config.START_MONTH <= m <= config.END_MONTH) else ignored).append(p)
    return keep, ignored


def validate_raw(csvs):
    if not csvs:
        _fail(f"no CSV files for {config.START_MONTH}..{config.END_MONTH} in {config.RAW}")
    if config.MODE == "real" and config.SYNTHETIC_MARKER.exists():
        _fail(f"{config.SYNTHETIC_MARKER} exists inside the REAL raw folder - synthetic files must never be here")
    found = {csv_month(p.name) for p in csvs if csv_month(p.name)}
    missing = [m for m in config.months_in_range() if m not in found]
    if found and missing:
        _fail(f"months missing from raw data: {missing}")
    for p in csvs:
        if p.stat().st_size == 0:
            _fail(f"{p.name} is empty")
    return {"months_found": sorted(found), "n_files": len(csvs)}


def validate_clean(con, path):
    rows, mn, mx = con.execute(f"SELECT count(*), min(started_at), max(started_at) FROM '{config.sql_path(path)}'").fetchone()
    if not rows:
        _fail("cleaned trips table is empty")
    if pd.Timestamp(mn) < pd.Timestamp(config.RANGE_START) or pd.Timestamp(mx) >= pd.Timestamp(config.RANGE_END_EXCL):
        _fail(f"cleaned trips outside requested range: {mn} .. {mx}")
    by_month = dict(con.execute(
        f"SELECT strftime(started_at, '%Y-%m'), count(*) FROM '{config.sql_path(path)}' GROUP BY 1").fetchall())
    short = {m: by_month.get(m, 0) for m in config.months_in_range() if by_month.get(m, 0) < config.MIN_ROWS_PER_MONTH}
    if short:
        _fail(f"months with fewer than {config.MIN_ROWS_PER_MONTH:,} trips after cleaning: {short}")
    return {"rows": rows, "first_trip": str(mn), "last_trip": str(mx), "rows_by_month": by_month}


def weather_target_end():
    last_day = config.RANGE_END_EXCL - timedelta(days=1)
    return min(last_day, date.today() - timedelta(days=config.ARCHIVE_LAG_DAYS))


def validate_weather(df, end_date=None):
    end_date = end_date or weather_target_end()
    if end_date < config.RANGE_START:
        _fail("requested range is entirely newer than the Open-Meteo archive")
    expected = pd.date_range(pd.Timestamp(config.RANGE_START), pd.Timestamp(end_date) + pd.Timedelta(hours=23), freq="h")
    have = pd.DatetimeIndex(pd.to_datetime(df["hour_ts"]).drop_duplicates())
    missing_share = 1 - len(expected.intersection(have)) / len(expected)
    if missing_share > config.MAX_MISSING_WEATHER_SHARE:
        _fail(f"weather covers only {100 * (1 - missing_share):.2f}% of {config.RANGE_START}..{end_date}")
    return {"hours_expected": len(expected), "hours_present": int(len(have)), "missing_share": missing_share,
            "covered_through": str(end_date)}


def validate_features(con, feat_path, clean_path):
    f, c = config.sql_path(feat_path), config.sql_path(clean_path)
    n_f, n_null = con.execute(f"SELECT count(*), count(*) FILTER (WHERE temp_c IS NULL) FROM '{f}'").fetchone()
    n_c = con.execute(f"SELECT count(*) FROM '{c}'").fetchone()[0]
    if n_f != n_c:
        _fail(f"feature table has {n_f:,} rows but cleaned table has {n_c:,} (join fan-out or loss)")
    share = n_null / n_f
    if share > config.MAX_MISSING_WEATHER_SHARE:
        _fail(f"{100 * share:.2f}% of trips have no weather match (limit {100 * config.MAX_MISSING_WEATHER_SHARE:.1f}%). "
              f"Weather cache may not cover the trip dates.")
    return {"rows": n_f, "no_weather_share": share}
