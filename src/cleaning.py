"""PHASE 1 / step 2 - Clean raw CSVs with DuckDB and produce an EXCLUSION REPORT.

Every raw row gets exactly ONE outcome (first failing rule wins), so the report is a true waterfall:
  kept | unparseable_timestamp | outside_requested_range | negative_duration | duration_below_min |
  duration_above_max | missing_start_coords | start_outside_nyc_box | end_outside_nyc_box | implausible_speed |
  duplicate_ride_id (found after merging files)
Missing END coordinates are handled intentionally: the trip is KEPT (it is a real trip) but flagged
`missing_end_coords`, has no distance/speed, and is excluded from the duration model.
Only CSVs inside the requested range are read; outputs live under the run tag, so nothing stale is reused.
"""
import pandas as pd
import duckdb

import config
import provenance
import validate
from config import sql_path

COLUMN_MAP = {
    "ride_id": ["ride_id"], "rideable_type": ["rideable_type"],
    "started_at": ["started_at", "starttime", "start_time"], "ended_at": ["ended_at", "stoptime", "stop_time"],
    "start_station_name": ["start_station_name", "start station name"],
    "start_station_id": ["start_station_id", "start station id"],
    "end_station_name": ["end_station_name", "end station name"],
    "end_station_id": ["end_station_id", "end station id"],
    "start_lat": ["start_lat", "start station latitude"], "start_lng": ["start_lng", "start station longitude"],
    "end_lat": ["end_lat", "end station latitude"], "end_lng": ["end_lng", "end station longitude"],
    "user_type": ["member_casual", "usertype"], "bike_id": ["bikeid"],
}
COORDS = ("start_lat", "start_lng", "end_lat", "end_lng")
USER_FIX = ("CASE lower({c}) WHEN 'subscriber' THEN 'member' "
            "WHEN 'customer' THEN 'casual' ELSE lower({c}) END")
HAV = """6371 * 2 * asin(sqrt(
  pow(sin(radians(end_lat - start_lat) / 2), 2) +
  cos(radians(start_lat)) * cos(radians(end_lat)) *
  pow(sin(radians(end_lng - start_lng) / 2), 2)))"""
REASONS = ["unparseable_timestamp", "outside_requested_range", "negative_duration", "duration_below_min",
           "duration_above_max", "missing_start_coords", "start_outside_nyc_box", "end_outside_nyc_box",
           "implausible_speed"]
REQUIRED = ["started_at", "ended_at"]


def build_select(cols):
    lower = {c.lower().strip(): c for c in cols}
    missing = [r for r in REQUIRED if not any(o in lower for o in COLUMN_MAP[r])]
    if missing:
        raise validate.ValidationError(f"CSV lacks required column(s) {missing}; columns seen: {sorted(lower)}")
    parts = []
    for std, options in COLUMN_MAP.items():
        src = next((lower[o] for o in options if o in lower), None)
        if src is None:
            parts.append(f"CAST(NULL AS {'DOUBLE' if std in COORDS else 'VARCHAR'}) AS {std}")
            continue
        q = f'"{src}"'
        if std in ("started_at", "ended_at"):
            parts.append(f"try_cast({q} AS TIMESTAMP) AS {std}")
        elif std in COORDS:
            parts.append(f"try_cast({q} AS DOUBLE) AS {std}")
        elif std == "user_type":
            parts.append(USER_FIX.format(c=q) + " AS user_type")
        else:
            parts.append(f"nullif(trim({q}), '') AS {std}")
    return ",\n          ".join(parts)


def count_data_lines(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def clean_file(con, csv_path):
    rd = f"read_csv('{sql_path(csv_path)}', all_varchar=true, sample_size=-1, ignore_errors=true)"
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rd}").fetchall()]
    la, lb = config.NYC_LAT
    lo, hi = config.NYC_LON
    box = lambda a, b: f"{a} BETWEEN {la} AND {lb} AND {b} BETWEEN {lo} AND {hi}"
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE labeled AS
    WITH src AS (SELECT {build_select(cols)} FROM {rd}),
    step AS (
      SELECT *,
        date_diff('second', started_at, ended_at) AS trip_duration_s,
        (end_lat IS NULL OR end_lng IS NULL) AS missing_end_coords,
        CASE WHEN start_lat IS NOT NULL AND start_lng IS NOT NULL AND end_lat IS NOT NULL AND end_lng IS NOT NULL
              AND {box('start_lat', 'start_lng')} AND {box('end_lat', 'end_lng')}
             THEN {HAV} END AS trip_distance_km
      FROM src)
    SELECT *,
      trip_distance_km / nullif(trip_duration_s / 3600.0, 0) AS avg_speed_kmh,
      CASE
        WHEN started_at IS NULL OR ended_at IS NULL THEN 'unparseable_timestamp'
        WHEN started_at < TIMESTAMP '{config.RANGE_START}' OR started_at >= TIMESTAMP '{config.RANGE_END_EXCL}'
             THEN 'outside_requested_range'
        WHEN trip_duration_s < 0 THEN 'negative_duration'
        WHEN trip_duration_s < {config.MIN_DURATION_S} THEN 'duration_below_min'
        WHEN trip_duration_s > {config.MAX_DURATION_S} THEN 'duration_above_max'
        WHEN start_lat IS NULL OR start_lng IS NULL THEN 'missing_start_coords'
        WHEN NOT ({box('start_lat', 'start_lng')}) THEN 'start_outside_nyc_box'
        WHEN NOT missing_end_coords AND NOT ({box('end_lat', 'end_lng')}) THEN 'end_outside_nyc_box'
        WHEN trip_distance_km / nullif(trip_duration_s / 3600.0, 0) > {config.MAX_SPEED_KMH} THEN 'implausible_speed'
      END AS exclusion_reason
    FROM step""")
    counts = dict(con.execute("SELECT coalesce(exclusion_reason, 'kept'), count(*) FROM labeled GROUP BY 1").fetchall())
    parsed = sum(counts.values())
    info = con.execute("""SELECT count(*) FILTER (WHERE missing_end_coords),
                                 count(*) FILTER (WHERE start_station_id IS NULL),
                                 count(*) FILTER (WHERE end_station_id IS NULL)
                          FROM labeled WHERE exclusion_reason IS NULL""").fetchone()
    out = config.CLEAN_DIR / (csv_path.stem + ".parquet")
    con.execute(f"""
    COPY (
      SELECT * EXCLUDE (exclusion_reason),
        coalesce(start_station_name ILIKE '%depot%' OR end_station_name ILIKE '%depot%'
         OR start_station_name ILIKE '%warehouse%' OR end_station_name ILIKE '%warehouse%'
         OR start_station_name ILIKE '%test%' OR end_station_name ILIKE '%test%', false) AS is_staff_trip,
        '{csv_path.name}' AS source_file
      FROM labeled WHERE exclusion_reason IS NULL
    ) TO '{sql_path(out)}' (FORMAT parquet)""")
    lines = count_data_lines(csv_path)
    stats = {"file": csv_path.name, "csv_data_lines": lines, "rows_parsed": parsed,
             "rows_unparsed_approx": max(lines - parsed, 0), "kept": counts.get("kept", 0),
             "kept_missing_end_coords": info[0], "kept_missing_start_station_id": info[1],
             "kept_missing_end_station_id": info[2], **{r: counts.get(r, 0) for r in REASONS}}
    print(f"{csv_path.name}: parsed {parsed:,} -> kept {stats['kept']:,}  "
          f"(unparsed lines ~{stats['rows_unparsed_approx']:,})")
    return stats, out


def main():
    provenance.init_run()
    csvs, ignored = validate.select_csvs()
    if ignored:
        print("ignoring CSVs outside the requested range:", [p.name for p in ignored])
    print("raw coverage:", validate.validate_raw(csvs))
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='duckdb_tmp'")
    per_file, outs = [], []
    for f in csvs:
        s, o = clean_file(con, f)
        per_file.append(s)
        outs.append(o)

    files_sql = "[" + ", ".join(f"'{sql_path(o)}'" for o in outs) + "]"       # ONLY this run's files, no globbing
    con.execute(f"""
    COPY (
      SELECT * EXCLUDE (rn) FROM (
        SELECT *, row_number() OVER (PARTITION BY coalesce(ride_id,
                 concat_ws('|', started_at, start_station_id, end_station_id, bike_id))) AS rn
        FROM (
          SELECT * REPLACE (
            regexp_replace(trim(start_station_name), '\\s+', ' ', 'g') AS start_station_name,
            regexp_replace(trim(end_station_name),   '\\s+', ' ', 'g') AS end_station_name)
          FROM read_parquet({files_sql}, union_by_name=true))
      ) WHERE rn = 1
    ) TO '{sql_path(config.TRIPS_CLEAN)}' (FORMAT parquet)""")
    final = con.execute(f"SELECT count(*) FROM '{sql_path(config.TRIPS_CLEAN)}'").fetchone()[0]
    kept_pre_dedup = sum(s["kept"] for s in per_file)
    dupes = kept_pre_dedup - final

    # ---- exclusion report (waterfall) ----
    n_lines = sum(s["csv_data_lines"] for s in per_file)
    rows = [("csv_data_lines (all files)", n_lines),
            ("unparsed_or_malformed_lines (approx, ignore_errors)", sum(s["rows_unparsed_approx"] for s in per_file)),
            ("rows_parsed", sum(s["rows_parsed"] for s in per_file))]
    rows += [(f"excluded: {r}", sum(s[r] for s in per_file)) for r in REASONS]
    rows += [("excluded: duplicate_ride_id", dupes), ("FINAL rows kept", final),
             ("info (kept): missing_end_coords", sum(s["kept_missing_end_coords"] for s in per_file)),
             ("info (kept): missing_start_station_id", sum(s["kept_missing_start_station_id"] for s in per_file)),
             ("info (kept): missing_end_station_id", sum(s["kept_missing_end_station_id"] for s in per_file))]
    rep = pd.DataFrame(rows, columns=["step", "rows"])
    parsed_total = rep.loc[rep.step == "rows_parsed", "rows"].iloc[0]
    rep["pct_of_parsed"] = (100 * rep.rows / max(parsed_total, 1)).round(3)
    rep.to_csv(config.EXCLUSION_REPORT, index=False)
    print("\n" + rep.to_string(index=False))
    assert parsed_total - sum(r[1] for r in rows if r[0].startswith("excluded:")) == final, "waterfall does not add up"

    con.execute(f"""
    COPY (SELECT start_station_id AS station_id, mode(start_station_name) AS station_name,
                 median(start_lat) AS lat, median(start_lng) AS lng, count(*) AS n_trips
          FROM '{sql_path(config.TRIPS_CLEAN)}' WHERE start_station_id IS NOT NULL GROUP BY 1
    ) TO '{sql_path(config.STATION_LOOKUP)}' (FORMAT parquet)""")

    cov = validate.validate_clean(con, config.TRIPS_CLEAN)
    provenance.update("cleaning", {
        "rules": {"min_duration_s": config.MIN_DURATION_S, "max_duration_s": config.MAX_DURATION_S,
                  "max_speed_kmh": config.MAX_SPEED_KMH, "nyc_box_lat": config.NYC_LAT, "nyc_box_lon": config.NYC_LON,
                  "missing_end_coords": "kept, flagged, excluded from distance/speed/duration model",
                  "missing_start_coords": "excluded"},
        "csv_files": [s["file"] for s in per_file], "rows_parsed": int(parsed_total), "rows_final": int(final),
        "duplicates_removed": int(dupes), "coverage": cov})
    print(f"\nrun {config.MODE}/{config.RUN_TAG}: {parsed_total:,} parsed -> {final:,} kept")


if __name__ == "__main__":
    main()
