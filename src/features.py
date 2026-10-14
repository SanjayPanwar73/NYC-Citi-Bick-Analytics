"""PHASE 1 / step 4 - Feature engineering + weather join -> trips_features.parquet (validated)."""
import duckdb
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import config
import provenance
import validate
from config import sql_path


def main():
    provenance.require_stage("cleaning")
    provenance.require_stage("weather")
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='duckdb_tmp'")
    hol = USFederalHolidayCalendar().holidays(start=str(config.RANGE_START), end=str(config.RANGE_END_EXCL))
    con.register("holidays", pd.DataFrame({"hol_date": pd.Series(list(hol), dtype="datetime64[ns]")}))
    con.execute(f"""
    COPY (
      SELECT t.*,
        date_trunc('hour', t.started_at) AS hour_ts, t.started_at::DATE AS trip_date,
        hour(t.started_at) AS hour, isodow(t.started_at) AS day_of_week, month(t.started_at) AS month,
        year(t.started_at) AS year, isodow(t.started_at) >= 6 AS is_weekend,
        (h.hol_date IS NOT NULL) AS is_holiday, t.trip_duration_s / 60.0 AS trip_duration_min,
        w.temp_c, w.humidity, w.precip_mm, w.snow_cm, w.wind_kmh
      FROM '{sql_path(config.TRIPS_CLEAN)}' t
      LEFT JOIN holidays h ON CAST(h.hol_date AS DATE) = t.started_at::DATE
      LEFT JOIN read_parquet('{sql_path(config.WEATHER_PQ)}') w ON w.hour_ts = date_trunc('hour', t.started_at)
    ) TO '{sql_path(config.TRIPS_FEAT)}' (FORMAT parquet)""")
    stats = validate.validate_features(con, config.TRIPS_FEAT, config.TRIPS_CLEAN)
    provenance.update("features", stats)
    print(f"{stats['rows']:,} rows; {100 * stats['no_weather_share']:.3f}% without weather")


if __name__ == "__main__":
    main()
