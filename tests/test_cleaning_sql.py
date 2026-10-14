"""Needs DuckDB (skipped otherwise). Verifies every exclusion rule fires exactly once per crafted row."""
import tempfile
import unittest
from pathlib import Path
import _env  # noqa: F401

try:
    import duckdb
except ImportError:
    duckdb = None

HEADER = "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual"
ROWS = {
    "ok":            "A1,classic_bike,2024-01-05 08:00:00,2024-01-05 08:20:00,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "ok_no_end":     "A2,classic_bike,2024-01-05 08:00:00,2024-01-05 08:20:00,A,1,,,40.75,-73.99,,,casual",
    "bad_ts":        "A3,classic_bike,garbage,2024-01-05 08:20:00,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "outside_range": "A4,classic_bike,2023-12-31 23:00:00,2023-12-31 23:20:00,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "negative":      "A5,classic_bike,2024-01-05 08:20:00,2024-01-05 08:00:00,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "too_short":     "A6,classic_bike,2024-01-05 08:00:00,2024-01-05 08:00:30,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "too_long":      "A7,classic_bike,2024-01-05 08:00:00,2024-01-06 08:00:00,A,1,B,2,40.75,-73.99,40.76,-73.98,member",
    "no_start":      "A8,classic_bike,2024-01-05 08:00:00,2024-01-05 08:20:00,A,1,B,2,,,40.76,-73.98,member",
    "start_outside": "A9,classic_bike,2024-01-05 08:00:00,2024-01-05 08:20:00,A,1,B,2,0,0,40.76,-73.98,member",
    "end_outside":   "B1,classic_bike,2024-01-05 08:00:00,2024-01-05 08:20:00,A,1,B,2,40.75,-73.99,41.9,-73.98,member",
    "too_fast":      "B2,classic_bike,2024-01-05 08:00:00,2024-01-05 08:02:00,A,1,B,2,40.60,-74.0,40.90,-73.7,member",
}
EXPECT = {"kept": 2, "unparseable_timestamp": 1, "outside_requested_range": 1, "negative_duration": 1, "duration_below_min": 1,
          "duration_above_max": 1, "missing_start_coords": 1, "start_outside_nyc_box": 1, "end_outside_nyc_box": 1, "implausible_speed": 1}


@unittest.skipUnless(duckdb, "duckdb not installed - run this test on your machine")
class CleaningSQL(unittest.TestCase):
    def test_waterfall(self):
        import cleaning, config
        csv = Path(tempfile.mkdtemp()) / "202401-citibike-tripdata_1.csv"
        csv.write_text(HEADER + "\n" + "\n".join(ROWS.values()) + "\n")
        stats, out = cleaning.clean_file(duckdb.connect(), csv)
        for k, v in EXPECT.items():
            self.assertEqual(stats[k] if k != "kept" else stats["kept"], v, k)
        self.assertEqual(stats["rows_parsed"], len(ROWS))
        self.assertEqual(stats["kept_missing_end_coords"], 1)             # kept, flagged, counted
        n = duckdb.connect().execute(f"SELECT count(*), count(*) FILTER (WHERE trip_distance_km IS NULL) FROM '{out.as_posix()}'").fetchone()
        self.assertEqual(n, (2, 1))                                        # the no-end-coords trip has no distance


if __name__ == "__main__":
    unittest.main()
