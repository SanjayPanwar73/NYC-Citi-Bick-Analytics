import tempfile
import unittest
import zipfile
from pathlib import Path
import _env  # noqa: F401
import quarantine_legacy as q


class Quarantine(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        for rel in ("data/processed", "data/external", "reports/figures", "reports/data", "data/raw"):
            (self.root / rel).mkdir(parents=True)
        (self.root / "data/processed/trips_features.parquet").write_bytes(b"x" * 10)
        (self.root / "data/external/weather_hourly.parquet").write_bytes(b"y")
        (self.root / "reports/figures/01.png").write_bytes(b"z")
        (self.root / "data/raw/202401-citibike-tripdata_1.csv").write_text("ride_id,start_station_name\nA,12 St & 3 Ave #44\n")
        (self.root / "data/raw/202402-real.csv").write_text("ride_id,start_station_name\nB,W 21 St & 6 Ave\n")
        with zipfile.ZipFile(self.root / "data/raw/good.zip", "w") as z:
            z.writestr("a.csv", "1")
        (self.root / "data/raw/truncated.zip").write_bytes(b"PK\x03\x04junk")

    def count(self):
        return sum(1 for p in self.root.rglob("*") if p.is_file())

    def test_dry_run_changes_nothing(self):
        before = self.count(); r = q.main(self.root, apply=False)
        self.assertFalse(r["applied"]); self.assertEqual(self.count(), before)

    def test_apply_moves_without_deleting_and_keeps_real_raw(self):
        before = self.count(); r = q.main(self.root, apply=True)
        self.assertTrue(r["applied"])
        self.assertEqual(self.count(), before + 1)                         # + MOVE_MANIFEST.json, nothing lost
        self.assertTrue((self.root / "data/raw/202402-real.csv").exists())  # non-synthetic raw untouched
        self.assertTrue((self.root / "data/raw/good.zip").exists())
        self.assertFalse((self.root / "data/raw/202401-citibike-tripdata_1.csv").exists())   # synthetic-looking CSV quarantined
        self.assertFalse((self.root / "data/processed/trips_features.parquet").exists())
        dest = Path(r["dest"]); self.assertTrue((dest / "data/processed/trips_features.parquet").exists())
        status = {k["file"]: k.get("zip") for k in r["kept"]}
        self.assertEqual(status["good.zip"], "valid"); self.assertTrue(status["truncated.zip"].startswith("INVALID"))


if __name__ == "__main__":
    unittest.main()
