import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import _env  # noqa: F401
import pandas as pd
import config
import provenance
import validate

SRC = Path(_env.SRC)
sys.path.insert(0, str(SRC.parent))   # for run_all


def run_py(code, **env):
    e = {**os.environ, "CITIBIKE_ROOT": tempfile.mkdtemp(), **env}
    return subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, r'{SRC}'); {code}"],
                          capture_output=True, text=True, env=e)


class Modes(unittest.TestCase):
    def test_paths_are_separated_by_mode_and_range(self):
        root = tempfile.mkdtemp()
        out = lambda mode, s, e: run_py("import config; print(config.RAW, config.PROCESSED, config.REPORT_DIR, config.WEATHER_PQ)",
                                        DATA_MODE=mode, START_MONTH=s, END_MONTH=e, CITIBIKE_ROOT=root).stdout.split()
        real, sample = out("real", "2024-01", "2024-03"), out("sample", "2024-01", "2024-03")
        other_range = out("real", "2024-01", "2024-06")
        for a, b in zip(real, sample):
            self.assertNotEqual(a, b)
        self.assertIn("/data/real/", real[0].replace("\\", "/")); self.assertIn("/data/sample/", sample[0].replace("\\", "/"))
        self.assertNotEqual(real[1], other_range[1])                 # processed dir keyed by range
        self.assertNotEqual(real[3], other_range[3])                 # weather cache keyed by range
        self.assertEqual(real[0], other_range[0])                    # raw shared across ranges (never recomputed)

    def test_invalid_config_rejected(self):
        for env in ({"DATA_MODE": "prod"}, {"START_MONTH": "2024-13"}, {"START_MONTH": "2024-05", "END_MONTH": "2024-01"}):
            self.assertNotEqual(run_py("import config", **{"DATA_MODE": "real", "START_MONTH": "2024-01", "END_MONTH": "2024-03", **env}).returncode, 0)

    def test_sample_generator_refuses_real_mode(self):
        r = run_py("exec(open(r'%s').read())" % (SRC / "make_sample_data.py"), DATA_MODE="real")
        self.assertNotEqual(r.returncode, 0); self.assertIn("Refusing", r.stdout + r.stderr)

    def test_download_refuses_sample_mode(self):
        r = run_py("import download_data; download_data.main()", DATA_MODE="sample")
        self.assertNotEqual(r.returncode, 0); self.assertIn("REAL mode only", r.stdout + r.stderr)

    def test_months_in_range_and_boundaries(self):
        self.assertEqual(config.months_in_range(), ["2024-01", "2024-02", "2024-03"])
        self.assertEqual(str(config.RANGE_END_EXCL), "2024-04-01")
        r = run_py("import config; print(config.RANGE_END_EXCL, config.months_in_range())", DATA_MODE="real", START_MONTH="2024-11", END_MONTH="2024-12")
        self.assertIn("2025-01-01", r.stdout)


class Manifest(unittest.TestCase):
    def test_manifest_records_and_refuses_mixing(self):
        m = provenance.init_run()
        self.assertFalse(m["is_synthetic"]); self.assertEqual(m["mode"], "real")
        provenance.update("cleaning", {"rows_final": 5})
        self.assertEqual(provenance.load()["stages"]["cleaning"]["rows_final"], 5)
        self.assertIn("python", provenance.load()["environment"])
        self.assertIsNone(provenance.require_stage("cleaning") and None)
        with self.assertRaises(SystemExit):
            provenance.require_stage("modeling")
        bad = provenance.load(); bad["end_month"] = "2024-06"; provenance._write(bad)   # simulate foreign manifest
        with self.assertRaises(SystemExit):
            provenance.init_run()
        bad["end_month"] = "2024-03"; provenance._write(bad)


class Validation(unittest.TestCase):
    def wx(self, drop=0.0):
        end = validate.weather_target_end()
        idx = pd.date_range(pd.Timestamp(config.RANGE_START), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
        df = pd.DataFrame({"hour_ts": idx, "temp_c": 1.0})
        return df.sample(frac=1 - drop, random_state=0) if drop else df

    def test_weather_coverage_gate(self):
        self.assertAlmostEqual(validate.validate_weather(self.wx())["missing_share"], 0.0)
        with self.assertRaises(SystemExit):
            validate.validate_weather(self.wx(drop=0.2))                      # stale / partial cache is rejected
        with self.assertRaises(SystemExit):
            validate.validate_weather(self.wx().iloc[:200])

    def test_weather_duplicates_do_not_inflate_coverage(self):
        d = self.wx().iloc[:500]
        with self.assertRaises(SystemExit):
            validate.validate_weather(pd.concat([d, d, d]))

    def test_raw_selection_and_gates(self):
        raw = config.RAW
        for n in ("202401-citibike-tripdata_1.csv", "202402-citibike-tripdata_1.csv", "202405-citibike-tripdata_1.csv"):
            (raw / n).write_text("a\n1\n")
        keep, ignored = validate.select_csvs()
        self.assertEqual(sorted(p.name[:6] for p in keep), ["202401", "202402"])
        self.assertEqual([p.name[:6] for p in ignored], ["202405"])           # out-of-range file never read
        with self.assertRaises(SystemExit):
            validate.validate_raw(keep)                                       # 2024-03 missing -> fail loudly
        (raw / "202403-citibike-tripdata_1.csv").write_text("a\n1\n")
        keep, _ = validate.select_csvs()
        self.assertEqual(validate.validate_raw(keep)["months_found"], ["2024-01", "2024-02", "2024-03"])
        config.SYNTHETIC_MARKER.write_text("x")                               # synthetic marker inside REAL raw
        with self.assertRaises(SystemExit):
            validate.validate_raw(keep)
        config.SYNTHETIC_MARKER.unlink()


if __name__ == "__main__":
    unittest.main()


class RunAllStatus(unittest.TestCase):
    def test_completed_stages_and_resume_mapping(self):
        provenance.init_run()
        provenance.update("cleaning", {"x": 1}); provenance.update("weather", {"y": 2})
        self.assertEqual(provenance.completed_stages()[-2:], ["cleaning", "weather"])
        import run_all
        every_script = {s for v in run_all.PHASES.values() for s in v}
        self.assertEqual(every_script - set(run_all.STAGE_OF), set())          # every script maps to a stage
        self.assertIn("station_info.py", run_all.OPTIONAL)
