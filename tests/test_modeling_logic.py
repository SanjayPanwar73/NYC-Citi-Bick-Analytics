import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import _env  # noqa: F401

try:
    import sklearn, scipy, matplotlib, seaborn, joblib  # noqa: F401,E401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "scikit-learn/scipy/matplotlib/seaborn not installed")
class ModelingLogic(unittest.TestCase):
    """duckdb is stubbed: checks backtesting, baselines, metrics and artifacts (NOT the SQL)."""
    @classmethod
    def setUpClass(cls):
        env = {**os.environ, "CITIBIKE_ROOT": tempfile.mkdtemp(), "DATA_MODE": "sample", "START_MONTH": "2024-01", "END_MONTH": "2024-03"}
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "_modeling_stub_runner.py"), str(_env.SRC)],
                           capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        cls.o = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT="))[7:])

    def test_models_beat_trivial_baselines_on_known_signal(self):
        self.assertLess(self.o["a_ols_mae"], self.o["a_median_mae"])
        self.assertLess(self.o["b1_ols_mae"], self.o["b1_mean_mae"])
        self.assertTrue(self.o["skill_finite"])

    def test_backtest_structure(self):
        self.assertEqual(self.o["a_folds"], 4)
        self.assertEqual(len(self.o["backtest_files"]), 3)
        self.assertIn("Baseline: same hour last week", self.o["b1_baselines"])
        self.assertIn("Baseline: station x hour-of-week mean", self.o["b1_baselines"])

    def test_zero_safe_metrics_for_counts(self):
        self.assertTrue(self.o["b1_has_zero_share"]); self.assertTrue(self.o["b1_no_mape"])

    def test_horizon_specific_features_and_gap_documented(self):
        self.assertIn("lag_24h", self.o["b1_features"])
        self.assertNotIn("lag_24h", self.o["b24_features"])              # would duplicate lag_h at h=24
        self.assertTrue(self.o["b24_gap_note"])

    def test_uncertainty_interval_and_artifacts(self):
        self.assertTrue(0.6 < self.o["a_coverage"] < 0.95)
        self.assertIn("resid_q10_log", self.o["bundle_keys"]); self.assertIn("fixed_training_conditions", self.o["bundle_keys"])
        self.assertTrue(self.o["bundle_synth"])                           # sample mode marks the model synthetic
        self.assertIn("12_duration_residuals.png", self.o["figs"]); self.assertIn("15_demand_eval_h24.png", self.o["figs"])


if __name__ == "__main__":
    unittest.main()
