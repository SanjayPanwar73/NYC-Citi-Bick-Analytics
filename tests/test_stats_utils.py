import unittest
import _env  # noqa: F401
import numpy as np
import pandas as pd
from stats_utils import regression_metrics, smape, poisson_deviance, rolling_origin_splits, ols_hac


class Metrics(unittest.TestCase):
    def test_zero_demand_handled(self):
        y = np.array([0, 0, 3, 0, 1.0]); p = np.array([0, 0.2, 2.5, 0, 1.0])
        m = regression_metrics(y, p, count_data=True)
        for k in ("MAE", "RMSE", "WAPE_%", "sMAPE_%", "PoissonDeviance"):
            self.assertTrue(np.isfinite(m[k]), k)
        self.assertAlmostEqual(m["zero_share"], 0.6)
        self.assertNotIn("MAPE_%", m)                       # MAPE is deliberately not used for counts

    def test_smape_zero_over_zero_is_zero(self):
        self.assertEqual(smape([0, 0], [0, 0]), 0.0)

    def test_poisson_deviance_zero_for_perfect(self):
        self.assertAlmostEqual(poisson_deviance([1, 2, 3], [1, 2, 3]), 0.0, places=9)
        self.assertGreater(poisson_deviance([0, 0], [2, 2]), 0)

    def test_perfect_prediction(self):
        y = np.arange(1, 20.0); m = regression_metrics(y, y)
        self.assertEqual(m["MAE"], 0); self.assertEqual(m["R2"], 1.0)

    def test_baseline_mean_has_r2_zero(self):
        y = np.arange(1, 20.0); self.assertAlmostEqual(regression_metrics(y, np.full(len(y), y.mean()))["R2"], 0.0)


class Splits(unittest.TestCase):
    def setUp(self):
        self.t = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=40 * 24, freq="h"), 3))

    def test_no_leakage_and_gap(self):
        gap = pd.Timedelta(hours=24)
        splits = rolling_origin_splits(self.t, n_folds=4, initial_frac=0.5, gap=gap)
        self.assertEqual(len(splits), 4)
        prev_end = None
        for tr, te, info in splits:
            self.assertFalse((tr & te).any())
            self.assertLess(self.t[tr].max(), self.t[te].min() - gap + pd.Timedelta(seconds=1))
            if prev_end is not None:
                self.assertGreater(self.t[te].min(), prev_end)          # test blocks are consecutive and later
            prev_end = self.t[te].max()
        self.assertGreaterEqual(self.t[splits[0][0]].nunique(), 1)

    def test_expanding_window(self):
        sizes = [int(tr.sum()) for tr, _, _ in rolling_origin_splits(self.t, 4, 0.5)]
        self.assertEqual(sizes, sorted(sizes))

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            rolling_origin_splits(pd.Series(pd.date_range("2024-01-01", periods=3)), 4)


class Inference(unittest.TestCase):
    def test_ols_hac_recovers_coefficients(self):
        rng = np.random.default_rng(1); x = rng.normal(size=500); y = 2 + 3 * x + rng.normal(size=500)
        r = ols_hac(x[:, None], y, names=["x"]).set_index("term")
        self.assertAlmostEqual(r.loc["x", "coef"], 3, delta=0.2)
        self.assertGreater(r.loc["x", "se_hac"], 0)
        self.assertLess(r.loc["x", "p_value"], 1e-6)

    def test_hac_wider_than_naive_under_autocorrelation(self):
        rng = np.random.default_rng(2); n = 600
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.9 * e[i - 1] + rng.normal()
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.normal()
        y = 1 + 0.0 * x + e
        self.assertGreater(ols_hac(x[:, None], y, lags=20).set_index("term").loc["x0", "se_hac"],
                           ols_hac(x[:, None], y, lags=0).set_index("term").loc["x0", "se_hac"])


if __name__ == "__main__":
    unittest.main()
