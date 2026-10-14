"""Smoke test for app/dashboard.py with FAKE streamlit/altair/pydeck.
Validates the Python control flow on realistic report data (every tab body runs, empty states, graceful chart
failures, synthetic banner). It does NOT validate Altair/pydeck specs or the visual rendering - open the dashboard to check those."""
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import _env  # noqa: F401
import numpy as np
import pandas as pd
import requests

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

DASH = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


# ------------------------------------------------------------------ fake streamlit
class W:
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def radio(self, label, options, index=0, **k): return list(options)[index]
    def slider(self, label, *a, **k): return k.get("value", a[2] if len(a) > 2 else a[0])
    def text_input(self, label, value="", **k): return value
    def checkbox(self, label, value=False, **k): return value
    def button(self, *a, **k): return False


class Ctx(W):
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):                       # markdown/caption/... inside containers & sidebar
        return getattr(self._st, name)


class Cache:
    def __call__(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)
    def clear(self): pass


class FakeSt(W):
    def __init__(self):
        self.calls, self.html = {}, []
        self.sidebar, self.cache_data, self.cache_resource = Ctx(self), Cache(), Cache()

    def _rec(self, name): self.calls[name] = self.calls.get(name, 0) + 1

    def markdown(self, body, *a, **k): self._rec("markdown"); self.html.append(str(body))
    def caption(self, body, *a, **k): self._rec("caption"); self.html.append(str(body))
    def tabs(self, labels): self.tab_labels = list(labels); return [Ctx(self) for _ in labels]
    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def container(self, **k): return Ctx(self)
    def expander(self, *a, **k): return Ctx(self)
    def stop(self): raise SystemExit("st.stop")
    def __getattr__(self, name):
        def f(*a, **k): self._rec(name)
        return f


def build_modules(altair_fails=False):
    st = FakeSt()
    alt = mock.MagicMock()
    if altair_fails:
        alt.Chart.side_effect = ValueError("boom")
    pdk = mock.MagicMock()
    return st, {"streamlit": st, "altair": alt, "pydeck": pdk}


# ------------------------------------------------------------------ report data builder
def make_run(root, synthetic=False, full=True):
    tag = "2024-06_2024-06"
    rep = Path(root) / "reports" / ("sample" if synthetic else "real") / tag
    D = rep / "data"; (rep / "figures").mkdir(parents=True, exist_ok=True); D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    man = {"mode": "sample" if synthetic else "real", "is_synthetic": synthetic, "data_source": "test",
           "start_month": "2024-06", "end_month": "2024-06", "run_tag": tag, "created_at": "2026-10-09T10:00:00+00:00",
           "environment": {"python": "3.12", "packages": {"duckdb": "1.5.6"}},
           "stages": {"cleaning": {"rows_parsed": 1000, "rows_final": 950, "duplicates_removed": 3, "csv_files": ["a.csv"],
                                   "coverage": {"first_trip": "2024-06-01 00:00:03", "last_trip": "2024-06-30 23:59:00",
                                                "rows_by_month": {"2024-06": 950}}},
                      "weather": {"source": "cache", "hours_present": 720, "hours_expected": 720, "missing_share": 0.0,
                                  "covered_through": "2024-06-30"},
                      "features": {"rows": 950, "no_weather_share": 0.0}}}
    (D / "run_manifest.json").write_text(json.dumps(man))
    if not full:
        return D
    dates = pd.date_range("2024-06-01", periods=30)
    pd.DataFrame({"trip_date": dates, "trips": rng.integers(100_000, 170_000, 30), "avg_duration_min": 12.0, "avg_distance_km": 1.9,
                  "is_weekend": (dates.dayofweek >= 5).astype(int), "is_holiday": 0, "temp_c": rng.normal(22, 3, 30),
                  "precip_mm": rng.exponential(2, 30), "snow_cm": 0.0, "wind_kmh": rng.gamma(4, 3, 30)}).to_pickle(D / "daily.parquet")
    hr = [{"hour": h, "is_weekend": we, "avg_trips": float(1000 + 800 * np.sin(h / 4)), "day_type": "Weekend" if we else "Weekday"}
          for h in range(24) for we in (False, True)]
    pd.DataFrame(hr).to_csv(D / "hourly.csv", index=False)
    pd.DataFrame([{"day_of_week": d, "hour": h, "trips": int(rng.integers(100, 5000))} for d in range(1, 8) for h in range(24)]
                 ).to_csv(D / "heat.csv", index=False)
    pd.DataFrame({"user_type": ["member", "casual"], "trips": [770, 230], "med_duration_min": [8.9, 14.1], "med_distance_km": [1.5, 1.7]}
                 ).to_csv(D / "user_summary.csv", index=False)
    pd.DataFrame([{"user_type": u, "hour": h, "trips": int(rng.integers(10, 900)), "med_duration_min": 10 + rng.random()}
                  for u in ("member", "casual") for h in range(24)]).to_csv(D / "user_hourly.csv", index=False)
    pd.DataFrame({"rideable_type": ["classic_bike", "electric_bike"], "trips": [400, 600], "med_duration_min": [10.0, 9.0],
                  "med_speed_kmh": [11.0, 15.0]}).to_csv(D / "bike_type.csv", index=False)
    n = 80; starts = rng.integers(50, 3000, n); ends = rng.integers(50, 3000, n)
    stn = pd.DataFrame({"station_id": [f"S{i}" for i in range(n)], "station_name": [f"{i} St & {i % 9} Ave" for i in range(n)],
                        "lat": rng.uniform(40.7, 40.8, n), "lng": rng.uniform(-74.0, -73.93, n), "starts": starts, "ends": ends})
    stn["activity"] = stn.starts + stn.ends; stn["net_outflow"] = stn.starts - stn.ends
    stn["imbalance_ratio"] = stn.net_outflow / stn.activity; stn["eligible_for_ranking"] = stn.activity >= stn.activity.median()
    stn.to_pickle(D / "stations.parquet")
    pd.DataFrame({"start_station_name": ["A", "B"], "end_station_name": ["B", "C"], "trips": [500, 400]}).to_csv(D / "top_od_pairs.csv", index=False)
    pd.DataFrame({"rain_bin": ["none", "0-0.5mm", "0.5-2.5mm", ">2.5mm"], "median": [1.0, .95, .8, .6], "mean": [1, .9, .8, .6],
                  "count": [500, 60, 40, 20]}).to_csv(D / "rain_ratio.csv", index=False)
    pd.DataFrame({"temp_bin": ["<0C", "0-10C", "10-20C", "20-30C", ">30C"], "median": [.5, .8, 1, 1.1, .9], "mean": [.5, .8, 1, 1.1, .9],
                  "count": [5, 50, 200, 400, 30]}).to_csv(D / "temp_ratio.csv", index=False)
    terms = ["const", "temp_per_5C", "precip_per_5mm_day", "wind_per_10kmh", "is_weekend", "is_holiday", "trend_per_30d"]
    pd.DataFrame({"term": terms, "coef": 0.0, "se_hac": .1, "z": 0, "p_value": .5, "ci95_low": -.1, "ci95_high": .1,
                  "pct_effect": [0, -2.5, -1, 0.5, -10, 0, 1], "pct_ci95_low": [0, -6, -3, -2, -15, -1, -1], "pct_ci95_high": [0, 1.3, 1, 3, -5, 1, 3],
                  "n_days": 30, "reliability": "exploratory only (<60 days)"}).to_csv(D / "weather_adjusted.csv", index=False)
    pd.DataFrame({"step": ["csv_data_lines (all files)", "rows_parsed", "excluded: duration_below_min", "excluded: end_outside_nyc_box",
                           "excluded: duplicate_ride_id", "FINAL rows kept", "info (kept): missing_end_coords"],
                  "rows": [1000, 1000, 40, 7, 3, 950, 12], "pct_of_parsed": [100, 100, 4, .7, .3, 95, 1.2]}).to_csv(D / "exclusion_report.csv", index=False)
    (rep / "EDA_insights.md").write_text("# notes\n\n- Busiest day 2024-06-01\n- Weekday peak hour 17:00\n- Members make 77% of trips\n- Weather (association) ...\n")

    def metr(names, base_mae, count):
        out = {}
        for i, nm in enumerate(names):
            mae = base_mae * (1.0 + 0.1 * i) if nm.startswith("Baseline") else base_mae * 0.8
            m = {k: {"mean": v, "std": 0.02} for k, v in {"MAE": mae, "RMSE": mae * 1.5, "R2": .3, "bias": 0.01, "WAPE_%": 40.0}.items()}
            m.update({"sMAPE_%": {"mean": 50.0, "std": 1.0}, "PoissonDeviance": {"mean": .9, "std": .1}, "zero_share": {"mean": .6, "std": .01}} if count
                     else {"MedianAE": {"mean": mae * .7, "std": .01}, "MAPE_%": {"mean": 40.0, "std": 2.0}})
            out[nm] = m
        return out
    dur_names = ["Baseline: global median", "Baseline: median by user x bike", "Baseline: distance x median pace", "Linear (OLS)", "Ridge", "Lasso", "Poly2+Ridge"]
    dem_names = ["Baseline: station mean", "Baseline: persistence (lag h)", "Linear (OLS)", "Ridge"]
    res = {"meta": {"mode": "real", "is_synthetic": synthetic, "run": tag, "n_folds": 4, "validation": "rolling-origin"},
           "duration_model": {"summary": metr(dur_names, 4.0, False), "best_baseline": "Baseline: distance x median pace",
                              "skill_mae_vs_best_baseline": .12, "interval_80": {"q10_log": -.4, "q90_log": .5, "empirical_coverage_last_block": .79},
                              "assumptions": ["a", "b"], "folds": []},
           "demand_model": {f"h{h}": {"summary": metr(dem_names, 0.9, True), "best_baseline": "Baseline: station mean", "horizon_hours": h,
                                      "skill_mae_vs_best_baseline": -.01, "n_stations": 150, "capacity_note": "no capacity", "assumptions": ["x"]}
                            for h in (1, 24)}}
    (D / "model_results.json").write_text(json.dumps(res))
    bt = pd.DataFrame([{"fold": f, "model": m, "MAE": 1 + .1 * f} for f in range(1, 5) for m in dur_names])
    bt.to_csv(D / "backtest_duration.csv", index=False)
    for h in (1, 24):
        bt.to_csv(D / f"backtest_demand_h{h}.csv", index=False)
    if HAVE_SK:
        num = ["log_dist", "hour_sin", "hour_cos", "is_weekend", "is_holiday", "temp_c", "humidity", "precip_mm", "wind_kmh"]
        cat = ["user_type", "rideable_type", "dow"]
        m = 300
        X = pd.DataFrame({"log_dist": rng.random(m) * 2, "hour_sin": rng.normal(size=m), "hour_cos": rng.normal(size=m), "is_weekend": rng.integers(0, 2, m),
                          "is_holiday": 0, "temp_c": rng.normal(20, 5, m), "humidity": rng.uniform(30, 90, m), "precip_mm": rng.exponential(.3, m),
                          "wind_kmh": rng.gamma(4, 3, m), "user_type": rng.choice(["member", "casual"], m),
                          "rideable_type": rng.choice(["classic_bike", "electric_bike"], m), "dow": rng.choice([str(i) for i in range(1, 8)], m)})
        pipe = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat)])),
                         ("m", LinearRegression())]).fit(X[num + cat], 1.5 + X.log_dist)
        joblib.dump({"pipeline": pipe, "features": num + cat, "resid_q10_log": -.4, "resid_q90_log": .5, "test_coverage_80": .79,
                     "is_synthetic": synthetic, "run": tag, "trained_through": "2024-06-24 00:00:00",
                     "fixed_training_conditions": "distance uses the ACTUAL end location"}, D / "duration_model.joblib")
    return D


def run_dashboard(root, altair_fails=False):
    os.environ["CITIBIKE_ROOT"] = str(root)
    st, mods = build_modules(altair_fails)
    with mock.patch.dict(sys.modules, mods), \
         mock.patch.object(pd, "read_parquet", lambda p, *a, **k: pd.read_pickle(p)), \
         mock.patch.object(requests, "get", side_effect=requests.exceptions.ConnectionError("offline")):
        runpy.run_path(str(DASH), run_name="__main__")
    st.mods = mods
    return st


class DashboardSmoke(unittest.TestCase):
    def setUp(self):
        self._root_before = os.environ.get("CITIBIKE_ROOT")

    def tearDown(self):
        os.environ["CITIBIKE_ROOT"] = self._root_before

    def test_full_real_run_renders_every_tab(self):
        root = tempfile.mkdtemp(); make_run(root)
        st = run_dashboard(root)
        self.assertEqual(len(st.tab_labels), 9)
        page = "\n".join(st.html)
        self.assertIn("REAL DATA", page); self.assertNotIn("SYNTHETIC DATA", page)
        self.assertIn("Busiest day", page)                                   # key-fact cards from EDA notes
        self.assertIn("Skill vs baseline", page)                             # models tab KPIs
        self.assertIn("Predicted typical duration", page)                    # predict tab
        self.assertIn("Why rows were excluded", page)                        # data-quality tab
        self.assertIn("Could not reach the live feed", page)                 # live tab degrades gracefully offline
        self.assertNotIn("chart is unavailable", page)                       # no builder failed silently
        self.assertGreaterEqual(st.calls.get("altair_chart", 0), 15)
        layers = st.mods["pydeck"].Layer.call_args_list
        self.assertTrue(layers)
        for call in layers:                                                  # NaN in map data would blank the map in the browser
            self.assertEqual(int(call.kwargs["data"].isna().sum().sum()), 0)
        self.assertGreaterEqual(st.calls.get("pydeck_chart", 0), 1)

    @unittest.skipUnless(HAVE_SK, "scikit-learn missing")
    def test_synthetic_run_is_loudly_marked(self):
        root = tempfile.mkdtemp(); make_run(root, synthetic=True)
        page = "\n".join(run_dashboard(root).html)
        self.assertIn("SYNTHETIC DATA", page); self.assertNotIn("REAL DATA", page)
        self.assertIn("trained on SYNTHETIC data", page)

    def test_chart_failures_never_break_the_page(self):
        root = tempfile.mkdtemp(); make_run(root)
        st = run_dashboard(root, altair_fails=True)
        page = "\n".join(st.html)
        self.assertIn("chart is unavailable", page)
        self.assertIn("Predicted typical duration", page)                    # later tabs still rendered
        self.assertEqual(st.calls.get("altair_chart", 0), 0)

    def test_unreadable_model_file_only_breaks_the_predict_tab(self):
        root = tempfile.mkdtemp(); D = make_run(root)
        (D / "duration_model.joblib").write_bytes(b"not a pickle")            # e.g. sklearn version mismatch on a server
        st = run_dashboard(root)
        page = "\n".join(st.html)
        self.assertGreaterEqual(st.calls.get("error", 0), 1)
        self.assertIn("Why rows were excluded", page)                         # tabs AFTER Predict still rendered

    def test_unfinished_run_shows_empty_states_not_errors(self):
        root = tempfile.mkdtemp(); make_run(root, full=False)                # manifest only (eda/modeling not done)
        page = "\n".join(run_dashboard(root).html)
        self.assertIn("No EDA output for this run yet", page)
        self.assertIn("Modeling has not run", page)
        self.assertIn("python src/modeling.py", page)

    def test_no_runs_at_all(self):
        root = tempfile.mkdtemp()
        os.environ["CITIBIKE_ROOT"] = root
        st, mods = build_modules()
        with mock.patch.dict(sys.modules, mods):
            with self.assertRaises(SystemExit):                              # st.stop()
                runpy.run_path(str(DASH), run_name="__main__")
        self.assertIn("No finished run found", "\n".join(st.html))


if __name__ == "__main__":
    unittest.main()
