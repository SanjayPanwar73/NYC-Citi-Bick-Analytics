"""Executed in a subprocess by test_modeling_logic.py. duckdb is STUBBED: this validates the Python-side
backtesting / metrics / artifact logic only - it does NOT test the SQL."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
rng = np.random.default_rng(0)


def duration_frame(n=40000):
    d = pd.to_datetime(rng.choice(pd.date_range("2024-01-01", periods=60), n))
    dist = rng.gamma(2, 1.0, n) + 0.2
    bike = rng.choice(["classic_bike", "electric_bike"], n)
    user = rng.choice(["member", "casual"], n, p=[.8, .2])
    dur = dist / np.where(bike == "electric_bike", 16, 11) * 60 * np.exp(rng.normal(0, .3, n)) + np.where(user == "casual", 4, 0) + 2
    hour = rng.integers(0, 24, n)
    return pd.DataFrame({"trip_date": d, "hour": hour, "day_of_week": d.dayofweek + 1, "is_weekend": (d.dayofweek >= 5).astype(int),
                         "is_holiday": 0, "user_type": user, "rideable_type": bike, "trip_distance_km": dist,
                         "temp_c": rng.normal(10, 8, n), "humidity": rng.uniform(30, 90, n),
                         "precip_mm": rng.exponential(.3, n) * (rng.random(n) < .2), "wind_kmh": rng.gamma(4, 3, n),
                         "trip_duration_min": dur})


def demand_frame(H):
    hrs = pd.date_range("2024-01-01", periods=60 * 24, freq="h")
    g = pd.MultiIndex.from_product([[f"S{i}" for i in range(20)], hrs], names=["sid", "hour_ts"]).to_frame(index=False)
    hour = g.hour_ts.dt.hour.to_numpy()
    base = g.sid.str[1:].astype(int).to_numpy() % 5 + 0.5
    temp = 10 + 8 * np.sin(2 * np.pi * g.hour_ts.dt.dayofyear.to_numpy() / 30)
    g["demand"] = rng.poisson(base * (0.3 + 1.2 * np.exp(-((hour - 17) ** 2) / 18)) * (1 + 0.02 * temp).clip(0.3))
    grp = g.groupby("sid").demand
    g["lag_h"] = grp.shift(H)
    if H != 24:
        g["lag_24h"] = grp.shift(24)
    g["lag_168h"] = grp.shift(168)
    g["roll_24h"] = grp.transform(lambda s: s.shift(H).rolling(24, min_periods=1).mean())
    g["hour"] = hour
    g["dow_i"] = g.hour_ts.dt.dayofweek + 1
    g["is_weekend"] = (g.dow_i >= 6).astype(int)
    g["temp_c"], g["humidity"] = temp, 60.0
    g["precip_mm"], g["wind_kmh"] = 0.0, 10.0
    return g.dropna(subset=["lag_168h"]).reset_index(drop=True)


class Res:
    def __init__(self, d): self.d = d
    def df(self): return self.d
    def fetchone(self): return (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))


class Con:
    def execute(self, sql, *a): return Res(duration_frame() if "reservoir" in sql else None)


fake = types.ModuleType("duckdb"); fake.connect = lambda *a, **k: Con(); sys.modules["duckdb"] = fake

import config            # noqa: E402
import provenance        # noqa: E402
provenance.update("features", {"stub": True})
import modeling          # noqa: E402
modeling.fetch_demand_sql = lambda H, n: (demand_frame(H), "stub-cut")     # SQL part stubbed; calendar features are REAL code

a = modeling.model_duration()
b1, b24 = modeling.model_demand(1), modeling.model_demand(24)
bundle = __import__("joblib").load(config.REP_DATA / "duration_model.joblib")
out = {
    "a_ols_mae": a["summary"]["Linear (OLS)"]["MAE"]["mean"], "a_median_mae": a["summary"]["Baseline: global median"]["MAE"]["mean"],
    "a_coverage": a["interval_80"]["empirical_coverage_last_block"], "a_folds": len(a["folds"]),
    "bundle_keys": sorted(bundle), "bundle_synth": bundle["is_synthetic"],
    "b1_ols_mae": b1["summary"]["Linear (OLS)"]["MAE"]["mean"], "b1_mean_mae": b1["summary"]["Baseline: station mean"]["MAE"]["mean"],
    "b1_has_zero_share": "zero_share" in b1["summary"]["Linear (OLS)"], "b1_no_mape": "MAPE_%" not in b1["summary"]["Linear (OLS)"],
    "b1_baselines": [m for m in b1["summary"] if m.startswith("Baseline")],
    "b24_features": pd.read_csv(config.REP_DATA / "coef_demand_h24.csv").feature.tolist(),
    "b1_features": pd.read_csv(config.REP_DATA / "coef_demand_h1.csv").feature.tolist(),
    "b24_gap_note": [x for x in b24["assumptions"] if "gap" in x.lower()],
    "figs": sorted(p.name for p in config.FIG.glob("*.png")),
    "backtest_files": sorted(p.name for p in config.REP_DATA.glob("backtest_*.csv")),
    "skill_finite": bool(np.isfinite(a["skill_mae_vs_best_baseline"]) and np.isfinite(b1["skill_mae_vs_best_baseline"])),
}
print("RESULT=" + json.dumps(out))
