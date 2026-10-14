"""PHASE 3a - Linear-regression models with leakage-aware TEMPORAL BACKTESTING.

Model A  trip duration (log target)  - "given origin/destination distance, rider/bike type, time and weather"
Model B  hourly station demand       - forecast HORIZON h in {1, 24} hours ahead

Validation design (both models)
  * rolling-origin expanding window: first 50% of the period is the initial training window, the remaining 50% is
    split into N_FOLDS consecutive test blocks; each fold trains ONLY on earlier data
  * Model B leaves a gap equal to the horizon between train end and test start (targets not yet realised at origin)
  * Model B's station universe is chosen from the first 50% of the period only (no selection leakage)
  * baselines are fitted on the training window only
  * scalers/encoders live inside the sklearn Pipeline and are refit per fold
Metrics: MAE/RMSE/R2/bias/WAPE; zero-safe sMAPE and Poisson deviance for counts (no silent dropping of y=0).
Coefficients describe associations in this sample, not causal effects.
"""
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

import config
import provenance
from config import FIG, REP_DATA, sql_path
from stats_utils import regression_metrics, rolling_origin_splits

provenance.require_stage("features")
sns.set_theme(style="whitegrid")
N_FOLDS = 4
HORIZONS = (1, 24)
con = duckdb.connect()
con.execute("PRAGMA temp_directory='duckdb_tmp'")
TRIPS = f"read_parquet('{sql_path(config.TRIPS_FEAT)}')"


def savefig(name):
    plt.tight_layout()
    provenance.stamp_figure()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def make_pipe(model, num, cat, poly_cols=None):
    if poly_cols:
        pc = [c for c in num if c in poly_cols]
        tf = [("num", StandardScaler(), [c for c in num if c not in pc]),
              ("poly", Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(2, include_bias=False))]), pc)]
    else:
        tf = [("num", StandardScaler(), num)]
    if cat:
        tf.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat))
    return Pipeline([("prep", ColumnTransformer(tf)), ("m", model)])


def coef_table(pipe, num, cat):
    names = list(num)
    if cat:
        names += list(pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat))
    return pd.DataFrame({"feature": names, "coef": pipe.named_steps["m"].coef_}).sort_values("coef", key=abs, ascending=False)


def summarize(fold_df):
    metric_cols = [c for c in fold_df.columns if c not in ("fold", "model", "test_start", "test_end", "n_train", "n_test")]
    g = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    return {m: {c: {"mean": float(g.loc[m, (c, "mean")]), "std": float(g.loc[m, (c, "std")]) if len(fold_df[fold_df.model == m]) > 1 else None}
                for c in metric_cols} for m in g.index}


def skill(summary, model, baselines):
    best = min(baselines, key=lambda b: summary[b]["MAE"]["mean"])
    return best, float(1 - summary[model]["MAE"]["mean"] / summary[best]["MAE"]["mean"])


# =====================================================================================================
# MODEL A - trip duration
# =====================================================================================================
def baselines_duration(tr, te):
    med = tr.trip_duration_min.median()
    g = tr.groupby(["user_type", "rideable_type"]).trip_duration_min.median().rename("m")
    pace = (tr.trip_duration_min / tr.trip_distance_km.clip(lower=0.05)).groupby(
        [tr.user_type, tr.rideable_type]).median().rename("pace")
    p = te.join(pace, on=["user_type", "rideable_type"])["pace"].fillna(pace.median())
    return {"Baseline: global median": np.full(len(te), med),
            "Baseline: median by user x bike": te.join(g, on=["user_type", "rideable_type"])["m"].fillna(med).to_numpy(),
            "Baseline: distance x median pace": np.clip(p.to_numpy() * te.trip_distance_km.to_numpy(), 1, None)}


def model_duration(n_sample=1_000_000):
    print("\n=== Model A: trip duration ===")
    df = con.execute(f"""
      SELECT * FROM (
        SELECT trip_date, hour, day_of_week, is_weekend::INT AS is_weekend, is_holiday::INT AS is_holiday,
               user_type, rideable_type, trip_distance_km, temp_c, humidity, precip_mm, wind_kmh, trip_duration_min
        FROM {TRIPS}
        WHERE trip_distance_km IS NOT NULL AND temp_c IS NOT NULL AND user_type IS NOT NULL
          AND rideable_type IS NOT NULL AND NOT is_staff_trip) USING SAMPLE reservoir({n_sample} ROWS)""").df()
    df["trip_date"] = pd.to_datetime(df.trip_date)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    df["dow"] = df.day_of_week.astype(str)
    df["log_dist"] = np.log1p(df.trip_distance_km)
    df["y"] = np.log1p(df.trip_duration_min)
    num = ["log_dist", "hour_sin", "hour_cos", "is_weekend", "is_holiday", "temp_c", "humidity", "precip_mm", "wind_kmh"]
    cat = ["user_type", "rideable_type", "dow"]
    X = num + cat
    poly_cols = ["log_dist", "hour_sin", "hour_cos", "temp_c"]
    models = {"Linear (OLS)": lambda: make_pipe(LinearRegression(), num, cat),
              "Ridge": lambda: make_pipe(Ridge(alpha=10), num, cat),
              "Lasso": lambda: make_pipe(Lasso(alpha=1e-4), num, cat),
              "Poly2+Ridge": lambda: make_pipe(Ridge(alpha=10), num, cat, poly_cols)}
    splits = rolling_origin_splits(df.trip_date, N_FOLDS, 0.5, gap=pd.Timedelta(0))
    rows = []
    for tr_m, te_m, info in splits:
        tr, te = df[tr_m], df[te_m]
        preds = baselines_duration(tr, te)
        for name, mk in models.items():
            pipe = mk().fit(tr[X], tr.y)
            preds[name] = np.expm1(pipe.predict(te[X]))          # back to minutes
        for name, p in preds.items():
            rows.append({**info, "model": name, **regression_metrics(te.trip_duration_min, p)})
        print(f"  fold {info['fold']}/{len(splits)}: train {info['n_train']:,} / test {info['n_test']:,}", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(REP_DATA / "backtest_duration.csv", index=False)
    summary = summarize(folds)
    baselines = [m for m in summary if m.startswith("Baseline")]
    best_b, sk = skill(summary, "Linear (OLS)", baselines)
    print(f"  OLS MAE {summary['Linear (OLS)']['MAE']['mean']:.2f} min vs best baseline ({best_b}) "
          f"{summary[best_b]['MAE']['mean']:.2f} -> skill {sk:+.1%}")

    # final model = last fold's training window; uncertainty = empirical log-residual quantiles, coverage checked on the test block
    tr_m, te_m, info = splits[-1]
    tr, te = df[tr_m], df[te_m]
    final = models["Linear (OLS)"]().fit(tr[X], tr.y)
    q10, q90 = np.quantile(tr.y - final.predict(tr[X]), [0.1, 0.9])
    pl = final.predict(te[X])
    coverage = float(np.mean((te.y >= pl + q10) & (te.y <= pl + q90)))
    joblib.dump({"pipeline": final, "features": X, "resid_q10_log": float(q10), "resid_q90_log": float(q90),
                 "test_coverage_80": coverage, "is_synthetic": config.IS_SYNTHETIC, "run": f"{config.MODE}/{config.RUN_TAG}",
                 "trained_through": info["test_start"],
                 "fixed_training_conditions": "distance uses the ACTUAL end location; weather is observed at trip start"},
                REP_DATA / "duration_model.joblib")
    coef_table(final, num, cat).to_csv(REP_DATA / "coef_duration.csv", index=False)

    resid = te.y.to_numpy() - pl
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].scatter(pl[:20000], resid[:20000], s=2, alpha=.3); ax[0].axhline(0, c="r")
    ax[0].set(title="Residuals vs fitted (log, last test block)", xlabel="fitted")
    sns.histplot(resid, bins=80, ax=ax[1]); ax[1].set_title("Residual distribution")
    stats.probplot(resid[:20000], dist="norm", plot=ax[2]); ax[2].set_title("Q-Q plot")
    savefig("12_duration_residuals.png")
    ea = te.assign(abs_err=np.abs(te.trip_duration_min.to_numpy() - np.expm1(pl)), rainy=np.where(te.precip_mm > 0.1, "rain", "dry"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ea.groupby("hour").abs_err.mean().plot(ax=ax[0], title="MAE by hour (min)")
    ea.groupby("user_type").abs_err.mean().plot.bar(ax=ax[1], title="MAE by user type")
    ea.groupby("rainy").abs_err.mean().plot.bar(ax=ax[2], title="MAE by weather")
    savefig("13_duration_error_analysis.png")
    top = coef_table(final, num, cat).head(12).iloc[::-1]
    plt.figure(figsize=(8, 5)); plt.barh(top.feature, top.coef); plt.title("Largest standardised coefficients (log-duration; associations)")
    savefig("14_duration_coefficients.png")
    return {"target": "trip_duration_min (back-transformed from log1p)", "n_rows_sampled": len(df),
            "folds": [i for _, _, i in splits], "summary": summary, "best_baseline": best_b,
            "skill_mae_vs_best_baseline": sk,
            "interval_80": {"q10_log": float(q10), "q90_log": float(q90), "empirical_coverage_last_block": coverage},
            "assumptions": [
                "Distance is straight-line from the ACTUAL end location, so the model assumes the destination is known at prediction time.",
                "Weather features are observed at trip start (a stand-in for a forecast).",
                "Random 1M-row sample of eligible trips (distance and weather available, non-staff); trips without end coordinates are excluded.",
                "Coefficients are associations; duration is dominated by unobserved behaviour (sightseeing vs commuting)."]}


# =====================================================================================================
# MODEL B - hourly station demand, h-hour-ahead forecast
# =====================================================================================================
def fetch_demand_sql(H, n_stations):
    """SQL only: station x hour grid with lags/rolling features. Returns (frame, station-selection cutoff)."""
    lo, hi = con.execute(f"SELECT min(hour_ts), max(hour_ts) FROM {TRIPS}").fetchone()
    sel_cut = (pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) * 0.5).floor("h")
    lag24 = "" if H == 24 else ", lag(demand, 24) OVER (PARTITION BY sid ORDER BY hour_ts) AS lag_24h"
    lag24_sel = "" if H == 24 else "l.lag_24h,"
    df = con.execute(f"""
      WITH top AS (SELECT start_station_id AS sid FROM {TRIPS}
                   WHERE start_station_id IS NOT NULL AND hour_ts < TIMESTAMP '{sel_cut}'
                   GROUP BY 1 ORDER BY count(*) DESC LIMIT {n_stations}),
      bounds AS (SELECT min(hour_ts) AS lo, max(hour_ts) AS hi FROM {TRIPS}),
      hours AS (SELECT unnest(generate_series(lo, hi, INTERVAL 1 HOUR)) AS hour_ts FROM bounds),
      grid AS (SELECT sid, hour_ts FROM top CROSS JOIN hours),
      cnt AS (SELECT start_station_id AS sid, hour_ts, count(*) AS demand FROM {TRIPS}
              WHERE start_station_id IN (SELECT sid FROM top) GROUP BY 1, 2),
      w AS (SELECT * FROM read_parquet('{sql_path(config.WEATHER_PQ)}')),
      base AS (SELECT g.sid, g.hour_ts, coalesce(c.demand, 0) AS demand FROM grid g LEFT JOIN cnt c USING (sid, hour_ts)),
      lagged AS (
        SELECT *, lag(demand, {H}) OVER (PARTITION BY sid ORDER BY hour_ts) AS lag_h {lag24},
          lag(demand, 168) OVER (PARTITION BY sid ORDER BY hour_ts) AS lag_168h,
          avg(demand) OVER (PARTITION BY sid ORDER BY hour_ts ROWS BETWEEN {H + 23} PRECEDING AND {H} PRECEDING) AS roll_24h
        FROM base)
      SELECT l.sid, l.hour_ts, l.demand, l.lag_h, {lag24_sel} l.lag_168h, l.roll_24h,
             hour(l.hour_ts) AS hour, isodow(l.hour_ts) AS dow_i, (isodow(l.hour_ts) >= 6)::INT AS is_weekend,
             w.temp_c, w.humidity, w.precip_mm, w.wind_kmh
      FROM lagged l JOIN w ON w.hour_ts = l.hour_ts
      WHERE l.lag_168h IS NOT NULL""").df()
    return df, str(sel_cut)


def add_calendar_features(df):
    df = df.copy()
    df["hour_ts"] = pd.to_datetime(df.hour_ts)
    hol = USFederalHolidayCalendar().holidays(start=str(config.RANGE_START), end=str(config.RANGE_END_EXCL))
    df["is_holiday"] = df.hour_ts.dt.normalize().isin(hol).astype(int)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    df["dow"] = df.dow_i.astype(str)
    return df


def load_demand_frame(H, n_stations):
    df, cut = fetch_demand_sql(H, n_stations)
    return add_calendar_features(df), cut


def baselines_demand(tr, te):
    gm = tr.demand.mean()
    sm = tr.groupby("sid").demand.mean().rename("sm")
    how = tr.groupby(["sid", "dow_i", "hour"]).demand.mean().rename("how")
    st_mean = te.join(sm, on="sid")["sm"].fillna(gm).to_numpy()
    how_p = te.join(how, on=["sid", "dow_i", "hour"])["how"].to_numpy()
    out = {"Baseline: station mean": st_mean,
           "Baseline: station x hour-of-week mean": np.where(np.isnan(how_p), st_mean, how_p),
           "Baseline: persistence (lag h)": te.lag_h.to_numpy(),
           "Baseline: same hour last week": te.lag_168h.to_numpy()}
    if "lag_24h" in te:
        out["Baseline: same hour yesterday"] = te.lag_24h.to_numpy()
    return out


def model_demand(H, n_stations=150):
    print(f"\n=== Model B: hourly station demand, horizon {H}h ===")
    df, sel_cut = load_demand_frame(H, n_stations)
    num = ["lag_h", "lag_168h", "roll_24h", "hour_sin", "hour_cos", "is_weekend", "is_holiday",
           "temp_c", "humidity", "precip_mm", "wind_kmh"]
    if "lag_24h" in df:
        num.insert(1, "lag_24h")
    cap_note = "capacity feature not used"
    if config.STATION_INFO.exists():
        cap = pd.read_parquet(config.STATION_INFO)[["station_id", "capacity"]].drop_duplicates("station_id")
        cap["station_id"] = cap.station_id.astype(str)
        df = df.merge(cap.rename(columns={"station_id": "sid"}), on="sid", how="left")
        df["capacity"] = df.capacity.fillna(df.capacity.median())
        num.append("capacity")
        cap_note = ("capacity = " + ("FAKE synthetic" if config.IS_SYNTHETIC else "CURRENT GBFS snapshot (not historical)"))
    cat = ["dow"]
    X = num + cat
    models = {"Linear (OLS)": lambda: make_pipe(LinearRegression(), num, cat), "Ridge": lambda: make_pipe(Ridge(alpha=10), num, cat)}
    splits = rolling_origin_splits(df.hour_ts, N_FOLDS, 0.5, gap=pd.Timedelta(hours=H))
    rows = []
    for tr_m, te_m, info in splits:
        tr, te = df[tr_m], df[te_m]
        print(f"  h={H} fold {info['fold']}/{len(splits)}: train {info['n_train']:,} / test {info['n_test']:,}", flush=True)
        preds = baselines_demand(tr, te)
        for name, mk in models.items():
            preds[name] = np.clip(mk().fit(tr[X], tr.demand).predict(te[X]), 0, None)
        for name, p in preds.items():
            rows.append({**info, "model": name, **regression_metrics(te.demand, p, count_data=True)})
    folds = pd.DataFrame(rows)
    folds.to_csv(REP_DATA / f"backtest_demand_h{H}.csv", index=False)
    summary = summarize(folds)
    baselines = [m for m in summary if m.startswith("Baseline")]
    best_b, sk = skill(summary, "Linear (OLS)", baselines)
    print(f"  OLS MAE {summary['Linear (OLS)']['MAE']['mean']:.3f} vs best baseline ({best_b}) "
          f"{summary[best_b]['MAE']['mean']:.3f} -> skill {sk:+.1%}")

    tr_m, te_m, info = splits[-1]
    tr, te = df[tr_m], df[te_m]
    lin = models["Linear (OLS)"]().fit(tr[X], tr.demand)
    pred = np.clip(lin.predict(te[X]), 0, None)
    coef_table(lin, num, cat).to_csv(REP_DATA / f"coef_demand_h{H}.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    dec = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"pred": pred, "actual": te.demand.to_numpy(), "d": dec}).groupby("d").mean()
    ax[0].plot(cal.pred, cal.actual, "o-"); ax[0].plot([0, cal.pred.max()], [0, cal.pred.max()], "r--")
    ax[0].set(xlabel="mean predicted", ylabel="mean actual", title=f"Calibration by prediction decile (h={H})")
    pd.Series(np.abs(te.demand.to_numpy() - pred), index=te.hour.to_numpy()).groupby(level=0).mean().plot(ax=ax[1], title="MAE by hour")
    te.assign(pred=pred).groupby("hour")[["demand", "pred"]].mean().plot(ax=ax[2], title="Mean actual vs predicted by hour")
    savefig(f"15_demand_eval_h{H}.png")
    return {"target": "trips starting at station in the hour", "horizon_hours": H, "n_stations": int(df.sid.nunique()),
            "n_rows": len(df), "folds": [i for _, _, i in splits], "summary": summary, "best_baseline": best_b,
            "skill_mae_vs_best_baseline": sk, "station_selection_cutoff": sel_cut, "capacity_note": cap_note,
            "assumptions": [
                f"Forecast origin is {H}h before the target hour; all demand lags/rolling means use only data at or before the origin.",
                "Weather is the OBSERVED value at the target hour, standing in for a perfect weather forecast; real forecast error would reduce accuracy.",
                f"Station universe = top {n_stations} by trips in the first 50% of the period (selection is leakage-free); stations opened later are not modelled.",
                f"Backtest gap between train end and test start = {H}h.",
                "Demand counts trips STARTED (observed rentals), not unmet demand; empty stations censor demand.",
                "Zeros are kept (grid is complete); sMAPE (0/0=0), WAPE and Poisson deviance are reported because MAPE is undefined at 0."]}


if __name__ == "__main__":
    out = {"meta": {"mode": config.MODE, "is_synthetic": config.IS_SYNTHETIC, "run": config.RUN_TAG,
                    "n_folds": N_FOLDS, "validation": "rolling-origin expanding window, 50% initial train"},
           "duration_model": model_duration(),
           "demand_model": {f"h{h}": model_demand(h) for h in HORIZONS}}
    (REP_DATA / "model_results.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    provenance.update("modeling", {"results": "model_results.json", "n_folds": N_FOLDS, "horizons": list(HORIZONS)})
    provenance.publish_to_reports()
    print("\nsaved", REP_DATA / "model_results.json")
