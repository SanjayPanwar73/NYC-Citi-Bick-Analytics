"""Dependency-light statistics helpers (numpy/pandas only) so they can be unit-tested anywhere."""
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- metrics
def regression_metrics(y, p, count_data=False):
    """MAE/RMSE/R2/bias/WAPE always. sMAPE (0/0 := 0) and Poisson deviance for count targets, since MAPE is
    undefined at y=0 and silently dropping zeros would bias the metric toward busy hours."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {"MAE": float(np.mean(np.abs(err))),
           "RMSE": float(np.sqrt(np.mean(err ** 2))),
           "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
           "bias": float(np.mean(err)),
           "WAPE_%": float(100 * np.sum(np.abs(err)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else float("nan")}
    if count_data:
        out["sMAPE_%"] = smape(y, p)
        out["PoissonDeviance"] = poisson_deviance(y, p)
        out["zero_share"] = float(np.mean(y == 0))
    else:
        out["MedianAE"] = float(np.median(np.abs(err)))
        out["MAPE_%"] = float(100 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9)))   # durations are >= 1 min
    return out


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(p - y) / np.where(denom == 0, 1, denom))
    return float(100 * ratio.mean())


def poisson_deviance(y, mu, eps=1e-3):
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), eps, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


# ---------------------------------------------------------------- temporal validation
def rolling_origin_splits(times, n_folds=4, initial_frac=0.5, gap=pd.Timedelta(0)):
    """Expanding-window backtest. Fold k trains on rows with time < test_start - gap and tests on one
    contiguous later block. `gap` should equal the forecast horizon so no training target is 'in the future'
    relative to the forecast origin. Returns list of (train_mask, test_mask, info)."""
    t = pd.to_datetime(pd.Series(np.asarray(times))).reset_index(drop=True)
    u = np.sort(t.unique())
    n_init = int(len(u) * initial_frac)
    rest = u[n_init:]
    if len(rest) < n_folds or n_init < 1:
        raise ValueError(f"Not enough distinct time points ({len(u)}) for {n_folds} folds.")
    edges = np.array_split(rest, n_folds)
    out = []
    for i, block in enumerate(edges):
        start, end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        train = (t < start - gap).to_numpy()
        test = ((t >= start) & (t <= end)).to_numpy()
        out.append((train, test, {"fold": i + 1, "test_start": str(start), "test_end": str(end),
                                  "n_train": int(train.sum()), "n_test": int(test.sum())}))
    return out


# ---------------------------------------------------------------- inference
def ols_hac(X, y, names=None, lags=7, add_const=True):
    """OLS with Newey-West (HAC) standard errors - appropriate for autocorrelated daily series."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["const"] + list(names or [f"x{i}" for i in range(X.shape[1] - 1)])
    n, k = X.shape
    if n <= k:
        raise ValueError("More parameters than observations.")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        G = xe[l:].T @ xe[:-l]
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv * n / (n - k)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = beta / np.where(se == 0, np.nan, se)
    p = np.array([math.erfc(abs(v) / math.sqrt(2)) if np.isfinite(v) else np.nan for v in z])
    return pd.DataFrame({"term": names, "coef": beta, "se_hac": se, "z": z, "p_value": p,
                         "ci95_low": beta - 1.96 * se, "ci95_high": beta + 1.96 * se})
