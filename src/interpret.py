"""Generate reports/<mode>/<run>/INTERPRETATION.md from SAVED results only.
Wording is conservative (associations, backtest skill vs baselines); nothing here is written by hand."""
import json
import pandas as pd

import config
import provenance

D = config.REP_DATA


def _f(x, n=2):
    return f"{x:.{n}f}"


def verdict(sk):
    if sk > 0.02:
        return f"improves MAE by {sk:.1%} over the best baseline"
    if sk >= -0.02:
        return f"does not meaningfully improve on the best baseline ({sk:+.1%} MAE)"
    return f"is WORSE than the best baseline ({sk:+.1%} MAE)"


def metric_table(summary, cols):
    rows = []
    for m, v in summary.items():
        rows.append([m] + [f"{v[c]['mean']:.3f}" + (f" ± {v[c]['std']:.3f}" if v[c]["std"] is not None else "") for c in cols])
    return "| model | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    man = provenance.load()
    res = json.loads((D / "model_results.json").read_text())
    L = [f"# Interpretation (auto-generated from saved results)", f"**Run:** `{provenance.banner()}`", ""]
    if config.IS_SYNTHETIC:
        L += ["> **SYNTHETIC DATA.** Everything below describes the random generator used for pipeline testing. "
              "None of it is a finding about Citi Bike or New York.", ""]
    cl = man["stages"].get("cleaning", {})
    if cl:
        L += ["## Data and cleaning",
              f"- {cl['rows_parsed']:,} rows parsed from {len(cl['csv_files'])} CSV file(s); {cl['rows_final']:,} kept "
              f"({100 * (1 - cl['rows_final'] / max(cl['rows_parsed'], 1)):.1f}% excluded). See `exclusion_report.csv` for the per-rule waterfall.", ""]
    a = res["duration_model"]
    L += ["## Model A - trip duration (minutes)",
          f"Backtest: {res['meta']['n_folds']} rolling-origin folds on a {a['n_rows_sampled']:,}-trip random sample (mean ± sd across folds).", "",
          metric_table(a["summary"], ["MAE", "RMSE", "R2", "WAPE_%"]), "",
          f"- OLS {verdict(a['skill_mae_vs_best_baseline'])} ('{a['best_baseline']}').",
          f"- Nominal 80% prediction interval covered {a['interval_80']['empirical_coverage_last_block']:.0%} of trips in the last test block"
          + (" (reasonably calibrated)." if abs(a['interval_80']['empirical_coverage_last_block'] - .8) <= .05 else " (MISCALIBRATED - do not rely on the interval)."),
          "- Low R2 is expected: duration is driven by behaviour (commute vs leisure) that these features do not observe.", ""]
    L += ["## Model B - hourly station demand"]
    for key, b in res["demand_model"].items():
        s = b["summary"]["Linear (OLS)"]
        L += [f"### Horizon {b['horizon_hours']} h ({b['n_stations']} stations)", "",
              metric_table(b["summary"], ["MAE", "RMSE", "WAPE_%", "PoissonDeviance", "bias"]), "",
              f"- OLS {verdict(b['skill_mae_vs_best_baseline'])} ('{b['best_baseline']}').",
              f"- Share of zero-demand station-hours in test blocks: {s['zero_share']['mean']:.0%} (why MAPE is not used).", ""]
    wa = D / "weather_adjusted.csv"
    if wa.exists():
        w = pd.read_csv(wa)
        L += ["## Weather (associations only)",
              f"Daily log-trips regressed on weather, weekend, holiday and trend (n = {int(w.n_days.iloc[0])} days, HAC errors). "
              f"Status: **{w.reliability.iloc[0]}**.", "",
              "```\n" + w[["term", "pct_effect", "pct_ci95_low", "pct_ci95_high"]].round(1).to_string(index=False) + "\n```", "",
              "These are associations in this sample. Temperature is entangled with season, ridership growth and calendar effects; "
              "no causal effect of weather is claimed.", ""]
    L += ["## Limitations", "- See the Assumptions listed in `model_results.json` and the README 'Known limitations' section.", ""]
    (config.REPORT_DIR / "INTERPRETATION.md").write_text("\n".join(L))
    provenance.update("interpretation", {"file": "INTERPRETATION.md"})
    provenance.publish_to_reports()
    print("\n".join(L))


if __name__ == "__main__":
    main()
