"""Run the pipeline (resumable: stages already recorded in the run manifest are skipped unless --force).

    python run_all.py status                 show which stages are done / pending for the current run
    python run_all.py all [--force]          REAL data, every phase
    python run_all.py phase1|phase2|phase3   one phase of the REAL pipeline
    python run_all.py sample [--force]       SYNTHETIC test run (DATA_MODE=sample; separate folders, watermarked)
    python run_all.py check                  unit tests (no data needed)

Range: set START_MONTH / END_MONTH (YYYY-MM) first, e.g.  set START_MONTH=2024-06  &  set END_MONTH=2024-06
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
PHASES = {
    "phase1": ["download_data.py", "cleaning.py", "weather.py", "features.py", "station_info.py"],
    "phase2": ["eda.py"],
    "phase3": ["modeling.py", "interpret.py"],
    "sample": ["make_sample_data.py", "cleaning.py", "weather.py", "features.py", "eda.py", "modeling.py", "interpret.py"],
}
STAGE_OF = {"download_data.py": "download", "cleaning.py": "cleaning", "weather.py": "weather", "features.py": "features",
            "station_info.py": "station_info", "eda.py": "eda", "modeling.py": "modeling",
            "interpret.py": "interpretation", "make_sample_data.py": "source"}
OPTIONAL = {"station_info.py"}          # needs network + only adds a capacity feature; failure never stops the run


def done_stages(env):
    os.environ.update({k: env[k] for k in ("DATA_MODE", "START_MONTH", "END_MONTH") if k in env})
    sys.path.insert(0, str(SRC))
    import config, provenance
    return config, provenance.completed_stages()


def run(scripts, env, force):
    config, done = done_stages(env)
    print(f"Run: [{config.MODE}] {config.START_MONTH}..{config.END_MONTH}  |  already done: {done or 'nothing'}")
    for script in scripts:
        if not force and STAGE_OF[script] in done:
            print(f"\n--- skip {script} (stage '{STAGE_OF[script]}' already done; use --force to redo)")
            continue
        print(f"\n{'=' * 10} {script} [{config.MODE}] {'=' * 10}", flush=True)
        r = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC, env=env)
        if r.returncode != 0:
            if script in OPTIONAL:
                print(f"WARNING: optional step {script} failed and was skipped.")
                continue
            sys.exit(f"\nSTOPPED: {script} failed (exit {r.returncode}). Fix it, then rerun the same command - "
                     f"completed stages are skipped.")


def status(env, scripts):
    config, done = done_stages(env)
    print(f"Run: [{config.MODE}] {config.START_MONTH}..{config.END_MONTH}")
    nxt = None
    for s in scripts:
        ok = STAGE_OF[s] in done
        print(f"  [{'x' if ok else ' '}] {s}")
        nxt = nxt or (None if ok else s)
    print(f"\nNext: python src/{nxt}" if nxt else "\nAll stages done. Dashboard: python -m streamlit run app/dashboard.py")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    which = args[0] if args else "all"
    if which == "check":
        suite = unittest.defaultTestLoader.discover(str(HERE / "tests"))
        sys.exit(0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1)
    env = os.environ.copy()
    if which == "sample":
        env["DATA_MODE"] = "sample"
    else:
        if env.get("DATA_MODE", "real") != "real":
            sys.exit("Real phases require DATA_MODE=real (or unset). Use `python run_all.py sample` for synthetic runs.")
        env["DATA_MODE"] = "real"
    real_all = [s for k, v in PHASES.items() if k != "sample" for s in v]
    if which == "status":
        status(env, PHASES["sample"] if env.get("DATA_MODE") == "sample" else real_all)
        sys.exit(0)
    if which not in PHASES and which != "all":
        sys.exit(__doc__)
    todo = PHASES["sample"] if which == "sample" else (real_all if which == "all" else PHASES[which])
    run(todo, env, force)
    print("\nDone. Dashboard: python -m streamlit run app/dashboard.py")
