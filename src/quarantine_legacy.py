"""Separate LEGACY (pre-mode-separation, unlabeled) artifacts from the new real/sample layout. NOTHING IS DELETED.

Legacy layout:  data/raw  data/processed  data/external  reports/figures  reports/data
 * derived artifacts (processed/external/figures/report data) are MOVED to data/_legacy_unlabeled_<timestamp>/
   because their provenance is unknown (they may be synthetic - e.g. the 291,155-trip file matches the generator)
 * raw files are inspected: synthetic-looking CSVs are moved too; zips are CRC-tested and REPORTED; real/unknown files stay
Default is a dry run.  python src/quarantine_legacy.py [--apply] [--root PATH]
"""
import argparse
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

SYNTH_RE = re.compile(r"\d+ St & \d+ Ave #\d+")
LEGACY_DERIVED = ["data/processed", "data/external", "reports/figures", "reports/data"]


def looks_synthetic(csv):
    try:
        return bool(SYNTH_RE.search(Path(csv).read_text(errors="ignore")[:20000]))
    except OSError:
        return False


def zip_status(p):
    try:
        if not zipfile.is_zipfile(p):
            return "INVALID (not a zip / truncated)"
        with zipfile.ZipFile(p) as z:
            return "INVALID (CRC failure)" if z.testzip() else "valid"
    except (OSError, zipfile.BadZipFile) as e:
        return f"INVALID ({e})"


def plan(root):
    root = Path(root)
    moves, report = [], []
    for rel in LEGACY_DERIVED:
        d = root / rel
        if d.is_dir():
            for item in sorted(d.iterdir()):
                if item.name.startswith("."):
                    continue
                moves.append((item, rel + "/" + item.name, "legacy derived artifact (unlabeled provenance)"))
    raw = root / "data" / "raw"
    if raw.is_dir():
        for p in sorted(raw.iterdir()):
            if p.suffix == ".csv" and looks_synthetic(p):
                moves.append((p, "data/raw/" + p.name, "synthetic-looking CSV (matches make_sample_data.py pattern)"))
            elif p.suffix == ".zip":
                report.append({"file": p.name, "bytes": p.stat().st_size, "zip": zip_status(p), "action": "left in place"})
            elif p.is_file():
                report.append({"file": p.name, "bytes": p.stat().st_size, "action": "left in place (not recognised as synthetic)"})
    return moves, report


def main(root=None, apply=False):
    root = Path(root or Path(__file__).resolve().parents[1])
    moves, report = plan(root)
    print(f"{'APPLY' if apply else 'DRY RUN'} - {len(moves)} item(s) would be moved, 0 deleted")
    for src, rel, why in moves:
        print(f"  move {rel}   [{why}]")
    for r in report:
        print("  keep", r)
    if not apply or not moves:
        return {"moves": [(str(m[0]), m[2]) for m in moves], "kept": report, "applied": False}
    dest = root / "data" / f"_legacy_unlabeled_{datetime.now():%Y%m%d_%H%M%S}"
    log = []
    for src, rel, why in moves:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        log.append({"from": rel, "to": str(target.relative_to(root)), "reason": why})
    (dest / "MOVE_MANIFEST.json").write_text(json.dumps({"moved": log, "kept_in_place": report}, indent=2))
    print("moved ->", dest)
    return {"moves": log, "kept": report, "applied": True, "dest": str(dest)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root")
    a = ap.parse_args()
    main(a.root, a.apply)
