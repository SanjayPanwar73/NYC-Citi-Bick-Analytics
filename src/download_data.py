"""PHASE 1 / step 1 - Safe, resumable download of monthly NYC Citi Bike files (REAL mode only).

Safety rules
  * object sizes come from the S3 listing; a file is 'complete' only if it passes the size check AND a full
    ZIP CRC test, and is then renamed atomically from  <name>.part  and marked with  <name>.ok
  * interrupted downloads resume from the .part file (HTTP Range); transient errors are retried with backoff
  * a corrupt/short file is renamed to <name>.corrupt (kept for inspection, never silently deleted) and refetched
  * CSV extraction goes through temp files; existing CSVs are never overwritten unless their size differs
  * the run fails loudly if any requested month has no CSV afterwards
"""
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

import requests

import config
import provenance

BUCKET = "https://s3.amazonaws.com/tripdata/"
PROGRESS_STEP = 50_000_000          # print progress every ~50 MB


# ------------------------------------------------------------------ listing
def list_objects(session=requests):
    """-> {key: size_bytes}"""
    objs, marker = {}, ""
    while True:
        r = session.get(BUCKET, params={"marker": marker}, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        last = None
        for c in root.findall(".//s:Contents", ns):
            last = c.find("s:Key", ns).text
            objs[last] = int(c.find("s:Size", ns).text)
        if root.find("s:IsTruncated", ns).text != "true" or last is None:
            return objs
        marker = last


def _ym_bounds():
    return config.START_MONTH.replace("-", ""), config.END_MONTH.replace("-", "")


def wanted(key):
    """Monthly (YYYYMM-citibike-tripdata*) or yearly (YYYY-citibike-tripdata*) NYC files; never JC-."""
    if key.startswith("JC-"):
        return False
    lo, hi = _ym_bounds()
    m = re.match(r"(\d{4})(\d{2})-citibike-tripdata", key)
    if m:
        return lo <= m.group(1) + m.group(2) <= hi
    y = re.match(r"(\d{4})-citibike-tripdata", key)
    return bool(y) and lo[:4] <= y.group(1) <= hi[:4]


def csv_in_range(name):
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", name.split("/")[-1])
    if not m:
        return True            # unlabelled: kept; cleaning filters by timestamp anyway
    lo, hi = _ym_bounds()
    return lo <= m.group(1) + m.group(2) <= hi


# ------------------------------------------------------------------ integrity
def validate_zip(path):
    if not zipfile.is_zipfile(path):
        raise zipfile.BadZipFile(f"{path.name}: not a valid ZIP (truncated or HTML error page?)")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()                      # reads every member and verifies CRC
        if bad is not None:
            raise zipfile.BadZipFile(f"{path.name}: CRC failure in member {bad}")
        if not z.namelist():
            raise zipfile.BadZipFile(f"{path.name}: empty archive")


def quarantine(path):
    """Rename a bad file to <name>.corrupt (or .corrupt.1, .corrupt.2 ...) - never overwrite earlier evidence.
    Windows' rename() fails if the target exists, so pick a free name first."""
    target = path.with_name(path.name + ".corrupt")
    n = 0
    while target.exists():
        n += 1
        target = path.with_name(f"{path.name}.corrupt.{n}")
    path.rename(target)
    return target


def _marker(dest):
    return dest.with_name(dest.name + ".ok")


def is_complete(dest, expected_size):
    """True only if file + marker exist and sizes match. A file without marker is re-validated (not trusted)."""
    if not dest.exists():
        return False
    if dest.stat().st_size != expected_size:
        quarantine(dest)
        return False
    mk = _marker(dest)
    if mk.exists() and json.loads(mk.read_text()).get("size") == expected_size:
        return True
    try:
        if dest.suffix == ".zip":
            validate_zip(dest)
    except (zipfile.BadZipFile, OSError):
        quarantine(dest)
        return False
    _write_marker(dest, expected_size)
    return True


def _write_marker(dest, size):
    _marker(dest).write_text(json.dumps({"size": size, "validated_at": datetime.now(timezone.utc).isoformat(),
                                         "zip_crc_ok": dest.suffix == ".zip"}))


def download_file(url, dest, expected_size, session=requests, retries=4, backoff=2.0, chunk=1 << 20,
                  sleep=time.sleep):
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > expected_size:
                quarantine(part)
                have = 0
            if have < expected_size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with session.get(url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    resumed = bool(have) and r.status_code == 206
                    done = have if resumed else 0
                    next_report = (done // PROGRESS_STEP + 1) * PROGRESS_STEP
                    with open(part, "ab" if resumed else "wb") as f:
                        for block in r.iter_content(chunk):
                            if block:
                                f.write(block)
                                done += len(block)
                                if done >= next_report:
                                    print(f"  {dest.name}: {done / 1e6:,.0f} / {expected_size / 1e6:,.0f} MB "
                                          f"({100 * done / expected_size:.0f}%)", flush=True)
                                    next_report += PROGRESS_STEP
            size = part.stat().st_size
            if size != expected_size:
                raise OSError(f"{dest.name}: got {size:,} bytes, expected {expected_size:,} (will resume)")
            if dest.suffix == ".zip":
                try:
                    validate_zip(part)
                except zipfile.BadZipFile:
                    quarantine(part)     # full size but bad -> restart clean
                    raise
            os.replace(part, dest)                    # atomic: dest only ever appears fully validated
            _write_marker(dest, size)
            return size
        except (OSError, zipfile.BadZipFile) as e:   # requests exceptions subclass OSError
            last = e
            print(f"  attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                sleep(backoff ** attempt)
    raise RuntimeError(f"Download failed for {dest.name} after {retries} attempts: {last}")


# ------------------------------------------------------------------ extraction
def extract_csvs(zf, out_dir):
    """Recursive (zip-in-zip, folders, several CSVs per month). Returns list of CSV file names written/verified."""
    done = []
    for name in zf.namelist():
        base = name.split("/")[-1]
        if "__MACOSX" in name or base.startswith("._"):
            continue
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                done += extract_csvs(inner, out_dir)
        elif name.lower().endswith(".csv") and csv_in_range(name):
            info = zf.getinfo(name)
            out = out_dir / base
            if not (out.exists() and out.stat().st_size == info.file_size):
                tmp = out.with_name(base + ".tmp")
                tmp.write_bytes(zf.read(name))
                os.replace(tmp, out)
            done.append(base)
    return done


def month_of(csv_name):
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", csv_name)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def main():
    if config.IS_SYNTHETIC:
        raise SystemExit("download_data.py is for REAL mode only (DATA_MODE=real).")
    provenance.init_run()
    objs = {k: s for k, s in list_objects().items() if wanted(k)}
    print(f"{len(objs)} archive(s) matched for {config.START_MONTH}..{config.END_MONTH}: {sorted(objs)}")
    files, extracted = {}, []
    for key, size in sorted(objs.items()):
        dest = config.RAW / key
        if is_complete(dest, size):
            print("ok (already complete):", key)
        else:
            print(f"downloading {key} ({size / 1e6:,.0f} MB)")
            download_file(BUCKET + key, dest, size)
        files[key] = size
        if key.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                names = extract_csvs(z, config.RAW)
            print(f"  {key}: {len(names)} CSV file(s)")
            extracted += names
    csvs = sorted(p.name for p in config.RAW.glob("*.csv") if csv_in_range(p.name))
    have_months = {month_of(n) for n in csvs if month_of(n)}
    missing = [m for m in config.months_in_range() if m not in have_months]
    provenance.update("download", {"archives": files, "csv_files": csvs, "months_found": sorted(have_months),
                                   "months_missing": missing})
    if missing:
        raise SystemExit(f"No CSV found for month(s) {missing}. Check https://s3.amazonaws.com/tripdata/ or place "
                         f"validated CSVs named like YYYYMM-citibike-tripdata_1.csv in {config.RAW}")
    print("download complete ->", config.RAW)


if __name__ == "__main__":
    main()
