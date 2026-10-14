"""OPTIONAL live component - poll GBFS station_status and append snapshots to parquet.
Run:  python src/realtime.py --minutes 5 --snapshots 12     (Ctrl+C to stop)
"""
import argparse
import time
from datetime import datetime
import pandas as pd
import requests
from config import LIVE as EXTERNAL

EXTERNAL.mkdir(parents=True, exist_ok=True)

STATUS = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
INFO = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"


def snapshot():
    st = pd.DataFrame(requests.get(STATUS, timeout=30).json()["data"]["stations"])
    info = pd.DataFrame(requests.get(INFO, timeout=30).json()["data"]["stations"])
    cols = ["station_id", "name", "lat", "lon", "capacity"]
    df = st.merge(info[cols], on="station_id", how="left")
    df["snapshot_ts"] = datetime.now()
    return df[["snapshot_ts", "station_id", "name", "lat", "lon", "capacity",
               "num_bikes_available", "num_docks_available", "num_ebikes_available"]
              if "num_ebikes_available" in df else
              ["snapshot_ts", "station_id", "name", "lat", "lon", "capacity",
               "num_bikes_available", "num_docks_available"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--snapshots", type=int, default=12)
    a = ap.parse_args()
    out = EXTERNAL / "live_snapshots.parquet"
    frames = [pd.read_parquet(out)] if out.exists() else []
    for i in range(a.snapshots):
        frames.append(snapshot())
        pd.concat(frames).to_parquet(out, index=False)
        print(f"[{i + 1}/{a.snapshots}] saved {len(frames[-1])} stations")
        if i < a.snapshots - 1:
            time.sleep(a.minutes * 60)
