"""Fetch the CURRENT station capacity snapshot (GBFS). Capacity is NOT historical - disclosed in model outputs."""
from datetime import datetime, timezone

import pandas as pd
import requests

import config
import provenance

URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"

if __name__ == "__main__":
    if config.IS_SYNTHETIC:
        raise SystemExit("Sample mode uses fake capacities created by make_sample_data.py.")
    s = pd.DataFrame(requests.get(URL, timeout=30).json()["data"]["stations"])
    out = s[["short_name", "name", "lat", "lon", "capacity"]].rename(columns={"short_name": "station_id"})
    out["station_id"] = out["station_id"].astype(str)
    out.to_parquet(config.STATION_INFO, index=False)
    provenance.update("station_info", {"rows": len(out), "snapshot_utc": datetime.now(timezone.utc).isoformat(),
                                       "caveat": "current capacity snapshot, not capacity at trip time"})
    print(len(out), "stations ->", config.STATION_INFO)
