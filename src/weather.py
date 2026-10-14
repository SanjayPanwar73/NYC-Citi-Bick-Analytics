"""PHASE 1 / step 3 - Hourly NYC weather (Open-Meteo archive, free, no key).
The cache file is keyed by the requested range and re-validated on every run; it is never trusted blindly."""
import pandas as pd
import requests

import config
import provenance
import validate

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = "temperature_2m,relative_humidity_2m,precipitation,snowfall,wind_speed_10m,weather_code"


def fetch(end_date):
    r = requests.get(URL, params={
        "latitude": config.WEATHER_LAT, "longitude": config.WEATHER_LON,
        "start_date": str(config.RANGE_START), "end_date": str(end_date),
        "hourly": VARS, "timezone": "America/New_York"}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={
        "time": "hour_ts", "temperature_2m": "temp_c", "relative_humidity_2m": "humidity",
        "precipitation": "precip_mm", "snowfall": "snow_cm", "wind_speed_10m": "wind_kmh"})
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df.drop_duplicates("hour_ts")        # DST repeats must not fan out the trip join


def main():
    provenance.init_run()
    end = validate.weather_target_end()
    if config.WEATHER_PQ.exists():
        df = pd.read_parquet(config.WEATHER_PQ)
        try:
            cov = validate.validate_weather(df, end)
            print("weather cache valid ->", config.WEATHER_PQ.name)
            provenance.update("weather", {"file": config.WEATHER_PQ.name, "source": "cache", **cov})
            return
        except SystemExit as e:
            if config.IS_SYNTHETIC:
                raise
            print("cache failed validation, refetching:", e)
    if config.IS_SYNTHETIC:
        raise SystemExit("Synthetic weather is created by make_sample_data.py - run it first.")
    print("fetching weather", config.RANGE_START, "->", end)
    df = fetch(end)
    cov = validate.validate_weather(df, end)               # validate BEFORE persisting
    tmp = config.WEATHER_PQ.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(config.WEATHER_PQ)
    provenance.update("weather", {"file": config.WEATHER_PQ.name, "source": "open-meteo archive", **cov,
                                  "lat": config.WEATHER_LAT, "lon": config.WEATHER_LON})
    print(len(df), "hourly rows ->", config.WEATHER_PQ)


if __name__ == "__main__":
    main()
