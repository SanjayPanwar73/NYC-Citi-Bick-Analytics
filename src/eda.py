"""PHASE 2 - Exploratory analysis (descriptive + association analysis; NO causal claims).

Normalisation / confounding choices:
  * hourly demand curves are per-day averages (weekday and weekend have different numbers of days)
  * station imbalance is a RATIO (net outflow / activity) among stations above median activity
  * weather vs demand: (a) unadjusted daily scatter - confounded by season/weekday, shown only as such;
    (b) hour-level demand relative to the same month x weekday x hour norm, by rain/temperature bin;
    (c) daily log-trips regression with weekend, holiday, trend and HAC (Newey-West) errors.
    Results are ASSOCIATIONS. Short ranges cannot separate temperature from trend/season.
"""
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import folium

import config
import provenance
from config import FIG, REP_DATA, sql_path
from stats_utils import ols_hac

provenance.require_stage("features")
sns.set_theme(style="whitegrid", context="notebook")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
con = duckdb.connect()
con.execute("PRAGMA temp_directory='duckdb_tmp'")
T = f"read_parquet('{sql_path(config.TRIPS_FEAT)}')"
W = f"read_parquet('{sql_path(config.WEATHER_PQ)}')"
notes = [f"**Run:** `{provenance.banner()}`"]


def q(sql):
    return con.execute(sql).df()


def save(name):
    plt.tight_layout()
    provenance.stamp_figure()
    plt.savefig(FIG / f"{name}.png", dpi=130)
    plt.close()
    print("saved", name)


# ---------------------------------------------------------------- coverage
cov = q(f"SELECT count(*) AS rows, min(started_at) AS first_trip, max(started_at) AS last_trip, "
        f"count(DISTINCT trip_date) AS n_days, count(DISTINCT start_station_id) AS n_start_stations FROM {T}").iloc[0]
notes.append(f"**Coverage:** {int(cov.rows):,} trips, {int(cov.n_days)} days ({str(cov.first_trip)[:10]} to "
             f"{str(cov.last_trip)[:10]}), {int(cov.n_start_stations)} start stations.")

# ---------------------------------------------------------------- 1. volume over time
daily = q(f"""SELECT trip_date, count(*) AS trips, avg(trip_duration_min) AS avg_duration_min,
                     avg(trip_distance_km) AS avg_distance_km, max(is_weekend::INT) AS is_weekend,
                     max(is_holiday::INT) AS is_holiday
              FROM {T} GROUP BY 1 ORDER BY 1""")
wd = q(f"""SELECT hour_ts::DATE AS trip_date, avg(temp_c) AS temp_c, sum(precip_mm) AS precip_mm,
                  sum(snow_cm) AS snow_cm, avg(wind_kmh) AS wind_kmh FROM {W} GROUP BY 1""")
daily["trip_date"] = pd.to_datetime(daily["trip_date"])
wd["trip_date"] = pd.to_datetime(wd["trip_date"])
daily = daily.merge(wd, on="trip_date", how="left")
daily.to_parquet(REP_DATA / "daily.parquet", index=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.trip_date, daily.trips, lw=1)
ax.plot(daily.trip_date, daily.trips.rolling(7, center=True).mean(), lw=2.5, label="7-day mean")
ax.set(title="Daily trips", ylabel="trips"); ax.legend()
save("01_daily_trips")
notes.append(f"- Busiest day {daily.loc[daily.trips.idxmax(), 'trip_date']:%Y-%m-%d} ({daily.trips.max():,} trips); "
             f"quietest {daily.loc[daily.trips.idxmin(), 'trip_date']:%Y-%m-%d} ({daily.trips.min():,}).")

# ---------------------------------------------------------------- 2. distributions (random sample, documented)
s = q(f"""SELECT * FROM (SELECT trip_duration_min, trip_distance_km, avg_speed_kmh, temp_c, humidity,
                 precip_mm, wind_kmh, hour FROM {T}) USING SAMPLE reservoir(1000000 ROWS)""")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(np.log10(s.trip_duration_min), bins=60, ax=ax[0]); ax[0].set(title="Trip duration (log10 min)")
sns.histplot(s.trip_distance_km.dropna().clip(upper=10), bins=60, ax=ax[1]); ax[1].set(title="Straight-line distance (km, clipped at 10)")
sns.histplot(s.avg_speed_kmh.dropna(), bins=60, ax=ax[2]); ax[2].set(title="Average speed (km/h)")
save("02_distributions")
notes.append(f"- Median trip {s.trip_duration_min.median():.1f} min, {s.trip_distance_km.median():.2f} km straight-line "
             f"({s.trip_distance_km.notna().mean() * 100:.1f}% of sampled trips have a distance).")

# ---------------------------------------------------------------- 3. hourly curve + heatmap
hourly = q(f"""WITH d AS (SELECT is_weekend, count(DISTINCT trip_date) AS n_days FROM {T} GROUP BY 1)
               SELECT t.hour, t.is_weekend, count(*) * 1.0 / any_value(d.n_days) AS avg_trips
               FROM {T} t JOIN d USING (is_weekend) GROUP BY 1, 2 ORDER BY 1""")
hourly["day_type"] = np.where(hourly.is_weekend, "Weekend", "Weekday")
hourly.to_csv(REP_DATA / "hourly.csv", index=False)
plt.figure(figsize=(10, 4)); sns.lineplot(data=hourly, x="hour", y="avg_trips", hue="day_type", marker="o")
plt.title("Average trips per hour (per-day normalised)"); plt.xticks(range(0, 24))
save("03_hourly_weekday_weekend")
pk = hourly[hourly.day_type == "Weekday"].sort_values("avg_trips").iloc[-1]
notes.append(f"- Weekday peak hour {int(pk.hour)}:00 (~{pk.avg_trips:,.0f} trips/hour on an average weekday).")
heat = q(f"SELECT day_of_week, hour, count(*) AS trips FROM {T} GROUP BY 1, 2")
heat.to_csv(REP_DATA / "heat.csv", index=False)
pv = heat.pivot(index="day_of_week", columns="hour", values="trips"); pv.index = [DOW[i - 1] for i in pv.index]
plt.figure(figsize=(12, 4)); sns.heatmap(pv, cmap="viridis"); plt.title("Trips: day of week x hour (raw counts)")
save("04_heatmap_dow_hour")

# ---------------------------------------------------------------- 4. member vs casual, bike type
user = q(f"""SELECT user_type, count(*) AS trips, median(trip_duration_min) AS med_duration_min,
                    median(trip_distance_km) AS med_distance_km FROM {T}
             WHERE user_type IS NOT NULL GROUP BY 1""")
user.to_csv(REP_DATA / "user_summary.csv", index=False)
uh = q(f"""SELECT user_type, hour, count(*) AS trips, median(trip_duration_min) AS med_duration_min
           FROM {T} WHERE user_type IS NOT NULL GROUP BY 1, 2 ORDER BY 2""")
uh.to_csv(REP_DATA / "user_hourly.csv", index=False)      # used by the dashboard
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.lineplot(data=uh, x="hour", y="trips", hue="user_type", ax=ax[0]); ax[0].set_title("Trips by hour")
sns.lineplot(data=uh, x="hour", y="med_duration_min", hue="user_type", ax=ax[1]); ax[1].set_title("Median duration by hour (min)")
save("05_member_vs_casual")
mem = user.set_index("user_type")
if {"member", "casual"} <= set(mem.index):
    notes.append(f"- Members make {100 * mem.loc['member', 'trips'] / mem.trips.sum():.0f}% of trips; median duration "
                 f"{mem.loc['member', 'med_duration_min']:.1f} min vs {mem.loc['casual', 'med_duration_min']:.1f} min (casual).")
q(f"""SELECT rideable_type, count(*) AS trips, median(trip_duration_min) AS med_duration_min,
             median(avg_speed_kmh) AS med_speed_kmh FROM {T} WHERE rideable_type IS NOT NULL GROUP BY 1"""
  ).to_csv(REP_DATA / "bike_type.csv", index=False)

# ---------------------------------------------------------------- 5. stations (normalised imbalance)
st = q(f"""WITH s AS (SELECT start_station_id AS station_id, count(*) AS starts FROM {T}
                      WHERE start_station_id IS NOT NULL GROUP BY 1),
                e AS (SELECT end_station_id AS station_id, count(*) AS ends FROM {T}
                      WHERE end_station_id IS NOT NULL GROUP BY 1)
           SELECT l.station_id, l.station_name, l.lat, l.lng, coalesce(s.starts, 0) AS starts,
                  coalesce(e.ends, 0) AS ends FROM read_parquet('{sql_path(config.STATION_LOOKUP)}') l
           LEFT JOIN s USING (station_id) LEFT JOIN e USING (station_id)""")
st["activity"] = st.starts + st.ends
st["net_outflow"] = st.starts - st.ends
st["imbalance_ratio"] = np.where(st.activity > 0, st.net_outflow / st.activity.replace(0, np.nan), np.nan)
st["eligible_for_ranking"] = st.activity >= st.activity.median()
st.to_parquet(REP_DATA / "stations.parquet", index=False)
for col, name, colr in (("starts", "06_top_start_stations", None), ("ends", "07_top_end_stations", "darkorange")):
    top = st.nlargest(20, col).iloc[::-1]
    plt.figure(figsize=(8, 7)); plt.barh(top.station_name, top[col], color=colr); plt.title(f"Top 20 stations by {col}")
    save(name)
el = st[st.eligible_for_ranking].dropna(subset=["imbalance_ratio"])
imb = pd.concat([el.nlargest(10, "imbalance_ratio"), el.nsmallest(10, "imbalance_ratio")]).iloc[::-1]
plt.figure(figsize=(8, 7))
plt.barh(imb.station_name, imb.imbalance_ratio, color=np.where(imb.imbalance_ratio > 0, "firebrick", "steelblue"))
plt.title("Station imbalance ratio (starts - ends)/(starts + ends)\nstations above median activity; red = net bike drain")
save("08_station_imbalance")
notes.append(f"- Busiest start station: {st.loc[st.starts.idxmax(), 'station_name']} ({st.starts.max():,} starts). "
             f"Largest imbalance ratio (above-median-activity stations): {el.loc[el.imbalance_ratio.idxmax(), 'station_name']} "
             f"({el.imbalance_ratio.max():+.2f}). Imbalance is trips-based, not a measurement of bike inventory.")
m = folium.Map(location=[40.73, -73.98], zoom_start=12, tiles="OpenStreetMap")
if config.IS_SYNTHETIC:
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;top:10px;left:60px;z-index:9999;background:#fff;color:red;padding:6px;"
        "font-weight:bold'>SYNTHETIC DATA</div>"))
mx = max(st.starts.max(), 1)
for r in st.nlargest(600, "starts").itertuples():
    if pd.notna(r.lat):
        folium.CircleMarker([r.lat, r.lng], radius=2 + 10 * r.starts / mx, fill=True, weight=0.5,
                            popup=f"{r.station_name}: {r.starts:,} starts").add_to(m)
m.save(str(FIG / "09_station_map.html"))

# ---------------------------------------------------------------- 6. origin-destination
q(f"""SELECT start_station_name, end_station_name, count(*) AS trips FROM {T}
      WHERE start_station_id <> end_station_id AND start_station_name IS NOT NULL AND end_station_name IS NOT NULL
      GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20""").to_csv(REP_DATA / "top_od_pairs.csv", index=False)

# ---------------------------------------------------------------- 7. weather: ASSOCIATIONS
d = daily.dropna(subset=["temp_c", "precip_mm", "wind_kmh"]).copy()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.scatterplot(data=d, x="temp_c", y="trips", hue="is_weekend", ax=ax[0], legend=False)
ax[0].set_title("UNADJUSTED: daily trips vs temp\n(confounded by season & weekday)")
d["rain_bin"] = pd.cut(d.precip_mm, [-0.01, 0.1, 2, 10, 1000], labels=["dry", "light", "moderate", "heavy"])
sns.boxplot(data=d, x="rain_bin", y="trips", ax=ax[1]); ax[1].set_title("UNADJUSTED: daily trips by rainfall")
sns.scatterplot(data=d, x="temp_c", y="avg_duration_min", ax=ax[2]); ax[2].set_title("UNADJUSTED: avg duration vs temp")
save("10_weather_unadjusted")

hr = q(f"""WITH h AS (SELECT hour_ts, count(*) AS trips FROM {T} GROUP BY 1),
                j AS (SELECT h.hour_ts, h.trips, w.temp_c, w.precip_mm, hour(h.hour_ts) AS hr,
                             isodow(h.hour_ts) AS dow, month(h.hour_ts) AS mo
                      FROM h JOIN {W} w ON w.hour_ts = h.hour_ts)
           SELECT *, trips / avg(trips) OVER (PARTITION BY mo, dow, hr) AS ratio FROM j""")
hr["rain_bin"] = pd.cut(hr.precip_mm, [-0.01, 0, 0.5, 2.5, 1000], labels=["none", "0-0.5mm", "0.5-2.5mm", ">2.5mm"])
hr["temp_bin"] = pd.cut(hr.temp_c, [-100, 0, 10, 20, 30, 100], labels=["<0C", "0-10C", "10-20C", "20-30C", ">30C"])
rr = hr.groupby("rain_bin", observed=True).ratio.agg(["median", "mean", "count"]).reset_index()
tt = hr.groupby("temp_bin", observed=True).ratio.agg(["median", "mean", "count"]).reset_index()
rr.to_csv(REP_DATA / "rain_ratio.csv", index=False); tt.to_csv(REP_DATA / "temp_ratio.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=rr, x="rain_bin", y="median", ax=ax[0]); ax[0].axhline(1, c="k", ls="--"); ax[0].set_title("Hourly demand vs same month x weekday x hour norm\nby rainfall (ratio; association)")
sns.barplot(data=tt, x="temp_bin", y="median", ax=ax[1]); ax[1].axhline(1, c="k", ls="--"); ax[1].set_title("... by temperature (association)")
save("10b_weather_relative_demand")
if len(rr) and rr["count"].min() >= 30 and "none" in set(rr.rain_bin.astype(str)) and ">2.5mm" in set(rr.rain_bin.astype(str)):
    a = rr.set_index("rain_bin")["median"]
    notes.append(f"- Weather (association, hour level): hours with >2.5 mm rain have a median demand ratio of {a['>2.5mm']:.2f} "
                 f"vs {a['none']:.2f} in dry hours, relative to the same month/weekday/hour norm "
                 f"({int(rr.set_index('rain_bin').loc['>2.5mm', 'count'])} rainy hours). Not adjusted for within-cell seasonal drift.")

adj = None
dd = d[d.trips > 0].sort_values("trip_date")
if len(dd) >= 14:
    X = pd.DataFrame({"temp_per_5C": dd.temp_c / 5, "precip_per_5mm_day": dd.precip_mm / 5, "wind_per_10kmh": dd.wind_kmh / 10,
                      "is_weekend": dd.is_weekend, "is_holiday": dd.is_holiday,
                      "trend_per_30d": (dd.trip_date - dd.trip_date.min()).dt.days / 30})
    adj = ols_hac(X.values, np.log(dd.trips.values), names=list(X.columns), lags=7)
    adj["pct_effect"] = 100 * (np.exp(adj.coef) - 1)
    adj["pct_ci95_low"] = 100 * (np.exp(adj.ci95_low) - 1)
    adj["pct_ci95_high"] = 100 * (np.exp(adj.ci95_high) - 1)
    adj["n_days"] = len(dd)
    adj["reliability"] = "exploratory only (<60 days)" if len(dd) < 60 else "associational; HAC errors"
    adj.to_csv(REP_DATA / "weather_adjusted.csv", index=False)
    te = adj.set_index("term").loc["temp_per_5C"]
    notes.append(f"- Weather (association, daily, adjusted for weekend/holiday/trend; n={len(dd)} days): +5C is associated with "
                 f"{te.pct_effect:+.1f}% daily trips (95% HAC CI {te.pct_ci95_low:+.1f}% to {te.pct_ci95_high:+.1f}%). "
                 + ("Fewer than 60 days: treat as exploratory; temperature cannot be separated from trend/season." if len(dd) < 60
                    else "Season and temperature remain entangled over a single season; this is not a causal estimate."))
else:
    notes.append("- Adjusted weather regression skipped: fewer than 14 days of data.")

# ---------------------------------------------------------------- 8. correlations (rank-based, robust to skew)
plt.figure(figsize=(8, 6))
sns.heatmap(s.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Spearman correlation (1M-row random sample)")
save("11_correlation_matrix")

(config.REPORT_DIR / "EDA_insights.md").write_text("# Auto-generated EDA notes\n\n" + "\n".join(notes) + "\n")
provenance.update("eda", {"n_days": int(cov.n_days), "rows": int(cov.rows), "adjusted_weather_done": adj is not None})
provenance.publish_to_reports()
print("\n".join(notes))
