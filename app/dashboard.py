"""Citi Bike analytics dashboard.   python -m streamlit run app/dashboard.py

Reads only finished runs (reports/<mode>/<range>/data/...). Real and synthetic runs are shown
separately, synthetic ones are loudly marked. Every chart is wrapped so one failure never breaks the page.
Charts: Altair + pydeck (both ship with Streamlit - nothing extra to install).
"""
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
import streamlit as st

import altair as alt

try:
    import pydeck as pdk
except ImportError:                                  # maps degrade to a notice
    pdk = None

ROOT = Path(os.environ.get("CITIBIKE_ROOT") or Path(__file__).resolve().parents[1])
DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
DOW_ORDER = list(DOW.values())
C = dict(cyan="#22d3ee", indigo="#818cf8", pink="#f472b6", amber="#fbbf24", green="#34d399",
         red="#f87171", slate="#94a3b8", blue="#38bdf8", white="#f8fafc")

st.set_page_config(page_title="Citi Bike Analytics", page_icon="🚲", layout="wide", initial_sidebar_state="expanded")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
background:radial-gradient(1100px 600px at 8% -10%,rgba(34,211,238,.10),transparent 60%),radial-gradient(900px 500px at 100% 0%,rgba(129,140,248,.13),transparent 55%),#0b1020;}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1280px}
footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.hero{position:relative;padding:28px 32px;border-radius:22px;background:linear-gradient(135deg,rgba(34,211,238,.16),rgba(129,140,248,.18) 55%,rgba(244,114,182,.14));border:1px solid rgba(255,255,255,.10);box-shadow:0 20px 60px rgba(0,0,0,.35);margin-bottom:18px}
.hero h1{margin:8px 0 4px;font-size:2.4rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;padding:0}
.hero h1 span{background:linear-gradient(90deg,#22d3ee,#818cf8,#f472b6);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin:0;color:#b6c0dc;font-size:1rem}
.hero-top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.badge{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.08em;border:1px solid}
.badge i{width:8px;height:8px;border-radius:50%;display:inline-block}
.badge-real{color:#34d399;border-color:rgba(52,211,153,.5);background:rgba(52,211,153,.10)}
.badge-real i{background:#34d399;animation:pulse 2s infinite}
.badge-syn{color:#f87171;border-color:rgba(248,113,113,.6);background:rgba(248,113,113,.12)}
.badge-syn i{background:#f87171}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.6)}70%{box-shadow:0 0 0 10px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
.chip{padding:5px 12px;border-radius:999px;font-size:.78rem;color:#c9d1e8;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08)}
.hero-stats{display:flex;gap:30px;margin-top:18px;flex-wrap:wrap}
.hero-stats b{display:block;font-size:1.55rem;font-weight:800;color:#f8fafc}
.hero-stats small{color:#93a0c0;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
.synth{margin:0 0 16px;padding:12px 18px;border-radius:14px;background:repeating-linear-gradient(45deg,rgba(248,113,113,.16),rgba(248,113,113,.16) 12px,rgba(248,113,113,.08) 12px,rgba(248,113,113,.08) 24px);border:1px solid rgba(248,113,113,.5);color:#fecaca;font-weight:600}
.note{margin:6px 0 14px;padding:12px 16px;border-radius:12px;background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.35);color:#fde68a;font-size:.88rem;line-height:1.45}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:6px 0 18px}
.kpi{position:relative;padding:16px 18px 16px 22px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);overflow:hidden;transition:transform .15s ease,border-color .15s ease}
.kpi:hover{transform:translateY(-2px);border-color:var(--accent)}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent)}
.kpi-l{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:#93a0c0;font-weight:600}
.kpi-v{font-size:1.85rem;font-weight:800;color:#f8fafc;margin-top:4px;letter-spacing:-.02em}
.kpi-s{font-size:.78rem;color:#8b97b8;margin-top:2px}
.sec{margin:24px 0 10px}.sec h3{margin:0;font-size:1.15rem;font-weight:700;padding:0}.sec p{margin:2px 0 0;color:#93a0c0;font-size:.88rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin:8px 0 14px}
.card{padding:14px 16px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#d7def2;font-size:.9rem;line-height:1.45}
.card .ic{font-size:1.2rem;margin-right:6px}
.empty{padding:26px;border-radius:16px;border:1px dashed rgba(255,255,255,.22);text-align:center;color:#b6c0dc;margin:10px 0}
.empty code{background:rgba(255,255,255,.08);padding:3px 8px;border-radius:6px}
.predict{padding:22px 24px;border-radius:20px;background:linear-gradient(135deg,rgba(34,211,238,.14),rgba(129,140,248,.14));border:1px solid rgba(255,255,255,.12)}
.predict .big{font-size:3.3rem;font-weight:800;letter-spacing:-.03em;color:#f8fafc}
.predict .sub{color:#b6c0dc;font-size:.95rem}
.rangebar{position:relative;height:14px;border-radius:999px;background:rgba(255,255,255,.09);margin:18px 0 8px}
.rb-band{position:absolute;top:0;bottom:0;border-radius:999px;background:linear-gradient(90deg,#22d3ee,#818cf8)}
.rb-mid{position:absolute;top:-5px;width:4px;height:24px;border-radius:3px;background:#fff;box-shadow:0 0 12px rgba(255,255,255,.7)}
.rb-scale{display:flex;justify-content:space-between;color:#8b97b8;font-size:.72rem}
.stage{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:.85rem;color:#c9d1e8}
.stage.todo{color:#6b7794}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid rgba(255,255,255,.08)}
.stTabs [data-baseweb="tab"]{height:44px;padding:0 16px;border-radius:10px 10px 0 0;font-weight:600;color:#93a0c0}
.stTabs [aria-selected="true"]{color:#f8fafc;background:rgba(255,255,255,.05)}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#22d3ee,#818cf8);height:3px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1630,#0b1020);border-right:1px solid rgba(255,255,255,.06)}
.brand{font-size:1.25rem;font-weight:800;letter-spacing:-.01em;margin-bottom:2px}
.brand span{color:#22d3ee}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def esc(x):
    return html.escape(str(x))


def fmt_int(n):
    return f"{int(round(n)):,}"


def compact(n):
    n = float(n)
    return f"{n / 1e6:.2f}M" if n >= 1e6 else (f"{n / 1e3:.1f}K" if n >= 1e4 else f"{n:,.0f}")


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{esc(title)}</h3>' + (f"<p>{esc(sub)}</p>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def kpi_row(items):
    cards = "".join(
        f'<div class="kpi" style="--accent:{a}"><div class="kpi-l">{esc(l)}</div>'
        f'<div class="kpi-v">{esc(v)}</div><div class="kpi-s">{esc(s)}</div></div>' for l, v, s, a in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def empty(msg, cmd=None):
    st.markdown(f'<div class="empty">{esc(msg)}' + (f"<br><br><code>{esc(cmd)}</code>" if cmd else "") + "</div>",
                unsafe_allow_html=True)


def _first_ok(*attempts):
    """Run Streamlit-version-specific calls in order until one works (width/use_container_width API changed)."""
    last = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001
            last = e
    raise last


def show_df(df, **kw):
    _first_ok(lambda: st.dataframe(df, width="stretch", hide_index=True, **kw),
              lambda: st.dataframe(df, hide_index=True, **kw))


def show_img(path, caption=None):
    p = Path(path)
    if p.exists():
        _first_ok(lambda: st.image(str(p), caption=caption, width="stretch"),
                  lambda: st.image(str(p), caption=caption, use_container_width=True),
                  lambda: st.image(str(p), caption=caption))


def style(ch, h=300):
    """Common dark styling for Altair charts (applied once, on the top-level chart)."""
    return (ch.properties(height=h)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="rgba(148,163,184,0.12)", domainColor="rgba(148,163,184,0.25)",
                            labelColor="#9aa4bf", titleColor="#9aa4bf", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor="#c9d1e8", titleColor="#9aa4bf"))


def draw(build, fallback_png=None):
    """Render an Altair chart; on ANY failure show a small notice (and a PNG fallback) instead of breaking the page."""
    try:
        ch = build()
        _first_ok(lambda: st.altair_chart(ch, width="stretch"),
                  lambda: st.altair_chart(ch, use_container_width=True))
    except Exception as e:                          # noqa: BLE001
        st.caption(f"⚠️ This chart is unavailable ({type(e).__name__}).")
        if fallback_png:
            show_img(fallback_png)


def legend():
    return alt.Legend(title=None, orient="top", labelColor="#c9d1e8")


def hrule(y):
    return alt.Chart(pd.DataFrame({"y": [float(y)]})).mark_rule(strokeDash=[6, 4], color=C["white"], opacity=.6).encode(y="y:Q")


def show_map(deck):
    if deck is None:
        st.caption("Map unavailable (pydeck not installed).")
        return
    _first_ok(lambda: st.pydeck_chart(deck, width="stretch"),
              lambda: st.pydeck_chart(deck, use_container_width=True),
              lambda: st.pydeck_chart(deck))


def ratio_colors(r):
    """Imbalance ratio in [-1, 1] -> RGB: blue (bikes pile up) .. grey .. red (bikes drain away)."""
    r = np.clip(np.nan_to_num(np.asarray(r, float)), -1, 1)[:, None]
    blue, grey, red = np.array([56, 189, 248]), np.array([148, 163, 184]), np.array([248, 113, 113])
    return np.where(r >= 0, grey + (red - grey) * r, grey + (blue - grey) * (-r)).astype(int)


# ------------------------------------------------------------------ data loading (cache busts when the file changes)
@st.cache_data(show_spinner=False)
def _read(path, kind, mtime):
    if kind == "parquet":
        return pd.read_parquet(path)
    if kind == "csv":
        return pd.read_csv(path)
    return json.loads(Path(path).read_text())


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    kind = {".parquet": "parquet", ".csv": "csv", ".json": "json"}[p.suffix]
    return _read(str(p), kind, p.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _bundle(path, mtime):
    import joblib
    return joblib.load(path)


def parse_notes(path):
    if not Path(path).exists():
        return []
    return [ln[2:].replace("**", "").replace("`", "") for ln in Path(path).read_text().splitlines() if ln.startswith("- ")]


ICONS = [("busiest", "📈"), ("median", "⏱️"), ("peak", "🕔"), ("member", "👥"), ("station", "📍"), ("weather", "🌦️")]


def icon_for(t):
    tl = t.lower()
    return next((i for k, i in ICONS if k in tl), "💡")


# ------------------------------------------------------------------ run selection
manifests = sorted(ROOT.glob("reports/*/*/data/run_manifest.json"), key=lambda p: (p.parts[-4] != "real", p.parts[-3]))
if not manifests:
    st.markdown('<div class="hero"><h1>NYC Citi Bike <span>Analytics</span></h1><p>No finished run found yet.</p></div>',
                unsafe_allow_html=True)
    empty("The dashboard shows finished runs only. Check progress, then run the next stage.",
          "python run_all.py status")
    st.stop()

labels = [f"{p.parts[-4]}/{p.parts[-3]}" for p in manifests]
st.sidebar.markdown('<div class="brand">🚲 Citi Bike <span>Analytics</span></div>', unsafe_allow_html=True)
choice = st.sidebar.selectbox("Run", labels, help="Real and synthetic runs live in separate folders and are never mixed.")
D = manifests[labels.index(choice)].parent
F = D.parent / "figures"
man = json.loads((D / "run_manifest.json").read_text())
SYN = bool(man.get("is_synthetic"))

STAGES = [("download", "Download"), ("source", "Synthetic data"), ("cleaning", "Cleaning"), ("weather", "Weather"),
          ("features", "Features"), ("eda", "EDA"), ("modeling", "Modeling"), ("interpretation", "Interpretation")]
st.sidebar.markdown("**Pipeline**")
done = man.get("stages", {})
shown = [(k, n) for k, n in STAGES if (k != "source" or SYN) and (k != "download" or not SYN)]
st.sidebar.markdown("".join(
    f'<div class="stage{"" if k in done else " todo"}">{"✅" if k in done else "⬜"} {esc(n)}</div>' for k, n in shown),
    unsafe_allow_html=True)
st.sidebar.caption("Next step: `python run_all.py status`")

ctx = SimpleNamespace(
    D=D, F=F, man=man, syn=SYN,
    daily=load(D / "daily.parquet"), hourly=load(D / "hourly.csv"), heat=load(D / "heat.csv"),
    user=load(D / "user_summary.csv"), user_hourly=load(D / "user_hourly.csv"), bike=load(D / "bike_type.csv"),
    stations=load(D / "stations.parquet"), od=load(D / "top_od_pairs.csv"), rain=load(D / "rain_ratio.csv"),
    temp=load(D / "temp_ratio.csv"), adj=load(D / "weather_adjusted.csv"), res=load(D / "model_results.json"),
    excl=load(D / "exclusion_report.csv"), notes=parse_notes(D.parent / "EDA_insights.md"))

# ------------------------------------------------------------------ hero
cl = man["stages"].get("cleaning", {})
cov = cl.get("coverage", {})
n_trips = (ctx.daily.trips.sum() if ctx.daily is not None else cl.get("rows_final", 0)) or 0
n_days = len(ctx.daily) if ctx.daily is not None else 0
n_stn = int((ctx.stations.starts > 0).sum()) if ctx.stations is not None else 0
span = f"{cov['first_trip'][:10]} → {cov['last_trip'][:10]}" if cov else f"{man['start_month']} → {man['end_month']}"
badge = ('<span class="badge badge-syn"><i></i>SYNTHETIC DATA</span>' if SYN
         else '<span class="badge badge-real"><i></i>REAL DATA</span>')
st.markdown(
    f'<div class="hero"><div class="hero-top">{badge}<span class="chip">{esc(span)}</span>'
    f'<span class="chip">run {esc(man["run_tag"])}</span></div>'
    f'<h1>NYC Citi Bike <span>Analytics</span></h1>'
    f'<p>Demand patterns, weather associations and linear-regression forecasts - with full data provenance.</p>'
    f'<div class="hero-stats"><div><b>{compact(n_trips)}</b><small>trips</small></div>'
    f'<div><b>{n_days}</b><small>days</small></div><div><b>{fmt_int(n_stn)}</b><small>active stations</small></div></div></div>',
    unsafe_allow_html=True)
if SYN:
    st.markdown('<div class="synth">⚠️ SYNTHETIC DATA - generated by src/make_sample_data.py for pipeline testing. '
                'Nothing here is a finding about Citi Bike or New York.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Rhythm", "Stations", "Weather", "Models", "Predict", "Data quality", "Live", "About"])


# ================================================================== OVERVIEW
def tab_overview(c):
    if c.daily is None:
        return empty("No EDA output for this run yet.", "python src/eda.py")
    d = c.daily.copy()
    d["trip_date"] = pd.to_datetime(d.trip_date)
    d = d.sort_values("trip_date")
    d["day_type"] = np.where(d.is_weekend.astype(int) == 1, "Weekend", "Weekday")
    d["avg7"] = d.trips.rolling(7, center=True, min_periods=1).mean()
    peak = None
    if c.hourly is not None and (c.hourly.day_type == "Weekday").any():
        wk = c.hourly[c.hourly.day_type == "Weekday"].sort_values("avg_trips").iloc[-1]
        peak = (f"{int(wk.hour):02d}:00", f"~{fmt_int(wk.avg_trips)} trips / hour")
    share = None
    if c.user is not None and {"member", "casual"} <= set(c.user.user_type):
        u = c.user.set_index("user_type")
        share = (f"{100 * u.loc['member', 'trips'] / u.trips.sum():.0f}%",
                 f"median {u.loc['member', 'med_duration_min']:.1f} vs {u.loc['casual', 'med_duration_min']:.1f} min casual")
    best = d.loc[d.trips.idxmax()]
    kpi_row([
        ("Total trips", compact(d.trips.sum()), f"{len(d)} days", C["cyan"]),
        ("Average per day", fmt_int(d.trips.mean()), f"busiest {best.trip_date:%b %d}: {fmt_int(best.trips)}", C["indigo"]),
        ("Weekday peak", peak[0] if peak else "n/a", peak[1] if peak else "", C["amber"]),
        ("Member share", share[0] if share else "n/a", share[1] if share else "", C["green"]),
        ("Avg duration", f"{d.avg_duration_min.mean():.1f} min", f"avg distance {d.avg_distance_km.mean():.2f} km", C["pink"])])

    section("Daily trips", "Bars: trips per day (weekend highlighted). Line: 7-day moving average.")

    def build():
        base = alt.Chart(d).encode(x=alt.X("trip_date:T", title=None))
        bars = base.mark_bar(opacity=.88, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("trips:Q", title="Trips per day"),
            color=alt.Color("day_type:N", scale=alt.Scale(domain=["Weekday", "Weekend"], range=[C["cyan"], C["indigo"]]),
                            legend=legend()),
            tooltip=[alt.Tooltip("trip_date:T", title="Date"), alt.Tooltip("trips:Q", title="Trips", format=","), "day_type:N"])
        line = base.mark_line(color=C["amber"], strokeWidth=3, interpolate="monotone").encode(y="avg7:Q")
        return style(bars + line, 320)
    draw(build, c.F / "01_daily_trips.png")

    left, right = st.columns([3, 2])
    with left:
        section("Average by weekday")
        dd = d.assign(dow=d.trip_date.dt.dayofweek.map(lambda i: DOW_ORDER[i])).groupby("dow", as_index=False).trips.mean()

        def build2():
            ch = alt.Chart(dd).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=C["cyan"]).encode(
                x=alt.X("dow:N", sort=DOW_ORDER, title=None), y=alt.Y("trips:Q", title="Avg trips"),
                tooltip=[alt.Tooltip("dow:N", title="Day"), alt.Tooltip("trips:Q", title="Avg trips", format=",.0f")])
            return style(ch, 260)
        draw(build2)
    with right:
        section("Rider mix")
        if c.user is not None:
            def build3():
                ch = alt.Chart(c.user).mark_arc(innerRadius=62, stroke="#0b1020", strokeWidth=3).encode(
                    theta=alt.Theta("trips:Q"),
                    color=alt.Color("user_type:N", scale=alt.Scale(domain=["member", "casual"], range=[C["cyan"], C["pink"]]),
                                    legend=legend()),
                    tooltip=[alt.Tooltip("user_type:N", title="Rider"), alt.Tooltip("trips:Q", title="Trips", format=",")])
                return style(ch, 260)
            draw(build3)
    if c.notes:
        section("Key facts", "Auto-generated from this run's EDA (descriptive, not causal).")
        st.markdown('<div class="cards">' + "".join(
            f'<div class="card"><span class="ic">{icon_for(t)}</span>{esc(t)}</div>' for t in c.notes) + "</div>",
            unsafe_allow_html=True)


# ================================================================== RHYTHM
def tab_rhythm(c):
    if c.hourly is None:
        return empty("No EDA output for this run yet.", "python src/eda.py")
    section("A day in the city", "Average trips per hour. Weekday and weekend are per-day averages, so they are comparable.")

    def build():
        base = alt.Chart(c.hourly).encode(
            x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("avg_trips:Q", title="Avg trips"),
            color=alt.Color("day_type:N", scale=alt.Scale(domain=["Weekday", "Weekend"], range=[C["cyan"], C["amber"]]),
                            legend=legend()))
        area = base.mark_area(opacity=.12, interpolate="monotone")
        line = base.mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=55), interpolate="monotone").encode(
            tooltip=[alt.Tooltip("day_type:N", title="Day type"), alt.Tooltip("hour:Q", title="Hour"),
                     alt.Tooltip("avg_trips:Q", title="Avg trips", format=",.0f")])
        return style(area + line, 320)
    draw(build, c.F / "03_hourly_weekday_weekend.png")

    if c.heat is not None:
        section("Weekly heatmap", "When does the system breathe?")
        mode = st.radio("Show", ["Share of each day's trips", "Raw counts"], horizontal=True)
        h = c.heat.copy()
        h["dow"] = h.day_of_week.map(DOW)
        h["value"] = h.trips / h.groupby("dow").trips.transform("sum") * 100 if mode.startswith("Share") else h.trips
        title = "% of day" if mode.startswith("Share") else "trips"

        def build2():
            ch = alt.Chart(h).mark_rect(cornerRadius=3).encode(
                x=alt.X("hour:O", title="Hour"), y=alt.Y("dow:N", sort=DOW_ORDER, title=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), legend=alt.Legend(title=title)),
                tooltip=[alt.Tooltip("dow:N", title="Day"), alt.Tooltip("hour:O", title="Hour"),
                         alt.Tooltip("trips:Q", title="Trips", format=",")])
            return style(ch, 280)
        draw(build2, c.F / "04_heatmap_dow_hour.png")

    section("Members vs casual riders")
    if c.user_hourly is not None:
        uh = c.user_hourly
        a, b = st.columns(2)
        pal = alt.Scale(domain=["member", "casual"], range=[C["cyan"], C["pink"]])
        with a:
            draw(lambda: style(alt.Chart(uh).mark_line(strokeWidth=3, interpolate="monotone").encode(
                x=alt.X("hour:Q", title="Hour"), y=alt.Y("trips:Q", title="Trips"),
                color=alt.Color("user_type:N", scale=pal, legend=legend()),
                tooltip=["user_type:N", "hour:Q", alt.Tooltip("trips:Q", format=",")]), 260))
        with b:
            draw(lambda: style(alt.Chart(uh).mark_line(strokeWidth=3, interpolate="monotone").encode(
                x=alt.X("hour:Q", title="Hour"), y=alt.Y("med_duration_min:Q", title="Median duration (min)"),
                color=alt.Color("user_type:N", scale=pal, legend=legend()),
                tooltip=["user_type:N", "hour:Q", alt.Tooltip("med_duration_min:Q", format=".1f")]), 260))
    else:
        show_img(c.F / "05_member_vs_casual.png")
        st.caption("Re-run `python run_all.py phase2 --force` to get interactive charts here.")
    if c.bike is not None and len(c.bike):
        section("Bike types")
        bk = c.bike.copy()
        bk["share_%"] = bk.trips / bk.trips.sum() * 100
        show_df(bk.rename(columns={"rideable_type": "bike type", "med_duration_min": "median min",
                                   "med_speed_kmh": "median km/h"}).round(2))


# ================================================================== STATIONS
def tab_stations(c):
    if c.stations is None:
        return empty("No station output for this run yet.", "python src/eda.py")
    stn = c.stations.dropna(subset=["lat", "lng"]).copy()
    stn = stn[stn.starts > 0]
    if stn.empty:
        return empty("No stations with trips.")
    section("Station map", "Dot size = starts. Colour = imbalance: red drains bikes away, blue collects them.")
    cc = st.columns([2, 2, 3])
    view = cc[0].radio("View", ["Dots", "3D density"], horizontal=True)
    n_max = int(min(2000, len(stn)))
    n = cc[1].slider("Top N stations", 50, n_max, min(500, n_max)) if n_max > 50 else n_max
    top = stn.nlargest(n, "starts").copy()
    top["radius"] = 40 + 260 * np.sqrt(top.starts / top.starts.max())
    rgb = ratio_colors(top.imbalance_ratio.fillna(0).to_numpy())
    top["cr"], top["cg"], top["cb"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    top = top.rename(columns={"lng": "lon"})
    top["station_name"] = top.station_name.fillna("").astype(str)
    # only clean, NaN-free columns go to the browser (NaN would make the deck.gl JSON invalid -> blank map)
    top = top[["station_name", "lat", "lon", "starts", "ends", "radius", "cr", "cg", "cb"]].fillna(0)
    if pdk is not None:
        if view == "Dots":
            layer = pdk.Layer("ScatterplotLayer", data=top, get_position="[lon, lat]", get_radius="radius",
                              get_fill_color="[cr, cg, cb, 190]", pickable=True, opacity=0.9)
            deck = pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=40.735, longitude=-73.98, zoom=11, pitch=0),
                            tooltip={"html": "<b>{station_name}</b><br/>{starts} starts · {ends} ends", "style": {"color": "white"}},
                            map_style="dark")
        else:
            layer = pdk.Layer("HexagonLayer", data=top, get_position="[lon, lat]", radius=160, elevation_scale=6,
                              elevation_range=[0, 300], extruded=True, coverage=0.92, pickable=True,
                              get_elevation_weight="starts", elevation_aggregation="SUM",
                              get_color_weight="starts", color_aggregation="SUM")
            deck = pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=40.735, longitude=-73.98, zoom=11, pitch=50, bearing=-15),
                            map_style="dark")
        show_map(deck)
    else:
        show_map(None)
    st.caption("Imbalance reflects trip counts, not bike inventory (rebalancing trucks are invisible).")

    section("Rankings")
    metric = st.radio("Rank by", ["Starts", "Ends", "Imbalance ratio"], horizontal=True)
    if metric == "Imbalance ratio":
        pool = stn[stn.eligible_for_ranking.astype(bool)].dropna(subset=["imbalance_ratio"])
        rk = pd.concat([pool.nlargest(8, "imbalance_ratio"), pool.nsmallest(8, "imbalance_ratio")]).drop_duplicates("station_id")
        rk = rk.assign(value=rk.imbalance_ratio)
        sub = "Among stations above median activity: positive = more starts than ends."
    else:
        col = metric.lower()
        rk = stn.nlargest(15, col).assign(value=lambda x: x[col])
        sub = f"Top 15 stations by {col}."
    st.caption(sub)
    rk = rk.rename(columns={"station_name": "station"})

    def build():
        enc_color = (alt.Color("value:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[C["blue"], C["slate"], C["red"]]), legend=None)
                     if metric == "Imbalance ratio" else alt.value(C["cyan"]))
        ch = alt.Chart(rk).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
            y=alt.Y("station:N", sort=alt.SortField("value", order="descending"), title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("value:Q", title=metric), color=enc_color,
            tooltip=[alt.Tooltip("station:N"), alt.Tooltip("value:Q", title=metric, format=",.2f")])
        return style(ch, max(260, 26 * len(rk)))
    draw(build, c.F / "06_top_start_stations.png")

    section("Find a station")
    q = st.text_input("Search by name", placeholder="e.g. Chambers")
    view_df = stn
    if q:
        view_df = stn[stn.station_name.str.contains(q, case=False, na=False, regex=False)]
    cols = ["station_name", "starts", "ends", "activity", "imbalance_ratio"]
    show_df(view_df.sort_values("starts", ascending=False)[cols].head(60).rename(columns={"station_name": "station"}).round(3))
    if c.od is not None:
        section("Most common routes")
        show_df(c.od.rename(columns={"start_station_name": "from", "end_station_name": "to"}))


# ================================================================== WEATHER
RAIN_ORDER = ["none", "0-0.5mm", "0.5-2.5mm", ">2.5mm"]
TEMP_ORDER = ["<0C", "0-10C", "10-20C", "20-30C", ">30C"]
NAMES = {"temp_per_5C": "+5 °C", "precip_per_5mm_day": "+5 mm rain / day", "wind_per_10kmh": "+10 km/h wind",
         "is_weekend": "Weekend", "is_holiday": "Holiday", "trend_per_30d": "+30 days (trend)"}


def tab_weather(c):
    note("Associations, not causes. Temperature moves together with season, ridership growth and the calendar, "
         "and a short date range cannot separate them.")
    if c.rain is None and c.adj is None and c.daily is None:
        return empty("No weather analysis for this run yet.", "python src/eda.py")
    section("Demand relative to a normal hour", "Ratio to the same month × weekday × hour norm (1 = typical). Based on hours, so counts per bin matter.")
    a, b = st.columns(2)
    for col, df, order, field, ttl in ((a, c.rain, RAIN_ORDER, "rain_bin", "Rainfall in the hour"),
                                       (b, c.temp, TEMP_ORDER, "temp_bin", "Temperature")):
        with col:
            if df is None:
                continue
            d = df.rename(columns={field: "bin"})

            def build(d=d, order=order, ttl=ttl):
                bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=C["cyan"]).encode(
                    x=alt.X("bin:N", sort=order, title=ttl), y=alt.Y("median:Q", title="Median ratio"),
                    tooltip=[alt.Tooltip("bin:N", title=ttl), alt.Tooltip("median:Q", title="Median ratio", format=".2f"),
                             alt.Tooltip("count:Q", title="Hours", format=",")])
                return style(bars + hrule(1.0), 280)
            draw(build)
    if c.adj is not None:
        w = c.adj[c.adj.term != "const"].copy()
        w["label"] = w.term.map(NAMES).fillna(w.term)
        w = w.rename(columns={"pct_effect": "pct", "pct_ci95_low": "lo", "pct_ci95_high": "hi"})
        rel = str(c.adj.reliability.iloc[0])
        section("Adjusted daily association", f"% change in daily trips (95% HAC interval) · n = {int(c.adj.n_days.iloc[0])} days · {rel}")

        def build2():
            base = alt.Chart(w).encode(y=alt.Y("label:N", sort=None, title=None))
            rule = base.mark_rule(strokeWidth=3, color=C["indigo"]).encode(x=alt.X("lo:Q", title="% change in daily trips"), x2="hi:Q")
            pts = base.mark_point(filled=True, size=130, color=C["cyan"]).encode(
                x="pct:Q", tooltip=["label:N", alt.Tooltip("pct:Q", format="+.1f"), alt.Tooltip("lo:Q", format="+.1f"), alt.Tooltip("hi:Q", format="+.1f")])
            zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[6, 4], color=C["white"], opacity=.6).encode(x="x:Q")
            return style(rule + pts + zero, 260)
        draw(build2)
        if "exploratory" in rel.lower():
            note("Exploratory only: fewer than 60 days. Intervals crossing 0 mean no detectable association in this window.")
    if c.daily is not None and {"temp_c", "trips"} <= set(c.daily.columns):
        with st.expander("Unadjusted view (confounded by season and weekday)"):
            d = c.daily.dropna(subset=["temp_c"]).assign(day_type=lambda x: np.where(x.is_weekend.astype(int) == 1, "Weekend", "Weekday"))

            def build3():
                pts = alt.Chart(d).mark_point(filled=True, size=70, opacity=.8).encode(
                    x=alt.X("temp_c:Q", title="Mean temperature (°C)", scale=alt.Scale(zero=False)), y=alt.Y("trips:Q", title="Trips"),
                    color=alt.Color("day_type:N", scale=alt.Scale(domain=["Weekday", "Weekend"], range=[C["cyan"], C["indigo"]]), legend=legend()))
                return style(pts, 300)
            draw(build3)


# ================================================================== MODELS
def tab_models(c):
    res = c.res
    if res is None:
        return empty("Modeling has not run for this run yet.", "python src/modeling.py")
    st.caption(f"Validation: {res['meta']['validation']} · {res['meta']['n_folds']} folds · values are mean ± sd across folds.")
    opts = {"Trip duration": (res["duration_model"], "backtest_duration.csv", "minutes", True)}
    for k, b in res["demand_model"].items():
        opts[f"Station demand · {b['horizon_hours']} h ahead"] = (b, f"backtest_demand_h{b['horizon_hours']}.csv",
                                                                   "trips / station-hour", False)
    pick = st.radio("Model", list(opts), horizontal=True)
    m, bt_file, unit, is_dur = opts[pick]
    s = m["summary"]
    lin = s["Linear (OLS)"]
    bb = m["best_baseline"]
    sk = m["skill_mae_vs_best_baseline"]
    third = (("R²", f"{lin['R2']['mean']:.3f}", "variance explained", C["amber"]) if is_dur else
             ("WAPE", f"{lin['WAPE_%']['mean']:.1f}%", "weighted abs. % error", C["amber"]))
    kpi_row([("Linear MAE", f"{lin['MAE']['mean']:.3f}", unit, C["cyan"]),
             ("Best baseline MAE", f"{s[bb]['MAE']['mean']:.3f}", bb.replace("Baseline: ", ""), C["slate"]),
             ("Skill vs baseline", f"{sk:+.1%}", "positive = linear model is better", C["green"] if sk > 0 else C["red"]),
             third])
    if sk <= 0.02:
        note("The linear model does not clearly beat the best baseline here - judge it by that, not by raw error.")
    rows = []
    for name, v in s.items():
        sd = v["MAE"]["std"] or 0.0
        rows.append({"model": name.replace("Baseline: ", "baseline · "), "MAE": v["MAE"]["mean"],
                     "lo": max(v["MAE"]["mean"] - sd, 0.0), "hi": v["MAE"]["mean"] + sd,
                     "kind": "Baseline" if name.startswith("Baseline") else "Linear model"})
    mdf = pd.DataFrame(rows)
    section("Error by model (lower is better)", "Bars: mean MAE across folds. Whisker: ± 1 sd.")

    def build():
        base = alt.Chart(mdf).encode(y=alt.Y("model:N", sort="x", title=None, axis=alt.Axis(labelLimit=260)))
        bars = base.mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
            x=alt.X("MAE:Q", title=f"MAE ({unit})"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Baseline", "Linear model"], range=[C["slate"], C["cyan"]]), legend=legend()),
            tooltip=["model:N", alt.Tooltip("MAE:Q", format=".3f")])
        err = base.mark_rule(color=C["white"], strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
        return style(bars + err, max(260, 34 * len(mdf)))
    draw(build)

    bt = load(c.D / bt_file)
    if bt is not None:
        section("Stability across folds", "Each backtest block is a later slice of time. Flat lines = stable; crossing lines = unreliable ranking.")
        bt = bt.assign(model=bt.model.str.replace("Baseline: ", "baseline · ", regex=False))

        def build2():
            ch = alt.Chart(bt).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("fold:O", title="Test block (time →)"), y=alt.Y("MAE:Q", title="MAE"),
                color=alt.Color("model:N", legend=alt.Legend(title=None, orient="right", labelLimit=240)),
                tooltip=["model:N", "fold:O", alt.Tooltip("MAE:Q", format=".3f")])
            return style(ch, 300)
        draw(build2)
    section("All metrics")
    cols = [k for k in lin if k != "zero_share"]
    table = pd.DataFrame({n: {k: (f"{v[k]['mean']:.3f}" + (f" ± {v[k]['std']:.3f}" if v[k]["std"] is not None else "")) for k in cols}
                          for n, v in s.items()}).T.reset_index().rename(columns={"index": "model"})
    show_df(table)
    if is_dur:
        iv = m["interval_80"]
        st.markdown(f"**80% prediction interval** covered **{iv['empirical_coverage_last_block']:.0%}** of trips in the last test block "
                    f"({'well calibrated' if abs(iv['empirical_coverage_last_block'] - .8) <= .05 else 'MISCALIBRATED - do not rely on it'}).")
        for n in ("12_duration_residuals.png", "13_duration_error_analysis.png", "14_duration_coefficients.png"):
            show_img(c.F / n)
    else:
        show_img(c.F / f"15_demand_eval_h{m['horizon_hours']}.png")
        st.caption(m.get("capacity_note", ""))
    with st.expander("Assumptions and caveats"):
        st.markdown("\n".join(f"- {x}" for x in m["assumptions"]))


# ================================================================== PREDICT
def tab_predict(c):
    p = c.D / "duration_model.joblib"
    if not p.exists():
        return empty("The duration model has not been trained for this run yet.", "python src/modeling.py")
    bundle = _bundle(str(p), p.stat().st_mtime)
    if bundle.get("is_synthetic"):
        st.markdown('<div class="synth">⚠️ This model was trained on SYNTHETIC data - predictions are illustrative only.</div>',
                    unsafe_allow_html=True)
    st.caption(f"Trained on run {bundle['run']} through {bundle['trained_through'][:10]}. {bundle['fixed_training_conditions']}.")
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Trip**")
            a, b = st.columns(2)
            dist = a.slider("Straight-line distance (km)", 0.2, 15.0, 2.0, 0.1)
            hour = b.slider("Start hour", 0, 23, 8)
            a, b = st.columns(2)
            dow = a.selectbox("Day of week", DOW_ORDER, 0)
            holiday = b.checkbox("US federal holiday", False)
            a, b = st.columns(2)
            user = a.radio("Rider", ["member", "casual"], horizontal=True)
            bike = b.radio("Bike", ["classic_bike", "electric_bike"], horizontal=True)
        with st.container(border=True):
            st.markdown("**Weather** (all model inputs are shown - none are hidden)")
            a, b = st.columns(2)
            temp = a.slider("Temperature (°C)", -10, 38, 18)
            hum = b.slider("Humidity (%)", 10, 100, 60)
            a, b = st.columns(2)
            rain = a.slider("Rain (mm/hour)", 0.0, 15.0, 0.0, 0.5)
            wind = b.slider("Wind (km/h)", 0, 50, 10)
    dnum = DOW_ORDER.index(dow) + 1

    def make_x(dist_v, hour_v):
        dist_v, hour_v = np.atleast_1d(np.asarray(dist_v, float)), np.atleast_1d(np.asarray(hour_v, float))
        n = max(len(dist_v), len(hour_v))
        dv, hv = np.resize(dist_v, n), np.resize(hour_v, n)
        x = pd.DataFrame({"log_dist": np.log1p(dv), "hour_sin": np.sin(2 * np.pi * hv / 24),
                          "hour_cos": np.cos(2 * np.pi * hv / 24), "is_weekend": int(dnum >= 6),
                          "is_holiday": int(holiday), "temp_c": temp, "humidity": hum, "precip_mm": rain, "wind_kmh": wind,
                          "user_type": user, "rideable_type": bike, "dow": str(dnum)})
        return x[bundle["features"]]

    def predict(dist_v, hour_v):
        z = bundle["pipeline"].predict(make_x(dist_v, hour_v))
        return (np.expm1(z + bundle["resid_q10_log"]), np.expm1(z), np.expm1(z + bundle["resid_q90_log"]))

    lo, mid, hi = (float(v[0]) for v in predict(dist, hour))
    top_scale = max(hi * 1.15, 10.0)
    with right:
        st.markdown(
            f'<div class="predict"><div class="sub">Predicted typical duration</div><div class="big">{mid:.1f} min</div>'
            f'<div class="sub">80% range: <b>{lo:.1f} – {hi:.1f} min</b> · tested coverage {bundle["test_coverage_80"]:.0%}</div>'
            f'<div class="rangebar"><div class="rb-band" style="left:{100 * lo / top_scale:.1f}%;width:{100 * (hi - lo) / top_scale:.1f}%"></div>'
            f'<div class="rb-mid" style="left:{100 * mid / top_scale:.1f}%"></div></div>'
            f'<div class="rb-scale"><span>0</span><span>{top_scale:.0f} min</span></div></div>', unsafe_allow_html=True)
        st.caption("Individual trips vary a lot (commuting vs sightseeing); a low R² is expected.")
    section("What-if curves", "Same inputs as above; only the x-axis variable changes.")
    d1 = np.linspace(0.2, 12, 45)
    l1, m1, u1 = predict(d1, hour)
    hrs = np.arange(24)
    l2, m2, u2 = predict(dist, hrs)
    c1, c2 = st.columns(2)

    def band(df, x, xt, cur):
        base = alt.Chart(df).encode(x=alt.X(f"{x}:Q", title=xt))
        area = base.mark_area(opacity=.22, color=C["indigo"]).encode(y=alt.Y("lo:Q", title="Duration (min)"), y2="hi:Q")
        line = base.mark_line(color=C["cyan"], strokeWidth=3).encode(y="mid:Q")
        dot = alt.Chart(pd.DataFrame({x: [float(cur)], "mid": [float(np.interp(cur, df[x], df["mid"]))]})).mark_point(
            filled=True, size=160, color=C["amber"]).encode(x=f"{x}:Q", y="mid:Q")
        return style(area + line + dot, 280)
    with c1:
        draw(lambda: band(pd.DataFrame({"dist": d1, "lo": l1, "mid": m1, "hi": u1}), "dist", "Distance (km)", dist))
    with c2:
        draw(lambda: band(pd.DataFrame({"hour": hrs, "lo": l2, "mid": m2, "hi": u2}), "hour", "Start hour", hour))


# ================================================================== DATA QUALITY
def tab_quality(c):
    cl = c.man["stages"].get("cleaning", {})
    wx = c.man["stages"].get("weather", {})
    ft = c.man["stages"].get("features", {})
    if cl:
        parsed, kept = cl.get("rows_parsed", 0), cl.get("rows_final", 0)
        kpi_row([("Rows parsed", compact(parsed), "from raw CSV files", C["cyan"]),
                 ("Rows kept", compact(kept), f"{100 * kept / max(parsed, 1):.1f}% of parsed", C["green"]),
                 ("Duplicates removed", fmt_int(cl.get("duplicates_removed", 0)), "same ride_id", C["amber"]),
                 ("Weather coverage", f"{100 * (1 - wx['missing_share']):.2f}%" if wx else "n/a",
                  f"{fmt_int(wx['hours_present'])} hours" if wx else "", C["indigo"]),
                 ("Trips w/o weather", f"{100 * ft['no_weather_share']:.3f}%" if ft else "n/a", "after join", C["pink"])])
    section("Provenance")
    st.markdown(f"- **Source:** {c.man['data_source']}\n- **Requested range:** {c.man['start_month']} → {c.man['end_month']} · created {c.man['created_at']}\n"
                f"- **Files:** {', '.join(cl.get('csv_files', [])) or 'n/a'}")
    if c.excl is not None:
        ex = c.excl[c.excl.step.str.startswith("excluded:")].copy()
        ex["reason"] = ex.step.str.replace("excluded: ", "", regex=False)
        ex = ex[ex.rows > 0].sort_values("rows", ascending=False)
        section("Why rows were excluded", "Every raw row has exactly one outcome (first failing rule wins).")
        if len(ex):
            def build():
                ch = alt.Chart(ex).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color=C["amber"]).encode(
                    y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                    x=alt.X("rows:Q", title="Rows"),
                    tooltip=["reason:N", alt.Tooltip("rows:Q", format=","), alt.Tooltip("pct_of_parsed:Q", format=".3f", title="% of parsed")])
                return style(ch, max(220, 34 * len(ex)))
            draw(build)
        else:
            st.caption("No exclusions.")
        with st.expander("Full exclusion report"):
            show_df(c.excl)
        st.caption("Missing END coordinates are kept and flagged: they are real trips, but have no distance and are excluded from the duration model only.")
    if cl.get("coverage", {}).get("rows_by_month"):
        section("Coverage by month")
        rbm = cl["coverage"]["rows_by_month"]
        mm = pd.DataFrame({"month": list(rbm), "trips": list(rbm.values())})
        draw(lambda: style(alt.Chart(mm).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=C["cyan"]).encode(
            x=alt.X("month:N", title=None), y=alt.Y("trips:Q", title="Trips"), tooltip=["month:N", alt.Tooltip("trips:Q", format=",")]), 240))
    section("Pipeline log")
    st.markdown('<div class="cards">' + "".join(
        f'<div class="card"><b>✅ {esc(k)}</b><br><span style="color:#8b97b8">{esc(str(v.get("completed_at", ""))[:19].replace("T", " "))} UTC</span></div>'
        for k, v in c.man["stages"].items()) + "</div>", unsafe_allow_html=True)
    with st.expander("Environment (exact package versions)"):
        st.json(c.man.get("environment", {}))


# ================================================================== LIVE
@st.cache_data(ttl=60, show_spinner=False)
def live_feed():
    base = "https://gbfs.citibikenyc.com/gbfs/en/"
    s = pd.DataFrame(requests.get(base + "station_status.json", timeout=20).json()["data"]["stations"])
    i = pd.DataFrame(requests.get(base + "station_information.json", timeout=20).json()["data"]["stations"])
    return s.merge(i[["station_id", "name", "lat", "lon", "capacity"]], on="station_id")


def tab_live(c):
    st.caption("Current conditions from the official GBFS feed - independent of the historical run above.")
    if st.button("Refresh now"):
        st.cache_data.clear()
    try:
        lv = live_feed()
    except Exception as e:                          # noqa: BLE001
        return empty(f"Could not reach the live feed ({type(e).__name__}). Check your connection and refresh.")
    kpi_row([("Bikes available", fmt_int(lv.num_bikes_available.sum()), "right now", C["cyan"]),
             ("Empty docks", fmt_int(lv.num_docks_available.sum()), "right now", C["indigo"]),
             ("Stations with 0 bikes", fmt_int((lv.num_bikes_available == 0).sum()), f"of {fmt_int(len(lv))}", C["red"])])
    if pdk is not None:
        lv = lv.assign(frac=(lv.num_bikes_available / lv.capacity.replace(0, np.nan)).fillna(0).clip(0, 1))
        lv["cr"], lv["cg"], lv["cb"] = (248 - 192 * lv.frac).astype(int), (113 + 76 * lv.frac).astype(int), (113 + 135 * lv.frac).astype(int)
        lv["radius"] = 45 + 4 * lv.num_bikes_available.clip(upper=40)
        lv["name"] = lv.name.fillna("").astype(str)
        lv = lv[["name", "lat", "lon", "num_bikes_available", "num_docks_available", "radius", "cr", "cg", "cb"]].fillna(0)
        layer = pdk.Layer("ScatterplotLayer", data=lv, get_position="[lon, lat]", get_radius="radius",
                          get_fill_color="[cr, cg, cb, 190]", pickable=True)
        show_map(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=40.735, longitude=-73.98, zoom=11),
                          tooltip={"html": "<b>{name}</b><br/>{num_bikes_available} bikes · {num_docks_available} docks"}, map_style="dark"))
        st.caption("Colour: red = nearly empty, blue = nearly full.")


# ================================================================== ABOUT
def tab_about(c):
    section("How to read this dashboard")
    st.markdown('<div class="cards">'
                '<div class="card"><span class="ic">🧭</span><b>Descriptive vs causal.</b> Weather and demand co-move with season, growth and the calendar. Everything labelled association is not a causal effect.</div>'
                '<div class="card"><span class="ic">🧪</span><b>Backtests, not one split.</b> Models are scored on several later time blocks and compared with simple baselines; skill vs the best baseline is what matters.</div>'
                '<div class="card"><span class="ic">🔒</span><b>Provenance first.</b> Real and synthetic runs are stored apart; the Data quality tab shows every exclusion and coverage gap.</div>'
                '</div>', unsafe_allow_html=True)
    section("Known limitations")
    st.markdown("""
* **Distance** is straight-line from actual end coordinates; trips without end coordinates are excluded from the duration model.
* **NYC filter** is a bounding box (it also contains Hoboken and Jersey City); borough polygons would be exact.
* **Weather** is one NYC-centre point; observed values stand in for forecasts.
* **Demand** counts started trips, not unmet demand; empty stations censor it.
* **Station capacity** (if used) is a current snapshot, not historical.
* **Linear models** cannot capture interactions or non-linearity.
* **One short period** cannot show seasonal generalisation.
""")
    section("Commands")
    st.code("python run_all.py status          # what is done / next\npython run_all.py all             # run everything (resumable)\n"
            "python run_all.py phase2 --force  # redo EDA only\npython -m streamlit run app/dashboard.py", language="bash")


for tab, fn in zip(tabs, (tab_overview, tab_rhythm, tab_stations, tab_weather, tab_models, tab_predict, tab_quality, tab_live, tab_about)):
    with tab:
        fn(ctx)
