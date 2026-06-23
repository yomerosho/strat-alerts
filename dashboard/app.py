"""
dashboard/app.py
-----------------
Streamlit dashboard for the Strat Scanner, built around
a 0DTE workflow: FTFC computed across the higher timeframes (15Min/30Min/
1H/4H/1D by default), entries confirmed on the fast timeframes (5Min/15Min).

This app does NOT do any scanning itself -- GitHub Actions does that in the
background (see ../.github/workflows/strat-scanner.yml) and commits
`latest_scan.json` to the repo after every run. This app just fetches that
file straight from GitHub's raw content URL on every page load/refresh, so
it's always showing the latest committed scan -- no redeploy-lag dependency.

Required secrets (Streamlit Cloud: App settings -> Secrets):
        GITHUB_OWNER   = "your-github-username"
    GITHUB_REPO    = "strat-scanner"
    GITHUB_BRANCH  = "main"
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Strat Scanner", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------
# Dark theme styling (OBI-style)
# --------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0b0e14; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
h1, h2, h3, p, span, label, .stMarkdown { color: #e2e8f0; }
[data-testid="stCaptionContainer"] { color: #64748b !important; }
.stCheckbox label p { color: #94a3b8 !important; font-size: 13px; }
div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
.strat-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.strat-table th {
    text-align: left; padding: 8px 10px; color: #64748b; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid #1e293b;
}
.strat-table td {
    padding: 10px 10px; border-bottom: 1px solid #161b26; color: #cbd5e1;
}
.strat-table tr:hover td { background-color: #11151f; }
.ticker-cell { color: #f1f5f9; font-weight: 700; }
.badge {
    display: inline-block; padding: 2px 7px; border-radius: 5px; font-weight: 700;
    font-size: 11px; margin-right: 3px;
}
.confluence-card {
    background: #0f1320; border: 1px solid #1e293b; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 8px;
}
.confluence-card.bull { border-left: 3px solid #22c55e; }
.confluence-card.bear { border-left: 3px solid #ef4444; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.continuity-label { font-size: 11px; font-weight: 700; margin-left: 4px; letter-spacing: 0.03em; }

/* Streamlit's native widgets default to a light theme regardless of our
   custom CSS above -- override buttons, expanders, and text inputs so
   they're actually readable against the dark background. */
.stButton button, [data-testid="stBaseButton-secondary"] {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}
.stButton button:hover, [data-testid="stBaseButton-secondary"]:hover {
    background-color: #273449 !important;
    border-color: #475569 !important;
    color: #f1f5f9 !important;
}
[data-testid="stExpander"] {
    background-color: #0f1320 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    background-color: #0f1320 !important;
    color: #e2e8f0 !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #161b26 !important;
}
[data-testid="stExpanderDetails"] {
    background-color: #0f1320 !important;
    color: #cbd5e1 !important;
}
[data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] strong {
    color: #cbd5e1 !important;
}
.stTextInput input {
    background-color: #161b26 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}
</style>
""", unsafe_allow_html=True)

STRAT_BADGE_STYLE = {
    "1":  ("#3f2f12", "#fbbf24"),   # inside bar -- amber
    "2U": ("#052e1b", "#4ade80"),   # directional up -- green
    "2D": ("#2e0a0a", "#f87171"),   # directional down -- red
    "3":  ("#2e1065", "#c4b5fd"),   # outside bar -- purple
}
DIR_COLOR = {"bull": "#22c55e", "bear": "#ef4444", "neutral": "#475569"}





# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
@st.cache_data(ttl=0, show_spinner=False)
def fetch_snapshot(owner: str, repo: str, branch: str, _bust: int = 0) -> dict | None:
    # _bust is a timestamp passed in on every call so Streamlit never
    # serves a cached version. The real CDN cache-bust happens via the
    # ?nocache= query param on the GitHub raw URL -- without it, GitHub's
    # CDN can serve a stale file for 5-10 minutes after a new commit even
    # if you hit "Refresh now".
    url = (
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/latest_scan.json"
        f"?nocache={_bust}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Couldn't fetch scan data from GitHub: {exc}")
        return None


owner = st.secrets.get("GITHUB_OWNER", "")
repo = st.secrets.get("GITHUB_REPO", "")
branch = st.secrets.get("GITHUB_BRANCH", "main")

if not owner or not repo:
    st.error("GITHUB_OWNER / GITHUB_REPO are not configured in this app's Secrets yet.")
    st.stop()

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.title("📊 Strat Scanner")
    st.caption("THE STRAT · FTFC CONFLUENCE · 5m / 15m / 30m / 1H / 4H / 1D")
with col_refresh:
    if st.button("🔄 Refresh now"):
        fetch_snapshot.clear()  # also forces a new _bust value on next call

data = fetch_snapshot(owner, repo, branch, _bust=int(time.time()))

if not data:
    st.warning(
        "No scan data yet. Either the GitHub Actions workflow hasn't run "
        "yet, or `latest_scan.json` hasn't been committed. Trigger a manual "
        "run from the Actions tab in your repo."
    )
    st.stop()

generated_at = data.get("generated_at_utc", "unknown")
states = data.get("states", [])
all_timeframes = data.get("timeframes", ["5Min", "15Min", "30Min", "1H", "4H", "1D"])
ftfc_timeframes = data.get("ftfc_timeframes", ["15Min", "30Min", "1H", "4H", "1D"])
entry_timeframes = data.get("entry_timeframes", ["5Min", "15Min"])
pattern_watch_timeframes = data.get("pattern_watch_timeframes", ["1H", "4H", "1D"])

# Continuity thresholds -- must match config.py exactly so the dashboard
# shows only signals that would actually send a Telegram alert.
min_continuity_watch = data.get("min_continuity_watch", 4)
min_continuity_entry = data.get("min_continuity_entry", 4)
min_continuity_entry_failed2 = data.get("min_continuity_entry_failed2", 3)

try:
    gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    age_min = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 60
    st.caption(f"Last scan: {generated_at}  ·  ({age_min:.0f} min ago)")
except Exception:
    st.caption(f"Last scan: {generated_at}")

if not states:
    st.info("Scan ran but returned no states (not enough bar history yet for any symbol).")
    st.stop()

df = pd.DataFrame(states)
df["strat_sequence"] = df["last_three_labels"].apply(lambda labels: list(labels))
df["last_label"] = df["last_three_labels"].apply(lambda labels: labels[-1] if labels else None)
# "patterns" won't exist in latest_scan.json files generated by the old
# scanner (before the pattern detection rebuild). Default to empty list so
# the dashboard still renders -- patterns will populate once the new
# scanner.py/main.py are deployed and the workflow runs once.
if "patterns" not in df.columns:
    df["patterns"] = [[] for _ in range(len(df))]
df["patterns"] = df["patterns"].apply(lambda p: p if isinstance(p, list) else [])
df["actionable_patterns"] = df["patterns"].apply(lambda ps: [p for p in ps if p.get("actionable")])
df["pmg_patterns"] = df["patterns"].apply(lambda ps: [p for p in ps if p.get("name") == "PMG"])


def direction(row) -> str:
    """bull / bear / neutral -- a live trigger wins, otherwise fall back to
    the last completed bar's label. Mirrors scanner.StratState.direction."""
    if row["trigger"] == "bullish_trigger":
        return "bull"
    if row["trigger"] == "bearish_trigger":
        return "bear"
    if row["last_label"] == "2U":
        return "bull"
    if row["last_label"] == "2D":
        return "bear"
    return "neutral"


df["direction"] = df.apply(direction, axis=1)
df["is_inside_setup"] = (df["last_label"] == "1") & (df["trigger"].isna())
df["is_fired"] = df["trigger"].notna()
df["has_pattern"] = df["actionable_patterns"].apply(lambda ps: len(ps) > 0)
df["is_actionable"] = df["is_inside_setup"] | df["is_fired"] | df["has_pattern"]

# --------------------------------------------------------------------------
# Continuity score -- "X/5" agreement across ftfc_timeframes, computed per
# symbol+direction. This is bias CONTEXT now, not a strict gate -- matches
# main.py's compute_continuity_score exactly, so what you see here is what
# would appear in any alert.
# --------------------------------------------------------------------------
def continuity_score(symbol: str, target_direction: str) -> str:
    sym_df = df[(df["symbol"] == symbol) & (df["timeframe"].isin(ftfc_timeframes))]
    if sym_df.empty:
        return "0/0"
    agree = (sym_df["direction"] == target_direction).sum()
    return f"{agree}/{len(sym_df)}"


def dominant_direction(symbol: str) -> str:
    """For display purposes: which direction has more FTFC agreement."""
    sym_df = df[(df["symbol"] == symbol) & (df["timeframe"].isin(ftfc_timeframes))]
    if sym_df.empty:
        return "neutral"
    counts = sym_df["direction"].value_counts()
    bull = counts.get("bull", 0)
    bear = counts.get("bear", 0)
    if bull == 0 and bear == 0:
        return "neutral"
    return "bull" if bull >= bear else "bear"


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_badge(label: str) -> str:
    bg, fg = STRAT_BADGE_STYLE.get(label, ("#1e293b", "#94a3b8"))
    return f'<span class="badge" style="background:{bg};color:{fg};">{label}</span>'


def render_sequence(labels: list[str]) -> str:
    return "".join(render_badge(l) for l in labels)


def render_signal(trigger: str | None) -> str:
    if trigger == "bullish_trigger":
        return '<span style="color:#4ade80;font-weight:700;">▲ TRIGGER</span>'
    if trigger == "bearish_trigger":
        return '<span style="color:#f87171;font-weight:700;">▼ TRIGGER</span>'
    return '<span style="color:#475569;">—</span>'


def render_patterns(actionable_patterns: list[dict], pmg_patterns: list[dict]) -> str:
    if not actionable_patterns and not pmg_patterns:
        return '<span style="color:#475569;">—</span>'
    chips = []
    for p in actionable_patterns:
        color = "#4ade80" if p["direction"] == "bull" else "#f87171"
        bg = "#052e1b" if p["direction"] == "bull" else "#2e0a0a"
        chips.append(
            f'<span class="badge" style="background:{bg};color:{color};" title="{p.get("note","")}">{p["name"]}</span>'
        )
    for p in pmg_patterns:
        chips.append(
            f'<span class="badge" style="background:#3f2f12;color:#fbbf24;" title="{p.get("note","")}">⚠ PMG</span>'
        )
    return "".join(chips)


def format_bar_time(iso_str: str) -> str:
    """Human-readable version of an ISO timestamp. Daily bars land exactly
    on midnight, so just show the date for those; intraday bars show
    date + time."""
    try:
        dt = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return iso_str
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt.strftime("%b %d, %Y")
    return dt.strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")


def render_continuity(symbol: str) -> str:
    direction_for_score = dominant_direction(symbol)
    score = continuity_score(symbol, direction_for_score)
    color = DIR_COLOR.get(direction_for_score, "#94a3b8")
    word = {"bull": "BULL", "bear": "BEAR", "neutral": "FLAT"}[direction_for_score]
    return f'<span style="color:{color};font-weight:700;">{score}</span> <span style="color:{color};font-size:11px;">{word}</span>'


def render_table(rows: pd.DataFrame) -> str:
    header = (
        "<tr><th>Ticker</th><th>Price</th><th>Strat Sequence</th><th>Signal</th>"
        "<th>Pattern</th><th>Continuity</th><th>Break ↑</th><th>Break ↓</th><th>Last Bar</th></tr>"
    )
    body_rows = []
    for _, r in rows.iterrows():
        body_rows.append(
            "<tr>"
            f'<td class="ticker-cell">{r["symbol"]}</td>'
            f'<td>{r["current_price"]:.2f}</td>'
            f'<td>{render_sequence(r["strat_sequence"])}</td>'
            f'<td>{render_signal(r["trigger"])}</td>'
            f'<td>{render_patterns(r["actionable_patterns"], r["pmg_patterns"])}</td>'
            f'<td>{render_continuity(r["symbol"])}</td>'
            f'<td>{r["last_completed_high"]:.2f}</td>'
            f'<td>{r["last_completed_low"]:.2f}</td>'
            f'<td style="color:#64748b;">{format_bar_time(r["last_bar_time"])}</td>'
            "</tr>"
        )
    return f'<table class="strat-table">{header}{"".join(body_rows)}</table>'


# --------------------------------------------------------------------------
# Top sections: Watch signals (anticipatory, higher timeframes) and Entry
# signals (the "go" moment, entry timeframes) -- mirrors main.py's alert
# logic exactly, so this is what would have messaged you on Telegram/WhatsApp.
# --------------------------------------------------------------------------
def render_signal_card(symbol: str, tf: str, pattern: dict, kind: str) -> None:
    direction_word = "BULLISH" if pattern["direction"] == "bull" else "BEARISH"
    emoji = "🟢" if pattern["direction"] == "bull" else "🔴"
    score = continuity_score(symbol, pattern["direction"])
    icon = "👀" if kind == "watch" else "🎯"
    stop_text = f' · stop {pattern["stop_level"]:.2f}' if pattern.get("stop_level") is not None else ""
    st.markdown(
        f'<div class="confluence-card {pattern["direction"]}">'
        f'{icon} <span style="font-size:16px;font-weight:800;color:#f1f5f9;">{emoji} {symbol}</span> '
        f'<span style="color:{DIR_COLOR[pattern["direction"]]};font-weight:700;">{direction_word} {pattern["name"]}</span> '
        f'on <b>{tf}</b> · continuity {score}{stop_text}'
        f'<div style="color:#64748b;font-size:12px;margin-top:4px;">{pattern.get("note","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def passes_threshold(pattern: dict, kind: str, score: str) -> bool:
    """Mirrors main.py's threshold logic exactly -- only show signals on the
    dashboard that would actually send a Telegram alert."""
    try:
        agree = int(score.split("/")[0])
    except (ValueError, IndexError):
        agree = 0
    if kind == "watch":
        return agree >= min_continuity_watch
    if pattern.get("name") == "Failed-2":
        return agree >= min_continuity_entry_failed2
    return agree >= min_continuity_entry


st.subheader("👀 Watch Signals")
st.caption("Named setups forming on the higher timeframes — anticipatory, no entry trigger yet.")
watch_rows = df[df["timeframe"].isin(pattern_watch_timeframes)]
watch_found = False
for _, r in watch_rows.iterrows():
    for p in r["actionable_patterns"]:
        score = continuity_score(r["symbol"], p["direction"])
        if not passes_threshold(p, "watch", score):
            continue
        render_signal_card(r["symbol"], r["timeframe"], p, "watch")
        watch_found = True
if not watch_found:
    st.caption("No higher-timeframe setups meeting the continuity threshold right now.")

st.divider()

st.subheader("🎯 Entry Signals")
st.caption("Named setups printing directly on your entry timeframes — the actual \"go\" moment.")
entry_rows = df[df["timeframe"].isin(entry_timeframes)]
entry_found = False
for _, r in entry_rows.iterrows():
    for p in r["actionable_patterns"]:
        score = continuity_score(r["symbol"], p["direction"])
        if not passes_threshold(p, "entry", score):
            continue
        render_signal_card(r["symbol"], r["timeframe"], p, "entry")
        entry_found = True
if not entry_found:
    st.caption("No entry-timeframe setups meeting the continuity threshold right now.")

st.divider()

# --------------------------------------------------------------------------
# Filters + tabs (per timeframe, for manual inspection)
# --------------------------------------------------------------------------
tf_tabs = st.tabs(all_timeframes)

for tf, tab in zip(all_timeframes, tf_tabs):
    with tab:
        tf_df = df[df["timeframe"] == tf].copy()
        if tf_df.empty:
            st.info(f"No {tf} data in the latest scan.")
            continue

        badges = []
        if tf in entry_timeframes:
            badges.append("⚡ entry timeframe")
        if tf in pattern_watch_timeframes:
            badges.append("👀 pattern-watch timeframe")
        if badges:
            st.caption(" · ".join(badges))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        f_actionable = c1.checkbox("Actionable", key=f"act_{tf}")
        f_inside = c2.checkbox("Inside bars", key=f"ins_{tf}")
        f_fired = c3.checkbox("Fired", key=f"fire_{tf}")
        f_bull = c4.checkbox("Bull", key=f"bull_{tf}")
        f_bear = c5.checkbox("Bear", key=f"bear_{tf}")
        f_pattern = c6.checkbox("Has pattern", key=f"pat_{tf}")

        if f_actionable:
            tf_df = tf_df[tf_df["is_actionable"]]
        if f_inside:
            tf_df = tf_df[tf_df["is_inside_setup"]]
        if f_fired:
            tf_df = tf_df[tf_df["is_fired"]]
        if f_bull:
            tf_df = tf_df[tf_df["direction"] == "bull"]
        if f_bear:
            tf_df = tf_df[tf_df["direction"] == "bear"]
        if f_pattern:
            tf_df = tf_df[tf_df["has_pattern"]]

        if tf_df.empty:
            st.info("No rows match the current filters.")
            continue

        tf_df = tf_df.sort_values(
            by=["has_pattern", "trigger", "symbol"],
            key=lambda col: col.isna() if col.name == "trigger" else col,
            ascending=[False, True, True],
        )
        st.markdown(render_table(tf_df), unsafe_allow_html=True)

st.divider()
with st.expander("Watchlist tickers in this scan"):
    st.write(", ".join(data.get("tickers", [])))
with st.expander("About Watch / Entry signals and continuity"):
    st.markdown(
        f"**Watch signals** are named Strat setups (Failed-2, 2-1-2, 3-1-2, 3-2-2, 1-2-2 Rev Strat) forming on "
        f"the higher timeframes ({', '.join(pattern_watch_timeframes)}) -- anticipatory, before any entry-timeframe "
        f"trigger exists. **Entry signals** are the same named setups printing directly on your entry timeframes "
        f"({', '.join(entry_timeframes)}) -- the actual \"go\" moment.\n\n"
        f"**Continuity** shows how many of the {len(ftfc_timeframes)} FTFC timeframes "
        f"({', '.join(ftfc_timeframes)}) agree with the dominant direction shown -- bias context, not a strict "
        f"gate. 4/5 or 5/5 in the alert's direction is a green light; 3/5 means proceed with tighter targets; "
        f"2/5 or below means wait.\n\n"
        f"**PMG** (⚠ badge) means 6+ consecutive same-direction bars on that timeframe -- a warning that the move "
        f"is stretched and due for a snapback, not a standalone entry signal by itself."
    )
