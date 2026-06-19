"""
dashboard/app.py
-----------------
Password-protected Streamlit dashboard for the Strat Scanner, built around
a 0DTE workflow: FTFC computed across the higher timeframes (15Min/30Min/
1H/4H/1D by default), entries confirmed on the fast timeframes (5Min/15Min).

This app does NOT do any scanning itself -- GitHub Actions does that in the
background (see ../.github/workflows/strat-scanner.yml) and commits
`latest_scan.json` to the repo after every run. This app just fetches that
file straight from GitHub's raw content URL on every page load/refresh, so
it's always showing the latest committed scan -- no redeploy-lag dependency.

Required secrets (Streamlit Cloud: App settings -> Secrets):
    APP_PASSWORD   = "whatever-you-want"
    GITHUB_OWNER   = "your-github-username"
    GITHUB_REPO    = "strat-scanner"
    GITHUB_BRANCH  = "main"
"""

from __future__ import annotations

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
# Password gate
# --------------------------------------------------------------------------
def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.title("📊 Strat Scanner")
    pwd = st.text_input("Password", type="password")
    if st.button("Unlock") or pwd:
        expected = st.secrets.get("APP_PASSWORD")
        if not expected:
            st.error("APP_PASSWORD is not configured in this app's Secrets yet.")
            return False
        if pwd == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_snapshot(owner: str, repo: str, branch: str) -> dict | None:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/latest_scan.json"
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
        fetch_snapshot.clear()

data = fetch_snapshot(owner, repo, branch)

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
df["is_actionable"] = df["is_inside_setup"] | df["is_fired"]

# --------------------------------------------------------------------------
# FTFC -- computed per symbol across ftfc_timeframes, independent of which
# tab is selected. Entry signal -- a live trigger on an entry timeframe
# matching the FTFC direction. This mirrors main.py's alert logic exactly,
# so what you see here is what would have triggered a Telegram/WhatsApp alert.
# --------------------------------------------------------------------------
continuity_map: dict[str, dict[str, str]] = {}
for symbol in df["symbol"].unique():
    sym_df = df[df["symbol"] == symbol]
    continuity_map[symbol] = {
        tf: (sym_df[sym_df["timeframe"] == tf]["direction"].iloc[0]
             if not sym_df[sym_df["timeframe"] == tf].empty else "neutral")
        for tf in ftfc_timeframes
    }


def ftfc_status(symbol: str) -> str:
    dirs = set(continuity_map.get(symbol, {}).values())
    if dirs == {"bull"}:
        return "bull"
    if dirs == {"bear"}:
        return "bear"
    return "mixed"


def entry_signal(symbol: str) -> tuple[str, str] | None:
    """Returns (timeframe, trigger) if an entry timeframe has a trigger
    matching this symbol's FTFC direction, else None."""
    status = ftfc_status(symbol)
    if status == "mixed":
        return None
    wanted = "bullish_trigger" if status == "bull" else "bearish_trigger"
    sym_df = df[df["symbol"] == symbol]
    for tf in entry_timeframes:
        row = sym_df[sym_df["timeframe"] == tf]
        if not row.empty and row.iloc[0]["trigger"] == wanted:
            return (tf, wanted)
    return None


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
    dots = "".join(
        f'<span class="dot" style="background:{DIR_COLOR.get(continuity_map[symbol].get(tf, "neutral"), "#475569")};" '
        f'title="{tf}"></span>'
        for tf in ftfc_timeframes
    )
    status = ftfc_status(symbol)
    label_color = DIR_COLOR.get(status, "#94a3b8")
    label_text = {"bull": "BULL", "bear": "BEAR", "mixed": "MIXED"}[status]
    return f'{dots}<span class="continuity-label" style="color:{label_color};">{label_text}</span>'


def render_table(rows: pd.DataFrame) -> str:
    header = (
        f"<tr><th>Ticker</th><th>Price</th><th>Strat Sequence</th><th>Signal</th>"
        f"<th>FTFC ({'/'.join(ftfc_timeframes)})</th><th>Break ↑</th><th>Break ↓</th><th>Last Bar</th></tr>"
    )
    body_rows = []
    for _, r in rows.iterrows():
        body_rows.append(
            "<tr>"
            f'<td class="ticker-cell">{r["symbol"]}</td>'
            f'<td>{r["current_price"]:.2f}</td>'
            f'<td>{render_sequence(r["strat_sequence"])}</td>'
            f'<td>{render_signal(r["trigger"])}</td>'
            f'<td>{render_continuity(r["symbol"])}</td>'
            f'<td>{r["last_completed_high"]:.2f}</td>'
            f'<td>{r["last_completed_low"]:.2f}</td>'
            f'<td style="color:#64748b;">{format_bar_time(r["last_bar_time"])}</td>'
            "</tr>"
        )
    return f'<table class="strat-table">{header}{"".join(body_rows)}</table>'


# --------------------------------------------------------------------------
# Top section: live confluence signals -- exactly what would have fired an
# alert. This is the primary view for a 0DTE workflow: FTFC agrees AND an
# entry timeframe has a live trigger.
# --------------------------------------------------------------------------
st.subheader("🎯 Live Confluence Signals")
confluence_rows = []
for symbol in sorted(df["symbol"].unique()):
    status = ftfc_status(symbol)
    sig = entry_signal(symbol)
    if status != "mixed" and sig is not None:
        confluence_rows.append((symbol, status, sig))

if not confluence_rows:
    st.caption("No symbol currently has full FTFC agreement *and* a live entry trigger. Check back, or browse by timeframe below.")
else:
    for symbol, status, (entry_tf, trigger) in confluence_rows:
        sym_df = df[(df["symbol"] == symbol) & (df["timeframe"] == entry_tf)]
        price = sym_df.iloc[0]["current_price"] if not sym_df.empty else None
        emoji = "🟢" if status == "bull" else "🔴"
        word = "BULLISH" if status == "bull" else "BEARISH"
        price_text = f"{price:.2f}" if price is not None else "—"
        st.markdown(
            f'<div class="confluence-card {status}">'
            f'<span style="font-size:16px;font-weight:800;color:#f1f5f9;">{emoji} {symbol}</span> '
            f'<span style="color:{DIR_COLOR[status]};font-weight:700;">{word} FTFC</span> '
            f'+ entry trigger on <b>{entry_tf}</b> · price {price_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

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

        if tf in entry_timeframes:
            st.caption(f"⚡ {tf} is one of your entry timeframes.")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        f_actionable = c1.checkbox("Actionable", key=f"act_{tf}")
        f_inside = c2.checkbox("Inside bars", key=f"ins_{tf}")
        f_fired = c3.checkbox("Fired", key=f"fire_{tf}")
        f_bull = c4.checkbox("Bull", key=f"bull_{tf}")
        f_bear = c5.checkbox("Bear", key=f"bear_{tf}")
        f_ftfc = c6.checkbox("FTFC only", key=f"ftfc_{tf}")

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
        if f_ftfc:
            tf_df = tf_df[tf_df["symbol"].apply(lambda s: ftfc_status(s) != "mixed")]

        if tf_df.empty:
            st.info("No rows match the current filters.")
            continue

        tf_df = tf_df.sort_values(
            by=["trigger", "symbol"], key=lambda col: col.isna() if col.name == "trigger" else col
        )
        st.markdown(render_table(tf_df), unsafe_allow_html=True)

st.divider()
with st.expander("Watchlist tickers in this scan"):
    st.write(", ".join(data.get("tickers", [])))
with st.expander("About FTFC + entry confluence"):
    st.markdown(
        f"**FTFC** ({', '.join(ftfc_timeframes)}) shows directional bias regardless of "
        f"which tab you're viewing -- the classic Strat **Full Timeframe Continuity** "
        f"concept. Green = bullish, red = bearish, gray = neutral/inside-bar/outside-bar.\n\n"
        f"**Entry timeframes** ({', '.join(entry_timeframes)}) are where actual triggers "
        f"are checked against the FTFC direction. The **Live Confluence Signals** section "
        f"at the top shows exactly what would have sent you a Telegram/WhatsApp alert -- "
        f"FTFC agreement *and* a live trigger on an entry timeframe, matching direction."
    )
