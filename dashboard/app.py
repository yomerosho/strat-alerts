"""
dashboard/app.py
-----------------
Password-protected Streamlit dashboard for the Strat Scanner.

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

STRAT_LABEL_COLORS = {
    "1": "#a78bfa",   # inside bar -- purple
    "2U": "#4ade80",  # directional up -- green
    "2D": "#f87171",  # directional down -- red
    "3": "#fbbf24",   # outside bar -- amber
}


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
@st.cache_data(ttl=60, show_spinner=False)
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
    st.caption("THE STRAT · CONTINUITY · LIVE 4H / DAILY / WEEKLY / MONTHLY")
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
all_timeframes = data.get("timeframes", ["4H", "1D", "1W", "1M"])

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
df["strat_sequence"] = df["last_three_labels"].apply(lambda labels: "-".join(labels))
df["signal"] = df["trigger"].map({
    "bullish_trigger": "🟢 Bullish trigger",
    "bearish_trigger": "🔴 Bearish trigger",
}).fillna("—")


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
tf_tabs = st.tabs(all_timeframes)

for tf, tab in zip(all_timeframes, tf_tabs):
    with tab:
        tf_df = df[df["timeframe"] == tf].copy()
        if tf_df.empty:
            st.info(f"No {tf} data in the latest scan.")
            continue

        actionable_only = st.checkbox("Actionable only (live triggers)", key=f"actionable_{tf}")
        if actionable_only:
            tf_df = tf_df[tf_df["trigger"].notna()]

        tf_df = tf_df.sort_values(
            by=["trigger", "symbol"], key=lambda col: col.isna() if col.name == "trigger" else col
        )

        display_df = tf_df[[
            "symbol", "current_price", "strat_sequence", "signal",
            "last_completed_high", "last_completed_low", "last_bar_time",
        ]].rename(columns={
            "symbol": "Ticker",
            "current_price": "Price",
            "strat_sequence": "Strat Sequence",
            "signal": "Signal",
            "last_completed_high": "Break ↑",
            "last_completed_low": "Break ↓",
            "last_bar_time": "Last Bar",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Break ↑": st.column_config.NumberColumn(format="%.2f"),
                "Break ↓": st.column_config.NumberColumn(format="%.2f"),
            },
        )

st.divider()
with st.expander("Watchlist tickers in this scan"):
    st.write(", ".join(data.get("tickers", [])))
