"""
dashboard/app.py
----------------
The board. Armed levels sorted by how close price is to triggering them.

The top of this screen is what is about to happen. That's the whole design.
v3's dashboard showed you completed patterns -- a museum of trades you'd
already missed. This shows you loaded guns.
"""

import time

import pandas as pd
import requests
import streamlit as st

RAW_URL = "https://raw.githubusercontent.com/yomerosho/strat-alerts/main/latest_scan.json"

st.set_page_config(page_title="Strat — Armed Levels", page_icon="🎯", layout="wide")


@st.cache_data(ttl=0)
def load(nocache: int):
    # GitHub's raw CDN caches aggressively; the query param busts it.
    r = requests.get(f"{RAW_URL}?nocache={nocache}", timeout=15)
    r.raise_for_status()
    return r.json()


try:
    data = load(int(time.time()))
except Exception as e:
    st.error(f"Couldn't load scan data: {e}")
    st.stop()

if data.get("version") != 4:
    st.warning("This scan file was written by an older scanner version. Re-run the scanner.")

levels = data.get("armed_levels", [])
gen_et = data.get("generated_at_et", data.get("generated_at_utc", "?"))

# ---------------------------------------------------------------- header
st.title("🎯 Armed Levels")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Armed", len(levels))
c2.metric("Tier 1", sum(1 for l in levels if l["tier"] == "TIER1"))
c3.metric("Tier 2", sum(1 for l in levels if l["tier"] == "TIER2"))
c4.metric("Setup TFs", " / ".join(data.get("setup_timeframes", [])))
st.caption(f"Last scan: {gen_et[:19].replace('T', ' ')} ET · {len(data.get('tickers', []))} tickers")

st.divider()

if not levels:
    st.info("Nothing armed right now. No inside bars on 2H/4H, no live Failed-2s.")
    st.caption("An empty board is a real answer — it means the scanner ran and found no setups.")
    st.stop()

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filter")
    tf_pick = st.multiselect("Setup timeframe", ["2H", "4H"], default=["2H", "4H"])
    tier_pick = st.multiselect(
        "Tier", ["ARMED", "TIER1", "TIER2"], default=["ARMED", "TIER1", "TIER2"]
    )
    dir_pick = st.multiselect("Direction", ["bull", "bear"], default=["bull", "bear"])
    max_dist = st.slider("Max distance from trigger (%)", 0.1, 5.0, 2.0, 0.1)

rows = [
    l for l in levels
    if l["setup_tf"] in tf_pick
    and l["tier"] in tier_pick
    and l["direction"] in dir_pick
    and abs(l["distance_pct"]) <= max_dist
]

if not rows:
    st.info("Nothing matches those filters.")
    st.stop()

TIER_RANK = {"TIER2": 2, "TIER1": 1, "ARMED": 0}
rows.sort(key=lambda d: (-TIER_RANK[d["tier"]], abs(d["distance_pct"])))

# ---------------------------------------------------------------- cards
TIER_STYLE = {
    "TIER2": ("🎯", "#16a34a", "Conviction — 15m closed through"),
    "TIER1": ("⚡", "#eab308", "Early — 5m closed through"),
    "ARMED": ("👀", "#64748b", "Loaded — not triggered"),
}

for l in rows:
    icon, color, blurb = TIER_STYLE[l["tier"]]
    arrow = "🟢" if l["direction"] == "bull" else "🔴"
    side = l["trigger_side"]
    hot = l["family"] == "f2" and l["tier"] == "TIER1"

    through = l["distance_pct"] > 0
    dist_txt = (
        f"**{abs(l['distance_pct']):.2f}% through**" if through
        else f"{abs(l['distance_pct']):.2f}% away"
    )

    with st.container(border=True):
        head, body = st.columns([1, 3])

        with head:
            st.markdown(
                f"### {icon} {arrow} {l['symbol']}"
                f"<br><span style='color:{color};font-weight:600'>{l['tier']}</span>"
                f"<br><small>{l['setup_tf']} · {l['pattern']}</small>",
                unsafe_allow_html=True,
            )
            if hot:
                st.markdown("🔥 **F2 — act on this tier**")

        with body:
            st.markdown(
                f"Needs a close **{side} {l['level']:.2f}** · "
                f"now **{l['current_price']:.2f}** ({dist_txt})"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trigger", f"{l['level']:.2f}")
            m2.metric("Stop", f"{l['invalidation']:.2f}" if l["invalidation"] else "—")
            m3.metric("Target", f"{l['target']:.2f}" if l["target"] else "—")
            m4.metric("Continuity", l["continuity"])

            bits = [blurb]
            if l["tier"] == "TIER1" and l.get("minutes_to_next_15m") is not None:
                m = l["minutes_to_next_15m"]
                bits.append("15m closing now" if m <= 1 else f"15m closes in {m} min")
            if l.get("tier1_time"):
                bits.append(f"T1 {l['tier1_time'][11:16]}")
            if l.get("tier2_time"):
                bits.append(f"T2 {l['tier2_time'][11:16]}")
            if l.get("setup_bar_closes_at"):
                bits.append(f"{l['setup_tf']} bar closes {l['setup_bar_closes_at'][11:16]}")
            st.caption(" · ".join(bits))

# ---------------------------------------------------------------- table
st.divider()
with st.expander("Raw table"):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
