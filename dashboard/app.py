"""
dashboard/app.py
----------------
The board. Armed levels sorted by how close price is to triggering them.

Design intent: this is a terminal, not a report. The top of the screen is what
is about to happen.

Direction carries the hue -- green/red, non-negotiable for a trader, you read
it before you read it. Tier carries the luminance: muted slate when a level is
merely loaded, amber the moment a 5m confirms, full brightness at Tier 2.

The signature element is the DISTANCE RAIL on each card. The trigger is a fixed
tick at centre; the price marker slides toward it and the gap fills in as price
closes on the level. You SEE how close a setup is instead of parsing a decimal.
Cards sort by that distance, so the rails at the top are the ones nearly
touching.
"""

import time

import pandas as pd
import requests
import streamlit as st

RAW_URL = "https://raw.githubusercontent.com/yomerosho/strat-alerts/main/latest_scan.json"

st.set_page_config(page_title="Armed Levels", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter+Tight:wght@400;600;700;800&display=swap');

:root {
  --bg:#0e141b; --surface:#151d27; --surface-2:#1b2532; --line:#26303f;
  --text:#dbe4ee; --dim:#71829a;
  --bull:#3fd07f; --bear:#ff5f56; --amber:#f0b429; --slate:#55657c;
}
.stApp { background:var(--bg); }
html, body, [class*="css"] { font-family:'Inter Tight',system-ui,sans-serif; color:var(--text); }
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
h1 { font-weight:800; letter-spacing:-.025em; color:var(--text); margin-bottom:0; }

.strip { display:flex; gap:2.6rem; padding:.9rem 0 1.3rem; border-bottom:1px solid var(--line);
         margin-bottom:1.5rem; flex-wrap:wrap; }
.stat .k { display:block; font-size:.66rem; text-transform:uppercase; letter-spacing:.12em;
           color:var(--dim); font-weight:600; margin-bottom:.15rem; }
.stat .v { display:block; font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
           font-weight:600; line-height:1.05; font-variant-numeric:tabular-nums; }
.v.amber{color:var(--amber);} .v.live{color:var(--bull);} .v.zero{color:var(--slate);}

.card { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--slate);
        border-radius:7px; padding:1rem 1.15rem; margin-bottom:.7rem; }
.card.t1 { border-left-color:var(--amber);
           background:linear-gradient(90deg,rgba(240,180,41,.06),var(--surface) 24%); }
.card.t2bull { border-left-color:var(--bull);
           background:linear-gradient(90deg,rgba(63,208,127,.09),var(--surface) 24%); }
.card.t2bear { border-left-color:var(--bear);
           background:linear-gradient(90deg,rgba(255,95,86,.09),var(--surface) 24%); }
.card.weak { opacity:.48; }

.row { display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }
.sym { font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:700; }
.sym.bull{color:var(--bull);} .sym.bear{color:var(--bear);}
.tag { font-size:.62rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
       padding:.17rem .45rem; border-radius:3px; }
.tag.armed{background:rgba(85,101,124,.22);color:var(--slate);}
.tag.t1{background:rgba(240,180,41,.16);color:var(--amber);}
.tag.t2bull{background:rgba(63,208,127,.16);color:var(--bull);}
.tag.t2bear{background:rgba(255,95,86,.16);color:var(--bear);}
.setup { font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:var(--dim); }

.thesis { margin:.55rem 0 0; font-size:.9rem; color:var(--text); }
.thesis b { font-family:'IBM Plex Mono',monospace; font-weight:600; }

.rail { position:relative; height:24px; margin:.8rem 0 .3rem; }
.rail .track{position:absolute;top:11px;left:0;right:0;height:2px;background:var(--surface-2);border-radius:2px;}
.rail .fill {position:absolute;top:11px;height:2px;border-radius:2px;}
.rail .tick {position:absolute;top:3px;left:50%;width:2px;height:18px;background:var(--dim);}
.rail .dot  {position:absolute;top:6px;width:11px;height:11px;border-radius:50%;
             border:2px solid var(--bg);transform:translateX(-50%);}
.rail .lbl  {position:absolute;top:-2px;font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:var(--dim);}

.nums { display:flex; gap:2rem; margin-top:.75rem; flex-wrap:wrap; }
.num .k{display:block;font-size:.61rem;text-transform:uppercase;letter-spacing:.1em;
        color:var(--dim);font-weight:600;margin-bottom:.1rem;}
.num .v{display:block;font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;
        color:var(--text);font-variant-numeric:tabular-nums;}
.num .v.bad{color:var(--bear);} .num .v.good{color:var(--bull);}

.foot { margin-top:.7rem; font-size:.72rem; color:var(--dim); font-family:'IBM Plex Mono',monospace; }
.foot .hot { color:var(--bear); font-weight:700; }
.warn { margin-top:.5rem; font-size:.74rem; color:var(--amber); }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=0)
def load(nocache: int):
    r = requests.get(f"{RAW_URL}?nocache={nocache}", timeout=15)
    r.raise_for_status()
    return r.json()


st.markdown("# Armed Levels")

try:
    data = load(int(time.time()))
except Exception as e:
    st.error(f"Scan data didn't load: {e}")
    st.caption("Check that the scanner workflow ran and committed latest_scan.json.")
    st.stop()

levels = data.get("armed_levels", [])
min_rr = float(data.get("min_risk_reward", 1.0))
gen = str(data.get("generated_at_et", data.get("generated_at_utc", "")))[:16].replace("T", " ")

n1 = sum(1 for l in levels if l["tier"] == "TIER1")
n2 = sum(1 for l in levels if l["tier"] == "TIER2")

st.markdown(f"""
<div class="strip">
  <span class="stat"><span class="k">Armed</span><span class="v">{len(levels)}</span></span>
  <span class="stat"><span class="k">Tier 1 · 5m</span>
    <span class="v {'amber' if n1 else 'zero'}">{n1}</span></span>
  <span class="stat"><span class="k">Tier 2 · 15m</span>
    <span class="v {'live' if n2 else 'zero'}">{n2}</span></span>
  <span class="stat"><span class="k">Setup</span>
    <span class="v">{' / '.join(data.get('setup_timeframes', []))}</span></span>
  <span class="stat"><span class="k">Last scan</span><span class="v">{gen[-5:]}</span></span>
</div>
""", unsafe_allow_html=True)

if not levels:
    st.markdown('<div class="card"><span class="setup">No inside bars on 2H or 4H, '
                'and no live Failed-2s. Nothing is loaded.</span></div>',
                unsafe_allow_html=True)
    st.caption(f"Scanner ran {gen} ET across {len(data.get('tickers', []))} tickers.")
    st.stop()

with st.sidebar:
    st.markdown("### Filter")
    tf_pick = st.multiselect("Setup timeframe", ["2H", "4H"], default=["2H", "4H"])
    tier_pick = st.multiselect("Tier", ["ARMED", "TIER1", "TIER2"],
                               default=["ARMED", "TIER1", "TIER2"])
    dir_pick = st.multiselect("Direction", ["bull", "bear"], default=["bull", "bear"])
    max_dist = st.slider("Max distance from trigger (%)", 0.1, 5.0, 1.5, 0.1)
    hide_weak = st.checkbox(f"Hide R:R below {min_rr:g}", value=True)

rows = [
    l for l in levels
    if l["setup_tf"] in tf_pick and l["tier"] in tier_pick and l["direction"] in dir_pick
    and abs(l["distance_pct"]) <= max_dist
    and not (hide_weak and (l.get("risk_reward") if l.get("risk_reward") is not None else 99) < min_rr)
]

if not rows:
    st.markdown('<div class="card"><span class="setup">Nothing matches those filters.</span></div>',
                unsafe_allow_html=True)
    st.stop()

RANK = {"TIER2": 2, "TIER1": 1, "ARMED": 0}
rows.sort(key=lambda d: (-RANK[d["tier"]], abs(d["distance_pct"])))

for l in rows:
    d = l["direction"]
    tier = l["tier"]
    hue = "var(--bull)" if d == "bull" else "var(--bear)"
    rr = l.get("risk_reward")
    hot = l["family"] == "f2" and tier == "TIER1"
    weak = rr is not None and min_rr > 0 and rr < min_rr

    cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": ""}[tier] + (" weak" if weak else "")
    tag_cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": "armed"}[tier]
    tag_txt = {"TIER2": "Tier 2", "TIER1": "Tier 1", "ARMED": "Armed"}[tier]

    # ---- distance rail ----
    scale = max(max_dist, 0.1)
    off = max(-1.0, min(1.0, l["distance_pct"] / scale)) * 46
    dot = 50 + off
    through = l["distance_pct"] > 0
    fill_col = hue if through else "var(--slate)"
    lo, hi = (min(50, dot), max(50, dot))
    dist_txt = (f"{abs(l['distance_pct']):.2f}% through" if through
                else f"{abs(l['distance_pct']):.2f}% away")

    stop_txt = f"{l['invalidation']:.2f}" if l.get("invalidation") is not None else "—"
    tgt_txt = f"{l['target']:.2f}" if l.get("target") is not None else "—"
    rr_txt = f"{rr:.1f}" if rr is not None else "—"
    rr_cls = "bad" if weak else ("good" if (rr is not None and rr >= 2) else "")

    foot = [{"TIER2": "15m closed through", "TIER1": "5m closed through",
             "ARMED": "loaded, not triggered"}[tier]]
    if tier == "TIER1" and l.get("minutes_to_next_15m") is not None:
        m = l["minutes_to_next_15m"]
        foot.append("15m closing now" if m <= 1 else f"15m closes in {m}m")
    if l.get("tier1_time"):
        foot.append(f"T1 {l['tier1_time'][11:16]}")
    if l.get("tier2_time"):
        foot.append(f"T2 {l['tier2_time'][11:16]}")
    if l.get("setup_bar_closes_at"):
        foot.append(f"{l['setup_tf']} bar ends {l['setup_bar_closes_at'][11:16]}")

    hot_html = '<span class="hot"> · F2 — act on this tier</span>' if hot else ""
    warn_html = (f'<div class="warn">R:R {rr:.2f} — the target sits on top of the trigger. '
                 f'The pattern is real; the trade probably isn\'t.</div>') if weak else ""

    st.markdown(f"""
<div class="card {cls}">
  <div class="row">
    <span class="sym {d}">{l['symbol']}</span>
    <span class="tag {tag_cls}">{tag_txt}</span>
    <span class="setup">{l['setup_tf']} · {l['pattern']}</span>
  </div>
  <div class="thesis">Needs a close <b>{l['trigger_side']} {l['level']:.2f}</b> · now <b>{l['current_price']:.2f}</b></div>
  <div class="rail">
    <div class="track"></div>
    <div class="fill" style="left:{lo}%;width:{hi-lo}%;background:{fill_col};"></div>
    <div class="tick"></div>
    <div class="dot" style="left:{dot}%;background:{fill_col};"></div>
    <div class="lbl" style="left:0;">{dist_txt}</div>
    <div class="lbl" style="right:0;">trigger {l['level']:.2f}</div>
  </div>
  <div class="nums">
    <span class="num"><span class="k">Trigger</span><span class="v">{l['level']:.2f}</span></span>
    <span class="num"><span class="k">Stop</span><span class="v">{stop_txt}</span></span>
    <span class="num"><span class="k">Target</span><span class="v">{tgt_txt}</span></span>
    <span class="num"><span class="k">R : R</span><span class="v {rr_cls}">{rr_txt}</span></span>
    <span class="num"><span class="k">Continuity</span><span class="v">{l['continuity']}</span></span>
  </div>
  <div class="foot">{' · '.join(foot)}{hot_html}</div>
  {warn_html}
</div>""", unsafe_allow_html=True)

st.divider()
with st.expander("Raw table"):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
