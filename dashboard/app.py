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
  --bg:#0d1319; --surface:#18212c; --surface-2:#243040; --line:#31405280;
  --text:#f1f6fb;      /* near-white: primary numbers must PUNCH */
  --dim:#a8bacd;       /* labels: readable, not decorative */
  --faint:#7f93a8;
  --bull:#3fe08a; --bear:#ff6b62; --amber:#ffc043; --slate:#7d90a8;
}
.stApp { background:var(--bg); }
html, body, [class*="css"] { font-family:'Inter Tight',system-ui,sans-serif; }
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:var(--text) !important; }

/* Streamlit wraps our HTML and colours it. Force our own. */
[data-testid="stMarkdownContainer"] h1 { color:var(--text) !important; font-weight:800;
  letter-spacing:-.03em; font-size:2.3rem; margin-bottom:.2rem; }

.strip { display:flex; gap:2.8rem; padding:1rem 0 1.4rem; border-bottom:1px solid var(--line);
         margin-bottom:1.6rem; flex-wrap:wrap; }
.stat .k { display:block !important; font-size:.7rem !important; text-transform:uppercase;
           letter-spacing:.13em; color:var(--dim) !important; font-weight:700; margin-bottom:.25rem; }
.stat .v { display:block !important; font-family:'IBM Plex Mono',monospace;
           font-size:1.85rem !important; font-weight:700; line-height:1;
           color:var(--text) !important; font-variant-numeric:tabular-nums; }
.stat .v.amber{color:var(--amber) !important;}
.stat .v.live {color:var(--bull) !important;}
.stat .v.zero {color:var(--faint) !important;}

.card { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--slate);
        border-radius:8px; padding:1.15rem 1.3rem; margin-bottom:.8rem; }
.card.t1 { border-left-color:var(--amber);
           background:linear-gradient(90deg,rgba(255,192,67,.09),var(--surface) 26%); }
.card.t2bull { border-left-color:var(--bull);
           background:linear-gradient(90deg,rgba(63,224,138,.11),var(--surface) 26%); }
.card.t2bear { border-left-color:var(--bear);
           background:linear-gradient(90deg,rgba(255,107,98,.11),var(--surface) 26%); }
.card.weak { opacity:.62; }

.row { display:flex; align-items:center; gap:.65rem; flex-wrap:wrap; }
.sym { font-family:'IBM Plex Mono',monospace; font-size:1.45rem !important; font-weight:700; }
.sym.bull{color:var(--bull) !important;} .sym.bear{color:var(--bear) !important;}
.tag { font-size:.64rem !important; font-weight:800; text-transform:uppercase; letter-spacing:.11em;
       padding:.22rem .5rem; border-radius:4px; }
.tag.armed{background:rgba(125,144,168,.25);color:#c3d0de !important;}
.tag.t1{background:rgba(255,192,67,.22);color:var(--amber) !important;}
.tag.t2bull{background:rgba(63,224,138,.22);color:var(--bull) !important;}
.tag.t2bear{background:rgba(255,107,98,.22);color:var(--bear) !important;}
.setup { font-family:'IBM Plex Mono',monospace; font-size:.85rem !important;
         color:var(--dim) !important; font-weight:500; }

.thesis { margin:.7rem 0 0; font-size:1rem !important; color:var(--dim) !important; }
.thesis b { font-family:'IBM Plex Mono',monospace; font-weight:700; color:var(--text) !important; }

/* ---- trade geometry bar ----
   Laid out in real price order: STOP | trigger | TARGET, with segment widths
   drawn to actual price distance. The ratio of the red block to the green
   block IS the reward:risk. You see it instead of reading it. The dot is
   where price actually is right now on that same axis. */
.geo { position:relative; height:46px; margin:1.15rem 0 .3rem; }
.geo .bar   { position:absolute; top:16px; left:0; right:0; height:14px;
              border-radius:4px; overflow:hidden; display:flex; }
.geo .risk  { background:rgba(255,107,98,.42); height:100%; }
.geo .rew   { background:rgba(63,224,138,.38); height:100%; }
.geo .trig  { position:absolute; top:9px; width:3px; height:28px;
              background:var(--text); border-radius:2px; transform:translateX(-1px); z-index:3; }
.geo .dot   { position:absolute; top:15px; width:16px; height:16px; border-radius:50%;
              border:3px solid var(--bg); transform:translateX(-50%); z-index:4;
              box-shadow:0 0 0 1px rgba(255,255,255,.25); }
.geo .end   { position:absolute; top:34px; font-family:'IBM Plex Mono',monospace;
              font-size:.7rem !important; font-weight:600; }
.geo .end.l { left:0;  color:var(--bear) !important; }
.geo .end.r { right:0; color:var(--bull) !important; }
.geo .tlab  { position:absolute; top:-4px; font-family:'IBM Plex Mono',monospace;
              font-size:.7rem !important; color:var(--text) !important; font-weight:700;
              transform:translateX(-50%); white-space:nowrap; }
.geo .plab  { position:absolute; top:34px; font-family:'IBM Plex Mono',monospace;
              font-size:.7rem !important; color:var(--dim) !important;
              transform:translateX(-50%); white-space:nowrap; }

.nums { display:flex; gap:2.2rem; margin-top:.9rem; flex-wrap:wrap; }
.num .k{display:block !important;font-size:.66rem !important;text-transform:uppercase;
        letter-spacing:.11em;color:var(--dim) !important;font-weight:700;margin-bottom:.2rem;}
.num .v{display:block !important;font-family:'IBM Plex Mono',monospace;
        font-size:1.15rem !important;font-weight:700;color:var(--text) !important;
        font-variant-numeric:tabular-nums;}
.num .v.bad {color:var(--bear) !important;}
.num .v.good{color:var(--bull) !important;}

.foot { margin-top:.85rem; font-size:.78rem !important; color:var(--faint) !important;
        font-family:'IBM Plex Mono',monospace; }
.foot .hot { color:var(--bear) !important; font-weight:700; }
.warn { margin-top:.6rem; font-size:.82rem !important; color:var(--amber) !important; font-weight:500; }
.stale { background:rgba(255,192,67,.1); border:1px solid rgba(255,192,67,.35);
         border-radius:8px; padding:.8rem 1rem; margin-bottom:1.2rem;
         color:var(--amber) !important; font-size:.88rem; }
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

if levels and all("risk_reward" not in l for l in levels):
    st.markdown(
        '<div class="stale">This scan was written by the old scanner — '
        'R:R is missing and the 2H bars still include the 30-minute stub. '
        'Push the new code and re-run the workflow.</div>',
        unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("#### Reward : Risk")
    rr_min = st.slider(
        "Minimum R:R", 0.0, 5.0, float(min_rr), 0.25,
        help="Trigger→target vs trigger→stop. Below 1.0 the target sits on top "
             "of the trigger — the pattern is real but the trade isn't worth taking.",
    )
    show_unrated = st.checkbox(
        "Include levels with no R:R", value=False,
        help="Levels with no computable target. Old scan files also land here.",
    )
    st.caption(f"Scanner suppresses alerts below R:R {min_rr:g}. "
               f"This slider only filters the board.")

def passes_rr(l) -> bool:
    rr = l.get("risk_reward")
    if rr is None:
        return show_unrated
    return rr >= rr_min


rows = [
    l for l in levels
    if l["setup_tf"] in tf_pick and l["tier"] in tier_pick and l["direction"] in dir_pick
    and abs(l["distance_pct"]) <= max_dist and passes_rr(l)
]

if not rows:
    hidden_by_rr = sum(1 for l in levels if not passes_rr(l))
    msg = "Nothing matches those filters."
    if hidden_by_rr:
        msg += (f" {hidden_by_rr} level(s) are being hidden by the R:R filter — "
                f"lower the minimum, or tick 'Include levels with no R:R' if this "
                f"scan predates the R:R field.")
    st.markdown(f'<div class="card"><span class="setup">{msg}</span></div>',
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
    weak = rr is not None and rr < 1.0

    cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": ""}[tier] + (" weak" if weak else "")
    tag_cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": "armed"}[tier]
    tag_txt = {"TIER2": "Tier 2", "TIER1": "Tier 1", "ARMED": "Armed"}[tier]

    # ---- trade geometry: STOP | trigger | TARGET, to scale ----
    # Position everything on one axis running stop -> target. Works for bear
    # too: there stop is ABOVE and target BELOW, so the axis simply descends
    # in price. The maths is identical, and the picture reads the same way:
    # left is where you're wrong, right is where you're paid.
    stop_p = l.get("invalidation")
    tgt_p = l.get("target")
    trig_p = l["level"]
    px = l["current_price"]
    through = l["distance_pct"] > 0

    geo_html = ""
    if stop_p is not None and tgt_p is not None and (tgt_p - stop_p) != 0:
        def pos(v):
            return max(0.0, min(100.0, (v - stop_p) / (tgt_p - stop_p) * 100))

        t_pos = pos(trig_p)          # trigger's place on the axis
        p_pos = pos(px)              # where price actually is
        dot_col = hue if through else "var(--slate)"

        geo_html = f"""
  <div class="geo">
    <div class="bar">
      <div class="risk" style="width:{t_pos}%"></div>
      <div class="rew"  style="width:{100 - t_pos}%"></div>
    </div>
    <div class="trig" style="left:{t_pos}%"></div>
    <div class="tlab" style="left:{t_pos}%">TRIGGER {trig_p:.2f}</div>
    <div class="dot"  style="left:{p_pos}%;background:{dot_col};"></div>
    <div class="plab" style="left:{p_pos}%">now {px:.2f}</div>
    <div class="end l">STOP {stop_p:.2f}</div>
    <div class="end r">TARGET {tgt_p:.2f}</div>
  </div>"""

    stop_txt = f"{l['invalidation']:.2f}" if l.get("invalidation") is not None else "—"
    tgt_txt = f"{l['target']:.2f}" if l.get("target") is not None else "—"
    rr_txt = f"{rr:.1f}" if rr is not None else "—"
    rr_cls = "bad" if weak else ("good" if (rr is not None and rr >= 2) else "")

    dist_txt = (f"{abs(l['distance_pct']):.2f}% through trigger" if through
                else f"{abs(l['distance_pct']):.2f}% from trigger")
    foot = [dist_txt,
            {"TIER2": "15m closed through", "TIER1": "5m closed through",
             "ARMED": "not triggered"}[tier]]
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
  {geo_html}
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
