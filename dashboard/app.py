"""
dashboard/app.py
----------------
The board. Armed levels sorted by how close price is to triggering them.

Design intent: this is a terminal, not a report. The top of the screen is what
is about to happen.

v5 board
========
This reflects the v5 gate stack, not the old R:R model:

  * Levels NOMINATE on 4H / Daily (2H no longer arms).
  * Tier 1 = price is through the level intrabar, the 15m has NOT closed yet
    (a heads-up). The old 5-minute tier is gone.
  * Tier 2 = a 15m bar closed through the level (the entry).
  * The trade plan is a SCALE-OUT: take half at +1R, pull the stop to
    breakeven, run the rest to the first gating rung (`target`). The geometry
    bar draws the +1R scale point, not just stop/target.
  * What actually gates a level is RUNWAY (nearest gating rung >= min_runway R)
    and FTFC (>= min_ftfc of 1H/2H/4H/1D/1W aligned) -- NOT reward:risk. R:R
    was deleted from the scanner; it is not shown as a gate here.

Direction carries the hue -- green/red. Tier carries the luminance: muted slate
when a level is merely loaded, amber the moment it's live through, full
brightness at Tier 2.
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

/* ---- trade geometry bar (scale-out) ----
   Axis runs stop -> runner target, drawn to real price distance. The white
   tick is the trigger; the amber tick is the +1R scale point where you take
   half off and pull the stop to breakeven; the dot is price now. Red block =
   risk (stop->trigger), green block = the runner's room (trigger->target). */
.geo { position:relative; height:38px; margin:1.1rem 0 .2rem; }
.geo .bar  { position:absolute; top:4px; left:0; right:0; height:16px;
             border-radius:4px; overflow:hidden; display:flex; }
.geo .risk { background:rgba(255,107,98,.45); height:100%; }
.geo .rew  { background:rgba(63,224,138,.42); height:100%; }
.geo .trig { position:absolute; top:0; width:3px; height:24px; background:#fff;
             border-radius:2px; transform:translateX(-1px); z-index:3; }
.geo .scale{ position:absolute; top:1px; width:3px; height:22px; background:var(--amber);
             border-radius:2px; transform:translateX(-1px); z-index:3; }
.geo .dot  { position:absolute; top:2px; width:20px; height:20px; border-radius:50%;
             border:3px solid var(--bg); transform:translateX(-50%); z-index:4; }
.geo .cap  { position:absolute; top:24px; font-family:'IBM Plex Mono',monospace;
             font-size:.66rem !important; font-weight:700; letter-spacing:.08em; }
.geo .cap.l { left:0;  color:var(--bear) !important; }
.geo .cap.r { right:0; color:var(--bull) !important; }

.plan { margin:.55rem 0 0; font-size:.82rem !important; color:var(--dim) !important;
        font-family:'IBM Plex Mono',monospace; }
.plan b { color:var(--text) !important; font-weight:700; }
.plan .amber { color:var(--amber) !important; }

.legend { display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap;
          margin:-.4rem 0 1.3rem; font-size:.75rem; color:var(--faint) !important;
          font-family:'IBM Plex Mono',monospace; }
.legend i { display:inline-block; width:11px; height:11px; border-radius:2px;
            margin-right:.35rem; vertical-align:-1px; }

.nums { display:flex; gap:2.0rem; margin-top:.9rem; flex-wrap:wrap; }
.num .k{display:block !important;font-size:.66rem !important;text-transform:uppercase;
        letter-spacing:.11em;color:var(--dim) !important;font-weight:700;margin-bottom:.2rem;}
.num .v{display:block !important;font-family:'IBM Plex Mono',monospace;
        font-size:1.15rem !important;font-weight:700;color:var(--text) !important;
        font-variant-numeric:tabular-nums;}
.num .v.good{color:var(--bull) !important;}
.num .v.amber{color:var(--amber) !important;}

.foot { margin-top:.85rem; font-size:.78rem !important; color:var(--faint) !important;
        font-family:'IBM Plex Mono',monospace; }
.foot .hot { color:var(--bear) !important; font-weight:700; }
.stale { background:rgba(255,192,67,.1); border:1px solid rgba(255,192,67,.35);
         border-radius:8px; padding:.8rem 1rem; margin-bottom:1.2rem;
         color:var(--amber) !important; font-size:.88rem; }
@media (max-width: 640px) {
  [data-testid="stMarkdownContainer"] h1 { font-size:1.8rem; }
  .strip { gap:1.5rem; padding:.8rem 0 1rem; margin-bottom:1.1rem; }
  .stat .v { font-size:1.35rem !important; }
  .stat .k { font-size:.62rem !important; letter-spacing:.09em; }
  .legend { gap:.8rem; font-size:.68rem; margin-bottom:1rem; }
  .card { padding:.95rem 1rem; }
  .sym { font-size:1.25rem !important; }
  .thesis { font-size:.9rem !important; }
  .plan { font-size:.74rem !important; }
  .nums { gap:1.25rem; margin-top:.75rem; }
  .num .v { font-size:1rem !important; }
  .num .k { font-size:.6rem !important; }
  .foot { font-size:.72rem !important; }
}
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
setup_tfs = data.get("setup_timeframes", ["4H", "1D"])
min_runway = float(data.get("min_runway_r", 2.0))
min_ftfc_gate = int(data.get("min_ftfc", 4))
ftfc_tfs = data.get("ftfc_timeframes", ["1H", "2H", "4H", "1D", "1W"])
ftfc_n = len(ftfc_tfs) or 5
gen = str(data.get("generated_at_et", data.get("generated_at_utc", "")))[:16].replace("T", " ")

# A pre-v5 scan has no decision block and no scale_level. Warn rather than
# render half-broken cards.
if levels and all(l.get("decision") is None and "scale_level" not in l for l in levels):
    st.markdown(
        '<div class="stale">This scan predates v5 — no gate decision, runway, '
        'or scale-out plan. Merge v5-gates to main and re-run the workflow to '
        'refresh latest_scan.json.</div>',
        unsafe_allow_html=True)

n1 = sum(1 for l in levels if l["tier"] == "TIER1")
n2 = sum(1 for l in levels if l["tier"] == "TIER2")

st.markdown(f"""
<div class="strip">
  <span class="stat"><span class="k">Armed</span><span class="v">{len(levels)}</span></span>
  <span class="stat"><span class="k">Tier 1 · live</span>
    <span class="v {'amber' if n1 else 'zero'}">{n1}</span></span>
  <span class="stat"><span class="k">Tier 2 · 15m</span>
    <span class="v {'live' if n2 else 'zero'}">{n2}</span></span>
  <span class="stat"><span class="k">Nominate</span>
    <span class="v">{' / '.join(setup_tfs)}</span></span>
  <span class="stat"><span class="k">Last scan</span><span class="v">{gen[-5:]}</span></span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span><i style="background:rgba(255,107,98,.55)"></i>risk (stop → trigger)</span>
  <span><i style="background:rgba(63,224,138,.5)"></i>runner (trigger → target)</span>
  <span><i style="background:#fff;width:3px;border-radius:1px;"></i>trigger</span>
  <span><i style="background:var(--amber);width:3px;border-radius:1px;"></i>+1R scale → breakeven</span>
  <span><i style="background:var(--slate);border-radius:50%"></i>price now</span>
</div>
""", unsafe_allow_html=True)

if not levels:
    st.markdown(f'<div class="card"><span class="setup">No inside bars on '
                f'{" or ".join(setup_tfs)}, and no live Failed-2s. Nothing is loaded.</span></div>',
                unsafe_allow_html=True)
    st.caption(f"Scanner ran {gen} ET across {len(data.get('tickers', []))} tickers.")
    st.stop()

with st.sidebar:
    st.markdown("### Filter")
    # Options come from the scan's own setup_timeframes, so the board can never
    # silently hide a timeframe the scanner actually nominates (the v4 board
    # hardcoded 2H/4H and dropped every Daily level).
    tf_pick = st.multiselect("Setup timeframe", setup_tfs, default=setup_tfs)
    tier_pick = st.multiselect("Tier", ["ARMED", "TIER1", "TIER2"],
                               default=["ARMED", "TIER1", "TIER2"])
    dir_pick = st.multiselect("Direction", ["bull", "bear"], default=["bull", "bear"])
    max_dist = st.slider("Max distance from trigger (%)", 0.1, 5.0, 1.5, 0.1)

    st.markdown("---")
    st.markdown("#### Quality")
    runway_min = st.slider(
        "Min runway (R)", 0.0, 8.0, 0.0, 0.5,
        help="R to the furthest untouched gating rung — how far the trade can "
             "run before higher-timeframe magnitude is spent.",
    )
    ftfc_min = st.slider(
        "Min FTFC aligned", 0, ftfc_n, 0, 1,
        help=f"How many of the {ftfc_n} watched frames "
             f"({', '.join(ftfc_tfs)}) agree with the trade.",
    )
    st.caption(
        f"Every level shown already cleared the scanner's gates: runway ≥ "
        f"{min_runway:g}R, FTFC ≥ {min_ftfc_gate}/{ftfc_n}, hard 4H+1D "
        f"continuity. These sliders only filter the board."
    )


def dec_of(l) -> dict:
    return l.get("decision") or {}


def passes_quality(l) -> bool:
    d = dec_of(l)
    rw = d.get("runway_r")
    fa = d.get("ftfc_aligned")
    if runway_min > 0 and (rw is None or rw < runway_min):
        return False
    if ftfc_min > 0 and (fa is None or fa < ftfc_min):
        return False
    return True


rows = [
    l for l in levels
    if l["setup_tf"] in tf_pick and l["tier"] in tier_pick and l["direction"] in dir_pick
    and abs(l["distance_pct"]) <= max_dist and passes_quality(l)
]

if not rows:
    hidden = sum(1 for l in levels if not passes_quality(l))
    msg = "Nothing matches those filters."
    if hidden:
        msg += (f" {hidden} level(s) are hidden by the runway/FTFC sliders — "
                f"lower them to see more.")
    st.markdown(f'<div class="card"><span class="setup">{msg}</span></div>',
                unsafe_allow_html=True)
    st.stop()

RANK = {"TIER2": 2, "TIER1": 1, "ARMED": 0}
# Score first (the scanner's own ranking), then tier, then proximity.
rows.sort(key=lambda l: (-(dec_of(l).get("score") or 0), -RANK[l["tier"]], abs(l["distance_pct"])))

for l in rows:
    d = l["direction"]
    tier = l["tier"]
    dec = dec_of(l)
    hot = l["family"] == "f2" and tier == "TIER1"

    cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": ""}[tier]
    tag_cls = {"TIER2": f"t2{d}", "TIER1": "t1", "ARMED": "armed"}[tier]
    tag_txt = {"TIER2": "Tier 2", "TIER1": "Tier 1", "ARMED": "Armed"}[tier]

    # ---- geometry: STOP | trigger | +1R scale | TARGET (runner), to scale ----
    stop_p = l.get("invalidation")
    tgt_p = l.get("target")
    scale_p = l.get("scale_level")
    trig_p = l["level"]
    px = l["current_price"]
    through = l["distance_pct"] > 0

    geo_html = ""
    if stop_p is not None and tgt_p is not None and (tgt_p - stop_p) != 0:
        def pos(v):
            return max(0.0, min(100.0, (v - stop_p) / (tgt_p - stop_p) * 100))

        t_pos = pos(trig_p)
        p_pos = pos(px)
        dot_col = ("var(--bull)" if d == "bull" else "var(--bear)") if through else "var(--slate)"
        scale_html = (f'<div class="scale" style="left:{pos(scale_p)}%"></div>'
                      if scale_p is not None else "")
        geo_html = f"""
  <div class="geo">
    <div class="bar">
      <div class="risk" style="width:{t_pos}%"></div>
      <div class="rew"  style="width:{100 - t_pos}%"></div>
    </div>
    <div class="trig" style="left:{t_pos}%"></div>
    {scale_html}
    <div class="dot"  style="left:{p_pos}%;background:{dot_col};"></div>
    <div class="cap l">STOP</div>
    <div class="cap r">TARGET</div>
  </div>"""

    # ---- scale-out plan line ----
    if scale_p is not None and tgt_p is not None:
        runner_r = l.get("runner_r")
        rr_txt = f" · {runner_r:.1f}R" if runner_r is not None else ""
        plan_html = (f'<div class="plan">half at <b class="amber">+1R {scale_p:.2f}</b>'
                     f' → stop to breakeven → runner <b>{tgt_p:.2f}</b>{rr_txt}</div>')
    else:
        plan_html = ""

    # ---- numbers ----
    stop_txt = f"{stop_p:.2f}" if stop_p is not None else "—"
    scale_txt = f"{scale_p:.2f}" if scale_p is not None else "—"
    tgt_txt = f"{tgt_p:.2f}" if tgt_p is not None else "—"
    runway = dec.get("runway_r")
    runway_txt = f"{runway:.1f}R" if runway is not None else "—"
    fa = dec.get("ftfc_aligned")
    ftfc_map = dec.get("ftfc") or {}
    ftfc_total = len(ftfc_map) or ftfc_n
    ftfc_txt = f"{fa}/{ftfc_total}" if fa is not None else l.get("continuity", "—")
    ftfc_cls = "good" if (fa is not None and fa >= min_ftfc_gate) else ""
    score = dec.get("score")
    score_txt = str(score) if score is not None else "—"

    dist_txt = (f"{abs(l['distance_pct']):.2f}% through trigger" if through
                else f"{abs(l['distance_pct']):.2f}% from trigger")
    foot = [dist_txt,
            {"TIER2": "15m closed through",
             "TIER1": "live — through, 15m not closed yet",
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

    st.markdown(f"""
<div class="card {cls}">
  <div class="row">
    <span class="sym {d}">{l['symbol']}</span>
    <span class="tag {tag_cls}">{tag_txt}</span>
    <span class="setup">{l['setup_tf']} · {l['pattern']}</span>
  </div>
  <div class="thesis">Needs a close <b>{l['trigger_side']} {l['level']:.2f}</b> · now <b>{l['current_price']:.2f}</b></div>
  {geo_html}
  {plan_html}
  <div class="nums">
    <span class="num"><span class="k">Trigger</span><span class="v">{l['level']:.2f}</span></span>
    <span class="num"><span class="k">Stop</span><span class="v">{stop_txt}</span></span>
    <span class="num"><span class="k">+1R scale</span><span class="v amber">{scale_txt}</span></span>
    <span class="num"><span class="k">Runner</span><span class="v">{tgt_txt}</span></span>
    <span class="num"><span class="k">Runway</span><span class="v good">{runway_txt}</span></span>
    <span class="num"><span class="k">FTFC</span><span class="v {ftfc_cls}">{ftfc_txt}</span></span>
    <span class="num"><span class="k">Score</span><span class="v">{score_txt}</span></span>
  </div>
  <div class="foot">{' · '.join(foot)}{hot_html}</div>
</div>""", unsafe_allow_html=True)

st.divider()
with st.expander("Raw table"):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
