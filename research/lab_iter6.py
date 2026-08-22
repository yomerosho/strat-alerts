"""
lab_iter6.py -- 15-min ORB entry-style shootout: intrabar BREAK vs candle CLOSE.

BREAK entry: buy-stop at the ORB high (sell-stop at ORB low) -- fills the moment
a 15m candle crosses the level. Entry price = the level itself.
Same-bar resolution is conservative: target-touch after an upward cross is
provable (win); any stop-zone touch on the entry bar counts as a loss.

CLOSE entry: wait for the first 15m candle to CLOSE beyond the level; enter at
that close (worse price, fewer fakeouts).

Both: stop = ORB midpoint, targets 0.35R/0.5R, first break per side only,
entries 09:45-10:45 bars, optional PDH/PDL + daily-EMA21 gate, flat 15:45.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy_lab as lab
from strategy_lab import fetch_bars, daily_context, stats, START

lab.EOD_HM = "15:45"
lab.RESOLVE_HOOK = None


def strat_orb15_entry(m15, ctx, style, r_mult, gate):
    """style: 'break' (stop order at level) | 'close' (confirmed close)."""
    out = []
    for date, day in m15.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "pdh"]):
            continue
        pdh, pdl, ema21 = ctx.loc[date, ["pdh", "pdl", "ema21"]]
        day = day.reset_index(drop=True)
        orb = day[day["hm"] == "09:30"]
        if orb.empty:
            continue
        orbH, orbL = orb.iloc[0]["h"], orb.iloc[0]["l"]
        orbM = (orbH + orbL) / 2
        done_l = done_s = False
        for i, bar in day.iterrows():
            if bar["hm"] < "09:45" or bar["hm"] > "10:45" or (done_l and done_s):
                continue
            for side in ("L", "S"):
                if side == "L" and done_l or side == "S" and done_s:
                    continue
                lvl = orbH if side == "L" else orbL
                crossed = bar["h"] > lvl if side == "L" else bar["l"] < lvl
                closed = bar["c"] > lvl if side == "L" else bar["c"] < lvl
                if style == "break":
                    if not crossed:
                        continue
                    entry = lvl
                else:
                    if not closed:
                        continue
                    entry = bar["c"]
                # gate uses info known at entry time
                ref = entry
                if gate:
                    ok = (ref > pdh and ref > ema21) if side == "L" else (ref < pdl and ref < ema21)
                    if not ok:
                        if side == "L":
                            done_l = True          # first break consumed either way
                        else:
                            done_s = True
                        continue
                stop = orbM
                risk = abs(entry - stop)
                if risk <= 0:
                    if side == "L":
                        done_l = True
                    else:
                        done_s = True
                    continue
                tgt = entry + r_mult * risk if side == "L" else entry - r_mult * risk
                res = None
                if style == "break":
                    # entry bar itself, conservative
                    hit_t = bar["h"] >= tgt if side == "L" else bar["l"] <= tgt
                    hit_s = bar["l"] <= stop if side == "L" else bar["h"] >= stop
                    if hit_s:
                        res = (0, -risk)
                    elif hit_t:
                        res = (1, r_mult * risk)
                if res is None:
                    res = lab.resolve(day, i, side, entry, stop, tgt)
                if res:
                    out.append(dict(date=date, win=res[0], pnl=res[1]))
                if side == "L":
                    done_l = True
                else:
                    done_s = True
    return out


M15 = fetch_bars("QQQ", "15Min", START)
CTX = daily_context(fetch_bars("QQQ", "1Day", "2021-01-01"))

for gate in (False, True):
    g = "gated" if gate else "raw"
    print(f"== {g} ==")
    for style in ("break", "close"):
        for r in (0.35, 0.5):
            stats(strat_orb15_entry(M15, CTX, style, r, gate), f"orb15 {style} {g} {r}R")
    print()
