"""
lab_iter5.py -- The full strategy suite re-run on the 15-MIN chart (QQQ 2022→now).

Adaptations for 15m bars: opening range = the single 09:30 bar; EOD exit at
the 15:45 bar (last of the session); everything else identical to the
3m/5m runs so numbers are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy_lab as lab
from strategy_lab import fetch_bars, daily_context, stats, START
from lab_iter4 import strat_structure

lab.EOD_HM = "15:45"
lab.RESOLVE_HOOK = None

M15 = fetch_bars("QQQ", "15Min", START)
CTX = daily_context(fetch_bars("QQQ", "1Day", "2021-01-01"))
print(f"15m bars: {len(M15)}  days: {M15['date'].nunique()}\n")


def strat_orb_retest_15m(m15, ctx, r_mult=0.5):
    """Champion ORB retest on 15m execution bars (OR = the 09:30 bar)."""
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
        up_break = dn_break = None
        long_done = short_done = False
        taken = 0
        for i, bar in day.iterrows():
            if bar["hm"] <= "09:30" or taken >= 2:
                continue
            in_win = "09:45" <= bar["hm"] <= "10:45"
            if up_break is None and bar["c"] > orbH:
                up_break = i
                continue
            if dn_break is None and bar["c"] < orbL:
                dn_break = i
                continue
            side = None
            if (not long_done and up_break is not None and i > up_break
                    and bar["l"] <= orbH and bar["c"] > orbH and in_win
                    and bar["c"] > pdh and bar["c"] > ema21):
                side, long_done = "L", True
            elif (not short_done and dn_break is not None and i > dn_break
                    and bar["h"] >= orbL and bar["c"] < orbL and in_win
                    and bar["c"] < pdl and bar["c"] < ema21):
                side, short_done = "S", True
            if side is None:
                continue
            entry, stop = bar["c"], orbM
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            tgt = entry + r_mult * risk if side == "L" else entry - r_mult * risk
            res = lab._res()(day, i, side, entry, stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1]))
                taken += 1
    return out


print("== ORB family on 15m ==")
stats(strat_orb_retest_15m(M15, CTX, 0.5), "ORB retest 15m 0.5R")
stats(lab.strat_orb_9ema(M15, CTX, 0.35, True, "orbmid", min_or_bars=1), "orb9ema 15m gated .35R")
stats(lab.strat_orb_9ema(M15, CTX, 0.5, True, "orbmid", min_or_bars=1), "orb9ema 15m gated 0.5R")

print("\n== Structure breaks on 15m ==")
stats(strat_structure(M15, CTX, 0.5, True, "all"), "bos+choch 15m gated .5R")
stats(strat_structure(M15, CTX, 0.35, True, "all"), "bos+choch 15m gated .35")
stats(strat_structure(M15, CTX, 0.5, True, "bos"), "bosPure 15m gated 0.5R")
stats(strat_structure(M15, CTX, 0.5, True, "choch"), "choch 15m gated 0.5R")

print("\n== Mean reversion on 15m ==")
for kind in ("rsi2", "bb", "pctb", "zscore"):
    stats(lab.strat_meanrev(M15, CTX, kind, 0.5, True), f"{kind} 15m gated 0.5R")

print("\n== Pivot + cross on 15m ==")
stats(lab.strat_pivot(M15, CTX, 0.5), "pivot 15m 0.5R")
stats(lab.strat_pivot(M15, CTX, 1.0), "pivot 15m 1.0R")
stats(lab.strat_cross(M15), "ema50/200 cross 15m")
