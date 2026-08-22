"""
lab_iter4.py -- Split intraday structure breaks into BOS (continuation) vs
CHoCH (change of character / reversal) on 3-min QQQ, 2022→now.

State machine: confirmed 3/3 swing points; breaking a swing high flips
structure bullish, breaking a swing low flips bearish. A break WITH the
prevailing state = BOS; the first break AGAINST it = CHoCH. Structure evolves
all session; entries obey window/gate/max-trades.

Also re-tests the user's live trim model on the best CHoCH config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy_lab as lab
from strategy_lab import fetch_bars, daily_context, stats, START
from lab_iter3 import make_trim_resolver


def strat_structure(m3, ctx, r_mult, gate, kind, win_start="09:45", win_end="15:30"):
    """kind: 'bos' (with-trend break), 'choch' (reversal break), 'all' (mixed)."""
    out = []
    for date, day in m3.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "ema21"]):
            continue
        ema21 = ctx.loc[date, "ema21"]
        day = day.reset_index(drop=True)
        h, l = day["h"].to_numpy(), day["l"].to_numpy()
        n = len(day)
        sw_hi = np.full(n, np.nan)
        sw_lo = np.full(n, np.nan)
        last_hi = last_lo = np.nan
        for i in range(n):
            j = i - 3
            if j >= 3 and h[j] == max(h[j - 3:min(j + 4, n)]):
                last_hi = h[j]
            if j >= 3 and l[j] == min(l[j - 3:min(j + 4, n)]):
                last_lo = l[j]
            sw_hi[i], sw_lo[i] = last_hi, last_lo

        state = 0                     # 0 neutral, +1 bullish, -1 bearish
        broken_hi = broken_lo = np.nan
        taken = 0
        for i, bar in day.iterrows():
            shi, slo = sw_hi[i], sw_lo[i]
            event = None
            if np.isfinite(shi) and bar["c"] > shi and (not np.isfinite(broken_hi) or shi != broken_hi):
                event, broken_hi = "up", shi
            elif np.isfinite(slo) and bar["c"] < slo and (not np.isfinite(broken_lo) or slo != broken_lo):
                event, broken_lo = "dn", slo
            if event is None:
                continue
            if event == "up":
                label = "choch" if state == -1 else "bos"
                state = 1
                side, stop_ref = "L", slo
            else:
                label = "choch" if state == 1 else "bos"
                state = -1
                side, stop_ref = "S", shi
            # ── entry filters (structure state updated regardless) ──
            if kind != "all" and label != kind:
                continue
            if taken >= 3 or not (win_start <= bar["hm"] <= win_end):
                continue
            if gate and ((side == "L") != (bar["c"] > ema21)):
                continue
            rng = bar["h"] - bar["l"]
            stop = stop_ref if np.isfinite(stop_ref) else (
                bar["c"] - 2 * rng if side == "L" else bar["c"] + 2 * rng)
            risk = (bar["c"] - stop) if side == "L" else (stop - bar["c"])
            if not np.isfinite(risk) or risk <= 0:
                continue
            tgt = bar["c"] + r_mult * risk if side == "L" else bar["c"] - r_mult * risk
            res = lab._res()(day, i, side, bar["c"], stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1],
                                ts=bar["ts"], dir=side, label=label,
                                day_open=day.iloc[0]["o"]))
                taken += 1
    return out


if __name__ == "__main__":
    M3 = fetch_bars("QQQ", "3Min", START)
    CTX = daily_context(fetch_bars("QQQ", "1Day", "2021-01-01"))
    
    lab.RESOLVE_HOOK = None
    print("== CHoCH (reversal breaks), full exit ==")
    stats(strat_structure(M3, CTX, 0.5, False, "choch"), "choch raw 0.5R")
    stats(strat_structure(M3, CTX, 0.5, True, "choch"), "choch gated 0.5R")
    stats(strat_structure(M3, CTX, 0.35, True, "choch"), "choch gated 0.35R")
    stats(strat_structure(M3, CTX, 0.35, True, "choch", "09:45", "12:00"), "choch gated .35R AM")
    stats(strat_structure(M3, CTX, 0.5, True, "choch", "09:45", "12:00"), "choch gated 0.5R AM")
    
    print("\n== BOS purified (continuation-only), full exit ==")
    stats(strat_structure(M3, CTX, 0.35, True, "bos", "09:45", "12:00"), "bosPure gated .35R AM")
    stats(strat_structure(M3, CTX, 0.5, True, "bos", "09:45", "12:00"), "bosPure gated 0.5R AM")
    print("   (mixed-version reference: 75.5% / PF 1.19 / +$0.060 @ .35R AM)")
    
    print("\n== best configs under user's live trim model (80% @0.5R, BE, runner) ==")
    lab.RESOLVE_HOOK = make_trim_resolver(0.5, 0.8)
    stats(strat_structure(M3, CTX, 0.5, True, "choch", "09:45", "12:00"), "choch AM trim80/20")
    stats(strat_structure(M3, CTX, 0.5, True, "bos", "09:45", "12:00"), "bosPure AM trim80/20")
    lab.RESOLVE_HOOK = make_trim_resolver(0.5, 0.5)
    stats(strat_structure(M3, CTX, 0.5, True, "choch", "09:45", "12:00"), "choch AM trim50/50")
    stats(strat_structure(M3, CTX, 0.5, True, "bos", "09:45", "12:00"), "bosPure AM trim50/50")
    lab.RESOLVE_HOOK = None
