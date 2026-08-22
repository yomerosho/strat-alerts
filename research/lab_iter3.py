"""
lab_iter3.py -- Test the user's live trim model across validated strategies.

Trim model (options-mapped to share space):
  entry -> T1 at +t1_mult*R (40-50% premium gain ~ 0.4-0.5R at our stop widths)
  bank trim_pct there, move stop to ENTRY (breakeven), runner rides to 15:55.
  Once T1 hits, the trade cannot lose: win rate == P(T1 before stop).

Runs: ORB retest (champion), BOS 3m AM gated, ORB->9EMA (mid-stop), each under
full-exit vs 80/20 trim vs 50/50 trim; then the combined portfolio.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import strategy_lab as lab
from strategy_lab import fetch_bars, daily_context, stats, START


# ── Champion ORB retest generator (port of the validated config) ─────────────

def strat_orb_retest(m5: pd.DataFrame, ctx: pd.DataFrame) -> list[dict]:
    out = []
    for date, day in m5.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "pdh"]):
            continue
        pdh, pdl, ema21 = ctx.loc[date, ["pdh", "pdl", "ema21"]]
        day = day.reset_index(drop=True)
        orb = day[day["hm"].isin(["09:30", "09:35", "09:40"])]
        if len(orb) < 3:
            continue
        orbH, orbL = orb["h"].max(), orb["l"].min()
        orbM = (orbH + orbL) / 2
        up_break = dn_break = None
        long_done = short_done = False
        taken = 0
        for i, bar in day.iterrows():
            if bar["hm"] <= "09:40" or taken >= 2:
                continue
            in_win = "09:45" <= bar["hm"] <= "10:55"
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
            tgt = entry + 0.5 * risk if side == "L" else entry - 0.5 * risk
            res = lab._res()(day, i, side, entry, stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1]))
                taken += 1
    return out


# ── Trim exit model ───────────────────────────────────────────────────────────

def make_trim_resolver(t1_mult: float, trim_pct: float):
    """resolve()-compatible: bank trim_pct at t1_mult*R, BE stop, runner to EOD."""
    def resolve_trim(day_bars, i, side, entry, stop, tgt_unused):
        risk = abs(entry - stop)
        t1 = entry + t1_mult * risk if side == "L" else entry - t1_mult * risk
        banked = None
        for j in range(i + 1, len(day_bars)):
            fb = day_bars.iloc[j]
            eod = fb["hm"] >= "15:55"
            if banked is None:
                if eod:
                    pnl = (fb["c"] - entry) if side == "L" else (entry - fb["c"])
                    return (int(pnl > 0), pnl)
                hit_t = fb["h"] >= t1 if side == "L" else fb["l"] <= t1
                hit_s = fb["l"] <= stop if side == "L" else fb["h"] >= stop
                if hit_s:                                   # conservative
                    return (0, -risk)
                if hit_t:
                    banked = trim_pct * t1_mult * risk
                    # same-bar BE-touch after T1: runner scratched (conservative)
                    be_touch = fb["l"] <= entry if side == "L" else fb["h"] >= entry
                    if be_touch:
                        return (1, banked)
            else:
                be_touch = fb["l"] <= entry if side == "L" else fb["h"] >= entry
                if be_touch and not eod:
                    return (1, banked)
                if eod:
                    run = (fb["c"] - entry) if side == "L" else (entry - fb["c"])
                    return (1, banked + (1 - trim_pct) * max(run, 0.0))
        return None
    return resolve_trim


# ── Runs ──────────────────────────────────────────────────────────────────────

def run_all(label: str):
    print(f"\n════ exit model: {label} ════")
    rows = []
    rows.append(stats(strat_orb_retest(M5, CTX), "ORB retest gated"))
    rows.append(stats(lab.strat_bos(M3, CTX, 0.35, True, "09:45", "12:00"), "BOS 3m AM gated"))
    rows.append(stats(lab.strat_orb_9ema(M3, CTX, 0.35, True, "orbmid"), "ORB->9EMA gated mid"))
    combo = []
    lab.RESOLVE_HOOK = lab.RESOLVE_HOOK  # unchanged
    for fn in (lambda: strat_orb_retest(M5, CTX),
               lambda: lab.strat_bos(M3, CTX, 0.35, True, "09:45", "12:00"),
               lambda: lab.strat_orb_9ema(M3, CTX, 0.35, True, "orbmid")):
        combo.extend(fn())
    stats(combo, "PORTFOLIO (all 3)")
    dfc = pd.DataFrame(combo)
    per_day = dfc.groupby("date").size()
    all_days = np.busday_count(dfc["date"].min(), dfc["date"].max())
    print(f"  portfolio days with >=1 signal: {len(per_day)}/{all_days} "
          f"({len(per_day)/all_days:.0%}), avg {per_day.mean():.1f} sig/active-day")


if __name__ == "__main__":
    M5 = fetch_bars("QQQ", "5Min", START)
    M3 = fetch_bars("QQQ", "3Min", START)
    CTX = daily_context(fetch_bars("QQQ", "1Day", "2021-01-01"))
    
    lab.RESOLVE_HOOK = None
    run_all("FULL EXIT at target (baseline)")
    
    for t1, trim in ((0.4, 0.8), (0.5, 0.8), (0.5, 0.5)):
        lab.RESOLVE_HOOK = make_trim_resolver(t1, trim)
        run_all(f"TRIM {int(trim*100)}% at {t1}R, BE stop, runner to EOD")
    lab.RESOLVE_HOOK = None
