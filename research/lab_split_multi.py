"""
lab_split_multi.py -- TRAIN(2022-24)/TEST(2025→now) split for SPY and IWM,
frozen QQQ-derived configs (cross-instrument out-of-sample).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import strategy_lab as lab
from strategy_lab import fetch_bars, daily_context, START
from lab_iter3 import strat_orb_retest
from lab_iter4 import strat_structure

SPLIT = dt.date(2025, 1, 1)


def seg_stats(trades, lo=None, hi=None):
    df = pd.DataFrame(trades)
    if df.empty:
        return "n=0"
    if lo:
        df = df[df["date"] >= lo]
    if hi:
        df = df[df["date"] < hi]
    if df.empty:
        return "n=0"
    wins, losses = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
    return (f"n={len(df):4d} win={df['win'].mean():.1%} PF={pf:4.2f} "
            f"exp={df['pnl'].mean() - lab.COST:+.3f}")


def both(trades, name):
    print(f"{name:26s} TRAIN {seg_stats(trades, hi=SPLIT)}   |  TEST {seg_stats(trades, lo=SPLIT)}")


for SYM in ("SPY", "IWM"):
    print(f"\n════════════ {SYM} ════════════")
    m5 = fetch_bars(SYM, "5Min", START)
    m3 = fetch_bars(SYM, "3Min", START)
    m15 = fetch_bars(SYM, "15Min", START)
    ctx = daily_context(fetch_bars(SYM, "1Day", "2021-01-01"))
    lab.RESOLVE_HOOK = None
    lab.EOD_HM = "15:55"

    print("-- train-side gate check (BOS .35R AM) --")
    for gate in (False, True):
        t = strat_structure(m3, ctx, 0.35, gate, "all", "09:45", "12:00")
        print(f"  gate={gate!s:5s}  {seg_stats(t, hi=SPLIT)}")

    print("-- frozen QQQ configs: TRAIN vs TEST --")
    both(strat_orb_retest(m5, ctx), "ORB retest gated 0.5R (5m)")
    both(strat_structure(m3, ctx, 0.35, True, "all", "09:45", "12:00"), "BOS+CHoCH gated .35R AM")
    both(strat_structure(m3, ctx, 0.5, True, "all", "09:45", "12:00"), "BOS+CHoCH gated 0.5R AM")
    lab.EOD_HM = "15:45"
    both(lab.strat_orb_9ema(m15, ctx, 0.35, True, "orbmid", min_or_bars=1), "orb9ema 15m gated .35R")
    lab.EOD_HM = "15:55"
    both(lab.strat_cross(m5), "ema50/200 cross (5m)")
