"""
lab_split.py -- Hold-out validation: TRAIN = 2022-01-01..2024-12-31,
TEST = 2025-01-01..now (untouched by any config choice if train-side
selection confirms the same picks).

Part 1: would TRAIN alone have selected our configs? (gate, R, window, stop)
Part 2: frozen champion configs, train vs test side by side.
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


M5 = fetch_bars("QQQ", "5Min", START)
M3 = fetch_bars("QQQ", "3Min", START)
M15 = fetch_bars("QQQ", "15Min", START)
CTX = daily_context(fetch_bars("QQQ", "1Day", "2021-01-01"))
lab.RESOLVE_HOOK = None

print("═══ Part 1: config selection using TRAIN data only ═══")
print("-- BOS: gate on/off (0.35R AM) --")
for gate in (False, True):
    t = strat_structure(M3, CTX, 0.35, gate, "all", "09:45", "12:00")
    print(f"  gate={gate!s:5s}  {seg_stats(t, hi=SPLIT)}")
print("-- BOS: window (gated 0.35R) --")
for ws, we, lbl in (("09:45", "12:00", "AM"), ("09:45", "15:30", "all-day"), ("13:00", "15:30", "PM")):
    t = strat_structure(M3, CTX, 0.35, True, "all", ws, we)
    print(f"  {lbl:8s}  {seg_stats(t, hi=SPLIT)}")
print("-- BOS: target R (gated AM) --")
for r in (0.25, 0.35, 0.5, 1.0):
    t = strat_structure(M3, CTX, r, True, "all", "09:45", "12:00")
    print(f"  {r}R  {seg_stats(t, hi=SPLIT)}")
print("-- orb9ema (15m): stop mode (gated .35R) --")
lab.EOD_HM = "15:45"
for sm in ("rej", "orbmid"):
    t = lab.strat_orb_9ema(M15, CTX, 0.35, True, sm, min_or_bars=1)
    print(f"  stop={sm:7s}  {seg_stats(t, hi=SPLIT)}")
lab.EOD_HM = "15:55"

print("\n═══ Part 2: frozen champion configs -- TRAIN vs TEST ═══")
both(strat_orb_retest(M5, CTX), "ORB retest gated 0.5R (5m)")
both(strat_structure(M3, CTX, 0.35, True, "all", "09:45", "12:00"), "BOS+CHoCH gated .35R AM")
both(strat_structure(M3, CTX, 0.5, True, "all", "09:45", "12:00"), "BOS+CHoCH gated 0.5R AM")
both(strat_structure(M3, CTX, 0.5, True, "choch", "09:45", "15:30"), "CHoCH gated 0.5R all-day")
lab.EOD_HM = "15:45"
both(lab.strat_orb_9ema(M15, CTX, 0.35, True, "orbmid", min_or_bars=1), "orb9ema 15m gated .35R")
lab.EOD_HM = "15:55"
both(lab.strat_cross(M5), "ema50/200 cross (5m)")
