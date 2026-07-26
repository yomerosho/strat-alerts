"""
f2d_tracker.py
--------------
Forward-tracking and alerting for the F2D-Deep strategy.

WHY THIS EXISTS
===============
The strategy was selected from many in-sample configurations (see
research/strat_patterns/F2_REPORT.md). One out-of-sample pass looked good
(66% win, +0.22R, PF 1.72), but the only test that cannot be gamed is the one
where the rules are frozen BEFORE the data exists. This module is that test:
every signal is logged the day it fires, resolved by the same conservative
fill rules as the backtest, and the running forward record is printed on every
alert. If the forward numbers rot, the strategy is dead and the log will say so.

THE FROZEN RULES (do not tune these; that would defeat the point)
=================================================================
Universe: SPY, QQQ, IWM. Long only -- the bearish mirror failed OOS.
Signal day D (completed daily bar):
    - 2D against day P = D-1:  low < P.low  AND  high <= P.high
    - undercut depth (P.low - D.low) >= 0.35 x ATR14 (ATR through P)
    - closes green (close > open) in the top 40% of its range
Entry: next session at the open. Skip if the open gaps at/beyond the runner
    target or at/below the stop.
Stop: D.low. Scale-out: half at +1R, stop to breakeven; runner targets P.high.
Time stop: close of the 2nd session after entry.
Fills are conservative: any 5m bar touching both stop and target counts as
STOP; a same-bar runner fill is ignored when breakeven was also touched.

HOW THE JOBS SHARE THE WORK
===========================
The premarket workflow (09:00 ET, contents: read) runs `--alert-only`: it can
send the morning BUY plan but cannot persist anything. The scanner workflow
(every 5m in RTH, contents: write) runs the default mode: it records signals,
resolves finished windows, sends resolution alerts, and commits the state file.
Double-alerts are prevented structurally: a plan alert is only sent while the
entry session has NOT yet opened, and the state-writing runs all happen after
09:30, so they record the signal without re-alerting it. Resolution alerts are
sent only by state-writing runs, which persist the flag that dedups them.

A day's own bar is only trusted once 16:05 ET has passed, so a partially
settled close can never produce a signal.

USAGE
=====
    python f2d_tracker.py                # update state + send due alerts
    python f2d_tracker.py --alert-only   # send due plan alerts, write nothing
    python f2d_tracker.py --dry-run      # print everything, send/write nothing

State lives in f2d_forward.json, committed by the scanner workflow the same
way scanner_state.db is. Signals are keyed by (symbol, signal_date), so every
mode is idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alerting import AlertManager
from bars import ET, BarProvider
from config import CONFIG

logger = logging.getLogger("strat_scanner.f2d")

SYMBOLS = ("SPY", "QQQ", "IWM")
DEPTH_MIN = 0.35
CLOSE_POS_MIN = 0.60
ATR_LEN = 14
SCAN_BACK = 12          # completed sessions to (re)check for signals
SETTLE = pd.Timedelta(hours=16, minutes=5)  # trust today's bar after 16:05 ET
STATE_FILE = Path(__file__).parent / "f2d_forward.json"


# ------------------------------------------------------------- state --------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"strategy": "F2D-Deep", "frozen": "2026-07-25", "signals": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def forward_record(state: dict) -> str:
    done = [s for s in state["signals"]
            if s["status"] == "resolved" and s["origin"] == "live"
            and s["outcome"] != "skipped_gap"]
    if not done:
        return "Forward record: no live resolutions yet."
    rs = [s["r"] for s in done]
    wins = sum(1 for r in rs if r > 0)
    return (f"Forward record (live): {len(done)} resolved, "
            f"{wins / len(done) * 100:.0f}% win, {np.mean(rs):+.2f}R avg")


# ----------------------------------------------------------- detection ------

def completed_daily(frames: dict, now: pd.Timestamp) -> pd.DataFrame | None:
    """The 1D frame restricted to bars whose session has settled."""
    d = frames.get("1D")
    if d is None or len(d) < ATR_LEN + 3:
        return None
    d = d.copy()
    d.index = d.index.tz_convert(ET).normalize()
    if now < d.index[-1] + SETTLE:
        d = d.iloc[:-1]
    if len(d) < ATR_LEN + 3:
        return None
    tr = np.maximum(
        d["high"] - d["low"],
        np.maximum((d["high"] - d["close"].shift()).abs(),
                   (d["low"] - d["close"].shift()).abs()),
    )
    d["atr"] = tr.rolling(ATR_LEN).mean()
    return d


def detect_signals(symbol: str, d: pd.DataFrame) -> list[dict]:
    """F2D-Deep signals among the last SCAN_BACK completed daily bars."""
    out = []
    for i in range(max(1, len(d) - SCAN_BACK), len(d)):
        p, x = d.iloc[i - 1], d.iloc[i]
        atr = d["atr"].iloc[i - 1]
        rng = x.high - x.low
        if not np.isfinite(atr) or atr <= 0 or rng <= 0:
            continue
        is_2d = x.low < p.low and x.high <= p.high
        depth = (p.low - x.low) / atr
        strong_green = x.close > x.open and (x.close - x.low) / rng >= CLOSE_POS_MIN
        if is_2d and strong_green and depth >= DEPTH_MIN:
            out.append(dict(
                symbol=symbol,
                signal_date=str(d.index[i].date()),
                depth=round(float(depth), 2),
                stop=round(float(x.low), 2),
                target=round(float(p.high), 2),
                status="signaled",
                origin="live",
                alerted=False,
                result_alerted=False,
            ))
    return out


def entry_not_yet_open(sig: dict, now: pd.Timestamp) -> bool:
    """True while the signal's entry session (next trading day) hasn't opened.

    Uses a weekday calendar; a holiday directly after a signal can suppress the
    evening/weekend alert, but the premarket run on the true entry day still
    fires (it runs at 09:00, before any open).
    """
    sd = pd.Timestamp(sig["signal_date"], tz=ET)
    if now.normalize() == sd:
        return True  # evening of the signal day
    nxt = sd + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return now < nxt + pd.Timedelta(hours=9, minutes=30)


# ----------------------------------------------------------- resolution -----

def resolve(sig: dict, d: pd.DataFrame, df5: pd.DataFrame) -> bool:
    """Try to advance a signal's status in place. True if it changed."""
    sd = pd.Timestamp(sig["signal_date"], tz=ET)
    after = d[d.index > sd]
    if after.empty:
        return False  # entry session hasn't completed yet

    entry_day = after.index[0]
    entry = float(after["open"].iloc[0])
    stop, target = sig["stop"], sig["target"]

    if entry <= stop or entry >= target:
        sig.update(status="resolved", entry=round(entry, 2),
                   entry_date=str(entry_day.date()), outcome="skipped_gap", r=0.0)
        return True

    if len(after) < 3:
        # window still open. Early stops/targets could be resolved sooner, but
        # resolving only complete windows keeps this replay byte-identical to
        # the backtest's rules.
        if sig["status"] != "entered":
            sig.update(status="entered", entry=round(entry, 2),
                       entry_date=str(entry_day.date()))
            return True
        return False

    exit_day = after.index[2]
    w = df5.loc[entry_day + pd.Timedelta(hours=9, minutes=30):
                exit_day + pd.Timedelta(hours=16)]
    if w.empty or w.index[-1].normalize() < exit_day:
        return False  # 5m tape doesn't cover the window yet

    risk = entry - stop
    one_r = entry + risk
    half_done, stop_cur = False, stop
    outcome, r = None, None
    for _, bar in w.iterrows():
        h, l = bar.high, bar.low
        if not half_done:
            if l <= stop_cur:
                outcome, r = "stop", -1.0
                break
            if h >= one_r:
                half_done, stop_cur = True, entry
                if h >= target and not l <= entry:
                    outcome, r = "target", 0.5 + 0.5 * (target - entry) / risk
                    break
            continue
        if l <= stop_cur:
            outcome, r = "breakeven", 0.5
            break
        if h >= target:
            outcome, r = "target", 0.5 + 0.5 * (target - entry) / risk
            break
    if outcome is None:
        close = float(d.loc[exit_day, "close"])
        r_open = (close - entry) / risk
        outcome = "time_half" if half_done else "time"
        r = 0.5 + 0.5 * r_open if half_done else r_open

    sig.update(status="resolved", outcome=outcome, r=round(float(r), 3))
    return True


# ------------------------------------------------------------- alerts -------

def signal_message(sig: dict, state: dict) -> str:
    return (
        f"🟩 F2D-DEEP — {sig['symbol']}\n"
        f"Signal {sig['signal_date']}: deep failed 2D "
        f"({sig['depth']:.2f}×ATR undercut, strong green close)\n"
        f"Plan — BUY next open:\n"
        f"  stop {sig['stop']} (signal low)\n"
        f"  half off at +1R, stop → breakeven\n"
        f"  runner target {sig['target']} (prior high)\n"
        f"  time stop: close of 2nd day after entry\n"
        f"Skip if the open gaps ≤ stop or ≥ target.\n"
        f"{forward_record(state)}"
    )


def result_message(sig: dict, state: dict) -> str:
    tag = {"target": "🎯", "stop": "🛑", "breakeven": "➖",
           "time": "⏱", "time_half": "⏱", "skipped_gap": "🚫"}.get(sig["outcome"], "")
    return (
        f"{tag} F2D-DEEP result — {sig['symbol']} "
        f"(signal {sig['signal_date']}, entry {sig.get('entry_date')}): "
        f"{sig['outcome']} {sig.get('r', 0):+.2f}R\n"
        f"{forward_record(state)}"
    )


# --------------------------------------------------------------- run --------

async def run(update_state: bool, send_alerts: bool) -> None:
    provider = BarProvider(CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key,
                           CONFIG.alpaca_data_feed)
    alerts = AlertManager(CONFIG.telegram_bot_token, CONFIG.telegram_chat_id)
    now = pd.Timestamp.now(tz=ET)
    first_run = not STATE_FILE.exists()
    state = load_state()
    known = {(s["symbol"], s["signal_date"]): s for s in state["signals"]}
    outbox: list[str] = []

    for sym in SYMBOLS:
        frames = provider.fetch(sym, lookback_days=30)
        d = completed_daily(frames, now) if frames else None
        df5 = frames.get("5Min") if frames else None
        if d is None or df5 is None:
            logger.warning("%s: no data; skipping this cycle", sym)
            continue

        # -- new signals
        for sig in detect_signals(sym, d):
            if (sig["symbol"], sig["signal_date"]) in known:
                continue
            if first_run and not entry_not_yet_open(sig, now):
                sig["origin"] = "backfill"  # seed the log, never alert it
            state["signals"].append(sig)
            known[(sig["symbol"], sig["signal_date"])] = sig
            logger.info("new signal: %s %s depth %.2f (%s)",
                        sig["symbol"], sig["signal_date"], sig["depth"],
                        sig["origin"])

        # -- plan alerts: only while the entry session hasn't opened
        for sig in state["signals"]:
            if (sig["symbol"] == sym and sig["origin"] == "live"
                    and sig["status"] == "signaled" and not sig["alerted"]
                    and entry_not_yet_open(sig, now)):
                outbox.append(signal_message(sig, state))
                if update_state:
                    sig["alerted"] = True

        # -- resolution (results alerted only by state-writing runs: the flag
        #    that dedups them must be persistable)
        for sig in state["signals"]:
            if sig["symbol"] != sym or sig["status"] == "resolved":
                continue
            if resolve(sig, d, df5) and sig["status"] == "resolved":
                logger.info("resolved: %s %s -> %s %+.2fR", sig["symbol"],
                            sig["signal_date"], sig["outcome"], sig["r"])
        for sig in state["signals"]:
            if (sig["symbol"] == sym and sig["status"] == "resolved"
                    and sig["origin"] == "live" and not sig["result_alerted"]
                    and update_state):
                outbox.append(result_message(sig, state))
                sig["result_alerted"] = True

    for msg in outbox:
        print("\n" + msg + "\n")
        if send_alerts:
            await alerts.send(msg)
    if not outbox:
        logger.info("nothing to send. %s", forward_record(state))

    if update_state:
        save_state(state)
        logger.info("state saved: %d signals tracked", len(state["signals"]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="F2D-Deep forward tracker (see module docstring)")
    ap.add_argument("--alert-only", action="store_true",
                    help="send due plan alerts but do not write f2d_forward.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print everything; write nothing, send nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Windows consoles default to cp1252, which cannot print the alert emoji.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    asyncio.run(run(update_state=not (args.alert_only or args.dry_run),
                    send_alerts=not args.dry_run))


if __name__ == "__main__":
    main()
