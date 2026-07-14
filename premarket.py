"""
premarket.py
------------
The pre-open brief. Run this between roughly 08:00 and 09:29 ET.

WHAT THIS IS NOT
================
It is not a premarket F2 scanner. An F2 cannot exist before the bell, and it
would be dishonest to pretend otherwise. An F2 arms off the FORMING setup bar:
price pokes through the prior bar's extreme, then fails back. Before 09:30 there
is no forming 4H (or Daily) bar. There is nothing to poke through.

WHAT IT ACTUALLY IS
===================
It tells you which F2 is SET UP to happen at the open, and where -- on BOTH the
4H and the Daily -- annotated with the timeframe stack coming into the session
so you know whether the set-up is with-trend or fighting it.

The prior 4H and Daily highs/lows are fixed and known before the bell. Premarket
price relative to those levels determines what the 09:30 bar opens as:

    premarket ABOVE the prior high  -> opens as a 2U attempt; fail back = F2D.
    premarket BELOW the prior low   -> opens as a 2D attempt; reclaim = F2U.
    premarket INSIDE the range      -> opens as a 1; watch the edges.

Plus: any inside bar that closed yesterday (4H or Daily) is armed from the first
tick, and the FTFC stack (1H/2H/4H/1D/1W, as it closed) tells you whether the
setup runs with the trend or against it -- the difference between the AAPL long
you take and the countertrend short you skip.

Premarket bars are used ONLY to read current price. They are NEVER fed into the
candles -- letting extended-hours tape into the buckets is precisely the bug
that made v3's higher-timeframe signals worthless.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd

import magnitude
from alerting import AlertManager
from bars import ET, BarProvider
from config import CONFIG, load_tickers
from levels import split_closed_forming
from strat import label_bars

logger = logging.getLogger("strat_scanner.premarket")

# The stack we read for bias, coming into the open (matches the scanner's FTFC).
FTFC_TFS: tuple[str, ...] = ("1H", "2H", "4H", "1D", "1W")
# Timeframes whose prior extremes we surface as overnight magnets.
MAGNET_TFS: tuple[str, ...] = ("4H", "1D", "1W")


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def _state(frames: dict, now: pd.Timestamp):
    """(closed_by_tf, forming_by_tf) for every real frame, labelled."""
    closed_by_tf: dict[str, pd.DataFrame] = {}
    forming_by_tf: dict[str, Optional[pd.Series]] = {}
    for tf, df in frames.items():
        if tf.startswith("_"):
            continue
        lab = label_bars(df)
        c, f = split_closed_forming(lab, now)
        closed_by_tf[tf] = c
        forming_by_tf[tf] = f
    return closed_by_tf, forming_by_tf


def _tf_map(prior: pd.Series, ref: float) -> dict:
    """What the next bar of this timeframe opens as, and which F2 that sets up."""
    hi, lo = float(prior["high"]), float(prior["low"])
    lab = prior["label"] if isinstance(prior["label"], str) else "?"
    if ref > hi:
        return dict(high=hi, low=lo, label=lab, opens_as="2U",
                    f2="F2D", trigger=hi, inside=lab == "1")
    if ref < lo:
        return dict(high=hi, low=lo, label=lab, opens_as="2D",
                    f2="F2U", trigger=lo, inside=lab == "1")
    return dict(high=hi, low=lo, label=lab, opens_as="1",
                f2=None, trigger=None, inside=lab == "1")


def _stack_bias(closed_by_tf: dict) -> tuple[int, int]:
    """How many of the FTFC frames CLOSED bullish (green). The bias into the
    open -- there is no forming intraday bar premarket, so we read the last
    completed bar of each frame."""
    bull = total = 0
    for tf in FTFC_TFS:
        c = closed_by_tf.get(tf)
        if c is None or c.empty:
            continue
        total += 1
        last = c.iloc[-1]
        if float(last["close"]) > float(last["open"]):
            bull += 1
    return bull, total


def _trend_note(f2: Optional[str], bull: int, total: int) -> str:
    """Does the stack support the set-up F2 direction? F2U is bullish, F2D
    bearish."""
    if f2 is None or total == 0:
        return ""
    agree = bull if f2 == "F2U" else (total - bull)
    if agree >= 4:
        return "with-trend ✓"
    if agree <= 1:
        return "⚠ counter-trend"
    return "mixed stack"


def brief_symbol(provider: BarProvider, symbol: str, now: pd.Timestamp) -> Optional[dict]:
    frames = provider.fetch(symbol)
    if not frames or "4H" not in frames or "1D" not in frames or "5Min" not in frames:
        return None

    closed_by_tf, _ = _state(frames, now)
    c4 = closed_by_tf.get("4H")
    cD = closed_by_tf.get("1D")
    c5 = closed_by_tf.get("5Min")
    if c4 is None or len(c4) < 2 or cD is None or len(cD) < 2 or c5 is None or c5.empty:
        return None

    # Price now (full tape, incl. premarket) vs the last RTH close.
    last = frames.get("_last")
    px = float(last["close"].iloc[-1]) if (last is not None and not last.empty) else None
    rth_close = float(c5["close"].iloc[-1])
    has_pre = px is not None and last is not None and last.index[-1] > c5.index[-1]
    ref = px if px is not None else rth_close

    m4 = _tf_map(c4.iloc[-1], ref)   # 4H map
    mD = _tf_map(cD.iloc[-1], ref)   # Daily map

    bull, total = _stack_bias(closed_by_tf)

    # Overnight magnets: untouched higher-TF prior extremes beyond price.
    up_mag, dn_mag = [], []
    for tf in MAGNET_TFS:
        c = closed_by_tf.get(tf)
        if c is None or c.empty:
            continue
        ph, pl = float(c.iloc[-1]["high"]), float(c.iloc[-1]["low"])
        if ph > ref:
            up_mag.append((tf, ph))
        if pl < ref:
            dn_mag.append((tf, pl))
    up_mag.sort(key=lambda x: x[1])          # nearest above first
    dn_mag.sort(key=lambda x: -x[1])         # nearest below first

    return {
        "symbol": symbol,
        "premarket": px,
        "has_premarket": has_pre,
        "rth_close": rth_close,
        "gap_pct": ((px - rth_close) / rth_close * 100) if px else 0.0,
        "m4": m4,
        "mD": mD,
        "ftfc_bull": bull,
        "ftfc_total": total,
        "up_mag": up_mag,
        "dn_mag": dn_mag,
    }


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _stack_str(bull: int, total: int) -> str:
    if total == 0:
        return "stack `?`"
    if bull >= total - bull:
        return f"stack `{bull}/{total} bull`"
    return f"stack `{total - bull}/{total} bear`"


def _tf_line(tag: str, m: dict, bull: int, total: int) -> Optional[str]:
    """One line for a 4H or Daily map, or None if it's just sitting inside."""
    if m["f2"]:
        note = _trend_note(m["f2"], bull, total)
        note = f"  {note}" if note else ""
        fail = "fail back below" if m["f2"] == "F2D" else "reclaim"
        dbl = "  (also the breakout trigger)" if m["inside"] else ""
        return (f"    {tag} opens `{m['opens_as']}` · *{m['f2']}* at "
                f"`{m['trigger']:.2f}` ({fail}){dbl}{note}")
    if m["inside"]:
        return (f"    {tag} inside bar armed · bull above `{m['high']:.2f}` · "
                f"bear below `{m['low']:.2f}`")
    return None


def format_brief(rows: list[dict], now: pd.Timestamp) -> str:
    lines = [
        f"🌅 *Pre-open brief* — {now:%a %d %b}, {now:%H:%M} ET",
        "",
        "_What the 09:30 4H & Daily bars open as, which F2 that sets up, and_",
        "_whether the timeframe stack backs it. An F2 can't exist before the bell —_",
        "_this is the map, not the signal._",
        "",
    ]

    def is_live(r):
        return r["m4"]["f2"] or r["mD"]["f2"] or r["m4"]["inside"] or r["mD"]["inside"]

    live = [r for r in rows if is_live(r)]
    quiet = [r for r in rows if not is_live(r)]

    # With-trend F2s first, then bigger gaps.
    def sort_key(r):
        best = ""
        for m in (r["mD"], r["m4"]):        # Daily weighted first
            n = _trend_note(m["f2"], r["ftfc_bull"], r["ftfc_total"])
            if "with-trend" in n:
                best = "0"
            elif best == "" and n:
                best = "1"
        return (best or "2", -abs(r["gap_pct"]))
    live.sort(key=sort_key)

    for r in live:
        px = r["premarket"] if r["premarket"] is not None else r["rth_close"]
        no_tape = "" if r["has_premarket"] else "  _(no premkt tape — yday close)_"
        gap = r["ftfc_bull"] >= (r["ftfc_total"] - r["ftfc_bull"])
        head_arrow = "🟢" if gap else "🔴"
        lines.append(
            f"{head_arrow} *{r['symbol']}* — premkt `{px:.2f}` "
            f"({r['gap_pct']:+.2f}%) · {_stack_str(r['ftfc_bull'], r['ftfc_total'])}{no_tape}"
        )
        for tag, m in (("1D", r["mD"]), ("4H", r["m4"])):
            ln = _tf_line(tag, m, r["ftfc_bull"], r["ftfc_total"])
            if ln:
                lines.append(ln)
        # magnets in the direction(s) that have a set-up
        mag_bits = []
        if r["dn_mag"]:
            mag_bits.append("↓ " + " · ".join(f"{tf} `{p:.2f}`" for tf, p in r["dn_mag"][:3]))
        if r["up_mag"]:
            mag_bits.append("↑ " + " · ".join(f"{tf} `{p:.2f}`" for tf, p in r["up_mag"][:3]))
        if mag_bits:
            lines.append("    magnets: " + "   ".join(mag_bits))
        lines.append("")

    if quiet:
        lines.append("_Inside their prior 4H & Daily ranges, nothing set up: "
                     + ", ".join(r["symbol"] for r in quiet) + "_")
        lines.append("")

    if not live:
        lines.append("_Nothing set up at the open. Everything is inside its prior ranges._")
        lines.append("")

    lines.append("_Levels are RTH 4H/Daily bars. Premarket tape is NOT in the candles._")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

async def run_brief(dry_run: bool = False) -> None:
    now = pd.Timestamp.now(tz=ET)
    provider = BarProvider(
        CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key, CONFIG.alpaca_data_feed
    )

    rows = []
    for symbol in load_tickers():
        try:
            r = await asyncio.to_thread(brief_symbol, provider, symbol, now)
        except Exception:
            logger.exception("Pre-open brief failed for %s", symbol)
            continue
        if r:
            rows.append(r)

    if not rows:
        logger.warning("No data for the pre-open brief.")
        return

    msg = format_brief(rows, now)
    print(msg)

    if not dry_run:
        alerts = AlertManager(CONFIG.telegram_bot_token, CONFIG.telegram_chat_id)
        await alerts.send(msg)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run_brief(dry_run="--dry-run" in sys.argv))
