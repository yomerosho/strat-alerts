"""
premarket.py
------------
The pre-open brief. Run this between roughly 08:00 and 09:29 ET.

WHAT THIS IS NOT
================
It is not a premarket F2 scanner. An F2 cannot exist before the bell, and it
would be dishonest to pretend otherwise.

An F2 arms off the FORMING setup bar: price has to poke through the prior 4H
bar's high, then fail back below it. Before 09:30 there is no forming 4H bar.
There is nothing to poke through. Any tool claiming to show you "a premarket
Failed-2" is showing you a fiction.

WHAT IT ACTUALLY IS
===================
It tells you which F2 is SET UP to happen at the open, and where.

The prior 4H bar's high and low are fixed and known before the bell -- they
were set yesterday afternoon. Premarket price relative to those two numbers
determines what the 09:30 bar opens as:

    premarket ABOVE prior 4H high   -> the new bar opens as a 2U attempt.
                                       If it fails back below that high, that
                                       is an F2D, and you already know the
                                       exact trigger price.

    premarket BELOW prior 4H low    -> opens as a 2D attempt.
                                       Failure back above = F2U.

    premarket INSIDE the prior range-> opens as a 1. Nothing to trap anyone
                                       with. Watch the edges.

Plus: any inside bar that closed on the 4H yesterday is still armed this
morning. Those levels carry over and are live from the first tick.

Premarket bars are used ONLY to read current price. They are never fed into
the 4H candles -- letting extended-hours tape into the buckets is precisely
the bug that made v3's higher-timeframe signals worthless.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alerting import AlertManager
from bars import ET, BarProvider, filter_rth, resample_session
from config import CONFIG, load_tickers
from levels import split_closed_forming
from strat import label_bars

logger = logging.getLogger("strat_scanner.premarket")

PREMARKET_OPEN = "04:00"
PREMARKET_END = "09:29"


def premarket_price(raw5: pd.DataFrame, session_date) -> Optional[float]:
    """Last traded price in today's premarket window (04:00-09:29 ET)."""
    today = raw5[[t.date() == session_date for t in raw5.index]]
    if today.empty:
        return None
    pre = today.between_time(PREMARKET_OPEN, PREMARKET_END)
    if pre.empty:
        return None
    return float(pre["close"].iloc[-1])


def brief_symbol(provider: BarProvider, symbol: str, now: pd.Timestamp) -> Optional[dict]:
    raw5 = provider._request(symbol, TimeFrame(5, TimeFrameUnit.Minute), 12)
    if raw5.empty:
        return None

    rth = filter_rth(raw5)
    if rth.empty:
        return None

    bars4 = label_bars(resample_session(rth, 240))
    closed, _ = split_closed_forming(bars4, now)
    if len(closed) < 2:
        return None

    prior = closed.iloc[-1]          # yesterday's final 4H bar
    prior_high = float(prior["high"])
    prior_low = float(prior["low"])
    prior_label = prior["label"] if isinstance(prior["label"], str) else "?"

    px = premarket_price(raw5, now.date())
    rth_close = float(rth["close"].iloc[-1])   # yesterday's 16:00 print

    out = {
        "symbol": symbol,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "prior_label": prior_label,
        "prior_bar_time": prior.name,
        "rth_close": rth_close,
        "premarket": px,
        "has_premarket": px is not None,
        "gap_pct": ((px - rth_close) / rth_close * 100) if px else 0.0,
    }

    ref = px if px is not None else rth_close

    # --- what does the 09:30 bar open as, and what F2 does that set up? ---
    if ref > prior_high:
        out["opens_as"] = "2U"
        out["f2_watch"] = "F2D"
        out["f2_trigger"] = prior_high
        out["f2_note"] = (
            f"Opening above the prior 4H high. If price fails back below "
            f"{prior_high:.2f}, breakout buyers are trapped -> F2D."
        )
    elif ref < prior_low:
        out["opens_as"] = "2D"
        out["f2_watch"] = "F2U"
        out["f2_trigger"] = prior_low
        out["f2_note"] = (
            f"Opening below the prior 4H low. If price reclaims "
            f"{prior_low:.2f}, breakdown sellers are trapped -> F2U."
        )
    else:
        out["opens_as"] = "1"
        out["f2_watch"] = None
        out["f2_trigger"] = None
        out["f2_note"] = (
            f"Inside the prior 4H range ({prior_low:.2f}-{prior_high:.2f}). "
            f"No one is trapped yet. A break of either edge is the first move."
        )

    # --- inside bar carried over from yesterday: armed from the first tick ---
    out["carried_inside_bar"] = prior_label == "1"

    return out


def format_brief(rows: list[dict], now: pd.Timestamp) -> str:
    """
    One entry per symbol, not one per phenomenon.

    The subtlety worth surfacing: when yesterday's final 4H bar is an INSIDE
    bar and premarket has gapped above it, the breakout trigger and the F2
    trigger are THE SAME PRICE. That isn't a duplicate -- it's the single most
    useful fact on the page. One level, two opposite resolutions:

        close above it and hold  -> the 2-1-2 breakout is on
        poke above and fail back -> everyone who bought the gap is trapped, F2D

    Printing those as two separate list items with identical numbers looks
    like a bug and buries the point. So each symbol gets one block that says
    what the level is and both ways it can go.
    """
    lines = [
        f"🌅 *Pre-open brief* — {now:%a %d %b}, {now:%H:%M} ET",
        "",
        "_What the 09:30 4H bar opens as, and which F2 that sets up._",
        "_An F2 can't exist before the bell — this is the map, not the signal._",
        "",
    ]

    live = [r for r in rows if r["f2_watch"] or r["carried_inside_bar"]]
    quiet = [r for r in rows if not r["f2_watch"] and not r["carried_inside_bar"]]

    live.sort(key=lambda r: (r["f2_watch"] is None, -abs(r["gap_pct"])))

    for r in live:
        px = r["premarket"] or r["rth_close"]
        no_tape = "" if r["has_premarket"] else "  _(no premkt tape — yday close)_"
        hi, lo = r["prior_high"], r["prior_low"]

        if r["f2_watch"]:
            arrow = "🔴" if r["f2_watch"] == "F2D" else "🟢"
            trig = r["f2_trigger"]
            lines.append(
                f"{arrow} *{r['symbol']}* — opens `{r['opens_as']}`  "
                f"premkt `{px:.2f}` ({r['gap_pct']:+.2f}%){no_tape}"
            )
            if r["carried_inside_bar"]:
                # Same price, two opposite outcomes. This is the whole point.
                hold = "above" if r["f2_watch"] == "F2D" else "below"
                fail = "back below" if r["f2_watch"] == "F2D" else "back above"
                lines.append(
                    f"    ⚡ *`{trig:.2f}` is a double-edged level* (4H inside bar):\n"
                    f"       · 15m closes {hold} it and holds → breakout is on\n"
                    f"       · pokes through, then closes {fail} → *{r['f2_watch']}*, "
                    f"gap buyers trapped"
                )
            else:
                lines.append(f"    F2 trigger `{trig:.2f}` → watch *{r['f2_watch']}*")
        else:
            lines.append(
                f"🎯 *{r['symbol']}* — 4H inside bar, armed from the first tick  "
                f"premkt `{px:.2f}` ({r['gap_pct']:+.2f}%){no_tape}"
            )
            lines.append(f"    bull above `{hi:.2f}` · bear below `{lo:.2f}`")
        lines.append("")

    if quiet:
        lines.append(f"_Sitting inside their prior 4H range, nothing set up: "
                     f"{', '.join(r['symbol'] for r in quiet)}_")
        lines.append("")

    if not live:
        lines.append("_Nothing set up at the open. Everything is inside its "
                     "prior 4H range._")
        lines.append("")

    lines.append("_Levels are RTH 4H bars. Premarket tape is NOT in the candles._")
    return "\n".join(lines)


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
