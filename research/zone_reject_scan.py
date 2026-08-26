"""
zone_reject_scan.py
-------------------
Scan a watchlist for the simplest, highest-clarity setup:

    price arrives at a supply/demand zone AND rejects from it.

A ZONE is the base (small candles) that price departed from with an impulse.
Unfilled institutional orders are presumed to still sit there, so the first
return to an untested zone is where reactions cluster.

A REJECTION is a bar that traded INTO the zone and closed back OUT of it:
    demand zone -> wick below, close back above  = the Strat F2D (trapped shorts)
    supply zone -> wick above, close back below  = the Strat F2U (trapped longs)

Bars come from the project's own session-aligned BarProvider path (RTH-only,
09:30-anchored buckets), so 1H/4H here match the scanner and the chart.

Usage:  python research/zone_reject_scan.py [--tf 1H,4H] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from bars import filter_rth, resample_session  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from alpaca.data.enums import DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402

ET = "America/New_York"


def load_symbols(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.split(":")[-1])
    # de-dupe, keep order
    seen, res = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res


def fetch_5min(client, symbols: list[str], feed, days: int) -> dict[str, pd.DataFrame]:
    """One request per chunk of symbols instead of one per symbol."""
    frames: dict[str, pd.DataFrame] = {}
    start = datetime.now() - timedelta(days=days)
    CHUNK = 40
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            feed=feed,
        )
        try:
            df = client.get_stock_bars(req).df
        except Exception as exc:  # noqa: BLE001
            print(f"  ! chunk {i//CHUNK}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        for sym in chunk:
            try:
                sub = df.xs(sym, level=0)
            except KeyError:
                continue
            sub = sub.copy()
            sub.index = sub.index.tz_convert(ET)
            frames[sym] = sub[["open", "high", "low", "close", "volume"]].sort_index()
        print(f"  fetched {min(i+CHUNK, len(symbols))}/{len(symbols)}", file=sys.stderr)
    return frames


def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def find_zones(df: pd.DataFrame, a: float) -> list[dict]:
    """Base-before-impulse zones. Returns zones with freshness (# retests)."""
    O, H, L, C = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    body = lambda i: abs(C[i] - O[i])  # noqa: E731
    zones = []
    for imp in range(3, n - 1):
        if body(imp) < 0.9 * a:
            continue
        up = C[imp] > O[imp]
        base = []
        j = imp - 1
        while j >= 0 and len(base) < 4 and body(j) <= 0.55 * a:
            base.insert(0, j)
            j -= 1
        if not base:
            continue
        z_hi = max(H[i] for i in base)
        z_lo = min(L[i] for i in base)
        if z_hi - z_lo > 2.5 * a or z_hi <= z_lo:
            continue
        ext = 0.0
        for k in range(imp, min(n, imp + 8)):
            ext = max(ext, (H[k] - z_hi) if up else (z_lo - L[k]))
        if ext < 1.3 * a:
            continue
        # distinct visits: consecutive bars inside the zone count as ONE test
        tests, prev = 0, -99
        for k in range(imp + 1, n):
            if L[k] <= z_hi and H[k] >= z_lo:
                if k - prev > 2:
                    tests += 1
                prev = k
        zones.append({
            "kind": "demand" if up else "supply",
            "hi": round(float(z_hi), 2), "lo": round(float(z_lo), 2),
            "tests": tests, "ext": round(ext / a, 1), "idx": imp,
            "when": str(df.index[base[0]])[:16],
        })
    return zones


def strat_label(df: pd.DataFrame, i: int) -> str:
    h, l = df["high"].iloc[i], df["low"].iloc[i]
    ph, pl = df["high"].iloc[i - 1], df["low"].iloc[i - 1]
    up, dn = h > ph, l < pl
    if up and dn:
        return "3"
    if up:
        return "2U"
    if dn:
        return "2D"
    return "1"


def analyse(sym: str, df: pd.DataFrame, tf: str) -> list[dict]:
    if len(df) < 40:
        return []
    a = atr(df)
    if not a or a <= 0:
        return []
    zones = find_zones(df, a)
    if not zones:
        return []
    last = float(df["close"].iloc[-1])
    hits = []
    # only consider the most recent 2 closed bars for a live reaction
    for back in (1, 2):
        if len(df) < back + 2:
            continue
        i = len(df) - back
        bar = df.iloc[i]
        lbl = strat_label(df, i)
        for z in zones:
            if z["idx"] >= i - 1:
                continue  # zone must predate the bar testing it
            if z["tests"] > 3:
                continue  # chopped-through area, not a zone
            entered = bar["low"] <= z["hi"] and bar["high"] >= z["lo"]
            if z["kind"] == "demand":
                rejected = entered and bar["close"] > z["hi"] and bar["close"] > bar["open"]
                dist = (last - z["hi"]) / a
                side = "LONG"
                trap = "F2D" if lbl == "2D" and bar["close"] > bar["open"] else None
            else:
                rejected = entered and bar["close"] < z["lo"] and bar["close"] < bar["open"]
                dist = (z["lo"] - last) / a
                side = "SHORT"
                trap = "F2U" if lbl == "2U" and bar["close"] < bar["open"] else None
            approaching = (not entered) and 0 <= dist <= 1.2
            if rejected and dist > 1.0:
                continue  # move already happened; no entry left
            if not (rejected or approaching):
                continue
            wick = (
                (min(bar["open"], bar["close"]) - bar["low"]) if z["kind"] == "demand"
                else (bar["high"] - max(bar["open"], bar["close"]))
            )
            score = 0.0
            if rejected:
                score += 5
            if z["tests"] == 0:
                score += 3          # fresh zone
            elif z["tests"] <= 2:
                score += 1
            if trap:
                score += 3          # Strat trap confirms the rejection
            score += min(z["ext"], 4)
            score += min(wick / a, 2) * 1.5
            if back == 1:
                score += 1          # most recent bar
            hits.append({
                "sym": sym, "tf": tf, "side": side, "state": "REJECT" if rejected else "approach",
                "zone": f'{z["lo"]}-{z["hi"]}', "kind": z["kind"], "tests": z["tests"],
                "ext": z["ext"], "trap": trap or "-", "bar": lbl, "bars_ago": back,
                "last": round(last, 2), "dist_atr": round(float(dist), 2),
                "wick_atr": round(float(wick / a), 2), "when": z["when"],
                "score": round(score, 1), "atr": round(a, 2),
            })
    # best hit per (side,state) to avoid spamming one symbol
    hits.sort(key=lambda x: -x["score"])
    out, seen = [], set()
    for h in hits:
        k = (h["side"], h["state"])
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=str(Path(__file__).parent / "strat_list.txt"))
    ap.add_argument("--tf", default="1H,4H")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    symbols = load_symbols(Path(args.symbols))
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"scanning {len(symbols)} symbols, tf={args.tf}", file=sys.stderr)

    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing")
    feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}.get(
        (os.getenv("ALPACA_DATA_FEED") or "iex").lower(), DataFeed.IEX)
    client = StockHistoricalDataClient(key, sec)

    raw = fetch_5min(client, symbols, feed, args.days)
    print(f"got 5min data for {len(raw)} symbols", file=sys.stderr)

    tf_map = {"1H": 60, "2H": 120, "4H": 240}
    results = []
    for sym, df5 in raw.items():
        rth = filter_rth(df5)
        if rth.empty:
            continue
        for tf in [t.strip() for t in args.tf.split(",")]:
            mins = tf_map.get(tf)
            if not mins:
                continue
            try:
                bars = resample_session(rth, mins)
            except Exception:  # noqa: BLE001
                continue
            results.extend(analyse(sym, bars, tf))

    results.sort(key=lambda x: -x["score"])
    rej = [r for r in results if r["state"] == "REJECT"]
    app = [r for r in results if r["state"] == "approach"]

    def show(rows, title):
        print(f"\n===== {title} ({len(rows)}) =====")
        if not rows:
            print("  none")
            return
        print(f'{"SYM":<6}{"TF":<4}{"SIDE":<6}{"ZONE":<20}{"fresh":<7}{"trap":<6}{"bar":<5}'
              f'{"ago":<5}{"last":>9}{"dATR":>7}{"wick":>7}{"dep":>6}{"score":>7}  when')
        for r in rows[:25]:
            fresh = "FRESH" if r["tests"] == 0 else f'{r["tests"]}x'
            print(f'{r["sym"]:<6}{r["tf"]:<4}{r["side"]:<6}{r["zone"]:<20}{fresh:<7}{r["trap"]:<6}'
                  f'{r["bar"]:<5}{r["bars_ago"]:<5}{r["last"]:>9}{r["dist_atr"]:>7}'
                  f'{r["wick_atr"]:>7}{r["ext"]:>6}{r["score"]:>7}  {r["when"]}')

    show(rej, "REJECTIONS (zone tested and refused)")
    show(app, "APPROACHING (within 1.2 ATR of a zone)")


if __name__ == "__main__":
    main()
