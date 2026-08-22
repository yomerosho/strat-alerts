"""
strategy_lab.py -- Multi-strategy intraday backtest lab (QQQ, 2022→now).

Tests, with one shared outcome engine (entry at signal-bar close, conservative
stop-first on ambiguous bars, flat 15:55 ET):

  1. orb_9ema      -- ORB break, retrace to 9EMA, rejection candle (3-min)
  2. rsi2 / rsi14  -- RSI mean reversion (5-min)
  3. bb_rev        -- Bollinger(20,2) reversion (5-min)
  4. pctb_rev      -- %B reversion (5-min)         } same family as bb_rev
  5. zscore_rev    -- z-score(20) reversion (5-min) } by construction
  6. pivot_bounce  -- prior-day floor pivot S1/R1/P rejection (5-min)
  7. cross_5m      -- EMA50/200 golden-death cross, intraday (5-min)
  8. bos_3m        -- break of structure on 3-min swings, daily-EMA21 gated

Stats per variant: n, win%, PF, expectancy/share (gross + cost-adj),
trades/day, and 2022-only (bear) win% for regime robustness.
"""
from __future__ import annotations

import datetime as dt
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_HOST = "https://data.alpaca.markets"
ET = "America/New_York"
CACHE = Path(__file__).parent / ".lab_cache"
CACHE.mkdir(exist_ok=True)
START = "2022-01-01"
COST = 0.025  # $/share round trip (slippage + commission), same scale as prior studies

with open(r"C:\Users\yshob\GexMetric\.streamlit\secrets.toml", "rb") as f:
    _sec = tomllib.load(f)
KEY, SECRET = _sec["alpaca"]["key"], _sec["alpaca"]["secret"]
HDRS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}


def fetch_bars(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    fp = CACHE / f"{symbol}_{timeframe}_{start}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = f"{DATA_HOST}/v2/stocks/bars"
    params = {"symbols": symbol, "timeframe": timeframe, "start": start,
              "end": dt.date.today().isoformat(), "adjustment": "raw",
              "limit": 10000, "sort": "asc"}
    rows, token = [], None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(url, headers=HDRS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for b in (data.get("bars", {}) or {}).get(symbol, []):
            rows.append((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
        token = data.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df["date"] = df["ts"].dt.date
    df["hm"] = df["ts"].dt.strftime("%H:%M")
    if timeframe != "1Day":
        df = df[(df["hm"] >= "09:30") & (df["hm"] < "16:00")].reset_index(drop=True)
    df.to_parquet(fp)
    return df


def rsi(series: pd.Series, n: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def daily_context(d1: pd.DataFrame) -> pd.DataFrame:
    d1 = d1.copy()
    d1["ema21"] = d1["c"].ewm(span=21, adjust=False).mean()
    d1["P"] = (d1["h"] + d1["l"] + d1["c"]) / 3
    d1["S1"] = 2 * d1["P"] - d1["h"]
    d1["R1"] = 2 * d1["P"] - d1["l"]
    ctx = d1.set_index("date")[["h", "l", "ema21", "P", "S1", "R1"]].shift(1)
    ctx.columns = ["pdh", "pdl", "ema21", "P", "S1", "R1"]
    return ctx


RESOLVE_HOOK = None   # set to a resolve-compatible fn to swap exit models
EOD_HM = "15:55"      # last-exit bar stamp; set "15:45" for 15-min charts


def _res():
    return RESOLVE_HOOK or resolve


def resolve(day_bars: pd.DataFrame, i: int, side: str, entry: float,
            stop: float, tgt: float) -> tuple[int, float] | None:
    """Walk bars after i. -> (win 0/1, pnl_per_share) or None if no bars left."""
    for _, fb in day_bars.iloc[i + 1:].iterrows():
        if fb["hm"] >= EOD_HM:
            pnl = (fb["c"] - entry) if side == "L" else (entry - fb["c"])
            return (int(pnl > 0), pnl)
        if side == "L":
            hit_t, hit_s = fb["h"] >= tgt, fb["l"] <= stop
        else:
            hit_t, hit_s = fb["l"] <= tgt, fb["h"] >= stop
        if hit_s:                       # conservative: stop first on both-hit bars
            return (0, -(abs(entry - stop)))
        if hit_t:
            return (1, abs(tgt - entry))
    return None


def stats(trades: list[dict], name: str) -> dict | None:
    if not trades:
        print(f"{name:22s}  no trades")
        return None
    df = pd.DataFrame(trades)
    ndays = df["date"].nunique()
    span_days = np.busday_count(df["date"].min(), df["date"].max()) or 1
    wins, losses = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
    exp_g = df["pnl"].mean()
    exp_n = exp_g - COST
    bear = df[(df["date"] >= dt.date(2022, 1, 1)) & (df["date"] <= dt.date(2022, 12, 31))]
    bear_wr = bear["win"].mean() if len(bear) >= 10 else np.nan
    row = dict(name=name, n=len(df), win=df["win"].mean(), pf=pf,
               exp_gross=exp_g, exp_net=exp_n, per_day=len(df) / span_days,
               bear_wr=bear_wr, bear_n=len(bear))
    print(f"{name:22s}  n={row['n']:5d}  win={row['win']:.1%}  PF={pf:5.2f}  "
          f"exp/net={exp_n:+.3f}$  t/day={row['per_day']:.2f}  "
          f"2022win={'' if np.isnan(bear_wr) else f'{bear_wr:.1%}'}({len(bear)})")
    return row


# ══════════════════════════════ strategies ══════════════════════════════════

def strat_orb_9ema(m3: pd.DataFrame, ctx: pd.DataFrame, r_mult: float, gate: bool,
                   stop_mode: str = "rej", min_or_bars: int = 5) -> list[dict]:
    """ORB(15m) break -> retrace to 3-min 9EMA -> rejection candle entry."""
    m3 = m3.copy()
    m3["ema9"] = m3.groupby("date")["c"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    out = []
    for date, day in m3.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "pdh"]):
            continue
        pdh, pdl, ema21 = ctx.loc[date, ["pdh", "pdl", "ema21"]]
        day = day.reset_index(drop=True)
        orb = day[day["hm"] < "09:45"]
        if len(orb) < min_or_bars:
            continue
        orbH, orbL = orb["h"].max(), orb["l"].min()
        broke_up = broke_dn = False
        done_l = done_s = False
        for i, bar in day.iterrows():
            if bar["hm"] < "09:45" or bar["hm"] > "11:57":
                continue
            rng = bar["h"] - bar["l"]
            if rng <= 0:
                continue
            if bar["c"] > orbH and not broke_up:
                broke_up = True
                continue
            if bar["c"] < orbL and not broke_dn:
                broke_dn = True
                continue
            # long: after up-break, bar dips to/below 9EMA, rejects (close above
            # ema, close in upper 40% of range), still above ORB high zone
            if (broke_up and not done_l and bar["l"] <= bar["ema9"]
                    and bar["c"] > bar["ema9"] and (bar["c"] - bar["l"]) / rng >= 0.6
                    and (not gate or (bar["c"] > pdh and bar["c"] > ema21))):
                stop = bar["l"] if stop_mode == "rej" else (orbH + orbL) / 2
                if bar["c"] - stop <= 0:
                    continue
                tgt = bar["c"] + r_mult * (bar["c"] - stop)
                res = _res()(day, i, "L", bar["c"], stop, tgt)
                if res:
                    out.append(dict(date=date, win=res[0], pnl=res[1]))
                done_l = True
            elif (broke_dn and not done_s and bar["h"] >= bar["ema9"]
                    and bar["c"] < bar["ema9"] and (bar["h"] - bar["c"]) / rng >= 0.6
                    and (not gate or (bar["c"] < pdl and bar["c"] < ema21))):
                stop = bar["h"] if stop_mode == "rej" else (orbH + orbL) / 2
                if stop - bar["c"] <= 0:
                    continue
                tgt = bar["c"] - r_mult * (stop - bar["c"])
                res = _res()(day, i, "S", bar["c"], stop, tgt)
                if res:
                    out.append(dict(date=date, win=res[0], pnl=res[1]))
                done_s = True
    return out


def strat_meanrev(m5: pd.DataFrame, ctx: pd.DataFrame, kind: str, r_mult: float,
                  gate: bool, max_day: int = 2) -> list[dict]:
    """RSI / Bollinger / %B / z-score reversion on 5-min bars."""
    m5 = m5.copy()
    m5["rsi2"] = rsi(m5["c"], 2)
    m5["rsi14"] = rsi(m5["c"], 14)
    m5["sma20"] = m5["c"].rolling(20).mean()
    m5["sd20"] = m5["c"].rolling(20).std()
    m5["lower"] = m5["sma20"] - 2 * m5["sd20"]
    m5["upper"] = m5["sma20"] + 2 * m5["sd20"]
    m5["z"] = (m5["c"] - m5["sma20"]) / m5["sd20"]
    m5["pctb"] = (m5["c"] - m5["lower"]) / (m5["upper"] - m5["lower"])
    m5["swlo"] = m5["l"].rolling(12).min()
    m5["swhi"] = m5["h"].rolling(12).max()

    if kind == "rsi2":
        lsig = (m5["rsi2"] < 10) & (m5["rsi2"].shift(1) >= 10)
        ssig = (m5["rsi2"] > 90) & (m5["rsi2"].shift(1) <= 90)
    elif kind == "rsi14":
        lsig = (m5["rsi14"] < 30) & (m5["rsi14"].shift(1) >= 30)
        ssig = (m5["rsi14"] > 70) & (m5["rsi14"].shift(1) <= 70)
    elif kind == "bb":
        lsig = (m5["c"] > m5["lower"]) & (m5["c"].shift(1) < m5["lower"].shift(1))
        ssig = (m5["c"] < m5["upper"]) & (m5["c"].shift(1) > m5["upper"].shift(1))
    elif kind == "pctb":
        lsig = (m5["pctb"] > 0.05) & (m5["pctb"].shift(1) <= 0.0)
        ssig = (m5["pctb"] < 0.95) & (m5["pctb"].shift(1) >= 1.0)
    else:  # zscore
        lsig = (m5["z"] > -2.0) & (m5["z"].shift(1) <= -2.2)
        ssig = (m5["z"] < 2.0) & (m5["z"].shift(1) >= 2.2)

    m5["lsig"], m5["ssig"] = lsig, ssig
    out = []
    for date, day in m5.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "ema21"]):
            continue
        ema21 = ctx.loc[date, "ema21"]
        day = day.reset_index(drop=True)
        taken = 0
        for i, bar in day.iterrows():
            if taken >= max_day or not ("10:00" <= bar["hm"] <= "15:30"):
                continue
            side = "L" if bar["lsig"] else ("S" if bar["ssig"] else None)
            if side is None:
                continue
            if gate and ((side == "L") != (bar["c"] > ema21)):
                continue
            stop = bar["swlo"] if side == "L" else bar["swhi"]
            risk = (bar["c"] - stop) if side == "L" else (stop - bar["c"])
            if not np.isfinite(risk) or risk <= 0:
                continue
            tgt = bar["c"] + r_mult * risk if side == "L" else bar["c"] - r_mult * risk
            res = _res()(day, i, side, bar["c"], stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1]))
                taken += 1
    return out


def strat_pivot(m5: pd.DataFrame, ctx: pd.DataFrame, r_mult: float) -> list[dict]:
    """Rejection candle off prior-day floor pivots (S1/P for longs, R1/P shorts)."""
    out = []
    for date, day in m5.groupby("date", sort=True):
        if date not in ctx.index or pd.isna(ctx.loc[date, "P"]):
            continue
        P, S1, R1 = ctx.loc[date, ["P", "S1", "R1"]]
        day = day.reset_index(drop=True)
        taken = 0
        for i, bar in day.iterrows():
            if taken >= 2 or not ("09:45" <= bar["hm"] <= "15:30"):
                continue
            rng = bar["h"] - bar["l"]
            if rng <= 0:
                continue
            side = None
            for lvl in (S1, P):
                if bar["l"] <= lvl <= bar["h"] and bar["c"] > lvl and (bar["c"] - bar["l"]) / rng >= 0.6 and bar["o"] > lvl:
                    side, stop = "L", bar["l"] - 0.02
                    break
            if side is None:
                for lvl in (R1, P):
                    if bar["l"] <= lvl <= bar["h"] and bar["c"] < lvl and (bar["h"] - bar["c"]) / rng >= 0.6 and bar["o"] < lvl:
                        side, stop = "S", bar["h"] + 0.02
                        break
            if side is None:
                continue
            risk = (bar["c"] - stop) if side == "L" else (stop - bar["c"])
            if risk <= 0:
                continue
            tgt = bar["c"] + r_mult * risk if side == "L" else bar["c"] - r_mult * risk
            res = _res()(day, i, side, bar["c"], stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1]))
                taken += 1
    return out


def strat_cross(m5: pd.DataFrame) -> list[dict]:
    """Intraday EMA50/200 golden-death cross, exit on opposite cross or EOD."""
    m5 = m5.copy()
    m5["e50"] = m5["c"].ewm(span=50, adjust=False).mean()
    m5["e200"] = m5["c"].ewm(span=200, adjust=False).mean()
    m5["gc"] = (m5["e50"] > m5["e200"]) & (m5["e50"].shift(1) <= m5["e200"].shift(1))
    m5["dc"] = (m5["e50"] < m5["e200"]) & (m5["e50"].shift(1) >= m5["e200"].shift(1))
    out = []
    for date, day in m5.groupby("date", sort=True):
        day = day.reset_index(drop=True)
        pos = None
        for i, bar in day.iterrows():
            if bar["hm"] >= EOD_HM:
                if pos:
                    pnl = (bar["c"] - pos[1]) if pos[0] == "L" else (pos[1] - bar["c"])
                    out.append(dict(date=date, win=int(pnl > 0), pnl=pnl))
                    pos = None
                break
            if bar["gc"] or bar["dc"]:
                if pos:
                    pnl = (bar["c"] - pos[1]) if pos[0] == "L" else (pos[1] - bar["c"])
                    out.append(dict(date=date, win=int(pnl > 0), pnl=pnl))
                pos = ("L", bar["c"]) if bar["gc"] else ("S", bar["c"])
        else:
            pass
    return out


def strat_bos(m3: pd.DataFrame, ctx: pd.DataFrame, r_mult: float, gate: bool,
              win_start: str = "09:45", win_end: str = "15:30") -> list[dict]:
    """3-min break of structure: close beyond last confirmed 3/3 swing point."""
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
            j = i - 3                       # candidate confirmed 3 bars later
            if j >= 3 and h[j] == max(h[j - 3:min(j + 4, n)]):
                last_hi = h[j]
            if j >= 3 and l[j] == min(l[j - 3:min(j + 4, n)]):
                last_lo = l[j]
            sw_hi[i], sw_lo[i] = last_hi, last_lo
        taken = 0
        broken_hi = broken_lo = -np.inf
        for i, bar in day.iterrows():
            if taken >= 3 or not (win_start <= bar["hm"] <= win_end):
                continue
            shi, slo = sw_hi[i], sw_lo[i]
            side = None
            if np.isfinite(shi) and bar["c"] > shi and shi > broken_hi and (not gate or bar["c"] > ema21):
                side, stop = "L", (slo if np.isfinite(slo) and slo < bar["c"] else bar["c"] - 2 * (bar["h"] - bar["l"]))
                broken_hi = shi
            elif np.isfinite(slo) and bar["c"] < slo and -slo > broken_lo and (not gate or bar["c"] < ema21):
                side, stop = "S", (shi if np.isfinite(shi) and shi > bar["c"] else bar["c"] + 2 * (bar["h"] - bar["l"]))
                broken_lo = -slo
            if side is None:
                continue
            risk = (bar["c"] - stop) if side == "L" else (stop - bar["c"])
            if not np.isfinite(risk) or risk <= 0:
                continue
            tgt = bar["c"] + r_mult * risk if side == "L" else bar["c"] - r_mult * risk
            res = _res()(day, i, side, bar["c"], stop, tgt)
            if res:
                out.append(dict(date=date, win=res[0], pnl=res[1]))
                taken += 1
    return out


# ══════════════════════════════ runner ═══════════════════════════════════════

def main():
    print("Fetching bars (cached after first run)...")
    m5 = fetch_bars("QQQ", "5Min", START)
    m3 = fetch_bars("QQQ", "3Min", START)
    d1 = fetch_bars("QQQ", "1Day", "2021-01-01")
    ctx = daily_context(d1)
    print(f"  5m={len(m5)}  3m={len(m3)}  days={m5['date'].nunique()}\n")
    rows = []

    print("── ORB -> 9EMA rejection (3-min) ──")
    for gate in (False, True):
        for r in (0.5, 1.0):
            rows.append(stats(strat_orb_9ema(m3, ctx, r, gate),
                              f"orb9ema {'gated' if gate else 'raw'} {r}R"))

    print("\n── Mean reversion family (5-min) ──")
    for kind in ("rsi2", "rsi14", "bb", "pctb", "zscore"):
        for gate in (False, True):
            rows.append(stats(strat_meanrev(m5, ctx, kind, 0.5, gate),
                              f"{kind} {'gated' if gate else 'raw'} 0.5R"))

    print("\n── Pivot bounce (5-min) ──")
    for r in (0.5, 1.0):
        rows.append(stats(strat_pivot(m5, ctx, r), f"pivot {r}R"))

    print("\n── Golden/death cross (5-min intraday) ──")
    rows.append(stats(strat_cross(m5), "ema50/200 cross"))

    print("\n── Break of structure (3-min) ──")
    for gate in (False, True):
        for r in (0.5, 1.0):
            rows.append(stats(strat_bos(m3, ctx, r, gate),
                              f"bos3m {'gated' if gate else 'raw'} {r}R"))

    lead = pd.DataFrame([r for r in rows if r])
    lead.to_csv(CACHE / "lab_results.csv", index=False)
    print("\n=== Leaderboard (win% desc, n>=200) ===")
    top = lead[lead["n"] >= 200].sort_values("win", ascending=False)
    for _, r in top.iterrows():
        print(f"  {r['name']:22s} win={r['win']:.1%}  PF={r['pf']:5.2f}  n={r['n']}  t/day={r['per_day']:.2f}  exp_net={r['exp_net']:+.3f}")


if __name__ == "__main__":
    main()
