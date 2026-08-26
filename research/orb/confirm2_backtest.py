"""
ORB break + 2-body-close confirmation entry, with volume-filter attribution.

User's rule: after price breaks the 15-min opening range, wait for 2
consecutive candles whose bodies close higher (long) / lower (short) than
the preceding candle, then enter. Question: does a volume confirmation
reduce the fakeouts?

Tested here on the same data/cost/exit framework as orb_backtest.py:
- 5-min bars, SPY/QQQ/IWM, IS 2022-2024, OOS 2025+.
- Breakout bar = first 5-min close beyond ORH/ORL, 09:45..13:55.
- Confirmation defs (both tested):
    C1 "close>close": 2 consecutive bars each closing beyond the prior
       bar's close (in trade direction). The breakout bar may count as
       the first of the two if it closed beyond the bar before it.
    C2 "close>extreme": stricter -- each bar's close beyond the prior
       bar's HIGH (long) / LOW (short).
  Entry at the close of the 2nd confirming bar; the entry close must
  still be beyond the OR level, and entry time <= 14:00 ET.
- Exit = V1 from the original test (stop at opposite OR level, target =
  entry + 1x OR range, time exit 15:45). Costs 1c/side.
- Fakeout metric: price closes back inside the OR within 6 bars after
  entry (independent of exit outcome).

Volume filters (decision-time safe, RVOL = bar vol / 20-day same
time-of-day mean):
    none        -- no filter
    brk_rvol    -- breakout bar RVOL >= 1.5
    conf_rvol1  -- both confirmation bars RVOL >= 1.0
    conf_rvol15 -- both confirmation bars RVOL >= 1.5
    any_rvol    -- max RVOL over (break, conf1, conf2) >= 1.5
    vol_expand  -- entry bar volume > mean volume of prior 6 bars today
    cum_rvol    -- session cumulative volume through entry bar >= 1.1x
                   the 20-day average cumulative volume to that time
"""

from pathlib import Path

import numpy as np
import pandas as pd

from orb_backtest import ET, IS_END, IS_START, OOS_START, SLIP, load_symbol, simulate_exit

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SYMBOLS = ["SPY", "QQQ", "IWM"]
ENTRY_CUTOFF = pd.Timestamp("14:00").time()   # entry bar close must be <= this
LAST_EXIT_BAR = pd.Timestamp("15:40").time()
T945 = pd.Timestamp("09:45").time()
T1355 = pd.Timestamp("13:55").time()


def build_day_context(df5):
    """Add cumulative-session RVOL to the frame."""
    df5 = df5.copy()
    day_key = df5.index.normalize()
    df5["cum_vol"] = df5.groupby(day_key)["volume"].cumsum()
    df5["avg_cum_tod"] = (
        df5.groupby("tod")["cum_vol"]
        .transform(lambda s: s.rolling(20, min_periods=20).mean().shift(1))
    )
    df5["cum_rvol"] = df5["cum_vol"] / df5["avg_cum_tod"]
    # rolling mean of prior 6 bars' volume within the session
    df5["prior6_vol"] = (
        df5.groupby(day_key)["volume"]
        .transform(lambda s: s.rolling(6, min_periods=3).mean().shift(1))
    )
    return df5


def find_confirmation(day, start_ts, direction, level, mode):
    """Walk bars from the breakout bar; return (entry_ts, conf1_ts, conf2_ts)
    for the first run of 2 consecutive confirming closes, else None.

    mode 'C1': close beyond prior bar's close.
    mode 'C2': close beyond prior bar's high (long) / low (short).
    """
    sign = 1 if direction == "long" else -1
    bars = day[day.index >= start_ts]
    idx = day.index.get_indexer([start_ts])[0]
    streak = []
    for k, (ts, b) in enumerate(bars.iterrows()):
        j = idx + k
        if j == 0:
            continue
        prev = day.iloc[j - 1]
        if mode == "C1":
            ref = prev["close"]
        else:
            ref = prev["high"] if sign == 1 else prev["low"]
        ok = sign * (b["close"] - ref) > 0
        if ok:
            streak.append(ts)
        else:
            streak = []
        if len(streak) >= 2:
            entry_ts = ts
            if day.loc[entry_ts, "tod"] > ENTRY_CUTOFF:
                return None
            # entry close must still be beyond the OR level
            if sign * (day.loc[entry_ts, "close"] - level) <= 0:
                return None
            return entry_ts, streak[-2], streak[-1]
    return None


def build_trades(sym, df5, d, mode):
    rows = []
    t930, t935, t940 = (pd.Timestamp(x).time() for x in ("09:30", "09:35", "09:40"))
    for date, day in df5.groupby(df5.index.normalize()):
        times = set(day["tod"])
        if not {t930, t935, t940} <= times:
            continue
        orb = day[day["tod"] < T945]
        orh, orl = orb["high"].max(), orb["low"].min()
        or_range = orh - orl
        if or_range <= 0 or date not in d.index:
            continue
        dd = d.loc[date]
        if pd.isna(dd["prev_close"]) or pd.isna(dd["atr_d"]):
            continue

        sig_bars = day[(day["tod"] >= T945) & (day["tod"] <= T1355)]

        for direction in ("long", "short"):
            sign = 1 if direction == "long" else -1
            level = orh if direction == "long" else orl
            mask = sign * (sig_bars["close"] - level) > 0
            if not mask.any():
                continue
            brk_ts = mask.idxmax()
            conf = find_confirmation(day, brk_ts, direction, level, mode)
            if conf is None:
                continue
            entry_ts, c1_ts, c2_ts = conf
            ebar = day.loc[entry_ts]
            entry = ebar["close"]
            if pd.isna(ebar["atr5"]):
                continue

            after = day[(day.index > entry_ts) & (day["tod"] <= LAST_EXIT_BAR)]

            brk = day.loc[brk_ts]
            c1, c2 = day.loc[c1_ts], day.loc[c2_ts]
            rows.append(dict(
                symbol=sym, date=date.date(), direction=direction,
                signal_ts=entry_ts, brk_ts=brk_ts,
                bars_brk_to_entry=int((entry_ts - brk_ts) / pd.Timedelta(minutes=5)),
                entry=entry, orh=orh, orl=orl, or_range=or_range,
                chase=sign * (entry - level),          # distance paid past the level
                brk_rvol=brk["rvol"],
                c1_rvol=c1["rvol"], c2_rvol=c2["rvol"],
                entry_vol=ebar["volume"], prior6_vol=ebar["prior6_vol"],
                cum_rvol=ebar["cum_rvol"],
                _after=after, _day=day,
            ))
    return pd.DataFrame(rows)


def apply_v1_exit(tr):
    out = tr.copy()
    sign = np.where(out["direction"] == "long", 1, -1)
    out["stop"] = np.where(sign == 1, out["orl"], out["orh"])
    out["target"] = out["entry"] + sign * out["or_range"]

    res = [simulate_exit(r, r["stop"], r["target"]) for _, r in out.iterrows()]
    out["exit"], out["exit_ts"], out["exit_reason"] = zip(*res)

    out["risk_ps"] = (out["entry"] - out["stop"]).abs()
    out["gross_ps"] = sign * (out["exit"] - out["entry"])
    out["net_ps"] = out["gross_ps"] - 2 * SLIP
    out["r_mult"] = out["net_ps"] / out["risk_ps"]
    out["s1_pnl"] = out["net_ps"] * (10000.0 / out["entry"])

    # fakeout: close back inside the OR within 6 bars after entry
    fake = []
    for _, r in out.iterrows():
        sgn = 1 if r["direction"] == "long" else -1
        level = r["orh"] if sgn == 1 else r["orl"]
        nxt = r["_after"].head(6)
        fake.append(bool((sgn * (nxt["close"] - level) < 0).any()) if len(nxt) else False)
    out["fakeout"] = fake
    return out.drop(columns=["_after", "_day"])


FILTERS = {
    "none":        lambda t: pd.Series(True, index=t.index),
    "brk_rvol":    lambda t: t["brk_rvol"].fillna(0) >= 1.5,
    "conf_rvol1":  lambda t: (t["c1_rvol"].fillna(0) >= 1.0) & (t["c2_rvol"].fillna(0) >= 1.0),
    "conf_rvol15": lambda t: (t["c1_rvol"].fillna(0) >= 1.5) & (t["c2_rvol"].fillna(0) >= 1.5),
    "any_rvol":    lambda t: t[["brk_rvol", "c1_rvol", "c2_rvol"]].fillna(0).max(axis=1) >= 1.5,
    "vol_expand":  lambda t: t["entry_vol"] > t["prior6_vol"].fillna(np.inf),
    "cum_rvol":    lambda t: t["cum_rvol"].fillna(0) >= 1.1,
}


def stats(g):
    n = len(g)
    if n == 0:
        return dict(n=0, win=np.nan, pf=np.nan, avg_r=np.nan, pnl=0.0, fake=np.nan)
    pnl = g["s1_pnl"]
    wins, losses = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    return dict(
        n=n, win=(pnl > 0).mean() * 100,
        pf=(wins / abs(losses)) if losses != 0 else np.inf,
        avg_r=g["r_mult"].mean(), pnl=pnl.sum(),
        fake=g["fakeout"].mean() * 100,
    )


def fmt(name, s):
    if s["n"] == 0:
        return f"| {name} | 0 | - | - | - | - | - |"
    pf = "inf" if np.isinf(s["pf"]) else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']} | {s['win']:.1f}% | {pf} | {s['avg_r']:+.3f} | "
            f"${s['pnl']:+,.0f} | {s['fake']:.1f}% |")


HDR = ("| config | trades | win% | PF | avg R | net P&L | fakeout% |\n"
       "|---|---|---|---|---|---|---|")


def main():
    lines = []
    W = lines.append
    W("# ORB 2-close confirmation entry + volume filters\n")
    W("Exit: V1 (stop = opposite OR level, target = entry + 1x OR range, "
      "time exit 15:45). $10k notional, 1c/side.\n")

    logs = {}
    for mode, label in (("C1", "close > prior close"), ("C2", "close > prior high/low")):
        parts = []
        for sym in SYMBOLS:
            df5, d = load_symbol(sym)
            df5 = build_day_context(df5)
            tr = build_trades(sym, df5, d, mode)
            if len(tr):
                parts.append(apply_v1_exit(tr))
            print(f"{mode} {sym}: {len(tr)} trades")
        allm = pd.concat(parts, ignore_index=True)
        allm = allm[pd.to_datetime(allm["date"]) >= IS_START].reset_index(drop=True)
        logs[mode] = allm
        allm.to_csv(OUT / f"confirm2_trades_{mode}.csv", index=False)

        dt = pd.to_datetime(allm["date"])
        spans = (("IS 2022-2024", (dt >= IS_START) & (dt <= IS_END)),
                 ("OOS 2025+", dt >= OOS_START))
        W(f"\n## Confirmation {mode}: {label}\n")
        W(f"Median bars break->entry: {allm['bars_brk_to_entry'].median():.0f}; "
          f"median chase past level: {allm['chase'].median():.2f} "
          f"(median OR range {allm['or_range'].median():.2f})\n")
        for span_name, m in spans:
            W(f"\n### {span_name}\n")
            W(HDR)
            sub = allm[m]
            for fname, fmask in FILTERS.items():
                W(fmt(fname, stats(sub[fmask(sub)])))

        # what does each filter remove?
        W(f"\n### {mode} filter removal (full period)\n")
        W("| filter | removed | removed win% | removed avg R | removed fakeout% "
          "| kept | kept win% | kept avg R |")
        W("|---|---|---|---|---|---|---|---|")
        for fname, fmask in FILTERS.items():
            if fname == "none":
                continue
            keep = fmask(allm)
            rem, kept = allm[~keep], allm[keep]
            if len(rem) == 0 or len(kept) == 0:
                continue
            W(f"| {fname} | {len(rem)} | {(rem['s1_pnl'] > 0).mean()*100:.1f}% | "
              f"{rem['r_mult'].mean():+.3f} | {rem['fakeout'].mean()*100:.1f}% | "
              f"{len(kept)} | {(kept['s1_pnl'] > 0).mean()*100:.1f}% | "
              f"{kept['r_mult'].mean():+.3f} |")

        # per ticker, no filter
        W(f"\n### {mode} per ticker (no filter, full period)\n")
        W(HDR)
        for sym in SYMBOLS:
            W(fmt(sym, stats(allm[allm["symbol"] == sym])))
        # long vs short
        for dr in ("long", "short"):
            W(fmt(dr, stats(allm[allm["direction"] == dr])))

    # baseline comparison: plain break entry from the original run
    base_p = OUT / "trades_V1.csv"
    if base_p.exists():
        W("\n## Baseline: plain ORB break entry (original trades_V1.csv)\n")
        base = pd.read_csv(base_p)
        dt = pd.to_datetime(base["date"])
        W("| span | trades | win% | avg R | net P&L |")
        W("|---|---|---|---|---|")
        for span_name, m in (("IS", (dt >= IS_START) & (dt <= IS_END)),
                             ("OOS", dt >= OOS_START)):
            g = base[m]
            W(f"| {span_name} | {len(g)} | {(g['s1_pnl'] > 0).mean()*100:.1f}% | "
              f"{g['r_mult'].mean():+.3f} | ${g['s1_pnl'].sum():+,.0f} |")

    # does volume even correlate with fakeouts?  (C1 log)
    W("\n## Do fakeouts have lower volume? (C1 log, full period)\n")
    c1 = logs["C1"]
    W("| group | n | brk RVOL med | conf RVOL med (avg of 2) | cum RVOL med |")
    W("|---|---|---|---|---|")
    c1 = c1.assign(conf_rvol=(c1["c1_rvol"] + c1["c2_rvol"]) / 2)
    for flag, nm in ((True, "fakeout"), (False, "clean")):
        g = c1[c1["fakeout"] == flag]
        W(f"| {nm} | {len(g)} | {g['brk_rvol'].median():.2f} | "
          f"{g['conf_rvol'].median():.2f} | {g['cum_rvol'].median():.2f} |")

    (OUT / "confirm2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written -> out/confirm2_report.md")


if __name__ == "__main__":
    main()
