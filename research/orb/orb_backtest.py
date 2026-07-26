"""
15-Minute Opening Range Breakout backtest -- exact spec implementation.

Produces per-exit-variant unfiltered trade logs with per-trade filter flags
(F1..F4), so every filter combination is a row-subset of the same log.

Conventions (all decision-time safe, stated in the report):
- Signal bar: first 5-min bar with close strictly beyond the OR level,
  bars starting 09:45 through 13:55 (entry at bar close, <= 14:00 ET).
- If a filter invalidates the first breakout bar, that direction is dead
  for the day (no re-arming on a later bar).
- Trailing stats (14d OR avg, ATR_D, 6-mo percentile, prev close, RVOL
  baseline) use data through the PRIOR session/bar only.
- ATR uses Wilder smoothing. ATR_5m on the continuous RTH 5-min series,
  valued at the signal bar close.
- Exits scanned from the bar after the signal bar. Fill at the level, or
  at the bar open if it gapped past. Bar touches both -> STOP.
- Time exit: close of the 15:40 bar (price at 15:45). Early-close days:
  last bar of the session.
- Costs: 1c/share per side (2c round trip), $0 commission.
- S1 sizing: fractional shares, exactly $10,000 notional.
- S2 sizing: shares = $100 / stop distance, capped at $25,000 notional.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SYMBOLS = ["SPY", "QQQ", "IWM"]
IS_START, IS_END = "2022-01-01", "2024-12-31"
OOS_START = "2025-01-01"

SLIP = 0.01          # per side, per share
ENTRY_CUTOFF = pd.Timestamp("14:00").time()   # entry (bar close) must be <= this
LAST_EXIT_BAR = pd.Timestamp("15:40").time()  # bar whose close is the 15:45 time exit


def wilder_atr(h, l, c, n):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def load_symbol(sym):
    df5 = pd.read_parquet(DATA / f"{sym}_5min.parquet")
    df5 = df5.between_time("09:30", "15:55")           # RTH bar starts only
    daily = pd.read_parquet(DATA / f"{sym}_daily.parquet")
    daily.index = daily.index.normalize()

    # --- daily context, all shifted to decision-time (prior session) ---
    d = pd.DataFrame(index=daily.index)
    d["prev_close"] = daily["close"].shift(1)
    atr_d = wilder_atr(daily["high"], daily["low"], daily["close"], 14)
    pctl20 = atr_d.rolling(126, min_periods=100).quantile(0.20)
    d["atr_d"] = atr_d.shift(1)                        # ATR_D known at entry
    d["f4_skip"] = (atr_d < pctl20).shift(1).fillna(True)  # low-vol regime
    day_range = daily["high"] - daily["low"]
    d["day_range"] = day_range                          # ex-post, reporting only
    d["avg14_day_range"] = day_range.rolling(14).mean().shift(1)

    # --- intraday context ---
    df5 = df5.copy()
    df5["atr5"] = wilder_atr(df5["high"], df5["low"], df5["close"], 14)
    tod = df5.index.time
    df5["tod"] = tod
    df5["avg_vol_tod"] = (
        df5.groupby("tod")["volume"]
        .transform(lambda s: s.rolling(20, min_periods=20).mean().shift(1))
    )
    df5["rvol"] = df5["volume"] / df5["avg_vol_tod"]
    return df5, d


def build_trades(sym, df5, d):
    """One row per (day, direction) candidate signal, unfiltered, with flags."""
    t930, t935, t940, t945 = (pd.Timestamp(x).time() for x in
                              ("09:30", "09:35", "09:40", "09:45"))
    rows = []
    or_ranges = {}   # session date -> OR_range, for the 14-day average

    for date, day in df5.groupby(df5.index.normalize()):
        times = set(day["tod"])
        if not {t930, t935, t940} <= times:
            continue
        orb = day[day["tod"] < t945]
        orh, orl = orb["high"].max(), orb["low"].min()
        or_range = orh - orl
        or_ranges[date] = or_range
        if or_range <= 0 or date not in d.index:
            continue
        dd = d.loc[date]
        if pd.isna(dd["prev_close"]) or pd.isna(dd["atr_d"]):
            continue

        sig_bars = day[(day["tod"] >= t945) & (day["tod"] <= pd.Timestamp("13:55").time())]
        post_945 = day[day["tod"] >= t945]

        for direction in ("long", "short"):
            if direction == "long":
                mask = sig_bars["close"] > orh
            else:
                mask = sig_bars["close"] < orl
            if not mask.any():
                continue
            sig_ts = mask.idxmax()
            sig = day.loc[sig_ts]
            entry = sig["close"]
            atr5 = sig["atr5"]
            if pd.isna(atr5):
                continue

            # bars available for exit scanning (after signal, to 15:40 incl.)
            after = day[(day.index > sig_ts) & (day["tod"] <= LAST_EXIT_BAR)]

            rows.append(dict(
                symbol=sym, date=date.date(), direction=direction,
                signal_ts=sig_ts, entry_time=(sig_ts + pd.Timedelta(minutes=5)).time(),
                entry=entry, orh=orh, orl=orl, or_range=or_range,
                atr_d=dd["atr_d"], atr5=atr5, rvol=sig["rvol"],
                prev_close=dd["prev_close"],
                f4_skip=bool(dd["f4_skip"]),
                day_range=dd["day_range"], avg14_day_range=dd["avg14_day_range"],
                _after=after,
            ))

    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr

    # F1: OR_range < 0.3 x mean of prior 14 sessions' OR_range
    ors = pd.Series(or_ranges).sort_index()
    avg14 = ors.rolling(14, min_periods=14).mean().shift(1)
    f1_map = (ors < 0.3 * avg14) | avg14.isna()
    tr["f1_skip"] = [bool(f1_map.loc[pd.Timestamp(dt, tz=ET)]) for dt in tr["date"]]

    # F2: longs only above prev close, shorts only below
    tr["f2_pass"] = np.where(tr["direction"] == "long",
                             tr["entry"] > tr["prev_close"],
                             tr["entry"] < tr["prev_close"])
    # F3: RVOL >= 1.5 on the signal bar
    tr["f3_pass"] = tr["rvol"].fillna(0) >= 1.5
    tr["trend_day"] = tr["day_range"] > tr["avg14_day_range"]
    return tr


def simulate_exit(row, stop, target):
    """Scan bars after the signal bar. Returns (exit_price, exit_ts, reason)."""
    is_long = row["direction"] == "long"
    after = row["_after"]
    if len(after) == 0:
        return row["entry"], row["signal_ts"], "time"
    for ts, b in after.iterrows():
        o, h, l = b["open"], b["high"], b["low"]
        if is_long:
            if o <= stop:
                return o, ts, "stop"
            if o >= target:
                return o, ts, "target"
            hit_stop, hit_tgt = l <= stop, h >= target
        else:
            if o >= stop:
                return o, ts, "stop"
            if o <= target:
                return o, ts, "target"
            hit_stop, hit_tgt = h >= stop, l <= target
        if hit_stop and hit_tgt:          # both touched -> STOP (spec)
            return stop, ts, "stop_both"
        if hit_stop:
            return stop, ts, "stop"
        if hit_tgt:
            return target, ts, "target"
    last = after.iloc[-1]
    return last["close"], after.index[-1], "time"


def apply_exit_variant(tr, variant):
    out = tr.copy()
    stops, targets = [], []
    for _, r in tr.iterrows():
        e, sign = r["entry"], (1 if r["direction"] == "long" else -1)
        if variant == "V1":
            stop = r["orl"] if sign == 1 else r["orh"]
            target = e + sign * r["or_range"]
        elif variant == "V2":
            stop = e - sign * 0.5 * r["atr5"]
            target = e + sign * 0.15 * r["atr_d"]
        else:  # V3: tighter of the two stops, V1 target
            atr_stop = e - sign * 0.5 * r["atr5"]
            stop = max(r["orl"], atr_stop) if sign == 1 else min(r["orh"], atr_stop)
            target = e + sign * r["or_range"]
        stops.append(stop)
        targets.append(target)
    out["stop"], out["target"] = stops, targets

    res = [simulate_exit(r, r["stop"], r["target"]) for _, r in out.iterrows()]
    out["exit"], out["exit_ts"], out["exit_reason"] = zip(*res)

    sign = np.where(out["direction"] == "long", 1, -1)
    out["risk_ps"] = (out["entry"] - out["stop"]).abs()
    out["gross_ps"] = sign * (out["exit"] - out["entry"])
    out["net_ps"] = out["gross_ps"] - 2 * SLIP
    out["r_mult"] = out["net_ps"] / out["risk_ps"]

    out["s1_shares"] = 10000.0 / out["entry"]
    out["s2_shares"] = np.minimum(100.0 / out["risk_ps"], 25000.0 / out["entry"])
    out["s1_pnl"] = out["net_ps"] * out["s1_shares"]
    out["s1_gross"] = out["gross_ps"] * out["s1_shares"]
    out["s2_pnl"] = out["net_ps"] * out["s2_shares"]
    out["variant"] = variant
    return out.drop(columns=["_after"])


# ---------------------------------------------------------------- reporting

FILTER_SETS = {
    "none":   lambda t: pd.Series(True, index=t.index),
    "F1+F2":  lambda t: ~t["f1_skip"] & t["f2_pass"],
    "F3":     lambda t: t["f3_pass"],
    "F4":     lambda t: ~t["f4_skip"],
    "all":    lambda t: ~t["f1_skip"] & t["f2_pass"] & t["f3_pass"] & ~t["f4_skip"],
}


def stats(trades, sessions, pnl_col="s1_pnl"):
    n = len(trades)
    ndays = len(sessions)
    if n == 0:
        return dict(trades=0, trades_per_day=0, win_rate=np.nan, pf=np.nan,
                    avg_r=np.nan, total_pnl=0.0, mdd=0.0, sharpe=np.nan)
    pnl = trades[pnl_col]
    wins, losses = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    daily = trades.groupby("date")[pnl_col].sum()
    daily = daily.reindex(sessions, fill_value=0.0)
    eq = daily.cumsum()
    mdd = (eq - eq.cummax()).min()
    sharpe = np.nan
    if daily.std() > 0:
        sharpe = daily.mean() / daily.std() * np.sqrt(252)
    return dict(
        trades=n, trades_per_day=n / ndays,
        win_rate=(pnl > 0).mean() * 100,
        pf=(wins / abs(losses)) if losses != 0 else np.inf,
        avg_r=trades["r_mult"].mean(),
        total_pnl=pnl.sum(), mdd=mdd, sharpe=sharpe,
    )


def fmt_row(name, s):
    pf = "inf" if np.isinf(s["pf"]) else f"{s['pf']:.2f}"
    return (f"| {name} | {s['trades']} | {s['trades_per_day']:.2f} | "
            f"{s['win_rate']:.1f}% | {pf} | {s['avg_r']:+.3f} | "
            f"${s['total_pnl']:+,.0f} | ${s['mdd']:,.0f} | {s['sharpe']:.2f} |")


HDR = ("| config | trades | tr/day | win% | PF | avg R | net P&L | maxDD | Sharpe |\n"
       "|---|---|---|---|---|---|---|---|---|")


def main():
    all_trades = {}
    daily_sessions = None
    for sym in SYMBOLS:
        df5, d = load_symbol(sym)
        tr = build_trades(sym, df5, d)
        all_trades[sym] = tr
        if daily_sessions is None:
            daily_sessions = d.index  # same US-equity calendar for all
        print(f"{sym}: {len(tr)} candidate signals")

    variants = {}
    for v in ("V1", "V2", "V3"):
        parts = [apply_exit_variant(all_trades[s], v) for s in SYMBOLS]
        allv = pd.concat(parts, ignore_index=True)
        # drop warm-up-period trades (Nov-Dec 2021 buffer exists only to seed
        # the trailing windows)
        allv = allv[pd.to_datetime(allv["date"]) >= IS_START].reset_index(drop=True)
        variants[v] = allv
        allv.to_csv(OUT / f"trades_{v}.csv", index=False)
        print(f"{v}: {len(allv)} trades logged")

    # convenience masks
    def period(tr, a, b):
        dt = pd.to_datetime(tr["date"])
        return tr[(dt >= a) & (dt <= b)]

    sess = daily_sessions.tz_localize(None) if daily_sessions.tz else daily_sessions
    is_sessions = [s.date() for s in sess if IS_START <= str(s.date()) <= IS_END]
    oos_sessions = [s.date() for s in sess if str(s.date()) >= OOS_START]

    lines = []
    W = lines.append

    W("# ORB-15 Backtest Results\n")
    W(f"IS: {IS_START}..{IS_END} ({len(is_sessions)} sessions) | "
      f"OOS: {OOS_START}..{max(oos_sessions)} ({len(oos_sessions)} sessions)\n")

    for label, sset in (("IN-SAMPLE", is_sessions), ("OUT-OF-SAMPLE", oos_sessions)):
        a, b = (IS_START, IS_END) if label == "IN-SAMPLE" else (OOS_START, "2099-01-01")
        W(f"\n## {label}\n")
        for v in ("V1", "V2", "V3"):
            tr = period(variants[v], a, b)
            W(f"\n### Exit {v} -- filter attribution (S1, combined)\n")
            W(HDR)
            for fname, fmask in FILTER_SETS.items():
                sub = tr[fmask(tr)]
                W(fmt_row(fname, stats(sub, sset)))
            W(f"\n### Exit {v} -- per ticker (S1)\n")
            for fname in ("none", "all"):
                W(f"\n**filters={fname}**\n")
                W(HDR)
                sub = tr[FILTER_SETS[fname](tr)]
                for sym in SYMBOLS:
                    W(fmt_row(sym, stats(sub[sub["symbol"] == sym], sset)))
                W(fmt_row("ALL", stats(sub, sset)))

    # ---- removed-trade analysis (V1, in-sample, vs no-filter baseline) ----
    W("\n## Filter removal analysis (V1, in-sample, S1)\n")
    tr = period(variants["V1"], IS_START, IS_END)
    base = tr
    W("| filter | removed | removed win% | removed avg R | kept | kept win% |")
    W("|---|---|---|---|---|---|")
    for fname, fmask in (("F1", lambda t: ~t["f1_skip"]), ("F2", lambda t: t["f2_pass"]),
                         ("F3", lambda t: t["f3_pass"]), ("F4", lambda t: ~t["f4_skip"])):
        keep = fmask(base)
        rem, kept = base[~keep], base[keep]
        rw = (rem["s1_pnl"] > 0).mean() * 100 if len(rem) else np.nan
        kw = (kept["s1_pnl"] > 0).mean() * 100 if len(kept) else np.nan
        ra = rem["r_mult"].mean() if len(rem) else np.nan
        W(f"| {fname} | {len(rem)} | {rw:.1f}% | {ra:+.3f} | {len(kept)} | {kw:.1f}% |")

    # ---- yearly / monthly (V1 none + all) ----
    for fname in ("none", "all"):
        W(f"\n## V1 filters={fname}: net P&L by year/month (S1, combined)\n")
        t = variants["V1"]
        sub = t[FILTER_SETS[fname](t)].copy()
        dt = pd.to_datetime(sub["date"])
        sub["year"], sub["month"] = dt.dt.year, dt.dt.month
        piv = sub.pivot_table(values="s1_pnl", index="year", columns="month",
                              aggfunc="sum").round(0)
        piv["TOTAL"] = piv.sum(axis=1)
        cols = list(piv.columns)
        W("| year | " + " | ".join(str(c) for c in cols) + " |")
        W("|" + "---|" * (len(cols) + 1))
        for y, r in piv.iterrows():
            cells = ["" if pd.isna(x) else f"{x:+,.0f}" for x in r]
            W(f"| {y} | " + " | ".join(cells) + " |")
        W("\n| year | trades | win% | avg R |")
        W("|---|---|---|---|")
        for y, g in sub.groupby("year"):
            W(f"| {y} | {len(g)} | {(g['s1_pnl'] > 0).mean()*100:.1f}% | "
              f"{g['r_mult'].mean():+.3f} |")

    # ---- trend vs range days (V1) ----
    W("\n## Trend vs range days (V1, S1, in-sample)\n")
    W("| filters | day type | trades | win% | PF | avg R | net P&L |")
    W("|---|---|---|---|---|---|---|")
    tr = period(variants["V1"], IS_START, IS_END)
    for fname in ("none", "all"):
        sub = tr[FILTER_SETS[fname](tr)]
        for flag, nm in ((True, "trend"), (False, "range")):
            g = sub[sub["trend_day"] == flag]
            if len(g) == 0:
                continue
            wins, losses = g.loc[g["s1_pnl"] > 0, "s1_pnl"].sum(), g.loc[g["s1_pnl"] <= 0, "s1_pnl"].sum()
            pf = wins / abs(losses) if losses else np.inf
            W(f"| {fname} | {nm} | {len(g)} | {(g['s1_pnl']>0).mean()*100:.1f}% | "
              f"{pf:.2f} | {g['r_mult'].mean():+.3f} | ${g['s1_pnl'].sum():+,.0f} |")

    # ---- winning exit variant: S1 vs S2 sizing ----
    # winner = best in-sample combined Sharpe across all (variant, filter) configs
    best = None
    for v in ("V1", "V2", "V3"):
        tr = period(variants[v], IS_START, IS_END)
        for fname, fmask in FILTER_SETS.items():
            s = stats(tr[fmask(tr)], is_sessions)
            if not np.isnan(s["sharpe"]) and (best is None or s["sharpe"] > best[2]):
                best = (v, fname, s["sharpe"])
    winner = best[0]
    W(f"\n## Sizing: S1 vs S2 -- winning exit variant = {winner} "
      f"(best IS combined Sharpe: {best[2]:.2f} at filters={best[1]})\n")
    W("Winner criterion: highest in-sample combined Sharpe over the five "
      "filter configs. S2 = $100 risk / stop distance, capped $25k notional.\n")
    W(HDR)
    for label, a, b, sset in (("IS", IS_START, IS_END, is_sessions),
                              ("OOS", OOS_START, "2099-01-01", oos_sessions)):
        tr = period(variants[winner], a, b)
        for fname in ("none", "all"):
            sub = tr[FILTER_SETS[fname](tr)]
            W(fmt_row(f"{label} {fname} S1", stats(sub, sset, "s1_pnl")))
            W(fmt_row(f"{label} {fname} S2", stats(sub, sset, "s2_pnl")))

    # ---- exit reason distribution ----
    W("\n## Exit reason distribution (no filters, 2022->present)\n")
    W("| variant | target | stop | stop(both-touched bar) | time |")
    W("|---|---|---|---|---|")
    for v in ("V1", "V2", "V3"):
        rc = variants[v]["exit_reason"].value_counts()
        W(f"| {v} | {rc.get('target', 0)} | {rc.get('stop', 0)} | "
          f"{rc.get('stop_both', 0)} | {rc.get('time', 0)} |")

    # ---- explicit spec flags ----
    W("\n## Spec flags\n")
    tr_is = period(variants["V1"], IS_START, IS_END)
    tr_oos = period(variants["V1"], OOS_START, "2099-01-01")
    for fname in ("none", "all"):
        si = stats(tr_is[FILTER_SETS[fname](tr_is)], is_sessions)
        so = stats(tr_oos[FILTER_SETS[fname](tr_oos)], oos_sessions)
        gross_is = tr_is[FILTER_SETS[fname](tr_is)]["s1_gross"].sum()
        W(f"- **V1 filters={fname}** -- survives costs in-sample: "
          f"{'YES' if si['total_pnl'] > 0 else 'NO'} "
          f"(gross ${gross_is:+,.0f} -> net ${si['total_pnl']:+,.0f}); "
          f"IS avg R {si['avg_r']:+.3f} vs OOS avg R {so['avg_r']:+.3f}; "
          f"IS PF {si['pf']:.2f} vs OOS PF {so['pf']:.2f}")
        W(f"  - trades/day across all tickers: IS {si['trades_per_day']:.2f}, "
          f"OOS {so['trades_per_day']:.2f}"
          + ("  ** BELOW the 2/day reporting threshold **"
             if min(si["trades_per_day"], so["trades_per_day"]) < 2 else ""))

    # ---- gross vs net (costs) ----
    W("\n## Cost impact (combined, 2022-01-01 -> present)\n")
    W("| variant | filters | gross P&L | net P&L | cost drag |")
    W("|---|---|---|---|---|")
    for v in ("V1", "V2", "V3"):
        for fname in ("none", "all"):
            sub = variants[v][FILTER_SETS[fname](variants[v])]
            g, n = sub["s1_gross"].sum(), sub["s1_pnl"].sum()
            W(f"| {v} | {fname} | ${g:+,.0f} | ${n:+,.0f} | ${g-n:,.0f} |")

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written")

    # equity curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_sessions = is_sessions + oos_sessions
    fig, ax = plt.subplots(figsize=(12, 6))
    for v in ("V1", "V2", "V3"):
        for fname, style in (("none", "--"), ("all", "-")):
            t = variants[v]
            sub = t[FILTER_SETS[fname](t)]
            daily = sub.groupby("date")["s1_pnl"].sum().reindex(all_sessions, fill_value=0)
            eq = daily.cumsum()
            ax.plot(pd.to_datetime(eq.index), eq.values, style, lw=1.2,
                    label=f"{v} {fname}")
            eq.to_csv(OUT / f"equity_{v}_{fname}.csv", header=["cum_pnl"])
    ax.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, alpha=0.6)
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1] * 0.95, " OOS", fontsize=9)
    ax.set_title("ORB-15 cumulative net P&L, $10k/trade (S1), combined SPY+QQQ+IWM")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=130)
    print("equity curves written")


if __name__ == "__main__":
    main()
