"""
ORB fakeout-fade backtest.

Hypothesis (user): when an ORB break fails -- price closes back inside the
opening range -- fading the failed break is "typically very rewarding,
1.5-3R" with the stop at the fakeout extreme.

Rules (decision-time safe, same framework as orb_backtest.py):
- OR = first three 5-min bars (09:30-09:45). Break bar = first close beyond
  ORH/ORL, 09:45..13:55.
- Fakeout trigger = first subsequent bar closing back INSIDE the range
  (below ORH for a failed upside break). Entry at that bar's close,
  entry bar close <= 14:00 ET. Direction = against the failed break.
- Stop = fakeout extreme (highest high from break bar through trigger bar
  for a failed upside break). Risk = |extreme - entry|.
- Targets tested: 1R / 1.5R / 2R / 3R, plus "opposite OR level".
- Entry variants:
    A  enter on the failure close itself.
    B  after the failure close, additionally require the user's own
       2-close confirmation in the fade direction (each close beyond the
       prior bar's low/high, C2-style), enter on 2nd confirming close.
- Time exit: close of the 15:40 bar. Costs 1c/side, $10k notional.
- MFE_R recorded per trade (max favorable excursion / risk before stop or
  session end) to test the "1.5-3R available" claim directly.
- Conditioning recorded: bars break->fail, spike RVOL (max RVOL of break..
  trigger bars), spike extension past level in OR-range units, trend day
  (ex-post, reporting only).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from orb_backtest import IS_END, IS_START, OOS_START, SLIP, load_symbol, simulate_exit

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SYMBOLS = ["SPY", "QQQ", "IWM"]
ENTRY_CUTOFF = pd.Timestamp("14:00").time()
LAST_EXIT_BAR = pd.Timestamp("15:40").time()
T945 = pd.Timestamp("09:45").time()
T1355 = pd.Timestamp("13:55").time()

TARGETS = {"1R": 1.0, "1.5R": 1.5, "2R": 2.0, "3R": 3.0, "oppOR": None}


def daily_trend_flags(sym):
    d = pd.read_parquet(HERE / "data" / f"{sym}_daily.parquet")
    d.index = d.index.normalize()
    rng = d["high"] - d["low"]
    return (rng > rng.rolling(14).mean().shift(1))


def confirm2_down(day, start_ts, fade_sign):
    """User's C2 confirmation applied in the fade direction, starting at the
    failure bar: 2 consecutive closes beyond the prior bar's low (fade
    short) / high (fade long). Returns entry_ts or None."""
    bars = day[day.index >= start_ts]
    idx = day.index.get_indexer([start_ts])[0]
    streak = 0
    for k, (ts, b) in enumerate(bars.iterrows()):
        j = idx + k
        if j == 0:
            continue
        prev = day.iloc[j - 1]
        ref = prev["low"] if fade_sign == -1 else prev["high"]
        if fade_sign * (b["close"] - ref) > 0:
            streak += 1
        else:
            streak = 0
        if streak >= 2:
            if day.loc[ts, "tod"] > ENTRY_CUTOFF:
                return None
            return ts
    return None


def build_trades(sym, df5, d, variant):
    rows = []
    t930, t935, t940 = (pd.Timestamp(x).time() for x in ("09:30", "09:35", "09:40"))
    for date, day in df5.groupby(df5.index.normalize()):
        if not {t930, t935, t940} <= set(day["tod"]):
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

        for brk_dir in ("up", "down"):
            brk_sign = 1 if brk_dir == "up" else -1
            level = orh if brk_dir == "up" else orl
            mask = brk_sign * (sig_bars["close"] - level) > 0
            if not mask.any():
                continue
            brk_ts = mask.idxmax()

            # first close back inside the range after the break bar
            post = day[day.index > brk_ts]
            back = brk_sign * (post["close"] - level) < 0
            if not back.any():
                continue
            fail_ts = back.idxmax()
            if day.loc[fail_ts, "tod"] > ENTRY_CUTOFF:
                continue

            fade_sign = -brk_sign
            if variant == "A":
                entry_ts = fail_ts
            else:
                entry_ts = confirm2_down(day, fail_ts, fade_sign)
                if entry_ts is None:
                    continue

            ebar = day.loc[entry_ts]
            entry = ebar["close"]

            spike = day[(day.index >= brk_ts) & (day.index <= fail_ts)]
            extreme = spike["high"].max() if brk_dir == "up" else spike["low"].min()
            risk = abs(extreme - entry)
            if risk <= 0:
                continue

            after = day[(day.index > entry_ts) & (day["tod"] <= LAST_EXIT_BAR)]
            if len(after) == 0:
                continue

            # MFE in R before the stop is hit (or session end)
            mfe = 0.0
            for _, b in after.iterrows():
                if brk_dir == "up" and b["high"] >= extreme:
                    break
                if brk_dir == "down" and b["low"] <= extreme:
                    break
                fav = (entry - b["low"]) if fade_sign == -1 else (b["high"] - entry)
                mfe = max(mfe, fav)
            rows.append(dict(
                symbol=sym, date=date.date(),
                direction="short" if fade_sign == -1 else "long",
                signal_ts=entry_ts, brk_ts=brk_ts, fail_ts=fail_ts,
                bars_to_fail=int((fail_ts - brk_ts) / pd.Timedelta(minutes=5)),
                entry=entry, orh=orh, orl=orl, or_range=or_range,
                extreme=extreme, risk_ps=risk,
                ext_or=abs(extreme - level) / or_range,   # spike depth in OR units
                spike_rvol=spike["rvol"].max(),
                mfe_r=mfe / risk,
                _after=after,
            ))
    return pd.DataFrame(rows)


def apply_exits(tr, tgt_name, tgt_mult):
    out = tr.copy()
    sign = np.where(out["direction"] == "long", 1, -1)
    out["stop"] = out["extreme"]
    if tgt_mult is None:
        out["target"] = np.where(sign == 1, out["orh"], out["orl"])  # opposite OR level
        # fade short after failed UP break: target = ORL; fade long: ORH
        out["target"] = np.where(sign == -1, out["orl"], out["orh"])
    else:
        out["target"] = out["entry"] + sign * tgt_mult * out["risk_ps"]

    res = [simulate_exit(r, r["stop"], r["target"]) for _, r in out.iterrows()]
    out["exit"], out["exit_ts"], out["exit_reason"] = zip(*res)
    out["gross_ps"] = sign * (out["exit"] - out["entry"])
    out["net_ps"] = out["gross_ps"] - 2 * SLIP
    out["r_mult"] = out["net_ps"] / out["risk_ps"]
    out["s1_pnl"] = out["net_ps"] * (10000.0 / out["entry"])
    out["tgt"] = tgt_name
    return out.drop(columns=["_after"])


def stats(g):
    n = len(g)
    if n == 0:
        return dict(n=0, win=np.nan, pf=np.nan, avg_r=np.nan, pnl=0.0)
    pnl = g["s1_pnl"]
    w, l = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    return dict(n=n, win=(pnl > 0).mean() * 100,
                pf=(w / abs(l)) if l != 0 else np.inf,
                avg_r=g["r_mult"].mean(), pnl=pnl.sum())


def fmt(name, s):
    if s["n"] == 0:
        return f"| {name} | 0 | - | - | - | - |"
    pf = "inf" if np.isinf(s["pf"]) else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']} | {s['win']:.1f}% | {pf} | "
            f"{s['avg_r']:+.3f} | ${s['pnl']:+,.0f} |")


HDR = ("| config | trades | win% | PF | avg R | net P&L |\n"
       "|---|---|---|---|---|---|")


def main():
    lines = []
    W = lines.append
    W("# ORB fakeout-fade backtest\n")
    W("Fade the failed ORB break on the close back inside the range. Stop = "
      "fakeout extreme. $10k notional, 1c/side, time exit 15:45.\n")

    trend = {s: daily_trend_flags(s) for s in SYMBOLS}
    base_logs = {}

    for variant, vlabel in (("A", "enter on failure close"),
                            ("B", "failure close + 2-close confirmation (C2)")):
        parts = []
        for sym in SYMBOLS:
            df5, d = load_symbol(sym)
            tr = build_trades(sym, df5, d, variant)
            if len(tr):
                tf = trend[sym]
                tr["trend_day"] = [bool(tf.get(pd.Timestamp(dt).tz_localize(tf.index.tz), False))
                                   for dt in tr["date"]]
                parts.append(tr)
            print(f"{variant} {sym}: {len(tr)} setups")
        allv = pd.concat(parts, ignore_index=True)
        allv = allv[pd.to_datetime(allv["date"]) >= IS_START].reset_index(drop=True)
        base_logs[variant] = allv

        W(f"\n## Variant {variant}: {vlabel}\n")
        W(f"Setups: {len(allv)}. Median risk (extreme-entry): "
          f"{allv['risk_ps'].median():.2f} = "
          f"{(allv['risk_ps'] / allv['or_range']).median():.2f} OR units. "
          f"Median bars break->fail: {allv['bars_to_fail'].median():.0f}\n")

        # MFE distribution -- the user's 1.5-3R claim
        W("### How far does the fade actually run before stop/EOD? (MFE in R)\n")
        q = allv["mfe_r"].quantile([0.25, 0.5, 0.75, 0.9]).round(2)
        W(f"- MFE_R quartiles: p25 {q[0.25]}, median {q[0.5]}, p75 {q[0.75]}, "
          f"p90 {q[0.9]}")
        for thr in (1.0, 1.5, 2.0, 3.0):
            W(f"- reaches >= {thr}R before stop: "
            f"{(allv['mfe_r'] >= thr).mean()*100:.1f}%")
        W("")

        dt = pd.to_datetime(allv["date"])
        spans = (("IS 2022-2024", (dt >= IS_START) & (dt <= IS_END)),
                 ("OOS 2025+", dt >= OOS_START))
        for span_name, m in spans:
            W(f"\n### {span_name}\n")
            W(HDR)
            sub = allv[m]
            for tname, tmult in TARGETS.items():
                ex = apply_exits(sub, tname, tmult)
                W(fmt(f"target {tname}", stats(ex)))

        # fixed representative target for the splits: 1.5R
        ex = apply_exits(allv, "1.5R", 1.5)
        W(f"\n### Variant {variant} splits (target 1.5R, full period)\n")
        W(HDR)
        for sym in SYMBOLS:
            W(fmt(sym, stats(ex[ex["symbol"] == sym])))
        for dr in ("long", "short"):
            W(fmt(f"fade {dr}", stats(ex[ex["direction"] == dr])))
        for flag, nm in ((True, "trend day"), (False, "range day")):
            W(fmt(nm, stats(ex[ex["trend_day"] == flag])))
        W(fmt("fail<=3 bars", stats(ex[ex["bars_to_fail"] <= 3])))
        W(fmt("fail 4-6 bars", stats(ex[(ex["bars_to_fail"] > 3) & (ex["bars_to_fail"] <= 6)])))
        W(fmt("fail>6 bars", stats(ex[ex["bars_to_fail"] > 6])))
        W(fmt("spike RVOL>=1.5", stats(ex[ex["spike_rvol"] >= 1.5])))
        W(fmt("spike RVOL<1.5", stats(ex[ex["spike_rvol"] < 1.5])))
        W(fmt("shallow spike (<0.5 OR)", stats(ex[ex["ext_or"] < 0.5])))
        W(fmt("deep spike (>=0.5 OR)", stats(ex[ex["ext_or"] >= 0.5])))

        ex.to_csv(OUT / f"fade_trades_{variant}_15R.csv", index=False)

    (OUT / "fade_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written -> out/fade_report.md")


if __name__ == "__main__":
    main()
