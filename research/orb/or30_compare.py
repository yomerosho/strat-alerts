"""
ORB duration comparison: 15-min vs 30-min opening range, same confirmed
entry (C2 2-close confirmation + vol_expand), with and without the gate.

Same framework as confirm2_backtest.py / c2_gated_test.py: V1 exits
(stop = opposite OR level, target = entry + 1x OR range, time exit 15:45),
$10k notional, 1c/side, IS 2022-2024 / OOS 2025+. Gate rows are restricted
to the four_gate.csv overlap (starts 2022-03-30).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from orb_backtest import IS_END, IS_START, OOS_START, SLIP, load_symbol, simulate_exit

HERE = Path(__file__).parent
OUT = HERE / "out"
SYMBOLS = ["SPY", "QQQ", "IWM"]
ENTRY_CUTOFF = pd.Timestamp("14:00").time()
LAST_EXIT_BAR = pd.Timestamp("15:40").time()
T1355 = pd.Timestamp("13:55").time()


def add_vol_context(df5):
    df5 = df5.copy()
    day_key = df5.index.normalize()
    df5["prior6_vol"] = (
        df5.groupby(day_key)["volume"]
        .transform(lambda s: s.rolling(6, min_periods=3).mean().shift(1))
    )
    return df5


def find_confirmation(day, start_ts, direction, level):
    """C2: 2 consecutive closes beyond the prior bar's high (long) / low
    (short), from the break bar onward. Returns entry_ts or None."""
    sign = 1 if direction == "long" else -1
    bars = day[day.index >= start_ts]
    idx = day.index.get_indexer([start_ts])[0]
    streak = 0
    for k, (ts, b) in enumerate(bars.iterrows()):
        j = idx + k
        if j == 0:
            continue
        prev = day.iloc[j - 1]
        ref = prev["high"] if sign == 1 else prev["low"]
        if sign * (b["close"] - ref) > 0:
            streak += 1
        else:
            streak = 0
        if streak >= 2:
            if day.loc[ts, "tod"] > ENTRY_CUTOFF:
                return None
            if sign * (day.loc[ts, "close"] - level) <= 0:
                return None
            return ts
    return None


def build_trades(sym, df5, d, or_end):
    t_end = pd.Timestamp(or_end).time()
    rows = []
    t930 = pd.Timestamp("09:30").time()
    for date, day in df5.groupby(df5.index.normalize()):
        orb = day[(day["tod"] >= t930) & (day["tod"] < t_end)]
        n_or_bars = int((pd.Timestamp(or_end) - pd.Timestamp("09:30")).seconds / 300)
        if len(orb) < n_or_bars:
            continue
        orh, orl = orb["high"].max(), orb["low"].min()
        or_range = orh - orl
        if or_range <= 0 or date not in d.index:
            continue
        dd = d.loc[date]
        if pd.isna(dd["prev_close"]) or pd.isna(dd["atr_d"]):
            continue

        sig_bars = day[(day["tod"] >= t_end) & (day["tod"] <= T1355)]

        for direction in ("long", "short"):
            sign = 1 if direction == "long" else -1
            level = orh if direction == "long" else orl
            mask = sign * (sig_bars["close"] - level) > 0
            if not mask.any():
                continue
            brk_ts = mask.idxmax()
            entry_ts = find_confirmation(day, brk_ts, direction, level)
            if entry_ts is None:
                continue
            ebar = day.loc[entry_ts]
            after = day[(day.index > entry_ts) & (day["tod"] <= LAST_EXIT_BAR)]

            rows.append(dict(
                symbol=sym, date=date.date(), direction=direction,
                signal_ts=entry_ts, entry=ebar["close"],
                orh=orh, orl=orl, or_range=or_range,
                vol_expand=bool(ebar["volume"] > ebar["prior6_vol"])
                    if not pd.isna(ebar["prior6_vol"]) else False,
                _after=after,
            ))
    return pd.DataFrame(rows)


def apply_v1(tr):
    out = tr.copy()
    sign = np.where(out["direction"] == "long", 1, -1)
    out["stop"] = np.where(sign == 1, out["orl"], out["orh"])
    out["target"] = out["entry"] + sign * out["or_range"]
    res = [simulate_exit(r, r["stop"], r["target"]) for _, r in out.iterrows()]
    out["exit"], out["exit_ts"], out["exit_reason"] = zip(*res)
    out["risk_ps"] = (out["entry"] - out["stop"]).abs()
    out["net_ps"] = sign * (out["exit"] - out["entry"]) - 2 * SLIP
    out["r_mult"] = out["net_ps"] / out["risk_ps"]
    out["s1_pnl"] = out["net_ps"] * (10000.0 / out["entry"])
    return out.drop(columns=["_after"])


def stats(g):
    n = len(g)
    if n == 0:
        return "| 0 | - | - | - | - |"
    pnl = g["s1_pnl"]
    w, l = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    pf = w / abs(l) if l != 0 else np.inf
    pf = "inf" if np.isinf(pf) else f"{pf:.2f}"
    return (f"| {n} | {(pnl > 0).mean()*100:.1f}% | {pf} | "
            f"{g['r_mult'].mean():+.3f} | ${pnl.sum():+,.0f} |")


def main():
    gates = pd.read_csv(HERE.parent / "smc" / "four_gate.csv")
    gates = gates[gates["sym"].isin(SYMBOLS)].rename(columns={"sym": "symbol"})
    gates["date"] = pd.to_datetime(gates["day"]).dt.date
    gates = gates[["symbol", "date", "score"]]

    logs = {}
    for or_end, tag in (("09:45", "OR15"), ("10:00", "OR30")):
        parts = []
        for sym in SYMBOLS:
            df5, d = load_symbol(sym)
            df5 = add_vol_context(df5)
            tr = build_trades(sym, df5, d, or_end)
            if len(tr):
                parts.append(apply_v1(tr))
            print(f"{tag} {sym}: {len(tr)} trades")
        allv = pd.concat(parts, ignore_index=True)
        allv = allv[pd.to_datetime(allv["date"]) >= IS_START].reset_index(drop=True)
        allv = allv.merge(gates, on=["symbol", "date"], how="left")
        logs[tag] = allv

    lines = ["# ORB duration: 15 vs 30 minutes (C2 confirmation entry)\n",
             "V1 exits, $10k, 1c/side. Gate rows use the four_gate.csv overlap.\n"]
    W = lines.append
    for tag in ("OR15", "OR30"):
        t = logs[tag]
        dt = pd.to_datetime(t["date"])
        W(f"\n## {tag}  (median OR range {t['or_range'].median():.2f})\n")
        W("| span | config | trades | win% | PF | avg R | net P&L |")
        W("|---|---|---|---|---|---|---|")
        for span, m in (("IS", (dt >= IS_START) & (dt <= IS_END)),
                        ("OOS", dt >= OOS_START)):
            sub = t[m]
            has_gate = ~sub["score"].isna()
            vx = sub["vol_expand"]
            W(f"| {span} | C2 only {stats(sub)}")
            W(f"| {span} | C2+vx {stats(sub[vx])}")
            W(f"| {span} | C2+vx gate>=1 {stats(sub[vx & has_gate & (sub['score'] >= 1)])}")

    (OUT / "or30_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written -> out/or30_report.md")


if __name__ == "__main__":
    main()
