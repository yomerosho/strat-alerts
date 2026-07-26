"""
Strat pattern backtest -- P1-P5 on 4H & Daily, P6/P7 (failed 2s) on Daily.

Spec: user brief 2026-07-25. Conventions confirmed with user:
  * Continuation targets (P2/P5): highest high of the 5 HTF bars preceding the
    pattern's first bar, must sit beyond the trigger, else the setup is skipped
    (counted as skipped_no_magnitude).
  * P4 target: high of the bar BEFORE the 2D setup bar (mirror for shorts).
  * Setup expiry: trigger must fire during the single HTF bar immediately after
    the pattern completes.

Other documented conventions (spec-driven or repo convention):
  * 4H bars are session-anchored 09:30-13:30 + 13:30-16:00 (bars.py, matches TV).
  * Classification uses completed HTF bars only; triggers are detected intrabar
    on the 5m tape (strict cross: high > trigger for longs, low < trigger shorts).
  * E1 fill = trigger level, or the 5m open when the bar gaps through it.
  * E2 confirmation must occur inside the trigger window (the single HTF bar);
    otherwise the setup is a "missed" trade for that variant.
  * Exits are evaluated on 5m bars. Any single 5m bar touching both stop and
    target counts as STOP (spec's ambiguity rule applied at 5m granularity).
    The E1 entry bar itself uses the same conservative rule.
  * Time stop: close of the 3rd HTF bar after the bar containing entry
    (2nd daily bar for P6/P7).
  * Win = R > 0. Stops fill at the stop price (R = -1 exactly).
  * FTFC at trigger time: trigger price vs today's official daily open and the
    current week's open (Monday's daily open).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from bars import filter_rth, resample_session, resample_weekly, session_close  # noqa: E402
from strat import label_bars  # noqa: E402

ET = "America/New_York"
DATA = ROOT / "research" / "orb" / "data"
OUT = Path(__file__).parent
SYMBOLS = ["SPY", "QQQ", "IWM"]

IS_START = pd.Timestamp("2022-01-01", tz=ET)
OOS_START = pd.Timestamp("2025-01-01", tz=ET)

STRUCT_LOOKBACK = 5  # bars before the pattern for continuation targets
ENTRY_CUTOFF = pd.Timedelta(hours=14)  # P6/P7: no entries after 14:00 ET


def ns(ts) -> np.datetime64:
    """tz-aware Timestamp -> UTC datetime64[ns] comparable with .values arrays."""
    return np.datetime64(pd.Timestamp(ts).value, "ns")


# ---------------------------------------------------------------- data ------

class Tape:
    """5m/15m RTH arrays for fast window scans, plus HTF frames."""

    def __init__(self, sym: str):
        df5 = pd.read_parquet(DATA / f"{sym}_5min.parquet")
        df5 = filter_rth(df5)
        self.df5 = df5
        self.t5 = df5.index.values  # ns int64 comparable
        self.o5 = df5["open"].to_numpy()
        self.h5 = df5["high"].to_numpy()
        self.l5 = df5["low"].to_numpy()
        self.c5 = df5["close"].to_numpy()

        df15 = resample_session(df5, 15)
        self.df15 = df15
        self.t15 = df15.index.values
        self.end15 = df15["bar_end"].values
        self.o15 = df15["open"].to_numpy()
        self.h15 = df15["high"].to_numpy()
        self.l15 = df15["low"].to_numpy()
        self.c15 = df15["close"].to_numpy()

        self.h4 = label_bars(resample_session(df5, 240))

        daily = pd.read_parquet(DATA / f"{sym}_daily.parquet")
        daily.index = daily.index.tz_convert(ET).normalize()
        daily["bar_end"] = [session_close(ts) for ts in daily.index]
        self.daily = label_bars(daily)

        weekly = resample_weekly(daily.drop(columns=["bar_end"]))
        self.weekly = weekly

        # date -> (day_open, week_open) for FTFC
        self.day_open = daily["open"]
        mondays = daily.index.normalize() - pd.to_timedelta(
            daily.index.weekday, unit="D"
        )
        wk_open = weekly["open"].reindex(mondays).to_numpy()
        self.week_open = pd.Series(wk_open, index=daily.index)

        # daily ATR14 (simple TR mean), value at date D uses bars up to D
        tr = np.maximum(
            daily["high"] - daily["low"],
            np.maximum(
                (daily["high"] - daily["close"].shift()).abs(),
                (daily["low"] - daily["close"].shift()).abs(),
            ),
        )
        self.atr14 = tr.rolling(14).mean()

    def w5(self, start, end):
        """[start, end) positions in the 5m arrays."""
        return np.searchsorted(self.t5, ns(start)), np.searchsorted(self.t5, ns(end))

    def ftfc_ok(self, ts: pd.Timestamp, price: float, side: str) -> bool:
        d = ts.normalize()
        try:
            do = self.day_open.loc[d]
            wo = self.week_open.loc[d]
        except KeyError:
            return False
        if side == "L":
            return price > do and price > wo
        return price < do and price < wo


# ------------------------------------------------------------- setups ------

def gen_setups(H: pd.DataFrame, tf: str, sym: str) -> list[dict]:
    lab = H["label"].tolist()
    hi = H["high"].to_numpy()
    lo = H["low"].to_numpy()
    starts = H.index
    ends = H["bar_end"].to_numpy()
    n = len(H)
    setups = []

    def struct_tgt(i_first: int, side: str, trig: float):
        """Highest high (lowest low) of the 5 bars before the pattern's first bar."""
        a = i_first - STRUCT_LOOKBACK
        if a < 0:
            return None
        if side == "L":
            v = hi[a:i_first].max()
            return v if v > trig else None
        v = lo[a:i_first].min()
        return v if v < trig else None

    for i in range(1, n - 1):
        wstart, wend = starts[i + 1], ends[i + 1]
        if pd.Timestamp(wstart) < IS_START - pd.Timedelta(days=45):
            continue

        def add(pat, side, trig, stop, tgt, joint=None):
            skip = None
            if tgt is None:
                skip = "no_magnitude"
            elif side == "L" and tgt <= trig:
                skip = "no_magnitude"
            elif side == "S" and tgt >= trig:
                skip = "no_magnitude"
            elif abs(trig - stop) <= 0:
                skip = "zero_risk"
            setups.append(
                dict(symbol=sym, tf=tf, pattern=pat, side=side, trigger=trig,
                     stop=stop, target=tgt, wstart=wstart, wend=wend,
                     entry_bar_idx=i + 1, joint=joint, skip=skip)
            )

        a, b = lab[i - 1], lab[i]
        if b == "1":
            if a == "2D":
                add("P1", "L", hi[i], lo[i], hi[i - 1])
                add("P2", "S", lo[i], hi[i], struct_tgt(i - 1, "S", lo[i]))
            elif a == "2U":
                add("P1", "S", lo[i], hi[i], lo[i - 1])
                add("P2", "L", hi[i], lo[i], struct_tgt(i - 1, "L", hi[i]))
            elif a == "3":
                key = f"{sym}|{tf}|{i}"
                add("P3", "L", hi[i], lo[i], hi[i - 1], joint=key)
                add("P3", "S", lo[i], hi[i], lo[i - 1], joint=key)
        if b == "2D":
            add("P4", "L", hi[i], lo[i], hi[i - 1])
            if a == "1":
                add("P5", "S", lo[i], hi[i], struct_tgt(i - 1, "S", lo[i]))
        elif b == "2U":
            add("P4", "S", lo[i], hi[i], lo[i - 1])
            if a == "1":
                add("P5", "L", hi[i], lo[i], struct_tgt(i - 1, "L", hi[i]))
    return setups


# ---------------------------------------------------------- simulation ------

def first_cross(tape: Tape, p0: int, p1: int, trig: float, side: str) -> int:
    """First 5m bar in [p0,p1) strictly crossing the trigger. -1 if none."""
    if p0 >= p1:
        return -1
    if side == "L":
        m = tape.h5[p0:p1] > trig
    else:
        m = tape.l5[p0:p1] < trig
    if not m.any():
        return -1
    return p0 + int(np.argmax(m))


def simulate_exit(tape: Tape, k0: int, include_entry_bar: bool, entry: float,
                  stop: float, tgt: float, side: str, tstop_end) -> dict:
    """Walk 5m bars from k0; return exit info. tstop_end = HTF bar_end of time stop."""
    truncated = ns(tstop_end) > tape.t5[-1] + np.timedelta64(5, "m")
    kend = np.searchsorted(tape.t5, ns(tstop_end))
    k = k0 if include_entry_bar else k0 + 1
    risk = abs(entry - stop)
    sgn = 1.0 if side == "L" else -1.0
    mfe = 0.0  # best excursion toward target, in price
    outcome, exit_px, exit_t = None, None, None
    while k < min(kend, len(tape.t5)):
        h, lo_ = tape.h5[k], tape.l5[k]
        mfe = max(mfe, (h - entry) if side == "L" else (entry - lo_))
        if side == "L":
            s_hit, t_hit = lo_ <= stop, h >= tgt
        else:
            s_hit, t_hit = h >= stop, lo_ <= tgt
        if s_hit and t_hit:
            outcome, exit_px = "stop", stop
        elif s_hit:
            outcome, exit_px = "stop", stop
        elif t_hit:
            outcome, exit_px = "target", tgt
        if outcome:
            exit_t = tape.t5[k]
            break
        k += 1
    if outcome is None:
        outcome = "truncated" if truncated else "time"
        k = min(kend, len(tape.t5)) - 1
        exit_px = tape.c5[k]
        exit_t = tape.t5[k]
    r = sgn * (exit_px - entry) / risk
    mag = abs(tgt - entry)
    return dict(outcome=outcome, exit_px=exit_px, exit_time=pd.Timestamp(exit_t, tz="UTC").tz_convert(ET),
                r=r, mfe_pct=min(mfe / mag, 1.0) if mag > 0 else np.nan)


def run_setup(tape: Tape, s: dict, H: pd.DataFrame) -> list[dict]:
    """Simulate E1 / E2-5m / E2-15m for one armed setup. Returns trade rows."""
    rows = []
    p0, p1 = tape.w5(s["wstart"], s["wend"])
    trig, stop, tgt, side = s["trigger"], s["stop"], s["target"], s["side"]
    xp = first_cross(tape, p0, p1, trig, side)

    if xp < 0:
        return rows  # never triggered; setup expired

    trig_time = pd.Timestamp(tape.t5[xp], tz="UTC").tz_convert(ET)
    base = dict(symbol=s["symbol"], tf=s["tf"], pattern=s["pattern"], side=side,
                trigger=trig, stop=stop, target=tgt, when=trig_time)

    # P3 joint handling: opposite side may have crossed first / same bar
    if s["joint"] is not None:
        opp = "S" if side == "L" else "L"
        opp_trig = stop  # for P3 the opposite trigger is this side's stop (the 1's other end)
        xo = first_cross(tape, p0, p1, opp_trig, opp)
        if xo >= 0 and xo < xp:
            return rows  # other direction broke first
        if xo == xp:
            rows.append(dict(base, variant="E1", status="ambiguous_trigger"))
            return rows

    ftfc = tape.ftfc_ok(trig_time, trig, side)

    # time stop: close of 3rd HTF bar after the entry bar
    i_e = s["entry_bar_idx"]
    i_ts = min(i_e + 3, len(H) - 1)
    tstop_end = H["bar_end"].iloc[i_ts]

    # ---- E1
    o = tape.o5[xp]
    e1_px = trig if (o <= trig if side == "L" else o >= trig) else o
    gap_skip = (side == "L" and e1_px >= tgt) or (side == "S" and e1_px <= tgt)
    e1_res = None
    if gap_skip:
        rows.append(dict(base, variant="E1", status="skipped_gap_through_target"))
    else:
        e1_res = simulate_exit(tape, xp, True, e1_px, stop, tgt, side, tstop_end)
        rows.append(dict(base, variant="E1", status="filled", entry=e1_px,
                         entry_time=trig_time, ftfc=ftfc, slippage_r=0.0,
                         **e1_res))

    e1_won_target = e1_res is not None and e1_res["outcome"] == "target"

    # ---- E2-5m: first 5m close beyond trigger, inside the window
    if side == "L":
        m = tape.c5[xp:p1] > trig
    else:
        m = tape.c5[xp:p1] < trig
    if m.any():
        q = xp + int(np.argmax(m))
        e_px = tape.c5[q]
        e_t = pd.Timestamp(tape.t5[q], tz="UTC").tz_convert(ET) + pd.Timedelta(minutes=5)
        if (side == "L" and e_px >= tgt) or (side == "S" and e_px <= tgt):
            rows.append(dict(base, variant="E2-5m", status="skipped_gap_through_target"))
        else:
            res = simulate_exit(tape, q, False, e_px, stop, tgt, side, tstop_end)
            slip = (e_px - trig) / abs(trig - stop) * (1 if side == "L" else -1)
            rows.append(dict(base, variant="E2-5m", status="filled", entry=e_px,
                             entry_time=e_t, ftfc=ftfc, slippage_r=slip, **res))
    else:
        rows.append(dict(base, variant="E2-5m",
                         status="missed_winner" if e1_won_target else "missed"))

    # ---- E2-15m: first 15m close beyond trigger, bar must END inside window
    j0 = np.searchsorted(tape.t15, ns(s["wstart"]))
    j1 = np.searchsorted(tape.end15, ns(s["wend"]), side="right")
    got = False
    for j in range(j0, j1):
        if tape.end15[j] <= tape.t5[xp]:
            continue  # 15m bar fully before the trigger touch
        cx = tape.c15[j]
        if (side == "L" and cx > trig) or (side == "S" and cx < trig):
            e_px = cx
            e_t = pd.Timestamp(tape.end15[j], tz="UTC").tz_convert(ET)
            k0 = np.searchsorted(tape.t5, tape.end15[j])
            if (side == "L" and e_px >= tgt) or (side == "S" and e_px <= tgt):
                rows.append(dict(base, variant="E2-15m", status="skipped_gap_through_target"))
            else:
                res = simulate_exit(tape, k0, True, e_px, stop, tgt, side, tstop_end)
                slip = (e_px - trig) / abs(trig - stop) * (1 if side == "L" else -1)
                rows.append(dict(base, variant="E2-15m", status="filled", entry=e_px,
                                 entry_time=e_t, ftfc=ftfc, slippage_r=slip, **res))
            got = True
            break
    if not got:
        rows.append(dict(base, variant="E2-15m",
                         status="missed_winner" if e1_won_target else "missed"))
    return rows


# ------------------------------------------------------ P6/P7 failed 2s -----

def run_failed_twos(tape: Tape, sym: str) -> list[dict]:
    rows = []
    D = tape.daily
    dates = D.index
    for di in range(1, len(D)):
        d, p = dates[di], dates[di - 1]
        if d < IS_START - pd.Timedelta(days=5):
            continue
        p_hi, p_lo = D["high"].iloc[di - 1], D["low"].iloc[di - 1]
        atr = tape.atr14.iloc[di - 1]
        # 15m bars of day d
        j0 = np.searchsorted(tape.t15, ns(d + pd.Timedelta(hours=9, minutes=30)))
        j1 = np.searchsorted(tape.t15, ns(d + pd.Timedelta(hours=16)))
        if j1 <= j0:
            continue
        day_open = tape.o15[j0]
        cutoff = d + ENTRY_CUTOFF

        for side, brk_lvl, tgt in (("L", p_lo, p_hi), ("S", p_hi, p_lo)):
            # find the 2D (2U) break: first 15m bar beyond brk_lvl with the day
            # still on the 2-side (hasn't taken the other side's level first)
            run_hi, run_lo = -np.inf, np.inf
            brk_j = -1
            for j in range(j0, j1):
                if side == "L":
                    if run_hi > p_hi:  # became outside first -> not a clean 2D
                        break
                    run_hi = max(run_hi, tape.h15[j])
                    if tape.l15[j] < p_lo:
                        if tape.h15[j] > p_hi:
                            break  # outside bar day
                        brk_j = j
                        break
                else:
                    if run_lo < p_lo:
                        break
                    run_lo = min(run_lo, tape.l15[j])
                    if tape.h15[j] > p_hi:
                        if tape.l15[j] < p_lo:
                            break
                        brk_j = j
                        break
            if brk_j < 0:
                continue
            pat = "P6-F2D" if side == "L" else "P7-F2U"

            # T1 / T2 reclaim confirm: first 15m bar (>= break bar) closing back
            # beyond the level, bar end <= 14:00 ET
            for var, lvl in (("T1", brk_lvl), ("T2", day_open)):
                conf_j = -1
                for j in range(brk_j, j1):
                    end_j = pd.Timestamp(tape.end15[j], tz="UTC").tz_convert(ET)
                    if end_j > cutoff:
                        break
                    c = tape.c15[j]
                    if (side == "L" and c > lvl) or (side == "S" and c < lvl):
                        conf_j = j
                        break
                base = dict(symbol=sym, tf="D", pattern=pat, side=side,
                            variant=var, trigger=lvl, target=tgt, when=d)
                if conf_j < 0:
                    rows.append(dict(base, status="no_reclaim"))
                    continue
                e_t_end = tape.end15[conf_j]
                e_px = tape.c15[conf_j]
                k_entry = np.searchsorted(tape.t5, e_t_end)  # first 5m bar after entry
                k_day0 = np.searchsorted(tape.t5, ns(d + pd.Timedelta(hours=9, minutes=30)))
                if side == "L":
                    fail = tape.l5[k_day0:k_entry].min()
                    stop = fail
                    bad = e_px >= tgt or e_px <= stop
                    undercut = (p_lo - fail) / atr if atr and atr > 0 else np.nan
                else:
                    fail = tape.h5[k_day0:k_entry].max()
                    stop = fail
                    bad = e_px <= tgt or e_px >= stop
                    undercut = (fail - p_hi) / atr if atr and atr > 0 else np.nan
                if bad:
                    rows.append(dict(base, status="skipped_no_magnitude"))
                    continue
                i_ts = min(di + 2, len(D) - 1)
                tstop_end = D["bar_end"].iloc[i_ts]
                res = simulate_exit(tape, k_entry, True, e_px, stop, tgt, side, tstop_end)
                # false trigger: new extreme beyond the failure level later same day
                k_eod = np.searchsorted(tape.t5, ns(d + pd.Timedelta(hours=16)))
                if side == "L":
                    false_trig = k_entry < k_eod and tape.l5[k_entry:k_eod].min() < fail
                else:
                    false_trig = k_entry < k_eod and tape.h5[k_entry:k_eod].max() > fail
                e_time = pd.Timestamp(e_t_end, tz="UTC").tz_convert(ET)
                rows.append(dict(base, status="filled", entry=e_px, stop=stop,
                                 entry_time=e_time, undercut_atr=undercut,
                                 false_trigger=false_trig,
                                 ftfc=tape.ftfc_ok(e_time, e_px, side), **res))

        # ---- T3: close-confirmed, from official daily bars
        lab = D["label"].iloc[di]
        o_, h_, l_, c_ = D[["open", "high", "low", "close"]].iloc[di]
        rng = h_ - l_
        if rng <= 0 or di + 1 >= len(D):
            continue
        pos_in_range = (c_ - l_) / rng
        for side, pat in (("L", "P6-F2D"), ("S", "P7-F2U")):
            if side == "L":
                ok = lab == "2D" and c_ > o_ and pos_in_range >= 0.6
                stop, tgt2 = l_, p_hi
            else:
                ok = lab == "2U" and c_ < o_ and pos_in_range <= 0.4
                stop, tgt2 = h_, p_lo
            if not ok:
                continue
            e_px = D["open"].iloc[di + 1]
            base = dict(symbol=sym, tf="D", pattern=pat, side=side, variant="T3",
                        trigger=np.nan, target=tgt2, stop=stop, when=d)
            if (side == "L" and (e_px >= tgt2 or e_px <= stop)) or (
                side == "S" and (e_px <= tgt2 or e_px >= stop)
            ):
                rows.append(dict(base, status="skipped_no_magnitude"))
                continue
            e_time = dates[di + 1] + pd.Timedelta(hours=9, minutes=30)
            k_entry = np.searchsorted(tape.t5, ns(e_time))
            i_ts = min(di + 3, len(D) - 1)
            tstop_end = D["bar_end"].iloc[i_ts]
            res = simulate_exit(tape, k_entry, True, e_px, stop, tgt2, side, tstop_end)
            if side == "L":
                undercut = (p_lo - l_) / atr if atr and atr > 0 else np.nan
            else:
                undercut = (h_ - p_hi) / atr if atr and atr > 0 else np.nan
            rows.append(dict(base, status="filled", entry=e_px, entry_time=e_time,
                             undercut_atr=undercut, false_trigger=False,
                             ftfc=tape.ftfc_ok(e_time, e_px, side), **res))
    return rows


# --------------------------------------------------------------- main -------

def main():
    all_rows, skip_rows = [], []
    for sym in SYMBOLS:
        print(f"== {sym}")
        tape = Tape(sym)
        for tf, H in (("4H", tape.h4), ("D", tape.daily)):
            setups = gen_setups(H, tf, sym)
            n_skip = sum(1 for s in setups if s["skip"])
            print(f"   {tf}: {len(setups)} setups armed ({n_skip} skipped pre-trigger)")
            for s in setups:
                if s["skip"]:
                    skip_rows.append(dict(symbol=sym, tf=tf, pattern=s["pattern"],
                                          side=s["side"], reason=s["skip"]))
                    continue
                all_rows.extend(run_setup(tape, s, H))
        f2 = run_failed_twos(tape, sym)
        print(f"   P6/P7: {len(f2)} rows")
        all_rows.extend(f2)

    trades = pd.DataFrame(all_rows)
    when = pd.to_datetime(trades["when"], utc=True).dt.tz_convert(ET)
    trades["sample"] = np.where(
        when >= OOS_START, "OOS", np.where(when >= IS_START, "IS", "pre")
    )
    trades = trades[trades["sample"] != "pre"].reset_index(drop=True)
    trades.to_parquet(OUT / "trades.parquet")
    pd.DataFrame(skip_rows).to_parquet(OUT / "skips.parquet")
    print(f"\ntotal rows: {len(trades)}  (filled: {(trades.status=='filled').sum()})")
    print(trades.groupby(["pattern", "variant"])["status"].value_counts().head(60))


if __name__ == "__main__":
    main()
