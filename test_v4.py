"""
test_v4.py
----------
No network. Synthetic 5-minute tape, run through the real code paths.

Checks:
  1. Session buckets are anchored to 09:30 and contain the open (the v3 bug).
  2. Extended-hours bars are excluded.
  3. An inside bar on 4H arms BOTH edges.
  4. A 5m close through an edge promotes to TIER1; a 15m close -> TIER2.
  5. F2 arms mid-bar and only counts closes AFTER the breach.
"""

import sys
import pandas as pd

from bars import bucket_end, bucket_start, filter_rth, resample_session, session_open
from levels import (
    ARMED, TIER1, TIER2,
    arm_failed_two, arm_inside_bar, evaluate_tiers, split_closed_forming,
)
from strat import label_bars

ET = "America/New_York"
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_5m(day: str, ohlc_by_time: dict) -> pd.DataFrame:
    """Build a 5m frame. ohlc_by_time: {"09:30": (o,h,l,c), ...}"""
    rows, idx = [], []
    for t, (o, h, l, c) in ohlc_by_time.items():
        idx.append(pd.Timestamp(f"{day} {t}", tz=ET))
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).sort_index()
    df["bar_end"] = df.index + pd.Timedelta(minutes=5)
    return df


def flat_session(day: str, o, h, l, c, start="09:30", end="15:55") -> dict:
    """Every 5m bar in [start, end] gets the same OHLC, except we force the
    session's true high/low/open/close onto specific bars."""
    times = pd.date_range(f"{day} {start}", f"{day} {end}", freq="5min", tz=ET)
    out = {}
    mid = len(times) // 2
    for i, t in enumerate(times):
        key = t.strftime("%H:%M")
        if i == 0:
            out[key] = (o, o, o, o)
        elif i == mid:
            out[key] = (o, h, l, o)          # the bar that makes the range
        elif i == len(times) - 1:
            out[key] = (o, max(o, c), min(o, c), c)
        else:
            lo = max(l, min(o, c)); hi = min(h, max(o, c))
            out[key] = (o, hi, lo, o)
    return out


# ==========================================================================
print("\n1. SESSION BUCKETING  (the v3 bug)")
# ==========================================================================

day = "2026-07-09"
# Include pre-market and after-hours to prove they're stripped
ext = make_5m(day, {
    "04:00": (99, 99, 99, 99),
    "08:00": (99, 99, 99, 99),
    "09:30": (100, 101, 99, 100),
    "09:35": (100, 100, 100, 100),
    "13:30": (100, 100, 100, 100),
    "15:55": (100, 100, 100, 100),
    "17:00": (200, 200, 200, 200),
    "19:30": (200, 200, 200, 200),
})

rth = filter_rth(ext)
check("pre/post-market bars dropped", len(rth) == 4, f"{len(rth)} RTH bars kept of {len(ext)}")
check("after-hours price 200 not present", 200 not in rth["high"].values)

b4 = resample_session(rth, 240)
first = b4.iloc[0]
check("first 4H bar starts at 09:30",
      b4.index[0].strftime("%H:%M") == "09:30", b4.index[0].strftime("%H:%M"))
check("first 4H bar CONTAINS the 09:30 open",
      first["open"] == 100 and first["high"] == 101,
      f"open={first['open']} high={first['high']}")
check("only 2 session 4H bars (no phantom evening bar)",
      len(b4) == 2, f"{len(b4)} bars: {[t.strftime('%H:%M') for t in b4.index]}")
check("2nd 4H bar ends at 16:00 (clamped, not 17:30)",
      b4.iloc[1]["bar_end"].strftime("%H:%M") == "16:00",
      b4.iloc[1]["bar_end"].strftime("%H:%M"))

b2 = resample_session(rth, 120)
check("2H buckets anchor 09:30/11:30/13:30/15:30",
      [t.strftime("%H:%M") for t in b2.index] == ["09:30", "13:30", "15:30"],
      str([t.strftime("%H:%M") for t in b2.index]))
check("2H stub bar clamps to 16:00",
      b2.iloc[-1]["bar_end"].strftime("%H:%M") == "16:00")


# ==========================================================================
print("\n2. INSIDE BAR ARMS BOTH EDGES  (4H 2-1-2)")
# ==========================================================================
# 5m tape engineered to produce this 4H sequence:
#   B0  100-105            (reference)
#   B1  101-110   -> 2U    (the '2')
#   B2  103-108   -> 1     (the '1')   <- LAST CLOSED, arms here
#   B3  forming

d1, d2 = "2026-07-06", "2026-07-07"

tape = {}
# d1 AM = B0 : 100-105
tape.update({f"{d1} {k}": v for k, v in flat_session(d1, 101, 105, 100, 104, "09:30", "13:25").items()})
# d1 PM = B1 : 101-110  -> 2U (high > 105, low >= 100)
tape.update({f"{d1} {k}": v for k, v in flat_session(d1, 104, 110, 101, 106, "13:30", "15:55").items()})
# d2 AM = B2 : 103-108  -> INSIDE B1 (108 <= 110, 103 >= 101)
tape.update({f"{d2} {k}": v for k, v in flat_session(d2, 106, 108, 103, 105, "09:30", "13:25").items()})

rows, idx = [], []
for stamp, (o, h, l, c) in tape.items():
    idx.append(pd.Timestamp(stamp, tz=ET))
    rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
df5 = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).sort_index()
df5["bar_end"] = df5.index + pd.Timedelta(minutes=5)

# "Now" = 14:00 on d2, so the 13:30 4H bar is forming and B2 is the last closed
now = pd.Timestamp(f"{d2} 14:00", tz=ET)

bars4 = label_bars(resample_session(df5, 240))
closed4, forming4 = split_closed_forming(bars4, now)

print("   4H bars:")
for ts, r in bars4.iterrows():
    state = "closed" if r["bar_end"] <= now else "FORMING"
    print(f"     {ts:%m-%d %H:%M}  O{r['open']:.0f} H{r['high']:.0f} "
          f"L{r['low']:.0f} C{r['close']:.0f}  [{r['label']}]  {state}")

check("last closed 4H bar is an inside bar",
      closed4["label"].iloc[-1] == "1", f"label={closed4['label'].iloc[-1]}")

price = 105.0
armed = arm_inside_bar("TEST", "4H", closed4, forming4, price)
check("inside bar armed exactly 2 levels", len(armed) == 2, f"got {len(armed)}")

bull = next((l for l in armed if l.direction == "bull"), None)
bear = next((l for l in armed if l.direction == "bear"), None)
check("bull level = inside bar HIGH", bull and bull.level == 108.0, f"{bull.level if bull else None}")
check("bear level = inside bar LOW", bear and bear.level == 103.0, f"{bear.level if bear else None}")
check("bull trigger_side is 'above'", bull.trigger_side == "above")
check("bear trigger_side is 'below'", bear.trigger_side == "below")
check("bull pattern labelled 2-1-2", bull.pattern == "2-1-2", bull.pattern)
check("bull invalidation = the other edge", bull.invalidation == 103.0)
check("bull target = prior bar high (110)", bull.target == 110.0, str(bull.target))


# ==========================================================================
print("\n3. TIER PROMOTION")
# ==========================================================================
# Append post-arm 5m tape that closes ABOVE 107 -> should hit TIER1 then TIER2
post = {}
for t in pd.date_range(f"{d2} 13:30", f"{d2} 13:55", freq="5min", tz=ET):
    post[t] = (107.5, 109.5, 107.0, 109.0)   # closes 109 > 108

extra = pd.DataFrame(
    [{"open": o, "high": h, "low": l, "close": c, "volume": 100} for (o, h, l, c) in post.values()],
    index=pd.DatetimeIndex(list(post.keys())),
)
extra["bar_end"] = extra.index + pd.Timedelta(minutes=5)
df5b = pd.concat([df5, extra]).sort_index()

bars15 = resample_session(df5b, 15)

bull2 = arm_inside_bar("TEST", "4H", closed4, forming4, 109.0)[0]
evaluate_tiers(bull2, df5b, bars15, now)

check("5m close above 108 -> promoted past ARMED", bull2.tier != ARMED, bull2.tier)
check("TIER2 reached (15m also closed above)", bull2.tier == TIER2, bull2.tier)
check("tier1_time recorded", bull2.tier1_time is not None, str(bull2.tier1_time))
check("tier2_time >= tier1_time", bull2.tier2_time >= bull2.tier1_time)
check("distance_pct positive when through", bull2.distance_pct > 0, f"{bull2.distance_pct:+.2f}%")

# A level whose confirming close came BEFORE arm_time must not count
bear2 = arm_inside_bar("TEST", "4H", closed4, forming4, 109.0)[1]
evaluate_tiers(bear2, df5b, bars15, now)
check("bear side did NOT trigger (price went up)", bear2.tier == ARMED, bear2.tier)


# ==========================================================================
print("\n4. FAILED-2  (arms mid-bar, breach time respected)")
# ==========================================================================
# Prior closed 4H bar high = 110. Forming bar pokes to 112, then falls back.
# A 5m bar that closed below 110 BEFORE the poke must not count as the failure.

d = "2026-07-10"
f2 = {}
# 09:30-13:25 : the prior 4H bar, high 110, low 100
for t in pd.date_range(f"{d} 09:30", f"{d} 13:25", freq="5min", tz=ET):
    f2[t] = (105, 106, 104, 105)
f2[pd.Timestamp(f"{d} 11:00", tz=ET)] = (105, 110, 100, 105)   # sets the range

# 13:30 forming bar:
f2[pd.Timestamp(f"{d} 13:30", tz=ET)] = (105, 106, 104, 105)   # below 110, BEFORE any poke
f2[pd.Timestamp(f"{d} 13:35", tz=ET)] = (106, 112, 106, 111)   # THE POKE above 110
f2[pd.Timestamp(f"{d} 13:40", tz=ET)] = (111, 111, 108, 109)   # closes back below 110 -> failure
f2[pd.Timestamp(f"{d} 13:45", tz=ET)] = (109, 109, 107, 108)

rows = [{"open": o, "high": h, "low": l, "close": c, "volume": 100} for (o, h, l, c) in f2.values()]
df5f = pd.DataFrame(rows, index=pd.DatetimeIndex(list(f2.keys()))).sort_index()
df5f["bar_end"] = df5f.index + pd.Timedelta(minutes=5)

now_f2 = pd.Timestamp(f"{d} 13:52", tz=ET)
bars4f = label_bars(resample_session(df5f, 240))
closed4f, forming4f = split_closed_forming(bars4f, now_f2)

check("prior 4H bar high is 110", float(closed4f["high"].iloc[-1]) == 110.0)
check("forming 4H bar poked to 112", float(forming4f["high"]) == 112.0)

lv = arm_failed_two("TEST", "4H", closed4f, forming4f, df5f, current_price=108.0)
check("F2D armed", len(lv) == 1 and lv[0].pattern == "F2D", str([x.pattern for x in lv]))

f2d = lv[0]
check("F2D level = prior bar high", f2d.level == 110.0, str(f2d.level))
check("F2D trigger_side is 'below' (reversal, not breakout)", f2d.trigger_side == "below")
check("F2D direction bear", f2d.direction == "bear")
check("arm_time is the BREACH bar's end (13:40), not the setup bar start",
      f2d.arm_time == pd.Timestamp(f"{d} 13:40", tz=ET), str(f2d.arm_time))

bars15f = resample_session(df5f, 15)
evaluate_tiers(f2d, df5f, bars15f, now_f2)
check("F2D promoted to TIER1 by the 13:40 close below 110",
      f2d.tier in (TIER1, TIER2), f2d.tier)
check("tier1_time is 13:45 (end of the 13:40 bar), after the breach",
      f2d.tier1_time == pd.Timestamp(f"{d} 13:45", tz=ET), str(f2d.tier1_time))


# ==========================================================================
print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
