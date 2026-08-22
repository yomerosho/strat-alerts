"""Slice the SMC event table and print the tables that answer the question."""
import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
T = pd.read_csv('research/smc/events.csv')
T['ts'] = pd.to_datetime(T.ts, utc=True).dt.tz_convert('America/New_York')
B = pd.read_csv('research/smc/baseline.csv')
B['ts'] = pd.to_datetime(B.ts, utc=True).dt.tz_convert('America/New_York')


def stats(df, label):
    if len(df) == 0:
        return None
    return dict(setup=label, n=len(df),
                hit1R=f"{df.w1.mean()*100:.0f}%",
                hit2R=f"{df.w2.mean()*100:.0f}%",
                stopped=f"{df.stopped.mean()*100:.0f}%" if 'stopped' in df else '-',
                unresolved=f"{(1-df.w1.mean()-(df.stopped.mean() if 'stopped' in df else 0)):.0%}" if 'stopped' in df else '-',
                exp_1R=round(df.r_1r.mean(), 3),
                exp_2R=round(df.r_2r.mean(), 3),
                exp_scale=round(df.r_so.mean(), 3),
                medMFE=round(df.mfe.median(), 2),
                stopPct=round(df.Rpct.median(), 2))


def show(title, rows):
    rows = [r for r in rows if r]
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- 1
long = T[T.dir == 1]
short = T[T.dir == -1]
show("1  Every 1H structure event, split by how the higher timeframes were leaning",
     [stats(long, "LONG events - all"),
      stats(long[long.stacked == 1], "LONG - 4H+1D+1W all bullish (stacked)"),
      stats(long[long.against == 1], "LONG - 4H+1D+1W all bearish (counter)"),
      stats(short, "SHORT events - all"),
      stats(short[short.stacked == 1], "SHORT - 4H+1D+1W all bearish (stacked)"),
      stats(short[short.against == 1], "SHORT - 4H+1D+1W all bullish (counter)")])

# ---------------------------------------------------------------- 2
sl = long[long.stacked == 1]
show("2  HTF stacked bullish: what BOS vs CHoCH actually does on the 1H",
     [stats(sl[sl.kind == 'BOS'], "BOS  (1H already bullish, continuation)"),
      stats(sl[sl.kind == 'CHoCH'], "CHoCH (1H was bearish, flips up)"),
      stats(B[B.dir == 1], "BASELINE: long any 1H close, stack bullish")])

# ---------------------------------------------------------------- 3
show("3  HTF stacked bullish + did a sell-side liquidity sweep precede it?",
     [stats(sl[(sl.kind == 'CHoCH') & (sl.swept == 1)], "CHoCH  AFTER a sweep"),
      stats(sl[(sl.kind == 'CHoCH') & (sl.swept == 0)], "CHoCH  no sweep"),
      stats(sl[(sl.kind == 'BOS') & (sl.swept == 1)], "BOS    AFTER a sweep"),
      stats(sl[(sl.kind == 'BOS') & (sl.swept == 0)], "BOS    no sweep")])

# ---------------------------------------------------------------- 4
show("4  Sweep + CHoCH long: how much HTF agreement do you actually need?",
     [stats(long[(long.kind == 'CHoCH') & (long.swept == 1) & (long.bw == 1) & (long.bd == 1) & (long.b4 == 1)], "1W + 1D + 4H all bullish"),
      stats(long[(long.kind == 'CHoCH') & (long.swept == 1) & (long.bw == 1) & (long.bd == 1)], "1W + 1D bullish (4H free)"),
      stats(long[(long.kind == 'CHoCH') & (long.swept == 1) & (long.bw == 1)], "1W bullish only"),
      stats(long[(long.kind == 'CHoCH') & (long.swept == 1) & (long.bw != 1)], "1W NOT bullish"),
      stats(long[(long.kind == 'CHoCH') & (long.swept == 1)], "any HTF state")])

# ---------------------------------------------------------------- 5
show("5  Same question on the short side (HTF stacked bearish)",
     [stats(short[(short.stacked == 1) & (short.kind == 'CHoCH') & (short.swept == 1)], "CHoCH after buy-side sweep"),
      stats(short[(short.stacked == 1) & (short.kind == 'CHoCH') & (short.swept == 0)], "CHoCH no sweep"),
      stats(short[(short.stacked == 1) & (short.kind == 'BOS')], "BOS"),
      stats(B[B.dir == -1], "BASELINE: short any 1H close, stack bearish")])

# ---------------------------------------------------------------- 6
best = sl[(sl.kind == 'CHoCH') & (sl.swept == 1)]
print(f"\n{'='*104}\n6  Year-by-year for the headline setup (stacked-bull sweep -> 1H bullish CHoCH)\n{'='*104}")
if len(best):
    yr = best.assign(y=best.ts.dt.year).groupby('y').agg(
        n=('r_so', 'size'), hit1R=('w1', 'mean'), hit2R=('w2', 'mean'),
        exp_scale=('r_so', 'mean'), exp_2R=('r_2r', 'mean'))
    yr['hit1R'] = (yr.hit1R * 100).round(0)
    yr['hit2R'] = (yr.hit2R * 100).round(0)
    print(yr.round(3).to_string())

# ---------------------------------------------------------------- 7
print(f"\n{'='*104}\n7  Where the money is: MFE distribution, stacked-bull longs\n{'='*104}")
for lbl, g in (("CHoCH+sweep", best),
               ("CHoCH no sweep", sl[(sl.kind == 'CHoCH') & (sl.swept == 0)]),
               ("BOS", sl[sl.kind == 'BOS'])):
    if not len(g):
        continue
    q = g.mfe.quantile([.25, .5, .75, .9]).round(2).tolist()
    print(f"{lbl:<16} n={len(g):>3}  MFE p25/p50/p75/p90 = {q}   "
          f"never got to +0.5R: {(g.mfe < 0.5).mean()*100:.0f}%   "
          f"got +3R: {g.w3.mean()*100:.0f}%")

# ---------------------------------------------------------------- 8
print(f"\n{'='*104}\n8  Time of day of the 1H event (stacked-bull longs, scale-out expectancy)\n{'='*104}")
tod = sl.assign(hr=sl.ts.dt.strftime('%H:%M')).groupby('hr').agg(
    n=('r_so', 'size'), hit1R=('w1', 'mean'), exp_scale=('r_so', 'mean'))
tod['hit1R'] = (tod.hit1R * 100).round(0)
print(tod.round(3).to_string())
