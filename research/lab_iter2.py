"""lab_iter2.py -- iteration 2: refine the two leaders + fix orb9ema geometry."""
from strategy_lab import fetch_bars, daily_context, stats, strat_orb_9ema, strat_bos, START

m5 = fetch_bars("QQQ", "5Min", START)
m3 = fetch_bars("QQQ", "3Min", START)
d1 = fetch_bars("QQQ", "1Day", "2021-01-01")
ctx = daily_context(d1)

print("== orb9ema with ORB-mid stop (wider stop, learned geometry) ==")
stats(strat_orb_9ema(m3, ctx, 0.5, False, "orbmid"), "orb9ema raw 0.5R mid")
stats(strat_orb_9ema(m3, ctx, 0.5, True, "orbmid"), "orb9ema gated 0.5R mid")
stats(strat_orb_9ema(m3, ctx, 0.35, True, "orbmid"), "orb9ema gated .35R mid")

print("\n== BOS 3m refinements (leader: gated 0.5R = 67.1%/PF1.13) ==")
stats(strat_bos(m3, ctx, 0.35, True), "bos gated 0.35R")
stats(strat_bos(m3, ctx, 0.25, True), "bos gated 0.25R")
stats(strat_bos(m3, ctx, 0.5, True, "09:45", "12:00"), "bos gated 0.5R AM-only")
stats(strat_bos(m3, ctx, 0.35, True, "09:45", "12:00"), "bos gated .35R AM-only")
stats(strat_bos(m3, ctx, 0.5, True, "13:00", "15:30"), "bos gated 0.5R PM-only")
