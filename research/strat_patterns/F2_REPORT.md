# Failed-2 Depth Strategy (F2D-Deep) -- final spec + validation

Built from the P6/P7 pattern study (see REPORT.md). Sweeps run on IS only
(2022-2024); OOS (2025-01..2026-07) touched once, on the locked spec.

## Locked spec: F2D-Deep (long only)
- Universe: SPY, QQQ, IWM (daily signal)
- Signal day D: goes 2D (undercuts yesterday's low without having taken
  yesterday's high), UNDERCUT DEPTH >= 0.35 x ATR14, and closes GREEN with
  close in the top 40% of the day's range (T3 close-confirmation).
- Detectable at the 16:00 close; enter next day at the open.
- Stop: signal day's low (no buffer -- buffers only trade expectancy for win rate).
- Exit (primary, scale-out): half off at +1R, stop to breakeven, runner to
  yesterday's high; time stop close of 2nd day after entry.
- Exit (aggressive alt, hold3): no target, stop active, exit close of 3rd day.
- The bearish mirror (P7/F2U short) FAILED OOS -> not traded.

## Validation (per-spec, IS then OOS)

### F2D-Deep scaleout  (depth >= 0.35 ATR)
| sample | n | win% | avg R | PF | maxDD (R) | trades/mo |
|---|---|---|---|---|---|---|
| IS | 75 | 60.0 | 0.089 | 1.26 | -10.9 | 2.1 |
| OOS | 50 | 66.0 | 0.218 | 1.72 | -6.3 | 2.6 |

OOS per symbol: IWM: n=18, win 55.6%, avg R 0.092; QQQ: n=14, win 64.3%, avg R 0.162; SPY: n=18, win 77.8%, avg R 0.388

### F2D-Deep hold3  (depth >= 0.35 ATR)
| sample | n | win% | avg R | PF | maxDD (R) | trades/mo |
|---|---|---|---|---|---|---|
| IS | 75 | 44.0 | 0.288 | 1.58 | -15.0 | 2.1 |
| OOS | 50 | 54.0 | 0.262 | 1.62 | -9.2 | 2.6 |

OOS per symbol: IWM: n=18, win 44.4%, avg R 0.229; QQQ: n=14, win 50.0%, avg R 0.224; SPY: n=18, win 66.7%, avg R 0.324

### P7-T1 scaleout (rejected)  (depth >= 0.1 ATR)
| sample | n | win% | avg R | PF | maxDD (R) | trades/mo |
|---|---|---|---|---|---|---|
| IS | 409 | 55.3 | 0.105 | 1.24 | -17.5 | 11.4 |
| OOS | 231 | 52.4 | 0.009 | 1.02 | -16.6 | 12.2 |

OOS per symbol: IWM: n=76, win 55.3%, avg R 0.076; QQQ: n=83, win 49.4%, avg R -0.073; SPY: n=72, win 52.8%, avg R 0.033

## Why it works (mechanism)
A deep (>0.35 ATR) undercut that closes strong is a genuine liquidity grab /
trapped-seller day, not a shallow poke. Depth was monotonic in the original
study (18% win at <0.1 ATR vs 56% at >0.5). The T3 close-confirmation filters
out the 50%+ intraday false reclaims that killed T1, and the 2.4R median
target distance is what lets ~60% win rates carry real expectancy.

## Caveats
- n=50 OOS trades; avg R +0.22 is modest. Friction matters: budget 2-3 bps
  on shares; on options, spread + overnight theta eat a meaningful share.
- Sweeps examined ~dozens of IS cells before locking; one OOS pass is honest
  but not immune to selection. Live paper-tracking is the real next gate.
- Code: failed2_strategy.py (events + sims), f2_events.parquet.