"""Mirror the Pine logic exactly (per-symbol rolling gap percentile, self-tracked
levels and averages) and confirm it reproduces the study's lift."""
import glob, os, numpy as np, pandas as pd

LOOKBACK, GAPWIN, GAPPCT, PMMULT, ORMULT = 10, 60, 66.7, 1.75, 3.6
STOP_ATR, TGT_ATR, HOLD = 1.0, 1.5, 5
rows = []
for path in sorted(glob.glob('research/smc/data/*_5m_ext.parquet')):
    sym = os.path.basename(path).split('_')[0]
    d5 = pd.read_parquet(path); d5['d'] = d5.index.date
    rth = d5.between_time('09:30','15:59'); pm = d5.between_time('04:00','09:29')
    daily = rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),
                                 c=('close','last'),v=('volume','sum'))
    daily = daily[daily.v>0]
    pmv = pm.groupby('d').volume.sum(); pmc = pm.groupby('d').close.last()
    tr = pd.concat([daily.h-daily.l,(daily.h-daily.c.shift()).abs(),(daily.l-daily.c.shift()).abs()],axis=1).max(axis=1)
    daily['atr14'] = tr.rolling(14).mean()
    days = list(daily.index)
    # rolling per-symbol gap history, exactly as the indicator builds it
    gaps = pd.Series({d: (pmc.get(d,np.nan)/daily.c.shift().get(d,np.nan)-1)*100 for d in days})
    for k in range(60, len(days)-1):
        day = days[k]; prior = daily.iloc[:k]
        lvl = prior.h.iloc[-LOOKBACK:].max(); pc = prior.c.iloc[-1]; atr = prior.atr14.iloc[-1]
        if not np.isfinite(atr) or atr<=0 or lvl<=pc: continue
        sess = rth[rth.d==day]
        if len(sess)<30: continue
        cross = sess[sess.high>=lvl]
        if len(cross)==0: continue
        t0 = cross.index[0]
        stop, tgt = lvl-STOP_ATR*atr, lvl+TGT_ATR*atr
        fwd = rth[(rth.index>t0)&(rth.d<=days[min(k+HOLD,len(days)-1)])]
        win = np.nan
        for _,b in fwd.iterrows():
            if b.low<=stop: win=0; break
            if b.high>=tgt: win=1; break
        if np.isnan(win): win = 1 if (len(fwd) and fwd.close.iloc[-1]>lvl) else 0
        # --- gates, computed the way the Pine does ---
        gh = gaps.iloc[max(0,k-GAPWIN):k].dropna()
        thr = np.percentile(gh, GAPPCT) if len(gh)>4 else np.nan
        gnow = gaps.iloc[k]
        pmv20 = pmv.reindex(days).iloc[max(0,k-20):k].mean()
        v20 = prior.v.iloc[-20:].mean()
        o15 = sess.iloc[:3]
        rows.append(dict(sym=sym, day=day, win=win,
            g1=int(np.isfinite(gnow) and np.isfinite(thr) and gnow>=thr),
            g2=int(pmv20>0 and pmv.get(day,0)/pmv20 >= PMMULT),
            g3=int(sess.open.iloc[0] >= lvl),
            g4=int(o15.volume.sum()/(v20*15/390) >= ORMULT)))
V = pd.DataFrame(rows); V['score'] = V[['g1','g2','g3','g4']].sum(axis=1)
V.to_csv('research/smc/pine_verify.csv', index=False)
B = V.win.mean()
print("="*92); print(f"Pine-logic replication   n={len(V)}   base {B:.1%}"); print("="*92)
print(f"{'score':<12}{'n':>6}{'sig/name/yr':>13}{'win':>8}")
for s in range(5):
    g = V[V.score==s]
    if len(g)>15: print(f"{s} of 4{'':<6}{len(g):>6}{len(g)/4.6/V.sym.nunique():>13.1f}{g.win.mean():>8.0%}")
for s in (2,3,4):
    g = V[V.score>=s]
    print(f">= {s} of 4{'':<3}{len(g):>6}{len(g)/4.6/V.sym.nunique():>13.1f}{g.win.mean():>8.0%}")
rng = np.random.default_rng(5); base = V.win.values.astype(float)
for s in (3,4):
    a = V[V.score>=s].win.values.astype(float)
    d = rng.choice(a,(8000,len(a)),replace=True).mean(1)-rng.choice(base,(8000,len(base)),replace=True).mean(1)
    print(f"  >={s}: lift {a.mean()-B:+.1%}  95% CI [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]")
print("\nper-name, score >= 3")
g = V[V.score>=3]; print(g.groupby('sym').agg(n=('win','size'),win=('win','mean')).round(3).to_string())
print("\nby year, score >= 3")
print(g.assign(y=pd.to_datetime(g.day).dt.year).groupby('y').agg(n=('win','size'),win=('win','mean')).round(3).to_string())
print("\nindividual gate fire rates:", {c: f"{V[c].mean():.0%}" for c in ['g1','g2','g3','g4']})
tsla = V[(V.sym=='TSLA')]
print("\nTSLA most recent 3 breakout events:")
print(tsla.tail(3)[['day','g1','g2','g3','g4','score','win']].to_string(index=False))
