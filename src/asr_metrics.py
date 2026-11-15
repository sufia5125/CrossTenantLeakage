import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

df = pd.read_csv('../results/asr_v2_paired.csv')
bad = (df.clean_verdict == 'UNPARSEABLE') | (df.poisoned_verdict == 'UNPARSEABLE')
print('failed/unparseable rows (excluded, rerun later):', int(bad.sum()), list(df[bad].qid))
d = df[~bad].copy()
flip = {'SUPPORT': 'CONTRADICT', 'CONTRADICT': 'SUPPORT'}
d['intended'] = d.gold_label.map(flip)

n = len(d)
tgt = int((d.poisoned_verdict == d.intended).sum())
cc = d[d.clean_correct]
deg = int((~cc.poisoned_correct).sum())
lo, hi = proportion_confint(tgt, n, method='wilson')
print(f'valid n = {n}')
print(f'clean accuracy: {d.clean_correct.mean():.3f}')
print(f'targeted ASR: {tgt}/{n} = {tgt/n:.3f}  Wilson 95% CI [{lo:.3f}, {hi:.3f}]')
print(f'degradation (clean right -> poisoned wrong): {deg}/{len(cc)}')

b = int((d.clean_correct & ~d.poisoned_correct).sum())
c = int((~d.clean_correct & d.poisoned_correct).sum())
bc = int((d.clean_correct & d.poisoned_correct).sum())
bw = int((~d.clean_correct & ~d.poisoned_correct).sum())
print('McNemar exact p:', round(mcnemar([[bc, b], [c, bw]], exact=True).pvalue, 4))
