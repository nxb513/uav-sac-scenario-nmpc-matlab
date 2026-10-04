#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank tests of the final evaluation (experiments/d1_final_eval.m).

For every chain's P against each baseline (LQR, MPC), in each condition, on the SAME
flights (same reference, plant and wind), the difference d = RMSE_P - RMSE_baseline of the
position RMSE is tested with the two-sided Wilcoxon signed-rank test (Wilcoxon 1945;
scipy.stats.wilcoxon, zero differences dropped as in Wilcoxon's original procedure). Two
flight sets are reported:
  (a) completed : flights that both controllers completed (no restart, error never at 5 m);
  (b) all       : every flight, with the recorded capped RMSE (a diverged step counts 5 m).
Reported per test: number of pairs, flights where P is better, median of d, the statistic,
the p-value, the matched-pairs rank-biserial correlation r = (R- - R+)/(R- + R+) (positive
when P has the lower RMSE; Kerby 2014) and the p-value adjusted over all tests of the table
with Holm's step-down method (Holm 1979).
A (cond, tag, seed, flight, ctrl) row found in several files counts once (the first file in
path order), as in summarize_final_eval.py.

Usage: paired_tests.py <input_dir>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

CAP_M = 5.0
KEY = ['cond', 'tag', 'seed', 'flight', 'ctrl']


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return float('nan')
    r = rankdata(np.abs(d))
    rneg, rpos = r[d < 0].sum(), r[d > 0].sum()
    return (rneg - rpos) / (rneg + rpos)


def main(folder):
    fs = sorted(glob.glob(os.path.join(folder, '**', 'final_eval_*.csv'), recursive=True))
    if not fs:
        print('No result CSVs found.'); return 1
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates(subset=KEY, keep='first')
    df['done'] = (df['ok'] == 1) & (df['restarts'] == 0) & (df['pos_max'] < CAP_M)
    rows = []
    chains = sorted(df[df['ctrl'] == 'P']['tag'].unique())
    for cond in ['train', 'ood']:
        d = df[df['cond'] == cond]
        for tag in chains:
            p = d[(d['ctrl'] == 'P') & (d['tag'] == tag)].set_index('flight')
            for base in ['LQR', 'MPC']:
                b = d[(d['ctrl'] == base) & (d['tag'] == 'base')].set_index('flight')
                j = p[['pos_rmse', 'done']].join(b[['pos_rmse', 'done']], lsuffix='_p', rsuffix='_b',
                                                  how='inner')
                for subset, sel in [('completed', j['done_p'] & j['done_b']), ('all', j.index == j.index)]:
                    x = j[sel]
                    dd = (x['pos_rmse_p'] - x['pos_rmse_b']).to_numpy()
                    nz = dd[dd != 0]
                    if len(nz) == 0:
                        continue
                    res = wilcoxon(dd, zero_method='wilcox', alternative='two-sided')
                    rows.append(dict(cond=cond, chain=tag, baseline=base, flights=subset, n=len(dd),
                                     p_better=int((dd < 0).sum()), median_d=float(np.median(dd)),
                                     W=float(res.statistic), p=float(res.pvalue), r=rank_biserial(dd)))
    t = pd.DataFrame(rows)
    t['p_holm'] = holm(t['p'])
    print('| condition | chain | baseline | flights | pairs | P better | median d [m] | W | p | p (Holm) | r |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in t.iterrows():
        print(f"| {r['cond']} | {r['chain']} | {r['baseline']} | {r['flights']} | {r['n']} | {r['p_better']} | "
              f"{r['median_d']:+.4f} | {r['W']:.0f} | {r['p']:.2e} | {r['p_holm']:.2e} | {r['r']:+.3f} |")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'results'))
