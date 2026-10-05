"""Fig. paired RMSE ratio: P / LQR on the same flights (final evaluation, measured wind).

Data: final-evaluation CSVs of the two CI runs (37191803718 first, so its baseline rows
are kept; the duplicate baseline of 37193673007 is dropped), as in docs/D1_runs.md.
Output: Figure_3.pdf (vector, TrueType fonts embedded).

Usage: python fig3_paired_rmse.py <data_root> <out.pdf>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
KEY = ['cond', 'tag', 'seed', 'flight', 'ctrl']
BLUE = '#2a78d6'
REF = '#52514e'
TEXT = '#0b0b0b'
GRID = '#e4e3df'
CAP = 5.0

def ordered(pattern):
    """Files under ROOT matching pattern; run 37191803718 first (its baseline rows are kept), local timing excluded."""
    fs = glob.glob(os.path.join(ROOT, '**', pattern), recursive=True)
    return sorted((f for f in fs if 'local_timing' not in f), key=lambda f: ('37191803718' not in f, f))


files = ordered('final_eval_*.csv')
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(subset=KEY, keep='first')
d['done'] = (d['ok'] == 1) & (d['restarts'] == 0) & (d['pos_max'] < CAP)

plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': REF, 'axes.labelcolor': TEXT,
                     'xtick.color': TEXT, 'ytick.color': TEXT})
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), sharex=True, sharey=True)
chains = [('c261003001', 'P, Bryson base'), ('c261003101', 'P, random base')]
for i, (tag, name) in enumerate(chains):
    for j, cond in enumerate(['train', 'ood']):
        ax = axes[i, j]
        x = d[(d['cond'] == cond) & (d['ctrl'] == 'LQR') & (d['tag'] == 'base')].set_index('flight')
        y = d[(d['cond'] == cond) & (d['ctrl'] == 'P') & (d['tag'] == tag)].set_index('flight')
        jn = x[['pos_rmse', 'done']].join(y[['pos_rmse', 'done']], lsuffix='_l', rsuffix='_p', how='inner')
        both = jn['done_l'] & jn['done_p']
        r = jn['pos_rmse_p'] / jn['pos_rmse_l']
        ax.axhline(1.0, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.scatter(jn.loc[both, 'pos_rmse_l'], r[both], s=14, color=BLUE, alpha=0.75, linewidths=0, zorder=3,
                   label='both completed (%d)' % both.sum())
        ax.scatter(jn.loc[~both, 'pos_rmse_l'], r[~both], s=26, facecolors='none', edgecolors=BLUE,
                   linewidths=1.0, marker='s', zorder=3, label='restart or 5 m cap in either (%d)' % (~both).sum())
        better = int((r < 1).sum())
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.06, 6.0); ax.set_ylim(0.2, 40.0)
        ax.set_title('(%s) %s, %s' % ('abcd'[2 * i + j], name, cond), fontsize=8.5, loc='left', color=TEXT)
        ax.text(0.98, 0.96, 'ratio < 1 on %d of %d flights\nmedian ratio (both completed) %.2f'
                % (better, len(jn), float(np.median(r[both]))), transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, color=TEXT)
        ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(loc='lower left', frameon=False, fontsize=7, handletextpad=0.3)
        if i == 1:
            ax.set_xlabel('LQR position RMSE [m]')
        if j == 0:
            ax.set_ylabel('RMSE ratio P / LQR (same flight)')
fig.tight_layout()
out = OUT
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
