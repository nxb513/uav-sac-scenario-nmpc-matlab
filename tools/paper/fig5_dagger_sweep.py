"""Fig. DAgger outcome versus the tuning iteration of the teacher (both runs).

Data: sweep_seed<s>_iter<NNNN>.csv of the DAgger sweeps (runs 37171887012 and 37191810147, teacher checkpoints
9-99) and of the final-test DAgger runs (37190192312 Bryson base, 37186800408 random base, checkpoint 100), with
the student files next to them (candidate spectral radii). A checkpoint found in several folders counts once.
Output: Figure_5.pdf (vector, TrueType fonts embedded) and a printed summary.

Usage: python fig5_dagger_sweep.py <data_root> <out.pdf>
"""
import csv
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
BLUE, ORANGE = '#2a78d6', '#e8710a'   # blue/orange: distinguishable with color-vision deficiency
REF, TEXT, GRID = '#52514e', '#0b0b0b', '#e4e3df'
CHAINS = [(261003001, 'Bryson base'), (261003101, 'random base')]


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, '**', 'sweep_seed*_iter*.csv'), recursive=True)):
        with open(f, newline='') as h:
            for r in csv.DictReader(h):
                key = (int(r['seed']), int(r['sac_iter']))
                if key in rows:
                    continue
                stu = os.path.join(os.path.dirname(f),
                                   os.path.basename(f).replace('sweep_', 'student_').replace('.csv', '.mat'))
                r['n_unstable'] = ''
                if os.path.exists(stu):
                    with h5py.File(stu, 'r') as hh:
                        st = hh['stu/stableAll'][()].ravel()
                        r['n_unstable'] = int((st == 0).sum())
                rows[key] = r
    return rows


rows = load()
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 8, 'font.family': 'serif'})
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
for j, (seed, name) in enumerate(CHAINS):
    rs = sorted((v for (s, _), v in rows.items() if s == seed), key=lambda r: int(r['sac_iter']))
    it = np.array([int(r['sac_iter']) for r in rs])
    st = np.array([float(r['student_val']) for r in rs])
    te = np.array([float(r['teacher_val']) for r in rs])
    lq = np.array([float(r['lqr_val']) for r in rs])
    sel = np.array([int(r['selected']) for r in rs])
    nun = [r['n_unstable'] for r in rs]
    ax = axes[0, j]
    ax.plot(it, lq, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=1, label='LQR')
    ax.plot(it, te, color=ORANGE, lw=0.7, ls=(0, (3, 1.5)), marker='s', ms=1.8, zorder=2, label='teacher')
    ax.plot(it, st, color=BLUE, lw=0.7, marker='o', ms=1.8, zorder=3, label='student (selected candidate)')
    ax.set_title('(%s) %s' % ('ab'[j], name), fontsize=8.5, loc='left', color=TEXT)
    ax2 = axes[1, j]
    ax2.scatter(it[sel == 1], sel[sel == 1], s=9, color=BLUE, zorder=3, label='candidate 1 (teacher-flown data only)')
    ax2.scatter(it[sel > 1], sel[sel > 1], s=9, facecolors='none', edgecolors=BLUE, linewidths=0.8, zorder=3,
                label='later candidate')
    ax2.set_ylim(0.3, 10.7); ax2.set_yticks([1, 4, 7, 10])
    ax2.set_xlabel('tuning iteration of the teacher')
    for x in (ax, ax2):
        x.grid(True, color=GRID, lw=0.5); x.set_axisbelow(True)
        for sp in ('top', 'right'):
            x.spines[sp].set_visible(False)
    n = len(rs)
    print('%s: %d checkpoints (%d-%d), missing %s' % (name, n, it.min(), it.max(),
          sorted(set(range(it.min(), it.max() + 1)) - set(it.tolist()))))
    print('  student < LQR on %d/%d, student < teacher on %d/%d' % ((st < lq).sum(), n, (st < te).sum(), n))
    print('  student val median %.3f (min %.3f max %.3f); teacher median %.3f (min %.3f max %.3f); LQR %.3f' % (
        np.median(st), st.min(), st.max(), np.median(te), te.min(), te.max(), lq.min()))
    print('  LQR val range %.6f-%.6f' % (lq.min(), lq.max()))
    print('  selected candidate 1 on %d/%d; counts %s' % ((sel == 1).sum(), n,
          {k: int((sel == k).sum()) for k in range(1, 11) if (sel == k).any()}))
    known = [x for x in nun if x != '']
    print('  checkpoints with >=1 unstable candidate: %d/%d (student files found %d); unstable candidates total %d' % (
        sum(1 for x in known if x > 0), n, len(known), sum(known)))
    print('  iter100: student %.3f teacher %.3f selected %d' % (st[it == 100][0], te[it == 100][0], sel[it == 100][0]))
axes[0, 0].set_ylabel('validation position RMSE [m]')
axes[1, 0].set_ylabel('selected candidate')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
axes[1, 0].legend(loc='center right', frameon=False, fontsize=7)
fig.tight_layout(h_pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.95))
out = OUT
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
