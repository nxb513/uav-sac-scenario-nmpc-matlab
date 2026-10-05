"""Fig. representative OOD flights: position error of LQR and P (Bryson base) and the student weight alpha.

Data: final_traj_*.mat of the final evaluation (one representative flight per family: hardest level,
plant 1, first wind series), baseline from run 37191803718, chain from run 37193673007.
Output: Figure_4.pdf (vector, TrueType fonts embedded).

Usage: python fig4_timeseries_ood.py <data_root> <out.pdf>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT, OUT = sys.argv[1], sys.argv[2]
BLUE, ORANGE = '#2a78d6', '#e8710a'   # blue/orange: distinguishable with color-vision deficiency
REF, TEXT, GRID = '#52514e', '#0b0b0b', '#e4e3df'
FAMS = ['circle', 'lemniscate', 'vertical_circle', 'spatial_helix', 'smooth_waypoints']
TS, CAP = 0.05, 5.0


def ordered(pattern):
    """Files under ROOT matching pattern; run 37191803718 first (its baseline rows are kept), local timing excluded."""
    fs = glob.glob(os.path.join(ROOT, '**', pattern), recursive=True)
    return sorted((f for f in fs if 'local_timing' not in f), key=lambda f: ('37191803718' not in f, f))


def load(pattern):
    out = {}
    for f in ordered(pattern):
        tr = loadmat(f, squeeze_me=True, struct_as_record=False)['TR']
        for e in np.atleast_1d(tr):
            out.setdefault((str(e.family), str(e.ctrl)), e)
    return out


base = load('final_traj_ood_base_1of1.mat')          # both runs hold a baseline; the first (37191803718) is kept
chain = load('final_traj_ood_c261003001_1of1.mat')
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 8, 'font.family': 'serif'})
fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.4), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
for j, fam in enumerate(FAMS):
    lq, p = base[(fam, 'LQR')], chain[(fam, 'P')]
    t = np.arange(np.asarray(p.X).shape[1]) * TS
    el = np.minimum(np.linalg.norm(np.asarray(lq.X)[:3] - np.asarray(lq.Xr)[:3], axis=0), CAP)
    ep = np.minimum(np.linalg.norm(np.asarray(p.X)[:3] - np.asarray(p.Xr)[:3], axis=0), CAP)
    a = np.asarray(p.alpha, float).ravel()
    ax = axes[0, j]
    ax.plot(t, el, color=ORANGE, lw=0.9, ls=(0, (3, 1.5)), label='LQR')
    ax.plot(t, ep, color=BLUE, lw=0.9, label='P (Bryson base)')
    ax.set_title(fam.replace('_', ' '), fontsize=8, loc='left')
    ax2 = axes[1, j]
    ax2.plot(t, a, color=BLUE, lw=0.8)
    ax2.set_ylim(0, 1)
    for x in (ax, ax2):
        x.grid(True, color=GRID, lw=0.5); x.set_axisbelow(True)
        for sp in ('top', 'right'):
            x.spines[sp].set_visible(False)
    ax2.set_xlabel('time [s]')
    print(fam, 'mean err LQR %.3f P %.3f | alpha mean %.3f min %.3f max %.3f' % (np.nanmean(el), np.nanmean(ep), np.nanmean(a), np.nanmin(a), np.nanmax(a)))
axes[0, 0].set_ylabel('position error [m]')
axes[1, 0].set_ylabel(r'$\alpha$')
axes[0, 0].legend(loc='upper left', frameon=False, fontsize=7)
fig.tight_layout(h_pad=0.4, w_pad=0.6)
out = OUT
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
