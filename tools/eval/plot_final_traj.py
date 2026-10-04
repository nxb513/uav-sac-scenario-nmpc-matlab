#!/usr/bin/env python3
"""Trajectory figures of the final evaluation (experiments/d1_final_eval.m).

Reads every final_traj_<cond>_<tag>_<k>of<K>.mat below <input_dir> (one representative
flight per family: hardest level, plant 1, first wind; tag 'base' = LQR/MPC, tag
'c<seed>' = Teacher/P of that chain) and writes, per condition and chain:

  traj_<cond>_c<seed>.png   3D flight paths of the reference, LQR, MPC, Teacher and P
                            (one column per family) and their position error over time;
  alpha_<cond>_c<seed>.png  where the student acts in P: the 3D path of P coloured by the
                            recorded weight alpha = c_S * g_L(c_LQR) (capped by the
                            contraction bound; 0 = pure LQR), alpha over time, and the
                            position error of LQR and P over time.

Entries found in several input folders (for example the baseline flown in two runs) are
counted once. The 3D panels use equal axis scales over the reference extent plus a margin
of 15 % of its largest span; a path leaving that box is cut there and named in the panel
title. Position errors are capped at 5 m as in the metrics (d1_track_err).
Usage: plot_final_traj.py <input_dir> <output_dir>
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.io import loadmat

# reference palette (categorical slots 1-4 in fixed order; validated light mode)
COL = {'P': '#2a78d6', 'Teacher': '#eb6834', 'LQR': '#1baf7a', 'MPC': '#eda100'}
LS = {'P': '-', 'Teacher': '-', 'LQR': '-', 'MPC': (0, (5, 2))}
LAB = {'P': 'P (proposed)', 'Teacher': 'Teacher (oracle, knows wind)', 'LQR': 'LQR', 'MPC': 'MPC (N = 20)'}
ORDER = ['LQR', 'MPC', 'Teacher', 'P']
REF = '#52514e'      # secondary ink: reference path
TEXT = '#0b0b0b'
GRID = '#e4e3df'
FAMS = ['circle', 'lemniscate', 'vertical_circle', 'spatial_helix', 'smooth_waypoints']
# sequential blue ramp 100 -> 700 for alpha
ALPHA_CMAP = LinearSegmentedColormap.from_list('alpha', ['#86b6ef', '#3987e5', '#256abf', '#184f95',
                                                         '#0d366b'])
TS = 0.05
CAP_M = 5.0
MARGIN = 0.15


def box_of(Xr):
    lo, hi = np.nanmin(Xr[:3], axis=1), np.nanmax(Xr[:3], axis=1)
    m = MARGIN * np.max(hi - lo)
    return lo - m, hi + m


def clip(X, box):
    """Path inside the box (points outside -> NaN) and whether any point was outside."""
    P = X[:3].copy()
    out = np.any((P < box[0][:, None]) | (P > box[1][:, None]), axis=0) & np.all(np.isfinite(P), axis=0)
    P[:, out] = np.nan
    return P, bool(out.any())


def set_box(ax, box):
    ax.set_xlim(box[0][0], box[1][0]); ax.set_ylim(box[0][1], box[1][1]); ax.set_zlim(box[0][2], box[1][2])
    ax.set_box_aspect(tuple(box[1] - box[0]))


def perr(e):
    return np.minimum(np.linalg.norm(e['X'][:3] - e['Xr'][:3], axis=0), CAP_M)


def style2(x):
    x.grid(True, color=GRID, lw=0.8); x.set_axisbelow(True); x.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        x.spines[sp].set_visible(False)


def load(src):
    """{(cond, flight): {'family', 'id', 'ctrl:<tag>': entry}} with entry = dict(X, Xr, alpha)."""
    out = {}
    pat = re.compile(r'final_traj_(train|ood)_(.+?)_(\d+)of(\d+)\.mat$')
    for f in sorted(glob.glob(os.path.join(src, '**', 'final_traj_*.mat'), recursive=True)):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        cond, tag = m.group(1), m.group(2)
        tr = loadmat(f, squeeze_me=True, struct_as_record=False).get('TR')
        if tr is None:
            continue
        for e in np.atleast_1d(tr):
            if not hasattr(e, 'ctrl'):
                continue
            k = (cond, int(e.flight))
            d = out.setdefault(k, {'family': str(e.family), 'id': str(e.id)})
            d.setdefault('%s:%s' % (e.ctrl, tag), dict(X=np.asarray(e.X, float), Xr=np.asarray(e.Xr, float),
                                                      alpha=np.asarray(e.alpha, float).ravel()))
    return out


def style(ax):
    ax.set_xlabel('x [m]', color=TEXT, fontsize=8, labelpad=-2)
    ax.set_ylabel('y [m]', color=TEXT, fontsize=8, labelpad=-2)
    ax.set_zlabel('z [m]', color=TEXT, fontsize=8, labelpad=-2)
    ax.tick_params(labelsize=7, pad=-1)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_major_locator(MaxNLocator(4))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor('#fcfcfb'); a._axinfo['grid']['color'] = GRID


def get(d, ctrl, tag):
    return d.get('%s:%s' % (ctrl, tag))


def fig_paths(fl, cond, tag, out):
    n = len(fl)
    fig = plt.figure(figsize=(4.2 * n, 7.6))
    gs = fig.add_gridspec(2, n, height_ratios=[2.2, 1], hspace=0.12, wspace=0.25)
    for i, (fam, d) in enumerate(fl):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax2 = fig.add_subplot(gs[1, i])
        Xr = next(v for k, v in d.items() if ':' in k)['Xr']
        box = box_of(Xr)
        ax.plot(Xr[0], Xr[1], Xr[2], color=REF, lw=1.5, ls=(0, (3, 2)), label='Reference')
        left = []
        for c in ORDER:
            e = get(d, c, 'base' if c in ('LQR', 'MPC') else tag)
            if e is None:
                continue
            P, o = clip(e['X'], box)
            if o:
                left.append(c)
            ax.plot(P[0], P[1], P[2], color=COL[c], lw=1.4, ls=LS[c], label=LAB[c])
            ax2.plot(np.arange(e['X'].shape[1]) * TS, perr(e), color=COL[c], lw=1.1, ls=LS[c], label=LAB[c])
        set_box(ax, box)
        ax.set_title(fam + ('\n(left the plotted box: %s)' % ', '.join(left) if left else ''),
                     color=TEXT, fontsize=10, loc='left')
        style(ax)
        style2(ax2)
        ax2.set_xlabel('time [s]', color=TEXT, fontsize=9)
        if i == 0:
            ax2.set_ylabel('position error [m]\n(capped at 5 m)', color=TEXT, fontsize=9)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=len(lab), frameon=False, fontsize=9)
    fig.suptitle('%s, chain %s: representative flight per family (hardest level, plant 1, first wind)'
                 % (cond, tag), color=TEXT, fontsize=10, y=0.003, va='bottom')
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.1)
    fig.savefig(os.path.join(out, 'traj_%s_%s.png' % (cond, tag)), dpi=150)
    plt.close(fig)


def fig_alpha(fl, cond, tag, out):
    n = len(fl)
    fig = plt.figure(figsize=(4.2 * n, 9.6))
    gs = fig.add_gridspec(3, n, height_ratios=[2.2, 1, 1], hspace=0.22, wspace=0.25)
    norm = Normalize(0, 1)
    for i, (fam, d) in enumerate(fl):
        p = get(d, 'P', tag)
        lq = get(d, 'LQR', 'base')
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax3 = fig.add_subplot(gs[2, i])
        ax2 = fig.add_subplot(gs[1, i], sharex=ax3)
        ax.set_title(fam, color=TEXT, fontsize=10, loc='left')
        if p is None:
            continue
        X, Xr, a = p['X'], p['Xr'], p['alpha']
        box = box_of(Xr)
        P, o = clip(X, box)
        if o:
            ax.set_title(fam + '\n(P left the plotted box)', color=TEXT, fontsize=10, loc='left')
        ax.plot(Xr[0], Xr[1], Xr[2], color=REF, lw=1.2, ls=(0, (3, 2)))
        pts = P.T.reshape(-1, 1, 3)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ok = np.all(np.isfinite(seg), axis=(1, 2))
        lc = Line3DCollection(seg[ok], cmap=ALPHA_CMAP, norm=norm, linewidths=2)
        lc.set_array(np.nan_to_num(a[:-1][ok]))
        ax.add_collection3d(lc)
        set_box(ax, box)
        style(ax)
        t = np.arange(X.shape[1]) * TS
        ax2.plot(t, a, color=COL['P'], lw=1.2)
        ax2.set_ylim(0, 1.02)
        if lq is not None:
            ax3.plot(t, perr(lq), color=COL['LQR'], lw=1.1, label=LAB['LQR'])
        ax3.plot(t, perr(p), color=COL['P'], lw=1.1, label=LAB['P'])
        ax3.set_xlabel('time [s]', color=TEXT, fontsize=9)
        style2(ax2); style2(ax3)
        plt.setp(ax2.get_xticklabels(), visible=False)
        if i == 0:
            ax2.set_ylabel('alpha (student weight)', color=TEXT, fontsize=9)
            ax3.set_ylabel('position error [m]\n(capped at 5 m)', color=TEXT, fontsize=9)
            ax3.legend(frameon=False, fontsize=8, loc='upper right')
    cax = fig.add_axes([0.35, 0.965, 0.3, 0.01])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ALPHA_CMAP), cax=cax, orientation='horizontal')
    cb.set_label('P path coloured by alpha = c_S * g_L(c_LQR) (0 = pure LQR); dashed gray = reference',
                 color=TEXT, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('%s, chain %s: where the student acts (representative flight per family)' % (cond, tag),
                 color=TEXT, fontsize=10, y=0.003, va='bottom')
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.08)
    fig.savefig(os.path.join(out, 'alpha_%s_%s.png' % (cond, tag)), dpi=150)
    plt.close(fig)


def main(src, out):
    data = load(src)
    if not data:
        print('no final_traj_*.mat found'); return 1
    os.makedirs(out, exist_ok=True)
    tags = sorted({k.split(':')[1] for d in data.values() for k in d if ':' in k} - {'base'})
    for cond in ('train', 'ood'):
        fl = sorted(((d['family'], d) for (c, _), d in data.items() if c == cond),
                    key=lambda x: FAMS.index(x[0]) if x[0] in FAMS else 99)
        if not fl:
            continue
        for tag in tags:
            if not any(k.endswith(':' + tag) for _, d in fl for k in d):
                continue
            fig_paths(fl, cond, tag, out)
            fig_alpha(fl, cond, tag, out)
            for fam, d in fl:
                p = get(d, 'P', tag)
                if p is not None:
                    a = p['alpha'][np.isfinite(p['alpha'])]
                    print('%s %s %-17s alpha mean %.3f, share of steps with alpha > 0: %.3f'
                          % (cond, tag, fam, a.mean(), (a > 0).mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
