#!/usr/bin/env python3
"""Performance-versus-SAC-iteration curve of the DAgger sweep (d1-sweep.yml).

Reads every sweep_seed<s>_iter<NNNN>.csv written by d1_dagger_run (one line per SAC
checkpoint: frozen teacher, LQR and selected DAgger student on the same validation
flights) and, when it lies next to the CSV, the student file student_seed<s>_iter<NNNN>.mat
(confidences learned by d1_consolidate: c_LQR accuracy and contracting share, c_S accuracy
and mean soft label). Writes sweep_summary.csv, sweep_curve.png (one column per chain;
second row = confidences when available) and prints a Markdown table. A milestone found
in several input folders is counted once.

Usage: plot_sweep.py <input_dir> <output_dir>
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDENT = '#2a78d6'   # categorical slot 1 (reference palette)
TEACHER = '#eb6834'   # categorical slot 2
REF = '#52514e'       # secondary ink: reference line (LQR)
TEXT = '#0b0b0b'
GRID = '#e4e3df'


CONF = ['c_lqr_acc', 'c_lqr_base', 'c_s_acc', 'c_s_mean']


def read_conf(csv_path):
    """conf.LQR.acc/base and conf.S.acc/base of the student file next to the CSV (v7.3)."""
    stu = os.path.join(os.path.dirname(csv_path),
                       os.path.basename(csv_path).replace('sweep_', 'student_').replace('.csv', '.mat'))
    try:
        import h5py
        with h5py.File(stu, 'r') as h:
            g = h['conf']
            v = [g['LQR/acc'], g['LQR/base'], g['S/acc'], g['S/base']]
            return dict(zip(CONF, [float(x[()].ravel()[0]) for x in v]))
    except (ImportError, OSError, KeyError):
        return {}


def main(src, out):
    rows = []
    for f in sorted(glob.glob(os.path.join(src, '**', 'sweep_seed*_iter*.csv'), recursive=True)):
        conf = read_conf(f)
        with open(f, newline='') as h:
            for r in csv.DictReader(h):
                r.update(conf)
                rows.append(r)
    if not rows:
        print('no sweep CSV found'); return 1
    os.makedirs(out, exist_ok=True)
    keys = ['seed', 'random_qr', 'sac_iter', 'teacher_val', 'lqr_val', 'student_val',
            'selected', 'rho_max', 'lambda'] + CONF
    rows = list({(int(r['seed']), int(r['sac_iter'])): r for r in rows}.values())
    rows.sort(key=lambda r: (int(r['seed']), int(r['sac_iter'])))
    with open(os.path.join(out, 'sweep_summary.csv'), 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    seeds = sorted({int(r['seed']) for r in rows})
    has_conf = any(r.get('c_s_mean') not in (None, '') for r in rows)
    nr = 2 if has_conf else 1
    fig, axes = plt.subplots(nr, len(seeds), figsize=(6.4 * len(seeds), 4.2 * nr), sharey='row',
                             squeeze=False)
    for ax, s in zip(axes[0], seeds):
        rs = [r for r in rows if int(r['seed']) == s]
        it = [int(r['sac_iter']) for r in rs]
        st = [float(r['student_val']) for r in rs]
        te = [float(r['teacher_val']) for r in rs]
        lq = float(rs[0]['lqr_val'])
        base = 'random Q,R base' if rs[0]['random_qr'] == '1' else 'Bryson base'
        ax.axhline(lq, color=REF, lw=1.5, ls='--', zorder=1)
        ax.plot(it, te, color=TEACHER, lw=2, marker='o', ms=4, zorder=2, label='Teacher (a = mu)')
        ax.plot(it, st, color=STUDENT, lw=2, marker='o', ms=4, zorder=3, label='Student (DAgger)')
        ax.text(it[0], lq, 'LQR', color=TEXT, va='bottom', ha='left', fontsize=9)
        ax.set_title('seed %d (%s)' % (s, base), color=TEXT, fontsize=11, loc='left')
        ax.set_xlabel('SAC iteration (checkpoint)', color=TEXT)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel('validation position RMSE [m]\n(15 flights, alpha = 1 for the student)', color=TEXT)
    axes[0][0].legend(frameon=False, loc='upper right')
    if has_conf:
        series = [('c_s_mean', 'c_S: mean label s (student tracking quality)', STUDENT, '-'),
                  ('c_s_acc', 'c_S: accuracy (p > 0.5 vs s > 0.5)', STUDENT, ':'),
                  ('c_lqr_base', 'c_LQR: share of contracting LQR windows', TEACHER, '-'),
                  ('c_lqr_acc', 'c_LQR: accuracy', TEACHER, ':')]
        for ax, s in zip(axes[1], seeds):
            rs = [r for r in rows if int(r['seed']) == s and r.get('c_s_mean') not in (None, '')]
            it = [int(r['sac_iter']) for r in rs]
            for k, lab, col, ls in series:
                ax.plot(it, [float(r[k]) for r in rs], color=col, ls=ls, lw=2, marker='o', ms=3, label=lab)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel('SAC iteration (checkpoint)', color=TEXT)
            ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axes[1][0].set_ylabel('confidences (training wind)', color=TEXT)
        axes[1][0].legend(frameon=False, loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'sweep_curve.png'), dpi=150)

    def f3(v):
        return '-' if v in (None, '') else '%.3f' % float(v)
    print('| seed | SAC iter | teacher | student | LQR | selected | rho_max | c_S mean s | c_LQR acc |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print('| %s | %s | %.4f | %.4f | %.4f | %s | %.3f | %s | %s |' % (
            r['seed'], r['sac_iter'], float(r['teacher_val']), float(r['student_val']),
            float(r['lqr_val']), r['selected'], float(r['rho_max']),
            f3(r.get('c_s_mean')), f3(r.get('c_lqr_acc'))))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
