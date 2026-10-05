"""Share of converged teacher solves on completed and not completed flights (final evaluation).

For each condition and chain, the mean over flights of the per-flight share of converged solves
(column teacher_conv of the final-evaluation CSVs) on the flights the teacher completed and on those it did not
complete (restart or 5 m cap). Rows found in several files count once, the first in the order of fig3/fig4
(run 37191803718 first), as in tools/eval/summarize_final_eval.py.

Usage: python teacher_solves.py <data_root>
"""
import glob
import os
import sys

import pandas as pd

ROOT = sys.argv[1]
KEY = ['cond', 'tag', 'seed', 'flight', 'ctrl']
fs = [f for f in glob.glob(os.path.join(ROOT, '**', 'final_eval_*.csv'), recursive=True) if 'local_timing' not in f]
fs.sort(key=lambda f: ('37191803718' not in f, f))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(subset=KEY, keep='first')
d['done'] = (d['ok'] == 1) & (d['restarts'] == 0) & (d['pos_max'] < 5.0)
t = d[d['ctrl'] == 'Teacher']
for (cond, tag), x in t.groupby(['cond', 'tag']):
    nd = x[~x['done']]
    print('%-5s %s: not completed %3d | converged share on not completed %s, on completed %.3f' % (
        cond, tag, len(nd), '%.3f' % nd['teacher_conv'].mean() if len(nd) else '-', x[x['done']]['teacher_conv'].mean()))
