"""DAgger selection and confidence statistics of the iteration-100 students (final test).

Prints, per chain: the selected candidate, its ridge lambda and spectral radius, the validation RMSE of the
student, its teacher and the LQR on the 15 validation flights, the number of DAgger labels, the validation RMSE,
spectral radius and stability flag of every candidate, and the accuracy and base rate of the confidences c_LQR
and c_S (mean soft label for c_S).

Usage: python student_stats.py <data_root>
"""
import glob
import os
import sys

import h5py

ROOT = sys.argv[1]


def v(x):
    return float(x[()].ravel()[0])


for f in sorted(glob.glob(os.path.join(ROOT, '**', 'student_seed*_iter0100.mat'), recursive=True)):
    with h5py.File(f, 'r') as h:
        g, s = h['conf'], h['stu']
        print(os.path.basename(f))
        print('  c_LQR acc %.3f base %.3f n %d | c_S acc %.3f mean s %.3f n %d' % (
            v(g['LQR/acc']), v(g['LQR/base']), v(g['LQR/n']), v(g['S/acc']), v(g['S/base']), v(g['S/n'])))
        print('  selected %d lambda %.3g rho %.3f val %.4f teacherVal %.4f lqrVal %.4f n_labels %d' % (
            v(s['selected']), v(s['lambda']), v(s['rho']), v(s['val']), v(s['teacherVal']), v(s['lqrVal']), v(s['n'])))
        print('  valAll   ', [round(float(x), 3) for x in s['valAll'][()].ravel()])
        print('  rhoAll   ', [round(float(x), 3) for x in s['rhoAll'][()].ravel()])
        print('  stableAll', [int(x) for x in s['stableAll'][()].ravel()])
