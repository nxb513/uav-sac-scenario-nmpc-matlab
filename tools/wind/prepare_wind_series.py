#!/usr/bin/env python3
"""Build world-frame wind-force series (t, Fx, Fy, Fz [N]) for experiments/d1_wind_eval.m.

The raw data are downloaded at run time from their original sources and are NOT stored
in this repository or uploaded as artifacts (Neural-Fly data: personal/educational use
only, written permission required for further use).

Sources
  Neural-Fly  (O'Connell et al., Science Robotics 7(66) eabm6597, 2022)
              github.com/aerorobotics/neural-fly  data/{experiment,training}/*.csv
              residual aerodynamic force `fa` [N] measured on a 2.53 kg quadrotor, 50 Hz,
              Caltech Real Weather Wind Tunnel, 0 ... 12.1 m/s and 8.5 + 2.4 sin(t) m/s.
  SWUF-3D     Zenodo record 17700905 (CC-BY 4.0), 2025_field_measurements.zip
              Thies 3D sonic anemometer wind (u, v, w) at a wind farm next to small
              multicopters (Holybro QAV250), 4-6 Hz, ~590 s per flight.

Conversion (same ACCELERATION as the source vehicle; our nominal mass m = 0.486 kg):
  nf_*    F(t) = (m / 2.53) * (fa(t) - mean(fa | no wind, same trajectory type))
  swuf_*  F(t) = (m / 2.53) * (c1 + c2 |w(t)|) w(t),  (c1, c2) = least-squares fit of the
          Neural-Fly steady-wind mean force  F_x(V) = c1 V + c2 V^2  (bias removed).
          Window: first 49 s (one flight lasts 979 x 0.05 s = 48.95 s).
"""
import argparse, ast, glob, os
import numpy as np, pandas as pd, h5py

M_NF, G = 2.53, 9.81
SPEED = {'nowind': 0.0, '10wind': 1.3, '20wind': 2.5, '30wind': 3.7, '35wind': 4.2, '40wind': 4.9,
         '50wind': 6.1, '70wind': 8.5, '70p20sint': 8.5, '100wind': 12.1}


def load_nf(fn):
    df = pd.read_csv(fn)
    t = df['t'].to_numpy(float)
    fa = np.array([ast.literal_eval(s) for s in df['fa']], dtype=float)
    return t - t[0], fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nf', required=True, help='folder with Neural-Fly custom_*_baseline_*.csv')
    ap.add_argument('--swuf', required=True, help='folder with SWUF-3D *_sonic_data.nc')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mass', type=float, default=0.486, help='nominal mass of our quadrotor [kg]')
    ap.add_argument('--window', type=float, default=49.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k_m = a.mass / M_NF

    bias = {}
    for traj in ('figure8', 'random3'):
        _, fa = load_nf(os.path.join(a.nf, f'custom_{traj}_baseline_nowind.csv'))
        bias[traj] = fa.mean(axis=0)
    V, Fx = [], []
    for fn in sorted(glob.glob(os.path.join(a.nf, 'custom_*_baseline_*.csv'))):
        parts = os.path.basename(fn)[:-4].split('_')
        traj, cond = parts[1], parts[3]
        if cond == 'nowind':
            continue
        t, fa = load_nf(fn)
        d = fa - bias[traj]
        if cond != '70p20sint':
            V.append(SPEED[cond]); Fx.append(-d[:, 0].mean())
        F = k_m * d
        pd.DataFrame({'t': t, 'Fx': F[:, 0], 'Fy': F[:, 1], 'Fz': F[:, 2]}).to_csv(
            os.path.join(a.out, f'nf_{cond}.csv'), index=False)
        print(f'nf_{cond:10s} V={SPEED[cond]:5.1f} m/s  mean|F| {np.linalg.norm(F.mean(0)):.3f} N '
              f'({100 * np.linalg.norm(F.mean(0)) / (a.mass * G):4.1f}% hover)')

    V, Fx = np.array(V), np.array(Fx)
    A = np.c_[V, V ** 2]
    c, *_ = np.linalg.lstsq(A, Fx, rcond=None)
    r2 = 1 - ((Fx - A @ c) ** 2).sum() / ((Fx - Fx.mean()) ** 2).sum()
    print(f'drag fit F = {c[0]:.4f} V + {c[1]:.4f} V^2  (R^2 = {r2:.3f})')

    for fn in sorted(glob.glob(os.path.join(a.swuf, '**', '*uas_13*_sonic_data.nc'), recursive=True)):
        fl = os.path.basename(fn).split('__')[0].replace('2025_flight_', 'f')
        with h5py.File(fn, 'r') as f:
            t = f['time'][:]; w = np.c_[f['u'][:], f['v'][:], f['w'][:]]
        t = t - t[0]; sel = t <= a.window + 1.0
        t, w = t[sel], w[sel]
        F = k_m * (c[0] + c[1] * np.linalg.norm(w, axis=1, keepdims=True)) * w
        pd.DataFrame({'t': t, 'Fx': F[:, 0], 'Fy': F[:, 1], 'Fz': F[:, 2],
                      'wu': w[:, 0], 'wv': w[:, 1], 'ww': w[:, 2]}).to_csv(
            os.path.join(a.out, f'swuf_{fl}.csv'), index=False)
        print(f'swuf_{fl:4s} wind mean {np.linalg.norm(w[:, :2], axis=1).mean():5.2f} m/s  '
              f'mean|F| {np.linalg.norm(F.mean(0)):.3f} N ({100 * np.linalg.norm(F.mean(0)) / (a.mass * G):4.1f}% hover)')


if __name__ == '__main__':
    main()
