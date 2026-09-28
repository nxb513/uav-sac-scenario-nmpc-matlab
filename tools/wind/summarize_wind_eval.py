#!/usr/bin/env python3
"""Summarize experiments/d1_wind_eval.m result CSVs into a Markdown report.

A flight counts as completed when the state stayed finite and the max position error
stayed below 5 m. Paired comparisons use the same (plant set, flight) for both controllers.
"""
import glob, os, sys
import pandas as pd

FAIL_M = 5.0


def level(ref):
    return 'v4/a2' if '|v4|' in ref else ('v8/a5' if '|v8|' in ref else 'v12/a9')


def main(folder):
    fs = sorted(glob.glob(os.path.join(folder, '**', 'wind_eval_*.csv'), recursive=True))
    if not fs:
        print('No result CSVs found.'); return
    df = pd.concat([pd.read_csv(f).assign(plantset=os.path.basename(f).split('_')[2]) for f in fs],
                   ignore_index=True)
    df['done'] = (df['ok'] == 1) & (df['pos_max'] < FAIL_M)
    df['level'] = df['ref'].map(level)
    ctrls = list(dict.fromkeys(df['ctrl']))
    base = 'LQI' if 'LQI' in ctrls else ctrls[0]
    print('# D1 real-wind evaluation\n')
    print(f'Files: {len(fs)} | rows: {len(df)} | completed = finite and max position error < {FAIL_M} m. '
          f'Paired baseline: **{base}**.\n')
    for ps in ['nom', 'train', 'ood']:
        d = df[df['plantset'] == ps]
        if d.empty:
            continue
        nfl = d['flight'].nunique()
        print(f'## True plant = `{ps}` ({nfl} flights; every controller designed on the nominal model)\n')
        print(f'| controller | completed | pos RMSE med [m] | pos RMSE mean | vel RMSE med [m/s] | max pos err med | '
              f'better than {base} (pos) | mean alpha |')
        print('|---|---|---|---|---|---|---|---|')
        b = d[d['ctrl'] == base].set_index('flight')
        for c in ctrls:
            x = d[d['ctrl'] == c].set_index('flight'); ok = x[x['done']]
            if c == base:
                wins = '—'
            else:
                j = ok.join(b[b['done']][['pos_rmse']], rsuffix='_b', how='inner')
                wins = f"{int((j['pos_rmse'] < j['pos_rmse_b']).sum())}/{len(j)}"
            am = ok['alpha_mean'].mean()
            print(f"| {c} | {len(ok)}/{len(x)} | {ok['pos_rmse'].median():.4f} | {ok['pos_rmse'].mean():.4f} | "
                  f"{ok['vel_rmse'].median():.4f} | {ok['pos_max'].median():.3f} | {wins} | "
                  f"{'—' if pd.isna(am) else f'{am:.3f}'} |")
        print('\nPer reference level: pos RMSE median (failures)\n')
        print('| level | ' + ' | '.join(ctrls) + ' |')
        print('|---|' + '---|' * len(ctrls))
        for L in ['v4/a2', 'v8/a5', 'v12/a9']:
            cells = []
            for c in ctrls:
                x = d[(d['ctrl'] == c) & (d['level'] == L)]
                cells.append(f"{x[x['done']]['pos_rmse'].median():.3f} ({int((~x['done']).sum())})")
            print(f'| {L} | ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
