#!/usr/bin/env python3
"""Summarize experiments/d1_final_eval.m CSVs into a Markdown report.

Controller label: baseline controllers keep their name (LQR, LQI, MPC); chain-specific
ones are tagged with the chain seed suffix and checkpoint iteration, e.g. Teacher@201(i202).
A flight counts as completed when the state stayed finite and the max position error
stayed below 5 m. Paired comparisons use the same (condition, flight).
"""
import glob, os, sys
import pandas as pd

FAIL_M = 5.0
FAMS = ['circle', 'lemniscate', 'vertical_circle', 'spatial_helix', 'smooth_waypoints']


def label(r):
    if r['tag'] == 'base':
        return r['ctrl']
    it = '' if pd.isna(r['ckpt_iter']) else f"(i{int(r['ckpt_iter'])})"
    return f"{r['ctrl']}@{str(int(r['seed']))[-3:]}{it}"


def main(folder):
    fs = sorted(glob.glob(os.path.join(folder, '**', 'final_eval_*.csv'), recursive=True))
    if not fs:
        print('No result CSVs found.'); return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df['done'] = (df['ok'] == 1) & (df['pos_max'] < FAIL_M)
    df['label'] = df.apply(label, axis=1)
    print('# D1 final evaluation (real wind, train / OOD, 5 families)\n')
    print(f'Files: {len(fs)} | rows: {len(df)} | completed = finite and max position error < {FAIL_M} m. '
          'All controllers know only the nominal model.\n')
    for cond in ['train', 'ood']:
        d = df[df['cond'] == cond]
        if d.empty:
            continue
        print(f"## Condition `{cond}` ({d['flight'].nunique()} flights)\n")
        base = d[(d['label'] == 'LQI') & d['done']].set_index('flight')['pos_rmse']
        stats = []
        for lab, x in d.groupby('label'):
            ok = x[x['done']]
            j = ok.set_index('flight')['pos_rmse'].to_frame().join(base.rename('b'), how='inner')
            stats.append(dict(label=lab, n=len(x), done=len(ok),
                              pos_med=ok['pos_rmse'].median(), pos_mean=ok['pos_rmse'].mean(),
                              pmax_med=ok['pos_max'].median(), vel_med=ok['vel_rmse'].median(),
                              du_med=ok['du_rms'].median(), t_med=x['t_med_us'].median(),
                              t_p99=x['t_p99_us'].median(), alpha=ok['alpha_mean'].mean(),
                              conv=x['teacher_conv'].mean(),
                              wins='—' if lab == 'LQI' or base.empty else f"{int((j['pos_rmse'] < j['b']).sum())}/{len(j)}"))
        st = pd.DataFrame(stats).sort_values(['done', 'pos_med'], ascending=[False, True])
        print('| controller | completed | pos RMSE med [m] | pos RMSE mean | max pos err med | vel RMSE med | '
              'du RMS med | t/step med [us] | t/step p99 [us] | better than LQI | mean alpha | NMPC conv |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|')
        f = lambda v, p=4: '—' if pd.isna(v) else f'{v:.{p}f}'
        for _, r in st.iterrows():
            print(f"| {r['label']} | {r['done']}/{r['n']} | {f(r['pos_med'])} | {f(r['pos_mean'])} | {f(r['pmax_med'], 3)} | "
                  f"{f(r['vel_med'])} | {f(r['du_med'])} | {f(r['t_med'], 1)} | {f(r['t_p99'], 1)} | {r['wins']} | "
                  f"{f(r['alpha'], 3)} | {f(r['conv'], 3)} |")
        print('\nPer family: pos RMSE median (failures)\n')
        print('| controller | ' + ' | '.join(FAMS) + ' |')
        print('|---|' + '---|' * len(FAMS))
        for lab in st['label']:
            cells = []
            for fam in FAMS:
                x = d[(d['label'] == lab) & (d['family'] == fam)]
                cells.append('—' if x.empty else f"{x[x['done']]['pos_rmse'].median():.3f} ({int((~x['done']).sum())})")
            print(f'| {lab} | ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
