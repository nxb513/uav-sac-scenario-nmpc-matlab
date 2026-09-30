#!/usr/bin/env python3
"""Summarize experiments/d1_final_eval.m CSVs into a Markdown report.

Controller label: baseline controllers keep their name (LQR, LQI, MPC); chain-specific
ones are tagged with the chain seed suffix and checkpoint iteration, e.g. Teacher@201(i500).

Flights always run their full length (common flight rules of the pipeline): a divergence
restarts the plant on the reference and is counted in `restarts`. Position errors are
capped at 5 m per step (a diverged step counts as 5 m). A flight is COMPLETED when it had
no restart and never reached the cap. Paired comparisons use the same (condition, flight).
"""
import glob, os, sys
import pandas as pd

CAP_M = 5.0
FAMS = ['circle', 'lemniscate', 'vertical_circle', 'spatial_helix', 'smooth_waypoints']


def label(r):
    if r['tag'] == 'base':
        return r['ctrl']
    it = '' if pd.isna(r['ckpt_iter']) else f"(i{int(r['ckpt_iter'])})"
    return f"{r['ctrl']}@{str(int(r['seed']))[-3:]}{it}"


def fmt(v, p=4):
    return '—' if pd.isna(v) else f'{v:.{p}f}'


def main(folder):
    fs = sorted(glob.glob(os.path.join(folder, '**', 'final_eval_*.csv'), recursive=True))
    if not fs:
        print('No result CSVs found.'); return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df['done'] = (df['ok'] == 1) & (df['restarts'] == 0) & (df['pos_max'] < CAP_M)
    df['label'] = df.apply(label, axis=1)
    print('# D1 final evaluation (real wind, train / OOD, 5 families)\n')
    print(f'Files: {len(fs)} | rows: {len(df)} | completed = no restart and per-step error never at '
          f'the {CAP_M} m cap. All controllers know only the nominal model; every flight runs its '
          'full length.\n')
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
            stats.append(dict(label=lab, n=len(x), done=len(ok), restarts=int(x['restarts'].sum()),
                              pos_med=ok['pos_rmse'].median(), pos_mean=ok['pos_rmse'].mean(),
                              pos_all=x['pos_rmse'].median(),
                              pmax_med=ok['pos_max'].median(), vel_med=ok['vel_rmse'].median(),
                              du_med=ok['du_rms'].median(), t_med=x['t_med_us'].median(),
                              t_p99=x['t_p99_us'].median(), alpha=ok['alpha_mean'].mean(),
                              conv=x['teacher_conv'].mean(),
                              wins='—' if lab == 'LQI' or base.empty else f"{int((j['pos_rmse'] < j['b']).sum())}/{len(j)}"))
        st = pd.DataFrame(stats).sort_values(['done', 'pos_med'], ascending=[False, True])
        print('| controller | completed | restarts | pos RMSE med (completed) [m] | pos RMSE mean (completed) '
              '| pos RMSE med (all, capped) | max pos err med | vel RMSE med | du RMS med | t/step med [us] '
              '| t/step p99 [us] | better than LQI | mean alpha | NMPC conv |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in st.iterrows():
            print(f"| {r['label']} | {r['done']}/{r['n']} | {r['restarts']} | {fmt(r['pos_med'])} | "
                  f"{fmt(r['pos_mean'])} | {fmt(r['pos_all'])} | {fmt(r['pmax_med'], 3)} | {fmt(r['vel_med'])} | "
                  f"{fmt(r['du_med'])} | {fmt(r['t_med'], 1)} | {fmt(r['t_p99'], 1)} | {r['wins']} | "
                  f"{fmt(r['alpha'], 3)} | {fmt(r['conv'], 3)} |")
        print('\nPer family: pos RMSE median over completed flights (not completed)\n')
        print('| controller | ' + ' | '.join(FAMS) + ' |')
        print('|---|' + '---|' * len(FAMS))
        for lab in st['label']:
            cells = []
            for fam in FAMS:
                x = d[(d['label'] == lab) & (d['family'] == fam)]
                cells.append('—' if x.empty else f"{fmt(x[x['done']]['pos_rmse'].median(), 3)} ({int((~x['done']).sum())})")
            print(f'| {lab} | ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
