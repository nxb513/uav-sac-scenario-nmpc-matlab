# D1 runs (official list)

Only the GitHub Actions runs listed here are results of the current D1 code and setup
(`docs/D1_method.pdf`). **Every other run in the Actions history is a draft or superseded
and must not be used**: all runs created before 2026-10-03 15:00 UTC (earlier D1 setups,
the hang diagnostics, the teacher grid, the draft DAgger and final-evaluation runs, and
every run of the legacy workflows now in `legacy/workflows/`). They are kept, not deleted,
so that they can be inspected if needed.

## Code

- SAC training code: commit `a707c1e` (unbounded action, warm-up 10, parallel flights,
  checkpoint every iteration). The chain runs executed at `f2bb6d9` and `3dd161f`; in the
  SAC path these differ from `a707c1e` only in comments.
- DAgger and sweep code: commit `7f59e98` (teacher/LQR reference on the validation flights,
  sweep CSV); `3dd161f` differs from it only in a README line.
- Final test at SAC iteration 100: DAgger on tag `d1-final-iter100` (= `3dd161f`); final
  evaluation on tag `d1-final-eval-iter100`, which differs from it in the D1 code only in
  the evaluation script, its workflow and its summary (LQI removed, linear MPC N = 20 instead of 5) and
  in comments, plus the move of unused files to `legacy/`.
- Solver: official release acados v0.6.0 in every run listed here.

## SAC chains

`stop_iter=100`, `ckpt_every=1`, `workers=4`, `solver=SQP`, `wind=1`, 20 flights per
iteration, `wall_seconds=20000`. Each run resumes from the previous one; the artifact
`d1-joint-ckpt-seed<seed>-<run_id>` of the latest run holds `checkpoint_seed<s>.mat` and
every milestone `checkpoint_seed<s>_iter<NNNN>.mat` up to its last iteration.

| Chain | Seed | `random_qr` | Runs (iterations) |
|---|---|---|---|
| Bryson base | 261003001 | 0 | 37131706114 (1-49), 37171575767 (50-95), 37187904877 (96-100) |
| Random base | 261003101 | 1 | 37131714239 (1-44), 37171580144 (45-100) |

## DAgger at SAC iteration 100 (`ckpt_suffix=_iter0100`)

| Chain | Run | Resumes SAC run |
|---|---|---|
| Bryson base | 37190192312 | 37187904877 |
| Random base | 37186800408 | 37171580144 |

Run 37190195267 is an accidental duplicate of the Bryson DAgger dispatched on `main`; it
was cancelled and is not used.

## Final evaluation at SAC iteration 100

`d1-final-eval.yml` on tag `d1-final-eval-iter100`, suffix `_iter0100` (run = the DAgger
runs above), conditions `train` and `ood`, controllers LQR, MPC (N = 20), Teacher, P. Each
chain was dispatched as soon as its DAgger run ended, so the evaluation is split over two
runs on the same flights; the combined summary is `tools/eval/summarize_final_eval.py`
over the CSVs of both runs.

| Run | Controllers |
|---|---|
| 37191803718 | LQR, MPC; Teacher and P of the random-base chain |
| 37193673007 (dispatched by `d1-final-eval-after.yml` run 37192056531 when 37190192312 ended) | Teacher and P of the Bryson-base chain |

The empty `base_ctrls` of 37193673007 was replaced by the default, so it also flew LQR and
MPC. That duplicate equals the baseline of 37191803718 to within 2e-13 m in position RMSE on
497 of 500 rows; the three exceptions are flights near a divergence (train flight 89,
`vertical_circle|v12|a9`, LQR and MPC; ood flight 55, `vertical_circle|v16|a9|OOD`, LQR, one
restart in one run and none in the other). The combined summary uses the baseline of
37191803718.
Since then `base_ctrls=none` skips the baseline job.

## Performance versus SAC iteration (DAgger sweep)

| Run | Chain | Milestones | SAC artifact |
|---|---|---|---|
| 37171887012 | Bryson base | 9-49 | 37131706114 |
| 37171887012 | Random base | 9-44 | 37131714239 |
| 37191810147 | Bryson base | 50-99 | 37187904877 |
| 37191810147 | Random base | 45-99 | 37171580144 |

Run 37191810147 runs on tag `d1-final-iter100` (same DAgger code). The point at iteration
100 is the final-test DAgger of each chain (table above), whose artifact holds the same
`sweep_seed<s>_iter0100.csv`.

One job of 37171887012 (Bryson, iteration 12) was cancelled by the runner and rerun as
attempt 2 of the same run.

## Computation time on one local machine (not a CI run)

`experiments/d1_local_timing.m` on 2026-10-04 (code `585b2ba`; `d1_final_eval.m` and
`src/joint` identical to tag `d1-final-eval-iter100`), with the iter-100 checkpoint and
student files of the DAgger runs above. Machine: Intel Core i9-13900HX, Windows 11 Home,
MATLAB R2024a, one computational thread, process pinned to logical processor 0 (affinity
mask 1), on AC power. All three controllers run as interpreted MATLAB; the teacher is not
timed. Time per control step in microseconds; per flight the median, p99 and maximum over
its 979 steps, then summarized over the flights:

| Condition | Controller | Flights | Median of medians | Median of p99 | Median of maxima | Max over flights | MPC QP iterations / step |
|---|---|---|---|---|---|---|---|
| train | LQR | 150 | 0.7 | 1.6 | 5.3 | 15795.9 | |
| train | MPC (N = 20) | 150 | 97.8 | 217.8 | 485.9 | 121999.3 | 1.071 |
| train | P (Bryson base) | 150 | 16.9 | 29.8 | 141.2 | 85168.4 | |
| train | P (random base) | 150 | 17.2 | 27.8 | 171.9 | 1295.5 | |
| ood | LQR | 100 | 0.7 | 1.4 | 5.5 | 378.7 | |
| ood | MPC (N = 20) | 100 | 99.5 | 148.9 | 729.0 | 138906.1 | 1.699 |
| ood | P (Bryson base) | 100 | 16.6 | 23.6 | 44.0 | 1702.9 | |
| ood | P (random base) | 100 | 16.7 | 23.0 | 41.6 | 1196.3 | |

The largest values come from the first flights of a MATLAB session (train LQR flight 3,
train MPC flight 1, train P Bryson flights 1-2: first calls) and, for MPC, from flights
where it diverged (ood flight 51: 139 ms in one step; the active-set iterations grow). The
CSVs stay local (`results/local_timing`, not committed); the same flights also give P of
both chains, which is compared with the CI results only as a consistency check.
