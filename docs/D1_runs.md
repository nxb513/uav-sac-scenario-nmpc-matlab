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

`d1-final-eval.yml` on tag `d1-final-eval-iter100`, both chains with suffix `_iter0100`
(run = the DAgger runs above), conditions `train` and `ood`, controllers LQR, MPC
(N = 20), Teacher, P: dispatched after both DAgger runs end.

## Performance versus SAC iteration (DAgger sweep)

| Run | Chain | Milestones | SAC artifact |
|---|---|---|---|
| 37171887012 | Bryson base | 9-49 | 37131706114 |
| 37171887012 | Random base | 9-44 | 37131714239 |

One job of 37171887012 (Bryson, iteration 12) was cancelled by the runner and rerun as
attempt 2 of the same run.
