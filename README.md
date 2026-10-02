# UAV SAC-Scenario-NMPC (MATLAB) — method D1

The active method of this repository is **D1**. Its complete description (Vietnamese)
is `docs/D1_method.pdf` (source `docs/D1_method.tex`); the document is kept in sync with
the code, and the code is authoritative. Earlier pipelines are superseded and described
only in `docs/legacy_pipeline.md`. `docs/D1_student_design.pdf` is the approved design of
the linear student + DAgger; it is now implemented and merged into `docs/D1_method`.

## Method in one page

- **Deployed controller (P):** `u = sat( sat(u_LQR) + alpha * W*phi )`,
  `alpha = c_S * g_L(c_LQR)`, `g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1)`,
  `c_low = 0.3`, `c_high = 0.7`. `u_LQR = u_h - K (x - x_ref)` is a Bryson LQR on the
  nominal hover model. No optimization is solved online; the cost per step is of the same
  order as the LQR (108 multiply-adds for `W*phi`, 48 for the LQR).
- **Teacher (training only, and oracle in evaluation):** SAC-tuned scenario NMPC
  (acados, M = 5 model scenarios, N = 20, Nc = 5, SQP with at most 50 SQP and 100 QP
  iterations per solve). It is told the
  exact current wind force (privileged information, simulation only), which enters its
  prediction model and a wind-consistent flat reference. It is not deployable.
- **Linear student:** `Du = W*phi`, `W` is 4 x 27, no bias. `phi` holds the state error
  (12), the external force estimated from the last step `F_hat` (3), the reference flat
  feed-forward `u_ref - u_h` (4), and its preview at `k+Nc` and `k+N` (8). The student
  never sees the wind.
- **DAgger** (Ross, Gordon, Bagnell 2011), after SAC, with the teacher frozen:
  1. Iteration 1: the teacher flies.
  2. Later iterations: the student flies at alpha = 1, and the teacher is only queried at
     the states the student visits.
  3. After every iteration, ridge regression in closed form on all data (lambda by
     case-grouped 5-fold CV), followed by a linearized stability check.
  4. The selected `W*` is the stable candidate with the best validation RMSE.
- **Confidences:** `c_S` (soft-label logistic of recent tracking quality of the student) and
  `c_LQR` (logistic of LQR Lyapunov contraction over H = 20 steps), both learned in the
  training wind.
- **Training environment:** nominal plant plus random synthetic wind (mean 1-10 m/s,
  Dryden low-altitude gusts, linear rotor drag). Measured wind data are **never** used in
  training; they are only used in the final evaluation.

## Code map

| File | Role |
|---|---|
| `experiments/run_d1_joint_pipeline.m` | SAC phase (teacher tuning), DAgger phase (`D1_DAGGER=1`), diagnostic modes |
| `experiments/d1_final_eval.m` | Final evaluation: LQR, LQI, linear MPC (N = 5), Teacher (oracle), P under measured wind |
| `src/joint/d1_*.m` | The single shared definitions used by both scripts (see `src/joint/README.md`) |
| `src/joint/d1_dagger_run.m` | DAgger loop, selection, confidences, student file |
| `src/joint/d1_teacher_build_solver.m` | acados scenario-NMPC teacher with the wind-force parameter |
| `tools/wind/` | Download and convert measured wind (validation only, inside CI jobs) |
| `tools/eval/summarize_final_eval.py` | Markdown summary of the final-evaluation CSVs |
| `tools/ci/hang_watchdog.sh` | Diagnostic watchdog of the training step |
| `experiments/d1_diag_replay.m`, `.github/workflows/d1-diag-replay.yml`, `tools/ci/d1_set_ftz.c`, `tools/ci/diag_watchdog.sh`, `tools/ci/diag_gdb.py` | Diagnostic replay of a hung case with stall snapshots (not part of the method) |

All other scripts in `experiments/` belong to earlier stages and are not part of the current
pipeline. This includes the older D1-stage scripts `run_d1_teacher_grid`, `select_d1_teacher`,
`build_d1_*`, `d1_acados_teacher_verify` and `test_d1_lyapunov_contraction`.

## Common flight rules (all training, DAgger, diagnostic and evaluation flights)

- Every flight lasts 979 steps (`Ts = 0.05 s`) and always runs to the end.
- A divergence (integration error, non-finite state or `|p| > 1e4 m`) restarts the plant
  on the reference, resets the controller's internal state and is counted.
- The per-step position error is capped at 5 m; a diverged step counts as 5 m.
- A teacher solve is *usable* when its status is 0 or 2 and the result is finite. Usable
  solves are applied (and are DAgger labels inside the flight envelope). Otherwise the last
  input is held and the solver is reset. The SAC reward's failure rate counts the unusable steps.

## GitHub workflows

- **D1 joint pipeline** (`.github/workflows/d1-joint-pipeline.yml`).
  - **SAC phase** (default). Main inputs: `seeds`, `random_qr`, `logmult_dec`,
    `wall_seconds`, `stop_iter`, `ckpt_every`, `wind`, `solver`, and `resume_run_id`
    (empty means a fresh start).
  - **Artifact per seed:** `d1-joint-ckpt-seed<seed>-<run_id>`. It holds
    `checkpoint_seed<s>.mat` and the milestone copies `checkpoint_seed<s>_iter<NNNN>.mat`
    every 50 iterations.
  - **DAgger phase:** `dagger=1` with `resume_run_id` set to the run that holds the SAC
    checkpoint, and `ckpt_suffix` (for example `_iter0500`). It writes
    `student_seed<s><suffix>.mat`, the deployed controller, and the resumable
    `dagger_seed<s><suffix>.mat`. To continue an unfinished DAgger, run it again with
    `resume_run_id` set to the previous DAgger run.
  - **Diagnostic mode inputs:** `diag`, `surr_eval` (student evaluation), `compare` (with
    `hard`, `plant_perturb`), `consolidate`. The gate-grid mode (`D1_GATE_GRID=1`) exists
    in the code but has no workflow input.
- **D1 final evaluation** (`.github/workflows/d1-final-eval.yml`).
  - **Inputs:** `chains` (JSON list of `{seed, rqr, dec, run, suffix}`, where `run` is the
    DAgger run, whose artifact holds both the SAC checkpoint and the student file), `conds`
    (`train`, `ood`), `chain_ctrls` (`Teacher,P`) and `base_ctrls` (`LQR,LQI,MPC`).
  - **Outputs:** CSVs, representative trajectories and a Markdown summary.

Both D1 workflows build acados at the pinned commit `3edf4435e7d88d8a4d1c5c3a4f613f032aae658f`
(CasADi 3.6.7), so every job of every chain uses the same solver. A resume fails if the
checkpoint cannot be downloaded; it never silently restarts a chain.

A diagnostic watchdog (`tools/ci/hang_watchdog.sh`) watches the training step. If no file is
written for 60 minutes, it saves the stacks of the MATLAB process (`hang_backtrace_*.txt` in
the artifact) and stops the step, so the last checkpoint is uploaded. It never changes the
training itself.

Other `matlab-*.yml` workflows belong to the legacy pipeline.

## Data

Measured wind for validation: Neural-Fly (O'Connell et al., 2022; personal/educational
use only, not redistributed) and SWUF-3D (Zenodo 17700905, CC-BY 4.0). Both are downloaded
from their sources inside the CI job and are never committed or uploaded. See
`DATA_PROVENANCE.md`.

## Scope and licensing

This repository is a public computational runner, not the complete research project. It
contains no papers, manuscript, trained checkpoints or private credentials. Trained
checkpoints live only in the GitHub Actions artifacts of the runs. No reuse license has
been selected yet; publication on GitHub does not by itself grant an open-source license.

MATLAB R2024a runs through the official `matlab-actions/setup-matlab@v3` and
`matlab-actions/run-command@v3`. MathWorks licenses supported products for workflows in
public repositories; no license file or token is included.
