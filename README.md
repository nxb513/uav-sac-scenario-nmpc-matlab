# UAV SAC-Scenario-NMPC (MATLAB) — method D1

The active method of this repository is **D1**. Its complete description (Vietnamese)
is `docs/D1_method.pdf` (source `docs/D1_method.tex`); the document is kept in sync with
the code, and the code is authoritative. `docs/D1_student_design.pdf` is the design note of the
linear student + DAgger (approved 2026-10-02, implemented, and rewritten to match the code;
its first box lists where the code differs from the approved proposal); `docs/D1_method`
contains the same content within the full method.

- **Official runs:** `docs/D1_runs.md` lists the GitHub Actions runs of the current code;
  every other run in the Actions history is a draft or superseded (kept, not deleted).
- **Earlier pipelines** are superseded. Their code, workflows, tests and results are kept,
  not deleted, in `legacy/` (see `legacy/README.md`; the former README is
  `legacy/legacy_pipeline.md`). Nothing in `legacy/` is on the D1 MATLAB path.

## Method in one page

- **Deployed controller (P):** `u = sat( sat(u_LQR) + alpha * W*phi )`,
  `alpha = c_S * g_L(c_LQR)`, `g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1)`,
  `c_low = 0.3`, `c_high = 0.7`. `u_LQR = u_h - K (x - x_ref)` is a Bryson LQR on the
  nominal hover model. No optimization is solved online. Per control step, as counted in
  `docs/D1_method.pdf` (computation cost): P 254 multiplications, 46 divisions, 327
  additions and 33 elementary functions; LQR 48 multiplications and 60 additions.
  Computation time is not taken from CI runners (shared VMs, CPU unspecified); it is
  measured on one documented machine with `experiments/d1_local_timing.m`.
- **Teacher (training only, and oracle in evaluation):** SAC-tuned scenario NMPC
  (acados, M = 5 model scenarios, N = 20, Nc = 5, SQP with at most 50 SQP and 100 QP
  iterations per solve). SAC's action is the log10 of six multipliers of the base Q, R
  (Bryson, or a random base), unbounded (no search-range parameter); one SAC iteration
  scores one sampled Q, R on 20 training flights. It is told the
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
| `experiments/d1_final_eval.m` | Final evaluation: LQR, linear MPC (N = 20), Teacher (oracle), P under measured wind |
| `experiments/d1_local_timing.m` | Computation time of LQR, MPC and P on one documented machine (same flights and code as the final evaluation, one thread) |
| `src/joint/d1_*.m` | The single shared definitions used by both scripts (see `src/joint/README.md`) |
| `src/joint/d1_dagger_run.m` | DAgger loop, teacher/LQR reference on the validation flights, selection, confidences, student file, sweep CSV line |
| `src/joint/d1_teacher_build_solver.m` | acados scenario-NMPC teacher with the wind-force parameter |
| `tools/wind/` | Download and convert measured wind (validation only, inside CI jobs) |
| `tools/eval/summarize_final_eval.py` | Markdown summary of the final-evaluation CSVs |
| `tools/eval/paired_tests.py` | Paired Wilcoxon signed-rank tests of P against LQR and MPC on the same flights (Holm-adjusted) |
| `tools/eval/plot_final_traj.py` | Final-evaluation trajectory figures: 3D paths and position error of every controller; where the student acts in P (path coloured by alpha) |
| `src/joint/d1_teacher_pool.m` | Local parallel workers for the flights of one SAC / DAgger iteration |
| `tools/ci/hang_watchdog.sh` | Diagnostic watchdog of the training step |
| `tools/paper/` | Figures, tables and quoted numbers of the D1 paper from the release data (see `tools/paper/README.md`) |

The diagnostic replay used to find the solver hang (`experiments/d1_diag_replay.m`,
`d1-diag-replay.yml`, `tools/ci/d1_set_ftz.c`, `diag_watchdog.sh`, `diag_gdb.py`) is kept
in the history at commit `7f60828`; it reads only the earlier checkpoint format.

Outside `legacy/`, the repository holds only the MATLAB files that the two D1 entry points
call (static call graph). Older scripts, including the older D1-stage scripts
`run_d1_teacher_grid`, `select_d1_teacher`, `build_d1_*`, `d1_acados_teacher_verify` and
`test_d1_lyapunov_contraction`, are in `legacy/experiments/`.

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
  - **SAC phase** (default). Main inputs: `seeds`, `random_qr`, `wall_seconds`,
    `stop_iter`, `ckpt_every`, `workers`, `wind`, `solver`, and `resume_run_id` (empty
    means a fresh start). The 20 flights of an iteration run in parallel on `workers`
    (default 4) local MATLAB workers; their cases and winds are drawn in order by the
    client, so these draws do not depend on the number of workers. A worker reuses its
    solver for several flights and the flight-to-worker assignment varies, so results are
    not bit-for-bit reproducible between runs (observed).
  - **Research setup (2026-10-03):** two chains, `stop_iter=100`, `ckpt_every=1`, SAC
    warm-up 10 iterations (updates from iteration 10); the checkpoint of every iteration is
    kept so that the performance-versus-iterations curve can be built afterwards: Bryson base (seed 261003001,
    `random_qr=0`) and random base (seed 261003101, `random_qr=1`). In the SAC phase the
    two chains share the exploration noise, case draws and winds (the reference-bank build
    reseeds the global random stream per case), so they form a paired comparison; the seed
    sets only the teacher's model scenarios and the random base.
  - **Final test at SAC iteration 100:** both chains; DAgger (`ckpt_suffix=_iter0100`) runs
    on tag `d1-final-iter100`, the final evaluation (LQR, MPC, Teacher, P) on tag
    `d1-final-eval-iter100`; run ids in `docs/D1_runs.md`.
  - **Artifact per seed:** `d1-joint-ckpt-seed<seed>-<run_id>`. It holds
    `checkpoint_seed<s>.mat` (saved after every iteration) and the milestone copies
    `checkpoint_seed<s>_iter<NNNN>.mat` (every iteration with `ckpt_every=1`). The SAC
    buffer in the checkpoint holds (action, reward) of every iteration, so the learning
    curve can be rebuilt from iteration 1.
  - **DAgger phase:** `dagger=1` with `resume_run_id` set to the run that holds the SAC
    checkpoint, and `ckpt_suffix` (for example `_iter0100`). It writes
    `student_seed<s><suffix>.mat`, the deployed controller, and the resumable
    `dagger_seed<s><suffix>.mat`. To continue an unfinished DAgger, run it again with
    `resume_run_id` set to the previous DAgger run.
  - **Diagnostic mode inputs:** `diag`, `surr_eval` (student evaluation), `compare` (with
    `hard`, `plant_perturb`), `consolidate`. The gate-grid mode (`D1_GATE_GRID=1`) exists
    in the code but has no workflow input.
- **D1 DAgger sweep** (`.github/workflows/d1-sweep.yml`): performance versus SAC iteration.
  - **Input:** `chains`, a JSON list of `{seed, rqr, run, from, to}`, where `run` is the SAC
    run whose artifact holds the milestones `checkpoint_seed<s>_iter<NNNN>.mat`.
  - **One job per milestone iteration:** the DAgger phase on that frozen teacher, plus the
    teacher and the LQR on the same 15 validation flights.
  - **Outputs:** per milestone the student file, the DAgger state and
    `sweep_seed<s>_iter<NNNN>.csv`; the summary job writes `sweep_summary.csv` and
    `sweep_curve.png` (`tools/eval/plot_sweep.py`; a second row shows the confidences of
    each milestone's student file: c_S mean label and accuracy, c_LQR contracting share and
    accuracy).
- **D1 final evaluation** (`.github/workflows/d1-final-eval.yml`).
  - **Inputs:** `chains` (JSON list of `{seed, rqr, run, suffix}`, where `run` is the
    DAgger run, whose artifact holds both the SAC checkpoint and the student file), `conds`
    (`train`, `ood`), `chain_ctrls` (`Teacher,P`) and `base_ctrls` (`LQR,MPC`).
  - **Outputs:** CSVs, representative trajectories and a Markdown summary.
  - **D1 final evaluation after a run ends** (`d1-final-eval-after.yml`): waits on GitHub
    until a given run (for example a DAgger run) has succeeded, then dispatches
    `d1-final-eval.yml` with the given inputs.

All D1 workflows build the official release acados v0.6.0 (commit
`503364817c872d474ab5bed219c26760ac267769`, unmodified; CasADi 3.6.7), so every job of every
chain uses the same solver. The previously pinned unreleased master commit carried an HPIPM
regularization loop without an iteration limit that never ended on non-finite QP data (the
multi-hour "hangs"); see the known-issue note in `docs/D1_method.pdf`. A resume fails if the
checkpoint cannot be downloaded; it never silently restarts a chain.

A diagnostic watchdog (`tools/ci/hang_watchdog.sh`) watches the training step. If no file is
written for 60 minutes, it saves the stacks of every MATLAB process, client and workers
(`hang_backtrace_*.txt` in the artifact), and stops the step, so the last checkpoint is
uploaded. It never changes the
training itself.

The former `matlab-*.yml` workflows of the legacy pipeline are in `legacy/workflows/`;
GitHub no longer offers them, and their runs remain in the Actions history.

## Reproducing the paper

`tools/paper/README.md` maps every table, figure and quoted number of the D1 paper to the
script that produces it from the archives of the release `d1-paper-v1.0`.

## Data

Measured wind for validation: Neural-Fly (O'Connell et al., 2022; personal/educational
use only, not redistributed) and SWUF-3D (Zenodo 17700905, CC-BY 4.0). Both are downloaded
from their sources inside the CI job and are never committed or uploaded. See
`DATA_PROVENANCE.md`.

## Scope and licensing

This repository is a public computational runner, not the complete research project. It
contains no papers, manuscript, trained checkpoints or private credentials. Trained
checkpoints, student files and evaluation outputs are GitHub Actions artifacts of the runs,
which GitHub keeps for 30 days; the author keeps a copy of the artifacts of every official run
(`docs/D1_runs.md`), available on request. The outputs of the final test (SAC iteration 100:
checkpoints, DAgger and student files of both chains, final-evaluation CSVs and trajectories,
local timing) and the sweep CSV and student files of every SAC iteration are attached permanently
to the release
[`d1-paper-v1.0`](https://github.com/nxb513/uav-sac-scenario-nmpc-matlab/releases/tag/d1-paper-v1.0).

The code is released under the MIT License (`LICENSE`).

MATLAB R2024a runs through the official `matlab-actions/setup-matlab@v3` and
`matlab-actions/run-command@v3`. MathWorks licenses supported products for workflows in
public repositories; no license file or token is included.
