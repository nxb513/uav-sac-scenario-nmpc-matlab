# UAV SAC-Scenario-NMPC (MATLAB) — method D1

The active method of this repository is **D1**. Its complete description (Vietnamese)
is `docs/D1_method.pdf` (source `docs/D1_method.tex`); the document is kept in sync with
the code, and the code is authoritative. Earlier pipelines are superseded and described
only in `docs/legacy_pipeline.md`.

## Method in one page

- **Deployed controller (P):** `u = sat( sat(u_LQR) + alpha * Du )`,
  `alpha = c_S * g_L(c_LQR)`, `g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1)`,
  `c_low = 0.3`, `c_high = 0.7`. `u_LQR = u_h - K (x - x_ref)` is a Bryson LQR on the
  nominal hover model. No optimization is solved online.
- **Teacher (training only, and oracle in evaluation):** SAC-tuned scenario NMPC
  (acados, M = 5 model scenarios, N = 20, Nc = 5, SQP). It is told the exact current wind
  force (privileged information, simulation only), which enters its prediction model and
  a wind-consistent flat reference (offset-free MPC target). It is not deployable.
- **Surrogate:** one network with two heads. The residual head is trained on
  `Du* = u_teacher - sat(u_LQR)` at the teacher's states; the `c_S` head predicts recent
  tracking quality. Its 208-D input holds `x_{k-3..k}` (current state included),
  `u_{k-4..k-1}`, `x_ref,k..k+10` and the nominal one-step prediction residual
  `r_k = x_k - Phi_nom(x_{k-1}, u_{k-1})`. The surrogate never sees the wind; it has to
  infer it from `r_k`.
- **Confidences:** `c_S` (surrogate head, trained at alpha = 1) and `c_LQR` (logistic
  model of LQR Lyapunov contraction over H = 20 steps), both learned in the training wind.
- **Training environment:** nominal plant plus random synthetic wind (mean 1-10 m/s,
  Dryden low-altitude gusts, linear rotor drag). Measured wind data are **never** used in
  training; they are only used in the final evaluation.

## Code map

| File | Role |
|---|---|
| `experiments/run_d1_joint_pipeline.m` | Training (SAC + teacher + streaming surrogate), consolidation (`c_S`, `c_LQR`), diagnostic modes |
| `experiments/d1_final_eval.m` | Final evaluation: LQR, LQI, linear MPC (N = 5), Teacher (oracle), P under measured wind |
| `src/joint/d1_*.m` | The single shared definitions used by both scripts (see `src/joint/README.md`) |
| `src/joint/d1_teacher_build_solver.m` | acados scenario-NMPC teacher with the wind-force parameter |
| `tools/wind/` | Download and convert measured wind (validation only, inside CI jobs) |
| `tools/eval/summarize_final_eval.py` | Markdown summary of the final-evaluation CSVs |

All other scripts in `experiments/` belong to earlier stages and are not part of the current
pipeline. This includes the older D1-stage scripts `run_d1_teacher_grid`, `select_d1_teacher`,
`build_d1_*`, `d1_acados_teacher_verify` and `test_d1_lyapunov_contraction`.

## Common flight rules (all training, diagnostic and evaluation flights)

- Every flight lasts 979 steps (`Ts = 0.05 s`) and always runs to the end.
- A divergence (integration error, non-finite state or `|p| > 1e4 m`) restarts the plant
  on the reference, resets the controller's internal state and is counted.
- The per-step position error is capped at 5 m; a diverged step counts as 5 m.
- A teacher solve is *usable* when its status is 0 or 2 and the result is finite. Usable
  solves are applied and used as labels. Otherwise the last input is held and the solver
  is reset. The SAC reward's failure rate counts the unusable steps.

## GitHub workflows

- **D1 joint pipeline** (`.github/workflows/d1-joint-pipeline.yml`). Main inputs:
  `seeds`, `random_qr`, `logmult_dec`, `wall_seconds`, `stop_iter`, `ckpt_every`, `wind`,
  `solver`, and `resume_run_id` (empty means a fresh start). Artifact per seed:
  `d1-joint-ckpt-seed<seed>-<run_id>`. It contains `checkpoint_seed<s>.mat` and
  `conf_seed<s>.mat` (the deployed pair), plus milestone copies
  `checkpoint_seed<s>_iter<NNNN>.mat` and `conf_seed<s>_iter<NNNN>.mat` every 50
  iterations. Diagnostic mode inputs: `diag`, `surr_eval`, `compare` (with `hard`,
  `plant_perturb`), `consolidate`. The gate-grid mode (`D1_GATE_GRID=1`) exists in the
  code but has no workflow input.
- **D1 final evaluation** (`.github/workflows/d1-final-eval.yml`). Inputs: `chains`
  (JSON list of `{seed, rqr, dec, run, suffix}`), `conds` (`train`, `ood`), `chain_ctrls`
  (`Teacher,P`) and `base_ctrls` (`LQR,LQI,MPC`). It writes CSVs, representative
  trajectories and a Markdown summary.

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
