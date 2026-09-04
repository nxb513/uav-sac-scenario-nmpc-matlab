# UAV SAC-Scenario-NMPC MATLAB Checkpoint Stream

This public research package contains an episode-based, resumable training
stream for a continuous SAC policy that configures a scenario-NMPC teacher for
a 12-state uncertain quadrotor. The research budget is counted in episodes,
not transitions. The current candidate ceiling is 1,500 episodes and remains
subject to review.

## Training definition

- State: `[px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]`.
- Input: `[T, tau_phi, tau_theta, tau_psi]`.
- SAC action: six normalized NMPC weight groups, prediction horizon `N`, and
  control horizon `Nc`.
- Horizon candidates: `N in {5,10,15,20}` and
  `Nc in {2,5,10,15,20}`, constrained by `Nc <= N` (14 feasible pairs).
- Scenario count: fixed at `M=5`.
- Parallelization: one central SAC agent/replay stream and three synchronous
  environment workers.
- Episode: at most 200 control steps at `Ts=0.05 s`.
- Checkpoint: exact-resume artifact every 50 episodes.
- Contexts: the locked C012 hard-event rule uses `H=20`, position `0.10 m`,
  attitude `5 deg`, velocity `0.30 m/s`, body rate `2 rad/s`, growth factor
  `2`, and five consecutive growth steps. Growth-only remains auxiliary.
- Disturbance and parametric uncertainty are both active in the context bank.
- NMPC solve time has zero reward weight.

## Approved reference envelope

The in-distribution speed anchors are
`{0.5,1,2,4,6,8,10,12} m/s` for circle, lemniscate, vertical circle,
spatial helix and smooth waypoints. Geometry is scaled using the approved
speed-feasible load levels. At `0.5 m/s`, the resolved targets are approximately
`{0.20,0.46,0.83} m/s^2`; at `1 m/s` they are
`{0.74,1.85,3.33} m/s^2`; from `2 m/s` upward the full `{2,5,9} m/s^2`
targets apply, subject to the vertical-circle workspace limit. Every generated
reference must meet the realized-speed and realized-acceleration tolerances.
The candidate OOD anchors `{14,16} m/s` remain locked.

## Solver guard

The GitHub workflow applies a candidate wall-time guard to each `fmincon`
solve through `NMPC_MAX_WALL_SECONDS`. Its default is 60 seconds and the
manual workflow offers 30, 60, or 120 seconds. This is an execution safety
setting for the runner, not a final research hyperparameter. A stopped
solve is logged and follows the existing solver-failure termination path.

The underlying MATLAB configuration defaults to `Inf`; therefore the guard
is not silently imposed on other experiments.

## Run on GitHub Actions

The obsolete low-speed context bank has been removed. Frozen LQR `R_089` and
its SHA-256 provenance are committed as a small reproducibility input. C012
was selected only from independent LQR validation run `33501219369`; the SAC
workflow does not reuse those episodes. Instead, every long SAC job rebuilds
the teacher-training bank deterministically from seed `300830301`: 2,700
source episodes over 135 cells, 400 LQR steps per episode, and enough reference
continuation for a 200-step SAC episode plus `Nmax=20`.

The builder stores hard-event contexts for specialist training, growth-only
contexts as an explicitly labeled curriculum boundary stratum, and safe
contexts for audit only. Stages 1-3 may use hard plus growth-only contexts;
stage 4 uses hard contexts only. Growth-only is never relabeled as a hard
confidence target.

Run `MATLAB finalize targeted LQR selection` after the 250-candidate design
artifact is available. This inexpensive stage rebuilds only the corrected
1,350-episode selection bank and re-evaluates the frozen design top five. It
prevents a reference outside the declared state workspace from influencing the
final LQR while avoiding an unnecessary repeat of the clean design phase.

After v6 succeeds, run `MATLAB screen targeted LQR weakness` with the v6 run
ID. It builds a fresh 2,700-episode bank from seed `300830201`, runs only the
frozen LQR for 200 steps per episode, and saves full traces plus global,
per-family and per-speed/load summaries. The reported 5/10/20-step prevalence
grid is diagnostic; final divergence thresholds are selected afterward and
are never tuned on OOD/test data.

1. Keep this repository public.
2. Open **Actions > MATLAB retune strong targeted LQR** and dispatch it.
3. Download and review the `lqr-retune-realized-coverage-v5-*` artifact.
4. Open **Actions > MATLAB SAC-NMPC checkpoint stream** only after reviewing
   the locally measured context-bank coverage and the SAC runtime probe.
5. For a fresh SAC run, select `fresh_start=true`. For a resume, enter the prior
   run ID and select `fresh_start=false`.
6. Choose the candidate per-solve guard and run the workflow.
7. The job rebuilds and audits C012 contexts before training. Download the
   `sac-nmpc-checkpoint-stream-*` artifact when the job ends.

The job uses the official
[`matlab-actions/setup-matlab@v3`](https://github.com/matlab-actions/setup-matlab)
and
[`matlab-actions/run-command@v3`](https://github.com/matlab-actions/run-command)
actions with MATLAB R2024a. MathWorks automatically licenses supported
products for workflows in public repositories. No local MATLAB installation,
license file, batch token, MATLAB Compiler, or MATLAB Coder is included.

## Expected outputs

The LQR retune writes:

```text
results/targeted_lqr_weak_rebuild_v1/lqr_retune_realized_coverage_v5/
```

It contains the two reproducible bank manifests, all candidate metrics,
rankings, frozen `selected_lqr.mat`, report and completion marker. SAC outputs
are written under:

```text
results/targeted_lqr_weak_rebuild_v1/specialist_sac_v1/
  checkpoint_stream_runs/run_<github-run-id>/
```

The artifact includes configuration, episode manifests/logs, MATLAB diary,
training statistics, and recoverable `AgentK.mat` checkpoints.
`SolverTimedOut` is recorded at every environment step.

## Continuous checkpoint stream

The manual workflow continues one central SAC agent with three synchronous
workers. The MATLAB training step runs until its five-hour GitHub execution
limit or the 1,500-episode candidate ceiling, whichever occurs first, and
saves an exact-resume `AgentK.mat` every 50 episodes.

Provide the previous GitHub run ID when dispatching the workflow. The code
recursively selects the numerically largest valid `AgentK.mat`. If execution
stops after episode 146, for example, only `Agent100.mat` is resumed; episodes
101--146 are intentionally discarded.

The five-hour limit is an execution boundary. The 1,500-episode ceiling is a
candidate study bound, not a claim that this is the optimal training budget.
Each new run is dispatched manually after checking the preceding artifact;
the workflow does not chain jobs automatically.

## Scope and licensing

This repository is a public computational runner, not the complete research
project. It contains no papers, manuscript, OOD test set, hybrid-controller
results, legacy checkpoints, or private credentials. No reuse license has
been selected yet; publication on GitHub does not by itself grant an
open-source license.
