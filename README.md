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
- Contexts: balanced pre-divergence LQR contexts after their thresholds and
  revised reference bank have been approved.
- Disturbance and parametric uncertainty are both active in the context bank.
- NMPC solve time has zero reward weight.

## Approved reference envelope

The in-distribution speed anchors are
`{0.5,1,2,4,6,8,10,12} m/s` for circle, lemniscate, vertical circle,
spatial helix and smooth waypoints. Geometry is scaled using the approved
acceleration targets `{2,5,9} m/s^2`; speed is not increased while keeping an
infeasibly small radius fixed. Every 10-replicate factorial cell is stratified
to contain all eight speed anchors and all three load targets. The candidate
OOD anchors `{14,16} m/s` remain locked.

## Solver guard

The GitHub workflow applies a candidate wall-time guard to each `fmincon`
solve through `NMPC_MAX_WALL_SECONDS`. Its default is 60 seconds and the
manual workflow offers 30, 60, or 120 seconds. This is an execution safety
setting for the runner, not a final research hyperparameter. A stopped
solve is logged and follows the existing solver-failure termination path.

The underlying MATLAB configuration defaults to `Inf`; therefore the guard
is not silently imposed on other experiments.

## Run on GitHub Actions

The obsolete low-speed context bank has been removed. Do not dispatch the SAC
checkpoint stream until the revised LQR/reference bank and divergence
thresholds are frozen and a replacement context artifact is added. The
contract-validation and LQR probe workflows are safe to run now.

1. Keep this repository public.
2. Open **Actions > MATLAB SAC-NMPC checkpoint stream**.
3. For a fresh run, select `fresh_start=true`. For a resume, enter the prior
   run ID and select `fresh_start=false`.
4. Choose the candidate per-solve guard and run the workflow.
5. Download the `sac-nmpc-checkpoint-stream-*` artifact when the job ends.

The job uses the official
[`matlab-actions/setup-matlab@v3`](https://github.com/matlab-actions/setup-matlab)
and
[`matlab-actions/run-command@v3`](https://github.com/matlab-actions/run-command)
actions with MATLAB R2024a. MathWorks automatically licenses supported
products for workflows in public repositories. No local MATLAB installation,
license file, batch token, MATLAB Compiler, or MATLAB Coder is included.

## Expected outputs

Outputs are written under:

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
