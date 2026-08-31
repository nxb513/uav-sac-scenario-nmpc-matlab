# UAV SAC-Scenario-NMPC MATLAB Pilot

This public research package runs a bounded, reproducible timing pilot for
a continuous SAC policy that configures a scenario-NMPC teacher for a
12-state uncertain quadrotor.

The pilot is intentionally limited to **100 global transitions**. It is used
to measure whether the workload is computationally feasible before any final
training budget or experiment grid is approved. It is not a trained policy
result: replay warm-up is 256 transitions, so no gradient update is expected.

## Pilot definition

- State: `[px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]`.
- Input: `[T, tau_phi, tau_theta, tau_psi]`.
- SAC action: six normalized NMPC weight groups, prediction horizon `N`, and
  control horizon `Nc`.
- Horizon candidates: `N in {5,10,15,20}` and
  `Nc in {2,5,10,15,20}`, constrained by `Nc <= N` (14 feasible pairs).
- Scenario count: fixed at `M=5`.
- Parallelization: one central SAC agent/replay stream and three synchronous
  environment workers.
- Contexts: one representative context from each of 135 balanced factorial
  cells is selected deterministically from the included 1,350-context bank.
- Disturbance and parametric uncertainty are both active in the context bank.
- NMPC solve time has zero reward weight.

## Solver guard

The GitHub workflow applies a candidate wall-time guard to each `fmincon`
solve through `NMPC_MAX_WALL_SECONDS`. Its default is 60 seconds and the
manual workflow offers 30, 60, or 120 seconds. This is an execution safety
setting for the timing pilot, not a final research hyperparameter. A stopped
solve is logged and follows the existing solver-failure termination path.

The underlying MATLAB configuration defaults to `Inf`; therefore the guard
is not silently imposed on other experiments.

## Run on GitHub Actions

1. Keep this repository public.
2. Open **Actions > MATLAB SAC-NMPC 100-transition pilot**.
3. Select **Run workflow** and choose the candidate solve limit.
4. Download the `sac-nmpc-sync3-pilot-*` artifact when the job ends.

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
  actual_sac_sync3_100_v1/
```

The artifact includes configuration, episode manifests/logs, MATLAB diary,
runtime summary, training statistics, and initial/final agent checkpoints.
`SolverTimedOut` is recorded at every environment step.

## Continuous checkpoint stream

After the timing pilot is validated, the manual workflow **MATLAB SAC-NMPC
checkpoint stream** continues the same central SAC agent with three
synchronous workers. It deliberately has no approved research episode or
transition budget. The MATLAB training step runs until its five-hour GitHub
execution limit and saves an exact-resume `AgentK.mat` every 10 episodes.

Provide the previous GitHub run ID when dispatching the workflow. The code
recursively selects the numerically largest valid `AgentK.mat`. If the
previous artifact is the corrected pilot and has no periodic checkpoint, it
starts from `final_agent.mat`. If execution stops after episode 96, for
example, only `Agent90.mat` is resumed; episodes 91--96 are discarded.

The five-hour limit is an execution boundary, not a final SAC budget. Each
new run is dispatched manually after checking the preceding artifact; the
workflow does not chain jobs automatically.

## Scope and licensing

This repository is a public computational pilot, not the complete research
project. It contains no papers, manuscript, OOD test set, hybrid-controller
results, legacy checkpoints, or private credentials. No reuse license has
been selected yet; publication on GitHub does not by itself grant an
open-source license.
