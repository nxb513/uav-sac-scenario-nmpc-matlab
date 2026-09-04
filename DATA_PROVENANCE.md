# Data Provenance

## Training-bank reconstruction

The obsolete low-speed specialist context bank remains removed. The active
bank is reconstructed rather than committed: seed `300830301` generates 2,700
independent teacher-training source episodes over 135 factorial cells. Each
source trace contains 400 LQR steps so a selected intervention state retains a
200-step SAC continuation and the maximum 20-step prediction horizon.

Candidate C012 was locked using validation seed `300830201`, not the training
seed. Its hard target is a future absolute-envelope, saturation, state/tilt or
nonfinite event within 20 steps. Growth-only warnings are saved in a separate
auxiliary stratum and are not hard-failure labels. The deterministic builder
is `experiments/build_targeted_c012_context_bank.m`; its audit is
`experiments/audit_targeted_c012_context_bank.m`.

## Publication boundary

The repository contains no account token, license material, user directory,
paper PDF, manuscript, surrogate dataset, OOD confirmation data or prior
controller result.

## Nominal reference-feasibility artifacts

`results/targeted_lqr_weak_rebuild_v1/reference_feasibility_v1/` contains the
first 120-case nominal screen. It is superseded because its low-speed load
labels were not dynamically distinct and its randomized bank did not enforce
realized-speed tolerance.

`results/targeted_lqr_weak_rebuild_v1/reference_feasibility_v2_realized_coverage/`
is the active 120-case screen. All 120 rows pass the physical and robust-train
reference gates. Maximum realized speed/acceleration errors are approximately
`0.27%/18.37%` in this deterministic screen. Both artifact directories include
SHA-256 manifests; neither contains hidden test/OOD realizations or closed-loop
performance results.

The subsequently approved bank sampler uses the same eight speed anchors and
three speed-feasible load levels, with geometry jitter and deterministic
within-cell stratification. At `0.5-1 m/s`, the load accelerations are reduced
according to `v^2/Rmin`; from `2 m/s` upward, the full `{2,5,9} m/s^2` targets
apply, except for a vertical-circle workspace floor. Contract tests rebuild a
1,350-episode manifest in memory and verify all eight speeds, all three load
levels, realized-speed tolerance and realized-acceleration tolerance in each
of the 135 factorial cells.

The v4 GitHub artifact from run `33472674604` is retained only as audit
provenance. It was computationally complete but selected some lemniscate rows
by target-speed labels rather than realized speed, and its low-speed load
labels were not dynamically distinct. It must not be used downstream.

The v5 design bank from run `33495255190` passes realized speed/load coverage
and contains no reference or initial-state bound violation. Its selection bank
contains one vertical-circle reference above the altitude bound and therefore
is not frozen directly. The v6 finalization workflow reuses only the clean
250-candidate design ranking and rebuilds the complete selection bank after
adding explicit reference/X0 state-bound gates.

The weakness-screen workflow uses fresh seed `300830201` and 20 replicates in
each of the 135 family/disturbance/uncertainty cells, totaling 2,700 episodes.
It does not reuse either LQR model-selection bank and does not open OOD/test
data. Its output is diagnostic LQR-only evidence; no NMPC teacher outcome is
used to select a weak context at this stage.

## Strong-LQR selection artifact

The manual LQR workflow rebuilds independent design and selection banks from
seeds `300830801` and `300830901`. It evaluates 125 coarse candidates, 125
local-refinement candidates and the top five on the independent selection
bank. This is validation-only model selection; no OOD confirmation realization
is opened. The selected `R_089` controller from successful v6 run
`33500607984` is committed at
`results/targeted_lqr_weak_rebuild_v1/lqr_retune_realized_coverage_v6/selected_lqr.mat`
so deterministic context reconstruction does not depend on expiring Actions
artifacts. Its SHA-256 digest is recorded beside the file; the large design and
selection banks remain excluded.
