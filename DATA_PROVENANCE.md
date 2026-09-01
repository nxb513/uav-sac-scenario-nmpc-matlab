# Data Provenance

## Training-bank gate

The obsolete low-speed specialist context bank has been removed. No SAC
training context bank is published in this revision. A replacement may be
generated only after the approved `0.5-12 m/s` reference envelope, strong LQR
and 20-step divergence thresholds have been frozen. The episode checkpoint
stream therefore is source-only in this revision and must not be dispatched.

## Publication boundary

The repository contains no account token, license material, user directory,
paper PDF, manuscript, surrogate dataset, OOD confirmation data or prior
controller result.

## Nominal reference-feasibility artifact

`results/targeted_lqr_weak_rebuild_v1/reference_feasibility_v1/` contains the
120-case nominal screen over five trajectory families, eight speed anchors and
three acceleration targets. It includes no hidden test/OOD realization and is
not a closed-loop performance result. `SHA256SUMS.txt` binds the CSV, MAT,
report and completion marker included in the repository.

The subsequently approved bank sampler uses the same eight speed anchors and
three acceleration targets, with geometry jitter and deterministic
within-cell stratification. Contract tests rebuild a 1,350-episode manifest in
memory and verify all eight speeds and all three load targets in each of the
135 factorial cells.
