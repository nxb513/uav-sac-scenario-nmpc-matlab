# Data Provenance

## Included artifact

`results/targeted_lqr_weak_rebuild_v1/specialist_local_context_banks_v2/training_local_context_bank.mat`

- Size: `84,344,993 bytes`.
- SHA-256: `B53F8A5FAD0F4E882BBC5D525FD61DF671E6470DD878D7B2742EC21445DF4101`.
- MATLAB format: `-v7.3`.
- Source bank: the balanced local-context construction stage of the
  `targeted_lqr_weak_rebuild_v1` pipeline.
- Total contexts in the source bank: `1,350`.
- Balanced factorial cells: `135`.
- Pilot selection: first stable occurrence of each `CellId`, producing
  exactly `135` active contexts.
- The validation bank and locked OOD set are not published or used here.

## Factorial coverage

The context bank balances reference family, disturbance type/level, and
parametric-uncertainty stratum. Every context stores its initial state,
previous input, local reference window, plant realization, disturbance
realization, source episode index, and LQR-local difficulty metadata.

The 100-transition pilot forces curriculum stage 4 so all represented context
types are eligible. Context sampling and uncertainty scenarios use fixed
seeds specified in `configs/targeted_specialist_sac_config.m`.

## Publication boundary

This artifact is included solely to reproduce the timing pilot. It contains
no account token, license material, user directory, paper PDF, manuscript,
surrogate dataset, OOD confirmation data, or prior controller result.

