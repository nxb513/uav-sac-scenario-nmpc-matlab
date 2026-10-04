# H=20 predictive LQR threshold analysis

- Status: `diagnostic_grid_requires_user_approval_before_threshold_freeze`.
- Final thresholds locked: `false`.
- Episodes/cells: 2700/135.
- Candidate start steps: 5 to 181.
- Candidate step contexts: 477900.
- Threshold combinations: 384.
- Analysis wall time: 61.744 s.
- Saturation episodes: 7.
- Constraint-violation episodes: 7.
- Union of hard episodes: 13.

## Future-error distribution

- position_m: q95 0.0895583, q99 0.116615, q99.9 0.172167, max 0.257219.
- attitude_deg: q95 2.97025, q99 5.26905, q99.9 13.2082, max 22.8119.
- velocity_mps: q95 0.114505, q99 0.185632, q99.9 0.518099, max 0.9822.
- body_rate_radps: q95 0.436941, q99 1.41282, q99.9 3.07916, max 4.54606.

## Class-balance representatives

- Target 1.0% -> C360: p/a/v/r 0.25 m / 15 deg / 1 m/s / 1 rad/s, growth 2 for 5 steps; observed 1.864% eligible steps, 201 episodes, 80 cells.
- Target 2.5% -> C299: p/a/v/r 0.25 m / 5 deg / 0.5 m/s / 1 rad/s, growth 2 for 3 steps; observed 2.501% eligible steps, 405 episodes, 112 cells.
- Target 5.0% -> C088: p/a/v/r 0.1 m / 20 deg / 0.75 m/s / 1 rad/s, growth 1.5 for 5 steps; observed 4.978% eligible steps, 808 episodes, 114 cells.
- Target 10.0% -> C019: p/a/v/r 0.1 m / 5 deg / 1 m/s / 1 rad/s, growth 1.25 for 3 steps; observed 7.343% eligible steps, 1080 episodes, 123 cells.

The shortlist is diagnostic only. It selects the grid row nearest each requested class-balance anchor; it does not approve an LQR-weak threshold. The 5% row is used only to export a coverage table. No NMPC, SAC, surrogate, test or OOD result entered this analysis.

