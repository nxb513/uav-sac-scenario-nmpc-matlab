# Targeted reference feasibility screen

- Cases: `120`.
- Wall time: `0.497206 s`.
- Speed anchors: `0.5, 1, 2, 4, 6, 8, 10, 12 m/s`.
- Load rule: `speed_feasible_fraction_of_full_envelope_target`.
- Realized speed/acceleration relative tolerances: `3.0% / 25.0%`.
- Physical gate: tilt <= 70.0 deg, input fraction <= 1.00, state bounds valid, p95 dynamics residual <= 0.250.
- Robust-train candidate gate additionally requires tilt <= 50.0 deg and input fraction <= 0.75.

## Family summary

- `circle`: physical `24/24`, robust candidate `24/24`, maximum physical/robust speed `12.00/12.00 m/s`.
- `lemniscate`: physical `24/24`, robust candidate `24/24`, maximum physical/robust speed `12.00/12.00 m/s`.
- `vertical_circle`: physical `24/24`, robust candidate `24/24`, maximum physical/robust speed `12.00/12.00 m/s`.
- `spatial_helix`: physical `24/24`, robust candidate `24/24`, maximum physical/robust speed `12.00/12.00 m/s`.
- `smooth_waypoints`: physical `24/24`, robust candidate `24/24`, maximum physical/robust speed `12.00/12.00 m/s`.

## Interpretation guardrails

- This screen changes geometry with speed so that speed is not confounded with an impossible centripetal acceleration.
- A passing row is only a nominal-reference feasibility candidate; it is not yet an approved training condition.
- Uncertainty, disturbance and closed-loop tracking are evaluated after the reference envelope is approved.
- OOD speed anchors remain locked and were not evaluated by this run.
