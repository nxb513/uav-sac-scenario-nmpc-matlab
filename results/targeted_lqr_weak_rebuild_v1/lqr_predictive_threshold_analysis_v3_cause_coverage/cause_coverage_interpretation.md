# Cause-coverage interpretation

Status: diagnostic recommendation; final thresholds are not locked.

The approved body-rate sensitivity grid contains 1,536 combinations over
477,900 candidate H=20 contexts. The main body-rate comparison holds the other
values at position `0.10 m`, attitude `5 deg`, velocity `0.30 m/s`, growth
factor `2` and five consecutive growth steps:

| Candidate | Body-rate threshold | Hard-event step fraction | Hard episodes | Hard cells | First body-rate events |
|---|---:|---:|---:|---:|---:|
| C006 | 1 rad/s | 2.605% | 301 | 96 | 6,722 |
| C012 | 2 rad/s | 1.720% | 249 | 86 | 950 |
| C018 | 3 rad/s | 1.644% | 246 | 84 | 10 |
| C024 | 4 rad/s | 1.644% | 246 | 84 | 0 |

The transition from 1 to 2 rad/s removes the body-rate-dominated labeling.
Increasing the threshold from 2 to 3 rad/s changes the hard-event count by only
4.4%, and 3 to 4 rad/s is effectively flat. Thus 2 rad/s is the diagnostic
elbow and retains meaningful body-rate events without letting them dominate.

Candidate C012 provides:

- 8,026 hard-event steps among 466,632 eligible steps (1.720%);
- 601 hard-event onsets;
- 249/2,700 hard-event episodes (9.22%);
- 86/135 hard-event factorial cells across all five trajectory families;
- first-event counts of 4,087 position, 2,675 attitude, 116 velocity, 950 body
  rate, 395 constraint and 296 saturation events. Counts are nonexclusive when
  limits cross simultaneously.

C012 also has 7,760 growth-only steps. These are not observed envelope/safety
crossings. The recommended formulation is therefore:

- hard binary LQR confidence target: absolute envelope, saturation,
  state/tilt or nonfinite event within H=20;
- growth-only: separate auxiliary/boundary stratum, not silently merged into
  the hard failure label.

This recommendation uses only frozen LQR validation traces. It does not use
NMPC, SAC, surrogate, test or OOD outcomes and requires user approval before
the context bank is frozen.
