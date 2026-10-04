# Publication Manifest

## Method D1 (active)

The public package contains, for method D1 (`docs/D1_method.pdf`):

- the pipeline `experiments/run_d1_joint_pipeline.m` (SAC phase, DAgger phase,
  diagnostic modes) and the final evaluation `experiments/d1_final_eval.m`;
- the shared D1 definitions `src/joint/d1_*.m` (configuration, flight rules, acados
  teacher with wind parameter, parallel teacher pool, teacher target/step, linear student features, DAgger,
  ridge fit, stability check, confidences, deployed blend) and the plant/reference
  utilities they call (`configs/step1_plant_config.m`, `quad_dynamics`, `quad_step_rk4`,
  reference generators, `targeted_lqr_weak_config`);
- the workflows `d1-joint-pipeline.yml`, `d1-sweep.yml` (DAgger on every SAC milestone:
  performance versus SAC iteration) and `d1-final-eval.yml` (the diagnostic replay of the
  solver hang is in the history at commit `7f60828`);
- `tools/wind/` (download + conversion of measured wind inside CI jobs),
  `tools/eval/summarize_final_eval.py`, `tools/eval/plot_sweep.py` and
  `tools/ci/hang_watchdog.sh`;
- the method document `docs/D1_method.tex/.pdf`, the design note
  `docs/D1_student_design.tex/.pdf` and the list of official runs `docs/D1_runs.md`.

Outside `legacy/`, only the MATLAB files called by the two D1 entry points remain.

Explicitly excluded for D1: SAC checkpoints and student files (they exist only as
GitHub Actions artifacts of the runs), measured wind data (Neural-Fly is not
redistributable; SWUF-3D is fetched from Zenodo), and evaluation outputs.

## Legacy pipeline (superseded, kept for provenance; see `legacy/legacy_pipeline.md`)

Since 2026-10-04 all of it is under `legacy/` with its former relative paths (see
`legacy/README.md`), except the files that D1 still calls, which stay in place. Nothing
was deleted. The legacy package contains only:

- six configuration functions required by the active pipeline;
- quadrotor plant and NMPC helper functions required at runtime;
- the targeted SAC environment, action map, 328D causal observation map,
  feature/prediction-error helpers, and warm-start resize helper;
- the episode-based continuous sync-3 checkpoint-stream entry point;
- the five-family reference-feasibility screen and contract tests;
- the approved speed/load-stratified reference sampler and bank builder;
- the full 250-candidate strong-LQR retune runner;
- the state-bound-gated top-five LQR selection-finalization workflow;
- the fresh 2,700-episode LQR-only weakness-screen runners and workflow;
- the frozen 0.29 MB `R_089` LQR artifact and SHA-256 provenance;
- the locked C012 causal-v2 independent context-bank builder and audit,
  without the generated context MAT file;
- manual GitHub Actions workflows for contract validation, probes and the
  staged checkpoint stream;
- the bounded 210-solve C012 NMPC convergence/feasibility screen;
- the resumable 90-task paired H=20 teacher closed-loop capability gate;
- the original and corrected nominal reference-feasibility CSV/MAT artifacts
  and reports, with the original explicitly marked superseded;
- documentation and repository metadata.

Explicitly excluded:

- all paper PDFs and journal templates;
- the manuscript and Methods section;
- local MATLAB/license information;
- all existing SAC checkpoints and failed-run artifacts;
- the obsolete low-speed SAC context bank and generated C012 context MAT file;
- surrogate, confidence, hybrid-controller, validation, and OOD results;
- unrelated scripts, tests, figures, and temporary files.
