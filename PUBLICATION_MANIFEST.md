# Publication Manifest

## Method D1 (active)

The public package contains, for method D1 (`docs/D1_method.pdf`):

- the training pipeline `experiments/run_d1_joint_pipeline.m` and the final evaluation
  `experiments/d1_final_eval.m`;
- the shared D1 definitions `src/joint/d1_*.m` (configuration, flight rules, acados
  teacher with wind parameter, teacher target/step, surrogate history/feature, deployed
  blend) and the plant/reference utilities they call (`configs/step1_plant_config.m`,
  `quad_dynamics`, `quad_step_rk4`, reference generators, `targeted_lqr_weak_config`);
- the workflows `d1-joint-pipeline.yml` and `d1-final-eval.yml`;
- `tools/wind/` (download + conversion of measured wind inside CI jobs) and
  `tools/eval/summarize_final_eval.py`;
- the method document `docs/D1_method.tex/.pdf`.

Explicitly excluded for D1: trained checkpoints and deployed pairs (they exist only as
GitHub Actions artifacts of the runs), measured wind data (Neural-Fly is not
redistributable; SWUF-3D is fetched from Zenodo), and evaluation outputs.

## Legacy pipeline (superseded, kept for provenance; see `docs/legacy_pipeline.md`)

The public package contains only:

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
