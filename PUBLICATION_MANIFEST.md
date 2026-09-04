# Publication Manifest

The public package contains only:

- six configuration functions required by the active pipeline;
- quadrotor plant and NMPC helper functions required at runtime;
- the targeted SAC environment, action map, observation map, and warm-start
  resize helper;
- the episode-based continuous sync-3 checkpoint-stream entry point;
- the five-family reference-feasibility screen and contract tests;
- the approved speed/load-stratified reference sampler and bank builder;
- the full 250-candidate strong-LQR retune runner;
- the state-bound-gated top-five LQR selection-finalization workflow;
- the fresh 2,700-episode LQR-only weakness-screen runners and workflow;
- the frozen 0.29 MB `R_089` LQR artifact and SHA-256 provenance;
- the locked C012 independent context-bank builder and audit, without the
  generated 76 MB context MAT file;
- manual GitHub Actions workflows for contract validation, probes and the
  staged checkpoint stream;
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
