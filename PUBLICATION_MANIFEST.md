# Publication Manifest

The public package contains only:

- four configuration functions required by the pilot;
- quadrotor plant and NMPC helper functions required at runtime;
- the targeted SAC environment, action map, observation map, and warm-start
  resize helper;
- the fixed 100-transition continuous sync-3 pilot entry point;
- one training context bank;
- one manual GitHub Actions workflow;
- documentation and repository metadata.

Explicitly excluded:

- all paper PDFs and journal templates;
- the manuscript and Methods section;
- local MATLAB/license information;
- all existing SAC checkpoints and failed-run artifacts;
- surrogate, confidence, hybrid-controller, validation, and OOD results;
- unrelated scripts, tests, figures, and temporary files.

