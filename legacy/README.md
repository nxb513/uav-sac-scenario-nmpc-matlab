# legacy (superseded, kept for inspection; NOT method D1)

Everything in this folder belongs to the stages before method D1. It was moved here on
2026-10-04 so that it cannot be mistaken for D1; nothing was deleted. No active workflow
adds this folder to the MATLAB path, so none of it runs in the D1 pipeline.

- Files keep their former relative paths: `legacy/<path>` was `<path>` (for example
  `legacy/experiments/run_d1_teacher_grid.m` was `experiments/run_d1_teacher_grid.m`).
  The files that D1 still calls (`configs/step1_plant_config.m`,
  `configs/targeted_lqr_weak_config.m`, the plant, reference and analysis functions)
  stay in place, so legacy scripts are not runnable from here without restoring the
  former layout.
- `legacy/workflows/`: the former `matlab-*.yml` workflows. Outside `.github/workflows`
  GitHub no longer offers them; the runs they produced remain in the Actions history.
- `legacy/legacy_pipeline.md`: the former top-level README (formerly
  `docs/legacy_pipeline.md`).
- `legacy/results/`: the committed artifacts of the targeted-LQR stages.
- `legacy/tests/`: contract tests of the legacy pipeline.
- `legacy/src/common/README.md`, `legacy/src/plant/README.md`: the former READMEs of these
  folders.
- The older D1-stage scripts (`run_d1_teacher_grid`, `select_d1_teacher`, `build_d1_*`,
  `d1_acados_teacher_verify`, `test_d1_lyapunov_contraction`, `d1_load_lyapunov_P`,
  `d1_regenerate_reference`) predate the current D1 pipeline and are not used by it.

The list was obtained from the static call graph of the two D1 entry points
(`experiments/run_d1_joint_pipeline.m`, `experiments/d1_final_eval.m`): every `.m` file
whose name does not occur in that call graph was moved. To restore a file, `git mv` it
back to the path without the `legacy/` prefix.
