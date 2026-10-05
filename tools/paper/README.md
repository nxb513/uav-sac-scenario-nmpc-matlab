# Reproducing the figures, tables and numbers of the D1 paper

The scripts below read the outputs of the official runs. Download the three archives of the release
[`d1-paper-v1.0`](https://github.com/nxb513/uav-sac-scenario-nmpc-matlab/releases/tag/d1-paper-v1.0) and
extract them into one folder, `<data_root>`:

- `d1_final_controllers_iter0100/`: the SAC checkpoints, DAgger and student files at iteration 100
- `d1_final_evaluation/`: the final-evaluation CSVs and trajectories, and the local timing
- `d1_dagger_sweep/`: the sweep CSV and student files for every teacher checkpoint

The scripts search `<data_root>` recursively, so the folder of downloaded run artifacts works as
well. Rows that two runs both hold count once, taking the first file with run 37191803718 first
(`docs/D1_runs.md`). Python needs numpy, pandas, scipy, h5py and matplotlib.

| Paper item | Source |
|---|---|
| Table 1 (parameters, input limits, perturbation ranges) | `src/joint/d1_config.m`, `src/joint/d1_joint_plant_params.m`, `src/plant/` |
| Table 2 (training configuration) | `src/joint/d1_config.m`; teacher multipliers `10^mu`: `teacher_multipliers('<data_root>')` (MATLAB) |
| Table 3 (final evaluation) | `python tools/eval/summarize_final_eval.py <data_root>/d1_final_evaluation` |
| Table 4 (paired tests) | `python tools/eval/paired_tests.py <data_root>/d1_final_evaluation` |
| Table 5 (computation) | operation counts: `docs/D1_method.pdf`, section on computation cost; times: `python tools/eval/summarize_final_eval.py --timing <data_root>/d1_final_evaluation/local_timing` (measured with `experiments/d1_local_timing.m`) |
| Figures 1-2 (diagrams) | `python tools/paper/diagrams_drawio.py <out_dir>`, then export with the draw.io desktop app |
| Figure 3 (per-flight RMSE ratio) | `python tools/paper/fig3_paired_rmse.py <data_root> Figure_3.pdf` |
| Figure 4 (OOD time series) | `python tools/paper/fig4_timeseries_ood.py <data_root> Figure_4.pdf` |
| Figure 5 (DAgger at every tuning iteration) | `python tools/paper/fig5_dagger_sweep.py <data_root> Figure_5.pdf`, which also prints the counts quoted in the text |
| DAgger selection, labels, validation errors, confidences at iteration 100 | `python tools/paper/student_stats.py <data_root>` |
| Converged teacher solves on completed and not completed flights | `python tools/paper/teacher_solves.py <data_root>` |

Note that `summarize_final_eval.py` and `paired_tests.py` read every CSV under their input folder. Run them
on `d1_final_evaluation`: its CI-run folders sort before `local_timing`, so the CI rows are kept; only the file and
duplicate counts in the header differ from a run on a folder that holds only the two `ci_run_*` folders.
