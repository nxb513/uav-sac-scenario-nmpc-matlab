# src/joint — shared definitions of method D1

Every function here is used by **both** `experiments/run_d1_joint_pipeline.m` (training,
consolidation, diagnostics) and `experiments/d1_final_eval.m` (final evaluation), so
training and evaluation cannot drift apart. Do not copy these functions into the scripts.
The method is documented in `docs/D1_method.pdf`.

| Function | Role |
|---|---|
| `d1_config` | the single configuration (constants, env overrides, actuator box, hover input) |
| `d1_joint_plant_params`, `d1_bryson_weights` | nominal plant parameters, Bryson Q0, R0 |
| `d1_build_lqr` | Bryson LQR on the nominal hover linearization (K, Riccati P) |
| `d1_train_cases` | the 120-case training reference bank |
| `d1_sample_scenarios`, `d1_teacher_build_solver` | teacher model scenarios; acados scenario NMPC with the wind-force parameter |
| `d1_action_to_QR`, `d1_set_teacher_weights` | SAC action to teacher Q, R |
| `d1_teacher_target` | wind-consistent flat reference (x̄, ū) of the teacher |
| `d1_teacher_step`, `d1_teacher_reset` | one teacher step (usable rule), solver reset |
| `d1_wind_now` | the current wind force (the teacher's privileged information) |
| `d1_sample_wind`, `d1_dryden` | random training wind (synthetic) |
| `d1_case_len`, `d1_plant_step`, `d1_track_err`, `d1_sat` | common flight rules |
| `d1_hist_init`, `d1_hist_push`, `d1_hist_feature`, `d1_feature_scale` | surrogate input history, prediction residual, 208-D feature and its scale |
| `d1_sur_predict_du`, `d1_sur_predict_cs` | surrogate heads |
| `d1_blend_control` | the deployed controller |
| `d1_conf_feature`, `d1_predict_logistic`, `d1_alpha_bar` | c_LQR feature and model, optional contraction cap |
| `d1_load_deployed` | load the deployed pair (consolidated surrogate and conf) |
