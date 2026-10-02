# src/joint — shared definitions of method D1

Every function here is used by `experiments/run_d1_joint_pipeline.m` (SAC phase, DAgger
phase, diagnostics) and/or `experiments/d1_final_eval.m` (final evaluation), so training
and evaluation cannot drift apart. Do not copy these functions into the scripts. The method
is documented in `docs/D1_method.pdf`.

| Function | Role |
|---|---|
| `d1_config` | the single configuration (constants, env overrides, actuator box, hover input, DAgger settings) |
| `d1_joint_plant_params`, `d1_bryson_weights` | nominal plant parameters, Bryson Q0, R0 |
| `d1_build_lqr` | Bryson LQR on the nominal hover linearization (K, Riccati P) |
| `d1_train_cases` | the 120-case training reference bank (with flat feed-forward Uref) |
| `d1_sample_scenarios`, `d1_teacher_build_solver` | teacher model scenarios; acados scenario NMPC with the wind-force parameter (budget: 50 SQP / 100 QP iterations) |
| `d1_action_to_QR`, `d1_set_teacher_weights` | SAC action to teacher Q, R |
| `d1_teacher_target` | wind-consistent flat reference of the teacher |
| `d1_teacher_step`, `d1_teacher_reset` | one teacher step (usable rule, solve time), solver reset |
| `d1_wind_now` | the current wind force (the teacher's privileged information) |
| `d1_sample_wind`, `d1_dryden` | random training wind (synthetic; optional dedicated random stream) |
| `d1_case_len`, `d1_plant_step`, `d1_track_err`, `d1_sat` | common flight rules |
| `d1_fhat`, `d1_student_init`, `d1_student_push`, `d1_student_feature` | the linear student's force estimate, memory and 27 features |
| `d1_fly_student` | teacher-free flight of the student (alpha = 1), the blend, or the LQR |
| `d1_dagger_case`, `d1_dagger_run` | one DAgger data flight; the DAgger loop, selection, confidences, student file |
| `d1_ridge_fit`, `d1_student_stability`, `d1_val_set` | closed-form ridge with case-grouped CV; linearized stability check; validation flights |
| `d1_consolidate`, `d1_fit_logistic`, `d1_cs_feature`, `d1_conf_feature`, `d1_predict_logistic` | c_S and c_LQR |
| `d1_blend_control` | the deployed controller |
| `d1_alpha_bar` | optional contraction cap (alpha_safe) |
| `d1_load_deployed` | load the student file (W and confidences) |
