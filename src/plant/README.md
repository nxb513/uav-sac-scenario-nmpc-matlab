# src/plant

Plant phi tuyến 12 trạng thái `x = [p; eta; v; omega]`, đầu vào `u = [T; tau_phi; tau_theta; tau_psi]`,
dùng bởi D1:

- `quad_dynamics` (gọi `quad_rotm_zyx`, `quad_euler_rates_zyx`, `quad_saturate_input`,
  `quad_disturbance`), `quad_step_rk4`: mô hình và tích phân RK4 (`d1_plant_step`,
  `d1_joint_plant_params`, LQR `d1_build_lqr`, teacher `d1_teacher_build_solver`).
- `quad_sample_uncertainty` (gọi `quad_apply_uncertainty`, `quad_inertia_consistent`): mẫu
  plant bất định của đánh giá cuối (`d1_final_eval`).

Các hàm plant cũ không còn dùng đã chuyển sang `legacy/src/plant/`.
