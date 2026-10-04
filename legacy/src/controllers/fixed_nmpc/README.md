# src/controllers/fixed_nmpc

Baseline fixed NMPC với `Q`, `R`, `N` cố định.

Mục tiêu: tạo chuẩn so sánh trước khi thêm RL tuning.

File chính:

- `fixed_nmpc_solve.m`: giải một bước receding-horizon trên nominal dynamics.
- `fixed_nmpc_rollout.m`: chạy closed-loop fixed NMPC với `thetaPlant` ẩn tách khỏi nominal model.
