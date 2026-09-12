# src

Code MATLAB chính của dự án.

- `plant/`: plant 12-state theo phương pháp mới.
- `controllers/`: fixed NMPC và scenario/uncertain NMPC teacher.
- `rl/`: environment và policy tune `Q`, `R`, `N`.
- `learning/`: surrogate, residual NN và confidence estimator.
- `common/`: tiện ích chung.
- `metrics/`: metric và báo cáo.

Không để script nháp, figure xuất tạm, hoặc dữ liệu sinh ra trong `src/`.
