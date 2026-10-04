# src

Code MATLAB của phương pháp D1 (chỉ gồm các file D1 gọi tới).

- `joint/`: **phương pháp D1** — mọi định nghĩa dùng chung của pipeline huấn luyện và đánh giá
  cuối (xem `joint/README.md`, `docs/D1_method.pdf`).
- `plant/`: plant 12 trạng thái (`quad_dynamics`, `quad_step_rk4`, mẫu plant bất định).
- `common/`: sinh/hoàn thiện quỹ đạo tham chiếu.
- `analysis/`: `d1_finite_horizon_contraction` (nhãn c_LQR) và `lqr_lyapunov_metric`.

Code của pipeline cũ (`controllers/`, `rl/`, `learning/`, `metrics/` và các hàm không dùng)
nằm ở `legacy/src/` (xem `legacy/README.md`).

Không để script nháp, figure xuất tạm, hoặc dữ liệu sinh ra trong `src/`.
