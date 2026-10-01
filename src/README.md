# src

Code MATLAB chính của dự án.

- `joint/`: **phương pháp D1 (đang dùng)** — mọi định nghĩa dùng chung của pipeline huấn luyện
  và đánh giá cuối (xem `joint/README.md`, `docs/D1_method.pdf`).
- `plant/`: plant 12-state (D1 dùng `quad_dynamics`, `quad_step_rk4`).
- `common/`: tiện ích chung (D1 dùng các hàm sinh/hoàn thiện quỹ đạo tham chiếu).
- `analysis/`: D1 dùng `d1_finite_horizon_contraction` (nhãn c_LQR).
- `controllers/`, `rl/`, `learning/`, `metrics/`: thuộc **pipeline cũ** (trước D1, xem
  `docs/legacy_pipeline.md`); D1 không dùng (học trò tuyến tính của D1 nằm trong `joint/`).

Không để script nháp, figure xuất tạm, hoặc dữ liệu sinh ra trong `src/`.
