# src/common

Tiện ích quỹ đạo tham chiếu dùng bởi D1:

- `quad_sample_targeted_reference_options`, `quad_targeted_reference_trajectory`,
  `quad_reference_trajectory`: sinh quỹ đạo tham chiếu theo họ/tốc độ/gia tốc (ngân hàng
  huấn luyện `d1_train_cases`, cấu hình `d1_config`, đánh giá cuối `d1_final_eval`).
- `quad_complete_flat_reference`: hoàn thiện tham chiếu phẳng (trạng thái + đầu vào), cũng
  dùng trong đích nhất quán với gió của teacher (`d1_teacher_target`).
- `d1_case_seed`: seed cố định theo tên quỹ đạo (`d1_train_cases`).

Các tiện ích cũ không còn dùng đã chuyển sang `legacy/src/common/`.
