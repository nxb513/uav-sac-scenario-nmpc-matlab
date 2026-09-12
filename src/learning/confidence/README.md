# src/learning/confidence

Confidence estimator học xác suất thực nghiệm rằng bounded surrogate trực tiếp
giữ sai số trong envelope trong `Hc` bước tới.

Calibrated confidence `c_k` mô tả độ tin cậy của surrogate; arbitration weight
`alpha_k` là đại lượng riêng được tạo từ confidence qua mapping chọn bằng
validation.

Tuyệt đối không đưa future error vào feature; future error chỉ dùng tạo nhãn.

Full Step-6 artifact hiện tại dùng primary `sac150_d336`, feature F3 219D,
envelope `Hc=20`, position `0.15 m`, attitude `2 deg`, architecture
`[128,64]` SiLU/dropout 0.1 và temperature scaling. Artifact deploy được lưu
tại `results/step6_confidence_full_v1_sac150_d336/frozen_confidence_artifact.mat`.

`T_ID` chỉ dùng báo cáo sau freeze. `V_arb` được giữ cho mapping confidence sang
arbitration weight ở bước hybrid; OOD vẫn khóa cho bước 7.
