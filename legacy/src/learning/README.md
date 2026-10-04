# src/learning

Các khối học từ teacher:

- `surrogate/`: học `u_teacher` từ feature online.
- `confidence/`: ước lượng xác suất surrogate giữ sai số trong envelope tương lai.

`residual/` chỉ giữ decision note lịch sử. Bounded residual correction NN đã
bị loại khỏi active pipeline theo quyết định `DEC-044`; không triển khai hoặc
train module này.

Train/validation/test phải chia theo episode.
