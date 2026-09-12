# src/learning/surrogate

Surrogate controller nhẹ học theo teacher RL-NMPC.

Input feature dự kiến:

`chi_k = [x_history, u_history, future_reference, prediction_residual]`

Output phải bị giới hạn theo actuator constraints.
