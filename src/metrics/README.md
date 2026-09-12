# src/metrics

Metric và báo cáo:

- tracking RMSE/P95/max;
- imitation error;
- control effort/smoothness;
- constraint violation;
- computation time;
- confidence calibration;
- risk-coverage.

File Gate A:

- `summarize_control_episode.m`: metric tracking, input, constraint, solver và solve time.
- `estimate_pipeline_workload.m`: ngoại suy thời gian SAC và sinh dataset teacher từ runtime đo được.
- `write_gate_a_summary.m`: ghi báo cáo Markdown cô đọng cho một run chính thức.
- `summarize_disturbance_episode.m`: peak response, recovery time và force/torque disturbance metrics.
