# src/plant

Plant phi tuyến 12 trạng thái sẽ được triển khai ở đây:

`x = [p; eta; v; omega]`

Input controller:

`u = [T; tau_phi; tau_theta; tau_psi]`

Plant mới không dùng lại các file `uav2_*.m` cũ.

`quad_generate_disturbance_episode.m` sinh realization có seed cho constant, gust, sinusoidal và colored-stochastic external force/moment; realization này chỉ truyền vào plant và được giữ ẩn với NMPC.
