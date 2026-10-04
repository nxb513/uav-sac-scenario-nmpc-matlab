# src/controllers/scenario_nmpc

Uncertain/scenario NMPC teacher.

Controller chỉ biết nominal parameters và miền bất định, không biết `theta_plant` thật của episode.

File chính:

- `scenario_nmpc_solve.m`: giải một bước scenario NMPC với cùng chuỗi input cho nhiều scenario.
- `scenario_nmpc_teacher_rollout.m`: chạy closed-loop episode để tạo trajectory teacher.
