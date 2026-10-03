function [nW, teacherC] = d1_teacher_pool(cfg, scen, teacher)
%D1_TEACHER_POOL Local workers for the parallel flights of one SAC / DAgger iteration.
% teacher is the client's solver (built by d1_teacher_build_solver, which generated and
% compiled the acados code in cfg.codegenDir). Each worker loads that code once into its
% own solver (d1_teacher_build_solver(..., reuse = true)) through a parallel.pool.Constant,
% so no two flights ever share a solver. Every flight starts with d1_teacher_reset and sets
% the teacher weights itself, and its wind is drawn by the client beforehand, so a flight's
% result does not depend on which worker flies it or on the number of workers.
% nW = number of workers for parfor (0 = serial on the client); teacherC.Value = the solver
% to use inside the parfor body.
nW = 0; teacherC = struct('Value', teacher);
if cfg.nWorkers < 1, return; end
if ~license('test', 'Distrib_Computing_Toolbox') || isempty(ver('parallel'))
    fprintf('POOL Parallel Computing Toolbox unavailable -> serial flights\n'); return;
end
p = gcp('nocreate');
if isempty(p)
    env = {'ACADOS_INSTALL_DIR', 'ACADOS_SOURCE_DIR', 'LD_LIBRARY_PATH', 'CASADI_DIR', 'ENV_RUN'};
    env = env(~cellfun(@(v) isempty(getenv(v)), env));
    % the default profile allows one worker per PHYSICAL core; the CI runner has 4 vCPUs
    clu = parcluster('Processes'); clu.NumWorkers = max(clu.NumWorkers, cfg.nWorkers);
    p = parpool(clu, cfg.nWorkers, 'IdleTimeout', Inf, 'EnvironmentVariables', env);
end
nW = p.NumWorkers;
teacherC = parallel.pool.Constant(@() d1_teacher_build_solver(cfg, scen, true));
fprintf('POOL %d workers, each with its own teacher solver loaded from %s\n', nW, cfg.codegenDir);
end
