function d1_teacher_reset(teacher, Xref, k, cfg)
%D1_TEACHER_RESET Clear ALL solver memory (iterates, multipliers, QP warm start) and
% re-seed the initial guess along the reference from step k. Called at the start of
% every flight, after a divergence restart and after an unusable solve: a reused solver
% must never carry a failed solve's state into later solves.
teacher.reset();
M = cfg.M; N = cfg.N; nc = size(Xref,2);
for j = 0:N
    teacher.set('init_x', [repmat(Xref(:,min(k+j,nc)),M,1); cfg.uh], j);
end
for j = 0:N-1
    teacher.set('init_u', zeros(4,1), j);
end
end
