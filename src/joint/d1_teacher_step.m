function [u, status, usable, tsolve] = d1_teacher_step(teacher, x, uprev, Xref, k, F, cfg)
%D1_TEACHER_STEP One step of the SAC-NMPC teacher with PRIVILEGED wind knowledge: the
% current external force F (world frame; exact, simulation only) enters the prediction
% model (acados parameter, held constant over the horizon) and the wind-consistent target
% (d1_teacher_target). The teacher knows neither future gusts nor the true plant
% parameters. The iterate is applied when the solver converged (status 0) or hit its
% iteration cap (status 2) with a finite result (= "usable": applied, used as a label,
% not a failure); otherwise u_prev is held and the solver is reset before the next step.
% tsolve = wall time of the solve [s] (logged only; it never changes the result).
N = cfg.N; M = cfg.M;
[Yx, Yu] = d1_teacher_target(Xref, k, F, cfg);
for s = 0:N-1
    teacher.set('cost_y_ref', [repmat(Yx(:,s+1),M,1); Yu(:,s+1)], s);
end
teacher.set('cost_y_ref_e', repmat(Yx(:,N+1),M,1));
for s = 0:N
    teacher.set('p', F(:), s);
end
teacher.set('constr_x0', [repmat(x,M,1); uprev]);
t0 = tic;
teacher.solve();
tsolve = toc(t0);
status = teacher.get('status'); du0 = teacher.get('u', 0);
usable = any(status == [0 2]) && all(isfinite(du0));
if usable
    u = uprev + du0;
else
    u = uprev;
    d1_teacher_reset(teacher, Xref, min(k+1, size(Xref,2)-N), cfg);
end
end
