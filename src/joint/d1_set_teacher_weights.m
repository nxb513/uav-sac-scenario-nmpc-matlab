function d1_set_teacher_weights(solver, Q, R, cfg)
%D1_SET_TEACHER_WEIGHTS Stage weights W = blkdiag(Q/M x M, R) on stages 0..N-1
% (terminal weight stays the build-time Bryson Q0/M).
Wblk = repmat({Q/cfg.M}, 1, cfg.M); Wblk{end+1} = R;
W = blkdiag(Wblk{:});                                 % (M*12+4) x (M*12+4)
for s = 0:cfg.N-1, solver.set('cost_W', W, s); end
end
