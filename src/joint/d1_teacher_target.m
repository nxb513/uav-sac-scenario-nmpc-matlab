function [Yx, Yu] = d1_teacher_target(Xref, k, F, cfg)
%D1_TEACHER_TARGET Teacher tracking target for stages k..k+N, consistent with the known
% external force F (world frame, held constant over the horizon): positions and
% velocities are the reference's; attitude, body rates and the feedforward input are
% re-completed by differential flatness (quad_complete_flat_reference, nominal model)
% with the thrust vector m (a_ref + g e3) + Dv v_ref - F. This is the steady-state
% target of offset-free MPC: with a target that ignored F, the attitude/thrust cost
% would pull against the tilt that cancels the wind and leave a position offset.
% F = 0 gives the reference's own attitude/rates and its flat feedforward input.
% Yx: 12 x (N+1) (stages 0..N), Yu: 4 x N (stages 0..N-1).
N = cfg.N; nc = size(Xref, 2);
mg = 3;                                  % margin: 3 nested finite differences stay centered
i0 = max(1, k - mg); i1 = min(nc, k + N + mg);
[W, Uw] = quad_complete_flat_reference(Xref(:, i0:i1), cfg.Ts, cfg.plant.nominal, ...
    cfg.refYaw, F(:));
c = (k:k+N) - i0 + 1;
Yx = W(:, c); Yu = Uw(:, c(1:N));
end
