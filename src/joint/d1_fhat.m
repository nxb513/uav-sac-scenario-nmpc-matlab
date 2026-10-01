function F = d1_fhat(xPrev, uPrev, x, cfg)
%D1_FHAT External world-frame force estimated from the last step (the student's view of
% the wind). Nominal translational dynamics m v' = alphaT T R(eta) e3 - m g e3 - Dv v + F,
% integrated over one step with the trapezoid rule (thrust T_{k-1} held):
%   F_k = m (v_k - v_{k-1})/Ts + m g e3 - alphaT T_{k-1} (R(eta_{k-1}) + R(eta_k)) e3 / 2
%         + Dv (v_{k-1} + v_k) / 2
% with nominal m, alphaT, Dv. Exact zero at hover equilibrium without wind.
nom = cfg.plant.nominal; m = nom.m; Ts = cfg.Ts;
F = m*(x(7:9) - xPrev(7:9))/Ts + [0; 0; m*nom.g] ...
    - nom.alphaT*uPrev(1)*(thrust_dir(xPrev(4:6)) + thrust_dir(x(4:6)))/2 ...
    + nom.Dv(:).*(xPrev(7:9) + x(7:9))/2;
end

function c = thrust_dir(eta)
% third column of the ZYX rotation R(phi, theta, psi): body z axis in the world frame
cp = cos(eta(1)); sp = sin(eta(1)); ct = cos(eta(2)); st = sin(eta(2));
cy = cos(eta(3)); sy = sin(eta(3));
c = [cy*st*cp + sy*sp; sy*st*cp - cy*sp; ct*cp];
end
