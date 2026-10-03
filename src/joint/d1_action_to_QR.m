function [Q, R] = d1_action_to_QR(a, cfg)
%D1_ACTION_TO_QR SAC action a in R^6 (unbounded) -> teacher weights: a = log10 of the
% multipliers of 6 weight groups (pos, att, vel, rate, thrust, torques) of the base Q0, R0,
% i.e. mult = 10.^a; a = 0 gives exactly the base.
[Q0, R0] = qr_base(cfg);                            % Bryson, or random (no-Bryson)
mult = 10.^a(:);
q = diag(Q0);
q(1:3)=q(1:3)*mult(1); q(4:6)=q(4:6)*mult(2);
q(7:9)=q(7:9)*mult(3); q(10:12)=q(10:12)*mult(4);
r = diag(R0); r(1)=r(1)*mult(5); r(2:4)=r(2:4)*mult(6);
Q = diag(q); R = diag(r);
end

function [Q0, R0] = qr_base(cfg)
% Default = Bryson (1/e_allow^2, 1/du_allow^2). D1_RANDOM_QR=1 -> random diagonal
% weights, log-uniform 10^cfg.rqrLog per element, deterministic per seed. Only the
% TEACHER base changes; the LQR keeps Bryson in both.
if cfg.randomQR
    rs = RandStream('twister', 'Seed', cfg.seed + 90210);   % independent of global rng
    lo = cfg.rqrLog(1); span = cfg.rqrLog(2) - cfg.rqrLog(1);
    Q0 = diag(10.^(lo + span*rand(rs,12,1)));
    R0 = diag(10.^(lo + span*rand(rs,4,1)));
else
    [Q0, R0] = d1_bryson_weights(cfg.plant);
end
end
