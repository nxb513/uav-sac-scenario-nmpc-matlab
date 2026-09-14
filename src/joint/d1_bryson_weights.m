function [Q0, R0] = d1_bryson_weights(P)
%D1_BRYSON_WEIGHTS Bryson-rule seed weights Q0,R0 (physical normalization).
% Q0_ii = 1/e_allow_i^2, R0_jj = 1/du_allow_j^2. Seed only; SAC tunes around this.
eAllow = [0.10;0.10;0.10; deg2rad(5)*[1;1;1]; 0.30;0.30;0.30; 2;2;2];
duAllow = [P.m*P.g; 0.5; 0.5; 0.25];
Q0 = diag(1 ./ eAllow.^2);
R0 = diag(1 ./ duAllow.^2);
end
