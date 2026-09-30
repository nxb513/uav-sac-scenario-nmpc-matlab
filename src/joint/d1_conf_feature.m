function F = d1_conf_feature(E)
%D1_CONF_FEATURE c_LQR feature of error states E (12 x n): the 12 errors with wrapped
% Euler angles + 4 group magnitudes (pos / att / vel / rate) -> 16 x n.
Ew = E; Ew(4:6,:) = mod(Ew(4:6,:)+pi, 2*pi) - pi;
pos = vecnorm(Ew(1:3,:)); att = vecnorm(Ew(4:6,:));
vel = vecnorm(Ew(7:9,:)); rate = vecnorm(Ew(10:12,:));
F = [Ew; pos; att; vel; rate];
end
