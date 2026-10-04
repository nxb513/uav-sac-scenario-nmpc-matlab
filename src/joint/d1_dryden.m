function [sg, L] = d1_dryden(U, cfg)
%D1_DRYDEN Low-altitude turbulence intensities and scale lengths (MIL-F-8785C, Sec. 3.7.3.4,
% Figs. 10-11) for mean wind U [m/s] at the
% reference height h = cfg.windHft [ft]: sigma_w = 0.1 U,
% sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4,
% L_u = L_v = h / (0.177 + 0.000823 h)^1.2, L_w = h.
% sg = [sigma_u sigma_v sigma_w] [m/s], L = [L_u L_v L_w] [m].
h = cfg.windHft; a = 0.177 + 0.000823*h; ft = 0.3048;
sw = 0.1*U; su = sw/a^0.4;
Lu = h/a^1.2*ft; Lw = h*ft;
sg = [su, su, sw]; L = [Lu, Lu, Lw];
end
