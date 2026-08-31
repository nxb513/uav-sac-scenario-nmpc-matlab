function uHover = quad_hover_input(theta)
%QUAD_HOVER_INPUT Generalized hover input for the z-up plant.

theta = require_theta(theta);
uHover = [theta.m * theta.g / theta.alphaT; 0; 0; 0];
end

function theta = require_theta(theta)
if nargin == 0 || isempty(theta)
    cfg = step1_plant_config();
    theta = cfg.nominal;
end
end
