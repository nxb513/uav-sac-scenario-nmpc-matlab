function theta = quad_apply_uncertainty(nominal, xi, rho)
%QUAD_APPLY_UNCERTAINTY Map normalized xi in [-1,1] to plant parameters.

xi = xi(:);
rho = rho(:);
if numel(xi) ~= 14 || numel(rho) ~= 14
    error('quad_apply_uncertainty:BadInputSize', ...
          'xi and rho must each have 14 elements.');
end
if any(abs(xi) > 1 + 1e-12)
    error('quad_apply_uncertainty:OutOfRange', ...
          'All normalized uncertainty samples xi must be in [-1,1].');
end

scale = 1 + rho .* xi;
if any(scale <= 0)
    error('quad_apply_uncertainty:NonPositiveScale', ...
          'Uncertainty scale produced a non-positive physical parameter.');
end

theta = nominal;
theta.m = nominal.m * scale(1);

Jdiag = diag(nominal.J);
Jdiag = Jdiag .* scale(2:4);
theta.J = diag(Jdiag);

theta.Dv = nominal.Dv(:) .* scale(5:7);
theta.Domega = nominal.Domega(:) .* scale(8:10);
theta.alphaT = nominal.alphaT * scale(11);
theta.alphaTau = nominal.alphaTau(:) .* scale(12:14);

theta.inputLimits.T = [0.0; 4.0 * theta.m * theta.g];
end
