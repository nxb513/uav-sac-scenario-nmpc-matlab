function warmStart = rl_nmpc_resize_warm_start(previous, horizon, theta)
%RL_NMPC_RESIZE_WARM_START Resize a prior input sequence to a new horizon.

if isempty(previous)
    warmStart = nmpc_default_warm_start(theta, horizon);
    return;
end
if size(previous, 2) >= horizon
    warmStart = previous(:, 1:horizon);
else
    warmStart = [previous, repmat(previous(:, end), 1, ...
        horizon - size(previous, 2))];
end
end
