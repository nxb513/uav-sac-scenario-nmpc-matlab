function F = d1_wind_now(ds, t, x, u, theta)
%D1_WIND_NOW External world-frame force the plant receives at time t in state x -- the
% teacher's privileged wind information (exact in simulation, unavailable to the deployed
% controllers). Zero without wind.
F = zeros(3,1);
if ~isempty(ds)
    d = ds(t, x, u, theta); F = d.force(:);
end
end
