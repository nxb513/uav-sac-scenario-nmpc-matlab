function [Xref, options, groupId] = d1_regenerate_reference( ...
        family, speed, accel, rep, refCfg, theta, sampleTime, sampleCount)
%D1_REGENERATE_REFERENCE Deterministically rebuild one D1 case reference.
%
% Seeds the RNG from the case group id (d1_case_seed) then draws the reference
% options and builds the 12-by-(sampleCount+1) reference trajectory through the
% SAME pipeline the S0 bank used, so every stage reconstructs the identical
% reference from (family, speed, accel, rep) alone.

groupId = sprintf('%s|v%g|a%g|r%d', family, speed, accel, rep);
rng(d1_case_seed(groupId), 'twister');
options = quad_sample_targeted_reference_options(family, refCfg, speed, accel);
Xref = quad_targeted_reference_trajectory(family, sampleTime, sampleCount, ...
    options, theta);
end
