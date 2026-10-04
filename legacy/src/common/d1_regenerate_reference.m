function [Xref, options, groupId, Uref] = d1_regenerate_reference( ...
        family, speed, accel, rep, refCfg, theta, sampleTime, sampleCount)
%D1_REGENERATE_REFERENCE Deterministically rebuild one D1 case reference.
%
% Seeds the RNG from the case group id (d1_case_seed) then draws the reference
% options and builds the 12-by-N reference trajectory through the SAME pipeline
% the S0 bank used, so every stage reconstructs the identical reference from
% (family, speed, accel, rep) alone. The 4th output Uref (4-by-N) is the nominal
% differential-flatness feedforward input aligned column-wise with Xref; it is
% used as the input-reference for the teacher's control-deviation penalty.

groupId = sprintf('%s|v%g|a%g|r%d', family, speed, accel, rep);
rng(d1_case_seed(groupId), 'twister');
options = quad_sample_targeted_reference_options(family, refCfg, speed, accel);
[Xref, ~, Uref] = quad_targeted_reference_trajectory(family, sampleTime, ...
    sampleCount, options, theta);
end
