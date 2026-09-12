function seed = d1_case_seed(groupId)
%D1_CASE_SEED Deterministic uint32 RNG seed from a case group id (via MD5).
%
% Lets any D1 stage regenerate the exact reference for a case from its group id
% alone (family|v..|a..|r..), independent of generation order, so the S0 bank,
% the S3 teacher grid, the S4 LQR screen and the S5 audit all see the identical
% reference for a given case.

md = java.security.MessageDigest.getInstance('MD5');
raw = mod(double(md.digest(uint8(char(groupId)))), 256);
seed = raw(1) + raw(2) * 256 + raw(3) * 65536 + raw(4) * 16777216;  % < 2^32
seed = uint32(mod(seed, 2^32 - 1) + 1);  % avoid 0
end
