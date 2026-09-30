function [sur, conf] = d1_load_deployed(S, confFile, required)
%D1_LOAD_DEPLOYED The deployed controller of a checkpoint = the consolidated surrogate and
% conf saved TOGETHER in conf_seed<s>[_iter<NNNN>].mat by the pipeline's consolidation.
% S: loaded checkpoint struct. required = true (final evaluation) -> a missing conf file is
% an error; otherwise (diagnostics) the training surrogate is used with an untrained c_S
% head and no c_LQR (g_L = 1), with a warning.
if nargin < 3, required = false; end
sur = S.sur; conf = [];
if ~isempty(confFile) && isfile(confFile)
    C = load(confFile); conf = C.conf;
    if isfield(C, 'sur'), sur = C.sur; end
elseif required
    error('d1_load_deployed:noconf', 'deployed pair (conf file) missing: %s', confFile);
else
    fprintf('WARN no conf file (%s): training surrogate, untrained c_S head, no c_LQR (g_L = 1)\n', confFile);
end
end
