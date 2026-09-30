function v = d1_getenv_num(name, dflt)
%D1_GETENV_NUM Numeric environment variable, or dflt when unset.
s = getenv(name); if isempty(s), v = dflt; else, v = str2double(s); end
end
