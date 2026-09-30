function v = d1_getenv_str(name, dflt)
%D1_GETENV_STR String environment variable, or dflt when unset.
s = getenv(name); if isempty(s), v = dflt; else, v = s; end
end
