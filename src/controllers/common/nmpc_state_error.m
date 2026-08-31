function E = nmpc_state_error(X, Xref)
%NMPC_STATE_ERROR Tracking error with wrapped Euler-angle components.

E = X - Xref;
E(4:6, :) = mod(E(4:6, :) + pi, 2*pi) - pi;
end
