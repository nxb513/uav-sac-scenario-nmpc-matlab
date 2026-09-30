function e = d1_track_err(X, Xr)
%D1_TRACK_ERR Per-step position tracking error ||p_k - p_ref,k+1|| (X(:,k) is the state
% after step k, Xr(:,k) = Xref(:,k+1)), capped at 5 m; a diverged step (NaN state)
% counts as the cap. Same definition as the training reward.
e = vecnorm(X(1:3,:) - Xr(1:3,:));
e(~isfinite(e)) = 5; e = min(e, 5);
end
