function du = d1_sur_predict_du(sur, feat)
%D1_SUR_PREDICT_DU Predicted residual Delta_u (physical units) from a raw feature.
z = single(feat) ./ sur.featScale;
p = predict(sur.net, dlarray(z,'CB'), 'Outputs', 'du');
du = double(extractdata(p(:))) .* sur.resHalf;
end
