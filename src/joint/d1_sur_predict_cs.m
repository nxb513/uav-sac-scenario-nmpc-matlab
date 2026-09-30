function c = d1_sur_predict_cs(sur, feat)
%D1_SUR_PREDICT_CS Predicted surrogate confidence c_S in [0,1] from a raw feature.
z = single(feat) ./ sur.featScale;
p = predict(sur.net, dlarray(z,'CB'), 'Outputs', 'cs');
c = double(extractdata(p(1)));
end
