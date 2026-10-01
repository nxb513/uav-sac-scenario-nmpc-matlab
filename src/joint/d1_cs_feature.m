function psi = d1_cs_feature(e, phi)
%D1_CS_FEATURE The 18 cheap features of the c_S logistic: the 16 c_LQR error features
% (d1_conf_feature) + |F_hat| + |u_ref - u_h| (both read from the student features phi).
psi = [d1_conf_feature(e); norm(phi(13:15)); norm(phi(16:19))];
end
