function s = d1_student_init(x, cfg)
%D1_STUDENT_INIT Student memory at the start of a flight or after a divergence restart:
% no previous step yet, so the force estimate is zero until the first step is pushed.
s.xPrev = x; s.uPrev = cfg.uh; s.valid = false;
end
