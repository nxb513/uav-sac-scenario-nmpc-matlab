function s = d1_student_push(s, x, u)
%D1_STUDENT_PUSH Remember the state x at which input u was applied (call after a
% non-diverged plant step); the next feature uses it for the force estimate.
s.xPrev = x; s.uPrev = u; s.valid = true;
end
