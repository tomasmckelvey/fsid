function [y,x] = dtlsim(sys,u,x0);
if nargin<3,
    x0 = [];
end
[a,b,c,d] = sys{:};
n = size(a,1);
if nargin<3,
    x = ltitr(a,b,u);
else
    x = ltitr(a,b,u,x0);
end
y = u*d.' + x*c.'; 
