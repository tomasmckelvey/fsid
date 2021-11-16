function [fddata, wnew] = ffdata2fddata(z,ffdata,w)
% function [znew, fddata, wnew] = ffdata(zw,ffdata,w)
if nargin>2
    error('w not implemented yet!')
end
nd = ndims(ffdata);
if nd == 3
    [nw,p,m] = size(ffdata);
elseif nd == 2
    [nw,p] = size(ffdata);
    m = 1;
elseif nd == 1
    nw = size(ffdata);
    m = 1;
    p = 1;
else
    error('ffdata wrong format');
end
if nargin<3
    w=[];
end

u = zeros(nw*m, m);
y = zeros(nw*m, p);
znew = zeros(nw*m,1);
b0 = eye(m);
if nd==3
    for midx = 0:m-1
        y(midx*nw+1:(midx+1)*nw, :) = ffdata(:, :, midx+1);
        u(midx*nw+1:(midx+1)*nw, :) = kron(ones(nw,1),b0(midx+1, :));
        znew(midx*nw+1:(midx+1)*nw) = z(:);
    end
elseif nd==1 || nd==2
    y = ffdata;
    u = ones(nw,1);
    znew = z(:);
end
fddata = {znew, y, u};


