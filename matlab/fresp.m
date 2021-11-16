function fkern =  fresp(z, a, b, c, d)
%function fkern =  fresp(a, b, c, d, z)
%    Calculates the (nz,p,m) size frequency function
%        fkern[i,:,:] = d+c*inv(z[i]*eye(n)-a)*b    
%
m = size(b, 2);
p = size(c,1);
nz = length(z);
fkern = zeros(nz,p,m);    
for midx=1:m
    fkern(:,:,midx) = (c*ltifr(a,b(:,midx),z)).';
end
dd = reshape(d,1,p,m);
for i=1:nz,
        fkern(i,:,:) = fkern(i,:,:) + dd;
end
