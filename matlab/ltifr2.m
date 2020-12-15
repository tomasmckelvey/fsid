function fkern =  ltifr2(a, b, z)
%    Calculates the (nz,n,m) size frequency kernel
%        fkern[i,:,:] = inv(z[i]*eye(n)-a)*b    
%    """
n = size(a, 1);
m = size(b, 2);
nz = length(z);
fkern = zeros(nz,n,m);    
for midx=1:m
    fkern(:,:,midx) = ltifr(a,b(:,midx),z).';
end