function fkern = ltifd(a,b,u,z)
% function fkern = ltifd(a,b,u,z)
%    """Calculates the nz x n frequency kernel with input u
%        fkern[:, i] = inv(eye(n)*z[i] - a)*b*u[i, :] 
n = size(a,1);
m = size(b,2);
nz = length(z);
fk = ltifr2(a,b,z);
fkern = zeros(n,nz);
for zidx = 1:length(z)
    fkern(:,zidx)=reshape(fk(zidx,:,:),n,m)*u(zidx,:).';
end
