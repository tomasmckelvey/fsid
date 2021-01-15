function [c, d] = fdestim_cd(z, yd, ud, a, b, xt, dtype, estimd, w)
%
%function fdestim_cd(z, yd, ud, a, b, xt, dtype, estimd, w)
%    """Estimate c and d matrices given z, yd, ud and a, b, and (optionally xt) matrices
%    
%    Calulates the c and d matrices for a state-space representation of a rational function 
%    given the a and b matrices and samples of the rational matrix function data in the form of 
%    output vectors yd[i,:] and input vectors ud[i,:] corresponding to the rational function 
%    evaluated at complex value z[i]. It solves 
%
%    if estimd=true and xt~=0    
%    min_{c,d} sum_i || ([d 0] + c*inv(z[i]*I-a)*[b xt])*[ud[i,:]; z[i]]  - yd[i,:] |^2_w[i,:,:]
%
%    if estimd=False and and xt~=0     
%    min_{c} sum_i|| (c*inv(z[i]*I-a)*[b xt])*[ud[i,:]; z[i]]  - yd[i,:] |^2_w[i,:,:]
%
%    if estimd=True and xt==0    
%    min_{c,d} sum_i || (d+ c*inv(z[i]*I-a)*b)*ud[i,:]  - yd[i,:] |^2_w[i,:,:]
%
%    if estimd=False and xt==0    
%    min_{c} sum_i|| (c*inv(z[i]*I-a)*b)*ud[i,:]  - yd[i,:] |^2_w[i,:,:]
%    
%    The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w^H 
%    is a positive definite matrix.  
%    If the noise on y[i,:].T is a zero mean rv with covariance r[i,:,:] a BLUE estimator will be obtained if 
%    w[i,:,:] is selected as linalg.cholesky(linalg.inv(r[i,:,:]])).T.conj()
%
%
%    if dtype='Real' a real valued solution is calulated. if dtype='Complex' 
%    the solution is complex valued
%    
%    Parameters
%    ----------
%    yd : matrix_like 
%        frequency data packed in a matrix. yd[i,:] is the output vector 
%        corresponding to sample i
%    ud : matrix_like 
%        frequency data packed in a matrix. ud[i,:] is the output vector 
%        corresponding to sample i
%    z : matrix_like
%        vector with the complex data z[i].
%    a : matrix_like
%        the a matrix
%    b : matrix_like
%        the b matrix
%    dtype : string, optional
%        data type of model either 'Real' or 'Complex'
%    estimd : logical, optional
%        if set to false no d matrix is esimated and a zero d matrix is returned
%      w: matrix like, optional
%        w[i,:,:] is the weighting matrix for data sample i, See above. 
%    
%    Returns 
%    -------
%    
%    c : matrix_like
%        the LS-optimal c matrix
%    d : matrix_like
%        the LS-optimal d matrix
%
%    """
    if  nargin<9,
        w = [];
    end
    if norm(xt)<eps
        estTrans = false;
    else
        estTrans = true;
    end
    p = size(yd, 2);
    m = size(ud, 2);
    n = size(a, 1);
    z = z(:);
    nz = length(z);
    if estTrans
        ude = [ud, z];
        be = [b, xt];
    else
        ude = ud;
        be = b;
    end
    
    fkern = ltifd(a, be, ude, z);

    if estimd
        r = zeros(n+m, nz);
        r(1:n, :) = fkern;
        r(n+1:end, :) = ud.';
    else
        r = fkern;
    end
    if ~isempty(w)
        nr = size(r,1);
        rw = zeros(p*nz, nr*p);
        ydw = zeros(p*nz,1);
        for zidx = 0:nz-1 %Vectorize data and apply pre-whitening filter
            ydw(p*zidx+1:p*(zidx+1)) = squeeze(w(zidx+1,:,:))*yd(zidx+1,:).';
            rw(p*zidx+1:p*(zidx+1),:) = kron(r(:,zidx+1).', squeeze(w(zidx+1,:,:)));
        end
        if isequal(dtype,'Real')
            lh = [real(ydw);  imag(ydw)];
            rh = [real(rw);  imag(rw)];
        else
            lh = ydw;
            rh = rw;
        end
        vecCD = rh\lh;
        c = zeros(p,n);
        c(:) = vecCD(1:n*p);
        if estimd
            d = zeros(p,m);
            d(:) = vecCD(n*p+1:end);
        else
            d = zeros(p,m);
        end
    else
        if isequal(dtype,'Real')
            rh = [real(r), imag(r)];
            lh = [real(yd); imag(yd)];
        else
            lh = yd;
            rh = r;
        end
        cd = (rh.'\ lh).';
        c = cd(:,1:n);
        if estimd
            d = cd(:,n+1:end);
        else
            d = zeros(p,m);
        end
    end
end