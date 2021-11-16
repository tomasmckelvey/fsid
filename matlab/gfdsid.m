function [sys, s] = gfdsid(fddata, n, q, estTrans, dtype, estimd, w)
% function [sys, s] = gfdsid(fddata, n, q, estTrans, dtype, estimd, w)
%    Estimate a state-space model from vector valued I/O rational data
%   
%    Determines the (a,b,c,d,xt) parameters such that 
%     sum_i || y(i,:) - d*u(i, :) + c*inv(z(i)*eye(n)-A)*[b,  xt]* [u(i, :); z(i)] ||^2_w(i,:,:)
%     is minimized 
%    If estrTrans=False the following problem is solved for the (a,b,c,d) parametrers
%     sum_i ||y(i,:) - d*u(i, :) + c*inv(z(i)*eye(n)-A)*b * u(i, :) ||^2_w(i,:,:)
%     is minimized 
%
%    The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w' 
%    is a positive definite matrix.  
%    If the noise on y(i,:) is a zero mean rv with covariance r(i,:,:) a BLUE estimator will be obtained if 
%    w(i,:,:) is selected such that w(i,:,:)*r(i,:,:)*w(i,:,:)' = eye(p)
%    (for example by calulating the cholesky factor of r and
%    inverting it.)
%  
%  Parameters
%  ==========
%  fddata:   cell array with elements 
%              fddata{1} = z 
%                  vector of complex scalars,
%              fddata{2} = y, 
%                  matrix of the output vector data where y(i,:) corresponds to z(i),
%              fddata{3} = u 
%                  matrix of the input vector data where u(i,:) corresponding to z(i)
%  n:          the model order of the ss-model
%  q:          the numer of block rows used in the intermediate matrix. Must satisfy q>n
%  Optional
%  --------
%  estTrans:   if true, a compensation for the transient term will be estimated (default)
%  type:       if type = 'Real'  a real valued solution (a,b,c,d) ...
%              is returned. (default)
%              if type =  'Complex' a complex valued solution is returned.
%  estimd:     if set to false no d matrix is esimated and a zero ...
%              d matrix is returned (default is true)
%  Returns
%  =======
%  sys: cell array
%    sys{1} = a:          the estimated a matrix  
%    sys{2} = b:          the estimated b matrix  
%    sys{3} = c:          the estimated c matrix  
%    sys{4} = d:          the estimated d matrix (or zero matrix if estimd=false)  
%    sys{5} = xt:         vector of the transient compensation
%  s:                     vector of the singular values   
    if nargin<7
        w=[];
    end
    if nargin<6
        estimd = [];
    end
    if isempty(estimd)
        estimd = true;
    end
    if nargin<5
        dtype = [];
    end
    if isempty(dtype)
        dtype = 'Real';
    end
    if nargin<4
        estTrans = [];
    end
    if isempty(estTrans)
        estTrans = true;
    end
    
    [z, yd, ud]  = fddata{:};
    z = z(:);

    [nwy, p] = size(yd);
    [nwu, m] = size(ud);

    if estTrans
        ude = [ud z];
        me = m + 1;
    else
        ude = ud;
        me = m;
    end
    nw = size(z, 1);

    if nw ~= nwy
        error('Error: z and Y sizes does not match!')
    end
    if nw ~= nwu
        error('Error: z and Y sizes does not match!')
    end 
       
    y = zeros(p*q,nw);
    if estimd
       u = zeros(me*q,nw);
    else
       u = zeros(me*(q-1),nw);
    end
    wf = ones(nw,1);
    if ~isempty(w)
        for widx = 1:nw
            wf(widx) = trace(squeeze(w(widx,:,:)))/p;
        end
    end
    for widx=1:nw
        y(1:p, widx) = wf(widx)*yd(widx, :);
        u(1:me, widx) = wf(widx)*ude(widx, :);
        zx = z(widx);
        for qidx=2:q
            y((qidx-1)*p + 1:qidx*p, widx) = zx*wf(widx)*yd(widx, :);
            if estimd || qidx<q
                u((qidx-1)*me + 1:qidx*me, widx) = zx*wf(widx)*ude(widx, :);
            end
            zx = zx*z(widx);
        end
    end
    if isequal(dtype, 'Real')
        hu = [real(u) imag(u)];
        hy = [real(y) imag(y)];
    else
        hu = u;
        hy = y;
    end

    h = [hu; hy];
    r = triu(qr(h.')).';
    if estimd
        r22 = r(me*q + 1:end, me*q + 1:end);
    else
        r22 = r(me*(q-1) + 1:end, me*(q-1) + 1:end);
    end
    [u, s, ~] = svd(r22,0);
    c = u(1:p, 1:n);
    lh = u(1:p*(q-1), 1:n);
    rh = u(p+1:p*q, 1:n);
    s = diag(s);

    a = lh\rh;

    if estTrans
        [b, d, xt] = fdestim_bd(z, yd, ud, a, c,  estTrans, ...
                                       dtype, estimd, w);
        [c, d] = fdestim_cd(z, yd, ud, a, b, xt,   dtype, ...
                                   estimd, w);
        [b, d, xt] = fdestim_bd(z, yd, ud, a, c,  estTrans, ...
                                       dtype, estimd, w);
        [c, d] = fdestim_cd(z, yd, ud, a, b, xt,   dtype, ...
                                   estimd, w);
    else
        [b, d] = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w);
        [c, d] = fdestim_cd(z, yd, ud, a, b, 0, dtype, estimd, w);
        [b, d] = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w);
        [c, d] = fdestim_cd(z, yd, ud, a, b, 0, dtype, estimd, w);
        xt = zeros(n, 1);
    end

    sys = {a, b, c, d, xt};
    
end
