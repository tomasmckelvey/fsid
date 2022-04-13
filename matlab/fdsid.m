function [sys, s] = fdsid(fddata, n, q, estTrans, dtype, estimd, CT, T, w)
% function [sys, s] = fdsid(fddata, n, q, estTrans, dtype, estimd, CT, T, w)
%      Estimate a DT or CT state-space model from I/O frequency data
%      
%      Determines the (a,b,c,d,xt) parametrers such that (DT case)
%       sum_i   || y(i,:) - d*u[i, :] + c*inv(z(i)*eye(n)-A)*[b,  xt]* [u(i, :); z(i)]||^2_w(i,:,:)
%      is small where z(i) = np.exp(1j*w(i))
%      and CT Case
%       sum_i   ||  y(i,:) - d*u(i, :) + c*inv(1j*w(i)*eye(n)-A)*b* u(i, :) ||^2_w(i,:,:)
%  
%      The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w^H 
%      is a positive definite matrix.  
%      If the noise on y(i,:).T is a zero mean rv with covariance r(i,:,:) a BLUE estimator will be obtained if 
%      w(i,:,:) is selected as the square root of the inverse of the
%      covariance matrix r(i,:,:)
%   
%  Parameters
%  ==========
%  fddata:   a cell array with elements 
%              fddata{1} = w 
%                  a vector of frequencies, if CT=false rad/sample if CT=true, rad/s
%              fddata{2} = y, 
%                  a matrix of the output frequency data where y(i,:) corresponds to z(i),
%              fddata{3} = u 
%                  a matrix of the input frrequency data where u(i,:) corresponding to z(i)
%  n:          the model order of the ss-model
%  q:          the numer of block rows used in the intermediate matrix. Must satisfy q>n
%  Optional
%  --------
%  estTrans:   if true, a compensation for the transient term will be estimated (default)
%  dtype:      if dtype = 'Real'  a real valued solution (a,b,c,d) ...
%                 is returned. (default)
%              if dtype =  'Complex' a complex valued solution is returned.
%  estimd:     if set to false no d matrix is esimated and a zero ...
%               d matrix is returned (default is true)
%  CT:         if true a CT model is estimated and estTrans is forced false. If false (default) a DT model is estimated
%  T:          a frequency scaling factor for the bilinear transformation used when CT=true. 
%              Default is 1. If CT=false parameter T is disregarded 
%  
%  Returns
%  =======
%  sys:             cell array
%    sys{1} = a:          the estimated a matrix  
%    sys{2} = b:          the estimated b matrix  
%    sys{3} = c:          the estimated c matrix  
%    sys{4} = d:          the estimated d matrix (or zero matrix if estimd=false) 
%    sys{5} = x:          vector of the transient compensation
%  s:                a vector of the singular values   
    if nargin<9
        w=[];
    end
    if nargin<8
        T=[];
    end
    if nargin<7
        CT=false;
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

    if CT
        if isempty(T)
            T = 0.1/max(z/2/pi);
        end
        estTrans = false;
        [sys, s] = fdsid({cf2df(z,T), yd, ud}, n, q, ...
                                      estTrans, dtype, true, false, ...
                                      [], w);
        sys = bilinear_d2c(sys, T);
        [a,b,c,d] = sys{:};
        [b, d] = fdestim_bd(1j*z, yd, ud, a, c, estTrans, dtype, estimd, w);
        [c, d] = fdestim_cd(1j*z, yd, ud, a, b, 0, dtype, estimd, w);
        [b, d] = fdestim_bd(1j*z, yd, ud, a, c, estTrans, dtype, estimd, w);
        [c, d] = fdestim_cd(1j*z, yd, ud, a, b, 0, dtype, estimd, w);
        xt = zeros(n, 1);
        sys = {a,b,c,d,xt};
    else
        [sys, s] = gfdsid({exp(1j*z), yd, ud}, n, q, estTrans, dtype, ...
                          estimd, w);
    end
end

function wd = cf2df(wc,T)
    wd = 2*atan(wc*T/2);
end

function sys =  bilinear_d2c(sys, T)
%    """ Calculates the bilinear transformation D->C for ss-system sys """

    [a, b, c, d] = sys{:};
    n =  size(a,1);
    ainv = inv(eye(n)+a);
    ac = ainv * (a-eye(n))*2/T;
    bc = ainv*b*2/sqrt(T);
    cc = (c*ainv)*2/sqrt(T);
    dc = d - c*ainv*b; 
    sys =  {ac, bc, cc, dc};
end
