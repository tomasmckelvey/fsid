function [sys, s] = ffsid(w, ff, n, q, dtype, estimd, CT, T, W)
% function [sys, s] = ffsid(fddata, n, q, dtype, estimd, CT, T, W)
% $$$ Estimate a state-space model from frequency function data
% $$$ 
% $$$     Estimate a DT or CT state-space model from I/O frequency data
% $$$     
% $$$     Determines the (a,b,c,d,xt) parametrers such that (DT case)
% $$$      sum_i   || ff[i,:,:] - d + c*inv(z[i]*eye(n)-A)* b ||^2_w[i,:,:]
% $$$     is small where z[i] = np.exp(1j*w[i])
% $$$     and CT Case
% $$$      sum_i   ||  ff[i,:,:] - d - c*inv(1j*w[i]*eye(n)-A)*b ||^2_w[i,:,:]
% $$$ 
% $$$     The weighted norm  || x ||^2_w is defined as  || w * x ||_F^2 where w is a square matrix such that w*w^H 
% $$$     is a positive definite matrix.  
% $$$     If the noise on ff[i,:,j] j=1..m is a zero mean rv with covariance R(i,:,:) a BLUE estimator will be obtained if 
% $$$     W(i,:,:) is selected as the square root of the inverse of the
% $$$     covariance matrix R(i,:,:) 
%         i.e. W(i,:,:)*R(i,:,:)*W(i,:,:)' = eye(p)
% $$$  
%     
%     Parameters
%     ==========
%     w : 
%         vector of frequencies in rad/sample [-pi,pi] or [0, 2pi]
%         if CT = True unit in radians/s (-inf, +inf)
%     ff: 
%         matrix of frequency function data. ffdata[i,:,:] is the frequency response matrix
%         at frequency w[i]
%     n : integer
%         the model order of the ss-model (a,b,c,d), i.e. a is a size (n x n) matrix
%     q : integer
%         the numer of block rows used in the intermediate matrix. Must satisfy q>n
%     dtype : data type, optional
%         if dtype = 'Real' a real valued solution (a,b,c,d) is returned.
%         if dtype = 'Complex' a complex valued solution (a,b,c,d) is returned.
%     estimd: optional
%         if set to False no d matrix is esimated and a zero d matrix is returned
%     CT: if set to true a continuous time (CT) model is esimated
%          if set to false a discrete time (DT) model is esimated (default)
%     T :  a frequency scaling factor for the bilinear transformation used when CT=True. 
%         Default is 1. If CT=False parameter T is disregarded 
%     W : weight matrices W(i,:,:) is the wight matrix for ffdata index i 
%
% $$$ Returns
% $$$ =======
% $$$ sys : cell array
% $$$ `sys{1} = a`:          the estimated `a` matrix  
% $$$ `sys{2} = b`:          the estimated `b` matrix  
% $$$ `sys{3} = c`:          the estimated `c` matrix  
% $$$ `sys{4} = d`:          the estimated `d` matrix (or zero matrix if `estimd=False`)  
% $$$ `sys{5} = x`:          vector set to zero 
% $$$  `s`:                  vector of the singular values   
  

estTrans = false;
if nargin<9
    W = [];
end
if nargin<8
    T = [];
end
if nargin<7
    CT = false;
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

fddata = ffdata2fddata(w,ff);
[sys, s] = fdsid(fddata , n, q, false, dtype, estimd, CT, T, W);

    