function [b, d, xt] = fdestim_bd(z, yd, ud, a, c,  estTrans, dtype, estimd, w)
%  function [b, d, xt, resid] = fdestim_bd(z, yd, ud, a, c,  estTrans, dtype, estimd, w)
%  Estimate b and d matrices ( and optionally xt) given yd, ud, a and c matrices
%      
%  Calulates the b and d matrices (and optimally xt) for a linear dynamic system in state-space form given the a and c 
%  matrices and samples of frequency domain function data. It solves 
%  
%  if estimd = true and estTrans = true
%    min_{b,d,xt} sum_i || ([d 0] + c*inv(z(i)*I-a)*[b xt])[ud(i,:); z(i)]  - yd(i,:,:) |^2_w(i,:,:)
%  
%  if estimd=false and estTrans = true 
%    min_{b,xt} sum_i|| (c*inv(z(i)*I-a)*[b xt])[ud(i,:); w(i)]  - yd(i,:,:) |^2__w(i,:,:)
%  
%  if estimd = true and estTrans=false 
%    min_{b,d} sum_i || (d+ c*inv(z(i)*I-a)*b)ud(i,:)  - yd(i,:,:) |^2__w(i,:,:)
%  
%  if estimd=false and estTrans=false \\
%    min_{b} sum_i|| (c*inv(z(i)*I-a)*b)ud(i,:)  - yd(i,:,:) |^2_w(i,:,:)
% 
%      The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square full rank matrix matrix.
%      If the noise on y(i,:).' is a zero mean rv with covariance r(i,:,:) a BLUE estimator will be obtained if 
%      w(i,:,:) is selected as inv(chol(r(i,:,:))') 
%  
%  if dtype='Real' a real valued solution is calulated. if dtype = 'Complex' 
%  the solution is complex valued
%      
%  Parameters
%  ----------
%  z:          vector with the samples of the function argument where z(i) is argument for index i
%  yd:         output frequency data yd(i,:)
%  ud:         input frequency data ud(i,:)
%  a:          matrix
%  c:          matrix
%  
%  Optional
%  --------  
%  estTrans:   if set to true also an xt vector will be estimated capturing the transient effect
%  dtype:       data type of model, either 'Real' or 'Complex'
%  estimd:     if set to false no d matrix is esimated and a zero d matrix is returned
%  w:          w(i,:,:) is weighting matrix as described above. if w = [ ]. No weighting is used.  
%  
%  Returns 
%  -------
%  b:          the LS-optimal b matrix
%  d:          LS-optimal d matrix  zeros matrix if estimd=false
%  xt:         LS-optimal xt vector if estTrans=true
% 
%  T. McKelvey, Nov 2020    
    
    if  nargin<9,
        w = [];
    end
    p = size(yd, 2);
    m = size(ud, 2);
    n = size(a, 1);
    z = z(:);
    nw = length(z);
    ydw = zeros(size(yd));
    if ~isempty(w)
        for widx=1:nw
            ydw(widx,:) = yd(widx,:)*squeeze(w(widx,:,:)).';
        end
    else
        ydw = yd;
    end
    

    if estTrans
        ude = [ud z];
        me = m + 1;
    else
        ude = ud;
        me = m;
    end
    fkern1 = ltifr2(a.', c.', z); 

    fkern = permute(fkern1, [3, 2, 1]); 

    eyep = eye(p);

    if estimd      
        r = zeros(nw*p, me*n+m*p); %  room for B xt and D
        if isempty(w)
            for widx=1:nw
                for midx=1:me
                    r(p*(widx-1) + 1:p*widx, n*(midx-1) + 1:n*midx) = ude(widx, midx) * fkern(:, :, widx);
                end
                for midx=1:m
                    r(p*(widx-1) + 1:p*widx, me*n + p*(midx-1) + 1:me*n+p*midx) = ud(widx, midx) * eyep;
                end
            end
        else
            for widx=1:nw
                for midx=1:me
                    r(p*(widx-1) + 1:p*widx, n*(midx-1) + 1:n*midx) = squeeze(w(widx,:,:)) *ude(widx, midx) * fkern(:, :, widx);
                end
                for midx=1:m
                    r(p*(widx-1) + 1:p*widx, me*n + p*(midx-1) + 1:me*n+p*midx) = squeeze(w(widx,:,:))*(ud(widx, midx) * eyep);
                end
            end
        end
    else
        r = zeros(nw*p, me*n); %# room for B xt   
        if isempty(w)
            for widx=1:nw
                for midx=1:me
                    r(p*(widx-1) + 1:p*widx, (n*(midx-1)) + 1:n*midx) = ude(widx, midx) * fkern(:, :, widx);
                end
            end
        else
            for widx=1:nw
                for midx=1:me
                    r(p*(widx-1) + 1:p*widx, (n*(midx-1)) + 1:n*midx) = squeeze(w(widx,:,:))*ude(widx, midx) * fkern(:, :, widx);
                end
            end
        end
        
    end

    if isequal(dtype,'Real')
        rh = [real(r); imag(r)];
        lh = [real(reshape(ydw.',numel(ydw),1)); imag(reshape(ydw.',numel(ydw),1))];
    else
        rh = r;
        lh = reshape(ydw.',numel(ydw),1);
    end
    
    vecBD = rh\lh;
    resid = rh*vecBD - lh;
    vecB = vecBD(1:n*me);

    if estimd
        vecD = vecBD(n*me + 1:end);
    else
        vecD = zeros(m*p, 1);
    end

    b = zeros(n, m);
    d = zeros(p, m);
    b(:) = vecB(1:n*m);
    d(:) = vecD;

    if estTrans
        xt = vecB(n*m+1:n*(m+1));
    else
        xt = zeros(n,1);
    end
end
