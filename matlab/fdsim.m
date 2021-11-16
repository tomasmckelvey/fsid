function [y,x] = fdsim(sys, u, z, xt)
% $$$ Calculates the output given input and state-space model in Fourier domain.
% $$$ 
% $$$   ``x[i,:] = inv(z[i]*I-a)*[B xt]*[u[i,:]; z[i]]``\\
% $$$   ``y[i,:] = d*u[i,:] + c*x[i,:]``
% $$$ 
% $$$ Parameters
% $$$ ==========
% $$$ `sys`:      typle `sys = {a, b, c, d}` or
% $$$             `sys = {a, b, c}`
% $$$             where
% $$$             `a` is a  square matrix of size (n,n), `b` is 
% $$$             a matrix of size (n,m), `c` is a matrix of size (p,n)
% $$$             and (optionally) `d` is a matrix of size (p,m).\\
% $$$ `u`:        an array of input vectors such that `u[i,:]` is 
% $$$             the input vector of size m at sample index `i`. 
% $$$             The array has size (N,m)\\
% $$$ `z`:        vector with the samples of the frequency function argument\\
% $$$ xt:         transient vector of size n, Default value is the zero vector. 
% $$$     
% $$$ Returns
% $$$ =======
% $$$ `y`:          the resulting output sequence of size (N,p)\\
% $$$ `x`:          the resulting state sequence of size (N,p)
% $$$ 
    [nwu, m] = size(u);
    nn = length(sys);
    if nn == 3,
        [a, b, c] = sys{:}
        [p, nc] = size(c);
        [nr, m] = size(b);
        d = zeros(p, m);
    elseif nn == 4 
        [a, b, c, d] = sys{:};
        [p, nc] = size(c);
        [nr, m] = size(b);
    else
        error("fdsim: Incorrect number of matrices in sys.");
    end

    y = zeros(nwu, p);
    
    if nargin == 4,
        ue = [u, z(:)];
        be = [b, xt];
        x = ltifd(a, be, ue, z);
    else
        x = ltifd(a, b, u, z);
    end

    for widx = 1:nwu,
        y(widx, :) = (c * x(:, widx)) + (d * u(widx, :).');
    end
end
