@doc raw"""
    lrm(u, y, n = 2, Nw = nothing)


"""
function lrm(u::AbstractArray, y::AbstractArray; n = 2, Nw = nothing)
    us = size(u)
    if length(us) == 1
        m = 1
        u = reshape(u, (us[1], 1))
    else
        m = us[2]
    end
    Nu = us[1]

    ys = size(y)
    if length(ys) == 1
        p = 1
        y = reshape(y, (ys[1], 1))
    else
        p = ys[2]
    end
    Ny = ys[1]

    if Nu != Ny
        println("lrm: Incorrect number of rows in u and y.")
        return false
    end

    if Nw === nothing
        Nw = 2Int(ceil(((m + 2)*n + 1)/2))
    end

    if Nw < Int(ceil(((m + 2)*n + 1)/2))
        println("Error: Nw too small.")
        return false
    end

    yf = fft(y, 1)
    uf = fft(u, 1)
    ff = zeros(Complex{typeof(first(u))}, Ny, p, m)
    iset = -Nw:Nw
    R = vander(iset, n+1, true)
    yfe = vcat(yf[end-Nw+1:end, :], yf, yf[1:Nw, :])
    ufe = vcat(uf[end-Nw+1:end, :], uf, uf[1:Nw, :])

    for i = 1:Ny
        for pidx = 1:p
            yy = yfe[Nw .+ i .+ iset, pidx]
            RR = hcat(R, diagm(-yy) * R[:, 1:n])

            for midx = m:-1:1
                RR = hcat(RR, diagm(ufe[Nw .+ i .+ iset, midx]) * R)
            end

            ht = RR\yy
            ht = reverse(ht; dims = 1)
            ff[i, pidx, :] = ht[1:(n + 1):(n + 1)*m]
        end
    end

    return ff
end

@doc raw"""
    kung_realization(mp::AbstractArray{T}, n::Int, q::Int = 0) where T <: Number

Calulates the state-space realization `(a, b, c)` from Markov parameters using Kung's relization algorithm.

Parameters:
* `m`: array of Marov parameters `m[i, j - 1, k - 1]` hold row `j` and column `k` of the Markov parameter of sample `i`
* `n`: model order state-space system
Optional:
* `q`: number of rows in Hankel matrix. The default is `n + 1`

Returns:
* `a`: system matrix
* `b`: input matrix
* `c`: output matrix
"""
function kung_realization(mp::AbstractArray{T}, n::Int, q::Int = n + 1) where T <: Number
    N = size(mp, 1)
    p = size(mp, 2)
    m = size(mp, 3)
    ncol = N - q + 1

    if ncol < n
        println("n, q and N are not compatible. N >= q + n - 1 must be satisfied")
        return false
    end

    H = Matrix{T}(undef, p*q, ncol*m)
    for j = 1:ncol
        for i = 1:q
            H[p*(i - 1) + 1:i*p, m*(j - 1) + 1:j*m] = mp[i + j - 1, :, :]
        end
    end

    u, s, vh = svd(H)
    c = u[1:p, 1:n]
    lh = u[1:p*(q-1), 1:n]
    rh = u[p+1:end, 1:n]
    a = lh\rh
    b = diagm(s[1:n]) * vh[1:m, 1:n]'

    return a, b, c, H
end

@doc raw"""
    markov(sys::Tuple, N::Int)

Calculate markov parameters from `sys = (a,b,c)`.

Parameters:
* `sys`: `(a, b, c)` state-space matrices
* `N`: numer of Markov parameters to generate

Returns:
* `mp`: `mp[i, :, :]` is Markov parameter `C(A^i)B`
"""
function markov(sys::Tuple, N::Int)
    a, b, c = sys
    n = size(a, 1)
    m = size(b, 2)
    p = size(c, 1)
    mp = Array{typeof(first(a))}(undef, N, p, m) # change this to function acting on type

    aa = Matrix{typeof(first(a))}(I, n, n) # and this too
    for i in 1:N
        mp[i, :, :] = c * (aa * b)
        aa = aa * a
    end

    return mp
end


@doc raw"""
    make_sys_real(sys::Tuple)

Convert realization sys into a real-valued realization.

Parameters:
* `sys`: `(a, b, c)` or `(a, b, c, d)`

Returns:
* `sysr`: `(ar, br, cr, dr)` the realization with real valued matrices
"""
function make_sys_real(sys::Tuple)
    a, b, c = sys[1:3]
    n = size(a, 1)
    mp = markov(sys, 2n)
    mpr = real.(mp)
    a, b, c = kung_realization(mpr, n)

    if length(sys) == 3
        return (a, b, c)
    else
        return (a, b, c, real.(sys[4]))
    end
end


@doc raw"""
    make_obs_real(a::Matrix{<:Number}, c::Matrix{<:Number})

Convert `(a, c)`` into real-valued matrices by approximating a real valued observability range space to the original one.

Parameters:
* `a`: complex matrix
* `b`: complex matrix

Returns:
* `ar`: real valued matrix
* `br`: real valued matrix
"""
function make_obs_real(a::Matrix{<:Number}, c::Matrix{<:Number})

    n = size(a, 1)
    p = size(c, 1)
    obs = Matrix{Complex}(undef, p*(n + 1), n)
    obs[1:p, :] = c

    for i in 1:n
        obs[p*(i + 1):p*(i + 2), :] = obs[p*i:p*(i+1), :] * a
    end

    obsr = hcat(real.(obs), imag.(obs))
    u, s, vh = svd(obsr)
    c = u[1:p, 1:n]
    lh = u[1:p*n, 1:n]
    rh = u[p:end, 1:n]
    lsres = lh\rh
    a = lsres[1]

    return a, c
end


"""
    moebius(sys; alpha = 1, beta = 0, gamma = 0, delta = 1)

Calculates the bilinear transformation D->C for ss-system sys.
"""
function moebius(sys; alpha = 1, beta = 0, gamma = 0, delta = 1)
    a, b, c, d = sys
    n = size(a, 1)
    ainv = inv(alpha * I(n) - gamma*a)
    ac = (delta * a - beta * I(n)) * ainv
    bc = (alpha * delta - gamma * beta) * ainv * b
    cc = c * ainv
    dc = d + gamma * multidot(c, ainv, b)

    return ac, bc, cc, dc
end


"""
    moebius_arg(z; alpha = 1, beta = 0, gamma = 0, delta = 1)


"""
function moebius_arg(z; alpha = 1, beta = 0, gamma = 0, delta = 1)
    nz = size(z, 1)
    s = Vector{contype(Complex)}(undef, nz)

    for idx = 1:nz
        s[idx] = (alpha*z[idx] + beta)/(gamma*z[idx] + delta)
    end

    return s
end


"""
    moebius_inv(sys; alpha = 1, beta = 0, gamma = 0, delta = 1)

Calculates the bilinear transformation D->C for ss-system sys.
"""
function moebius_inv(sys; alpha = 1, beta = 0, gamma = 0, delta = 1)
    a, b, c, d = sys
    n =  size(a, 1)
    ainv = inv(delta * I(n) + gamma*a)
    ac = (alpha * a +beta * I(n)) * ainv
    bc = ainv * b
    cc = -(gamma * beta - alpha * delta) * (c * ainv)
    dc = d - gamma * multidot(c, ainv, b)

    return ac, bc, cc, dc
end


"""
    moebius_arg_inv(s; alpha = 1, beta = 0, gamma = 0, delta = 1)


"""
function moebius_arg_inv(s; alpha = 1, beta = 0, gamma = 0, delta = 1)
    nz = size(s, 1)
    z = Vector{contype(Complex)}(undef, nz)

    for idx = 1:nz
        z[idx]= (beta - delta*s[idx])/(gamma*s[idx] - alpha)
    end

    return z
end


"""
    uq_cond(z, q)

"""
function uq_cond(z, q)
    m = 1
    nw = size(z, 1)
    u = Matrix{contype(Complex)}(undef, m*q, nw*m)

    for widx = 1:nw
        u[1:m, (widx - 1)*m + 1:widx*m] = Matrix(I, m, m)
        zx = z[widx]
        for qidx = 1:q
            u[(qidx-1)*m + 1:qidx*m, (widx-1)*m + 1:widx*m] = zx*Matrix(I, m ,m)
            zx *= z[widx]
        end
    end

    return cond(u)
end


@doc raw"""
    function fresp_slow!(
        frsp::AbstractArray{<:Number, 3},
        z::AbstractVector{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        c::AbstractMatrix{<:Number},
        d::AbstractMatrix{<:Number}
    )

Frequency response of state-space model `(a, b, c, d)` (slow version) calulated as

``fresp[i,:,:] = d + c * (I*z[i]-a)^{-1} * b``
"""
function fresp_slow!(
    frsp::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number},
    d::AbstractMatrix{<:Number}
)
    n = size(a, 1)
    m = size(b, 2)
    p = size(c, 1)
    nw = size(z, 1)

    for widx in 1:nw
        frsp[widx, :, :] = multidot(c, ( (I * z[widx] - a) \ b) ) + d
    end

    return nothing
end

"""
    function fresp_fast!(
       frsp::AbstractArray{<:Number, 3},
        z::AbstractVector{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        c::AbstractMatrix{<:Number},
        d::AbstractMatrix{<:Number},
        eig::Eigen
    )

Frequency response of ss-model `(a, b, c, d)` (fast version)

``fresp[i,:,:] = d+c*inv(I*z[i]-a)*b``
"""
function fresp_fast!(
    frsp::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number},
    d::AbstractMatrix{<:Number},
    eig::Eigen
)
    n, m = size(b)
    nw = length(z)
    o = ones(n)
    b = eig.vectors\ b
    c = c * eig.vectors

    for widx in 1:nw
        frsp[widx, :, :] = multidot(c, diagm(0 => (1 ./ (o * z[widx] - eig.values)) ), b) + d
    end

    return nothing
end

"""
    function fresp(
        z::AbstractVector{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        c::AbstractMatrix{<:Number},
        d::AbstractMatrix{<:Number},
        noWarning::Bool = true
    )

Frequency response of ss-model `(a,b,c,d)` a rational matrix function

``fresp[i,:,:] = d+c*inv(z[i]*I-a)*b``

`fresp[i,:,:]` is the function value of the rational matrix function evaluated at `z[i]`

Parameters:
* `z`: vector with samples of the function argument
* `a`: matrix
* `b`: matrix
* `c`: matrix
* `d`: matrix
* `noWarning`: information message is suppressed if set to true

Returns:
* `frsp`: frequency response
"""
function fresp(
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number},
    d::AbstractMatrix{<:Number},
    noWarning::Bool = true
)
    n, m = size(b)
    p = size(c, 1)
    nw = size(z, 1)
    eig = eigen(a)
    frsp = zeros(ComplexF64, nw, p, m)

    if rank(eig.vectors) < n
        if !noWarning
            println("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        end

        fresp_slow!(frsp, z, a, b, c, d)
    else
        fresp_fast!(frsp, z, a, b, c, d, eig)
    end

    return frsp
end

"""
    function ltifr_slow!(
        fkern::AbstractArray{<:Number, 3},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number}
    )

Calculates the frequency kernel in place as (slow version)

``fkern[i,:,:] = inv(z[i]*I-a)*b``
"""
function ltifr_slow!(
    fkern::AbstractArray{<:Number, 3},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number}
)
    n, m = size(b)
    nw = length(z)

    for widx in 1:nw
        fkern[widx, :, :] = (I * z[widx] - a)\b
    end

    return nothing
end

"""
    function ltifr_fast!(
        fkern::AbstractArray{<:Number, 3},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number},
        eig::Eigen
    )

Calculates the frequency kernel in place (fast version)

``fkern[i,:,:] = inv(z[i]*I-a)*b``
"""
function ltifr_fast!(
    fkern::AbstractArray{<:Number, 3},
    a::AbstractMatrix{<:Number}, # why?
    b::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number},
    eig::Eigen
)
    n, m = size(b)
    nw = length(z)
    o = ones(n)

    bb = eig.vectors\b
    for widx in 1:nw
        da = o * z[widx] - eig.values
        fkern[widx, :, :] = multidot(eig.vectors, diagm(0 => (1 ./ da)), bb)
    end

    return nothing
end

"""
    function ltifr(
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number},
        noWarning::Bool = true
    )

Calculates the frequency kernel

``fkern[i,:,:] = inv(z[i]*I-a)*b``

Parameters:
* `a`: matrix
* `b`: matrix
* `z`: vector with the samples of the function argument
* `noWarning`: if true suppress information message

Returns:
* `fkern`: frequency response
"""
function ltifr(
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number},
    noWarning::Bool = true
)
    n, m = size(b)
    nw = length(z)
    fkern = zeros(ComplexF64, nw, n, m)

    eig = eigen(copy(a))

    if rank(eig.vectors) < n
        if !noWarning
            println("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        end

        ltifr_slow!(fkern, a, b, z)
    else
        ltifr_fast!(fkern, a, b, z, eig)
    end

    return fkern
end

"""
    function ltifd_slow!(
        fkern::AbstractMatrix{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        u::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number}
    )

Calculates the frequency kernel with input `u` in place (slow version)

``fkern[:, i] = inv(z[i]*I - a)*b*u[i, :]``

Parameters:
* `fkern`: frequency kernel
* `a`: matrix
* `b`: matrix
* `u`: matrix with input vectors
* `z`: vector with the frequecy data
"""
function ltifd_slow!(
    fkern::AbstractMatrix{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    u::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number}
)
    n = size(a, 0)
    nw = length(z)
    eyen = Matrix{Float64}(I, n, n)

    for widx in 1:nw
        fkern[:, widx] = (eyen * z[widx] - a)\(b * u[widx, :])
    end

    return nothing
end


"""
    function ltifd_fast!(
        fkern::AbstractMatrix{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        u::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number},
        eig::Eigen
    )

Calculates the frequency kernel with input `u` in place (fast version)

``fkern[:, i] = inv(z[i]*I - a)*b*u[i, :] ``

Parameters:
* `fkern`: frequency kernel
* `a`: matrix
* `b`: matrix
* `u`: matrix with input vectors
* `z`: vector with the frequecy data
* `eig`: eigendecomposition of `a`
"""
function ltifd_fast!(
    fkern::AbstractMatrix{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    u::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number},
    eig::Eigen
)
    n, m = size(b)
    nw = length(z)
    o = ones(n)

    bb = eig.vectors\b
    for widx in 1:nw
        da = o * z[widx] - eig.values
        fkern[:, widx] = multidot(eig.vectors, diagm(0 => (1 ./ da)), bb * u[widx, :])
    end

    return nothing
end


"""
    function ltifd(
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        u::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number},
        noWarning::Bool = true
    )

Calculates the frequency kernel with input `u`

``fkern[:, i] = inv(z[i]*I - a)*b*u[i, :]``

Parameters:
* `a`: matrix
* `b`: matrix
* `u`: input vectors
* `z`: vector with samples of the function argument
* `noWarning`: if true suppress information message

Returns:
* `fkern`: frequency kernel
"""
function ltifd(
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    u::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number},
    noWarning::Bool = true
)
    n, m = size(b)
    nw = length(z)
    fkern = zeros(ComplexF64, n, nw)

    eig = eigen(a)

    if rank(eig.vectors) < n
        if !noWarning
            println("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        end
        ltifd_slow!(fkern, a, b, u, z)
    else
        ltifd_fast!(fkern, a, b, u, z, eig)
    end

    return fkern
end

"""
    ffdata2fddata(ffdata::AbstractArray{<:Number, 3}, z::AbstractVector{<:Number})

Converts `ffdata` to `fddata`.

Converts frequency function data `ffdata` to input/output data format `fddata`.

Parameters:
* `ffdata`: frequency function data in the format such that `ffdata[i,:,:]` corresponds to the frequency function matrix of size `(p,m)` at frequency index `i` corresponding to function argument `z[i]` at a total number of samples `nz`
* `z`: array with the corresponding frequency function argument of size `nz`

Returns:
* `u`: Fourier transform of input of size `(m*nz, m)`
* `y`: Fourier transform of output of size `(m*nz, p)`
* `zn`: frequency function argument vector of length `m*nw`
"""
function ffdata2fddata(
    ffdata::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number}
)
    nz, p, m = size(ffdata)
    u = Matrix{ComplexF64}(undef, nz*m, m)
    y = Matrix{ComplexF64}(undef, nz*m, p)
    zn = Vector{Float64}(undef, nz*m)
    b0 = Matrix{Float64}(I, m, m)

    for midx in 1:m
        y[(midx-1)*nz+1:midx*nz, :] = ffdata[:, :, midx]
        u[(midx-1)*nz+1:midx*nz, :] = repeat(transpose(b0[midx, :]), nz, 1)
        zn[(midx-1)*nz+1:midx*nz] = z[:]
    end

    return u, y, zn
end

"""
    function ltitr(
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number},
        u::AbstractMatrix{<:Number},
        x0::Union{AbstractVector{<:Number}, Number} = 0;
        type::Union{Type{<:Real}, Type{<:Complex}} = Real
    )

Calculates the time domain input to state respone.

Calculates the time domain state response

``x[i+1,:] = a*x[i,:]) + b*u[i,:]``

Parmeters:
* `a`: a square matrix of size `(n, n)`
* `b`: a matrix of size `(n, m)`
* `u`: an array of input vectors such that `u[i,:]` is the input vector of size `m` at time index `i`. The array has size `(N, m)`
* `x0`: intial vector of size `n`, i.e. `x[0,:] = x0`. Default value is the zero vector

Returns:
* `x`: the resulting state-sequence of size `(N, n)`
* `x[k,:]`: is the state at sample `k`
"""
function ltitr(
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number},
    u::AbstractMatrix{<:Number},
    x0::Union{AbstractVector{<:Number}, Number} = 0;
    type::Union{Type{<:Real}, Type{<:Complex}} = Real
)
    type = contype(type)

    n = size(a, 1)
    N, m = size(u)
    x = Matrix{type}(undef, N, n)
    x0 = x0 == 0 ? zeros(type, n, 1) : x0
    x[1, :] = x0

    for nidx in 1:N-1
        x[nidx+1, :] = a * x[nidx, :] + b * u[nidx, :]
    end

    return x
end

"""
    function lsim(
        sys::Tuple,
        u::AbstractMatrix{<:Number},
        x0::Union{AbstractVector{<:Number}, Number} = 0;
        type::Union{Type{<:Real}, Type{<:Complex}} = Real
    )

Calculates the time-domain output given input sequence and state-space model `sys = (a, b, c, d)`

``x[i+1,:] = a*x[i,:]) + b*u[i,:]``

``y[i,:] =  c*x[i,:]) + d*u[i,:]``

Parameters:
* `sys`: a typle `sys = (a, b, c, d)` or `sys = (a, b, c)` where `a` is a square matrix of size (n,n), `b` is a matrix of size (n,m), `c` is a matrix of size (p,n) and (optionally) `d` is a matrix of size `(p, m)``
* `u`: an array of input vectors such that `u[i,:]` is the input vector of size m at time index `i`. The array has size `(N, m)`
*Optional:*
* `x0`: intial vector of size n, i.e. `x[0,:] = x0`. Default value is the zero vector

Returns:
* `y`: the resulting output sequence of size (N, p)
* `x`: the resulting state sequence of size (N, n)
"""
function lsim(
    sys::Tuple,
    u::AbstractMatrix{<:Number},
    x0::Union{AbstractVector{<:Number}, Number} = 0;
    type::Union{Type{<:Real}, Type{<:Complex}} = Real
)
    type = contype(type)

    nn = size(sys, 1)
    if nn == 3
        a, b, c = sys
        p, nc = size(c)
        nr, m = size(b)
        d = zeros(p, m)
    elseif nn == 4
        a, b, c, d = sys
        p, nc = size(c)
        nr, m = size(b)
    else
        println("lsim: Incorrect number of matrices in sys.")
        return False
    end

    x = ltitr(a, b, u, x0; type=type)
    nu = size(u, 1)
    y = Matrix{type}(undef, nu, p)

    for idx in 1:nu
        y[idx, :] = c * x[idx, :] + d * u[idx, :]
    end

    return y, x
end

"""
    function fdsim(
        sys::Tuple,
        u::AbstractMatrix{<:Number},
        z::AbstractVector{<:Number},
        xt::AbstractVector{<:Number} = Vector{Float64}(undef, 0)
    )

Calculates the output given input and state-space model in Fourier domain

``x[i,:] = inv(z[i]*I-a)*[B xt]*[u[i,:]; z[i]]``

``y[i,:] = d*u[i,:] + c*x[i,:]``

Parameters:
* `sys`: typle `sys = (a, b, c, d)` or `sys = (a, b, c)` where `a` is a  square matrix of size (n,n), `b` is  a matrix of size (n, m), `c` is a matrix of size (p, n) and (optionally) `d` is a matrix of size (p, m).
* `u`: an array of input vectors such that `u[i,:]` is the input vector of size m at sample index `i`. The array has size (N, m)
* `z`: vector with the samples of the frequency function argument\\
* `xt`: transient vector of size n, Default value is the zero vector.

Returns:
* `y`: the resulting output sequence of size (N,p)
* `x`: the resulting state sequence of size (N,p)
"""
function fdsim(
    sys::Tuple,
    u::AbstractMatrix{<:Number},
    z::AbstractVector{<:Number},
    xt::AbstractVector{<:Number} = Vector{Float64}(undef, 0)
)
    nwu, m = size(u)
    nn = size(sys, 1)
    if nn == 3
        a, b, c = sys
        p, nc = size(c)
        nr, m = size(b)
        d = zeros(p, m)
    elseif nn == 4
        a, b, c, d = sys
        p, nc = size(c)
        nr, m = size(b)
    else
        println("fdsim: Incorrect number of matrices in sys.")
        return false
    end

    y = Matrix{ComplexF64}(undef, nwu, p)

    if length(xt) > 0
        ue = hcat(u, reshape(z, nwu, 1))
        be = hcat(b, reshape(xt, nr, 1))
        x = ltifd(a, be, ue, z)
    else
        x = ltifd(a, b, u, z)
    end

    for widx in 1:nwu
        y[widx, :] = (c * x[:, widx]) + (d * u[widx, :])
    end

    return y, x
end

@doc raw"""
    bilinear_d2c(sys::Tuple, T::Real = 1)

Calculates the bilinear transformation D->C for state-space system `sys`.

Parameters:
* `sys`: tuple of system matrices `(a, b, c, d)`
* `T`: frequency scaling factor

Returns:
* `a`: system matrix
* `b`: input matrix
* `c`: output matrix
* `d`: direct feedthrough matrix
"""
function bilinear_d2c(sys::Tuple, T::Real = 1)
    a, b, c, d = sys
    n = size(a, 1)
    ainv = inv(I + a)
    ac = (ainv * (a - I)) * 2/T
    bc = (ainv * b) * 2/sqrt(T)
    cc = (c * ainv) * 2/sqrt(T)
    dc = d .- multidot(c, ainv, b)

    return ac, bc, cc, dc
end

"""
    bilinear_c2d(sys::Tuple, T::Real = 1)

Calculates the bilinear transformation C->D for ss-system `sys`.

Parameters:
* `sys`: tuple of system matrices `(a, b, c, d)`
* `T`: frequency scaling factor

Returns:
* `a`: system matrix
* `b`: intput matrix
* `c`: output matrix
* `d`: direct feedthrough matrix
"""
function bilinear_c2d(sys::Tuple, T::Real = 1)
    a, b, c, d = sys
    n =  size(a, 1)
    ainv = inv(I * 2/T - a)
    ad = (I * 2/T + a) * ainv
    bd = (ainv * b) * 2/sqrt(T)
    cd = (c * ainv) * 2/sqrt(T)
    dd = d .+ multidot(c, ainv, b)

    return ad, bd, cd, dd
end

"""
    cf2df(wc::AbstractVector{<:Number}, T::Real)

Calculates the bilinear transformation frequency mapping C->D for frequency vector `wc`.

Parameters:
* `wc`: vector of frequencies
* `T`: frequency scaling factor

Returns:
* `wd`: vector of transformed frequencies
"""
cf2df(wc::AbstractVector{<:Number}, T::Real) = 2*atan.(wc*T/2)

@doc raw"""
    df2cf(wd::AbstractVector{<:Number}, T::Real)

Calculates the bilinear transformation frequency mapping D->C for frequency vector `wd`.

Parameters:
* `wd`: vector of frequencies
* `T`: frequency scaling factor

Returns:
* `wc`: vector of transformed frequencies
"""
df2cf(wd::AbstractVector{<:Number}, T::Real) = 2*tan.(wd/2)/T

@doc raw"""
    contype(type::Type{<:Real}, dtype = Float64)
    contype(type::Type{<:Complex}, dtype = Float64)

Convert type to default concrete type.

Parameters:
 * `type`: a concrete or abstract type

Returns:
 * default concrete type `Float64`
"""
contype(type::Type{<:Real}, dtype = Float64) = isconcretetype(type) ? type : dtype
contype(type::Type{<:Complex}, dtype = Float64) = isconcretetype(type) ? type : Complex{dtype}
