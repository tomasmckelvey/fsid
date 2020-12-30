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
            print("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
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
        xt::AbstractMatrix{<:Number} = Matrix{Float64}(undef, 0, 0)
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
    xt::AbstractMatrix{<:Number} = Matrix{Float64}(undef, 0, 0)
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
        print("fdsim: Incorrect number of matrices in sys.")
        return false
    end

    y = Matrix{ComplexF64}(nwu, p)
    
    if size(xt) > 0
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
