@doc raw"""
    function ffsid(
        w::AbstractVector{<:Number},
        ffdata::AbstractArray{<:Number, 3},
        n::Integer,
        q::Integer;
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true,
        CT::Bool = false,
        T::Real = 1
    )

Estimate a state-space model `(a, b, c, d)` from frequency function data.

Parameters:
* `w`: vector of frequencies in rad/sample [-pi,pi] or [0, 2pi]. if `CT = true` unit in radians/s (-inf, +inf)
* `ffdata`: matrix of frequency function data. `ffdata[i,:,:]` is the frequency response matrix at frequency `w[i]`
* `n`: the model order of the ss-model `(a, b, c, d)`, i.e. `a` is a size (`n` x `n`) matrix
* `q`: the numer of block rows used in the intermediate matrix. Must satisfy `q > n`
*Optional:*
* `type`:
    * if `type = Real` a real valued solution `(a, b, c, d)` is returned
    * if `type = Complex` a complex valued solution `(a,b,c,d)` is returned
* `estimd`: if set to False no `d` matrix is esimated and a zero `d` matrix is returned
* `CT`:
    * if set to `true` a continuous time (CT) model is esimated
    * if set to `false` a discrete time (DT) model is esimated (default)
* `T`: a frequency scaling factor for the bilinear transformation used when `CT = true`. Default is 1. If `CT = false` parameter `T` is disregarded

Returns:
* `a`: the estimated `a` matrix
* `b`: the estimated `b` matrix
* `c`: the estimated `c` matrix
* `d`: the estimated `d` matrix (or zero matrix if `estimd = false`)
* `s`: a vector of the singular values   
"""
function ffsid(
    w::AbstractVector{<:Number},
    ffdata::AbstractArray{<:Number, 3},
    n::Integer,
    q::Integer;
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true,
    CT::Bool = false,
    T::Real = 1
)
    if CT
        ad, bd, cd, dd, s  = ffsid(cf2df(w, T), ffdata, n, q;
            type = type, estimd = true, CT = false, T = T)
        a, be, ce, de = bilinear_d2c((ad, bd, cd, dd), T)
        if !estimd
            be, de = estimate_bd(ffdata, im*w, a, ce; type = type, estimd = estimd)
            ce, de = estimate_cd(ffdata, im*w, a, be; type = type, estimd = estimd)
            be, de = estimate_bd(ffdata, im*w, a, ce; type = type, estimd = estimd)
            ce, de = estimate_cd(ffdata, im*w, a, be; type = type, estimd = estimd)
        end       
        return a, be, ce, de, s
    end

    return gffsid(exp.(im*w), ffdata, n, q; type = type, estimd = estimd)
end

@doc raw"""
    function gffsid(
        z::AbstractVector{<:Number},
        ffdata::AbstractArray{<:Number, 3},
        n::Integer,
        q::Integer;
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true
    )

Estimate a state-space model `(a, b, c, d)` from general frequency function data by mininizing the Frobenius norm 

``\sum_i \| ffdata[i,:,:] - d - c*inv(z[i]*eye(n)-a)*b \|^2_F``

Parameters:
* `z`: vector of complex numbers
* `ffdata`: matrix of frequency function data. `ffdata[i,:,:]` is the frequency response matrix at complex point `z[i]`
* `n`: the model order of the ss-model `(a, b, c, d)`, i.e. a is a size (`n` x `n`) matrix
* `q`: the numer of block rows used in the intermediate matrix. Must satisfy `q > n`
*Optional:*
* `type`:
    * if `type = Real` a real valued solution `(a, b, c, d)` is returned
    * if `type = Complex` a complex valued solution is returned
* `estimd`: if set to `false` no `d` matrix is esimated and a zero `d` matrix is returned

Returns:
* `a`: the estimated `a` matrix
* `b`: the estimated `b` matrix
* `c`: the estimated `c` matrix
* `d`: the estimated `d` matrix (or zero matrix if `estimd = false`)
* `s`: a vector of the singular values   
"""
function gffsid(
    z::AbstractVector{<:Number},
    ffdata::AbstractArray{<:Number, 3},
    n::Integer,
    q::Integer;
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true
)
    type = contype(type)

    nwfd, p, m = size(ffdata)
    nw = size(z, 1)

    if q < n+1
        println("Error: q must satidfy q > n.")
        return false
    end
    if nw != nwfd
        println("Error: w and ffdata sizes does not match!")
        return false
    end

    y = zeros(ComplexF64, p*q, nw*m) # G
    if estimd
        u = zeros(ComplexF64, m*q, nw*m) # W
    else
        u = zeros(ComplexF64, m*(q-1), nw*m) # W
    end    
    eyem = Matrix{Float64}(I, m, m)

    for widx in 1:nw
        y[1:p, (widx-1)*m + 1:widx*m] = ffdata[widx, :, :]
        u[1:m, (widx-1)*m + 1:widx*m] = eyem
        zx = z[widx]

        for qidx in 2:q
            y[(qidx-1)*p + 1:qidx*p, (widx-1)*m + 1:widx*m] = zx*ffdata[widx, :, :]
            if estimd || qidx<q
                u[(qidx-1)*m + 1:qidx*m, (widx-1)*m + 1:widx*m] = zx*eyem
            end
            zx *= z[widx]
        end
    end

    if type <: Real
        hU = [real(u) imag(u)]
        hY = [real(y) imag(y)]
    else
        hU = u
        hY = y
    end

    h = [hU; hY]
    r = transpose(qr(transpose(h)).R)
    if estimd
        r22 = r[m*q+1:end, m*q+1:end]
    else
        r22 = r[m*(q-1)+1:end, m*(q-1)+1:end]
    end
    u, s, v = svd(r22)
    c = u[1:p, 1:n]
    lh = u[1:p*(q-1), 1:n]
    rh = u[p+1:p*q, 1:n]
    a = lh\rh
    be, de = estimate_bd(ffdata, z, a, c; type=type, estimd=estimd)
    ce, de = estimate_cd(ffdata, z, a, be; type=type, estimd=estimd)
    be, de = estimate_bd(ffdata, z, a, ce; type=type, estimd=estimd)
    ce, de = estimate_cd(ffdata, z, a, be; type=type, estimd=estimd)
    return a, be, ce, de, s
end

@doc raw"""    
    function fdsid(
        fddata::Tuple,
        n::Integer,
        q::Integer;
        estTrans::Bool = true,
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true,
        CT::Bool = false,
        T::Real = 1
    )

Estimate a DT or CT state-space model from I/O frequency data.

Determines the `(a,b,c,d,xt)` parametrers such that (DT case)

``sum_i   || y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-a)*[b,  xt]* [u[i, :]; z[i]]||^_F``

is small where `z[i] = exp(im*w[i])` and CT Case

``sum_i   ||  y[i,:] - d*u[i, :] + c*inv(im*w[i]*eye(n)-a)*b* u[i, :] ||^2_F``

Parameters:
* `fddata`: a tuple with elements 
    * `fddata[0] = w`: a vector of frequencies in radians/sample (rad/s in CT case)
    * `fddata[1] = y`: a matrix of the output transform data where `y[i,:]` is the DFT of the time domain output signal corresponding to frequency `w[i]`
    * `fddata[2] = u`: a matrix of the input transform data where `u[i,:]` is the DFT of the time domain input signal corresponding to frequency `w[i]`
* `n`: the model order of the ss-model
* `q`: the numer of block rows used in the intermediate matrix. Must satisfy `q > n`
*Optional:*
* `estTrans`: if `true`, a compensation for the transient term will be estimated (default)
* `type`:
    * if `type = Real` a real valued solution `(a, b, c, d)` is returned.
    * if `type = Complex` a complex valued solution is returned.
* `estimd`: if set to `false` no `d` matrix is esimated and a zero `d` matrix is returned
* `CT`: if `true` a CT model is estimated and `estTrans` is forced `false`
* `T`: a frequency scaling factor for the bilinear transformation used when `CT = true`.  Default is 1. If `CT = false` parameter `T` is disregarded

Returns:
* `a`: the estimated `a` matrix
* `b`: the estimated `b` matrix
* `c`: the estimated `c` matrix
* `d`: the estimated `d` matrix (or zero matrix if `estimd = false`)
* `xt`: vector of the transient compensation
* `s`: a vector of the singular values   
"""
function fdsid(
    fddata::Tuple,
    n::Integer,
    q::Integer;
    estTrans::Bool = true,
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true,
    CT::Bool = false,
    T::Real = 1
)
    type = contype(type)

    w = fddata[1]
    yd = fddata[2]
    ud = fddata[3]

    if CT
        estTrans=False
        ad, bd, cd, dd, xt, s  = fdsid((cf2df(w,T), yd, ud), n, q;
            estTrans=False, type=type, estimd = true, CT=False, T=T)
        a, b, c, d = bilinear_d2c((ad, bd, cd, dd), T)
        if !estimd
            b, d, resid = fdestim_bd(w*im, yd, ud, a, c;
                estTrans = estTrans, type = type, estimd = estimd)
            c, d, resid = fdestim_cd(w*im, yd, ud, a, b, 0;
                type = type, estimd = estimd)
            b, d, resid = fdestim_bd(w*im, yd, ud, a, c;
                estTrans = estTrans, type = type, estimd = estimd)
            c, d, resid = fdestim_cd(w*im, yd, ud, a, b, 0;
                type = type, estimd = estimd)
            xt = zeros(type, n, 1)
        end
        return a, b, c, d, xt, s
    end

    return gfdsid((exp.(im*w), yd, ud), n, q; estTrans = estTrans, type = type, estimd = estimd)
end


@doc raw"""
    function gfdsid(
        fddata::Tuple,
        n::Integer,
        q::Integer;
        estTrans::Bool = true,
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true
    )

Estimate a state-space model from I/O frequency data

Determines the `(a, b, c, d, xt)` parameters such that

``\sum_i \| y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-a)*[b,  xt]* [u[i, :]; z[i]] \|^2_F ``

is minimized.

If `estrTrans = false` the following problem is solved

``\sum_i \| y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-a)*b * u[i, :] \|^2_F ``

is minimized.

Parameters:
* `fddata`: a tuple with elements 
    * `fddata[0] = z`: a vector of complex scalars,
    * `fddata[1] = y`: a matrix of the output frequency data where `y[i,:]` corresponds to `z[i]`
    * `fddata[2] = u`: a matrix of the input frrequency data where `u[i,:]` corresponding to `z[i]`
* `n`: the model order of the ss-model
* `q`: the numer of block rows used in the intermediate matrix. Must satisfy `q > n`
*Optional:*
* `estTrans`: if `true`, a compensation for the transient term will be estimated (default)
* `type`:
    * if `type = Real` a real valued solution `(a, b, c, d)` is returned.
    * if `type = Complex` a complex valued solution is returned.
* `estimd`: if set to 'false' no `d` matrix is esimated and a zero `d` matrix is returned

Returns:
* `a`: the estimated `a` matrix
* `b`: the estimated `b` matrix
* `c`: the estimated `c` matrix
* `d`: the estimated `d` matrix (or zero matrix if `estimd = false`)
* `x`: vector of the transient compensation
* `s`: a vector of the singular values   
"""
function gfdsid(
    fddata::Tuple,
    n::Integer,
    q::Integer;
    estTrans::Bool = true,
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true
)
    type = contype(type)

    z = fddata[1]
    yd = fddata[2]
    ud = fddata[3]

    nwy, p = size(yd)
    nwu, m = size(ud)

    if estTrans
        ude = [ud z]
#        ude = [ud ones(nwu,1)]
        me = m + 1
    else
        ude = ud
        me = m
    end
    nw = size(z, 1)

    if nw != nwy
        println("Error: z and Y sizes does not match!")
        return false
    end
    if nw != nwu
        println("Error: z and U sizes does not match!")
        return false
    end 
       
    y = Matrix{Complex}(undef, p*q, nw)
   if estimd
       u = Matrix{Complex}(undef, me*q, nw)
   else
       u = Matrix{Complex}(undef, me*(q-1), nw)
   end
    for widx in 1:nw
        y[1:p, widx] = yd[widx, :]
        u[1:me, widx] = ude[widx, :]
        zx = z[widx]

        for qidx in 2:q
            y[(qidx-1)*p + 1:qidx*p, widx] = zx*yd[widx, :]
            if estimd || qidx<q
                u[(qidx-1)*me + 1:qidx*me, widx] = zx*ude[widx, :]
            end
            zx *= z[widx]
        end
    end
    if type <: Real
        hu = [real(u) imag(u)]
        hy = [real(y) imag(y)]
    else
        hu = u
        hy = y
    end

    h = [hu; hy]
    r = transpose(qr(transpose(h)).R)
    if estimd
        r22 = r[me*q + 1:end, me*q + 1:end]
    else
        r22 = r[me*(q-1) + 1:end, me*(q-1) + 1:end]
    end
    u, s, v = svd(r22)
    c = u[1:p, 1:n]
    lh = u[1:p*(q-1), 1:n]
    rh = u[p+1:p*q, 1:n]

    a = lh\rh

    if estTrans
        b, d, xt, resid = fdestim_bd(z, yd, ud, a, c; estTrans=estTrans, type=type, estimd=estimd)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, xt;  type=type, estimd=estimd)
        b, d, xt, resid = fdestim_bd(z, yd, ud, a, c; estTrans=estTrans, type=type, estimd=estimd)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, xt;  type=type, estimd=estimd)
    else
        b, d, resid = fdestim_bd(z, yd, ud, a, c; estTrans=estTrans, type=type, estimd=estimd)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, 0; type=type, estimd=estimd)
        b, d, resid = fdestim_bd(z, yd, ud, a, c; estTrans=estTrans, type=type, estimd=estimd)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, 0; type=type, estimd=estimd)
        xt = zeros(type, n, 1)
    end

    return a, b, c, d, xt, s
end
