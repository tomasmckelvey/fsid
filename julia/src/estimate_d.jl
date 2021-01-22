@doc raw"""
    fdestim_cd(
        z::AbstractVector{<:Number},
        yd::AbstractArray{<:Number},
        ud::AbstractArray{<:Number},
        a::AbstractArray{<:Number},
        b::AbstractArray{<:Number},
        xt::Union{AbstractVector{<:Number}, Number} = 0;
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true
    )

Estimate `c` and `d` matrices given `z`, `yd`, `ud` and `a`, `c`, and (optionally `xt`) matrices.
    
Calulates the `c` and `d` matrices for a linear dynamic system on state-space form given the `a` and `b` matrices and samples of rational function data. It solves
if `estimd = true`

  ``\min_{c,d} \sum_i \| ([d 0] + c*inv(z[i]*I-a)*[b xt])[ud[i,:]; z[i]]  - ffdata[i,:,:] \|^2_F``

if `estimd = false`

  ``\min_{c} \sum_i \| (c*inv(z[i]*I-a)*[b xt])[ud[i,:]; w[i]]  - ffdata[i,:,:] \|^2_F``

if `type = Real` a real valued solution is calulated. if `type = Complex` the solution is complex valued

Parameters:
* `ffdata`: data packed in a matrix. `ffdata[i,:,:]` is the frequency function matrix (sample of the rationl function) corresponding to sample `i`
* `w`: vector with the rational function arguments. `w[i]` is the function arguemnt for sample `i`
* `a`: matrix
* `b`: matrix
*Optional:*
* `xt`: vector
* `type`: data type of model either `Real` or `Complex`
* `estimd`: if set to false no d matrix is esimated and a zero d matrix is returned
  
Returns :
* `c`: LS-optimal c matrix
* `d`: LS-optimal d matrix
"""
function fdestim_cd(
    z::AbstractVector{<:Number},
    yd::AbstractArray{<:Number},
    ud::AbstractArray{<:Number},
    a::AbstractArray{<:Number},
    b::AbstractArray{<:Number},
    xt::Union{AbstractVector{<:Number}, Number} = 0;
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true
)
    type = contype(type)

    p = size(yd, 2)
    m = size(ud, 2)
    n = size(a, 1)
    nw = size(z, 1)
    if xt != 0
        ude = [ud z]
        be = [b xt]
    else
        ude = ud
        be = b
    end
    
    fkern = ltifd(a, be, ude, z)
    
    if estimd
        r = zeros(ComplexF64, n + m, nw)
        r[1:n, :] = fkern
        r[n+1:end, :] = transpose(ud)
    else
        r = fkern
    end

    if type <: Real
        rh = [real(r) imag(r)]
        lh = [real(yd); imag(yd)]
    else
        lh = yd
        rh = r
    end

    vecCD = transpose(transpose(rh)\lh)
    resCD = transpose(rh)*transpose(vecCD) - lh

    if estimd
        return vecCD[:, 1:n], vecCD[:, n+1:end], resCD # Return c and d and residuals
    else
        return vecCD, zeros(type, p, m), resCD # Return c and d and residuals
    end
end

@doc raw"""
    fdestim_bd(
        z::AbstractVector{<:Number},
        yd::AbstractMatrix{<:Number},
        ud::AbstractMatrix{<:Number},
        a::AbstractMatrix{<:Number},
        c::AbstractMatrix{<:Number};
        estTrans::Bool = false,
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true
    )

Estimate `b` and `d` matrices (and optionally `xt`) given `yd`, `ud`, `a` and `c` matrices.
    
Calulates the `b` and `d` matrices (and optimally `xt`) for a linear dynamic system in state-space form given the `a` and `c` matrices and samples of frequency domain function data. It solves 
if `estimd = true` and `estTrans = true`

``\min_{b,d,xt} \sum_i \| ([d 0] + c*inv(z[i]*I-a)*[b xt])[ud[i,:]; z[i]]  - ffdata[i,:,:] \|^2_F``

if `estimd = false` and `estTrans = true`

``\min_{b,xt} \sum_i \| (c*inv(z[i]*I-a)*[b xt])[ud[i,:]; w[i]]  - ffdata[i,:,:] \|^2_F``

if `estimd = true` and `estTrans = false`

``\min_{b,d} \sum_i  \| (d+ c*inv(z[i]*I-a)*b)ud[i,:]  - ffdata[i,:,:] \|^2_F``

if `estimd = false` and `estTrans = false`

``\min_{b} \sum_i \| (c*inv(z[i]*I-a)*b)ud[i,:]  - ffdata[i,:,:] \|^2_F``

if `type = Real` a real valued solution is calulated. if `type = Complex` 
the solution is complex valued
    
Parameters:
* `z`: vector with the samples of the function argument where `z[i]` is argument for index `i`
* `yd`: output frequency data `yd[i,:]`
* `ud`: input frequency data `ud[widx,:]`
* `a`: system matrix
* `c`: output matrix
*Optional:*
* `estTrans`: if set to true also an xt vector will be estimated capturing the transient effect
* `type`: data type of model, either Real or Complex
* `estimd`: if set to false no `d` matrix is esimated and a zero `d` matrix is returned

Returns:
* `b`: the LS-optimal `b` matrix
* `d`: LS-optimal `d` matrix  zeros matrix if `estimd = false`
* `xt`: LS-optimal `xt` vector if `estTrans = true`
"""
function fdestim_bd(
    z::AbstractVector{<:Number},
    yd::AbstractMatrix{<:Number},
    ud::AbstractMatrix{<:Number},
    a::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number};
    estTrans::Bool = false,
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true
)
    type = contype(type)

    p = size(yd, 2)
    m = size(ud, 2)
    n = size(a, 1)
    nw = size(z, 1)

    if estTrans
        ude = [ud z]
        me = m + 1
    else
        ude = ud
        me = m
    end

    fkern = permutedims(ltifr(copy(transpose(a)), copy(transpose(c)), z), (3, 2, 1))
    eyep = Matrix{Float64}(I, p, p)

    if estimd      
        r = zeros(ComplexF64, nw*p, me*n+m*p) # room for B xt and D
        for widx in 1:nw
            for midx in 1:me
                r[p*(widx-1) + 1:p*widx, n*(midx-1) + 1:n*midx] = ude[widx, midx] * fkern[:, :, widx]
            end
            for midx in 1:m
                r[p*(widx-1) + 1:p*widx, me*n + p*(midx-1) + 1:me*n+p*midx] = ud[widx, midx] * eyep
            end
        end
    else
        r = zeros(ComplexF64, nw*p, me*n) # room for B xt   
        for widx in 1:nw
            for midx in 1:me
                r[p*(widx-1) + 1:p*widx, (n*(midx-1)) + 1:n*midx] = ude[widx, midx] * fkern[:, :, widx] 
            end
        end
    end

    if type <: Real
        rh = [real(r); imag(r)]
        lh = [real(vcat(transpose(yd)...)); imag(vcat(transpose(yd)...))]
    else
        rh = r
        lh = transpose(hcat(transpose(yd)...))
    end

    vecBD = rh\lh
    resBD = rh*vecBD - lh
    vecB = vecBD[1:n*me]

    if estimd
        vecD = vecBD[n*me + 1:end]
    else
        vecD = zeros(m*p, m*p)
    end

    b = Matrix{type}(undef, n, m)
    d = Matrix{type}(undef, p, m)
    for midx in 1:m
        b[:, midx] = vecB[n*(midx-1) + 1:n*midx]
        d[:, midx] = vecD[p*(midx-1) + 1:p*midx]
    end

    if estTrans
        xt = vecB[n*m+1:n*(m+1)]
        return b, d, xt, resBD
    else
        return b, d, resBD
    end
end
