@doc raw"""
    estimate_cd(
        ffdata::AbstractArray{<:Number, 3},
        z::AbstractVector{<:Number},
        a::AbstractMatrix{<:Number},
        b::AbstractMatrix{<:Number};
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true,
    )

Estimates the `c` and `d` matrices given `a and `b` matrices and frequency function data `ffdata`.

Solve the optimization problem

``\min_{c,d} \sum_i \| d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] \|^2_F``

or if `estimd == false`

``\min_{c} \sum_i \| c*inv(z[i]*I-a)*b - ffdata[i,:,:] \|^2_F``

if `type = Real` a real valued solution is calulated. If `type = Complex` the solution is complex valued

Parameters:
* `ffdata`: frequency data packed in a matrix. `ffdata[i,:,:]` is the frequency function matrix corresponding to sample `i`
* `z`: vector with complex scalars
* `a`: square matrix
* `b`: matrix
*Optional:*
* `type`: data type of model, either `Real` or `Complex`
* `estimd`: if set to `false` no d matrix is esimated and a zero d matrix is returned

Returns :
* `c`: LS-optimal `c` matrix
* `d`: LS-optimal `d` matrix
"""
function estimate_cd(
    ffdata::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number};
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true,
)
    type = contype(type)

    n = size(b, 1)
    nw, p, m = size(ffdata)
    fk = ltifr(a, b, z)
    fkstack = reshape(permutedims(fk, (2, 3, 1)), n, :) # regressor for c matrix
    
    if estimd
        dreg = repeat(Matrix{type}(I, m, m), 1, nw) # regressor for d matrix
        R = vcat(fkstack, dreg) # complete regressor
    else
        R = fkstack
    end
    RH = reshape(permutedims(ffdata, (2, 3, 1)), p, :) # right hand side in LS

    if type <: Real
        R = [real(R) imag(R)]
        RH = [real(RH) imag(RH)]
    end
    cd = RH/R
    
    return cd[:, 1:n], estimd ? cd[:, n+1:end] : zeros(type, p, m)
end

@doc raw""" 
    estimate_bd(
        ffdata::AbstractArray{<:Number, 3},
        z::AbstractVector{<:Number},
        a::AbstractMatrix{<:Number},
        c::AbstractMatrix{<:Number};
        type::Union{Type{<:Real}, Type{<:Complex}} = Real,
        estimd::Bool = true,
    )

Estimates the `b` and `d` matrices given `a`, `c` matrices and frequency function data `ffdata`.

Solve the optimization problem

``\min_{b,d} \sum_i \| d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] \|^2_F ``

or if `estimd == false`

``\min_{b} \sum_i \| c*inv(z[i]*I-a)*b - ffdata[i,:,:] \|^2_F``

if `type = Real` a real valued solution is calulated. If `type = Complex` the solution is complex valued

Parameters:
* `ffdata`: frequency data packed in a matrix. `ffdata[i,:,:]` is the frequency function matrix corresponding to sample `i`
* `z`: vector with complex scalars
* `a`: square matrix
* `c`: matrix
*Optional_*
* `type`: data type of model, either `Real` or `Complex`
* `estimd`: if set to false no `d` matrix is esimated and a zero `d` matrix is returned

Returns:
* `b`: LS-optimal `b` matrix
* `d`: LS-optimal `d` matrix
"""
function estimate_bd(
    ffdata::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number};
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true,
)
    fdt = transpose_ffdata(ffdata)
    bt, dt = estimate_cd(fdt, z, transpose(a), transpose(c);type=type, estimd=estimd)
    
    transpose(bt), transpose(dt)
end
 
@doc raw"""
	transpose_ffdata(ffdata::AbstractArray{<:Number, 3})

Transposes ffdata (changes inputs and outputs).
"""
function transpose_ffdata(ffdata::AbstractArray{<:Number, 3})
    # nw, p, m = size(ffdata)
    # fdt = zeros(T, nw, m, p)

    # for idx in 1:nw
    #     fdt[idx, :, :] = transpose(ffdata[idx, :, :])
    # end

    return permutedims(ffdata, (1, 3, 2))
end
