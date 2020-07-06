"""
    estimate_cd(ffdata, z, a, b; type = Real, estimd = true)

Estimates the c and d matrices given a, b matrices and frequency function data ffdata.

min_{c,d} sum_i || d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

if estimd == false 
min_{c} sum_i || c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

if type='float' a real valued solution is calulated. if type='complex' 
the solution is complex valued

Parameters
----------
ffdata:     frequency data packed in a matrix. fftata[i,:,:] is the frequency
            function matrix corresponding to sample i
z:          vector with the frequecy points to evaluate, i.e. for DT z = exp(j*w)
            where w is frequencies in radians per sample
a:          a matrix
b:          b matrix

Optional
type:       data type of model either 'Real' or 'Complex'
estimd:     if set to false no d matrix is esimated and a zero d matrix is returned

Returns 
-------
c:          LS-optimal c matrix
d:          LS-optimal d matrix
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

""" 
    estimate_bd(ffdata, z, a, c; type = Real, estimd = true)
    
Estimates the b and d matrices given a, c matrices and frequency function data 'ffdata'.

Calulates the b and d matrices for a linear dynamic system given the a and b 
matrices and samples of frequency function data. It solves 

min_{b,d} sum_i || d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

if estimd == false 
min_{b} sum_i || c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

if type='float' a real valued solution is calulated. if type='complex' 
the solution is complex valued

Parameters
----------
ffdata:     frequency data packed in a matrix. fftata[i,:,:] is the frequency
            function matrix corresponding to sample i
z:          vector with the frequecy points to evaluate, i.e. for DT z = exp(j*w)
            where w is frequencies in radians per sample
a:          a matrix
c:          c matrix

Optional
type:       data type of model either 'float' or 'complex'
estimd:     if set to false no d matrix is esimated and a zero d matrix is returned

Returns 
-------
b:          LS-optimal b matrix
d:          LS-optimal d matrix
"""
function estimate_bd(
    ffdata::AbstractArray{<:Number, 3},
    z::AbstractVector{<:Number},
    a::AbstractMatrix{<:Number},
    c::AbstractMatrix{<:Number};
    type::Union{Type{<:Real}, Type{<:Complex}} = Real,
    estimd::Bool = true
)
    fdt = transpose_ffdata(ffdata)
    bt, dt = estimate_cd(fdt, z, transpose(a), transpose(c); type, estimd)
    
    transpose(bt), transpose(dt)
end

"""
	transpose_ffdata(ffdata)

Transposes ffdata (changes inputs and outputs)
"""
function transpose_ffdata(ffdata::AbstractArray{<:Number, 3})
    # nw, p, m = size(ffdata)
    # fdt = zeros(T, nw, m, p)

    # for idx in 1:nw
    #     fdt[idx, :, :] = transpose(ffdata[idx, :, :])
    # end

    return permutedims(ffdata, (1, 3, 2))
end