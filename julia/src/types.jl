@doc raw"""
    SS

State-space object with fields:

* `a`: system matrix ``n \times n``
* `b`: input matrix ``n \times m``
* `c`: output matrix ``r \times n``
* `d`: direct feedthrough matrix ``r \times m``
"""
struct SS{T <: AbstractMatrix}
    "System matrix."
    a::T
    "Input matrix."
    b::T
    "Output matrix."
    c::T
    "Direct feedhtrough matrix."
    d::T
    SS{T}() where T = new()
    function SS{T}(a::T, b::T) where T
        size(a, 2) != size(b, 1) || throw(ArgumentError("System and input matrix dimensions mismatch."))
        SS(a, b, zeros(size(a, 1), 1), zeros(1, 1))
    end
    function SS{T}(a::T, b::T, c::T) where T
        size(a, 2) != size(b, 1) || throw(ArgumentError("System and input matrix dimensions mismatch."))
        size(a, 2) != size(c, 2) || throw(ArgumentError("System and output matrix dimensions mismatch."))
        SS(a, b, c, zeros(size(c, 1), size(b, 2)))
    end
    function SS{T}(a::T, b::T, c::T, d::T) where T
        size(a, 2) != size(b, 1) || throw(ArgumentError("System and input matrix dimensions mismatch."))
        size(a, 2) != size(c, 2) || throw(ArgumentError("System and output matrix dimensions mismatch."))
        size(b, 2) != size(d, 2) || throw(ArgumentError("Input and throughput matrix dimensions mismatch."))
        size(c, 1) != size(c, 1) || throw(ArgumentError("Output and throughput matrix dimensions mismatch."))
        new(a, b, c, d)
    end
end
"""
    SS()

Create empty SS object.
"""
SS() = SS{Matrix{Float64}}()
"""
    SS(a::AbstractMatrix, b::AbstractMatrix)

Create SS object without output and direct throughput.
"""
SS(a::T, b::T) where T = SS{Matrix{Float64}}(a, b)
"""
    SS(a::AbstractMatrix, b::AbstractMatrix, c::AbstractMatrix)

Create SS object without direct throughput.
"""
SS(a::T, b::T, c::T) where T = SS{Matrix{Float64}}(a, b, c)
"""
    SS(a::AbstractMatrix, b::AbstractMatrix, c::AbstractMatrix, d::AbstractMatrix)

Create SS object.
"""
SS(a::T, b::T, c::T, d::T) where T = SS{Matrix{Float64}}(a, b, c, d)

@doc raw"""
    FRDC

Continuous-time frequency response data object with fields:

* `ffdata`: matrix of frequency response data. `ffdata[i,:,:]` is the frequency function matrix (sample of the rationl function) corresponding to sample `i`
* `w`: vector with angular frequencies `rad/s`
"""
struct FRDC{T <: AbstractArray, S <: AbstractVector}
    ffdata::T
    w::S
    FRDC{T,S}() where {T,S} = new() 
    FRDC{T,S}(ffdata::T, w::S) where {T,S} = new(ffdata, w) 
end
"""
    FRDC()

Create empty FRDC object.
"""
FRDC() = FRDC{Array{Complex{Float64}, 3}, Vector{Float64}}()
"""
    FRDC(ffdata::AbstractArray, w::AbstractVector)

Create FRDC object.
"""
FRDC(ffdata::T, w::S) where {T,S} = FRDC{T, S}(ffdata, w)

@doc raw"""
    FRDD

Discrete-time frequency response data object with fields:

* `ffdata`: matrix of frequency response data. `ffdata[i,:,:]` is the frequency function matrix (sample of the rationl function) corresponding to sample `i`
* `z`: vector with the samples of the function argument where `z[i]` is argument for index `i`
"""
struct FRDD{T <: AbstractArray, S <: AbstractVector}
    ffdata::T
    z::S
    FRDD{T,S}() where {T,S} = new() 
    FRDD{T,S}(ffdata::T, z::S) where {T,S} = new(ffdata, z) 
end
"""
    FRDD()

Create empty FRDD object.
"""
FRDD() = FRDD{Array{Complex{Float64}, 3}, Vector{Float64}}()
"""
    FRDD(ffdata::AbstractArray, z::AbstractVector)

Create FRDD object.
"""
FRDD(ffdata::T, z::S) where {T,S} = FRDD{T, S}(ffdata, z)