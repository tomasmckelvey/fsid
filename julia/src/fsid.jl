module fsid

using LinearAlgebra, FFTW, SpecialMatrices, Documenter

include("types.jl")
include("math.jl")
include("estimate_c.jl")
include("estimate_d.jl")
include("misc.jl")
include("n4sid.jl")

end
