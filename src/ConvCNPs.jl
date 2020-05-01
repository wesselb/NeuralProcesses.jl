module ConvCNPs

using Distributions
using Flux
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics

if Flux.has_cuarrays()
    include("gpu.jl")
else
    const CuOrVector = Vector
    const CuOrMatrix = Matrix
    const CuOrArray = Array
end

include("util.jl")
include("discretisation.jl")
include("setconv.jl")
include("conv.jl")
include("model.jl")
include("data.jl")

include("experiment/experiment.jl")

end
