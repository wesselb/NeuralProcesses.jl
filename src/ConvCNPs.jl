module ConvCNPs

using Distributions
using Flux
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics

include("util.jl")
include("discretisation.jl")
include("setconv.jl")
include("conv.jl")
include("model.jl")
include("data.jl")

if Flux.has_cuarrays()
    include("gpu.jl")
end

include("experiment.jl")

end
