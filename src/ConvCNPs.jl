module ConvCNPs

using Distributions
using Flux
using GPUArrays
using NNlib
using Printf
using Random
using Statistics
using Stheno

GPUArrays.allowscalar(false)

include("util.jl")
include("discretisation.jl")
include("setconv.jl")
include("conv.jl")
include("model.jl")
include("data.jl")

end
