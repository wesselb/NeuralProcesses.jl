module ConvCNPs

using Distributions
using Flux
using NNlib
using Printf
using Random
using Statistics
using Stheno

include("util.jl")
include("discretisation.jl")
include("setconv.jl")
include("conv.jl")
include("model.jl")
include("data.jl")

end
