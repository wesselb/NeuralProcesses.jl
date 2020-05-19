module ConvCNPs

using Distributions
using Flux
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics
using StatsBase

if Flux.has_cuarrays()
    include("gpu.jl")
else
    const CuOrVector = Vector
    const CuOrMatrix = Matrix
    const CuOrArray = Array
end

include("util.jl")
include("data.jl")
include("conv.jl")

include("discretisation.jl")
include("setconv.jl")
include("attention.jl")
include("model/cnp/convcnp.jl")
include("model/cnp/correlatedconvcnp.jl")
include("model/np/np.jl")
include("model/np/convnp.jl")
include("model/np/anp.jl")

include("experiment/experiment.jl")

end
