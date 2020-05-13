using ConvCNPs
using ConvCNPs.Experiment
using Distributions
using FiniteDifferences
using Flux
using Flux.Tracker
using LinearAlgebra
using NNlib
using StatsFuns
using Stheno
using Test

@testset "ConvCNPs.jl" begin
    include("util.jl")
    include("discretisation.jl")
    include("setconv.jl")
    include("attention.jl")
    include("data.jl")
    include("model.jl")
end
