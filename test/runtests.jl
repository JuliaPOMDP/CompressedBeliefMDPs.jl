using CompressedBeliefMDPs
using Random
using Test

using POMDPs, POMDPModels, POMDPTools
using ParticleFilters
using MCTS

using Distances

# TODO: set global seed?

@testset "CompressedBeliefMDPs.jl" begin
    include("compressor_tests.jl")
    include("solver_tests.jl")
end
