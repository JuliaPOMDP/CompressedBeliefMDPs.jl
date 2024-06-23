using CompressedBeliefMDPs
using Random
using Test

using POMDPs, POMDPModels, POMDPTools
using ParticleFilters
using MCTS

using Distances

Random.seed!(1)

@testset "CompressedBeliefMDPs.jl" begin
    include("compressor_tests.jl")
    include("solver_tests.jl")
    include("circular_tests.jl")
end
