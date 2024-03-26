using CompressedBeliefMDPs
using Test

using POMDPs, POMDPModels, POMDPTools
# TODO: also test w/ FA solver
using MCTS

@testset "CompressedBeliefMDPs.jl" begin
    include("mv_stats_tests.jl")
    include("solver_tests.jl")
end
