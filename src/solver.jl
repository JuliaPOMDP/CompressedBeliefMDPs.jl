using POMDPs

using POMDPModelTools: GenerativeBeliefMDP
using LocalFunctionApproximation
using LocalApproximationValueIteration



struct CompressedSolver <: POMDPs.Solver
    sampler::Sampler
    compressor::Compressor
    approximator::LocalFunctionApproximator
    updater::POMDPs.Updater
end

# TODO: perhaps the solve method isn't even needed!!
function POMDPs.solve(solver::CompressedSolver, pomdp::POMDP)
    # sample beliefs to fit the compressor and approximator
    B = sample(solver.sampler, pomdp)
    fit!(solver.compressor, B)
    
    # make a local function approximator
    points = [S]

    # make compressed belief-state MDP
    bmdp = GenerativeBeliefMDP(pomdp, solver.up)
    compressed_bmdp = CompressedBeliefPOMDPs(bmdp, solver.compressor)

    # solve the belief MDP with fitted value approximation
    approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=true)
    approx_policy = solve(approx_solver, compressed_bmdp)
    # TODO: perhaps remap some stuff
    return return approx_policy
end