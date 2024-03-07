using NearestNeighbors
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration: LocalApproximationValueIterationSolver, LocalApproximationValueIterationPolicy


struct CompressedSolver <: POMDPs.Solver
    sampler::Sampler
    compressor::Compressor
    approximator::LocalFunctionApproximator
    updater::POMDPs.Updater
end

struct CompressedSolverPolicy <: Policy
    compressor::Compressor
    base_policy::LocalApproximationValueIterationPolicy
end


function POMDPs.action(p::CompressedSolverPolicy, b)
    b̃ = encode(p.base_policy.mdp.bmdp.pomdp, p.compressor, b)
    return action(p.base_policy, b̃)
end

function POMDPs.value(p::CompressedSolverPolicy, b)
    b̃ = encode(p.base_policy.mdp.bmdp.pomdp, p.compressor, b)
    # return value(p.base_policy, vec(b̃))
    return value(p.base_policy, b̃)
end


function CompressedSolver(pomdp::POMDP, sampler::Sampler, compressor::Compressor; n_samples::Integer=100, k::Integer=1)
    # sample and compress beliefs
    B = sample(sampler, pomdp; n_samples=n_samples)
    fit!(compressor, B)
    B̃ = compress(compressor, B)

    # make function approximator
    data = [SVector(row...) for row in eachrow(B̃)]
    tree = KDTree(data)
    approximator = LocalNNFunctionApproximator(tree, data, k)

    return CompressedSolver(sampler, compressor, approximator, DiscreteUpdater(pomdp))
end


function POMDPs.solve(solver::CompressedSolver, pomdp::POMDP; verbose=false, max_iterations=1000)   
    mdp = CompressedBeliefMDP(pomdp, solver.updater, solver.compressor)
    approx_solver = LocalApproximationValueIterationSolver(solver.approximator; verbose=verbose, max_iterations=max_iterations, is_mdp_generative=true, n_generative_samples=10)
    approx_policy = solve(approx_solver, mdp)
    return CompressedSolverPolicy(solver.compressor, approx_policy)
end

