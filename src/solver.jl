using NearestNeighbors
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration: LocalApproximationValueIterationSolver, LocalApproximationValueIterationPolicy


struct CompressedSolver <: POMDPs.Solver
    sampler::Sampler
    compressor::Compressor
    approximator::LocalFunctionApproximator
    updater::Updater
end

struct CompressedSolverPolicy <: Policy
    m::CompressedBeliefMDP
    base_policy::LocalApproximationValueIterationPolicy
end

function POMDPs.action(p::CompressedSolverPolicy, b)
    s = encode(p.m, b)
    return action(p.base_policy, s)
end

function POMDPs.value(p::CompressedSolverPolicy, b)
    s = encode(p.m, b)
    return value(p.base_policy, s)
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
    cbmdp = CompressedBeliefMDP(pomdp, solver.updater, solver.compressor)
    approx_solver = LocalApproximationValueIterationSolver(solver.approximator; verbose=verbose, max_iterations=max_iterations, is_mdp_generative=true, n_generative_samples=10)
    approx_policy = solve(approx_solver, cbmdp)
    return CompressedSolverPolicy(cbmdp, approx_policy)
end

