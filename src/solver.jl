using NearestNeighbors
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration: LocalApproximationValueIterationSolver, LocalApproximationValueIterationPolicy

# TODO: add references to the appropriate local approx libraries in the docstring 
struct CompressedSolver <: POMDPs.Solver
    sampler::Sampler
    compressor::Compressor
    approximator::LocalFunctionApproximator
    updater::Updater
    base_solver::LocalApproximationValueIterationSolver
end

struct CompressedSolverPolicy <: Policy
    m::CompressedBeliefMDP
    base_policy::LocalApproximationValueIterationPolicy
end

# TODO: use macro forwarding
function POMDPs.action(p::CompressedSolverPolicy, b)
    s = encode(p.m, b)
    return action(p.base_policy, s)
end

function POMDPs.value(p::CompressedSolverPolicy, b)
    s = encode(p.m, b)
    return value(p.base_policy, s)
end

POMDPs.updater(policy::CompressedSolverPolicy) = policy.m.bmdp.updater

function CompressedSolver(
    pomdp::POMDP, 
    sampler::BaseSampler, 
    compressor::Compressor; 
    n_samples::Integer=100, 
    k::Integer=1, 
    verbose=false, 
    n_generative_samples=500,
    max_iterations=1000
)
    # sample and compress beliefs
    B = sample(sampler, pomdp; n_samples=n_samples)
    fit!(compressor, B)
    B̃ = compress(compressor, B)

    # make function approximator
    data = [SVector(row...) for row in eachrow(B̃)]
    tree = KDTree(data)
    approximator = LocalNNFunctionApproximator(tree, data, k)

    base_solver = LocalApproximationValueIterationSolver(
        approximator; 
        verbose=verbose, 
        max_iterations=max_iterations, 
        is_mdp_generative=true, 
        n_generative_samples=n_generative_samples
    )

    return CompressedSolver(sampler, compressor, approximator, sampler.updater, base_solver)
end

function POMDPs.solve(solver::CompressedSolver, pomdp::POMDP)   
    cbmdp = CompressedBeliefMDP(pomdp, solver.updater, solver.compressor)
    approx_policy = solve(solver.base_solver, cbmdp)
    return CompressedSolverPolicy(cbmdp, approx_policy)
end

