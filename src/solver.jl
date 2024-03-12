using NearestNeighbors
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration: LocalApproximationValueIterationSolver, LocalApproximationValueIterationPolicy

# TODO: add references to the appropriate local approx libraries in the docstring 
"""
    struct CompressedSolver <: POMDPs.Solver

A solver for compressed belief MDPs, implementing the POMDPs.Solver interface.

# Fields
- `sampler::Sampler`: The sampler used for Monte Carlo simulations.
- `compressor::Compressor`: The compressor used to compress beliefs in the MDP.
- `approximator::LocalFunctionApproximator`: The local function approximator for value function estimation.
- `updater::Updater`: The belief updater used in the solver.

## Example
```julia
sampler = MySampler()  # replace with your specific sampler type
compressor = MyCompressor()  # replace with your specific compressor type
approximator = MyFunctionApproximator()  # replace with your specific function approximator type
updater = MyUpdater()  # replace with your specific updater type

solver = CompressedSolver(sampler, compressor, approximator, updater)
```
"""
struct CompressedSolver <: POMDPs.Solver
    sampler::Sampler
    compressor::Compressor
    approximator::LocalFunctionApproximator
    updater::Updater
end


"""
    struct CompressedSolverPolicy <: Policy

A policy for decision-making in a CompressedBeliefMDP using a local approximation value iteration policy.

# Fields
- `m::CompressedBeliefMDP`: The compressed belief MDP associated with the policy.
- `base_policy::LocalApproximationValueIterationPolicy`: The base policy used for decision-making.

## Example
```julia
mdp = CompressedBeliefMDP(...)  # replace with your specific CompressedBeliefMDP instantiation
base_policy = LocalApproximationValueIterationPolicy(...)  # replace with your specific base policy instantiation

policy = CompressedSolverPolicy(mdp, base_policy)
```
"""
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

