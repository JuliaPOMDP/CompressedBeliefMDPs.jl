### POLICY ###

"""
    CompressedBeliefPolicy

Maps a base policy for the compressed belief-state MDP to a policy for the
true POMDP.

## Fields
- `m::CompressedBeliefMDP`: The compressed belief-state MDP.
- `base_policy::Policy`: The base policy used for decision-making in the compressed belief-state MDP.

## Constructors
    CompressedBeliefPolicy(m::CompressedBeliefMDP, base_policy::Policy)

Constructs a `CompressedBeliefPolicy` using the specified compressed belief-state MDP and base policy.

## Example Usage
```julia
policy = solve(solver, pomdp)
s = initialstate(pomdp)
a = action(policy, s) # returns the approximately optimal action for state s
v = value(policy, s)  # returns the approximately optimal value for state s
```
"""
struct CompressedBeliefPolicy <: POMDPs.Policy
    m::CompressedBeliefMDP
    base_policy::Policy
end

function POMDPs.action(p::CompressedBeliefPolicy, s)
    b = initialize_belief(p.m.bmdp.updater, s)
    b̃ = encode(p.m, b)
    a = action(p.base_policy, b̃)
    return a
end

function POMDPs.value(p::CompressedBeliefPolicy, s)
    b = initialize_belief(p.m.bmdp.updater, s)
    b̃ = encode(p.m, b)
    v = value(p.base_policy, b̃)
    return v
end

function POMDPs.updater(p::CompressedBeliefPolicy)
    up = p.m.bmdp.updater
    return up
end

### SOLVER ###

"""
    CompressedBeliefSolver

The `CompressedBeliefSolver` struct represents a solver for compressed belief-state MDPs. It combines a compressed belief-state MDP with a base solver to approximate the value function.

## Fields
- `m::CompressedBeliefMDP`: The compressed belief-state MDP.
- `base_solver::Solver`: The base solver used to solve the compressed belief-state MDP.

## Constructors
    CompressedBeliefSolver(pomdp::POMDP, base_solver::Solver; updater::Updater=DiscreteUpdater(pomdp), sampler::Sampler=BeliefExpansionSampler(pomdp), compressor::Compressor=PCACompressor(1))
    CompressedBeliefSolver(pomdp::POMDP; updater::Updater=DiscreteUpdater(pomdp), sampler::Sampler=BeliefExpansionSampler(pomdp), compressor::Compressor=PCACompressor(1), interp::Union{Nothing, LocalFunctionApproximator}=nothing, k::Int=1, verbose::Bool=false, max_iterations::Int=1000, n_generative_samples::Int=10, belres::Float64=1e-3)

Constructs a `CompressedBeliefSolver` using the specified POMDP, base solver, updater, sampler, and compressor. Alternatively, you can omit the base solver
in which case a `LocalApproximationValueIterationSolver`(https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl) will be created instead. For example, different base solvers
are needed if the POMDP state and action space are continuous.

## Example Usage

```julia-repl
julia> pomdp = TigerPOMDP();
julia> solver = CompressedBeliefSolver(pomdp; verbose=true, max_iterations=10);
julia> solve(solver, pomdp);
[Iteration 1   ] residual:       8.51 | iteration runtime:    635.870 ms, (     0.636 s total)
[Iteration 2   ] residual:       3.63 | iteration runtime:      0.504 ms, (     0.636 s total)
[Iteration 3   ] residual:       10.1 | iteration runtime:      0.445 ms, (     0.637 s total)
[Iteration 4   ] residual:       15.2 | iteration runtime:      0.494 ms, (     0.637 s total)
[Iteration 5   ] residual:       6.72 | iteration runtime:      0.432 ms, (     0.638 s total)
[Iteration 6   ] residual:       7.38 | iteration runtime:      0.508 ms, (     0.638 s total)
[Iteration 7   ] residual:       6.03 | iteration runtime:      0.495 ms, (     0.639 s total)
[Iteration 8   ] residual:       5.73 | iteration runtime:      0.585 ms, (     0.639 s total)
[Iteration 9   ] residual:       4.02 | iteration runtime:      0.463 ms, (      0.64 s total)
[Iteration 10  ] residual:       7.28 | iteration runtime:      0.576 ms, (      0.64 s total)
```
"""
struct CompressedBeliefSolver <: Solver
    m::CompressedBeliefMDP
    base_solver::Solver
end


"""
    make_cache(B, B̃)

Helper function that creates a cache that maps each unique belief from the set `B` to its corresponding compressed representation in `B̃`.

# Arguments
- `B::Vector{<:Any}`: A vector of beliefs.
- `B̃::Matrix{Float64}`: A matrix where each row corresponds to the compressed representation of the beliefs in `B`.

# Returns
- `Dict{<:Any, Vector{Float64}}`: A dictionary mapping each unique belief in `B` to its corresponding compressed representation in `B̃`.

# Example Usage
```julia
B = [belief1, belief2, belief3]
B̃ = [compressed_belief1; compressed_belief2; compressed_belief3]
ϕ = make_cache(B, B̃)
```
"""
function make_cache(B, B̃)
    ϕ = Dict(unique(t->t[2], zip(B, eachrow(B̃))))
    return ϕ
end

"""
    make_numerical(B, pomdp)

Helper function that converts a set of beliefs `B` into a numerical matrix representation suitable for processing by numerical algorithms/compressors.

# Arguments
- `B::Vector{<:Any}`: A vector of beliefs.
- `pomdp::POMDP`: The POMDP model associated with the beliefs.

# Returns
- `Matrix{Float64}`: A matrix where each row corresponds to a numerical representation of a belief in `B`.

# Example Usage
```julia
B = [belief1, belief2, belief3]
B_numerical = make_numerical(B, pomdp)
```
"""
function make_numerical(B, pomdp)
    B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
    return B_numerical
end


function _make_compressed_belief_MDP(
    pomdp::POMDP, 
    sampler::Sampler, 
    updater::Updater, 
    compressor::Compressor
)
    # sample beliefs
    B = sampler(pomdp)

    # compress beliefs and cache mapping
    B_numerical = make_numerical(B, pomdp)
    fit!(compressor, B_numerical)
    B̃ = compressor(B_numerical)
    ϕ = make_cache(B, B̃)

    # construct the compressed belief-state MDP
    m = CompressedBeliefMDP(pomdp, updater, compressor)
    merge!(m.ϕ, ϕ)  # update the compression cache

    return m, B̃
end

function CompressedBeliefSolver(
    pomdp::POMDP,
    base_solver::Solver;
    updater::Updater=DiscreteUpdater(pomdp),
    sampler::Sampler=BeliefExpansionSampler(pomdp),
    compressor::Compressor=PCACompressor(1)
)
    m, _ = _make_compressed_belief_MDP(pomdp, sampler, updater, compressor)
    return CompressedBeliefSolver(m, base_solver)
end

function CompressedBeliefSolver(
    pomdp::POMDP;
    updater::Updater=DiscreteUpdater(pomdp),
    sampler::Sampler=BeliefExpansionSampler(pomdp),
    compressor::Compressor=PCACompressor(1),

    # base policy arguments
    interp::Union{Nothing, LocalFunctionApproximator}=nothing,
    k=1,  # k nearest neighbors; only used if interp is nothing
    verbose=false,
    max_iterations=1000,  # for value iteration
    n_generative_samples=10,  # number of steps to look ahead when calculated expected reward
    belres::Float64=1e-3
)
    # Type assertions
    @assert k > 0 "k must be greater than 0"
    @assert max_iterations > 0 "max_iterations must be greater than 0"
    @assert n_generative_samples > 0 "n_generative_samples must be greater than 0"
    @assert belres > 0.0 "Belman residual (belres) must be greater than 0.0"

    m, B̃ = _make_compressed_belief_MDP(pomdp, sampler, updater, compressor)

    # define the interpolator for the solver
    if isnothing(interp)
        data = map(row->SVector(row...), eachrow(B̃))
        tree = KDTree(data)
        interp = LocalNNFunctionApproximator(tree, data, k)
    end
    
    # build the base solver
    base_solver = LocalApproximationValueIterationSolver(
        interp,
        max_iterations=max_iterations,
        belres=belres,
        verbose=verbose,
        is_mdp_generative=true,
        n_generative_samples=n_generative_samples
    )

    return CompressedBeliefSolver(m, base_solver)
end

function POMDPs.solve(
    solver::CompressedBeliefSolver, 
    pomdp::POMDP
)
    if solver.m.bmdp.pomdp !== pomdp
        @warn "Got $pomdp, but solver.m.bmdp.pomdp $(solver.m.bmdp.pomdp) isn't identical"
    end

    base_policy = solve(solver.base_solver, solver.m)
    return CompressedBeliefPolicy(solver.m, base_policy)
end