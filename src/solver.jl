### POLICY ###

struct CompressedBeliefPolicy <: POMDPs.Policy
    m::CompressedBeliefMDP
    base_policy::Policy
end

function POMDPs.action(p::CompressedBeliefPolicy, s)
    b = initialize_belief(p.m.bmdp.updater, s)
    action(p.base_policy, encode(p.m, b))
end

function POMDPs.value(p::CompressedBeliefPolicy, s)
    b = initialize_belief(p.m.bmdp.updater, s)
    value(p.base_policy, encode(p.m, b))
end

POMDPs.updater(p::CompressedBeliefPolicy) = p.m.bmdp.updater

### SOLVER ###

struct CompressedBeliefSolver <: Solver
    m::CompressedBeliefMDP
    base_solver::Solver
end

function _make_compressed_belief_MDP(
    pomdp::POMDP, explorer::Union{ExplorationPolicy, Policy}, 
    updater::Updater,
    compressor::Compressor,
    n::Integer,
    expansion::Bool,
    fit_kwargs::Union{Nothing, Dict}=nothing,
    metric::NearestNeighbors.MinkowskiMetric=Euclidean()
)
    # sample beliefs
    if expansion
        B = exploratory_belief_expansion(pomdp, updater; n=n, metric=metric)
    else
        B = sample(pomdp, explorer, updater, n)
    end

    # compress beliefs and cache mapping
    B_numerical = Matrix(mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)')
    isnothing(fit_kwargs) ? fit!(compressor, B_numerical) : fit!(compressor, B_numerical; fit_kwargs...)
    B̃ = compressor(B_numerical)
    ϕ = Dict(unique(t->t[2], zip(B, eachrow(B̃))))

    # construct the compressed belief-state MDP
    m = CompressedBeliefMDP(pomdp, updater, compressor)
    merge!(m.ϕ, ϕ)  # update the compression cache

    return m, B̃
end

# TODO: add seeding
function CompressedBeliefSolver(
    pomdp::POMDP;

    # sampling arguments
    explorer::Union{Policy, ExplorationPolicy}=RandomPolicy(pomdp),  # explorer policy; only used if expansion is false
    updater::Updater=DiscreteUpdater(pomdp),  # only works for discrete S
    compressor::Compressor=PCACompressor(1),
    expansion=true,  # only works for discrete S, A, O
    n::Integer=5,  # if expansion, then n is the number of times we expand; otherwise, n is max number of belief samples
    metric::NearestNeighbors.MinkowskiMetric=Euclidean(),
    fit_kwargs::Union{Nothing, Dict}=nothing,

    # base policy arguments
    interp::Union{Nothing, LocalFunctionApproximator}=nothing,
    k=1,  # k nearest neighbors; only used if interp is nothing
    verbose=false,
    max_iterations=1000,  # for value iteration
    n_generative_samples=10,  # number of steps to look ahead when calculated expected reward
    belres::Float64=1e-3
)
    m, B̃ = _make_compressed_belief_MDP(pomdp, explorer, updater, compressor, n, expansion, fit_kwargs, metric)

    # define the interpolator for the solver
    if isnothing(interp)
        data = map(row->SVector(row...), eachrow(B̃))
        tree = KDTree(data)
        interp = LocalNNFunctionApproximator(tree, data, k)  # TODO: check that we need this
    end
    
    # build the based solver
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

function CompressedBeliefSolver(
    pomdp::POMDP,
    base_solver::Solver;

    # sampling arguments
    explorer::Union{Policy, ExplorationPolicy}=RandomPolicy(pomdp),  # explorer policy; only used if expansion is false
    updater::Updater=DiscreteUpdater(pomdp),  # only works for discrete S
    compressor::Compressor=PCACompressor(1),
    expansion=true,  # only works for discrete S, A, O
    n::Integer=5,  # if expansion, then n is the number of times we expand; otherwise, n is max number of belief samples
    metric::NearestNeighbors.MinkowskiMetric=Euclidean(),
    fit_kwargs::Union{Nothing, Dict}=nothing
)
    m, _ = _make_compressed_belief_MDP(pomdp, explorer, updater, compressor, n, expansion, fit_kwargs, metric)
    return CompressedBeliefSolver(m, base_solver)
end


function POMDPs.solve(solver::CompressedBeliefSolver, pomdp::POMDP)
    if solver.m.bmdp.pomdp !== pomdp
        @warn "Got $pomdp, but solver.m.bmdp.pomdp $(solver.m.bmdp.pomdp) isn't identical"
    end

    base_policy = solve(solver.base_solver, solver.m)
    return CompressedBeliefPolicy(solver.m, base_policy)
end