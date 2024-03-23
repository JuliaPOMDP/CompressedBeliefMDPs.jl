struct CompressedBeliefSolver <: Solver
    explorer::Union{Policy, ExplorationPolicy}
    updater::Updater
    compressor::Compressor
    base_solver::Solver
    n::Integer
end

function CompressedBeliefSolver(
    explorer::Union{Policy, ExplorationPolicy},
    updater::Updater,
    compressor::Compressor,
    base_solver::Solver;
    n=100
)
    return CompressedBeliefSolver(explorer, updater, compressor, base_solver, n)
end

# TODO: make compressed solver that infers everything 
# TODO: make compressed solver that uses local FA solver

struct CompressedBeliefPolicy <: Policy
    m::CompressedBeliefMDP
    base_policy::Policy
end

POMDPs.action(p::CompressedBeliefPolicy, b) = action(p.base_policy, encode(m, b))
POMDPs.value(p::CompressedBeliefPolicy, b) = value(p.base_policy, encode(m, b))
POMDPs.updater(p::CompressedBeliefPolicy) = p.m.bmdp.updater

function POMDPs.solve(solver::CompressedBeliefSolver, pomdp::POMDP)
    B = sample(pomdp, solver.explorer, solver.updater, solver.n)
    B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)'
    fit!(solver.compressor, B_numerical)
    B̃ = compress(solver.compressor, B_numerical)
    m = CompressedBeliefMDP(pomdp, solver.updater, solver.compressor)
    ϕ = Dict(unique(t->t[2], zip(B, eachrow(B̃))))
    merge!(m.ϕ, ϕ)  # update compression cache
    base_policy = solve(solver.base_solver, m)
    return CompressedBeliefPolicy(m, base_policy)
end