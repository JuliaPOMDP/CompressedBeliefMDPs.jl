import POMDPs

struct HistoryUpdater <: POMDPs.Updater end

POMDPs.initialize_belief(up::HistoryUpdater, d) = Any[d]

function POMDPs.update(up::HistoryUpdater, b, a, o)
    bp = copy(b)
    push!(bp, a)
    push!(bp, o)
    return bp
end