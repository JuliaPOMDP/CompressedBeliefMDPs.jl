using Distances


struct BeliefExpansionSampler <: Sampler
    updater::Updater
    metric::NearestNeighbors.MinkowskiMetric
    n::Integer
end

function BeliefExpansionSampler(
    pomdp::POMDP; 
    updater::Updater=DiscreteUpdater(pomdp), 
    metric::NearestNeighbors.MinkowskiMetric=Euclidean(),
    n::Integer=3
)
    return BeliefExpansionSampler(updater, metric, n)
end

function _make_numeric(b, pomdp::POMDP)
    b = convert_s(AbstractArray{Float64}, b, pomdp)
    return SVector{length(b)}(b)
end

"""
Adapted from PointBasedValueIteration.jl: https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/blob/master/src/pbvi.jl
"""
function _successors(pomdp::POMDP, b, updater::Updater)
    succs = []
    for a in actions(pomdp, b), o in observations(pomdp)            
        s = update(updater, b, a, o)
        push!(succs, s)
    end
    return unique!(succs)
end

"""
Effecient adaptation of algorithm 21.13 from AFDM that uses KDTree.
Only works for finite S, A, O.
"""
function _exploratory_belief_expansion!(
    pomdp::POMDP, 
    B::Set, 
    B_numeric,
    s::BeliefExpansionSampler
)
    tree = KDTree(B_numeric, s.metric)
    B_new = typeof(B)()
    for b in B
        succs = _successors(pomdp, b, s.updater)
        succs_numeric = map(s->_make_numeric(s, pomdp), succs)
        if !isempty(succs)
            _, dists = nn(tree, succs_numeric)
            i = argmax(dists)
            b_new = succs[i]
            b_numeric_new = succs_numeric[i]
            if !in(b_new, B)
                push!(B_new, b_new)
                push!(B_numeric, b_numeric_new)
            end
        end
    end
    union!(B, B_new)
end

function (s::BeliefExpansionSampler)(pomdp::POMDP)
    b0 = initialize_belief(s.updater, initialstate(pomdp))
    b0_numeric = _make_numeric(b0, pomdp)
    B = Set([b0])
    B_numeric = [b0_numeric]
    for _ in 1:s.n
        _exploratory_belief_expansion!(pomdp, B, B_numeric, s)
    end
    return B
end