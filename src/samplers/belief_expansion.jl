using Distances


### Utilities ###

function _make_numeric(b, pomdp::POMDP)
    b = convert_s(AbstractArray{Float64}, b, pomdp)
    return SVector{length(b)}(b)
end

"""
Adapted from PointBasedValueIteration.jl: https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/blob/master/src/pbvi.jl
"""
function _successors(pomdp::POMDP, b, updater::Updater)
    succs = []
    for a in actions(pomdp, b)
        for o in observations(pomdp)            
            s = update(updater, b, a, o)
            push!(succs, s)
        end
    end
    return unique!(succs)
end

### Body ###

"""
Effecient adaptation of algorithm 21.13 from AFDM that uses KDTree.
Only works for finite S, A, O.
"""
function exploratory_belief_expansion!(pomdp::POMDP, B::Set, B_numeric, updater::Updater; metric::NearestNeighbors.MinkowskiMetric=Euclidean())
    tree = KDTree(B_numeric, metric)
    B_new = typeof(B)()
    for b in B
        if isterminal(pomdp, b)
            println("woops")
        end
        succs = _successors(pomdp, b, updater)
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

"""
Wrapper for exploratory_belief_expansion!.
Creates an initial belief set and calls exploratory_belief_expansion! on POMDP n times.
"""
function exploratory_belief_expansion(pomdp::POMDP, updater::Updater; n::Integer=10, metric::NearestNeighbors.MinkowskiMetric=Euclidean())
    b0 = initialize_belief(updater, initialstate(pomdp))
    b0_numeric = _make_numeric(b0, pomdp)
    B = Set([b0])
    B_numeric = [b0_numeric]
    for _ in 1:n
        exploratory_belief_expansion!(pomdp, B, B_numeric, updater; metric)
    end
    return B
end