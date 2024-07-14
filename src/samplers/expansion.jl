using Distances


"""
    BeliefExpansionSampler

Fast extension of exploratory belief expansion (Algorithm 21.13 in [Algorithms for Decision Making](https://algorithmsbook.com/)) that uses
``k``-d trees.

## Fields
- `updater::Updater`: The updater used to update beliefs.
- `metric::NearestNeighbors.MinkowskiMetric`: The metric used to measure distances between beliefs.
It must be a [Minkowski metric](https://en.wikipedia.org/wiki/Minkowski_distance).
- `n::Integer`: The number of belief expansions to perform.

## Constructors
    BeliefExpansionSampler(pomdp::POMDP; updater::Updater=DiscreteUpdater(pomdp),
    metric::NearestNeighbors.MinkowskiMetric=Euclidean(), n::Integer=3)

## Methods
    (s::BeliefExpansionSampler)(pomdp::POMDP)

Creates an initial belief and performs exploratory belief expansion. Returns the unique belief states. 
Only works for POMDPs with discrete state, action, and observation spaces.

## Example Usage

```julia-repl
julia> pomdp = TigerPOMDP();
julia> sampler = BeliefExpansionSampler(pomdp; n=2);
julia> beliefs = sampler(pomdp)
Set{DiscreteBelief{TigerPOMDP, Bool}} with 4 elements:
  DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.15000000000000002, 0.85])
  DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.5, 0.5])
  DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.85, 0.15000000000000002])
  DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.9697986577181208, 0.030201342281879207])
```
"""
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
    @assert n > 0 "n must be greater than 0"
    return BeliefExpansionSampler(updater, metric, n)
end

function _make_numeric(
    b, 
    pomdp::POMDP
)
    b = convert_s(AbstractArray{Float64}, b, pomdp)
    return SVector{length(b)}(b)
end

function _successors(
    pomdp::POMDP, 
    b, 
    updater::Updater
)
    # Adapted from PointBasedValueIteration.jl: https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/blob/master/src/pbvi.jl
    succs = []
    for a in actions(pomdp, b)
        for o in observations(pomdp)
            s = update(updater, b, a, o)
            push!(succs, s)
        end
    end
    return unique!(succs)
end


function _successors(
    pomdp::POMDP, 
    b, 
    updater::DiscreteUpdater
)
    # Adapted from PointBasedValueIteration.jl: https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/blob/master/src/pbvi.jl
    succs = []
    for a in actions(pomdp, b)
        for o in observations(pomdp)
            try
                s = update(updater, b, a, o)
                push!(succs, s)
            catch e
                # skip impossible observations
                if isa(e, ErrorException) && occursin("Failed discrete belief update", e.msg)
                    continue
                else
                    rethrow(e)  # Re-throw if it's a different error
                end
            end
        end
    end
    return unique!(succs)
end


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
    s0 = initialstate(pomdp)
    b0 = initialize_belief(s.updater, s0)
    b0_numeric = _make_numeric(b0, pomdp)
    B = Set([b0])
    B_numeric = [b0_numeric]
    for _ in 1:s.n
        _exploratory_belief_expansion!(pomdp, B, B_numeric, s)
    end
    return B
end