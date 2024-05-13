struct CircularCorridorState
    row::Integer
    index::Integer
end


struct CircularCorridorPOMDP <: POMDP{CircularCorridorState, Integer, Integer}
    corridor_length::Integer
    num_corridors::Integer
    distribution::VonMises
    goals::AbstractArray{CircularCorridorState}
end


function CircularCorridorPOMDP(corridor_length::Integer; num_corridors::Integer=2)
    goals = []
    goal_indices = rand(1:corridor_length, num_corridors)
    for (row, index) in enumerate(goal_indices)
        goal = CircularCorridorState(row, index)
        push!(goals, goal)
    end
    distribution = VonMises()
    pomdp = CircularCorridorPOMDP(corridor_length, num_corridors, distribution, goals)
    return pomdp
end


# actions
const LEFT = 0
const RIGHT = 1
const SENSE_CORRIDOR = 2
const DECLARE_GOAL = 3


function POMDPs.states(p::CircularCorridorPOMDP)
    return nothing
end


function POMDPs.stateindex(p::CircularCorridorPOMDP, s::CircularCorridorState)
    return nothing
end


function _sample_distribution(p::CircularCorridorsPOMDP)
    sample = rand(p.distribution)
    min_ = minimum(p.distribution)
    max_ = maximum(p.distribution)
    step = (max_ - min_) / p.corridor_length
    bins = collect(min_:step:max_)
    i = searchsortedfirst(bins, sample)  # TODO: replace this w/ NN search? # FIXME: searchsortedfirst doesn't work as needed
    return i
end


function POMDPs.observation(p::CircularCorridorsPOMDP, a::Integer, sp::CircularCorridorsState)
    if a == SENSE_CORRIDOR
        return sp.row
    else
        μ = sp.index
        sample = _sample_distribution(p)
        index = (μ + sample) % p.corridor_length
        return index
    end
end


function POMDPs.transition(p::CircularCorridorsPOMDP)
    return nothing
end



