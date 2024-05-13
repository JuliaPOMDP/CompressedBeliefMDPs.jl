struct CircularCorridorState
    row::Integer
    index::Integer
end


struct CircularCorridorPOMDP <: POMDP{Union{CircularCorridorState, TerminalState}, Integer, Integer}
    corridor_length::Integer
    num_corridors::Integer
    distribution::VonMises
    goals::AbstractArray{CircularCorridorState}
    discount_factor::Float64
end


function CircularCorridorPOMDP(; corridor_length::Integer=200, num_corridors::Integer=2, discount_factor::Float64=0.95)
    goals = []
    goal_indices = rand(1:corridor_length, num_corridors)
    for (row, index) in enumerate(goal_indices)
        goal = CircularCorridorState(row, index)
        push!(goals, goal)
    end
    distribution = VonMises()
    pomdp = CircularCorridorPOMDP(corridor_length, num_corridors, distribution, goals, discount_factor)
    return pomdp
end


# actions
const LEFT = 0
const RIGHT = 1
const SENSE_CORRIDOR = 2
const DECLARE_GOAL = 3


function POMDPs.actions(::CircularCorridorPOMDP)
    A = [
        LEFT,
        RIGHT,
        SENSE_CORRIDOR,
        DECLARE_GOAL
    ]
    return A
end


function POMDPs.initial_belief(p::CircularCorridorPOMDP)
    num_states = p.num_corridors * p.corridor_length
    belief = DiscreteBelief(num_states)
    return belief
end


function POMDPs.states(p::CircularCorridorPOMDP)
    S = []
    for row in 1:p.num_corridors
        for index in 1:p.corridor_length
            state = CircularCorridorState(row, index)
            push!(S, state)
        end
    end
    return S
end


function POMDPs.stateindex(p::CircularCorridorPOMDP, s::CircularCorridorState)
    i = p.corridor_length * s.row + s.index
    return i
end


function _sample_distribution(p::CircularCorridorsPOMDP, rng)
    sample = rand(rng, p.distribution)
    min_ = minimum(p.distribution)
    max_ = maximum(p.distribution)
    step = (max_ - min_) / p.corridor_length
    bins = collect(min_:step:max_)
    i = searchsortedfirst(bins, sample)  # TODO: replace this w/ NN search? # FIXME: searchsortedfirst doesn't work as needed
    return i
end


function POMDPs.observation(p::CircularCorridorsPOMDP, a::Integer, sp::CircularCorridorsState)
    ImplicitDistribution() do rng
        if a == SENSE_CORRIDOR
            obs = sp.row
        else
            # TODO: how to represent n-modal distributions??
            μ = sp.index
            sample = _sample_distribution(p, rng)
            index = (μ + sample) % p.corridor_length
            obs = index
        end
        return obs
    end
end


function POMDPs.transition(p::CircularCorridorsPOMDP, s::CircularCorridorState, a)
    ImplicitDistribution() do rng
        if a == DECLARE_GOAL
            sp = TerminalState()
        else
            if a == LEFT
                μ = abs(s.index - 1) % p.corridor_length
            elseif a == RIGHT
                μ = (s.index + 1) % p.corridor_length
            else
                μ = s.index
            end
            sample = _sample_distribution(p, rng)
            index = (μ + sample) % p.corridor_length
            row = s.row
            sp = CircularCorridorState(row, index)
        end
        return sp
    end
end


function POMDPs.reward(p::CircularCorridorPOMDP, s::CircularCorridorState, a)
    if a == DECLARE_GOAL && s in p.goals
        r = 1
    else
        r = 0
    end
    return r
end
