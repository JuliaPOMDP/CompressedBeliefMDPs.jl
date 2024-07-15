"""
    CircularMazeState(corridor::Integer, x::Integer)

The `CircularMazeState` struct represents the state of an agent in a circular maze.

# Fields
- `corridor::Integer`: The corridor number. The value ranges from 1 to `n_corridors`.
- `x::Integer`: The position of the state within the corridor. The value ranges from 1 to the `corridor_length`.
"""
struct CircularMazeState
    corridor::Integer  # corridor number ∈ [1, ..., n_corridors]
    x::Integer  # position in corridor ∈ [1, ..., corridor_length]
end

"""
    CircularMaze(n_corridors::Integer, corridor_length::Integer, discount::Float64, r_findgoal::Float64, r_timestep_penalty::Float64)
    CircularMaze(n_corridors::Integer, corridor_length::Integer; kwargs...)
    CircularMaze()
        
    
A Partially Observable Markov Decision Process (POMDP) representing a circular maze environment.

# Fields
- `n_corridors::Integer`: Number of corridors in the circular maze.
- `corridor_length::Integer`: Length of each corridor.
- `probabilities::AbstractArray`: Probability masses for creating von Mises distributions.
- `center::Integer`: The central position in the maze.
- `discount::Float64`: Discount factor for future rewards.
- `r_findgoal::Float64`: Reward for finding the goal.
- `r_timestep_penalty::Float64`: Penalty for each timestep taken.
- `states::AbstractArray`: Array of all possible states in the maze.
- `goals::AbstractArray`: Array of goal states in the maze.

# Example
```julia
using CompressedBeliefMDPs

n_corridors = 8
corridor_length = 25
maze = CircularMaze(n_corridors, corridor_length)
"""
struct CircularMaze <: POMDP{
    Union{CircularMazeState, TerminalState}, 
    Integer, 
    Integer
}
    n_corridors::Integer  # number of corridors
    corridor_length::Integer  # corridor length
    probabilities::AbstractArray  # probability masses for creating von Mises distributions
    center::Integer
    discount::Float64

    r_findgoal::Float64
    r_timestep_penalty::Float64

    states::AbstractArray
    goals::AbstractArray
end

# get the mass of each state
function _get_mass(d, x1, x2)
    @assert x2 >= x1 "x2 ($x2) should be greater than or equal to x1 ($x1)"
    @assert minimum(d) <= x1 <= maximum(d) """
            x1 ($x1) should be within the distribution's range
            (minimum: $(minimum(d)), maximum: $(maximum(d)))
            """
    @assert minimum(d) <= x2 <= maximum(d) """
            x2 ($x2) should be within the distribution's range
            (minimum: $(minimum(d)), maximum: $(maximum(d)))
            """

    c1 = cdf(d, x1)
    c2 = cdf(d, x2)
    m = c2 - c1
    return m
end

# returns an array of probability masses for a single corridor
function _make_probabilities(corridor_length::Integer)
    @assert corridor_length >= 0

    d = VonMises()
    min_ = minimum(d)
    max_ = maximum(d)    
    a = range(min_, max_, length=corridor_length + 1)
    a1 = a[1:end-1]
    a2 = a[2:end]
    probabilities = []
    for (x1, x2) in zip(a1, a2)
        m = _get_mass(d, x1, x2)
        push!(probabilities, m)
    end

    return probabilities
end

function _make_states(n_corridors, corridor_length)
    space = Union{CircularMazeState, TerminalState}[]
    for i ∈ 1:n_corridors
        for j ∈ 1:corridor_length
            state = CircularMazeState(i, j)
            push!(space, state)
        end
    end
    push!(space, terminalstate)
    return space
end

function CircularMaze(
    n_corridors::Integer, 
    corridor_length::Integer; 
    discount::Float64 = 0.99,
    r_findgoal::Float64 = 1.0,
    r_timestep_penalty::Float64 = 0.0,
    rng::AbstractRNG = MersenneTwister()
)
    @assert n_corridors > 0 "Number of corridors must be a positive integer."
    @assert corridor_length > 0 "Corridor length must be a positive integer."
    @assert 0.0 <= discount <= 1.0 "Discount factor must be between 0 and 1."
    @assert r_findgoal >= 0.0 "Reward for finding the goal must be non-negative."
    @assert r_timestep_penalty >= 0.0 "The timestep penalty must be non-negative."

    if typeof(n_corridors) != typeof(corridor_length)
        type1 = typeof(n_corridors)
        type2 = typeof(corridor_length)
        @warn "n_corridors ($type1) and corridor_length ($type2) are not of the same type."
    end

    probabilities = _make_probabilities(corridor_length)
    center = div(corridor_length, 2) + 1

    # make states
    states = _make_states(n_corridors, corridor_length)

    # make goals
    goals = []
    positions = rand(rng, 1:corridor_length, n_corridors)
    for (corridor, x) in enumerate(positions)
        s = CircularMazeState(corridor, x)
        push!(goals, s)
    end

    pomdp = CircularMaze(
        n_corridors, 
        corridor_length, 
        probabilities,
        center, 
        discount, 
        r_findgoal, 
        r_timestep_penalty, 
        states,
        goals
    )
    return pomdp
end

# conveience constructors
function CircularMaze(
    n_corridors::Integer, 
    corridor_length::Integer, 
    discount::Float64,
    r_findgoal::Float64,
    r_timestep_penalty::Float64,
)
    pomdp = CircularMaze(
        n_corridors, 
        corridor_length;
        discount,
        r_findgoal=r_findgoal, 
        r_timestep_penalty=r_timestep_penalty, 
    )
    return pomdp
end

function CircularMaze()
    pomdp = CircularMaze(2, 100)
    return pomdp
end

const CMAZE_LEFT = 1
const CMAZE_RIGHT = 2
const CMAZE_SENSE_CORRIDOR = 3
const CMAZE_DECLARE_GOAL = 4

const ACTIONS = [
    CMAZE_LEFT,
    CMAZE_RIGHT,
    CMAZE_SENSE_CORRIDOR,
    CMAZE_DECLARE_GOAL
]

function POMDPs.actions(::CircularMaze)
    A = ACTIONS
    return A
end

function POMDPs.actionindex(::CircularMaze, a::Integer)
    index = a
    return index
end

function POMDPs.states(pomdp::CircularMaze)
    space = pomdp.states
    return space
end

function POMDPs.stateindex(pomdp::CircularMaze, s::CircularMazeState)
    index = (s.corridor - 1) * pomdp.corridor_length + s.x
    return index
end

function POMDPs.stateindex(pomdp::CircularMaze, ::TerminalState)
    index = pomdp.n_corridors * pomdp.corridor_length + 1
    return index
end

# the initial state distribution is a von Mises distributions each over corridor with a mean at the center
function POMDPs.initialstate(pomdp::CircularMaze)
    probabilities = repeat(pomdp.probabilities ./ pomdp.n_corridors, pomdp.n_corridors)
    values = states(pomdp)
    push!(probabilities, 0)  # OBOE from terminal state
    d = SparseCat(values, probabilities)
    return d
end

function _center_probabilities(pomdp::CircularMaze, x::Integer)
    shifts = pomdp.center - x
    centered_probabilities = circshift(pomdp.probabilities, shifts)
    return centered_probabilities
end

# observations identify the current state modulo 100 with a mean equal to the true state s.x (modulo 100)
function POMDPs.observation(
    pomdp::CircularMaze, 
    s::CircularMazeState, 
    a::Integer, 
    ::CircularMazeState
)
    @assert a in actions(pomdp) "Unrecognized action $a"
    if a == CMAZE_SENSE_CORRIDOR
        obs = Deterministic(s.corridor)
    else
        values = states(pomdp)
        probabilities = _center_probabilities(pomdp, s.x)
        probabilities = repeat(pomdp.probabilities, pomdp.n_corridors)
        probabilities /= pomdp.n_corridors  # normalize values to sum to 1
        push!(probabilities, 0)  # address OBOE from terminalstate
        d = SparseCat(values, probabilities)
        obs = d
    end
    return obs
end

function POMDPs.observation(::CircularMaze, s::TerminalState)
    return Deterministic(s)
end

function POMDPs.observations(pomdp::CircularMaze)
    states = pomdp.states
    corridors = collect(1:pomdp.n_corridors)
    O = vcat(states, corridors)
    O = convert(Vector{Union{CircularMazeState, TerminalState, Integer}}, O)
    return O
end

# TODO: maybe implement POMDPs.obsindex

function POMDPs.transition(
    pomdp::CircularMaze, 
    s::CircularMazeState, 
    a::Integer
)
    @assert a in actions(pomdp) "Unrecognized action $a"
    if a == CMAZE_DECLARE_GOAL
        # env resets when goal is declared regardless of whether agent is actually at the goal
        d = Deterministic(terminalstate)
    else
        # move CMAZE_LEFT/CMAZE_RIGHT with some von Mises noise
        if a == CMAZE_LEFT
            x = s.x - 1
            if x < 1
                x = pomdp.corridor_length
            end
        elseif a == CMAZE_RIGHT
            x = s.x + 1
            if x > pomdp.corridor_length
                x = 1
            end
        else
            x = s.x
        end
        corridor = s.corridor
        states = pomdp.states
        start = (corridor - 1) * pomdp.corridor_length + 1
        stop = start + pomdp.corridor_length - 1
        corridor_states = states[start:stop]
        probabilities = _center_probabilities(pomdp, x)
        d = SparseCat(corridor_states, probabilities)
    end
    return d
end

function POMDPs.transition(
    pomdp::CircularMaze, 
    ::TerminalState, 
    a::Integer
)
    @assert a in actions(pomdp) "Unrecognized action $a"
    terminal = Deterministic(terminalstate)
    return terminal
end
   
# NOTE: https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/model/#Terminal-State -- no type unions
function POMDPs.reward(
    pomdp::CircularMaze, 
    s::CircularMazeState,
    a::Integer
)
    @assert a in actions(pomdp) "Unrecognized action $a"
    if s ∈ pomdp.goals && a == CMAZE_DECLARE_GOAL
        r = pomdp.r_findgoal
    else
        r = 0
    end
    r -= pomdp.r_timestep_penalty
    return r
end

function POMDPs.reward(
    ::CircularMaze, 
    ::TerminalState,
    ::Integer
)
    r = 0
    return r
end

function POMDPs.discount(pomdp::CircularMaze)
    disc = pomdp.discount
    return disc
end

## hack to avoid exploring terminal states
CMAZE_TERMINAL_FLAG = false
function POMDPTools.ModelTools.gbmdp_handle_terminal(::CircularMaze, ::Updater, b, s, a, rng)
    global CMAZE_TERMINAL_FLAG = true
    return b
end