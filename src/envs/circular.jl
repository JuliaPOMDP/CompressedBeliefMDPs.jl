struct CircularMazeState
    corridor::Integer  # corridor number ∈ [1, ..., n_corridors]
    x::Integer  # position in corridor ∈ [1, ..., corridor_length]
end

struct CircularMaze <: POMDP{Union{CircularMazeState, TerminalState}, Integer, Integer}
    n_corridors::Integer  # number of corridors
    corridor_length::Integer  # corridor length
    probabilities::AbstractArray  # probability masses for creating von Mises distributions
    center::Integer
    discount::Float64

    r_findgoal::Float64
    goals::AbstractArray
    # TODO: add RNG support
end

function _get_mass(d, x1, x2)
    @assert x2 >= x1
    c1 = cdf(d, x1)
    c2 = cdf(d, x2)
    m = c2 - c1
    return m
end

# get the probability masses for each state in a discretized von Mises distribution
function _make_probabilities(corridor_length)
    d = VonMises()  # von Mises distribution with zero mean and unit concentration
    min_ = minimum(d)  # default to -π
    max_ = maximum(d)  # defaults to π
    step = (max_ - min_) / corridor_length
    probabilities = []
    if corridor_length % 2 == 0  # offset indices by step / 2 when corridor_length is even
        for x1 in (min_ + step / 2):step:(max_ - 1.5 * step)
            m = _get_mass(d, x1,  x1 + step)
            push!(probabilities, m)
        end
        m1 = _get_mass(d, max_ - step / 2, max_)
        m2 = _get_mass(d, min_, min_ + step / 2)
        m = m1 + m2
        push!(probabilities, m)
    else
        for x1 in min_:step:(max_ - step)
            m = _get_mass(d, x1, x1 + step)
            push!(probabilities, m)
        end
    end
    return probabilities
end

function CircularMaze(
    n_corridors::Integer, 
    corridor_length::Integer, 
    discount::Float64;
    r_findgoal::Float64 = 1.0,
    rng::AbstractRNG = MersenneTwister()
)
    probabilities = _make_probabilities(corridor_length)
    center = div(corridor_length, 2) + 1
    goals = []
    positions = rand(rng, 1:corridor_length, n_corridors)
    for (corridor, x) in enumerate(positions)
        s = CircularMazeState(corridor, x)
        push!(goals, s)
    end
    pomdp = CircularMaze(n_corridors, corridor_length, probabilities, center, discount, r_findgoal, goals)
    return pomdp
end

function CircularMaze()
    pomdp = CircularMaze(2, 200, 0.99)
    return pomdp
end

const CMAZE_LEFT = 0
const CMAZE_RIGHT = 1
const CMAZE_SENSE_CORRIDOR = 2
const CMAZE_DECLARE_GOAL = 3

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
    space = statetype(pomdp)[]
    for i ∈ 1:pomdp.n_corridors
        for j ∈ 1:pomdp.corridor_length
            state = CircularMazeState(i, j)
            push!(space, state)
        end
    end
    push!(space, terminalstate)
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
    probabilities = repeat(pomdp.probabilities, pomdp.n_corridors)
    probabilities /= pomdp.n_corridors  # normalize values to sum to 1
    values = states(pomdp)
    d = SparseCat(values, probabilities)
    return d
end

function _center_probabilities(pomdp::CircularMaze, x::Integer)
    shifts = pomdp.center - x
    centered_probabilities = circshift(pomdp.probabilities, shifts)
    return centered_probabilities
end

# observations identify the current state modulo 100 with a mean equal to the true state s.x (modulo 100)
function POMDPs.observation(pomdp::CircularMaze, s::CircularMazeState, a::Integer, sp::CircularMazeState)
    if a == CMAZE_SENSE_CORRIDOR
        obs = Deterministic(s.corridor)
    else
        values = 1:pomdp.corridor_length
        probabilities = _center_probabilities(pomdp, s.x)
        d = SparseCat(values, probabilities)
        obs = d
    end
    return obs
end

function POMDPs.observations(pomdp::CircularMaze)
    # NOTE: In JuliaPOMDPs, an observation space is NOT the set of possible distributions, but rather union of the support of all possible observations
    corridors = 1:pomdp.n_corridors  # from CMAZE_SENSE_CORRIDOR
    perms = permutations(pomdp.probabilities)
    space = chain(corridors, perms)  # generator
    return space
end

# TODO: maybe implement POMDPs.obsindex


function POMDPs.transition(pomdp::CircularMaze, s::CircularMazeState, a::Integer)
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
            x = (s.x + 1) % pomdp.corridor_length
        else
            x = s.x
        end
        corridor = s.corridor
        corridor_states = []
        for x_ ∈ 1:pomdp.corridor_length
            s_ = CircularMazeState(corridor, x_)
            push!(corridor_states, s_)
        end
        probabilities = _center_probabilities(pomdp, x)
        d = SparseCat(corridor_states, probabilities)
    end
    return d
end
    
function POMDPs.reward(pomdp::CircularMaze, s::Union{CircularMazeState, TerminalState}, a::Integer)
    @assert a in actions(pomdp) "Unrecognized action $a"
    if s ∈ pomdp.goals && a == CMAZE_DECLARE_GOAL
        r = pomdp.r_findgoal
    else
        r = 0
    end
    return r
end
