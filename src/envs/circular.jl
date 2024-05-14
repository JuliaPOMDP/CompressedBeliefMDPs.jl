struct CircularMazeState
    corridor::Integer  # corridor number
    x::Integer  # position in corridor
end

struct CircularMaze <: POMDP{Union{CircularMazeState, TerminalState}, Integer, Integer}
    n_corridors::Integer  # number of corridors
    corridor_length::Integer  # corridor length
    probabilities::AbstractArray  # probability masses for creating von Mises distributions
    center::Integer
    discount::Float64
end

function _get_mass(d, x1, x2)
    @assert x2 >= x1
    c1 = cdf(d, x1)
    c2 = cdf(d, x2)
    m = c2 - c1
    return m
end

function CircularMaze(
    n_corridors::Integer, 
    corridor_length::Integer, 
    discount::Float64,
)
    d = VonMises()
    min_ = minimum(d)
    max_ = maximum(d)
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
    center = div(corridor_length, 2) + 1
    pomdp = CircularMaze(n_corridors, corridor_length, probabilities, center, discount)
    return pomdp
end

function CircularMaze()
    pomdp = CircularMaze(2, 200, 0.99)
    return pomdp
end

const LEFT = 0
const RIGHT = 1
const SENSE_CORRIDOR = 2
const DECLARE_GOAL = 3

function POMDPs.actions(::CircularMaze)
    A = [
        LEFT,
        RIGHT,
        SENSE_CORRIDOR,
        DECLARE_GOAL
    ]
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

function POMDPs.stateindex(::CircularMaze, s::CircularMazeState)
    index = (s.corridor - 1) + s.x
    return index
end

function POMDPs.stateindex(pomdp::CircularMaze, ::TerminalState)
    index = pomdp.n_corridors * pomdp.corridor_length + 1
    return index
end

function POMDPs.initialstate(pomdp::CircularMaze)
    # TODO
    return nothing
end

function _make_sparse_cat(pomdp::CircularMaze, x::Integer)
    values = states(pomdp)
    shifts = pomdp.center - s.x
    probabilities = circshift(pomdp.probabilities, shifts)
    d = SparseCat(values, probabilities)
    return d
end

function POMDPs.observation(pomdp::CircularMaze, s::CircularMazeState, a::Integer)
    if a == SENSE_CORRIDOR
        obs = s.corridor
    else
        obs = _make_sparse_cat(pomdp, s.x)
    end
    return obs
end

function POMDPs.observations(pomdp::CircularMaze)
    corridors = 1:pomdp.n_corridors  # from SENSE_CORRIDOR
    distribution_observations = permutations(pomdp.probabilities)
    space = chain(corridors, distribution_observations)  # generator
    return space
end

# TODO: maybe implement POMDPs.obsindex

function POMDPs.transition(pomdp::CircularMaze, s::CircularMazeState, a::Integer)
    if a == DECLARE_GOAL
        d = Deterministic(terminalstate)
    else
        if a == LEFT
            x = s.x - 1
            if x < 1
                x = pomdp.corridor_length
            end
        elseif a == RIGHT
            x = (s.x + 1) % pomdp.corridor_length
        else
            x = s.x
        end
        d = _make_sparse_cat(pomdp, x)
    end
    return d
end
    