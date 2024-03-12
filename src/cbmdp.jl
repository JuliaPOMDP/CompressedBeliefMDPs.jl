# TODO: add particle filters --> AbstractParticleBelief https://juliapomdp.github.io/ParticleFilters.jl/latest/beliefs/


"""
    struct CompressedBeliefMDP{B, A} <: MDP{B, A}

A struct representing a Markov Decision Process (MDP) with compressed beliefs.

# Fields
- `bmdp::GenerativeBeliefMDP`: The underlying GenerativeBeliefMDP that defines the original MDP.
- `compressor::Compressor`: The compressor used to compress beliefs in the MDP.

## Type Parameters
- `B`: Type of the belief space.
- `A`: Type of the action space.

## Example
```julia
pomdp = TigerPOMDP()
updater = DiscreteUpdater(pomdp)
compressor = PCACompressor(1)
generative_mdp = GenerativeBeliefMDP(pomdp, updater)
mdp = CompressedBeliefMDP(generative_mdp, compressor)
```
"""
struct CompressedBeliefMDP{B, A} <: MDP{B, A}
    bmdp::GenerativeBeliefMDP
    compressor::Compressor
end

"""
    struct CompressedBeliefMDPState

A struct representing the state of a CompressedBeliefMDP.

# Fields
- `b̃::AbstractArray{<:Real}`: Compressed belief vector.
"""
struct CompressedBeliefMDPState
    b̃::AbstractArray{<:Real}
end

# TODO: merge docstring of constructor w/ docstring for the struct proper
"""
    CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)

Create a CompressedBeliefMDP based on a given POMDP, updater, and compressor.

This function initializes a `GenerativeBeliefMDP` using the provided `pomdp` and `updater`.
It then constructs a `CompressedBeliefMDP` with the specified `compressor`.

# Arguments
- `pomdp::POMDP`: The original partially observable Markov decision process (POMDP).
- `updater::Updater`: The belief updater used in the GenerativeBeliefMDP.
- `compressor::Compressor`: The compressor used to compress beliefs in the MDP.

# Returns
- `CompressedBeliefMDP{CompressedBeliefMDPState, actiontype(bmdp)}`: The compressed belief MDP.

## Example
```julia
pomdp = MyPOMDP()  # replace with your specific POMDP type
updater = MyUpdater()  # replace with your specific updater type
compressor = MyCompressor()  # replace with your specific compressor type

mdp = CompressedBeliefMDP(pomdp, updater, compressor)
```
"""
function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    return CompressedBeliefMDP{CompressedBeliefMDPState, actiontype(bmdp)}(bmdp, compressor)
end


function decode(m::CompressedBeliefMDP, s::CompressedBeliefMDPState)
    b = decompress(m.compressor, s.b̃)
    b = normalize(abs.(b), 1)  # make valid probability distribution
    b = convert_s(statetype(m.bmdp), b, m.bmdp.pomdp)
    return b
end

function encode(m::CompressedBeliefMDP, b)
    b = convert_s(AbstractArray, b, m.bmdp.pomdp)
    b̃ = compress(m.compressor, b)
    s = CompressedBeliefMDPState(b̃)
    return s
end

function POMDPs.gen(m::CompressedBeliefMDP, s::CompressedBeliefMDPState, a, rng::Random.AbstractRNG)
    b = decode(m, s)
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    sp = encode(m, bp)
    return (sp=sp, r=r)
end

POMDPs.actions(m::CompressedBeliefMDP, s::CompressedBeliefMDPState) = actions(m.bmdp, decode(m, s))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.isterminal(m::CompressedBeliefMDP, s::CompressedBeliefMDPState) = isterminal(m.bmdp, decode(m, s))
POMDPs.discount(m::CompressedBeliefMDP) = discount(m.bmdp)
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m, initialstate(m.bmdp))
POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp, a)  # TODO: figure out if can just wrap m.bmdp

# POMDPs.convert_s(::Type{V}, s::CompressedBeliefMDPState, m::CompressedBeliefMDP) where V<:AbstractArray = convert_s(V, s.b̃, m)
function POMDPs.convert_s(::Type{V}, s::CompressedBeliefMDPState, m::CompressedBeliefMDP) where V<:AbstractArray
    # TODO: come up w/ more elegant solution here
    return convert_s(V, s.b̃ isa Matrix ? vec(s.b̃) : s.b̃, m)
end
POMDPs.convert_s(::Type{CompressedBeliefMDPState}, v::AbstractArray, m::CompressedBeliefMDP) = CompressedBeliefMDPState(v)
POMDPs.convert_s(::Type{CompressedBeliefMDPState}, v, m::CompressedBeliefMDP) = CompressedBeliefMDPState(convert_s(Vector, v, m.bmdp.pomdp))

# convenience methods
POMDPs.convert_s(::Type{<:AbstractArray}, s::DiscreteBelief, pomdp::POMDP) = s.b
POMDPs.convert_s(::Type{<:DiscreteBelief}, v, pomdp::POMDP) = DiscreteBelief(pomdp, vec(v))

ExplicitDistribution = Union{SparseCat, BoolDistribution, Deterministic, Uniform}  # distributions w/ explicit PDF from POMDPs.jl (https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/distributions/#Implemented-Distributions)
POMDPs.convert_s(::Type{<:AbstractArray}, b::ExplicitDistribution, pomdp::POMDP) = [pdf(b, s) for s in states(pomdp)]

function POMDPs.convert_s(::Type{<:SparseCat}, vec, pomdp::POMDP)
    @assert length(vec) == length(states(pomdp))
    values = []
    probabilities = []
    for (s, p) in zip(states(pomdp), vec)
        if p != 0
            push!(values, s)
            push!(probabilities, p)
        end
    end
    dist = SparseCat(values, probabilities)
    return dist
end

POMDPs.convert_s(::Type{<:BoolDistribution}, vec, pomdp::POMDP) = BoolDistribution(vec[1])
# TODO: add conversions to Uniform + Deterministic