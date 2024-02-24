using POMDPs
using POMDPModelTools: GenerativeBeliefMDP
using POMDPTools: DiscreteBelief

using Random: AbstractRNG
using LinearAlgebra

# TEMP
using StaticArrays

"""
    decode(m::CompressedBeliefMDP, b̃)

Return the decoded belief with type statetype(m.bdmp).
"""
function decode end

"""
    encode(m::CompressedBeliefMDP, b::AbstractArray)

Return the encoded belief.
"""
function encode end


# TODO: perhaps change parameters to make less confusing since our state type is a vector and bmdp is a belief
struct CompressedBeliefMDP{B, A} <: MDP{B, A}
    bmdp::GenerativeBeliefMDP
    compressor::Compressor
end

function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    # NOTE: hack to determine belief type
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    b0 = initialize_belief(updater, initialstate(pomdp))
    # TODO: make more general later w/ encode rather than compress or make more encode methods (probably have to use multiple dispatch here); something like encode(pomdp, compressor, )
    # b̃0 = encode(pomdp, b0)
    b̃0 = vec(compress(compressor, b0.b'))
    return CompressedBeliefMDP{typeof(b̃0), actiontype(bmdp)}(bmdp, compressor)
end

function POMDPs.gen(m::CompressedBeliefMDP, b̃::V, a, rng::AbstractRNG) where V<:AbstractArray
    b = decode(m, b̃)
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    b̃p = encode(m, bp)
    return (sp=b̃p, r=r)
end

POMDPs.actions(m::CompressedBeliefMDP, b̃) = actions(m.bmdp, decode(m, b̃))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.isterminal(m::CompressedBeliefMDP, b̃) = isterminal(m.bmdp, decode(m, b̃))
POMDPs.discount(m::CompressedBeliefMDP{B, A}) where {B, A} = discount(m.bmdp)
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m, initialstate(m.bmdp))
# TODO: fix
# POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp)  # Why doesn't BabyPOMDP have an action index?
POMDPs.actionindex(m::CompressedBeliefMDP, a) = 1

########################
# MOVE EVERYTHING BELOW HERE SOMEWHERE ELSE LATER
########################

# TODO: use parameterized types here
function decode(m::CompressedBeliefMDP, b̃)
    b = decompress(m.compressor, b̃)
    b = normalize(b, 1)  # TODO: make sure they are all positive
    return DiscreteBelief(m.bmdp.pomdp, vec(b))
end

# TODO: perhaps store belief type as parameter of the struct for more general decoding
function encode(m::CompressedBeliefMDP, b::DiscreteBelief)
    return vec(compress(m.compressor, b.b'))
end

# TODO: try to make more general
# TODO: fix
# function POMDPs.convert_s(::Type{V}, s, problem::CompressedBeliefMDP) where V<:Vector
#     return [0.5, 0.5]
# end

# TODO: figure out how to make more general
# TODO: fix
# function POMDPs.convert_s(::Type{S}, vec::V, problem::CompressedBeliefMDP) where {S, V<:AbstractArray}
#     return decode(problem, vec)
# end
