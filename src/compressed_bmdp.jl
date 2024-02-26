using POMDPs
using POMDPModelTools: GenerativeBeliefMDP
using POMDPTools: DiscreteBelief

using Random: AbstractRNG
using LinearAlgebra

"""
    decode(m::POMDP, c::Compressor, b̃)

Return the decoded belief.
"""
function decode end

"""
    encode(m::POMDP, c::Compressor, b)

Return the encoded belief.
"""
function encode end


struct CompressedBeliefMDP{B, A} <: MDP{B, A}
    bmdp::GenerativeBeliefMDP
    compressor::Compressor
end


function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    # NOTE: hack to determine belief type
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    b0 = initialize_belief(updater, initialstate(pomdp))
    b̃0 = encode(pomdp, compressor, b0)
    return CompressedBeliefMDP{typeof(b̃0), actiontype(bmdp)}(bmdp, compressor)
end


function POMDPs.gen(m::CompressedBeliefMDP, b̃::V, a, rng::AbstractRNG) where V<:AbstractArray
    b = decode(m.bmdp.pomdp, m.compressor, b̃)
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    b̃p = encode(m.bmdp.pomdp, m.compressor, bp)
    return (sp=b̃p, r=r)
end


POMDPs.actions(m::CompressedBeliefMDP, b̃) = actions(m.bmdp, decode(m.bmdp.pomdp, m.compressor, b̃))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.isterminal(m::CompressedBeliefMDP, b̃) = isterminal(m.bmdp, decode(m.bmdp.pomdp, m.compressor, b̃))
POMDPs.discount(m::CompressedBeliefMDP{B, A}) where {B, A} = discount(m.bmdp)
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m.bmdp.pomdp, m.compressor, initialstate(m.bmdp))
POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp, a)


# Convenience methods
decode(m::POMDP, c::Compressor, b̃) = DiscreteBelief(m, vec(normalize(decompress(c, b̃), 1)))
encode(m::POMDP, c::Compressor, b::DiscreteBelief) = vec(compress(c, b.b'))
