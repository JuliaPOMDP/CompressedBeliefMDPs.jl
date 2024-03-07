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

# TODO: fix?
function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    # NOTE: hack to determine belief type
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    # b0 = initialize_belief(updater, initialstate(pomdp))
    # b̃0 = encode(pomdp, compressor, b0)
    # # return CompressedBeliefMDP{typeof(b̃0), actiontype(bmdp)}(bmdp, compressor)
    return CompressedBeliefMDP{Vector{Float64}, actiontype(bmdp)}(bmdp, compressor)
end


function POMDPs.gen(m::CompressedBeliefMDP, b̃::V, a, rng::Random.AbstractRNG) where V<:AbstractArray
    b = decode(m.bmdp.pomdp, m.compressor, b̃)  # convert compressed vector into belief
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    b̃p = encode(m.bmdp.pomdp, m.compressor, bp)  # convert belief into compressed vector
    return (sp=b̃p, r=r)
end

POMDPs.actions(m::CompressedBeliefMDP, b̃) = actions(m.bmdp, decode(m.bmdp.pomdp, m.compressor, b̃))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.isterminal(m::CompressedBeliefMDP, b̃) = isterminal(m.bmdp, decode(m.bmdp.pomdp, m.compressor, b̃))
POMDPs.discount(m::CompressedBeliefMDP) = discount(m.bmdp)
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m.bmdp.pomdp, m.compressor, initialstate(m.bmdp))
POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp, a)
POMDPs.states(m::CompressedBeliefMDP) = states(m.bmdp.pomdp)  # GenerativeBeliefMDP doesn't implement states(...), so we bypass it.

# Convenience methods
process(x) = normalize(abs.(x), 1)

function decode(m::POMDP, c::Compressor, b̃; postprocessing=false)
    b = decompress(c, b̃')
    if postprocessing
        b = process(x)
    end
    b0 = initialstate(m)
    b = convert_s(typeof(b0), vec(b), m)  # TODO: remove processing from convert_s methods
    return b
end

function encode(m::POMDP, c::Compressor, b)
    b = convert_s(Vector, b, m)
    b̃ = compress(c, b')
    return vec(b̃)
end

# TODO clean up
function POMDPs.convert_s(T::Type{<:AbstractArray}, s::StaticArray, pomdp::POMDP)
    return convert_s(T, s, pomdp.bmdp.pomdp)
end

function POMDPs.convert_s(::Type{<:AbstractArray}, s::DiscreteBelief, pomdp::POMDP)
    return s.b
end

# TODO: add support for all the implemented distributions in pomdps.jl by using a union type like Union{SparseCat, Uniform, ...} (https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/distributions/#Implemented-Distributions)
function POMDPs.convert_s(::Type{<:AbstractArray}, b::Union{SparseCat, BoolDistribution}, pomdp::POMDP)
    return [pdf(b, s) for s in states(pomdp)]
end

function POMDPs.convert_s(::Type{<:SparseCat}, vec, pomdp::POMDP)
    @assert length(vec) == length(states(pomdp))
    values = []
    probabilities = []
    for (s, p) in zip(states(pomdp), process(vec))
        if p != 0
            push!(values, s)
            push!(probabilities, p)
        end
    end
    dist = SparseCat(values, probabilities)
    return dist
end

POMDPs.convert_s(::Type{<:BoolDistribution}, vec, pomdp::POMDP) = BoolDistribution(process(vec)[1])