struct CompressedBeliefMDP{B, A} <: MDP{B, A}
    bmdp::GenerativeBeliefMDP
    compressor::Compressor
    ϕ::Bijection  # ϕ: belief ↦ compress(compressor, belief); NOTE: While compressions aren't usually injective, we cache compressed beliefs on a first-come, first-served basis, so the *cache* is effectively bijective.
end


# TODO: merge docstring of constructor w/ docstring for the struct proper
function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    # Hack to determine typeof(b̃)
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    b = initialstate(bmdp).val
    b̃ = compress(compressor, convert_s(AbstractVector{Float64}, b, bmdp.pomdp))
    B = typeof(b)
    B̃ = typeof(b̃)
    ϕ = Bijection{B, B̃}()
    return CompressedBeliefMDP{B̃, actiontype(bmdp)}(bmdp, compressor, ϕ)
end

function decode(m::CompressedBeliefMDP, b̃)
    b = m.ϕ(b̃)
    return b
end

function encode(m::CompressedBeliefMDP, b)
    b = convert_s(AbstractVector{Float64}, b, m)
    b̃ = get!(m.ϕ, b) do
        b = convert_s(AbstractArray{Float64}, b, m)  # TODO: not sure if I need a `let b = ...` here
        compress(m.compressor, b)  # NOTE: compress is only called if b ∉ domain(m.ϕ)
    end
    return b̃
end

function POMDPs.gen(m::CompressedBeliefMDP, b̃, a, rng::Random.AbstractRNG)
    b = decode(m, b̃)
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    b̃p = encode(m, bp)
    return (sp=b̃p, r=r)
end

# TODO: use macro forwarding
# TODO: read about orthogonalized code on julia documetation
POMDPs.actions(m::CompressedBeliefMDP, b̃) = actions(m.bmdp, decode(m, b̃))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.isterminal(m::CompressedBeliefMDP, b̃) = isterminal(m.bmdp, decode(m, b̃))
POMDPs.discount(m::CompressedBeliefMDP) = discount(m.bmdp)
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m, initialstate(m.bmdp))
POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp, a)

POMDPs.convert_s(t::Type, s, m::CompressedBeliefMDP) = convert_s(t, s, m.bmdp.pomdp)
POMDPs.convert_s(t::Type{<:AbstractArray}, s::AbstractArray, m::CompressedBeliefMDP) = convert_s(t, s, m.bmdp.pomdp)  # NOTE: this second implementation is b/c to get around a requirement from POMDPLinter

# TODO: maybe don't include sparsecat
ExplicitDistribution = Union{SparseCat, BoolDistribution, Deterministic, Uniform}  # distributions w/ explicit PDFs from POMDPs.jl (https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/distributions/#Implemented-Distributions)
POMDPs.convert_s(::Type{<:AbstractArray}, s::ExplicitDistribution, m::POMDP) = [pdf(s, x) for x in states(m)]


# function POMDPs.convert_s(t::Type{V}, s, m::CompressedBeliefMDP) where V<:AbstractArray
#     convert_s(t, s, m.bmdp.pomdp)
# end

# function POMDPs.convert_s(t::Type{S}, v::V, m::CompressedBeliefMDP) where {S, V<:AbstractArray}
#     convert_s(t, v, m.bmdp.pomdp)
# end

# TODO: try to add issue to github for this since this is only to get around POMDPLinter





# POMDPs.convert_s(::Type{V}, s::CompressedBeliefMDPState, m::CompressedBeliefMDP) where V<:AbstractArray = convert_s(V, s.b̃, m)
# function POMDPs.convert_s(::Type{V}, s, m::CompressedBeliefMDP) where V<:AbstractArray
#     # TODO: come up w/ more elegant solution here
#     return convert_s(V, s.b̃ isa Matrix ? vec(s.b̃) : s.b̃, m)
# end
# # POMDPs.convert_s(::Type{CompressedBeliefMDPState}, v::AbstractArray, m::CompressedBeliefMDP) = CompressedBeliefMDPState(v)
# POMDPs.convert_s(::Type{CompressedBeliefMDPState}, v, m::CompressedBeliefMDP) = CompressedBeliefMDPState(convert_s(Vector, v, m.bmdp.pomdp))

# convenience methods
POMDPs.convert_s(::Type{V}, s::DiscreteBelief, p::POMDP) where V<:AbstractArray = s.b
# POMDPs.convert_s(::Type{S}, v::V, p::POMDP) where {S, V<:AbstractArray} = DiscreteBelief(p, _process(convert(Vector{Float64}, vec(v))))  # TODO: is there Julian shorthand for type conversions?
# POMDPs.convert_s(::Type{<:AbstractArray}, s::DiscreteBelief, pomdp::POMDP) = s.b
# POMDPs.convert_s(::Type{<:DiscreteBelief}, v, problem::Union{POMDP, MDP) = DiscreteBelief(pomdp, vec(v))

# ExplicitDistribution = Union{SparseCat, BoolDistribution, Deterministic, Uniform}  # distributions w/ explicit PDF from POMDPs.jl (https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/distributions/#Implemented-Distributions)
# POMDPs.convert_s(::Type{<:AbstractArray}, b::ExplicitDistribution, pomdp::POMDP) = [pdf(b, s) for s in states(pomdp)]

# function POMDPs.convert_s(::Type{<:SparseCat}, vec, pomdp::POMDP)
#     @assert length(vec) == length(states(pomdp))
#     values = []
#     probabilities = []
#     for (s, p) in zip(states(pomdp), vec)
#         if p != 0
#             push!(values, s)
#             push!(probabilities, p)
#         end
#     end
#     dist = SparseCat(values, probabilities)
#     return dist
# end

# POMDPs.convert_s(::Type{<:BoolDistribution}, vec, pomdp::POMDP) = BoolDistribution(vec[1])
# # TODO: add conversions to Uniform + Deterministic