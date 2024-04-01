"""
    CompressedBeliefMDP{B, A}

The `CompressedBeliefMDP` struct is a generalization of the compressed belief-state MDP presented in 
[Exponential Family PCA for Belief Compression in POMDPs](https://papers.nips.cc/paper_files/paper/2002/hash/a11f9e533f28593768ebf87075ab34f2-Abstract.html).

## Type Parameters
- `B`: The type of compressed belief states.
- `A`: The type of actions.

## Fields
- `bmdp::GenerativeBeliefMDP`: The generative belief-state MDP.
- `compressor::Compressor`: The compressor used to compress belief states.
- `ϕ::Bijection`: A bijection representing the mapping from uncompressed belief states to compressed belief states. See notes. 

## Constructors
    CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)

Constructs a `CompressedBeliefMDP` using the specified POMDP, updater, and compressor.

## Example Usage

```julia
pomdp = TigerPOMDP()
updater = DiscreteUpdater(pomdp)
compressor = PCACompressor(1)
mdp = CompressedBeliefMDP(pomdp, updater, compressor)
```

For continuous POMDPs, see [ParticleFilters.jl](https://juliapomdp.github.io/ParticleFilters.jl/latest/basic/).

## Notes
- While compressions aren't usually injective, we cache beliefs and their compressions on a first-come, first-served basis, so we can effectively use a bijection without loss of generality.
"""
struct CompressedBeliefMDP{B, A} <: MDP{B, A}
    bmdp::GenerativeBeliefMDP
    compressor::Compressor
    ϕ::Bijection  # ϕ: belief ↦ compressor(belief); NOTE: While compressions aren't usually injective, we cache compressed beliefs on a first-come, first-served basis, so the *cache* is effectively bijective.
end

function CompressedBeliefMDP(pomdp::POMDP, updater::Updater, compressor::Compressor)
    # Hack to determine typeof(b̃)
    bmdp = GenerativeBeliefMDP(pomdp, updater)
    b = initialstate(bmdp).val
    b̃ = compressor(convert_s(AbstractVector{Float64}, b, bmdp.pomdp))
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
    if b ∈ domain(m.ϕ)
        b̃ = m.ϕ[b]
    else
        b_numerical = convert_s(AbstractArray{Float64}, b, m)
        b̃ = m.compressor(b_numerical)
        if b̃ ∉ image(m.ϕ)
            m.ϕ[b] = b̃
        end
    end
    return b̃
end

function POMDPs.gen(m::CompressedBeliefMDP, b̃, a, rng::Random.AbstractRNG)
    b = decode(m, b̃)
    bp, r = @gen(:sp, :r)(m.bmdp, b, a, rng)
    b̃p = encode(m, bp)
    return (sp=b̃p, r=r)
end

POMDPs.states(m::CompressedBeliefMDP) = [encode(m, initialize_belief(m.bmdp.updater, s)) for s in states(m.bmdp.pomdp)]
POMDPs.initialstate(m::CompressedBeliefMDP) = encode(m, initialstate(m.bmdp))
POMDPs.actions(m::CompressedBeliefMDP, b̃) = actions(m.bmdp, decode(m, b̃))
POMDPs.actions(m::CompressedBeliefMDP) = actions(m.bmdp)
POMDPs.actionindex(m::CompressedBeliefMDP, a) = actionindex(m.bmdp.pomdp, a)
POMDPs.isterminal(m::CompressedBeliefMDP, b̃) = isterminal(m.bmdp, decode(m, b̃))
POMDPs.discount(m::CompressedBeliefMDP) = discount(m.bmdp)

POMDPs.convert_s(t::Type, s, m::CompressedBeliefMDP) = convert_s(t, s, m.bmdp.pomdp)
POMDPs.convert_s(t::Type{<:AbstractArray}, s::AbstractArray, m::CompressedBeliefMDP) = convert_s(t, s, m.bmdp.pomdp)  # NOTE: this second implementation is b/c to get around a requirement from POMDPLinter

# TODO: maybe exclude include sparsecat; e.g., for sparsecat do [pdf(s, x) for x in support(s)]
ExplicitDistribution = Union{SparseCat, BoolDistribution, Deterministic, Uniform}  # distributions w/ explicit PDFs from POMDPs.jl (https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/distributions/#Implemented-Distributions)
POMDPs.convert_s(::Type{<:AbstractArray}, s::ExplicitDistribution, m::POMDP) = [pdf(s, x) for x in states(m)]
POMDPs.convert_s(::Type{V}, s::DiscreteBelief, p::POMDP) where V<:AbstractArray = s.b
POMDPs.convert_s(::Type{<:AbstractArray}, s::ParticleCollection, p::POMDP) = weights(s)