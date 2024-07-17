"""
    make_cache(B, B̃)

Helper function that creates a cache that maps each unique belief from the set `B` to its corresponding compressed representation in `B̃`.

# Arguments
- `B::Vector{<:Any}`: A vector of beliefs.
- `B̃::Matrix{Float64}`: A matrix where each row corresponds to the compressed representation of the beliefs in `B`.

# Returns
- `Dict{<:Any, Vector{Float64}}`: A dictionary mapping each unique belief in `B` to its corresponding compressed representation in `B̃`.

# Example Usage
```julia
B = [belief1, belief2, belief3]
B̃ = [compressed_belief1; compressed_belief2; compressed_belief3]
ϕ = make_cache(B, B̃)
```
"""
function make_cache(B, B̃)
    ϕ = Dict(unique(t->t[2], zip(B, eachrow(B̃))))
    return ϕ
end

"""
    make_numerical(B, pomdp)

Helper function that converts a set of beliefs `B` into a numerical matrix representation suitable for processing by numerical algorithms/compressors.

# Arguments
- `B::Vector{<:Any}`: A vector of beliefs.
- `pomdp::POMDP`: The POMDP model associated with the beliefs.

# Returns
- `Matrix{Float64}`: A matrix where each row corresponds to a numerical representation of a belief in `B`.

# Example Usage
```julia
B = [belief1, belief2, belief3]
B_numerical = make_numerical(B, pomdp)
```
"""
function make_numerical(B, pomdp)
    B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
    return B_numerical
end

"""
    compress_POMDP(pomdp, sampler, updater, compressor)

Creates a compressed belief-state MDP by sampling, compressing, and caching beliefs from the given POMDP.

# Arguments
- `pomdp::POMDP`: The POMDP model to be compressed.
- `sampler::Sampler`: A sampler to generate a set of beliefs from the POMDP.
- `updater::Updater`: An updater to initialize beliefs from states.
- `compressor::Compressor`: A compressor to reduce the dimensionality of the beliefs.

# Returns
- `CompressedBeliefMDP`: The constructed compressed belief-state MDP.
- `Matrix{Float64}`: A matrix where each row corresponds to the compressed representation of the sampled beliefs.

# Example Usage
```julia
pomdp = TigerPOMDP()
sampler = BeliefExpansionSampler(pomdp)
updater = DiscreteUpdater(pomdp)
compressor = PCACompressor(2)
m, B̃ = compress_POMDP(pomdp, sampler, updater, compressor)
"""
function compress_POMDP(
    pomdp::POMDP, 
    sampler::Sampler, 
    updater::Updater, 
    compressor::Compressor
)
    # sample beliefs
    B = sampler(pomdp)

    # compress beliefs and cache mapping
    B_numerical = make_numerical(B, pomdp)
    fit!(compressor, B_numerical)
    B̃ = compressor(B_numerical)
    ϕ = make_cache(B, B̃)

    # construct the compressed belief-state MDP
    m = CompressedBeliefMDP(pomdp, updater, compressor)
    merge!(m.ϕ, ϕ)  # update the compression cache

    return m, B̃
end