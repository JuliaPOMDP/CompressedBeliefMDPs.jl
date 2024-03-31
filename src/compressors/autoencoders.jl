using Flux

struct AutoencoderCompressor <: Compressor
    encoder
    model
    optimizer
    epochs
end

function AutoencoderCompressor(input_dim::Integer, latent_dim::Integer; opt=Adam(), epochs=10)
    encoder = Dense(input_dim, latent_dim, sigmoid) |> f64
    decoder = Chain(Dense(latent_dim => input_dim), softmax)
    model = Chain(encoder, decoder) |> f64
    return AutoencoderCompressor(encoder, model, opt, epochs)
end

function fit!(c::AutoencoderCompressor, beliefs)
    opt_state = Flux.setup(c.optimizer, c.model)
    data = [(beliefs', beliefs')]
    loss(m, x, y) = Flux.kldivergence(m(x), y)
    # @showprogress for _ in 1:c.epochs
    for _ in 1:c.epochs
        Flux.train!(loss, c.model, data, opt_state)
    end
end

function (c::AutoencoderCompressor)(beliefs)
    return ndims(beliefs) == 2 ? c.encoder(beliefs')' : c.encoder(beliefs)
end


# struct VAECompressor <: Compressor
#     encoder
#     model
#     optimizer
#     epochs
# end

# # custom split layer from https://fluxml.ai/Flux.jl/dev/models/advanced/#Multiple-outputs:-a-custom-Split-layer
# struct Split{T}
#     paths::T
#   end
  
# Split(paths...) = Split(paths)
  
# Flux.@layer Split
  
# (m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


# "
# Adapted from: https://github.com/FlyingWorkshop/DiffusionGNNTutorial
# "
# function VAECompressor(input_dim::Integer, hidden_dim::Integer=1, latent_dim::Integer; opt=Adam(), epochs=10)
#     encoder = Chain(
#         Dense(input_dim => hidden_dim, relu),
#         Split(Dense(hidden_dim => latent_dim), Dense(hidden_dim => latent_dim))
#     ) |> f64

#     function model(x)
#         μ, σ = encoder(x)
#         ϵ = randn(size(μ)...)

#     end

#     encoder = Dense(input_dim, latent_dim, sigmoid) |> f64
#     decoder = Chain(Dense(latent_dim => input_dim), softmax)
#     model = Chain(encoder, decoder) |> f64
#     return AutoencoderCompressor(encoder, model, opt, epochs)
# end