using Flux


struct AutoencoderCompressor <: Compressor
    encoder
    model
    optimizer
    epochs
end

"""
Implements an autoencoder in Flux.
"""
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
    for _ in 1:c.epochs
        Flux.train!(loss, c.model, data, opt_state)
    end
end

function (c::AutoencoderCompressor)(beliefs)
    if ndims(beliefs) == 2
        result = c.encoder(beliefs')'
    else
        result = c.encoder(beliefs)
    end
    return result
end
