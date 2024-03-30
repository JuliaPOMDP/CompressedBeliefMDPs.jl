using Flux

struct AutoencoderCompressor <: Compressor
    encoder
    model
    optimizer
    epochs
end

function AutoencoderCompressor(input_dim::Integer, latent_dim::Integer; opt=ADAM(0.01), epochs=10)
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
    # TODO: try to remove this pattern b/c it shows up in every freakin compressor
    return ndims(beliefs) == 2 ? c.encoder(beliefs')' : c.encoder(beliefs)
end

# struct VAECompressor <: Compressor
#     latent::Integer
# end