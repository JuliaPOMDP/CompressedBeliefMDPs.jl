using Flux


"
Adapted from: 
- https://github.com/FlyingWorkshop/DiffusionGNNTutorial
- https://github.com/FluxML/model-zoo/tree/master/vision/vae_mnist
"


struct Encoder
    linear
    μ
    logσ
end

Flux.@layer Encoder

Encoder(input_dim::Integer, latent_dim::Integer, hidden_dim::Integer) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
) |> f64

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Integer, latent_dim::Integer, hidden_dim::Integer) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
) |> f64

function reconstuct(encoder, decoder, x)
    μ, logσ = encoder(x)
    z = μ + randn(size(logσ)...) .* exp.(logσ)
    return μ, logσ, decoder(z)
end

function model_loss(encoder, decoder, x)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x)
    kl_q_p = 0.5f0 * sum(@. (exp(2logσ) + μ^2 - 1 - 2logσ))
    logp_x_z = -Flux.logitbinarycrossentropy(decoder_z, x, agg=sum)
    return -logp_x_z + kl_q_p
end

struct VAECompressor <: Compressor
    encoder
    decoder
    optimizer
    epochs
    verbose
end


"""
Implements a [VAE](https://arxiv.org/abs/1312.6114) in Flux.
"""
function VAECompressor(input_dim::Integer, latent_dim::Integer; hidden_dim::Integer=2, optimizer=Adam(), epochs::Integer=10, verbose=false)
    @assert input_dim > 0 "Input dimension must be a positive integer."
    @assert latent_dim > 0 "Latent dimension must be a positive integer."
    @assert hidden_dim > 0 "Hidden dimension must be a positive integer."
    @assert epochs > 0 "Number of epochs must be a positive integer."
    
    encoder = Encoder(input_dim, latent_dim, hidden_dim)
    decoder = Decoder(input_dim, latent_dim, hidden_dim)
    VAECompressor(encoder, decoder, optimizer, epochs, verbose)
end

function fit!(c::VAECompressor, beliefs)
    encoder, decoder = c.encoder, c.decoder
    opt_enc = Flux.setup(c.optimizer, encoder)
    opt_dec = Flux.setup(c.optimizer, decoder)

    if c.verbose
        println("Start Training, total $(c.epochs) epochs")
    end

    for epoch = 1:c.epochs
        if c.verbose
            println("Epoch $(epoch)")
        end

        for b in eachrow(beliefs)
            loss, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                model_loss(enc, dec, b)
            end

            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)

            # progress meter
            if c.verbose
                @show loss
            end
        end
    end
end

function (c::VAECompressor)(beliefs)
    if ndims(beliefs) == 2
        B̃ = c.encoder(beliefs')[1]'
    else
        B̃ = c.encoder(beliefs)[1]
    end
    return B̃
end