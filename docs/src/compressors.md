# Compressors

## Defining a Belief Compressor

In this section, we outline the requirements and guidelines for defining a belief `Compressor`.

### Interface

The `Compressor` interface is extremely minimal. It only supports two methods: `fit!` and the associated [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects). For example, if you wanted to implement your own `Compressor`, you could write something like this

```julia
struct MyCompressor <: Compressor
    foo
    bar
end

# functor definition
function (c::MyCompressor)(beliefs)
    # YOUR CODE HERE
    return compressed_beliefs
end

function fit!(c::MyCompressor, beliefs)
    # YOUR CODE HERE
end
```

#### Implementation Tips
* For robustness, both the functor and `fit!` should be able to handle `AbstractVector` and `AbstractMatrix` inputs. 
* `fit!` is called only once after beliefs are sampled from the POMDP.
* `CompressedBeliefSolver` will attempt to convert each belief state (often of type [`DiscreteBelief`](https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/beliefs/#POMDPTools.BeliefUpdaters.DiscreteBelief)) into an `AbstractArray{Float64}` using [`convert_s`](https://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.convert_s). As a convenience, CompressedBeliefMDP implements conversions for commonly used belief types; however, if the POMDP has a custom belief state, then it is the users' responsibility to implement the appropriate conversion. See the source code for help. 

## Implemented Compressors

CompressedBeliefMDPs currently provides wrappers for the following compression types:
- a principal component analysis (PCA) compressor,
- a kernel PCA compressor,
- a probabilistic PCA compressor,
- a factor analysis compressor,
- an isomap compressor,
- an autoencoder compressor
- a variational auto-encoder (VAE) compressor

### Principal Component Analysis (PCA)
```@docs 
PCACompressor
```

### Kernel PCA
```@docs 
KernelPCACompressor
```

### Probabilistic PCA
```@docs 
PPCACompressor
```

### Factor Analysis
```@docs 
FactorAnalysisCompressor
```

### Isomap
```@docs
IsomapCompressor
```

### Autoencoder
```@docs
AutoencoderCompressor
```

### Variational Auto-Encoder (VAE)
```@docs
VAECompressor
```

!!! warning 
    Some compression algorithms aren't optimized for large belief spaces. While they pass our unit tests, they may fail on large POMDPs or without seeding. For large POMDPs, users may want a custom `Compressor`.