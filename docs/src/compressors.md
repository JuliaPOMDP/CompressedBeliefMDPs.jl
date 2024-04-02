# Implemented Compressors

CompressedBeliefMDPs currently provides wrappers for the following compression types:
- a principal component analysis (PCA) compressor,
- a kernel PCA compressor,
- a probabilistic PCA compressor,
- a factor analysis compressor,
- an isomap compressor,
- an autoencoder compressor
- a variational auto-encoder (VAE) compressor

## Principal Component Analysis (PCA)
```@docs 
PCACompressor
```

## Kernel PCA
```@docs 
KernelPCACompressor
```

## Probabilistic PCA
```@docs 
PPCACompressor
```

## Factor Analysis
```@docs 
FactorAnalysisCompressor
```

## Isomap
```@docs
IsomapCompressor
```

## Autoencoder
```@docs
AutoencoderCompressor
```

### Variational Auto-Encoder (VAE)
```@docs
VAECompressor
```

!!! warning 
    Some compression algorithms aren't optimized for large belief spaces. While they pass our unit tests, they may fail on large POMDPs or without seeding. For large POMDPs, users may want a custom `Compressor`.