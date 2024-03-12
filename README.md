# CompressedBeliefMDPs

[![Build Status](https://github.com/FlyingWorkshop/CompressedBeliefMDPs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FlyingWorkshop/CompressedBeliefMDPs.jl/actions/workflows/CI.yml?query=branch%3Amain)

CompressedBeliefMDP.jl provides a simple interface for solving large POMDPs with sparse belief spaces.

# Installation
```
add CompressedBeliefMDPs
```

# Quickstart

CompressedBeliefMDPs.jl is compatible with the [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/) ecosystem.
```julia
using CompressedBeliefMDPs

using POMDPs
using POMDPModels
using POMDPTools

pomdp = TMaze(200, 0.99)
sampler = DiscreteRandomSampler(pomdp)
compressor = PCACompressor(2)
solver = CompressedSolver(pomdp, sampler, compressor)
policy = solve(solver, pomdp)
```

The solver finds an _approximate_ policy for the POMDP.

```julia
v = value(policy, s)
a = action(policy, s)
```
# Sampling

Compression is handled by the `Sampler` abstract type.

# Compression

Compression is handled by the `Compressor` abstract type.

## [MultivariateStats.jl](https://juliapackages.com/p/multivariatestats) Wrappers

As a convenience, we provide several wrappers for compression schemes from MultivariateStats.jl.

# Limitations

`CompressedBeliefMDP` is built off [`GenerativeBeliefMDP`](https://juliapomdp.github.io/POMDPs.jl/stable/POMDPTools/model/#POMDPTools.ModelTools.GenerativeBeliefMDP) which means the reward returned is the reward for a _single state sampled from the belief_ rather than the expected cumulative discounted reward.