# CompressedBeliefMDPs.jl

## Introduction

Welcome to CompressedBeliefMDPs.jl! This package is part of the [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/) ecosystem and takes inspiration from [Exponential Family PCA for Belief Compression in POMDPs](https://papers.nips.cc/paper_files/paper/2002/hash/a11f9e533f28593768ebf87075ab34f2-Abstract.html). 

This package provides a general framework for applying belief compression in large POMDPs with generic compression, sampling, and planning algorithms.

## Installation

You can install CompressedBeliefMDPs.jl using Julia's package manager. Open the Julia REPL (press `]` to enter the package manager mode) and run the following command:

```julia-repl
pkg> add CompressedBeliefMDPs
```

## Quickstart

Using belief compression is easy. Simplify pick a `Sampler`, `Compressor`, and a base `Policy` and then use the standard POMDPs.jl interface.

```julia
using POMDPs, POMDPTools, POMDPModels
using CompressedBeliefMDPs

pomdp = BabyPOMDP()
compressor = PCACompressor(1)
updater = DiscreteUpdater(pomdp)
sampler = BeliefExpansionSampler(pomdp)
solver = CompressedBeliefSolver(
    pomdp;
    compressor=compressor,
    sampler=sampler,
    updater=updater,
    verbose=true, 
    max_iterations=100, 
    n_generative_samples=50, 
    k=2
)
policy = solve(solver, pomdp)
```

