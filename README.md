# CompressedBeliefMDPs

[![Build Status](https://github.com/JuliaPOMDP/CompressedBeliefMDPs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPOMDP/CompressedBeliefMDPs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev-Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaPOMDP.github.io/CompressedBeliefMDPs.jl/dev/)
[![codecov](https://codecov.io/gh/JuliaPOMDP/CompressedBeliefMDPs.jl/graph/badge.svg?token=FXmEi9Fscd)](https://codecov.io/gh/JuliaPOMDP/CompressedBeliefMDPs.jl)
[![status](https://joss.theoj.org/papers/e47a49f030233de4fbada2c389de9b34/status.svg)](https://joss.theoj.org/papers/e47a49f030233de4fbada2c389de9b34)

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

### Continuous Example

This example demonstrates using CompressedBeliefMDP in a continuous setting with the `LightDark1D` POMDP. It combines particle filters for belief updating and Monte Carlo Tree Search (MCTS) as the solver. While compressing a 1D space is trivial toy problem, this architecture can be easily scaled to larger POMDPs with continuous state and action spaces.

```julia
using POMDPs, POMDPModels, POMDPTools
using ParticleFilters
using MCTS
using CompressedBeliefMDPs

pomdp = LightDark1D()
pomdp.movement_cost = 1
base_solver = MCTSSolver(n_iterations=10, depth=50, exploration_constant=5.0)
updater = BootstrapFilter(pomdp, 100)
solver = CompressedBeliefSolver(
    pomdp,
    base_solver;
    updater=updater,
    sampler=PolicySampler(pomdp; updater=updater)
)
policy = solve(solver, pomdp)
rs = RolloutSimulator(max_steps=50)
r = simulate(rs, pomdp, policy)
```

### Large Example

In this example, we tackle a more realistic scenario with the TMaze POMDP, which has 123 states. To handle the larger state space efficiently, we employ a variational auto-encoder (VAE) to compress the belief simplex. By leveraging the VAE's ability to learn a compact representation of the belief state, we focus computational power on the relevant (compressed) belief states during each Bellman update.

```julia
using POMDPs, POMDPModels, POMDPTools
using CompressedBeliefMDPs

pomdp = TMaze(60, 0.9)
solver = CompressedBeliefSolver(
    pomdp;
    compressor=VAECompressor(123, 6; hidden_dim=10, verbose=true, epochs=2),
    sampler=PolicySampler(pomdp, n=500),
    verbose=true, 
    max_iterations=1000, 
    n_generative_samples=30,
    k=2
)
policy = solve(solver, pomdp)
rs = RolloutSimulator(max_steps=50)
r = simulate(rs, pomdp, policy)
```

