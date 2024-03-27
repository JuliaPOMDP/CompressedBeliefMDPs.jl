# CompressedBeliefMDPs

[![Build Status](https://github.com/FlyingWorkshop/CompressedBeliefMDPs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FlyingWorkshop/CompressedBeliefMDPs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev-Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://flyingworkshop.github.io/CompressedBeliefMDPs.jl/dev/)

CompressedBeliefMDP.jl provides a simple interface for solving large POMDPs with sparse belief spaces.

# Installation
```
add CompressedBeliefMDPs
```

# Quickstart

CompressedBeliefMDPs.jl is compatible with the [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/) ecosystem.
```julia
using POMDPs, POMDPModels
using CompressedBeliefMDPs

pomdp = BabyPOMDP()
solver = CompressedBeliefSolver(pomdp)
policy = POMDPs.solve(solver, pomdp)
s = initialstate(pomdp)
v = value(policy, s)
a = action(policy, s)
```

The solver finds an _approximate_ policy for the POMDP.

```julia
v = value(policy, s)
a = action(policy, s)
```
# Sampling

There are two ways to collect belief samples: belief expansion or policy rollouts.

## Belief Expansion

CompressedBeliefMDPs.jl implements a fast version of exploratory belief expansion (Algorithm 21.13 from [Algorithms for Decision Making](https://algorithmsbook.com/)) that uses [$k$-d trees](https://en.wikipedia.org/wiki/K-d_tree) from [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl). Belief expansion is supported for POMDPs with finite state, action, and observation spaces.

## Policy Rollouts 



# Compressors

As a convenience, we provide several wrappers for compression schemes from [MultivariateStats.jl](https://juliastats.org/MultivariateStats.jl/stable/) and [ManifoldLearning.jl](https://wildart.github.io/ManifoldLearning.jl/stable/).