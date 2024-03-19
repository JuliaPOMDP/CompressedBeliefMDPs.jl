---
title: 'CompressedBeliefMDPs.jl: A Julia Package for Solving Large POMDPs with Belief Compression'
tags:
  - POMDP
  - MDP
  - Julia
  - sequential decision making
  - RL
  - compression
  - dimensionality reduction
  - open-source
authors:
  - name: Logan Mondal Bhamidipaty
    orcid: 0009-0001-3978-9462
    affiliation: 1
  - name: Mykel J. Kochenderfer
    orcid: 0000-0002-7238-9663
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 16 March 2024
bibliography: paper.bib
---

# Summary

A standard mathematical framework for specifying a sequential decision problem with state and outcome uncertainty is the partially observable Markov decision process (POMDP) [@AFDM], with applications to areas such as medicine [@drugs], sustainability [@carbon], and aerospace [@planes]. Unfortunately, solving real-world POMDPs with traditional methods is often computationally intractable [@complexity1; @complexity2]. Belief compression [@Roy] is a powerful technique for overcoming this limitation that's particularly effective when uncertainty is sparse or concentrated.

# Statement of Need

CompressedBeliefMDPs.jl is a Julia package [@Julia] for solving large POMDPs in the POMDPs.jl ecosystem [@POMDPs.jl] with belief compression [@Roy]. CompressedBeliefMDPs.jl provides an interface that generalizes the belief compression algorithm presented in @Roy. In particular, while @Roy describe belief compression using Poisson exponential-family principal component analysis (PCA), CompressedBeliefMDPs.jl supports arbitrary compression techniques and function approximators. This flexibility enables the development and application of new, more powerful compression approaches as research in this area continues to evolve.

As far as we are aware, no prior Julia or Python package implements POMDP belief compression, though there is a similar package in MATLAB [@epca-MATLAB].

# Belief Compression

The belief compression algorithm in @Roy can be generalized into the following steps:

1. sample reachable beliefs;
2. compress the beliefs;
3. create a function approximator for the compressed beliefs;
4. create the compressed belief-state Markov decision process (MDP);
5. and, solve the MDP with local approximation value iteration.

For steps 1. and 2., CompressedBeliefMDPS.jl defines two abstract types `Sampler` and `Compressor`. For step 3., we use the `LocalFunctionApproximator` abstract type from [LocalApproximationValueIteration.jl](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl). Note that we need function approximation because it gives us an error bound on the estimated value function: our estimate is no longer guaranteed to converge to the optimum since the value function is not necessarily convex over the compressed belief simplex. As a convenience, CompressedBeliefMDPs.jl defines several concrete subtypes which we describe later. 

For step 4., we define a new generative `POMDP` type called `CompressedBeliefMDP` that wraps [`GenerativeBeliefMDP`](https://juliapomdp.github.io/POMDPModelTools.jl/stable/model_transformations/#Generative-Belief-MDP). While @Roy builds the compressed belief-state MDP directly from the interpolated values, CompressedBeliefMDPs.jl delegates local approximation to the solver. This makes it easier to benchmark different approximators against the same compressed belief-state MDP. Finally, for step 5., we define `CompressedSolver <: POMDPs.Solver` that wraps the entire belief compression pipeline.

# Example

Using CompressedBeliefMDPs.jl is simple.

```julia
using POMDPs, POMDPModels
using CompressedBeliefMDPs

pomdp = BabyPOMDP()
sampler = DiscreteRandomSampler(pomdp)
compressor = PCACompressor(2)
approx_solver = CompressedSolver(pomdp, sampler, compressor)
approx_policy = POMDPs.solve(approx_solver, pomdp)
s = initialstate(pomdp)
v = value(approx_policy, s)
a = action(approx_policy, s)
```

![We see that that the compressed solver recovers a similar value function to SARSOP [@SARSOP].](./images/baby_benchmark.svg){height="200pt"}

## Function Approximators

CompressedBeliefMDPs.jl is compatible with any `LocalFunctionApproximator`. It supports grid interpolations [@grid] through [GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl) and $k$-nearest neighbors [@kNN] through [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl). For more details, see [LocalFunctionApproximation.jl](https://github.com/sisl/LocalFunctionApproximation.jl)

## Compressors

CompressedBeliefMDPs.jl provides several wrappers for commonly used compressors. Through [MultiVariateStats.jl](https://juliastats.org/MultivariateStats.jl/stable/), we include PCA [@PCA], kernel PCA [@kernelPCA], and probabilistic PCA [@PPCA]. Through a forthcoming package `ExpFamilyPCA.jl`, we also include exponential family PCA [@EPCA]. This is more general than the Poisson exponential family PCA in @Roy: ExpFamilyPCA.jl supports compression objectives induced from _any_ convex function, not just the Poisson link function.

# Acknowledgements

We thank Arec Jamgochian and Robert Moss for their advice.

# References