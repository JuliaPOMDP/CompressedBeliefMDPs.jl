---
title: 'CompressedBeliefMDPs.jl: A Julia Package for Solving Large POMDPs with Belief Compression'
tags:
  - POMDP
  - MDP
  - Julia
  - sequential decision making
  - RL
  - compression
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

Partially observable Markov decision processes (POMDPs) are a common framework in reinforcement learning and decision making under uncertainty with applications across medicine [@drugs], sustainability [@carbon], economics [@markets], aerospace [@planes], and more. Unfortunately, solving real-world POMDPs with traditional methods is often computationally intractable due to the so-called "curse of dimensionality" [@AFDM]. Belief compression [@Roy] is a powerful technique for overcoming this curse that's particular effective when uncertainty is sparse or concentrated.

# Statement of need

CompressedBeliefMDPs.jl is a Julia package [@Julia] for solving large POMDPs in the JuliaPOMDPs ecosystem [@POMDPs.jl] with belief compression [@Roy]. CompressedBeliefMDPs.jl exports an interface that generalizes the belief compression algorithm presented in @Roy. In particular, while @Roy describe belief compression using Poisson exponential-family PCA, CompressedBeliefMDPs.jl supports arbitrary compression techniques and function approximators. This flexibility enables development and application of new, more powerful compression approaches as research in this area continues to evolve.

As far as we are aware, no prior Julia or Python package implements POMDP belief compression, though there is a similar package in MATLAB [@epca-MATLAB].

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
```

![We see that that the compressed solver performs similarly with SARSOP [@SARSOP].](./images/baby_benchmark.svg){height="200pt"}


# ExpFamilyPCA.jl

ExpFamilyPCA.jl is a forthcoming package that complements CompressedBeliefMDPs.jl. It implements not only the Poisson exponential family PCA in @Roy, but also the more general exponential family PCA from @EPCA.

# Acknowledgements

We thank Arec Jamgochian and Robert Moss for their advice.

# References