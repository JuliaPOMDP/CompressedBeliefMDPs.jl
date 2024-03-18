---
title: 'CompressedBeliefMDPs: A Julia Package for Solving Large POMDPs'
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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

CompressedBeliefMDPs.jl is a Julia package for solving large partially observable Markov decision processes (POMDPs). It's a part of the widely-used JuliaPOMDPs ecosystem and generalizes the belief compression model presented in @Roy.

# Statement of need

POMDPs are a common framework in reinforcement learning and decision making under uncertainty, with applications across medicine, sustainability, finance, aerospace, and more. Unfortunately, solving real-world POMDPs is often computationally intractable since the complexity of traditional methods scales exponentially with the size of the state space. One method for tackling the so-called "curse of dimensionality" is belief compression. By compressing the belief distribution (i.e., the probability that the POMDP solver is in any given state at a given time), we can focus computation on the relevant belief states and solve large POMDPs. Belief compression is particularly useful when a POMDP has many states, but uncertainty is concentrated or sparse.

CompressedBeliefMDPs.jl provides an interface that connects the JuliaPOMDPs ecosystem with the arsenal of dimensionality reduction techniques in the wider Julia community. In particular, it exports a `CompressedBeliefMDP` and `CompressedSolver` that generalize the methods in @Roy.


# Example

Using CompressedBeliefMDPs.jl is simple.
```julia
using POMDPs
using POMDPModels
using CompressedBeliefMDPs

pomdp = TigerPOMDP()
sampler = DiscreteRandomSampler(pomdp)
compressor = PCACompressor(2)
solver = CompressedSolver(pomdp, sampler, compressor)
policy = POMDPs.solve(solver, pomdp)
```

# ExpFamilyPCA.jl

Another package 

# Acknowledgements

We thank Arec Jamgochian and Robert Moss for their advice.

# References