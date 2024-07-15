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
date: 13 April 2024
bibliography: paper.bib
---

# Summary

Partially observable Markov decision processes (POMDPs) are a standard mathematical model for sequential decision making under state and outcome uncertainty [@AFDM]. They commonly feature in reinforcement learning research and have applications spanning medicine [@drugs], sustainability [@carbon], and aerospace [@planes]. Unfortunately, real-world POMDPs often require bespoke solutions since they are too large to be tractable with traditional methods [@complexity1; @complexity2]. Belief compression [@Roy] is a general-purpose technique that focuses planning on relevant belief states, thereby making it feasible to solve complex, real-world POMDPs more efficiently.

# Statement of Need

## Research Purpose

CompressedBeliefMDPs.jl is a Julia package [@Julia] for solving large POMDPs in the POMDPs.jl ecosystem [@POMDPs.jl] with belief compression. It offers a simple interface for effeciently sampling and compressing beliefs and for constructing and solving belief-state MDPs. The package can be used to solve complex POMDPs. It can also be used to benchmark sampling, compressing (dimensionality reduction), and solving techniques.

## Relation to Prior Work

### Other Methods for Solving Large POMDPs

While traditional tabular methods like policy and value iteration scale poorly on real-world POMDPs, there are many modern techniques that are effective at solving large-scale POMDPs like point-based methods [@PBVI; @perseus; @hsvi; @SARSOP] and online planners [@AEMS; @despot; @mcts; @pomcp; @sunberg2018online]. Belief compression is an effective but often overlooked technique that finds an effecient belief representation during planning. One of the primary goals of CompressedBeliefMDPs.jl is to make benchmarking and exploring belief compression easier, so that it can be studied further.

### Belief Compression

CompressedBeliefMDPs.jl abstracts the belief compression algorithm of @Roy into four basic steps:

1. sample reachable beliefs,
2. compress the samples,
3. construct the compressed belief-state MDP, and
4. solve using an MDP solver.

In our package, each step is handled by a struct or abstract type. Step (1) is handled by the `Sampler` abstract type; step (2) by the `Compressor` abstract type; step (3) by the `CompressedBeliefMDP` struct; and step (4) by the `CompressedBeliefSolver` and `CompressedBeliefPolicy` structs. In contrast, @Roy use a specific sampler, compressor, and solver. They use a heuristic controller for sampling beliefs; exponential family principal component analysis with Poisson loss for compression [@EPCA]; and local approximation value iteration for the base solver. @Roy does not explore how belief compression might generalize with alternative techniques.

### Related Packages

As far as we are aware, no prior Julia or Python package implements POMDP belief compression, though there is a similar package in MATLAB [@epca-MATLAB] that focuses specifically on Poisson exponential family principal component analysis. 
 
# Sampling

The `Sampler` abstract type handles sampling in CompressedBeliefMDPs.jl. CompressedBeliefMDPs.jl supports sampling with policy rollouts through `PolicySampler` and `ExplorationSampler` which wrap `Policy` and `ExplorationPolicy` from POMDPs.jl respectively. You can use these objects if you want to collect beliefs with a random or $\epsilon$-greedy policy for instance. 

CompressedBeliefMDPs.jl also supports fast exploratory belief expansion on POMDPs with discrete state, action, and observation spaces. Our implementation is an adaptation of Algorithm 21.13 in @AFDM. We use $k$-d trees [@kd-trees] to efficiently find the furthest belief sample.

# Compression

The `Compressor` abstract type handles compression in CompressedBeliefMDPs.jl. CompressedBeliefMDPs.jl provides seven off-the-shelf compressors:

1. principal component analysis (PCA) [@PCA],
2. kernel PCA [@kernelPCA],
3. probabilistic PCA [@PPCA],
4. factor analysis [@factor],
5. Isomap [@isomap],
6. autoencoder [@autoencoder], and
7. variational auto-encoder (VAE) [@VAE].

The first four are supported through [MultivariateState.jl](https://juliastats.org/MultivariateStats.jl/stable/); Isomap is supported through [ManifoldLearning.jl](https://wildart.github.io/ManifoldLearning.jl/stable/); and the last two are implemented in Flux.jl [@flux].

# Compressed Belief-State MDPs

## Definition

Before we discuss *compressed* belief-state MDPs, it is important to understand vanilla belief-state MDPs.

Any POMDP can be viewed as a belief-state MDP [@belief-state-MDP] where each state is a belief and transitions are defined with belief updates (e.g., Bayesian or Kalman filters). Formally, a POMDP $\langle S, A, T, R, \Omega, O, \gamma \rangle$—with $S$ being the state space, $A$ the action space, $T: S \times A \times S \to \mathbb{R}$ the transition model, $R: S \times A \to \mathbb{R}$ the reward moel, $\Omega$ the observation space, $O: \Omega \times S \times A \to \mathbb{R}$ the observation model, and $\gamma \in [0, 1)$ the discount factor—is said to induce the belief-state MDP $\langle B, A, T', R', \gamma \rangle$ where $B$ is the POMDP belief space, $T': B \times A \times B \to \mathbb{R}$ the belief update model, and $R': B \times A \to \mathbb{R}$ the reward model ($A$ and $\gamma$ are identical).

We define the corresponding *compressed belief-state MDP* as $\langle \tilde{B}, A, \tilde{T}, \tilde{R}, \gamma \rangle$ where $\tilde{B}$ is the compressed belief space obtained from the compression $\phi: B \to \tilde{B}$. Then $\tilde{R}(\tilde{b}, a) = R(\phi^{-1}(\tilde{b}), a)$ and $\tilde{T}(\tilde{b}, a, \tilde{b}') = T(\phi^{-1}(\tilde{b}), a, \phi^{-1}(\tilde{b}'))$. When $\phi$ is lossy, $\phi$ may not be invertible. In practice, we circumvent this issue by caching compressions on a first-come-first-serve basis (or under an arbitrary ranking over $B$ if the compression is parallel), so that if $\phi(b_1) = \phi(b_2) = \tilde{b}$ we have $\phi^{-1}(\tilde{b}) = b_1$ if $b_1$ was ranked higher than $b_2$ for $b_1, b_2 \in B$ and $\tilde{b} \in \tilde{B}$.

## Implementation

We implement compressed belief MDPs with the `CompressedBeliefMDP` struct. `CompressedBeliefMDP` contains a [`GenerativeBeliefMDP`](https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/model/#POMDPTools.ModelTools.GenerativeBeliefMDP), a `Compressor`, and a cache `ϕ` that recovers the original belief. The default construct handles the `GenerativeBeliefMDP` and `ϕ` creation, so in practice `CompressedBeliefMDP`s can be easily constructed with only a POMDPs.jl `POMDP`, `Updater`, and CompressedBeliefMDPs.jl `Compressor`.

```julia
using POMDPs, POMDPModels, POMDPTools
using CompressedBeliefMDPs

n_components = 1
my_compressor = PCACompressor(n_components)
# some code to fit! the compressor
cbmdp = CompressedBeliefMDP(BabyPOMDP(), DiscreteUpdater(), my_compressor)
```

# Solvers

`CompressedBeliefMDP` can be solved by any POMDPs.jl MDP solver.

```julia
solver = MyMDPSolver()::POMDPs.Solver
policy = solve(solver, cbmdp)
```

For convenience, we also provide `CompressedBeliefSolver`and `CompressedBeliefPolicy` which wraps the entire belief compression pipeline including sampling beliefs and fitting the compressor.

```julia
using POMDPs, POMDPModels, POMDPTools
using CompressedBeliefMDPs

pomdp = BabyPOMDP()
base_solver = MyMDPSolver()
solver = CompressedBeliefSolver(
  pomdp,
  base_solver;
  updater=DiscreteUpdater(pomdp),
  sampler=BeliefExpansionSampler(pomdp),
  compressor=PCACompressor(1),
)
policy = POMDPs.solve(solver, pomdp)  # CompressedBeliefPolicy
s = initialstate(pomdp)
v = value(policy, s)
a = action(policy, s)
```

Following @Roy, we use local value approximation as our default base solver because it provides an error bound on our value estimate [@error_bound].[^1]

```julia
using POMDPs, POMDPTools, POMDPModels
using CompressedBeliefMDPs

pomdp = BabyPOMDP()
solver = CompressedBeliefSolver(pomdp)
policy = solve(solver, pomdp)
```

The generality of the base solver in CompressedBeliefMDPs.jl offers a major improvement over the belief compression of @Roy because it supports continuous state, action, and observation spaces. More details, examples, and instructions on implementing custom components can be found in the [documentation](https://juliapomdp.github.io/CompressedBeliefMDPs.jl/dev/).

[^1]: The value function is no longer guaranteed to be convex over the compressed belief space $\tilde{B}$.

# Circular Maze

CompressedBeliefMDPs.jl also includes the Circular Maze POMDP from @Roy and scripts to recreate figures from the original paper. More details can be found in the [documentation](https://juliapomdp.github.io/CompressedBeliefMDPs.jl/dev/).

```julia
using CompressedBeliefMDPs

n_corridors = 2
corridor_length = 100
pomdp = CircularMaze(n_corridors, corridor_length)
```

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, Dylan Asmar, and Zachary Sunberg for their help and guidance.

# References