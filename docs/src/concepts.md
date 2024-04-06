# Concepts and Architecture

CompressedBeliefMDPs.jl aims to implement a generalization of the [belief compression algorithm](https://papers.nips.cc/paper_files/paper/2002/hash/a11f9e533f28593768ebf87075ab34f2-Abstract.html) for solving large POMDPs. The algorithm has four steps:
1. collect belief samples,
2. compress the samples,
3. create the compressed belief-state MDP,
4. solve the MDP.

Each step is handled by `Sampler`, `Compressor`, `CompressedBeliefMDP`, and `CompressedBeliefSolver` respectively.

For more details, please see the rest of the documentation or the associated paper.