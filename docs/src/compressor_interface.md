# Defining a Belief Compressor

In this section, we outline the requirements and guidelines for defining a belief `Compressor`.

## Interface

The `Compressor` interface is extremely minimal. It only supports two methods: `fit!` and the associated [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects). For example, if you wanted to implement your own `Compressor`, you could write something like this

```julia
struct MyCompressor <: Compressor
    foo
    bar
end

# functor definition
function (c::MyCompressor)(beliefs)
    # YOUR CODE HERE
end

function fit!(c::MyCompressor, beliefs)
    # YOUR CODE HERE
end
```

## Implementation Tips
* For robustness, both the functor and `fit!` should be able to handle `AbstractVector` and `AbstractMatrix` inputs. 
* `fit!` is called only once after beliefs are sampled from the POMDP.
* `CompressedBeliefSolver` will attempt to convert each belief state (often of type [`DiscreteBelief`](https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/beliefs/#POMDPTools.BeliefUpdaters.DiscreteBelief)) into an `AbstractArray{Float64}` using [`convert_s`](https://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.convert_s). As a convenience, CompressedBeliefMDP implements conversions for commonly used belief types; however, if the POMDP has a custom belief state, then it is the users' responsibility to implement the appropriate conversion. See the source code for help. 