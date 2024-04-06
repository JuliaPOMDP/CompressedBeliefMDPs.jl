# Defining a Sampler

In this section, we outline the requirements and guidelines for defining a belief `Sampler`.

## Interface

The `Sampler` interface only has one method: the [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects). For example, if you wanted to implement your own `Sampler`, you could write something like this

```julia
struct MySampler <: Compressor
    foo
    bar
end

# functor definition
function (c::MySampler)(pomdp::POMDP)
    # YOUR CODE HERE
end
```

