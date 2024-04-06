using Documenter
using CompressedBeliefMDPs

makedocs(
    sitename = "CompressedBeliefMDPs",
    format = Documenter.HTML(),
    modules = [CompressedBeliefMDPs]
)

makedocs(
    sitename = "CompressedBeliefMDPs",
    checkdocs = :exports,
    format = Documenter.HTML(),
    modules = [CompressedBeliefMDPs],
    pages = [
        "CompressedBeliefMDPs.jl" => "index.md",
        "Concepts and Architecture" => "concepts.md",
        "Defining a Sampler" => "sampler_interface.md",
        "Implemented Samplers" => "samplers.md",
        "Defining a Compressor" => "compressor_interface.md",
        "Implemented Compressors" => "compressors.md",
        "API Documentation" => "api.md"
        # "Subsection" => [
        #     ...
        # ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
deploydocs(
    repo = "github.com/FlyingWorkshop/CompressedBeliefMDPs.jl.git",
)
