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
        "Samplers" => "samplers.md",
        "Compressors" => "compressors.md",
        "Environments" => "circular.md",
        "API Documentation" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
deploydocs(
    repo = "github.com/JuliaPOMDP/CompressedBeliefMDPs.jl.git",
)
