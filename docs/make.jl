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
        "Samplers" => "samplers.md",
        "Compressors" => "compressors.md",
        "CompressedBeliefMDP" => "cbmdp.md",
        "Base Solvers" => "solvers.md",
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
