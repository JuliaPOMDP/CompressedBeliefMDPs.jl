using Documenter
using CompressedBeliefMDPs

makedocs(
    sitename = "CompressedBeliefMDPs",
    format = Documenter.HTML(),
    modules = [CompressedBeliefMDPs]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/FlyingWorkshop/CompressedBeliefMDPs.jl.git"
)
