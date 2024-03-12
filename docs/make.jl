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
# ENV["GITHUB_EVENT_NAME"] = "push"
# ENV["GITHUB_REF"] = "main"
deploydocs(
    repo = "github.com/FlyingWorkshop/CompressedBeliefMDPs.jl.git",
    # devbranch = "main",
    # deploy_config = Documenter.GitHubActions()
)
