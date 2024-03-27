function test_compressor(C::Function, maxoutdim::Int; kwargs...)
    pomdp = BabyPOMDP()
    compressor = C(maxoutdim)
    solver = CompressedBeliefSolver(pomdp; compressor=compressor, kwargs...)
    policy = solve(solver, pomdp)
    s = initialstate(pomdp)
    _ = action(policy, s)
    _ = value(policy, s)
    return policy
end

MVS_COMPRESSORS = (
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor,
    FactorAnalysisCompressor,
    # MDSCompressor
)

MANIFOLD_COMPRESSORS = (
    IsomapCompressor,
    # LLECompressor,
    # HLLECompressor,
    # LEMCompressor,
    # LTSACompressor,
    # DiffMapCompressor
)

@testset "Compressor Tests" begin
    @testset "MultivariateStats Compressors" begin
        @testset "$C" for C in MVS_COMPRESSORS
            @test_nowarn test_compressor(C, 1; n=3)
            @test_nowarn test_compressor(C, 2; n=3)
        end
    end
    @testset "ManifoldLearning Compressors" begin
        @testset "$C" for C in MANIFOLD_COMPRESSORS
            @test_nowarn test_compressor(C, 1; expansion=false, n=50, fit_kwargs=Dict(:k => 5))
            @test_nowarn test_compressor(C, 2; expansion=false, n=50, fit_kwargs=Dict(:k => 5))
        end
    end
end
