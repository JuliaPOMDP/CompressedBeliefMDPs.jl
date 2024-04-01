function test_compressor(pomdp::POMDP, sampler::Sampler, compressor::Compressor)
    solver = CompressedBeliefSolver(pomdp; compressor=compressor, sampler=sampler)
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
)

FLUX_COMPRESSORS = (
    AutoencoderCompressor,
)

@testset "Compressor Tests" begin
    pomdp = BabyPOMDP()
    sampler = BeliefExpansionSampler(pomdp; n=5)
    @testset "MultivariateStats Compressors" begin
        @testset "$C" for C in MVS_COMPRESSORS
            @test_nowarn test_compressor(pomdp, sampler, C(1))
            @test_nowarn test_compressor(pomdp, sampler, C(2))
        end
    end
    @testset "ManifoldLearning Compressors" begin
        @testset "$C" for C in MANIFOLD_COMPRESSORS
            @test_nowarn test_compressor(pomdp, sampler, C(1; k=5))
            @test_nowarn test_compressor(pomdp, sampler, C(2; k=5))
        end
    end
    @testset "Flux Compressors" begin
        @testset "$C" for C in FLUX_COMPRESSORS
            @test_nowarn test_compressor(pomdp, sampler, C(2, 1))
            @test_nowarn test_compressor(pomdp, sampler, C(2, 2))
        end
    end
end
