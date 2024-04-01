@testset "Solver/Sampler Tests" begin
    compressor = PCACompressor(1)
    @testset "Discrete S, A, O" begin
        @testset "$pomdp" for pomdp in (BabyPOMDP(), TigerPOMDP())  # TODO: include TMaze once it works
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=PolicySampler(pomdp)), pomdp)
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=ExplorationPolicySampler(pomdp)), pomdp)
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=BeliefExpansionSampler(pomdp)), pomdp)
        end
    end
    
    @testset "Continuous S, A" begin
        @testset "$pomdp" for pomdp in (LightDark1D(),)
            # NOTE: have to use a different base solver since LocalApproximationValueIterationSolver only supports discrete S, A
            # TODO: check if MCTS is appropriate for belief compression
            base_solver = MCTSSolver(n_iterations=10, depth=5, exploration_constant=5.0)
            updater = BootstrapFilter(pomdp, 100)
            solver = CompressedBeliefSolver(pomdp, base_solver; sampler=PolicySampler(pomdp; updater=updater), updater=updater)
            @test_nowarn test_solver(solver, pomdp)
        end 
    end
end