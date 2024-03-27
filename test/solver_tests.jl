@testset "Solver Tests" begin
    compressor = PCACompressor(1)
    @testset "Discrete S, A, O" begin
        @testset "$pomdp" for pomdp in (BabyPOMDP(), TigerPOMDP())  # NOTE: TMaze doesn't work b/c we don't define behavior for terminal states ourselves; this is expected behavior
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; n=5, expansion=false), pomdp)  # test POMDPs.Policy
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; n=5, expansion=false, explorer=EpsGreedyPolicy(pomdp, 0.5)), pomdp)  # test POMDPs.ExplorationPolicy
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; n=5), pomdp)  # test exploratory_belief_expansion
        end
    end
    @testset "Continuous S, A" begin
        @testset "$pomdp" for pomdp in (LightDark1D(),)
            # NOTE: have to use a different base solver since LocalApproximationValueIterationSolver only supports discrete S, A
            base_solver = MCTSSolver(n_iterations=10, depth=5, exploration_constant=5.0)
            updater = BootstrapFilter(pomdp, 100)
            solver = CompressedBeliefSolver(pomdp, base_solver; n=10, expansion=false, updater=updater)
            @test_nowarn test_solver(solver, pomdp)
        end 
    end
end