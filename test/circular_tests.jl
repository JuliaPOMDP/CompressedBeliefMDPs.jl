let
    pomdp = CircularMaze(2, 5, 0.99)

    policy = RandomPolicy(pomdp, rng=MersenneTwister(2))
    sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)

    @test has_consistent_distributions(pomdp)

    # TODO: test transitions, simulations, RNG, rewards for non goal states

    for s in pomdp.goals
        @test reward(pomdp, s, CMAZE_LEFT) == 0
        @test reward(pomdp, s, CMAZE_LEFT) == 0
        @test reward(pomdp, s, CMAZE_SENSE_CORRIDOR) == 0
        @test reward(pomdp, s, CMAZE_DECLARE_GOAL) == pomdp.r_findgoal
    end
end