# using POMDPModels
# using POMDPTools
# using Test

# let
#     pomdp = TigerPOMDP()

#     pomdp2 = TabularPOMDP(T, R, O, 0.95)

#     policy = RandomPolicy(pomdp, rng=MersenneTwister(2))
#     sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)

#     simulate(sim, pomdp1, policy, updater(policy), initialstate(pomdp1))

#     o = last(observations(pomdp1))
#     @test o == 1
#     # test vec
#     ov = convert_o(Array{Float64}, true, pomdp1)
#     @test ov == [1.]
#     o = convert_o(Bool, ov, pomdp1)
#     @test o == true

#     @test has_consistent_distributions(pomdp)

#     @test reward(pomdp, TIGER_LEFT, TIGER_OPEN_LEFT) == pomdp1.r_findtiger
#     @test reward(pomdp1, TIGER_LEFT, TIGER_OPEN_RIGHT) == pomdp1.r_escapetiger
#     @test reward(pomdp1, TIGER_RIGHT, TIGER_OPEN_RIGHT) == pomdp1.r_findtiger
#     @test reward(pomdp1, TIGER_RIGHT, TIGER_OPEN_LEFT) == pomdp1.r_escapetiger
#     @test reward(pomdp1, TIGER_RIGHT, TIGER_LISTEN) == pomdp1.r_listen

#     for s in states(pomdp1)
#         @test pdf(transition(pomdp1, s, TIGER_LISTEN), s) == 1.0
#         @test pdf(transition(pomdp1, s, TIGER_OPEN_LEFT), s) == 0.5
#         @test pdf(transition(pomdp1, s, TIGER_OPEN_RIGHT), s) == 0.5
#     end

#     for s in states(pomdp1)
#         @test pdf(observation(pomdp1, TIGER_LISTEN, s), s) == pomdp1.p_listen_correctly
#         @test pdf(observation(pomdp1, TIGER_OPEN_LEFT, s), s) == 0.5
#         @test pdf(observation(pomdp1, TIGER_OPEN_RIGHT, s), s) == 0.5
#     end
# end