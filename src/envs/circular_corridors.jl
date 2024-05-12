struct CircularCorridorsState
    is_top::Bool
    index::Integer
end


struct CircularCorridorsPOMDP <: POMDP{CircularCorridorsState, Integer, Integer}
    top_goal::CircularCorridorsState
    bot_goal::CircularCorridorsState
end


# actions
const LEFT = 0
const RIGHT = 1
const SENSE_CORRIDOR = 2
const DECLARE_GOAL = 3


function states(p::CircularCorridorsPOMDP)
    return nothing
end


function stateindex(p::CircularCorridorsPOMDP, s::CircularCorridorsState)
    return nothing
end


# function observation(p::CircularCorridorsPOMDP, a::Integer, sp::CircularCorridorsState)
#     if a == SENSE_CORRIDOR
#         if sp.is_top
#             return sp === p.top_goal
#         else
#             return sp === p.bot_goal
#         end
#     else
#         # TODO: von Mises distribution
#     end
# end



