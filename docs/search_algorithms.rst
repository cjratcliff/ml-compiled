"""""""""""""""
Search
"""""""""""""""
In the context of reinforcement learning and most commonly in games, search refers to trying to find the value of an action in a particular state by looking ahead into the future, imagining possible moves and countermoves. Search is also sometimes referred to as lookahead search.

Alpha-beta pruning
-------------------
A technique to reduce the number of nodes that need to be evaluated when doing search.

Minimax algorithm
--------------------
Recursive algorithm for computing the value of a state in a two-player zero-sum game.

It models the players in this way:

1. It (the first player) always picks the move that will lead to the state that maximizes its value function.
2. Its opponent always picks the move that will lead to the state which minimizes the value function.

It alternately picks and evaluates moves for itself and its opponent up to some maximum depth using depth-first search.

Rollout
---------
A simulation of a possible future game trajectory.

Monte Carlo rollout
______________________
Searches to maximum depth without branching by sampling long sequences of actions with a policy. Can average over these to achieve super-human performance in backgammon and Scrabble.

Monte Carlo Tree Search
------------------------
Uses Monte Carlo rollouts to estimate the value of each state in a search tree in order to improve a policy. The policy and value networks will be evaluated multiple times in each branch of the tree search. Converges to optimal play.
