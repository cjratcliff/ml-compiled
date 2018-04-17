"""""""""""""""
Search
"""""""""""""""
In the context of reinforcement learning and most commonly in games, search refers to trying to find the value of an action in a particular state by looking ahead into the future, simulating possible countermoves.

Alpha-beta pruning
-------------------
TO DO

Rollout
---------
A simulation of a possible future game trajectory.

Monte Carlo rollout
______________________
Searches to maximum depth without branching by sampling long sequences of actions with a policy. Can average over these to achieve super-human performance in backgammon and Scrabble.

Monte Carlo Tree Search
------------------------
Uses Monte Carlo rollouts to estimate the value of each state in a search tree in order to improve a policy. The policy and value networks will be evaluated multiple times in each branch of the tree search. Converges to optimal play.
