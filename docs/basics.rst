"""""""""""
Basics
"""""""""""

Absorbing state
----------------
Terminal state.

Action space
--------------
The space of all possible actions. May be discrete as in Chess or continuous as in many robotics tasks.

Behaviour distribution
-----------------------
The probability distribution over sequences of states and actions.

Bellman equation
------------------
Computes the value of a state given a policy. Represents the intuition that if the value at the next timestep is known for all possible actions, the optimal strategy is to select the action that maximizes that value plus the immediate reward.

.. math::

    Q^*(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q^*(s',a')|s,a]

Breadth
---------
In the context of games with a discrete action space like Chess and Go, breadth is the average number of possible moves.

Control policy
---------------
See policy.

Credit assignment problem
---------------------------
The problem of not knowing which actions helped and which hindered in getting to a particular reward.

Depth
-----------
Length of the game on average.

Discount factor
----------------
Between 0 and 1. Values closer to 0 make the agent concentrate on short-term rewards.

Episode
------------
Analogous to a game. Ends when a terminal state is reached or after a predetermined number of steps.

Policy
----------
A function, :math:`\pi` that maps states to actions.

Regret
-------
The difference in the cumulative reward between performing optimally and executing the given policy.

Reward function
------------------
Maps state-action pairs to rewards.

Trajectory
--------------
The sequence of states and actions experienced by the agent.

Transition function
---------------------
Maps a state and an action to a new state.

Value function
----------------
The value of a state is equal to the expectation of the reward function given the state and the policy. 

.. math::

    V(s) = E[R|s,\pi]

