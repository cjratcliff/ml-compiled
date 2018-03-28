"""""""""""
Basics
"""""""""""

Absorbing state
----------------
Terminal state.

Behaviour distribution
-----------------------
The probability distribution over sequences of states and actions.

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

Reward function
------------------
Maps state-action pairs to rewards.

Value function
----------------
The value of a state is equal to the expectation of the reward function given the state and the policy. 

.. math::

    V(s) = E[R|s,\pi]
