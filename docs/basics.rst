"""""""""""
Basics
"""""""""""

Absorbing state
----------------
See `terminal state <https://ml-compiled.readthedocs.io/en/latest/basics.html#terminal-state>`_.

Action space
--------------
The space of all possible actions. May be discrete as in Chess or continuous as in many robotics tasks.

Behaviour distribution
-----------------------
The probability distribution over sequences of state-action pairs that describes the behaviour of an agent.

Bellman equation
------------------
Computes the value of a state given a policy. Represents the intuition that if the value at the next timestep is known for all possible actions, the optimal strategy is to select the action that maximizes that value plus the immediate reward.

.. math::

    V(s,a) = \mathbb{E}_{s'}[r(s,a) + \gamma \max_{a'} V(s',a')]

Where :math:`r` is the immediate reward, :math:`\gamma` is the discount rate, :math:`s` and :math:`s'` are states and :math:`a` and :math:`a'` are actions. :math:`V(s,a)` is the value function for executing action :math:`a` in state :math:`s`.

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

Markov Decision Process (MDP)
-----------------------------------
Models the environment using Markov chains, extended with actions and rewards. 

Partially Observable Markov Decision Process (POMDP)
----------------------------------------------------------
Generalization of the MDP. The agent cannot directly observe the underlying state.

Policy
----------
A function, :math:`\pi` that maps states to actions.

Regret
-------
The difference in the cumulative reward between performing optimally and executing the given policy.

Reward function
------------------
Maps state-action pairs to rewards.

REINFORCE
------------
Simple policy learning algorithm.

If a policy :math:`\pi_\theta` executes action :math:`a` in state :math:`s` with some corresponding value :math:`v` the update rule is:

.. math::

  \Delta \theta = \alpha \nabla_\theta \pi_\theta(s,a) v
  
Where :math:`\nabla_\theta` means the derivative with respect to :math:`\theta`.

| **Proposed in**
| `Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams (1992) <http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf>`_

Terminal state
----------------
A state which ends the episode when reached. No further actions by the agent are possible.

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

