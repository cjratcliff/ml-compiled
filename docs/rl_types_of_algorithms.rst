""""""""""""""""""""""""""""""""""""""
Types of policy-learning algorithms
""""""""""""""""""""""""""""""""""""""

Model-based reinforcement learning
-------------------------------------
Models the environment in order to predict the distribution over states that will result from a given state-action pair.

Model-free reinforcement learning
-------------------------------------
Algorithms that learn the policy without requiring a model of the environment. Q-learning is an example.

Off-policy learning
---------------------
The behaviour distribution does not follow the policy. Typically a more exploratory behaviour distribution is chosen. An example is `Q-learning <https://ml-compiled.readthedocs.io/en/latest/td.html#q-learning>`_.

On-policy learning
--------------------
The policy determines the samples the network is trained on. Can introduce bias to the estimator. An example is `SARSA <https://ml-compiled.readthedocs.io/en/latest/td.html#sarsa>`_.

Policy-based method
----------------------
Does not use a value function. Learns the policy explicitly, unlike value-based methods.

Policy gradient method
-------------------------
Policy learning algorithm. Iteratively alternates between improving the policy given the value function and the value function under the current policy.

Value-based methods
-------------------------
Have an implicit policy based on choosing the action which maximises the value function. May use an epsilon-greedy policy, for example. Can’t learn stochastic policies.
