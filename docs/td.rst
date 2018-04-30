""""""""""""""""""""""""""""""""""
Temporal-difference learning
""""""""""""""""""""""""""""""""""

The model is optimized to make predictions of the total return more similar to other, more accurate, predictions. These latter predictions are more accurate because they were made at a later point in time, closer to the end. Uses the recursive Bellman equation. Q-learning is an example of TD learning. 

The TD error is defined as:

.. math::

    r_t+V(s_{t+1})-V(s_t)
    
Action-value function
-----------------------
Another term for the Q-function.

Actor-critic method
----------------------
Type of on-policy temporal-difference method. Also a policy-gradient algorithm. The policy is the actor and the value function is the critic, with the criticism being TD error. If the error is positive, it suggests the chosen action should be taken more often and vice versa if the error is negative. Unlike pure policy or value based methods, actor-critic learns both a policy and a value function. Apart from being off-policy, Q-learning is different as it estimates the value as a function of the state and the action, not just the state.

Asynchronous Advantage Actor-Critic (A3C)
----------------------------------------------
An on-policy asynchronous RL algorithm. Can train both feedforward and recurrent agents. Recurrent agents do not require pooling as they act in the generative fashion.

Maintains a policy (the actor) and an estimate of the value function (the critic) :math:`V(s_t;\theta_v)` The policy and value functions are updated after :math:`t_{max}` steps or when a terminal state is reached. The two functions share all parameters apart from those in the final output layers. The policy network has a softmax over all actions (in the discrete case) and the value network has a single linear output.

Both global and local versions of the parameters are maintained for the policy and value nets.

The advantage function for doing action :math:`a_t` in state :math:`s_t` is the sum of discounted rewards plus the difference in the value functions between the states:

.. math::

    A(s_t,a_t;\theta,\theta_v) = \sum_{i=0}^{k-1}\gamma^i r_{t+i} + \gamma^k V(s_{t+k};\theta_v)-V(s_t;\theta_v), k \leq t_{max}


The loss function for the policy network is:

.. math::

    L =(a_t|s_t;\theta')(R_t-V(s_t;\theta_v)) + \beta H(\pi(s_t;\theta)) 

Where

.. math::

    R_t=\sum_{k=0}^{\inf}\gamma^k r_{t+k}

:math:`R_t-V(s_t;v)` is the temporal difference term. 

It’s multiplied by the probability assigned by the policy for the action at time :math:`t`. This means policies which are more certain will be penalized more heavily for incorrectly estimating the value function. The final term is the entropy of the policy's distribution over actions.

It is optimized by RMSProp with the moving average of gradients shared between threads.

`Asynchronous Methods for Deep Reinforcement Learning, Mnih et al. (2016) <https://arxiv.org/abs/1602.01783>`_

Q-learning
----------------
Model-free iterative algorithm to find the optimal policy and a form of temporal difference learning. 

Uses the update rule:

.. math::

    Q(s,a) := Q(s,a) + \alpha(r + \gamma \max_{a'}Q(s',a'))

where :math:`Q(a,s)` is the value of performing action a in state s and performing optimally thereafter. :math:`s'` is the state that results from performing action :math:`a` in state :math:`s`.

The Q-function
'''''''''''''''''''''
Also known as the action-value function. Eventually converges to the optimal policy in any finite MDP. In its simplest form it uses tables to store values for the Q function, although this only works for very small state and action spaces. An off-policy learner.

The expected total reward from taking the action in the state and following the policy thereafter.

.. math::

    Q^\pi(s,a) = E[R|s,a,\pi]
    
Deep Q-learning
''''''''''''''''''''
Was used for the Atari games by DeepMind. A CNN is used to approximate the optimal action-value function:

% TODO: check the formulae in this section

.. math::

    Q^*(s,a) = \max_\pi \mathbb{E}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...| s_t = s, a_t = a, \pi]

The CNN takes an image of the game state as input and outputs a Q-value for each action in that state. This is more computationally efficient than having the action as an input to the network. The action with the largest corresponding Q-value is chosen.

Uses the loss function:

.. math::

    L = \mathbb{E}_{s,a}[(y - Q(s,a;\theta))^2]

where the target, :math:`y` is defined as:

.. math::

    y = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q(s',a';\theta)|s,a]

This means the target depends on the network weights, unlike in supervised learning. The loss function tries to change the parameters such that the estimate and the true Q-values are as close as possible, making forecasts of action-values more accurate.

A replay memory and periodically freezing the target Q network prevents oscillations or divergence in the learning process. The use of a replay memory means it is necessary to learn off-policy, hence the choice of Q-learning. Clipping is used to ensure the gradients are well-conditioned.

`Playing Atari with Deep Reinforcement Learning, Mnih et al. (2013) <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_

`Human-level control through deep reinforcement learning, Mnih et al. (2015) <https://www.nature.com/articles/nature14236>`_

SARSA
-------
An algorithm for learning a policy. Stands for state-action-reward-state-action. On-policy. Unlike Q-learning, SARSA is an on-policy algorithm and thus learns the Q-values associated with the policy it follows itself. Q-learning on the other hand is an off-policy algorithm and therefore learns the value function while following an exploitation/exploration policy. 

The update rule is:

.. math::

    Q(s_t,a_t) := Q(s_t,a_t) + \alpha (r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)) 

