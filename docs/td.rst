""""""""""""""""""""""""""""""""""
Temporal-difference learning
""""""""""""""""""""""""""""""""""
Temporal-difference learning optimizes the model to make predictions of the total return more similar to other, more accurate, predictions. These latter predictions are more accurate because they were made at a later point in time, closer to the end.

The TD error is defined as:

.. math::

    r_t+V(s_{t+1})-V(s_t)
    
Q-learning is an example of TD learning. 
    
Action-value function
-----------------------
Another term for the `Q-function <https://ml-compiled.readthedocs.io/en/latest/td.html#the-q-function>`_.

Actor-critic method
----------------------
A type of on-policy temporal-difference method, as well as a policy-gradient algorithm. 

The policy is the actor and the value function is the critic, with the 'criticism' being the TD error. If the TD error is positive the value of the action was greater than expected, suggesting the chosen action should be taken more often. If the TD error was negative the action had a lower value than expected, and so will be done less often in future states which are similar.

Unlike pure policy or value-based methods, actor-critic learns both a policy and a value function. 

Apart from being off-policy, Q-learning is different as it estimates the value as a function of the state and the action, not just the state.

Asynchronous Advantage Actor-Critic (A3C)
_____________________________________________
An on-policy asynchronous RL algorithm. Can train both feedforward and recurrent agents. Recurrent agents do not require pooling as they act in the generative fashion.

Maintains a policy (the actor) :math:`\pi(a_t|s_t;\theta)` and a the value function (the critic) :math:`V(s_t;\theta_v)` The policy and value functions are updated after :math:`t_{max}` steps or when a terminal state is reached. The policy and value functions share all parameters apart from those in the final output layers. The policy network has a softmax over all actions (in the discrete case) and the value network has a single linear output.

The advantage function for doing action :math:`a_t` in state :math:`s_t` is the sum of discounted rewards plus the difference in the value functions between the states:

.. math::

    A(s_t,a_t;\theta,\theta_v) = \sum_{i=0}^{k-1}\gamma^i r_{t+i} + \gamma^k V(s_{t+k};\theta_v)-V(s_t;\theta_v), k \leq t_{max}


The loss function is:

.. math::

    L = \log \pi(a_t|s_t;\theta)(R_t-V(s_t;\theta_v)) + \beta H(\pi(s_t;\theta)) 

Where

.. math::

    R_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
    
and :math:`H(\pi(s_t;\theta)` is the entropy of the policy. This term is used to incentivize exploration. :math:`\beta` is a hyperparameter.

:math:`R_t-V(s_t;\theta_v)` is the temporal difference term. 

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
_________________
Expresses the expected total reward from taking the action in the given state and following the policy thereafter. Also known as the action-value function.

.. math::

    Q^\pi(s,a) = E[R|s,a,\pi]
    
Eventually converges to the optimal policy in any finite MDP. In its simplest form it uses tables to store values for the Q function, although this only works for very small state and action spaces.
    
Deep Q-learning
____________________
A neural network is used to approximate the optimal action-value function, :math:`Q(s,a)` and the actions which maximise Q are chosen. 

The action-value function is defined according to the `Bellman equation <https://ml-compiled.readthedocs.io/en/latest/basics.html#bellman-equation>`_.

The CNN takes an image of the game state as input and outputs a Q-value for each action in that state. This is more computationally efficient than having the action as an input to the network. The action with the largest corresponding Q-value is chosen.

Uses the loss function:

.. math::

    L = \mathbb{E}_{s,a}[(y - Q(s,a;\theta))^2]

where the target, :math:`y` is defined as:

.. math::

    y = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q(s',a';\theta)|s,a]

This means the target depends on the network weights, unlike in supervised learning. The loss function tries to change the parameters such that the estimate and the true Q-values are as close as possible, making forecasts of action-values more accurate.

Periodically freezing the target Q network helps prevent oscillations or divergence in the learning process.

`Playing Atari with Deep Reinforcement Learning, Mnih et al. (2013) <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_

`Human-level control through deep reinforcement learning, Mnih et al. (2015) <https://www.nature.com/articles/nature14236>`_

`Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al. (2017) <https://arxiv.org/pdf/1710.02298.pdf>`_

Experience Replay
'''''''''''''''''''
Sample experiences :math:`(s_t, a_t, r_t, s_{t+1})` to update the Q-function from a **replay memory** which retains the last N experiences. `Mnih et al. (2013) <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_ set N to 1 million when training over a total of 10 million frames.

Contrast this with `on-policy learning algorithms <https://ml-compiled.readthedocs.io/en/latest/rl_types_of_algorithms.html#on-policy-learning>`_ learn from events as they experience them. This can cause two problems:

1. Most gradient descent algorithms rely on the assumption that updates are identically and independently distributed. Learning on-policy can break that assumption since the update at time t influences the state at the next timestep.
2. Events are forgotten quickly. This can be particularly harmful in the case of rare but important events.

Both of these problems are solved by using experience replay.

The use of a replay memory means it is necessary to learn off-policy.

`Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching, Lin (1992) <http://www.incompleteideas.net/lin-92.pdf>`_

`Playing Atari with Deep Reinforcement Learning, Mnih et al. (2013) <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_

Prioritized Experience Replay
''''''''''''''''''''''''''''''''
Samples from the `replay memory <https://ml-compiled.readthedocs.io/en/latest/td.html#experience-replay>`_ according to a function of the loss. In contrast, in the standard approach (eg `Mnih et al. (2013) <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_) past experiences are selected uniformly at random from the replay memory.

TODO

`Prioritized Experience Replay, Schaul et al. (2015) <https://arxiv.org/abs/1511.05952>`_

Distributional Q-learning
''''''''''''''''''''''''''''''
Models the distribution of the value function, rather than simply its expectation.

`A Distributional Perspective on Reinforcement Learning, Bellemare et al. (2017) <https://arxiv.org/abs/1707.06887>`_

Multi-step bootstrap targets
''''''''''''''''''''''''''''''

`Asynchronous Methods for Deep Reinforcement Learning, Mnih et al. (2016) <https://arxiv.org/abs/1602.01783>`_

`Learning to Predict by the Methods of Temporal Differences, Sutton (1988) <https://pdfs.semanticscholar.org/9c06/865e912788a6a51470724e087853d7269195.pdf>`_

Noisy parameters
'''''''''''''''''''
A method for helping exploration when training that can be more effective than traditional `epsilon-greedy <https://ml-compiled.readthedocs.io/en/latest/explore_exploit.html#epsilon-greedy-policy>`_ appraoch. The linear component :math:`y = wx + b` of the layers in the network are replaced with:

.. math::

  y = (\mu_w + \sigma_w * \epsilon_w)x + (\mu_b + \sigma_b * \epsilon_b)
  
where :math:`\mu_w` and :math:`\sigma_w` are learned parameter matrices of the same shape as :math:`w` in the original equation. Similarly, :math:`\mu_b` and :math:`\sigma_b` are learned parameter vectors and have the same shape as :math:`b`. :math:`\epsilon_w` and :math:`\epsilon_b` also have the same shape as :math:`w` and :math:`b` respectively, but are not learnt - they are random variables.

Since the amount of noise is learnt no hyperparameter-tuning is required, unlike epsilon-greedy, for example.

`Noisy Networks for Exploration, Fortunato et al. (2017) <https://arxiv.org/abs/1706.10295>`_

SARSA
-------
An algorithm for learning a policy. Stands for state-action-reward-state-action.

The update rule for learning the Q-function is:

.. math::

    Q(s_t,a_t) := Q(s_t,a_t) + \alpha (r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)) 

Where :math:`0 < \alpha < 1` is the learning rate.

Pseudocode:

.. code-block:: none

      1. Randomly initialize Q(s,a)
      2. While not converged:
      3.   Choose the action that maximizes Q(s,a)
      4.   Compute the next state, given s and a.
      5.   Apply the update rule for the Q-function.
    
Unlike Q-learning, SARSA is an on-policy algorithm and thus learns the Q-values associated with the policy it follows itself. Q-learning on the other hand is an off-policy algorithm and learns the value function while following an exploitation/exploration policy. 

