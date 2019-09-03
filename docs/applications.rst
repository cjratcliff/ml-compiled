"""""""""""""""
Applications
"""""""""""""""

Atari
------

Notable results
''''''''''''''''
Below is the median human-normalized performance on the 57 Atari games dataset with human starts. The numbers are from `Hessel et al. (2017) <https://arxiv.org/pdf/1710.02298.pdf>`_.

* 153% - `Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al. (2017) <https://arxiv.org/pdf/1710.02298.pdf>`_
* 128% - `Prioritized Experience Replay, Schaul et al. (2015) <https://arxiv.org/abs/1511.05952>`_
* 125% - `A Distributional Perspective on Reinforcement Learning, Bellemare et al. (2017) <https://arxiv.org/abs/1707.06887>`_
* 117% - `Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. (2015) <https://arxiv.org/abs/1511.06581>`_
* 116% - `Asynchronous Methods for Deep Reinforcement Learning, Minh et al. (2016) <https://arxiv.org/pdf/1602.01783.pdf>`_ 
* 110% - `Deep Reinforcement Learning with Double Q-learning, Hassely et al. (2015) <https://arxiv.org/abs/1509.06461>`_
* 102% - `Noisy Networks for Exploration, Fortunato et al. (2017) <https://arxiv.org/abs/1706.10295>`_
* 68% - `Human-level control through deep reinforcement learning, Mnih et al. (2015) <https://www.nature.com/articles/nature14236>`_

Human starts are used in the evaluation to avoid rewarding agents that have overfitted to their own trajectories.

Go
----

AlphaGo
'''''''''
Go-playing algorithm by Google DeepMind.

First learns a supervised policy network that predicts moves by expert human players.
A reinforcement learning policy network is initialized to this network and then improved by policy gradient learning against previous versions of the policy network.

Finally, a supervised value-network is trained to predict the outcome (which player wins) from positions in the self-play dataset.

The value and policy networks are combined in an `Monte Carlo Tree Search (MCTS) <https://ml-compiled.readthedocs.io/en/latest/search_algorithms.html#monte-carlo-tree-search>`_ algorithm that selects actions by lookahead search.
Both the value and policy networks are composed of many convolutional layers.

AlphaGo Zero
'''''''''''''''
An advanced version of AlphaGo that beat its predecessor 100-0 without having been trained on any data from human games.

Note: AlphaGo Zero is not Alpha Zero applied to Go. They are different algorithms. AlphaGo Zero has some features specific to Go that Alpha Zero does not.

Training
__________
AlphaGo Zero is trained entirely from self-play. The key idea is to learn a policy which can no longer be improved by MCTS. The algorithm maintains a 'best network' which is updated when a new network beats it in at least 55% of games.

During training moves are picked stochastically, with the amount of noise being decreased over time. This aids exploration.

Architecture and loss functions
____________________________________
The core network is a 20-layer `ResNet <https://ml-compiled.readthedocs.io/en/latest/convolutional.html#residual-network>`_ with batch norm and ReLUs. It has two outputs:

The first predicts the value of the current game position. This is trained with a mean-squared error from the actual outcomes of played games. 1 if the player won and -1 if they lost.
The second predicts the policy, given the current game position. This is trained with a cross-entropy loss and the policy resulting from MCTS.

Paper
________
`Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, Silver et al. (2017) <https://arxiv.org/abs/1712.01815>`_

Blog posts
_________________
http://www.inference.vc/alphago-zero-policy-improvement-and-vector-fields

Poker
--------
Unlike games like Chess and Go, Poker is an imperfect information game. This means that as well as having to maintain a probability distribution over the hidden state of the game, strategies like bluffing must also be considered.

Due to the amount of luck involved who is the better of two Poker players can take a very large number of games to evaluate.

Heads up no limit Texas Hold 'em
'''''''''''''''''''''''''''''''''''
* Two players
* The cards start off dealt face down to each player.
* Cards in later rounds are dealt face up.
* The bets can be of any size, subject to an overall limit on the amount wagered in the game.

DeepStack
'''''''''''''
DeepStack is an AI for playing sequential imperfect-information games, most notably applied to heads up no-limit Texas Hold 'em Poker. It was the first algorithm to beat human professional players with statistical significance.

https://www.deepstack.ai/

`DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker, Moravcik et al. (2017) <https://arxiv.org/abs/1701.01724>`_

Starcraft
-----------
Compared to games like Go, Starcraft is hard for the following reasons:

1. Continuous action space. Means the conventional tree-search method cannot be applied. Even if it could be, the number of states to search through would be far too large.
2. Imperfect information. Not all of the environment is visible at once. This means the agent must not only seek to improve its position but also explore the environment.
3. More complex rules. There are multiple types of units, all of which have different available actions and interact in different ways.
4. Requires learning both low-level tasks (like positioning troops) and high-level strategy.
5. May require feints and bluffs.

2 and 5 may have been solved by the DeepStack poker playing system.
