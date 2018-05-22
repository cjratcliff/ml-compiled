"""""""""""""""""""""""""""""""""""
Probabilistic graphical models
"""""""""""""""""""""""""""""""""""

Bayesian network
------------------
A directed acyclic graph where the nodes represent random variables.

Not to be confused with Bayesian neural networks.

The chain rule for Bayesian networks
______________________________________

The joint distribution for all the variables in a network is equal to the product of the distributions for all the individual variables, conditional on their parents.

.. math::

    P(X_1,...,X_n) = \prod_i P(X_i|Par(X_i))

where :math:`Par(X_i)` denotes the parents of the node :math:`X_i` in the graph.

Clique
-------
A subset of a graph where the nodes are fully-connected, ie each node has an edge with every other node in the set.

Conditional Random Field (CRF)
---------------------------------

Hidden Markov Model (HMM)
---------------------------

Markov chain
--------------
A simple state transition model where the next state depends only on the current state.

Markov property
--------------------
A process is said to have the Markov property if the next state depends only on the current state, not any of the previous ones.

Markov Random Field (MRF)
---------------------------
A type of undirected graph which defines the joint probability distribution over a set of variables. Each variable is represented by one node in the graph.
