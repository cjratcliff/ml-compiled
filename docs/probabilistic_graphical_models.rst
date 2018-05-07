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

Conditional Random Field (CRF)
---------------------------------

Markov Random Field (MRF)
---------------------------
