"""""""""""""""""""""""""""""""""""
Graphical models
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

Boltzmann Machines
----------------------

Restricted Boltzmann Machines (RBMs)
______________________________________
Trained with contrastive divergence. 


Deep Belief Networks (DBNs)
______________________________


Deep Belief Machines (DBMs)
______________________________


Clique
-------
A subset of a graph where the nodes are fully-connected, ie each node has an edge with every other node in the set.

Conditional Random Field (CRF)
---------------------------------
Discriminative model that can be seen as a generalization of logistic regression.

Common applications of CRFs include `image segmentation <https://ml-compiled.readthedocs.io/en/latest/computer_vision.html#semantic-segmentation>`_ and `named entity recognition <https://ml-compiled.readthedocs.io/en/latest/natural_language_processing.html#named-entity-recognition-ner>`_.

| **Used in**
| `Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation, Kolesnikov and Lampert (2016) <https://arxiv.org/abs/1603.06098>`_

Linear Chain CRFs
___________________
A simple sequential CRF.


Hidden Markov Model (HMM)
---------------------------
A simple generative sequence model in which there is an observable state and a latent state, which must be inferred. 

At each time step the model is in a latent state :math:`x_t` and outputs an observation :math:`y_t`. The observation is solely a function of the latent state, as is the probability distribution over the next state, :math:`x_{t+1}`. Hence the model obeys the `Markov property <https://ml-compiled.readthedocs.io/en/latest/probabilistic_graphical_models.html#markov-property>`_.

The model is defined by:

* A matrix :math:`T` of transition probabilities where :math:`T_{ij}` is the probability of going from state i to state j.
* A matrix :math:`E` of emission probabilities where :math:`E_{ij}` is the probability of emitting observation j in state i.

The parameters can be learnt with the Baum-Welch algorithm.

Markov chain
--------------
A simple state transition model where the next state depends only on the current state. At any given time, if the current state is node i, there is a probability :math:`T_{ij}` of transitioning to node j, where :math:`T` is the transition matrix.

.. figure:: ../img/markov_chain.PNG
  :align: center
  
  Source: `Wikipedia <https://en.wikipedia.org/wiki/Markov_chain#/media/File:Markovkate_01.svg>`_

Markov property
--------------------
A process is said to have the Markov property if the next state depends only on the current state, not any of the previous ones.

Markov Random Field (MRF)
---------------------------
A type of undirected graphical model which defines the joint probability distribution over a set of variables. Each variable is represented by one node in the graph.

One use for an MRF could be to model the distribution over the pixel values for a set of images. In order to keep the model tractable edges are only drawn between neighbouring pixels.

Naive Bayes Model
-------------------
A simple classifier that models all of the features as independent, given the label.

.. math::

  P(Y|X_1,...,X_n) = P(Y)\prod_{i=1}^n P(Y|X_i)
