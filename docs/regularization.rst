===============
Regularization
===============

General principles
Small changes in the inputs should not produce large changes in the outputs.
Sparsity. Most features should be inactive most of the time.
It should be possible to model the data well using a relatively low dimensional distribution of independent latent factors.

"""""""
Methods
"""""""
* Dropout
* Weight decay
* Early stopping
* Unsupervised pre-training
* Dataset augmentation
* Semi-supervised learning
* Noise injection
* Bagging and ensembling
* Optimisation algorithms like SGD that prefer wide minima


Dropout
-------
Regularization method. For each training case, omit each hidden unit with some constant probability. This results in a network for each training case, the outputs of which are combined through averaging. If a unit is not omitted, its value is shared across all the models. Prevents units from co-adapting too much.

Dropout’s effectiveness could be due to:
* An ensembling effect. ‘Training a neural network with dropout can be seen as training a collection of :math:`2^n` thinned networks with extensive weight sharing’ - Srivastava et al. (2014)
* Restricting the network’s ability to co-adapt weights. The idea is that if a node is not reliably included, it would be ineffective for nodes in the next layer to rely on it’s output. Weights that depend strongly on each other correspond to a sharp local minimum as a small change in the weights is likely to damage accuracy significantly. Conversely, nodes that take input from a variety of sources will be more resilient and reside in a shallower local minimum.

Can be interpreted as injecting noise inside the network.

`Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Srivastava et al. (2014) <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_


Weight decay
------------

""""""""""""""
L1 weight decay
"""""""""""""""
Regularization method. Adds the following term to the cost function:

.. math::

    C \sum_{i=1}^k |\theta_i|

:math:`C > 0` is a hyperparameter.

"""""""""""""""
L2 weight decay
"""""""""""""""
Regularization method. Adds the following term to the loss function:

.. math::

    C \sum_{i=1}^k {\theta_i}^2

:math:`C > 0` is a hyperparameter.

Intuition
Weight decay works by making large parameters costly. Therefore during optimisation the most important parameters will tend to have the largest magnitude. The unimportant ones will be close to zero.

Sometimes referred to as ridge regression or Tikhonov regularisation in statistics.
