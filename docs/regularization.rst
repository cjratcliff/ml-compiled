===============
Regularization
===============
Used to reduce overfitting and improve generalization to data that was not seen during the training process.

`Identifying Generalization Properties in Neural Networks, Wang et al. (2018) <https://arxiv.org/abs/1809.07402v1>`_

`Understanding Deep Learning Requires Rethinking Generalization, Zhang et al. (2016) <https://arxiv.org/pdf/1611.03530.pdf>`_

General principles
""""""""""""""""""""
* Small changes in the inputs should not produce large changes in the outputs.
* Sparsity. Most features should be inactive most of the time.
* It should be possible to model the data well using a relatively low dimensional distribution of independent latent factors.

Methods
""""""""
* `Dropout <https://ml-compiled.readthedocs.io/en/latest/regularization.html#dropout>`_
* `Weight decay <https://ml-compiled.readthedocs.io/en/latest/regularization.html#weight-decay>`_
* `Early stopping <https://ml-compiled.readthedocs.io/en/latest/optimizers.html#early-stopping>`_
* `Unsupervised pre-training <https://ml-compiled.readthedocs.io/en/latest/training_with_limited_data.html#unsupervised-pre-training>`_
* `Data augmentation <https://ml-compiled.readthedocs.io/en/latest/computer_vision.html#data-augmentation>`_
* `Semi-supervised learning <https://ml-compiled.readthedocs.io/en/latest/training_with_limited_data.html#semi-supervised-learning>`_
* Noise injection
* `Bagging and ensembling <https://ml-compiled.readthedocs.io/en/latest/ensemble_models.html>`_
* Optimisation algorithms like SGD that prefer wide minima 
* `Batch normalization <https://ml-compiled.readthedocs.io/en/latest/layers.html?highlight=batch%20normalization#batch-normalization>`_
* Label smoothing

Dropout
""""""""
For each training case, omit each hidden unit with some constant probability. This results in a network for each training case, the outputs of which are combined through averaging. If a unit is not omitted, its value is shared across all the models. Prevents units from co-adapting too much.

Dropout’s effectiveness could be due to:

* An ensembling effect. ‘Training a neural network with dropout can be seen as training a collection of :math:`2^n` thinned networks with extensive weight sharing’ - `Srivastava et al. (2014) <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_
* Restricting the network’s ability to co-adapt weights. The idea is that if a node is not reliably included, it would be ineffective for nodes in the next layer to rely on it’s output. Weights that depend strongly on each other correspond to a sharp local minimum as a small change in the weights is likely to damage accuracy significantly. Conversely, nodes that take input from a variety of sources will be more resilient and reside in a shallower local minimum.

Can be interpreted as injecting noise inside the network.

`Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Srivastava et al. (2014) <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_

Variational dropout
-----------------------
Applied to RNNs. Unlike normal dropout, the same dropout mask is retained over all timesteps, rather than sampling a new one each time the cell is called. Compared to normal dropout, this is less likely to disrupt the RNN’s ability to learn long-term dependencies.

`Variational Dropout and the Local Reparameterization Trick, Kingma et al. (2015) <https://arxiv.org/abs/1506.02557>`_

Generalization error
"""""""""""""""""""""""
The difference between the training error and the test error.

Label smoothing
"""""""""""""""""
Replaces the labels with a weighted average of the true labels and the uniform distribution.

Weight decay
"""""""""""""""

L1 weight decay
---------------------
Adds the following term to the loss function:

.. math::

    C \sum_{i=1}^k |\theta_i|

:math:`C > 0` is a hyperparameter.

L1 weight decay is mathematically equivalent to `MAP estimation <https://ml-compiled.readthedocs.io/en/latest/probability.html#map-estimation>`_ with a Laplacian prior on the parameters.

L2 weight decay
-----------------
Adds the following term to the loss function:

.. math::

    C \sum_{i=1}^k {\theta_i}^2

:math:`C > 0` is a hyperparameter.

L2 weight decay is mathematically equivalent to doing `MAP estimation <https://ml-compiled.readthedocs.io/en/latest/probability.html#map-estimation>`_ where the prior on the parameters is Gaussian:

.. math::

  q(\theta) = N(0,C^{-1})

Intuition
-------------
Weight decay works by making large parameters costly. Therefore during optimisation the most important parameters will tend to have the largest magnitude. The unimportant ones will be close to zero.

Sometimes referred to as ridge regression or Tikhonov regularisation in statistics.
