"""""""""""""""""""
Initialization
"""""""""""""""""""

He initialization
--------------------
The parameters are drawn from the following normal distribution:

.. math::

  \theta^{(i)} \sim N(0, \sqrt{2/n_i})
  
where :math:`\theta^{(i)}` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

`Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification , He et al. (2015) <https://arxiv.org/abs/1502.01852>`_

Orthogonal initialization
----------------------------
Useful for training very deep networks.
Can be used to help with vanishing and exploding gradients in RNNs.

`All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_

`Explaining and illustrating orthogonal initialization for recurrent neural networks, Merity (2016) <https://smerity.com/articles/2016/orthogonal_init.html>`_

Orthonormal initialization
____________________________
Initialise the matrix by first generating a matrix where every entry is drawn from a normal distribution with mean 0 and variance 1. Then perform a `QR decomposition <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#qr-decomposition>`_ and use Q as the initialization matrix. Alternatively, do `SVD <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#singular-value-decomposition-svd>`_ and pick U or V as the initialization matrix.

Described in `Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, Saxe et al. (2013) <https://arxiv.org/abs/1312.6120>`_

Xavier initialization
-----------------------
Sometimes referred to as Glorot initialization.

.. math::

  \theta^{(i)} \sim U(-\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}},\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}})
  
where :math:`\theta^{(i)}` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

`Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
