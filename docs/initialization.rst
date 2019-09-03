"""""""""""""""""""
Initialization
"""""""""""""""""""

He initialization
--------------------
The weights are drawn from the following normal distribution:

.. math::

  \theta^{(i)} \sim N(0, \sqrt{2/n_i})
  
where :math:`\theta^{(i)}` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

The biases are initialized to zero as usual.

| **Proposed in** 
| `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification , He et al. (2015) <https://arxiv.org/abs/1502.01852>`_

Orthogonal initialization
----------------------------
Useful for training very deep networks.
Can be used to help with vanishing and exploding gradients in RNNs.

`Explaining and illustrating orthogonal initialization for recurrent neural networks, Merity (2016) <https://smerity.com/articles/2016/orthogonal_init.html>`_

LSUV initialization
______________________
Layer-sequential unit-variance initialization. An iterative initialization procedure:

1. Pre-initialize the layers with orthonormal matrices as proposed in.
2. For each layer:
3. While :math:`Var() - 1` \geq Tol_{var}` and :math:`T_i < T_\max`:
4.    :math:`\theta^(i) = \theta^(i) / \sqrt{Var(O^(i))}`

| **Proposed in**
| `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_

Orthonormal initialization
____________________________

1. Initialise the weights from a standard normal distribution: :math:`\theta_i \sim N(0, 1)`.
2. Perform a `QR decomposition <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#qr-decomposition>`_ and use Q as the initialization matrix. Alternatively, do `SVD <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#singular-value-decomposition-svd>`_ and pick U or V as the initialization matrix.

| **Proposed in**
| `Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, Saxe et al. (2013) <https://arxiv.org/abs/1312.6120>`_

Xavier initialization
-----------------------
Sometimes referred to as Glorot initialization.

.. math::

  \theta^{(i)} \sim U(-\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}},\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}})
  
where :math:`\theta^{(i)}` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

| **Proposed in** 
| `Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
