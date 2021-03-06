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

Effectiveness
_______________
Was used to improve the state of the art for image classification (`He et al., 2015 <https://arxiv.org/abs/1502.01852>`_) but the improvement over ReLU activations with Xavier initialization was very small, reducing top-1 error on ImageNet from 33.9% to 33.8%. 

| **Proposed in** 
| `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification , He et al. (2015) <https://arxiv.org/abs/1502.01852>`_

Initialization with zeros
-----------------------------
All of the weights are initialised to zero. Used for bias vectors since the weight matrix, which is initialized with random weights, provides the symmetry breaking.

Orthogonal initialization
----------------------------
Initializes the weights as an orthogonal matrix. Useful for training very deep networks. Can be used to help with vanishing and exploding gradients in RNNs.

The procedure is as follows:

.. code-block:: none

  1. Generate a matrix of random numbers, X (eg from the normal distribution)
  2. Perform the QR decomposition X = QR, resulting in an orthogonal matrix Q and an upper triangular matrix R.
  3. Initialise with Q.

| **Further reading**
| `Explaining and illustrating orthogonal initialization for recurrent neural networks, Merity (2016) <https://smerity.com/articles/2016/orthogonal_init.html>`_

LSUV initialization
______________________
Layer-sequential unit-variance initialization. An iterative initialization procedure:

.. code-block:: none

  1. t_max = 10
  2. tol_var = 0.05
  3. pre-initialize the layers with orthonormal matrices as proposed in Saxe et al. (2013)
  4. for each layer:
  5.    let w be the weights of the layer
  6.    let b be the output of the layer 
  7.    for i in range(t_max):
  8.        w = w / sqrt(var(b))
  9.        if abs(var(b) - 1) < tol_var:
  10.            break

| **Proposed in**
| `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_

Orthonormal initialization
____________________________

1. Initialise the weights from a standard normal distribution: :math:`\theta_i \sim N(0, 1)`.
2. Perform a `QR decomposition <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#qr-decomposition>`_ and use Q as the initialization matrix. Alternatively, do `SVD <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#singular-value-decomposition-svd>`_ and pick U or V as the initialization matrix.

| **Proposed in**
| `Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, Saxe et al. (2013) <https://arxiv.org/abs/1312.6120>`_
|
| **Used by**
| `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, Cho et al. (2014) <https://arxiv.org/pdf/1406.1078.pdf>`_

Symmetry breaking
------------------
An essential property of good initialization for fully connected layers. In a fully connected layer every hidden node has exactly the same set of inputs so if all nodes are initialised to the same value their gradients will also be identical. Thus they will never take on different values.

Xavier initialization
-----------------------
Sometimes referred to as Glorot initialization.

.. math::

  \theta^{(i)} \sim U(-\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}},\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}})
  
where :math:`\theta^{(i)}` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

Xavier initialization's derivation assumes linear activations. Despite this it has been observed to work well in practice for networks that whose activations are nonlinear.

| **Proposed in** 
| `Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
