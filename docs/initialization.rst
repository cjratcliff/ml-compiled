"""""""""""""""""""
Initialization
"""""""""""""""""""

Orthogonal initialization
----------------------------
Useful for training very deep networks.
Can be used to help with vanishing and exploding gradients in RNNs.

`All you need is a good init, Mishkin and Matas (2016) <https://arxiv.org/abs/1511.06422>`_

`Explaining and illustrating orthogonal initialization for recurrent neural networks, Merity (2016) <https://smerity.com/articles/2016/orthogonal_init.html>`_

Xavier initialization
-----------------------
Sometimes referred to as Glorot initialization.

.. math::

  \theta^(i) U(-\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}},\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}})
  
where :math:`\theta^(i)` are the parameters for layer :math:`i` of the network and :math:`n_i` is the size of layer :math:`i` of the network.

`Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
