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

  \theta \textasciitilde U[-\frac{\sqrt{6}}{\sqrt{m+n}},\frac{\sqrt{6}}{\sqrt{m+n}}]

`Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
