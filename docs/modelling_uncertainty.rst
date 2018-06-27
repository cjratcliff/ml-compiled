""""""""""""""""""""""
Modelling uncertainty
""""""""""""""""""""""

Calibration
---------------
The problem of getting accurate estimates of the uncertainty of the prediction(s) of a classifier or regressor.

Measuring uncertainty
----------------------

Classification
________________
The uncertainty for a predicted probability distribution over a set of classes can be measured by calculating its `entropy <https://ml-compiled.readthedocs.io/en/latest/entropy.html#entropy>`_.

Regression
______________
Unlike in classification we do not normally output a probability distribution when making predictions for a regression problem. Therefore modifications must be made:

The network outputs two numbers: 

* The mean :math:`\mu`, outputted by a fully-connected layer with a linear activation.
* The variance :math:`\sigma^2`, outputted by a fully-connected layer with a `softplus activation <https://ml-compiled.readthedocs.io/en/latest/activations.html#softplus>`_. Using the softplus ensures the variance is always positive.

The loss function is:

TODO

Example paper
________________
`Asynchronous Methods for Deep Reinforcement Learning, Mnih et al. (2016) <https://arxiv.org/abs/1602.01783>`_ - Section 9
