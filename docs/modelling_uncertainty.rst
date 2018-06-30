""""""""""""""""""""""
Modelling uncertainty
""""""""""""""""""""""

Calibration
---------------
The problem of getting accurate estimates of the uncertainty of the prediction(s) of a classifier or regressor.

For example, if a binary classifier gives scores of 0.9 and 0.1 for classes A and B that does not necessarily mean it has a 90% chance of being correct. If the actual probability of being correct (class A) is far from 90% we say that the classifier is **poorly calibrated**. On the other hand, if the model if it really does have a close to 90% chance of being correct we can say the classifier is **well calibrated**.

Classification
_________________
The method below describes how a binary classifier can be calibrated but it can easily be extended to the multiclass case by repeating the process after step 1 for each class.

1. Train the classifier :math:`\hat{y} = f(x)` in the normal way
2. Construct a dataset with, for each row in the original dataset, the predicted score and the actual label.
3. Fit an `isotonic regression <https://ml-compiled.readthedocs.io/en/latest/regression.html#isotonic-regression>`_ :math:`\bar{y} = g(\hat{y})` to this data, trying to predict the label given the score. :math:`\bar{y}` can be used as a well-calibrated estimate of the true probability.

Measuring uncertainty
----------------------
This section describes methods for estimating the uncertainty of a classifier. Note that additional methods may be necessary to ensure that this estimate is well-calibrated.

Classification
________________
The uncertainty for a predicted probability distribution over a set of classes can be measured by calculating its `entropy <https://ml-compiled.readthedocs.io/en/latest/entropy.html#entropy>`_.

Regression
______________
Unlike in classification we do not normally output a probability distribution when making predictions for a regression problem. Therefore modifications must be made.

The network outputs two numbers describing the Normal distribution :math:`N(\mu,\sigma^2)`. :math:`\mu` is the predicted value and :math:`\sigma^2` describes the level of uncertainty.

* The mean :math:`\mu`, outputted by a fully-connected layer with a linear activation.
* The variance :math:`\sigma^2`, outputted by a fully-connected layer with a `softplus activation <https://ml-compiled.readthedocs.io/en/latest/activations.html#softplus>`_. Using the softplus ensures the variance is always positive without having zero gradients when the input is below zero, as with the `ReLU <https://ml-compiled.readthedocs.io/en/latest/activations.html#relu>`_.

The loss function is the negative log likelihood of the observation under the predicted distribution:  

.. math::

  L(y,\mu,\sigma) = - \frac{1}{2}\log(\sigma^2) - \frac{1}{2n \sigma^2}\sum_{i=1}^n (y - \mu)^2

Example paper
________________
`Asynchronous Methods for Deep Reinforcement Learning, Mnih et al. (2016) <https://arxiv.org/abs/1602.01783>`_ - Section 9
