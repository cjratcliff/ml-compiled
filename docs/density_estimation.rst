""""""""""""""""""""
Density estimation
""""""""""""""""""""
The problem of estimating the probability density function for a given set of observations. Once this is known it can be used to generate new samples from the distribution.

Mixture Model
------------------------
Estimates the density as a weighted sum of parametric distributions.

.. math::
  = \sum_{i=1}^k w_i f(x;\theta_i)

Where :math:`k` is the number of distributions and each distribution is parameterised by `\theta`. It is also weighted by a single scalar :math:`w_i` where :math:`\sum_{i=1}^k w_i = 1`.

The Gaussian is a common choice for the distribution.

All of the parameters can be learnt using Expectation-Maximization, except for :math:`k` which is a hyperparameter.


