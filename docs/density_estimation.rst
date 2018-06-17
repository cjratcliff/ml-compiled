""""""""""""""""""""
Density estimation
""""""""""""""""""""
The problem of estimating the probability density function for a given set of observations. Once this is known it can be used to generate new samples from the distribution.

Empirical distribution function
---------------------------------

Histogram
-----------


Mixture Model
------------------------
Estimates the density as a weighted sum of parametric distributions.

.. math::
  = \sum_{i=1}^k w_i f(x;\theta_i)

Where :math:`k` is the number of distributions and each distribution, :math:`f`, is parameterised by `\theta`. It is also weighted by a single scalar :math:`w_i` where :math:`\sum_{i=1}^k w_i = 1`.

The Gaussian is a common choice for the distribution. In this case the estimator is known as a **Gaussian Mixture Model**.

All of the parameters can be learnt using Expectation-Maximization, except for :math:`k` which is a hyperparameter.


