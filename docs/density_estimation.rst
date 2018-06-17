""""""""""""""""""""
Density estimation
""""""""""""""""""""
The problem of estimating the probability density function for a given set of observations. Once this is known it can be used to generate new samples from the distribution.

Gaussian Mixture Model
------------------------
Estimates the density as a weighted sum of Gaussian distributions.

.. math::
  = \sum_{i=1}^k w_i N(\mu_i,\Sigma_i)

Where :math:`k` is the number of Gaussians and each Gaussian is parameterised by a mean vector :math:`\mu_i` and a covariance matrix :math:`\Sigma_i`. It is also weighted by a single scalar :math:`w_i` where :math:`\sum_{i=1}^k w_i = 1`.

All of the parameters can be learnt using Expectation-Maximization, except for :math:`k` which is a hyperparameter.
