""""""""""""""""""""
Density estimation
""""""""""""""""""""
The problem of estimating the probability density function for a given set of observations. Once this is known it can be used to generate new samples from the distribution.

Empirical distribution function
---------------------------------

Histogram
-----------

Kernel Density Estimation
---------------------------

.. math::

  \hat{f}(x) = \frac{1}{n}\sum_{i=1}^n K_h(x - x_i)

Mixture Model
------------------------
Estimates the density as a weighted sum of parametric distributions. The predicted density function for a sample :math:`x` is:

.. math::

  \hat{f}(x) = \sum_{i=1}^k w_i \phi(x;\theta_i)

Where :math:`k` is the number of distributions and each distribution, :math:`\phi`, is parameterised by :math:`\theta`. It is also weighted by a single scalar :math:`w_i` where :math:`\sum_{i=1}^k w_i = 1`.

The Gaussian is a common choice for the distribution. In this case the estimator is known as a **Gaussian Mixture Model**.

All of the parameters can be learnt using Expectation-Maximization, except for :math:`k` which is a hyperparameter.


