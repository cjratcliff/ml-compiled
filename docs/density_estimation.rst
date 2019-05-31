""""""""""""""""""""
Density estimation
""""""""""""""""""""
The problem of estimating the probability density function for a given set of observations.

Empirical distribution function
---------------------------------
Compute the empirical CDF and numerically differentiate it.

Histogram
-----------
Take the range of the sample and split it up into n bins, where n is a hyperparameter. Then assign a probability to each bin according to the proportion of the sample that fell within its bounds.

Isolation Forest
-------------------
An ensemble of decision trees. The key idea is that points in less dense areas will require fewer splits to be uniquely identified since they are surrounded by fewer points.

Features and split values are randomly chosen, with the split value being somewhere between the min and the max observed values of the feature.

Kernel Density Estimation
---------------------------
The predicted density function given an a sample :math:`x` is:

.. math::

  \hat{f}(x) = \frac{1}{n}\sum_{i=1}^n K_h(x - x_i)
  
Where :math:`K` is the kernel and :math:`h > 0` is a smoothing parameter.

.. math::

  K_h(x) = \frac{1}{h}K\big(\frac{x}{h}\big)

A variety of kernels can be used. A common one is the Gaussian, defined as:

.. math::

  K(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}
  
Disadvantages
_______________
The complexity at inference time is linear in the size of the sample.

Local Outlier Factor
-----------------------
A nearest-neighbour model.

Mixture Model
------------------------
Estimates the density as a weighted sum of parametric distributions. The predicted density function for a sample :math:`x` is:

.. math::

  \hat{f}(x) = \sum_{i=1}^k w_i \phi(x;\theta_i)

Where :math:`k` is the number of distributions and each distribution, :math:`\phi`, is parameterised by :math:`\theta`. It is also weighted by a single scalar :math:`w_i` where :math:`\sum_{i=1}^k w_i = 1`.

The Gaussian is a common choice for the distribution. In this case the estimator is known as a **Gaussian Mixture Model**.

All of the parameters can be learnt using Expectation-Maximization, except for :math:`k` which is a hyperparameter.

One-Class SVM
----------------

