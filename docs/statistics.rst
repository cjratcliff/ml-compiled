Statistics
"""""""""""""

Arithmetic mean
--------------------

.. math::

  A(x_1,x_2,...,x_n) = \frac{1}{n}\sum_{i=1}^n x_i

Bernoulli distribution
------------------------
Distribution for a random variable which is 1 with probability :math:`p` and zero with probability :math:`1-p`.

Special case of the Binomial distribution, which generalizes the Bernoulli to multiple trials.

.. math::

  P(x = k;p) = 
  \begin{cases}
    p, & \text{if } k = 1\\
    1-p, & \text{if } k = 0
  \end{cases}

Binomial distribution
-----------------------
Distribution for the number of successes in n trials, each with probability p of success and 1-p of failure.

.. math::
  
  P(x = k;n,p) = {n\choose k} p^k (1-p)^{n-k}

Categorical distribution
--------------------------
Generalizes the Bernoulli distribution to more than two categories.

.. math::

  P(x = k;p) = p_k

Covariance matrix
----------------------
There are three types of covariance matrix:

* Full - All entries are specified. Has :math:`O(n^2)` parameters.
* Diagonal - The matrix is diagonal, meaning all off-diagonal entries are zero. Variances can differ across dimensions but there is no interplay between the dimensions. Has :math:`O(n)` parameters.
* Spherical - The matrix is equal to the identity matrix multiplied by a constant. This means the variance is the same in all dimensions. Has :math:`O(1)` parameters.

A valid covariance matrix is always symmetric and positive semi-definite.

Geometric mean
----------------

.. math::

    G(x_1,x_2,...,x_n) = \sqrt[\leftroot{-2}\uproot{2}n]{x_1x_2...x_n}

Only applicable to positive numbers.

Gumbel distribution
---------------------
Used to model the distribution of the maximum (or the minimum) of a number of samples of various distributions.

`Categorical Reparameterization with Gumbel-Softmax, Jang et al. (2016) <https://arxiv.org/abs/1611.01144>`_

Harmonic mean
---------------

.. math::

    H(x_1,x_2,...,x_n) = n/\sum_{i=1}^n \frac{1}{x_i} 
    
Moments
--------
* 1st moment - Arithmetic mean
* 2nd moment - Variance
* 3rd moment - Skewness
* 4th moment - Kurtosis
    
Point estimate
----------------
An estimate for a parameter.

Zipf distribution
---------------------
For a population of size n, the frequency of the kth most frequent item is:

.. math::

  \frac{1/{k^s}}{\sum_{i=1}^n 1/i^s}
  
where :math:`s \geq 0` is a hyperparameter
