Statistics
"""""""""""""

Arithmetic mean
--------------------

.. math::

  A(x_1,x_2,...,x_n) = \frac{1}{n}\sum_{i=1}^n x_i
  
Correlation
--------------

.. math::

  \text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{V(X)V(Y)}}

Covariance matrix
----------------------
There are three types of covariance matrix:

* Full - All entries are specified. Has :math:`O(n^2)` parameters for :math:`n` variables.
* Diagonal - The matrix is diagonal, meaning all off-diagonal entries are zero. Variances can differ across dimensions but there is no interplay between the dimensions. Has :math:`O(n)` parameters.
* Spherical - The matrix is equal to the identity matrix multiplied by a constant. This means the variance is the same in all dimensions. Has :math:`O(1)` parameters.

A valid covariance matrix is always symmetric and positive semi-definite.

Geometric mean
----------------

.. math::

    G(x_1,x_2,...,x_n) = \sqrt[\leftroot{-2}\uproot{2}n]{x_1x_2...x_n}

Only applicable to positive numbers.

Harmonic mean
---------------

.. math::

    H(x_1,x_2,...,x_n) = n/\sum_{i=1}^n \frac{1}{x_i} 
    
Moments
--------
* 1st moment - `Arithmetic mean <https://ml-compiled.readthedocs.io/en/latest/statistics.html#arithmetic-mean>`_
* 2nd moment - `Variance <https://ml-compiled.readthedocs.io/en/latest/statistics.html#variance>`_
* 3rd moment - `Skewness <https://ml-compiled.readthedocs.io/en/latest/statistics.html#skewness>`_
* 4th moment - Kurtosis

Moving average
-----------------

Exponential moving average (EMA)
___________________________________

.. math::

  m_t = \alpha x_t + (1 - \alpha)m_{t-1}
  
:math:`0 \leq \alpha \leq 1` is a hyperparameter.
    
Point estimate
----------------
An estimate for a parameter.

Skewness
----------
Measures the asymmetry of a probability distribution.

Variance
---------

.. math::

  V(X) = E[(X-\mu)^2]
  
where :math:`\mu` is the mean of X.

Zipf distribution
---------------------
For a population of size n, the frequency of the kth most frequent item is:

.. math::

  \frac{1/{k^s}}{\sum_{i=1}^n 1/i^s}
  
where :math:`s \geq 0` is a hyperparameter
