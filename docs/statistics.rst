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

Covariance
-------------

.. math::

  \text{Cov}(X,Y) = \frac{1}{n}\sum_{i=1}^n (x_i - \mu_x)(y_i - \mu_y)

Covariance matrix
________________________
A square matrix :math:`\Sigma` where :math:`\Sigma_{ij} = Cov(X_i,X_j)` and :math:`X_i` and :math:`X_j` are two variables.

There are three types of covariance matrix:

* Full - All entries are specified. Has :math:`O(n^2)` parameters for :math:`n` variables.
* Diagonal - The matrix is diagonal, meaning all off-diagonal entries are zero. Variances can differ across dimensions but there is no interplay between the dimensions. Has :math:`O(n)` parameters.
* Spherical - The matrix is equal to the identity matrix multiplied by a constant. This means the variance is the same in all dimensions. Has :math:`O(1)` parameters.

A valid covariance matrix is always symmetric and positive semi-definite.

Geometric mean
----------------

.. math::

    G(x_1,x_2,...,x_n) = \sqrt[\leftroot{-2}\uproot{2}n]{x_1x_2...x_n}

Only applicable to positive numbers since otherwise it may involve taking the root of a negative number.

Harmonic mean
---------------

.. math::

    H(x_1,x_2,...,x_n) = n/\sum_{i=1}^n \frac{1}{x_i}
    
Cannot be computed if one of the numbers is zero since that would necessitate dividing by zero.
    
Heteroscedasticity
--------------------
When the error of a model is correlated with one or more of the features.
    
Moments
--------
* 1st moment - `Arithmetic mean <https://ml-compiled.readthedocs.io/en/latest/statistics.html#arithmetic-mean>`_
* 2nd moment - `Variance <https://ml-compiled.readthedocs.io/en/latest/statistics.html#variance>`_
* 3rd moment - `Skewness <https://ml-compiled.readthedocs.io/en/latest/statistics.html#skewness>`_
* 4th moment - Kurtosis

Moving average
-----------------
A moving average smooths a sequence of observations.

Exponential moving average (EMA)
___________________________________
A type of moving average in which the influence of past observations on the current average diminishes exponentially with time.

.. math::

  m_t = \alpha m_{t-1} + (1 - \alpha) x_t
  
:math:`m_t` is the moving average at time :math:`t`, :math:`x_t` is the input at time :math:`t` and :math:`0 < \alpha < 1` is a hyperparameter. As :math:`\alpha` decreases, the moving average weights recent observations more strongly.

Bias correction
==================
If we initialise the EMA to equal zero (:math:`m_0 = 0`) it will be very biased towards zero around the start. To correct this we can start with :math:`\alpha` being close to 0 and gradually increase it. This effect can be achieved by rewriting the formula as:

.. math::

  m_t = \frac{1}{1 - \alpha^t}(\alpha m_{t-1} + (1 - \alpha) x_t)

See `Adam: A Method for Stochastic Optimization, Kingma et al. (2015) <https://arxiv.org/pdf/1412.6980.pdf>`_ for an example of this bias correction being used in practice.
    
Point estimate
----------------
An estimate for a parameter.

Skewness
----------
Measures the asymmetry of a probability distribution.

.. math::
  = E\bigg[\bigg(\frac{X - \mu}{\sigma}\bigg)^3\bigg]
  
Standard deviation
--------------------
The square root of the variance. The formula is:

.. math::

  \sigma = \sqrt{E[(X-\mu)^2]}
  
where :math:`\mu` is the mean of X.
  
Sample standard deviation
_____________________________

.. math::

  s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n(x_i-\mu)^2}
  
Note that the above is the biased estimator for the sample standard deviation. Estimators which are unbiased exist but they each only apply to some population distributions.

Variance
---------
The variance of :math:`X=\{x_1, ..., x_n\}` is:

.. math::

  V(X) = E[(X-\mu)^2]
  
where :math:`\mu` is the mean of X.
  
The formula can also be written as:

.. math::

  V(X) = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2

Sample variance
__________________
When it is impractical to compute the variance over the entire population, we can take a sample instead and compute the sample variance. The formula for the unbiased sample variance is:

.. math::

  V(X) = \frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2

