Statistics
"""""""""""""

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
    
Point estimate
----------------
An estimate for a parameter.
