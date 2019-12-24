Stability
""""""""""""
This page gives some numerically stable equivalents for common formulae.

Log-sum-exponential
---------------------

.. math::

  LSE(x_1, ..., x_n) = \log \sum_{i=1}^n \exp(x_i)
  
Is equivalent to:

.. math::

  LSE(x_1, ..., x_n) = x^* +  \log \sum_{i=1}^n \exp(x_i - x^*)
  
where :math:`x^* = \max{x_1, ..., x_n}
  
