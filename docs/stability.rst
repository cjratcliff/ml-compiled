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
  
where :math:`x^* = \max{x_1, ..., x_n}`

Softmax
----------
For background read the main article on `softmax <https://ml-compiled.readthedocs.io/en/latest/activations.html#softmax>`_.

.. math:: 

    f(x)_j = \frac{e^{x_j}}{\sum_{k=1}^K e^{x_k}}, j \in {1,...,K}

Is equivalent to:

.. math::

    f(x)_j = \frac{x^* e^{x_j}}{x^* \sum_{k=1}^K e^{x_k}} = \frac{e^{x_j + \log x^*}}{\sum_{k=1}^K e^{x_k + \log x^*}}, j \in {1,...,K}

where :math:`x^* = -\max{x_1, ..., x_n}`


  
