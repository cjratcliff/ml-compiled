"""""""""""""""""""""""""""
Gaussian processes
"""""""""""""""""""""""""""

Gaussian processes model a probability distribution over functions. 

Let :math:`f(x)` be some function mapping vectors to vectors. Then we can write:

.. math::

  f(x) ~ GP(m(x),k(x,x'))

where :math:`m(x)` represents the mean vector:

.. math::

  m(x) = \mathbb{E}[f(x)]
  
and :math:`k(x,x')` is the kernel function:
  
.. math::

  k(x,x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))^T]
  
The kernel function
----------------------
The kernel represents the covariance function for the Gaussian process and can be thought of as a prior for the shape of the function.

Linear kernel
_______________

.. math::

  k(x,x') = x \cdot x'
  
Polynomial kernel
___________________

.. math::

  k(x,x') = (x \cdot x' + a)^b
  
Gaussian kernel
________________

.. math::

  k(x,x') = \exp{{-||x - x'||}_2^2}
  
Laplacian kernel
_________________

.. math::

  k(x,x') = \exp{{-||x - x'||}_2}
  
Gaussian process regression
------------------------------
