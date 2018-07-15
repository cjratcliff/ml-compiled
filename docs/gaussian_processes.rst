"""""""""""""""""""""""""""
Gaussian processes
"""""""""""""""""""""""""""

Gaussian processes model a probability distribution over functions. 

Let :math:`f(x)` be some function mapping vectors to vectors. Then we can write:

.. math::

  f(x) \textasciitilde GP(m(x),k(x,x'))

where:

.. math::

  m(x) = \mathbb{E}[f(x)]
  
.. math::

  k(x,x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))^T]
