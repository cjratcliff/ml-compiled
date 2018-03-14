===============
Loss functions
===============

""""""""""""""""
Contrastive loss
""""""""""""""""

""""""""""""""""""""""""""""""""
Cross-entropy loss
""""""""""""""""""""""""""""""""

""""""""""""""""
Hinge loss
""""""""""""""""

""""""""""""""""
Huber loss
""""""""""""""""
A loss function used for regression. It is less sensitive to outliers than the squared loss.

Definition

.. math::

  L(e;\delta) = 
          \begin{cases}
              \frac{1}{2}e^2, & \ |e| \leq \delta \\
              \delta(|e| - \frac{1}{2}\delta), & \text{otherwise}
          \end{cases}

""""""""""""""""
Squared loss
""""""""""""""""
A loss function used for regression. 

.. math::

  L(y,\^{y}) = \sum_i (y - \^{y})^2
