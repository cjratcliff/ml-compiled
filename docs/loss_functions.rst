===============
Loss functions
===============

""""""""""""""""
Contrastive loss
""""""""""""""""

""""""""""""""""""""""""""""""""
Cross-entropy loss
""""""""""""""""""""""""""""""""
Loss function for classification.

.. math::

  L(y,\hat{y}) = -\sum_i y_i \log(\hat{y}_i)


""""""""""""""""
Hinge loss
""""""""""""""""

""""""""""""""""
Huber loss
""""""""""""""""
A loss function used for regression. It is less sensitive to outliers than the squared loss.

.. math::

  L(y,\hat{y};\delta) = 
          \begin{cases}
              \frac{1}{2}(y_i - \hat{y}_i)^2, & \ |y_i - \hat{y}_i| \leq \delta \\
              \delta(|y_i - \hat{y}_i| - \frac{1}{2}\delta), & \text{otherwise}
          \end{cases}

""""""""""""""""
Squared loss
""""""""""""""""
A loss function used for regression. 

.. math::

  L(y,\hat{y}) = \sum_i (y_i - \hat{y}_i)^2
  
""""""""""""""""
Triplet loss
""""""""""""""""
Used for training triplet networks. A triplet is composed of an anchor, a positive example and a negative example.

.. math::

  L = \sum_i \max(0, m - d(a_i,p_i) + d(a_i,n_i))
  
Where :math:`m` is a hyperparameter called the margin.
