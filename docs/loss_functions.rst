===============
Loss functions
===============
For classification problems, :math:`y` is equal to 1 if the example is a positive and 0 if it is a negative. :math:`\hat{y}` can take on any value (although predicting outside of the (0,1) interval is unlikely to be useful).

""""""""""""""""
Contrastive loss
""""""""""""""""
Loss function for learning embeddings.

.. math::

  L(x_1,x_2,y) = (1-y_i)d(x_1,x_2) + y_i \max\{0, m - d(x_1,x_2)\}
  
Where :math:`x_1` and :math:`x_2` are the embeddings for the two examples and :math:`m` is a hyperparameter called the margin. :math:`d(x,y)` is a distance function, usually the Euclidean or cosine distance.

""""""""""""""""""""""""""""""""
Cross-entropy loss
""""""""""""""""""""""""""""""""
Loss function for classification.

.. math::

  L(y,\hat{y}) = -\sum_i \sum_c y_{i,c} \log(\hat{y}_{i,c})

where c are the classes.

""""""""""""""""
Hinge loss
""""""""""""""""
Loss function for classification.

.. math::

  L(y,\hat{y}) = \sum_i \max\{0, m - f(y_i)\hat{y}_i\}
  
Where :math:`f(y)` remaps y so that negatives are labelled as :math:`-1` rather than :math:`0`.

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

""""""""""""""""""""""""""""""""
Noise Contrastive Estimation
""""""""""""""""""""""""""""""""
Loss functions for efficient learning when the number of output classes is large. Useful for language modelling.

A binary classification task is created to disambiguate pairs that are expected to be close to each other from ‘noisy’ examples put together at random. Makes training time at the output layer independent of the number of classes. It remains linear in time at evaluation, however.

Embeddings
----------------------
When only learning embeddings a simpler formula can be used. It is:

.. math::

  L(a,b,y) = \sum_i y_i\log \sigma(a_i \cdot b_i) + (1-y_i)\log(1-\sigma(a_i \cdot b_i))

where :math:`a` and :math:`b` are embeddings and :math:`y = 1` if the pair :math:`(a,b)` are expected to be similar and :math:`y = 0` if not (because they have been sampled from the noise distribution). The dot product measures the distance between the two embeddings and the sigmoid function transforms it into a probability.

This means maximising the probability that actual samples are in the dataset and that noise samples aren’t in the dataset. Parameter update complexity is linear in the size of the vocabulary. The model is improved by having more noise than training samples, with around 15 times more being optimal.

Classification
----------------

  TODO  
      
k is a hyperparameter, denoting the number of noise samples for each real sample.

`Noise Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models, Gutmann and Hyvarinen (2010) <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`_

`Learning Word Embeddings Efficiently with Noise Contrastive Estimation, Mnih and Kavukcuoglu (2013) <https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation>`_

`RNNLM Training with NCE for Speech Recognition, Chen et al. (2015) <https://www.repository.cam.ac.uk/bitstream/handle/1810/247439/Chen_et_al-2015-ICASSP.pdf?sequence=1>`_

""""""""""""""""
Squared loss
""""""""""""""""
A loss function used for regression. 

.. math::

  L(y,\hat{y}) = \sum_i (y_i - \hat{y}_i)^2
  
""""""""""""""""
Triplet loss
""""""""""""""""
Used for training triplet networks. A triplet is composed of an anchor (:math:`a`), a positive example (:math:`p`) and a negative example (:math:`n`).

.. math::

  L(a,p,n) = \sum_i \max\{0, m - d(a_i,p_i) + d(a_i,n_i)\}
  
Where :math:`m` is a hyperparameter called the margin. :math:`d(x,y)` is a distance function, usually the Euclidean or cosine distance.
