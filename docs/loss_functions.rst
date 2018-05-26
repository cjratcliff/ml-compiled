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

Sometimes referred to as the negative log-likelihood loss.

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

"""""""""""""""""""""""""""""
Negative sampling
"""""""""""""""""""""""""""""
A technique for efficient learning when the number of output classes is large. Useful for language modelling. The softmax over the vocabulary is removed and the problem is reframed as a binary classification problem.

Embeddings
------------

.. math::

  L(x_0,x_1,y) = y\log \sigma(f(x_0) \cdot f(x_1)) + (1-y_i)\log(\sigma(-f(x_0) \cdot f(x_1)))
  
where :math:`x_0` and :math:`x_1` are two examples, :math:`f` is the learned embedding function and :math:`y = 1` if the pair :math:`(x_0,x_1)` are expected to be similar and :math:`y = 0` otherwise. The dot product measures the distance between the two embeddings.

""""""""""""""""""""""""""""""""
Noise Contrastive Estimation
""""""""""""""""""""""""""""""""
Like negative sampling, this is a technique for efficient learning when the number of output classes is large. Useful for language modelling.

A binary classification task is created to disambiguate pairs that are expected to be close to each other from ‘noisy’ examples put together at random. 

In essence, rather than estimating :math:`P(y|x)`, NCE estimates :math:`P(C=1|x,y)` where :math:`C = 1` if :math:`y` has been sampled from the real distribution and :math:`C = 0` if :math:`y` has been sampled from the noise distribution.

NCE makes training time at the output layer independent of the number of classes. It remains linear in time at evaluation, however.

Embeddings
----------------------
When only learning embeddings a simpler formula can be used. It is:

.. math::

  L(x_0,x_1,y) = y\log \sigma(f(x_0) \cdot f(x_1)) + (1-y_i)\log(1-\sigma(f(x_0) \cdot f(x_1)))

where :math:`x_0` and :math:`x_1` are two examples, :math:`f` is the learned embedding function and :math:`y = 1` if the pair :math:`(x_0,x_1)` are expected to be similar and :math:`y = 0` if not (because they have been sampled from the noise distribution). The dot product measures the distance between the two embeddings and the sigmoid function transforms it to be between 0 and 1 so it can be interpreted as a prediction for a binary classifier.

This means maximising the probability that actual samples are in the dataset and that noise samples aren’t in the dataset. Parameter update complexity is linear in the size of the vocabulary. The model is improved by having more noise than training samples, with around 15 times more being optimal.

Classification
----------------
When learning for multi-class classification rather than only embeddings the formula is:

.. math::

  L(x,y) = -\sum_i \log(P(C_i=1|x_i,y_i)) + \sum_{j = 1}^k \log(1 - P(C_i=1|x_i,y^n_j))
      
:math:`k` is a hyperparameter, denoting the number of noise samples for each real sample. :math:`y_i` is a label sampled from the data distribution and :math:`y^n_j` is one sampled from the noise distribution. :math:`C_i = 1` if the pair :math:`(x,y)` was drawn from the data distribution and 0 otherwise.

`Noise Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models, Gutmann and Hyvarinen (2010) <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`_

`Learning Word Embeddings Efficiently with Noise Contrastive Estimation, Mnih and Kavukcuoglu (2013) <https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation>`_

`RNNLM Training with NCE for Speech Recognition, Chen et al. (2015) <https://www.repository.cam.ac.uk/bitstream/handle/1810/247439/Chen_et_al-2015-ICASSP.pdf?sequence=1>`_

""""""""""""""""
Squared loss
""""""""""""""""
A loss function used for regression. 

.. math::

  L(y,\hat{y}) = \sum_i (y_i - \hat{y}_i)^2
  
Disadvantages
---------------
The squaring means this loss function weights large errors more than smaller ones, relative to the magnitude of the error. This can be particularly harmful in the case of outliers.
  
""""""""""""""""
Triplet loss
""""""""""""""""
Used for training embeddings with triplet networks. A triplet is composed of an anchor (:math:`a`), a positive example (:math:`p`) and a negative example (:math:`n`). The positive examples are similar to the anchor and the negative examples are dissimilar.

.. math::

  L(a,p,n) = \sum_i \max\{0, m - d(a_i,p_i) + d(a_i,n_i)\}
  
Where :math:`m` is a hyperparameter called the margin. :math:`d(x,y)` is a distance function, usually the Euclidean or cosine distance.
