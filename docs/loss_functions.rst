===============
Loss functions
===============

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

  L(y,\hat{y}) = -\sum_i y_i \log(\hat{y}_i)


""""""""""""""""
Hinge loss
""""""""""""""""
Loss function for classification.


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

Learning embeddings
----------------------
When only learning embeddings a simpler formula can be used. It is:

.. math::

  L(a,b,y) = \sum_{i=1}^n y_i\log \sigma(a_i \cdot b_i) + (1-y_i)\log(1-\sigma(a_i \cdot b_i))

where :math:`a` and :math:`b` are embeddings and :math:`y = 1` if the pair :math:`(a,b)` are expected to be similar and :math:`y = 0` if not (because they have been sampled from the negative distribution). The dot product measures the distance between the two embeddings and the sigmoid function transforms it into a probability.

This means maximising the probability that actual samples are in the dataset and that noise samples aren’t in the dataset. Parameter update complexity is linear in the size of the vocabulary. The model is improved by having more noise than training samples, with around 15 times more being optimal.

Classification
----------------

.. math::

    L() = -1/N_w \sum_{i=1}^{N_w}\log P(C_{w_i}^{RNN}=1|w_i,h_i) + \sum_{j=1}^k \log P(C^n_{w_{ij}}=1|w_{ij},h_i)

where

.. math::

    P(C_{w_i}^{RNN}=1|w_i,h_i) = \frac{P^{NCE}_{RNN}(w|h_i)}{P^{NCE}_{RNN}(w|h_i) + kP_n(w|h_i)}

    P(C_{w}^{n}=1|w_i,h_i) = \frac{kP_n(w|h_i)}{P^{NCE}_{RNN}(w|h_i) + kP_n(w|h_i)})      
      
$w_{ij}$ is the word sampled from the noise distribution for the training set word 

Explanation for learning embeddings, Newell (2016)
RNNLM Training with NCE for Speech Recognition, Chen et al. (2015)
Noise Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models, Gutmann and Hyvarinen (2010)
Learning Word Embeddings Efficiently with Noise Contrastive Estimation, Mnih and Kavukcuoglu (2013)

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
