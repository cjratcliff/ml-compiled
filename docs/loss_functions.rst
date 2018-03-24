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
A method used for learning language models over large vocabularies efficiently. A binary classification task is created to disambiguate groups of words that are actually near each other from ‘noisy’ words put together at random. Makes training time at the output layer independent of vocabulary size. It remains linear in time at evaluation, however.

.. math::

    L() = -1/N_w \sum_{i=1}^{N_w}\ln P(C_{w_i}^{RNN}=1|w_i,h_i) + \sum_{j=1}^k \ln P(C^n_{w_{ij}}=1|w_{ij},h_i)

where

.. math::

    P(C_{w_i}^{RNN}=1|w_i,h_i) = \frac{P^{NCE}_{RNN}(w|h_i)}{P^{NCE}_{RNN}(w|h_i) + kP_n(w|h_i)}

    P(C_{w}^{n}=1|w_i,h_i) = \frac{kP_n(w|h_i)}{P^{NCE}_{RNN}(w|h_i) + kP_n(w|h_i)})      
      
$w_{ij}$ is the word sampled from the noise distribution for the training set word 

Alternatives are hierarchical softmax and self-normalizing partition functions. Simpler to implement than the former. Appears to be more widely used than the latter, which only works with certain loss functions.

Experimental comparison
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
