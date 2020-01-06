""""""""""""""""""""""""""""
Information theory and complexity
""""""""""""""""""""""""""""

Akaike Information Criterion (AIC)
------------------------------------
A measure of the quality of a model that combines accuracy with the number of parameters. Smaller AIC values mean the model is better. The formula is:

.. math::

  \text{AIC}(x,\theta) = 2|\theta| - 2 \ln L(\theta,x)
  
Where :math:`x` is the data and :math:`L` is the `likelihood function <https://ml-compiled.readthedocs.io/en/latest/probability.html#likelihood>`_.

Capacity
----------
The capacity of a machine learning model describes the complexity of the functions it can learn. If the model can learn highly complex functions it is said to have a high capacity. If it can only learn simple functions it has a low capacity.

Entropy
-------------
The entropy of a discrete probability distribution :math:`p` is:

.. math::

    H(X) = -\sum_{x \in X} p(x) \log p(x)

Joint entropy
_______________
Measures the entropy of a joint distribution.

.. math::

    H(X,Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(x,y)


Conditional entropy
_____________________

.. math::

    H(X|Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(y|x)


Finite-sample expressivity
----------------------------
The ability of a model to memorize the training set.

Fisher Information Matrix
---------------------------
An :math:`N \times N` matrix of second-order partial derivatives where :math:`N` is the number of parameters in a model.

The matrix is defined as:

.. math::

  I(\theta)_{ij} = E[\frac{\partial \log f(X;\theta)}{\partial \theta_i} \frac{\partial \log f(X;\theta)}{\partial \theta_j}|\theta]
  
The Fisher Information Matrix is equal to the negative expected `Hessian <https://ml-compiled.readthedocs.io/en/latest/calculus.html#hessian-matrix>`_ of the log likelihood.


Information bottleneck
-------------------------
An objective for training compressed representations.

.. math::

  \min I(X,T) - \beta I(T,Y)
  
Where :math:`I(X,T)` and :math:`I(T,Y)` represent the `mutual information <https://ml-compiled.readthedocs.io/en/latest/entropy.html#mutual-information>`_ between their respective arguments. :math:`X` is the input features, :math:`Y` is the labels and :math:`T` is a representation of the input such as the activations of a hidden layer in a neural network. :math:`\beta` is a hyperparameter controlling the trade-off between compression and predictive power.

When the expression is minimised there is very little mutual information between the compressed representation and the input. At the same time, there is a lot of mutual information between the representation and the output, meaning the representation is useful for prediction.

| **Proposed in**
| `The information bottleneck method, Tishby et al. (2000) <https://arxiv.org/pdf/physics/0004057.pdf>`_

Jensen-Shannon divergence
---------------------------
A symmetric version of the KL-divergence. This means that :math:`JS(P,Q) = JS(Q,P)`, which is not true for the KL-divergence.

.. math::

    JS(P,Q) = \frac{1}{2}(D_{KL}(P||M) + D_{KL}(M||Q))

where :math:`M` is a mixture distribution equal to :math:`\frac{1}{2}(P + Q)`

See also: `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/high_dimensionality.html#wasserstein-distance>`_
    
Kullback-Leibler divergence
----------------------------------
A measure of the difference between two probability distributions. Also known as the KL-divergence and the relative entropy. In the usual use case one distribution is the true distribution of the data and the other is an approximation of it. 

For discrete distributions it is given as:

.. math::

    D_{KL}(P||Q) = -\sum_i P_i \log \frac{Q_i}{P_i}

Note that if a point is outside the support of Q (:math:`Q_i = 0`), the KL-divergence will explode since :math:`\log (0)` is undefined. This can be dealt with by adding some random noise to Q. However, this introduces a degree of error and a lot of noise is often needed for convergence when using the KL-divergence for MLE. The `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/high_dimensionality.html#wasserstein-distance>`_, which also measures the distance between two distributions, does not have this problem.

Properties
______________

* The KL-divergence is not symmetric.
* A KL-Divergence of 0 means the distributions are identical. As the distributions become more different the divergence becomes more negative.

Mutual information
-----------------------
Measures the dependence between two random variables.

.. math::

    I(X,Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
   
If the variables are independent :math:`I(X,Y) = 0`. If they are completely dependent :math:`I(X,Y) = H(X) = H(Y)`.
   
Rademacher complexity
-------------------------
TODO

Total variation distance
-----------------------------
Like the Kullback-Leibler divergence, it is also a way of measuring the difference between two different probability distributions.

See also: `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/geometry.html#wasserstein-distance>`_

VC dimension
--------------
Vapnik–Chervonenkis dimension is a measure of the `capacity <https://ml-compiled.readthedocs.io/en/latest/entropy.html#capacity>`_ of a model.
