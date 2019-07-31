""""""""""""""""""""""""""""
Information theory and complexity
""""""""""""""""""""""""""""

Capacity
----------
The capacity of a machine learning model describes the complexity of the functions it can learn.

Entropy
-------------
The entropy of a discrete probability distribution :math:`p` is:

.. math::

    H(X) = -\sum_{x \in X} p(x) \log p(x)


Joint entropy
_______________

.. math::

    H(X,Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(x,y)


Conditional entropy
_____________________

.. math::

    H(X|Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(y|x)
    
Jensen-Shannon divergence
---------------------------
Symettric version of the KL-divergence.

.. math::

    JS(P,Q) = \frac{1}{2}(D_{KL}(P||M) + D_{KL}(M||Q))

where :math:`M` is a mixture distribution equal to :math:`\frac{1}{2}(P + Q)`

See also: `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/high_dimensionality.html#wasserstein-distance>`_
    
Kullback-Leibler divergence
----------------------------------
A measure of the difference between two probability distributions. Also known as the relative entropy. In the usual use case one distribution is the true distribution of the data and the other is a model of it. 

For discrete distributions it is given as:

.. math::

    D_{KL}(P||Q) = -\sum_i P_i \log \frac{Q_i}{P_i}

Note that if a point is outside the support of Q (:math:`Q_i = 0`), the KL-divergence will explode. This can be dealt with by adding some random noise to Q. However, this introduces a degree of error and a lot of noise is often needed for convergence when using the KL-divergence for MLE. The `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/high_dimensionality.html#wasserstein-distance>`_, which also measures the distance between two distributions, does not have this problem.

The KL-divergence is not symmetric.

Mutual information
-----------------------
Measures the dependence of two random variables.

.. math::

   I(X,Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
   
Rademacher complexity
-------------------------
TODO

Total variation distance
-----------------------------
Like the Kullback-Leibler divergence, it is also a way of measuring the difference between two different probability distributions.

See also: `Wasserstein distance <https://ml-compiled.readthedocs.io/en/latest/geometry.html#wasserstein-distance>`_

VC dimension
--------------
Vapnik–Chervonenkis dimension is a measure of capacity.
