""""""""""""""
Entropy
""""""""""""""

The entropy of a discrete probability distribution :math:`p` is:

.. math::

    H(X) = -\sum_{x \in X} p(x) \log p(x)


Joint entropy
-----------------

.. math::

    H(X,Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(x,y)


Conditional entropy
---------------------

.. math::

    H(X|Y) = -\sum_{x \in X} \sum_{y \in Y} p(x,y) \log p(y|x)
