Ranking
""""""""""
Given a query retrieve the most relevant documents from a set. If the ranking is personalized a context including user history or location may also be taken into account. Often referred to as 'learning to rank'.

Inversion
-----------
An instance where two documents have been ranked in the wrong order given the ground truth. That is to say the less relevant document is ranked above the more relevant one.

LambdaLoss
------------

| **Proposed in**
`The LambdaLoss Framework for Ranking Metric Optimization, Wang et al. (2018) <https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf>`_


LambdaMART
------------
Combines the boosted tree model `MART (Friedman, 1999) <https://statweb.stanford.edu/~jhf/ftp/trebst.pdf>`_ with LambdaRank.

| **Further reading**
| `From RankNet to LambdaRank to LambdaMART: An Overview, Burges (2010) <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf>`_

LambdaRank
-----------

Builds upon RankNet. 

The loss is a function of the labeled relevance :math:`y` and the predicted score :math:`s`, summing over pairs of relevance labels where :math:`y_i > y_j`:

.. math::

  L(y,s) = \sum_{y_i > y_j} \Delta NDCG(i,j) \log(1 + \exp(-\sigma(s_i - s_j)))
  
where :math:`\Delta NDCG(i,j)` is the change in NDCG that would result from the ranking of documents i and j being swapped:

.. math::

  \Delta NDCG(i,j) = |G_i - G_j| |\frac{1}{D_i} - \frac{1}{D_j}|
  
:math:`G` and :math:`D` are the gain and discount functions:

.. math::

  G_i = \frac{2^{y_i} - 1}{maxDCG}
  
.. math::

  D_i = \log(1+i)

| **Proposed in**
| `Learning to Rank with Nonsmooth Cost Functions, Burges et al. (2006) <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`_
|
| **Further reading**
| `From RankNet to LambdaRank to LambdaMART: An Overview, Burges (2010) <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf>`_
`The LambdaLoss Framework for Ranking Metric Optimization, Wang et al. (2018) <https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf>`_

Listwise ranking
-----------------
The loss function is defined over the list of documents.

| **Example papers**

`Loss functions <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#ranking>`_
------------------------------------------------------------------------------------------------

Metrics
-----------------

See `the main section on metrics <https://ml-compiled.readthedocs.io/en/latest/metrics.html#ranking>`_ or jump to one of its subsections:

* `Cumulative Gain <https://ml-compiled.readthedocs.io/en/latest/metrics.html#cumulative-gain>`_
* `Discounted Cumulative Gain (DCG) <https://ml-compiled.readthedocs.io/en/latest/metrics.html#discounted-cumulative-gain-dcg>`_
* `Mean Reciprocal Rank (MRR) <https://ml-compiled.readthedocs.io/en/latest/metrics.html#mean-reciprocal-rank-mrr>`_
* `Normalized Discounted Cumulative Gain (NDCG) <https://ml-compiled.readthedocs.io/en/latest/metrics.html#normalized-discounted-cumulative-gain-ndcg>`_
* `Precision@k <https://ml-compiled.readthedocs.io/en/latest/metrics.html#precision-k>`_

Pairwise ranking
--------------------
Learning to rank is seen as a classification problem where the task is to predict whether a document A is more relevant than some other document B given a query.

Simple to train using the cross-entropy loss but requires more computational effort at inference time since there are :math:`O(n^2)` possible comparisons in a list of :math:`n` items.

| **Example papers**
| `Learning to Rank using Gradient Descent, Burges et al. (2005) <https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_
| `Learning to Rank with Nonsmooth Cost Functions, Burges et al. (2006) <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`_

Pointwise ranking
----------------------
Poses learning to rank as a regression problem where a relevance score must be predicted given a document and query.

| **Example papers**

RankNet
--------

A pairwise ranking algorithm. Can be built using any differentiable model such as neural networks or boosted trees. For a given pair of documents i and j the model computes the probability that i should be ranked higher than j:

.. math::

  P_{ij} = P(y_i > y_j) = \frac{1}{1 + \exp(-\sigma(s_i - s_j))}
  
Given the prediction, the model is then trained using the cross-entropy loss.

| **Proposed in**
| `Learning to Rank using Gradient Descent, Burges et al. (2005) <https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_
|
| **Further reading**
| `From RankNet to LambdaRank to LambdaMART: An Overview, Burges (2010) <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf>`_
