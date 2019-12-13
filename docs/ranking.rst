Ranking
""""""""""
Given a query retrieve the most relevant documents from a set. If the ranking is personalized a context including user history or location may also be taken into account.

LambdaMART
------------
Combines the boosted tree model `MART (Friedman, 1999) <https://statweb.stanford.edu/~jhf/ftp/trebst.pdf>`_ with LambdaRank.

LambdaRank
-----------

| **Proposed in**
| `Learning to Rank with Nonsmooth Cost Functions, Burges et al. (2006) <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`_

Listwise ranking
-----------------

`Loss functions <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#ranking>`_
------------------------------------------------------------------------------------------------

`Metrics <https://ml-compiled.readthedocs.io/en/latest/metrics.html#ranking>`_
-----------------

Pairwise ranking
--------------------
Learning to rank is seen as a classification problem where the task is to predict whether a document A is more relevant than some other document B given a query. 

Pointwise ranking
----------------------
Poses learning to rank as a regression problem where a relevance score must be predicted given a document and query.

RankNet
--------

A pairwise ranking algorithm.

| **Proposed in**
| `Learning to Rank using Gradient Descent, Burges et al. (2005) <https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_
