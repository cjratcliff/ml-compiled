""""""""""""""""""""""""
Ensemble models
""""""""""""""""""""""""

AdaBoost
---------
A boosting algorithm. Each classifier in the ensemble attempts to correctly predict the instances misclassified by the previous iteration.

Decision trees are often used as the weak learners.

Bagging
--------
A way to reduce overfitting by building several models independently and averaging their predictions. As bagging reduces variance it is well suited to models like decision trees as they are prone to overfitting.

Boosting
----------
Build models sequentially, each one trying to reduce the bias of the combined estimator. AdaBoost is an example.

Gradient boosting
___________________
Learns a weighted sum of weak learners:

.. math::

  \hat{y}_i = \sum_{i=1}^M \gamma_i h_i(x)
  
where :math:`\gamma_i` is the weight associated with the weak learner :math:`h_i`. :math:`M` is the total number of weak learners.

The first learner predicts a constant for all examples. All subsequent learners try to predict the residuals.

Can learn with any differentiable loss function. 

Weak learner
--------------
The individual algorithms that make up the ensemble.

