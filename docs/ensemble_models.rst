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
Can learn with any differentiable loss function.

Weak learner
--------------
The individual algorithms that make up the ensemble.

