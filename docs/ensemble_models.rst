""""""""""""""""""""""""
Ensemble models
""""""""""""""""""""""""

Bagging
--------
A way to reduce overfitting by building several models independently and averaging their predictions. As bagging reduces variance it is well suited to models like decision trees that are complex and prone to overfitting.

Boosting
----------
Build models sequentially, each one trying to reduce the bias of the combined estimator. AdaBoost is an example.

Random forest
---------------
Each tree is built from a sample drawn with replacement. Randomises how splits in the tree are chosen (rather than simply choosing the best), decreasing variance at the expense of bias, but with a positive overall effect on accuracy.
