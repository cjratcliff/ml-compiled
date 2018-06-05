""""""""""""""""""""""""""""""
Hyperparameter optimization
""""""""""""""""""""""""""""""

Cross-validation
------------------

k-fold cross validation
_________________________

.. code-block:: none

      1. Randomly split the dataset into K equally sized parts
      2. For i = 1,...,K
      3.     Train the model on every part apart from part i
      4.     Evaluate the model on part i
      5. Report the average of the K accuracy statistics

Grid search
-------------

Random search
----------------
