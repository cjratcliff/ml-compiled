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
A simple algorithm for exhaustively testing different combinations of hyperparameters.

It starts by manually specifying the hyperparameters to be evaluated. For example:

.. code-block:: none

    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    num_layers = [12, 16, 20, 24]
    
Then every combination is tested one by one by training a model with those settings and calculating the accuracy on the validation set.

Random search
----------------
A simple algorithm that tests random combinations of hyperparameters.

As with grid search, it begins by deciding the hyperparameters to be evaluated. For example:

.. code-block:: none

    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    num_layers = [12, 16, 20, 24]
    
Then random combinations of hyperparameters are chosen. For each one we train a model and calculate the accuracy on the validation set.

`Random Search for Hyper-Parameter Optimization, Bergstra and Bengio (2012) <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_

Reinforcement learning
-------------------------
Hyperparameter optimisation can be framed as a problem for reinforcement learning by letting the accuracy on the validation set be the reward and training with a standard algorithm like REINFORCE.

`Neural Architecture Search with Reinforcement Learning, Zoph and Le (2016) <https://arxiv.org/abs/1611.01578>`_