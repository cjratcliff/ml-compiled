""""""""""""""""""""""""""""""
Hyperparameter optimization
""""""""""""""""""""""""""""""
A hyperparameter is a parameter of the model which is set according to the design of the model rather than learnt through the training process. Examples of hyperparameters include the learning rate, the dropout rate and the number of layers. Since they cannot be learnt by gradient descent hyperparameter optimization is a difficult problem.

`Gaussian processes <https://ml-compiled.readthedocs.io/en/latest/gaussian_processes.html#gaussian-processes>`_

Bayesian optimization
----------------------

Note that much of the below explanation references states. These are irrelevant for hyperparameter optimisation since each training run is initialized in the same way.

Acquisition function
_________________________
A function that decides the next point to sample while trying to maximize the cumulative reward, balancing exploration and exploitation.

https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

Probability of improvement
'''''''''''''''''''''''''''
Pick the action which maximises the chance of getting to a state with a value greater than the current best state. The reward is 1 if the new state is better and 0 otherwise. This means that it will eschew possible large improvements in favour of more certain small ones.

If all nearby states are known to be worse this strategy can lead to getting stuck in local optima.

Expected improvement
''''''''''''''''''''''
Pick the action which maximises the expected improvement of that new state over the current best state. The reward is the difference between the values if the new state is better than the old one and zero otherwise.

A higher expected improvement can be obtained either by increasing either the variance or the mean of the value distribution of the next state.

Upper confidence bound
'''''''''''''''''''''''''''
Calculate the upper bound of the confidence interval for the rewards from each action in a given state. Pick the action for which the upper bound of the reward is greatest. This will lead to actions with greater uncertainty being chosen since their confidence interval will be larger.

Using a Gaussian distribution gives a simple expression for the bound, that it is :math:`\beta` standard deviations away from the mean of the distribution of rewards given an action in some state:

.. math::

  UCB_{s,a} = \mu_{s,a} + \beta \sigma_{s,a}
  
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

Effectiveness
________________
Grid search is believed to be less efficient than random search, particularly when tuning a large number of parameters. (`Bergstra and Bengio (2012) <http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf>`_. 

The reasoning is that typically in neural networks a few hyperparameters matter a great deal and most do not change the results much. Grid search looks at exponentially fewer values of each hyperparameter than random search does. In essence, grid search wastes far more time evaluating combinations of variables that don't matter.

Neural Architecture Search
----------------------------
The automatic design of the architecture of neural networks. Typically involves deciding aspects like the size, connections and type of layers as well as their activations.

| **Notable papers**
| `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Tan and Le (2019) <https://arxiv.org/abs/1905.11946>`_
| `Efficient Neural Architecture Search via Parameter Sharing, Pham et al. (2018) <https://arxiv.org/abs/1802.03268>`_
| `Regularized Evolution for Image Classifier Architecture Search, Real et al. (2018) <https://arxiv.org/abs/1802.01548>`_
| `Learning Transferable Architectures for Scalable Image Recognition, Zoph et al. (2017) <https://arxiv.org/pdf/1707.07012.pdf>`_
| `Neural Architecture Search with Reinforcement Learning, Zoph and Le (2016) <https://arxiv.org/abs/1611.01578>`_

Random search
----------------
A simple algorithm that tests random combinations of hyperparameters.

As with grid search, it begins by deciding the hyperparameters to be evaluated. For example:

.. code-block:: none

    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    num_layers = [12, 16, 20, 24]
    
Then random combinations of hyperparameters are chosen. For each one we train a model and calculate the accuracy on the validation set.

Extremely simple to implement and easy to parallelize.

`Random Search for Hyper-Parameter Optimization, Bergstra and Bengio (2012) <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_

Reinforcement learning
-------------------------
Hyperparameter optimisation can be framed as a problem for reinforcement learning by letting the accuracy on the validation set be the reward and training with a standard algorithm like REINFORCE.

| `Neural Architecture Search with Reinforcement Learning, Zoph and Le (2016) <https://arxiv.org/abs/1611.01578>`_
| `Efficient Neural Architecture Search via Parameter Sharing, Pham et al. (2018) <https://arxiv.org/abs/1802.03268>`_
