"""""""""""""""""""""""""
Decision Trees
"""""""""""""""""""""""""

Advantages
------------
* Easy to interpret.
* Efficient inference. Inference time is proportional to the depth of the tree, not its size.
* No risk of local optima during training. 

Disadvantages
--------------
* Can easily overfit.
* Requires a quantity of data that is exponential in the depth of the tree. This means learning a complex function can require a prohibitive amount of data.

Training
---------------
The simplest approach for training decision trees is:

* At each node find the optimal variable to split on. This is the variable whose split yields the largest information gain (decrease in entropy).

Regularization
----------------
Some options for avoiding overfitting when using decision trees include:

* Specifying a maximum depth for the tree
* Setting a minimum number of samples to create a split.
