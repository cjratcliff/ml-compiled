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


Regularization
----------------
Some options for avoiding overfitting when using decision trees include:

* Specifying a maximum depth for the tree
* Setting a minimum number of samples to create a split.
