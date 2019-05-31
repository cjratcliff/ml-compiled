"""""""""""""""""""""""""
SVMs
"""""""""""""""""""""""""

Support Vector Machines.

Kernels
----------
* Linear
* Polynomial
* Sigmoid
* RBF

Advantages
-------------

Disadvantages
----------------
* Cannot naturally learn multiclass classification problems. Applying an SVM to these requires reformulating the problem as a series of binary classification tasks, either :math:`n` one-vs-all or :math:`n^2` one-vs-one tasks. Learning these separately is inefficient and poor for generalisation.
