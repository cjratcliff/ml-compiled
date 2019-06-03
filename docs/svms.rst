"""""""""""""""""""""""""
SVMs
"""""""""""""""""""""""""

Support Vector Machines. 

Binary classifier. Their objective is to find a hyperplane that optimally separates the two classes (maximises the margin).

Kernels
----------
The kernel is used to map the data into a high-dimensional space in which it is easier to separate it linearly. This is known as the 'kernel trick'.

Linear
_______

.. math::

  k(x,y) = x \cdot y

Polynomial
_____________

.. math::

  k(x,y) = (a x \cdot y + b)^d

Sigmoid
________

.. math::

  k(x,y) = \tanh(a x \cdot y + b)


RBF
______

.. math::

  k(x,y) = \exp (-||x-y||^2/2 \sigma^2)



Advantages
-------------
* The optimisation problem is convex so local optima are not a problem.

Disadvantages
----------------
* Cannot naturally learn multiclass classification problems. Applying an SVM to these requires reformulating the problem as a series of binary classification tasks, either :math:`n` one-vs-all or :math:`n^2` one-vs-one tasks. Learning these separately is inefficient and poor for generalisation.


`One-Class SVMs <https://ml-compiled.readthedocs.io/en/latest/density_estimation.html#one-class-svm>`_
---------------------------------------------------------------------------------------------------------
