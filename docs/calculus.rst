""""""""""""
Calculus
""""""""""""

Euler's method
=================
An iterative method for solving differential equations (ie integration).

Hessian matrix
====================
Let :math:`f:\mathbb{R}^n \rightarrow \mathbb{R}` be a function mapping vectors onto real numbers. Then the Hessian is defined as the matrix of second order partial derivatives:

.. math::

  H_{ij} = \frac{\partial^2 f}{\partial x_i x_j}

Applied to neural networks
---------------------------------
In the context of neural networks, :math:`f` is usually the loss function and :math:`x` is the parameter vector so we have:

.. math::

  H_{ij} = \frac{\partial^2 L}{\partial \theta_i \theta_j}

The size and therefore cost to compute of the Hessian is quadratic in the number of parameters. This makes it infeasible to compute for most problems. 

However, it is of theoretical interest as its properties can tell us a lot about the nature of the loss function we are trying to optimize:

* If the Hessian at a point on the loss surface has no negative eigenvalues the point is a local minimum.
* If the Hessian is `ill-conditioned <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#condition-number>`_, the loss function may be hard to optimize with gradient descent. 

Jacobian matrix
======================
Let :math:`f:\mathbb{R}^n \rightarrow \mathbb{R}^m` be a function. Then the Jacobian of :math:`f` can be defined as the matrix of partial derivatives:

.. math::

  J_{ij} = \frac{\partial f_i}{\partial x_j}

Applied to neural networks
---------------------------------
It is common in machine learning to compute the Jacobian of the loss function of a network with respect to its parameters. Then :math:`m = 1` and the Jacobian is a vector representing the gradients of the network:

.. math::

  J_i = \frac{\partial L}{\partial \theta_i}

Partial derivative
=====================
The derivative of a function of many variables with respect to one of those variables. 

The notation for the partial derivative of y with respect to x is :math:`\frac{\partial y}{\partial x}`

Rules of differentiation
========================

Sum rule
--------
.. math:: (f+g)' = f' + g'

Product rule
-------------
.. math:: (fg)' = fg' + f'g

Quotient rule
----------------
.. math:: (f/g)' = (f'g + fg')/g^2

Reciprocal rule
----------------
.. math:: (1/f)' = -f'/f^2

Power rule
------------
.. math:: (x^a)' = ax^{a-1}

Exponentials
--------------
.. math:: (a^{bx})' = a^{bx} \cdot b\log(a)

Logarithms
--------------
.. math:: (\log_a x)' = 1/(x \ln a)

Chain rule
----------------
.. math:: \frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}

Multivariate chain rule
------------------------
Used to calculate total derivatives.

.. math:: \frac{dy}{dx} = \frac{dy}{da} \cdot \frac{da}{dx} + \frac{dy}{db} \cdot \frac{db}{dx}

The derivative of a function wrt a function
-----------------------------------------------
Can be done using the chain rule. For example, :math:`\partial x^6/\partial x^2` can be found by setting :math:`y=x^6` and :math:`z=x^2`. Then do :math:`\partial y/\partial z = \partial y/\partial x \cdot \partial x/\partial z = 6x^5 \cdot 1/{2x} = 3x^4`.

Inverse relationship
------------------------
In general :math:`dy/dx` is the inverse of :math:`dx/dy`.

Matrix differentiation
-----------------------
TODO

Total derivative
======================
The derivative of a function of many arguments with respect to one of those arguments, taking into account any indirect effects via the other arguments.

The total derivative of :math:`z(x,y)` with respect to :math:`x` is:

.. math::

  \frac{dz}{dx} = \frac{\partial z}{\partial x} + \frac{\partial z}{\partial y} \frac{dy}{dx}
