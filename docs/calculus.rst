""""""""""""
Calculus
""""""""""""

Euler's method
=================
An iterative method for solving differential equations (ie integration).

Hessian matrix
====================

Square matrix of second-order partial derivatives of a scalar-valued function. Its size and therefore cost to compute is quadratic in the number of parameters. This makes it infeasible to compute for most problems. If the Hessian at a point on the loss surface has no negative eigenvalues the point is a local minimum.

Jacobian matrix
======================

Matrix of all first-order derivatives of a vector-valued function. Singular values are important.

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

Chain rule
----------------
.. math:: \frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}

Multivariate chain rule
------------------------
Used to calculate total derivatives.

.. math:: \frac{dy}{dx} = \frac{dy}{da} \cdot \frac{da}{dx} + \frac{dy}{db} \cdot \frac{db}{dx}

Function wrt function
------------------------
Can be done using the chain rule. For example, :math:`\partial x^6/\partial x^2` can be found by setting :math:`y=x^6` and :math:`z=x^2`. Then do :math:`\partial y/\partial z = \partial y/\partial x \cdot \partial x/\partial z = 6x^5 \cdot 1/{2x} = 3x^4`.

Inverse relationship
------------------------
In general dy/dx is the inverse of dx/dy. However there are some conditions attached (unknown at the time of writing).

Matrix differentiation
-----------------------
TODO

Total derivative
======================
The derivative of a function of many arguments with respect to one of those arguments, taking into account any indirect effects via the other arguments. The total derivative of y with respect to x is written as :math:`\frac{dy}{dx}`.
