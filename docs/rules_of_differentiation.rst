============
Rules of differentiation
============

Sum rule
.. math::
    `(f+g)' = f' + g'`

Product rule
.. math::
    (fg)' = fg' + f'g

Quotient rule
.. math::
    (f/g)' = (f'g + fg')/g^2

Reciprocal rule
.. math::
    (1/f)' = -f'/f^2

Chain rule
.. math::
    \frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}

Multivariate chain rule
.. math::
    \frac{dy}{dx} = \frac{dy}{da} \cdot \frac{da}{dx} + \frac{dy}{db} \cdot \frac{db}{dx}

Used to calculate total derivatives.

Function wrt function
Can be done using the chain rule. For example, $\partial x^6/\partial x^2$ can be found by setting $y=x^6$ and $z=x^2$. Then do $\partial y/\partial z = \partial y/\partial x \cdot \partial x/\partial z = 6x^5 \cdot 1/{2x} = 3x^4$.

Inverse relationship
In general dy/dx is the inverse of dx/dy. However there are some conditions attached (unknown at the time of writing).
