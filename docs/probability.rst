Probability
"""""""""""""

“Admits a density/distribution”
---------------------------------
If a variable ‘admits a distribution’, that means it can be described by a probability density function. Contrast with

.. math::

  P(X=a) = 0.5 if a is 0 or 1, 0 otherwise.

which cannot be described by a pdf.

Bayesian network
------------------

Bayes' rule
-------------

.. math::

  P(A|B) = \frac{P(B|A)P(A)}{P(B)}

Chain rule of probability
---------------------------

Conjugate prior
----------------

Improper prior
----------------
A prior whose probability distribution has infinitesimal density over an infinitely large range. For example, the distribution for picking an integer at random.

Informative and uninformative priors
---------------------------------------
Examples:

Informative:
* The temperature is normally distributed with mean 20 and variance 3.

Uninformative:
* The temperature is positive.
* The temperature is less than 200.
* All temperatures are equally likely.

'Uninformative' can be a misnomer. 'Not very informative' would be more accurate.

Likelihood
-----------

Marginal likelihood
----------------------

MAP estimation
----------------

Maximum likelihood estimation (MLE)
-------------------------------------

Prior
------

Posterior
----------
