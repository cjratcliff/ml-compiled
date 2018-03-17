Probability
"""""""""""""

“Admits a density/distribution”
---------------------------------
If a variable ‘admits a distribution’, that means it can be described by a probability density function. Contrast with

.. math::

  P(X=a) = 
    \begin{cases} 
      1 ,& \text{if } a = 0 \\
      0 ,& \text{otherwise}
    \end{cases}

which cannot be described by a pdf.

Bayesian network
------------------
A directed acyclic graph where the nodes represent random variables.

The chain rule for Bayesian networks
''''''''''''''''''''''''''''''''''''''''''

The joint distribution for all the variables in a network is equal to the product of the distributions for all the individual variables, conditional on their parents.

.. math::

    P(X_1,...,X_n) = \prod_i P(X_i|Par(X_i))

where :math:`Par(X_i)` denotes the parents of :math:`X_i`

Bayes' rule
-------------

.. math::

  P(A|B) = \frac{P(B|A)P(A)}{P(B)}

Chain rule of probability
---------------------------

.. math::
  P(A_3,A_2,A_1) = P(A_3|A_2,A_1) \cdot P(A_2|A_1) \cdot P(A_1)

Conjugate prior
----------------
A prior for a likelihood function is conjugate if it is from the same family of distributions (eg Gaussian) as the posterior.

====================== ======================
 Likelihood             Conjugate prior
====================== ======================
 Bernoulli               Beta
 Binomial                Beta
 Negative binomial        Beta
 Categorical               Dirichlet
 Multinomial              Dirichlet
 Poisson                  Gamma
====================== ======================

Improper prior
----------------
A prior whose probability distribution has infinitesimal density over an infinitely large range. For example, the distribution for picking an integer at random.

Inference
-----------
Probabilistic inference is the task of determining the probability of a particular outcome.

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

Law of total probability
--------------------------

.. math::

  P(X) = \sum_i P(X|Y=y_i)P(Y=y_i)

Likelihood
-----------
The likelihood of the parameters given the data is equal to the probability of the data given the parameters.

.. math::

    L(\theta|O) = P(O|\theta)


Marginal distribution
---------------------------------------
The most basic sort of probability, :math:`P(x)`. Contrast with the conditional distribution :math:`P(x|y)` or the joint :math:`P(x,y)`.


Marginal likelihood
----------------------
A likelihood function in which some variable has been marginalised out (removed by summation).

MAP estimation
----------------
A type of point estimate. Can be seen as a regularization of MLE since it also incorporates a prior distribution. Uses Bayes rule to incorporate a prior over the parameters and find the parameters that are most likely given the data (rather than the other way around). Unlike with MLE (which is a bit of a simplification), the most likely parameters given the data are exactly what we want to find.

.. math::

    \hat{\theta}_{MAP}(O) = \arg \max_\theta p(\theta|O) = \arg \max_\theta \frac{p(\theta|O)q(\theta)}{\int_{\theta'} p(\theta'|O)q(\theta') d\theta'}=  \arg \max_\theta p(\theta|O)q(\theta)

In the equation above the denominator vanishes since it does not depend on :math:`\theta`.

Maximum likelihood estimation (MLE)
-------------------------------------
Finds the set of parameters that are most likely, given the data. Since priors over parameters are not taken into account unless MAP estimation is taking place, this is equivalent to finding the parameters that maximize the probability of the data given the parameters.

Prior
------

Posterior
----------