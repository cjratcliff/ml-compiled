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

Bayes' rule
-------------

.. math::

  P(A|B) = \frac{P(B|A)P(A)}{P(B)}
  
Bayesian inference
--------------------
The use of Bayes' rule to update a probability distribution as the amount of evidence changes.

Chain rule of probability
--------------------------
.. math::
  P(A_n, ..., A_1) = \prod_{i=1}^{n}P(A_i|A_1,...,A_{i-1})

For three variables this looks like:

.. math::
  P(A_3,A_2,A_1) = P(A_3|A_2,A_1) \cdot P(A_2|A_1) \cdot P(A_1)

Change of variables
----------------------
In the context of probability densities the change of variables formula describes how one distribution :math:`p(y)` can be given in terms of another, :math:`p(x)`:

.. math::

  p(y) = {|\frac{\partial f(x)}{\partial x}|}^{-1} p(x)
  
Where :math:`f` is an invertible function.

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

Distributions
---------------

Bernoulli
____________
Distribution for a random variable which is 1 with probability :math:`p` and zero with probability :math:`1-p`.

Special case of the Binomial distribution, which generalizes the Bernoulli to multiple trials.

.. math::

  P(x = k;p) = 
  \begin{cases}
    p, & \text{if } k = 1\\
    1-p, & \text{if } k = 0
  \end{cases}
  
Beta
_______
Family of distributions defined over :math:`[0,1]`.

Binomial
___________
Distribution for the number of successes in n trials, each with probability p of success and 1-p of failure. The probability density function is:

.. math::
  
  P(x = k;n,p) = {n\choose k} p^k (1-p)^{n-k}
  
Is closely approximated by the Poisson distribution when n is large and p is small.

Boltzmann
____________
.. math::

  P(x_i;T,\epsilon) = \frac{1}{Q} e^{-\epsilon_i / T}
  
where :math:`\epsilon_i` is the energy of :math:`x_i`, :math:`T` is the temperature of the system and :math:`Q` is a normalising constant.

Categorical
_____________
Generalizes the Bernoulli distribution to more than two categories.

.. math::

  P(x = k;p) = p_k

Dirichlet
___________
Multivariate version of the Beta distribution.

Conjugate prior of the categorical and multinomial distributions. 

Gamma
______
Can be used to model the amount of something a particular period, area or volume. For example, the amount of rainfall in an area in a month. This is as opposed to the Poisson which models the distribution for the number of discrete events.

  
Geometric
___________
Special case of the Negative Binomial distribution.

Gibbs
________
See `Boltzmann Distribution <https://ml-compiled.readthedocs.io/en/latest/probability.html#boltzmann>`_.
  
Gumbel
__________
Used to model the distribution of the maximum (or the minimum) of a number of samples of various distributions.

`Categorical Reparameterization with Gumbel-Softmax, Jang et al. (2016) <https://arxiv.org/abs/1611.01144>`_


Hypergeometric
_______________
Models the probability of k successes in n draws without replacement from a population of size N, where K of the objects in the population have the desired characteristic. Similar to the Binomial, except that the draws are made without replacement which means they are no longer independent.

Multinomial
______________
The distribution for n trials, each with k possible outcomes.

When n and k take on specific values or ranges the Multinomial distribution has specific names.

+------------------------+-----------------+------------------+
|                        | :math:`k = 2`   | :math:`k \geq 2` |
+========================+=================+==================+
| :math:`n = 1`          | Bernoulli       | Categorical      |
+------------------------+-----------------+------------------+
| :math:`n \geq 1`       | Binomial        | Multinomial      |
+------------------------+-----------------+------------------+

Negative Binomial
__________________
Distribution of the number of successes before a given number of failures occur.


Poisson
_________
Used to model the number of events which occur within a particular period, area or volume.


Zipf 
_______
A distribution that has been observed to be a good model for things like the frequency of words in a language, where there are a few very popular words and a long tail of lesser known ones.

For a population of size n, the frequency of the kth most frequent item is:

.. math::

  \frac{1/{k^s}}{\sum_{i=1}^n 1/i^s}
  
where :math:`s \geq 0` is a hyperparameter

Inference
-----------
Probabilistic inference is the task of determining the probability of a particular outcome.

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
Maximum a posteriori estimation. A type of point estimate. Can be seen as a regularization of MLE since it also incorporates a prior distribution. Uses Bayes rule to incorporate a prior over the parameters and find the parameters that are most likely given the data (rather than the other way around). Unlike with MLE (which is a bit of a simplification), the most likely parameters given the data are exactly what we want to find.

.. math::

    \hat{\theta}_{MAP}(O) = \arg \max_\theta p(\theta|O) = \arg \max_\theta \frac{p(O|\theta)q(\theta)}{\int_{\theta'} p(O|\theta')q(\theta') d\theta'} = \arg \max_\theta p(O|\theta)q(\theta)

Where :math:`q(\theta)` is the prior for the parameters.

In the equation above the denominator vanishes since it does not depend on :math:`\theta`.

Maximum likelihood estimation (MLE)
-------------------------------------
Finds the set of parameters that are most likely, given the data. Since priors over parameters are not taken into account unless MAP estimation is taking place, this is equivalent to finding the parameters that maximize the probability of the data given the parameters.

.. math::

    \hat{\theta}_{MLE}(O) = \arg \max_\theta p(O|\theta)

Normalizing flow
------------------
A function that can be used to transform one random variable into another. The function must be invertible and have a tractable Jacobian.

Extensively used for density estimation.

Prior
------
A probability distribution before any evidence is taken into account. For example the probability that it will rain where there are no observations such as cloud cover.

Improper prior
_________________
A prior whose probability distribution has infinitesimal density over an infinitely large range. For example, the distribution for picking an integer at random.

Informative and uninformative priors
______________________________________
Examples:

Informative:

* The temperature is normally distributed with mean 20 and variance 3.

Uninformative:

* The temperature is positive.
* The temperature is less than 200.
* All temperatures are equally likely.

'Uninformative' can be a misnomer. 'Not very informative' would be more accurate.

Posterior
----------
A conditional probability distribution that takes evidence into account. For example, the probability that it will rain, given that it is cloudy.
