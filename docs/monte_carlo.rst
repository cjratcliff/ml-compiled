""""""""""""""""""""""""""""""
Monte Carlo methods
""""""""""""""""""""""""""""""

Gibbs sampling
--------------------

A simple MCMC algorithm, used for sampling from the joint distribution when it cannot be calculated directly but the conditional can be.

An example use case is in generative image models. The joint distribution over all the pixels is intractable but the conditional distribution for one pixel given the rest is not.

Pseudocode:

.. code-block:: none

      Randomly initialise x.
      For  i = 1,...,d
        Sample the ith dimension of x given the values in all the other dimensions.

Importance sampling
------------------------
Monte Carlo method that attempts to estimate the mean of a distribution with zero density almost everywhere that would make simple Monte Carlo methods ineffective. Does this by sampling from a distribution that does not have this property then adjusting to compensate.

Can be used to deal with the computational problems of very large vocabularies in NLP but suffers from stability problems.

`Quick Training of Probabilistic Neural Nets by Importance Sampling, Bengio and Senecal (2003)  <http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/21>`_

Deep Learning, Section 17.2

MCMC (Markov Chain Monte Carlo)
---------------------------------
A class of algorithm which is useful for sampling from and computing expectations of highly complex and high-dimensional probability distributions. For example, the distribution for images which contain dogs, as described by their pixel values.

High probability areas are very low proportion of the total space due to the high dimensionality, meaning that rejection sampling won’t work. Instead, MCMC uses a random walk (specifically a Markov chain) that attempts to stay close to areas of high probability in the space.

MCMC algorithms do not generate independent samples.

Metropolis-Hastings algorithm
---------------------------------
A simple MCMC algorithm.

Pseudocode:

.. code-block:: none

    Randomly initialise x
    For t = 1,...,T_max
        Generate a candidate for the next sample from a normal distribution centered on the current point.
        Calculate the acceptance ratio, the probability that the new candidate will be retained. This is equal to the density at the current point, divided by the density at the candidate point.
        Either accept or reject the candidate, based on a random sample from the distribution (a, 1-a).

The proposal distribution is the distribution over the possible points to sample next.

Mixing
----------
The samples from an MCMC algorithm are said to be well mixed if they are independent of each other.

Poor mixing can be caused by getting stuck in local minima of the density function.
