=====================
Activation functions
=====================

"""
ELU
"""
An activation function with the form:

.. math:: 

    f(x) = 
    \begin{cases}
      x, & x > 0 \\
      \alpha (exp(x) - 1), & x \leq 0
    \end{cases}

The first derivative is:

.. math:: 

    f(x) = 
    \begin{cases}
      1, &  x > 0 \\
      f(x) + \alpha, & x \leq 0
    \end{cases}

In practice the hyperparameter :math:`\alpha` is always set to 1.

Compared to ReLUs, ELUs have a mean activation closer to zero which is helpful. However, this advantage is probably nullified by batch normalization.

The more gradual decrease of the gradient should also make them less susceptible to the dying ReLU problem, although they will suffer from the vanishing gradients problem instead.

`Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs), Clevert et al. (2015) <https://arxiv.org/abs/1511.07289>`_

""""""
Maxout
""""""
An activation function used with dropout. Can be a piecewise linear approximation for arbitrary convex activation functions. This means it can approximate ReLU, LReLU, ELU and linear activations but not tanh or sigmoid.

.. math::

  f(x) = \max_{j \in [1,k]} x^T W_j + b_j

Was used to get state of the art performance on MNIST, SVHN, CIFAR-10 and CIFAR-100.

`Maxout Networks, Goodfellow et al. (2013) <https://arxiv.org/pdf/1302.4389.pdf>`_

""""
ReLU
""""
Rectified Linear Unit. The non-saturating activation function :math:`f(x)=\max\{0,x\}` where x is the input to the neuron.

The fact that the gradient is 1 when the input is positive means it does not suffer from vanishing and exploding gradients. However, it suffers from its own 'dying ReLU problem' instead.

The Dying ReLU Problem
-------------------------
When the input to a neuron is negative, the gradient will be zero. This means that gradient descent will not update the weights so long as the input remains negative.
A smaller learning rate helps solve this problem.
The Leaky ReLU and the Parametric ReLU (PReLU) attempt to solve this problem by using :math:`f(x)=max\{ax,x\}` where a is a small constant like 0.1. However, this small gradient when the input in negative means vanishing gradients are once again a problem.

`Rectified Linear Units Improve Restricted Boltzmann Machines, Nair and Hinton (2010) <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf>`_

"""""""
Sigmoid
"""""""
Activation function that maps outputs to be between 0 and 1.

.. math::

  f(x) = \frac{e^x}{e^x + 1}

Has problems with saturation. This makes vanishing and exploding gradients a problem and initialization extremely important.

"""""""
Softmax
"""""""
All entries in the output vector are in the range (0,1) and sum to 1, making the result a valid probability distribution.

.. math:: 

    f(z)_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}, j \in {1,...,K}
    
Unlike most other activation functions, the softmax does not apply the same function to each item in the input independently. The requirement that the output vector sums to 1 means that if one of the inputs is increased the others must decrease in the output.

""""
Tanh
""""
Activation function that is used in the GRU and LSTM.
Has problems with saturation like the sigmoid. This makes vanishing and exploding gradients a problem and initialization extremely important.
tanh(x) is between -1 and 1.
Centered around 0, unlike the sigmoid.
