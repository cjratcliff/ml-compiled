===============
Optimization
===============

--------------------------
Automatic differentiation
--------------------------
Has two distinct modes - forward and reverse.

Forward mode takes an input to the graph and evaluates the derivative of all subsequent nodes with respect to it.

Reverse mode takes an output (eg the loss) and differentiates it with respect to all inputs. This is usually more useful in neural networks since it can be used to get the derivatives for all the parameters in one pass.

--------------------------
Backpropagation
--------------------------
Naively summing the product of derivatives over all paths to a node is computationally intractable because the number of paths increases exponentially with depth.

Instead, the sum over paths is calculated by merging paths back together at the nodes. Derivatives can be computed either forward or backward with this method.

http://colah.github.io/posts/2015-08-Backprop/

"""""""""""""""""""""""""""""""""""""
Backpropagation through time (BPTT)
"""""""""""""""""""""""""""""""""""""
Used to train RNNs. The RNN is unfolded through time.

When dealing with long sequences (hundreds of inputs), a truncated version of BPTT is often used to reduce the computational cost. This stops backpropagating the errors after a fixed number of steps, limiting the length of the dependencies that can be learned.

-------------
Batch size
-------------
Pros of large batch sizes:

* Decreases the variance of the updates, making convergence more stable. From this perspective, increasing the batch size has very similar effects to decreasing the learning rate.
* Matrix computation is more efficient.

Cons of large batch sizes:

* Very large batch sizes may not fit in memory.
* Smaller number of updates for processing the same amount of data, slowing training.
* Hypothesized by Keskar et al. (2016) to have worse generalization performance since they result in sharper local minima being reached.

`On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima, Keskar et al. (2016) <https://arxiv.org/abs/1609.04836>`_

`Coupling Adaptive Batch Sizes with Learning Rates (2016) <https://arxiv.org/abs/1612.05086>`_

`Big Batch SGD: Automated Inference using Adaptive Batch Sizes (2016) <https://arxiv.org/abs/1610.05792>`_

--------------------------
Curriculum learning
--------------------------
Training the classifier with easy examples initially and gradually transitioning to the harder ones. Useful for architectures which are very hard to train.

---------
Depth
---------
Depth increases the representational power of a network exponentially, for a given number of parameters. However, deeper networks can also be considerably harder to train, due to vanishing and exploding gradients or dying ReLUs.

Potential solutions include:

* Using a smaller learning rate
* Skip connections
* Auxiliary loss functions (eg `Szegedy et al. (2016) <https://arxiv.org/pdf/1409.4842.pdf>`_)
* Orthogonal initialization

-------------
End-to-end
-------------
The entire model is trained in one process, not as separate modules. For example, a pipeline consisting of object recognition and description algorithms that are trained individually would not be trained end-to-end.

-------------
Epoch
-------------
A single pass through the training data.

--------------
Error surface
--------------
The surface obtained by plotting the weights of the network against the loss. For a linear network with a squared error function, the surface is a quadratic bowl.

----------------------------
Exploding gradient problem
----------------------------
The gradient grows exponentially as we move backward through the layers.

Gradient clipping can be an effective antidote.

`On the difficulty of training recurrent neural networks, Pascanu et al. (2012) <https://arxiv.org/pdf/1211.5063.pdf>`_

----------------------------
Gradient clipping
----------------------------
Used to avoid exploding gradients in very deep networks by normalizing the gradients of the parameter vector. Clipping can be done either by value or by norm.

"""""""""""""""""""""""""""""""""""""
Clipping by value
"""""""""""""""""""""""""""""""""""""
.. math::

  g_i = \min\{a,\max\{b,g_i\}\}
  
Where :math:`g_i` is the gradient of the parameter :math:`\theta_i` and :math:`a` and :math:`b` are hyperparameters.

"""""""""""""""""""""""""""""""""""""
Clipping by norm
"""""""""""""""""""""""""""""""""""""
.. math::

  g_i = g_i*a/||g||_2

Where :math:`g_i` is the gradient of the parameter :math:`\theta_i` and :math:`a` is a hyperparameter.

`On the difficulty of training recurrent neural networks, Pascanu et al. (2012) <https://arxiv.org/pdf/1211.5063.pdf>`_

----------------------------
Learning rate
----------------------------

-------------
Optimizers
-------------

""""""""
AdaBoost
""""""""

""""""""
AdaDelta
""""""""
Adadelta is a gradient descent based learning algorithm that adapts the learning rate per parameter over time. It was proposed as an improvement over AdaGrad, which is more sensitive to hyperparameters and may decrease the learning rate too aggressively. Adadelta It is similar to rmsprop and can be used instead of vanilla SGD.

`AdaDelta: An Adaptive Learning Rate Method, Zeiler (2012) <https://arxiv.org/abs/1212.5701>`_

""""""""
Adam
""""""""
Adam is an adaptive learning rate algorithm similar to RMSProp, but updates are directly estimated using EMAs of the first and uncentered second moment of the gradient. Designed to combine the advantages of RMSProp and AdaGrad.

First moment - mean. Second moment - variance. This means the entire expression can be interpreted as a signal-to-noise ratio, with the step-size increasing when the signal is higher, relative to the noise. This leads to the step-size naturally becoming smaller over time. Using the square root for the variance term means it can be seen as computing the EMA of :math:`g/|g|`. This reduces the learning rate when the gradient is a mixture of positive and negative values as they cancel out in the EMA to produce a number closer to 0.

The bias correction term counteracts bias caused by initializing the moment estimates with zeros.

Does not require a stationary objective and works with sparse gradients. Is invariant to the scale of the gradients.

`Adam: A Method for Stochastic Optimization, Kingma et al. (2015) <https://arxiv.org/pdf/1412.6980.pdf>`_

""""""""""""""""""""""""
Averaged SGD (ASGD)
""""""""""""""""""""""""
Runs like normal SGD but replaces the parameters with their average over time at the end.

""""""""
BFGS
""""""""
Iterative method for solving nonlinear optimization problems that approximates Newton’s method.
BFGS stands for Broyden–Fletcher–Goldfarb–Shanno.
L-BFGS is a popular memory-limited version of the algorithm.

""""""""""""""""""""""""
Conjugate gradient
""""""""""""""""""""""""
Iterative algorithm for solving SLEs where the matrix is symmetric and positive-definite.

""""""""""""""""""""""""""""""""
Krylov subspace descent
""""""""""""""""""""""""""""""""
Second-order optimization method. Inferior to SGD.

`Krylov Subspace Descent for Deep Learning, Vinyals and Povey (2011) <https://arxiv.org/abs/1111.4259>`_

""""""""
Momentum
""""""""
Adds a fraction of the update from the previous time step to the current time step. 

Deep architectures often have deep ravines in their landscape near local optimas. They can lead to slow convergence with vanilla SGD since the negative gradient will point down one of the steep sides rather than towards the optimum. Momentum pushes optimization to the minimum faster. Commonly set to 0.9.

""""""""""""""""
Natural gradient
""""""""""""""""
At each iteration attempts to perform the update which minimizes the loss function subject to the constraint that the KL-divergence between the probability distribution output by the network before and after the update is equal to a constant.

`Revisiting natural gradient for deep networks, Pascanu and Bengio (2014) <https://arxiv.org/abs/1301.3584>`_

""""""""""""""""
Newton’s method
""""""""""""""""
An iterative method for finding the roots of an equation.

.. math::

    x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}

In the context of gradient descent, Newton’s method is applied to the derivative of the function to find the points where the derivative is equal to zero (the local optima). Therefore in this context it is a second order method.

:math:`x_t=H_{t-1}g_t` where :math:`H_{t-1}` is the inverse of the Hessian matrix at iteration :math:`t-1`.

Picks the optimal step size for quadratic problems but is also prohibitively expensive to compute for large models due to the size of the Hessian matrix, which is quadratic in the number of parameters.

""""""""""""""""""""""""
Nesterov’s method
""""""""""""""""""""""""
Attempts to solve instabilities that can arise from using momentum by keeping the history of previous update steps and combining this with the next gradient step.

""""""""
RMSProp
""""""""
Similar to Adagrad, but introduces an additional decay term to counteract AdaGrad’s rapid decrease in the learning rate. Divides the gradient by a running average of its recent magnitude. 0.001 is a good default value for the learning rate (:math:`\eta`) and 0.9 is a good default value for :math:`\alpha`. The name comes from Root Mean Square Propagation.

.. math::

  \mu_t = \alpha \mu_{t-1} + (1 - \alpha) g_t^2
  
  u_t = - \eta \frac{g_t}{\sqrt{\mu_t + \epsilon}}

http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

http://ruder.io/optimizing-gradient-descent/index.html#rmsprop

-------------------
Saddle points
-------------------

Gradients around saddle points are close to zero which makes learning slow. The problem can be partially solved by using a noisy estimate of the gradient, which SGD does implicitly.

`Identifying and attacking the saddle point problem in high-dimensional non-convex optimization, Dauphin et al. (2014) <https://arxiv.org/abs/1406.2572>`_

