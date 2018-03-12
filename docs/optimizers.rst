""""""""
AdaBoost
""""""""

""""""""
AdaDelta
""""""""
Adadelta is a gradient descent based learning algorithm that adapts the learning rate per parameter over time. It was proposed as an improvement over AdaGrad, which is more sensitive to hyperparameters and may decrease the learning rate too aggressively. Adadelta It is similar to rmsprop and can be used instead of vanilla SGD.

AdaDelta: An Adaptive Learning Rate Method, Zeiler (2012)

""""""""
Adam
""""""""
Adam is an adaptive learning rate algorithm similar to RMSProp, but updates are directly estimated using EMAs of the first and uncentered second moment of the gradient. Designed to combine the advantages of RMSProp and AdaGrad.

First moment - mean. Second moment - variance. This means the entire expression can be interpreted as a signal-to-noise ratio, with the step-size increasing when the signal is higher, relative to the noise. This leads to the step-size naturally becoming smaller over time. Using the square root for the variance term means it can be seen as computing the EMA of $g/|g|$. This reduces the learning rate when the gradient is a mixture of positive and negative values as they cancel out in the EMA to produce a number closer to 0.

The bias correction term counteracts bias caused by initializing the moment estimates with zeros.

Does not require a stationary objective and works with sparse gradients. Is invariant to the scale of the gradients.

Adam: A Method for Stochastic Optimization

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

Krylov Subspace Descent for Deep Learning, Vinyals and Povey (2011)

""""""""
Momentum
""""""""
Adds a fraction of the update from the previous time step to the current time step. 

Deep architectures often have deep ravines in their landscape near local optimas. They can lead to slow convergence with vanilla SGD since the negative gradient will point down one of the steep sides rather than towards the optimum. Momentum pushes optimization to the minimum faster. Commonly set to 0.9.

""""""""""""""""
Natural gradient
""""""""""""""""
At each iteration attempts to perform the update which minimizes the loss function subject to the constraint that the KL-divergence between the probability distribution output by the network before and after the update is equal to a constant.

Revisiting natural gradient for deep networks, Pascanu and Bengio (2014)

""""""""""""""""""""""""
Nesterov’s method
""""""""""""""""""""""""
Attempts to solve instabilities that can arise from using momentum by keeping the history of previous update steps and combining this with the next gradient step.

""""""""
RMSProp
""""""""
Similar to Adagrad, but introduces an additional decay term to counteract AdaGrad’s rapid decrease in the learning rate. Divides the gradient by a running average of its recent magnitude. Parameters are the learning rate, alpha and epsilon. 0.001 is a good default value for the learning rate. The name comes from Root Mean Square Propagation.

% TODO

\url{http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf}
