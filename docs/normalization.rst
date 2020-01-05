Normalization
""""""""""""""""""""

Batch normalization
-------------------------
Normalizes the input vector to a layer to have zero mean and unit variance, making training more efficient. Training deep neural networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization. This phenomenon is referred to as internal covariate shift.

Adding :math:`\beta` to the normalized input and scaling it by :math:`\gamma` ensures the model does not lose representational power as a result of the normalization.

Batch Normalization is often found to improve generalization performance (`Zhang et al. (2016) <https://arxiv.org/pdf/1611.03530.pdf>`_).

| **Proposed in** 
| `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015) <https://arxiv.org/abs/1502.03167>`_

Training
_________________
The batch-normalized version of the inputs, :math:`x \in \mathbb{R}^{n \times d}`, to a layer is:

.. math::

  BN(x) = \gamma \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta
  
Where :math:`\gamma` and :math:`\beta` are learned and :math:`\epsilon` is a small hyperparameter that prevents division by zero. If there are multiple batch normalization layers a separate :math:`\gamma` and :math:`\beta` will be learned for each of them.

:math:`\mu_x \in \mathbb{R}^{d}` and :math:`\sigma_x^2 \in \mathbb{R}^{d}` are moving averages of the mean and variance of :math:`x`. They do not need to be learned. The moving averages are calculated independently for each feature in :math:`x`.

Batch normalization does not work well with small batch sizes (`Wu and He, 2018 <https://arxiv.org/abs/1803.08494>`_). Small batches cause the statistics to become inaccurate. This can cause problems when training models with large images where large batches will not fit in memory.

Inference
___________
Batch normalization's stabilizing effect is helpful during training but unnecessary at inference time. Therefore, once the network is trained the population mean and variance are used for normalization, rather than the batch mean and variance. This means the networks output can depend only on the input, not also on other examples in the batch.

Application to RNNs
____________________
Batch normalization is difficult to apply to RNNs since it requires storing the batch statistics for every time step in the sequence. This can be problematic if a sequence input during inference is longer than those seen during training.

`Coojimans et al. (2016) <https://arxiv.org/abs/1603.09025>`_ propose a variant of the LSTM that applies batch normalization to the hidden-to-hidden transitions.

`Recurrent Batch Normalization, Coojimans et al. (2016) <https://arxiv.org/abs/1603.09025>`_

Conditional batch normalization
________________________________
The formula is exactly the same as normal batch normalization except :math:`\gamma` and :math:`\beta` are not learned parameters, but rather the outputs of functions.

Was used to achieve `state of the art results <https://arxiv.org/pdf/1707.03017.pdf>`_ on the CLEVR visual reasoning benchmark.

`Learning Visual Reasoning Without Strong Priors, Perez et al. (2017) <https://arxiv.org/pdf/1707.03017.pdf>`_

Feature normalization
-----------------------

This class of normalizations refers to methods that transform the inputs to the model, as opposed to the activations within it.

Feature scaling
__________________
Examples include min-max and z-score normalization.

Min-max normalization
_______________________

Rescales the features so they have a specified minimum and maximum.

To rescale to between a and b:

.. math::

  x_{ij} := \frac{(x_{ij} - min_j x_{ij})(b - a)}{max_j x_{ij} - min_j x_{ij}}
  
When computing the min and max be sure to use only the training data, as opposed to calculating these statistics on the entire dataset.

Principal Component Analysis (PCA)
_____________________________________
Decomposes a matrix :math:`X \in \mathbb{R}^{n \times m}` into a set of :math:`k` orthogonal vectors. The matrix :math:`X` represents a dataset with :math:`n` examples and :math:`m` features.

Method for PCA via eigendecomposition:

1. Center the data by subtracting the mean for each dimension.
2. Compute the covariance matrix on the centered data :math:`C = (X^TX)/(n-1)`.
3. Do eigendecomposition of the covariance matrix to get :math:`C = Q \Lambda Q^*`.
4. Take the k largest eigenvalues and their associated eigenvectors. These eigenvectors are the 'principal components'.
5. Construct the new matrix from the principal components by multiplying the centered :math:`X` by the truncated :math:`Q`.

PCA can also be done via SVD.

Whitening
____________
The process of transforming the inputs so that they have zero mean and a covariance matrix which is the identity. This means the features will be linearly uncorrelated with each other and have variances equal to 1.

ZCA
_____
Like PCA, ZCA converts the data to have zero mean and an identity covariance matrix. Unlike PCA, it does not reduce the dimensionality of the data and tries to create a whitened version that is minimally different from the original.

Z-score normalization
_______________________

The features are transformed by subtracting their mean and dividing by their standard deviation:

.. math::

  x_{ij} := \frac{x_{ij} - \mu_i}{\sigma_i}
  
where :math:`x_{ij}` is the jth instance of feature i and :math:`\mu_i` and :math:`\sigma_i` are the mean and standard deviation of feature x_i respectively.

Ensure that the mean and standard deviation are calculated on the training set, not on the entire dataset.

Group normalization
----------------------

Group normalization implements the same formula as batch normalization but takes the average over the feature dimension(s) rather than the batch dimension. This means it can be used with small batch sizes, unlike batch normalization, which is useful for many computer vision applications where memory-consuming high resolution images naturally restrict the batch size. 

.. math::

  GN(x) = \gamma \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta
  
Where :math:`\gamma` and :math:`\beta` are learned and :math:`\epsilon` is a small hyperparameter that prevents division by zero. Separate \gamma and \beta are learned for each group normalization layer. :math:`\beta` and :math:`\gamma` make sure the model does not lose any representational power from the normalization.

| **Proposed in** 
| `Group Normalization, Wu and He. (2018) <https://arxiv.org/abs/1803.08494>`_

Layer normalization
----------------------
Can be easily applied to RNNs, unlike batch normalization.

If the hidden state at time :math:`t` of an RNN is given by:

.. math::

  h_t = f(W x_t + b) = f(a_t + b)

Then the layer normalized version is:

.. math::

  h_t = f(\frac{g}{\sigma_t}*(a - \mu_t) + b)
  
where :math:`\mu_t` and :math:`\sigma_t` are the mean and variance of :math:`a_t`.

| **Proposed in** 
| `Layer Normalization, Ba et al. (2016) <https://arxiv.org/abs/1607.06450>`_
| 
| **Used in**
| `Attention is All You Need, Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_


Weight normalization
----------------------
The weights of the network are reparameterized as:

.. math::

  w = \frac{g}{||v||}v
  
where :math:`g` is a learnt scalar and :math:`v` is a learnt vector.

This guarantees that :math:`||w|| = g` without the need for explicit normalization. 

Simple to use in RNNs, unlike batch normalization.

Unlike batch normalization, weight normalization only affects the weights - it does not normalize the activations of the network.

| **Proposed in**
| `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, Salimans and Kingma (2016) <https://arxiv.org/abs/1602.07868>`_
