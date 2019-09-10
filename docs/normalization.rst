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

Group normalization
----------------------

Group normalization implements the same formula as batch normalization but takes the average over the feature dimension(s) rather than the batch dimension. This means it can be used with small batch sizes, unlike batch normalization, which is useful for many computer vision applications where memory-consuming high resolution images naturally restrict the batch size. 

.. math::

  GN(x) = \gamma \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta
  
Where \gamma and \beta are learned and \epsilon is a small hyperparameter that prevents division by zero. Separate \gamma and \beta are learned for each group normalization layer. :math:`\beta` and :math:`\gamma` make sure the model does not lose any representational power from the normalization.

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
