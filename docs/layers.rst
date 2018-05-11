"""""""""""""""
Layers
"""""""""""""""

Affine layer
--------------
Synomym for fully-connected layer.

Attention
------------
Has been used to improve image classification, image captioning, speech recognition, generative models and learning algorithmic tasks, but it has probably had the largest impact on neural machine translation.

In translation, rather than creating a fixed-length vector from the outputs of encoder, it retains them all and concatenates them into a ‘memory tensor’. At each step of the decoder, a weighted average over the memory tensor is computed, allowing the decoder to ‘focus’ on different parts of the input. A similar logic applies in the image captioning task, focusing on areas of the image instead.

In translation, each output word depends on a weighted combination of all input words. Computing these weights can take time proportional to the product of the length of the input and output sequences. In content-based attention the weights are computed as the dot product between the items in the sequence and the ‘query’ outputted by the attending RNN.

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Soft attention
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
The standard form of attention, as proposed in Bahdanau et al. (2015).

Let :math:`x = \{x_1,...,x_T\}` be the input sequence and :math:`y = \{y_1,...,y_U\}` be the output sequence.

There is an encoder RNN whose hidden state at index i we refer to as :math:`h_i`. The decoder RNN's state at index i is :math:`s_i`.

Attention is calculated over all the words in the sequence form a weighted sum, known as the context vector. This is defined as:

.. math::

  c_i = \sum_{j=1}^{T} \alpha_{ij} h_j
  
where :math:`\alpha_{ij}` is the jth element of the softmax of :math:`e_i`.

The attention given to a particular input word depends on the hidden states of the encoder and decoder RNNs.

.. math::

  e_{ij} = a(s_{i-1}, h_j) 
  
The decoder's hidden state is computed according to the following expression, where :math:`f` represents the decoder.

.. math::

  s_i = f(s_{i-1},y_{i-1},c_i)

To predict the output sequence we take the decoder hidden state and the context vector and feed them into a fully connected softmax layer :math:`g` which gives a distribution over the output vocabulary.

.. math::

  y_i = g(s_i,c_i)

Training
__________
Soft attention is differentiable and can therefore be trained with standard back-propagation.

Computational complexity
_______________________________
When using two RNNs (an encoder and a decoder) to translate a sequence of length :math:`n` the time complexity is :math:`O(n)`.

However, a soft attention mechanism must look over every item in the input sequence for every item in the output sequence, resulting in a quadratic complexity:  :math:`O(n^2)`.


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Hard attention
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Trained using the REINFORCE algorithm, since it is not differentiable.

`Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al. (2015) <https://arxiv.org/abs/1409.0473>`_

Batch normalization
-------------------------
Normalizes the input vector to a layer to have zero mean and unit variance. Training deep neural networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization. This phenomenon is referred to as internal covariate shift.

The batch-normalized version of a layer, :math:`x`, is:

.. math::

  BN(x) = \gamma \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta
  
Where :math:`\gamma` and :math:`\beta` are learned and :math:`\epsilon` is a hyperparameter. 

:math:`\mu_x` and :math:`\sigma_x^2` are moving averages of the mean and variance of :math:`x`. They do not need to be learned.

`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015) <https://arxiv.org/abs/1502.03167>`_

Convolutional layer
-----------------------
Transforms an image according to the convolution operation shown below, where the image on the left is the input and the image being created on the right is the output:

TODO

Let :math:`x` be a matrix representing the image and :math:`k` be another representing the kernel, which is of size NxN. :math:`c(x,k)` is the matrix that results from convolving them together. Then, formally, convolution applies the following formula:

.. math::

  c(x,k)_{ij} = \sum_{r=-M}^{M} \sum_{s=-M}^{M} x_{i+r,j+s} k_{r+M,s+M}
  
Where :math:`M = (N - 1)/2`.

'''''''''''''''''''''''''''''
Padding
'''''''''''''''''''''''''''''
Applying the kernel to pixels near or at the edges of the image will result in needing pixel values that do not exist. There are two ways of resolving this:

* Only apply the kernel to pixels where the operation is valid. For a kernel of size k this will reduce the image by (k-1)/2 pixels on each side.
* Pad the image with zeros to allow the operation to be defined.

'''''''''''''''''''''''''''''
Efficiency
'''''''''''''''''''''''''''''
The same convolution operation is applied to every pixel in the image, resulting in a considerable amount of weight sharing. This means convolutional layers are quite efficient in terms of parameters. Additionally, if a fully connected layer was used to represent the functionality of a convolutional layer most of its parameters would be zero since the convolution is a local operation. This further increases efficiency.

The number of parameters can be further reduced by setting a stride so the convolution operation is only applied every m pixels.

'''''''''''''''''''''''''''''
1x1 convolutions
'''''''''''''''''''''''''''''
These are actually matrix multiplications, not convolutions. They are a useful way of increasing the depth of the neural network since they are equivalent to :math:`f(hW)`, where :math:`f` is the activation function.

If the number of channels decreases from one layer to the next they can be also be used for dimensionality reduction.

http://iamaaditya.github.io/2016/03/one-by-one-convolution/

'''''''''''''''''''''''''''''
Separable convolution/filter
'''''''''''''''''''''''''''''
A filter or kernel is separable if it (a matrix) can be expressed as the product of a row vector and a column vector. This decomposition can reduce the computational cost of the convolution. Examples include the Sobel edge detection and Gaussian blur filters.

.. math::

  K = xx^T, x \in \mathbb{R}^{n \times 1}

Dense layer
--------------
Synomym for fully-connected layer.

Fully-connected layer
-----------------------
Applies the following function:

.. math::

  h' = f(hW + b)
  
:math:`f` is the activation function. :math:`h` is the output of the previous hidden layer. :math:`W` is the weight matrix and :math:`b` is known as the bias vector.

Hierarchical softmax
----------------------
A layer designed to improve efficiency when the number of output classes is large. Its complexity is logarithmic in the number of classes rather than linear, as for a standard softmax layer.

A tree is constructed where the leaves are the output classes.

Alternative methods include `Noise Contrastive Estimation <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#noise-contrastive-estimation>`_ and `Negative Sampling <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#negative-sampling>`_.

`Classes for Fast Maximum Entropy Training, Goodman (2001) <https://arxiv.org/abs/cs/0108006>`_

Inception layer
--------------------
At each layer of a traditional CNN we can choose it to be either a convolutional or a pooling layer. If it is convolutional we then need to choose the kernel size (1x1, 3x3, 5x5 etc.). The inception module negates this choice by choosing them all and concatenating the results.

Padding can ensure the different convolution sizes still have the same size of output. The pooling component can be concatenated by using a stride of length 1 for the pooling.

9 are used in GoogLeNet, a 22-layer deep network and state of the art solution for ILSVRC 2014. The width of the filters increases from 256 to 1024 from the start to the end of GoogLeNet. Due to the removal of final fully connected layers it only has 5 million parameters and takes less than twice as long as AlexNet to train.

5x5 convolutions are expensive so the `1x1 convolutions <https://ml-compiled.readthedocs.io/en/latest/layers.html#x1-convolutions>`_ make the architecture computationally viable. The 1x1 convolutions perform dimensionality reduction by reducing the number of filters. This is not a characteristic necessarily found in all 1x1 convolutions. Rather, the authors have specified to have the number of output filters less than the number of input filters.

Pooling layer
---------------

'''''''''''''''''''''''''''''
Max pooling
'''''''''''''''''''''''''''''
Transforms the input by taking the max along a particular dimension. In sequence processing this is usually the length of the sequence.

'''''''''''''''''''''''''''''
Mean pooling
'''''''''''''''''''''''''''''
Also known as average pooling. Identical to max-pooling except the mean is used instead of the max.

'''''''''''''''''''''''''''''
RoI pooling
'''''''''''''''''''''''''''''
Used to solve the problem that the regions of interest (RoI) identified by the bounding boxes can be different shapes in object recognition. The CNN requires all inputs to have the same dimensions.

The RoI is divided into a number of rectangles of fixed size (except at the edges). If doing 3x3 RoI pooling there will be 9 rectangles in each RoI. We do max-pooling over each RoI to get 3x3 numbers.

Softmax layer
----------------
A fully-connected layer with a `softmax <https://ml-compiled.readthedocs.io/en/latest/activations.html#softmax>`_ activation function.

Upsampling layer
-----------------
Used in convolutional autoencoders to go from the the bottleneck layer up to full image.
