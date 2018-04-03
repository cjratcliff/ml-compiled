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

Usage for decoding in translation:

* The decoder (an RNN) is initialized.
* The attention given to a particular input word depends on the encoding of that input and the hidden state of the RNN.
* Attention is calculated over all words at each timestep to form a weighted sum, known as the context vector.
* The context vector is the input to the decoder RNN at the next time-step.
* The decoder then takes the context vector and its own previous hidden state to produce a softmax over possible words as the translated text for that timestep.
* Instead of just predicting the most likely word it is also possible to use the top k predictions and do beam search.

Trained using the REINFORCE algorithm when using hard attention, since it is not differentiable. Can be trained with standard back-propagation when using deterministic soft attention.

`Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al. (2015) <https://arxiv.org/abs/1409.0473>`_

Batch normalization
-------------------------
Normalizes the input vector to a layer to have zero mean and unit variance. Training deep neural networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization. This phenomenon is referred to as internal covariate shift.

.. math::

  BN(x) = \gamma \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta
  
Where :math:`\gamma` and :math:`\beta` are learned and :math:`\epsilon` is a hyperparameter.

`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015) <https://arxiv.org/abs/1502.03167>`_

Convolutional layer
-----------------------
Transforms an image according to the convolution operation shown below, where the image on the left is the input and the image being created on the right is the output:

TODO

Applying the kernel to pixels near or at the edges of the image will result in needing pixel values that do not exist. There are two ways of resolving this:

* Only apply the kernel to pixels where the operation is valid. For a kernel of size k this will reduce the image by (k-1)/2 pixels on each side.
* Pad the image with zeros to allow the operation to be defined.

The same convolution operation is applied to every pixel in the image, resulting in a considerable amount of weight sharing. This means convolutional layers are quite efficient in terms of parameters.

The number of parameters can be further reduced by setting a stride so the convolution operation is only applied every m pixels.

Can be represented by a fully-connected layer in theory. Such a layer would be mostly zeros as the effects are local. This is especially true if the layer is replicating multiple filters.

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

Inception layer
--------------------
At each layer of a traditional CNN we can choose it to be either a convolutional or a pooling layer. If it is convolutional we then need to choose the kernel size (1x1, 3x3, 5x5 etc.). The inception module negates this choice by choosing them all and concatenating the results.

Padding can ensure the different convolution sizes still have the same size of output. The pooling component can be concatenated by using a stride of length 1 for the pooling.

9 are used in GoogLeNet, a 22-layer deep network and state of the art solution for ILSVRC 2014. The width of the filters increases from 256 to 1024 from the start to the end of GoogLeNet. Due to the removal of final fully connected layers it only has 5 million parameters and takes less than twice as long as AlexNet to train.

5x5 convolutions are expensive so the 1x1 convolutions make the architecture computationally viable. The 1x1 convolutions perform dimensionality reduction by reducing the number of filters. This is not a characteristic necessarily found in all 1x1 convolutions. Rather, the authors have specified to have the number of output filters less than the number of input filters. See also convolutional layer > 1x1 convolution.

RoI pooling
--------------
Used to solve the problem that the regions of interest (RoI) identified by the bounding boxes can be different shapes in object recognition. The CNN requires all inputs to have the same dimensions.

The RoI is divided into a number of rectangles of fixed size (except at the edges). If doing 3x3 RoI pooling there will be 9 rectangles in each RoI. We do max-pooling over each RoI to get 3x3 numbers.

Upsampling layer
-----------------
Used in convolutional autoencoders to go from the the bottleneck layer up to full image.
