""""""""""""""""""""""""""
Convolutional networks
""""""""""""""""""""""""""

AlexNet
--------
Performed considerably better than the state of the art at the time. Has 60 million parameters, 650,000 neurons and includes five convolutional layers.

The two 'streams' shown in the paper only exist to allow training on two GPUs.

| **Proposed in**
| `ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky et al. (2012) <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

GoogLeNet
-------------
CNN that won the ILSVRC 2014 challenge. Composed of 9 inception layers.

| **Proposed in**
| `Going Deeper with Convolutions, Szegedy et al. (2014) <https://arxiv.org/abs/1409.4842>`_

LeNet5
--------
A basic convolutional network, historically used for the MNIST dataset.

| **Proposed in**
| `Gradient-based learning applied to document recognition, LeCun et al. (1998) <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

Residual network (ResNet)
---------------------------
An architecture that uses skip connections to create very deep networks. The `original paper <https://arxiv.org/abs/1512.03385>`_ achieved 152 layers, 8 times deeper than VGG nets. Used for image recognition, winning first place in the ILSVRC 2015 classification task. Residual connections can also be used to create deeper RNNs such as Google’s 16-layer RNN encoder-decoder (Wu et al., 2016).

Uses shortcut connections performing the identity mapping, which are added to the outputs of the stacked layers. Each residual block uses the equation:

.. math::

  x = f(x) + x

where :math:`f` is a sequence of layers such as convolutions and nonlinearities.

Motivation
_____________
There are a number of hypothesized reasons for why residual networks are effective:

* Shorter paths: The skip connections provide short paths between the input and output, making residual networks able to avoid the vanishing gradient problem more easily.
* Increased depth: As a result of the reduced vanishing gradients problem ResNets can be trained with more layers, enabling more sophisticated functions to be learnt.
* Ensembling effect: `Veit et al. (2016) <https://arxiv.org/pdf/1605.06431.pdf>`_ demonstrate that a residual network can be seen as an ensemble of sub-networks of different lengths.

Comparison with Highway Networks
___________________________________
`Highway Networks, Srivastava et al (2015) <https://arxiv.org/abs/1505.00387>`_ also use skip connections to attempt to make it easier to train very deep networks. In contrast to Residual Networks their connections are gated as follows:

.. math::

  y = H(x, W_H) \cdot T(x, W_T) + x \cdot (1 - T(x, W_T))

Comparisons between the accuracies of the two approaches suggest the gating is not useful and so is detrimental overall as it increases the number of parameters and the computational complexity of the network.

| **Proposed in**
| `Deep Residual Learning for Image Recognition, He et al. (2015) <https://arxiv.org/abs/1512.03385>`_

VGG
----
A CNN that secured the first and second place in the 2014 ImageNet localization and classification tracks, respectively. VGG stands for the team which submitted the model, Oxford’s Visual Geometry Group. The VGG model consists of 16–19 weight layers and uses small convolutional filters of size 3x3 and 1x1.

| **Proposed in**
| `Very deep convolutional networks for large-scale image recognition, Simonyan and Zisserman (2015) <https://arxiv.org/abs/1409.1556>`_
