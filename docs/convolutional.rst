""""""""""""""""""""""""""
Convolutional networks
""""""""""""""""""""""""""

AlexNet
--------
Performed considerably better than the state of the art at the time. Has 60 million parameters, 650,000 neurons and includes five convolutional layers.

The two 'streams' only exist to allow training on two GPUs.

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

Residual network
-------------------
An architecture that uses skip connections to create very deep networks. The `original paper <https://arxiv.org/abs/1512.03385>`_ achieved 152 layers, 8 times deeper than VGG nets. Used for image recognition, winning first place in the ILSVRC 2015 classification task. Residual connections can also be used to create deeper RNNs such as Google’s 16-layer RNN encoder-decoder (Wu et al., 2016).

Uses shortcut connections performing the identity mapping, which are added to the outputs of the stacked layers. Each residual block uses the equation:

``
residual_block(x) = relu(conv(relu(conv(x))) + x)
``

Similar but superior to `Highway Networks <https://arxiv.org/abs/1505.00387>`_ as they do not introduce any extra parameters.

| **Proposed in**
| `Deep Residual Learning for Image Recognition, He et al. (2015) <https://arxiv.org/abs/1512.03385>`_

VGG
----
A CNN that secured the first and second place in the 2014 ImageNet localization and classification tracks, respectively. VGG stands for the team which submitted the model, Oxford’s Visual Geometry Group. The VGG model consists of 16–19 weight layers and uses small convolutional filters of size 3x3 and 1x1.

| **Proposed in**
| `Very deep convolutional networks for large-scale image recognition, Simonyan and Zisserman (2015) <https://arxiv.org/abs/1409.1556>`_
