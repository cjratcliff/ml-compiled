""""""""""""""""""""""""""
Convolutional networks
""""""""""""""""""""""""""

AlexNet
--------
Performed considerably better than the state of the art at the time. Has 60 million parameters, 650,000 neurons and includes five convolutional layers.

The two 'streams' only exist to allow training on two GPUs.

`ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky et al. (2012) <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

GoogLeNet
-------------
CNN that won the ILSVRC 2014 challenge. Composed of 9 inception layers.

`Going Deeper with Convolutions, Szegedy et al. (2014) <https://arxiv.org/abs/1409.4842>`_

LeNet5
--------

Residual network
-------------------
Used for training very deep networks. The original paper achieved 152 layers, 8 times deeper than VGG nets. Used for image recognition, winning first place in the ILSVRC 2015 classification task. Residual connections can also be used to create deeper RNNs such as Google’s 16-layer RNN encoder-decoder (Wu et al., 2016).

Uses shortcut connections performing the identity mapping, which are added to the outputs of the stacked layers. Based on the theory that  is easier to optimise than . Each layer uses the equation 

Similar but superior to highway networks as they do not introduce any extra parameters.

VGG
----
A CNN that secured the first and second place in the 2014 ImageNet localization and classification tracks, respectively. VGG stands for the team which submitted the model, Oxford’s Visual Geometry Group. The VGG model consists of 16–19 weight layers and uses small convolutional filters of size 3x3 and 1x1.

`Very deep convolutional networks for large-scale image recognition, Simonyan and Zisserman (2015) <https://arxiv.org/abs/1409.1556>`_