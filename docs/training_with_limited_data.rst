"""""""""""""""""""""""""""
Training with limited data
"""""""""""""""""""""""""""

Active learning
----------------
The learning algorithm requests examples to be labelled as part of the training process. Useful when there is a small set of labelled examples and a larger set of unlabelled examples and labelling is expensive.

Class imbalance problem
--------------------------
When one or more classes occur much more frequently in the dataset than others. This can lead to classifiers maximising their objective by predicting the majority class(es) all of the time, ignoring the features.

Datasets
----------

Omniglot
__________
1623 handwritten characters from 50 alphabets. Useful for one-shot learning.

**Notable results**

* 93.8% - `Matching Networks for One Shot Learning, Vinyals et al. (2016) <https://arxiv.org/abs/1606.04080>`_

One-shot learning
------------------
Classification where only one member of that class has been seen before. Matching Networks achieve 93.2% top-5 accuracy on ImageNet compared to 96.5% for Inception v3.

Semi-supervised learning
---------------------------
Training using a limited set of labelled data and a (usually much larger) set of unlabelled data.

Ladder Network
_______________
A network designed for semi-supervised learning that also works very well for permutation invariant MNIST.

Simultaneously minimize the sum of supervised and unsupervised cost functions by backpropagation, avoiding the need for layer-wise pre-training. The learning task is similar to that of a denoising autoencoder, but minimizing the reconstruction error at every layer, not just the inputs. Each layer contributes a term to the loss function.

The architecture is an autoencoder with skip-connections from the encoder to the decoder. Can work with both fully-connected and convolutional layers.

There are two encoders - one for clean and one for noisy data. The clean one is used to predict labels and get the supervised loss. The noisy one links with the decoder and helps create the unsupervised losses. Both encoders have the same parameters.

The loss is the sum of the supervised and the unsupervised losses. The supervised cost is the cross-entropy loss as normal. The unsupervised cost (reconstruction error) is the squared difference.

The hyperparameters are the weight for the denoising cost of each layer as well as the amount of noise to be added within the corrupted encoder.

Achieved state of the art performance for semi-supervised MNIST and CIFAR-10 and permutation invariant MNIST.

`Semi-Supervised Learning with Ladder Networks, Rasmus et al. (2015) <https://arxiv.org/abs/1507.02672>`_

Self-training
_______________
Method for semi-supervised learning. A model is trained on the labelled data and then used to classify the unlabelled data, creating more labelled examples. This process then continues iteratively. Usually only the most confident predictions are used at each stage.

Unsupervised pre-training
____________________________
Layers are first trained using an auto-encoder and then fine tuned over labelled data. Improves the initialization of the weights, making optimization faster and reducing overfitting. Most useful in semi-supervised learning.

Transfer learning
-------------------
The process of taking results (usually weights) that have been obtained on one dataset and applying them to another to improve accuracy on that one.

Useful for reducing the amount of training time and data required.

Zero-shot learning
----------------------
Learning without any training examples. This is made possible by generalising from a wider dataset.

An example is learning to recognise a cat having only read information about them - no images of cats are seen. This could be done by using Wikipedia with a dataset like ImageNet to learn a joint embedding between words and images.

`Zero-Shot Learning Through Cross-Modal Transfer, Socher et al. (2013) <https://nlp.stanford.edu/~socherr/SocherGanjooManningNg_NIPS2013.pdf>`_
