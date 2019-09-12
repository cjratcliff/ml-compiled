"""""""""""""""""""""""""""
Training with limited data
"""""""""""""""""""""""""""

Active learning
----------------
The learning algorithm requests examples to be labelled as part of the training process. Useful when there is a small set of labelled examples and a larger set of unlabelled examples and labelling is expensive.

Class imbalance problem
--------------------------
When one or more classes occur much more frequently in the dataset than others. This can lead to classifiers maximising their objective by predicting the majority class(es) all of the time, ignoring the features.

Methods for addressing the problem include:

* Focal loss
* Oversampling the minority class
* Undersampling the majority class

Datasets
----------

Omniglot
__________
1623 handwritten characters from 50 alphabets with 20 examples of each character. Useful for one-shot learning. Introduced in `One shot learning of simple visual concepts, Lake et al. (2011) <https://cims.nyu.edu/~brenden/papers/LakeEtAl2011CogSci.pdf>`_.

| **Notable results**
| 20-way one shot accuracies are reported. This means one labelled example is provided from each of the 20 classes that were not in the training set. The task is then to classify unlabelled examples into these 20 classes.

* 98.2% - `Object-Level Representation Learning for Few-Shot Image Classification, Long et al. (2018) <https://arxiv.org/pdf/1805.10777.pdf>`_
* 93.8% - `Matching Networks for One Shot Learning, Vinyals et al. (2016) <https://arxiv.org/abs/1606.04080>`_
* 88.1% - `Siamese Neural Networks for One-shot Image Recognition, Koch et al. (2015) <https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf>`_

miniImageNet
______________
60,000 84x84 images from 100 classes, each with 600 examples. There are 80 classes in the training set and 20 in the test set. Much harder than Omniglot.

Introduced in `Vinyals et al. (2016) <https://arxiv.org/abs/1606.04080>`_.

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

`Why Does Unsupervised Pre-training Help Deep Learning?, Erhan et al. (2010) <http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf>`_

Transfer learning
-------------------
The process of taking results (usually weights) that have been obtained on one dataset and applying them to another to improve accuracy on that one.

Useful for reducing the amount of training time and data required.

Zero-shot learning
----------------------
Learning without any training examples. This is made possible by generalising from a wider dataset.

An example is learning to recognise a cat having only read information about them - no images of cats are seen. This could be done by using Wikipedia with a dataset like ImageNet to learn a joint embedding between words and images.

`Zero-Shot Learning Through Cross-Modal Transfer, Socher et al. (2013) <https://nlp.stanford.edu/~socherr/SocherGanjooManningNg_NIPS2013.pdf>`_
