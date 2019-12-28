"""""""""""""""""""
Computer vision
"""""""""""""""""""

Tasks which have an image or video as their input. This includes:

* Image captioning
* Image classification
* `Image segmentation <https://ml-compiled.readthedocs.io/en/latest/computer_vision.html#image-segmentation>`_
* `Image-to-image translation <https://ml-compiled.readthedocs.io/en/latest/computer_vision.html#image-to-image-translation>`_
* `Object detection <https://ml-compiled.readthedocs.io/en/latest/computer_vision.html#object-detection>`_

Challenges
------------

* Parts of the object may be obscured.
* Photos can be taken at different angles.
* Different lighting conditions. Both the direction and amount of light may differ, as well as the number of light sources.
* Objects belonging to one class can come in a variety of forms.

Data augmentation
--------------------
The images in the training set are randomly altered in order to improve the generalization of the network.

`Cubuk et al. (2018) <https://arxiv.org/pdf/1805.09501.pdf>`_, who evaluate a number of different data augmentation techniques, use the following transforms:

* Blur - The entire image is blurred by a random amount.
* Brightness
* Color balance
* Contrast
* Cropping - The image is randomly cropped and the result is fed into the network instead.
* Cutout - Mask a random square region of the image, replacing it with grey. Was used to get state of the art results on the CIFAR-10, CIFAR-100 and SVHN datasets. Proposed in `Improved Regularization of Convolutional Neural Networks with Cutout, DeVries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_
* Equalize - Perform histogram equalization on the image. This adjusts the contrast.
* Flipping - The image is flipped with probability 0.5 and left as it is otherwise. Normally only horizontal flipping is used but vertical flipping can be used where it makes sense - satellite imagery for example.
* Posterize - Decrease the bits per pixel
* Rotation
* Sample pairing - Combine two random images into a new synthetic image. See `Data Augmentation by Pairing Samples for Images Classification, Inoue (2018) <https://arxiv.org/pdf/1801.02929.pdf>`_.
* Shearing
* Solarize - Pixels above a random value are inverted.
* Translation

Datasets
---------

CIFAR-10/100
______________
60000 32x32 colour images in 10 (100) classes with 6000 (600) images each. 50000 images in the training set and 10000 in the test.

Notable results - CIFAR-10

* 98.9% - `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Tan and Le (2019) <https://arxiv.org/abs/1905.11946>`_
* 98.5% - `AutoAugment: Learning Augmentation Strategies from Data, Cubuk et al. (2018) <https://arxiv.org/pdf/1805.09501.pdf>`_
* 97.6% - `Learning Transferable Architectures for Scalable Image Recognition, Zoph et al. (2017) <https://arxiv.org/pdf/1707.07012.pdf>`_
* 97.4% - `Improved Regularization of Convolutional Neural Networks with Cutout, DeVries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_
* 96.1% - `Wide Residual Networks, Zagoruyko and Komodakis (2016) <https://arxiv.org/pdf/1605.07146.pdf>`_
* 94.2% - `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_
* 93.6% - `Deep Residual Learning for Image Recognition, He et al. (2015) <https://arxiv.org/abs/1512.03385>`_
* 93.5% - `Fast and Accurate Deep Network Learning by Exponential Linear Units, Clevert et al. (2015) <https://arxiv.org/abs/1511.07289>`_

Notable results - CIFAR-100

* 91.7% - `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Tan and Le (2019) <https://arxiv.org/abs/1905.11946>`_
* 89.3% - `AutoAugment: Learning Augmentation Strategies from Data, Cubuk et al. (2018) <https://arxiv.org/pdf/1805.09501.pdf>`_
* 84.8% - `Improved Regularization of Convolutional Neural Networks with Cutout, de Vries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_
* 81.1% - `Wide Residual Networks, Zagoruyko and Komodakis (2016) <https://arxiv.org/pdf/1605.07146.pdf>`_
* 75.7% - `Fast and Accurate Deep Network Learning by Exponential Linear Units, Clevert et al. (2015) <https://arxiv.org/abs/1511.07289>`_
* 72.3% - `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_

https://keras.io/datasets/#cifar10-small-image-classification

COCO
_________
Common Objects in COntext. A dataset for image recognition, segmentation and captioning.

Detection task - Notable results (mAP):

* 51.0% - `EfficientDet: Scalable and Efficient Object Detection, Tan et al. (2019) <https://arxiv.org/abs/1911.09070v1>`_
* 48.3% - `NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection, Ghaisi et al. (2019) <https://arxiv.org/pdf/1904.07392.pdf>`_
* 42.1% - `AutoAugment: Learning Augmentation Strategies from Data, Cubuk et al. (2018) <https://arxiv.org/pdf/1805.09501.pdf>`_
* 35.9% - `Fast R-CNN, Girshick et al. (2015) <https://arxiv.org/abs/1504.08083>`_

ImageNet (ILSVRC)
___________________
ILSVRC stands for Imagenet Large Scale Recognition Challenge. Popular image classification task in which the algorithm must use a dataset of ~1.4m images to classify 1000 classes.

Notable results (top-1 accuracy):

* 87.4% - `Self-training with Noisy Student improves ImageNet classification, Xie et al. (2019) <https://arxiv.org/pdf/1911.04252v1.pdf>`_
* 85.0% - `RandAugment: Practical data augmentation with no separate search, Cubuk et al. (2019) <https://arxiv.org/pdf/1909.13719v1.pdf>`_
* 84.4% - `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Tan and Le (2019) <https://arxiv.org/abs/1905.11946>`_
* 83.9% - `Regularized Evolution for Image Classifier Architecture Search, Real et al. (2018) <https://arxiv.org/pdf/1802.01548.pdf>`_
* 83.5% - `AutoAugment: Learning Augmentation Strategies from Data, Cubuk et al. (2018) <https://arxiv.org/pdf/1805.09501.pdf>`_
* 82.7% - `Learning Transferable Architectures for Scalable Image Recognition, Zoph et al. (2017) <https://arxiv.org/pdf/1707.07012.pdf>`_
* 78.6% - `Deep Residual Learning for Image Recognition, He et al. (2015) <https://arxiv.org/abs/1512.03385>`_
* 76.3% - `Very deep convolutional networks for large-scale image recognition, Simonyan and Zisserman (2014) <https://arxiv.org/abs/1409.1556>`_
* 62.5% - `ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky et al. (2012) <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

NB: `Xie et al. (2019) <https://arxiv.org/pdf/1911.04252v1.pdf>`_ also use unlabeled data.

MNIST
________
70000 28x28 pixel grayscale images of handwritten digits (10 classes), 60000 in the training set and 10000 in the test set.

http://yann.lecun.com/exdb/mnist/

https://keras.io/datasets/#mnist-database-of-handwritten-digits

Pascal VOC
____________
`PASCAL Visual Object Classes Homepage <http://host.robots.ox.ac.uk/pascal/VOC/>`_

SVHN
______
Street View House Numbers.

Face recognition
--------------------
The name of the general topic. Includes face identification and verification.

The normal face recognition pipeline is:

* Face detection - Identifying the area of the photo that corresponds to the face.
* Face alignment - Often done by detecting facial landmarks like the nose, eyes and mouth.
* Feature extraction and similarity calculation

Challenges
______________
In addition to the standard challenges in computer vision facial recognition also encounters the following problems:

* Changes in facial hair.
* Glasses, which may not always be worn.
* People aging over time.

Datasets
_________

* LFW
* YouTube-Faces
* CASIA-Webface
* CelebA

Face identification
______________________
Multiclass classification problem. Given an image of a face, determine the identity of the person.

Face verification
___________________
Binary classification problem. Given two images of faces, assess whether they are from the same person or not.

Commonly used architectures for solving this problem include Siamese and Triplet networks.

Image segmentation
--------------------
Partitions an object into meaningful parts with associated labels. May also be referred to as per-pixel classification.

| **Further reading**
| `U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al. (2015) <https://arxiv.org/abs/1505.04597>`_

Instance segmentation
_______________________
Unlike semantic segmentation, different instances of the same object type have to be labelled as separate objects (eg person 1, person 2). Harder than semantic segmentation.

Semantic segmentation
_______________________
Unlike instance segmentation, in semantic segmentation it is only necessary to predict what class each pixel belongs to, not separate out different instances of the same class.

Weakly-supervised segmentation
_________________________________
Learning to segment from only image-level labels. The labels will describe the classes that exist within the image but not what the class is for every pixel.

The results from weak-supervision are generally poorer than otherwise but datasets tend to be much cheaper to acquire. 

When the dataset is only weakly-supervised it can be very hard to correctly label highly-correlated objects that are usually only seen together, such as a train and rails.

Image-to-image translation
---------------------------
Examples:

* Daytime to nighttime
* Greyscale to colour
* Streetmap to satellite view

`Image-to-Image Translation with Conditional Adversarial Networks, Isola et al. (2016) <https://arxiv.org/abs/1611.07004>`_

Object detection
-------------------

One-stage detector
_____________________

Contrast with two-stage detectors.

| **Example papers**
| `Focal Loss for Dense Object Detection, Lin et al. (2017) <https://arxiv.org/pdf/1708.02002.pdf>`_
| `YOLO9000: Better, Faster, Stronger, Redmon and Farhadi (2016) <https://arxiv.org/abs/1612.08242>`_
| `You Only Look Once: Unified, Real-Time Object Detection, Redmon et al. (2015) <https://arxiv.org/abs/1506.02640>`_
| `SSD: Single Shot MultiBox Detector, Liu et al. (2015) <https://arxiv.org/abs/1512.02325>`_

Region of interest
_______________________
See 'region proposal'.

Region proposal
________________
A region in an image (usually defined by a rectangle) identified as containing an object of interest with high probability, relative to the background.

Two-stage detector
____________________
The first stage proposes regions that may contain objects of interest. The second stage classifies these regions as either background or one of the classes. 

There is often a significant class-imbalance problem since background regions greatly outnumber the other classes.

Contrast with one-stage detectors.

| **Example papers for the first stage**
| `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Ren et al. (2015) <https://arxiv.org/abs/1506.01497>`_
| `Edge Boxes: Locating Object Proposals from Edges, Zitnick and Dollar (2014) <https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf>`_
| `Selective Search for Object Recognition, Uijlings et al. (2012) <http://www.huppelen.nl/publications/selectiveSearchDraft.pdf>`_
|
| **Example papers for the second stage**
| `Mask R-CNN, He et al. (2017) <https://arxiv.org/abs/1703.06870>`_
| `Fast R-CNN, Girshick et al. (2015) <https://arxiv.org/abs/1504.08083>`_

Saliency map
---------------
A heatmap over an image which shows each pixel's importance for the classification.

