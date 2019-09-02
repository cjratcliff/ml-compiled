"""""""""""""""""""
Computer vision
"""""""""""""""""""

Data augmentation
--------------------
The images in the training set are randomly altered in order to improve the generalization of the network.

Random flipping
___________________
The image is flipped with probability 0.5 and left as it is otherwise. Normally only horizontal flipping is used but vertical flipping can be used where it makes sense, satellite imagery for example.

Random cropping
______________________
The image is cropped and the result is fed into the network instead. 

Cutout
________
Regularization method that masks a random square region of the image, replacing it with grey.

Was used to get new state of the art methods on the CIFAR-10, CIFAR-100 and SVHN datasets (DeVries and Taylor, 2017).

| **Proposed in**
| `Improved Regularization of Convolutional Neural Networks with Cutout, DeVries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_

Datasets
---------

CIFAR-10/100
______________
60000 32x32 colour images in 10 (100) classes with 6000 (600) images each. 50000 images in the training set and 10000 in the test.

Notable results - CIFAR-10

* 97.6% - `Learning Transferable Architectures for Scalable Image Recognition, Zoph et al. (2017) <https://arxiv.org/pdf/1707.07012.pdf>`_
* 97.4% - `Improved Regularization of Convolutional Neural Networks with Cutout, de Vries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_
* 96.1% - `Wide Residual Networks, Zagoruyko and Komodakis (2016) <https://arxiv.org/pdf/1605.07146.pdf>`_
* 94.2% - `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_
* 93.6% - `Deep Residual Learning for Image Recognition, He et al. (2015) <https://arxiv.org/abs/1512.03385>`_
* 93.5% - `Fast and Accurate Deep Network Learning by Exponential Linear Units, Clevert et al. (2015) <https://arxiv.org/abs/1511.07289>`_

Notable results - CIFAR-100

* 84.8% - `Improved Regularization of Convolutional Neural Networks with Cutout, de Vries and Taylor (2017) <https://arxiv.org/pdf/1708.04552.pdf>`_
* 81.1% - `Wide Residual Networks, Zagoruyko and Komodakis (2016) <https://arxiv.org/pdf/1605.07146.pdf>`_
* 75.7% - `Fast and Accurate Deep Network Learning by Exponential Linear Units, Clevert et al. (2015) <https://arxiv.org/abs/1511.07289>`_
* 72.3% - `All you need is a good init, Mishkin and Matas (2015) <https://arxiv.org/abs/1511.06422>`_

https://keras.io/datasets/#cifar10-small-image-classification

COCO
_________
Common Objects in COntext. A dataset for image recognition, segmentation and captioning.

MNIST
________
70000 28x28 pixel grayscale images of handwritten digits (10 classes), 60000 in the training set and 10000 in the test set.

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
* Photos being taken at different angles.
* Different lighting conditions.
* Changes in facial hair.
* Glasses.
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

ILSVRC
-------
Imagenet Large Scale Recognition Challenge. Popular image classification task in which the algorithm must use a dataset of ~1.4m images to classify 1000 classes.

Notable results
_________________
* AlexNet
* GoogLeNet
* ResNet 
* NASNet

Image segmentation
--------------------
Partitions an object into meaningful parts with associated labels. May also be referred to as per-pixel classification.

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

R-CNN
------
Type of network for object detection. Stands for Region-based CNN. 

| **Further reading**
| `Fast R-CNN, Girshick et al. (2015) <https://arxiv.org/abs/1504.08083>`_
| `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Ren et al. (2015) <https://arxiv.org/abs/1506.01497>`_
| `Mask R-CNN, He et al. (2017) <https://arxiv.org/abs/1703.06870>`_

Region of interest
--------------------
A region in an image (usually defined by a rectangle) identified as containing an object of interest with high probability, relative to the background.

Saliency map
---------------
A heatmap over an image which shows each pixel's importance for the classification.

