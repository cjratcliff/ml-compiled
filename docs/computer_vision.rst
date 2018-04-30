"""""""""""""""""""
Computer vision
"""""""""""""""""""

Data augmentation
--------------------
The images in the training set are randomly altered in order to improve the generalization of the network.

Examples:

* Random flipping - the image is flipped with probability 0.5 and left as it is otherwise. Normally only horizontal flipping is used but vertical flipping can be used where it makes sense, satellite imagery for example.
* Random cropping

Face recognition
--------------------
The name of the general topic. Includes face identification and verification.

Face identification
______________________
Multiclass classification problem. Given an image of a face, determine the identity of the person.

Face verification
___________________
Binary classification problem. Given two images of faces, assess whether they are from the same person or not.

Commonly used architectures for solving this problem include Siamese and Triplet networks.

Instance segmentation
------------------------
Unlike semantic segmentation, different instances of the same object type have to be labelled as separate objects (eg person 1, person 2). Harder than semantic segmentation.

Region of interest
--------------------
A region in an image (usually defined by a rectangle) identified as containing an object of interest with high probability, relative to the background.

Saliency map
---------------
A heatmap over an image which shows each pixel's importance for the classification.

Semantic segmentation
------------------------
Partitions an object into meaningful parts with associated labels. May also be referred to as per-pixel classification. Contrast with instance segmentation.

Weakly supervised segmentation
--------------------------------
Segmentation trained only on images with one or more class labels. There are no ground truth segmentations available.
