"""""""""""""""
Layers
"""""""""""""""

1x1 convolutions
------------------------
These are actually matrix multiplications, not convolutions. They are a useful way of increasing the depth of the neural network since they are equivalent to f(Wh), where f is the activation function.

If the number of channels decreases from one layer to the next they can be also be used for dimensionality reduction.

Network in Network, Lin et al. (2014)

http://iamaaditya.github.io/2016/03/one-by-one-convolution/

Affine layer
--------------
A fully-connected layer.

Batch normalization
-------------------------
Normalizes the input vector to a layer to have zero mean and unit variance. Training deep neural networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization. This phenomenon is referred to as internal covariate shift.

Convolutional layer
-----------------------
Transforms an image according to the convolution operation shown below, where the image on the left is the input and the image being created on the right is the output:

TODO

Applying the kernel to pixels near or at the edges of the image will result in needing pixel values that do not exist. There are two ways of resolving this:

Only apply the kernel to pixels where the operation is valid. For a kernel of size k this will reduce the image by (k-1)/2 pixels on each side.
Pad the image with zeros to allow the operation to be defined.
The same convolution operation is applied to every pixel in the image, resulting in a considerable amount of weight sharing. This means convolutional layers are quite efficient in terms of parameters.

The number of parameters can be further reduced by setting a stride so the convolution operation is only applied every m pixels.

Can be represented by a fully-connected layer in theory. Such a layer would be mostly zeros as the effects are local. This is especially true if the layer is replicating multiple filters.

'''''''''''''''''''''''''''''
Separable convolution/filter
'''''''''''''''''''''''''''''
A filter or kernel is separable if it (a matrix) can be expressed as the product of a row vector and a column vector. This decomposition can reduce the computational cost of the convolution. Examples include the Sobel edge detection and Gaussian blur filters.

Inception layer
--------------------

RoI pooling
--------------
Used to solve the problem that the regions of interest (RoI) identified by the bounding boxes can be different shapes in object recognition. The CNN requires all inputs to have the same dimensions.

The RoI is divided into a number of rectangles of fixed size (except at the edges). If doing 3x3 RoI pooling there will be 9 rectangles in each RoI. We do max-pooling over each RoI to get 3x3 numbers.

Upsampling layer
-----------------
Used in convolutional autoencoders to go from the the bottleneck layer up to full image.
