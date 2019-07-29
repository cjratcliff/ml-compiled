Geometry
"""""""""""""""""""""""

Affine transformation
-----------------------
A linear mapping that preserves points, lines and planes. Examples include translation, scale, rotation or shear.

Cosine similarity
-----------------------
Measures the similarity of two vectors by calculating the cosine of the angle between them. The similarity is 1 if the vectors are pointing in the same direction, 0 if they are orthogonal, and -1 if they are pointing in exactly opposite directions.

.. math ::

    c(x,y) = x \cdot y/(||x||_2 \cdot ||y||_2)

Where :math:`x \cdot y` is the dot product.

Relationship with the Euclidean distance
'''''''''''''''''''''''''''''''''''''''''''
The major differences between the Euclidean distance and cosine similarity are as follows:

* The Euclidean distance takes magnitude of the two vectors into account. The cosine similarity ignores it.
* Unlike the Euclidean distance, the cosine distance does not suffer from the curse of dimensionality, making it useful for comparing high-dimensional feature vectors.
* The cosine similarity is not a metric in the mathematical sense.
* The cosine similarity is bounded between -1 and 1, whereas the Euclidean distance must be between 0 and infinity.
* The Euclidean distance for two identical vectors is zero. The cosine similarity in this case is 1.
* The cosine similarity does not satisfy the triangle inequality.
* The cosine similarity is undefined if one of the vectors is all zeros.

There is a linear relationship between the cosine similarity of two vectors and the squared Euclidean distance if the vectors first undergo L2 normalization.

The proof is as follows:

Let :math:`x` and :math:`y` be two vectors that have been normalized such that :math:`||x||_2 = ||y||_2 = 1`. Then the expression for the squared Euclidean distance is:

.. math::

  ||x - y||_2^2 

.. math::
  = (x-y)^T(x-y)
  
.. math::
  = x^Tx -2x^Ty + y^Ty
  
.. math::
  = ||x||_2 - 2x^Ty + ||y||_2
  
.. math::
  = 2 - 2x^Ty
  
.. math::
  = 2 - 2c(x,y)
  

Euclidean distance
-----------------------
Measures the distance between two vectors.

.. math::

  d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}

Disadvantages
'''''''''''''''''''''''
The Euclidean distance can have poor performance under high dimensionality. For points randomly distributed in space, the distribution of distances between random pairs of points falls tightly around the mean. This is because the Euclidean distance is the nth root of the sum of distances along each dimension. So this becomes close to the mean, just as for any sufficiently large sample.

The ratio between the distance between the two furthest points and the distance between the two closest approaches 1 as the dimensionality increases.

High dimensionality
--------------------

Euclidean distance
'''''''''''''''''''''''
* For points randomly distributed in space, the distribution of the distances between them falls tightly around the mean.
* For this reason the usefulness of the Euclidean distance is limited in high dimensions.
* This also means the ratio between the distance between the two furthest points and the distance between the two closest approaches 1 for high dimensions.

Gaussian distribution
'''''''''''''''''''''''
* Although the value remains highest at the origin, there is very little volume there. Most points are in the ‘tails’. This reflects the intuition that over many dimensions, any given point is likely to be anomalous in at least one aspect.

Sphere
'''''''''''''''''''''''
* There is almost no interior volume. This follows the same intuition as for the Gaussian distribution - a random point is likely to be near the edge in at least one dimension, which is sufficient to call it exterior.
* The volume is mostly contained in a thin ring around the equator at the surface.
* The surface area is almost all at the equator.

Interpolation
'''''''''''''''''''''''
* Linearly interpolating between two high-dimensional vectors will produce something that doesn't look much like either. The entries will tend to be atypically close to the mean. Polar interpolation should be used instead.

Inner product of random samples
''''''''''''''''''''''''''''''''''''''''''''''
* Two random high-dimensional vectors are likely to be close to orthogonal. This is because orthogonality is measured by the inner product, which is the sum of the elementwise products. Over a large number of dimensions, this will tend towards the mean of the products which will be zero, so long as the mean of the sampling distribution is also zero.

https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/chap1-high-dim-space.pdf  

http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble

Lebesgue measure
-------------------
The concept of volume, generalised to an arbitrary number of dimensions. In one dimension it is the same as length and in two it is the same as area.

Manifold
-----------
Type of topological space. Includes lines, circles, planes, spheres and tori.

Metric
--------
A metric :math:`d(x,y)` must have the following properties:

.. math::

    d(x,y) \geq 0

    d(x,y) = 0 	\Leftrightarrow x = y

    d(x,y) = d(y,x)    

    d(x,z) \leq d(x,y) + d(y,z)    
    
Polar interpolation
-----------------------
For two vectors x and y, linear interpolation is :math:`px + (1-p)y`, where :math:`0 \leq p \leq 1`.

Polar interpolation by contrast, is:

.. math::

    \sqrt{p}x + \sqrt{1-p}y


Unlike linear interpolation, the sum of the coefficients can exceed 1.

http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/

Wasserstein distance
-------------------------
Also known as the earth mover distance. Like the Kullback-Leibler divergence, it is a way of measuring the difference between two different probability distributions.

Intuition
'''''''''''''''''''''''
If the two probability distributions are visualised as mounds of earth, the Wasserstein distance is the amount of effort required to turn one mound into the other. That is, the amount of earth mutliplied by the distance it has to be moved.

Defining the Wasserstein distance
''''''''''''''''''''''''''''''''''''''''''''''
There are many different ways to move the earth so calculating the Wasserstein distance requires solving an optimisation problem, in general.

An exact solution exists if both distributions are normal.

Properties
'''''''''''''''''''''''
Unlike the Kullback-Leibler divergence, Jensen-Shannon divergence and total variation distance, this metric does not have zero gradients when the supports of P and Q are disjoint (the probability distributions have no overlap).

Exact computation of the Wasserstein distance is intractable.

https://vincentherrmann.github.io/blog/wasserstein/

    
