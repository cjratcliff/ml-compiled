Geometry
"""""""""""""""""""""""

Affine transformation
-----------------------
A linear mapping that preserves points, lines and planes. Examples include translation, scale, rotation or shear.

Cosine similarity
-----------------------
Measures the similarity of two vectors by calculating the cosine of the angle between them. The similarity is 1 if the vectors are pointing in the same direction, 0 if they are orthogonal, and -1 if they are pointing in exactly opposite directions.

.. math ::

    c(x,y) = xy/(||x||_2 \cdot ||y||_2)


This means it is distinct from the Euclidean distance, which takes magnitude into account. The squared euclidean distance of two L2-normalized vectors is closely related but not identical to the cosine similarity of those vectors.

The major differences between the Euclidean and cosine similarity are as follows:

* Unlike the Euclidean distance, the cosine distance does not suffer from the curse of dimensionality, making it useful for comparing high-dimensional feature vectors.
* The cosine distance ‘wraps around’. This means it does not satisfy the triangle inequality. Requires one extra dimension in the vectors, relative to the Euclidean distance, to store the same amount of information. * The 2D cosine similarity is a circle, which is 1D.

Euclidean distance
-----------------------
Can have poor performance under high dimensionality.

For points randomly distributed in space, the distribution of distances between them falls tightly around the mean. This is because the Euclidean distance is the nth root of the sum of distances along each dimension. So this becomes close to the mean, just as for any sufficiently large sample.
For this reason the Euclidean distance is less useful.
This also means the ratio between the distance between the two furthest points and the distance between the two closest approaches 1 for high dimensions.

High dimensionality
--------------------

Euclidean distance
'''''''''''''''''''''''
* For points randomly distributed in space, the distribution of distances between them falls tightly around the mean. This is because the Euclidean distance is the nth root of the sum of distances along each dimension. So this becomes close to the mean, just as for any sufficiently large sample.
* For this reason the Euclidean distance is less useful.
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
* Two random vectors from a high-dimensional space are likely to be close to orthogonal. This is because orthogonality is measured by the inner product, which is the sum of elementwise products. Over a large number of dimensions, this will tend towards the mean of the products which will be zero, so long as the mean of the sampling distribution is also zero.

https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/chap1-high-dim-space.pdf  

http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble

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

    
