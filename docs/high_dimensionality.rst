High dimensionality
"""""""""""""""""""""""

Euclidean distance
-------------------
* For points randomly distributed in space, the distribution of distances between them falls tightly around the mean. This is because the Euclidean distance is the nth root of the sum of distances along each dimension. So this becomes close to the mean, just as for any sufficiently large sample.
* For this reason the Euclidean distance is less useful.
* This also means the ratio between the distance between the two furthest points and the distance between the two closest approaches 1 for high dimensions.

Gaussian distribution
------------------------
* Although the value remains highest at the origin, there is very little volume there. Most points are in the ‘tails’. This reflects the intuition that over many dimensions, any given point is likely to be anomalous in at least one aspect.

Sphere
--------
* There is almost no interior volume. This follows the same intuition as for the Gaussian distribution - a random point is likely to be near the edge in at least one dimension, which is sufficient to call it exterior.
* The volume is mostly contained in a thin ring around the equator at the surface.
* The surface area is almost all at the equator.

Interpolation
--------------
* Linearly interpolating between two high-dimensional vectors will produce something that doesn't look much like either. The entries will tend to be atypically close to the mean. Polar interpolation should be used instead.

Inner product of random samples
--------------------------------
* Two random vectors from a high-dimensional space are likely to be close to orthogonal. This is because orthogonality is measured by the inner product, which is the sum of elementwise products. Over a large number of dimensions, this will tend towards the mean of the products which will be zero, so long as the mean of the sampling distribution is also zero.

https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/chap1-high-dim-space.pdf  

http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble
