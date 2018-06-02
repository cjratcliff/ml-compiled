Linear algebra
""""""""""""""""

Adjoint
=======
Another term for the conjugate transpose. Identical to the transpose if the matrix is real.

Affine combination
=====================
A linear combination of vectors where the weights sum to 1. Unlike a convex combination, the weights can be negative.

Cholesky decomposition
============================
:math:`A = LL^*`, where A is Hermitian and positive-definite, L is lower-triangular and :math:`L^*` is its conjugate transpose. Can be used for solving SLEs.

Conjugate transpose
=====================
The matrix obtained by taking the transpose followed by the complex conjugate of each entry.

Dot product
=============

.. math::

  a \cdot b = \sum_{i=1}^n a_i b_i

Eigendecomposition
=====================

.. math::

    A = Q \Lambda Q^*

Where the columns of Q are the eigenvectors of A. :math:`\Lambda` is a diagonal matrix in which :math:`\Lambda_{ii}` is the i'th eigenvalue of A.

Eigenvalues and eigenvectors
===============================
Let :math:`A` be a square matrix. Then the eigenvalues and eigenvectors of the matrix are the vectors :math:`v` and scalars :math:`\lambda` respectively that satisfy the equation:

.. math::

  Av = \lambda v 
  
Satisfy the following properties:

* The trace of A is the sum of its eigenvalues.
* The determinant of A is the product of its eigenvalues.

Gaussian elimination
=====================
An algorithm for solving SLEs that iteratively transforms the matrix into an upper triangular one in row echelon form.

Hermitian matrix
=====================
The complex equivalent of a symmetric matrix. :math:`A = A^*`, where * represents the conjugate transpose.

Hadamard product
=====================
Synonymous with elementwise-multiplication.

Invertible
===========
A matrix :math:`A` is invertible if and only if there exists a matrix :math:`B` such that :math:`AB = BA = I`.

LU Decomposition
=====================
A = LU, where L is lower triangular and U is upper triangular. Can be used to solve SLEs.

Normal matrix
==============
:math:`A^*A = AA^*` where :math:`A^*` is the conjugate transpose of :math:`A`.

Orthogonal matrix
=====================

.. math:: 

    A^TA = AA^T = I
    
Orthonormal vectors
====================
Two vectors are orthonormal if they are orthogonal and both unit vectors.

Outer product
==============

Positive (semi-)definite
============================
A matrix :math:`A \in \mathbb{R}^{n \times n}` is positive definite if:

.. math::

    z^TAz > 0, \forall z \in \mathbb{R}^n, z \neq 0 

Positive semi-definite matrices are defined analogously, except with :math:`z^TAz \geq 0`

Negative (semi-)definite matrices are the same but with the inequality round the other way.

Principal Component Analysis (PCA)
=====================================
Approximates a dataset with a set of smaller linearly uncorrelated variables. These variables can be found through eigenvalue decomposition.

.. TODO: Formula

QR decomposition
=====================
Decomposes a real square matrix :math:`A` such that :math:`A = QR`. :math:`Q` is an `orthogonal matrix <http://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#orthogonal-matrix>`_ and :math:`R` is upper triangular.

Rank
=======

Matrix rank
------------
The number of linearly independent columns.

Tensor rank
------------
When the term is applied to tensors, the rank refers to the dimensionality:
* Rank 0 is a scalar
* Rank 1 is a vector
* Rank 2 is a matrix etc.

Singular matrix
=====================
A square matrix which is not invertible. A matrix is singular if and only if the determinant is zero.

Singular value
=====================

Singular value decomposition (SVD)
===================================
Matrix factorization algorithm.

.. math::

    A = U\Sigma V^*

where :math:`U` is a unitary matrix, :math:`\Sigma` is a rectangular diagonal matrix containing the singular values and :math:`V` is a unitary matrix.

Can be used for computing the sum of squares or the pseudoinverse.

Span
=======
The span of a matrix is the set of all points that can be obtained as a linear combination of the vectors in the matrix.

Spectral radius
=====================
The maximum of the magnitudes of the eigenvalues.

Spectrum
==============
The set of eigenvalues of a matrix.

System of Linear Equations (SLE)
======================================

.. math::
 
  Ax = b
  
Can be solved by Gaussian elimination. 

Trace
=======
The sum of the elements along the main diagonal of a square matrix.

.. math::

  \text{tr}(A) = \sum_{i=1}^n A_{ii}
  
Satisfies the following properties:

.. math::

  \text{tr}(A) = \text{tr}(A^T)
  
  \text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)
  
  \text{tr}(cA) = c\text{tr}(A)

Transpose
==============

.. math::

  (A^T)_{ij} = A_{ji}

Satisfies the following properties:

.. math::

    (A+B)^T = A^T + B^T

    (AB)^T = B^TA^T

    (A^T)^{-1} = (A^{-1})^T

Unitary matrix
=====================
A matrix where its inverse is the same as its complex conjugate. The complex version of an orthogonal matrix.

.. math::

  A^*A = AA^* = I
  
ZCA
=====
Like PCA, ZCA converts the data to have zero mean and an identity covariance matrix. Unlike PCA, it does not reduce the dimensionality of the data and tries to create a whitened version that is minimally different from the original.
  
