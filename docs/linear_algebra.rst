""""""""""""""""
Linear algebra
""""""""""""""""

Adjoint
--------
Another term for the conjugate transpose. Identical to the transpose if the matrix is real.

Affine combination
--------------------
A linear combination of vectors where the weights sum to 1. Unlike a convex combination, the weights can be negative.

Condition number
------------------
The condition number of a matrix :math:`A` is defined as:

.. math::

  \kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}

where :math:`\sigma_{\max}(A)` and :math:`\sigma_{\min}(A)` are the largest and smallest singular values of :math:`A` respectively.

If :math:`\kappa(A)` is high, the matrix :math:`A` is said to be **ill-conditioned**. Conversely, if the condition number is very low (ie close to 0) we say :math:`A` is **well-conditioned**.

Since singular values are always non-negative, condition numbers are also always non-negative.

Conjugate transpose
--------------------
The matrix obtained by taking the transpose followed by the complex conjugate of each entry.

Dot product
----------------

.. math::

  a \cdot b = \sum_{i=1}^n a_i b_i

Eigenvalues and eigenvectors
----------------------------------
Let :math:`A` be a square matrix. Then the eigenvalues and eigenvectors of the matrix are the vectors :math:`v` and scalars :math:`\lambda` respectively that satisfy the equation:

.. math::

  Av = \lambda v 
  
Properties
_____________
The trace of A is the sum of its eigenvalues:

.. math::

  \text{tr}(A) = \sum_i \lambda_i

The determinant of A is the product of its eigenvalues.

.. math::

  \text{det}(A) = \prod_i \lambda_i

Gaussian elimination
-----------------------
An algorithm for solving SLEs that iteratively transforms the matrix into an upper triangular one in row echelon form.

Hadamard product
------------------
Synonymous with elementwise-multiplication.

Inverse
------------
The inverse of a matrix :math:`A` is written as :math:`A^{-1}`.

A matrix :math:`A` is invertible if and only if there exists a matrix :math:`B` such that :math:`AB = BA = I`.

The inverse can be found using:
* Gaussian elimination
* LU decomposition
* Gauss-Jordan elimination

Matrix decomposition
-----------------------
Also known as matrix factorization.

Cholesky decomposition
_________________________
:math:`A = LL^*`, where A is Hermitian and positive-definite, L is lower-triangular and :math:`L^*` is its conjugate transpose. Can be used for solving SLEs.

Eigendecomposition
______________________

.. math::

    A = Q \Lambda Q^*

Where the columns of Q are the eigenvectors of A. :math:`\Lambda` is a diagonal matrix in which :math:`\Lambda_{ii}` is the i'th eigenvalue of A.

LU decomposition
___________________
A = LU, where L is lower triangular and U is upper triangular. Can be used to solve SLEs.

Polar decomposition
_____________________

.. math::

  A = UP
  
where :math:`U` is unitary and :math:`P` is positive semi-definite and Hermitian.

QR decomposition
___________________
Decomposes a real square matrix :math:`A` such that :math:`A = QR`. :math:`Q` is an `orthogonal matrix <http://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#orthogonal-matrix>`_ and :math:`R` is upper triangular.

Singular value decomposition (SVD)
_______________________________________
Matrix factorization algorithm.

.. math::

    A = U\Sigma V^*

where :math:`U` is a unitary matrix, :math:`\Sigma` is a rectangular diagonal matrix containing the singular values and :math:`V` is a unitary matrix.

Can be used for computing the sum of squares or the pseudoinverse.
    
Orthonormal vectors
-----------------------
Two vectors are orthonormal if they are orthogonal and both unit vectors.

Outer product
-----------------
The outer product of two column vectors :math:`x` and :math:`y` is:

.. math::

  A = xy^T

Principal Component Analysis (PCA)
---------------------------------------
Approximates a dataset with a set of smaller linearly uncorrelated variables. These variables can be found through eigenvalue decomposition.

.. TODO: Formula

Rank
--------

Matrix rank
____________
The number of linearly independent columns.

Tensor rank
____________
When the term is applied to tensors, the rank refers to the dimensionality:
* Rank 0 is a scalar
* Rank 1 is a vector
* Rank 2 is a matrix etc.

Singular values
-----------------
For a matrix A the singular values are the set of numbers:

.. math::

  \sigma_1 \geq \sigma_2 \geq ... \geq \sigma_n \geq 0
  
where :math:`\sigma_i = \sqrt{\lambda_i}` and :math:`\lambda_i` is an eigenvalue of the matrix :math:`A^{T}A`.

Span
-------
The span of a matrix is the set of all points that can be obtained as a linear combination of the vectors in the matrix.

Spectral norm
-----------------
The maximum `singular value <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#singular_values>`_ of a matrix.

Spectral radius
----------------
The maximum of the magnitudes of the `eigenvalues <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#eigenvalues-and-eigenvectors>`_.

Spectrum
-------------
The set of `eigenvalues <https://ml-compiled.readthedocs.io/en/latest/linear_algebra.html#eigenvalues-and-eigenvectors>`_ of a matrix.

System of Linear Equations (SLE)
------------------------------------
A set of :math:`n` linear equations using a common set of :math:`m` variables. For example:

.. math::

  3x_0 + 4x_1 = 5
  
.. math::
  
  -2x_0 + x_1 = 11

In matrix form an SLE can be written as:

.. math::
 
  Ax = b
  
Where :math:`x` is the vector of unknowns to be determined, :math:`A` is a matrix of the coefficients from the left-hand side and the vector :math:`b` contains the numbers from the right-hand side of the equations.

Systems of linear equations can be solved in many ways. Gaussian elimination is one.

Underdetermined and overdetermined systems
_____________________________________________
* If the number of variables exceeds the number of equations the system is **underdetermined**.
* If the number of variables is less than the number of equations the system is **overdetermined**.

Trace
--------
The sum of the elements along the main diagonal of a square matrix.

.. math::

  \text{tr}(A) = \sum_{i=1}^n A_{ii}
  
Satisfies the following properties:

.. math::

  \text{tr}(A) = \text{tr}(A^T)
  
  \text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)
  
  \text{tr}(cA) = c\text{tr}(A)

Transpose
------------

.. math::

  (A^T)_{ij} = A_{ji}

Satisfies the following properties:

.. math::

    (A+B)^T = A^T + B^T

    (AB)^T = B^TA^T

    (A^T)^{-1} = (A^{-1})^T

Types of matrix
----------------

Diagonal matrix
________________
A matrix where :math:`A_{ij} = 0` if :math:`i \neq j`.

Can be written as :math:`\text{diag}(a)` where :math:`a` is a vector of values specifying the diagonal entries.

Diagonal matrices have the following properties:

.. math::

  \text{diag}(a) + \text{diag}(b) = \text{diag}(a + b)
  
  \text{diag}(a) \cdot \text{diag}(b) = \text{diag}(a * b)
  
  \text{diag}(a)^{-1} = \text{diag}(a_1^{-1},...,a_n^{-1})
  
  \text{det}(\text{diag}(a)) = \prod_i{a_i}
  
The eigenvalues of a diagonal matrix are the set of its values on the diagonal.


Hermitian matrix
__________________
The complex equivalent of a symmetric matrix. :math:`A = A^*`, where * represents the conjugate transpose.

Also known as a self-adjoint matrix.

Normal matrix
________________
:math:`A^*A = AA^*` where :math:`A^*` is the conjugate transpose of :math:`A`.

Orthogonal matrix
___________________

.. math:: 

    A^TA = AA^T = I

Positive and negative (semi-)definite
_____________________________
A matrix :math:`A \in \mathbb{R}^{n \times n}` is positive definite if:

.. math::

    z^TAz > 0, \forall z \in \mathbb{R}^n, z \neq 0 

Positive semi-definite matrices are defined analogously, except with :math:`z^TAz \geq 0`

Negative (semi-)definite matrices are the same but with the inequality round the other way.

Singular matrix
_______________
A square matrix which is not invertible. A matrix is singular if and only if the determinant is zero.

Symmetric matrix
____________________
A square matrix :math:`A` where :math:`A = A^T`.

Some properties of symmetric matrices are:

* All the eigenvalues of the matrix are real.

Triangular matrix
___________________
Either a lower triangular or an upper triangular matrix.

Lower triangular matrix
==============================
A square matrix where only the lower triangle is not composed of zeros. Formally:

.. math::

  A_{ij} = 0, \text{if} i < j

Upper triangular matrix
============================
A square matrix where only the upper triangle is not composed of zeros. Formally:

.. math::

  A_{ij} = 0, \text{if} i \geq j

Unitary matrix
___________________
A matrix where its inverse is the same as its complex conjugate. The complex version of an orthogonal matrix.

.. math::

  A^*A = AA^* = I
  
ZCA
------
Like PCA, ZCA converts the data to have zero mean and an identity covariance matrix. Unlike PCA, it does not reduce the dimensionality of the data and tries to create a whitened version that is minimally different from the original.
  
