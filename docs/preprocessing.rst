""""""""""""""""""
Preprocessing
""""""""""""""""""

Principal Component Analysis (PCA)
----------------------------------------
Approximates a dataset with a set of smaller linearly uncorrelated variables. These variables can be found through eigenvalue decomposition.

% TODO: Formula

Whitening
------------
Converts the data to have zero mean and an identity covariance matrix. 

Types of whitening include PCA and ZCA.

ZCA
-----
Like PCA, ZCA converts the data to have zero mean and an identity covariance matrix. Unlike PCA, it does not reduce the dimensionality of the data and tries to create a whitened version that is minimally different from the original.
