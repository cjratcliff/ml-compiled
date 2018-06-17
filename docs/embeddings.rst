''''''''''''
Embeddings
''''''''''''

""""""""""""""""""""""""""""""""""""
Distributed representation
""""""""""""""""""""""""""""""""""""
A representation in which each number in a vector is used to store information about some attribute of an object. For example, brightness or size.

Contrast this with one-hot representations. A one-hot vector of length n can store n states, whereas a distributed representation of the same length can store an exponential number of states.

""""""""""""""""""""""""
One-hot representation
""""""""""""""""""""""""
A vector which has zeros everywhere except for in the indices representing the class or classes which are present.

""""""""""""""""""""""""
Triplet network
""""""""""""""""""""""""
Architecture for learning embeddings for calculating similarities. Useful for tasks like `face verification <http://ml-compiled.readthedocs.io/en/latest/computer_vision.html#face-verification>`_.

During each batch in training, an 'anchor' example is supplied along with a positive that is similar to it and a negative that is not.

Uses the `triplet loss <http://ml-compiled.readthedocs.io/en/latest/loss_functions.html#triplet-loss>`_.

"""""""""""""
Word vectors
"""""""""""""
The meaning of a word is represented by a vector of fixed size.

Polysemous words (words with multiple meanings) cannot be modeled effectively by a single point.

CBOW (Continuous Bag of Words)
-----------------------------------
Used to create word embeddings. Predicts a word given its context. The context is the surrounding n words, as in the skip-gram model. Referred to as a bag of words model as the order of words within the window does not affect the embedding. Mikolov et al. (2013) use a window size of 4 on either side.

Several times faster to train than the skip-gram model and has slightly better accuracy for words which occur frequently.

`Efficient Estimation of Word Representations in Vector Space, Mikolov et al. (2013) <https://arxiv.org/abs/1301.3781>`_

GloVe
------
Method for learning word vectors.

https://nlp.stanford.edu/projects/glove/

`GloVe: Global Vectors for Word Representation, Pennington et al. (2014) <https://www.aclweb.org/anthology/D14-1162>`_

Skip-gram
-----------
Used to create word embeddings. Predicts the context given a word. For example, let the window size be 2. Then the relevant window is :math:`\{w_{i-2}, w_{i-1},w_i,w_{i+1},w_{i+2}\}`. The model picks a random word :math:`w_k \in \{w_{i-2},w_{i-1},w_{i+1},w_{i+2}\}` and trains a model that predicts :math:`w_k` given :math:`w_i`.

Increasing the window size improves the quality of the word vectors but also makes them more expensive to compute. Samples less from words that are far away from the known word, since the influence will be weaker. Works well with a small amount of data and can represent even rare words or phrases well.

The efficiency and quality of the skip-gram model is improved by two additions:

1. Subsampling frequent words. Words like ‘the’ and ‘is’ occur very frequently in most text corpora yet contain little useful semantic information about surrounding words. To reduce this inefficiency words are sampled according to :math:`P(w_i)=1-t/f_i` where :math:`f_i` is the frequency of word i and t is a manually set threshold, usually around 10-5.

2. Negative sampling, a simplification of noise-contrastive estimation.

With some minor changes, skip-grams can also be used to calculate embeddings for phrases such as ‘North Sea’. However, this can increase the size of the vocabulary dramatically.

`Efficient Estimation of Word Representations in Vector Space, Mikolov et al. (2013) <https://arxiv.org/abs/1301.3781>`_

Word2vec
---------
The name of the implementation of the CBOW and skip-gram architectures in Mikolov et al. (2013)

https://code.google.com/archive/p/word2vec/

`Efficient Estimation of Word Representations in Vector Space, Mikolov et al. (2013) <https://arxiv.org/abs/1301.3781>`_


