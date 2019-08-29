""""""""""""""""""""""""""""""""""""""""""
Natural language processing (NLP)
""""""""""""""""""""""""""""""""""""""""""

Datasets
-----------

Labelled
____________

* `bAbI <https://research.fb.com/downloads/babi/>`_ - Dataset for question answering
* `GLUE <https://gluebenchmark.com/>`_
* IMDB - Dataset of movie reviews, used for sentiment classification. Each review is labelled as either positive or negative.
* `RACE <https://www.cs.cmu.edu/~glai1/data/race/>`_ - Reading comprehension dataset.
* `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`_ - Stanford Question Answering Dataset
* `SuperGLUE <https://super.gluebenchmark.com/>`_ - Harder successor to the GLUE dataset.
* TIMIT - Speech corpus.
* WMT - https://nlp.stanford.edu/projects/nmt/

Unlabelled
________________
A list of some of the most frequently used unlabelled datasets and text corpora, suitable for tasks like generative text modelling and learning word embeddings.

* `1 Billion Word Language Model Benchmark <http://www.statmt.org/lm-benchmark/>`_
* `Common Crawl <http://commoncrawl.org/the-data/>`_
* `Gigaword 5 <https://catalog.ldc.upenn.edu/LDC2011T07>`_
* PTB - Stands for 'Penn Treebank'
* `Project Gutenberg <http://www.gutenberg.org/>`_
* `Shakespeare <https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt>`_


Entity linking
----------------
The task of finding the specific entity which words or phrases refer to. Not to be confused with Named Entity Recognition.

FastText
----------
A simple baseline method for text classification.

The architecture is as follows:

* The inputs are n-grams features from the original input sequence. Using n-grams means some of the word-order information is preserved without the large increase in computational complexity characteristic of recurrent networks.
* An embedding layer.
* A mean-pooling layer averages the features over the length of the inputs.
* A softmax layer gives the class probabilities.

The model is trained with the `cross-entropy loss <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#cross-entropy-loss>`_ as normal.

| **Proposed in** 
| `Bag of Tricks for Efficient Text Classification <https://arxiv.org/abs/1607.01759>`_


`Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_


Latent Dirichlet Allocation (LDA)
-----------------------------------
Topic modelling algorithm.

Each item/document is a finite mixture over the set of topics.
Each topic is a distribution over words.
The parameters can be estimated with expectation maximisation.
Unlike a simple clustering approach, LDA allows a document to be associated with multiple topics.

`Latent Dirichlet Allocation, Blei et al. (2003) <http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf>`_

Morpheme
----------
A word or a part of a word that conveys meaning on its own. For example, 'ing', 'un', 'dog' or 'cat'.

Named Entity Recognition (NER)
---------------------------------
Labelling words and word sequences with the type of entity they represent, such as person, place or time. 

Not to be confused with `entity linking <https://ml-compiled.readthedocs.io/en/latest/natural_language_processing.html#entity-linking>`_ which finds the specific entity (eg the city of London) rather than only the type (place).

Part of speech tagging (POS tagging)
------------------------------------------
Labelling words with ADV, ADJ, PREP etc. Correct labelling is dependent on context - ‘bananas’ can be a noun or an adjective.

Phoneme
---------
A unit of sound in a language, shorter than a syllabel. English has 44 phonemes. For example, the long 'a' sound in 'train' and 'sleigh' and the 't' sound in 'bottle' and 'sit'.

Stemming
----------
Reducing a word to its basic form. This often involves removing suffixes like 'ed', 'ing' or 's'.

