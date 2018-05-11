""""""""""""""""""""""""""""""""""""""""""
Natural language processing (NLP)
""""""""""""""""""""""""""""""""""""""""""

Entity linking
----------------
The task of finding the specific entity which words or phrases refer to. Not to be confused with Named Entity Recognition.

FastText
----------
A simple baseline method for text classification.

`Bag of Tricks for Efficient Text Classification <https://arxiv.org/abs/1607.01759>`_

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

Not to be confused with entity linking which finds the specific entity (eg the city of London) rather than only the type (place).

Part of speech tagging (POS tagging)
------------------------------------------
Labelling words with ADV, ADJ, PREP etc. Correct labelling is dependent on context - ‘bananas’ can be a noun or an adjective.

Phoneme
---------
A unit of sound in a language, shorter than a syllabel. English has 44 phonemes. For example, the long 'a' sound in 'train' and 'sleigh' and the 't' sound in 'bottle' and 'sit'.

Stemming
----------
Reducing a word to its basic form. This often involves removing suffixes like 'ed', 'ing' or 's'.
