""""""""""""""""""""""""""""""""""""""""""
Natural language processing (NLP)
""""""""""""""""""""""""""""""""""""""""""

Datasets
-----------

Labelled
__________

`WMT <https://nlp.stanford.edu/projects/nmt/>`_
''''''''''''''''''
Parallel corpora for translation. Aligned on the sentence level. 

Notable results in BLEU (higher is better):

English-to-German (2014)

* 28.4 - `Attention is All You Need, Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
* 24.7 - `Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, Wu et al. (2016)<https://arxiv.org/abs/1609.08144>`_
* 23.8 - `Neural Machine Translation in Linear Time, Kalchbrenner et al. (2016) <https://arxiv.org/pdf/1610.10099.pdf>`_

English-to-French (2014)

* 41.8 - `Attention is All You Need, Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
* 39.0 - `Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, Wu et al. (2016)<https://arxiv.org/abs/1609.08144>`_

Other datasets
''''''''''''''''

* `bAbI <https://research.fb.com/downloads/babi/>`_ - Dataset for question answering
* `GLUE <https://gluebenchmark.com/>`_ - Stands for General Language Understanding Evaluation. Assesses performance across 11 different tasks including sentiment analysis, question answering and entailment, more details of which can be found on `their website <https://gluebenchmark.com/tasks>`_. Leaderboard `here <https://gluebenchmark.com/leaderboard>`_.
* IMDB - Dataset of movie reviews, used for sentiment classification. Each review is labelled as either positive or negative.
* `RACE <https://www.cs.cmu.edu/~glai1/data/race/>`_ - Reading comprehension dataset. Leaderboard `here <http://www.qizhexie.com/data/RACE_leaderboard.html>`_.
* `RACE: Large-scale ReAding Comprehension Dataset From Examinations, Lai et al. (2017) <https://arxiv.org/pdf/1704.04683.pdf>`_
* `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`_ - Stanford Question Answering Dataset
* `SuperGLUE <https://super.gluebenchmark.com/>`_ - Harder successor to the GLUE dataset. Assesses performance across 10 different tasks (more details `here <https://super.gluebenchmark.com/tasks>`_). Leaderboard `here <https://super.gluebenchmark.com/leaderboard>`_.
* TIMIT - Speech corpus

Unlabelled
________________
A list of some of the most frequently used unlabelled datasets and text corpora, suitable for tasks like language modelling and learning word embeddings.

PTB
''''''''
Stands for 'Penn Treebank'. Notable results, given in word-level `perplexity <https://ml-compiled.readthedocs.io/en/latest/metrics.html#perplexity>`_ (lower is better):

* 35.8 - `Language Models are Unsupervised Multitask Learners, Radford et al. (2019) <https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
* 47.7 - `Breaking the Softmax Bottleneck: A High-Rank RNN Language Model, Yang et al. (2017) <https://arxiv.org/abs/1711.03953v4>`_
* 55.8 - `Efficient Neural Architecture Search via Parameter Sharing, Pham et al. (2018) <https://arxiv.org/abs/1802.03268>`_
* 62.4 - `Neural Architecture Search with Reinforcement Learning, Zoph and Le (2016) <https://arxiv.org/pdf/1611.01578v2.pdf>`_
* 68.7 - `Recurrent Neural Network Regularization, Zaremba et al. (2014) <https://arxiv.org/pdf/1409.2329v1.pdf>`_

Other datasets
''''''''''''''''
* `1 Billion Word Language Model Benchmark <http://www.statmt.org/lm-benchmark/>`_
* `Common Crawl <http://commoncrawl.org/the-data/>`_
* `Gigaword 5 <https://catalog.ldc.upenn.edu/LDC2011T07>`_
* `Project Gutenberg <http://www.gutenberg.org/>`_
* WikiText-2
* `Shakespeare <https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt>`_

Entailment
------------
The task of deciding whether one piece of text follows logically from another. 

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

Polysemy
-----------
The existence of multiple meanings for a word.

Stemming
----------
Reducing a word to its basic form. This often involves removing suffixes like 'ed', 'ing' or 's'.

