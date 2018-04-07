""""""""""""""""""""""""""
Sequence to sequence
""""""""""""""""""""""""""
Any machine learning task that takes one sequence and turns it into another.

Examples include:

* Translation
* Text-to-speech
* Speech-to-text
* Part of speech tagging (POS tagging)

Beam search
-------------
The beam width, k, is a hyperparameter. The k best solutions so far are kept in memory.

Pseudocode:

.. code-block:: none

  initialise candidates to be an empty dictionary
  for i = 1,...,seq_length:
    for each candidate in candidates:
      for each class in classes:
        compute the score
  remove all entries not in the top k from candidates

RNN Encoder-Decoder
-------------------------
Common architecture for translation.

Consists of two RNNs. One encodes the input sequence into a fixed-length vector representation, the other decodes it into an output sequence. Uses a special type of hidden unit that improves the model’s memory by adaptively remembering and forgetting.
Can be augmented with embeddings, sampled softmax, bucketing and padding.

`Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, Cho et al. (2014) <https://arxiv.org/pdf/1406.1078.pdf>`_
