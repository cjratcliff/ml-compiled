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
Search algorithm to find the most likely output sequence.

Motivation
_____________
In a sequence to sequence problem, the next element in the decoded sequence is highly dependent on the previous one. If the output vocabulary is of size :math:`n` and the sequence is of length :math:`m` the complexity of finding the best sequence is :math:`O(n^m)` by brute force. Therefore a good heuristic algorithm is needed.

Greedy search runs in :math:`O(mn)` time but has poor accuracy. Beam search is a compromise between these two extremes.

The algorithm
________________
In the context sequence-to-sequence beam search is a tree search algorithm. The inner nodes are partial solutions and the leaves are full solutions. At each iteration beam search keeps track of the k best partial solutions. The parameter k is known as the beam width.

Pseudocode:

.. code-block:: none

  # frontier maps partial solutions to scores
  initialise frontier to contain the root node with a score of 0
  
  while the end of the sequence has not been reached:
      select the candidate from frontier with the best score
      
      # expand the chosen candidate
      add all the children of the candidate to frontier
      compute the scores of all the new nodes in frontier
      
      # prune candidates        
      remove all entries not in the top k from frontier

https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

RNN Encoder-Decoder
-------------------------
Common architecture for translation.

Consists of two RNNs. One encodes the input sequence into a fixed-length vector representation, the other decodes it into an output sequence. Uses a special type of hidden unit that improves the model’s memory by adaptively remembering and forgetting.
Can be augmented with sampled softmax, bucketing and padding.

`Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, Cho et al. (2014) <https://arxiv.org/pdf/1406.1078.pdf>`_
