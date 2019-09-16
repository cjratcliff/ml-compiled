Sequence models
"""""""""""""""

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

  1. # frontier maps partial solutions to scores
  2. initialise frontier to contain the root node with a score of 0
  
  3. while the end of the sequence has not been reached:
  4.    select the candidate from frontier with the best score
      
  5.    # expand the chosen candidate
  6.    add all the children of the candidate to frontier
  7.    compute the scores of all the new nodes in frontier
      
  8.    # prune candidates        
  9.    remove all entries not in the top k from frontier

https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/


Bidirectional RNN
---------------------
Combines the outputs of two RNNs, one processing the input sequence from left to right (forwards in time) and the other from right to left (backwards in time). The two RNNs are stacked on top of each other and their states are typically combined by appending the two vectors. Bidirectional RNNs are often used in Natural Language problems, where we want to take the context from both before and after a word into account before making a prediction.

The basic bidirectional RNN can be defined as follows:

.. math::

  h^f_t = \tanh(W^f_h x_t + U^f_h h^f_{t-1})
  
  h^b_t = \tanh(W^b_h x_{T-t} + U^b_h h^b_{t-1})
  
  h_t = \text{concat}(h^f_t,h^b_t)
  
  o_t = V h_t
  
Where :math:`h^f_t` and :math:`h^b_t` are the hidden states in the forwards and backwards directions respectively. :math:`T` is the length of the sequence. Biases have been omitted for simplicity. :math:`x_t` and :math:`o_t` are the input and output states at time t, respectively.

| **Proposed in**
| `Bidirectional Recurrent Neural Networks, Schuster and Paliwal (1997) <https://ai.intel.com/wp-content/uploads/sites/53/2017/06/BRNN.pdf>`_

Differentiable Neural Computer (DNC)
-------------------------------------------
The memory is an NxW matrix. There are N locations, which can be selectively read and written to.
Read vectors are weighted sums over the memory locations.

The heads use three forms of differentiable attention which:

* Look up content.
* Record transitions between consecutively written locations in an NxN temporal link matrix L.
* Allocate memory for writing.

| **Proposed in**
| `Hybrid computing using a neural network with dynamic external memory, Graves et al. (2016) <https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz>`_

| **Further reading**
| https://deepmind.com/blog/article/differentiable-neural-computers


GRU (Gated Recurrent Unit)
-------------------------------
Variation of the LSTM that is simpler to compute and implement, mergeing the cell and the hidden state.

Comparable performance to LSTMs on a translation task. Has two gates, a reset gate :math:`r` and an update gate :math:`z`. Not reducible from LSTM as there is only one tanh nonlinearity. Cannot ‘count’ as LSTMs can. Partially negates the vanishing gradient problem, as LSTMs do.

The formulation is:

.. math::

    r_t = \sigma(W_r x_t + U_r h_{t-1})
    
    z_t = \sigma(W_z x_t + U_z h_{t-1})

    \tilde h_t = \tanh(Wx + U(h_{t-1}*r))

    h_t = z_t*h_{t-1} + (1-z) * \tilde h_t


Where :math:`*` represents element-wise multiplication and :math:`W_r`, :math:`U_r`, :math:`W_z`, :math:`U_z`, :math:`W` and :math:`U` are parameters to be learnt. Note the lack of bias terms, in contrast with the LSTM.

:math:`z` is used for constructing the new hidden vector and dictates which information is updated from the new output and which is remembered from the old hidden vector.
:math:`r` is used for constructing the output and decides which parts of the hidden vector will be used and which won’t be. The input for the current time-step is always used.

| **Proposed by**
| `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, Cho et al. (2014) <https://www.aclweb.org/anthology/D14-1179>`_

| **Further reading**
| `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling, Chung et al. (2014) <https://arxiv.org/abs/1412.3555>`_

LSTM (Long Short-Term Memory)
--------------------------------
A type of RNN with a memory cell as the hidden state. Uses a gating mechanism to ensure proper propagation of information through many timesteps. Traditional RNNs struggle to train for behaviour requiring long lags due to the exponential loss in error as back propagation proceeds through time (vanishing gradient problem). LSTMs store the error in the memory cell, making long memories possible. However, repeated access to the cell means the issue remains for many problems.

Can have multiple layers. The input gate determines when the input is significant enough to remember. The output gate decides when to output the value. The forget gate determines when the value should be forgotten.

The activations of the input, forget and output gates are :math:`i_t`, :math:`f_t` and :math:`o_t` respectively. The state of the memory cell is :math:`C_t`.

.. math::

    i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)

    f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)

    \tilde C_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)

    C_t = i_t*\tilde C_t + f_t*C_{t-1}

    o_t = \sigma(W_o x_t + U_o h_{t-1} + V_o C_t + b_o)

    h_t = o_t * \tanh(C_t)


Where :math:`*` represents element-wise multiplication.

Each of the input, output and forget gates is surrounded by a sigmoid nonlinearity. This squashes the input so it is between 0 (let nothing through the gate) and 1 (let everything through).

The new cell state is the candidate cell state scaled by the input gate activation, representing how much we want to remember each value and added to the old cell state, scaled by the forget gate activation, how much we want to forget each of those values.

The :math:`\tanh` functions serve to add nonlinearities.

Using an LSTM does not protect from exploding gradients. 

| **Proposed by**
| `Long Short-Term Memory, Hochreiter and Schmidhuber (1997) <https://www.bioinf.jku.at/publications/older/2604.pdf>`_

Forget bias initialization
____________________________________
Helpful to initialize the bias of the forget gate to 1 in order to reduce the scale of forgetting at the start of training. This is done by default in TensorFlow.


Weight tying
_________________
Tie the input and output embeddings. May only be applicable to generative models. Discriminative ones do not have an output embedding.

`Using the Output Embedding to Improve LMs, Press and Wolf (2016) <https://arxiv.org/abs/1608.05859>`_

Cell clipping
__________________
Clip the activations of the memory cells to a range such as [-3,3] or [-50,50]. Helps with convergence problems by preventing exploding gradients and saturation in the sigmoid/tanh nonlinearities.
Deep Recurrent Neural Networks for Acoustic Modelling, Chan and Lane (2015)
LSTM RNN Architectures for Large Scale Acoustic Modeling, Sak et al. (2014)

Peep-hole connections
___________________________
Allows precise timing to be learned, such as the frequency of a signal and other periodic patterns.
Learning Precise Timing with LSTM Recurrent Networks, Ger et al. (2002)
LSTM RNN Architectures for Large Scale Acoustic Modeling, Sak et al. (2014)

Neural Turing Machine (NTM)
------------------------------
Can infer simple algorithms like copying, sorting and associative recall. 

Has two principal components: 

1. A controller, an LSTM. Takes the inputs and emits the outputs for the NTM as a whole.
2. A memory matrix. 

The controller interacts with the memory via a number of read and write heads. Read and write operations are ‘blurry’. A read is a convex combination of ‘locations’ or rows in the memory matrix, according to a weight vector over locations assigned to the read head. Writing uses an erase vector and an add vector. Both content-based and location-based addressing systems are used.

Similarity between vectors is measured by the cosine similarity.

Location-based addressing is designed for both iteration across locations and random-access jumps.

Content addressing
___________________________
Compares a key vector to each location in memory, :math:`M_t(i)` to produce a normalised weighting, :math:`w_t^c(i)`. :math:`t>0` is the key strength, used to amplify or attenuate the focus.

Interpolation
__________________
Blends the weighting produced at the previous time step and the content weighting. An ‘interpolation gate’ is emitted by each head. If :math:`g_t=1` the addressing is entirely content-based. If :math:`g_t=0`, the addressing is entirely location-based.

Convolutional shift
___________________________
Provides a rotation to the weight vector :math:`w_t^g`. All index arithmetic is computed modulo N. The shift weighting :math:`s_t` is a vector emitted by each head and defines a distribution over the allowed integer shifts.

Sharpening
__________________
Combats possible dispersion of weightings over time.

.. math::

  w_t(i) := \frac{w_t(i)^{\gamma_t}}{\sum_j w_t(j)^{\gamma_t}}

| **Proposed in**
| `Neural Turing Machines, Graves et al. (2014) <https://arxiv.org/abs/1410.5401>`_


RNN (Recurrent Neural Network)
----------------------------------
A type of network which processes a sequence and outputs another of the same length. It maintains a hidden state which is updated as new inputs are read in.

.. .. image:: ../img/rnn.PNG
..    :align: center

The most basic type of RNN has the functional form:

.. math::

  h_t = \tanh(W_h x_t + U_h h_{t-1} + b_h)
  
  o_t = V h_t + b_o
  
Where :math:`x_t`, :math:`o_t` and :math:`h_t` are the input, output and hidden states at time t, respectively.


RNN Encoder-Decoder
-------------------------
Common architecture for translation.

Consists of two RNNs. One encodes the input sequence into a fixed-length vector representation, the other decodes it into an output sequence. The original, proposed in `Cho et al. (2014) <https://arxiv.org/pdf/1406.1078.pdf>`_, uses the `GRU <https://ml-compiled.readthedocs.io/en/latest/sequence.html#gru-gated-recurrent-unit>`_ to model sequential information using fewer parameters than the LSTM. Can be augmented with sampled softmax, bucketing and padding.

| **Proposed in**
| `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, Cho et al. (2014) <https://arxiv.org/pdf/1406.1078.pdf>`_


Sequence to sequence	
------------------------
Any machine learning task that takes one sequence and turns it into another.	

Examples include:
 
* Translation	
* Text-to-speech	
* Speech-to-text	
* Part of speech tagging (POS tagging)


Transformer
---------------
Sequence model notable for not using recurrence or convolutions - only attention.

Attained state of the art accuracy on translation tasks `(Vaswani et al., 2017) <https://arxiv.org/abs/1706.03762>`_
and has subsequently been used to get new records on a variety of other tasks (see 'Used in').

Attention layers use `scaled-dot product attention <https://ml-compiled.readthedocs.io/en/latest/layers.html#scaled-dot-product-attention>`_.

Both the encoder and decoder are comprised of multiple blocks each with a multi-ahead attention layer, two fully-connected layers and two layer-normalisation components.

Multi-head attention
________________________
Concatenates the output of multiple parallel attention layers. Each layer has the same inputs (Q, K and V) but different weights. `Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_ use 8 layers in each multi-head attention component but reduce the dimensionality of each from 512 to 64, which keeps the computational cost the same overall.

Positional encoding
_____________________
Positional encodings are added (summed, not concatenated) to the input embeddings to allow the model to be aware of the sequence order.

Self-attention
_________________


Usage in pre-trained language models
______________________________________
`Devlin et al. (2018) <https://arxiv.org/pdf/1810.04805.pdf>`_ pre-train a bidirectional transformer and use this model to attain state of the art accuracy on a variety of natural language tasks. The transformer is first pre-trained to predict masked out tokens and predict next sentences and then fine-tuned on the tasks to be evaluated.

| **Proposed in**
| `Attention is All You Need, Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
|
| **Used in**
| `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al. (2018) <https://arxiv.org/pdf/1810.04805.pdf>`_
| `Improving Language Understanding by Generative Pre-Training, Radford et al. (2018) <https://pdfs.semanticscholar.org/cd18/800a0fe0b668a1cc19f2ec95b5003d0a5035.pdf?_ga=2.41548523.873668896.1568045194-1332261.1564593648>`_
| `Language Models are Unsupervised Multitask Learners, Radford et al. (2019) <https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
|
| **Further reading**
| `The Annotated Transformer, Rush (2018) <http://nlp.seas.harvard.edu/2018/04/03/attention.html>`_


