""""""""""""""""""""""""
Evaluation metrics
""""""""""""""""""""""""

Classification
-----------------

AUC (Area Under the Curve)
____________________________
Summarises the `ROC curve <https://ml-compiled.readthedocs.io/en/latest/metrics.html#roc-curve>`_ with a single number, equal to the integral of the curve.

Sometimes referred to as AUROC (Area Under the Receiver Operating Characteristics).

F1-score
__________
The F1-score is the harmonic mean of the precision and the recall.

Using the harmonic mean has the effect that a good F1-score requires both a good precision and a good recall.

.. math:: 

  F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}

Precision
______________
The probability that an example is in fact a positive, given that it was classified as one.

.. math::

  \text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

Where TP is the number of true positives and FP is the number of false positives.

Recall
______________
The probability of classifying an example as a positive given that it is infact a positive.

.. math::

  \text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  
Where TP is the number of true positives and FN is the number of false negatives.

ROC curve
______________
Plots the true positive rate against the false positive rate for different values of the threshold in a binary classifier.

ROC stands for Receiver Operating Characteristic.

The ROC curve can be described with one number using the `ROC <https://ml-compiled.readthedocs.io/en/latest/metrics.html#auc-area-under-the-curve>`_.


Language modelling
---------------------

Bits per character (BPC)
__________________________
Used for assessing character-level language models.

Identical to the cross-entropy loss, but uses base 2 for the logarithm.

Perplexity
___________
Used to measure how well a probabilistic model predicts a sample. It is equivalent to the exponential of the cross-entropy loss.


Object detection
-------------------

Intersection over Union (IoU)
________________________________
An accuracy score for two bounding boxes, where one is the prediction and the other is the target. It is equal to the area of their intersection divided by the area of their union.

Mean Average Precision
__________________________
The main evaluation metric for object detection.

To calculate it first define the overlap criterion. This could be that the IoU for two bounding boxes be greater than 0.5. Since the ground truth is always that the class is present, this means each predicted box is either a true-positive or a false-positive. This means the precision can be calculated using TP/(TP+FN).


Ranking
----------

Cumulative Gain
_________________
A simple metric for ranking that does not take position into account.

.. math::

  CG_p = \sum_{i=1}^p r_i
  
Where :math:`r_i` is the relevance of document :math:`i`.

Discounted Cumulative Gain (DCG)
_____________________________________
Used for ranking. Takes the position of the documents in the ranking into account.

.. math::

  DCG_p = \sum_{i=1}^p \frac{r_i}{\log_2{(i+1)}}

Where :math:`r_i` is the relevance of the document in position :math:`i`.

Mean Reciprocal Rank (MRR)
____________________________

.. math::

  MRR = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{rank(q)}
  
Where :math:`q \in Q` is a query taken from a set of queries and :math:`rank(q)` is the rank of the first document that is relevant for query :math:`q`. 

Normalized Discounted Cumulative Gain (NDCG)
______________________________________________
Used for ranking. Normalizes the DCG by dividing by the score that would be achieved by a perfect ranking. NDCG is always between 0 and 1.

.. math::

  NDCG_p = \frac{DCG_p}{IDCG_p}

Where

.. math::

  DCG_p = \sum_{i=1}^p \frac{r_i}{\log_2{(i+1)}}
  
and IDCG is the Ideal Discounted Cumulative Gain, the DCG that would be produced by a perfect ranking:

.. math::

  IDCG_p = \sum_{i=1}^p \frac{2^{r_i} - 1}{\log_2{(i+1)}}
  
Precision @ k
________________

The proportion of documents returned in the top k results which are relevant. ie the number of relevant documents divided by k.
  
Regression
-------------

RMSE
_______
Root Mean Squared Error.

.. math::

  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}

R-squared
____________
A common metric for evaluating regression algorithms that is easier to interpret than the RMSE but only valid for linear models.

Intuitively, it is the proportion of the variance in the y variable that has been explained by the model. As long as the model contains an intercept term the R-squared should be between 0 and 1.

.. math::

  R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
  
where :math:`\bar{y} = \sum_{i=1}^n y_i`, the mean of y.

Translation
---------------

BLEU
_______
Score for assessing translation tasks. Also used for image captioning. Stands for BiLingual Evaluation Understudy.

Ranges from 0 to 1, where 1 corresponds to being identical to the reference translation.
Often uses multiple reference translations.

`BLEU: a Method for Automatic Evaluation of Machine Translation, Papineni et al. (2002) <https://www.aclweb.org/anthology/P02-1040.pdf>`_


