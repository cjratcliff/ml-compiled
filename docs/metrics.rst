""""""""""""
Metrics
""""""""""""

BLEU
------
Score for assessing translation tasks. Also used for image captioning. Stands for BiLingual Evaluation Understudy.

Ranges from 0 to 1, where 1 corresponds to being identical to the reference translation.
Often uses multiple reference translations.

BLEU: a Method for Automatic Evaluation of Machine Translation, Papineni et al. (2002)

F1-score
----------

Intersection over Union (IoU)
------------------------------
An accuracy score for two bounding boxes, where one is the prediction and the other is the target. It is equal to the area of their intersection divided by the area of their union.

Mean Average Precision
------------------------
The main evaluation metric for object detection.
To calculate it first define the overlap criterion. This could be that the IoU for two bounding boxes be greater than 0.5. Since the ground truth is always that the class is present, this means each predicted box is either a true-positive or a false-positive. This means the precision can be calculated using TP/(TP+FN).

Precision
------------

Recall
--------
