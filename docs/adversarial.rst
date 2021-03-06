"""""""""""""""""""""""""
Adversarial examples
"""""""""""""""""""""""""
Examples that are specially created so that image classification algorithms predict the wrong class with high confidence even though the image remains easy for humans to classify correctly. Only small perturbations in the pixel-values are necessary to create adversarial examples.

They can be created without knowledge of the weights of the classifier. The same adversarial example can be misclassified by many classifiers, trained on different subsets of the dataset and with different architectures.

The direction of perturbation, not the point itself matters most when generating adversarial examples. Adversarial perturbations generalize across different clean examples.

`Kurakin et al. (2016) <https://arxiv.org/abs/1607.02533>`_ showed that adversarial examples are still effective, even when perceived through a cellphone camera.

Generating adversarial examples
---------------------------------
Perform gradient descent on the image by taking the derivative of the score for the desired class with respect to the pixels.

Note that this is almost the same technique as was used by Google for understanding convnet predictions but without an additional constraint. They specify that the output image should look like a natural image (eg by having neighbouring pixels be correlated).

Explanations
---------------
Adversarial examples are made possible when the input has a large number of dimensions. This means many individually small effects can have a very large effect on the overall prediction.

`Goodfellow et al. (2015) <https://arxiv.org/abs/1412.6572>`_ suggest that the effectiveness of adversarial examples is down to the linearity of neural networks. While the function created by the network is indeed nonlinear, it is not as nonlinear as often thought. Goodfellow says “...neural nets are piecewise linear, and the linear pieces with non-negligible slope are much bigger than we expected.”

Mitigation techniques
-------------------------

* Regularization - `Karpathy (2015) <http://karpathy.github.io/2015/03/30/breaking-convnets/>`_ showed that regularization is effective for linear classifiers. It reduces the size of the weights so the image has to be changed more drastically in order to get the same misclassification. However, this comes at a cost in accuracy.
* Adding noise - Somewhat effective but hurts accuracy, `Gu et al. (2014) <https://arxiv.org/abs/1412.5068>`_
* Blurring - Somewhat effective but hurts accuracy, `Gu et al. (2014) <https://arxiv.org/abs/1412.5068>`_
* Binarization - Highly effective where it is applicable without hurting accuracy, such as reading text, `Graese et al. (2016) <https://arxiv.org/abs/1610.04256>`_
* Averaging over multiple crops - Can be sufficient to correctly classify the majority of adversarial examples.
* RBF networks `(Goodfellow et al. (2015)) <https://arxiv.org/abs/1412.6572>`_ are resistant to adversarial examples due to their non-linearity. In general using more non-linear models (trained with a better optimization algorithm to make them feasible) may be the best approach.

Papers
---------
| `Breaking Linear Classifiers on ImageNet, Karpathy (2015) <http://karpathy.github.io/2015/03/30/breaking-convnets/>`_
| `Explaining and Harnessing Adversarial Examples, Goodfellow et al. (2015) <https://arxiv.org/abs/1412.6572>`_
| `Intriguing Properties of Neural Networks, Szegedy et al. (2013) <https://arxiv.org/abs/1312.6199>`_
| `Towards Deep Neural Network Architectures Robust to Adversarial Examples, Gu et al. (2014) <https://arxiv.org/abs/1412.5068>`_
| `Adversarial examples in the physical world, Kurakin et al. (2016) <https://arxiv.org/abs/1607.02533>`_
| `Assessing Threat of Adversarial Examples on Deep Neural Networks, Graese et al. (2016) <https://arxiv.org/abs/1610.04256>`_
