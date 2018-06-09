"""""""""""""""""""""""""
Adversarial examples
"""""""""""""""""""""""""
Inputs formed by applying small perturbations to examples from the dataset, such that the perturbed input results in wrong classifications with high confidence. They can be created without knowledge of the weights of the classifier. The same adversarial example can be misclassified by many classifiers, trained on different subsets of the dataset and with different architectures.

The direction of perturbation, not the point itself matters most when generating adversarial examples. Adversarial perturbations generalize across different clean examples.

Generating adversarial examples
---------------------------------
Perform gradient descent on the image by taking the derivative of the score for the desired class with respect to the pixels.

Note that this is almost the same technique as was used by Google for understanding convnet predictions but without an additional constraint. They specify that the output image should look like a natural image (eg by having neighbouring pixels be correlated).

Explanations
---------------
Goodfellow et al. (2015) claim that the effectiveness of adversarial examples is down to the linearity of neural networks. While the function created by the network is indeed nonlinear, it is not as nonlinear as often thought. Goodfellow says “...neural nets are piecewise linear, and the linear pieces with non-negligible slope are much bigger than we expected.”

Mitigation techniques
-------------------------

* Regularization - [1] showed that regularization is effective for linear classifiers. It reduces the size of the weights so the image has to be changed more drastically in order to get the same misclassification. However, this comes at a cost in accuracy.
* Adding noise - Somewhat effective but hurts accuracy. [4]
* Blurring - Somewhat effective but hurts accuracy. [4]
* Binarization - Highly effective where it is applicable without hurting accuracy, such as reading text. [6]
* Averaging over multiple crops - 5 can be sufficient to correctly classify the majority of adversarial examples.
* RBF networks (Goodfellow et al., 2015) are resistant to adversarial examples due to their non-linearity. In general using more non-linear models (trained with a better optimization algorithm to make them feasible) may be the best approach.

[5] showed that adversarial examples are still effective, even when perceived through a cellphone camera.

[1] Breaking Linear Classifiers on ImageNet, Karpathy (2015)

[2] Explaining and Harnessing Adversarial Examples, Goodfellow et al. (2015)

[3] Intriguing Properties of Neural Networks, Szegedy et al. (2014)

[4] Towards Deep Neural Network Architectures Robust to Adversarial Examples, Gu et al. (2015)

[5] Adversarial examples in the physical world, Kurakin et al. (2016)

[6] Assessing Threat of Adversarial Examples on Deep Neural Networks, Graese (2016)
