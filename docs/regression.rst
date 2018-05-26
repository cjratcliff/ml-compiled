""""""""""""""
Regression
""""""""""""""

Confidence intervals
-----------------------


Isotonic regression
---------------------

Linear regression
---------------------
The simplest form of regression. Estimates a model with the equation:

.. math::

  \hat{y} = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n
  
where the :math:`\beta_i` are parameters to be estimated by the model and the :math:`x_i` are the features. 

Logistic regression
----------------------
Used for modelling probabilities. It uses the sigmoid function (:math:`\sigma`) to ensure the predicted values are between 0 and 1. Values outside of this range would not make sense when predicting a probability. The functional form is:

.. math::

  \hat{y} = \sigma{\beta_0 + \beta_1 x_1 + ... + \beta_n x_n}

P-values
----------
Measure the statistical significance of the coefficients of a regression.
