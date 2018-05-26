""""""""""""""
Regression
""""""""""""""

Confidence intervals
-----------------------
TODO

Isotonic regression
---------------------
Fits a step-wise monotonic function to the data. A useful way to avoid overfitting if there is a strong theoretical reason to believe that the function :math:`y = f(x)` is monotonic. For example, the relationship between the floor area of houses and their prices.

Linear regression
---------------------
The simplest form of regression. Estimates a model with the equation:

.. math::

  \hat{y} = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n
  
where the :math:`\beta_i` are parameters to be estimated by the model and the :math:`x_i` are the features. 

The loss function is usually the `squared error <https://ml-compiled.readthedocs.io/en/latest/loss_functions.html#squared-loss>`_.

Logistic regression
----------------------
Used for modelling probabilities. It uses the sigmoid function (:math:`\sigma`) to ensure the predicted values are between 0 and 1. Values outside of this range would not make sense when predicting a probability. The functional form is:

.. math::

  \hat{y} = \sigma(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)
  
Multicollinearity
-------------------
When one of the features is a linear function of one or more of the others. 

P-values
----------
Measure the statistical significance of the coefficients of a regression. The closer the p-value is to 0, the more statistically significant that result is.

The p-value is the probability of seeing an effect greater than or equal to the one observed if there is in fact no relationship.

In a regression the formula for calculating the p-value of a coefficient is:

TODO