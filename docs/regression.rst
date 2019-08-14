""""""""""""""
Regression
""""""""""""""

Confidence intervals
-----------------------
The confidence interval for a point estimate measures is the interval within which we have a particular degree of confidence the true value resides. For example, the 95% confidence interval for the mean height in a population may be [1.78m, 1.85m].

Confidence intervals can be calculated in this way:

1. Let :math:`\alpha` be the specified confidence level. eg :math:`\alpha = 0.95` for the 95% confidence level.
2. Let :math:`f(x; n-1)` be the pdf for Student's t distribution, parameterised by the number of degrees of freedom which is the sample size (n) minus 1.
3. Calculate :math:`t = f(1 - \alpha/2; n-1)`
4. Then the confidence interval for the point estimate is:

.. math::

  \bar{x} - t \frac{s}{\sqrt{n}} \leq x \leq \bar{x} + t \frac{s}{\sqrt{n}}
  
Where :math:`\bar{x}` is the estimated value of the statistic, :math:`x` is the true value and :math:`s` is the sample standard deviation.

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

Normal equation
___________________
The equation that gives the optimal parameters for a linear regression.

Rewrite the regression equation as:

.. math::

  \hat{y} = X \beta
  
Then the formula for :math:`beta` which minimizes the squared error is:

.. math::

  \beta = (X^T X)^{-1} X^T y

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
