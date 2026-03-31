import numpy as np
import pandas as pd

class RegressionSelectionMetrics:
    """
    A class for calculating regression model selection metrics.

    This class provides statistical metrics used to compare competing
    regression models by balancing goodness-of-fit and model complexity.
    Typical use cases include feature selection, model comparison,
    and likelihood-based evaluation.

    :param y_true: Observed (actual) target values
    :type y_true: pd.Series
    :param y_pred: Predicted target values from the regression model
    :type y_pred: pd.Series
    """

    def __init__(self,
                 y_true: pd.Series,
                 y_pred: pd.Series,
                 ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = (y_true - y_pred)

    def calculate_rss(self) -> float:
        """
        Calculate the residual sum of squares (RSS).

        RSS measures the total squared difference between observed
        and predicted values and is used in AIC, BIC, and likelihood
        calculations.

        :return: Residual sum of squares
        :rtype: float
        """
        return np.sum((self.y_true - self.y_pred) ** 2)

    def calculate_aic(self, n_features: int) -> float:
        """
        Calculate the Akaike Information Criterion (AIC).

        AIC evaluates model quality by balancing model fit and complexity.
        Lower AIC values indicate a better model.

        Formula:

            AIC = n * ln(RSS / n) + 2k

        where:

        - n = number of observations
        - RSS = residual sum of squares
        - k = number of parameters (features + intercept)

        :param n_features: Number of predictors in the regression model
        :type n_features: int

        :return: AIC value
        :rtype: float
        """
        n = len(self.y_true)
        rss = self.calculate_rss()

        if rss == 0:
            return -np.inf

        k = n_features + 1
        return n * np.log(rss / n) + 2 * k

    def calculate_bic(self, n_features: int) -> float:
        """
        Calculate the Bayesian Information Criterion (BIC).

        BIC penalizes model complexity more strongly than AIC,
        making it suitable for selecting simpler models.

        Formula:

            BIC = n * ln(RSS / n) + k * ln(n)

        where:

        - n = number of observations
        - RSS = residual sum of squares
        - k = number of parameters (features + intercept)

        :param n_features: Number of predictors in the regression model
        :type n_features: int

        :return: BIC value
        :rtype: float
        """
        n = len(self.y_true)
        rss = self.calculate_rss()

        if rss == 0:
            return -np.inf

        k = n_features + 1
        return n * np.log(rss / n) + k * np.log(n)

    def log_likelihood(self) -> float:
        """
        Calculate the log-likelihood of the regression model.

        Assumes normally distributed residuals and is commonly used
        for likelihood ratio tests and information criteria.

        Formula:

            logL = -n/2 * (ln(2π) + ln(RSS/n) + 1)

        :return: Log-likelihood value
        :rtype: float
        """
        n = len(self.y_true)
        rss = self.calculate_rss()

        if rss <= 0:
            return -np.inf

        return -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)