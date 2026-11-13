import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionPerformanceMetrics:
    """
    Regression performance metrics calculator.

    This class provides methods for evaluating the predictive performance
    of regression models. It focuses on error-based and goodness-of-fit
    metrics that quantify how well predicted values match observed values.

    The class is intended to be used internally by the
    ``RegressionMetrics`` facade but can also be used independently
    for performance evaluation.

    :param y_true: Observed (true) target values
    :type y_true: pd.Series

    :param y_pred: Predicted target values
    :type y_pred: pd.Series

    :ivar y_true: Observed target values
    :vartype y_true: pd.Series

    :ivar y_pred: Predicted target values
    :vartype y_pred: pd.Series

    :ivar residuals: Difference between observed and predicted values
    :vartype residuals: pd.Series

    :example:

        >>> perf = RegressionPerformanceMetrics(y_true, y_pred)
        >>> perf.calculate_nrmse()
        >>> perf.calculate_mape()
        >>> perf.calculate_adj_r2(n_features=4)
    """

    def __init__(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ):
        """
        Initialise RegressionPerformanceMetrics.

        :param y_true: Observed target values
        :type y_true: pd.Series

        :param y_pred: Predicted target values
        :type y_pred: pd.Series
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = y_true - y_pred

    def calculate_nrmse(
        self,
        normalisation_variable: float | None = None,
        normalisation_type: str | None = "mean",
    ) -> float:
        """
        Calculate Normalised Root Mean Squared Error (NRMSE).

        NRMSE expresses the RMSE relative to the scale of the target
        variable and returns the result in percentage.

        Three normalisation options are supported:

        - Custom normalisation variable
        - Mean of target values
        - Min-max range of target values

        :param normalisation_variable: Custom value used for normalisation
        :type normalisation_variable: float or None

        :param normalisation_type: Normalisation method ("mean" or "min-max")
        :type normalisation_type: str or None

        :return: Normalised RMSE in percentage
        :rtype: float

        :note:
            If ``normalisation_variable`` is provided,
            it overrides ``normalisation_type``.

        :note:
            Returns NaN if target values are empty or mean equals zero.
        """

        if len(self.y_true) == 0 or np.mean(self.y_true) == 0:
            return np.nan

        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))

        if normalisation_variable is not None:
            return (rmse / normalisation_variable) * 100

        elif normalisation_type == "min-max":
            return (rmse / (np.max(self.y_true) - np.min(self.y_true))) * 100

        else:
            return (rmse / np.mean(self.y_true)) * 100

    def calculate_nmae(
        self,
        normalisation_variable: float | None = None,
        normalisation_type: str | None = "mean",
    ) -> float:
        """
        Calculate Normalised Root Mean Squared Error (NRMSE).

        NRMSE expresses the RMSE relative to the scale of the target
        variable and returns the result in percentage.

        Three normalisation options are supported:

        - Custom normalisation variable
        - Mean of target values
        - Min-max range of target values

        :param normalisation_variable: Custom value used for normalisation
        :type normalisation_variable: float or None

        :param normalisation_type: Normalisation method ("mean" or "min-max")
        :type normalisation_type: str or None

        :return: Normalised RMSE in percentage
        :rtype: float

        :note:
            If ``normalisation_variable`` is provided,
            it overrides ``normalisation_type``.

        :note:
            Returns NaN if target values are empty or mean equals zero.
        """

        if len(self.y_true) == 0 or np.mean(self.y_true) == 0:
            return np.nan

        mae = mean_absolute_error(self.y_true, self.y_pred)

        if normalisation_variable is not None:
            return (mae / normalisation_variable) * 100

        elif normalisation_type == "min-max":
            return (mae / (np.max(self.y_true) - np.min(self.y_true))) * 100

        else:
            return (mae / np.mean(self.y_true)) * 100


    def calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        MAPE measures the average percentage difference between
        observed and predicted values.

        :return: Mean Absolute Percentage Error in percentage
        :rtype: float

        :note:
            Observations with zero true values are excluded
            to avoid division by zero.

        :note:
            Returns NaN if all target values are zero.
        """

        mask = self.y_true != 0

        if not np.any(mask):
            return np.nan

        return (
            np.mean(
                np.abs(
                    (self.y_true[mask] - self.y_pred[mask]) /
                    self.y_true[mask]
                )
            ) * 100
        )

    def calculate_rss(self) -> float:
        """
        Calculate Residual Sum of Squares (RSS).

        RSS measures the total squared error between observed
        and predicted values.

        :return: Residual sum of squares
        :rtype: float
        """

        return np.sum((self.y_true - self.y_pred) ** 2)

    def calculate_r2(self)->float:

        """
        Calculate the r2 score of the model

        Placed herefor the purposes of using a unified library

        """
        return r2_score(self.y_true, self.y_pred)


    def calculate_adj_r2(self, n_features: int) -> float:
        """
        Calculate Adjusted R-squared.

        Adjusted R² corrects R² for the number of predictors in the model
        and penalises unnecessary complexity.

        Formula:

            Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)

        where:

        - n = number of observations
        - k = number of predictors

        :param n_features: Number of predictors in the regression model
        :type n_features: int

        :return: Adjusted R-squared value
        :rtype: float

        :note:
            If the number of observations is too small
            (n <= n_features + 1), standard R² is returned.
        """

        n = len(self.y_true)
        r2 = r2_score(self.y_true, self.y_pred)

        if n <= n_features + 1:
            return r2

        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def calculate_predictive_r2(self,
                                reference_variance: float | None = None) -> float:
        """
        Calculate predictive R-squared (Q-squared) from out-of-fold predictions.

        Defined as ``1 - PRESS / SS_tot``, where PRESS is the sum of squared
        out-of-fold residuals and SS_tot the total sum of squares of the
        observed values. Unlike adjusted R-squared this carries no complexity
        penalty: the held-out predictions already exclude the in-sample
        optimism that the penalty is designed to remove, so applying one
        would correct for the same effect twice.

        Intended to be called once per cross-validation repeat on the pooled
        out-of-fold vector, never on an individual fold. Fold-local scoring
        recomputes SS_tot on a handful of observations, so the result reflects
        which samples happened to be grouped together as much as model quality.

        :param reference_variance: Optional fixed variance used in place of
            the sample variance of ``y_true`` when forming SS_tot. Supplying
            a cohort-independent value puts Q-squared from differently sized
            or differently spread cohorts on a common scale, mirroring the
            role of the fixed divisor in :meth:`calculate_nrmse`.
        :type reference_variance: float or None

        :return: Predictive R-squared. Unbounded below; negative values mean
            the model predicts held-out data worse than the cohort mean does.
        :rtype: float

        :note: Returns NaN when SS_tot is zero, which occurs if every observed
            value is identical.
        """
        y_true = np.asarray(self.y_true, dtype=float)
        y_pred = np.asarray(self.y_pred, dtype=float)

        press = np.sum((y_true - y_pred) ** 2)

        if reference_variance is not None:
            ss_tot = reference_variance * len(y_true)
        else:
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return np.nan

        return 1.0 - press / ss_tot