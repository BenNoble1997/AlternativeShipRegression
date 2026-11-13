import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from knowledge_layer.Metrics.regression_assumption_checks import RegressionAssumptionMetrics
from knowledge_layer.Metrics.regression_performance_metrics import RegressionPerformanceMetrics
from knowledge_layer.Metrics.regression_selection_metrics import RegressionSelectionMetrics
from config.loggermixin.logger_config import LoggerMixin

class RegressionMetrics(LoggerMixin):
    """
    Central interface for regression model evaluation.

    This class provides a unified framework for calculating regression
    performance metrics, model selection criteria, and regression
    assumption diagnostics.

    The class follows a composition-based design and internally uses:

    - RegressionPerformanceMetrics
    - RegressionSelectionMetrics
    - RegressionAssumptionMetrics

    to organise regression evaluation into clear functional categories.

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

    :ivar performance: Performance metrics calculator
    :vartype performance: RegressionPerformanceMetrics

    :ivar selection: Model selection metrics calculator
    :vartype selection: RegressionSelectionMetrics

    :ivar assumption: Regression diagnostic metrics calculator
    :vartype assumption: RegressionAssumptionMetrics

    :example:

        >>> metrics = RegressionMetrics(y_true, y_pred)
        >>> metrics.calculate_rrmse()
        >>> metrics.calculate_aic(n_features=4)
        >>> metrics.calculate_vif(X, feature_names)
    """

    def __init__(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        reference:float|None=None,
        verbose:bool=False,
        debug:bool=False):
        """
        Initialise RegressionMetrics object.

        :param y_true: Observed target values
        :type y_true: pd.Series

        :param y_pred: Predicted target values
        :type y_pred: pd.Series
        """

        self.performance = RegressionPerformanceMetrics(y_true, y_pred)
        self.selection = RegressionSelectionMetrics(y_true, y_pred)
        self.assumption = RegressionAssumptionMetrics(y_true, y_pred)

        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = (y_true - y_pred)
        self.reference = reference
        self.verbose = verbose
        self.set_log_level(verbose=self.verbose, debug=debug)

    def calculate_rrmse(self, y_mean: float | None = None) -> float:
        """
        Calculate Relative Root Mean Squared Error (RRMSE).

        RRMSE is the RMSE normalised by the mean of the target variable
        and expressed as a percentage.

        :param y_mean: Optional mean value used for normalisation
        :type y_mean: float or None

        :return: Relative RMSE in percentage
        :rtype: float
        """

        if self.reference is not None:
            if self.verbose:
                self.logger.info(f"Using global reference value: {self.reference}")
            return self.performance.calculate_nrmse(normalisation_variable=self.reference)
        else:
            return self.performance.calculate_nrmse(normalisation_variable=y_mean)

    def calculate_nmae(self, y_mean: float | None = None) -> float:
        """
        Calculate normalised mean absolute error

        NMAE is the mean absolute error normalised by a specified normalisation variable
        and expressed as a percentage.

        :param y_mean: Optional mean value used for normalisation
        :type y_mean: float or None

        :return: Relative MAE in percentage
        :rtype: float
        """

        if self.reference is not None:
            if self.verbose:
                self.logger.info(f"Using global reference value: {self.reference}")
            return self.performance.calculate_nmae(normalisation_variable=self.reference)
        else:
            return self.performance.calculate_nmae(normalisation_variable=y_mean)

    def calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        :return: Mean Absolute Percentage Error in percentage
        :rtype: float
        """
        return self.performance.calculate_mape()

    def calculate_rss(self) -> float:
        """
        Calculate Residual Sum of Squares (RSS).

        :return: Residual sum of squares
        :rtype: float
        """
        return np.sum((self.y_true - self.y_pred) ** 2)

    def calculate_aic(self, n_features: int) -> float:
        """
        Calculate Akaike Information Criterion (AIC).

        AIC balances model fit and model complexity.

        Formula:

            AIC = n * ln(RSS / n) + 2k

        where:

        - n = number of observations
        - RSS = residual sum of squares
        - k = number of model parameters

        :param n_features: Number of predictors in the model
        :type n_features: int

        :return: AIC value
        :rtype: float

        :note:
            Lower AIC indicates better model.
        """
        return self.selection.calculate_aic(n_features)

    def calculate_bic(self, n_features: int) -> float:
        """
        Calculate Bayesian Information Criterion (BIC).

        Formula:

            BIC = n * ln(RSS / n) + k * ln(n)

        :param n_features: Number of predictors in the model
        :type n_features: int

        :return: BIC value
        :rtype: float

        :note:
            BIC penalises model complexity more strongly than AIC.
        """
        return self.selection.calculate_bic(n_features)

    def log_likelihood(self) -> float:
        """
        Calculate log-likelihood of the regression model.

        :return: Log-likelihood value
        :rtype: float
        """
        return self.selection.log_likelihood()

    def calculate_predictive_metrics(self, n_features: int) -> dict:
        """
        Calculate predictive performance metrics.

        Intended for use in cross-validation and model evaluation.

        :param n_features: Number of predictors in the model
        :type n_features: int

        :return: Dictionary with predictive metrics
        :rtype: dict

        :returns dict with keys:

            - r2
            - adj_r2
            - rmse
            - rrmse
            - mae
            - mape
            - n_samples
            - n_features
        """

        n = len(self.y_true)

        return {
            "r2": r2_score(self.y_true, self.y_pred),
            "adj_r2": self.performance.calculate_adj_r2(n_features=n_features),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "rrmse": self.calculate_rrmse(),
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "mape": self.performance.calculate_mape(),
            "n_samples": n,
            "n_features": n_features,
        }

    def calculate_cv_metrics(self, reference_variance: float | None = None, reference_mean:float=None) -> dict:
        """
        Metrics appropriate for out-of-fold predictions.

        Excludes adjusted R-squared, AIC, BIC and log-likelihood: all four are
        in-sample selection criteria whose derivations assume residuals from
        the data the model was fitted on. Use
        :meth:`calculate_information_criteria` for model selection on a full
        cohort instead.

        :returns dict with keys: q2, rmse, rrmse, mae, nmae, mape, n_samples
        """
        return {
            "q2": self.performance.calculate_predictive_r2(reference_variance),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "rrmse": self.calculate_rrmse(y_mean=reference_mean),
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "nmae": self.calculate_nmae(),
            "mape": self.performance.calculate_mape(),
            "n_samples": len(self.y_true),
        }

    def calculate_information_criteria(self, n_features: int) -> dict:
        """
        Calculate model selection criteria.

        :param n_features: Number of predictors
        :type n_features: int

        :return: Dictionary containing AIC, BIC, and log-likelihood
        :rtype: dict
        """
        return {
            "aic": self.selection.calculate_aic(n_features),
            "bic": self.selection.calculate_bic(n_features),
            "log_likelihood": self.selection.log_likelihood(),
        }

    def calculate_all_metrics(self, n_features: int) -> dict:
        """
        Calculate full regression evaluation metrics.

        :param n_features: Number of predictors in the model
        :type n_features: int

        :return: Dictionary containing performance and selection metrics
        :rtype: dict
        """

        return {
            "r2": r2_score(self.y_true, self.y_pred),
            "adj_r2": self.performance.calculate_adj_r2(n_features=n_features),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "rrmse": self.calculate_rrmse(),
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "mape": self.performance.calculate_mape(),
            "aic": self.selection.calculate_aic(n_features),
            "bic": self.selection.calculate_bic(n_features),
            "log_likelihood": self.selection.log_likelihood(),
            'n_samples':len(self.y_true),
        }

    def calculate_vif(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | tuple,
        vif_threshold: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF).

        :param X: Design matrix
        :type X: pd.DataFrame

        :param feature_names: Predictor names
        :type feature_names: list[str] or tuple

        :param vif_threshold: Threshold for multicollinearity
        :type vif_threshold: int

        :return: DataFrame with VIF values
        :rtype: pd.DataFrame
        """
        return self.assumption.calculate_vif(
            X=X,
            feature_names=feature_names,
            vif_threshold=vif_threshold,
        )

    def combined_heteroscedasticity_test(
        self,
        X: np.ndarray,
        feature_names: list[str] | tuple,
    ) -> dict:

        """
        Combined heteroscedasticity test
        :param X:
        :param feature_names:
        :return:
        """

        results = {}

        # results['spearman'] = self.calculate_spearman_heteroscedasticity(X=X, feature_names=feature_names)['overall']['heteroscedastic']
        # results['breusch_pagan'] = self.breusch_pagan_test(X=X, feature_names=feature_names)['heteroscedastic']
        # results['white'] = self.white_test(X=X, feature_names=feature_names)['heteroscedastic']

        results['spearman'] = self.calculate_spearman_heteroscedasticity(X=X, feature_names=feature_names)
        results['breusch_pagan'] = self.breusch_pagan_test(X=X, feature_names=feature_names)
        results['white'] = self.white_test(X=X, feature_names=feature_names)

        results["conclusion"] = sum(results['spearman']['overall']['heteroscedastic'], results['spearman']['heteroscedastic'], results['white']['heteroscedastic'])

        return results


    def calculate_spearman_heteroscedasticity(
        self,
        X: np.ndarray,
        feature_names: list[str] | tuple,
    ) -> dict:
        """
        Test heteroscedasticity using Spearman correlation.

        :param X: Predictor matrix
        :type X: np.ndarray

        :param feature_names: Predictor names
        :type feature_names: list[str] or tuple

        :return: Spearman test results
        :rtype: dict
        """
        return self.assumption.calculate_spearman_heteroscedasticity(
            X=X,
            feature_names=feature_names,
        )

    def breusch_pagan_test(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> dict:
        """
        Perform Breusch-Pagan heteroscedasticity test.

        :param X: Predictor matrix
        :type X: pd.DataFrame

        :param feature_names: Predictor names
        :type feature_names: list[str] or None

        :return: Test results
        :rtype: dict
        """
        return self.assumption.breusch_pagan_test(X=X, feature_names=feature_names)

    def white_test(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> dict:
        """
        Perform White heteroscedasticity test.

        :param X: Predictor matrix
        :type X: pd.DataFrame

        :param feature_names: Predictor names
        :type feature_names: list[str] or None

        :return: Test results
        :rtype: dict
        """
        return self.assumption.white_test(X=X, feature_names=feature_names)

    def durbin_watson_test(self) -> dict:
        """
        Perform Durbin-Watson autocorrelation test.

        :return: Test results
        :rtype: dict
        """
        return self.assumption.durbin_watson_test()

    def ljungbox_test(
        self,
        lags: int | None = None,
        alpha: float = 0.05,
    ) -> dict:
        """
        Perform Ljung-Box autocorrelation test.

        :param lags: Number of lags
        :type lags: int or None

        :param alpha: Significance level
        :type alpha: float

        :return: Test results
        :rtype: dict
        """
        return self.assumption.ljungbox_test(lags=lags, alpha=alpha)