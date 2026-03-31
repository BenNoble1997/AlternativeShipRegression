import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox


class RegressionAssumptionMetrics:
    """
    A class for evaluating regression model assumptions.

    This class provides statistical tests and diagnostics used to
    validate regression assumptions such as:

    - multicollinearity
    - heteroscedasticity
    - autocorrelation
    - residual independence

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

    def calculate_vif(
            self,
            X: pd.DataFrame,
            feature_names: list[str] | tuple,
            vif_threshold: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate the Variance Inflation Factor (VIF) for each predictor.

        VIF measures multicollinearity between predictors.
        Higher VIF values indicate stronger correlation between features.

        Rule of thumb:

        - VIF < 5 → acceptable
        - VIF 5–10 → moderate multicollinearity
        - VIF > 10 → severe multicollinearity

        :param X: Design matrix containing predictors
        :type X: pd.DataFrame
        :param feature_names: Names of predictor variables
        :type feature_names: list[str] or tuple
        :param vif_threshold: Threshold for problematic VIF values
        :type vif_threshold: int

        :return: DataFrame with feature names and VIF values
        :rtype: pd.DataFrame
        """

        if isinstance(feature_names, tuple):
            feature_names = list(feature_names)

        if X.shape[1] < 2:
            # VIF requires at least 2 features
            return pd.DataFrame({'feature': X.columns, 'vif': [1.0] * X.shape[1]})

        # Handle missing values
        X_clean = X.dropna()

        if len(X_clean) < X.shape[1] + 1:
            # Not enough samples for VIF calculation
            return pd.DataFrame({'feature': X.columns, 'vif': [np.nan] * X.shape[1]})

        # Calculate VIF for each feature
        vif_data = []
        for i, col in enumerate(X_clean.columns):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_data.append({'feature': col, 'vif': vif})
            except Exception as e:
                vif_data.append({'feature': col, 'vif': np.nan})

        vif_df = pd.DataFrame(vif_data)

        # Check for problematic VIF values
        max_vif = vif_df['vif'].max()
        problematic_features = vif_df[vif_df['vif'] > vif_threshold]

        is_valid = max_vif <= vif_threshold

        vif_info = {
            'max_vif': max_vif,
            'vif_df': vif_df,
            'problematic_features': problematic_features.to_dict('records') if not problematic_features.empty else [],
            'predictors': feature_names
        }

        return pd.DataFrame(vif_data)

    def calculate_spearman_heteroscedasticity(
            self,
            X: np.ndarray,
            feature_names: list[str] | tuple,
    ) -> dict:
        """
        Detect heteroscedasticity using Spearman correlation.

        This test evaluates whether residual magnitude correlates with
        predicted values or predictors.

        Interpretation:

        - |rho| < 0.10 → negligible
        - |rho| < 0.30 → weak
        - |rho| < 0.50 → moderate
        - |rho| ≥ 0.50 → strong

        :param X: Predictor matrix
        :type X: np.ndarray
        :param feature_names: Names of predictors
        :type feature_names: list[str] or tuple

        :return: Dictionary containing overall and per-feature heteroscedasticity results
        :rtype: dict
        """

        if isinstance(feature_names, tuple):
            feature_names = list(feature_names)

        try:
            # Overall: predictions vs. absolute residuals
            rho_pred, p_pred = spearmanr(self.y_pred, np.abs(self.residuals))

            # ✅ Ensure we always return heteroscedastic key
            results = {
                'overall': {
                    'rho': float(rho_pred),
                    'p_value': float(p_pred),
                    'heteroscedastic': abs(rho_pred) > 0.3,  # ✅ CRITICAL: Add this key
                    'interpretation': self._interpret_spearman(rho_pred)
                }
            }
        except Exception as e:
            # Fallback if Spearman fails
            results = {
                'overall': {
                    'rho': 0.0,
                    'p_value': 1.0,
                    'heteroscedastic': False,  # ✅ CRITICAL: Fallback must have this key too
                    'interpretation': 'test failed'
                }
            }

        # Per predictor: check if specific predictor drives heteroscedasticity
        per_predictor = []

        try:
            # Handle case where X is 1D (single predictor)
            if len(X.shape) == 1 or X.shape[1] == 1:
                X_2d = X.reshape(-1, 1) if len(X.shape) == 1 else X
                rho, p = spearmanr(X_2d[:, 0], np.abs(self.residuals))
                per_predictor.append({
                    'feature': feature_names[0],
                    'rho': float(rho),
                    'p_value': float(p),
                    'heteroscedastic': abs(rho) > 0.3,
                    'interpretation': self._interpret_spearman(rho)
                })
            else:
                # Multiple predictors
                for i, name in enumerate(feature_names):
                    rho, p = spearmanr(X[:, i], np.abs(self.residuals))
                    per_predictor.append({
                        'feature': name,
                        'rho': float(rho),
                        'p_value': float(p),
                        'heteroscedastic': abs(rho) > 0.3,
                        'interpretation': self._interpret_spearman(rho)
                    })
        except Exception as e:
            # If per-predictor fails, use empty list
            per_predictor = []

        results['per_predictor'] = pd.DataFrame(per_predictor)
        results['by_feature'] = {row['feature']: row for row in per_predictor}

        return results

    def _interpret_spearman(self, rho):
        """Interpret Spearman's rho magnitude."""
        abs_rho = abs(rho)

        if abs_rho < 0.10:
            severity = "negligible"
        elif abs_rho < 0.30:
            severity = "weak"
        elif abs_rho < 0.50:
            severity = "moderate"
        else:
            severity = "strong"

        direction = "increasing" if rho > 0 else "decreasing"

        return f"{severity} {direction} variance"

    def breusch_pagan_test(self, X, feature_names=None) -> dict:
        """
        Perform the Breusch-Pagan test for heteroscedasticity.

        Tests whether residual variance depends linearly on predictors.

        Hypotheses:

        - H0: Homoscedasticity (constant variance)
        - H1: Heteroscedasticity present

        :param X: Predictor matrix
        :type X: np.ndarray
        :param feature_names: Optional predictor names
        :type feature_names: list[str] or None

        :return: Dictionary containing test statistic, p-value, and interpretation
        :rtype: dict
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        residuals = self.y_true - self.y_pred
        n = len(residuals)

        # Squared residuals (our dependent variable for auxiliary regression)
        residuals_sq = residuals ** 2

        # Auxiliary regression: residuals² ~ X

        aux_model = LinearRegression()
        aux_model.fit(X, residuals_sq)
        residuals_sq_pred = aux_model.predict(X)

        # Calculate BP statistic
        # BP = n * R² from auxiliary regression
        ss_total = np.sum((residuals_sq - residuals_sq.mean()) ** 2)
        ss_residual = np.sum((residuals_sq - residuals_sq_pred) ** 2)
        r_squared_aux = 1 - (ss_residual / ss_total)

        bp_statistic = n * r_squared_aux
        df = X.shape[1]  # Number of predictors
        p_value = 1 - stats.chi2.cdf(bp_statistic, df)

        # Test each feature individually
        by_feature = {}
        if feature_names is not None and len(feature_names) == X.shape[1]:
            for i, name in enumerate(feature_names):
                X_single = X[:, i].reshape(-1, 1)
                aux_model_single = LinearRegression()
                aux_model_single.fit(X_single, residuals_sq)
                pred_single = aux_model_single.predict(X_single)

                ss_total_single = np.sum((residuals_sq - residuals_sq.mean()) ** 2)
                ss_res_single = np.sum((residuals_sq - pred_single) ** 2)
                r2_single = 1 - (ss_res_single / ss_total_single)

                bp_stat_single = n * r2_single
                p_val_single = 1 - stats.chi2.cdf(bp_stat_single, 1)

                by_feature[name] = {
                    'statistic': bp_stat_single,
                    'p_value': p_val_single,
                    'heteroscedastic': p_val_single < 0.05
                }

        return {
            'test': 'Breusch-Pagan',
            'statistic': bp_statistic,
            'p_value': p_value,
            'df': df,
            'heteroscedastic': p_value < 0.05,
            'interpretation': 'heteroscedastic' if p_value < 0.05 else 'homoscedastic',
            'by_feature': by_feature
        }

    def white_test(self, X, feature_names=None) -> dict:
        """
        Perform White's test for heteroscedasticity.

        White's test detects both linear and non-linear heteroscedasticity
        by including squared and interaction terms in an auxiliary regression.

        :param X: Predictor matrix
        :type X: np.ndarray
        :param feature_names: Optional predictor names
        :type feature_names: list[str] or None

        :return: Dictionary with test statistic and interpretation
        :rtype: dict
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        residuals = self.y_true - self.y_pred
        n = len(residuals)
        residuals_sq = residuals ** 2

        # Create polynomial features (includes squares and interactions)
        # For ship data: includes DWT², LOA², DWT·LOA, etc.
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Auxiliary regression with all polynomial terms
        aux_model = LinearRegression()
        aux_model.fit(X_poly, residuals_sq)
        residuals_sq_pred = aux_model.predict(X_poly)

        # Calculate White statistic
        ss_total = np.sum((residuals_sq - residuals_sq.mean()) ** 2)
        ss_residual = np.sum((residuals_sq - residuals_sq_pred) ** 2)
        r_squared_aux = 1 - (ss_residual / ss_total)

        white_statistic = n * r_squared_aux
        df = X_poly.shape[1]  # Number of polynomial features
        p_value = 1 - stats.chi2.cdf(white_statistic, df)

        return {
            'test': 'White',
            'statistic': white_statistic,
            'p_value': p_value,
            'df': df,
            'n_aux_features': df,
            'heteroscedastic': p_value < 0.05,
            'interpretation': 'heteroscedastic' if p_value < 0.05 else 'homoscedastic',
            'note': 'Detects both linear and non-linear heteroscedasticity'
        }

    def durbin_watson_test(self) -> dict:
        """
        Perform Durbin-Watson test for residual autocorrelation.

        Interpretation:

        - DW ≈ 2 → no autocorrelation
        - DW < 2 → positive autocorrelation
        - DW > 2 → negative autocorrelation

        :return: Dictionary with statistic and interpretation
        :rtype: dict
        """
        residuals = self.y_true - self.y_pred

        # Handle NaN values
        residuals = residuals.dropna() if hasattr(residuals, 'dropna') else residuals[~np.isnan(residuals)]

        if len(residuals) < 3:
            return {
                'test': 'Durbin-Watson',
                'statistic': np.nan,
                'interpretation': 'insufficient data',
                'warning': 'Need at least 3 observations'
            }

        dw_statistic = durbin_watson(residuals)

        # More conservative thresholds (closer to 2 is better)
        if 1.8 <= dw_statistic <= 2.2:
            interpretation = 'independent (no significant autocorrelation)'
        elif dw_statistic < 1.8:
            interpretation = 'positive autocorrelation detected'
        else:
            interpretation = 'negative autocorrelation detected'

        return {
            'test': 'Durbin-Watson',
            'statistic': dw_statistic,
            'interpretation': interpretation,
            'note': 'Values near 2.0 indicate independence'
        }

    def ljungbox_test(self, lags=None, alpha: float = 0.05) -> dict:
        """
        Perform Ljung-Box test for residual autocorrelation.

        Hypotheses:

        - H0: Residuals are independently distributed
        - H1: Residuals are autocorrelated

        :param lags: Number of lags to test
        :type lags: int or None
        :param alpha: Significance level
        :type alpha: float

        :return: Dictionary containing test statistic and p-value
        :rtype: dict
        """
        residuals = self.y_true - self.y_pred

        # Handle NaN values
        residuals = residuals.dropna() if hasattr(residuals, 'dropna') else residuals[~np.isnan(residuals)]

        if len(residuals) < 3:
            return {
                'test': 'Ljung-Box',
                'statistic': np.nan,
                'p_value': np.nan,
                'interpretation': 'insufficient data',
                'warning': 'Need at least 3 observations'
            }

        # Auto-select lags if not provided
        if lags is None:
            lags = min(10, len(residuals) // 5)
            if lags < 1:
                lags = 1

        lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)

        lb_stat = lb_test['lb_stat'].iloc[0]
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]

        independent = lb_pvalue > alpha

        return {
            'test': 'Ljung-Box',
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'lags_tested': lags,
            'interpretation': 'independent' if independent else 'autocorrelation detected',
            'note': f'H0: no autocorrelation (p>{alpha} supports independence)'
        }

