"""
Shared formatting utilities for display classes.

Provides consistent formatting across all display classes for:
- Metrics (R², RRMSE, MAPE, etc.)
- Model names
- CV descriptions
- Tables and headers
"""

import numpy as np
from typing import Any, Optional


class DisplayFormatters:
    """Shared formatting utilities for consistent display output."""

    # ========================================================================
    # METRIC FORMATTING
    # ========================================================================

    @staticmethod
    def format_metric(value: float,
                      metric_type: str,
                      decimals: Optional[int] = None) -> str:
        """
        Format a metric value consistently.

        Parameters
        ----------
        value : float
            Metric value to format
        metric_type : str
            Type of metric: 'r2', 'adj_r2', 'rrmse', 'mape', 'rmse', 'mae', 'aic', 'bic'
        decimals : int, optional
            Override default decimal places

        Returns
        -------
        str
            Formatted metric string

        Examples
        --------
        >>> format_metric(0.92456, 'r2')
        '0.9246'
        >>> format_metric(12.456, 'rrmse')
        '12.46'
        >>> format_metric(np.nan, 'r2')
        'N/A'
        """
        # Handle NaN/Inf
        if np.isnan(value) or np.isinf(value):
            return 'N/A'

        # Default decimal places by metric type
        if decimals is None:
            decimals_map = {
                'r2': 4,
                'adj_r2': 4,
                'rrmse': 2,
                'mape': 2,
                'rmse': 2,
                'mae': 2,
                'aic': 1,
                'bic': 1,
                'vif': 2,
            }
            decimals = decimals_map.get(metric_type, 3)

        return f"{value:.{decimals}f}"

    @staticmethod
    def format_metric_with_std(value: float,
                               std: float,
                               metric_type: str) -> str:
        """
        Format metric with standard deviation.

        Parameters
        ----------
        value : float
            Mean metric value
        std : float
            Standard deviation
        metric_type : str
            Type of metric

        Returns
        -------
        str
            Formatted string: "value ± std"

        Examples
        --------
        >>> format_metric_with_std(0.92, 0.02, 'r2')
        '0.9200 ± 0.02'
        """
        if np.isnan(value) or np.isnan(std):
            return DisplayFormatters.format_metric(value, metric_type)

        val_str = DisplayFormatters.format_metric(value, metric_type)
        std_str = DisplayFormatters.format_metric(std, metric_type, decimals=2)

        return f"{val_str} ± {std_str}"

    # ========================================================================
    # MODEL NAME FORMATTING
    # ========================================================================

    @staticmethod
    def format_model_name(model_name: str, max_length: int = 18) -> str:
        """
        Format model name for display.

        Parameters
        ----------
        model_name : str
            Model type name
        max_length : int
            Maximum length (truncate if longer)

        Returns
        -------
        str
            Formatted model name

        Examples
        --------
        >>> format_model_name('polynomial-2')
        'polynomial-2'
        >>> format_model_name('very-long-model-name-here', max_length=10)
        'very-lo...'
        """
        if len(model_name) <= max_length:
            return model_name
        return model_name[:max_length - 3] + "..."

    @staticmethod
    def format_predictor_string(predictors, max_length: int = 35) -> str:
        """
        Format predictor combination string.

        Parameters
        ----------
        predictors : tuple or list
            Predictor names
        max_length : int
            Maximum length (truncate if longer)

        Returns
        -------
        str
            Formatted predictor string

        Examples
        --------
        >>> format_predictor_string(('DWT', 'LOA (M)'))
        'DWT + LOA (M)'
        >>> format_predictor_string(('DWT', 'LOA (M)', 'BEAM (M)', 'DRAFT (M)'), max_length=20)
        'DWT + LOA (M) + ...'
        """
        pred_str = " + ".join(predictors)

        if len(pred_str) <= max_length:
            return pred_str

        return pred_str[:max_length - 4] + " ..."

    # ========================================================================
    # CV DESCRIPTION FORMATTING
    # ========================================================================

    @staticmethod
    def format_cv_description(cv_method: Optional[str],
                              n_folds: int = 10) -> str:
        """
        Format cross-validation method description.

        Parameters
        ----------
        cv_method : str or None
            CV method: 'standard', 'stratified', 'grouped', or None
        n_folds : int
            Number of folds

        Returns
        -------
        str
            Formatted CV description

        Examples
        --------
        >>> format_cv_description('stratified', 10)
        ' (10-fold stratified CV)'
        >>> format_cv_description(None)
        ' (no CV)'
        """
        if cv_method is None:
            return " (no CV)"
        elif cv_method == 'standard':
            return f" ({n_folds}-fold CV)"
        elif cv_method == 'stratified':
            return f" ({n_folds}-fold stratified CV)"
        elif cv_method == 'grouped':
            return f" ({n_folds}-fold grouped CV)"
        else:
            return f" ({cv_method})"

    # ========================================================================
    # TABLE FORMATTING
    # ========================================================================

    @staticmethod
    def format_header(title: str, width: int = 140, char: str = '=') -> str:
        """
        Format a section header.

        Parameters
        ----------
        title : str
            Header title
        width : int
            Total width
        char : str
            Character for border

        Returns
        -------
        str
            Formatted header with borders

        Examples
        --------
        >>> print(format_header("RESULTS", width=40))
        ========================================
        RESULTS
        ========================================
        """
        border = char * width
        return f"\n{border}\n{title}\n{border}"

    @staticmethod
    def format_subheader(title: str, width: int = 140) -> str:
        """Format a subsection header."""
        return f"\n{title}\n{'-' * width}"

    @staticmethod
    def format_separator(width: int = 140, char: str = '-') -> str:
        """Format a separator line."""
        return char * width

    # ========================================================================
    # SPECIAL VALUE FORMATTING
    # ========================================================================

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """
        Format percentage value.

        Examples
        --------
        >>> format_percentage(0.25)
        '25.0%'
        >>> format_percentage(12.456, decimals=2)
        '12.46%'
        """
        if np.isnan(value):
            return 'N/A'
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_gap(value: float, metric_type: str = 'r2') -> str:
        """
        Format gap value with sign.

        Examples
        --------
        >>> format_gap(0.05, 'r2')
        '+0.0500'
        >>> format_gap(-2.3, 'rrmse')
        '-2.30'
        """
        if np.isnan(value):
            return 'N/A'

        decimals = 4 if metric_type == 'r2' else 2
        return f"{value:+.{decimals}f}"

    @staticmethod
    def format_sample_count(n: int) -> str:
        """
        Format sample count.

        Examples
        --------
        >>> format_sample_count(1234)
        '1,234'
        """
        return f"{n:,}"

    # ========================================================================
    # EFFECT CLASSIFICATION
    # ========================================================================

    @staticmethod
    def classify_effect(gap_r2: float,
                        gap_rrmse: float,
                        threshold_r2: float = 0.05,
                        threshold_rrmse: float = 1.5) -> tuple[str, str]:
        """
        Classify effect magnitude and return icon.

        Parameters
        ----------
        gap_r2 : float
            R² gap
        gap_rrmse : float
            RRMSE gap
        threshold_r2 : float
            R² significance threshold
        threshold_rrmse : float
            RRMSE significance threshold

        Returns
        -------
        effect : str
            Classification: 'MINIMAL', 'MODERATE', 'MAJOR'
        icon : str
            Visual indicator: '✅', '⚠️', '🚨'

        Examples
        --------
        >>> classify_effect(0.02, 5.0)
        ('MINIMAL', '✅')
        >>> classify_effect(0.18, 25.0)
        ('MAJOR', '🚨')
        """
        r2_significant = abs(gap_r2) >= threshold_r2
        rrmse_significant = abs(gap_rrmse) >= threshold_rrmse

        if not r2_significant and not rrmse_significant:
            return "MINIMAL", "✅"
        elif r2_significant and rrmse_significant:
            if abs(gap_r2) > 0.15 or abs(gap_rrmse) > 2:
                return "MAJOR", "🚨"
            else:
                return "MODERATE", "⚠️"
        else:
            return "MODERATE", "⚠️"

    # ========================================================================
    # COLOR FORMATTING (for terminal output)
    # ========================================================================

    @staticmethod
    def colorize(text: str, color: str) -> str:
        """
        Add ANSI color codes (optional, for terminal output).

        Parameters
        ----------
        text : str
            Text to colorize
        color : str
            Color name: 'red', 'green', 'yellow', 'blue', 'cyan', 'magenta'

        Returns
        -------
        str
            Colorized text (if terminal supports it)
        """
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'magenta': '\033[95m',
            'reset': '\033[0m'
        }

        if color in colors:
            return f"{colors[color]}{text}{colors['reset']}"
        return text

