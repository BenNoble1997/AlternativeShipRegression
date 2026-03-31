# residual_plotter.py
"""
Residual Plotter
================

Pure plotting functions for residual analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ResidualData:
    """Container for residual analysis data."""
    y_true: np.ndarray
    y_pred: np.ndarray
    dataset_name: str

    @property
    def residuals(self) -> np.ndarray:
        """Calculate residuals."""
        return self.y_true - self.y_pred


class ResidualPlotter:
    """
    Creates residual analysis plots.

    Pure plotting class - takes residual data and creates diagnostic plots.

    Examples
    --------
    >>> plotter = ResidualPlotter()
    >>>
    >>> train_residuals = ResidualData(
    ...     y_true=np.array([100, 200, 300]),
    ...     y_pred=np.array([105, 195, 310]),
    ...     dataset_name='train'
    ... )
    >>>
    >>> plotter.plot_four_way_residuals(
    ...     target_name='Main_Power',
    ...     model_description='polynomial-2',
    ...     train_data=train_residuals,
    ...     # ... other datasets ...
    ... )
    """

    def __init__(self):
        self.colors = {
            'train': '#2ecc71',
            'test': '#f39c12',
            'matched': '#e67e22',
            'alternative': '#e74c3c'
        }

    def plot_four_way_residuals(self,
                                target_name: str,
                                predictor_names: List[str],
                                model_description: str,
                                train_data: ResidualData,
                                test_data: ResidualData,
                                matched_data: ResidualData,
                                alternative_data: ResidualData,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 4×3 grid of residual plots.

        Each row = one dataset
        Columns = residuals vs predicted, histogram, Q-Q plot

        Parameters
        ----------
        target_name : str
            Name of target variable
        predictor_names : list of str
            Predictor names
        model_description : str
            Model type description
        train_data, test_data, matched_data, alternative_data : ResidualData
            Residual data for each dataset
        save_path : str, optional
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))

        datasets = [train_data, test_data, matched_data, alternative_data]

        for row_idx, data in enumerate(datasets):
            color = self.colors.get(data.dataset_name, 'steelblue')

            self._plot_residuals_vs_predicted(axes[row_idx, 0], data, color)
            self._plot_residual_histogram(axes[row_idx, 1], data, color)
            self._plot_qq(axes[row_idx, 2], data, color)

        # Overall title
        predictor_str = ' + '.join(predictor_names)
        fig.suptitle(f'Residual Analysis: {target_name}\n{predictor_str} ({model_description})',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_residuals_vs_predicted(self, ax: plt.Axes, data: ResidualData, color: str):
        """Plot residuals vs predicted values."""
        residuals = data.residuals
        y_pred = data.y_pred

        ax.scatter(y_pred, residuals, alpha=0.5, color=color, s=30,
                   edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Residuals', fontweight='bold')
        ax.set_title(f'{data.dataset_name.capitalize()}: Residuals vs Predicted',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(y_pred) > 10:
            try:
                from scipy.signal import savgol_filter
                sorted_idx = np.argsort(y_pred)
                window_length = min(11, len(residuals) // 2 * 2 - 1)
                if window_length >= 3:
                    smoothed = savgol_filter(residuals[sorted_idx], window_length, 2)
                    ax.plot(y_pred[sorted_idx], smoothed, 'r-',
                            linewidth=2, label='Trend', alpha=0.7)
                    ax.legend(fontsize=8)
            except:
                pass

    def _plot_residual_histogram(self, ax: plt.Axes, data: ResidualData, color: str):
        """Plot histogram of residuals with normal overlay."""
        residuals = data.residuals

        ax.hist(residuals, bins=20, color=color, alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{data.dataset_name.capitalize()}: Distribution',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Normal curve overlay
        mu, std = residuals.mean(), residuals.std()
        if std > 0:
            x = np.linspace(residuals.min(), residuals.max(), 100)
            y = stats.norm.pdf(x, mu, std) * len(residuals) * \
                (residuals.max() - residuals.min()) / 20
            ax.plot(x, y, 'r-', linewidth=2, label='Normal', alpha=0.7)
            ax.legend(fontsize=8)

    def _plot_qq(self, ax: plt.Axes, data: ResidualData, color: str):
        """Plot Q-Q plot with statistics."""
        residuals = data.residuals

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(color)
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(6)
        ax.set_title(f'{data.dataset_name.capitalize()}: Q-Q Plot',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        mu, std = residuals.mean(), residuals.std()

        stats_text = (f'RMSE: {rmse:.2f}\n'
                      f'MAE: {mae:.2f}\n'
                      f'Mean: {mu:.2f}\n'
                      f'Std: {std:.2f}')
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)