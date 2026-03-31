# prediction_scatter_plotter.py
"""
Prediction Scatter Plotter
===========================

Pure plotting functions for prediction vs actual scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionData:
    """Container for prediction data from one dataset."""
    y_true: np.ndarray
    y_pred: np.ndarray
    r2: float
    rrmse: float
    n_samples: int
    dataset_name: str


class PredictionScatterPlotter:
    """
    Creates scatter plots comparing predictions to actual values.

    Pure plotting class - just takes prediction data and plots it.

    Examples
    --------
    >>> plotter = PredictionScatterPlotter()
    >>>
    >>> train_data = PredictionData(
    ...     y_true=np.array([100, 200, 300]),
    ...     y_pred=np.array([105, 195, 310]),
    ...     r2=0.95,
    ...     rrmse=5.2,
    ...     n_samples=3,
    ...     dataset_name='train'
    ... )
    >>>
    >>> plotter.plot_four_way_scatter(
    ...     target_name='Main_Power',
    ...     model_description='polynomial-2',
    ...     train_data=train_data,
    ...     test_data=test_data,
    ...     matched_data=matched_data,
    ...     alternative_data=alternative_data
    ... )
    """

    def __init__(self):
        self.colors = {
            'train': '#2ecc71',
            'test': '#f39c12',
            'matched': '#e67e22',
            'alternative': '#e74c3c'
        }

    def plot_four_way_scatter(self,
                              target_name: str,
                              predictor_names: List[str],
                              model_description: str,
                              train_data: PredictionData,
                              test_data: PredictionData,
                              matched_data: PredictionData,
                              alternative_data: PredictionData,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2x2 grid of scatter plots for four datasets.

        Parameters
        ----------
        target_name : str
            Name of target variable
        predictor_names : list of str
            Predictor variable names
        model_description : str
            Model type description
        train_data, test_data, matched_data, alternative_data : PredictionData
            Prediction data for each dataset
        save_path : str, optional
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        datasets = [train_data, test_data, matched_data, alternative_data]

        for ax, data in zip(axes, datasets):
            self._plot_single_scatter(ax, data, target_name)

        # Overall title
        predictor_str = ' + '.join(predictor_names)
        fig.suptitle(f'{target_name}: {predictor_str} ({model_description})',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_single_scatter(self, ax: plt.Axes, data: PredictionData, target_name: str):
        """Plot scatter for a single dataset."""
        color = self.colors.get(data.dataset_name, 'steelblue')

        # Scatter plot
        ax.scatter(data.y_true, data.y_pred, alpha=0.6, s=50,
                   color=color, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(data.y_true.min(), data.y_pred.min())
        max_val = max(data.y_true.max(), data.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='Perfect prediction', alpha=0.7)

        # Labels and title
        ax.set_xlabel(f'Actual {target_name}', fontweight='bold')
        ax.set_ylabel(f'Predicted {target_name}', fontweight='bold')
        ax.set_title(f'{data.dataset_name.capitalize()}\n'
                     f'R²={data.r2:.3f}, RRMSE={data.rrmse:.1f}%',
                     fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

        # Sample count
        ax.text(0.98, 0.02, f'n={data.n_samples}',
                transform=ax.transAxes,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_aspect('equal', adjustable='datalim')