# domain_shift_plotter.py
"""
Domain Shift Plotter
====================

Pure plotting functions for domain shift visualization.
No dependencies on analyzer classes - just takes data and plots it.

Philosophy: "Tell me what to plot, I don't care how you got the data"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetMetrics:
    """Container for metrics from one dataset."""
    r2: float
    rrmse: float
    rmse: float
    mae: float
    mape: float
    n_samples: int


@dataclass
class FourWayMetrics:
    """Container for four-way comparison metrics."""
    train: DatasetMetrics
    test: DatasetMetrics
    matched: DatasetMetrics
    alternative: DatasetMetrics

    # Gap metrics
    gap_generalization_r2: float
    gap_distribution_r2: float
    gap_fuel_type_r2: float

    gap_generalization_rrmse: float
    gap_distribution_rrmse: float
    gap_fuel_type_rrmse: float

class MultiDatasetFourWayMetrics:
    """Container for multi-dataset comparison metrics."""
    dataset_name: str  # e.g., 'LNG', 'Methanol', 'Ammonia'
    metrics: FourWayMetrics


class DomainShiftPlotter:
    """
    Plots domain shift analysis for four-way comparison.

    Pure plotting class - no knowledge of analyzers.
    Just takes structured data and creates visualizations.

    """

    def __init__(self,
                 figsize_per_target: Tuple[float, float] = (16, 4),
                 verbose: bool = False):  # ← ADD this
        """
        Initialize plotter.

        Parameters
        ----------
        figsize_per_target : tuple
            Figure size for each target (width, height)
        verbose : bool
            Whether to print diagnostic messages
        """
        self.figsize_per_target = figsize_per_target
        self.verbose = verbose  # ← ADD this
        self.colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        self.dataset_labels = ['Train', 'Test', 'Matched\n(Conv)', 'Alternative\n(Alt Fuel)']

    def plot_single_target(self,
                           target_name: str,
                           predictor_names: List[str],
                           model_description: str,
                           metrics: FourWayMetrics,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot domain shift for a single target.

        Creates side-by-side R² and RRMSE plots with gap annotations.

        Parameters
        ----------
        target_name : str
            Name of target dimension (e.g., 'Main_Power')
        predictor_names : list of str
            Predictor names (e.g., ['DWT', 'LOA'])
        model_description : str
            Model type description (e.g., 'polynomial-2', 'Random Forest')
        metrics : FourWayMetrics
            All metrics for four datasets
        save_path : str, optional
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, (ax_r2, ax_rrmse) = plt.subplots(1, 2, figsize=self.figsize_per_target)

        # Plot R² (left)
        self._plot_r2_comparison(ax_r2, target_name, metrics)

        # Plot RRMSE (right)
        self._plot_rrmse_comparison(ax_rrmse, target_name, metrics)

        # Overall title
        predictor_str = ' + '.join(predictor_names)
        fig.suptitle(f'{target_name}: {predictor_str} ({model_description})',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_multiple_targets(self,
                              targets_data: Dict[str, Tuple[List[str], str, FourWayMetrics]],
                              save_path: str|Path|None = None) -> plt.Figure:
        """
        Plot domain shift for multiple targets in one figure.

        Parameters
        ----------
        targets_data : dict
            Dictionary mapping target names to (predictors, model_desc, metrics)
            Example: {
                'Main_Power': (['DWT', 'LOA'], 'polynomial-2', metrics1),
                'Speed': (['DWT'], 'linear', metrics2)
            }
        save_path : str, optional
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if isinstance(save_path,str):
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        n_targets = len(targets_data)

        if n_targets == 0:
            raise ValueError("No targets to plot")

        fig, axes = plt.subplots(n_targets, 2,
                                 figsize=(self.figsize_per_target[0],
                                          self.figsize_per_target[1] * n_targets))

        if n_targets == 1:
            axes = axes.reshape(1, -1)

        for idx, (target_name, (predictors, model_desc, metrics)) in enumerate(targets_data.items()):
            ax_r2, ax_rrmse = axes[idx]

            self._plot_r2_comparison(ax_r2, target_name, metrics)
            self._plot_rrmse_comparison(ax_rrmse, target_name, metrics)

        fig.suptitle('Domain Shift Analysis: R² vs RRMSE',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_r2_comparison(self, ax: plt.Axes, target_name: str, metrics: FourWayMetrics):
        """Plot R² comparison with gap annotations."""
        r2_values = [
            metrics.train.r2,
            metrics.test.r2,
            metrics.matched.r2,
            metrics.alternative.r2
        ]

        # Bar plot
        bars = ax.bar(self.dataset_labels, r2_values, color=self.colors,
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        # Gap annotations
        gaps = [
            metrics.gap_generalization_r2,
            metrics.gap_distribution_r2,
            metrics.gap_fuel_type_r2
        ]
        gap_labels = ['Gen', 'Dist', 'FUEL']

        for i, (gap, label) in enumerate(zip(gaps, gap_labels)):
            x = i + 0.5
            y = min(r2_values[i], r2_values[i + 1])
            ax.annotate(
                f'{label}\n{gap:+.3f}',
                xy=(x, y),
                xytext=(x, y - 0.15),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                                color='red' if label == 'FUEL' else 'gray')
            )

        # Formatting
        ax.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax.set_title(f'{target_name} - R² Performance', fontsize=11, fontweight='bold')
        ax.set_ylim([max(0, min(r2_values) - 0.3), 1.0])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)

        # Value labels
        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _plot_rrmse_comparison(self, ax: plt.Axes, target_name: str, metrics: FourWayMetrics):
        """Plot RRMSE comparison with gap annotations."""
        rrmse_values = [
            metrics.train.rrmse,
            metrics.test.rrmse,
            metrics.matched.rrmse,
            metrics.alternative.rrmse
        ]

        # Color scale based on error magnitude
        max_rrmse = max(rrmse_values)
        colors = [plt.cm.RdYlGn_r(val / max_rrmse) for val in rrmse_values]

        # Bar plot
        bars = ax.bar(self.dataset_labels, rrmse_values, color=colors,
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        # Gap annotations
        gaps = [
            metrics.gap_generalization_rrmse,
            metrics.gap_distribution_rrmse,
            metrics.gap_fuel_type_rrmse
        ]
        gap_labels = ['Gen', 'Dist', 'FUEL']

        for i, (gap, label) in enumerate(zip(gaps, gap_labels)):
            x = i + 0.5
            y = max(rrmse_values[i], rrmse_values[i + 1])
            ax.annotate(
                f'{label}\n{gap:+.1f}%',
                xy=(x, y),
                xytext=(x, y + max_rrmse * 0.1),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                                color='red' if label == 'FUEL' else 'gray')
            )

        # Formatting
        ax.set_ylabel('RRMSE (%)', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Error (RRMSE)', fontsize=11, fontweight='bold')
        ax.set_ylim([0, max_rrmse * 1.4])
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, value in zip(bars, rrmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    def plot_multi_dataset_comparison(self,
                                      target_name: str,
                                      predictor_names: List[str],
                                      model_description: str,
                                      datasets_metrics: Dict[str, FourWayMetrics],
                                      rrmse_threshold: Optional[float] = None,
                                      threshold_label: Optional[str] = None,
                                      show_average_lines: bool = False,
                                      save_path: str|Path|None = None) -> plt.Figure:

        """
            Compare domain shift across multiple alternative fuel datasets.

            Parameters
            ----------
            target_name : str
                Target dimension name
            predictor_names : list of str
                Predictor names
            model_description : str
                Model type description
            datasets_metrics : dict
                Dictionary mapping fuel type names to FourWayMetrics
            rrmse_threshold : float, optional
                Horizontal reference line value on RRMSE panels.
                Typically the median/mean conventional test RRMSE for the target's category.
            threshold_label : str, optional
                Label shown in the textbox explaining the threshold line.
                Example: "Median conv. test RRMSE"
            show_average_lines : bool
                If True, draws dotted average lines across bar groups.
            save_path : str, optional
                Path to save figure

            Returns
            -------
            matplotlib.figure.Figure
                The created figure
            """
        n_datasets = len(datasets_metrics)

        if isinstance(save_path,str):
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if n_datasets > 4:
            print(f"⚠️  Warning: {n_datasets} datasets is a lot for one plot.")
            print(f"   Consider breaking into multiple plots or using summary view.")

        # Create figure: 2 rows (R² and RRMSE) × n_datasets columns
        fig, axes = plt.subplots(2, n_datasets,
                                 figsize=(6 * n_datasets, 10))

        if n_datasets == 1:
            axes = axes.reshape(2, 1)

        # Plot each dataset
        for col_idx, (fuel_type, metrics) in enumerate(datasets_metrics.items()):
            ax_r2 = axes[0, col_idx]
            ax_rrmse = axes[1, col_idx]

            # Plot R² comparison
            self._plot_r2_comparison_single_column(
                ax_r2, fuel_type, metrics
            )

            # Plot RRMSE comparison
            self._plot_rrmse_comparison_single_column(
                ax_rrmse, fuel_type, metrics,
                rrmse_threshold=rrmse_threshold,
                threshold_label=threshold_label,
                show_average_lines=show_average_lines
            )

        # Overall title
        predictor_str = ' + '.join(predictor_names)
        fig.suptitle(
            f'Multi-Dataset Comparison: {target_name}\n'
            f'{predictor_str} ({model_description})',
            fontsize=16, fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_r2_comparison_single_column(self, ax: plt.Axes,
                                          fuel_type: str,
                                          metrics: FourWayMetrics):
        """Plot R² for one fuel type (used in multi-dataset comparison)."""
        r2_values = [
            metrics.train.r2,
            metrics.test.r2,
            metrics.matched.r2,
            metrics.alternative.r2
        ]

        # Use same colors as before
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        labels = ['Train', 'Test', 'Match', 'Alt']

        bars = ax.bar(labels, r2_values, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Highlight fuel type gap
        fuel_gap = metrics.gap_fuel_type_r2
        y_pos = min(metrics.matched.r2, metrics.alternative.r2)
        ax.annotate(
            f'FUEL\n{fuel_gap:+.3f}',
            xy=(2.5, y_pos),
            xytext=(2.5, y_pos - 0.15),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red' if abs(fuel_gap) > 0.1 else 'yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='red')
        )

        # Formatting
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f'{fuel_type}', fontsize=12, fontweight='bold')
        ax.set_ylim([max(0, min(r2_values) - 0.3), 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _plot_rrmse_comparison_single_column(self, ax: plt.Axes,
                                             fuel_type: str,
                                             metrics: FourWayMetrics,
                                             rrmse_threshold: Optional[float] = None,
                                             threshold_label: Optional[str] = None,
                                             show_average_lines: bool = False):
        """Plot RRMSE for one fuel type (used in multi-dataset comparison)."""
        rrmse_values = [
            metrics.train.rrmse,
            metrics.test.rrmse,
            metrics.matched.rrmse,
            metrics.alternative.rrmse
        ]

        max_rrmse = max(rrmse_values)
        colors = [plt.cm.RdYlGn_r(val / max_rrmse) for val in rrmse_values]
        labels = ['Train', 'Test', 'Match', 'Alt']

        bars = ax.bar(labels, rrmse_values, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Fuel type gap annotation
        fuel_gap = metrics.gap_fuel_type_rrmse
        y_pos = max(metrics.matched.rrmse, metrics.alternative.rrmse)
        ax.annotate(
            f'FUEL\n{fuel_gap:+.1f}%',
            xy=(2.5, y_pos),
            xytext=(2.5, y_pos + max_rrmse * 0.1),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red' if abs(fuel_gap) > 15 else 'yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='red')
        )

        # Threshold line
        if rrmse_threshold is not None:
            ax.axhline(
                y=rrmse_threshold,
                color='#E74C3C',
                linestyle='--',
                linewidth=1.8,
                alpha=0.8,
                zorder=10
            )

            # Textbox in top-left explaining the line
            label_text = threshold_label or f"Reference: {rrmse_threshold:.1f}%"
            box_text = f"{label_text}\n({rrmse_threshold:.1f}%)"

            ax.text(
                0.03, 0.97, box_text,
                transform=ax.transAxes,
                fontsize=8,
                va='top', ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='lightyellow',
                    edgecolor='#E74C3C',
                    alpha=0.9
                ),
                family='monospace',
                zorder=15
            )

        # Optional average lines (off by default - these caused the dotted line confusion)
        if show_average_lines:
            avg = np.nanmean(rrmse_values)
            if np.isfinite(avg):
                ax.axhline(y=avg, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Formatting
        ax.set_ylabel('RRMSE (%)', fontsize=11, fontweight='bold')
        ax.set_ylim([0, max(max_rrmse * 1.4,
                            rrmse_threshold * 1.2 if rrmse_threshold else max_rrmse * 1.4)])
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, value in zip(bars, rrmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    def plot_fuel_type_comparison_summary(self,
                                          targets_data: Dict[str, Dict[str, FourWayMetrics]],
                                          metric: str = 'gap_fuel_type_rrmse',
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create summary heatmap comparing fuel types across targets.

        **FIXED**: Now handles missing data gracefully - only plots targets/fuel combinations
        that have valid results.

        Parameters
        ----------
        targets_data : dict
            Nested dict: {target: {fuel_type: FourWayMetrics}}
            Example: {
                'Main_Power': {'LNG': metrics1, 'Methanol': metrics2},
                'Speed': {'LNG': metrics3}  # Methanol missing - OK!
            }
        metric : str
            Which metric to display
        save_path : str, optional
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """

        # =========================================================================
        # STEP 1: Validate and clean data
        # =========================================================================

        if not targets_data:
            raise ValueError("targets_data is empty - no data to plot")

        # Find all fuel types that appear in ANY target
        all_fuel_types = set()
        for fuel_dict in targets_data.values():
            all_fuel_types.update(fuel_dict.keys())

        fuel_types = sorted(list(all_fuel_types))

        if not fuel_types:
            raise ValueError("No fuel types found in targets_data")

        # Find targets that have at least one fuel type
        valid_targets = [t for t in targets_data.keys() if targets_data[t]]

        if not valid_targets:
            raise ValueError("No valid targets with data found")

        targets = sorted(valid_targets)

        if self.verbose:
            print(f"\n📊 Creating fuel type comparison summary:")
            print(f"  Targets: {len(targets)}")
            print(f"  Fuel types: {len(fuel_types)}")

            # Show missing combinations
            missing_count = 0
            for target in targets:
                for fuel_type in fuel_types:
                    if fuel_type not in targets_data[target]:
                        missing_count += 1

            if missing_count > 0:
                print(f"  ⚠️  Missing combinations: {missing_count}/{len(targets) * len(fuel_types)}")
                print(f"      (These will be shown as NaN/gray in the heatmap)")

        # =========================================================================
        # STEP 2: Create data matrix (with NaN for missing combinations)
        # =========================================================================

        data_matrix = np.full((len(targets), len(fuel_types)), np.nan)

        for i, target in enumerate(targets):
            for j, fuel_type in enumerate(fuel_types):
                # Check if this combination exists
                if fuel_type not in targets_data[target]:
                    continue  # Leave as NaN

                metrics = targets_data[target][fuel_type]

                # Extract the requested metric
                if metric == 'gap_fuel_type_rrmse':
                    data_matrix[i, j] = metrics.gap_fuel_type_rrmse
                elif metric == 'gap_fuel_type_r2':
                    data_matrix[i, j] = metrics.gap_fuel_type_r2
                elif metric == 'alternative_rrmse':
                    data_matrix[i, j] = metrics.alternative.rrmse
                elif metric == 'alternative_r2':
                    data_matrix[i, j] = metrics.alternative.r2
                else:
                    raise ValueError(f"Unknown metric: {metric}")

        # =========================================================================
        # STEP 3: Create heatmap
        # =========================================================================

        fig, ax = plt.subplots(figsize=(max(8, len(fuel_types) * 1.5),
                                        max(6, len(targets) * 0.8)))

        # Choose colormap based on metric
        if 'gap' in metric or 'rrmse' in metric.lower():
            cmap = 'RdYlGn_r'  # Red = bad (high error), Green = good (low error)
        else:
            cmap = 'RdYlGn'  # Red = bad (low R²), Green = good (high R²)

        # Create masked array to handle NaN values
        masked_data = np.ma.masked_invalid(data_matrix)

        im = ax.imshow(masked_data, cmap=cmap, aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(fuel_types)))
        ax.set_yticks(np.arange(len(targets)))
        ax.set_xticklabels(fuel_types)
        ax.set_yticklabels(targets)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Add text annotations
        for i in range(len(targets)):
            for j in range(len(fuel_types)):
                value = data_matrix[i, j]

                if np.isnan(value):
                    # Show "N/A" for missing data
                    text = ax.text(j, i, "N/A",
                                   ha="center", va="center",
                                   color="gray", fontsize=9, style='italic')
                else:
                    # Show value
                    text = ax.text(j, i, f'{value:.1f}',
                                   ha="center", va="center",
                                   color="black", fontweight='bold')

        # Labels and title
        ax.set_xlabel('Fuel Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Dimension', fontsize=12, fontweight='bold')

        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f'Fuel Type Comparison: {metric_name}',
                     fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig