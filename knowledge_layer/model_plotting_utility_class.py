"""
Model Plotting Utilities
========================

Reusable plotting infrastructure for model selection classes.

✅ UPDATED: Enhanced with ship_type and variable_name_map support
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import math


class ModelPlotter:
    """
    Reusable plotting utilities for model selection results.

    Uses composition pattern - selector classes pass their data
    to these methods rather than inheriting.
    """

    # ========================================================================
    # METRIC UTILITIES
    # ========================================================================

    @staticmethod
    def get_metric_label(metric: str) -> str:
        """Map metric names to display labels."""
        metric_labels = {
            'adj_r2': 'Adjusted R²',
            'r2': 'R²',
            'rrmse': 'RRMSE (%)',
            'mape': 'MAPE (%)',
            'rmse': 'RMSE',
            'mae': 'MAE'
        }
        return metric_labels.get(metric, metric)

    @staticmethod
    def get_cv_text(cv_method: Optional[str], n_folds: int = 10) -> str:
        """Generate CV method description text."""
        if cv_method is None:
            return " (No CV - Training Set Only)"
        elif cv_method == 'standard':
            return f" ({n_folds}-Fold CV)"
        elif cv_method == 'stratified':
            return f" ({n_folds}-Fold Stratified CV)"
        elif cv_method == 'grouped':
            return f" ({n_folds}-Fold Grouped CV)"
        else:
            return f" ({cv_method})"

    # ========================================================================
    # LAYOUT UTILITIES
    # ========================================================================

    @staticmethod
    def calculate_grid_layout(n_plots: int,
                              max_cols: int = 2) -> Tuple[int, int]:
        """
        Calculate optimal grid layout for subplots.

        Returns
        -------
        (rows, cols) : tuple
        """
        cols = min(max_cols, n_plots)
        rows = math.ceil(n_plots / cols)
        return rows, cols

    @staticmethod
    def setup_figure(n_plots: int,
                     figsize_per_plot: Tuple[float, float],
                     title: str,
                     max_cols: int = 2) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create figure with subplot grid.

        Returns
        -------
        (fig, axes) : Figure and flattened axes array
        """
        rows, cols = ModelPlotter.calculate_grid_layout(n_plots, max_cols)

        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(fig_width, fig_height))

        # Flatten axes for easier indexing
        if n_plots == 1:
            axes = np.array([axes])
        elif n_plots > 1:
            axes = axes.flatten()

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        return fig, axes

    @staticmethod
    def hide_extra_subplots(axes: np.ndarray, n_plots: int):
        """Hide unused subplots in grid."""
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

    # ========================================================================
    # BAR PLOT UTILITIES
    # ========================================================================

    @staticmethod
    def create_colored_bars(ax: plt.Axes,
                            x_pos: np.ndarray,
                            values: List[float],
                            n_predictors_list: List[int],
                            max_predictors: int,
                            add_error_bars: bool = False,
                            std_values: Optional[List[float]] = None):
        """
        Create bar plot with color gradient based on n_predictors.

        Parameters
        ----------
        ax : matplotlib Axes
        x_pos : array of x positions
        values : bar heights
        n_predictors_list : number of predictors for each bar
        max_predictors : max predictors (for color normalization)
        add_error_bars : whether to add error bars
        std_values : standard deviations for error bars

        Returns
        -------
        bars : BarContainer
        """
        # Create color gradient
        colors = plt.cm.viridis([n / max(max_predictors, 1)
                                 for n in n_predictors_list])

        # Create bars
        bars = ax.bar(x_pos, values, color=colors,
                      alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add error bars if requested
        if add_error_bars and std_values is not None:
            ax.errorbar(x_pos, values, yerr=std_values,
                        fmt='none', ecolor='red',
                        capsize=3, capthick=1, alpha=0.7)

        return bars

    @staticmethod
    def add_bar_labels(ax: plt.Axes,
                       bars,
                       values: List[float],
                       std_values: Optional[List[float]] = None,
                       include_std: bool = False,
                       fontsize: int = 7):
        """Add value labels on top of bars."""
        for bar, value, std in zip(bars, values,
                                   std_values or [0] * len(values)):
            height = bar.get_height()

            if include_std and std > 0:
                label_text = f'{value:.3f}\n±{std:.3f}'
            else:
                label_text = f'{value:.3f}'

            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    label_text,
                    ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def add_predictor_legend(ax: plt.Axes,
                             min_predictors: int,
                             max_predictors: int,
                             location: str = 'upper right'):
        """Add legend showing number of predictors."""
        from matplotlib.patches import Patch

        legend_elements = []
        for n in range(min_predictors, max_predictors + 1):
            color = plt.cm.viridis((n - min_predictors) /
                                   max(max_predictors - min_predictors, 1))
            legend_elements.append(
                Patch(facecolor=color, alpha=0.7,
                      label=f'{n} predictors')
            )

        ax.legend(handles=legend_elements, loc=location, fontsize=8)

    # ========================================================================
    # FORMATTING UTILITIES
    # ========================================================================

    @staticmethod
    def format_axes(ax: plt.Axes,
                    xlabel: str,
                    ylabel: str,
                    title: str,
                    x_labels: List[str],
                    rotation: int = 45):
        """Apply consistent formatting to axes."""
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=12)

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=rotation,
                           ha='right', fontsize=8)

        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # ========================================================================
    # SAVE UTILITIES
    # ========================================================================

    @staticmethod
    def save_and_show(fig: plt.Figure,
                      save_path: Optional[str] = None,
                      verbose: bool = True):
        """Save figure if path provided, then show."""
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✅ Plot saved to {save_path}")

        plt.show()


# ============================================================================
# METRICS COMPARISON PLOT - GENERIC VERSION
# ============================================================================

class MetricsComparisonPlotter:
    """
    Handles metrics comparison plots for any selector class.

    This is designed to work with different data structures
    from different selector classes.

    ✅ UPDATED: Enhanced with ship_type and variable_name_map support
    """

    @staticmethod
    def plot_metrics_comparison(
            results_by_target: Dict[str, List[Dict]],
            metric: str = 'rrmse',
            cv_method: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            include_only_vars: Optional[List[str]] = None,
            figsize_per_plot: Tuple[float, float] = (8, 6),
            title: str = "Model Performance Comparison",
            save_path: Optional[str] = None,
            verbose: bool = True
    ):
        """
        Create metrics comparison plot.

        Parameters
        ----------
        results_by_target : dict
            {target_name: [{'label': str, 'metric_value': float,
                             'std_value': float, 'n_predictors': int,
                             'predictors': tuple}, ...]}
        metric : str
            Metric to plot
        cv_method : str, optional
            CV method used (shown in subtitle)
        ship_type : str, optional
            Ship type shown in main title
        variable_name_map : dict, optional
            Maps variable names to display names
        include_only_vars : list of str, optional
            Only include combinations where ALL predictors are in this list.
            Combinations containing any variable outside the list are excluded.
            Example: ['DWT', 'GT'] excludes any combo using 'LOA (M)' etc.
        figsize_per_plot : tuple
            Size per subplot
        title : str
            Main title text
        save_path : str, optional
            Path to save figure
        verbose : bool
            Print save confirmation
        """
        # ------------------------------------------------------------------ #
        # 1. FILTER by include_only_vars                                       #
        # ------------------------------------------------------------------ #
        # Each result dict is expected to carry a 'predictors' tuple built by
        # _prepare_plot_data(). Combinations missing that key are kept (safe
        # fallback so old callers without the key still work).
        if include_only_vars is not None:
            allowed = set(include_only_vars)
            filtered = {}
            for target, results in results_by_target.items():
                kept = [
                    r for r in results
                    if all(p in allowed for p in r.get('predictors', ()))
                ]
                if kept:
                    filtered[target] = kept
            results_by_target = filtered

            if not results_by_target:
                print("No results remain after include_only_vars filtering!")
                return

        n_targets = len(results_by_target)
        if n_targets == 0:
            print("No results to plot!")
            return

        # ------------------------------------------------------------------ #
        # 2. BUILD TITLE COMPONENTS                                            #
        # ------------------------------------------------------------------ #
        # Main title: ship type + user title
        # Subtitle:   CV info + optional variable filter note
        # Keeping them separate lets us use different font sizes.

        main_title = f"{ship_type}: {title}" if ship_type else title

        subtitle_parts = []
        if cv_method is None:
            subtitle_parts.append("No CV — training set only")
        elif cv_method == 'standard':
            subtitle_parts.append("10-fold CV")
        elif cv_method == 'stratified':
            subtitle_parts.append("10-fold stratified CV")
        else:
            subtitle_parts.append(cv_method)

        if include_only_vars is not None:
            var_str = ", ".join(include_only_vars[:5])
            if len(include_only_vars) > 5:
                var_str += f" … (+{len(include_only_vars) - 5})"
            subtitle_parts.append(f"predictors: {var_str}")

        subtitle = " | ".join(subtitle_parts)

        # ------------------------------------------------------------------ #
        # 3. CREATE FIGURE                                                     #
        # ------------------------------------------------------------------ #
        rows, cols = ModelPlotter.calculate_grid_layout(n_targets)
        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        if n_targets == 1:
            axes = np.array([axes])
        elif n_targets > 1:
            axes = axes.flatten()

        # Two-line title: main title bold at top, subtitle smaller below
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)
        fig.text(0.5, 0.975, subtitle,
                 ha='center', va='top',
                 fontsize=9, style='italic', color='#444444')

        # ------------------------------------------------------------------ #
        # 4. PLOT EACH TARGET                                                  #
        # ------------------------------------------------------------------ #
        for idx, (target, results) in enumerate(results_by_target.items()):
            ax = axes[idx]
            MetricsComparisonPlotter._plot_single_target(
                ax, target, results, metric, cv_method is not None,
                variable_name_map=variable_name_map,
                is_first=(idx == 0)
            )

        ModelPlotter.hide_extra_subplots(axes, n_targets)

        # Extra top margin so the two-line header doesn't overlap subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✅ Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def _plot_single_target(ax, target, results, metric, has_cv,
                            variable_name_map=None, is_first=False):
        """Plot single target dimension."""
        if not results:
            ax.text(0.5, 0.5, f'No results for {target}',
                    ha='center', va='center', transform=ax.transAxes)

            # Format target name for title
            if variable_name_map:
                target_str = variable_name_map.get(target, target)
            else:
                target_str = target

            ax.set_title(target_str, fontweight='bold')
            return

        # Extract data
        labels = [r['label'] for r in results]
        values = [r['metric_value'] for r in results]
        stds = [r.get('std_value', 0) for r in results]
        n_preds = [r['n_predictors'] for r in results]

        x_pos = np.arange(len(labels))

        # Create bars
        bars = ModelPlotter.create_colored_bars(
            ax, x_pos, values, n_preds, max(n_preds),
            add_error_bars=has_cv, std_values=stds
        )

        # Add labels
        ModelPlotter.add_bar_labels(
            ax, bars, values, stds, include_std=has_cv
        )

        # Format target name for title
        if variable_name_map:
            target_str = variable_name_map.get(target, target)
        else:
            target_str = target

        # Format axes
        ModelPlotter.format_axes(
            ax,
            xlabel='Predictor Combination (Model)',
            ylabel=ModelPlotter.get_metric_label(metric),
            title=f'Target: {target_str}',
            x_labels=labels
        )

        # Add legend on first plot
        if is_first and len(set(n_preds)) > 1:
            ModelPlotter.add_predictor_legend(
                ax, min(n_preds), max(n_preds)
            )


# ============================================================================
# FEATURE IMPORTANCE PLOT (RF-SPECIFIC)
# ============================================================================

class FeatureImportancePlotter:
    """Handles feature importance visualization for Random Forest."""

    @staticmethod
    def plot_feature_importance(
            importance_by_target: Dict[str, Dict[str, float]],
            figsize_per_plot: Tuple[float, float] = (10, 6),
            save_path: Optional[str] = None,
            verbose: bool = True
    ):
        """
        Plot feature importance for each target.

        Parameters
        ----------
        importance_by_target : dict
            {target: {feature: importance}}
        """
        n_targets = len(importance_by_target)

        if n_targets == 0:
            print("No importance data to plot!")
            return

        fig, axes = ModelPlotter.setup_figure(
            n_targets, figsize_per_plot,
            'Feature Importance Analysis'
        )

        for idx, (target, importance) in enumerate(
                importance_by_target.items()
        ):
            ax = axes[idx]
            FeatureImportancePlotter._plot_single_importance(
                ax, target, importance
            )

        ModelPlotter.hide_extra_subplots(axes, n_targets)
        ModelPlotter.save_and_show(fig, save_path, verbose)

    @staticmethod
    def _plot_single_importance(ax, target, importance):
        """Plot importance for single target."""
        if not importance:
            ax.text(0.5, 0.5, 'No importance data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Target: {target}', fontweight='bold')
            return

        # Sort by importance
        sorted_items = sorted(importance.items(),
                              key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]

        # Create horizontal bars
        y_pos = np.arange(len(features))
        colors = plt.cm.RdYlGn(
            importances / max(importances) if importances else [1]
        )

        bars = ax.barh(y_pos, importances, color=colors,
                       edgecolor='black', linewidth=1)

        # Format
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance', fontsize=10)
        ax.set_title(f'Target: {target}', fontweight='bold', fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, importances):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2.,
                    f'{value:.3f}',
                    ha='left', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.7))