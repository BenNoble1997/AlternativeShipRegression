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

    LOWER_IS_BETTER = frozenset({
        'rrmse', 'nrmse', 'rmse', 'mae', 'mape', 'bic', 'aic', 'aicc'
    })

    # ========================================================================
    # METRIC UTILITIES
    # ========================================================================


    @staticmethod
    def metric_sign(metric: str) -> int:
        """
        Return the multiplier that turns a metric into a 'smaller is better' score.

        :param metric: Metric key as stored in the results dictionaries.
        :returns: ``1`` when lower values are better, ``-1`` otherwise.
        """
        return 1 if metric.lower() in ModelPlotter.LOWER_IS_BETTER else -1

    @staticmethod
    def find_best_index(results: List[Dict],
                        metric: str,
                        tolerance: float = 0.0) -> Tuple[Optional[int], List[int]]:
        """
        Identify the winning bar and any practically indistinguishable peers.

        :param results: Result dictionaries for a single target.
        :param metric: Metric key to compare on.
        :param tolerance: Relative tolerance for equivalence, e.g. ``0.02`` for
            2%. Zero disables tie detection.
        :returns: Index of the winner (``None`` if all values are NaN) and the
            indices of near-equivalent combinations, excluding the winner.
        """
        values = np.array([r.get('metric_value', np.nan) for r in results],
                          dtype=float)
        scores = ModelPlotter.metric_sign(metric) * values

        if np.all(np.isnan(scores)):
            return None, []

        best_idx = int(np.nanargmin(scores))
        best_score = scores[best_idx]

        if tolerance <= 0 or not np.isfinite(best_score) or best_score == 0:
            return best_idx, []

        margin = abs(best_score) * tolerance
        ties = [i for i, s in enumerate(scores)
                if i != best_idx and np.isfinite(s) and (s - best_score) <= margin]
        return best_idx, ties

    @staticmethod
    def get_metric_label(metric: str) -> str:
        """Map metric names to display labels."""
        metric_labels = {
            'adj_r2': 'Adjusted R²',
            'r2': 'R²',
            'rrmse': 'NRMSE (%)',
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

    HIGHLIGHT_COLOR = '#2CA02C'

    @staticmethod
    def highlight_bars(bars,
                       best_idx: Optional[int],
                       tie_idx: Optional[List[int]] = None,
                       color: str = HIGHLIGHT_COLOR):
        """
        Recolour the winning bar and outline any equivalent peers.

        :param bars: BarContainer returned by :meth:`create_colored_bars`.
        :param best_idx: Index of the winning bar, or ``None`` to do nothing.
        :param tie_idx: Indices of bars within the equivalence tolerance.
        :param color: Fill colour for the winner.
        """
        if best_idx is None:
            return

        bars[best_idx].set_color(color)
        bars[best_idx].set_alpha(0.9)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(1.2)

        for i in tie_idx or []:
            bars[i].set_edgecolor(color)
            bars[i].set_linewidth(1.8)
            bars[i].set_linestyle('--')

    @staticmethod
    def annotate_selected(ax: plt.Axes,
                          result: Dict,
                          metric: str,
                          model_name_map: Optional[Dict[str, str]] = None,
                          variable_name_map: Optional[Dict[str, str]] = None,
                          color: str = HIGHLIGHT_COLOR,
                          loc: str = 'upper left'):
        """
        Draw a text box naming the selected combination and its model.

        :param ax: Axes to annotate.
        :param result: Result dictionary of the winning combination.
        :param metric: Metric key, used for the value line.
        :param model_name_map: Optional display names for model types.
        :param variable_name_map: Optional display names for predictors.
        :param color: Border colour, matched to the highlighted bar.
        :param loc: ``'upper left'`` or ``'upper right'``. Choose the side that
            does not collide with the predictor legend.
        """
        predictors = result.get('predictors', ())
        if variable_name_map:
            predictors = [variable_name_map.get(p, p) for p in predictors]
        predictor_str = " + ".join(predictors) if predictors else result.get('label', '')

        model_name = result.get('model_name', 'unknown')
        model_label = (model_name_map or {}).get(model_name, model_name)

        value = result.get('metric_value', np.nan)
        std = result.get('std_value', 0) or 0
        value_line = f"{ModelPlotter.get_metric_label(metric)} = {value:.3f}"
        if std:
            value_line += f" $\\pm$ {std:.3f}"

        x, ha = (0.02, 'left') if loc == 'upper left' else (0.98, 'right')

        ax.text(x, 0.97,
                f"Selected\n{predictor_str}\n{model_label}\n{value_line}",
                transform=ax.transAxes, ha=ha, va='top', fontsize=8.5,
                bbox={'boxstyle': 'round,pad=0.45', 'facecolor': 'white',
                      'edgecolor': color, 'linewidth': 1.4, 'alpha': 0.92})

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
            highlight_best: bool = True,
            figsize:tuple|None = None,
            equivalence_tolerance: Optional[float] = None,
            model_name_map: Optional[Dict[str, str]] = None,
            annotation_loc:str="upper left",
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

        main_title = (f"{ship_type}:"
                      f"\n {title}") if ship_type else title

        subtitle_parts = []
        if cv_method == 'standard':
            subtitle_parts.append("10-fold CV")
        elif cv_method == 'stratified':
            subtitle_parts.append("10-fold stratified CV")
        else:
            subtitle_parts=[]

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

        if figsize is None:
            fig_width = figsize_per_plot[0] * cols
            fig_height = figsize_per_plot[1] * rows
        else:
            fig_width, fig_height = figsize

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
                is_first=(idx == 0),
                highlight_best=highlight_best,
                annotation_loc=annotation_loc,
                model_name_map=model_name_map,
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
                            variable_name_map=None, is_first=False,
                            highlight_best=True, model_name_map=None,
                            equivalence_tolerance=0.0,
                            annotation_loc='upper left'):
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

        best_idx, tie_idx = (None, [])
        if highlight_best:
            best_idx, tie_idx = ModelPlotter.find_best_index(
                results, metric, equivalence_tolerance
            )
            ModelPlotter.highlight_bars(bars, best_idx, tie_idx)

            # Headroom so the annotation box clears the bar value labels.
            finite = [v for v in values if np.isfinite(v)]
            if finite:
                ax.set_ylim(0, max(finite) * 1.35)

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

        if best_idx is not None:
            ModelPlotter.annotate_selected(
                ax, results[best_idx], metric,
                model_name_map=model_name_map,
                variable_name_map=variable_name_map,
                loc=annotation_loc
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