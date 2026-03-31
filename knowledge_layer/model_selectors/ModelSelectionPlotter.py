"""
Model Selection Comparison Plotter
===================================

Visualizes the model selection process by showing all tested models
and their performance metrics for a specific predictor-target combination.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict


class ModelSelectionPlotter:
    """
    Visualizes model selection decisions.

    **Concept:**
    Shows why a specific model was selected by comparing all tested models
    on the chosen selection metric. This helps understand:
    - How close the competition was
    - Whether the selected model clearly outperformed others
    - If multiple models were essentially equivalent
    """

    @staticmethod
    def plot_model_selection_comparison(
            selector,
            predictor_combo: Tuple[str, ...],
            target: str,
            metric: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            highlight_selected: bool = True,
            show_values: bool = True,
            show_heteroscedasticity: bool = True,  # NEW parameter
            figsize: Tuple[float, float] = (10, 6),
            save_path: Optional[str] = None
    ):
        """
        Plot comparison of all models tested for a specific combination.

        **New Feature**: Shows heteroscedasticity status for each model with color-coded indicators.

        Parameters
        ----------
        show_heteroscedasticity : bool, default=True
            Show heteroscedasticity indicators next to model names.
            Uses emoji/symbols: ✓ (none), ⚠ (weak/strong)

        ... (other parameters same as before) ...
        """

        # ====================================================================
        # 1. Get Results
        # ====================================================================

        key = (tuple(predictor_combo), target)

        if key not in selector.detailed_results:
            raise ValueError(
                f"No results found for {' + '.join(predictor_combo)} → {target}. "
                f"Run analyze_all_combinations() first."
            )

        model_results = selector.detailed_results[key]

        if not model_results:
            raise ValueError(f"No model results available for this combination")

        # Determine metric
        if metric is None:
            metric = selector.selection_metric

        # Get selected model
        selected_model = selector.model_recommendations.get(key)

        # ====================================================================
        # 2. Prepare Data
        # ====================================================================

        # Extract metric values for all models
        data = []
        for model_name, metrics in model_results.items():
            metric_value = metrics.get(metric)

            # Skip if metric not available
            if metric_value is None or np.isnan(metric_value):
                continue

            # NEW: Extract heteroscedasticity info
            het_status = metrics.get('heteroscedasticity', 'unknown')

            data.append({
                'model': model_name,
                'value': metric_value,
                'is_selected': (model_name == selected_model),
                'het_status': het_status  # NEW field
            })

        if not data:
            raise ValueError(f"No valid {metric} values found for any model")

        # Sort models
        lower_is_better = metric in ['bic', 'aic', 'rrmse', 'mape', 'rmse', 'mae']
        data.sort(key=lambda x: x['value'], reverse=not lower_is_better)

        # ====================================================================
        # 3. Create Plot
        # ====================================================================

        fig, ax = plt.subplots(figsize=figsize)

        # Extract data for plotting
        models = [d['model'] for d in data]
        values = [d['value'] for d in data]
        is_selected = [d['is_selected'] for d in data]
        het_statuses = [d['het_status'] for d in data]  # NEW

        # Determine colors
        if highlight_selected:
            colors = ['#2ecc71' if selected else '#3498db'
                      for selected in is_selected]
        else:
            colors = '#3498db'

        # Create horizontal bar chart
        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, values, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels
        if show_values:
            for bar, value, selected in zip(bars, values, is_selected):
                x_pos = bar.get_width()

                # Format value
                if metric in ['adj_r2', 'r2']:
                    label = f'{value:.4f}'
                elif metric in ['rrmse', 'mape']:
                    label = f'{value:.2f}%'
                elif metric in ['aic', 'bic']:
                    label = f'{value:.1f}'
                else:
                    label = f'{value:.3f}'

                # Add selected marker
                if selected and highlight_selected:
                    label = f'{label} ←'

                ax.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                        f' {label}',
                        va='center', ha='left', fontsize=10,
                        fontweight='bold' if selected else 'normal')

        # ====================================================================
        # 4. Format Plot - NEW: Add heteroscedasticity indicators
        # ====================================================================

        # Format variable names
        if variable_name_map:
            predictors_str = ' + '.join([
                variable_name_map.get(p, p) for p in predictor_combo
            ])
            target_str = variable_name_map.get(target, target)
        else:
            predictors_str = ' + '.join(predictor_combo)
            target_str = target

        # Y-axis (models) - NEW: Include heteroscedasticity status
        if show_heteroscedasticity:
            model_labels = []
            for model, het_status in zip(models, het_statuses):
                # Determine indicator
                if het_status == 'No':
                    indicator = '✓'
                    color_code = ''
                elif het_status in ['Strong', 'Weak']:
                    indicator = '⚠'
                    color_code = ''
                else:
                    indicator = '?'
                    color_code = ''

                label = f"{model} {indicator}"
                model_labels.append(label)
        else:
            model_labels = models

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_labels, fontsize=10)
        ax.set_ylabel('Model Type', fontsize=11, fontweight='bold')

        # X-axis (metric)
        metric_label = ModelSelectionPlotter._get_metric_label(metric)
        ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')

        # Title
        n_models = len(models)

        if ship_type:
            title_prefix = f'{ship_type}: '
        else:
            title_prefix = ''

        if selected_model:
            title = (f'{title_prefix}Model Selection: {predictors_str} → {target_str}\n'
                     f'Selected: {selected_model} ({n_models} models compared)')
        else:
            title = (f'{title_prefix}Model Comparison: {predictors_str} → {target_str}\n'
                     f'({n_models} models tested)')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add vertical line at "best" position
        if lower_is_better:
            best_value = min(values)
            ax.axvline(best_value, color='red', linestyle='--',
                       linewidth=2, alpha=0.5, label='Best value')
        else:
            best_value = max(values)
            ax.axvline(best_value, color='red', linestyle='--',
                       linewidth=2, alpha=0.5, label='Best value')

        # ====================================================================
        # 5. Legend - NEW: Include heteroscedasticity explanation
        # ====================================================================

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []

        if highlight_selected and selected_model:
            legend_elements.extend([
                Patch(facecolor='#2ecc71', edgecolor='black',
                      label='Selected model'),
                Patch(facecolor='#3498db', edgecolor='black',
                      label='Other models')
            ])

        if show_heteroscedasticity:
            legend_elements.extend([
                Line2D([0], [0], marker='$✓$', color='w',
                       markerfacecolor='black', markersize=10,
                       label='No heteroscedasticity'),
                Line2D([0], [0], marker='$⚠$', color='w',
                       markerfacecolor='orange', markersize=10,
                       label='Heteroscedastic')
            ])

        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right',
                      fontsize=9, framealpha=0.9)

        # ====================================================================
        # 6. Add Diagnostic Info (unchanged)
        # ====================================================================

        if metric in ['bic', 'aic'] and selected_model:
            selected_value = model_results[selected_model][metric]
            threshold = 2.0

            equivalent = [m for m in models
                          if abs(model_results[m][metric] - selected_value) < threshold]

            if len(equivalent) > 1:
                equiv_str = ', '.join(equivalent)
                info_text = (f'Note: {len(equivalent)} models within '
                             f'Δ{metric.upper()} < 2.0:\n{equiv_str}')

                ax.text(0.02, 0.98, info_text,
                        transform=ax.transAxes,
                        fontsize=8, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='yellow', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if hasattr(selector, 'verbose') and selector.verbose:
                print(f"✅ Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def _get_metric_label(metric: str) -> str:
        """Get display label for metric."""
        labels = {
            'adj_r2': 'Adjusted R² (higher is better)',
            'r2': 'R² (higher is better)',
            'bic': 'BIC (lower is better)',
            'aic': 'AIC (lower is better)',
            'rrmse': 'RRMSE % (lower is better)',
            'mape': 'MAPE % (lower is better)',
            'rmse': 'RMSE (lower is better)',
            'mae': 'MAE (lower is better)'
        }
        return labels.get(metric, metric)

    @staticmethod
    def plot_multiple_selections(
            selector,
            combinations: List[Tuple[Tuple[str, ...], str]],
            metric: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            highlight_selected: bool = True,
            show_values: bool = True,
            show_heteroscedasticity: bool = True,  # NEW parameter
            figsize_per_plot: Tuple[float, float] = (10, 5),
            save_path_template: Optional[str] = None
    ):
        """
        Plot selection comparisons for multiple combinations.

        **Use Case:**
        Compare model selection across different predictor combinations
        or targets to understand selection patterns.

        Parameters
        ----------
        selector : ParametricModelSelector
            Fitted selector
        combinations : list of (predictors, target) tuples
            List of combinations to plot
        metric : str, optional
            Metric to display
        ship_type : str, optional
            Ship type to include in overall title
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names
        highlight_selected : bool, default=True
            Highlight selected models
        show_values : bool, default=True
            Show values on bars
        figsize_per_plot : tuple, default=(10, 5)
            Size per subplot
        save_path_template : str, optional
            Template for save paths. Use {i} for index.
            Example: 'plots/selection_{i}.png'

        Examples
        --------
        >>> # Compare selection for same target, different predictors
        >>> variable_map = {
        ...     'DWT': 'DWT (t)',
        ...     'GT': 'GT (m³)',
        ...     'LOA (M)': 'L_OA (m)'
        ... }
        >>> combinations = [
        ...     (('DWT',), 'LOA (M)'),
        ...     (('GT',), 'LOA (M)'),
        ...     (('DWT', 'GT'), 'LOA (M)')
        ... ]
        >>> selector.plot_multiple_selections(
        ...     combinations,
        ...     ship_type='Container Ships',
        ...     variable_name_map=variable_map
        ... )
        """

        n_plots = len(combinations)

        # Calculate grid layout
        cols = min(2, n_plots)
        rows = int(np.ceil(n_plots / cols))

        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        metric_label = ModelSelectionPlotter._get_metric_label(
            metric or selector.selection_metric
        )

        # Build title with optional ship type
        if ship_type:
            fig_title = f'{ship_type}: Model Selection Comparison ({metric_label})'
        else:
            fig_title = f'Model Selection Comparison ({metric_label})'

        fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.998)

        for idx, (pred_combo, target) in enumerate(combinations):
            ax = axes[idx]

            # Use existing single-plot logic but on this axes
            ModelSelectionPlotter._plot_on_axes(
                ax, selector, pred_combo, target, metric,
                highlight_selected, show_values, variable_name_map,
                show_heteroscedasticity  # NEW argument
            )

            # Save individual if template provided
            if save_path_template:
                save_path = save_path_template.format(i=idx)
                # Save just this subplot
                extent = ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                fig.savefig(save_path, dpi=300, bbox_inches=extent.expanded(1.2, 1.2))

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_on_axes(ax, selector, predictor_combo, target, metric,
                      highlight_selected, show_values, variable_name_map=None,
                      show_heteroscedasticity=True):  # NEW parameter
        """Helper to plot on specific axes (for subplots)."""

        key = (tuple(predictor_combo), target)

        if key not in selector.detailed_results:
            ax.text(0.5, 0.5, 'No results',
                    ha='center', va='center', transform=ax.transAxes)
            return

        model_results = selector.detailed_results[key]

        if metric is None:
            metric = selector.selection_metric

        selected_model = selector.model_recommendations.get(key)

        # Prepare data
        data = []
        for model_name, metrics in model_results.items():
            metric_value = metrics.get(metric)
            if metric_value is None or np.isnan(metric_value):
                continue

            # NEW: Extract heteroscedasticity
            het_status = metrics.get('heteroscedasticity', 'unknown')

            data.append({
                'model': model_name,
                'value': metric_value,
                'is_selected': (model_name == selected_model),
                'het_status': het_status  # NEW
            })

        if not data:
            ax.text(0.5, 0.5, f'No {metric} values',
                    ha='center', va='center', transform=ax.transAxes)
            return

        lower_is_better = metric in ['bic', 'aic', 'rrmse', 'mape', 'rmse', 'mae']
        data.sort(key=lambda x: x['value'], reverse=not lower_is_better)

        # Plot
        models = [d['model'] for d in data]
        values = [d['value'] for d in data]
        is_selected = [d['is_selected'] for d in data]
        het_statuses = [d['het_status'] for d in data]  # NEW

        if highlight_selected:
            colors = ['#2ecc71' if selected else '#3498db'
                      for selected in is_selected]
        else:
            colors = '#3498db'

        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, values, color=colors,
                       edgecolor='black', linewidth=1, alpha=0.8)

        if show_values:
            for bar, value, selected in zip(bars, values, is_selected):
                x_pos = bar.get_width()

                if metric in ['adj_r2', 'r2']:
                    label = f'{value:.3f}'
                elif metric in ['rrmse', 'mape']:
                    label = f'{value:.1f}%'
                elif metric in ['aic', 'bic']:
                    label = f'{value:.0f}'
                else:
                    label = f'{value:.2f}'

                if selected and highlight_selected:
                    label = f'{label} ←'

                ax.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                        f' {label}',
                        va='center', ha='left', fontsize=8,
                        fontweight='bold' if selected else 'normal')

        # Format - NEW: Add heteroscedasticity to labels
        if show_heteroscedasticity:
            model_labels = []
            for model, het_status in zip(models, het_statuses):
                if het_status == 'No':
                    indicator = '✓'
                elif het_status in ['Strong', 'Weak']:
                    indicator = '⚠'
                else:
                    indicator = '?'
                model_labels.append(f"{model} {indicator}")
        else:
            model_labels = models

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_labels, fontsize=8)

        # Format variable names for title
        if variable_name_map:
            predictors_str = ' + '.join([
                variable_name_map.get(p, p) for p in predictor_combo
            ])
            target_str = variable_name_map.get(target, target)
        else:
            predictors_str = ' + '.join(predictor_combo)
            target_str = target

        ax.set_title(f'{predictors_str} → {target_str}\n{selected_model}',
                     fontsize=10, fontweight='bold')

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
