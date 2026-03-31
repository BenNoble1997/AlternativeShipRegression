"""
Predictor Combination Comparison Plotter
=========================================

Visualizes performance across different predictor combinations for a single target.
Useful for Random Forest or any selector that tests multiple predictor combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict


class PredictorCombinationPlotter:
    """
    Visualizes predictor combination selection decisions.

    **Concept:**
    Shows why a specific predictor combination was selected by comparing all
    tested combinations on the chosen selection metric. Helps understand:
    - Which predictor combinations perform best
    - Whether certain predictors consistently improve performance
    - If adding more predictors helps or hurts
    - Impact of feature engineering

    **Key Difference from ModelSelectionPlotter:**
    - ModelSelectionPlotter: Multiple models, single predictor combination
    - PredictorCombinationPlotter: Single model, multiple predictor combinations
    """

    @staticmethod
    def plot_combination_comparison(
            selector,
            target: str,
            metric: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            highlight_best: bool = True,
            show_values: bool = True,
            show_vif: bool = True,
            max_predictors: Optional[int] = None,
            filter_feature_engineered: Optional[bool] = None,
            sort_by_n_predictors: bool = True,
            figsize: Tuple[float, float] = (12, 8),
            save_path: Optional[str] = None
    ):
        """
        Plot comparison of all predictor combinations tested for a specific target.

        **Use Case:**
        You want to see which predictor combination performs best for predicting LOA.
        This plot shows all tested combinations (DWT alone, GT alone, DWT+GT, etc.)
        and their performance metrics.

        Parameters
        ----------
        selector : RandomForestSelector
            Fitted selector with results
        target : str
            Target variable to analyze
        metric : str, optional
            Metric to display. If None, uses selector's selection_metric.
            Options: 'adj_r2', 'rrmse', 'mape', 'r2'
        ship_type : str, optional
            Ship type to include in title, e.g., 'Container Ships'
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        highlight_best : bool, default=True
            Highlight the best-performing combination in different color
        show_values : bool, default=True
            Show metric values beside/above bars
        show_vif : bool, default=True
            Show VIF indicators for combinations with multiple predictors
        max_predictors : int, optional
            Limit to combinations with at most this many predictors
        filter_feature_engineered : bool, optional
            Filter to only raw (False) or engineered (True) features.
            If None, shows both.
        sort_by_n_predictors : bool, default=True
            Group combinations by number of predictors
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, optional
            Path to save figure

        Examples
        --------
        >>> # Basic usage
        >>> selector.plot_combination_comparison(target='LOA (M)')

        >>> # With ship type and formatted names
        >>> variable_map = {
        ...     'DWT': 'DWT (t)',
        ...     'LOA (M)': 'L_OA (m)',
        ...     'GT': 'GT (m³)',
        ...     'Beam (M)': 'B (m)'
        ... }
        >>> selector.plot_combination_comparison(
        ...     target='LOA (M)',
        ...     ship_type='Container Ships',
        ...     variable_name_map=variable_map,
        ...     max_predictors=2
        ... )

        >>> # Compare only raw features
        >>> selector.plot_combination_comparison(
        ...     target='LOA (M)',
        ...     filter_feature_engineered=False,
        ...     metric='rrmse'
        ... )
        """

        # ====================================================================
        # 1. Get Results for Target
        # ====================================================================

        if target not in selector.results:
            raise ValueError(
                f"No results found for target '{target}'. "
                f"Available targets: {list(selector.results.keys())}"
            )

        combinations = selector.results[target]

        if not combinations:
            raise ValueError(f"No combinations available for target '{target}'")

        # Determine metric
        if metric is None:
            metric = selector.selection_metric

        # ====================================================================
        # 2. Prepare and Filter Data
        # ====================================================================

        data = []
        for combo_name, combo_data in combinations.items():
            metrics = combo_data['metrics']
            predictors = combo_data['predictors']
            is_feature_eng = combo_data.get('feature_engineered', False)

            # Apply filters
            if max_predictors is not None and len(predictors) > max_predictors:
                continue

            if filter_feature_engineered is not None:
                if is_feature_eng != filter_feature_engineered:
                    continue

            metric_value = metrics.get(metric)
            if metric_value is None or np.isnan(metric_value):
                continue

            # Format predictor names
            if variable_name_map:
                formatted_predictors = [
                    variable_name_map.get(p, p) for p in predictors
                ]
            else:
                formatted_predictors = list(predictors)

            # Create label
            predictors_str = ' + '.join(formatted_predictors)
            if is_feature_eng:
                label = f"{predictors_str} (FE)"
            else:
                label = predictors_str

            data.append({
                'label': label,
                'value': metric_value,
                'n_predictors': len(predictors),
                'is_feature_eng': is_feature_eng,
                'max_vif': metrics.get('max_vif', 1.0),
                'combo_name': combo_name,
                'std_value': metrics.get(f'std_{metric}', 0)
            })

        if not data:
            raise ValueError(
                f"No valid combinations found for target '{target}' "
                f"after applying filters"
            )

        # ====================================================================
        # 3. Sort Data
        # ====================================================================

        lower_is_better = metric in ['rrmse', 'mape', 'rmse', 'mae']

        if sort_by_n_predictors:
            # Sort by: (1) n_predictors, (2) metric value
            data.sort(
                key=lambda x: (
                    x['n_predictors'],
                    x['value'] if lower_is_better else -x['value']
                )
            )
        else:
            # Sort by metric value only
            data.sort(key=lambda x: x['value'], reverse=not lower_is_better)

        # ====================================================================
        # 4. Find Best Combination
        # ====================================================================

        if lower_is_better:
            best_idx = min(range(len(data)), key=lambda i: data[i]['value'])
        else:
            best_idx = max(range(len(data)), key=lambda i: data[i]['value'])

        best_combo = data[best_idx]['combo_name']

        # Mark best in data
        for i, d in enumerate(data):
            d['is_best'] = (i == best_idx)

        # ====================================================================
        # 5. Create Plot
        # ====================================================================

        fig, ax = plt.subplots(figsize=figsize)

        # Extract data for plotting
        labels = [d['label'] for d in data]
        values = [d['value'] for d in data]
        is_best = [d['is_best'] for d in data]
        vif_values = [d['max_vif'] for d in data]

        # Determine colors
        if highlight_best:
            colors = ['#2ecc71' if best else '#3498db' for best in is_best]
        else:
            colors = '#3498db'

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels
        if show_values:
            for bar, value, best, std in zip(bars, values, is_best,
                                             [d['std_value'] for d in data]):
                x_pos = bar.get_width()

                # Format value
                if metric in ['adj_r2', 'r2']:
                    label = f'{value:.4f}'
                    if std > 0:
                        label += f' ±{std:.4f}'
                elif metric in ['rrmse', 'mape']:
                    label = f'{value:.2f}%'
                    if std > 0:
                        label += f' ±{std:.2f}%'
                else:
                    label = f'{value:.3f}'
                    if std > 0:
                        label += f' ±{std:.3f}'

                # Add best marker
                if best and highlight_best:
                    label = f'{label} ★'

                ax.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                        f' {label}',
                        va='center', ha='left', fontsize=9,
                        fontweight='bold' if best else 'normal')

        # ====================================================================
        # 6. Format Plot with VIF Indicators
        # ====================================================================

        # Y-axis (combinations) - Add VIF warnings
        if show_vif:
            y_labels = []
            for label, vif in zip(labels, vif_values):
                if vif > 10:
                    y_labels.append(f"{label} ⚠ VIF={vif:.1f}")
                elif vif > 5:
                    y_labels.append(f"{label} ⚡ VIF={vif:.1f}")
                else:
                    y_labels.append(label)
        else:
            y_labels = labels

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_ylabel('Predictor Combination', fontsize=11, fontweight='bold')

        # X-axis (metric)
        metric_label = PredictorCombinationPlotter._get_metric_label(metric)
        ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')

        # Format target name
        if variable_name_map:
            target_str = variable_name_map.get(target, target)
        else:
            target_str = target

        # Title
        n_combos = len(data)

        if ship_type:
            title_prefix = f'{ship_type}: '
        else:
            title_prefix = ''

        filter_desc = ""
        if filter_feature_engineered is True:
            filter_desc = " (Feature Engineered)"
        elif filter_feature_engineered is False:
            filter_desc = " (Raw Features)"

        title = (f'{title_prefix}Predictor Combination Comparison: {target_str}\n'
                 f'Best: {best_combo} ({n_combos} combinations tested{filter_desc})')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add vertical line at best value
        best_value = values[best_idx]
        ax.axvline(best_value, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label='Best value')

        # ====================================================================
        # 7. Add Legend
        # ====================================================================

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []

        if highlight_best:
            legend_elements.extend([
                Patch(facecolor='#2ecc71', edgecolor='black',
                      label='Best combination'),
                Patch(facecolor='#3498db', edgecolor='black',
                      label='Other combinations')
            ])

        if show_vif:
            legend_elements.extend([
                Line2D([0], [0], marker='$⚠$', color='w',
                       markerfacecolor='red', markersize=10,
                       label='VIF > 10 (High multicollinearity)'),
                Line2D([0], [0], marker='$⚡$', color='w',
                       markerfacecolor='orange', markersize=10,
                       label='VIF > 5 (Moderate)')
            ])

        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right',
                      fontsize=9, framealpha=0.9)

        # ====================================================================
        # 8. Add Summary Statistics Box
        # ====================================================================

        # Calculate summary stats
        n_single = sum(1 for d in data if d['n_predictors'] == 1)
        n_multi = sum(1 for d in data if d['n_predictors'] > 1)
        n_feat_eng = sum(1 for d in data if d['is_feature_eng'])
        best_n_preds = data[best_idx]['n_predictors']

        summary_text = (
            f"Summary:\n"
            f"  Single predictors: {n_single}\n"
            f"  Multi predictors: {n_multi}\n"
            f"  Feature engineered: {n_feat_eng}\n"
            f"  Best uses {best_n_preds} predictor(s)"
        )

        ax.text(0.02, 0.98, summary_text,
                transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='lightyellow', alpha=0.7),
                family='monospace')

        plt.tight_layout()

        # Save if requested
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
            'rrmse': 'RRMSE % (lower is better)',
            'mape': 'MAPE % (lower is better)',
            'rmse': 'RMSE (lower is better)',
            'mae': 'MAE (lower is better)'
        }
        return labels.get(metric, metric)

    @staticmethod
    def plot_multiple_targets(
            selector,
            targets: List[str],
            metric: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            highlight_best: bool = True,
            show_values: bool = True,
            show_vif: bool = True,
            max_predictors: Optional[int] = None,
            filter_feature_engineered: Optional[bool] = None,
            figsize_per_plot: Tuple[float, float] = (12, 6),
            save_path_template: Optional[str] = None
    ):
        """
        Plot combination comparisons for multiple targets.

        Parameters
        ----------
        selector : RandomForestSelector
            Fitted selector
        targets : list of str
            List of targets to plot
        metric : str, optional
            Metric to display
        ship_type : str, optional
            Ship type for titles
        variable_name_map : dict, optional
            Variable name formatting
        highlight_best : bool, default=True
            Highlight best combinations
        show_values : bool, default=True
            Show values on bars
        show_vif : bool, default=True
            Show VIF indicators
        max_predictors : int, optional
            Max predictors to include
        filter_feature_engineered : bool, optional
            Filter by feature engineering
        figsize_per_plot : tuple, default=(12, 6)
            Size per subplot
        save_path_template : str, optional
            Template for save paths. Use {target} as placeholder.
            Example: 'plots/combinations_{target}.png'

        Examples
        --------
        >>> # Compare combinations for multiple targets
        >>> selector.plot_multiple_targets(
        ...     targets=['LOA (M)', 'Beam (M)', 'Depth (M)'],
        ...     variable_name_map=variable_map,
        ...     save_path_template='plots/rf_combos_{target}.png'
        ... )
        """

        for target in targets:
            if save_path_template:
                save_path = save_path_template.format(
                    target=target.replace(' ', '_').replace('(', '').replace(')', '')
                )
            else:
                save_path = None

            try:
                PredictorCombinationPlotter.plot_combination_comparison(
                    selector=selector,
                    target=target,
                    metric=metric,
                    ship_type=ship_type,
                    variable_name_map=variable_name_map,
                    highlight_best=highlight_best,
                    show_values=show_values,
                    show_vif=show_vif,
                    max_predictors=max_predictors,
                    filter_feature_engineered=filter_feature_engineered,
                    figsize=figsize_per_plot,
                    save_path=save_path
                )
            except ValueError as e:
                print(f"⚠️  Skipped {target}: {e}")