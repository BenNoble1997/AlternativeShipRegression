"""
Model Selection Comparison Plotter
===================================

Visualizes the model selection process by showing all tested models
and their performance metrics for a specific predictor-target combination.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Mapping, Sequence, Any
from pathlib import Path
from dataclasses import dataclass

# =============================================================================
# Class attributes: add at the top of ModelSelectionPlotter
# =============================================================================




@dataclass(frozen=True)
class ModelDisplaySelection:
    """
    Declarative description of which fitted models to display and under what
    names.

    This object is deliberately ignorant of metrics, figures and axes. It maps
    a set of *available* raw model names onto an ordered subset of raw names,
    and provides a raw-name-to-label lookup. Keeping the two concerns (which
    models, what labels) inside one small value object means a plotting call
    site needs a single argument rather than several loosely-related ones, and
    the same specification can be reused across several figures or a results
    table.

    :param include: Raw model names to display. ``None`` means "all available
        models". Names not present in the results are handled according to
        :attr:`strict`.
    :type include: Sequence[str] | None
    :param exclude: Raw model names to drop. Applied *after* ``include``, so a
        name appearing in both is dropped.
    :type exclude: Sequence[str] | None
    :param name_map: Mapping from raw model name to display label, e.g.
        ``{"linear": "M1", "power_offset": "M2"}``. Any model absent from the
        mapping keeps its raw name.
    :type name_map: Mapping[str, str] | None
    :param preserve_include_order: If ``True`` and ``include`` is given, the
        resolved order follows ``include`` rather than the metric ordering
        applied later by the caller. Useful when the labels encode an intended
        sequence (M1, M2, M3 ...) that should not be permuted by performance.
    :type preserve_include_order: bool
    :param strict: If ``True``, requesting or excluding an unknown model name
        raises :class:`ValueError`. If ``False``, unknown names are silently
        ignored, which is convenient when one specification is shared across
        ship types whose candidate model sets differ.
    :type strict: bool
    """

    include: Sequence[str] | None = None
    exclude: Sequence[str] | None = None
    name_map: Mapping[str, str] | None = None
    preserve_include_order: bool = False
    strict: bool = False

    def resolve(self, available: Sequence[str], debug: bool = False) -> list[str]:
        """
        Reduce the available raw model names to the ordered subset to display.

        :param available: Raw model names present in the results dictionary.
        :type available: Sequence[str]
        :param debug: Emit a trace of the filtering decisions.
        :type debug: bool
        :returns: Raw model names to plot, in resolution order.
        :rtype: list[str]
        :raises ValueError: If :attr:`strict` is set and an unknown name is
            requested or excluded, or if the resolved set is empty.
        """
        available_set = set(available)

        if debug:
            print(f"[ModelDisplaySelection] available: {sorted(available_set)}")

        if self.include is None:
            kept = list(available)
        else:
            unknown = [m for m in self.include if m not in available_set]
            if unknown:
                msg = f"Unknown model name(s) in 'include': {unknown}"
                if self.strict:
                    raise ValueError(f"{msg}. Available: {sorted(available_set)}")
                if debug:
                    print(f"[ModelDisplaySelection] ignoring {msg}")

            if self.preserve_include_order:
                kept = [m for m in self.include if m in available_set]
            else:
                requested = set(self.include)
                kept = [m for m in available if m in requested]

        if self.exclude:
            unknown = [m for m in self.exclude if m not in available_set]
            if unknown and self.strict:
                raise ValueError(
                    f"Unknown model name(s) in 'exclude': {unknown}. "
                    f"Available: {sorted(available_set)}"
                )
            dropped = set(self.exclude)
            kept = [m for m in kept if m not in dropped]

        if not kept:
            raise ValueError(
                "Model display selection removed every model. "
                f"Available: {sorted(available_set)}"
            )

        if debug:
            print(f"[ModelDisplaySelection] kept ({len(kept)}): {kept}")
            print(f"[ModelDisplaySelection] preserve_include_order="
                  f"{self.preserve_include_order}")

        return kept

    def display_name(self, raw_name: str) -> str:
        """
        Return the label for a raw model name.

        :param raw_name: Raw model name as used as a key in the results.
        :type raw_name: str
        :returns: Mapped label, or ``raw_name`` if no mapping is defined.
        :rtype: str
        """
        if not self.name_map:
            return raw_name
        return self.name_map.get(raw_name, raw_name)

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

    #: Axes-fraction anchor (x, y, ha, va) for each corner. Used for ax.text boxes,
    #: which unlike a legend have no ``loc`` keyword.
    _CORNER_ANCHORS: Dict[str, Tuple[float, float, str, str]] = {
        'upper left': (0.02, 0.98, 'left', 'top'),
        'upper right': (0.98, 0.98, 'right', 'top'),
        'lower left': (0.02, 0.02, 'left', 'bottom'),
        'lower right': (0.98, 0.02, 'right', 'bottom'),
    }

    #: Automatic note-box corner for each legend corner. Bars in a horizontal bar
    #: chart grow from the left, so the left-hand corners only ever cover bar bases,
    #: while the value labels sit at the right-hand bar ends. The note box is
    #: therefore kept on the left whenever the legend does not already occupy it.
    _AUTO_INFO_CORNER: Dict[str, str] = {
        'upper left': 'lower left',
        'lower left': 'upper left',
        'upper right': 'upper left',
        'lower right': 'upper left',
    }

    #: All accepted values of ``legend_loc``.
    _LEGEND_LOCATIONS: Tuple[str, ...] = (
        *_CORNER_ANCHORS.keys(), 'best', 'outside right',
    )

    @staticmethod
    def plot_model_selection_comparison(
            selector,
            predictor_combo: Tuple[str, ...],
            target: str,
            metric: Optional[str] = None,
            ship_type: Optional[str] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            model_selection: Optional["ModelDisplaySelection"] = None,
            highlight_selected: bool = True,
            show_values: bool = True,
            show_heteroscedasticity: bool = True,
            legend_loc: str = 'upper left',
            info_box_loc: Optional[str] = None,
            figsize: Tuple[float, float] = (10, 6),
            save_path: str | Path | None = None,
            show_plot: bool = False,
            debug: bool = False,
    ):
        """
        Plot a comparison of the candidate models tested for one predictor
        combination and target.

        Which models appear, and under what names, is governed entirely by
        ``model_selection``. Raw model names remain the lookup keys into the
        selector's results throughout; display labels are applied only at
        render time, so renaming a model cannot desynchronise the figure from
        the underlying metrics.

        :param selector: Fitted selector exposing ``detailed_results``,
            ``model_recommendations`` and ``selection_metric``.
        :param predictor_combo: Predictor names identifying the combination.
        :type predictor_combo: Tuple[str, ...]
        :param target: Target variable name.
        :type target: str
        :param metric: Metric to compare on. Defaults to
            ``selector.selection_metric``.
        :type metric: str | None
        :param ship_type: Optional ship type used as a title prefix.
        :type ship_type: str | None
        :param variable_name_map: Mapping from raw predictor/target names to
            display names, used in the title only.
        :type variable_name_map: Dict[str, str] | None
        :param model_selection: Specification of which models to plot and how
            to label them. ``None`` plots every model under its raw name.
        :type model_selection: ModelDisplaySelection | None
        :param highlight_selected: Colour the selector's chosen model
            differently from the remainder.
        :type highlight_selected: bool
        :param show_values: Annotate each bar with its metric value.
        :type show_values: bool
        :param show_heteroscedasticity: Append a heteroscedasticity indicator
            to each y-tick label.
        :type show_heteroscedasticity: bool
        :param legend_loc: Position of the legend. One of ``'upper left'``
            (default), ``'upper right'``, ``'lower left'``, ``'lower right'``,
            ``'best'`` (matplotlib chooses the least crowded location) or
            ``'outside right'`` (outside the axes, never covers bars).
        :type legend_loc: str
        :param info_box_loc: Corner for the Delta-AIC/BIC note box. ``None``
            places it in a free left-hand corner.
        :type info_box_loc: str | None
        :param figsize: Figure size in inches.
        :type figsize: Tuple[float, float]
        :param save_path: If given, the figure is written here at 300 dpi.
        :type save_path: str | Path | None
        :param show_plot: Call ``plt.show()`` after rendering.
        :type show_plot: bool
        :param debug: Emit stage-by-stage diagnostics describing filtering,
            label resolution, sorting, annotation and placement decisions.
        :type debug: bool
        :returns: The figure and axes produced.
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        :raises ValueError: If no results exist for the combination, if the
            selection removes every model, if no finite metric values remain,
            or if ``legend_loc`` / ``info_box_loc`` is not supported.
        """

        # ====================================================================
        # 0. Validate placement options early (fail before any computation)
        # ====================================================================

        legend_kwargs = ModelSelectionPlotter._legend_kwargs(legend_loc)
        info_corner = ModelSelectionPlotter._resolve_info_box_corner(
            legend_loc, info_box_loc
        )

        if debug:
            print(f"[plot] legend_loc='{legend_loc}' -> legend kwargs {legend_kwargs}")
            print(f"[plot] info box corner='{info_corner}' "
                  f"({'explicit' if info_box_loc else 'auto'})")

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
            raise ValueError("No model results available for this combination")

        if metric is None:
            metric = selector.selection_metric

        selected_model = selector.model_recommendations.get(key)

        if debug:
            print(f"[plot] key={key}")
            print(f"[plot] metric='{metric}', selected_model='{selected_model}'")
            print(f"[plot] models in results: {list(model_results.keys())}")

        # ====================================================================
        # 2. Apply display selection (filter + rename)
        # ====================================================================

        if model_selection is None:
            model_selection = ModelDisplaySelection()

        models_to_plot = model_selection.resolve(
            list(model_results.keys()), debug=debug
        )

        if selected_model is not None and selected_model not in models_to_plot and debug:
            print(f"[plot] NOTE: selected model '{selected_model}' has been "
                  f"filtered out of the figure")

        # ====================================================================
        # 3. Prepare Data
        # ====================================================================

        data = []
        for model_name in models_to_plot:
            metrics = model_results[model_name]
            metric_value = metrics.get(metric)

            if metric_value is None or not np.isfinite(metric_value):
                if debug:
                    print(f"[plot] skipping '{model_name}': "
                          f"{metric}={metric_value!r} not finite")
                continue

            data.append({
                'raw': model_name,
                'label': model_selection.display_name(model_name),
                'value': float(metric_value),
                'is_selected': (model_name == selected_model),
                'het_status': metrics.get('heteroscedasticity', 'unknown'),
            })

        if not data:
            raise ValueError(
                f"No valid {metric} values found for the selected models"
            )

        lower_is_better = metric in ['bic', 'aic', 'rrmse', 'mape', 'rmse', 'mae']

        if not model_selection.preserve_include_order:
            data.sort(key=lambda x: x['value'], reverse=not lower_is_better)
        elif debug:
            print("[plot] preserve_include_order=True → metric sorting skipped")

        if debug:
            print(f"[plot] lower_is_better={lower_is_better}")
            for d in data:
                print(f"[plot]   {d['raw']:<20s} -> {d['label']:<8s} "
                      f"{metric}={d['value']:.4g} het={d['het_status']} "
                      f"selected={d['is_selected']}")

        # ====================================================================
        # 4. Create Plot
        # ====================================================================

        fig, ax = plt.subplots(figsize=figsize)

        raw_names = [d['raw'] for d in data]
        labels = [d['label'] for d in data]
        values = [d['value'] for d in data]
        is_selected = [d['is_selected'] for d in data]
        het_statuses = [d['het_status'] for d in data]

        if highlight_selected:
            colors = ['#2ecc71' if sel else '#3498db' for sel in is_selected]
        else:
            colors = '#3498db'

        y_pos = np.arange(len(raw_names))
        bars = ax.barh(y_pos, values, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        if show_values:
            for bar, value, sel in zip(bars, values, is_selected):
                x_pos = bar.get_width()

                if metric in ['adj_r2', 'r2']:
                    label = f'{value:.4f}'
                elif metric in ['rrmse', 'mape']:
                    label = f'{value:.2f}%'
                elif metric in ['aic', 'bic']:
                    label = f'{value:.1f}'
                else:
                    label = f'{value:.3f}'

                if sel and highlight_selected:
                    label = f'{label} ←'

                ax.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                        f' {label}',
                        va='center', ha='left', fontsize=10,
                        fontweight='bold' if sel else 'normal')

        # ====================================================================
        # 5. Format Plot
        # ====================================================================

        if variable_name_map:
            predictors_str = ' + '.join(
                [variable_name_map.get(p, p) for p in predictor_combo]
            )
            target_str = variable_name_map.get(target, target)
        else:
            predictors_str = ' + '.join(predictor_combo)
            target_str = target

        if show_heteroscedasticity:
            tick_labels = []
            for label, het_status in zip(labels, het_statuses):
                if het_status == 'No':
                    indicator = '✓'
                elif het_status in ['Strong', 'Weak']:
                    indicator = '⚠'
                else:
                    indicator = '?'
                tick_labels.append(f"{label} {indicator}")
        else:
            tick_labels = labels

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.set_ylabel('Model Type', fontsize=11, fontweight='bold')

        metric_label = ModelSelectionPlotter._get_metric_label(metric)
        ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')

        n_models = len(raw_names)
        n_available = len(model_results)
        title_prefix = f'{ship_type}: ' if ship_type else ''

        if selected_model in raw_names:
            selected_label = model_selection.display_name(selected_model)
            title = (f'{title_prefix}Model Selection: {predictors_str} → {target_str}\n'
                     f'Selected: {selected_label} ')
        else:
            title = (f'{title_prefix}Model Comparison: {predictors_str} → {target_str}\n'
                     f'({n_models} of {n_available} models shown)')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        best_value = min(values) if lower_is_better else max(values)
        ax.axvline(best_value, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label='Best value')

        if debug:
            print(f"[plot] best displayed {metric}={best_value:.4g}")

        # ====================================================================
        # 6. Legend
        # ====================================================================

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = []

        if highlight_selected and selected_model in raw_names:
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
            ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9,
                      **legend_kwargs)
            if debug:
                print(f"[plot] legend drawn with {len(legend_elements)} entries")
        elif debug:
            print("[plot] no legend entries, legend skipped")

        # ====================================================================
        # 7. Diagnostic Info (Delta-AIC/BIC note box)
        # ====================================================================

        if metric in ['bic', 'aic'] and selected_model in raw_names:
            selected_value = model_results[selected_model][metric]
            threshold = 2.0

            equivalent_raw = [
                m for m in raw_names
                if abs(model_results[m][metric] - selected_value) < threshold
            ]

            if len(equivalent_raw) > 1:
                equiv_str = ', '.join(
                    model_selection.display_name(m) for m in equivalent_raw
                )
                info_text = (f'Note: {len(equivalent_raw)} displayed models within '
                             f'Δ{metric.upper()} < 2.0:\n{equiv_str}')

                x, y, ha, va = ModelSelectionPlotter._CORNER_ANCHORS[info_corner]
                ax.text(x, y, info_text,
                        transform=ax.transAxes,
                        fontsize=8, va=va, ha=ha,
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='yellow', alpha=0.3))

                if debug:
                    print(f"[plot] Δ{metric} < 2.0 group (raw): {equivalent_raw}")
                    print(f"[plot] note box placed at '{info_corner}' "
                          f"(x={x}, y={y}, ha={ha}, va={va})")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if debug or getattr(selector, 'verbose', False):
                print(f"✅ Plot saved to {save_path}")

        if show_plot:
            plt.show()

        return fig, ax

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
            save_path_template: str|Path|None=None,
            show_plot: bool = False,
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

        if show_plot:

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

    @staticmethod
    def _legend_kwargs(legend_loc: str) -> Dict[str, Any]:
        """
        Translate a ``legend_loc`` option into keyword arguments for
        :meth:`matplotlib.axes.Axes.legend`.

        Corner names and ``'best'`` are passed straight through as ``loc``.
        ``'outside right'`` anchors the legend's upper-left corner just beyond the
        right-hand edge of the axes, so it never covers a bar; the figure is then
        widened by ``bbox_inches='tight'`` when saving.

        :param legend_loc: One of ``'upper left'``, ``'upper right'``,
            ``'lower left'``, ``'lower right'``, ``'best'`` or ``'outside right'``.
        :type legend_loc: str
        :returns: Keyword arguments for ``ax.legend``.
        :rtype: dict[str, Any]
        :raises ValueError: If ``legend_loc`` is not a supported option.
        """
        if legend_loc not in ModelSelectionPlotter._LEGEND_LOCATIONS:
            raise ValueError(
                f"legend_loc='{legend_loc}' not supported. "
                f"Choose from {list(ModelSelectionPlotter._LEGEND_LOCATIONS)}"
            )

        if legend_loc == 'outside right':
            return {'loc': 'upper left', 'bbox_to_anchor': (1.02, 1.0),
                    'borderaxespad': 0.0}

        return {'loc': legend_loc}

    @staticmethod
    def _resolve_info_box_corner(legend_loc: str,
                                 info_box_loc: Optional[str]) -> str:
        """
        Decide which corner the Delta-AIC/BIC note box occupies.

        If ``info_box_loc`` is given it is used as-is. Otherwise the box goes to a
        left-hand corner not occupied by the legend (see ``_AUTO_INFO_CORNER``).
        When the legend is not in a corner (``'best'`` or ``'outside right'``) the
        box defaults to the upper left, which matches the original behaviour.

        :param legend_loc: Resolved legend location (see :meth:`_legend_kwargs`).
        :type legend_loc: str
        :param info_box_loc: Explicit corner for the note box, or ``None`` for
            automatic placement.
        :type info_box_loc: str | None
        :returns: Corner name, a key of ``_CORNER_ANCHORS``.
        :rtype: str
        :raises ValueError: If ``info_box_loc`` is not a corner name.
        """
        corners = ModelSelectionPlotter._CORNER_ANCHORS

        if info_box_loc is not None:
            if info_box_loc not in corners:
                raise ValueError(
                    f"info_box_loc='{info_box_loc}' not supported. "
                    f"Choose from {list(corners)}"
                )
            if info_box_loc == legend_loc:
                print(
                    f"info_box_loc and legend_loc are both '{legend_loc}'; "
                    f"the note box and legend will overlap."
                )
            return info_box_loc

        return ModelSelectionPlotter._AUTO_INFO_CORNER.get(legend_loc, 'upper left')


