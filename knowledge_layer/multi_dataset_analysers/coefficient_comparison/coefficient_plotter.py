"""
Coefficient Change Visualization (UPDATED)
==========================================

Creates detailed plots showing how coefficients change across datasets
with flexible display options.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
import pandas as pd


class CoefficientChangePlotter:
    """
    Handles visualization of coefficient changes across datasets.

    Creates plots showing:
    - Distribution Effect (Full → Matched) - optional
    - Fuel Effect (Matched → Alternative) - optional
    For each target variable's best model.
    """

    # ── Unicode subscript digit map ───────────────────────────────────────────────
    # Maps integer indices to β labels with proper Unicode subscripts.
    # Handles multi-digit indices (e.g. 10 → β₁₀) by translating digit by digit.


    def __init__(self, analyzer):
        """
        Initialize plotter with analyzer instance.

        Parameters
        ----------
        analyzer : CoefficientComparisonAnalyzer
            Analyzer with comparison results
        """
        self.analyzer = analyzer

    def plot_coefficient_changes_all_targets(
            self,
            figsize_per_plot: Tuple[float, float] = (8, 5),
            save_path: Optional[str] = None,
            show_values: bool = True,
            min_change_threshold: float = 0.01,
            show_distribution: bool = True,
            show_fuel: bool = True,
            n_cols: int = 2,
            bar_height: float = 0.25,
            bar_spacing: float = 0.5,
            title_prefix: str = '',
            title_suffix: str = '',
            min_r2: Optional[float] = None,
            filter_dataset: str = 'alternative',
            min_n_samples: Optional[int] = None,
            variable_name_map: Optional[Dict[str, str]] = None,
            formula_name_map: Optional[Dict[str, str]] = None,
            show_sign_change_legend: bool = False,
            title_bold_spacing: float = 0.030,
            title_italic_spacing: float = 0.022,
            title_gap_below: float = 0.015,
            debug: bool = False,
    ):
        """
        Create coefficient change plots for all targets in a grid layout.

        Each subplot shows how a target's model coefficients shift across the
        three datasets (Full Conventional → Matched Conventional → Alternative),
        decomposed into a distribution effect and a fuel effect.

        :param figsize_per_plot: ``(width, height)`` in inches for each
            individual subplot cell.
        :type figsize_per_plot: tuple[float, float]

        :param save_path: File path at which to save the combined figure.
        :type save_path: str, optional

        :param show_values: Print numeric change values on bars.
        :type show_values: bool

        :param min_change_threshold: Minimum absolute coefficient change for
            a term to appear in the bar chart.
        :type min_change_threshold: float

        :param show_distribution: Include bars for the distribution effect
            (Full → Matched).
        :type show_distribution: bool

        :param show_fuel: Include bars for the fuel effect
            (Matched → Alternative).
        :type show_fuel: bool

        :param n_cols: Number of subplot columns.  Clamped to the number of
            retained targets when fewer targets pass the performance filter.
        :type n_cols: int

        :param bar_height: Height of each bar in data-axis units.
        :type bar_height: float

        :param bar_spacing: Vertical spacing between coefficient groups.
        :type bar_spacing: float

        :param title_prefix: Bold text placed as the first title line (above
            the main title).  Leave empty to omit.
        :type title_prefix: str

        :param title_suffix: Italic text placed as the last title line.
            Leave empty to omit.
        :type title_suffix: str

        :param min_r2: Minimum R² threshold for retaining a target.
        :type min_r2: float, optional

        :param filter_dataset: Which dataset's R² is compared against
            ``min_r2``.  One of ``'alternative'``, ``'conventional_full'``,
            ``'conventional_matched'``, ``'all'``, ``'any'``.
        :type filter_dataset: str

        :param min_n_samples: Minimum number of alternative-fuel ships
            required to retain a target.
        :type min_n_samples: int, optional

        :param variable_name_map: Mapping from internal variable names to
            display strings used in the **subplot title**.  The caller
            controls all formatting including units and Unicode.
            Example: ``{'BEAM (M)': 'Beam (m)', 'DWT': 'DWT (t)'}``.
        :type variable_name_map: dict[str, str], optional

        :param formula_name_map: Mapping from internal variable names to
            display strings used **inside the formula** shown below each
            subplot title.  Typically more compact than ``variable_name_map``.
            When ``None``, ``variable_name_map`` is used as the fallback.
            Example: ``{'BEAM (M)': 'B', 'DWT': 'DWT'}``.
        :type formula_name_map: dict[str, str], optional

        :param show_sign_change_legend: Add a legend explaining the green /
            red sign-change colour coding.
        :type show_sign_change_legend: bool

        :param title_bold_spacing: Vertical gap (in figure-fraction units)
            consumed by each **bold** title line.  Increase to open up space
            between bold lines and the element below them.  Default ``0.030``.
        :type title_bold_spacing: float

        :param title_italic_spacing: Vertical gap consumed by each **italic**
            title line.  Default ``0.022``.
        :type title_italic_spacing: float

        :param title_gap_below: Additional blank space between the bottom of
            the title block and the first subplot row, in figure-fraction
            units.  Increase when the subplots still feel cramped against the
            title.  Default ``0.015``.
        :type title_gap_below: float

        :param debug: Print intermediate filtering and sign-change details
            to stdout.
        :type debug: bool
        """
        if not show_distribution and not show_fuel:
            print("⚠️  Must show at least one effect type!")
            return

        if self.analyzer.comparison_results is None or len(self.analyzer.comparison_results) == 0:
            print("No results to plot.  Run analyze_all_targets() first.")
            return

        if debug:
            print("\n🔍 DEBUG: plot_coefficient_changes_all_targets")
            print(f"   Initial results count: {len(self.analyzer.comparison_results)}")
            print(f"   Requested n_cols: {n_cols}")
            print(f"   Filtering: min_r2={min_r2}, dataset={filter_dataset}")

        results = self.analyzer._filter_results_by_performance(
            results_df=self.analyzer.comparison_results,
            min_r2=min_r2,
            dataset=filter_dataset,
            min_n_samples=min_n_samples,
            verbose=self.analyzer.verbose or debug,
        )

        if debug:
            print(f"   Post-filtering count: {len(results)}")

        if len(results) == 0:
            print("⚠️  No targets meet performance criteria.")
            return

        n_targets = len(results)
        effective_n_cols = min(n_cols, n_targets)
        n_rows = int(np.ceil(n_targets / effective_n_cols))

        if debug:
            print(f"   Grid: {n_rows} rows × {effective_n_cols} cols for {n_targets} targets")

        fig, axes = plt.subplots(
            n_rows, effective_n_cols,
            figsize=(figsize_per_plot[0] * effective_n_cols,
                     figsize_per_plot[1] * n_rows),
        )

        # Normalise axes to a flat array regardless of grid shape
        if n_rows == 1 and effective_n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif effective_n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes_flat = axes.flatten()

        # Build the hierarchical title structure
        if show_distribution and show_fuel:
            base_title = 'Distribution Effect vs Fuel Effect'
        elif show_distribution:
            base_title = 'Distribution Effect (Full → Matched)'
        else:
            base_title = 'Fuel Effect (Matched → Alternative)'

        metadata = {}
        if min_r2 is not None:
            metadata['Min R²'] = f'{min_r2:.2f} ({filter_dataset})'
        if len(results) < len(self.analyzer.comparison_results):
            metadata['Targets shown'] = (
                f'{len(results)}/{len(self.analyzer.comparison_results)}'
            )

        title_structure = self._build_hierarchical_title(
            main_title=base_title,
            title_prefix=title_prefix,
            title_suffix=title_suffix,
            metadata=metadata,
        )

        # Render title and get the y-position where subplot content should begin
        top_margin = self._add_hierarchical_suptitle(
            fig, title_structure,
            bold_spacing=title_bold_spacing,
            italic_spacing=title_italic_spacing,
            gap_below=title_gap_below,
        )

        # Plot each target
        for idx, (_, row) in enumerate(results.iterrows()):
            self._plot_single_target_coefficients(
                axes_flat[idx], row, show_values, min_change_threshold,
                show_distribution, show_fuel, bar_height, bar_spacing,
                show_sign_change_legend,
                variable_name_map,
                formula_name_map if formula_name_map is not None else variable_name_map,
            )

        # Hide unfilled cells
        for idx in range(n_targets, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, top_margin])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.analyzer.verbose:
                print(f"✅ Coefficient change plot saved to {save_path}")

        plt.show()

    # =============================================================================
    # _apply_name_mapping  (unchanged, included for reference)
    # =============================================================================

    def _apply_name_mapping(
            self,
            names: List[str],
            name_map: Optional[Dict[str, str]],
    ) -> List[str]:
        """
        Apply a name map to a list of strings.

        Names absent from the map are returned unchanged.

        :param names: Original name strings.
        :type names: list[str]
        :param name_map: Mapping from internal names to display strings, or
            ``None`` to return ``names`` unchanged.
        :type name_map: dict[str, str], optional

        :returns: Mapped names.
        :rtype: list[str]
        """
        if name_map is None:
            return names
        return [name_map.get(n, n) for n in names]

    def _build_hierarchical_title(self,
                                  main_title: str,
                                  title_prefix: str = '',
                                  title_suffix: str = '',
                                  metadata: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Build hierarchical title structure for formatted display.

        Concept:
        --------
        Creates a multi-level title with visual hierarchy:
        - Line 1: Prefix (bold, large) - optional
        - Line 2: Main title (bold, large) - always present
        - Line 3: Metadata (italic, smaller) - optional
        - Line 4: Suffix (italic, smaller) - optional

        Returns dictionary with 'lines' and 'styles' for rendering.

        Parameters
        ----------
        main_title : str
            Primary title text (always shown in bold)
        title_prefix : str
            Text before main title (shown in bold if present)
        title_suffix : str
            Additional context (shown in italic if present)
        metadata : dict, optional
            Key-value pairs for filtering info (shown in italic)
            Example: {'Min R²': '0.60 (alternative)', 'Targets': '12/25'}

        Returns
        -------
        title_structure : dict
            {
                'lines': [('text', 'style'), ...],  # style = 'bold' or 'italic'
                'has_metadata': bool
            }
        """

        lines = []

        # Line 1: Prefix (bold, if present)
        if title_prefix:
            lines.append((title_prefix, 'bold'))

        # Line 2: Main title (bold, always present)
        lines.append((main_title, 'bold'))

        # Line 3: Metadata (italic, if present)
        if metadata:
            meta_str = ' | '.join([f'{k}: {v}' for k, v in metadata.items()])
            lines.append((meta_str, 'italic'))

        # Line 4: Suffix (italic, if present)
        if title_suffix:
            lines.append((title_suffix, 'italic'))

        return {
            'lines': lines,
            'has_metadata': bool(metadata) or bool(title_suffix)
        }

    """
    Improved title spacing for CoefficientChangePlotter
    =====================================================

    Replaces the fixed-spacing title system with one that:

    1.  Exposes ``title_pad_top`` and ``title_line_spacing`` parameters on
        ``plot_coefficient_changes_all_targets`` so callers can tune spacing
        without touching internals.

    2.  Computes ``top_margin`` (the rect passed to tight_layout) from the
        actual rendered title height rather than a lookup dict, so it works
        correctly for any number of title lines.

    3.  Separates bold-line and italic-line spacing into two distinct
        parameters and adds a small fixed gap between the last title line and
        the first subplot row.

    Methods changed
    ---------------
        plot_coefficient_changes_all_targets   — two new parameters
        _build_hierarchical_title              — unchanged (no edits needed)
        _add_hierarchical_suptitle             — now accepts spacing params,
                                                 returns the final y-position
                                                 so the caller can compute
                                                 top_margin accurately
    """

    from typing import Dict, List, Optional, Tuple
    import numpy as np

    # =============================================================================
    # _add_hierarchical_suptitle  (replacement)
    # =============================================================================

    def _add_hierarchical_suptitle(
            self,
            fig,
            title_structure: Dict,
            bold_spacing: float = 0.030,
            italic_spacing: float = 0.022,
            gap_below: float = 0.015,
    ) -> float:
        """
        Add a hierarchical figure title using :func:`~matplotlib.figure.Figure.text`.

        Each title line is placed individually with :meth:`fig.text` so that
        bold lines and italic lines can use different font sizes and spacing
        without the constraints of :meth:`fig.suptitle`.

        Lines are stacked top-to-bottom starting from ``y = 1.0`` (the top of
        the figure in figure-fraction coordinates):

        - Bold lines (prefix, main title): larger font, more vertical space.
        - Italic lines (metadata, suffix): smaller font, tighter spacing.

        :param fig: Target matplotlib figure.
        :param title_structure: Dict as returned by
            :meth:`_build_hierarchical_title`, with a ``'lines'`` key
            containing ``(text, style)`` tuples where ``style`` is
            ``'bold'`` or ``'italic'``.
        :type title_structure: dict

        :param bold_spacing: Vertical gap consumed by each bold line, in
            figure-fraction units.  Increase to add more breathing room
            between bold title lines and the next element.  Default ``0.030``.
        :type bold_spacing: float

        :param italic_spacing: Vertical gap consumed by each italic line.
            Smaller than ``bold_spacing`` because italic lines use a smaller
            font.  Default ``0.022``.
        :type italic_spacing: float

        :param gap_below: Additional blank space inserted after the last
            title line before the subplot area begins.  Increase if the first
            row of subplots still feels cramped.  Default ``0.015``.
        :type gap_below: float

        :returns: The y-position (in figure-fraction coordinates) of the
            bottom edge of the title block, including ``gap_below``.  Pass
            this value directly as the ``top`` argument of
            ``plt.tight_layout(rect=[0, 0, 1, <return_value>])``.
        :rtype: float
        """
        lines = title_structure['lines']
        y = 1.0  # start from very top; text is placed va='top' so it sits inside

        for text, style in lines:
            if style == 'bold':
                fig.text(
                    0.5, y, text,
                    ha='center', va='top',
                    fontsize=14, fontweight='bold',
                    transform=fig.transFigure,
                )
                y -= bold_spacing
            else:  # italic
                fig.text(
                    0.5, y, text,
                    ha='center', va='top',
                    fontsize=10, style='italic', color='#555555',
                    transform=fig.transFigure,
                )
                y -= italic_spacing

        # Reserve blank gap between title block and subplot content
        y -= gap_below
        return y

    def _plot_single_target_coefficients(self,
                                         ax,
                                         result_row,
                                         show_values: bool,
                                         min_change_threshold: float,
                                         show_distribution: bool,
                                         show_fuel: bool,
                                         bar_height: float = 0.25,
                                         bar_spacing: float = 0.5,
                                         show_sign_change_legend: bool = False,
                                         variable_name_map: Optional[Dict[str, str]] = None,  # For title
                                         formula_name_map: Optional[Dict[str, str]] = None):  # NEW: For formula
        """
        Plot coefficient changes for a single target.

        NEW: Uses separate name maps for subplot title vs formula.

        Data Flow:
        ----------
        1. Extract target and predictors
        2. Apply variable_name_map to target → subplot title (e.g., "Beam (m)")
        3. Get formula with β notation
        4. Apply formula_name_map to formula → compact notation (e.g., "B")
        5. Display subplot with both mappings applied

        Parameters
        ----------
        [existing parameters...]
        variable_name_map : dict, optional
            Mapping for subplot title (full names like "Beam (m)")
        formula_name_map : dict, optional
            Mapping for formula (compact like "B")
            If None, uses variable_name_map
        """
        target = result_row['target']
        model_type = result_row['model_type']
        predictors = result_row['predictors']
        coeff_comparison = result_row['coefficient_comparison']

        # Apply variable_name_map to target for subplot title
        target_display = self._apply_name_mapping([target], variable_name_map)[0]

        # ✅ NEW: Get formula with formula_name_map (compact notation)
        # If formula_name_map is None, fallback to variable_name_map
        formula = self._get_beta_formula(
            target, predictors, model_type, result_row,
            formula_name_map if formula_name_map is not None else variable_name_map
        )

        # Choose baseline based on what's being displayed
        if show_fuel and not show_distribution:
            coeffs_baseline = result_row['coefficients_matched']
            baseline_label = " (Matched Conv.)"
        elif show_distribution and not show_fuel:
            coeffs_baseline = result_row['coefficients_full']
            baseline_label = " (Full Conv.)"
        else:
            coeffs_baseline = result_row['coefficients_full']
            baseline_label = " (Full Conv.)"

        baseline_values = self._format_baseline_coefficients(coeffs_baseline, model_type)

        if baseline_values:
            baseline_values = baseline_values + baseline_label

        # Extract coefficient changes with β labels
        plot_data = self._prepare_plot_data(
            coeff_comparison, min_change_threshold, result_row
        )

        if not plot_data['beta_labels']:
            ax.text(0.5, 0.5, 'No significant coefficient changes',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target_display}\n({formula})\n{baseline_values}',
                         fontweight='bold', fontsize=11)
            return

        # Detect sign changes for color coding
        sign_changes = self._detect_sign_changes(coeff_comparison, plot_data)

        # Apply custom spacing between bars
        y_positions = np.arange(len(plot_data['beta_labels'])) * bar_spacing

        # Determine bar configuration
        if show_distribution and show_fuel:
            dist_offset = -bar_height / 2
            fuel_offset = +bar_height / 2
        else:
            dist_offset = 0
            fuel_offset = 0

        # Distribution effect with sign-change coloring
        if show_distribution:
            dist_colors = [self._get_bar_color(sign_changes['distribution'][i])
                           for i in range(len(plot_data['beta_labels']))]

            bars_dist = ax.barh(y_positions + dist_offset,
                                plot_data['distribution_effects'],
                                bar_height,
                                label='Distribution Effect (Full → Matched)',
                                color=dist_colors,
                                alpha=0.8,
                                edgecolor='black', linewidth=0.5)

            if show_values:
                self._add_value_labels(ax, bars_dist, plot_data['distribution_effects'])

        # Fuel effect with sign-change coloring
        if show_fuel:
            fuel_colors = [self._get_bar_color(sign_changes['fuel'][i])
                           for i in range(len(plot_data['beta_labels']))]

            bars_fuel = ax.barh(y_positions + fuel_offset,
                                plot_data['fuel_effects'],
                                bar_height,
                                label='Fuel Effect (Matched → Alternative)',
                                color=fuel_colors,
                                alpha=0.8,
                                edgecolor='black', linewidth=0.5)

            if show_values:
                self._add_value_labels(ax, bars_fuel, plot_data['fuel_effects'])

        # Format axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_data['beta_labels'], fontsize=10, fontweight='bold')

        ax.set_xlabel('Change (std dev)' if result_row['normalized'] else 'Change',
                      fontsize=10)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        # ✅ Title: Uses variable_name_map for target, formula_name_map for formula
        # Example: "Beam (m)" in title, but "B" in formula
        title = f"{target_display}\n({formula})\n{baseline_values}"
        ax.set_title(title, fontweight='bold', fontsize=11, pad=10)

        # Conditional legend
        if show_sign_change_legend:
            if show_distribution and show_fuel:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black',
                          label='Distribution Effect'),
                    Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black',
                          label='Fuel Effect'),
                    Patch(facecolor='green', alpha=0.8, edgecolor='black',
                          label='Sign preserved'),
                    Patch(facecolor='red', alpha=0.8, edgecolor='black',
                          label='Sign changed')
                ]
                ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)
            elif show_distribution:
                ax.text(0.02, 0.98, 'Distribution Effect\nGreen: sign preserved, Red: sign changed',
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif show_fuel:
                ax.text(0.02, 0.98, 'Fuel Effect\nGreen: sign preserved, Red: sign changed',
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.2, axis='x')
        ax.set_ylim(-bar_spacing * 0.5, len(plot_data['beta_labels']) * bar_spacing - bar_spacing * 0.5)
        ax.invert_yaxis()

    def _detect_sign_changes(
            self,
            coeff_comparison: Dict,
            plot_data: Dict,
            debug: bool = False,
    ) -> Dict[str, List[bool]]:
        """
        Determine which coefficients changed sign during each transition.

        A sign change indicates a fundamental qualitative shift in the
        relationship between a predictor and the target — for example, a
        coefficient switching from a positive to a negative value means the
        association reversed direction across datasets.

        Two transitions are checked for each coefficient:

        - **Distribution effect** (Full Conventional → Matched Conventional):
          captures size-distribution sampling artefacts.
        - **Fuel effect** (Matched Conventional → Alternative Fuel): captures
          genuine fuel-type differences.

        Sign-change detection logic:

        - If both the *from* and *to* values are non-zero (``|val| > 1e-10``),
          a sign change is declared when their product is negative (i.e. they
          have opposite signs).
        - If either value is effectively zero, a sign change is also declared
          conservatively, because the direction is ambiguous.

        :param coeff_comparison: Coefficient comparison dict as produced by
            :meth:`~CoefficientComparisonAnalyzer._compare_coefficients_threeway`,
            containing ``'distribution_effect'`` and ``'fuel_effect'`` sub-dicts.
        :type coeff_comparison: dict

        :param plot_data: Prepared plot data as produced by
            :meth:`_prepare_plot_data`, containing ``'coefficient_names'``.
        :type plot_data: dict

        :param debug: If ``True``, prints per-coefficient sign-change
            calculations to stdout.  Controlled by the caller's ``debug``
            parameter; independent of the class-level ``verbose`` flag so
            that normal verbose output is not polluted by sign-change
            internals.
        :type debug: bool

        :returns: Dict with keys ``'distribution'`` and ``'fuel'``, each a
            list of booleans (one per coefficient in
            ``plot_data['coefficient_names']``) where ``True`` means the sign
            changed.
        :rtype: dict[str, list[bool]]
        """
        dist_effect = coeff_comparison['distribution_effect']
        fuel_effect = coeff_comparison['fuel_effect']

        sign_changes: Dict[str, List[bool]] = {'distribution': [], 'fuel': []}

        if debug:
            print("\n" + "=" * 80)
            print("SIGN CHANGE DETECTION DEBUG")
            print("=" * 80)

        for idx, coef_name in enumerate(plot_data['coefficient_names']):

            if debug:
                print(f"\n[{idx}] Analysing: {coef_name}")

            # ── Distribution effect: Full → Matched ───────────────────────
            if coef_name in dist_effect:
                val_from = dist_effect[coef_name]['from']
                val_to = dist_effect[coef_name]['to']

                if debug:
                    print(f"  Distribution: {val_from:.6f} → {val_to:.6f}")

                if abs(val_from) > 1e-10 and abs(val_to) > 1e-10:
                    sign_changed = (val_from * val_to) < 0
                    if debug:
                        print(f"    product = {val_from * val_to:.6f} → "
                              f"sign changed: {sign_changed}")
                else:
                    sign_changed = True
                    if debug:
                        zero_side = 'from' if abs(val_from) <= 1e-10 else 'to'
                        print(f"    {zero_side} value near zero → treating as sign change")

                sign_changes['distribution'].append(sign_changed)
            else:
                if debug:
                    print(f"  Distribution: '{coef_name}' not found; "
                          f"available: {list(dist_effect.keys())}")
                sign_changes['distribution'].append(False)

            # ── Fuel effect: Matched → Alternative ───────────────────────
            if coef_name in fuel_effect:
                val_from = fuel_effect[coef_name]['from']
                val_to = fuel_effect[coef_name]['to']

                if debug:
                    print(f"  Fuel: {val_from:.6f} → {val_to:.6f}")

                if abs(val_from) > 1e-10 and abs(val_to) > 1e-10:
                    sign_changed = (val_from * val_to) < 0
                    if debug:
                        print(f"    product = {val_from * val_to:.6f} → "
                              f"sign changed: {sign_changed}")
                else:
                    sign_changed = True
                    if debug:
                        zero_side = 'from' if abs(val_from) <= 1e-10 else 'to'
                        print(f"    {zero_side} value near zero → treating as sign change")

                sign_changes['fuel'].append(sign_changed)
            else:
                if debug:
                    print(f"  Fuel: '{coef_name}' not found; "
                          f"available: {list(fuel_effect.keys())}")
                sign_changes['fuel'].append(False)

        if debug:
            print("\n" + "=" * 80)
            print("SIGN CHANGE SUMMARY")
            print("=" * 80)
            for idx, coef_name in enumerate(plot_data['coefficient_names']):
                beta = f'β{self._get_subscript(idx)}'
                print(f"{beta} ({coef_name}):")
                print(f"  Distribution : "
                      f"{'RED (changed)' if sign_changes['distribution'][idx] else 'GREEN (preserved)'}")
                print(f"  Fuel         : "
                      f"{'RED (changed)' if sign_changes['fuel'][idx] else 'GREEN (preserved)'}")
            print("=" * 80 + "\n")

        return sign_changes

    def _get_bar_color(self, sign_changed: bool) -> str:
        """
        Get bar color based on whether coefficient sign changed.

        Color Scheme:
        - Green: Sign preserved (relationship direction unchanged)
        - Red: Sign changed (relationship direction reversed)

        This highlights fundamental qualitative changes in relationships.

        Parameters
        ----------
        sign_changed : bool
            Whether the coefficient changed sign

        Returns
        -------
        str
            Hex color code
        """
        if sign_changed:
            return '#e74c3c'  # Red - sign changed (critical change!)
        else:
            return '#2ecc71'  # Green - sign preserved (expected change)

    # =============================================================================
    # _format_baseline_coefficients
    # =============================================================================

    def _format_baseline_coefficients(
            self,
            coeffs_full: Dict,
            model_type: str,
    ) -> str:
        """
        Format baseline coefficient values for display in the subplot title.

        Produces a compact string such as ``β0=12.4, β1=0.52, β2=−0.32``
        using plain ASCII β-labels (``β0``, ``β1``, …).  Any Unicode
        subscript or typographic preference should be applied by the caller
        via the formula name maps before the result is embedded in the title.

        For simple nonlinear models (``'power'``, ``'exponential'``, etc.)
        the named parameters ``a``, ``b``, ``c`` are mapped to ``β0``,
        ``β1``, ``β2`` for display consistency with other model types.

        :param coeffs_full: Coefficient dict from the full-conventional
            descriptor, e.g. ``{'intercept': 12.4, '√DWT': 0.52}``.
        :type coeffs_full: dict[str, float]
        :param model_type: Model-type identifier used to decide whether the
            ``a``/``b``/``c`` nonlinear mapping applies.
        :type model_type: str

        :returns: Comma-separated coefficient string, or ``''`` when no
            coefficients are present.  Long strings (> 80 characters with
            more than 4 terms) are split across two lines.
        :rtype: str
        """
        if not coeffs_full:
            return ''

        nonlinear_simple = {
            'exponential', 'exponential-offset',
            'power', 'power-offset',
            'logarithmic', 'logarithmic-power',
        }

        if model_type in nonlinear_simple:
            # Map a → β0, b → β1, c → β2
            pairs = []
            for key, beta_idx in [('a', 0), ('b', 1), ('c', 2)]:
                val = coeffs_full.get(key) or coeffs_full.get(f'coef_{key}')
                if val is not None:
                    pairs.append((beta_idx, val))
            parts = [f'β{i}={self._format_value(v)}' for i, v in pairs]
        else:
            parts = []
            # intercept → β0
            if 'intercept' in coeffs_full:
                parts.append(f'β0={self._format_value(coeffs_full["intercept"])}')
            # remaining terms in order
            for idx, (key, val) in enumerate(
                    ((k, v) for k, v in coeffs_full.items() if k != 'intercept'),
                    start=1,
            ):
                parts.append(f'β{idx}={self._format_value(val)}')

        if not parts:
            return ''

        result = ', '.join(parts)
        if len(result) > 80 and len(parts) > 4:
            mid = len(parts) // 2
            result = ', '.join(parts[:mid]) + ',\n' + ', '.join(parts[mid:])

        return result

    def _format_value(self, val: float) -> str:
        """
        Format a coefficient value based on its magnitude.

        Parameters
        ----------
        val : float
            Coefficient value

        Returns
        -------
        str
            Formatted value string
        """
        if abs(val) < 0.001 or abs(val) > 10000:
            # Scientific notation for very small/large values
            return f"{val:.2e}"
        elif abs(val) < 0.01:
            return f"{val:.4f}"
        elif abs(val) < 1:
            return f"{val:.3f}"
        elif abs(val) < 100:
            return f"{val:.2f}"
        else:
            return f"{val:.1f}"



    def plot_coefficient_changes_single_target(self,
                                               target: str,
                                               figsize: Tuple[float, float] = (12, 8),
                                               save_path: Optional[str] = None,
                                               show_values: bool = True,
                                               min_change_threshold: float = 0.01,
                                               show_distribution: bool = True,
                                               show_fuel: bool = True,
                                               bar_height: float = 0.35,
                                               bar_spacing: float = 0.6,
                                               title_prefix: str = '',  # NEW parameter
                                               title_suffix: str = ''):  # NEW parameter
        """
        Create detailed coefficient change plot for a single target.

        Larger format with more detail than the multi-target plot.

        Parameters
        ----------
        target : str
            Target variable to plot
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, optional
            Path to save figure
        show_values : bool, default=True
            Whether to show numeric values on bars
        min_change_threshold : float, default=0.01
            Minimum absolute change to display
        show_distribution : bool, default=True
            Whether to show distribution effect
        show_fuel : bool, default=True
            Whether to show fuel effect
        bar_height : float, default=0.35
            Height of bars (can use larger bars for single-target plots)
        bar_spacing : float, default=0.6
            Spacing between coefficient groups
        title_prefix : str, default=''
            Text to add before the main title
        title_suffix : str, default=''
            Text to add after the main title

        Examples
        --------
        >>> # Single target with custom title
        >>> analyzer.plot_single_target_coefficients(
        ...     target='MCR (KW)',
        ...     show_fuel=True,
        ...     show_distribution=False,
        ...     title_prefix='Bulk Carriers',
        ...     title_suffix='Pure Alternative',
        ...     save_path='mcr_fuel_effect.png'
        ... )
        """
        if not show_distribution and not show_fuel:
            print("⚠️  Must show at least one effect type!")
            return

        if self.analyzer.comparison_results is None:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        # Find target in results
        results = self.analyzer.comparison_results
        target_row = results[results['target'] == target]

        if len(target_row) == 0:
            print(f"Target '{target}' not found in results.")
            available = results['target'].tolist()
            print(f"Available targets: {available}")
            return

        result_row = target_row.iloc[0]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot with custom bar height and spacing
        self._plot_single_target_coefficients(
            ax, result_row, show_values, min_change_threshold,
            show_distribution, show_fuel, bar_height, bar_spacing
        )

        # ✅ UPDATED: Add custom title to the figure
        if title_prefix or title_suffix:
            # Get the current title from the axis
            current_title = ax.get_title()

            # Build new title
            if title_prefix and title_suffix:
                new_title = f'{title_prefix} - {current_title} - {title_suffix}'
            elif title_prefix:
                new_title = f'{title_prefix} - {current_title}'
            else:  # title_suffix only
                new_title = f'{current_title} - {title_suffix}'

            ax.set_title(new_title, fontweight='bold', fontsize=14, pad=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.analyzer.verbose:
                print(f"✅ Plot saved to {save_path}")

        plt.show()

    # =============================================================================
    # _get_beta_formula
    # =============================================================================

    def _beta_label(self,idx: int) -> str:
        """
        Return a β coefficient label with a Unicode subscript index.

        This is a module-level helper — no class state required — so it can
        be called anywhere in the plotter without ``self``.

        :param idx: Zero-based coefficient index.
        :type idx: int

        :returns: e.g. ``'β₀'``, ``'β₁'``, ``'β₁₀'``.
        :rtype: str

        The Unicode subscript digits (U+2080–U+2089) render correctly in
        matplotlib text, axis tick labels, and figure titles without any
        LaTeX backend.
        """

        _SUBSCRIPT_DIGITS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

        return 'β' + str(idx).translate(_SUBSCRIPT_DIGITS)

    # =============================================================================
    # _get_beta_formula
    # =============================================================================

    def _get_beta_formula(
            self,
            target: str,
            predictors: Tuple,
            model_type: str,
            result_row,
            formula_name_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Return the model formula with β notation for display in subplot titles.

        If the descriptor is available the formula is taken directly from
        :meth:`~ModelDescriptor.get_formula_string` with
        ``include_coefficients=False``.  Variable names inside the formula are
        then replaced using ``formula_name_map`` so that the caller controls
        display notation (e.g. mapping ``'BEAM (M)'`` to ``'B'`` or ``'B (m)'``).

        If no descriptor is available a fallback formula is constructed from
        the coefficient keys in ``result_row['coefficients_full']``.

        :param target: Internal target variable name.
        :type target: str
        :param predictors: Predictor variable names.
        :type predictors: tuple[str, ...]
        :param model_type: Model-type identifier string.
        :type model_type: str
        :param result_row: Row from the comparison results DataFrame.
        :param formula_name_map: Optional mapping applied to variable names
            *inside* the formula string.  The caller is fully responsible for
            the display strings — Unicode subscripts, LaTeX, plain text, etc.
            If ``None`` no substitution is applied.
        :type formula_name_map: dict[str, str], optional

        :returns: Formula string with β notation and substituted variable names.
        :rtype: str
        """
        key = (tuple(predictors), target)

        if key in self.analyzer.model_descriptors['conventional_full']:
            descriptor = self.analyzer.model_descriptors['conventional_full'][key]

            formula = descriptor.get_formula_string(
                include_coefficients=False,
                notation='math',
            )

            if formula_name_map:
                formula = self._apply_name_mapping_to_formula(formula, formula_name_map)

            return formula

        # Fallback: build formula from coefficient keys
        coeffs_full = result_row['coefficients_full']
        coeff_names = [k for k in coeffs_full if k != 'intercept']

        # Apply formula mapping to the target label
        target_display = (formula_name_map or {}).get(target, target)

        terms = [self._beta_label(0)]
        for i, coeff_name in enumerate(coeff_names, 1):
            display_name = (formula_name_map or {}).get(coeff_name, coeff_name)
            terms.append(f'{self._beta_label(i)}·{display_name}')

        return f"{target_display} = {' + '.join(terms)}".replace(' + -', ' - ')

    def _ensure_beta_subscripts(self, formula: str) -> str:
        """
        Ensure all beta coefficients have proper Unicode subscripts.

        Concept:
        --------
        ModelDescriptor.get_formula_string() may return formulas with:
        - "beta0", "beta1" (text notation)
        - "β0", "β1" (missing subscripts)
        - "β₀", "β₁" (correct Unicode subscripts)

        This function normalizes to the correct Unicode subscript format.

        Parameters
        ----------
        formula : str
            Formula string that may have inconsistent beta notation

        Returns
        -------
        str
            Formula with proper β subscripts

        Examples
        --------
        >>> _ensure_beta_subscripts("Y = beta0 + beta1·X")
        "Y = β₀ + β₁·X"
        >>> _ensure_beta_subscripts("Y = β0 + β1·X")
        "Y = β₀ + β₁·X"
        """
        import re

        # Replace "beta" with "β"
        formula = formula.replace('beta', 'β')

        # Find all patterns like "β0", "β1", "β2", etc. (without subscripts)
        # and replace with proper subscripts "β₀", "β₁", "β₂"
        def replace_beta_number(match):
            number = match.group(1)
            subscript = self._get_subscript(int(number))
            return f'β{subscript}'

        # Pattern: β followed by one or more digits (not already subscripted)
        # Negative lookbehind to avoid already-subscripted characters
        pattern = r'β(\d+)'
        formula = re.sub(pattern, replace_beta_number, formula)

        return formula

    def _ensure_beta_subscripts(self, formula: str) -> str:
        """
        Ensure all beta coefficients have proper Unicode subscripts.

        Concept:
        --------
        ModelDescriptor.get_formula_string() may return formulas with:
        - "beta0", "beta1" (text notation)
        - "β0", "β1" (missing subscripts)
        - "β₀", "β₁" (correct Unicode subscripts)

        This function normalizes to the correct Unicode subscript format.

        Parameters
        ----------
        formula : str
            Formula string that may have inconsistent beta notation

        Returns
        -------
        str
            Formula with proper β subscripts

        Examples
        --------
        >>> _ensure_beta_subscripts("Y = beta0 + beta1·X")
        "Y = β₀ + β₁·X"
        >>> _ensure_beta_subscripts("Y = β0 + β1·X")
        "Y = β₀ + β₁·X"
        """
        import re

        # Replace "beta" with "β"
        formula = formula.replace('beta', 'β')

        # Find all patterns like "β0", "β1", "β2", etc. (without subscripts)
        # and replace with proper subscripts "β₀", "β₁", "β₂"
        def replace_beta_number(match):
            number = match.group(1)
            subscript = self._get_subscript(int(number))
            return f'β{subscript}'

        # Pattern: β followed by one or more digits (not already subscripted)
        # Negative lookbehind to avoid already-subscripted characters
        pattern = r'β(\d+)'
        formula = re.sub(pattern, replace_beta_number, formula)

        return formula

    # =============================================================================
    # _apply_name_mapping_to_formula  (unchanged, included for reference)
    # =============================================================================

    def _apply_name_mapping_to_formula(
            self,
            formula: str,
            name_map: Dict[str, str],
    ) -> str:
        """
        Replace internal variable names in a formula string with display names.

        Substitutions are applied longest-first to avoid partial replacements
        (e.g. ``'LOA (M)'`` is replaced before ``'LOA'`` to prevent
        ``'LOA (m) (M)'``).

        The caller is responsible for the content of display strings; this
        method performs only plain string replacement with no additional
        formatting.

        :param formula: Formula string produced by
            :meth:`~ModelDescriptor.get_formula_string`.
        :type formula: str
        :param name_map: Mapping from internal names to display strings.
        :type name_map: dict[str, str]

        :returns: Formula with display names substituted in.
        :rtype: str
        """
        result = formula
        for key in sorted(name_map, key=len, reverse=True):
            result = result.replace(key, name_map[key])
        return result

    def _get_subscript(self, n: int) -> str:
        """
        Convert integer to subscript Unicode characters.

        Parameters
        ----------
        n : int
            Number to convert

        Returns
        -------
        str
            Subscript version (e.g., 0 → ₀, 10 → ₁₀)
        """
        subscripts = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        return str(n).translate(subscripts)

    # =============================================================================
    # _prepare_plot_data
    # =============================================================================

    def _prepare_plot_data(
            self,
            coeff_comparison: Dict,
            min_change_threshold: float,
            result_row,
    ) -> Dict:
        """
        Prepare coefficient change data for plotting, labelling each term β0,
        β1, β2, … using plain ASCII digits.

        The caller controls any Unicode subscript or other formatting by
        supplying appropriate strings in ``formula_name_map`` — this method
        generates no typography of its own.

        :param coeff_comparison: Coefficient comparison dict with
            ``'distribution_effect'`` and ``'fuel_effect'`` sub-dicts.
        :type coeff_comparison: dict
        :param min_change_threshold: Minimum absolute change for a coefficient
            to be included in the plot.
        :type min_change_threshold: float
        :param result_row: Row from the comparison results DataFrame; provides
            the coefficient order via ``'coefficients_full'``.

        :returns: Dict with keys ``'beta_labels'``, ``'coefficient_names'``,
            ``'distribution_effects'``, ``'fuel_effects'``.
        :rtype: dict
        """
        dist_effect = coeff_comparison['distribution_effect']
        fuel_effect = coeff_comparison['fuel_effect']
        coeffs_full = result_row['coefficients_full']

        # Preserve insertion order: intercept first, then remaining terms
        ordered = (
                      ['intercept'] if 'intercept' in coeffs_full else []
                  ) + [k for k in coeffs_full if k != 'intercept']

        result: Dict = {
            'beta_labels': [],
            'coefficient_names': [],
            'distribution_effects': [],
            'fuel_effects': [],
        }

        for idx, coef_name in enumerate(ordered):
            if coef_name not in dist_effect or coef_name not in fuel_effect:
                continue

            dist_change = dist_effect[coef_name]['absolute_change']
            fuel_change = fuel_effect[coef_name]['absolute_change']

            if (abs(dist_change) < min_change_threshold
                    and abs(fuel_change) < min_change_threshold):
                continue

            result['beta_labels'].append(self._beta_label(idx))
            result['coefficient_names'].append(coef_name)
            result['distribution_effects'].append(dist_change)
            result['fuel_effects'].append(fuel_change)

        return result

    def _add_value_labels(self, ax, bars, values):
        """
        Add value labels inside bars when possible, outside for small bars.

        Parameters
        ----------
        ax : matplotlib Axes
            Subplot axes
        bars : BarContainer
            Bar objects
        values : list
            Values to display
        """
        # Get x-axis range to determine "small bar" threshold
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        min_bar_width_for_inside = x_range * 0.05  # 5% of plot width

        for bar, value in zip(bars, values):
            if abs(value) < 0.01:  # Skip very small values
                continue

            width = bar.get_width()
            abs_width = abs(width)

            # ✅ Decide placement: inside for large bars, outside for small
            if abs_width >= min_bar_width_for_inside:
                # Large bar: place inside
                x_pos = width * 0.5
                ha = 'center'
                color = 'white'
                fontweight = 'bold'
            else:
                # Small bar: place outside
                if width >= 0:
                    x_pos = width + 0.02
                    ha = 'left'
                else:
                    x_pos = width - 0.02
                    ha = 'right'
                color = 'black'
                fontweight = 'normal'

            # Format value
            if abs(value) >= 1:
                label = f'{value:+.2f}'
            else:
                label = f'{value:+.3f}'

            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    label,
                    ha=ha,
                    va='center',
                    fontsize=7,
                    fontweight=fontweight,
                    color=color)

    def _apply_name_mapping_to_formula(self,
                                       formula: str,
                                       name_map: Dict[str, str]) -> str:
        """
        Apply variable name mapping to formula string.

        ✅ UPDATED: Now supports Unicode subscripts in mapped names.

        Concept:
        --------
        Replaces internal variable names in formula with display names.
        Display names can include Unicode subscripts for professional appearance.

        Uses whole-word matching and longest-first replacement to handle:
        - Nested names like "LOA (M)" before "LOA"
        - Names with subscripts like "L_{OA}" → "L_OA"

        Parameters
        ----------
        formula : str
            Original formula with internal names
        name_map : dict
            Mapping from internal to display names.
            Display names can include Unicode subscripts.

        Returns
        -------
        str
            Formula with display names (including subscripts)

        Examples
        --------
        >>> # Regular mapping
        >>> formula = "DWT = β₀ + β₁·LOA (M)"
        >>> name_map = {'LOA (M)': 'L'}
        >>> _apply_name_mapping_to_formula(formula, name_map)
        "DWT = β₀ + β₁·L"

        >>> # Mapping with subscripts
        >>> name_map = {'LOA (M)': 'L_OA', 'LBP (M)': 'L_BP'}
        >>> # where L_OA is actually "L" + subscript "OA"
        >>> # Use helper: _format_subscript_name('L', 'OA') → 'L_OA'
        """

        # Sort by length (longest first) to handle nested names
        sorted_names = sorted(name_map.keys(), key=len, reverse=True)

        mapped_formula = formula

        for internal_name in sorted_names:
            display_name = name_map[internal_name]

            # Replace all occurrences
            mapped_formula = mapped_formula.replace(internal_name, display_name)

        return mapped_formula

    def _format_subscript_name(self, main: str, subscript: str) -> str:
        """
        Create a variable name with Unicode subscript.

        Concept:
        --------
        Converts "L" + "OA" → "L_OA" where OA is in Unicode subscript.
        Useful for creating professional-looking variable names like L_OA, L_BP.

        Parameters
        ----------
        main : str
            Main variable symbol (e.g., 'L', 'V', 'T')
        subscript : str
            Subscript text (e.g., 'OA', 'BP', 'max')

        Returns
        -------
        str
            Variable name with Unicode subscript

        Examples
        --------
        >>> _format_subscript_name('L', 'OA')
        'LₒA'  # (with subscript O and A)
        >>> _format_subscript_name('V', 'max')
        'Vₘₐₓ'
        >>> _format_subscript_name('T', 'BP')
        'T_BP'
        """
        # Convert subscript text to Unicode subscripts
        subscript_unicode = self._text_to_subscript(subscript)
        return f'{main}{subscript_unicode}'

    def _text_to_subscript(self, text: str) -> str:
        """
        Convert text to Unicode subscript characters.

        Concept:
        --------
        Maps regular characters to their Unicode subscript equivalents.
        Supports digits (0-9) and some letters (a, e, o, x, h, k, l, m, n, p, s, t).

        Parameters
        ----------
        text : str
            Text to convert to subscript

        Returns
        -------
        str
            Text with subscript Unicode characters

        Examples
        --------
        >>> _text_to_subscript('OA')
        'ₒA'  # (subscript o, regular A - not all letters have subscripts)
        >>> _text_to_subscript('BP')
        'BP'  # (B and P don't have subscript versions)
        >>> _text_to_subscript('max')
        'ₘₐₓ'
        >>> _text_to_subscript('123')
        '₁₂₃'
        """
        # Mapping of characters to Unicode subscripts
        # Note: Not all letters have Unicode subscript versions!
        subscript_map = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ',
            'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ',
            's': 'ₛ', 't': 'ₜ',
            # Capital letters don't have subscript versions in Unicode
            # So they'll remain as regular characters
        }

        result = ''
        for char in text:
            # Convert to lowercase for lookup (subscripts are lowercase)
            result += subscript_map.get(char.lower(), char)

        return result
