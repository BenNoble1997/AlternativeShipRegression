"""
Display class for literature formula evaluation results.

Handles all formatting and printing for LiteratureFormulaEvaluator,
following the same pattern as SelectorResultsDisplay.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class LiteratureEvaluationDisplay:
    """
    Handles all display logic for literature formula evaluation.

    Separates presentation from computation, making the evaluator
    cleaner and more testable.
    """

    def __init__(self, evaluator):
        """
        Initialize display for an evaluator.

        Parameters
        ----------
        evaluator : LiteratureFormulaEvaluator
            Evaluator instance with results to display
        """
        self.evaluator = evaluator
        self.fmt = DisplayFormatters()

    # ========================================================================
    # INITIALIZATION DISPLAY
    # ========================================================================

    def print_initialization_summary(self):
        """Print summary at evaluator initialization."""

        default_source = self.evaluator.dataset_manager.get_datasets_for_source('default')
        conventional = default_source.get_dataset('conventional')
        matched = default_source.get_dataset('matched')  # or from self.evaluator.matched_datasets
        alternative = default_source.get_dataset('alternative')

        print(f"\nLiterature Formula Evaluator (Three-Dataset) initialized:")
        print(f"  Conventional full: {len(conventional)}")
        print(f"  Alternative ships: {len(alternative)}")
        print(f"  Formulae to test: {len(self.evaluator.formulae)}")

        if self.evaluator.target_dimensions:
            print(f"  Target filter: {self.evaluator.target_dimensions}")

        if self.evaluator.predictor_dimensions:
            print(f"  Predictor filter: {self.evaluator.predictor_dimensions}")

    def print_formulae_filtering_results(self,
                                         kept: int,
                                         excluded_count: int,
                                         excluded_reasons: Dict[str, int]):
        """Print results of formula filtering."""
        if excluded_count > 0:
            print(f"\n  Filtered formulae:")
            print(f"    Kept: {kept}")
            print(f"    Excluded: {excluded_count}")

            if excluded_reasons['target'] > 0:
                print(f"      - Target not in allowed list: {excluded_reasons['target']}")

            if excluded_reasons['predictor'] > 0:
                print(f"      - Predictor not in allowed list: {excluded_reasons['predictor']}")

            if excluded_reasons['both'] > 0:
                print(f"      - Both target and predictor excluded: {excluded_reasons['both']}")

    # ========================================================================
    # VALIDATION WARNINGS
    # ========================================================================

    def print_column_validation_warnings(self,
                                         missing_cols: set,
                                         available_cols: set,
                                         affected_formulae: list):
        """Print warnings about missing columns."""
        if not missing_cols:
            return

        print(f"\n  ⚠️  Warning: Some formula columns not found in data:")
        print(f"     Missing: {sorted(missing_cols)}")
        print(f"\n  Available columns in your data:")
        print(f"     {sorted(available_cols)}")

        print(f"\n  Formulae that will be skipped: {len(affected_formulae)}/{len(self.evaluator.formulae)}")

        if len(affected_formulae) <= 5:
            for name in affected_formulae:
                print(f"     - {name}")
        else:
            for name in affected_formulae[:3]:
                print(f"     - {name}")
            print(f"     ... and {len(affected_formulae) - 3} more")

    def print_data_quality_warnings(self):
        # Get datasets from manager
        default_source = self.evaluator.dataset_manager.get_datasets_for_source('default')

        # Get matched dataset
        try:
            matched_keys = list(self.evaluator.matched_datasets.keys())
            if matched_keys:
                matched = self.evaluator.matched_datasets[matched_keys[0]]
            else:
                matched = default_source.get_dataset('conventional')
        except:
            matched = default_source.get_dataset('conventional')

        for dataset_name, dataset in [
            ('conventional', default_source.get_dataset('conventional')),
            ('matched', matched),
            ('alternative', default_source.get_dataset('alternative'))
        ]:
            if len(dataset) < 3:
                print(f"  ⚠️  WARNING: {dataset_name} dataset is very small ({len(dataset)} ships)")

    def print_domain_exclusion(self,
                               formula_name: str,
                               dataset_name: str,
                               n_outside: int,
                               n_total: int,
                               domain_lower: Optional[float],
                               domain_upper: Optional[float]):
        """Print domain filtering information."""
        if n_outside == 0:
            return

        pct_outside = 100 * n_outside / n_total
        print(f"  Domain filter for {formula_name} on {dataset_name}: "
              f"excluded {n_outside}/{n_total} ({pct_outside:.1f}%) outside "
              f"[{domain_lower}, {domain_upper}]")

    # ========================================================================
    # EVALUATION RESULTS DISPLAY - CONFIGURABLE METRICS (SIMPLIFIED)
    # ========================================================================

    def print_data_quality_report(self, results_list: list):
        """
        Print Phase 1: Data quality report with bounds and exclusions.

        Parameters
        ----------
        results_list : list of dict
            All evaluation results
        """
        print("\n" + "=" * 120)
        print("DATA QUALITY REPORT: Bounds and Domain Filtering")
        print("=" * 120)

        has_content = False

        for result in results_list:
            formula_name = result.get('formula_name', 'Unknown')
            target = result.get('target', 'Unknown')

            # Check if this formula has bounds or exclusions to report
            has_bounds = result.get('bounds') is not None
            has_exclusions = any(
                result.get(f'{ds}_n_excluded_domain', 0) > 0
                for ds in ['conventional', 'matched', 'alternative']
            )

            if not has_bounds and not has_exclusions:
                continue

            has_content = True

            # Print formula header
            print(f"\n{formula_name} → {target}:")

            # Print bounds information
            if has_bounds:
                bounds_pct = result.get('bounds')
                print(f"  Bounds ({bounds_pct:.1f}% percentile):")

                # Get predictor bounds from result if stored
                predictors = result.get('predictors', [])
                for pred in predictors:
                    lower = result.get(f'bound_lower_{pred}')
                    upper = result.get(f'bound_upper_{pred}')
                    if lower is not None and upper is not None:
                        print(f"    {pred}: [{lower:.2f}, {upper:.2f}]")

            # Print domain filtering results for each dataset
            for dataset_name in ['conventional', 'matched', 'alternative']:
                n_excluded = result.get(f'{dataset_name}_n_excluded_domain', 0)
                pct_excluded = result.get(f'{dataset_name}_pct_excluded_domain', 0)

                if n_excluded > 0:
                    n_total = result.get(f'{dataset_name}_n', 0) + n_excluded
                    domain_lower = result.get('domain_lower')
                    domain_upper = result.get('domain_upper')

                    print(f"  {dataset_name.capitalize()}: excluded {n_excluded}/{n_total} "
                          f"({pct_excluded:.1f}%) outside domain [{domain_lower}, {domain_upper}]")

        if not has_content:
            print("\n  (No bounds or domain filtering applied)")

        print("\n" + "=" * 120)

    def print_performance_results_table(self,
                                       results_list: list,
                                       metrics: str | list = 'all'):
        """
        Print Phase 2: Performance results table with configurable metrics.

        Metrics automatically include both absolute values AND gaps.

        Parameters
        ----------
        results_list : list of dict
            All evaluation results
        metrics : str or list of str, default='all'
            Which metric families to display:
            - 'all': All metrics (R², RRMSE, MAPE with their gaps)
            - 'r2': R² values + R² gaps
            - 'rrmse': RRMSE values + RRMSE gaps
            - 'mape': MAPE values + MAPE gaps
            - ['r2', 'rrmse']: Multiple metric families (list)

        Examples
        --------
        >>> # Show R² performance + R² gaps
        >>> display.print_performance_results_table(results, metrics='r2')

        >>> # Show R² and RRMSE (both with their gaps)
        >>> display.print_performance_results_table(results, metrics=['r2', 'rrmse'])

        >>> # Show everything
        >>> display.print_performance_results_table(results, metrics='all')
        """
        # Normalize metrics to a list
        if isinstance(metrics, str):
            if metrics == 'all':
                metrics = ['r2', 'rrmse', 'mape']
            else:
                metrics = [metrics]

        # Validate metric choices
        valid_metrics = ['r2', 'rrmse', 'mape']
        invalid = [m for m in metrics if m not in valid_metrics]
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. Choose from: {valid_metrics}")

        # Build header and determine column layout
        header_parts, width = self._build_table_header(metrics)

        print("\n" + "=" * width)
        print("PERFORMANCE RESULTS: Three-Dataset Analysis")
        print("=" * width)
        print(header_parts)
        print("-" * width)

        # Print each result row
        for result in results_list:
            self._print_result_row_configurable(result, metrics)

        print("=" * width)

    def _build_table_header(self, metrics: list) -> tuple[str, int]:
        """
        Build table header based on selected metrics.

        Each metric automatically includes its gap analysis.

        Returns
        -------
        header : str
            Formatted header string
        width : int
            Total table width
        """
        # Base columns (now includes reference)
        parts = [
            f"{'Formula':<35s}",
            f"{'Target':<15s}",
            f"{'Reference':<25s}"  # NEW: Added reference column
        ]
        base_width = 35 + 3 + 15 + 3 + 25 + 3  # formula + separator + target + separator + reference + separator

        # Add metric-specific columns (values + gaps)
        metric_width = 0

        if 'r2' in metrics:
            # R² values (3 datasets)
            parts.extend([
                f"{'Conv R²':>8s}",
                f"{'Match R²':>8s}",
                f"{'Alt R²':>8s}"
            ])
            # R² gaps (distribution, fuel, total)
            parts.extend([
                f"{'Dist Gap':>9s}",
                f"{'FUEL Gap':>9s}",
                f"{'Total Gap':>10s}"
            ])
            metric_width += (8 + 3) * 3 + (9 + 3) * 2 + 10 + 3

        if 'rrmse' in metrics:
            # RRMSE values (3 datasets)
            parts.extend([
                f"{'Conv RRMSE%':>12s}",
                f"{'Match RRMSE%':>13s}",
                f"{'Alt RRMSE%':>12s}"
            ])
            # RRMSE gaps (distribution, fuel, total)
            parts.extend([
                f"{'Dist Gap%':>10s}",
                f"{'FUEL Gap%':>10s}",
                f"{'Total Gap%':>11s}"
            ])
            metric_width += (12 + 3 + 13 + 3 + 12 + 3) + (10 + 3 + 10 + 3 + 11 + 3)

        if 'mape' in metrics:
            # MAPE values (3 datasets)
            parts.extend([
                f"{'Conv MAPE%':>11s}",
                f"{'Match MAPE%':>12s}",
                f"{'Alt MAPE%':>11s}"
            ])
            # MAPE gaps (distribution, fuel, total)
            parts.extend([
                f"{'Dist Gap%':>10s}",
                f"{'FUEL Gap%':>10s}",
                f"{'Total Gap%':>11s}"
            ])
            metric_width += (11 + 3 + 12 + 3 + 11 + 3) + (10 + 3 + 10 + 3 + 11 + 3)

        header = " | ".join(parts)
        total_width = base_width + metric_width

        return header, total_width

    def _print_result_row_configurable(self, result: Dict, metrics: list):
        """Print a single result row with configurable metrics."""
        # Check if we have valid results
        required_keys = ['conventional_r2', 'matched_r2', 'alternative_r2']
        if not all(k in result for k in required_keys):
            return

        formula_name = result.get('formula_name', 'Unknown')
        target = result.get('target', 'Unknown')
        reference = result.get('reference', 'No reference')  # NEW: Extract reference

        # Truncate reference if too long
        if len(reference) > 23:
            reference = reference[:20] + "..."

        # Base columns (now includes reference)
        parts = [
            f"{formula_name:<35s}",
            f"{target:<15s}",
            f"{reference:<25s}"  # NEW: Added reference column
        ]

        # Add metric-specific columns (values + gaps)
        if 'r2' in metrics:
            # R² values
            parts.extend([
                f"{result['conventional_r2']:>8.3f}",
                f"{result['matched_r2']:>8.3f}",
                f"{result['alternative_r2']:>8.3f}"
            ])
            # R² gaps - CHANGED: Inverted signs so decrease shows as negative
            parts.extend([
                f"{-result['gap_distribution']:>+9.3f}",  # CHANGED: Added negative sign
                f"{-result['gap_fuel_type']:>+9.3f}",  # CHANGED: Added negative sign
                f"{-result['gap_total']:>+9.3f}"  # CHANGED: Added negative sign
            ])

        if 'rrmse' in metrics:
            # RRMSE values
            parts.extend([
                f"{result['conventional_rrmse']:>12.2f}",
                f"{result['matched_rrmse']:>13.2f}",
                f"{result['alternative_rrmse']:>12.2f}"
            ])

            parts.extend([
                f"{result['gap_distribution_rrmse']:>10.2f}",
                f"{result['gap_fuel_type_rrmse']:>10.2f}",
                f"{result['gap_total_rrmse']:>11.2f}"
            ])

        if 'mape' in metrics:
            # MAPE values
            parts.extend([
                f"{result['conventional_mape']:>11.2f}",
                f"{result['matched_mape']:>12.2f}",
                f"{result['alternative_mape']:>11.2f}"
            ])
            # MAPE gaps (no change - these already show increase as positive)
            parts.extend([
                f"{result['gap_distribution_mape']:>+10.2f}",
                f"{result['gap_fuel_type_mape']:>+10.2f}",
                f"{result['gap_total_mape']:>+11.2f}"
            ])

        print(" | ".join(parts))


    def _print_single_result_row(self, result: Dict):
        """Helper for backward compatibility - shows R² and RRMSE with gaps."""
        self._print_result_row_configurable(result, ['r2', 'rrmse'])

    # ========================================================================
    # LEGACY METHODS (for backward compatibility)
    # ========================================================================

    def print_evaluation_header(self):
        """Legacy method: Print header for evaluation results table."""
        print("\n" + "=" * 180)
        print("LITERATURE FORMULA EVALUATION (Three-Dataset Analysis)")
        print("=" * 180)

        # Column headers
        header = (f"{'Formula':<35s} | {'Target':<15s} | "
                 f"{'Conv R²':>8s} | {'Match R²':>8s} | {'Alt R²':>8s} | "
                 f"{'Conv RRMSE%':>12s} | {'Match RRMSE%':>13s} | {'Alt RRMSE%':>12s} | "
                 f"{'FUEL Gap R²':>12s} | {'FUEL Gap RRMSE':>15s}")
        print(header)
        print("-" * 180)

    def print_evaluation_row(self, formula_name: str, target: str, result: Dict):
        """Legacy method: Print a single formula's evaluation results."""
        self._print_single_result_row(result)

    def print_evaluation_footer(self):
        """Legacy method: Print footer after all evaluations."""
        print("=" * 180)

    # ========================================================================
    # INTERPRETATION DISPLAY
    # ========================================================================

    def print_interpretation_header(self):
        """Print header for interpretation section."""
        print("\n" + "=" * 120)
        print("INTERPRETATION: Fuel Type Effect on Literature Formulae")
        print("=" * 120)

    def print_formula_interpretation(self,
                                     row: pd.Series,
                                     effect: str,
                                     icon: str,
                                     significance_threshold_r2: float,
                                     significance_threshold_rrmse: float):
        """Print detailed interpretation for a single formula."""
        print(f"\n{icon} {row['formula_name']} → {row['target']}:")
        print(f"   Type: {row['formula_type']}")

        if row['bounds'] is not None and not pd.isna(row['bounds']):
            print(f"   Bounds: {row['bounds']:.1f}% percentile")

        # Performance metrics
        print(f"\n   Performance (R²):")
        print(f"     Conventional:         {row['conventional_r2']:.3f}")
        print(f"     Matched (Conv):       {row['matched_r2']:.3f}")
        print(f"     Alternative (Fuel):   {row['alternative_r2']:.3f}")

        print(f"\n   Performance (RRMSE %):")
        print(f"     Conventional:         {row['conventional_rrmse']:.2f}%")
        print(f"     Matched (Conv):       {row['matched_rrmse']:.2f}%")
        print(f"     Alternative (Fuel):   {row['alternative_rrmse']:.2f}%")

        # Gaps
        print(f"\n   Gaps (R²):")
        print(f"     Distribution:    {row['gap_distribution']:+.3f} (conv→matched)")
        print(f"     FUEL TYPE:       {row['gap_fuel_type']:+.3f} (matched→alternative) ← KEY")

        print(f"\n   Gaps (RRMSE %):")
        print(f"     Distribution:   \u00D7{row['gap_distribution_rrmse']:.2f} (conv→matched)")
        print(f"     FUEL TYPE:      \u00D7{row['gap_fuel_type_rrmse']:.2f} (matched→alternative) ← KEY")
        print(f"     Total change:   \u00D7{row['gap_total_rrmse']:.2f} (conventional→alternative)")


        print(f"\n   Effect Classification: {effect}")

        # Additional warnings
        if abs(row['gap_fuel_type_rrmse']) > 20:
            print(f"   ⚠️  Prediction error increases by {abs(row['gap_fuel_type_rrmse']):.1f}% "
                  f"for alternative fuel ships!")

    def print_interpretation_summary(self,
                                     valid_results: pd.DataFrame,
                                     significance_threshold_r2: float,
                                     significance_threshold_rrmse: float):
        """Print summary statistics for interpretation."""
        print(f"\n{'=' * 120}")
        print("SUMMARY:")

        # Average gaps
        mean_gap_r2 = valid_results['gap_fuel_type'].mean()
        mean_gap_rrmse = valid_results['gap_fuel_type_rrmse'].mean()
        mean_gap_dist_r2 = valid_results['gap_distribution'].mean()
        mean_gap_dist_rrmse = valid_results['gap_distribution_rrmse'].mean()

        print(f"  Average distribution shift (R²): {mean_gap_dist_r2:+.3f}")
        print(f"  Average distribution shift (RRMSE): {mean_gap_dist_rrmse:+.2f}%")
        print(f"  Average fuel-type effect (R²): {mean_gap_r2:+.3f}")
        print(f"  Average fuel-type effect (RRMSE): {mean_gap_rrmse:+.2f}%")

        # Count significant effects
        significant = valid_results[
            (abs(valid_results['gap_fuel_type']) >= significance_threshold_r2) |
            (abs(valid_results['gap_fuel_type_rrmse']) >= significance_threshold_rrmse)
        ]
        print(f"  Formulae with significant fuel-type effect: {len(significant)}/{len(valid_results)}")

    # ========================================================================
    # EXPORT CONFIRMATION
    # ========================================================================

    def print_export_confirmation(self, filename: str):
        """Print confirmation after exporting results."""
        print(f"\n✓ Results exported to {filename}")

    # ========================================================================
    # ERROR MESSAGES
    # ========================================================================

    def print_no_results_warning(self, context: str = "evaluate"):
        """Print warning when no results are available."""
        if context == "evaluate":
            print("No results to evaluate.")
        elif context == "plot":
            print("No results to plot.")
        elif context == "export":
            print("No results to export.")
        elif context == "interpret":
            print("No results to interpret. Running evaluate_all_formulae() by default first.")

    def print_evaluation_error(self, formula_name: str, dataset_name: str, error: Exception):
        """Print error message for failed evaluation."""
        print(f"  Error on {dataset_name} data for {formula_name}: {error}")