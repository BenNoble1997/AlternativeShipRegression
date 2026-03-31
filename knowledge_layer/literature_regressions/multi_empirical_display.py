# multi_dataset_literature_display.py
"""
Display class for multi-dataset literature formula evaluator results.

Handles display logic for comparing literature formulas across multiple
alternative fuel datasets (LNG, Methanol, Ammonia, etc.).
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class MultiDatasetLiteratureDisplay:
    """
    Handles display logic for multi-dataset literature formula evaluation.

    Creates comparison tables showing how literature formulas perform across
    different alternative fuel types.

    Examples
    --------
    >>> multi_evaluator = MultiDatasetLiteratureEvaluator(...)
    >>> multi_evaluator.evaluate_all_datasets()
    >>>
    >>> display = MultiDatasetLiteratureDisplay(multi_evaluator)
    >>> display.print_fuel_type_comparison()
    >>> display.print_formula_comparison('DWT_to_LBP_power')
    """

    def __init__(self, multi_evaluator):
        """
        Initialize display for multi-dataset evaluator.

        Parameters
        ----------
        multi_evaluator : MultiDatasetLiteratureEvaluator
            Multi-dataset evaluator with results to display
        """
        self.multi_evaluator = multi_evaluator
        self.fmt = DisplayFormatters()

    # ========================================================================
    # MAIN COMPARISON TABLES
    # ========================================================================

    def print_fuel_type_comparison(self,
                                   targets: Optional[List[str]] = None,
                                   show_r2: bool = True,
                                   show_rrmse: bool = True):
        """
        Print comparison tables across all fuel types.

        Creates separate tables for R² and RRMSE performance metrics,
        showing how each formula performs on different fuel types.

        Parameters
        ----------
        targets : list[str], optional
            Targets to include. If None, uses all targets.
        show_r2 : bool
            If True, shows R² comparison table
        show_rrmse : bool
            If True, shows RRMSE comparison table

        Table Format
        ------------
        Formula | Target | Reference | Conv | Match | FuelType1_Alt | FuelType1_FuelGap |
                FuelType2_Alt | FuelType2_FuelGap | ...
        """
        if self.multi_evaluator.comparison_results is None:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        df = self.multi_evaluator.comparison_results

        if targets is not None:
            df = df[df['target'].isin(targets)]

        if df.empty:
            print("No results to display.")
            return

        # Get fuel types (sorted for consistent ordering)
        fuel_types = sorted(df['fuel_type'].unique())

        print(self.fmt.format_header(
            "MULTI-DATASET LITERATURE FORMULA COMPARISON",
            width=200
        ))

        print(f"\n📊 Comparing {len(fuel_types)} fuel types: {', '.join(fuel_types)}")
        print(f"   Across {df['target'].nunique()} targets")
        print(f"   Testing {df['formula_name'].nunique()} formulas")
        print(f"   Total evaluations: {len(df):,}")

        # Print R² table
        if show_r2:
            self._print_metric_comparison_table(df, fuel_types, metric='r2')

        # Print RRMSE table
        if show_rrmse:
            self._print_metric_comparison_table(df, fuel_types, metric='rrmse')

    def _print_metric_comparison_table(self,
                                       df: pd.DataFrame,
                                       fuel_types: List[str],
                                       metric: str):
        """
        Print comparison table for a specific metric (R² or RRMSE).

        Parameters
        ----------
        df : pd.DataFrame
            Full comparison results
        fuel_types : list[str]
            Fuel types to compare
        metric : str
            'r2' or 'rrmse'
        """

        metric_name = 'R²' if metric == 'r2' else 'RRMSE (%)'

        print(f"\n\n{'=' * 220}")
        print(f"{metric_name} COMPARISON TABLE")
        print(f"{'=' * 220}")

        # Build header dynamically
        header_parts = [
            f"{'Formula':<35s}",
            f"{'Target':<15s}",
            f"{'Reference':<20s}",
            f"{'Conv':>8s}",
            f"{'Match':>8s}"
        ]

        # Add columns for each fuel type: Alt, FuelGap
        for fuel_type in fuel_types:
            # Abbreviate fuel type if too long
            fuel_abbr = fuel_type[:10] if len(fuel_type) <= 10 else fuel_type[:9] + "."

            header_parts.extend([
                f"{fuel_abbr:>12s}_Alt",
                f"{fuel_abbr:>12s}_FGap"  # Fuel Gap
            ])

        print(" | ".join(header_parts))
        print(self.fmt.format_separator(220))

        # Group by target and formula
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]

            # Get unique formulas for this target
            formulas = target_df['formula_name'].unique()

            for formula_name in formulas:
                formula_df = target_df[target_df['formula_name'] == formula_name]

                if formula_df.empty:
                    continue

                # Print row for this formula
                self._print_formula_row(
                    formula_name, target, formula_df, fuel_types, metric
                )

            # Add separator between targets
            if target != sorted(df['target'].unique())[-1]:
                print(self.fmt.format_separator(220, char='·'))

        print(self.fmt.format_separator(220))

    def _print_formula_row(self,
                           formula_name: str,
                           target: str,
                           formula_df: pd.DataFrame,
                           fuel_types: List[str],
                           metric: str):
        """
        Print one row for a formula with results across all fuel types.

        Parameters
        ----------
        formula_name : str
            Formula name
        target : str
            Target dimension name
        formula_df : pd.DataFrame
            All data for this formula
        fuel_types : list[str]
            Fuel types to show
        metric : str
            'r2' or 'rrmse'
        """
        if formula_df.empty:
            return

        # Get basic info (should be same across fuel types)
        first_row = formula_df.iloc[0]

        # Truncate formula name if too long
        if len(formula_name) > 33:
            formula_display = formula_name[:30] + "..."
        else:
            formula_display = formula_name

        # Truncate reference if too long
        reference = first_row.get('reference', 'No reference')
        if len(reference) > 18:
            reference_display = reference[:15] + "..."
        else:
            reference_display = reference

        # Start building row
        row_parts = [
            f"{formula_display:<35s}",
            f"{target:<15s}",
            f"{reference_display:<20s}"
        ]

        # Conv and Match are same for all fuel types (use first)
        conv_val = self.fmt.format_metric(first_row[f'conventional_{metric}'], metric)
        match_val = self.fmt.format_metric(first_row[f'matched_{metric}'], metric)

        row_parts.extend([
            f"{conv_val:>8s}",
            f"{match_val:>8s}"
        ])

        # Add columns for each fuel type: Alternative, Fuel Gap
        for fuel_type in fuel_types:
            fuel_row = formula_df[formula_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                # No data for this fuel type
                row_parts.extend([
                    f"{'N/A':>12s}",
                    f"{'N/A':>12s}"
                ])
            else:
                fuel_data = fuel_row.iloc[0]

                # Alternative value
                alt_val = self.fmt.format_metric(
                    fuel_data[f'alternative_{metric}'], metric
                )

                # Fuel gap: Matched → Alternative
                if metric == 'rrmse':
                    fuel_gap = fuel_data['gap_fuel_type_rrmse']
                else:
                    # For R², invert sign so decrease shows as negative
                    fuel_gap = -fuel_data['gap_fuel_type']

                fuel_gap_str = self.fmt.format_gap(fuel_gap, metric)

                row_parts.extend([
                    f"{alt_val:>12s}",
                    f"{fuel_gap_str:>12s}"
                ])

        print(" | ".join(row_parts))

    # ========================================================================
    # DETAILED FORMULA COMPARISON
    # ========================================================================

    def print_formula_comparison(self,
                                 formula_name: str,
                                 show_all_targets: bool = False):
        """
        Print detailed comparison for a specific formula across all fuel types.

        Parameters
        ----------
        formula_name : str
            Formula name to analyze
        show_all_targets : bool
            If True, shows all targets this formula applies to.
            If False, shows only the first target found.
        """
        if self.multi_evaluator.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_evaluator.comparison_results
        formula_df = df[df['formula_name'] == formula_name]

        if formula_df.empty:
            print(f"Formula '{formula_name}' not found in results.")
            available = df['formula_name'].unique()
            print(f"\nAvailable formulas ({len(available)}):")
            for fname in sorted(available)[:10]:
                print(f"  • {fname}")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
            return

        fuel_types = sorted(formula_df['fuel_type'].unique())
        targets = formula_df['target'].unique()

        print(self.fmt.format_header(
            f"DETAILED COMPARISON: {formula_name}",
            width=180
        ))

        # Formula metadata
        first_row = formula_df.iloc[0]
        print(f"\nFormula Type: {first_row['formula_type']}")
        print(f"Predictors: {', '.join(first_row['predictors'])}")
        print(f"Reference: {first_row.get('reference', 'No reference')}")
        print(f"Targets: {', '.join(targets)}")
        print(f"Fuel types tested: {', '.join(fuel_types)}")

        # Show comparison for each target
        for target in targets:
            target_df = formula_df[formula_df['target'] == target]

            print(f"\n{self.fmt.format_subheader(f'Target: {target}', width=180)}")

            self._print_formula_target_comparison(target_df, fuel_types)

            if not show_all_targets:
                break  # Only show first target

    def _print_formula_target_comparison(self,
                                         target_df: pd.DataFrame,
                                         fuel_types: List[str]):
        """Print comparison table for one formula-target combination."""

        header = (f"{'Fuel Type':<15s} | "
                  f"{'Conv R²':>8s} | {'Match R²':>8s} | {'Alt R²':>8s} | "
                  f"{'Conv RRMSE%':>12s} | {'Match RRMSE%':>13s} | {'Alt RRMSE%':>12s} | "
                  f"{'Dist Gap R²':>12s} | {'Fuel Gap R²':>12s} | "
                  f"{'Dist Gap RRMSE':>15s} | {'Fuel Gap RRMSE':>15s}")

        print(header)
        print(self.fmt.format_separator(180))

        for fuel_type in fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                print(f"{fuel_type:<15s} | {'N/A':<100s}")
                continue

            row = fuel_row.iloc[0]

            # R² values
            conv_r2 = self.fmt.format_metric(row['conventional_r2'], 'r2')
            match_r2 = self.fmt.format_metric(row['matched_r2'], 'r2')
            alt_r2 = self.fmt.format_metric(row['alternative_r2'], 'r2')

            # RRMSE values
            conv_rrmse = self.fmt.format_metric(row['conventional_rrmse'], 'rrmse')
            match_rrmse = self.fmt.format_metric(row['matched_rrmse'], 'rrmse')
            alt_rrmse = self.fmt.format_metric(row['alternative_rrmse'], 'rrmse')

            # Gaps (invert R² sign for consistency)
            dist_gap_r2 = self.fmt.format_gap(-row['gap_distribution'], 'r2')
            fuel_gap_r2 = self.fmt.format_gap(-row['gap_fuel_type'], 'r2')
            dist_gap_rrmse = self.fmt.format_gap(row['gap_distribution_rrmse'], 'rrmse')
            fuel_gap_rrmse = self.fmt.format_gap(row['gap_fuel_type_rrmse'], 'rrmse')

            print(f"{fuel_type:<15s} | "
                  f"{conv_r2:>8s} | {match_r2:>8s} | {alt_r2:>8s} | "
                  f"{conv_rrmse:>12s} | {match_rrmse:>13s} | {alt_rrmse:>12s} | "
                  f"{dist_gap_r2:>12s} | {fuel_gap_r2:>12s} | "
                  f"{dist_gap_rrmse:>15s} | {fuel_gap_rrmse:>15s}")

        print(self.fmt.format_separator(180))

    # ========================================================================
    # RANKING TABLES
    # ========================================================================

    def print_fuel_type_ranking(self,
                                metric: str = 'gap_fuel_type_rrmse',
                                ascending: bool = True):
        """
        Rank fuel types by average formula performance.

        Shows which fuel types have the largest/smallest deviations
        from matched conventional ships.

        Parameters
        ----------
        metric : str
            Metric to rank by: 'gap_fuel_type_rrmse', 'alternative_rrmse',
            'gap_fuel_type', 'alternative_r2'
        ascending : bool
            If True, lower is better (for gaps/errors).
            If False, higher is better (for R²).
        """
        if self.multi_evaluator.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_evaluator.comparison_results

        print(self.fmt.format_header(
            f"FUEL TYPE RANKING (by average {metric})",
            width=140
        ))

        # Calculate average metric for each fuel type
        fuel_stats = df.groupby('fuel_type').agg({
            metric: ['mean', 'std', 'count'],
            'alternative_r2': 'mean',
            'alternative_rrmse': 'mean',
            'gap_fuel_type': 'mean',
            'gap_fuel_type_rrmse': 'mean'
        }).round(4)

        # Sort
        fuel_stats = fuel_stats.sort_values(
            (metric, 'mean'),
            ascending=ascending
        )

        # Print header
        header = (f"{'Rank':<6s} | {'Fuel Type':<15s} | "
                  f"{metric + ' (mean)':>18s} | {metric + ' (std)':>18s} | "
                  f"{'n_formulas':>11s} | "
                  f"{'Avg Alt R²':>11s} | {'Avg Alt RRMSE%':>15s}")
        print(header)
        print(self.fmt.format_separator(140))

        # Print rankings
        for rank, (fuel_type, row) in enumerate(fuel_stats.iterrows(), 1):
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]
            count_val = int(row[(metric, 'count')])
            avg_r2 = row[('alternative_r2', 'mean')]
            avg_rrmse = row[('alternative_rrmse', 'mean')]

            metric_type = 'r2' if 'r2' in metric else 'rrmse'

            print(f"{rank:<6d} | {fuel_type:<15s} | "
                  f"{self.fmt.format_metric(mean_val, metric_type):>18s} | "
                  f"{self.fmt.format_metric(std_val, metric_type):>18s} | "
                  f"{count_val:>11d} | "
                  f"{self.fmt.format_metric(avg_r2, 'r2'):>11s} | "
                  f"{self.fmt.format_metric(avg_rrmse, 'rrmse'):>15s}")

        print(self.fmt.format_separator(140))

        # Interpretation
        print("\nInterpretation:")
        if 'gap' in metric:
            print("  • Lower gap = formula is more robust for this fuel type")
            print("  • Higher gap = formula breaks down more for this fuel type")
        else:
            if 'r2' in metric:
                print("  • Higher R² = better prediction accuracy")
            else:
                print("  • Lower RRMSE = better prediction accuracy")

    def print_target_difficulty_ranking(self,
                                        metric: str = 'gap_fuel_type_rrmse'):
        """
        Rank targets by how difficult they are to predict for alternative fuels.

        Shows which dimensions have the largest fuel type effects.

        Parameters
        ----------
        metric : str
            Metric to rank by: 'gap_fuel_type_rrmse', 'gap_fuel_type',
            'alternative_rrmse', 'alternative_r2'
        """
        if self.multi_evaluator.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_evaluator.comparison_results

        print(self.fmt.format_header(
            f"TARGET DIFFICULTY RANKING (by average {metric})",
            width=140
        ))

        # Calculate average metric for each target
        target_stats = df.groupby('target').agg({
            metric: ['mean', 'std', 'min', 'max'],
            'alternative_rrmse': 'mean',
            'gap_fuel_type_rrmse': 'mean'
        }).round(4)

        # Sort by mean (descending for gaps/errors = most difficult first)
        ascending = 'r2' in metric  # R² higher is better, so we want descending
        target_stats = target_stats.sort_values(
            (metric, 'mean'),
            ascending=ascending
        )

        # Print
        metric_type = 'r2' if 'r2' in metric else 'rrmse'

        header = (f"{'Rank':<6s} | {'Target':<15s} | "
                  f"{'Mean':>10s} | {'Std':>10s} | "
                  f"{'Min':>10s} | {'Max':>10s} | "
                  f"{'Avg RRMSE%':>12s} | {'Avg Gap RRMSE%':>15s}")
        print(header)
        print(self.fmt.format_separator(140))

        for rank, (target, row) in enumerate(target_stats.iterrows(), 1):
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]
            min_val = row[(metric, 'min')]
            max_val = row[(metric, 'max')]
            avg_rrmse = row[('alternative_rrmse', 'mean')]
            avg_gap = row[('gap_fuel_type_rrmse', 'mean')]

            print(f"{rank:<6d} | {target:<15s} | "
                  f"{self.fmt.format_metric(mean_val, metric_type):>10s} | "
                  f"{self.fmt.format_metric(std_val, metric_type):>10s} | "
                  f"{self.fmt.format_metric(min_val, metric_type):>10s} | "
                  f"{self.fmt.format_metric(max_val, metric_type):>10s} | "
                  f"{self.fmt.format_metric(avg_rrmse, 'rrmse'):>12s} | "
                  f"{self.fmt.format_metric(avg_gap, 'rrmse'):>15s}")

        print(self.fmt.format_separator(140))

    # ========================================================================
    # FORMULA ROBUSTNESS ANALYSIS
    # ========================================================================

    def print_robustness_classification(self,
                                        threshold_rrmse: float = 10.0,
                                        threshold_r2: float = 0.05):
        """
        Classify formulas by robustness across fuel types.

        Categories:
        - ROBUST: Works well for all fuel types (low gaps)
        - FUEL-SENSITIVE: Breaks down for some fuel types (high gaps)
        - CONSISTENTLY POOR: High gaps for all fuel types

        Parameters
        ----------
        threshold_rrmse : float
            RRMSE gap threshold for significance (default: 10%)
        threshold_r2 : float
            R² gap threshold for significance (default: 0.05)
        """
        if self.multi_evaluator.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_evaluator.comparison_results

        print(self.fmt.format_header(
            "FORMULA ROBUSTNESS CLASSIFICATION",
            width=140
        ))

        print(f"\nThresholds:")
        print(f"  • R² gap: ±{threshold_r2:.3f}")
        print(f"  • RRMSE gap: {threshold_rrmse:.1f}%")

        # Classify each formula
        robust = []
        sensitive = []
        poor = []

        for formula_name in df['formula_name'].unique():
            formula_df = df[df['formula_name'] == formula_name]

            # Get gaps for all fuel types
            gaps_rrmse = formula_df['gap_fuel_type_rrmse'].values
            gaps_r2 = formula_df['gap_fuel_type'].abs().values

            # Count significant gaps
            n_significant_rrmse = np.sum(np.abs(gaps_rrmse) > threshold_rrmse)
            n_significant_r2 = np.sum(gaps_r2 > threshold_r2)
            n_fuel_types = len(gaps_rrmse)

            # Get target and average performance
            target = formula_df['target'].iloc[0]
            mean_gap_rrmse = np.mean(np.abs(gaps_rrmse))
            max_gap_rrmse = np.max(np.abs(gaps_rrmse))

            entry = {
                'formula': formula_name,
                'target': target,
                'mean_gap_rrmse': mean_gap_rrmse,
                'max_gap_rrmse': max_gap_rrmse,
                'n_fuel_types': n_fuel_types,
                'n_significant': max(n_significant_rrmse, n_significant_r2)
            }

            # Classify
            if n_significant_rrmse == 0 and n_significant_r2 == 0:
                robust.append(entry)
            elif n_significant_rrmse == n_fuel_types or n_significant_r2 == n_fuel_types:
                poor.append(entry)
            else:
                sensitive.append(entry)

        # Print classifications
        self._print_robustness_category("✅ ROBUST", robust,
                                        "work well for all fuel types")
        self._print_robustness_category("⚠️  FUEL-SENSITIVE", sensitive,
                                        "break down for some fuel types")
        self._print_robustness_category("🚨 CONSISTENTLY POOR", poor,
                                        "fail for all fuel types")

        # Summary
        total = len(robust) + len(sensitive) + len(poor)
        print(f"\n{self.fmt.format_separator(140)}")
        print("SUMMARY:")
        print(f"  Total formulas: {total}")
        print(f"  • Robust:            {len(robust):3d} ({100 * len(robust) / total:5.1f}%)")
        print(f"  • Fuel-sensitive:    {len(sensitive):3d} ({100 * len(sensitive) / total:5.1f}%)")
        print(f"  • Consistently poor: {len(poor):3d} ({100 * len(poor) / total:5.1f}%)")

    def _print_robustness_category(self,
                                   title: str,
                                   entries: List[Dict],
                                   description: str):
        """Print one robustness category."""
        print(f"\n{title} FORMULAS ({description}): {len(entries)}")
        print(self.fmt.format_separator(140))

        if not entries:
            print("  (None found)")
            return

        # Sort by mean gap
        entries_sorted = sorted(entries, key=lambda x: x['mean_gap_rrmse'])

        header = (f"{'Formula':<45s} | {'Target':<15s} | "
                  f"{'Mean Gap RRMSE':>16s} | {'Max Gap RRMSE':>15s} | "
                  f"{'Significant/Total':>18s}")
        print(header)
        print(self.fmt.format_separator(140, char='-'))

        for entry in entries_sorted:
            formula_display = entry['formula']
            if len(formula_display) > 43:
                formula_display = formula_display[:40] + "..."

            sig_frac = f"{entry['n_significant']}/{entry['n_fuel_types']}"

            print(f"{formula_display:<45s} | {entry['target']:<15s} | "
                  f"{self.fmt.format_metric(entry['mean_gap_rrmse'], 'rrmse'):>16s} | "
                  f"{self.fmt.format_metric(entry['max_gap_rrmse'], 'rrmse'):>15s} | "
                  f"{sig_frac:>18s}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    def print_summary_statistics(self):
        """Print overall summary statistics across all datasets."""
        if self.multi_evaluator.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_evaluator.comparison_results

        print(self.fmt.format_header(
            "MULTI-DATASET LITERATURE EVALUATION SUMMARY",
            width=140
        ))

        # Dataset overview
        print(f"\n📊 Dataset Overview:")
        print(f"  Fuel types analyzed: {df['fuel_type'].nunique()}")
        print(f"    • {', '.join(sorted(df['fuel_type'].unique()))}")
        print(f"  Targets covered: {df['target'].nunique()}")
        print(f"  Formulas tested: {df['formula_name'].nunique()}")
        print(f"  Total evaluations: {len(df):,}")

        # Performance summary
        print(f"\n📈 Average Performance:")
        print(f"  Conventional R²:  {df['conventional_r2'].mean():.4f}")
        print(f"  Matched R²:       {df['matched_r2'].mean():.4f}")
        print(f"  Alternative R²:   {df['alternative_r2'].mean():.4f}")
        print(f"  ")
        print(f"  Conventional RRMSE: {df['conventional_rrmse'].mean():.2f}%")
        print(f"  Matched RRMSE:      {df['matched_rrmse'].mean():.2f}%")
        print(f"  Alternative RRMSE:  {df['alternative_rrmse'].mean():.2f}%")

        # Gap analysis
        print(f"\n📉 Average Gaps (Fuel Type Effect):")
        print(f"  R² gap:     {df['gap_fuel_type'].mean():+.4f}")
        print(f"  RRMSE gap:  {df['gap_fuel_type_rrmse'].mean():+.2f}%")

        # Best/Worst by fuel type
        print(f"\n🏆 Best Performance by Fuel Type:")
        for fuel_type in sorted(df['fuel_type'].unique()):
            fuel_df = df[df['fuel_type'] == fuel_type]
            best_idx = fuel_df['alternative_rrmse'].idxmin()
            best = fuel_df.loc[best_idx]

            print(f"  {fuel_type}:")
            print(f"    Formula: {best['formula_name']}")
            print(f"    Target: {best['target']}")
            print(f"    Alt RRMSE: {best['alternative_rrmse']:.2f}%")
            print(f"    Fuel Gap: {best['gap_fuel_type_rrmse']:+.2f}%")

        print(f"\n{self.fmt.format_separator(140)}")