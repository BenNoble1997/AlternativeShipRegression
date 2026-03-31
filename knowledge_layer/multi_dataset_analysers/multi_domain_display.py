# multi_dataset_display.py (NEW FILE)
"""
Display class for multi-dataset domain analyzer results.

Handles display logic for comparing multiple alternative fuel datasets,
showing side-by-side performance across fuel types.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class MultiDatasetDisplay:
    """
    Handles display logic for multi-dataset domain analysis.

    Creates comparison tables showing how models perform across
    different alternative fuel types (LNG, Methanol, Ammonia, etc.).

    Examples
    --------
    >>> multi_analyzer = MultiDatasetDomainAnalyzer(...)
    >>> multi_analyzer.analyze_all_datasets()
    >>>
    >>> display = MultiDatasetDisplay(multi_analyzer)
    >>> display.print_fuel_type_comparison()
    >>> display.print_target_comparison('Main_Power')
    """

    def __init__(self, multi_analyzer):
        """
        Initialize display for multi-dataset analyzer.

        Parameters
        ----------
        multi_analyzer : MultiDatasetDomainAnalyzer
            Multi-dataset analyzer with results to display
        """
        self.multi_analyzer = multi_analyzer
        self.fmt = DisplayFormatters()

    # ========================================================================
    # MAIN COMPARISON TABLES
    # ========================================================================
    # multi_dataset_display.py (UPDATE print_fuel_type_comparison method)

    def print_fuel_type_comparison(self,
                                   targets: Optional[List[str]] = None,
                                   show_r2: bool = True,
                                   show_rrmse: bool = True):
        """
        Print comparison tables across all fuel types.

        Creates separate tables for R² and RRMSE performance metrics.

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
        Target | Predictors | Model | Train | Test | FuelType1_Match | FuelType1_Alt |
               FuelType1_DistGap | FuelType1_FuelGap | FuelType2_Match | ...
        """
        if self.multi_analyzer.comparison_results is None:
            print("No results available. Run analyze_all_datasets() first.")
            return

        df = self.multi_analyzer.comparison_results

        if targets is not None:
            df = df[df['target'].isin(targets)]

        if df.empty:
            print("No results to display.")
            return

        # Get fuel types (sorted for consistent ordering)
        fuel_types = sorted(df['fuel_type'].unique())

        print(self.fmt.format_header(
            "MULTI-DATASET FUEL TYPE COMPARISON",
            width=200
        ))

        print(f"\n📊 Comparing {len(fuel_types)} fuel types: {', '.join(fuel_types)}")
        print(f"   Across {df['target'].nunique()} targets")
        print(f"   Total combinations: {len(df):,}")

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

        print(f"\n\n{'=' * 200}")
        print(f"{metric_name} COMPARISON TABLE")
        print(f"{'=' * 200}")

        # Build header dynamically
        header_parts = [
            f"{'Target':<15s}",
            f"{'Predictors':<30s}",
            f"{'Model':<15s}",
            f"{'Train':>8s}",
            f"{'Test':>8s}"
        ]

        # Add columns for each fuel type: Match, Alt, DistGap, FuelGap
        for fuel_type in fuel_types:
            # Abbreviate fuel type if too long
            fuel_abbr = fuel_type[:8] if len(fuel_type) <= 8 else fuel_type[:7] + "."

            header_parts.extend([
                f"{fuel_abbr:>10s}_Match",
            f"{fuel_abbr:>10s}_Alt",
            f"{fuel_abbr:>10s}_DGap",  # Distribution Gap
            f"{fuel_abbr:>10s}_FGap"])

            print(" | ".join(header_parts))
            print(self.fmt.format_separator(200))

            # Group by target and print best combination for each
            for target in sorted(df['target'].unique()):
                target_df = df[df['target'] == target]

                # Get best combination across all fuel types for this target
                # Use average performance across fuel types as criterion
                avg_metric_col = f'alternative_{metric}'
                best_combo = self._get_best_combination_for_target(
                    target_df, fuel_types, avg_metric_col
                )

                if best_combo is None:
                    continue

                # Print row for this target
                self._print_target_row(
                    target, best_combo, fuel_types, target_df, metric
                )

            print(self.fmt.format_separator(200))

    def _get_best_combination_for_target(self,
                                         target_df: pd.DataFrame,
                                         fuel_types: List[str],
                                         metric_col: str) -> Optional[str]:
        """
        Find best predictor combination for a target across all fuel types.

        Strategy: For each predictor combo, calculate average performance
        across all fuel types, then pick the combo with best average.

        Parameters
        ----------
        target_df : pd.DataFrame
            Data for one target
        fuel_types : list[str]
            Fuel types to consider
        metric_col : str
            Metric column to optimize

        Returns
        -------
        str or None
            Best predictor combination string, or None if no valid data
        """
        # Calculate average performance for each combination
        combo_scores = {}

        for combo in target_df['predictor_str'].unique():
            combo_df = target_df[target_df['predictor_str'] == combo]

            # Check if this combo has results for all fuel types
            has_all = all(
                fuel_type in combo_df['fuel_type'].values
                for fuel_type in fuel_types
            )

            if has_all:
                # Use average across fuel types
                avg_score = combo_df[metric_col].mean()
                combo_scores[combo] = avg_score

        if not combo_scores:
            # Fallback: use combo that appears most frequently
            return target_df['predictor_str'].mode()[0] if len(target_df) > 0 else None

        # Best combo depends on metric (lower RRMSE or higher R²)
        if 'rrmse' in metric_col:
            best_combo = min(combo_scores.items(), key=lambda x: x[1])[0]
        else:
            best_combo = max(combo_scores.items(), key=lambda x: x[1])[0]

        return best_combo

    def _print_target_row(self,
                          target: str,
                          best_combo: str,
                          fuel_types: List[str],
                          target_df: pd.DataFrame,
                          metric: str):
        """
        Print one row for a target with results across all fuel types.

        Parameters
        ----------
        target : str
            Target dimension name
        best_combo : str
            Best predictor combination string
        fuel_types : list[str]
            Fuel types to show
        target_df : pd.DataFrame
            All data for this target
        metric : str
            'r2' or 'rrmse'
        """
        # Get data for best combination across all fuel types
        combo_df = target_df[target_df['predictor_str'] == best_combo]

        if combo_df.empty:
            return

        # Get basic info (should be same across fuel types)
        first_row = combo_df.iloc[0]
        pred_str = self.fmt.format_predictor_string(
            first_row['predictors'], max_length=28
        )
        model_str = self.fmt.format_model_name(
            first_row['model_type'], max_length=13
        )

        # Start building row
        row_parts = [
            f"{target:<15s}",
            f"{pred_str:<30s}",
            f"{model_str:<15s}"
        ]

        # Train and Test are same for all fuel types (use first)
        train_val = self.fmt.format_metric(first_row[f'train_{metric}'], metric)
        test_val = self.fmt.format_metric(first_row[f'test_{metric}'], metric)

        row_parts.extend([
            f"{train_val:>8s}",
            f"{test_val:>8s}"
        ])

        # Add columns for each fuel type
        for fuel_type in fuel_types:
            fuel_row = combo_df[combo_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                # No data for this fuel type
                row_parts.extend([
                    f"{'N/A':>12s}",
                    f"{'N/A':>10s}",
                    f"{'N/A':>10s}",
                    f"{'N/A':>10s}"
                ])
            else:
                fuel_data = fuel_row.iloc[0]

                # Matched, Alternative, Distribution Gap, Fuel Gap
                matched_val = self.fmt.format_metric(
                    fuel_data[f'matched_{metric}'], metric
                )
                alt_val = self.fmt.format_metric(
                    fuel_data[f'alternative_{metric}'], metric
                )

                # Distribution gap: Test → Matched
                if metric == 'rrmse':
                    dist_gap = fuel_data['gap_distribution_rrmse']
                else:
                    dist_gap = fuel_data['gap_distribution']

                # Fuel gap: Matched → Alternative
                if metric == 'rrmse':
                    fuel_gap = fuel_data['gap_fuel_type_rrmse']
                else:
                    fuel_gap = fuel_data['gap_fuel_type']

                dist_gap_str = self.fmt.format_gap(dist_gap, metric)
                fuel_gap_str = self.fmt.format_gap(fuel_gap, metric)

                row_parts.extend([
                    f"{matched_val:>12s}",
                    f"{alt_val:>10s}",
                    f"{dist_gap_str:>10s}",
                    f"{fuel_gap_str:>10s}"
                ])

        print(" | ".join(row_parts))
    # ========================================================================
    # TARGET-SPECIFIC DETAILED COMPARISON
    # ========================================================================

    def print_target_comparison(self,
                                target: str,
                                metric: str = 'alternative_rrmse',
                                show_all_combinations: bool = False):
        """
        Print detailed comparison for a specific target across all fuel types.

        Parameters
        ----------
        target : str
            Target dimension to analyze
        metric : str
            Metric for sorting combinations
        show_all_combinations : bool
            If True, shows all tested combinations.
            If False, shows only best combination for each fuel type.
        """
        if self.multi_analyzer.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_analyzer.comparison_results
        target_df = df[df['target'] == target]

        if target_df.empty:
            print(f"No results found for target '{target}'")
            return

        fuel_types = sorted(target_df['fuel_type'].unique())

        print(self.fmt.format_header(
            f"DETAILED COMPARISON: {target}",
            width=180
        ))

        print(f"\nComparing across {len(fuel_types)} fuel types: {', '.join(fuel_types)}")

        if show_all_combinations:
            self._print_all_combinations_comparison(target_df, fuel_types, metric)
        else:
            self._print_best_combinations_comparison(target_df, fuel_types, metric)

    def _print_all_combinations_comparison(self,
                                           target_df: pd.DataFrame,
                                           fuel_types: List[str],
                                           metric: str):
        """Print all combinations across all fuel types."""

        # Get all unique predictor combinations
        all_combos = target_df['predictor_str'].unique()

        print(f"\nShowing all {len(all_combos)} predictor combinations:")
        print(f"\n{'Predictors':<30s} | {'Model':<12s} | ", end='')

        # Header with fuel types
        for fuel_type in fuel_types:
            print(f"{fuel_type} R²/{fuel_type[:3]} RRMSE% | ", end='')
        print()

        print(self.fmt.format_separator(180))

        # Sort combinations by average performance across fuel types
        combo_scores = {}
        for combo in all_combos:
            combo_df = target_df[target_df['predictor_str'] == combo]
            avg_score = combo_df[metric].mean()
            combo_scores[combo] = avg_score

        sorted_combos = sorted(combo_scores.items(),
                               key=lambda x: x[1],
                               reverse=('r2' in metric))

        for combo, _ in sorted_combos:
            combo_df = target_df[target_df['predictor_str'] == combo]

            if combo_df.empty:
                continue

            # Get model type (should be same across fuel types)
            model_type = combo_df.iloc[0]['model_type']

            pred_str = self.fmt.format_predictor_string(
                combo_df.iloc[0]['predictors'], max_length=28
            )
            model_str = self.fmt.format_model_name(model_type, max_length=12)

            print(f"{pred_str:<30s} | {model_str:<12s} | ", end='')

            for fuel_type in fuel_types:
                fuel_row = combo_df[combo_df['fuel_type'] == fuel_type]

                if fuel_row.empty:
                    print(f"{'N/A':>8s}/{'N/A':>10s} | ", end='')
                else:
                    r2 = self.fmt.format_metric(
                        fuel_row.iloc[0]['alternative_r2'], 'r2'
                    )
                    rrmse = self.fmt.format_metric(
                        fuel_row.iloc[0]['alternative_rrmse'], 'rrmse'
                    )
                    print(f"{r2:>8s}/{rrmse:>10s} | ", end='')

            print()

        print(self.fmt.format_separator(180))

    def _print_best_combinations_comparison(self,
                                            target_df: pd.DataFrame,
                                            fuel_types: List[str],
                                            metric: str):
        """Print best combination for each fuel type."""

        print("\nBest combination for each fuel type:")

        header_parts = [
            f"{'Fuel Type':<12s}",
            f"{'Predictors':<30s}",
            f"{'Model':<12s}",
            f"{'Train R²':>8s}",
            f"{'Test R²':>8s}",
            f"{'Match R²':>9s}",
            f"{'Alt R²':>8s}",
            f"{'Alt RRMSE%':>11s}",
            f"{'Fuel Gap R²':>12s}",
            f"{'Fuel Gap RRMSE':>15s}"
        ]

        print(" | ".join(header_parts))
        print(self.fmt.format_separator(180))

        for fuel_type in fuel_types:
            fuel_df = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_df.empty:
                print(f"{fuel_type:<12s} | {'N/A':<30s} | ...")
                continue

            # Get best for this fuel type
            if 'rrmse' in metric:
                best_row = fuel_df.loc[fuel_df[metric].idxmin()]
            else:
                best_row = fuel_df.loc[fuel_df[metric].idxmax()]

            pred_str = self.fmt.format_predictor_string(
                best_row['predictors'], max_length=28
            )
            model_str = self.fmt.format_model_name(
                best_row['model_type'], max_length=12
            )

            row_parts = [
                f"{fuel_type:<12s}",
                f"{pred_str:<30s}",
                f"{model_str:<12s}",
                f"{self.fmt.format_metric(best_row['train_r2'], 'r2'):>8s}",
                f"{self.fmt.format_metric(best_row['test_r2'], 'r2'):>8s}",
                f"{self.fmt.format_metric(best_row['matched_r2'], 'r2'):>9s}",
                f"{self.fmt.format_metric(best_row['alternative_r2'], 'r2'):>8s}",
                f"{self.fmt.format_metric(best_row['alternative_rrmse'], 'rrmse'):>11s}",
                f"{self.fmt.format_gap(best_row['gap_fuel_type'], 'r2'):>12s}",
                f"{self.fmt.format_gap(best_row['gap_fuel_type_rrmse'], 'rrmse'):>15s}"
            ]

            print(" | ".join(row_parts))

        print(self.fmt.format_separator(180))

    # ========================================================================
    # RANKING TABLES
    # ========================================================================

    def print_fuel_type_ranking(self,
                                metric: str = 'gap_fuel_type_rrmse',
                                ascending: bool = True):
        """
        Rank fuel types by average performance across all targets.

        Parameters
        ----------
        metric : str
            Metric to rank by: 'gap_fuel_type_rrmse', 'alternative_rrmse',
            'gap_fuel_type_r2', 'alternative_r2'
        ascending : bool
            If True, lower is better (for gaps/errors).
            If False, higher is better (for R²).
        """
        if self.multi_analyzer.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_analyzer.comparison_results

        print(self.fmt.format_header(
            f"FUEL TYPE RANKING (by average {metric})",
            width=120
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
                  f"{metric + ' (mean)':>15s} | {metric + ' (std)':>15s} | "
                  f"{'n_targets':>10s} | "
                  f"{'Avg Alt R²':>11s} | {'Avg Alt RRMSE%':>15s}")
        print(header)
        print(self.fmt.format_separator(120))

        # Print rankings
        for rank, (fuel_type, row) in enumerate(fuel_stats.iterrows(), 1):
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]
            count_val = int(row[(metric, 'count')])
            avg_r2 = row[('alternative_r2', 'mean')]
            avg_rrmse = row[('alternative_rrmse', 'mean')]

            metric_type = 'r2' if 'r2' in metric else 'rrmse'

            print(f"{rank:<6d} | {fuel_type:<15s} | "
                  f"{self.fmt.format_metric(mean_val, metric_type):>15s} | "
                  f"{self.fmt.format_metric(std_val, metric_type):>15s} | "
                  f"{count_val:>10d} | "
                  f"{self.fmt.format_metric(avg_r2, 'r2'):>11s} | "
                  f"{self.fmt.format_metric(avg_rrmse, 'rrmse'):>15s}")

        print(self.fmt.format_separator(120))

    def print_target_difficulty_ranking(self,
                                        metric: str = 'gap_fuel_type_rrmse'):
        """
        Rank targets by how difficult they are to predict for alternative fuels.

        Shows which dimensions have the largest fuel type effects.

        Parameters
        ----------
        metric : str
            Metric to rank by: 'gap_fuel_type_rrmse', 'gap_fuel_type_r2',
            'alternative_rrmse', 'alternative_r2'
        """
        if self.multi_analyzer.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_analyzer.comparison_results

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
    # SUMMARY STATISTICS
    # ========================================================================

    def print_summary_statistics(self):
        """Print overall summary statistics across all datasets."""
        if self.multi_analyzer.comparison_results is None:
            print("No results available.")
            return

        df = self.multi_analyzer.comparison_results

        print(self.fmt.format_header("MULTI-DATASET ANALYSIS SUMMARY", width=140))

        # Overall stats
        print(f"\n📊 Dataset Overview:")
        print(f"  Fuel types analyzed: {df['fuel_type'].nunique()}")
        print(f"    • {', '.join(sorted(df['fuel_type'].unique()))}")
        print(f"  Targets analyzed: {df['target'].nunique()}")
        print(f"  Total combinations tested: {len(df):,}")
        print(f"  Predictor combinations: {df['predictor_str'].nunique()}")

        # Performance summary
        print(f"\n📈 Performance Summary:")
        print(f"  Average Alternative R²: {df['alternative_r2'].mean():.4f}")
        print(f"  Average Alternative RRMSE: {df['alternative_rrmse'].mean():.2f}%")
        print(f"  Average Fuel Type Gap (R²): {df['gap_fuel_type'].mean():+.4f}")
        print(f"  Average Fuel Type Gap (RRMSE): {df['gap_fuel_type_rrmse'].mean():+.2f}%")

        # Best/Worst
        best_idx = df['alternative_rrmse'].idxmin()
        worst_idx = df['alternative_rrmse'].idxmax()

        best = df.loc[best_idx]
        worst = df.loc[worst_idx]

        print(f"\n🏆 Best Performance:")
        print(f"  Target: {best['target']}")
        print(f"  Fuel Type: {best['fuel_type']}")
        print(f"  Predictors: {' + '.join(best['predictors'])}")
        print(f"  Alternative RRMSE: {best['alternative_rrmse']:.2f}%")
        print(f"  Fuel Gap RRMSE: {best['gap_fuel_type_rrmse']:+.2f}%")

        print(f"\n⚠️  Worst Performance:")
        print(f"  Target: {worst['target']}")
        print(f"  Fuel Type: {worst['fuel_type']}")
        print(f"  Predictors: {' + '.join(worst['predictors'])}")
        print(f"  Alternative RRMSE: {worst['alternative_rrmse']:.2f}%")
        print(f"  Fuel Gap RRMSE: {worst['gap_fuel_type_rrmse']:+.2f}%")

        print(f"\n{'=' * 140}")

