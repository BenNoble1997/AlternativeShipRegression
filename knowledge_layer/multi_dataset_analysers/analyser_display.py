"""
Display class for domain analyzer results.

Handles all display/printing logic for PolynomialDomainAnalyzer and
RandomForestDomainAnalyzer, keeping core analyzer classes focused on computation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class AnalyzerResultsDisplay:
    """
    Handles all display logic for domain analyzers.

    This class separates presentation from computation for four-way
    domain shift analysis, allowing:
    - Silent computation with later display
    - Multiple display formats
    - Consistent formatting

    Examples
    --------
    >>> # Automatic display
    >>> analyzer = PolynomialDomainAnalyzer(...)
    >>> analyzer.analyze_all_combinations(display_results=True)

    >>> # Manual display
    >>> analyzer.analyze_all_combinations(display_results=False)
    >>> display = AnalyzerResultsDisplay(analyzer)
    >>> display.print_analysis_results()
    >>> display.print_interpretation()
    """

    def __init__(self, analyzer):
        """
        Initialize display for an analyzer.

        Parameters
        ----------
        analyzer : PolynomialDomainAnalyzer or RandomForestDomainAnalyzer
            Analyzer instance with results to display
        """
        self.analyzer = analyzer
        self.fmt = DisplayFormatters()

    # ========================================================================
    # INITIALIZATION DISPLAY
    # ========================================================================

    def print_initialization_summary(self):
        """Print configuration summary at analyzer initialization."""
        analyzer = self.analyzer

        print(f"\n{analyzer.__class__.__name__} initialized:")
        print(f"  Conventional full: {len(analyzer.conventional_full)} ships")
        print(f"  Conventional matched: {len(analyzer.conventional_matched)} ships")
        print(f"  Alternative: {len(analyzer.alternative)} ships")
        print(f"  Test size: {analyzer.test_size * 100:.0f}%")

        # Model recommendations info
        if hasattr(analyzer, 'model_recommendations'):
            print(f"  Model recommendations: {len(analyzer.model_recommendations)} combinations")

            # Breakdown by target
            targets_in_recs = set(tgt for (_, tgt) in analyzer.model_recommendations.keys())
            print(f"  Targets with recommendations: {len(targets_in_recs)}")
            for target in sorted(targets_in_recs):
                n_combos = sum(1 for (_, tgt) in analyzer.model_recommendations.keys()
                               if tgt == target)
                print(f"    • {target}: {n_combos} combinations")

        # Selector results info (Random Forest)
        elif hasattr(analyzer, 'selector_results'):
            total_combos = sum(len(combos) for combos in analyzer.detailed_results.values())
            print(f"  Selector results: {total_combos} combinations")
            print(f"  Targets with results: {len(analyzer.detailed_results)}")
            for target in sorted(analyzer.detailed_results.keys()):
                n_combos = len(analyzer.detailed_results[target])
                print(f"    • {target}: {n_combos} combinations")

    # ========================================================================
    # MAIN RESULTS DISPLAY
    # ========================================================================

    def print_analysis_results(self,
                               results_df: Optional[pd.DataFrame] = None,
                               group_by_target: bool = True,
                               show_all_metrics: bool = False):
        """
        Print formatted analysis results table.

        Parameters
        ----------
        results_df : pd.DataFrame, optional
            Results to print. If None, uses analyzer.results_df
        group_by_target : bool
            If True, prints results grouped by target with headers
        show_all_metrics : bool
            If True, shows all metrics. If False, shows compact view
        """
        if results_df is None:
            if not hasattr(self.analyzer, 'results_df'):
                print("No results to display. Run analyze_all_combinations() first.")
                return
            results_df = self.analyzer.results_df

        if results_df.empty:
            print("No results to display.")
            return

        # Print main header
        print(self.fmt.format_header(
            "DOMAIN SHIFT ANALYSIS RESULTS (Four-Way Evaluation)",
            width=180
        ))

        # Print grouped or ungrouped
        if group_by_target:
            for target in results_df['target'].unique():
                target_df = results_df[results_df['target'] == target]
                self._print_target_section(target, target_df, show_all_metrics)
        else:
            self._print_results_table(results_df, show_all_metrics)

        # Print summary
        print(f"\n{'=' * 180}")
        print(f"✓ Analysis complete: {len(results_df)} combinations tested "
              f"across {results_df['target'].nunique()} targets")
        print(f"{'=' * 180}")

    def _print_target_section(self,
                              target: str,
                              target_df: pd.DataFrame,
                              show_all_metrics: bool):
        """Print results section for a single target."""
        print(f"\n{'=' * 180}")
        print(f"Target: {target}")

        # Show available predictors
        all_predictors = set()
        for pred_list in target_df['predictors']:
            all_predictors.update(pred_list)
        print(f"Available predictors: {sorted(list(all_predictors))}")

        print(f"{'=' * 180}")

        # Print table
        self._print_results_table(target_df, show_all_metrics)

    def _print_results_table(self,
                             df: pd.DataFrame,
                             show_all_metrics: bool):
        """Print formatted results table."""
        if show_all_metrics:
            self._print_full_table(df)
        else:
            self._print_compact_table(df)

    def _print_compact_table(self, df: pd.DataFrame):
        """Print compact results table with key metrics only."""
        # Column headers
        header = (f"{'Predictors':<25s} | {'Model':<12s} | "
                  f"{'Train R²':>8s} | {'Test R²':>8s} | {'Match R²':>8s} | {'Alt R²':>8s} | "
                  f"{'Train RRMSE%':>12s} | {'Test RRMSE%':>11s} | "
                  f"{'Match RRMSE%':>12s} | {'Alt RRMSE%':>11s} | "
                  f"{'FUEL Gap R²':>12s} | {'FUEL Gap RRMSE':>15s}")

        print(header)
        print(self.fmt.format_separator(180))

        # Data rows
        for _, row in df.iterrows():
            pred_str = self.fmt.format_predictor_string(row['predictors'], max_length=23)
            model = self.fmt.format_model_name(row['model_type'], max_length=12)

            print(f"{pred_str:<25s} | {model:<12s} | "
                  f"{self.fmt.format_metric(row['train_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['test_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['matched_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['alternative_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['train_rrmse'], 'rrmse'):>12s} | "
                  f"{self.fmt.format_metric(row['test_rrmse'], 'rrmse'):>11s} | "
                  f"{self.fmt.format_metric(row['matched_rrmse'], 'rrmse'):>12s} | "
                  f"{self.fmt.format_metric(row['alternative_rrmse'], 'rrmse'):>11s} | "
                  f"{self.fmt.format_gap(row['gap_fuel_type'], 'r2'):>12s} | "
                  f"{self.fmt.format_gap(row['gap_fuel_type_rrmse'], 'rrmse'):>15s}")

    def _print_full_table(self, df: pd.DataFrame):
        """Print full results table with all metrics."""
        display_cols = [
            'predictor_str', 'model_type', 'n_predictors',
            'train_r2', 'test_r2', 'matched_r2', 'alternative_r2',
            'train_rrmse', 'test_rrmse', 'matched_rrmse', 'alternative_rrmse',
            'train_mae', 'test_mae', 'matched_mae', 'alternative_mae',
            'train_mape', 'test_mape', 'matched_mape', 'alternative_mape',
            'gap_generalization', 'gap_distribution', 'gap_fuel_type',
            'gap_generalization_rrmse', 'gap_distribution_rrmse', 'gap_fuel_type_rrmse'
        ]

        # Filter to columns that exist
        existing_cols = [col for col in display_cols if col in df.columns]

        print(df[existing_cols].to_string(index=False))

    # ========================================================================
    # INTERPRETATION DISPLAY
    # ========================================================================

    def print_interpretation(self,
                             significance_threshold_r2: float = 0.05,
                             significance_threshold_rrmse: float = 10.0):
        """
        Interpret which dimensions are affected by fuel type.

        Parameters
        ----------
        significance_threshold_r2 : float
            R² gap threshold to consider effect significant
        significance_threshold_rrmse : float
            RRMSE gap threshold (%) to consider effect significant
        """
        if not hasattr(self.analyzer, 'results') or not self.analyzer.results:
            print("No results available for interpretation.")
            return

        print(self.fmt.format_header(
            "INTERPRETATION: Fuel Type Effect Analysis",
            width=120
        ))

        summary = []

        for target, models in self.analyzer.results.items():
            if not models:
                continue

            # Get best model (lowest alternative RRMSE)
            best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo

            # Extract gaps
            fuel_gap_r2 = metrics['gap_fuel_type']
            fuel_gap_rrmse = metrics['gap_fuel_type_rrmse']
            gen_gap_r2 = metrics['gap_generalization']
            gen_gap_rrmse = metrics['gap_generalization_rrmse']
            dist_gap_r2 = metrics['gap_distribution']
            dist_gap_rrmse = metrics['gap_distribution_rrmse']

            # Classify effect
            effect, icon = self.fmt.classify_effect(
                fuel_gap_r2,
                fuel_gap_rrmse,
                significance_threshold_r2,
                significance_threshold_rrmse
            )

            # Print detailed analysis for this target
            print(f"\n{icon} {target}:")
            print(f"   Best predictors: {combo_name}")
            print(f"   Model: {metrics['model_type']}")

            print(f"\n   Performance (R²):")
            print(f"     Train:       {self.fmt.format_metric(metrics['train_r2'], 'r2')}")
            print(f"     Test:        {self.fmt.format_metric(metrics['test_r2'], 'r2')}")
            print(f"     Matched:     {self.fmt.format_metric(metrics['matched_r2'], 'r2')}")
            print(f"     Alternative: {self.fmt.format_metric(metrics['alternative_r2'], 'r2')}")

            # OOB score if available (Random Forest)
            if 'oob_r2' in metrics:
                print(f"     OOB:         {self.fmt.format_metric(metrics['oob_r2'], 'r2')}")

            print(f"\n   Performance (RRMSE %):")
            print(f"     Train:       {self.fmt.format_metric(metrics['train_rrmse'], 'rrmse')}%")
            print(f"     Test:        {self.fmt.format_metric(metrics['test_rrmse'], 'rrmse')}%")
            print(f"     Matched:     {self.fmt.format_metric(metrics['matched_rrmse'], 'rrmse')}%")
            print(f"     Alternative: {self.fmt.format_metric(metrics['alternative_rrmse'], 'rrmse')}%")

            print(f"\n   Gaps (R²):")
            print(f"     Generalization:  {self.fmt.format_gap(gen_gap_r2, 'r2')} (train→test)")
            print(f"     Distribution:    {self.fmt.format_gap(dist_gap_r2, 'r2')} (test→matched)")
            print(f"     FUEL TYPE:       {self.fmt.format_gap(fuel_gap_r2, 'r2')} (matched→alternative) ← KEY")

            print(f"\n   Gaps (RRMSE %):")
            print(f"     Generalization:  {self.fmt.format_gap(gen_gap_rrmse, 'rrmse')} (train→test)")
            print(f"     Distribution:    {self.fmt.format_gap(dist_gap_rrmse, 'rrmse')} (test→matched)")
            print(f"     FUEL TYPE:       {self.fmt.format_gap(fuel_gap_rrmse, 'rrmse')} (matched→alternative) ← KEY")

            print(f"\n   Effect Classification: {effect}")

            if abs(fuel_gap_rrmse) > 20:
                print(f"   ⚠️  Prediction error increases by {abs(fuel_gap_rrmse):.1f}% "
                      f"for alternative fuel ships!")

            # Feature importance if available (Random Forest)
            if 'feature_importance' in metrics:
                print(f"\n   Feature Importance:")
                feature_imp = metrics['feature_importance']
                for feat, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
                    print(f"     {feat}: {imp:.3f}")

            summary.append({
                'target': target,
                'effect': effect,
                'fuel_gap_r2': fuel_gap_r2,
                'fuel_gap_rrmse': fuel_gap_rrmse,
                'alternative_rrmse': metrics['alternative_rrmse'],
                'best_predictors': combo_name
            })

        # Print summary table
        self._print_effect_summary_table(summary)

    def _print_effect_summary_table(self, summary: List[Dict]):
        """Print summary table of effects across all targets."""
        print(f"\n{'=' * 140}")
        print("SUMMARY TABLE")
        print(f"{'=' * 140}")

        header = (f"{'Dimension':<15s} | {'Effect':<10s} | "
                  f"{'Fuel Gap R²':>12s} | {'Fuel Gap RRMSE%':>16s} | "
                  f"{'Alt RRMSE%':>11s} | {'Best Predictors'}")
        print(header)
        print(self.fmt.format_separator(140))

        # Sort by RRMSE gap magnitude
        for item in sorted(summary, key=lambda x: abs(x['fuel_gap_rrmse']), reverse=True):
            print(f"{item['target']:<15s} | {item['effect']:<10s} | "
                  f"{self.fmt.format_gap(item['fuel_gap_r2'], 'r2'):>12s} | "
                  f"{self.fmt.format_gap(item['fuel_gap_rrmse'], 'rrmse'):>16s} | "
                  f"{self.fmt.format_metric(item['alternative_rrmse'], 'rrmse'):>11s} | "
                  f"{item['best_predictors']}")

        print(f"\n{'=' * 140}")
        print("INTERPRETATION GUIDE:")
        print("  R² Gap: How much variance explanation is lost")
        print("  RRMSE Gap: How much prediction error increases (in %)")
        print(f"  Thresholds: R² gap > 0.05 OR RRMSE gap > 10% = significant fuel effect")
        print("=" * 140)

    # ========================================================================
    # BEST RESULTS DISPLAY
    # ========================================================================

    def print_best_combinations(self,
                                metric: str = 'alternative_rrmse',
                                ascending: bool = True):
        """
        Print best combination for each target.

        Parameters
        ----------
        metric : str
            Metric to optimize ('alternative_rrmse', 'alternative_r2', etc.)
        ascending : bool
            If True, lower is better (for RRMSE). If False, higher is better (for R²)
        """
        if not hasattr(self.analyzer, 'results_df'):
            print("No results available. Run analyze_all_combinations() first.")
            return

        best_results = self.analyzer.get_best_combinations(metric=metric, ascending=ascending)

        print(self.fmt.format_header(f"BEST COMBINATIONS (by {metric})", width=140))

        header = (f"{'Target':<15s} | {'Predictors':<30s} | {'Model':<15s} | "
                  f"{'Alt R²':>8s} | {'Alt RRMSE%':>11s} | {'Fuel Gap R²':>12s} | "
                  f"{'Fuel Gap RRMSE':>15s}")
        print(header)
        print(self.fmt.format_separator(140))

        for _, row in best_results.iterrows():
            pred_str = self.fmt.format_predictor_string(row['predictors'], max_length=28)
            model = self.fmt.format_model_name(row['model_type'], max_length=15)

            print(f"{row['target']:<15s} | {pred_str:<30s} | {model:<15s} | "
                  f"{self.fmt.format_metric(row['alternative_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['alternative_rrmse'], 'rrmse'):>11s} | "
                  f"{self.fmt.format_gap(row['gap_fuel_type'], 'r2'):>12s} | "
                  f"{self.fmt.format_gap(row['gap_fuel_type_rrmse'], 'rrmse'):>15s}")

        print("=" * 140)

    # ========================================================================
    # COMPARISON DISPLAYS
    # ========================================================================

    def print_model_comparison(self, target: str):
        """
        Compare all tested combinations for a specific target.

        Parameters
        ----------
        target : str
            Target dimension to analyze
        """
        if not hasattr(self.analyzer, 'results_df'):
            print("No results available.")
            return

        target_df = self.analyzer.results_df[
            self.analyzer.results_df['target'] == target
            ]

        if target_df.empty:
            print(f"No results found for target '{target}'")
            return

        print(self.fmt.format_header(f"MODEL COMPARISON: {target}", width=140))

        # Sort by alternative RRMSE
        target_df = target_df.sort_values('alternative_rrmse')

        self._print_compact_table(target_df)

        # Show best vs worst
        best = target_df.iloc[0]
        worst = target_df.iloc[-1]

        print(f"\n{'=' * 140}")
        print(f"BEST:  {best['predictor_str']:<30s} | "
              f"Alt RRMSE: {self.fmt.format_metric(best['alternative_rrmse'], 'rrmse')}%")
        print(f"WORST: {worst['predictor_str']:<30s} | "
              f"Alt RRMSE: {self.fmt.format_metric(worst['alternative_rrmse'], 'rrmse')}%")
        print(f"IMPROVEMENT: {(worst['alternative_rrmse'] - best['alternative_rrmse']):.1f}% "
              f"by choosing optimal predictors")
        print("=" * 140)

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
                f"{fuel_abbr:>12s}_Match",
            f"{fuel_abbr:> 10s}_Alt",
            f"{fuel_abbr:> 10s}_DGap",  # Distribution Gap
            f"{fuel_abbr:> 10s}_FGap"])

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