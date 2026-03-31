"""
Display class for coefficient comparison analysis.

Handles all display logic for CoefficientComparisonAnalyzer,
keeping the analyzer class focused on computation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class CoefficientComparisonDisplay:
    """
    Handles all display logic for coefficient comparison analysis.

    Separates presentation from computation, making both more maintainable.
    """

    def __init__(self, analyzer):
        """
        Initialize display for coefficient analyzer.

        Parameters
        ----------
        analyzer : CoefficientComparisonAnalyzer
            Analyzer instance with results to display
        """
        self.analyzer = analyzer
        self.fmt = DisplayFormatters()

    # ========================================================================
    # INITIALIZATION DISPLAY
    # ========================================================================

    def print_initialization_summary(self):
        """Print summary at analyzer initialization."""
        analyzer = self.analyzer

        print("\n" + "=" * 120)
        print("COEFFICIENT COMPARISON ANALYZER INITIALIZED")
        print("=" * 120)
        print(f"  Conventional ships (full):    {len(analyzer.conventional_data)}")
        print(f"  Alternative ships:             {len(analyzer.alternative_data)}")
        print(f"  Matched datasets created:      {len(analyzer.matched_datasets)}")
        print(f"  Targets to analyze:            {len(analyzer.selector.target_dimensions)}")
        print(f"  Normalization:                 {'Enabled ✓' if analyzer.normalize else 'Disabled'}")

        if analyzer.normalize:
            print(f"  Method:                        {analyzer.normalization_method}")
            print(f"  Interpretation:                Coefficients in standardized units (std dev)")

        # Show sample matched dataset sizes
        if analyzer.matched_datasets:
            print(f"\n  Sample Matched Dataset Sizes:")
            for i, (combo, matched_df) in enumerate(list(analyzer.matched_datasets.items())[:3]):
                combo_str = ' + '.join(combo)
                print(f"    • {combo_str}: {len(matched_df)} ships")

            if len(analyzer.matched_datasets) > 3:
                print(f"    • ... and {len(analyzer.matched_datasets) - 3} more")

    # ========================================================================
    # ANALYSIS PROGRESS DISPLAY
    # ========================================================================

    def print_analysis_header(self):
        """Print header at start of analysis."""
        print("\n" + "=" * 120)
        print("THREE-WAY COEFFICIENT COMPARISON ANALYSIS")
        print("Full Conventional → Matched Conventional → Alternative")
        print("=" * 120)

    def print_target_header(self, target: str, predictors: tuple, model_type: str):
        """Print header for single target analysis."""
        print(f"\n{'=' * 120}")
        print(f"Target: {target}")
        print(f"{'=' * 120}")
        print(f"  Best combination: {' + '.join(predictors)}")
        print(f"  Best model type:  {model_type}")

    # ========================================================================
    # SINGLE TARGET RESULTS DISPLAY
    # ========================================================================

    def print_single_target_results(self,
                                    target: str,
                                    predictors: tuple,
                                    model_type: str,
                                    conv_full_coeffs: Dict,
                                    conv_matched_coeffs: Dict,
                                    alt_coeffs: Dict,
                                    coeff_comparison: Dict,
                                    conv_full_metrics: Dict,
                                    conv_matched_metrics: Dict,
                                    alt_metrics: Dict,
                                    cross_evals: Dict):
        """Print detailed results for single target."""

        # Model info
        print(f"\n  Model Form: {model_type}")
        if self.analyzer.normalize:
            print(f"  Coefficients: Standardized units (std dev)")

        # Sample sizes
        self._print_sample_sizes(conv_full_metrics, conv_matched_metrics, alt_metrics)

        # Performance comparison
        self._print_performance_comparison(
            conv_full_metrics, conv_matched_metrics, alt_metrics
        )

        # Cross-evaluations
        if cross_evals:
            self._print_cross_evaluations(cross_evals)

        # Coefficient comparison
        self._print_coefficient_table(
            conv_full_coeffs, conv_matched_coeffs, alt_coeffs, coeff_comparison
        )

        # Interpretation
        if 'interpretation' in coeff_comparison:
            print(f"\n  💡 {coeff_comparison['interpretation']}")

    def _print_sample_sizes(self, conv_full_metrics, conv_matched_metrics, alt_metrics):
        """Print sample size information."""
        print(f"\n  Sample Sizes:")
        print(f"    Full Conventional:    {conv_full_metrics['n_samples']}")
        print(f"    Matched Conventional: {conv_matched_metrics['n_samples']}")
        print(f"    Alternative:          {alt_metrics['n_samples']}")

    def _print_performance_comparison(self,
                                      conv_full_metrics,
                                      conv_matched_metrics,
                                      alt_metrics):
        """Print in-sample performance comparison."""
        print(f"\n  In-Sample Performance:")
        print(f"  {'Dataset':<20s} | {'R²':>8s} | {'RRMSE %':>10s}")
        print(f"  {'-' * 42}")

        datasets = [
            ('Full Conventional', conv_full_metrics),
            ('Matched Conventional', conv_matched_metrics),
            ('Alternative', alt_metrics)
        ]

        for name, metrics in datasets:
            r2_str = self.fmt.format_metric(metrics['r2'], 'r2')
            rrmse_str = self.fmt.format_metric(metrics['rrmse'], 'rrmse')
            print(f"  {name:<20s} | {r2_str:>8s} | {rrmse_str:>10s}")

    def _print_cross_evaluations(self, cross_evals: Dict):
        """Print cross-evaluation results."""
        print(f"\n  Cross-Evaluation:")
        print(f"  {'Model → Data':<30s} | {'R²':>8s} | {'RRMSE %':>10s}")
        print(f"  {'-' * 52}")

        eval_labels = {
            'full_on_matched': 'Full → Matched',
            'full_on_alt': 'Full → Alternative',
            'matched_on_alt': 'Matched → Alternative'
        }

        for key, metrics in cross_evals.items():
            label = eval_labels.get(key, key.replace('_', ' → ').title())
            r2_str = self.fmt.format_metric(metrics['r2'], 'r2')
            rrmse_str = self.fmt.format_metric(metrics['rrmse'], 'rrmse')
            print(f"  {label:<30s} | {r2_str:>8s} | {rrmse_str:>10s}")

    def _print_coefficient_table(self,
                                 conv_full_coeffs: Dict,
                                 conv_matched_coeffs: Dict,
                                 alt_coeffs: Dict,
                                 coeff_comparison: Dict):
        """Print three-way coefficient comparison table."""
        print(f"\n  Three-Way Coefficient Comparison:")

        if self.analyzer.normalize:
            header = (f"  {'Coefficient':<15s} | {'Full (std)':>12s} | "
                      f"{'Matched (std)':>14s} | {'Alt (std)':>12s} | "
                      f"{'Dist Δ':>10s} | {'Fuel Δ':>10s} | {'Total Δ':>10s}")
        else:
            header = (f"  {'Coefficient':<15s} | {'Full':>12s} | "
                      f"{'Matched':>14s} | {'Alternative':>12s} | "
                      f"{'Dist Δ':>10s} | {'Fuel Δ':>10s} | {'Total Δ':>10s}")

        print(header)
        print(f"  {'-' * 100}")

        # Get common coefficients
        common_keys = sorted(set(conv_full_coeffs.keys()).intersection(
            set(conv_matched_coeffs.keys()).intersection(set(alt_coeffs.keys()))
        ))

        for key in common_keys:
            if key not in coeff_comparison['distribution_effect']:
                continue

            dist = coeff_comparison['distribution_effect'][key]
            fuel = coeff_comparison['fuel_effect'][key]
            total = coeff_comparison['total_effect'][key]

            print(f"  {key:<15s} | {dist['from']:>12.6f} | {dist['to']:>14.6f} | "
                  f"{fuel['to']:>12.6f} | {dist['absolute_change']:>+10.6f} | "
                  f"{fuel['absolute_change']:>+10.6f} | {total['absolute_change']:>+10.6f}")

    # ========================================================================
    # SUMMARY TABLE DISPLAY
    # ========================================================================

    def print_summary_table(self, results_df: pd.DataFrame):
        """Print summary table of all results."""
        print("\n" + "=" * 140)
        print("SUMMARY: Three-Way Coefficient Comparison")
        print("=" * 140)

        print(f"\n{'Target':<15s} | {'Model':<15s} | "
              f"{'Full R²':>8s} | {'Match R²':>9s} | {'Alt R²':>8s} | "
              f"{'Full→Alt':>9s} | {'Match→Alt':>10s}")
        print("-" * 140)

        for _, row in results_df.iterrows():
            r2_total = row['alt_r2'] - row['conv_full_r2']
            r2_fuel = row['alt_r2'] - row['conv_matched_r2']

            print(f"{row['target']:<15s} | {row['model_type']:<15s} | "
                  f"{self.fmt.format_metric(row['conv_full_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_metric(row['conv_matched_r2'], 'r2'):>9s} | "
                  f"{self.fmt.format_metric(row['alt_r2'], 'r2'):>8s} | "
                  f"{self.fmt.format_gap(r2_total, 'r2'):>9s} | "
                  f"{self.fmt.format_gap(r2_fuel, 'r2'):>10s}")

        print("=" * 140)

        # Summary statistics
        self._print_summary_statistics(results_df)

    def _print_summary_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics."""
        print(f"\nOverall Statistics:")
        print(f"  Average R² (Full Conv):       "
              f"{self.fmt.format_metric(results_df['conv_full_r2'].mean(), 'r2')}")
        print(f"  Average R² (Matched Conv):    "
              f"{self.fmt.format_metric(results_df['conv_matched_r2'].mean(), 'r2')}")
        print(f"  Average R² (Alternative):     "
              f"{self.fmt.format_metric(results_df['alt_r2'].mean(), 'r2')}")

        avg_r2_total = (results_df['alt_r2'] - results_df['conv_full_r2']).mean()
        avg_r2_fuel = (results_df['alt_r2'] - results_df['conv_matched_r2']).mean()

        print(f"\n  Average R² change (Total):    {self.fmt.format_gap(avg_r2_total, 'r2')}")
        print(f"  Average R² change (Fuel):     {self.fmt.format_gap(avg_r2_fuel, 'r2')}")

        if abs(avg_r2_total) > 1e-10:
            fuel_contribution = 100 * abs(avg_r2_fuel) / abs(avg_r2_total)
            print(f"  Fuel effect contribution:     {self.fmt.format_percentage(fuel_contribution)}")

    # ========================================================================
    # DECOMPOSITION DISPLAY
    # ========================================================================

    def print_decomposition_summary(self, decomp_df: pd.DataFrame, threshold: float):
        """Print formatted decomposition summary."""
        print("\n" + "=" * 140)
        print("COEFFICIENT CHANGE DECOMPOSITION")
        print("Distribution Effect (Full → Matched) vs Fuel Effect (Matched → Alternative)")
        print("=" * 140)

        # Effect size thresholds
        print(f"\nEffect Size Thresholds:")
        print(f"  Negligible: < {threshold} std")
        print(f"  Small:      {threshold} - 0.5 std")
        print(f"  Medium:     0.5 - 0.8 std")
        print(f"  Large:      > 0.8 std")

        # Table header
        print("\n" + "-" * 140)
        print(f"{'':2s} {'Target':<15s} | {'Coefficient':<15s} | "
              f"{'Full':>8s} → {'Match':>8s} → {'Alt':>8s} | "
              f"{'Dist Δ':>8s} | {'Fuel Δ':>8s} | {'Fuel %':>7s} | {'Effect':<12s}")
        print("-" * 140)

        # Rows
        for _, row in decomp_df.iterrows():
            print(f"{row['emoji']:2s} {row['target']:<15s} | {row['coefficient']:<15s} | "
                  f"{row['full']:>8.3f} → {row['matched']:>8.3f} → {row['alternative']:>8.3f} | "
                  f"{row['distribution_effect']:>+8.3f} | {row['fuel_effect']:>+8.3f} | "
                  f"{self.fmt.format_percentage(row['fuel_pct'], decimals=1):>7s} | "
                  f"{row['fuel_effect_size']:<12s}")

        print("=" * 140)

        # Summary stats
        self._print_decomposition_statistics(decomp_df)

    def _print_decomposition_statistics(self, decomp_df: pd.DataFrame):
        """Print decomposition summary statistics."""
        print(f"\nSummary:")
        print(f"  Total coefficients analyzed:      {len(decomp_df)}")
        print(f"  Fuel effect significant:          "
              f"{decomp_df['fuel_significant'].sum()} "
              f"({decomp_df['fuel_significant'].sum() / len(decomp_df) * 100:.1f}%)")
        print(f"  Average fuel contribution:        "
              f"{self.fmt.format_percentage(decomp_df['fuel_pct'].mean())}")
        print(f"  Average distribution contribution: "
              f"{self.fmt.format_percentage(decomp_df['distribution_pct'].mean())}")

        print(f"\n  Effect Size Distribution:")
        for effect in ['Negligible', 'Small', 'Medium', 'Large']:
            count = (decomp_df['fuel_effect_size'] == effect).sum()
            pct = count / len(decomp_df) * 100
            print(f"    {effect:<12s}: {count} ({pct:.1f}%)")

    # ========================================================================
    # VALIDATION DISPLAY
    # ========================================================================

    def print_matching_quality(self, matched_datasets: Dict):
        """Print matching quality assessment."""
        from scipy import stats as scipy_stats

        print("\n" + "=" * 120)
        print("MATCHING QUALITY ASSESSMENT")
        print("=" * 120)
        print("\nKolmogorov-Smirnov Test (distributions should be similar):")
        print(f"{'Predictor':<20s} | {'KS Statistic':>12s} | {'p-value':>10s} | {'Quality':<15s}")
        print("-" * 120)

        for pred_combo, matched_data in list(matched_datasets.items())[:5]:
            pred_str = ' + '.join(pred_combo)

            for predictor in pred_combo:
                conv_matched = matched_data[predictor].dropna()
                alt = self.analyzer.alternative_data[predictor].dropna()

                if len(conv_matched) < 3 or len(alt) < 3:
                    continue

                ks_stat, p_value = scipy_stats.ks_2samp(conv_matched, alt)

                if p_value > 0.05:
                    quality = "✓ Good match"
                elif p_value > 0.01:
                    quality = "⚠️ Moderate"
                else:
                    quality = "✗ Poor match"

                print(f"{predictor:<20s} | {ks_stat:>12.3f} | {p_value:>10.3f} | {quality:<15s}")

        print("=" * 120)