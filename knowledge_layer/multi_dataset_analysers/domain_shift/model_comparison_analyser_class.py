"""
Model Comparison Analyzer
=========================

Unified comparison framework for Polynomial/Linear vs Random Forest models.

Supports:
1. Selector Comparison: Which predictor combinations work best?
2. Domain Shift Comparison: How well do models generalize to alternative fuel?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# DIMENSION_DISPLAY_NAMES = {
#     'BEAM (M)': 'Beam (m)',
#     'LOA (M)': 'LOA (m)',
#     'LBP (M)': 'LBP (m)',
#     'DRAFT (M)': 'Draft (m)',
#     'DEPTH (M)': 'Depth (m)',
#     'DWT': 'DWT',
#     'MCR (KW)': 'MCR (kW)',
#     'Vs': 'Vs',
#     'FUEL_CONSUMPTION': 'Fuel Consumption (t/day)',
#     'LBD':'LBD',
#     'LBT':'LBT'
#     # ... add more as needed
# }
#
#
# def get_display_name(dimension: str) -> str:
#     """
#     Get formatted display name for a dimension.
#
#     Parameters
#     ----------
#     dimension : str
#         Internal dimension name (e.g., 'BEAM_(M)')
#
#     Returns
#     -------
#     str
#         Formatted display name (e.g., 'Beam (m)')
#         Falls back to title-cased version if not in dictionary
#     """
#     return DIMENSION_DISPLAY_NAMES.get(
#         dimension,
#         dimension.replace('_', ' ').title()  # fallback formatting
#     )

class ModelComparisonAnalyzer:
    """
    Compare Polynomial/Linear regression vs Random Forest across multiple dimensions.

    This class provides a unified interface for comparing:
    - Model selection results (ConstrainedModelSelector vs RandomForestSelector)
    - Domain shift results (ConstrainedDomainAnalyzer vs RandomForestDomainAnalyzer)

    Key Features:
    ------------
    - Side-by-side performance comparison
    - Comprehensive visualization suite
    - Statistical significance testing
    - Export capabilities
    """

    def __init__(self,
                 polynomial_selector=None,
                 rf_selector=None,
                 polynomial_domain_analyzer=None,
                 rf_domain_analyzer=None,
                 verbose: bool = True):
        """
        Parameters
        ----------
        polynomial_selector : ConstrainedModelSelector, optional
            Polynomial/linear model selector
        rf_selector : RandomForestSelector, optional
            Random Forest selector
        polynomial_domain_analyzer : ConstrainedDomainAnalyzer, optional
            Polynomial/linear domain shift analyzer
        rf_domain_analyzer : RandomForestDomainAnalyzer, optional
            Random Forest domain shift analyzer
        verbose : bool
            Print detailed output
        """

        self.poly_selector = polynomial_selector
        self.rf_selector = rf_selector
        self.poly_domain = polynomial_domain_analyzer
        self.rf_domain = rf_domain_analyzer
        self.verbose = verbose

        # Validate inputs
        if not any([polynomial_selector, rf_selector,
                    polynomial_domain_analyzer, rf_domain_analyzer]):
            raise ValueError("Must provide at least one selector or domain analyzer")

        # Determine what comparisons are possible
        self.can_compare_selectors = (polynomial_selector is not None and
                                      rf_selector is not None)
        self.can_compare_domain = (polynomial_domain_analyzer is not None and
                                   rf_domain_analyzer is not None)

        if verbose:
            print("\n" + "=" * 100)
            print("MODEL COMPARISON ANALYZER INITIALIZED")
            print("=" * 100)
            print(f"  Selector comparison available: {self.can_compare_selectors}")
            print(f"  Domain shift comparison available: {self.can_compare_domain}")

    # ========================================================================
    # SELECTOR COMPARISON (Training Performance)
    # ========================================================================

    def _default_variable_groupings(self):

        variable_groupings = {
            'Hull dimensions': ['LBP (M)', 'LOA (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)'],
            'Capacity': ['DWT', 'GT', 'TEU','Np', 'Nv'],
            'Operational': ['Vs', 'MCR (KW)'],
        }

        return variable_groupings

    def _get_display_name(self,
                          variable: str,
                          variable_name_map:dict[str,str]) -> str:
        """
        Get formatted display name for a dimension.

        Parameters
        ----------
        variable : str
            Internal dimension name (e.g., 'BEAM_(M)')

        Returns
        -------
        str
            Formatted display name (e.g., 'Beam (m)')
            Falls back to title-cased version if not in dictionary
        """
        return variable_name_map.get(
            variable,
            variable.replace('_', ' ').title()  # fallback formatting
        )

    def compare_selectors(self, metric: str = 'rrmse') -> pd.DataFrame:
        """
        Compare polynomial vs RF selector performance.

        Parameters
        ----------
        metric : str
            Metric to compare ('rrmse', 'adj_r2', 'mape')

        Returns
        -------
        pd.DataFrame
            Comparison summary
        """

        if not self.can_compare_selectors:
            raise ValueError("Both selectors must be provided for comparison")

        # Get best combinations from each
        poly_best = self._get_poly_selector_best()
        rf_best = self.rf_selector.get_best_combinations()

        # Build comparison dataframe
        comparison_data = []

        for target in poly_best.keys():
            if target not in rf_best:
                continue

            poly_predictors, poly_metrics = poly_best[target]
            rf_predictors, rf_data = rf_best[target]
            rf_metrics = rf_data['metrics']

            comparison_data.append({
                'target': target,
                'poly_predictors': poly_predictors,
                'rf_predictors': rf_predictors,
                'poly_metric': poly_metrics[metric],
                'rf_metric': rf_metrics[metric],
                'rf_advantage': (poly_metrics[metric] - rf_metrics[metric]
                                 if metric == 'rrmse'
                                 else rf_metrics[metric] - poly_metrics[metric]),
                'better_method': 'RF' if (poly_metrics[metric] > rf_metrics[metric]
                                          if metric == 'rrmse'
                                          else rf_metrics[metric] > poly_metrics[metric]) else 'Polynomial'
            })

        comparison_df = pd.DataFrame(comparison_data)

        if self.verbose:
            print("\n" + "=" * 120)
            print(f"SELECTOR COMPARISON - {metric.upper()}")
            print("=" * 120)
            print(comparison_df.to_string(index=False))

            rf_wins = (comparison_df['better_method'] == 'RF').sum()
            total = len(comparison_df)
            print(f"\nRF wins: {rf_wins}/{total} ({rf_wins / total * 100:.1f}%)")

        return comparison_df

    def _get_poly_selector_best(self) -> Dict:
        """
        Extract best combinations from polynomial selector.

        **ENHANCED DEBUG VERSION**
        """
        best_combos = {}

        print("\n" + "=" * 80)
        print("DEBUG: _get_poly_selector_best()")
        print("=" * 80)

        # ========================================================================
        # VALIDATION
        # ========================================================================

        if not hasattr(self.poly_selector, 'model_recommendations'):
            if self.verbose:
                print("⚠️  Warning: Polynomial selector missing 'model_recommendations'")
            return best_combos

        if not hasattr(self.poly_selector, 'detailed_results'):
            if self.verbose:
                print("⚠️  Warning: Polynomial selector missing 'detailed_results'")
            return best_combos

        if not self.poly_selector.model_recommendations:
            if self.verbose:
                print("⚠️  Warning: Polynomial selector has no recommendations")
            return best_combos

        print(f"✅ Selector has {len(self.poly_selector.model_recommendations)} recommendations")
        print(f"✅ Selector has {len(self.poly_selector.detailed_results)} detailed_results entries")

        # ========================================================================
        # EXTRACTION
        # ========================================================================

        results_by_target = {}

        for (predictors, target), best_model_name in self.poly_selector.model_recommendations.items():
            key = (predictors, target)

            print(f"\n  Processing: {predictors} → {target}")
            print(f"    Recommended model: {best_model_name}")

            if key not in self.poly_selector.detailed_results:
                print(f"    ❌ No detailed_results for this key")
                continue

            model_results = self.poly_selector.detailed_results[key]
            print(f"    ✅ Found detailed_results with {len(model_results)} model types")

            if best_model_name not in model_results:
                print(f"    ❌ Model '{best_model_name}' not in results")
                print(f"       Available: {list(model_results.keys())}")
                continue

            metrics = model_results[best_model_name]
            print(f"    ✅ Got metrics for {best_model_name}")
            print(f"       Metrics keys: {list(metrics.keys())}")

            if 'rrmse' not in metrics:
                print(f"    ❌ No RRMSE in metrics")
                continue

            print(f"       RRMSE: {metrics['rrmse']:.4f}")

            combo_name = ' + '.join(predictors)

            if target not in results_by_target:
                results_by_target[target] = {}

            results_by_target[target][combo_name] = metrics

        # ========================================================================
        # SELECTION
        # ========================================================================

        print(f"\n{'=' * 80}")
        print(f"SELECTING BEST BY RRMSE FOR EACH TARGET")
        print(f"{'=' * 80}")

        for target, combos in results_by_target.items():
            if not combos:
                continue

            print(f"\nTarget: {target}")
            print(f"  Candidates: {len(combos)}")

            # Show all candidates sorted
            sorted_combos = sorted(combos.items(), key=lambda x: x[1].get('rrmse', float('inf')))
            for i, (combo_name, metrics) in enumerate(sorted_combos[:5], 1):
                marker = "✅" if i == 1 else "  "
                print(f"    {marker} {i}. {combo_name:30s} RRMSE={metrics['rrmse']:.4f}")

            # Select best
            best_combo_name = min(combos.keys(), key=lambda c: combos[c].get('rrmse', float('inf')))
            best_metrics = combos[best_combo_name]

            best_combos[target] = (best_combo_name, best_metrics)

            print(f"  → SELECTED: {best_combo_name} (RRMSE: {best_metrics['rrmse']:.4f})")

        print(f"\n{'=' * 80}")
        print(f"✅ Extracted {len(best_combos)} best combinations")
        print(f"{'=' * 80}\n")

        return best_combos

    # ========================================================================
    # DOMAIN SHIFT COMPARISON (Generalization Performance)
    # ========================================================================

    def compare_domain_shift(self,
                             selection_metric: str = 'test_rrmse',
                             verbose_selection: bool = False,
                         raise_on_empty: bool = False) -> pd.DataFrame:
        """
        Compare parametric vs Random Forest domain shift performance.

        For each target present in both analyzers' results, the best
        combination is selected by ``selection_metric`` and the two methods
        are compared on conventional test, matched, and alternative
        performance.

        An empty result is a legitimate outcome, not an error: when the
        alternative dataset for a fuel category contains zero ships (e.g. a
        pure-alternative subset that is empty after fuel reclassification),
        both analyzers legitimately have no results and there is nothing to
        compare. In that case an empty DataFrame is returned (with a clear
        console message) unless ``raise_on_empty=True``.

        :param selection_metric: Metric used to pick the best combination per
            target. One of ``'test_rrmse'``, ``'alternative_rrmse'``, or
            ``'selector'`` (use the original selector recommendations).
        :type selection_metric: str
        :param verbose_selection: Print per-target selection details.
        :type verbose_selection: bool
        :param raise_on_empty: If ``True``, raise :class:`ValueError` when no
            comparison rows can be generated. If ``False`` (default), return
            an empty DataFrame so batch pipelines can skip gracefully.
        :type raise_on_empty: bool
        :returns: One row per comparable target with test/alternative RRMSE,
            fuel-type gaps, and RF-advantage columns. Empty if there is no
            comparable data.
        :rtype: pandas.DataFrame
        :raises ValueError: If both domain analyzers were not provided, if
            ``selection_metric`` is invalid, or (when ``raise_on_empty=True``)
            if no comparison data could be generated.
        """

        if not self.can_compare_domain:
            raise ValueError("Both domain analyzers must be provided for comparison")

        print("\n" + "=" * 100)
        print("DEBUG: compare_domain_shift()")
        print("=" * 100)
        print(f"  selection_metric: {selection_metric}")
        print(f"  verbose_selection: {verbose_selection}")

        # Validate selection_metric
        valid_metrics = ['test_rrmse', 'alternative_rrmse','train_rrmse', 'selector']
        if selection_metric not in valid_metrics:
            raise ValueError(f"selection_metric must be one of {valid_metrics}, got '{selection_metric}'")

        poly_results = self.poly_domain.results
        rf_results = self.rf_domain.results

        common_targets = sorted(set(poly_results.keys()) & set(rf_results.keys()))

        print(f"\n  Poly domain analyzer has results for {len(poly_results)} targets")
        print(f"  RF domain analyzer has results for {len(rf_results)} targets")
        print(f"  Common targets: {sorted(set(poly_results.keys()) & set(rf_results.keys()))}")

        # ── EARLY EXIT: nothing to compare ─────────────────────────────────
        # This happens legitimately when the alternative dataset is empty
        # (e.g. zero pure-alternative-fuel ships after fuel reclassification).
        if not common_targets:
            msg = ("No common targets between parametric and RF analyzers — "
                   "likely an empty alternative dataset for this fuel category. "
                   "No comparison possible.")
            if raise_on_empty:
                raise ValueError(msg)
            print(f"\n  ⚠️  {msg}")
            print("  Returning empty DataFrame.\n")
            return pd.DataFrame()

        comparison_data = []

        # Get selector results if using 'selector' mode
        if selection_metric == 'selector':
            print(f"\n  Using 'selector' mode - getting selector results...")
            try:
                poly_selector_best = self._get_poly_selector_best()
                rf_selector_best = self.rf_selector.get_best_combinations()

                print(f"    Poly selector: {len(poly_selector_best)} targets")
                print(f"    RF selector: {len(rf_selector_best)} targets")
            except Exception as e:
                raise ValueError(f"Failed to get selector results: {e}") from e

        for target in poly_results.keys():
            if target not in rf_results:
                if self.verbose:
                    print(f"  ⚠️  Skipping {target}: not in RF results")
                continue

            print(f"\n{'·' * 100}")
            print(f"Processing target: {target}")
            print(f"{'·' * 100}")

            # ================================================================
            # SELECT BEST COMBINATION FOR THIS TARGET
            # ================================================================

            if selection_metric == 'selector':
                print(f"  Selection method: SELECTOR")
                poly_best, rf_best = self._select_from_selector(
                    target,
                    poly_selector_best,
                    rf_selector_best,
                    poly_results,
                    rf_results,
                    verbose_selection=True  # Always show selection details
                )
            else:
                print(f"  Selection method: ANALYZER ({selection_metric})")

                print(f"\n  Poly analyzer results for {target}:")
                print(f"    Type: {type(poly_results[target])}")
                print(f"    Length: {len(poly_results[target])}")
                if isinstance(poly_results[target], dict):
                    print(f"    Keys (sample): {list(poly_results[target].keys())[:3]}")

                poly_best = self._select_best_from_analyzer(
                    poly_results[target],
                    selection_metric,
                    'Parametric',
                    target,
                    verbose=True  # Always show selection details
                )

                print(f"\n  RF analyzer results for {target}:")
                print(f"    Type: {type(rf_results[target])}")
                print(f"    Length: {len(rf_results[target])}")
                if isinstance(rf_results[target], dict):
                    print(f"    Keys (sample): {list(rf_results[target].keys())[:3]}")

                rf_best = self._select_best_from_analyzer(
                    rf_results[target],
                    selection_metric,
                    'Random Forest',
                    target,
                    verbose=True
                )

            # Skip if either selection failed
            if poly_best is None or rf_best is None:
                print(f"  ❌ Selection failed for {target}")
                continue

            # ================================================================
            # EXTRACT METRICS
            # ================================================================

            print(f"\n  Extracting metrics from selected combinations:")

            print(f"\n  Poly best keys: {list(poly_best.keys())}")
            print(f"  RF best keys: {list(rf_best.keys())}")

            # Conventional performance
            poly_test_rrmse = poly_best.get('test_rrmse', np.nan)
            rf_test_rrmse = rf_best.get('test_rrmse', np.nan)

            print(f"\n  Conventional Test RRMSE:")
            print(f"    Poly: {poly_test_rrmse:.4f}")
            print(f"    RF:   {rf_test_rrmse:.4f}")

            # Alternative performance
            poly_alt_rrmse = poly_best.get('alternative_rrmse', np.nan)
            rf_alt_rrmse = rf_best.get('alternative_rrmse', np.nan)

            print(f"\n  Alternative RRMSE:")
            print(f"    Poly: {poly_alt_rrmse:.4f}")
            print(f"    RF:   {rf_alt_rrmse:.4f}")

            # Fuel gaps
            poly_fuel_gap = poly_best.get('gap_fuel_type_rrmse', np.nan)
            rf_fuel_gap = rf_best.get('gap_fuel_type_rrmse', np.nan)

            print(f"\n  Fuel-Type Gap RRMSE:")
            print(f"    Poly: {poly_fuel_gap:+.4f}")
            print(f"    RF:   {rf_fuel_gap:+.4f}")

            # Predictor names
            poly_preds = poly_best.get('predictors', [])
            rf_preds = rf_best.get('predictors', [])

            if isinstance(poly_preds, tuple):
                poly_preds = list(poly_preds)
            if isinstance(rf_preds, tuple):
                rf_preds = list(rf_preds)

            poly_pred_str = ' + '.join(poly_preds) if poly_preds else 'Unknown'
            rf_pred_str = ' + '.join(rf_preds) if rf_preds else 'Unknown'

            print(f"\n  Predictors:")
            print(f"    Poly: {poly_pred_str}")
            print(f"    RF:   {rf_pred_str}")

            # Calculate advantages
            rf_advantage_test = poly_test_rrmse - rf_test_rrmse

            if not np.isnan(poly_alt_rrmse) and not np.isnan(rf_alt_rrmse):
                rf_advantage_alt = poly_alt_rrmse - rf_alt_rrmse
                better_on_alt = 'RF' if rf_alt_rrmse < poly_alt_rrmse else 'Parametric'
                testable = True
            else:
                rf_advantage_alt = np.nan
                better_on_alt = 'Unknown'
                testable = False

            better_on_test = 'RF' if rf_test_rrmse < poly_test_rrmse else 'Parametric'

            print(f"\n  Advantages:")
            print(f"    RF advantage (test): {rf_advantage_test:+.4f}")
            print(f"    RF advantage (alt):  {rf_advantage_alt:+.4f}")
            print(f"    Better on test: {better_on_test}")
            print(f"    Better on alt:  {better_on_alt}")

            # Store results
            comparison_data.append({
                'target': target,
                'poly_predictors': poly_pred_str,
                'rf_predictors': rf_pred_str,
                'n_poly_predictors': len(poly_preds),
                'n_rf_predictors': len(rf_preds),

                'poly_test_rrmse': poly_test_rrmse,
                'rf_test_rrmse': rf_test_rrmse,
                'rf_advantage_test': rf_advantage_test,
                'better_on_test': better_on_test,

                'poly_alt_rrmse': poly_alt_rrmse,
                'rf_alt_rrmse': rf_alt_rrmse,
                'rf_advantage_alt': rf_advantage_alt,
                'better_on_alt': better_on_alt,

                'poly_fuel_gap': poly_fuel_gap,
                'rf_fuel_gap': rf_fuel_gap,

                'poly_train_rrmse': poly_best.get('train_rrmse', np.nan),
                'rf_train_rrmse': rf_best.get('train_rrmse', np.nan),
                'poly_matched_rrmse': poly_best.get('matched_rrmse', np.nan),
                'rf_matched_rrmse': rf_best.get('matched_rrmse', np.nan),

                'testable_on_alternative': testable,
                'selection_metric_used': selection_metric
            })

        if not comparison_data:
            raise ValueError("No valid comparison data generated")

        comparison_df = pd.DataFrame(comparison_data)

        if self.verbose:
            self._print_comparison_summary(comparison_df, selection_metric)

        return comparison_df

        # ================================================================
        # DISPLAY RESULTS
        # ================================================================

        if self.verbose:
            self._print_comparison_summary(comparison_df, selection_metric)

        return comparison_df

    def _select_from_selector(self,
                              target: str,
                              poly_selector_best: Dict,
                              rf_selector_best: Dict,
                              poly_analyzer_results: Dict,
                              rf_analyzer_results: Dict,
                              verbose: bool) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Select combinations using selector's original recommendations.

        Returns
        -------
        tuple of (poly_best_dict, rf_best_dict) or (None, None)
        """
        # Check if target in selector results
        if target not in poly_selector_best:
            if self.verbose:
                print(f"⚠️  {target}: not in parametric selector results")
            return None, None

        if target not in rf_selector_best:
            if self.verbose:
                print(f"⚠️  {target}: not in RF selector results")
            return None, None

        # Get selector's choices
        poly_combo_str, poly_selector_metrics = poly_selector_best[target]
        rf_combo_data = rf_selector_best[target]

        # RF selector returns (combo_str, data_dict)
        if isinstance(rf_combo_data, tuple) and len(rf_combo_data) == 2:
            rf_combo_str, _ = rf_combo_data
        else:
            # Might be different format
            rf_combo_str = ' + '.join(rf_combo_data.get('predictors', []))

        if verbose:
            print(f"\n{target}:")
            print(f"  Selector chose: Poly={poly_combo_str}, RF={rf_combo_str}")

        # Find these combinations in analyzer results
        poly_best = self._find_combination_in_analyzer(
            poly_analyzer_results,
            target,
            poly_combo_str
        )

        rf_best = self._find_combination_in_analyzer(
            rf_analyzer_results,
            target,
            rf_combo_str
        )

        # Check if found
        if poly_best is None:
            if self.verbose:
                print(f"⚠️  {target}: Parametric selector's choice '{poly_combo_str}' "
                      f"not found in analyzer results (not testable)")
            return None, None

        if rf_best is None:
            if self.verbose:
                print(f"⚠️  {target}: RF selector's choice '{rf_combo_str}' "
                      f"not found in analyzer results (not testable)")
            return None, None

        if verbose:
            print(f"  Found in analyzer: test_rrmse Poly={poly_best.get('test_rrmse', 'N/A'):.2f}, "
                  f"RF={rf_best.get('test_rrmse', 'N/A'):.2f}")

        return poly_best, rf_best

    def _select_best_from_analyzer(self,
                                   target_results: Dict,
                                   selection_metric: str,
                                   model_name: str,
                                   target: str,
                                   verbose: bool) -> Optional[Dict]:
        """
        Select best combination from analyzer results by specified metric.

        **ENHANCED DEBUG VERSION**

        This is a helper method for compare_domain_shift() when selection_metric
        is 'test_rrmse' or 'alternative_rrmse' (not 'selector').
        """

        print(f"\n  {'─' * 70}")
        print(f"  _select_best_from_analyzer() for {model_name}")
        print(f"  {'─' * 70}")
        print(f"    Target: {target}")
        print(f"    Selection metric: {selection_metric}")

        if not target_results:
            print(f"    ❌ No results available")
            return None

        # Handle both dict and list structures
        if isinstance(target_results, dict):
            results_list = list(target_results.values())
            print(f"    Format: dict with {len(results_list)} combinations")
        elif isinstance(target_results, list):
            results_list = target_results
            print(f"    Format: list with {len(results_list)} combinations")
        else:
            print(f"    ❌ Unexpected structure: {type(target_results)}")
            return None

        if not results_list:
            print(f"    ❌ Results list is empty")
            return None

        # Show available metrics in first result
        if results_list:
            print(f"    Available metrics (sample): {list(results_list[0].keys())[:10]}")

        # Filter out results with missing selection metric
        valid_results = [
            r for r in results_list
            if not np.isnan(r.get(selection_metric, np.nan))
        ]

        print(f"    Valid results (have {selection_metric}): {len(valid_results)}/{len(results_list)}")

        if not valid_results:
            print(f"    ❌ No results with valid {selection_metric}")
            return None

        # Show all candidates sorted
        print(f"\n    All candidates (sorted by {selection_metric}):")
        sorted_valid = sorted(valid_results, key=lambda x: x[selection_metric])
        for i, r in enumerate(sorted_valid[:5], 1):
            preds = r.get('predictors', [])
            pred_str = ' + '.join(preds) if preds else 'Unknown'
            metric_val = r[selection_metric]
            marker = "✅" if i == 1 else "  "
            print(f"      {marker} {i}. {pred_str:30s} {selection_metric}={metric_val:.4f}")

        if len(sorted_valid) > 5:
            print(f"      ... and {len(sorted_valid) - 5} more")

        # Select best
        best = min(valid_results, key=lambda x: x[selection_metric])

        preds = best.get('predictors', [])
        pred_str = ' + '.join(preds) if preds else 'Unknown'
        metric_val = best[selection_metric]

        print(f"\n    ✅ SELECTED: {pred_str}")
        print(f"       {selection_metric}: {metric_val:.4f}")

        # Show all metrics for selected
        print(f"       All metrics:")
        for key, val in best.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                print(f"         {key:30s}: {val:.4f}")

        return best

    def _find_combination_in_analyzer(self,
                                      analyzer_results: Dict,
                                      target: str,
                                      combo_str: str) -> Optional[Dict]:
        """
        Find specific combination in analyzer results.

        Parameters
        ----------
        analyzer_results : dict
            Full analyzer results (all targets)
        target : str
            Target dimension
        combo_str : str
            Combination string like "DWT + GT" or "LBP (M)"

        Returns
        -------
        dict or None
            Metrics for this combination, or None if not found
        """
        if target not in analyzer_results:
            return None

        target_results = analyzer_results[target]

        # Handle dict structure
        if isinstance(target_results, dict):
            # Try direct key match
            if combo_str in target_results:
                return target_results[combo_str]

            # Try matching by predictors
            for key, metrics in target_results.items():
                if 'predictors' in metrics:
                    preds = metrics['predictors']
                    if isinstance(preds, (list, tuple)):
                        stored_combo_str = ' + '.join(preds)
                    else:
                        stored_combo_str = str(preds)

                    if stored_combo_str == combo_str:
                        return metrics

        # Handle list structure
        elif isinstance(target_results, list):
            for metrics in target_results:
                # Try predictor_str field
                if 'predictor_str' in metrics and metrics['predictor_str'] == combo_str:
                    return metrics

                # Try building from predictors
                if 'predictors' in metrics:
                    preds = metrics['predictors']
                    if isinstance(preds, (list, tuple)):
                        stored_combo_str = ' + '.join(preds)
                        if stored_combo_str == combo_str:
                            return metrics

        return None

    def _print_comparison_summary(self,
                                  comparison_df: pd.DataFrame,
                                  selection_metric: str):
        """Print formatted comparison summary."""

        print("\n" + "=" * 140)
        print("DOMAIN SHIFT COMPARISON - Alternative Fuel Prediction")
        print(f"Selection based on: {selection_metric.upper()}")
        print("=" * 140)

        # Main comparison table
        display_cols = [
            'target', 'poly_predictors', 'rf_predictors',
            'poly_test_rrmse', 'rf_test_rrmse',
            'poly_alt_rrmse', 'rf_alt_rrmse',
            'poly_fuel_gap', 'rf_fuel_gap',
            'rf_advantage_alt', 'better_on_alt'
        ]

        # Only show columns that exist
        display_cols = [c for c in display_cols if c in comparison_df.columns]

        print(comparison_df[display_cols].to_string(index=False))

        # Summary statistics
        print("\n" + "=" * 140)

        testable = comparison_df['testable_on_alternative'].sum()
        total = len(comparison_df)

        print(f"Testable on alternative: {testable}/{total} targets")

        if testable > 0:
            testable_df = comparison_df[comparison_df['testable_on_alternative']]

            # Test performance
            rf_wins_test = (testable_df['better_on_test'] == 'RF').sum()
            avg_rf_adv_test = testable_df['rf_advantage_test'].mean()

            print(f"\nConventional Test Performance:")
            print(f"  RF better: {rf_wins_test}/{testable} ({rf_wins_test / testable * 100:.1f}%)")
            print(f"  Average RF advantage: {avg_rf_adv_test:+.2f}% RRMSE")

            # Alternative performance
            rf_wins_alt = (testable_df['better_on_alt'] == 'RF').sum()
            avg_rf_adv_alt = testable_df['rf_advantage_alt'].mean()

            print(f"\nAlternative Fuel Performance:")
            print(f"  RF better: {rf_wins_alt}/{testable} ({rf_wins_alt / testable * 100:.1f}%)")
            print(f"  Average RF advantage: {avg_rf_adv_alt:+.2f}% RRMSE")

            # Fuel gaps
            avg_poly_gap = testable_df['poly_fuel_gap'].mean()
            avg_rf_gap = testable_df['rf_fuel_gap'].mean()

            print(f"\nFuel-Type Effect (Gap):")
            print(f"  Parametric avg gap: {avg_poly_gap:+.2f}% RRMSE")
            print(f"  Random Forest avg gap: {avg_rf_gap:+.2f}% RRMSE")

        print("=" * 140)

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================

    def plot_selector_comparison(self,
                                 metric: str = 'rrmse',
                                 figsize: Tuple[int, int] = (14, 6),
                                 save_path: Optional[str] = None):
        """
        Plot side-by-side comparison of selector performance.

        Parameters
        ----------
        metric : str
            Metric to compare
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """

        if not self.can_compare_selectors:
            print("Cannot plot: both selectors required")
            return

        comparison_df = self.compare_selectors(metric=metric)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Bar comparison
        targets = comparison_df['target'].values
        x = np.arange(len(targets))
        width = 0.35

        poly_values = comparison_df['poly_metric'].values
        rf_values = comparison_df['rf_metric'].values

        bars1 = ax1.bar(x - width / 2, poly_values, width,
                        label='Polynomial', color='royalblue', alpha=0.8,
                        edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width / 2, rf_values, width,
                        label='Random Forest', color='forestgreen', alpha=0.8,
                        edgecolor='black', linewidth=1)

        ax1.set_xlabel('Target Dimension', fontsize=12)

        metric_labels = {
            'adj_r2': 'Adjusted R²',
            'rrmse': 'NRMSE (%)',
            'mape': 'MAPE (%)'
        }
        ax1.set_ylabel(metric_labels.get(metric, metric),
                       fontsize=12)
        ax1.set_title('Model Selection Performance',
                      fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}',
                         ha='center', va='bottom', fontsize=8)

        # Plot 2: Advantage chart
        advantages = comparison_df['rf_advantage'].values
        colors = ['green' if x > 0 else 'red' for x in advantages]

        bars3 = ax2.barh(targets, advantages, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1)

        ax2.set_xlabel(f'RF Advantage ({metric.upper()})',
                       fontsize=12, fontweight='bold')
        ax2.set_title('RF Performance Advantage',
                      fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, advantages)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{val:+.2f}',
                     ha='left' if width > 0 else 'right',
                     va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Selector comparison plot saved to {save_path}")

        plt.show()

    def plot_domain_shift_comparison(
            self,
            figsize: Tuple[int, int] = (18, 11),
            save_path: Optional[str] = None,
            shiptype: 'ShipType|None' = None,
            dataset: str | None = None,
            rrmse_threshold: float | None = None,
            variable_groupings: dict[str, List[str]]|None = None,
            variable_name_map:dict[str, str] | None = None,
            threshold_metric: Optional[str] = 'median',
            # ── layout ──────────────────────────────────────────────────────
            hspace: float = 0.45,
            wspace: float = 0.30,
            height_ratios: Tuple[float, float, float] = (1.0, 1.4, 1.4),
            gridspec_top: float = 0.93,  # ← NEW: controls gap below suptitle
            # ── text ────────────────────────────────────────────────────────
            base_font_size: int = 9,
            tick_font_size: Optional[int] = None,  # ← NEW: overrides base for x-ticks
            bar_label_font_size: Optional[int] = None,  # ← NEW: overrides base for bar labels
            # ── bars ────────────────────────────────────────────────────────
            bar_width_scale: float = 0.80,
            show_bar_labels: bool = True,
            debug: bool = False,
    ):
        """
        Comprehensive domain shift comparison visualisation.

        Shows:

        1. Alternative fuel NRMSE comparison (top-left panel).
        2. RF advantage over parametric (top-right panel).
        3. Train → Test → Matched → Alternative cascade for parametric models.
        4. Train → Test → Matched → Alternative cascade for Random Forest models.

        Size and density are controlled through explicit layout/text/bar
        parameters so the caller can tune the figure without touching internal
        logic.

        :param figsize: Overall figure dimensions ``(width_in, height_in)``.
        :type figsize: tuple[int, int]
        :param save_path: File path to write the figure to (optional).
        :type save_path: str or None
        :param shiptype: Ship type object used for axis/title labelling.
        :param dataset: Dataset label string (underscores replaced, title-cased).
        :type dataset: str or None
        :param rrmse_threshold: Global NRMSE threshold drawn as a horizontal
            dashed line on the cascade panels when ``variable_groupings`` is
            ``None``.
        :type rrmse_threshold: float or None
        :param variable_groupings: Mapping of category name → list of target
            variable names. When provided, per-category reference lines are
            drawn instead of a single global threshold.
        :type variable_groupings: dict[str, list[str]] or None
        :param threshold_metric: Aggregation used for per-category reference
            lines: ``'median'``, ``'mean'``, or ``'best'``.
        :type threshold_metric: str or None
        :param hspace: Vertical space between subplot rows as a fraction of the
            average subplot height. Decrease to tighten rows.
        :type hspace: float
        :param wspace: Horizontal space between subplot columns. Decrease to
            tighten columns.
        :type wspace: float
        :param height_ratios: Relative heights of the three subplot rows.
            Increase the cascade row values relative to the summary row to give
            cascade panels more vertical room.
        :type height_ratios: tuple[float, float, float]
        :param base_font_size: Base font size in points. All text elements in
            the figure are scaled relative to this value, so changing it once
            resizes everything consistently.
        :type base_font_size: int
        :param bar_width_scale: Fraction of the available inter-group space
            taken up by the bar cluster (0–1). Reduce below 0.8 to add
            whitespace between groups.
        :type bar_width_scale: float
        :param show_bar_labels: Whether to annotate each bar with its numeric
            value. Disabling this substantially reduces clutter on small figures.
        :type show_bar_labels: bool
        :param debug: Print intermediate sizing values to stdout.
        :type debug: bool
        """

        if variable_groupings is None:
            variable_groupings = self._default_variable_groupings()

        if dataset:
            dataset = dataset.replace('_', ' ').title()

        ship_type_str = shiptype.string if shiptype else "Ship"

        if not self.can_compare_domain:
            print("Cannot plot: both domain analyzers required")
            return

        # ── derive font sizes from base, with optional overrides ────────────
        fs = {
            'bar_label': bar_label_font_size if bar_label_font_size is not None else base_font_size - 2,
            'tick': tick_font_size if tick_font_size is not None else base_font_size - 1,
            'axis_label': base_font_size,
            'legend': base_font_size - 1,
            'subtitle': base_font_size + 1,
            'suptitle': base_font_size + 3,
            'textbox': base_font_size - 2,
        }

        if debug:
            print(f"[plot_domain_shift_comparison] figsize={figsize}")
            print(f"[plot_domain_shift_comparison] font sizes: {fs}")
            print(f"[plot_domain_shift_comparison] hspace={hspace}, wspace={wspace}")
            print(f"[plot_domain_shift_comparison] height_ratios={height_ratios}")
            print(f"[plot_domain_shift_comparison] bar_width_scale={bar_width_scale}, "
                  f"show_bar_labels={show_bar_labels}")

        comparison_df = self.compare_domain_shift()

        # ── EARLY EXIT: nothing to plot ──────────────────────────────────
        if comparison_df.empty:
            print(f"⚠️  No comparison data for {ship_type_str} {dataset or ''} — "
                  f"skipping domain shift comparison plot.")
            return

        # ── figure / gridspec ───────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            3, 3,
            hspace=hspace,
            wspace=wspace,
            height_ratios=list(height_ratios),
            top=gridspec_top,  # ← leaves a fixed band at the top for suptitle
        )

        targets = comparison_df['target'].values
        x = np.arange(len(targets))
        bar_pair_w = bar_width_scale / 2  # width of each bar in a side-by-side pair

        if variable_name_map is not None:
            display_names = [self._get_display_name(t, variable_name_map) for t in targets]
        else:
            display_names = targets

        # ── panel 1: alternative fuel NRMSE ─────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])

        poly_alt = comparison_df['poly_alt_rrmse'].values
        rf_alt = comparison_df['rf_alt_rrmse'].values

        bars1 = ax1.bar(x - bar_pair_w / 2, poly_alt, bar_pair_w,
                        label='Parametric', color='royalblue', alpha=0.8,
                        edgecolor='black')
        bars2 = ax1.bar(x + bar_pair_w / 2, rf_alt, bar_pair_w,
                        label='Random Forest', color='forestgreen', alpha=0.8,
                        edgecolor='black')

        ax1.set_xlabel('Target Dimension', fontsize=fs['axis_label'])
        ax1.set_ylabel(f'NRMSE (%)', fontsize=fs['axis_label'])
        ax1.set_title(f'{ship_type_str} {dataset} Dataset Prediction Accuracy',
                      fontsize=fs['subtitle'], fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names,ha='right',
                            fontsize=fs['tick'])
        ax1.legend(fontsize=fs['legend'])
        ax1.grid(axis='y', alpha=0.3)

        if show_bar_labels:
            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., h,
                             f'{h:.1f}', ha='center', va='bottom',
                             fontsize=fs['bar_label'])

        if debug:
            print(f"[panel 1] poly_alt range: {poly_alt.min():.1f}–{poly_alt.max():.1f}")
            print(f"[panel 1] rf_alt  range: {rf_alt.min():.1f}–{rf_alt.max():.1f}")

        # ── panel 2: RF advantage ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])

        advantages = comparison_df['rf_advantage_alt'].values
        bar_colors = ['green' if v > 0 else 'red' for v in advantages]

        bars3 = ax2.barh(display_names, advantages, color=bar_colors,
                         alpha=0.7, edgecolor='black')

        ax2.set_xlabel('NRMSE Improvement', fontweight='bold', fontsize=fs['axis_label'])
        ax2.set_title('RF Advantage\n(Lower is Better)',
                      fontsize=fs['subtitle'], fontweight='bold')
        ax2.tick_params(axis='y', labelsize=fs['tick'])
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.grid(axis='x', alpha=0.3)

        if show_bar_labels:
            for bar, val in zip(bars3, advantages):
                w = bar.get_width()
                ax2.text(w, bar.get_y() + bar.get_height() / 2.,
                         f'{val:+.1f}',
                         ha='left' if w > 0 else 'right',
                         va='center', fontsize=fs['bar_label'], fontweight='bold')

        # ── cascade data & per-category thresholds ───────────────────────────
        poly_cascade = self._extract_poly_cascade_data()
        rf_cascade = self._extract_rf_cascade_data()

        poly_thresholds = None
        rf_thresholds = None

        if variable_groupings is not None and threshold_metric is not None:
            poly_thresholds = self._compute_cascade_thresholds(
                cascade_data=poly_cascade,
                variable_groupings=variable_groupings,
                metric_type=threshold_metric,
            )
            rf_thresholds = self._compute_cascade_thresholds(
                cascade_data=rf_cascade,
                variable_groupings=variable_groupings,
                metric_type=threshold_metric,
            )

            if debug:
                print(f"[thresholds] poly: {poly_thresholds}")
                print(f"[thresholds] rf:   {rf_thresholds}")

        # ── panel 3: parametric cascade ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_cascade(
            ax3, poly_cascade, 'Parametric', 'royalblue',
            rrmse_threshold=rrmse_threshold,
            category_thresholds=poly_thresholds,
            variable_groupings=variable_groupings,
            variable_name_map=variable_name_map,
            threshold_metric=threshold_metric,
            base_font_size=base_font_size,
            tick_font_size=tick_font_size,  # ← pass through
            bar_label_font_size=bar_label_font_size,  # ← pass through
            bar_width_scale=bar_width_scale,
            show_bar_labels=show_bar_labels,
            debug=debug,
        )

        # ── panel 4: RF cascade ──────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_cascade(
            ax4, rf_cascade, 'Random Forest', 'forestgreen',
            rrmse_threshold=rrmse_threshold,
            category_thresholds=rf_thresholds,
            variable_groupings=variable_groupings,
            variable_name_map=variable_name_map,
            threshold_metric=threshold_metric,
            base_font_size=base_font_size,
            tick_font_size=tick_font_size,  # ← pass through
            bar_label_font_size=bar_label_font_size,  # ← pass through
            bar_width_scale=bar_width_scale,
            show_bar_labels=show_bar_labels,
            debug=debug,
        )

        # ── suptitle ─────────────────────────────────────────────────────────
        title = (f'{ship_type_str} Performance Shift Analysis: '
                 f'Parametric vs Random Forest')
        if dataset:
            title += f' ({dataset})'
        fig.suptitle(title, fontsize=fs['suptitle'], fontweight='bold', y=1.0)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Performance shift comparison plot saved to {save_path}")

        plt.show()

    def _extract_poly_cascade_data(self) -> Dict:
        """Extract train→test→matched→alt cascade for polynomial."""
        cascade = {}

        for target, models in self.poly_domain.results.items():
            # Check if models dict is not empty
            if not models or not models.values():
                if self.verbose:
                    print(f"Warning: No polynomial results found for target '{target}', skipping")
                continue

            # ✅ FIX: Select by test_rrmse for consistency
            best = min(models.values(),
                       key=lambda x: x.get('test_rrmse', float('inf')))

            # Check if all required keys exist
            required_keys = ['train_rrmse', 'test_rrmse', 'matched_rrmse', 'alternative_rrmse']
            if not all(key in best for key in required_keys):
                if self.verbose:
                    print(f"Warning: Missing cascade values for target '{target}', skipping")
                continue

            cascade[target] = {
                'train': best['train_rrmse'],
                'test': best['test_rrmse'],
                'matched': best['matched_rrmse'],
                'alternative': best['alternative_rrmse']
            }

        return cascade

    def _extract_rf_cascade_data(self) -> Dict:
        """Extract train→test→matched→alt cascade for RF."""
        cascade = {}

        for target, rf_target_results in self.rf_domain.results.items():

            # Check if results exist and are not empty
            if not rf_target_results:
                if self.verbose:
                    print(f"Warning: No RF results found for target '{target}', skipping")
                continue

            # Handle both dict and list structures
            if isinstance(rf_target_results, dict):
                # Check if dict has values
                if not rf_target_results.values():
                    if self.verbose:
                        print(f"Warning: Empty RF results dict for target '{target}', skipping")
                    continue

                # ✅ FIX: Select by test_rrmse for consistency
                best = min(rf_target_results.values(),
                           key=lambda x: x.get('test_rrmse', float('inf')))

            elif isinstance(rf_target_results, list):
                # Check if list is empty
                if not rf_target_results:
                    if self.verbose:
                        print(f"Warning: Empty RF results list for target '{target}', skipping")
                    continue

                # ✅ FIX: Select by test_rrmse for consistency
                best = min(rf_target_results,
                           key=lambda x: x.get('test_rrmse', float('inf')))
            else:
                if self.verbose:
                    print(
                        f"Warning: Unexpected RF results structure for target '{target}': {type(rf_target_results)}, skipping")
                continue

            # Extract values with flexible key names
            rf_train = best.get('rf_train_rrmse', best.get('train_rrmse', None))
            rf_test = best.get('test_rrmse', None)
            rf_matched = best.get('rf_matched_rrmse', best.get('matched_rrmse', None))
            rf_alt = best.get('rf_alt_rrmse', best.get('alternative_rrmse', None))

            # Only add to cascade if we have all required values
            if rf_train is None or rf_test is None or rf_matched is None or rf_alt is None:
                if self.verbose:
                    print(f"Warning: Missing cascade values for target '{target}', skipping")
                continue

            cascade[target] = {
                'train': rf_train,
                'test': rf_test,
                'matched': rf_matched,
                'alternative': rf_alt
            }

        return cascade

    def _plot_cascade(
            self,
            ax,
            cascade_data: Dict,
            method_name: str,
            color: str,
            rrmse_threshold: float | None = None,
            category_thresholds: Optional[Dict[str, float]] = None,
            variable_groupings: dict[str, List[str]] = None,
            variable_name_map: dict[str, str]|None = None,
            threshold_metric: str = 'median',
            base_font_size: int = 9,
            tick_font_size: Optional[int] = None,  # ← NEW
            bar_label_font_size: Optional[int] = None,  # ← NEW
            bar_width_scale: float = 0.80,
            show_bar_labels: bool = True,
            debug: bool = False,
    ):
        """
        Render the four-stage (Train → Test → Matched → Alternative) cascade
        bar chart onto a pre-existing axes object.

        The four stages are shown as grouped bars for every target variable.
        Bars within each group use increasing alpha to visually encode the
        progression from training to out-of-domain performance.

        Reference lines can be drawn as either a single global threshold or
        per-category lines derived from the conventional test NRMSE values.

        :param ax: Matplotlib axes to draw on.
        :param cascade_data: Nested dict
            ``{target: {'train': float, 'test': float,
            'matched': float, 'alternative': float}}``.
        :type cascade_data: dict
        :param method_name: Label for the subplot title (e.g. ``'Parametric'``).
        :type method_name: str
        :param color: Base matplotlib colour string for all bars in this panel.
        :type color: str
        :param rrmse_threshold: Single global threshold line (only used when
            ``category_thresholds`` is ``None``).
        :type rrmse_threshold: float or None
        :param category_thresholds: Per-category threshold values as returned
            by :meth:`_compute_cascade_thresholds`. When provided, overrides
            ``rrmse_threshold``.
        :type category_thresholds: dict[str, float] or None
        :param variable_groupings: Category → variable-list mapping, needed to
            position per-category threshold lines correctly.
        :type variable_groupings: dict[str, list[str]] or None
        :param variable_name_map: Mapping from variable names to publishing name
        :param threshold_metric: Label used in the text-box annotation.
        :type threshold_metric: str
        :param base_font_size: Base font size; all text is offset from this.
        :type base_font_size: int
        :param bar_width_scale: Fraction of available group width used by bars.
        :type bar_width_scale: float
        :param show_bar_labels: Whether to print numeric values above bars.
        :type show_bar_labels: bool
        :param debug: Print per-bar values and threshold positions to stdout.
        :type debug: bool
        """

        # ── font size offsets ────────────────────────────────────────────────
        fs = {
            'bar_label': bar_label_font_size if bar_label_font_size is not None else base_font_size - 2,
            'tick': tick_font_size if tick_font_size is not None else base_font_size - 1,
            'axis_label': base_font_size,
            'legend': base_font_size - 1,
            'subtitle': base_font_size + 1,
            'textbox': base_font_size - 2,
        }

        targets = list(cascade_data.keys())
        stages = list(cascade_data[targets[0]].keys())  # ['train','test','matched','alternative']
        n_stages = len(stages)

        x = np.arange(len(targets))
        width = bar_width_scale / n_stages  # width of a single bar

        stage_alphas = {'train': 0.30, 'test': 0.50, 'matched': 0.70, 'alternative': 1.00}

        for i, stage in enumerate(stages):
            values = [cascade_data[t][stage] for t in targets]
            offset = (i - n_stages / 2) * width + width / 2
            alpha = stage_alphas.get(stage, 0.70)

            bars = ax.bar(x + offset, values, width,
                          label=stage.capitalize(),
                          color=color, alpha=alpha,
                          edgecolor='black', linewidth=0.5)

            if show_bar_labels:
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height(),
                            f'{val:.1f}',
                            ha='center', va='bottom',
                            fontsize=fs['bar_label'])

            if debug:
                print(f"[_plot_cascade] {method_name} | stage={stage} | "
                      f"values={[f'{v:.1f}' for v in values]}")

        # ── threshold lines ──────────────────────────────────────────────────
        if rrmse_threshold is not None and category_thresholds is None:
            ax.axhline(y=rrmse_threshold, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Threshold ({rrmse_threshold}%)', zorder=10)

        if category_thresholds is not None and variable_groupings is not None:
            category_colors = {
                'Hull dimensions': '#E74C3C',
                'Capacity': '#3498DB',
                'Operational': '#2ECC71',
                'Propulsion': '#F39C12',
                'Economic': '#9B59B6',
            }

            for category, threshold in category_thresholds.items():
                cat_vars = variable_groupings.get(category, [])
                positions = [i for i, t in enumerate(targets) if t in cat_vars]

                if not positions:
                    continue

                x_min = min(positions) - 0.5
                x_max = max(positions) + 0.5
                cat_color = category_colors.get(category, '#95A5A6')

                ax.hlines(y=threshold,
                          xmin=x_min, xmax=x_max,
                          colors=cat_color, linestyles='--',
                          linewidth=1.5, alpha=0.8,
                          label=f'{category} ({threshold:.1f}%)',
                          zorder=10)

                if debug:
                    print(f"[_plot_cascade] threshold: {category} = {threshold:.1f}% "
                          f"x=[{x_min:.1f}, {x_max:.1f}] "
                          f"targets={[targets[p] for p in positions]}")

            # Summary text box


            # metric_label = threshold_metric.capitalize()
            # lines = [f"Ref. lines ({metric_label} conv. test NRMSE):"]
            # for cat, val in sorted(category_thresholds.items()):
            #     lines.append(f"  {cat}: {val:.1f}%")
            #
            # ax.text(0.02, 0.98, '\n'.join(lines),
            #         transform=ax.transAxes,
            #         fontsize=fs['textbox'],
            #         va='top', ha='left',
            #         bbox=dict(boxstyle='round,pad=0.4',
            #                   facecolor='lightyellow',
            #                   edgecolor='gray', alpha=0.9),
            #         family='monospace', zorder=15)

        # ── axis decoration ──────────────────────────────────────────────────
        ax.set_xlabel('Target Dimension',  fontsize=fs['axis_label'])
        ax.set_ylabel('NRMSE (%)', fontsize=fs['axis_label'])
        ax.set_title(f'{method_name} — Performance shift results',
                     fontsize=fs['subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._get_display_name(t, variable_name_map) for t in targets], ha='right', fontsize=fs['tick'],
        )
        n_legend_cols = n_stages + len(category_thresholds or {})
        ax.legend(loc='upper left', fontsize=fs['legend'], ncol=n_legend_cols)
        ax.grid(axis='y', alpha=0.3)

    def plot_comprehensive_comparison(self,
                                      figsize: Tuple[int, int] = (20, 12),
                                      save_path: Optional[str] = None,
                                      ship_type:str|None=None,
                                      dataset:str|None=None,):
        """
        Create comprehensive comparison including both selector and domain shift.

        Only works if both selectors AND both domain analyzers are provided.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """

        if not (self.can_compare_selectors and self.can_compare_domain):
            print("Cannot create comprehensive plot: requires both selectors and domain analyzers")
            return

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Section 1: Selector comparison (top row)
        selector_df = self.compare_selectors(metric='rrmse')

        ax1 = fig.add_subplot(gs[0, :2])
        targets = selector_df['target'].values
        x = np.arange(len(targets))
        width = 0.35

        ax1.bar(x - width / 2, selector_df['poly_metric'], width,
                label='Polynomial', color='royalblue', alpha=0.8, edgecolor='black')
        ax1.bar(x + width / 2, selector_df['rf_metric'], width,
                label='Random Forest', color='forestgreen', alpha=0.8, edgecolor='black')

        ax1.set_ylabel('NRMSE (%) - Training', fontweight='bold')
        ax1.set_title('A. Model Selection Performance (Training Data)',
                      fontsize=12, fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Advantage panel
        ax2 = fig.add_subplot(gs[0, 2])
        adv_sel = selector_df['rf_advantage'].values
        colors_sel = ['green' if x > 0 else 'red' for x in adv_sel]
        ax2.barh(targets, adv_sel, color=colors_sel, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('RF Advantage', fontweight='bold')
        ax2.set_title('RF Advantage\n(Training)',
                      fontsize=11, fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=2)
        ax2.grid(axis='x', alpha=0.3)

        # Section 2: Domain shift comparison (middle & bottom rows)
        domain_df = self.compare_domain_shift()

        ax3 = fig.add_subplot(gs[1, :2])
        ax3.bar(x - width / 2, domain_df['poly_alt_rrmse'], width,
                label='Polynomial', color='royalblue', alpha=0.8, edgecolor='black')
        ax3.bar(x + width / 2, domain_df['rf_alt_rrmse'], width,
                label='Random Forest', color='forestgreen', alpha=0.8, edgecolor='black')

        ax3.set_ylabel('NRMSE (%) - Alternative Fuel', fontweight='bold')
        ax3.set_title('B. Domain Shift Performance (Alternative Fuel Ships)',
                      fontsize=12, fontweight='bold', loc='left')
        ax3.set_xticks(x)
        ax3.set_xticklabels(targets, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Advantage panel
        ax4 = fig.add_subplot(gs[1, 2])
        adv_domain = domain_df['rf_advantage'].values
        colors_domain = ['green' if x > 0 else 'red' for x in adv_domain]
        ax4.barh(targets, adv_domain, color=colors_domain, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('RF Advantage', fontweight='bold')
        ax4.set_title('RF Advantage\n(Alt Fuel)',
                      fontsize=11, fontweight='bold')
        ax4.axvline(x=0, color='black', linewidth=2)
        ax4.grid(axis='x', alpha=0.3)

        # Section 3: Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        summary_text = f"""
SUMMARY STATISTICS
═══════════════════════════════════════════════════════════════════════════════

MODEL SELECTION (Training Performance):
  • Polynomial Avg RRMSE:    {selector_df['poly_metric'].mean():.2f}%
  • RF Avg RRMSE:            {selector_df['rf_metric'].mean():.2f}%
  • RF Advantage:            {selector_df['rf_advantage'].mean():+.2f}%
  • RF Wins:                 {(selector_df['better_method'] == 'RF').sum()}/{len(selector_df)} dimensions

DOMAIN SHIFT (Alternative Fuel Generalization):
  • Polynomial Avg RRMSE:    {domain_df['poly_alt_rrmse'].mean():.2f}%
  • RF Avg RRMSE:            {domain_df['rf_alt_rrmse'].mean():.2f}%
  • RF Advantage:            {domain_df['rf_advantage'].mean():+.2f}%
  • RF Wins:                 {(domain_df['better_method'] == 'RF').sum()}/{len(domain_df)} dimensions

INTERPRETATION:
  • Training: RF {'outperforms' if selector_df['rf_advantage'].mean() > 0 else 'underperforms'} polynomial on training data
  • Generalization: RF {'generalizes better' if domain_df['rf_advantage'].mean() > 0 else 'generalizes worse'} to alternative fuel ships
  • {'RF recommended for this dataset' if domain_df['rf_advantage'].mean() > 5 else 'Methods perform similarly - choose based on interpretability needs'}
        """

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        fig.suptitle('Comprehensive Model Comparison: Polynomial vs Random Forest',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Comprehensive comparison plot saved to {save_path}")

        plt.show()

    def _compute_cascade_thresholds(self,
                                    cascade_data: Dict,
                                    variable_groupings: Dict[str, List[str]],
                                    metric_type: str = 'median') -> Dict[str, float]:
        """
        Compute per-category reference thresholds from cascade test RRMSE values.

        The test RRMSE represents model generalisation on held-out conventional
        ships - a natural baseline for what 'good' looks like before any domain
        shift is applied.

        Parameters
        ----------
        cascade_data : dict
            {target: {'train': float, 'test': float, 'matched': float, 'alternative': float}}
        variable_groupings : dict
            Category groupings e.g. {'Hull dimensions': ['LBP (M)', 'BEAM (M)'], ...}
        metric_type : str
            'median', 'mean', or 'best' (minimum)

        Returns
        -------
        dict
            {category_name: threshold_value}
        """
        thresholds = {}

        for category, variables in variable_groupings.items():
            # Collect test RRMSE for all targets in this category
            test_rrmse_vals = [
                cascade_data[t]['test']
                for t in variables
                if t in cascade_data and np.isfinite(cascade_data[t]['test'])
            ]

            if not test_rrmse_vals:
                continue

            vals = np.array(test_rrmse_vals)

            if metric_type == 'median':
                thresholds[category] = float(np.median(vals))
            elif metric_type == 'mean':
                thresholds[category] = float(np.mean(vals))
            elif metric_type == 'best':
                thresholds[category] = float(np.min(vals))
            else:
                raise ValueError(f"Unknown metric_type: '{metric_type}'. Use 'median', 'mean', or 'best'.")

        return thresholds

    # ========================================================================
    # EXPORT & SUMMARY METHODS
    # ========================================================================

    def export_all_comparisons(self, base_filename: str = 'model_comparison'):
        """
        Export all comparison results to CSV files.

        Parameters
        ----------
        base_filename : str
            Base name for output files
        """

        if self.can_compare_selectors:
            selector_df = self.compare_selectors(metric='rrmse')
            selector_file = f"{base_filename}_selectors.csv"
            selector_df.to_csv(selector_file, index=False)
            if self.verbose:
                print(f"✓ Selector comparison exported to {selector_file}")

        if self.can_compare_domain:
            domain_df = self.compare_domain_shift()
            domain_file = f"{base_filename}_domain_shift.csv"
            domain_df.to_csv(domain_file, index=False)
            if self.verbose:
                print(f"✓ Domain shift comparison exported to {domain_file}")

    def print_summary(self):
        """Print comprehensive summary of all comparisons."""

        print("\n" + "=" * 120)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 120)

        if self.can_compare_selectors:
            print("\n1. MODEL SELECTION (Training Performance)")
            print("-" * 120)
            selector_df = self.compare_selectors(metric='rrmse')

            print(f"\nOverall Results:")
            print(f"  Polynomial Avg NRMSE: {selector_df['poly_metric'].mean():.2f}%")
            print(f"  RF Avg RRMSE:         {selector_df['rf_metric'].mean():.2f}%")
            print(f"  RF Advantage:         {selector_df['rf_advantage'].mean():+.2f}%")

            print(f"\nBy Dimension:")
            for _, row in selector_df.iterrows():
                print(f"  {row['target']:<15s}: Poly={row['poly_metric']:6.2f}%, "
                      f"RF={row['rf_metric']:6.2f}%, "
                      f"Advantage={row['rf_advantage']:+6.2f}% "
                      f"[{row['better_method']}]")

        if self.can_compare_domain:
            print("\n2. DOMAIN SHIFT (Alternative Fuel Generalization)")
            print("-" * 120)
            domain_df = self.compare_domain_shift()

            print(f"\nOverall Results:")
            print(f"  Polynomial Avg Alt RRMSE: {domain_df['poly_alt_rrmse'].mean():.2f}%")
            print(f"  RF Avg Alt RRMSE:         {domain_df['rf_alt_rrmse'].mean():.2f}%")
            print(f"  RF Advantage:             {domain_df['rf_advantage'].mean():+.2f}%")

            print(f"\nBy Dimension:")
            for _, row in domain_df.iterrows():
                print(f"  {row['target']:<15s}: Poly={row['poly_alt_rrmse']:6.2f}%, "
                      f"RF={row['rf_alt_rrmse']:6.2f}%, "
                      f"Advantage={row['rf_advantage']:+6.2f}% "
                      f"[{row['better_method']}]")

        print("\n" + "=" * 120)