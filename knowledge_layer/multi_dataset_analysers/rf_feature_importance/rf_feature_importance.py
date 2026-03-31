"""
Random Forest Feature Importance Comparison Analyzer (REFACTORED WITH FOUNDATION)
=================================================================================

Compares how feature importance changes between conventional and alternative fuel ships
using Random Forest models, WITH MATCHED DATASET CONTEXT.

**REFACTORED to use RandomForestAnalysisFoundation for:**
- Selector results validation
- Matched dataset creation
- Data preparation utilities
- Diagnostic tools

Key Enhancement:
---------------
Three-way comparison:
- Conventional (Full): All conventional ships
- Conventional (Matched): Conventional ships matched to alternative distribution
- Alternative: Alternative fuel ships

This isolates:
- Distribution Effect: Full → Matched
- Fuel Effect: Matched → Alternative
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings

from sklearn.ensemble import RandomForestRegressor
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.multi_dataset_analysers.rf_analysis_foundation import RandomForestAnalysisFoundation
from config import LoggerMixin

warnings.filterwarnings('ignore')


class RFImportanceComparisonAnalyzer(LoggerMixin):
    """
    Analyze feature importance changes with matched dataset context.

    **REFACTORED:** Now uses RandomForestAnalysisFoundation via composition
    for common RF analysis functionality.

    Three-way comparison:
    1. Conventional (Full) - baseline with all conventional ships
    2. Conventional (Matched) - conventional ships matched to alternative distribution
    3. Alternative - alternative fuel ships

    This allows decomposition of importance changes into:
    - Distribution effect (how matching changes importance)
    - Fuel effect (pure fuel type effect after matching)
    """

    def __init__(self,
                 rf_selector,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 rf_params: Optional[Dict] = None,
                 calculate_permutation_importance: bool = False,
                 verbose: bool = True):
        """
        Initialize RF importance comparison analyzer with matched dataset support.

        **REFACTORED:** Now uses RandomForestAnalysisFoundation for common setup.

        Parameters
        ----------
        rf_selector : RandomForestSelector
            Fitted RF selector with results
        conventional_data : pd.DataFrame
            Conventional ship data (full dataset)
        alternative_data : pd.DataFrame
            Alternative fuel ship data
        rf_params : dict, optional
            RandomForestRegressor parameters. If None, uses selector's params.
        calculate_permutation_importance : bool, default=False
            If True, also calculates permutation importance (slower but more reliable).
        verbose : bool
            Print detailed output
        """

        self.selector = rf_selector
        self.calculate_permutation_importance = calculate_permutation_importance
        self.verbose = verbose

        # ✅ NEW: Create foundation to handle common RF analysis setup
        self.foundation = RandomForestAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            variable_dimensions=rf_selector.variable_dimensions,
            target_dimensions=rf_selector.target_dimensions,
            selector_results=rf_selector.results,
            rf_params=rf_params or rf_selector.rf_params,
            random_state=42,
            verbose=verbose
        )

        # Storage for THIS analyzer's specific results
        self.comparison_results = None
        self.fitted_models = {
            'conventional_full': {},
            'conventional_matched': {},
            'alternative': {}
        }

        if verbose:
            print("\n" + "=" * 120)
            print("RANDOM FOREST FEATURE IMPORTANCE COMPARISON ANALYZER (WITH MATCHED DATASETS)")
            print("=" * 120)
            print(f"  Conventional ships (full): {len(conventional_data)}")
            print(f"  Alternative ships:         {len(alternative_data)}")
            print(f"  Targets to analyze:        {len(self.foundation.target_dimensions)}")
            print(f"  RF parameters:             n_estimators={self.foundation.rf_params.get('n_estimators', 100)}, "
                  f"max_depth={self.foundation.rf_params.get('max_depth', 10)}")
            print(f"  Permutation test:          {'Enabled (slower)' if calculate_permutation_importance else 'Disabled (faster)'}")
            print(f"  Matched datasets:          {len(self.foundation.matched_data_sets)} created at initialization")

    # ========================================================================
    # MAIN ANALYSIS METHOD (WITH MATCHED DATASETS)
    # ========================================================================

    def analyze_all_targets(self,
                            min_alternative_samples: int = 10,
                            min_matched_samples: int = 10,
                            use_feature_engineering: bool = False,
                            display_results: bool = True) -> pd.DataFrame:
        """
        Analyze feature importance changes for all target dimensions.

        NOW WITH THREE-WAY COMPARISON:
        - Conventional (Full)
        - Conventional (Matched)
        - Alternative

        Workflow:
        1. For each target, get best RF model from selector
        2. Get matched conventional dataset from foundation
        3. Train RF on conventional full data
        4. Train RF on conventional matched data
        5. Train RF on alternative data
        6. Compare feature importances across all three
        7. Decompose into distribution vs fuel effects

        Parameters
        ----------
        min_alternative_samples : int
            Minimum number of alternative fuel ships required
        min_matched_samples : int
            Minimum number of matched conventional ships required
        use_feature_engineering : bool
            Whether to use feature-engineered versions (if available)
        display_results : bool
            Whether to print results during analysis

        Returns
        -------
        pd.DataFrame
            Comparison results with columns:
            - target, predictors, n_predictors
            - n_conventional_full, n_conventional_matched, n_alternative
            - conv_full_r2, conv_matched_r2, alt_r2
            - importance_conventional_full, importance_conventional_matched, importance_alternative
            - importance_changes (with distribution and fuel effects)
        """

        if display_results and self.verbose:
            print("\n" + "=" * 120)
            print("RANDOM FOREST FEATURE IMPORTANCE COMPARISON (THREE-WAY)")
            print("Full Conventional → Matched Conventional → Alternative")
            print("=" * 120)

        all_results = []

        for target in self.foundation.target_dimensions:
            if display_results and self.verbose:
                print(f"\n{'=' * 120}")
                print(f"Target: {target}")
                print(f"{'=' * 120}")

            # Get best combination for this target
            best_combo, best_is_engineered = self._get_best_combination_for_target(
                target, use_feature_engineering
            )

            if best_combo is None:
                if self.verbose:
                    print(f"  ⚠️  No results found for {target}")
                continue

            if display_results and self.verbose:
                print(f"  Best combination: {' + '.join(best_combo)}")
                if best_is_engineered:
                    print(f"  Feature engineered: Yes")

            # Analyze this target (three-way)
            result = self._analyze_single_target_threeway(
                target=target,
                predictors=best_combo,
                is_feature_engineered=best_is_engineered,
                min_alternative_samples=min_alternative_samples,
                min_matched_samples=min_matched_samples,
                display_results=display_results
            )

            if result is not None:
                all_results.append(result)

        # Create summary DataFrame
        if not all_results:
            print("\n⚠️  No valid results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Store results
        self.comparison_results = results_df

        if display_results and self.verbose:
            self._print_summary_table(results_df)

        return results_df

    # ========================================================================
    # SINGLE TARGET ANALYSIS (THREE-WAY) - REFACTORED
    # ========================================================================

    def _analyze_single_target_threeway(self,
                                       target: str,
                                       predictors: Tuple[str, ...],
                                       is_feature_engineered: bool,
                                       min_alternative_samples: int,
                                       min_matched_samples: int,
                                       display_results: bool) -> Optional[Dict]:
        """
        Analyze feature importance changes for a single target (three-way comparison).

        **REFACTORED:** Now uses foundation for data preparation and matched dataset access.

        Returns
        -------
        dict or None
            Results dictionary or None if analysis failed
        """

        # ====================================================================
        # PREPARE DATA - THREE DATASETS (using foundation)
        # ====================================================================

        # 1. Conventional (Full) - ✅ Use foundation's prepare_dataset
        conv_full_data = self.foundation.prepare_dataset(
            self.foundation.conventional_full,
            list(predictors),
            target,
            min_samples=5
        )
        if conv_full_data is None:
            if self.verbose:
                print(f"  ⚠️  Insufficient conventional data")
            return None
        X_conv_full, y_conv_full = conv_full_data

        # 2. Alternative - ✅ Use foundation's prepare_dataset
        alt_data = self.foundation.prepare_dataset(
            self.foundation.alternative,
            list(predictors),
            target,
            min_samples=5
        )
        if alt_data is None:
            if self.verbose:
                print(f"  ⚠️  Insufficient alternative data")
            return None
        X_alt, y_alt = alt_data

        # Check minimum samples
        if len(X_alt) < min_alternative_samples:
            if self.verbose:
                print(f"  ⚠️  Insufficient alternative samples: {len(X_alt)} < {min_alternative_samples}")
            return None

        # 3. Conventional (Matched) - ✅ Use foundation's matched datasets
        predictor_key = tuple(predictors)
        matched_df = self.foundation.get_matched_dataset(predictor_key)

        if matched_df is None:
            if self.verbose:
                print(f"  ⚠️  No matched dataset found for {' + '.join(predictors)}")
            return None

        conv_matched_data = self.foundation.prepare_dataset(
            matched_df,
            list(predictors),
            target,
            min_samples=5
        )

        if conv_matched_data is None:
            if self.verbose:
                print(f"  ⚠️  Insufficient matched conventional data")
            return None

        X_conv_matched, y_conv_matched = conv_matched_data

        # Check minimum matched samples
        if len(X_conv_matched) < min_matched_samples:
            if self.verbose:
                print(f"  ⚠️  Insufficient matched samples: {len(X_conv_matched)} < {min_matched_samples}")
            return None

        # ====================================================================
        # FIT THREE RANDOM FOREST MODELS - ✅ Use foundation's RF params
        # ====================================================================

        rf_params = self.foundation.get_rf_params()

        # 1. Fit on conventional (full)
        try:
            conv_full_rf = RandomForestRegressor(**rf_params)
            conv_full_rf.fit(X_conv_full, y_conv_full)
            y_pred_conv_full = conv_full_rf.predict(X_conv_full)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Conventional (full) RF fitting failed: {e}")
            return None

        # 2. Fit on conventional (matched)
        try:
            conv_matched_rf = RandomForestRegressor(**rf_params)
            conv_matched_rf.fit(X_conv_matched, y_conv_matched)
            y_pred_conv_matched = conv_matched_rf.predict(X_conv_matched)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Conventional (matched) RF fitting failed: {e}")
            return None

        # 3. Fit on alternative
        try:
            alt_rf = RandomForestRegressor(**rf_params)
            alt_rf.fit(X_alt, y_alt)
            y_pred_alt = alt_rf.predict(X_alt)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Alternative RF fitting failed: {e}")
            return None

        # Store fitted models
        key = (tuple(predictors), target)
        self.fitted_models['conventional_full'][key] = conv_full_rf
        self.fitted_models['conventional_matched'][key] = conv_matched_rf
        self.fitted_models['alternative'][key] = alt_rf

        # ====================================================================
        # EXTRACT FEATURE IMPORTANCES
        # ====================================================================

        conv_full_importance = dict(zip(predictors, conv_full_rf.feature_importances_))
        conv_matched_importance = dict(zip(predictors, conv_matched_rf.feature_importances_))
        alt_importance = dict(zip(predictors, alt_rf.feature_importances_))

        # Calculate metrics
        conv_full_metrics = self._calculate_metrics(y_conv_full, y_pred_conv_full, len(predictors))
        conv_matched_metrics = self._calculate_metrics(y_conv_matched, y_pred_conv_matched, len(predictors))
        alt_metrics = self._calculate_metrics(y_alt, y_pred_alt, len(predictors))

        # ====================================================================
        # CROSS-EVALUATION
        # ====================================================================

        cross_evals = {}
        try:
            # Full → Matched
            y_matched_by_full = conv_full_rf.predict(X_conv_matched)
            cross_evals['full_on_matched'] = self._calculate_metrics(
                y_conv_matched, y_matched_by_full, len(predictors)
            )

            # Full → Alternative
            y_alt_by_full = conv_full_rf.predict(X_alt)
            cross_evals['full_on_alt'] = self._calculate_metrics(
                y_alt, y_alt_by_full, len(predictors)
            )

            # Matched → Alternative
            y_alt_by_matched = conv_matched_rf.predict(X_alt)
            cross_evals['matched_on_alt'] = self._calculate_metrics(
                y_alt, y_alt_by_matched, len(predictors)
            )

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Cross-evaluation failed: {e}")

        # ====================================================================
        # OPTIONAL: PERMUTATION IMPORTANCE
        # ====================================================================

        perm_importance_full = None
        perm_importance_matched = None
        perm_importance_alt = None

        if self.calculate_permutation_importance:
            if self.verbose:
                print(f"  Calculating permutation importance (this may take a moment)...")

            try:
                from sklearn.inspection import permutation_importance

                # Full
                perm_full = permutation_importance(
                    conv_full_rf, X_conv_full, y_conv_full, n_repeats=10, random_state=42
                )
                perm_importance_full = dict(zip(predictors, perm_full.importances_mean))

                # Matched
                perm_matched = permutation_importance(
                    conv_matched_rf, X_conv_matched, y_conv_matched, n_repeats=10, random_state=42
                )
                perm_importance_matched = dict(zip(predictors, perm_matched.importances_mean))

                # Alternative
                perm_alt = permutation_importance(
                    alt_rf, X_alt, y_alt, n_repeats=10, random_state=42
                )
                perm_importance_alt = dict(zip(predictors, perm_alt.importances_mean))

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Permutation importance calculation failed: {e}")

        # ====================================================================
        # COMPARE IMPORTANCES (THREE-WAY)
        # ====================================================================

        importance_comparison = self._compare_importances_threeway(
            conv_full_importance, conv_matched_importance, alt_importance, predictors
        )

        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================

        if display_results and self.verbose:
            self._print_single_target_results(
                target, predictors, is_feature_engineered,
                conv_full_importance, conv_matched_importance, alt_importance,
                importance_comparison,
                conv_full_metrics, conv_matched_metrics, alt_metrics,
                cross_evals,
                perm_importance_full, perm_importance_matched, perm_importance_alt
            )

        # ====================================================================
        # BUILD RESULT DICTIONARY
        # ====================================================================

        result = {
            'target': target,
            'predictors': ' + '.join(predictors),
            'n_predictors': len(predictors),
            'feature_engineered': is_feature_engineered,

            # Sample sizes
            'n_conventional_full': len(X_conv_full),
            'n_conventional_matched': len(X_conv_matched),
            'n_alternative': len(X_alt),

            # Performance metrics
            'conv_full_r2': conv_full_metrics['r2'],
            'conv_full_rrmse': conv_full_metrics['rrmse'],
            'conv_matched_r2': conv_matched_metrics['r2'],
            'conv_matched_rrmse': conv_matched_metrics['rrmse'],
            'alt_r2': alt_metrics['r2'],
            'alt_rrmse': alt_metrics['rrmse'],

            # Cross-evaluation
            'cross_full_on_matched_r2': cross_evals.get('full_on_matched', {}).get('r2', np.nan),
            'cross_full_on_alt_r2': cross_evals.get('full_on_alt', {}).get('r2', np.nan),
            'cross_matched_on_alt_r2': cross_evals.get('matched_on_alt', {}).get('r2', np.nan),

            # Performance changes
            'r2_distribution_effect': conv_matched_metrics['r2'] - conv_full_metrics['r2'],
            'r2_fuel_effect': alt_metrics['r2'] - conv_matched_metrics['r2'],
            'r2_total_effect': alt_metrics['r2'] - conv_full_metrics['r2'],

            # Feature importances
            'importance_conventional_full': conv_full_importance,
            'importance_conventional_matched': conv_matched_importance,
            'importance_alternative': alt_importance,
            'importance_changes': importance_comparison,

            # Permutation importances (if calculated)
            'perm_importance_full': perm_importance_full,
            'perm_importance_matched': perm_importance_matched,
            'perm_importance_alternative': perm_importance_alt,

            # Top features
            'top_full': max(conv_full_importance, key=conv_full_importance.get),
            'top_matched': max(conv_matched_importance, key=conv_matched_importance.get),
            'top_alternative': max(alt_importance, key=alt_importance.get),
        }

        return result

    # ========================================================================
    # IMPORTANCE COMPARISON (THREE-WAY) - Unchanged
    # ========================================================================

    def _compare_importances_threeway(self,
                                     conv_full_importance: Dict,
                                     conv_matched_importance: Dict,
                                     alt_importance: Dict,
                                     predictors: Tuple[str, ...]) -> Dict:
        """
        Compare feature importances across three datasets.

        Decomposes changes into:
        - Distribution effect: Full → Matched
        - Fuel effect: Matched → Alternative

        Returns
        -------
        dict
            Feature-level comparison with distribution and fuel effects
        """
        comparison = {}

        # Get rankings for all three
        conv_full_ranked = sorted(conv_full_importance.items(), key=lambda x: x[1], reverse=True)
        conv_matched_ranked = sorted(conv_matched_importance.items(), key=lambda x: x[1], reverse=True)
        alt_ranked = sorted(alt_importance.items(), key=lambda x: x[1], reverse=True)

        conv_full_ranks = {feat: rank for rank, (feat, _) in enumerate(conv_full_ranked, 1)}
        conv_matched_ranks = {feat: rank for rank, (feat, _) in enumerate(conv_matched_ranked, 1)}
        alt_ranks = {feat: rank for rank, (feat, _) in enumerate(alt_ranked, 1)}

        for feature in predictors:
            full_val = conv_full_importance[feature]
            matched_val = conv_matched_importance[feature]
            alt_val = alt_importance[feature]

            # Distribution effect (Full → Matched)
            dist_change = matched_val - full_val
            dist_pct_points = dist_change * 100

            # Fuel effect (Matched → Alternative)
            fuel_change = alt_val - matched_val
            fuel_pct_points = fuel_change * 100

            # Total effect (Full → Alternative)
            total_change = alt_val - full_val
            total_pct_points = total_change * 100

            # Rank changes
            rank_full_to_matched = conv_matched_ranks[feature] - conv_full_ranks[feature]
            rank_matched_to_alt = alt_ranks[feature] - conv_matched_ranks[feature]
            rank_full_to_alt = alt_ranks[feature] - conv_full_ranks[feature]

            comparison[feature] = {
                # Values
                'full': full_val,
                'matched': matched_val,
                'alternative': alt_val,

                # Distribution effect
                'distribution_effect': dist_change,
                'distribution_pct_points': dist_pct_points,
                'rank_full': conv_full_ranks[feature],
                'rank_matched': conv_matched_ranks[feature],
                'rank_change_distribution': rank_full_to_matched,

                # Fuel effect
                'fuel_effect': fuel_change,
                'fuel_pct_points': fuel_pct_points,
                'rank_alternative': alt_ranks[feature],
                'rank_change_fuel': rank_matched_to_alt,

                # Total effect
                'total_effect': total_change,
                'total_pct_points': total_pct_points,
                'rank_change_total': rank_full_to_alt,

                # Direction indicators
                'distribution_direction': 'increased' if dist_change > 0 else 'decreased' if dist_change < 0 else 'unchanged',
                'fuel_direction': 'increased' if fuel_change > 0 else 'decreased' if fuel_change < 0 else 'unchanged',
                'total_direction': 'increased' if total_change > 0 else 'decreased' if total_change < 0 else 'unchanged'
            }

        return comparison

    # ========================================================================
    # IMPORTANCE CHANGE ANALYSIS - Unchanged
    # ========================================================================

    def get_importance_changes(self,
                               threshold: float = 0.05,
                               effect: str = 'fuel',
                               sort_by: str = 'absolute') -> pd.DataFrame:
        """
        Get summary of feature importance changes.

        Parameters
        ----------
        threshold : float, default=0.05
            Minimum change in importance (0-1 scale) to be considered significant.
            Default 0.05 = 5 percentage points.
        effect : str, default='fuel'
            Which effect to analyze:
            - 'distribution': Full → Matched
            - 'fuel': Matched → Alternative
            - 'total': Full → Alternative
        sort_by : str, default='absolute'
            How to sort results:
            - 'absolute': By absolute change magnitude
            - 'percentage': By percentage point change
            - 'rank': By rank change

        Returns
        -------
        pd.DataFrame
            Summary of importance changes
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run analyze_all_targets() first.")
            return pd.DataFrame()

        # Validate effect parameter
        valid_effects = ['distribution', 'fuel', 'total']
        if effect not in valid_effects:
            raise ValueError(f"effect must be one of {valid_effects}, got '{effect}'")

        change_data = []

        for _, row in self.comparison_results.iterrows():
            importance_comparison = row['importance_changes']

            for feature, comp in importance_comparison.items():
                # Select which effect to analyze
                if effect == 'distribution':
                    abs_change = comp['distribution_effect']
                    pct_points = comp['distribution_pct_points']
                    rank_change = comp['rank_change_distribution']
                    direction = comp['distribution_direction']
                elif effect == 'fuel':
                    abs_change = comp['fuel_effect']
                    pct_points = comp['fuel_pct_points']
                    rank_change = comp['rank_change_fuel']
                    direction = comp['fuel_direction']
                else:  # total
                    abs_change = comp['total_effect']
                    pct_points = comp['total_pct_points']
                    rank_change = comp['rank_change_total']
                    direction = comp['total_direction']

                change_data.append({
                    'target': row['target'],
                    'feature': feature,
                    'effect_type': effect,
                    'importance_full': comp['full'],
                    'importance_matched': comp['matched'],
                    'importance_alt': comp['alternative'],
                    'absolute_change': abs_change,
                    'percentage_point_change': pct_points,
                    'rank_change': rank_change,
                    'direction': direction,
                    'significant': abs(abs_change) >= threshold
                })

        changes_df = pd.DataFrame(change_data)

        # Sort based on user preference
        if sort_by == 'absolute':
            changes_df = changes_df.sort_values('absolute_change',
                                                key=abs,
                                                ascending=False)
        elif sort_by == 'percentage':
            changes_df = changes_df.sort_values('percentage_point_change',
                                                key=abs,
                                                ascending=False)
        elif sort_by == 'rank':
            changes_df = changes_df.sort_values('rank_change',
                                                key=abs,
                                                ascending=False)

        if self.verbose:
            self._print_importance_changes_summary(changes_df, threshold, effect)

        return changes_df

    def _print_importance_changes_summary(self, changes_df: pd.DataFrame,
                                         threshold: float, effect: str):
        """Print formatted summary of importance changes."""

        effect_labels = {
            'distribution': 'Distribution Effect (Full → Matched)',
            'fuel': 'Fuel Effect (Matched → Alternative)',
            'total': 'Total Effect (Full → Alternative)'
        }

        print("\n" + "=" * 120)
        print(f"FEATURE IMPORTANCE CHANGES - {effect_labels[effect].upper()}")
        print(f"(Threshold: {threshold * 100:.1f} percentage points)")
        print("=" * 120)

        # Filter to significant changes
        significant = changes_df[changes_df['significant']]

        if len(significant) == 0:
            print(f"\n  No features show {effect} changes > {threshold * 100:.1f} percentage points")
            return

        print(f"\n{'Target':<15s} | {'Feature':<20s} | {'Full':>8s} | {'Match':>8s} | {'Alt':>8s} | "
              f"{'Δ (pp)':>10s} | {'Rank Δ':>8s} | {'Direction':<12s}")
        print("-" * 120)

        for _, row in significant.iterrows():
            emoji = "📈" if row['absolute_change'] > 0 else "📉"

            print(f"{emoji} {row['target']:<13s} | {row['feature']:<20s} | "
                  f"{row['importance_full']:>8.3f} | {row['importance_matched']:>8.3f} | "
                  f"{row['importance_alt']:>8.3f} | "
                  f"{row['percentage_point_change']:>+10.2f} | {row['rank_change']:>+8d} | "
                  f"{row['direction']:<12s}")

        print("=" * 120)

        # Summary statistics
        gained = (significant['absolute_change'] > 0).sum()
        lost = (significant['absolute_change'] < 0).sum()

        print(f"\nSignificant Changes ({effect}): {len(significant)}")
        print(f"  Features gaining importance:  {gained}")
        print(f"  Features losing importance:   {lost}")

    # ========================================================================
    # DECOMPOSITION ANALYSIS - Unchanged
    # ========================================================================

    def decompose_importance_changes(self,
                                    effect_size_threshold: float = 0.05) -> pd.DataFrame:
        """
        Decompose importance changes into distribution vs fuel effects.

        Similar to coefficient decomposition in parametric analyzer.

        Parameters
        ----------
        effect_size_threshold : float
            Minimum change to be considered significant (in 0-1 scale)

        Returns
        -------
        pd.DataFrame
            Decomposition results
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run analyze_all_targets() first.")
            return pd.DataFrame()

        decompositions = []

        for _, row in self.comparison_results.iterrows():
            importance_comp = row['importance_changes']

            for feature, comp in importance_comp.items():
                dist_effect = comp['distribution_effect']
                fuel_effect = comp['fuel_effect']
                total_effect = comp['total_effect']

                # Calculate contribution percentages
                total_abs = abs(total_effect)
                if total_abs > 1e-10:
                    dist_pct = 100 * abs(dist_effect) / total_abs
                    fuel_pct = 100 * abs(fuel_effect) / total_abs
                else:
                    dist_pct = np.nan
                    fuel_pct = np.nan

                # Categorize fuel effect size
                fuel_abs = abs(fuel_effect)
                if fuel_abs < effect_size_threshold:
                    effect_size = "Negligible"
                    emoji = "✓"
                elif fuel_abs < 0.10:
                    effect_size = "Small"
                    emoji = "⚠️"
                elif fuel_abs < 0.20:
                    effect_size = "Medium"
                    emoji = "🔶"
                else:
                    effect_size = "Large"
                    emoji = "🚨"

                decompositions.append({
                    'target': row['target'],
                    'feature': feature,
                    'full': comp['full'],
                    'matched': comp['matched'],
                    'alternative': comp['alternative'],
                    'distribution_effect': dist_effect,
                    'fuel_effect': fuel_effect,
                    'total_effect': total_effect,
                    'distribution_pct': dist_pct,
                    'fuel_pct': fuel_pct,
                    'fuel_effect_size': effect_size,
                    'fuel_significant': fuel_abs >= effect_size_threshold,
                    'emoji': emoji
                })

        decomp_df = pd.DataFrame(decompositions)
        if self.verbose:
            self._print_decomposition_summary(decomp_df, effect_size_threshold)
        return decomp_df

    def _print_decomposition_summary(self, decomp_df: pd.DataFrame,
                                     threshold: float):
        """Print formatted decomposition summary."""

        print("\n" + "=" * 140)
        print("IMPORTANCE CHANGE DECOMPOSITION")
        print("Distribution Effect (Full → Matched) vs Fuel Effect (Matched → Alternative)")
        print("=" * 140)

        print(f"\nEffect Size Thresholds:")
        print(f"  Negligible: < {threshold * 100:.0f}pp")
        print(f"  Small:      {threshold * 100:.0f}-10pp")
        print(f"  Medium:     10-20pp")
        print(f"  Large:      > 20pp")

        print("\n" + "-" * 140)
        print(f"{'':2s} {'Target':<15s} | {'Feature':<15s} | "
              f"{'Full':>8s} → {'Match':>8s} → {'Alt':>8s} | "
              f"{'Dist Δ':>8s} | {'Fuel Δ':>8s} | {'Fuel %':>7s} | {'Effect':<12s}")
        print("-" * 140)

        for _, row in decomp_df.iterrows():
            print(f"{row['emoji']:2s} {row['target']:<15s} | {row['feature']:<15s} | "
                  f"{row['full']:>8.3f} → {row['matched']:>8.3f} → {row['alternative']:>8.3f} | "
                  f"{row['distribution_effect']:>+8.3f} | {row['fuel_effect']:>+8.3f} | "
                  f"{row['fuel_pct']:>7.1f} | "
                  f"{row['fuel_effect_size']:<12s}")

        print("=" * 140)

        # Summary stats
        print(f"\nSummary:")
        print(f"  Total features analyzed:      {len(decomp_df)}")
        print(f"  Fuel effect significant:      "
              f"{decomp_df['fuel_significant'].sum()} "
              f"({decomp_df['fuel_significant'].sum() / len(decomp_df) * 100:.1f}%)")
        print(f"  Average fuel contribution:    {decomp_df['fuel_pct'].mean():.1f}%")
        print(f"  Average distribution contrib: {decomp_df['distribution_pct'].mean():.1f}%")

        print(f"\n  Effect Size Distribution:")
        for effect in ['Negligible', 'Small', 'Medium', 'Large']:
            count = (decomp_df['fuel_effect_size'] == effect).sum()
            pct = count / len(decomp_df) * 100
            print(f"    {effect:<12s}: {count} ({pct:.1f}%)")

    # ========================================================================
    # HELPER METHODS - Simplified with foundation
    # ========================================================================

    def _get_best_combination_for_target(self,
                                         target: str,
                                         prefer_feature_engineered: bool) -> Tuple[Optional[Tuple], bool]:
        """Get best predictor combination for target from RF selector results."""

        if target not in self.selector.results:
            return None, False

        target_results = self.selector.results[target]

        if not target_results:
            return None, False

        # Get best combination (lowest RRMSE)
        best_combo_name = None
        best_rrmse = float('inf')
        best_is_engineered = False

        for combo_name, data in target_results.items():
            metrics = data['metrics']
            is_engineered = data.get('feature_engineered', False)

            # If user prefers feature-engineered, prioritize those
            if prefer_feature_engineered and not is_engineered:
                continue

            rrmse = metrics.get('rrmse', float('inf'))
            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_combo_name = combo_name
                best_is_engineered = is_engineered

        if best_combo_name is None:
            return None, False

        # Extract predictors from combo_name
        # Handle both "X + Y" and "X + Y (Feat Eng)" formats
        predictors_str = best_combo_name.replace(' (Feat Eng)', '')
        predictors = tuple(p.strip() for p in predictors_str.split('+'))

        return predictors, best_is_engineered

    def _calculate_metrics(self, y_true, y_pred, n_features) -> Dict:
        """Calculate regression metrics."""
        metric_calc = RegressionMetrics(y_true, y_pred)
        return metric_calc.calculate_all_metrics(n_features)

    # ========================================================================
    # NEW: Expose foundation diagnostic methods
    # ========================================================================

    def print_variable_diagnostics(self):
        """Print diagnostic information about variable availability."""
        self.foundation.print_variable_diagnostics()

    def get_combinations_summary(self) -> pd.DataFrame:
        """Get summary of available combinations from selector results."""
        return self.foundation.get_combinations_summary()

    def diagnose_missing_variables(self) -> pd.DataFrame:
        """Diagnose which variables from selector results are missing."""
        return self.foundation.diagnose_missing_variables()

    # ========================================================================
    # DISPLAY METHODS - Unchanged
    # ========================================================================

    def _print_single_target_results(self,
                                     target: str,
                                     predictors: Tuple[str, ...],
                                     is_feature_engineered: bool,
                                     conv_full_importance: Dict,
                                     conv_matched_importance: Dict,
                                     alt_importance: Dict,
                                     importance_comparison: Dict,
                                     conv_full_metrics: Dict,
                                     conv_matched_metrics: Dict,
                                     alt_metrics: Dict,
                                     cross_evals: Dict,
                                     perm_importance_full: Optional[Dict],
                                     perm_importance_matched: Optional[Dict],
                                     perm_importance_alt: Optional[Dict]):
        """Print detailed results for single target."""

        print(f"\n  Predictors: {' + '.join(predictors)}")
        if is_feature_engineered:
            print(f"  Feature engineered: Yes")

        print(f"  Samples: Full={conv_full_metrics['n_samples']}, "
              f"Matched={conv_matched_metrics['n_samples']}, "
              f"Alt={alt_metrics['n_samples']}")

        print(f"\n  Performance Comparison:")
        print(
            f"  {'Metric':<25s} | {'Full':>12s} | {'Matched':>12s} | {'Alt':>12s} | {'Dist Δ':>10s} | {'Fuel Δ':>10s}")
        print(f"  {'-' * 90}")

        r2_dist = conv_matched_metrics['r2'] - conv_full_metrics['r2']
        r2_fuel = alt_metrics['r2'] - conv_matched_metrics['r2']
        rrmse_dist = conv_matched_metrics['rrmse'] - conv_full_metrics['rrmse']
        rrmse_fuel = alt_metrics['rrmse'] - conv_matched_metrics['rrmse']

        print(f"  {'R²':<25s} | {conv_full_metrics['r2']:>12.4f} | {conv_matched_metrics['r2']:>12.4f} | "
              f"{alt_metrics['r2']:>12.4f} | {r2_dist:>+10.4f} | {r2_fuel:>+10.4f}")
        print(f"  {'RRMSE (%)':<25s} | {conv_full_metrics['rrmse']:>12.2f} | {conv_matched_metrics['rrmse']:>12.2f} | "
              f"{alt_metrics['rrmse']:>12.2f} | {rrmse_dist:>+10.2f} | {rrmse_fuel:>+10.2f}")

        if cross_evals:
            print(f"\n  Cross-Evaluation:")
            print(f"  {'Model → Data':<30s} | {'R²':>12s} | {'RRMSE %':>12s}")
            print(f"  {'-' * 58}")

            for key, metrics in cross_evals.items():
                label = key.replace('_', ' → ').replace('full', 'Full').replace('matched', 'Matched').replace('alt',
                                                                                                              'Alt')
                print(f"  {label:<30s} | {metrics['r2']:>12.4f} | {metrics['rrmse']:>12.2f}")

        print(f"\n  Three-Way Feature Importance Comparison:")
        print(f"  {'Feature':<20s} | {'Full':>10s} | {'Match':>10s} | {'Alt':>10s} | "
              f"{'Dist Δ':>10s} | {'Fuel Δ':>10s} | {'Total Δ':>10s}")
        print(f"  {'-' * 90}")

        # Sort by full importance
        sorted_features = sorted(conv_full_importance.items(), key=lambda x: x[1], reverse=True)

        for feature, full_imp in sorted_features:
            comp = importance_comparison[feature]

            print(f"  {feature:<20s} | {full_imp:>10.3f} | {comp['matched']:>10.3f} | "
                  f"{comp['alternative']:>10.3f} | "
                  f"{comp['distribution_effect']:>+10.3f} | {comp['fuel_effect']:>+10.3f} | "
                  f"{comp['total_effect']:>+10.3f}")

        # Interpretation
        print(f"\n  💡 Most Important Features:")
        print(f"     Full:        {max(conv_full_importance, key=conv_full_importance.get)} "
              f"({max(conv_full_importance.values()) * 100:.1f}%)")
        print(f"     Matched:     {max(conv_matched_importance, key=conv_matched_importance.get)} "
              f"({max(conv_matched_importance.values()) * 100:.1f}%)")
        print(f"     Alternative: {max(alt_importance, key=alt_importance.get)} "
              f"({max(alt_importance.values()) * 100:.1f}%)")

        # Biggest movers
        biggest_dist_gainer = max(importance_comparison.items(),
                                  key=lambda x: x[1]['distribution_effect'])
        biggest_fuel_gainer = max(importance_comparison.items(),
                                  key=lambda x: x[1]['fuel_effect'])

        if abs(biggest_dist_gainer[1]['distribution_effect']) > 0.05:
            print(f"     Biggest distribution shift: {biggest_dist_gainer[0]} "
                  f"({biggest_dist_gainer[1]['distribution_effect']:+.3f})")

        if abs(biggest_fuel_gainer[1]['fuel_effect']) > 0.05:
            print(f"     Biggest fuel effect:        {biggest_fuel_gainer[0]} "
                  f"({biggest_fuel_gainer[1]['fuel_effect']:+.3f})")

    def _print_summary_table(self, results_df: pd.DataFrame):
        """Print summary table of all results."""

        print("\n" + "=" * 140)
        print("SUMMARY: Three-Way Feature Importance Comparison")
        print("=" * 140)

        print(f"\n{'Target':<15s} | {'Full R²':>8s} | {'Match R²':>9s} | {'Alt R²':>8s} | "
              f"{'Top (Full)':<20s} | {'Top (Match)':<20s} | {'Top (Alt)':<20s} | {'Switched?':<10s}")
        print("-" * 140)

        for _, row in results_df.iterrows():
            top_full = row['top_full']
            top_matched = row['top_matched']
            top_alt = row['top_alternative']

            switched_dist = "Yes ⚠️" if top_full != top_matched else ""
            switched_fuel = "Yes ⚠️" if top_matched != top_alt else ""
            switched = f"{switched_dist}/{switched_fuel}".strip('/')

            print(f"{row['target']:<15s} | {row['conv_full_r2']:>8.4f} | {row['conv_matched_r2']:>9.4f} | "
                  f"{row['alt_r2']:>8.4f} | "
                  f"{top_full:<20s} | {top_matched:<20s} | {top_alt:<20s} | {switched:<10s}")

        print("=" * 140)

        # Summary statistics
        switched_dist = (results_df['top_full'] != results_df['top_matched']).sum()
        switched_fuel = (results_df['top_matched'] != results_df['top_alternative']).sum()
        switched_total = (results_df['top_full'] != results_df['top_alternative']).sum()

        print(f"\n  Top feature changes (Distribution): {switched_dist}/{len(results_df)}")
        print(f"  Top feature changes (Fuel):          {switched_fuel}/{len(results_df)}")
        print(f"  Top feature changes (Total):         {switched_total}/{len(results_df)}")

    # ========================================================================
    # PLOTTING METHODS - Unchanged (keeping all existing plots)
    # ========================================================================

    def plot_importance_comparison(self,
                                   target: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8),
                                   save_path: Optional[str] = None):
        """
        Plot feature importance comparison with matched dataset context.

        Creates grouped horizontal bar charts showing conventional (full),
        conventional (matched), and alternative importance for each feature.

        Parameters
        ----------
        target : str, optional
            Specific target to plot. If None, plots all.
        figsize : tuple
            Figure size per target
        save_path : str, optional
            Path to save figure
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        results_df = self.comparison_results

        if target is not None:
            results_df = results_df[results_df['target'] == target]
            if len(results_df) == 0:
                print(f"No results found for target '{target}'")
                return

        n_targets = len(results_df)

        # Create subplots - one per target
        n_cols = 2
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * 1.5, figsize[1] * n_rows))

        # Flatten axes array for easier iteration
        if n_targets == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle('Feature Importance Comparison: Full vs Matched vs Alternative',
                     fontsize=16, fontweight='bold', y=0.995)

        for idx, (_, row) in enumerate(results_df.iterrows()):
            ax = axes[idx]

            conv_full_imp = row['importance_conventional_full']
            conv_matched_imp = row['importance_conventional_matched']
            alt_imp = row['importance_alternative']

            # Sort by full conventional importance (highest first)
            sorted_features = sorted(conv_full_imp.items(), key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_features]

            full_vals = [conv_full_imp[f] for f in features]
            matched_vals = [conv_matched_imp[f] for f in features]
            alt_vals = [alt_imp[f] for f in features]

            # Set up bar positions (three bars per feature)
            y = np.arange(len(features))
            height = 0.15

            # Create three horizontal bars
            bars_full = ax.barh(y + height, full_vals, height,
                                label='Conventional (Full)',
                                color='royalblue', alpha=0.8, edgecolor='black')

            bars_matched = ax.barh(y, matched_vals, height,
                                   label='Conventional (Matched)',
                                   color='steelblue', alpha=0.8, edgecolor='black')

            bars_alt = ax.barh(y - height, alt_vals, height,
                               label='Alternative',
                               color='forestgreen', alpha=0.8, edgecolor='black')

            # Formatting
            ax.set_xlabel('Feature Importance', fontweight='bold', fontsize=11)
            ax.set_title(f'{row["target"]} - Importance Comparison',
                         fontweight='bold', fontsize=12)
            ax.set_yticks(y)
            ax.set_yticklabels(features)
            ax.tick_params(axis='y', pad=1)
            ax.set_ylim(-0.5, len(features) - 0.5)

            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels on bars
            for bars in [bars_full, bars_matched, bars_alt]:
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2.,
                            f'{width:.3f}',
                            ha='left', va='center', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Add change indicators (comparing matched → alt)
            for i, feature in enumerate(features):
                fuel_change = alt_vals[i] - matched_vals[i]

                if abs(fuel_change) > 0.05:
                    max_val = max(full_vals[i], matched_vals[i], alt_vals[i])
                    x_pos = max_val + 0.02
                    arrow = '↑' if fuel_change > 0 else '↓'
                    color = 'green' if fuel_change > 0 else 'red'
                    ax.text(x_pos, i, f'{arrow}{abs(fuel_change):.2f}',
                            ha='left', va='center', fontsize=9,
                            color=color, fontweight='bold')

            # Add sample size information
            n_full = row['n_conventional_full']
            n_matched = row['n_conventional_matched']
            n_alt = row['n_alternative']

            sample_text = f'n: Full={n_full}, Match={n_matched}, Alt={n_alt}'
            ax.text(0.02, 0.98, sample_text,
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if n_targets % 2 == 1:
            axes[-1].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Importance comparison plot saved to {save_path}")

        plt.show()

    def plot_importance_changes(self,
                                effect: str = 'fuel',
                                threshold: float = 0.05,
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None):
        """
        Plot feature importance changes.

        Parameters
        ----------
        effect : str
            Which effect to plot: 'distribution', 'fuel', or 'total'
        threshold : float
            Minimum change to display (in 0-1 scale)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """

        changes_df = self.get_importance_changes(threshold=threshold,
                                                 effect=effect,
                                                 sort_by='absolute')

        if len(changes_df) == 0:
            print(f"No {effect} changes exceed threshold of {threshold * 100:.1f} percentage points")
            return

        # Filter to significant changes
        significant = changes_df[changes_df['significant']]

        fig, ax = plt.subplots(figsize=figsize)

        # Create labels combining target and feature
        labels = [f"{row['target']}:{row['feature']}"
                  for _, row in significant.iterrows()]
        changes = significant['percentage_point_change'].values

        # Color code by direction
        colors = ['green' if x > 0 else 'red' for x in changes]

        bars = ax.barh(labels, changes, color=colors, alpha=0.7, edgecolor='black')

        effect_labels = {
            'distribution': 'Distribution Effect (Full → Matched)',
            'fuel': 'Fuel Effect (Matched → Alternative)',
            'total': 'Total Effect (Full → Alternative)'
        }

        ax.set_xlabel('Change in Importance (percentage points)', fontweight='bold', fontsize=12)
        ax.set_title(f'Feature Importance Changes - {effect_labels[effect]}\n'
                     f'(Threshold: ±{threshold * 100:.1f} percentage points)',
                     fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linewidth=2)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, changes):
            width = bar.get_width()
            label_x = width + (1 if width > 0 else -1)
            ax.text(label_x, bar.get_y() + bar.get_height() / 2.,
                    f'{val:+.1f}',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=9, fontweight='bold')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Increased importance'),
            Patch(facecolor='red', alpha=0.7, label='Decreased importance')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Importance changes plot saved to {save_path}")

        plt.show()

    def plot_cross_evaluation(self,
                              figsize: Tuple[int, int] = (16, 6),
                              save_path: Optional[str] = None):
        """
        Plot three-way cross-evaluation results.

        Shows how models transfer between Full/Matched/Alternative datasets.
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        results_df = self.comparison_results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        targets = results_df['target'].values
        x = np.arange(len(targets))
        width = 0.2

        # R² comparison
        full_r2 = results_df['conv_full_r2'].values
        matched_r2 = results_df['conv_matched_r2'].values
        alt_r2 = results_df['alt_r2'].values
        cross_full_alt = results_df['cross_full_on_alt_r2'].values
        cross_matched_alt = results_df['cross_matched_on_alt_r2'].values

        ax1.bar(x - 2 * width, full_r2, width, label='Full on Full',
                color='royalblue', alpha=0.8)
        ax1.bar(x - width, matched_r2, width, label='Matched on Matched',
                color='steelblue', alpha=0.8)
        ax1.bar(x, alt_r2, width, label='Alt on Alt',
                color='forestgreen', alpha=0.8)
        ax1.scatter(x + width, cross_full_alt, marker='x', s=100,
                    color='red', label='Full on Alt', zorder=5)
        ax1.scatter(x + 2 * width, cross_matched_alt, marker='+', s=100,
                    color='darkred', label='Matched on Alt', zorder=5)

        ax1.set_ylabel('R²', fontweight='bold', fontsize=12)
        ax1.set_title('Three-Way Model Transfer (R²)', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(axis='y', alpha=0.3)

        # RRMSE comparison
        full_rrmse = results_df['conv_full_rrmse'].values
        matched_rrmse = results_df['conv_matched_rrmse'].values
        alt_rrmse = results_df['alt_rrmse'].values

        ax2.plot(x, full_rrmse, 'o-', label='Full on Full',
                 color='royalblue', linewidth=2, markersize=8)
        ax2.plot(x, matched_rrmse, 's-', label='Matched on Matched',
                 color='steelblue', linewidth=2, markersize=8)
        ax2.plot(x, alt_rrmse, '^-', label='Alt on Alt',
                 color='forestgreen', linewidth=2, markersize=8)

        ax2.set_ylabel('RRMSE (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Three-Way Model Transfer (RRMSE)', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(targets, rotation=45, ha='right')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Random Forest Three-Way Cross-Evaluation',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Cross-evaluation plot saved to {save_path}")

        plt.show()

    # ========================================================================
    # EXPORT METHODS - Unchanged
    # ========================================================================

    def export_results(self, filename: str = 'rf_importance_comparison_threeway.csv'):
        """Export three-way comparison results to CSV."""

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to export. Run analyze_all_targets() first.")
            return

        # Flatten importance dictionaries
        export_df = self.comparison_results.copy()

        export_df['importance_conventional_full'] = export_df['importance_conventional_full'].apply(str)
        export_df['importance_conventional_matched'] = export_df['importance_conventional_matched'].apply(str)
        export_df['importance_alternative'] = export_df['importance_alternative'].apply(str)
        export_df['importance_changes'] = export_df['importance_changes'].apply(str)

        if 'perm_importance_full' in export_df.columns:
            export_df['perm_importance_full'] = export_df['perm_importance_full'].apply(str)
            export_df['perm_importance_matched'] = export_df['perm_importance_matched'].apply(str)
            export_df['perm_importance_alternative'] = export_df['perm_importance_alternative'].apply(str)

        export_df.to_csv(filename, index=False)

        if self.verbose:
            print(f"✅ Results exported to {filename}")
            print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def export_decomposition(self, filename: str = 'rf_importance_decomposition.csv'):
        """Export effect decomposition to CSV."""

        decomp_df = self.decompose_importance_changes()

        if len(decomp_df) > 0:
            decomp_df.to_csv(filename, index=False)
            if self.verbose:
                print(f"✅ Decomposition exported to {filename}")