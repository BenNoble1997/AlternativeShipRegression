"""
Random Forest Full Feature Importance Analyzer
==============================================

Analyzes feature importance using ALL available predictors (not just pairs)
for each target variable, comparing across conventional, matched, and
alternative fuel datasets.

Key Differences from RFImportanceComparisonAnalyzer:
---------------------------------------------------
- Uses ALL valid predictors for each target (not just best selector combinations)
- Demonstrates Random Forest's ability to handle many features
- Better suited for understanding overall feature hierarchy
- Not constrained by selector results

Workflow:
--------
1. For each target, identify all valid predictors (excluding forbidden)
2. Train RF on conventional (full) with all predictors
3. Train RF on conventional (matched) with all predictors
4. Train RF on alternative with all predictors
5. Compare feature importances across datasets
6. Decompose into distribution vs fuel effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import math

from sklearn.ensemble import RandomForestRegressor
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.variable_combination_manager_class import VariableCombinationManager
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from config import LoggerMixin


warnings.filterwarnings('ignore')


class RFFullFeatureImportanceAnalyzer(LoggerMixin):
    """
    Analyze feature importance using ALL available predictors for each target.

    Three-way comparison:
    1. Conventional (Full) - all conventional ships, all predictors
    2. Conventional (Matched) - matched conventional ships, all predictors
    3. Alternative - alternative fuel ships, all predictors

    This reveals:
    - Which features matter most when ALL information is available
    - How feature hierarchies change with fuel type
    - Random Forest's ability to handle high-dimensional problems
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: Optional[List[str]] = None,
                 relationship_context: Optional[Dict[str, List[str]]] = None,
                 forbidden_combinations: Optional[List[Tuple[str, ...]]] = None,
                 rf_params: Optional[Dict] = None,
                 calculate_permutation_importance: bool = False,
                 verbose: bool = True):
        """
        Initialize Full Feature Importance Analyzer.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_data : pd.DataFrame
            Alternative fuel ship dataset
        variable_dimensions : List[str]
            List of all potential dimensions
        target_dimensions : List[str], optional
            Target dimensions. If None, uses variable_dimensions.
        relationship_context : dict, optional
            Exclusion rules: {dimension: [excluded_dimensions]}
            Example: {'LOA (M)': ['LBP (M)', 'C_B']}
        forbidden_combinations : list of tuples, optional
            Specific forbidden predictor pairs
        rf_params : dict, optional
            RandomForestRegressor parameters
        calculate_permutation_importance : bool, default=False
            If True, calculates permutation importance (slower but more reliable)
        verbose : bool
            Print detailed output
        """
        self.verbose = verbose
        self.calculate_permutation_importance = calculate_permutation_importance

        # Setup combination manager for validation
        self.combination_manager = VariableCombinationManager(
            training_data=conventional_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            fixed_dimensions=[],  # No fixed dimensions for full feature analysis
            relationship_context=relationship_context,
            forbidden_combinations=forbidden_combinations,
            verbose=verbose
        )

        # Store datasets
        self.conventional_full = conventional_data[self.combination_manager.available_dimensions].copy()
        self.alternative = alternative_data[self.combination_manager.available_dimensions].copy()

        # Default RF parameters (tuned for many features)
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 15,  # Slightly deeper for many features
            'max_features': 'sqrt',  # sqrt(n_features) at each split
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        }

        # Storage
        self.comparison_results = None
        self.fitted_models = {
            'conventional_full': {},
            'conventional_matched': {},
            'alternative': {}
        }
        self.matched_datasets = {}

        # Validate data quality
        self._validate_data_quality()

        if verbose:
            print("\n" + "=" * 120)
            print("RANDOM FOREST FULL FEATURE IMPORTANCE ANALYZER")
            print("Uses ALL available predictors (not just pairs)")
            print("=" * 120)
            print(f"  Conventional ships (full): {len(conventional_data)}")
            print(f"  Alternative ships:         {len(alternative_data)}")
            print(f"  Available dimensions:      {len(self.combination_manager.available_dimensions)}")
            print(f"  Targets to analyze:        {len(self.combination_manager.target_dimensions)}")
            print(f"  RF parameters:             n_estimators={self.rf_params.get('n_estimators', 100)}, "
                  f"max_depth={self.rf_params.get('max_depth', 15)}, "
                  f"max_features={self.rf_params.get('max_features', 'sqrt')}")
            print(f"  Permutation test:          {'Enabled (slower)' if calculate_permutation_importance else 'Disabled (faster)'}")

    def _validate_data_quality(self):
        """Check data quality and warn about issues."""
        if not self.verbose:
            return

        for dataset_name, dataset in [
            ('conventional', self.conventional_full),
            ('alternative', self.alternative)
        ]:
            # Check for NaN
            nan_counts = dataset.isna().sum()
            if nan_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has missing values:")
                for col, count in nan_counts[nan_counts > 0].items():
                    pct = 100 * count / len(dataset)
                    print(f"    {col}: {count} ({pct:.1f}%)")

            # Check minimum size
            if len(dataset) < 10:
                print(f"\n  ⚠️  WARNING: {dataset_name} dataset is very small "
                      f"({len(dataset)} ships)")

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def _check_predictor_availability(self,
                                      predictors: List[str],
                                      target: str,
                                      min_coverage_pct: float = 70.0) -> Tuple[List[str], Dict]:
        """
        Filter predictors based on data availability in alternative dataset.

        Parameters
        ----------
        predictors : list of str
            Candidate predictor variables
        target : str
            Target variable
        min_coverage_pct : float, default=70.0
            Minimum percentage of alternative ships that must have data
            for both predictor and target

        Returns
        -------
        valid_predictors : list of str
            Predictors with sufficient data coverage
        diagnostics : dict
            Detailed information about filtering decisions
        """
        n_alt_total = len(self.alternative)

        # Check target availability first
        target_available = self.alternative[target].notna().sum()
        target_coverage = 100 * target_available / n_alt_total

        diagnostics = {
            'target': target,
            'n_alternative_total': n_alt_total,
            'target_available': target_available,
            'target_coverage_pct': target_coverage,
            'min_coverage_threshold': min_coverage_pct,
            'predictor_diagnostics': []
        }

        # If target itself has insufficient coverage, return empty
        if target_coverage < min_coverage_pct:
            diagnostics['target_insufficient'] = True
            diagnostics['reason'] = f"Target has only {target_coverage:.1f}% coverage (< {min_coverage_pct}%)"
            return [], diagnostics

        diagnostics['target_insufficient'] = False

        # Check each predictor
        valid_predictors = []

        for pred in predictors:
            # Count rows where BOTH predictor AND target are available
            both_available = (
                    self.alternative[pred].notna() &
                    self.alternative[target].notna()
            ).sum()

            coverage_pct = 100 * both_available / n_alt_total

            pred_info = {
                'predictor': pred,
                'n_available': both_available,
                'coverage_pct': coverage_pct,
                'sufficient': coverage_pct >= min_coverage_pct
            }

            diagnostics['predictor_diagnostics'].append(pred_info)

            if coverage_pct >= min_coverage_pct:
                valid_predictors.append(pred)

        # Sort diagnostics by coverage (lowest first for visibility)
        diagnostics['predictor_diagnostics'].sort(key=lambda x: x['coverage_pct'])

        return valid_predictors, diagnostics

    def _filter_results_by_performance(self,
                                       results_df: pd.DataFrame,
                                       min_r2: Optional[float] = None,
                                       dataset: str = 'alternative',
                                       min_n_samples: Optional[int] = None,
                                       verbose: bool = True) -> pd.DataFrame:
        """
        Filter results by model performance criteria.

        Concept:
        --------
        Removes targets where models perform poorly, as feature importance
        from low-quality models is unreliable. You can filter based on:
        - R² threshold (prediction quality)
        - Dataset choice (which model's performance matters)
        - Sample size (statistical reliability)

        Data Flow:
        ----------
        1. Start with full results_df
        2. Build boolean mask based on criteria
        3. Apply mask to filter rows
        4. Return filtered DataFrame

        Parameters
        ----------
        results_df : pd.DataFrame
            Full comparison results
        min_r2 : float, optional
            Minimum R² threshold. Filters targets where specified dataset
            R² < min_r2. If None, no R² filtering applied.
        dataset : str, default='alternative'
            Which dataset's R² to use for filtering:
            - 'alternative': Filter by alt_r2 (most restrictive)
            - 'conventional_full': Filter by conv_full_r2
            - 'conventional_matched': Filter by conv_matched_r2
            - 'all': Keep only if ALL three datasets meet threshold
            - 'any': Keep if ANY dataset meets threshold
        min_n_samples : int, optional
            Minimum sample size in alternative dataset. If None, no filtering.
        verbose : bool
            Print filtering summary

        Returns
        -------
        filtered_df : pd.DataFrame
            Results meeting performance criteria
        """

        if len(results_df) == 0:
            return results_df

        original_count = len(results_df)
        mask = pd.Series([True] * len(results_df), index=results_df.index)

        filtering_applied = []

        # R² filtering
        if min_r2 is not None:
            if dataset == 'alternative':
                r2_mask = results_df['alt_r2'] >= min_r2
                filtering_applied.append(f"alt_r2 >= {min_r2:.3f}")
            elif dataset == 'conventional_full':
                r2_mask = results_df['conv_full_r2'] >= min_r2
                filtering_applied.append(f"conv_full_r2 >= {min_r2:.3f}")
            elif dataset == 'conventional_matched':
                r2_mask = results_df['conv_matched_r2'] >= min_r2
                filtering_applied.append(f"conv_matched_r2 >= {min_r2:.3f}")
            elif dataset == 'all':
                r2_mask = (
                        (results_df['conv_full_r2'] >= min_r2) &
                        (results_df['conv_matched_r2'] >= min_r2) &
                        (results_df['alt_r2'] >= min_r2)
                )
                filtering_applied.append(f"all R² >= {min_r2:.3f}")
            elif dataset == 'any':
                r2_mask = (
                        (results_df['conv_full_r2'] >= min_r2) |
                        (results_df['conv_matched_r2'] >= min_r2) |
                        (results_df['alt_r2'] >= min_r2)
                )
                filtering_applied.append(f"any R² >= {min_r2:.3f}")
            else:
                raise ValueError(f"Invalid dataset: {dataset}. Must be one of: "
                                 "'alternative', 'conventional_full', 'conventional_matched', 'all', 'any'")

            mask &= r2_mask

        # Sample size filtering
        if min_n_samples is not None:
            n_mask = results_df['n_alternative'] >= min_n_samples
            mask &= n_mask
            filtering_applied.append(f"n_alternative >= {min_n_samples}")

        # Apply filter
        filtered_df = results_df[mask].copy()
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count

        # Report filtering results
        if verbose and (min_r2 is not None or min_n_samples is not None):
            print("\n" + "=" * 80)
            print("PERFORMANCE FILTERING APPLIED")
            print("=" * 80)
            print(f"Criteria: {' AND '.join(filtering_applied)}")
            print(f"\nResults:")
            print(f"  Original targets:  {original_count}")
            print(f"  Retained targets:  {filtered_count}")
            print(f"  Removed targets:   {removed_count}")

            if removed_count > 0:
                print(f"\nRemoved targets (poor performance):")
                removed_targets = results_df[~mask]['target'].tolist()
                for target in removed_targets:
                    row = results_df[results_df['target'] == target].iloc[0]
                    print(f"  • {target:<20s} | "
                          f"Full R²: {row['conv_full_r2']:.3f} | "
                          f"Matched R²: {row['conv_matched_r2']:.3f} | "
                          f"Alt R²: {row['alt_r2']:.3f} | "
                          f"n_alt: {row['n_alternative']}")

            print("=" * 80 + "\n")

        return filtered_df

    def analyze_all_targets(self,
                            min_alternative_samples: int = 10,
                            min_matched_samples: int = 10,
                            min_predictors: int = 3,
                            min_coverage_pct: float = 70.0,  # NEW PARAMETER
                            display_results: bool = True) -> pd.DataFrame:
        """
        Analyze feature importance for all targets using ALL available predictors.

        Parameters
        ----------
        min_alternative_samples : int
            Minimum number of alternative fuel ships required
        min_matched_samples : int
            Minimum number of matched conventional ships required
        min_predictors : int
            Minimum number of valid predictors required to proceed
        min_coverage_pct : float, default=70.0
            Minimum percentage of alternative ships that must have data
            for each predictor-target pair
        display_results : bool
            Whether to print results during analysis

        Returns
        -------
        pd.DataFrame
            Comparison results
        """

        if display_results and self.verbose:
            print("\n" + "=" * 120)
            print("FULL FEATURE IMPORTANCE COMPARISON (THREE-WAY)")
            print("Training models with ALL available predictors per target")
            print(f"Data coverage threshold: {min_coverage_pct:.0f}% of alternative ships")
            print("=" * 120)

        all_results = []
        all_diagnostics = []  # Store diagnostics for summary

        for target in self.combination_manager.target_dimensions:
            if display_results and self.verbose:
                print(f"\n{'=' * 120}")
                print(f"Target: {target}")
                print(f"{'=' * 120}")

            # Get valid predictors WITH diagnostics
            predictors, diagnostics = self._get_all_valid_predictors(
                target=target,
                min_coverage_pct=min_coverage_pct
            )

            # Store diagnostics
            all_diagnostics.append(diagnostics)

            # Display diagnostics
            if display_results and self.verbose:
                self._print_availability_diagnostics(diagnostics)

            # Check if we have enough predictors
            if len(predictors) < min_predictors:
                if self.verbose:
                    print(f"  ⚠️  Insufficient predictors after filtering: {len(predictors)} < {min_predictors}")
                continue

            # Analyze this target
            result = self._analyze_single_target_threeway(
                target=target,
                predictors=tuple(predictors),
                min_alternative_samples=min_alternative_samples,
                min_matched_samples=min_matched_samples,
                display_results=display_results
            )

            if result is not None:
                all_results.append(result)

        # Create summary DataFrame
        if not all_results:
            print("\n⚠️  No valid results generated")

            # Store diagnostics even if no results
            self.availability_diagnostics = all_diagnostics
            self._print_overall_availability_summary(all_diagnostics)

            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Store results and diagnostics
        self.comparison_results = results_df
        self.availability_diagnostics = all_diagnostics

        if display_results and self.verbose:
            self._print_summary_table(results_df)
            self._print_overall_availability_summary(all_diagnostics)

        return results_df

    # ========================================================================
    # PREDICTOR SELECTION
    # ========================================================================

    def _get_all_valid_predictors(self,
                                  target: str,
                                  min_coverage_pct: float = 70.0) -> Tuple[List[str], Dict]:
        """
        Get all valid predictors for a target with data availability filtering.

        Now returns both predictors and diagnostics!

        Parameters
        ----------
        target : str
            Target dimension
        min_coverage_pct : float, default=70.0
            Minimum percentage of alternative ships that must have data

        Returns
        -------
        valid_predictors : list of str
            Valid predictor dimensions with sufficient data
        diagnostics : dict
            Detailed filtering diagnostics
        """
        # Step 1: Get candidates (relationship context filtering)
        candidate_predictors = [d for d in self.combination_manager.available_dimensions
                                if d != target]

        # Remove relationship context exclusions
        if target in self.combination_manager.relationship_context:
            excluded = set(self.combination_manager.relationship_context[target])
            candidate_predictors = [p for p in candidate_predictors if p not in excluded]

        # Step 2: Filter by data availability in alternative dataset
        valid_predictors, diagnostics = self._check_predictor_availability(
            predictors=candidate_predictors,
            target=target,
            min_coverage_pct=min_coverage_pct
        )

        return valid_predictors, diagnostics

    # ========================================================================
    # SINGLE TARGET ANALYSIS (THREE-WAY)
    # ========================================================================

    def _analyze_single_target_threeway(self,
                                       target: str,
                                       predictors: Tuple[str, ...],
                                       min_alternative_samples: int,
                                       min_matched_samples: int,
                                       display_results: bool) -> Optional[Dict]:
        """
        Analyze feature importance for a single target using all predictors.

        Returns
        -------
        dict or None
            Results dictionary or None if analysis failed
        """

        # ====================================================================
        # PREPARE DATA - THREE DATASETS
        # ====================================================================

        # 1. Conventional (Full)
        conv_full_data = self._prepare_data(self.conventional_full, predictors, target)
        if conv_full_data is None:
            if self.verbose:
                print(f"  ⚠️  Insufficient conventional data")
            return None
        X_conv_full, y_conv_full = conv_full_data

        # 2. Alternative
        alt_data = self._prepare_data(self.alternative, predictors, target)
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

        # 3. Conventional (Matched) - create if needed
        predictor_key = tuple(predictors)

        if predictor_key not in self.matched_datasets:
            if self.verbose:
                print(f"  Creating matched dataset for {len(predictors)} predictors...")

            try:
                matched_dict = MatchedDataset.multiple_matched_datasets(
                    alternative_fleet=self.alternative,
                    conventional_fleet=self.conventional_full,
                    logger=self.logger,
                    predictor_groups=[predictors],
                    std_reference='conventional'
                )

                if predictor_key in matched_dict:
                    self.matched_datasets[predictor_key] = matched_dict[predictor_key]
                    if self.verbose:
                        print(f"  ✅ Matched dataset created: {len(matched_dict[predictor_key])} ships")
                else:
                    if self.verbose:
                        print(f"  ⚠️  Matched dataset creation returned no data")
                    return None

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Matched dataset creation failed: {e}")
                return None

        matched_df = self.matched_datasets[predictor_key]
        conv_matched_data = self._prepare_data(matched_df, predictors, target)

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
        # FIT THREE RANDOM FOREST MODELS
        # ====================================================================

        # 1. Fit on conventional (full)
        try:
            conv_full_rf = RandomForestRegressor(**self.rf_params)
            conv_full_rf.fit(X_conv_full, y_conv_full)
            y_pred_conv_full = conv_full_rf.predict(X_conv_full)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Conventional (full) RF fitting failed: {e}")
            return None

        # 2. Fit on conventional (matched)
        try:
            conv_matched_rf = RandomForestRegressor(**self.rf_params)
            conv_matched_rf.fit(X_conv_matched, y_conv_matched)
            y_pred_conv_matched = conv_matched_rf.predict(X_conv_matched)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Conventional (matched) RF fitting failed: {e}")
            return None

        # 3. Fit on alternative
        try:
            alt_rf = RandomForestRegressor(**self.rf_params)
            alt_rf.fit(X_alt, y_alt)
            y_pred_alt = alt_rf.predict(X_alt)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Alternative RF fitting failed: {e}")
            return None

        # Store fitted models
        key = (predictor_key, target)
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
                print(f"  Calculating permutation importance (this may take a while with {len(predictors)} features)...")

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
                target, predictors,
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
            'n_predictors': len(predictors),
            'predictors': predictors,

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

            # OOB scores if available
            'oob_full': conv_full_rf.oob_score_ if hasattr(conv_full_rf, 'oob_score_') else np.nan,
            'oob_matched': conv_matched_rf.oob_score_ if hasattr(conv_matched_rf, 'oob_score_') else np.nan,
            'oob_alt': alt_rf.oob_score_ if hasattr(alt_rf, 'oob_score_') else np.nan,
        }

        return result

    # ========================================================================
    # IMPORTANCE COMPARISON (THREE-WAY) - Same as RFImportanceComparisonAnalyzer
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
    # HELPER METHODS
    # ========================================================================

    def _prepare_data(self,
                      data: pd.DataFrame,
                      predictors: Tuple[str, ...],
                      target: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare X, y arrays with data cleaning."""
        try:
            X = data[list(predictors)].values
            y = data[target].values

            # Clean data
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                     np.isinf(X).any(axis=1) | np.isinf(y))

            X_clean = X[mask]
            y_clean = y[mask]

            if len(X_clean) < 10:  # Need at least 10 samples
                return None

            return X_clean, y_clean

        except KeyError as e:
            if self.verbose:
                print(f"  ⚠️  Missing column: {e}")
            return None

    def _calculate_metrics(self, y_true, y_pred, n_features) -> Dict:
        """Calculate regression metrics."""
        metric_calc = RegressionMetrics(y_true, y_pred)
        return metric_calc.calculate_all_metrics(n_features)

    # ========================================================================
    # ANALYSIS METHODS (Similar to RFImportanceComparisonAnalyzer)
    # ========================================================================

    def get_importance_changes(self,
                               threshold: float = 0.05,
                               effect: str = 'fuel',
                               sort_by: str = 'absolute',
                               top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get summary of feature importance changes.

        Parameters
        ----------
        threshold : float, default=0.05
            Minimum change in importance (0-1 scale) to be considered significant
        effect : str, default='fuel'
            Which effect to analyze: 'distribution', 'fuel', or 'total'
        sort_by : str, default='absolute'
            How to sort: 'absolute', 'percentage', or 'rank'
        top_n : int, optional
            If provided, returns only top N changes per target

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
                    'rank_full': comp['rank_full'],
                    'rank_matched': comp['rank_matched'],
                    'rank_alternative': comp['rank_alternative'],
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

        # Apply top_n filter per target if requested
        if top_n is not None:
            changes_df = (changes_df
                         .groupby('target').head(top_n).reset_index(drop=True))

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

        for _, row in significant.head(20).iterrows():  # Show top 20
            emoji = "📈" if row['absolute_change'] > 0 else "📉"

            print(f"{emoji} {row['target']:<13s} | {row['feature']:<20s} | "
                  f"{row['importance_full']:>8.3f} | {row['importance_matched']:>8.3f} | "
                  f"{row['importance_alt']:>8.3f} | "
                  f"{row['percentage_point_change']:>+10.2f} | {row['rank_change']:>+8d} | "
                  f"{row['direction']:<12s}")

        if len(significant) > 20:
            print(f"\n  ... and {len(significant) - 20} more significant changes")

        print("=" * 120)

        # Summary statistics
        gained = (significant['absolute_change'] > 0).sum()
        lost = (significant['absolute_change'] < 0).sum()

        print(f"\nSignificant Changes ({effect}): {len(significant)}")
        print(f"  Features gaining importance:  {gained}")
        print(f"  Features losing importance:   {lost}")

    def decompose_importance_changes(self,
                                     effect_size_threshold: float = 0.05,
                                     top_n_per_target: Optional[int] = 10) -> pd.DataFrame:
        """
        Decompose importance changes into distribution vs fuel effects.

        Parameters
        ----------
        effect_size_threshold : float
            Minimum change to be considered significant (in 0-1 scale)
        top_n_per_target : int, optional
            If provided, shows only top N features per target

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

        # Apply top_n filter if requested
        if top_n_per_target is not None:
            decomp_df = (decomp_df
                         .sort_values(['target', 'fuel_effect'], key=lambda x: abs(x) if x.name == 'fuel_effect' else x)
                         .groupby('target')
                         .head(top_n_per_target)
                         .reset_index(drop=True))

        if self.verbose:
            self._print_decomposition_summary(decomp_df, effect_size_threshold)

        return decomp_df

    def _print_decomposition_summary(self, decomp_df: pd.DataFrame,
                                     threshold: float):
        """Print formatted decomposition summary."""

        print("\n" + "=" * 140)
        print("IMPORTANCE CHANGE DECOMPOSITION (Full Feature Analysis)")
        print("Distribution Effect (Full → Matched) vs Fuel Effect (Matched → Alternative)")
        print("=" * 140)

        print(f"\nEffect Size Thresholds:")
        print(f"  Negligible: < {threshold * 100:.0f}pp")
        print(f"  Small:      {threshold * 100:.0f}-10pp")
        print(f"  Medium:     10-20pp")
        print(f"  Large:      > 20pp")

        print("\n" + "-" * 140)
        print(f"{'':2s} {'Target':<15s} | {'Feature':<20s} | "
              f"{'Full':>8s} → {'Match':>8s} → {'Alt':>8s} | "
              f"{'Dist Δ':>8s} | {'Fuel Δ':>8s} | {'Fuel %':>7s} | {'Effect':<12s}")
        print("-" * 140)

        # Show top 30
        for _, row in decomp_df.head(30).iterrows():
            print(f"{row['emoji']:2s} {row['target']:<15s} | {row['feature']:<20s} | "
                  f"{row['full']:>8.3f} → {row['matched']:>8.3f} → {row['alternative']:>8.3f} | "
                  f"{row['distribution_effect']:>+8.3f} | {row['fuel_effect']:>+8.3f} | "
                  f"{row['fuel_pct']:>7.1f} | "
                  f"{row['fuel_effect_size']:<12s}")

        if len(decomp_df) > 30:
            print(f"\n  ... and {len(decomp_df) - 30} more features")

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
    # DISPLAY METHODS
    # ========================================================================

    def _print_single_target_results(self,
                                     target: str,
                                     predictors: tuple[str, ...],
                                     conv_full_importance: dict,
                                     conv_matched_importance: dict,
                                     alt_importance: dict|None,
                                     importance_comparison: dict,
                                     conv_full_metrics: dict,
                                     conv_matched_metrics: dict,
                                     alt_metrics: dict|None,
                                     cross_evals: dict|None,
                                     perm_importance_full: dict|None,
                                     perm_importance_matched: dict|None,
                                     perm_importance_alt: dict|None):
        """Print detailed results for single target."""

        print(f"\n  Number of predictors: {len(predictors)}")
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

        print(f"\n  Top 10 Most Important Features (Full Conventional):")
        print(f"  {'Feature':<20s} | {'Full':>10s} | {'Match':>10s} | {'Alt':>10s} | "
              f"{'Fuel Δ':>10s}")
        print(f"  {'-' * 70}")

        # Sort by full importance
        sorted_features = sorted(conv_full_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        for feature, full_imp in sorted_features:
            comp = importance_comparison[feature]
            print(f"  {feature:<20s} | {full_imp:>10.3f} | {comp['matched']:>10.3f} | "
                  f"{comp['alternative']:>10.3f} | {comp['fuel_effect']:>+10.3f}")

    def _print_summary_table(self, results_df: pd.DataFrame):
        """Print summary table of all results."""

        print("\n" + "=" * 140)
        print("SUMMARY: Full Feature Importance Comparison")
        print("=" * 140)

        print(f"\n{'Target':<15s} | {'N Feat':>7s} | {'Full R²':>8s} | {'Match R²':>9s} | {'Alt R²':>8s} | "
              f"{'Top (Full)':<20s} | {'Top (Alt)':<20s} | {'Switched?':<10s}")
        print("-" * 140)

        for _, row in results_df.iterrows():
            top_full = row['top_full']
            top_alt = row['top_alternative']

            switched = "Yes ⚠️" if top_full != top_alt else ""

            print(
                f"{row['target']:<15s} | {row['n_predictors']:>7d} | {row['conv_full_r2']:>8.4f} | {row['conv_matched_r2']:>9.4f} | "
                f"{row['alt_r2']:>8.4f} | "
                f"{top_full:<20s} | {top_alt:<20s} | {switched:<10s}")

        print("=" * 140)

        # Summary statistics
        switched_total = (results_df['top_full'] != results_df['top_alternative']).sum()
        avg_features = results_df['n_predictors'].mean()
        avg_r2_full = results_df['conv_full_r2'].mean()
        avg_r2_alt = results_df['alt_r2'].mean()

        print(f"\n  Top feature changes (Full → Alt): {switched_total}/{len(results_df)}")
        print(f"  Average predictors per target:    {avg_features:.1f}")
        print(f"  Average R² (Full):                {avg_r2_full:.4f}")
        print(f"  Average R² (Alternative):         {avg_r2_alt:.4f}")
        print(f"  Average R² gap:                   {avg_r2_alt - avg_r2_full:+.4f}")

    # ========================================================================
    # PLOTTING METHODS (Similar to RFImportanceComparisonAnalyzer but adapted)
    # ========================================================================

    def plot_importance_comparison(self,
                                   target: Optional[str] = None,
                                   top_n: int = 15,
                                   min_r2: Optional[float] = 0.9,
                                   filter_dataset: str = 'alternative',
                                   min_n_samples: Optional[int] = None,
                                   figsize: Tuple[int, int] = (14, 10),
                                   save_path: Optional[str] = None,
                                   show_values: bool = False,
                                   value_format: str = '.3f',
                                   title_prefix: str = '',
                                   title_suffix: str = '',
                                   debug:bool=False):
        """
        Plot feature importance comparison (top N features only).

        Parameters
        ----------
        target : str, optional
            Specific target to plot. If None, plots all.
        top_n : int, default=15
            Number of top features to show per target
        figsize : tuple
            Figure size per target
        save_path : str, optional
            Path to save figure
        show_values : bool, default=False
            If True, display importance values next to bars
        value_format : str, default='.3f'
            Format string for value labels (e.g., '.3f', '.2%', '.4f')
        title_prefix : str, default=''
            Text to add before the main title
        title_suffix : str, default=''
            Text to add after the main title
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        # Debug: Initial state
        if debug:
            print("\n🔍 DEBUG: plot_importance_comparison")
            print(f"   Initial results count: {len(self.comparison_results)}")
            print(
                f"   Filtering parameters: min_r2={min_r2}, dataset={filter_dataset}, min_n_samples={min_n_samples}")

        # Apply performance filtering
        results_df = self._filter_results_by_performance(
            results_df=self.comparison_results,
            min_r2=min_r2,
            dataset=filter_dataset,
            min_n_samples=min_n_samples,
            verbose=self.verbose or debug
        )

        # Debug: Post-filtering state
        if debug:
            print(f"   Post-filtering count: {len(results_df)}")
            if len(results_df) > 0:
                print(f"   Retained targets: {results_df['target'].tolist()}")

        if len(results_df) == 0:
            print("⚠️  No targets meet performance criteria")
            return

        if target is not None:
            results_df = results_df[results_df['target'] == target]
            if len(results_df) == 0:
                print(f"No results found for target '{target}'")
                return

        n_targets = len(results_df)

        # Create subplots
        n_cols = 2
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * 1.5, figsize[1] * n_rows))

        if n_targets == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Build the main title with optional prefix/suffix
        main_title = f'Full Feature Importance Comparison (Top {top_n} Features)'
        if title_prefix:
            main_title = f'{title_prefix} - {main_title}'
        if title_suffix:
            main_title = f'{main_title} - {title_suffix}'

        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)

        for idx, (_, row) in enumerate(results_df.iterrows()):
            ax = axes[idx]

            conv_full_imp = row['importance_conventional_full']
            conv_matched_imp = row['importance_conventional_matched']
            alt_imp = row['importance_alternative']

            # Sort by full importance, take top N
            sorted_features = sorted(conv_full_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features = [f[0] for f in sorted_features]

            full_vals = [conv_full_imp[f] for f in features]
            matched_vals = [conv_matched_imp[f] for f in features]
            alt_vals = [alt_imp[f] for f in features]

            # Setup bar positions
            y = np.arange(len(features))
            height = 0.15

            # Create bars
            bars_full = ax.barh(y + height, full_vals, height,
                                label='Conventional (Full)',
                                color='royalblue', alpha=0.8, edgecolor='black')

            bars_matched = ax.barh(y, matched_vals, height,
                                   label='Conventional (Matched)',
                                   color='steelblue', alpha=0.8, edgecolor='black')

            bars_alt = ax.barh(y - height, alt_vals, height,
                               label='Alternative',
                               color='forestgreen', alpha=0.8, edgecolor='black')

            # Add value labels if requested
            if show_values:
                # Add labels to full bars
                for bar, val in zip(bars_full, full_vals):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f' {val:{value_format}}',
                            ha='left', va='center', fontsize=7,
                            fontweight='bold')

                # Add labels to matched bars
                for bar, val in zip(bars_matched, matched_vals):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f' {val:{value_format}}',
                            ha='left', va='center', fontsize=7,
                            fontweight='bold')

                # Add labels to alternative bars
                for bar, val in zip(bars_alt, alt_vals):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f' {val:{value_format}}',
                            ha='left', va='center', fontsize=7,
                            fontweight='bold')

            # Formatting
            ax.set_xlabel('Feature Importance', fontweight='bold', fontsize=11)
            ax.set_title(f'{row["target"]} - Top {len(features)} Features',
                         fontweight='bold', fontsize=12)
            ax.set_yticks(y)
            ax.set_yticklabels(features, fontsize=9)
            ax.tick_params(axis='y', pad=1)
            ax.set_ylim(-0.5, len(features) - 0.5)

            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='x', alpha=0.3)

            # Add sample size
            n_features_total = row['n_predictors']
            sample_text = f'Using {n_features_total} total features'
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

    # ========================================================================
    # EXPORT METHODS
    # ========================================================================

    def export_results(self, filename: str = 'rf_full_feature_importance.csv'):
        """Export full feature importance results to CSV."""

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to export. Run analyze_all_targets() first.")
            return

        # Flatten importance dictionaries
        export_df = self.comparison_results.copy()

        export_df['importance_conventional_full'] = export_df['importance_conventional_full'].apply(str)
        export_df['importance_conventional_matched'] = export_df['importance_conventional_matched'].apply(str)
        export_df['importance_alternative'] = export_df['importance_alternative'].apply(str)
        export_df['importance_changes'] = export_df['importance_changes'].apply(str)
        export_df['predictors'] = export_df['predictors'].apply(lambda x: ', '.join(x))

        if 'perm_importance_full' in export_df.columns:
            export_df['perm_importance_full'] = export_df['perm_importance_full'].apply(str)
            export_df['perm_importance_matched'] = export_df['perm_importance_matched'].apply(str)
            export_df['perm_importance_alternative'] = export_df['perm_importance_alternative'].apply(str)

        export_df.to_csv(filename, index=False)

        if self.verbose:
            print(f"✅ Results exported to {filename}")
            print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def export_decomposition(self, filename: str = 'rf_full_feature_decomposition.csv'):
        """Export effect decomposition to CSV."""

        decomp_df = self.decompose_importance_changes()

        if len(decomp_df) > 0:
            decomp_df.to_csv(filename, index=False)
            if self.verbose:
                print(f"✅ Decomposition exported to {filename}")

    def _print_availability_diagnostics(self, diagnostics: Dict):
        """
        Print detailed diagnostics about predictor availability filtering.

        Parameters
        ----------
        diagnostics : dict
            Output from _check_predictor_availability
        """
        print(f"\n  Data Availability Check:")
        print(f"  Alternative dataset size: {diagnostics['n_alternative_total']} ships")
        print(f"  Minimum coverage threshold: {diagnostics['min_coverage_threshold']:.0f}%")

        # Target availability
        print(f"\n  Target '{diagnostics['target']}':")
        print(f"    Available: {diagnostics['target_available']}/{diagnostics['n_alternative_total']} "
              f"({diagnostics['target_coverage_pct']:.1f}%)")

        if diagnostics.get('target_insufficient'):
            print(f"    ❌ {diagnostics['reason']}")
            return

        print(f"    ✅ Sufficient coverage")

        # Predictor availability
        pred_diag = diagnostics['predictor_diagnostics']

        insufficient = [p for p in pred_diag if not p['sufficient']]
        sufficient = [p for p in pred_diag if p['sufficient']]

        if insufficient:
            print(f"\n  ❌ Predictors with INSUFFICIENT coverage ({len(insufficient)}):")
            for p in insufficient:
                print(f"    • {p['predictor']:<20s}: {p['n_available']:>3d}/{diagnostics['n_alternative_total']} "
                      f"({p['coverage_pct']:>5.1f}%) - EXCLUDED")

        if sufficient:
            print(f"\n  ✅ Predictors with SUFFICIENT coverage ({len(sufficient)}):")
            for p in sufficient[:10]:  # Show first 10
                print(f"    • {p['predictor']:<20s}: {p['n_available']:>3d}/{diagnostics['n_alternative_total']} "
                      f"({p['coverage_pct']:>5.1f}%)")

            if len(sufficient) > 10:
                print(f"    • ... and {len(sufficient) - 10} more")

        print(f"\n  Final predictor count: {len(sufficient)}/{len(pred_diag)}")

    def _print_overall_availability_summary(self, all_diagnostics: List[Dict]):
        """
        Print summary of data availability across all targets.

        Parameters
        ----------
        all_diagnostics : list of dict
            Diagnostics from all targets
        """
        print("\n" + "=" * 120)
        print("DATA AVAILABILITY SUMMARY")
        print("=" * 120)

        # Aggregate statistics
        total_targets = len(all_diagnostics)
        targets_with_sufficient_data = sum(1 for d in all_diagnostics
                                           if not d.get('target_insufficient', False))

        print(f"\nTargets analyzed: {total_targets}")
        print(f"Targets with sufficient target coverage: {targets_with_sufficient_data}/{total_targets}")

        if targets_with_sufficient_data == 0:
            print("\n❌ No targets have sufficient data coverage in alternative dataset")
            print("Consider:")
            print("  1. Lowering min_coverage_pct threshold")
            print("  2. Collecting more alternative fuel ship data")
            print("  3. Using imputation for missing values")
            return

        # Predictor availability summary
        print(f"\n{'Target':<20s} | {'Coverage':>10s} | {'Valid Pred':>11s} | {'Excluded':>10s}")
        print("-" * 120)

        for diag in all_diagnostics:
            if diag.get('target_insufficient'):
                print(f"{diag['target']:<20s} | "
                      f"{diag['target_coverage_pct']:>9.1f}% | "
                      f"{'N/A':>11s} | {'N/A':>10s}  ❌ Insufficient target coverage")
            else:
                pred_diag = diag['predictor_diagnostics']
                n_valid = sum(1 for p in pred_diag if p['sufficient'])
                n_excluded = sum(1 for p in pred_diag if not p['sufficient'])

                status = "✅" if n_valid >= 3 else "⚠️"

                print(f"{diag['target']:<20s} | "
                      f"{diag['target_coverage_pct']:>9.1f}% | "
                      f"{n_valid:>11d} | "
                      f"{n_excluded:>10d}  {status}")

        print("=" * 120)

        # Most commonly excluded predictors
        predictor_exclusion_counts = {}

        for diag in all_diagnostics:
            if not diag.get('target_insufficient'):
                for pred_info in diag['predictor_diagnostics']:
                    if not pred_info['sufficient']:
                        pred = pred_info['predictor']
                        if pred not in predictor_exclusion_counts:
                            predictor_exclusion_counts[pred] = {
                                'count': 0,
                                'avg_coverage': []
                            }
                        predictor_exclusion_counts[pred]['count'] += 1
                        predictor_exclusion_counts[pred]['avg_coverage'].append(pred_info['coverage_pct'])

        if predictor_exclusion_counts:
            print("\nMost Frequently Excluded Predictors:")
            print(f"{'Predictor':<20s} | {'Excluded Count':>15s} | {'Avg Coverage':>13s}")
            print("-" * 60)

            sorted_exclusions = sorted(predictor_exclusion_counts.items(),
                                       key=lambda x: x[1]['count'],
                                       reverse=True)

            for pred, info in sorted_exclusions[:10]:
                avg_cov = np.mean(info['avg_coverage'])
                print(f"{pred:<20s} | {info['count']:>15d} | {avg_cov:>12.1f}%")

    def export_availability_diagnostics(self,
                                        filename: str = 'feature_availability_diagnostics.csv'):
        """
        Export data availability diagnostics to CSV.

        Parameters
        ----------
        filename : str
            Output filename
        """
        if not hasattr(self, 'availability_diagnostics'):
            print("No diagnostics available. Run analyze_all_targets() first.")
            return

        rows = []

        for diag in self.availability_diagnostics:
            target = diag['target']
            target_coverage = diag['target_coverage_pct']
            target_sufficient = not diag.get('target_insufficient', False)

            if target_sufficient:
                for pred_info in diag['predictor_diagnostics']:
                    rows.append({
                        'target': target,
                        'target_coverage_pct': target_coverage,
                        'target_sufficient': target_sufficient,
                        'predictor': pred_info['predictor'],
                        'n_available': pred_info['n_available'],
                        'coverage_pct': pred_info['coverage_pct'],
                        'sufficient': pred_info['sufficient'],
                        'threshold': diag['min_coverage_threshold']
                    })
            else:
                # Add row for insufficient target
                rows.append({
                    'target': target,
                    'target_coverage_pct': target_coverage,
                    'target_sufficient': False,
                    'predictor': None,
                    'n_available': None,
                    'coverage_pct': None,
                    'sufficient': False,
                    'threshold': diag['min_coverage_threshold']
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        if self.verbose:
            print(f"✅ Availability diagnostics exported to {filename}")
            print(f"   {len(df)} rows")

    def plot_importance_heatmap(self,
                                target: Optional[str] = None,
                                separate_figures: bool = False,
                                top_n: Optional[int] = None,
                                min_r2: Optional[float] = None,
                                filter_dataset: str = 'alternative',
                                min_n_samples: Optional[int] = None,
                                figsize: Tuple[int, int] = (10, 8),
                                save_path: Optional[str] = None,
                                title_prefix: str = '',
                                title_suffix: str = '',
                                cmap: str = 'viridis',
                                annotate: bool = True,
                                annotation_format: str = '.3f',
                                variable_name_map: Optional[Dict[str, str]] = None,
                                alternative_dataset_name: str = 'Alternative',
                                n_cols: int = 3,
                                debug: bool = False,
                                # ── text ──────────────────────────────────────
                                base_font_size: int = 11,
                                tick_font_size: Optional[int] = None,
                                annotation_font_size: Optional[int] = None,
                                suptitle_font_size: Optional[int] = None,
                                # ── layout ────────────────────────────────────
                                gridspec_top: Optional[float] = None,
                                ):
        """
        Plot feature importance comparison as a heatmap.

        Each cell shows the importance of a feature (row) for a given dataset
        (column: Conventional Full, Conventional Matched, Alternative). Colour
        intensity encodes importance magnitude.

        When ``separate_figures=False`` all targets are tiled into a single
        figure whose grid is controlled by ``n_cols``.

        :param target: Specific target to plot. If ``None``, plots all retained
            targets.
        :type target: str, optional
        :param separate_figures: If ``True`` each target gets its own figure;
            otherwise all targets share a single tiled figure.
        :type separate_figures: bool
        :param top_n: Maximum number of features (rows) to display per heatmap,
            ranked by conventional-full importance. ``None`` shows all features.
        :type top_n: int, optional
        :param min_r2: Minimum R² threshold for retaining a target.
        :type min_r2: float, optional
        :param filter_dataset: Which dataset's R² is compared against ``min_r2``.
        :type filter_dataset: str
        :param min_n_samples: Minimum number of alternative-fuel ships required.
        :type min_n_samples: int, optional
        :param figsize: ``(width, height)`` in inches for **each** subplot when
            tiling, or the single figure when ``separate_figures=True``.
        :type figsize: tuple[int, int]
        :param save_path: File path to save the figure.
        :type save_path: str, optional
        :param title_prefix: Text prepended to the main figure title.
        :type title_prefix: str
        :param title_suffix: Text appended as a subtitle.
        :type title_suffix: str
        :param cmap: Matplotlib colormap name.
        :type cmap: str
        :param annotate: If ``True``, importance values are printed inside cells.
        :type annotate: bool
        :param annotation_format: Python format string for cell annotations.
        :type annotation_format: str
        :param variable_name_map: Optional mapping from internal variable names
            to publication-friendly labels.
        :type variable_name_map: dict[str, str], optional
        :param alternative_dataset_name: Display label for the alternative-fuel
            dataset column.
        :type alternative_dataset_name: str
        :param n_cols: Number of columns in the tiled figure. Clamped to the
            number of retained targets.
        :type n_cols: int
        :param debug: Print intermediate counts and filtering decisions.
        :type debug: bool
        :param base_font_size: Base font size in points. Axis labels, subplot
            titles, colorbar labels, and the info text box all scale relative
            to this. Typical range 9–13.
        :type base_font_size: int
        :param tick_font_size: Font size for axis tick labels (feature names on
            the y-axis, dataset names on the x-axis). When ``None``, defaults
            to ``base_font_size - 1``. Set explicitly to make ticks larger or
            smaller than the rest of the text.
        :type tick_font_size: int, optional
        :param annotation_font_size: Font size for the numeric values printed
            inside heatmap cells. When ``None``, defaults to
            ``base_font_size - 1``. Set explicitly to increase readability on
            small figures.
        :type annotation_font_size: int, optional
        :param suptitle_font_size: Font size for the figure-level suptitle (the
            bold main title across all panels). When ``None``, defaults to
            ``base_font_size + 4``.
        :type suptitle_font_size: int, optional
        :param gridspec_top: Upper boundary of the subplot area as a fraction of
            figure height (0–1). Controls the gap between the suptitle and the
            first row of panels. When ``None``, it is derived automatically from
            the number of title lines. Typical range 0.88–0.96.
        :type gridspec_top: float, optional
        :returns: A single ``Figure`` when ``separate_figures=False``, or a
            ``list`` of figures when ``separate_figures=True``. ``None`` if no
            targets pass filtering.
        :rtype: matplotlib.figure.Figure or list[matplotlib.figure.Figure] or None
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return None

        if debug:
            print("\n🔍 DEBUG: plot_importance_heatmap")
            print(f"   Initial results count: {len(self.comparison_results)}")

        # ── derive font sizes from base, with optional overrides ────────────
        fs = {
            'tick': tick_font_size if tick_font_size is not None else base_font_size - 1,
            'annotation': annotation_font_size if annotation_font_size is not None else base_font_size - 1,
            'suptitle': suptitle_font_size if suptitle_font_size is not None else base_font_size + 4,
            'axis_label': base_font_size + 1,
            'subplot_title': base_font_size + 2,
            'colorbar_label': base_font_size + 1,
            'infobox': base_font_size - 2,
        }

        if debug:
            print(f"   Font sizes: {fs}")
            print(f"   figsize={figsize}, n_cols={n_cols}, gridspec_top={gridspec_top}")

        results_df = self._filter_results_by_performance(
            results_df=self.comparison_results,
            min_r2=min_r2,
            dataset=filter_dataset,
            min_n_samples=min_n_samples,
            verbose=self.verbose or debug
        )

        if debug:
            print(f"   Post-filtering count: {len(results_df)}")

        if len(results_df) == 0:
            print("⚠️  No targets meet performance criteria")
            return None

        if target is not None:
            results_df = results_df[results_df['target'] == target]
            if len(results_df) == 0:
                print(f"No results found for target '{target}'")
                return None

        if separate_figures:
            figures = []
            for _, row in results_df.iterrows():
                fig = self._create_single_heatmap(
                    row=row,
                    top_n=top_n,
                    figsize=figsize,
                    title_prefix=title_prefix,
                    title_suffix=title_suffix,
                    min_r2=min_r2,
                    filter_dataset=filter_dataset,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format,
                    variable_name_map=variable_name_map,
                    alternative_dataset_name=alternative_dataset_name,
                    fs=fs,
                    gridspec_top=gridspec_top,
                )
                figures.append(fig)

                if save_path:
                    if len(results_df) > 1:
                        base, ext = save_path.rsplit('.', 1)
                        target_path = f"{base}_{row['target'].replace(' ', '_')}.{ext}"
                    else:
                        target_path = save_path
                    fig.savefig(target_path, dpi=300, bbox_inches='tight')
                    if self.verbose:
                        print(f"✅ Heatmap saved to {target_path}")

            return figures

        else:
            n_targets = len(results_df)

            n_rows, effective_n_cols = self._optimal_grid(n_targets, n_cols)

            if debug:
                print(f"   Grid: {n_rows} rows × {effective_n_cols} cols for {n_targets} targets")

            fig = plt.figure(figsize=(figsize[0] * effective_n_cols,
                                      figsize[1] * n_rows),
                 layout='constrained')

            # Build hierarchical title
            main_title = 'Feature Importance Comparison - Heatmap View'
            if top_n is not None:
                main_title += f' (Top {top_n} Features)'

            metadata = {}
            if min_r2 is not None:
                metadata['Min R²'] = f'{min_r2:.2f} ({filter_dataset})'
            if len(results_df) < len(self.comparison_results):
                metadata['Targets shown'] = f'{len(results_df)}/{len(self.comparison_results)}'

            title_structure = self._build_hierarchical_title(
                main_title=main_title,
                title_prefix=title_prefix,
                title_suffix=title_suffix,
                metadata=metadata
            )

            self._add_hierarchical_suptitle(fig, title_structure,
                                            bold_font_size=fs['suptitle'],
                                            italic_font_size=fs['suptitle'] - 5)

            # Derive top margin: explicit override > auto from line count
            n_lines = len(title_structure['lines'])
            auto_top = {1: 0.97, 2: 0.95, 3: 0.93}.get(n_lines, 0.91)
            effective_top = gridspec_top if gridspec_top is not None else auto_top

            for idx, (_, row) in enumerate(results_df.iterrows()):
                ax = fig.add_subplot(n_rows, effective_n_cols, idx + 1)
                self._plot_heatmap_on_axis(
                    ax=ax,
                    row=row,
                    top_n=top_n,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format,
                    variable_name_map=variable_name_map,
                    alternative_dataset_name=alternative_dataset_name,
                    fs=fs,
                )

            total_cells = n_rows * effective_n_cols
            for empty_idx in range(n_targets + 1, total_cells + 1):
                ax = fig.add_subplot(n_rows, effective_n_cols, empty_idx)
                ax.set_visible(False)

            fig.get_layout_engine().set(rect=[0, 0, 1, effective_top])

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f"✅ Combined heatmap saved to {save_path}")

            plt.show()
            return fig



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
                'lines': [('text', 'style'), ...],  # style = 'bold' or 'italic' or 'normal'
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

    def _add_hierarchical_suptitle(self,
                                   fig,
                                   title_structure: Dict,
                                   bold_font_size: int = 16,
                                   italic_font_size: int = 11):
        """
        Add hierarchical title to figure using fig.text for formatting control.

        Renders the title structure produced by :meth:`_build_hierarchical_title`
        from top to bottom, using different font sizes and styles per line.
        Bold lines (prefix and main title) use ``bold_font_size``; italic lines
        (metadata and suffix) use ``italic_font_size``.

        :param fig: Figure to add the title to.
        :param title_structure: Output of :meth:`_build_hierarchical_title`.
        :type title_structure: dict
        :param bold_font_size: Font size for bold title lines.
        :type bold_font_size: int
        :param italic_font_size: Font size for italic subtitle/metadata lines.
        :type italic_font_size: int
        """

        lines = title_structure['lines']
        y_position = 0.985

        bold_line_spacing = 0.018
        italic_line_spacing = 0.015

        for text, style in lines:
            if style == 'bold':
                fig.text(0.5, y_position, text,
                         ha='center', va='top',
                         fontsize=bold_font_size, fontweight='bold',
                         transform=fig.transFigure)
                y_position -= bold_line_spacing
            elif style == 'italic':
                fig.text(0.5, y_position, text,
                         ha='center', va='top',
                         fontsize=italic_font_size, style='italic', color='gray',
                         transform=fig.transFigure)
                y_position -= italic_line_spacing

    def _create_single_heatmap(self,
                               row: pd.Series,
                               top_n: Optional[int],
                               figsize: Tuple[int, int],
                               title_prefix: str,
                               title_suffix: str,
                               min_r2: Optional[float],
                               filter_dataset: str,
                               cmap: str,
                               annotate: bool,
                               annotation_format: str,
                               variable_name_map: Optional[Dict[str, str]],
                               alternative_dataset_name: str,
                               fs: Dict,
                               gridspec_top: Optional[float] = None):
        """
        Create a single heatmap figure for one target dimension.

        Accepts a pre-computed font-size dict ``fs`` and an optional
        ``gridspec_top`` override so all sizing decisions flow from the
        public method rather than being re-derived here.

        :param fs: Font-size mapping produced in :meth:`plot_importance_heatmap`.
        :type fs: dict
        :param gridspec_top: Explicit top-margin override (0–1). When ``None``,
            derived automatically from the number of title lines.
        :type gridspec_top: float, optional
        """

        fig, ax = plt.subplots(figsize=figsize)

        self._plot_heatmap_on_axis(
            ax=ax,
            row=row,
            top_n=top_n,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format,
            variable_name_map=variable_name_map,
            alternative_dataset_name=alternative_dataset_name,
            fs=fs,
        )

        # Build hierarchical title
        target_display = self._apply_name_mapping([row['target']], variable_name_map)[0]
        main_title = target_display
        if top_n is not None:
            n_shown = len(list(row['importance_conventional_full'].keys())[:top_n])
            main_title += f' (Top {n_shown} Features)'

        metadata = {}
        if min_r2 is not None:
            metadata['Min R²'] = f'{min_r2:.2f} ({filter_dataset})'
        metadata['R² (Full/Match/Alt)'] = (
            f"{row['conv_full_r2']:.3f} / "
            f"{row['conv_matched_r2']:.3f} / "
            f"{row['alt_r2']:.3f}"
        )

        title_structure = self._build_hierarchical_title(
            main_title=main_title,
            title_prefix=title_prefix,
            title_suffix=title_suffix,
            metadata=metadata
        )

        self._add_hierarchical_suptitle(fig, title_structure,
                                        bold_font_size=fs['suptitle'],
                                        italic_font_size=fs['suptitle'] - 5)

        # Derive top margin
        n_lines = len(title_structure['lines'])
        auto_top = {1: 0.95, 2: 0.93, 3: 0.91}.get(n_lines, 0.89)
        effective_top = gridspec_top if gridspec_top is not None else auto_top

        plt.tight_layout(rect=[0, 0, 1, effective_top])
        plt.show()
        return fig

    def _plot_heatmap_on_axis(self,
                              ax,
                              row: pd.Series,
                              top_n: Optional[int],
                              cmap: str,
                              annotate: bool,
                              annotation_format: str,
                              variable_name_map: Optional[Dict[str, str]] = None,
                              alternative_dataset_name: str = 'Alternative',
                              fs: Optional[Dict] = None):
        """
        Plot heatmap on given axis for a single target.

        All font sizes are read from ``fs`` if provided, otherwise sensible
        hardcoded defaults are used so the method remains callable without
        the public-method wrapper (e.g. in tests).

        :param fs: Font-size mapping with keys ``'tick'``, ``'annotation'``,
            ``'axis_label'``, ``'subplot_title'``, ``'colorbar_label'``,
            ``'infobox'``. Produced by :meth:`plot_importance_heatmap`.
        :type fs: dict, optional
        """

        # Fall back to hardcoded defaults if called without fs
        if fs is None:
            fs = {
                'tick': 10,
                'annotation': 10,
                'axis_label': 12,
                'subplot_title': 14,
                'colorbar_label': 12,
                'infobox': 9,
            }

        # ── data preparation ────────────────────────────────────────────────
        conv_full_imp = row['importance_conventional_full']
        conv_matched_imp = row['importance_conventional_matched']
        alt_imp = row['importance_alternative']

        sorted_features = sorted(conv_full_imp.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            sorted_features = sorted_features[:top_n]

        features = [f[0] for f in sorted_features]
        features_display = self._apply_name_mapping(features, variable_name_map)

        dataset_names_short = ['Conv. (Full)', 'Conv. (Matched)', alternative_dataset_name]
        importance_matrix = np.zeros((len(features), 3))
        for i, feature in enumerate(features):
            importance_matrix[i, 0] = conv_full_imp[feature]
            importance_matrix[i, 1] = conv_matched_imp[feature]
            importance_matrix[i, 2] = alt_imp[feature]

        # ── heatmap ─────────────────────────────────────────────────────────
        im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap,
                       interpolation='nearest', vmin=0,
                       vmax=importance_matrix.max())

        r2_full = row['conv_full_r2']
        r2_matched = row['conv_matched_r2']
        r2_alt = row['alt_r2']

        x_labels = [
            f"{dataset_names_short[0]}\n$R^2_{{adj}}$ = {r2_full:.3f}",
            f"{dataset_names_short[1]}\n$R^2_{{adj}}$ = {r2_matched:.3f}",
            f"{dataset_names_short[2]}\n$R^2_{{adj}}$ = {r2_alt:.3f}",
        ]

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=fs['tick'])
        ax.set_yticklabels(features_display, fontsize=fs['tick'])

        # ── colorbar ────────────────────────────────────────────────────────
        cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
        cbar.set_label('Feature Importance', rotation=270, labelpad=20,
                       fontsize=fs['colorbar_label'])
        cbar.ax.tick_params(labelsize=fs['tick'])

        # ── cell annotations ────────────────────────────────────────────────
        if annotate:
            threshold = importance_matrix.max() * 0.5
            for i in range(len(features)):
                for j in range(3):
                    value = importance_matrix[i, j]
                    text_color = 'black' if value > threshold else 'white'
                    ax.text(j, i, f'{value:{annotation_format}}',
                            ha='center', va='center',
                            color=text_color,
                            fontsize=fs['annotation'],
                            fontweight='bold')

        # ── grid ────────────────────────────────────────────────────────────
        ax.set_xticks(np.arange(3) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(features)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        # ── axis decoration ─────────────────────────────────────────────────
        ax.set_xlabel('Dataset', fontsize=fs['axis_label'], fontweight='bold', labelpad=10)
        ax.set_ylabel('Features', fontsize=fs['axis_label'], fontweight='bold')

        target_display = self._apply_name_mapping([row['target']], variable_name_map)[0]
        title = target_display
        if top_n is not None:
            title += f' (Top {len(features)} Features)'
        ax.set_title(title, fontweight='bold', fontsize=fs['subplot_title'], pad=10)

        # ── info box ────────────────────────────────────────────────────────
        n_total = row['n_predictors']
        n_conv_full = row['n_conventional_full']
        n_conv_matched = row['n_conventional_matched']
        n_alt = row['n_alternative']

        # ax.text(0.02, 0.98,
        #         f'Total features: {n_total}\nSamples: {n_conv_full} / {n_conv_matched} / {n_alt}',
        #         transform=ax.transAxes,
        #         fontsize=fs['infobox'],
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    def plot_importance_change_heatmap(self,
                                       effect: str = 'fuel',
                                       separate_figures: bool = False,
                                       top_n: Optional[int] = None,
                                       figsize: Tuple[int, int] = (10, 8),
                                       save_path: Optional[str] = None,
                                       title_prefix: str = '',
                                       title_suffix: str = '',
                                       annotate: bool = True,
                                       annotation_format: str = '+.3f'):
        """
        Plot heatmap of importance CHANGES (delta values) across datasets.

        Concept:
        --------
        While plot_importance_heatmap shows absolute values, this shows the
        CHANGE in importance, revealing which features gain/lose importance
        when moving from conventional to alternative fuels.

        Uses diverging colormap (RdBu_r) where:
        - Red = feature gained importance
        - Blue = feature lost importance
        - White = no change

        Parameters
        ----------
        effect : str, default='fuel'
            Which effect to visualize: 'distribution', 'fuel', or 'total'
            - 'distribution': Full → Matched
            - 'fuel': Matched → Alternative
            - 'total': Full → Alternative
        separate_figures : bool, default=False
            Create separate figure per target or combined subplots
        top_n : int, optional
            Show only top N features by absolute change magnitude
        figsize : tuple
            Figure size for each heatmap
        save_path : str, optional
            Path to save figure
        title_prefix : str
            Text before main title
        title_suffix : str
            Text after main title
        annotate : bool
            Show change values in cells
        annotation_format : str
            Format for annotations (note: '+' prefix for signed values)

        Returns
        -------
        fig or list of fig
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return None

        valid_effects = ['distribution', 'fuel', 'total']
        if effect not in valid_effects:
            raise ValueError(f"effect must be one of {valid_effects}, got '{effect}'")

        results_df = self.comparison_results.copy()

        if separate_figures:
            figures = []
            for _, row in results_df.iterrows():
                fig = self._create_change_heatmap(
                    row=row,
                    effect=effect,
                    top_n=top_n,
                    figsize=figsize,
                    title_prefix=title_prefix,
                    title_suffix=title_suffix,
                    annotate=annotate,
                    annotation_format=annotation_format
                )
                figures.append(fig)

                if save_path and len(results_df) > 1:
                    base, ext = save_path.rsplit('.', 1)
                    target_path = f"{base}_{row['target'].replace(' ', '_')}.{ext}"
                    fig.savefig(target_path, dpi=300, bbox_inches='tight')

            return figures

        else:
            # Combined figure
            n_targets = len(results_df)
            n_cols = min(3, n_targets)
            n_rows = int(np.ceil(n_targets / n_cols))

            fig = plt.figure(figsize=(figsize[0] * n_cols, figsize[1] * n_rows))

            effect_labels = {
                'distribution': 'Distribution Effect',
                'fuel': 'Fuel Effect',
                'total': 'Total Effect'
            }

            main_title = f'Feature Importance Changes - {effect_labels[effect]}'
            if title_prefix:
                main_title = f'{title_prefix} - {main_title}'
            if title_suffix:
                main_title = f'{main_title} - {title_suffix}'

            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)

            for idx, (_, row) in enumerate(results_df.iterrows()):
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                self._plot_change_heatmap_on_axis(
                    ax=ax,
                    row=row,
                    effect=effect,
                    top_n=top_n,
                    annotate=annotate,
                    annotation_format=annotation_format
                )

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
            return fig

    def _create_change_heatmap(self,
                               row: pd.Series,
                               effect: str,
                               top_n: Optional[int],
                               figsize: Tuple[int, int],
                               title_prefix: str,
                               title_suffix: str,
                               annotate: bool,
                               annotation_format: str):
        """Create single change heatmap figure."""
        fig, ax = plt.subplots(figsize=figsize)

        self._plot_change_heatmap_on_axis(
            ax=ax,
            row=row,
            effect=effect,
            top_n=top_n,
            annotate=annotate,
            annotation_format=annotation_format
        )

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_change_heatmap_on_axis(self,
                                     ax,
                                     row: pd.Series,
                                     effect: str,
                                     top_n: Optional[int],
                                     annotate: bool,
                                     annotation_format: str):
        """
        Plot importance change heatmap on axis.

        Data Flow:
        ----------
        1. Extract importance_changes dict from row
        2. For each feature, get the requested effect type
        3. Build matrix with change values (can be negative!)
        4. Use diverging colormap centered at zero
        """

        importance_changes = row['importance_changes']

        # Extract change values based on effect type
        feature_changes = []
        for feature, changes in importance_changes.items():
            if effect == 'distribution':
                change_val = changes['distribution_effect']
            elif effect == 'fuel':
                change_val = changes['fuel_effect']
            else:  # total
                change_val = changes['total_effect']

            feature_changes.append((feature, change_val))

        # Sort by absolute change magnitude
        feature_changes.sort(key=lambda x: abs(x[1]), reverse=True)

        # Apply top_n filter
        if top_n is not None:
            feature_changes = feature_changes[:top_n]

        features = [f[0] for f in feature_changes]
        changes = np.array([f[1] for f in feature_changes]).reshape(-1, 1)

        # Use diverging colormap centered at zero
        vmax = max(abs(changes.min()), abs(changes.max()))

        im = ax.imshow(changes,
                       aspect='auto',
                       cmap='RdBu_r',  # Red=positive, Blue=negative
                       interpolation='nearest',
                       vmin=-vmax,
                       vmax=vmax)

        # Set ticks
        ax.set_xticks([0])
        ax.set_xticklabels(['Δ Importance'], fontsize=11, fontweight='bold')
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=9)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Change in Importance', rotation=270, labelpad=20, fontsize=10)

        # Annotations
        if annotate:
            for i in range(len(features)):
                value = changes[i, 0]

                # Color based on magnitude
                text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                text = f'{value:{annotation_format}}'

                ax.text(0, i, text,
                        ha='center', va='center',
                        color=text_color,
                        fontsize=9,
                        fontweight='bold')

        # Grid
        ax.set_yticks(np.arange(len(features)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2, axis='y')

        # Title
        effect_labels = {
            'distribution': 'Distribution Effect (Full → Matched)',
            'fuel': 'Fuel Effect (Matched → Alt)',
            'total': 'Total Effect (Full → Alt)'
        }

        title = f'{row["target"]} - {effect_labels[effect]}'
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

    def plot_feature_importance_correlation_heatmaps(self,
                                                     separate_figures: bool = True,
                                                     min_targets: int = 2,
                                                     figsize: Tuple[int, int] = (12, 10),
                                                     save_path: Optional[str] = None,
                                                     title_prefix: str = '',
                                                     title_suffix: str = '',
                                                     cmap: str = 'coolwarm',
                                                     annotate: bool = False,
                                                     annotation_format: str = '.2f',
                                                     cluster: bool = True):
        """
        Create pairwise feature importance correlation heatmaps for each dataset.

        Concept:
        --------
        This visualization shows how feature importances correlate ACROSS TARGETS.
        For example, if feature A is important for predicting Length and also
        important for predicting Beam, they will show positive correlation.

        This reveals:
        1. Feature synergies: Which features tend to be important together
        2. Feature specialization: Which features are important for specific targets only
        3. Structural changes: How alternative fuels change these relationships

        The analysis works by:
        1. Collecting feature importances across ALL targets
        2. Creating a feature × target matrix for each dataset
        3. Computing pairwise correlations between features
        4. Visualizing as heatmap (similar to correlation matrix)

        Data Flow:
        ----------
        comparison_results (DataFrame)
        → Extract importance dicts for each target
        → Build feature × target importance matrices (one per dataset)
        → Compute correlation between feature importance profiles
        → Three heatmaps showing feature-feature correlations

        Parameters
        ----------
        separate_figures : bool, default=True
            If True, create three separate figures (one per dataset).
            If False, create single figure with 3 subplots.
        min_targets : int, default=2
            Minimum number of targets a feature must appear in to be included.
            Prevents features that only predict one target from dominating.
        figsize : tuple
            Figure size for each heatmap
        save_path : str, optional
            Base path for saving. Will append dataset name if separate_figures=True.
        title_prefix : str
            Text before main title
        title_suffix : str
            Text after main title
        cmap : str, default='coolwarm'
            Colormap for correlation values (use diverging colormap)
        annotate : bool, default=False
            Show correlation values in cells (can be cluttered for many features)
        annotation_format : str
            Format string for annotations
        cluster : bool, default=True
            If True, reorder features using hierarchical clustering to group
            similar importance patterns together

        Returns
        -------
        fig or list of fig
            Single figure with 3 subplots or list of 3 separate figures
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return None

        # Build feature × target importance matrices for each dataset
        matrices = self._build_importance_matrices(min_targets=min_targets)

        if matrices is None:
            return None

        conv_full_matrix = matrices['conventional_full']
        conv_matched_matrix = matrices['conventional_matched']
        alt_matrix = matrices['alternative']
        features = matrices['features']

        if self.verbose:
            print(f"\n📊 Feature Importance Correlation Analysis")
            print(f"   Features included: {len(features)}")
            print(f"   Targets analyzed: {len(self.comparison_results)}")
            print(f"   Minimum target appearances: {min_targets}")

        # Compute correlation matrices
        conv_full_corr = np.corrcoef(conv_full_matrix)
        conv_matched_corr = np.corrcoef(conv_matched_matrix)
        alt_corr = np.corrcoef(alt_matrix)

        # Optional: Cluster features by similarity
        if cluster:
            from scipy.cluster.hierarchy import linkage, leaves_list

            # Use average of three correlation matrices for clustering
            avg_corr = (conv_full_corr + conv_matched_corr + alt_corr) / 3

            # Convert correlation to distance (ensuring symmetry)
            dist_matrix = 1 - np.abs(avg_corr)

            # Make sure it's perfectly symmetric (fix numerical precision issues)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2

            # Ensure diagonal is exactly zero
            np.fill_diagonal(dist_matrix, 0)

            # For hierarchical clustering, we need the condensed form
            # Extract upper triangle (excluding diagonal)
            n = dist_matrix.shape[0]
            condensed_dist = []
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_dist.append(dist_matrix[i, j])

            condensed_dist = np.array(condensed_dist)

            # Hierarchical clustering on condensed distance matrix
            linkage_matrix = linkage(condensed_dist, method='average')

            # Get optimal leaf ordering
            order = leaves_list(linkage_matrix)

            # Reorder features and matrices
            features = [features[i] for i in order]
            conv_full_corr = conv_full_corr[order, :][:, order]
            conv_matched_corr = conv_matched_corr[order, :][:, order]
            alt_corr = alt_corr[order, :][:, order]

        if separate_figures:
            figures = []
            dataset_info = [
                ('Conventional (Full)', conv_full_corr, 'full'),
                ('Conventional (Matched)', conv_matched_corr, 'matched'),
                ('Alternative', alt_corr, 'alternative')
            ]

            for dataset_name, corr_matrix, suffix in dataset_info:
                fig = self._create_correlation_heatmap_figure(
                    corr_matrix=corr_matrix,
                    features=features,
                    dataset_name=dataset_name,
                    figsize=figsize,
                    title_prefix=title_prefix,
                    title_suffix=title_suffix,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format
                )
                figures.append(fig)

                # Save if path provided
                if save_path:
                    base, ext = save_path.rsplit('.', 1)
                    dataset_path = f"{base}_{suffix}.{ext}"
                    fig.savefig(dataset_path, dpi=300, bbox_inches='tight')
                    if self.verbose:
                        print(f"✅ {dataset_name} correlation heatmap saved to {dataset_path}")

            return figures

        else:
            # Combined figure with 3 subplots
            fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))

            main_title = 'Feature Importance Correlation Across Targets'
            if title_prefix:
                main_title = f'{title_prefix} - {main_title}'
            if title_suffix:
                main_title = f'{main_title} - {title_suffix}'

            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

            dataset_info = [
                ('Conventional (Full)', conv_full_corr, 1),
                ('Conventional (Matched)', conv_matched_corr, 2),
                ('Alternative', alt_corr, 3)
            ]

            for dataset_name, corr_matrix, subplot_idx in dataset_info:
                ax = fig.add_subplot(1, 3, subplot_idx)
                self._plot_correlation_heatmap_on_axis(
                    ax=ax,
                    corr_matrix=corr_matrix,
                    features=features,
                    dataset_name=dataset_name,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format
                )

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f"✅ Combined correlation heatmap saved to {save_path}")

            plt.show()
            return fig

    def _build_importance_matrices(self, min_targets: int = 2) -> Optional[Dict]:
        """
        Build feature × target importance matrices for each dataset.

        Implementation Details:
        -----------------------
        1. Scan all results to find features appearing in >= min_targets
        2. Create matrices where:
           - Rows = features
           - Columns = targets
           - Values = importance scores
        3. Fill with importance values from each dataset's dict
        4. Return three aligned matrices (same features, same order)

        This ensures we're comparing the same features across datasets.

        Returns
        -------
        dict or None
            {
                'features': list of feature names,
                'targets': list of target names,
                'conventional_full': numpy array (n_features × n_targets),
                'conventional_matched': numpy array,
                'alternative': numpy array
            }
        """

        # Count how many targets each feature appears in
        feature_counts = {}
        all_targets = []

        for _, row in self.comparison_results.iterrows():
            target = row['target']
            all_targets.append(target)

            # Features are the same across all three importance dicts
            for feature in row['importance_conventional_full'].keys():
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Filter features by minimum target count
        valid_features = [f for f, count in feature_counts.items()
                          if count >= min_targets]

        if len(valid_features) == 0:
            print(f"⚠️ No features appear in >= {min_targets} targets")
            return None

        valid_features.sort()  # Alphabetical order
        n_features = len(valid_features)
        n_targets = len(all_targets)

        # Initialize matrices
        conv_full_matrix = np.zeros((n_features, n_targets))
        conv_matched_matrix = np.zeros((n_features, n_targets))
        alt_matrix = np.zeros((n_features, n_targets))

        # Fill matrices
        for target_idx, (_, row) in enumerate(self.comparison_results.iterrows()):
            conv_full_imp = row['importance_conventional_full']
            conv_matched_imp = row['importance_conventional_matched']
            alt_imp = row['importance_alternative']

            for feature_idx, feature in enumerate(valid_features):
                # Use .get() with default 0 in case feature doesn't exist for this target
                conv_full_matrix[feature_idx, target_idx] = conv_full_imp.get(feature, 0.0)
                conv_matched_matrix[feature_idx, target_idx] = conv_matched_imp.get(feature, 0.0)
                alt_matrix[feature_idx, target_idx] = alt_imp.get(feature, 0.0)

        return {
            'features': valid_features,
            'targets': all_targets,
            'conventional_full': conv_full_matrix,
            'conventional_matched': conv_matched_matrix,
            'alternative': alt_matrix
        }

    def _create_correlation_heatmap_figure(self,
                                           corr_matrix: np.ndarray,
                                           features: List[str],
                                           dataset_name: str,
                                           figsize: Tuple[int, int],
                                           title_prefix: str,
                                           title_suffix: str,
                                           cmap: str,
                                           annotate: bool,
                                           annotation_format: str):
        """Create single correlation heatmap figure."""

        fig, ax = plt.subplots(figsize=figsize)

        self._plot_correlation_heatmap_on_axis(
            ax=ax,
            corr_matrix=corr_matrix,
            features=features,
            dataset_name=dataset_name,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format
        )

        # Add title with prefix/suffix
        base_title = ax.get_title()
        if title_prefix:
            base_title = f'{title_prefix} - {base_title}'
        if title_suffix:
            base_title = f'{base_title} - {title_suffix}'
        ax.set_title(base_title, fontweight='bold', fontsize=14, pad=15)

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_correlation_heatmap_on_axis(self,
                                          ax,
                                          corr_matrix: np.ndarray,
                                          features: List[str],
                                          dataset_name: str,
                                          cmap: str,
                                          annotate: bool,
                                          annotation_format: str):
        """
        Plot correlation heatmap on given axis.

        Visualization Concept:
        ----------------------
        This is a symmetric matrix where:
        - Diagonal = 1.0 (feature correlates perfectly with itself)
        - Off-diagonal = correlation between features' importance profiles

        Interpretation:
        - High positive correlation (red): Features important for same targets
        - Low/zero correlation (white): Features important for different targets
        - Negative correlation (blue): When one is important, the other isn't

        Parameters
        ----------
        ax : matplotlib axis
        corr_matrix : numpy array
            Feature correlation matrix (n_features × n_features)
        features : list of str
            Feature names (length = n_features)
        dataset_name : str
            Name of dataset for title
        cmap : str
            Colormap (should be diverging)
        annotate : bool
            Show correlation values
        annotation_format : str
            Format for values
        """

        n_features = len(features)

        # Create heatmap
        im = ax.imshow(corr_matrix,
                       aspect='equal',
                       cmap=cmap,
                       interpolation='nearest',
                       vmin=-1,  # Correlation ranges from -1 to 1
                       vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(features, rotation=90, ha='right', fontsize=9)
        ax.set_yticklabels(features, fontsize=9)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=10)

        # Annotate cells if requested
        if annotate:
            for i in range(n_features):
                for j in range(n_features):
                    value = corr_matrix[i, j]

                    # Skip diagonal and redundant half to reduce clutter
                    if i >= j:
                        continue

                    # Color based on value
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    text = f'{value:{annotation_format}}'

                    ax.text(j, i, text,
                            ha='center', va='center',
                            color=text_color,
                            fontsize=7)

        # Grid
        ax.set_xticks(np.arange(n_features) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_features) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

        # Title
        ax.set_title(f'Feature Importance Correlations - {dataset_name}',
                     fontweight='bold', fontsize=12, pad=10)

        # Add interpretation guide
        guide_text = 'High correlation = features important for same targets'
        ax.text(0.5, -0.15, guide_text,
                transform=ax.transAxes,
                fontsize=8,
                ha='center',
                style='italic',
                color='gray')

    def analyze_importance_correlation_changes(self,
                                               min_targets: int = 2,
                                               threshold: float = 0.3) -> pd.DataFrame:
        """
        Analyze how feature importance correlations change from conventional to alternative.

        Concept:
        --------
        Quantifies structural changes in feature relationships. If two features
        are highly correlated in conventional dataset but uncorrelated in
        alternative dataset, this indicates the fuel type fundamentally changes
        which features work together for prediction.

        Data Flow:
        ----------
        1. Build importance matrices (as in plotting function)
        2. Compute correlation matrices for each dataset
        3. Calculate Δ correlation = alt_corr - conv_full_corr
        4. Identify feature pairs with large changes
        5. Return DataFrame of significant changes

        Parameters
        ----------
        min_targets : int
            Minimum targets a feature must appear in
        threshold : float
            Minimum absolute correlation change to report

        Returns
        -------
        pd.DataFrame
            Columns: feature_1, feature_2, corr_full, corr_matched, corr_alt,
                     delta_distribution, delta_fuel, delta_total, change_magnitude
        """

        matrices = self._build_importance_matrices(min_targets=min_targets)

        if matrices is None:
            return pd.DataFrame()

        features = matrices['features']

        # Compute correlations
        conv_full_corr = np.corrcoef(matrices['conventional_full'])
        conv_matched_corr = np.corrcoef(matrices['conventional_matched'])
        alt_corr = np.corrcoef(matrices['alternative'])

        # Analyze changes
        changes = []
        n_features = len(features)

        for i in range(n_features):
            for j in range(i + 1, n_features):  # Upper triangle only
                corr_full = conv_full_corr[i, j]
                corr_matched = conv_matched_corr[i, j]
                corr_alt = alt_corr[i, j]

                # Decompose change
                delta_distribution = corr_matched - corr_full
                delta_fuel = corr_alt - corr_matched
                delta_total = corr_alt - corr_full

                # Check if change exceeds threshold
                if abs(delta_total) >= threshold:
                    changes.append({
                        'feature_1': features[i],
                        'feature_2': features[j],
                        'corr_full': corr_full,
                        'corr_matched': corr_matched,
                        'corr_alt': corr_alt,
                        'delta_distribution': delta_distribution,
                        'delta_fuel': delta_fuel,
                        'delta_total': delta_total,
                        'change_magnitude': abs(delta_total),
                        'change_direction': 'strengthened' if delta_total > 0 else 'weakened'
                    })

        changes_df = pd.DataFrame(changes)

        if len(changes_df) > 0:
            changes_df = changes_df.sort_values('change_magnitude', ascending=False)

            if self.verbose:
                self._print_correlation_changes_summary(changes_df, threshold)

        return changes_df

    def _print_correlation_changes_summary(self, changes_df: pd.DataFrame, threshold: float):
        """Print formatted summary of correlation changes."""

        print("\n" + "=" * 120)
        print("FEATURE IMPORTANCE CORRELATION CHANGES")
        print(f"(Threshold: {threshold:.2f} correlation change)")
        print("=" * 120)

        print(f"\n{'Feature Pair':<45s} | {'Full':>7s} | {'Match':>7s} | {'Alt':>7s} | "
              f"{'Δ Total':>9s} | {'Direction':<12s}")
        print("-" * 120)

        for _, row in changes_df.head(20).iterrows():
            pair_name = f"{row['feature_1']} ↔ {row['feature_2']}"
            emoji = "📈" if row['change_direction'] == 'strengthened' else "📉"

            print(f"{emoji} {pair_name:<43s} | {row['corr_full']:>7.3f} | "
                  f"{row['corr_matched']:>7.3f} | {row['corr_alt']:>7.3f} | "
                  f"{row['delta_total']:>+9.3f} | {row['change_direction']:<12s}")

        if len(changes_df) > 20:
            print(f"\n  ... and {len(changes_df) - 20} more significant changes")

        print("=" * 120)

        # Summary stats
        strengthened = (changes_df['delta_total'] > 0).sum()
        weakened = (changes_df['delta_total'] < 0).sum()

        print(f"\nSignificant Changes: {len(changes_df)}")
        print(f"  Correlations strengthened: {strengthened}")
        print(f"  Correlations weakened:     {weakened}")
        print(f"  Average magnitude:         {changes_df['change_magnitude'].mean():.3f}")

    def plot_importance_across_targets(self,
                                       min_targets: int = 2,
                                       top_n_features: Optional[int] = None,
                                       min_r2: Optional[float] = None,
                                       filter_dataset: str = 'alternative',
                                       min_n_samples: Optional[int] = None,
                                       separate_figures: bool = True,
                                       figsize: Tuple[int, int] = (14, 10),
                                       save_path: Optional[str] = None,
                                       title_prefix: str = '',
                                       title_suffix: str = '',
                                       cmap: str = 'viridis',
                                       annotate: bool = True,  # UPDATED: Default to True
                                       annotation_format: str = '.2f',
                                       mask_diagonal: bool = True,
                                       variable_name_map: Optional[Dict[str, str]] = None,
                                       debug:bool=False):  # NEW parameter
        """
        Plot feature importance across different target variables for each dataset.

        Concept:
        --------
        This visualization shows how each feature's importance varies across different
        prediction tasks (targets). Unlike plot_importance_heatmap which compares
        datasets for a single target, this compares targets for a single dataset.

        Reveals:
        - Universal features: Important across many targets (dark horizontal bands)
        - Specialized features: Important for specific targets only (isolated dark cells)
        - Target-specific patterns: Which targets require unique feature sets (vertical patterns)
        - Consistency across fuel types: Do importance patterns change? (compare 3 heatmaps)

        Data Structure:
        - Y-axis: Features (predictors) - ordered to match target order
        - X-axis: Targets (predicted variables)
        - Color: Importance value (0 to max, consistent across all three heatmaps)
        - White diagonal: Where feature = target (self-prediction, excluded)

        The diagonal runs from top-left to bottom-right, with the color scale
        optimized to show variation (0 to maximum importance rounded up to nearest 0.05).

        Parameters
        ----------
        min_targets : int, default=2
            Minimum number of targets a feature must appear in to be included.
            Filters out features that only predict one or two targets.
        top_n_features : int, optional
            Show only top N most important features (by average importance).
            If None, shows all features that meet min_targets threshold.
            Note: Ordering is still maintained for diagonal alignment.
        separate_figures : bool, default=True
            If True, create three separate figures (one per dataset).
            If False, create single figure with 3 subplots.
        figsize : tuple, default=(14, 10)
            Figure size for each heatmap.
        save_path : str, optional
            Path to save figure(s). If separate_figures=True, appends dataset name.
        title_prefix : str
            Text to add before main title.
        title_suffix : str
            Text to add after main title.
        cmap : str, default='viridis'
            Colormap name. Use sequential colormaps (viridis, plasma, magma).
        annotate : bool, default=True
            If True, show importance values in cells.
        annotation_format : str, default='.2f'
            Format string for annotations.
        mask_diagonal : bool, default=True
            If True, show white for diagonal elements (feature predicting itself).
        variable_name_map : dict, optional
            Dictionary mapping internal variable names to publication-friendly names.
            Example: {'LOA (M)': 'LOA (m)', 'DWT': 'DWT (t)', 'DRAFT (M)': 'Draught (m)'}
            If None, uses original variable names.

        Returns
        -------
        fig or list of fig
            Single figure with 3 subplots or list of 3 separate figures.

        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return None

        if debug:
            print("\n🔍 DEBUG: plot_importance_across_targets")
            print(f"   Initial results count: {len(self.comparison_results)}")

            # Apply performance filtering FIRST
        filtered_results = self._filter_results_by_performance(
            results_df=self.comparison_results.copy(),
            min_r2=min_r2,
            dataset=filter_dataset,
            min_n_samples=min_n_samples,
            verbose=self.verbose or debug
        )

        if debug:
            print(f"   Post-filtering count: {len(filtered_results)}")

        if len(filtered_results) == 0:
            print("⚠️  No targets meet performance criteria")
            return None

        # Build matrices using FILTERED results
        # Temporarily replace self.comparison_results for matrix building
        original_results = self.comparison_results
        self.comparison_results = filtered_results

        try:
            matrices = self._build_importance_matrices(min_targets=min_targets)
        finally:
            # Restore original results
            self.comparison_results = original_results

        if matrices is None:
            return None

        features = matrices['features']
        targets = matrices['targets']
        conv_full_matrix = matrices['conventional_full']
        conv_matched_matrix = matrices['conventional_matched']
        alt_matrix = matrices['alternative']

        # Align features to match target order
        features_ordered, feature_indices = self._align_features_to_targets(
            features, targets
        )

        # Reorder matrices according to aligned features
        conv_full_matrix = conv_full_matrix[feature_indices, :]
        conv_matched_matrix = conv_matched_matrix[feature_indices, :]
        alt_matrix = alt_matrix[feature_indices, :]

        # Update features list
        features = features_ordered

        # Apply top_n_features filter if requested
        if top_n_features is not None and top_n_features < len(features):
            # Calculate average importance across all targets for each feature
            avg_importance_full = conv_full_matrix.mean(axis=1)
            avg_importance_matched = conv_matched_matrix.mean(axis=1)
            avg_importance_alt = alt_matrix.mean(axis=1)

            # Average across all three datasets
            avg_importance_overall = (avg_importance_full + avg_importance_matched + avg_importance_alt) / 3

            # Get indices of top N features
            top_indices = np.argsort(avg_importance_overall)[-top_n_features:][::-1]

            # Filter
            features = [features[i] for i in top_indices]
            conv_full_matrix = conv_full_matrix[top_indices, :]
            conv_matched_matrix = conv_matched_matrix[top_indices, :]
            alt_matrix = alt_matrix[top_indices, :]

            if self.verbose:
                print(f"\nFiltered to top {top_n_features} features by average importance")

        # Calculate optimal vmax across all three datasets
        vmax = self._calculate_optimal_vmax(
            conv_full_matrix, conv_matched_matrix, alt_matrix
        )

        # NEW: Apply variable name mapping if provided
        features_display = self._apply_name_mapping(features, variable_name_map)
        targets_display = self._apply_name_mapping(targets, variable_name_map)

        # Create diagonal mask if requested
        mask = None
        if mask_diagonal:
            mask = self._create_diagonal_mask(features, targets)  # Use original names for mask

        if self.verbose:
            print(f"\n📊 Feature Importance Across Targets")
            print(f"   Features shown: {len(features)}")
            print(f"   Targets analyzed: {len(targets)}")
            print(f"   Feature/target alignment: {'Diagonal' if len(features) == len(targets) else 'Partial'}")
            print(f"   Diagonal masking: {'Enabled' if mask_diagonal else 'Disabled'}")
            print(f"   Value annotations: {'Enabled' if annotate else 'Disabled'}")
            print(f"   Variable name mapping: {'Applied' if variable_name_map else 'None'}")
            print(f"   Color scale: 0.0 to {vmax:.2f} (optimized for visibility)")

        if separate_figures:
            figures = []
            dataset_info = [
                ('Conventional (Full)', conv_full_matrix, 'full'),
                ('Conventional (Matched)', conv_matched_matrix, 'matched'),
                ('Alternative', alt_matrix, 'alternative')
            ]

            for dataset_name, importance_matrix, suffix in dataset_info:
                fig = self._create_target_comparison_figure(
                    importance_matrix=importance_matrix,
                    features=features_display,  # UPDATED: Use display names
                    targets=targets_display,  # UPDATED: Use display names
                    dataset_name=dataset_name,
                    mask=mask,
                    vmax=vmax,
                    figsize=figsize,
                    title_prefix=title_prefix,
                    title_suffix=title_suffix,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format
                )
                figures.append(fig)

                # Save if path provided
                if save_path:
                    base, ext = save_path.rsplit('.', 1)
                    dataset_path = f"{base}_{suffix}.{ext}"
                    fig.savefig(dataset_path, dpi=300, bbox_inches='tight')
                    if self.verbose:
                        print(f"✅ {dataset_name} heatmap saved to {dataset_path}")

            return figures

        else:
            # Combined figure with 3 subplots
            fig = plt.figure(figsize=(figsize[0] * 1.2, figsize[1]))

            # Build title with newlines for stacking
            title_parts = []

            if title_prefix:
                title_parts.append(title_prefix)

            main_title = 'Feature Importance Across Target Variables'
            if top_n_features is not None:
                main_title += f' (Top {len(features)} Features)'
            title_parts.append(main_title)

            if min_r2 is not None:
                title_parts.append(f'Min R² ({filter_dataset}): {min_r2:.2f}')

            if title_suffix:
                title_parts.append(title_suffix)

            # Join with newlines
            stacked_title = '\n'.join(title_parts)

            fig.suptitle(stacked_title, fontsize=16, fontweight='bold', y=0.98)

            dataset_info = [
                ('Conventional (Full)', conv_full_matrix, 1),
                ('Conventional (Matched)', conv_matched_matrix, 2),
                ('Alternative', alt_matrix, 3)
            ]

            for dataset_name, importance_matrix, subplot_idx in dataset_info:
                ax = fig.add_subplot(1, 3, subplot_idx)
                self._plot_target_comparison_on_axis(
                    ax=ax,
                    importance_matrix=importance_matrix,
                    features=features_display,  # UPDATED: Use display names
                    targets=targets_display,  # UPDATED: Use display names
                    dataset_name=dataset_name,
                    mask=mask,
                    vmax=vmax,
                    cmap=cmap,
                    annotate=annotate,
                    annotation_format=annotation_format
                )

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f"✅ Combined heatmap saved to {save_path}")

            plt.show()
            return fig

    def _apply_name_mapping(self,
                            names: List[str],
                            name_map: Optional[Dict[str, str]]) -> List[str]:
        """
        Apply variable name mapping for publication-friendly display.

        Concept:
        --------
        Converts internal variable names (e.g., 'LOA (M)', 'DWT', 'DRAFT (M)')
        to publication-friendly versions (e.g., 'LOA (m)', 'DWT (t)', 'Draught (m)').

        If a variable is not in the mapping dictionary, it is left unchanged.

        Parameters
        ----------
        names : list of str
            Original variable names
        name_map : dict or None
            Mapping from original to display names.
            If None, returns names unchanged.

        Returns
        -------
        display_names : list of str
            Mapped names (or original if no mapping provided)

        Examples
        --------
        >>> names = ['DWT', 'LOA (M)', 'BEAM (M)']
        >>> name_map = {'DWT': 'DWT (t)', 'LOA (M)': 'LOA (m)', 'BEAM (M)': 'Beam (m)'}
        >>> _apply_name_mapping(names, name_map)
        ['DWT (t)', 'LOA (m)', 'Beam (m)']
        """

        if name_map is None:
            return names

        # Apply mapping, keeping original if not in map
        display_names = [name_map.get(name, name) for name in names]

        return display_names

    def _calculate_optimal_vmax(self,
                                conv_full_matrix: np.ndarray,
                                conv_matched_matrix: np.ndarray,
                                alt_matrix: np.ndarray) -> float:
        """
        Calculate optimal vmax for color scale across all datasets.

        Concept:
        --------
        To maximize visibility of differences while maintaining comparability,
        we use a consistent color scale across all three heatmaps. The scale is:
        - vmin = 0 (always)
        - vmax = maximum importance across ALL datasets, rounded up to nearest 0.05

        This ensures:
        1. All three heatmaps use the same scale (valid comparison)
        2. The scale is optimized for the actual data range (better contrast)
        3. The scale is "clean" (rounded to 0.05) for readability

        Example:
        --------
        If max importance across all datasets = 0.673, vmax = 0.70
        If max importance = 0.821, vmax = 0.85
        If max importance = 0.449, vmax = 0.45

        Parameters
        ----------
        conv_full_matrix : numpy array
            Conventional (full) importance matrix
        conv_matched_matrix : numpy array
            Conventional (matched) importance matrix
        alt_matrix : numpy array
            Alternative importance matrix

        Returns
        -------
        vmax : float
            Maximum value for color scale, rounded up to nearest 0.05
        """

        # Find maximum importance value across all three datasets
        max_full = np.nanmax(conv_full_matrix)  # Use nanmax to ignore NaN (masked values)
        max_matched = np.nanmax(conv_matched_matrix)
        max_alt = np.nanmax(alt_matrix)

        overall_max = max(max_full, max_matched, max_alt)

        # Round up to nearest 0.05
        vmax = np.ceil(overall_max / 0.05) * 0.05

        # Ensure vmax is at least 0.05 (in case all importances are near zero)
        vmax = max(vmax, 0.05)

        return vmax

    def _align_features_to_targets(self,
                                   features: List[str],
                                   targets: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Reorder features to match target order for diagonal alignment.

        Concept:
        --------
        To create a clear diagonal of white cells (feature predicting itself),
        we need features and targets in the same order. This function:
        1. Identifies which features also appear as targets
        2. Orders those features to match target order (for diagonal)
        3. Appends remaining features (not in targets) at the end

        Example:
        --------
        targets = ['DWT', 'LOA (M)', 'Beam (M)', 'Speed']
        features = ['GT', 'DWT', 'Beam (M)', 'LOA (M)', 'Power']

        Result:
        features_ordered = ['DWT', 'LOA (M)', 'Beam (M)', 'GT', 'Power']
                            ^---- matches target order ---^  ^- extras -^

        This creates diagonal at positions [0,0], [1,1], [2,2] for DWT, LOA, Beam.

        Parameters
        ----------
        features : list of str
            Original feature list (from _build_importance_matrices)
        targets : list of str
            Target list (from _build_importance_matrices)

        Returns
        -------
        features_ordered : list of str
            Features reordered to align with targets
        indices : numpy array
            Indices to reorder the original feature matrix
        """

        # Identify features that also appear as targets (for diagonal)
        features_in_targets = [f for f in features if f in targets]
        features_not_in_targets = [f for f in features if f not in targets]

        # Order the diagonal features to match target order
        features_diagonal_ordered = []
        for target in targets:
            if target in features_in_targets:
                features_diagonal_ordered.append(target)

        # Combine: diagonal-aligned features first, then extras
        features_ordered = features_diagonal_ordered + features_not_in_targets

        # Get indices for reordering the matrix
        indices = [features.index(f) for f in features_ordered]
        indices = np.array(indices)

        return features_ordered, indices

    def _create_diagonal_mask(self, features: List[str], targets: List[str]) -> np.ndarray:
        """
        Create a boolean mask for diagonal elements (feature predicting itself).

        Concept:
        --------
        When a feature appears as both predictor and target (e.g., DWT predicting DWT),
        the importance value is not meaningful for cross-target comparison. We mask
        these cells to show them as white, preventing confusion.

        With aligned ordering from _align_features_to_targets, these masked cells
        will form a diagonal line from top-left to bottom-right.

        Parameters
        ----------
        features : list of str
            Feature names (y-axis), should be ordered by _align_features_to_targets
        targets : list of str
            Target names (x-axis)

        Returns
        -------
        mask : numpy array (n_features × n_targets)
            Boolean mask where True indicates feature == target (should be masked)
        """
        n_features = len(features)
        n_targets = len(targets)

        mask = np.zeros((n_features, n_targets), dtype=bool)

        for i, feature in enumerate(features):
            for j, target in enumerate(targets):
                if feature == target:
                    mask[i, j] = True

        return mask

    def _create_target_comparison_figure(self,
                                         importance_matrix: np.ndarray,
                                         features: List[str],
                                         targets: List[str],
                                         dataset_name: str,
                                         mask: Optional[np.ndarray],
                                         vmax: float,
                                         figsize: Tuple[int, int],
                                         title_prefix: str,
                                         title_suffix: str,
                                         cmap: str,
                                         annotate: bool,
                                         annotation_format: str):
        """
        Create a single heatmap figure showing features vs. targets.

        Data Flow:
        ----------
        1. Receive importance_matrix (n_features × n_targets) - already reordered
        2. Receive vmax (shared across all three datasets)
        3. Receive display names (already mapped to publication-friendly format)
        4. Pass to _plot_target_comparison_on_axis for visualization
        5. Return completed figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        self._plot_target_comparison_on_axis(
            ax=ax,
            importance_matrix=importance_matrix,
            features=features,
            targets=targets,
            dataset_name=dataset_name,
            mask=mask,
            vmax=vmax,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format
        )

        # Add title with prefix/suffix
        base_title = ax.get_title()
        if title_prefix:
            base_title = f'{title_prefix} - {base_title}'
        if title_suffix:
            base_title = f'{base_title} - {title_suffix}'
        ax.set_title(base_title, fontweight='bold', fontsize=14, pad=15)

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_target_comparison_on_axis(self,
                                        ax,
                                        importance_matrix: np.ndarray,
                                        features: List[str],
                                        targets: List[str],
                                        dataset_name: str,
                                        mask: Optional[np.ndarray],
                                        vmax: float,
                                        cmap: str,
                                        annotate: bool,
                                        annotation_format: str):
        """
        Plot feature vs. target importance heatmap on given axis.

        Concept & Implementation:
        -------------------------
        This creates a heatmap where:
        - Rows (y-axis) = Features (predictors) - ordered to align with targets
        - Columns (x-axis) = Targets (predicted variables)
        - Cell color = Importance value (0 to vmax scale)
        - Cell annotations = Actual importance values (if annotate=True)
        - White diagonal = Masked cells where feature predicts itself

        The diagonal runs from top-left to bottom-right because features are
        ordered to match targets (via _align_features_to_targets).

        The color scale (0 to vmax) is consistent across all three datasets,
        where vmax is the maximum importance rounded up to nearest 0.05.
        This maximizes contrast while maintaining comparability.

        Interpretation:
        - Dark horizontal band = Feature important for many targets (universal)
        - Isolated dark cell = Feature specialized for specific target
        - Dark vertical band = Target requires many features (complex)
        - Light vertical band = Target predicted by few features (simple)

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot on
        importance_matrix : numpy array (n_features × n_targets)
            Feature importance values - rows already ordered by _align_features_to_targets
        features : list of str
            Feature names (y-axis labels) - publication-friendly format
        targets : list of str
            Target names (x-axis labels) - publication-friendly format
        dataset_name : str
            Name for title
        mask : numpy array, optional
            Boolean mask for diagonal elements
        vmax : float
            Maximum value for color scale (shared across all datasets)
        cmap : str
            Colormap name
        annotate : bool
            Show values in cells
        annotation_format : str
            Format for values
        """

        n_features = len(features)
        n_targets = len(targets)

        # Create masked array if mask provided
        if mask is not None:
            # Set masked values to NaN for white color
            plot_data = importance_matrix.copy()
            plot_data[mask] = np.nan
        else:
            plot_data = importance_matrix

        # Create heatmap with OPTIMIZED scale (0 to vmax)
        im = ax.imshow(plot_data,
                       aspect='auto',
                       cmap=cmap,
                       interpolation='nearest',
                       vmin=0.0,
                       vmax=vmax)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_targets))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(features, fontsize=9)

        # Add colorbar with optimized ticks
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Feature Importance', rotation=270, labelpad=20, fontsize=10)

        # Set colorbar ticks based on vmax
        # Create 5 evenly spaced ticks from 0 to vmax
        tick_values = np.linspace(0, vmax, 5)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.2f}' for val in tick_values])

        # UPDATED: Annotate cells with values if requested
        if annotate:
            # Determine threshold for text color (50% of vmax)
            threshold = vmax * 0.5

            for i in range(n_features):
                for j in range(n_targets):
                    # Skip masked cells
                    if mask is not None and mask[i, j]:
                        continue

                    value = importance_matrix[i, j]

                    # Choose text color based on background intensity
                    text_color = 'black' if value > threshold else 'white'

                    # Format the value
                    text = f'{value:{annotation_format}}'

                    ax.text(j, i, text,
                            ha='center', va='center',
                            color=text_color,
                            fontsize=8,
                            fontweight='bold')

        # Add grid for readability
        ax.set_xticks(np.arange(n_targets) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_features) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        # Labels and title
        ax.set_xlabel('Target Variables', fontsize=11, fontweight='bold')
        ax.set_ylabel('Features (Predictors)', fontsize=11, fontweight='bold')
        ax.set_title(f'Feature Importance - {dataset_name}',
                     fontweight='bold', fontsize=12, pad=10)

        # Guide text shows the actual scale
        guide_text = f'Diagonal (white) = self-prediction | Scale: 0.0-{vmax:.2f} (consistent across datasets)'
        ax.text(0.5, -0.12, guide_text,
                transform=ax.transAxes,
                fontsize=8,
                ha='center',
                style='italic',
                color='gray')

    def _optimal_grid(self, n_plots: int, max_cols: int) -> tuple[int, int]:
        """
        Find the column count c (1 ≤ c ≤ max_cols) that minimises empty
        grid cells, breaking ties by minimising the number of rows.

        Returns (n_rows, n_cols).
        """
        # best = None  # (empty_cells, n_rows, n_cols)
        # for c in range(1, max_cols + 1):
        #     rows = math.ceil(n_plots / c)
        #     empty = rows * c - n_plots
        #     candidate = (empty, rows, c)
        #     if best is None or candidate < best:
        #         best = candidate
        # _, n_rows, n_cols = best
        # return n_rows, n_cols

        min_rows = math.ceil(n_plots / max_cols)  # fewest possible rows
        best = None  # (empty_cells, neg_n_cols)
        for c in range(1, max_cols + 1):
            rows = math.ceil(n_plots / c)
            if rows > min_rows:  # skip layouts that need more rows than necessary
                continue
            empty = rows * c - n_plots
            candidate = (empty, -c)
            if best is None or candidate < best:
                best = candidate
        _, neg_n_cols = best
        n_cols = -neg_n_cols
        n_rows = math.ceil(n_plots / n_cols)
        return n_rows, n_cols