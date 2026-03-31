"""
Coefficient Comparison Analyzer (UPDATED with ModelDescriptor Integration)
==========================================================================

✅ NOW USES: ModelDescriptor for structure-preserving refitting
✅ ENSURES: Comparable coefficients across datasets
✅ SUPPORTS: MFP, Lasso, and all parametric models

Key Improvement:
- Discovers structure on Full Conventional dataset
- Refits with FIXED structure on Matched and Alternative datasets
- Enables meaningful coefficient comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
from knowledge_layer.parametric_regression.parametric_model_description_class import ModelDescriptor
from knowledge_layer.multi_dataset_analysers.parametric_analysis_foundation import ParametricAnalysisFoundation
from knowledge_layer.multi_dataset_analysers.coefficient_comparison.coefficient_comparison_display import CoefficientComparisonDisplay
from knowledge_layer.multi_dataset_analysers.coefficient_comparison.coefficient_plotter import CoefficientChangePlotter
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from config import LoggerMixin

warnings.filterwarnings('ignore')


class CoefficientComparisonAnalyzer(LoggerMixin):
    """
    Analyze coefficient changes using matched datasets with structure preservation.

    **Key Feature: Structure-Preserving Refitting**
    Uses ModelDescriptor to ensure same model structure across all three datasets:
    1. Discover structure on Full Conventional
    2. Refit with FIXED structure on Matched Conventional
    3. Refit with FIXED structure on Alternative

    This makes coefficient comparison meaningful!

    Three-Way Analysis:
    ------------------
    Full Conventional → Matched Conventional → Alternative Fuel
         (all ships)        (size-matched)        (alt fuel)

    Decomposition:
    -------------
    - Distribution Effect: Change from Full → Matched (size sampling effect)
    - Fuel Effect: Change from Matched → Alternative (fuel type effect)
    - Total Effect: Change from Full → Alternative (combined)
    """

    def __init__(self,
                 parametric_selector,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 normalize: bool = True,
                 normalization_method: str = 'zscore',
                 verbose: bool = True):
        """
        Initialize coefficient comparison analyzer.

        Parameters
        ----------
        parametric_selector : ParametricModelSelector
            Fitted selector with model_recommendations AND model_descriptors
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_data : pd.DataFrame
            Alternative fuel ships dataset
        normalize : bool, default=True
            Whether to normalize coefficients for comparison.
            **Highly recommended** - allows comparing coefficients across predictors.
        normalization_method : str, default='zscore'
            Normalization method: 'zscore' or 'minmax'
        verbose : bool, default=True
            Whether to print progress and results
        """

        # ✅ Initialize foundation for data management
        self.foundation = ParametricAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            independent_variables=parametric_selector.variable_dimensions,
            dependent_variables=parametric_selector.target_dimensions,
            model_recommendations=parametric_selector.model_recommendations,
            detailed_results=parametric_selector.detailed_results,  # ✅ Pass for descriptor lookup
            verbose=verbose
        )

        # Store selector reference
        self.selector = parametric_selector

        # ✅ NEW: Validate that selector has descriptors
        if not hasattr(parametric_selector, 'model_descriptors') or not parametric_selector.model_descriptors:
            if verbose:
                print("\n⚠️  WARNING: Selector has no model_descriptors!")
                print("    Descriptors are required for structure-preserving refitting.")
                print("    Creating descriptors now...")
            parametric_selector.create_missing_descriptors()

        # Coefficient-specific configuration
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.verbose = verbose

        # Storage for results
        self.normalization_params = {}  # Maps (predictors, target) → params
        self.comparison_results = None

        # ✅ NEW: Store descriptors for each dataset
        self.model_descriptors = {
            'conventional_full': {},      # (predictors, target) → descriptor
            'conventional_matched': {},   # (predictors, target) → descriptor
            'alternative': {}             # (predictors, target) → descriptor
        }

        # ✅ Initialize display class
        self.display = CoefficientComparisonDisplay(self)

        if verbose:
            self.display.print_initialization_summary()

    # ========================================================================
    # DELEGATE PROPERTIES (for backward compatibility)
    # ========================================================================

    @property
    def conventional_data(self) -> pd.DataFrame:
        """Full conventional dataset (for backward compatibility)."""
        return self.foundation.conventional_full

    @property
    def alternative_data(self) -> pd.DataFrame:
        """Alternative dataset (for backward compatibility)."""
        return self.foundation.alternative

    @property
    def matched_datasets(self) -> Dict:
        """Matched datasets (for backward compatibility)."""
        return self.foundation.matched_data_sets

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
        Removes targets where models perform poorly, as coefficient comparison
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

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_targets(self,
                            min_alternative_samples: int = 10,
                            min_matched_samples: int = 10,
                            max_predictors: Optional[int] = None,  # ✅ NEW
                            require_alternative_data: bool = True,  # ✅ NEW
                            display_results: bool = True) -> pd.DataFrame:
        """
        Analyze all targets with three-way comparison and structure preservation.

        ✅ Uses ModelDescriptor for consistent coefficient extraction across datasets.

        Parameters
        ----------
        min_alternative_samples : int, default=10
            Minimum alternative fuel ships required for analysis
        min_matched_samples : int, default=10
            Minimum matched conventional ships required
        max_predictors : int, optional
            Maximum number of predictors to consider.
            If None, uses all available combinations from selector.
            Useful for focusing on simpler, more stable models.
        require_alternative_data : bool, default=True
            If True, only selects combinations where alternative dataset has
            sufficient non-missing data. Recommended to avoid analyzing
            combinations that can't be properly evaluated.
        display_results : bool, default=True
            Whether to print results

        Returns
        -------
        pd.DataFrame
            Comprehensive results with decomposition

        Examples
        --------
        >>> # Analyze all combinations (default)
        >>> results = analyzer.analyze_all_targets()

        >>> # Focus on single-predictor models only
        >>> results = analyzer.analyze_all_targets(max_predictors=1)

        >>> # Allow 2-predictor models, ensure alternative data exists
        >>> results = analyzer.analyze_all_targets(
        ...     max_predictors=2,
        ...     require_alternative_data=True,
        ...     min_alternative_samples=15
        ... )

        >>> # Skip alternative data check (risky - may analyze unavailable combos)
        >>> results = analyzer.analyze_all_targets(
        ...     max_predictors=2,
        ...     require_alternative_data=False
        ... )
        """

        if display_results and self.verbose:
            self.display.print_analysis_header()

            # ✅ NEW: Print filtering configuration
            if max_predictors is not None:
                print(f"\n  📊 Filtering: max_predictors={max_predictors}")
            if require_alternative_data:
                print(f"  📊 Filtering: require_alternative_data=True (min_samples={min_alternative_samples})")

        all_results = []
        skipped_targets = []

        for target in self.foundation.target_dimensions:
            # ✅ UPDATED: Pass filtering parameters
            best_combo, best_model_type = self._get_best_combination_for_target(
                target=target,
                max_predictors=max_predictors,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

            if best_combo is None:
                if self.verbose:
                    print(f"\n  ⚠️  No valid combinations found for {target}")
                skipped_targets.append(target)
                continue

            if display_results and self.verbose:
                self.display.print_target_header(target, best_combo, best_model_type)

            # Analyze this target with structure preservation
            result = self._analyze_single_target_threeway(
                target=target,
                min_alternative_samples=min_alternative_samples,
                min_matched_samples=min_matched_samples,
                display_results=display_results,
                max_predictors=max_predictors,  # ✅ ADD
                require_alternative_data=require_alternative_data  # ✅ ADD
            )

            if result is not None:
                all_results.append(result)
            else:
                skipped_targets.append(target)

        # Create results DataFrame
        if not all_results:
            print("\n⚠️  No valid results generated")
            if skipped_targets:
                print(f"  Skipped targets: {', '.join(skipped_targets)}")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        self.comparison_results = results_df

        if display_results and self.verbose:
            self.display.print_summary_table(results_df)

            # ✅ NEW: Report skipped targets
            if skipped_targets:
                print(f"\n⚠️  Skipped {len(skipped_targets)} targets due to filtering:")
                print(f"  {', '.join(skipped_targets)}")

        return results_df

    def _get_best_combination_for_target(self,
                                         target: str,
                                         max_predictors: Optional[int] = None,
                                         require_alternative_data: bool = True,
                                         min_alternative_samples: int = 10) -> Tuple[
        Optional[Tuple[str, ...]], Optional[str]]:
        """
        Get best predictor combination for target with filtering.

        ✅ NEW: Filters by max_predictors and alternative data availability

        Parameters
        ----------
        target : str
            Target variable
        max_predictors : int, optional
            Maximum number of predictors to consider
        require_alternative_data : bool, default=True
            Whether to check if alternative dataset has sufficient data
        min_alternative_samples : int, default=10
            Minimum valid samples required in alternative dataset

        Returns
        -------
        tuple
            (predictors_tuple, model_type) or (None, None)
        """
        try:
            # ✅ UPDATED: Use foundation's method which now supports these parameters
            result = self.foundation.get_best_combination_for_target(
                target=target,
                max_predictors=max_predictors,
                filter_by_vif=True,
                vif_threshold=10.0,
                require_alternative_data=require_alternative_data  # ✅ NEW
            )

            if result is None:
                return None, None

            predictors, model_type = result

            # ✅ ADDITIONAL CHECK: Verify alternative dataset has data for this combination
            if require_alternative_data:
                # Double-check that we can actually prepare the alternative dataset
                alt_check = self.foundation.prepare_dataset(
                    self.foundation.alternative,
                    list(predictors),
                    target,
                    min_samples=min_alternative_samples
                )

                if alt_check is None:
                    if self.verbose:
                        print(f"  ⚠️  {target}: Alternative dataset insufficient for {predictors}")
                    return None, None

            return predictors, model_type

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Could not get best combination for {target}: {e}")
            return None, None

    # ========================================================================
    # SINGLE TARGET ANALYSIS WITH STRUCTURE PRESERVATION
    # ========================================================================

    def _analyze_single_target_threeway(
            self,
            target: str,
            min_alternative_samples: int,
            min_matched_samples: int,
            display_results: bool,
            max_predictors=None,
            require_alternative_data: bool = True,
    ) -> Optional[Dict]:
        """
        Analyze a single target with structure-preserving three-way decomposition.

        Uses :class:`~ModelDescriptor` to guarantee that the same model
        structure — MFP transformations, Lasso alpha, or nonlinear functional
        form — is applied to all three datasets (Full Conventional, Matched
        Conventional, Alternative Fuel).  Any difference in re-estimated
        coefficients therefore reflects a genuine population difference rather
        than an artefact of different model selection.

        Data flow
        ---------
        1. Retrieve best predictor combination and base descriptor from the
           selector.
        2. Normalise all three datasets using parameters fit on the full
           conventional data only.
        3. Clean data for nonlinear models (remove NaN / Inf / non-positive
           rows).
        4. Deep-copy the base descriptor and refit each copy on its respective
           dataset, preserving the discovered structure.
        5. Extract comparable coefficients and compute performance metrics.
        6. Perform cross-evaluations (full model on matched data, etc.).
        7. Compute the distribution effect (Full → Matched) and fuel effect
           (Matched → Alternative) decomposition.

        :param target: Target variable name.
        :type target: str
        :param min_alternative_samples: Minimum number of alternative-fuel
            ships required to proceed.
        :type min_alternative_samples: int
        :param min_matched_samples: Minimum matched conventional ships
            required.
        :type min_matched_samples: int
        :param display_results: Whether to print per-target results.
        :type display_results: bool
        :param max_predictors: Maximum number of predictors to consider.
        :type max_predictors: int, optional
        :param require_alternative_data: Require sufficient alternative data
            for the selected predictor combination.
        :type require_alternative_data: bool

        :returns: Result dictionary, or ``None`` if the target cannot be
            analysed.
        :rtype: dict or None
        """

        # ── Step 0: best combination ──────────────────────────────────────────
        best_combo, best_model_type = self._get_best_combination_for_target(
            target=target,
            max_predictors=max_predictors,
            require_alternative_data=require_alternative_data,
            min_alternative_samples=min_alternative_samples,
        )

        if best_combo is None:
            return None

        key = (tuple(best_combo), target)

        # ── Step 1: base descriptor ───────────────────────────────────────────
        base_descriptor = self._get_base_descriptor(key, best_model_type)

        if base_descriptor is None:
            if self.verbose:
                print(f"  ⚠️  No descriptor available for {best_combo} → {target}")
            return None

        # ── Step 2: normalised datasets ───────────────────────────────────────
        conv_full_prep = self.foundation.prepare_dataset_normalized(
            dataset=self.foundation.conventional_full,
            predictors=list(best_combo),
            target=target,
            fit_params=True,
            normalization_method=self.normalization_method if self.normalize else 'zscore',
        )

        if conv_full_prep is None:
            if self.verbose:
                print(f"  ⚠️  Insufficient full conventional data for {target}")
            return None

        if self.normalize:
            X_conv_full_norm, y_conv_full_norm, norm_params = conv_full_prep
        else:
            X_conv_full_norm, y_conv_full_norm = conv_full_prep
            norm_params = None

        matched_dataset = self.foundation.get_matched_dataset(best_combo)

        if matched_dataset is None or len(matched_dataset) < min_matched_samples:
            if self.verbose:
                print(f"  ⚠️  Insufficient matched data")
            return None

        conv_matched_prep = self.foundation.prepare_dataset_normalized(
            dataset=matched_dataset,
            predictors=list(best_combo),
            target=target,
            normalization_params=norm_params,
            normalization_method=self.normalization_method if self.normalize else 'zscore',
        )

        if conv_matched_prep is None:
            return None

        X_conv_matched_norm, y_conv_matched_norm = conv_matched_prep

        if len(self.foundation.alternative) < min_alternative_samples:
            if self.verbose:
                print(f"  ⚠️  Insufficient alternative data")
            return None

        alt_prep = self.foundation.prepare_dataset_normalized(
            dataset=self.foundation.alternative,
            predictors=list(best_combo),
            target=target,
            normalization_params=norm_params,
            normalization_method=self.normalization_method if self.normalize else 'zscore',
        )

        if alt_prep is None:
            return None

        X_alt_norm, y_alt_norm = alt_prep
        self.normalization_params[key] = norm_params

        # ── Step 3: clean data for nonlinear models ───────────────────────────
        X_conv_full_clean, y_conv_full_clean, mask_conv_full = (
            self._clean_data_for_nonlinear(
                X_conv_full_norm, y_conv_full_norm,
                best_model_type, "Full conventional",
            )
        )
        if X_conv_full_clean is None:
            return None

        X_conv_matched_clean, y_conv_matched_clean, mask_conv_matched = (
            self._clean_data_for_nonlinear(
                X_conv_matched_norm, y_conv_matched_norm,
                best_model_type, "Matched conventional",
            )
        )
        if X_conv_matched_clean is None:
            return None

        X_alt_clean, y_alt_clean, mask_alt = (
            self._clean_data_for_nonlinear(
                X_alt_norm, y_alt_norm,
                best_model_type, "Alternative",
            )
        )
        if X_alt_clean is None:
            return None

        # ── Step 4: structure-preserving refit ───────────────────────────────
        if self.verbose:
            print(f"\n  🔧 Refitting with preserved structure...")
            print(f"     {self._describe_mfp_transforms(base_descriptor)}"
                  if best_model_type == 'mfp'
                  else f"     model type: {best_model_type}")

        try:
            descriptor_full = self._copy_descriptor(base_descriptor)
            descriptor_matched = self._copy_descriptor(base_descriptor)
            descriptor_alt = self._copy_descriptor(base_descriptor)

            descriptor_full.refit(X_conv_full_clean, y_conv_full_clean, force_structure=True)
            descriptor_matched.refit(X_conv_matched_clean, y_conv_matched_clean, force_structure=True)
            descriptor_alt.refit(X_alt_clean, y_alt_clean, force_structure=True)

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Structure-preserving refit failed: {e}")
            return None

        self.model_descriptors['conventional_full'][key] = descriptor_full
        self.model_descriptors['conventional_matched'][key] = descriptor_matched
        self.model_descriptors['alternative'][key] = descriptor_alt

        # ── Step 5: predictions for metrics ──────────────────────────────────
        try:
            model_full, prep_full = ModelFactory.create_and_fit(
                best_model_type, X_conv_full_clean, y_conv_full_clean,
            )
            y_pred_conv_full_norm = ModelFactory.predict(
                model_full, prep_full, X_conv_full_clean, best_model_type,
            )

            model_matched, prep_matched = descriptor_matched.refit(
                X_conv_matched_clean, y_conv_matched_clean, force_structure=True,
            )
            y_pred_conv_matched_norm = ModelFactory.predict(
                model_matched, prep_matched, X_conv_matched_clean, best_model_type,
            )

            model_alt, prep_alt = descriptor_alt.refit(
                X_alt_clean, y_alt_clean, force_structure=True,
            )
            y_pred_alt_norm = ModelFactory.predict(
                model_alt, prep_alt, X_alt_clean, best_model_type,
            )

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Prediction failed: {e}")
            return None

        # ── Step 6: coefficient extraction ───────────────────────────────────
        conv_full_coeffs = self._extract_clean_coefficients(descriptor_full)
        conv_matched_coeffs = self._extract_clean_coefficients(descriptor_matched)
        alt_coeffs = self._extract_clean_coefficients(descriptor_alt)

        for coeffs, desc in [
            (conv_full_coeffs, descriptor_full),
            (conv_matched_coeffs, descriptor_matched),
            (alt_coeffs, descriptor_alt),
        ]:
            if desc.intercept is not None:
                coeffs['intercept'] = desc.intercept

        # ── Step 7: metrics (original scale) ─────────────────────────────────
        if self.normalize:
            def _orig(dataset, mask):
                prep = self.foundation.prepare_dataset(dataset, list(best_combo), target)
                return prep[1][mask]

            y_conv_full = _orig(self.foundation.conventional_full, mask_conv_full)
            y_conv_matched = _orig(matched_dataset, mask_conv_matched)
            y_alt = _orig(self.foundation.alternative, mask_alt)

            def _denorm(y_pred_norm):
                return self.foundation.denormalize_predictions(
                    y_pred_norm, norm_params, self.normalization_method,
                )

            y_pred_conv_full = _denorm(y_pred_conv_full_norm)
            y_pred_conv_matched = _denorm(y_pred_conv_matched_norm)
            y_pred_alt = _denorm(y_pred_alt_norm)

        else:
            y_conv_full = y_conv_full_clean
            y_conv_matched = y_conv_matched_clean
            y_alt = y_alt_clean
            y_pred_conv_full = y_pred_conv_full_norm
            y_pred_conv_matched = y_pred_conv_matched_norm
            y_pred_alt = y_pred_alt_norm

        conv_full_metrics = self._calculate_metrics(
            y_conv_full, y_pred_conv_full, descriptor_full.get_effective_parameters())
        conv_matched_metrics = self._calculate_metrics(
            y_conv_matched, y_pred_conv_matched, descriptor_matched.get_effective_parameters())
        alt_metrics = self._calculate_metrics(
            y_alt, y_pred_alt, descriptor_alt.get_effective_parameters())

        # ── Step 8: cross-evaluations ─────────────────────────────────────────
        cross_evals = self._perform_cross_evaluations_with_descriptors(
            descriptors={'full': descriptor_full,
                         'matched': descriptor_matched,
                         'alt': descriptor_alt},
            data_norm={'full': (X_conv_full_clean, y_conv_full_clean),
                       'matched': (X_conv_matched_clean, y_conv_matched_clean),
                       'alt': (X_alt_clean, y_alt_clean)},
            data_orig={'full': (None, y_conv_full),
                       'matched': (None, y_conv_matched),
                       'alt': (None, y_alt)},
            model_type=best_model_type,
            norm_params=norm_params,
        )

        # ── Step 9: coefficient decomposition ────────────────────────────────
        coeff_comparison = self._compare_coefficients_threeway(
            conv_full_coeffs, conv_matched_coeffs, alt_coeffs, best_model_type,
        )

        # ── Step 10: display ──────────────────────────────────────────────────
        if display_results and self.verbose:
            self.display.print_single_target_results(
                target, best_combo, best_model_type,
                conv_full_coeffs, conv_matched_coeffs, alt_coeffs,
                coeff_comparison,
                conv_full_metrics, conv_matched_metrics, alt_metrics,
                cross_evals,
            )

        # ── Step 11: result dict ──────────────────────────────────────────────
        return {
            'target': target,
            'predictors': best_combo,
            'n_predictors': len(best_combo),
            'model_type': best_model_type,
            'normalized': self.normalize,
            'normalization_method': self.normalization_method if self.normalize else None,
            'structure_preserved': True,

            'n_conventional_full': len(y_conv_full),
            'n_conventional_matched': len(y_conv_matched),
            'n_alternative': len(y_alt),

            'conv_full_r2': conv_full_metrics['r2'],
            'conv_full_rrmse': conv_full_metrics['rrmse'],
            'conv_matched_r2': conv_matched_metrics['r2'],
            'conv_matched_rrmse': conv_matched_metrics['rrmse'],
            'alt_r2': alt_metrics['r2'],
            'alt_rrmse': alt_metrics['rrmse'],

            'cross_full_on_matched_r2': cross_evals.get('full_on_matched', {}).get('r2', np.nan),
            'cross_full_on_matched_rrmse': cross_evals.get('full_on_matched', {}).get('rrmse', np.nan),
            'cross_full_on_alt_r2': cross_evals.get('full_on_alt', {}).get('r2', np.nan),
            'cross_full_on_alt_rrmse': cross_evals.get('full_on_alt', {}).get('rrmse', np.nan),
            'cross_matched_on_alt_r2': cross_evals.get('matched_on_alt', {}).get('r2', np.nan),
            'cross_matched_on_alt_rrmse': cross_evals.get('matched_on_alt', {}).get('rrmse', np.nan),

            'coefficients_full': conv_full_coeffs,
            'coefficients_matched': conv_matched_coeffs,
            'coefficients_alternative': alt_coeffs,
            'coefficient_comparison': coeff_comparison,
        }

    # ========================================================================
    # ✅ NEW HELPER METHODS FOR DESCRIPTOR HANDLING
    # ========================================================================

    def _get_base_descriptor(self, key: Tuple, model_type: str) -> Optional[ModelDescriptor]:
        """
        Get base descriptor from selector for structure preservation.

        Parameters
        ----------
        key : tuple
            (predictors, target) key
        model_type : str
            Model type string

        Returns
        -------
        ModelDescriptor or None
            Base descriptor with discovered structure
        """
        # Try to get from selector's descriptors
        if hasattr(self.selector, 'model_descriptors'):
            if key in self.selector.model_descriptors:
                if model_type in self.selector.model_descriptors[key]:
                    return self.selector.model_descriptors[key][model_type]

        # Fallback: create from fitted model if needed
        if self.verbose:
            print(f"  ⚠️  Creating descriptor on-the-fly for {key}")

        predictors, target = key

        # Get data
        data_subset = self.foundation.conventional_full[list(predictors) + [target]].dropna()
        if len(data_subset) < 20:
            return None

        X = data_subset[list(predictors)].values
        y = data_subset[target].values

        # Clean
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 20:
            return None

        # Fit and create descriptor
        try:
            model, preprocessor = ModelFactory.create_and_fit(model_type, X_clean, y_clean)

            descriptor = ModelDescriptor.from_fitted_model(
                model=model,
                preprocessor=preprocessor,
                model_type=model_type,
                independent_vars=list(predictors),
                dependent_var=target
            )

            return descriptor

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed to create descriptor: {e}")
            return None

    def _describe_mfp_transforms(self, descriptor: ModelDescriptor) -> str:
        """
        Return a compact description of the MFP transformation structure.

        Delegates to :meth:`~ModelDescriptor.get_formula_string` with
        ``include_coefficients=False``, which produces the β-symbol formula
        that already encodes the transformation for each variable (e.g.
        ``LOA = β₀ + β₁·√DWT + β₂·ln(GT)``).  This is more informative than
        manually reconstructing the transform names and avoids the deprecated
        ``descriptor._format_power()`` call that was previously used here.

        :param descriptor: Fitted MFP descriptor.
        :type descriptor: ModelDescriptor

        :returns: β-notation formula string showing the transformation
            structure, or ``'none'`` when no transformations are stored.
        :rtype: str
        """
        if not descriptor.transformations:
            return 'none'

        return descriptor.get_formula_string(
            include_coefficients=False,
            notation='math',
        )

    def _perform_cross_evaluations_with_descriptors(
            self,
            descriptors: Dict,
            data_norm: Dict,
            data_orig: Dict,
            model_type: str,
            norm_params: Optional[Dict],
    ) -> Dict:
        """
        Cross-evaluate each fitted structure on the other datasets.

        Three evaluations are performed:

        - **Full → Matched**: apply the full-conventional structure to the
          matched-conventional data.
        - **Full → Alternative**: apply the full-conventional structure to
          the alternative-fuel data.
        - **Matched → Alternative**: apply the matched-conventional structure
          to the alternative-fuel data.

        Each evaluation uses :meth:`~ModelDescriptor.refit` with
        ``force_structure=True`` on a deep copy of the source descriptor, so
        the source descriptor's coefficients are never modified.

        :param descriptors: Dict with keys ``'full'``, ``'matched'``,
            ``'alt'`` mapping to the three fitted descriptors.
        :type descriptors: dict[str, ModelDescriptor]
        :param data_norm: Dict with same keys mapping to
            ``(X_norm, y_norm)`` tuples of normalised data.
        :type data_norm: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        :param data_orig: Dict with same keys mapping to
            ``(None, y_orig)`` tuples of original-scale targets (``X`` is
            not needed here).
        :type data_orig: dict[str, tuple[None, numpy.ndarray]]
        :param model_type: Model-type identifier, forwarded to
            :meth:`~ModelFactory.predict`.
        :type model_type: str
        :param norm_params: Normalisation parameters used to back-transform
            predictions to the original scale.
        :type norm_params: dict, optional

        :returns: Dict with keys ``'full_on_matched'``,
            ``'full_on_alt'``, ``'matched_on_alt'``, each mapping to a
            metrics dict (or ``{}`` on failure).
        :rtype: dict
        """
        cross_evals = {}

        evaluations = [
            ('full_on_matched', 'full', 'matched'),
            ('full_on_alt', 'full', 'alt'),
            ('matched_on_alt', 'matched', 'alt'),
        ]

        for label, source_key, target_key in evaluations:
            try:
                X_norm, y_norm = data_norm[target_key]
                _, y_orig = data_orig[target_key]

                desc_copy = self._copy_descriptor(descriptors[source_key])
                model, prep = desc_copy.refit(X_norm, y_norm, force_structure=True)
                y_pred_norm = ModelFactory.predict(model, prep, X_norm, model_type)

                y_pred = (
                    self.foundation.denormalize_predictions(
                        y_pred_norm, norm_params, self.normalization_method,
                    )
                    if self.normalize
                    else y_pred_norm
                )

                cross_evals[label] = self._calculate_metrics(
                    y_orig, y_pred, desc_copy.get_effective_parameters(),
                )

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Cross-evaluation '{label}' failed: {e}")
                cross_evals[label] = {}

        return cross_evals

    # ========================================================================
    # EXISTING HELPER METHODS (unchanged)
    # ========================================================================

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          n_features: int) -> Dict:
        """Calculate regression metrics."""
        metric_calculator = RegressionMetrics(y_pred, y_true)
        metrics = metric_calculator.calculate_all_metrics(n_features)
        return metrics

    def _compare_coefficients_threeway(self,
                                      coeffs_full: Dict,
                                      coeffs_matched: Dict,
                                      coeffs_alt: Dict,
                                      model_type: str) -> Dict:
        """
        Compare coefficients across three datasets.

        Returns decomposition into distribution and fuel effects.
        """
        comparison = {
            'distribution_effect': {},  # Full → Matched
            'fuel_effect': {},          # Matched → Alt
            'total_effect': {},         # Full → Alt
            'interpretation': ''
        }

        # Find common coefficient keys
        common_keys = set(coeffs_full.keys()).intersection(
            set(coeffs_matched.keys()).intersection(set(coeffs_alt.keys()))
        )

        if not common_keys:
            comparison['interpretation'] = "No common coefficients found across all three models"
            return comparison

        for key in common_keys:
            val_full = coeffs_full[key]
            val_matched = coeffs_matched[key]
            val_alt = coeffs_alt[key]

            # Distribution effect (Full → Matched)
            dist_change = val_matched - val_full
            comparison['distribution_effect'][key] = {
                'from': val_full,
                'to': val_matched,
                'absolute_change': dist_change,
                'relative_change': (dist_change / val_full * 100) if abs(val_full) > 1e-10 else np.inf
                }
            # Fuel effect (Matched → Alt)
            fuel_change = val_alt - val_matched
            comparison['fuel_effect'][key] = {
                'from': val_matched,
                'to': val_alt,
                'absolute_change': fuel_change,
                'relative_change': (fuel_change / val_matched * 100) if abs(val_matched) > 1e-10 else np.inf
            }

            # Total effect (Full → Alt)
            total_change = val_alt - val_full
            comparison['total_effect'][key] = {
                'from': val_full,
                'to': val_alt,
                'absolute_change': total_change,
                'relative_change': (total_change / val_full * 100) if abs(val_full) > 1e-10 else np.inf
            }

        # Calculate fuel contribution percentage
        total_abs_change = sum(abs(comparison['total_effect'][k]['absolute_change'])
                               for k in common_keys)
        fuel_abs_change = sum(abs(comparison['fuel_effect'][k]['absolute_change'])
                              for k in common_keys)

        if total_abs_change > 1e-10:
            fuel_contribution = 100 * fuel_abs_change / total_abs_change
            comparison['interpretation'] = (
                f"Fuel effect accounts for {fuel_contribution:.1f}% of total coefficient changes"
            )
        else:
            comparison['interpretation'] = "Coefficients show minimal change across datasets"

        return comparison

    # ========================================================================
    # DECOMPOSITION METHOD
    # ========================================================================

    def decompose_coefficient_changes(self,
                                      effect_size_threshold: float = 0.2) -> pd.DataFrame:
        """
        Decompose coefficient changes into distribution and fuel effects.

        Uses display class for printing.
        """
        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run analyze_all_targets() first.")
            return pd.DataFrame()

        if not self.normalize and self.verbose:
            print("⚠️  WARNING: Normalization not enabled!")
            print("    Effect sizes may not be comparable across variables")

        decompositions = []

        for _, row in self.comparison_results.iterrows():
            coeff_comp = row['coefficient_comparison']

            coef_keys = [k for k in coeff_comp['fuel_effect'].keys()
                         if k != 'interpretation']

            for coef_name in coef_keys:
                dist_effect = coeff_comp['distribution_effect'][coef_name]
                fuel_effect = coeff_comp['fuel_effect'][coef_name]
                total_effect = coeff_comp['total_effect'][coef_name]

                total_abs = abs(total_effect['absolute_change'])
                if total_abs > 1e-10:
                    dist_pct = 100 * abs(dist_effect['absolute_change']) / total_abs
                    fuel_pct = 100 * abs(fuel_effect['absolute_change']) / total_abs
                else:
                    dist_pct = np.nan
                    fuel_pct = np.nan

                # Classify effect size
                fuel_abs = abs(fuel_effect['absolute_change'])
                if fuel_abs < effect_size_threshold:
                    effect_size = "Negligible"
                    emoji = "✓"
                elif fuel_abs < 0.5:
                    effect_size = "Small"
                    emoji = "⚠️"
                elif fuel_abs < 0.8:
                    effect_size = "Medium"
                    emoji = "🔶"
                else:
                    effect_size = "Large"
                    emoji = "🚨"

                decompositions.append({
                    'target': row['target'],
                    'model_type': row['model_type'],
                    'coefficient': coef_name,
                    'full': dist_effect['from'],
                    'matched': dist_effect['to'],
                    'alternative': fuel_effect['to'],
                    'distribution_effect': dist_effect['absolute_change'],
                    'fuel_effect': fuel_effect['absolute_change'],
                    'total_effect': total_effect['absolute_change'],
                    'distribution_pct': dist_pct,
                    'fuel_pct': fuel_pct,
                    'fuel_effect_size': effect_size,
                    'fuel_significant': fuel_abs >= effect_size_threshold,
                    'emoji': emoji
                })

        decomp_df = pd.DataFrame(decompositions)

        if self.verbose:
            self.display.print_decomposition_summary(decomp_df, effect_size_threshold)

        return decomp_df

    # ========================================================================
    # CONVENIENCE DISPLAY METHODS (delegate to display class)
    # ========================================================================

    def print_matching_quality(self):
        """Print matching quality assessment."""
        self.display.print_matching_quality(self.foundation.matched_data_sets)

    # ========================================================================
    # EXPORT METHODS
    # ========================================================================

    def export_results(self, filename: str = 'coefficient_comparison_threeway.csv'):
        """Export three-way comparison results to CSV."""
        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to export. Run analyze_all_targets() first.")
            return

        export_df = self.comparison_results.copy()

        # Convert tuple predictors to string
        export_df['predictors'] = export_df['predictors'].apply(
            lambda x: ' + '.join(x) if isinstance(x, tuple) else x
        )

        # Convert dict columns to JSON strings
        export_df['coefficients_full'] = export_df['coefficients_full'].apply(str)
        export_df['coefficients_matched'] = export_df['coefficients_matched'].apply(str)
        export_df['coefficients_alternative'] = export_df['coefficients_alternative'].apply(str)
        export_df['coefficient_comparison'] = export_df['coefficient_comparison'].apply(str)

        export_df.to_csv(filename, index=False)

        if self.verbose:
            print(f"✅ Three-way results exported to {filename}")
            print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def plot_coefficient_changes(self,
                                 figsize_per_plot: Tuple[float, float] = (8, 5),
                                 save_path: Optional[str] = None,
                                 show_values: bool = True,
                                 min_change_threshold: float = 0.01,
                                 show_distribution: bool = True,
                                 show_fuel: bool = True,
                                 n_cols: int = 3,
                                 bar_height: float = 0.25,
                                 bar_spacing: float = 0.5,
                                 title_prefix: str = '',
                                 title_suffix: str = '',
                                 min_r2: Optional[float] = None,
                                 filter_dataset: str = 'alternative',
                                 min_n_samples: Optional[int] = None,
                                 variable_name_map: Optional[Dict[str, str]] = None,  # For subplot titles
                                 formula_name_map: Optional[Dict[str, str]] = None,  # NEW: For formulas
                                 show_sign_change_legend: bool = False,
                                 debug: bool = False):
        """
        Plot coefficient changes for all targets in grid layout.

        NEW: Separate name maps for subplot titles vs formulas.

        Parameters
        ----------
        [existing parameters...]
        variable_name_map : dict, optional
            Mapping for subplot titles (target names).
            Example: {'BEAM (M)': 'Beam (m)', 'DWT': 'DWT (t)'}
        formula_name_map : dict, optional
            Mapping for formula display (compact notation).
            Example: {'BEAM (M)': 'B', 'LOA (M)': 'L', 'DWT': 'DWT'}
            If None, uses variable_name_map.

        Examples
        --------
        >>> # Define name maps
        >>> title_map = {
        ...     'DWT': 'DWT (t)',
        ...     'LOA (M)': 'LOA (m)',
        ...     'BEAM (M)': 'Beam (m)',
        ...     'DRAFT (M)': 'Draught (m)'
        ... }
        >>>
        >>> formula_map = {
        ...     'DWT': 'DWT',
        ...     'LOA (M)': 'L',
        ...     'BEAM (M)': 'B',
        ...     'DRAFT (M)': 'T'
        ... }
        >>>
        >>> # Plot with both mappings
        >>> analyzer.plot_coefficient_changes(
        ...     min_r2=0.6,
        ...     title_prefix='Bulk Carriers',
        ...     variable_name_map=title_map,  # For "Beam (m)" in subplot title
        ...     formula_name_map=formula_map,  # For "B" in formula
        ...     show_fuel=True
        ... )
        """
        plotter = CoefficientChangePlotter(self)
        plotter.plot_coefficient_changes_all_targets(
            figsize_per_plot=figsize_per_plot,
            save_path=save_path,
            show_values=show_values,
            min_change_threshold=min_change_threshold,
            show_distribution=show_distribution,
            show_fuel=show_fuel,
            n_cols=n_cols,
            bar_height=bar_height,
            bar_spacing=bar_spacing,
            title_prefix=title_prefix,
            title_suffix=title_suffix,
            min_r2=min_r2,
            filter_dataset=filter_dataset,
            min_n_samples=min_n_samples,
            variable_name_map=variable_name_map,
            formula_name_map=formula_name_map,  # NEW
            show_sign_change_legend=show_sign_change_legend,
            debug=debug
        )

    def plot_single_target_coefficients(self,
                                        target: str,
                                        figsize: Tuple[float, float] = (12, 8),
                                        save_path: Optional[str] = None,
                                        show_values: bool = True,
                                        min_change_threshold: float = 0.01,
                                        show_distribution: bool = True,
                                        show_fuel: bool = True,
                                        bar_height: float = 0.35,
                                        bar_spacing: float = 0.6,
                                        title_prefix: str = '',
                                        title_suffix: str = ''):
        """
        Plot coefficient changes for a single target (in analyzer class).

        NEW: This is just a wrapper that calls the plotter's method.
        """
        plotter = CoefficientChangePlotter(self)
        plotter.plot_coefficient_changes_single_target(
            target=target,
            figsize=figsize,
            save_path=save_path,
            show_values=show_values,
            min_change_threshold=min_change_threshold,
            show_distribution=show_distribution,
            show_fuel=show_fuel,
            bar_height=bar_height,
            bar_spacing=bar_spacing,
            title_prefix=title_prefix,
            title_suffix=title_suffix
        )

    def compare_filtering_strategies(self,
                                     max_predictors_options: List[Optional[int]] = [None, 2, 1],
                                     require_alt_data_options: List[bool] = [True, False]) -> pd.DataFrame:
        """
        Compare how different filtering strategies affect target coverage.

        Useful for understanding trade-offs between model complexity and data availability.

        Parameters
        ----------
        max_predictors_options : list, default=[None, 2, 1]
            Different max_predictors values to try
        require_alt_data_options : list, default=[True, False]
            Whether to require alternative data

        Returns
        -------
        pd.DataFrame
            Summary showing how many targets can be analyzed under each strategy

        Examples
        --------
        >>> # Compare different strategies
        >>> comparison = analyzer.compare_filtering_strategies()
        >>> print(comparison)
        """
        strategies = []

        for max_preds in max_predictors_options:
            for require_alt in require_alt_data_options:
                # Count how many targets can be analyzed
                valid_targets = []
                skipped_targets = []

                for target in self.foundation.target_dimensions:
                    result = self._get_best_combination_for_target(
                        target=target,
                        max_predictors=max_preds,
                        require_alternative_data=require_alt,
                        min_alternative_samples=10
                    )

                    if result[0] is not None:
                        valid_targets.append(target)
                    else:
                        skipped_targets.append(target)

                strategies.append({
                    'max_predictors': max_preds if max_preds is not None else 'None',
                    'require_alternative_data': require_alt,
                    'n_valid_targets': len(valid_targets),
                    'n_skipped_targets': len(skipped_targets),
                    'coverage_pct': 100 * len(valid_targets) / len(self.foundation.target_dimensions),
                    'valid_targets': ', '.join(valid_targets[:5]) + ('...' if len(valid_targets) > 5 else ''),
                    'skipped_targets': ', '.join(skipped_targets[:5]) + ('...' if len(skipped_targets) > 5 else '')
                })

        strategy_df = pd.DataFrame(strategies)

        if self.verbose:
            print("\n" + "=" * 120)
            print("FILTERING STRATEGY COMPARISON")
            print("=" * 120)
            print(f"\nTotal targets: {len(self.foundation.target_dimensions)}")
            print("\nStrategy effectiveness:")
            print(strategy_df[['max_predictors', 'require_alternative_data',
                               'n_valid_targets', 'coverage_pct']].to_string(index=False))

            # Recommend best strategy
            best_idx = strategy_df['n_valid_targets'].idxmax()
            best = strategy_df.iloc[best_idx]

            print(f"\n💡 Recommended strategy:")
            print(f"   max_predictors={best['max_predictors']}, "
                  f"require_alternative_data={best['require_alternative_data']}")
            print(f"   Covers {best['n_valid_targets']}/{len(self.foundation.target_dimensions)} targets "
                  f"({best['coverage_pct']:.1f}%)")

            print("=" * 120)

        return strategy_df

    def _extract_clean_coefficients(self, descriptor: ModelDescriptor) -> Dict[str, float]:
        """
        Extract coefficients excluding bias terms.

        For polynomial models, PolynomialFeatures creates a '1' feature
        that duplicates the intercept. This method excludes it.

        Parameters
        ----------
        descriptor : ModelDescriptor
            Fitted model descriptor

        Returns
        -------
        dict
            Clean coefficients {feature_name: value}
            Does NOT include 'intercept' or '1' bias term
        """
        if descriptor.coefficients is None:
            return {}

        # ✅ Filter out both intercept representations
        clean_coeffs = {
            name: value
            for name, value in descriptor.coefficients.items()
            if name not in ['intercept', '1']  # ← Skip bias terms
        }

        return clean_coeffs

    def create_subscript_name_map(self,
                                  mappings: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
        """
        Convenience function to create name maps with subscripts.

        Concept:
        --------
        Instead of manually creating Unicode subscripts, provide tuples of
        (main_symbol, subscript_text) and this function creates the proper mapping.

        Parameters
        ----------
        mappings : dict
            Mapping from internal names to (main, subscript) tuples.
            Example: {'LOA (M)': ('L', 'OA'), 'LBP (M)': ('L', 'BP')}

        Returns
        -------
        dict
            Name map with Unicode subscripts

        Examples
        --------
        >>> mappings = {
        ...     'LOA (M)': ('L', 'OA'),
        ...     'LBP (M)': ('L', 'BP'),
        ...     'BEAM (M)': ('B', ''),
        ...     'DRAFT (M)': ('T', ''),
        ...     'SPEED (KN)': ('V', '')
        ... }
        >>> name_map = create_subscript_name_map(mappings)
        >>> # Returns: {'LOA (M)': 'Lₒₐ', 'LBP (M)': 'L_BP', 'BEAM (M)': 'B', ...}
        """
        plotter = CoefficientChangePlotter(self)

        result = {}
        for internal_name, (main, subscript) in mappings.items():
            if subscript:
                result[internal_name] = plotter._format_subscript_name(main, subscript)
            else:
                result[internal_name] = main

        return result

    def _copy_descriptor(self, descriptor: ModelDescriptor) -> ModelDescriptor:
        """
        Return a deep copy of a :class:`~ModelDescriptor`.

        The copy is produced via the serialisation round-trip
        (:meth:`~ModelDescriptor.to_dict` → :meth:`~ModelDescriptor.from_dict`),
        which is the canonical way to clone a descriptor while preserving all
        structural fields (transformations, active features, regularisation
        parameters, coefficients).

        This helper exists to avoid repeating the verbose
        ``descriptor.__class__.from_dict(descriptor.to_dict())`` pattern at
        every refit call site.

        :param descriptor: Source descriptor to copy.
        :type descriptor: ModelDescriptor

        :returns: Independent copy of the descriptor.
        :rtype: ModelDescriptor
        """
        return ModelDescriptor.from_dict(descriptor.to_dict())


    def _clean_data_for_nonlinear(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        dataset_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Remove invalid rows from ``(X, y)`` for nonlinear model types.

        Nonlinear models fitted via :func:`scipy.optimize.curve_fit` fail when
        the data contains NaN or Inf values.  Log-based models additionally
        require all inputs and targets to be strictly positive (normalisation
        can introduce negative values even when the original data was positive).

        For linear and polynomial model types this method is a no-op: it returns
        the original arrays unchanged with an all-``True`` mask.

        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param model_type: Model-type identifier used to decide whether
            cleaning is required and which positivity constraints apply.
        :type model_type: str
        :param dataset_name: Human-readable label used in verbose output
            (e.g. ``'Full conventional'``, ``'Alternative'``).
        :type dataset_name: str

        :returns: A three-element tuple ``(X_clean, y_clean, mask)``.
            Returns ``(None, None, None)`` when fewer than 10 valid samples
            remain after cleaning, signalling that this target should be
            skipped.
        :rtype: tuple[numpy.ndarray | None, numpy.ndarray | None,
            numpy.ndarray | None]
        """
        nonlinear_models = {
            'power-log-multivariate', 'exponential-log-multivariate',
            'power-offset', 'exponential-offset',
            'power', 'exponential',
            'logarithmic', 'logarithmic-power',
        }

        if model_type not in nonlinear_models:
            return X, y, np.ones(len(X), dtype=bool)

        n_original = len(X)

        # Remove NaN / Inf
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        mask &= ~(np.isinf(X).any(axis=1) | np.isinf(y))

        # Log-based models require strictly positive values
        if 'log' in model_type or model_type in {'power', 'logarithmic', 'logarithmic-power'}:
            mask &= (X > 0).all(axis=1) & (y > 0)

        X_clean = X[mask]
        y_clean = y[mask]
        n_removed = n_original - len(X_clean)

        if len(X_clean) < 10:
            if self.verbose:
                print(f"  ⚠️  {dataset_name}: Insufficient clean data "
                      f"({len(X_clean)}/{n_original} samples remain)")
            return None, None, None

        if self.verbose and n_removed > 0:
            pct = 100 * n_removed / n_original
            print(f"  ℹ️  {dataset_name}: Cleaned data "
                  f"({n_original} → {len(X_clean)} samples, "
                  f"removed {n_removed} with NaN/Inf/negative [{pct:.1f}%])")

        return X_clean, y_clean, mask