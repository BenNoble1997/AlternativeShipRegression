# parametric_cross_dataset_analysis.py

"""
Cross-Dataset Performance Analyzer
==================================
Compares model performance when trained on different datasets.

Uses composition with ParametricAnalysisFoundation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from knowledge_layer.multi_dataset_analysers.parametric_analysis_foundation import ParametricAnalysisFoundation
from knowledge_layer.multi_dataset_analysers.small_dataset_evaluation import SmallDatasetEvaluator
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.multi_dataset_analysers.cross_dataset.ScaleReference import ScaleReference
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility

class CrossDatasetPerformanceAnalyzer:
    """
    Analyzes how model performance varies with training dataset.

    Uses ParametricAnalysisFoundation via composition.

    Key Question:
    ------------
    "Does training on alternative ships improve alternative ship predictions?"

    Analysis:
    --------
    For each target:
    1. Get best predictor combo + model type (from selector)
    2. Train on conventional, matched, and alternative datasets
    3. Evaluate each via k-fold CV
    4. Compare performance and stability
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 independent_variables: list[str],
                 dependent_variables: list[str]|None = None,
                 parametric_selector: 'ParametricModelSelector' = None,
                 model_recommendations: dict|None = None,
                 detailed_results: dict|None = None,
                 selection_metric: str = 'rrmse',
                 selection_strategy: str = 'simple',
                 target_predictor_constraints: dict[str, list[str]] | None = None,
                 random_state: int = 42,
                scale_reference:ScaleReference|None=None,
                 verbose: bool = True,
                 build_year_col:str|None=None,
                 year_edges:list|None=None,
                 build_eligibility:TemporalEligibility|None=None,
                 include_matched:bool=False):
        """Initialize analyzer with foundation and evaluator.

        :param target_predictor_constraints: Allowed predictors per target, forwarded
        to the foundation so that candidate filtering matches Experiment 2.
    """

        # Extract structure from the selector, but NOT the predictor-set metric.
        #
        # The selector's own `selection_metric` governs the model-FORM decision
        # (which functional form wins within a fixed predictor set), where every
        # candidate shares the same rows and AIC/BIC is comparable. It must not
        # govern the predictor-SET decision made here, where candidates are
        # fitted to different complete-case subsets and information criteria
        # reward missingness. That metric stays under the caller's control.
        if parametric_selector is not None:
            model_recommendations = parametric_selector.model_recommendations
            detailed_results = parametric_selector.detailed_results
            selection_strategy = parametric_selector.model_selector.selection_strategy
            self.model_descriptors = parametric_selector.model_descriptors

            if verbose:
                print(f"  Model forms selected under "
                      f"'{parametric_selector.selection_metric}' (from selector); "
                      f"predictor sets ranked by '{selection_metric}' (from caller)")
        else:
            self.model_descriptors = {}

        # Foundation for data management
        self.foundation = ParametricAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            independent_variables=independent_variables,
            dependent_variables=dependent_variables,
            model_recommendations=model_recommendations,
            detailed_results=detailed_results,
            selection_metric=selection_metric,
            selection_strategy=selection_strategy,
            target_predictor_constraints=target_predictor_constraints,
            random_state=random_state,
            verbose=verbose,
            defer_matched_sets=not include_matched,
            build_year_col=build_year_col,
            year_edges=year_edges,
            build_eligibility=build_eligibility
        )

        # Evaluator for multiple evaluation strategies
        self.evaluator = SmallDatasetEvaluator(
            random_state=random_state,
            verbose=verbose
        )

        self.include_matched=include_matched

        self.scale_reference = scale_reference
        self.random_state = random_state
        self.verbose = verbose

        # Storage
        self.results = {}
        self.results_df = None

    # ========================================================================
    # DELEGATE PROPERTIES
    # ========================================================================

    @property
    def target_dimensions(self) -> List[str]:
        return self.foundation.target_dimensions

    @property
    def conventional_full(self) -> pd.DataFrame:
        return self.foundation.conventional_full

    @property
    def alternative(self) -> pd.DataFrame:
        return self.foundation.alternative

    def _is_structure_discovering_model(self, model_type: str) -> bool:
        """Check if model discovers structure during fitting."""
        structure_discovering = [
            'mfp',
            'polynomial-2-lasso',
            'polynomial-3-lasso',
            'polynomial-2-lasso-cv',
            'polynomial-3-lasso-cv',
        ]
        return model_type in structure_discovering

    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze_best_combinations(self,
                                  evaluation_method: str = 'repeated-cv',
                                  n_folds: int = 5,
                                  n_repeats: int = 10,
                                  n_bootstraps: int = 100,
                                  display_results: bool = True,
                                  max_predictors: Optional[int] = None,
                                  require_alternative_data: bool = False,  # ✅ Keep False as default
                                  min_alternative_samples: int = 5  # ✅ NEW
                                  ) -> pd.DataFrame:
        """
        Analyze best combination for each target across datasets.

        Parameters
        ----------
        evaluation_method : str, default='repeated-cv'
            Evaluation strategy:
            - 'standard-cv': Simple k-fold CV
            - 'repeated-cv': k-fold CV repeated multiple times (recommended)
            - 'bootstrap': Bootstrap .632+ (best for very small datasets)
        n_folds : int, default=5
            Number of CV folds (for 'standard-cv' and 'repeated-cv')
        n_repeats : int, default=10
            Number of repetitions (for 'repeated-cv')
        n_bootstraps : int, default=100
            Number of bootstrap samples (for 'bootstrap')
        display_results : bool
            Print results table
        max_predictors : int, optional
            Maximum number of predictors to consider
        require_alternative_data : bool, default=False
            Whether to require sufficient alternative data.

            **Default False:** We want to evaluate performance even on small 
            alternative datasets - that's part of what we're testing!

            Set to True only if you want to exclude targets with insufficient
            alternative data entirely.
        min_alternative_samples : int, default=5
            Minimum samples required in alternative dataset.
            Only used when require_alternative_data=True.

        Returns
        -------
        pd.DataFrame
            Results with performance metrics

        Examples
        --------
        >>> # Standard: Test performance on all datasets (including small ones)
        >>> results = analyzer.analyze_best_combinations(
        ...     evaluation_method='repeated-cv',
        ...     max_predictors=2
        ... )

        >>> # Conservative: Only analyze targets with good alternative data
        >>> results = analyzer.analyze_best_combinations(
        ...     evaluation_method='repeated-cv',
        ...     max_predictors=2,
        ...     require_alternative_data=True,
        ...     min_alternative_samples=10
        ... )
        """
        if evaluation_method not in ['standard-cv', 'repeated-cv', 'bootstrap']:
            raise ValueError(
                f"evaluation_method must be 'standard-cv', 'repeated-cv', or 'bootstrap'. "
                f"Got: {evaluation_method}"
            )

        if display_results and self.verbose:
            print("\n" + "=" * 140)
            print("CROSS-DATASET PERFORMANCE ANALYSIS")
            print("=" * 140)

            if evaluation_method == 'repeated-cv':
                print(f"Evaluation: {n_folds}-fold CV repeated {n_repeats} times "
                      f"({n_folds * n_repeats} evaluations per model)")
            elif evaluation_method == 'bootstrap':
                print(f"Evaluation: Bootstrap .632+ ({n_bootstraps} samples per model)")
            else:
                print(f"Evaluation: {n_folds}-fold CV")

            print("Consistent methodology applied to all datasets")

            # ✅ NEW: Report filtering configuration
            if max_predictors is not None:
                print(f"  📊 Max predictors: {max_predictors}")
            if require_alternative_data:
                print(f"  📊 Requiring sufficient alternative data (min_samples={min_alternative_samples})")
            else:
                print(f"  📊 Testing on all combinations (including those with limited alternative data)")

            print("=" * 140)

        all_results = []
        skipped_targets = []  # ✅ Track skipped targets

        for target in self.target_dimensions:
            # ✅ UPDATED: Pass all filtering parameters
            best_combo, model_type = self._get_best_combination_for_target(
                target=target,
                max_predictors=max_predictors,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

            if best_combo is None:
                if self.verbose:
                    print(f"\n  ⚠️  No combinations found for {target}")
                skipped_targets.append(target)
                continue

            if self.verbose:
                print(f"\nAnalyzing {target}:")
                print(f"  Best predictors: {' + '.join(best_combo)}")
                print(f"  Model: {model_type}")

            # Analyze across datasets with selected method
            result = self._analyze_combination_across_datasets(
                predictors=list(best_combo),
                target=target,
                model_type=model_type,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps,
                include_matched=self.include_matched,
            )

            if result is not None:
                result['target'] = target
                result['predictor_str'] = " + ".join(best_combo)
                result['predictors'] = list(best_combo)
                result['n_predictors'] = len(best_combo)
                result['model_type'] = model_type
                result['evaluation_method'] = evaluation_method
                all_results.append(result)
            else:
                skipped_targets.append(target)

        # Create DataFrame
        if not all_results:
            if self.verbose:
                print("\n  ⚠️  No valid results generated!")
                if skipped_targets:
                    print(f"     Skipped targets: {', '.join(skipped_targets)}")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Reorder columns
        column_order = [
            'target', 'predictor_str', 'predictors', 'n_predictors', 'model_type',
            'evaluation_method',
            'conv_q2', 'conv_q2_std',
            'matched_q2', 'matched_q2_std',
            'alt_q2', 'alt_q2_std',
            'conv_rrmse', 'conv_rrmse_std',
            'matched_rrmse', 'matched_rrmse_std',
            'alt_rrmse', 'alt_rrmse_std',
        ]

        if evaluation_method == 'bootstrap':
            bootstrap_cols = [
                                 f'{prefix}_{metric}_{part}'
                                 for prefix in ('conv', 'matched', 'alt')
                                 for metric in ('q2', 'adj_q2', 'rrmse')
                                 for part in ('train', 'oob')
                             ] + [f'{prefix}_relative_overfitting' for prefix in ('conv', 'matched', 'alt')]
            column_order.extend(bootstrap_cols)

        column_order.extend(['conv_n', 'matched_n', 'alt_n'])

        existing_cols = [col for col in column_order if col in results_df.columns]
        remaining_cols = [col for col in results_df.columns if col not in column_order]
        results_df = results_df[existing_cols + remaining_cols]

        # Display
        if display_results and self.verbose:
            self._print_results_table(results_df, evaluation_method)

            # ✅ NEW: Report skipped targets
            if skipped_targets:
                print(f"\n⚠️  Skipped {len(skipped_targets)}/{len(self.target_dimensions)} targets:")
                print(f"    {', '.join(skipped_targets)}")
                print(f"    (Insufficient data or no valid combinations)")

        # Store
        self.results_df = results_df
        self.results = self._dataframe_to_dict(results_df)

        return results_df

    def _analyze_combination_across_datasets(self,
                                             predictors: List[str],
                                             target: str,
                                             model_type: str,
                                             evaluation_method: str,
                                             n_folds: int,
                                             n_repeats: int,
                                             n_bootstraps: int,
                                             include_matched:bool) -> Optional[Dict]:
        """Evaluate on three datasets using selected method."""
        # Get matched dataset
        datasets = {'conv': self.conventional_full}

        if self.include_matched:
            matched_dataset = self.foundation.get_matched_dataset(tuple(predictors))

            if matched_dataset is None:
                if self.verbose:
                    print(f"    ⚠️  No matched dataset for {predictors}")
                return None

            datasets['matched'] = matched_dataset

        datasets['alt'] = self.alternative

        results = {}
        failed_datasets = []  # ✅ Track failures

        # Evaluate on each dataset with SAME method
        for dataset_name, dataset in datasets.items():
            dataset_results = self._evaluate_on_dataset(
                dataset=dataset,
                predictors=predictors,
                target=target,
                model_type=model_type,
                dataset_name=dataset_name,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps
            )

            if dataset_results is None:
                failed_datasets.append(dataset_name)

                # ✅ DECISION: Skip entire target if ANY dataset fails completely
                # This ensures we only report complete three-way comparisons
                if self.verbose:
                    print(f"    ⚠️  Cannot complete three-way comparison for {target}:")
                    print(f"        {dataset_name} dataset has no valid data for {predictors}")
                return None

            # Flatten with dataset prefix
            for metric, value in dataset_results.items():
                if metric == 'relative_overfitting':
                    results[f'{dataset_name}_q_{metric}'] = value
                else:
                    results[f'{dataset_name}_{metric}'] = value

        return results

    def _evaluate_on_dataset(self,
                             dataset: pd.DataFrame,
                             predictors: List[str],
                             target: str,
                             model_type: str,
                             dataset_name: str,
                             evaluation_method: str,
                             n_folds: int,
                             n_repeats: int,
                             n_bootstraps: int,
                             debug:bool=False) -> Optional[Dict]:
        """
        Evaluate one model on one cohort using the selected strategy.

        :param debug: Emit the resolved reference scale and divisor provenance.
        :returns: Metric dictionary, or ``None`` if the cohort has no usable data.
        """
        # Prepare data
        data_prep = self.foundation.prepare_dataset(dataset, predictors, target)

        if data_prep is None:
            if self.verbose:
                print(f"    ⚠️  {dataset_name}: Insufficient data")
            return None

        X, y = data_prep
        n_samples = len(X)

        reference_scale = (
            None if self.scale_reference is None
            else self.scale_reference.for_target(target)
        )

        if debug:
            local_mean = float(np.mean(y))
            if reference_scale is None:
                print(f"      [scale] {dataset_name}/{target}: cohort-local "
                      f"divisor={local_mean:.3f}")
            else:
                ratio = local_mean / reference_scale
                print(f"      [scale] {dataset_name}/{target}: fixed "
                      f"divisor={reference_scale:.3f}, cohort mean={local_mean:.3f} "
                      f"(ratio {ratio:.3f})")

        if self.verbose:
            print(f"    {dataset_name}: {evaluation_method} (n={n_samples})")

        # Check if model needs structure fixing
        key = (tuple(predictors), target)
        is_structure_discovering = self._is_structure_discovering_model(model_type)

        if is_structure_discovering and key in self.model_descriptors:
            descriptor = self.model_descriptors[key].get(model_type)

            if descriptor is None:
                if self.verbose:
                    print(f"    ⚠️  {dataset_name}: No descriptor found for {model_type}")
                return None

            result = self._evaluate_with_fixed_structure(
                X, y, descriptor,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps,
                reference_scale=reference_scale
            )
        else:
            result = self._evaluate_standard(
                X, y, model_type,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps,
                reference_scale=reference_scale
            )

        if result is None:
            return None

        # Add sample size
        result['n'] = n_samples
        result['reference_scale'] = reference_scale

        # Standardize output format
        if evaluation_method == 'bootstrap':
            standardized = {}
            for key, value in result.items():
                if key.endswith('_632plus'):
                    base_metric = key.replace('_632plus', '')
                    standardized[base_metric] = value
                else:
                    standardized[key] = value
            result = standardized

        if self.verbose:
            r2_val = result.get('q2', np.nan)
            r2_std = result.get('q2_std', 0)
            rrmse_val = result.get('rrmse', np.nan)
            rrmse_std = result.get('rrmse_std', 0)

            print(f"        → R²={r2_val:.3f}±{r2_std:.3f}, "
                  f"RRMSE={rrmse_val:.1f}±{rrmse_std:.1f}%")

        return result

    # ========================================================================
    # STRUCTURE-PRESERVING EVALUATION (unchanged from before)
    # ========================================================================

    def _evaluate_with_fixed_structure(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       descriptor: 'ModelDescriptor',
                                       evaluation_method: str,
                                       n_folds: int,
                                       n_repeats: int,
                                       n_bootstraps: int,
                                       reference_scale:float|None) -> dict|None:
        """Evaluate structure-discovering model with FIXED structure."""
        if evaluation_method == 'standard-cv':
            return self._fixed_structure_standard_cv(X, y, descriptor, n_folds, reference_scale=reference_scale)
        elif evaluation_method == 'repeated-cv':
            return self._fixed_structure_repeated_cv(X, y, descriptor, n_folds, n_repeats, reference_scale=reference_scale)
        elif evaluation_method == 'bootstrap':
            return self._fixed_structure_bootstrap(X, y, descriptor, n_bootstraps, reference_scale=reference_scale)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")

    def _fixed_structure_standard_cv(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     descriptor: 'ModelDescriptor',
                                     n_folds: int,
                                     reference_scale: float | None = None,
                                     min_coverage: float = 0.8) -> Optional[Dict]:
        """
        Single-partition k-fold CV with fixed structure, scored on pooled
        out-of-fold predictions.

        Each sample is held out exactly once, so the k folds together produce one
        prediction per sample. The metric is computed once on that pooled vector.
        No dispersion is returned: a single partition provides one estimate, and
        the fold-to-fold spread reported previously measured how unevenly the
        shuffle divided the cohort rather than any property of the model.
        """
        from sklearn.model_selection import KFold

        n_samples = len(y)
        oof_pred = np.full(n_samples, np.nan)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kfold.split(X):
            try:
                model, prep = descriptor.refit(X=X[train_idx], y=y[train_idx],
                                               force_structure=True)
                oof_pred[test_idx] = ModelFactory.predict(
                    model, prep, X[test_idx], descriptor.model_type
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"      Fold failed: {e}")
                continue

        predicted = ~np.isnan(oof_pred)
        coverage = predicted.sum() / n_samples

        if coverage < min_coverage:
            if self.verbose:
                print(f"      Discarded: only {coverage:.0%} of samples predicted")
            return None

        metric_calc = RegressionMetrics(y_true=y[predicted],
                                        y_pred=oof_pred[predicted],
                                        reference=reference_scale)
        metrics = metric_calc.calculate_cv_metrics()
        metrics['coverage'] = coverage
        metrics['n_evaluations'] = 1

        return metrics

    def _fixed_structure_repeated_cv(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     descriptor: 'ModelDescriptor',
                                     n_folds: int,
                                     n_repeats: int,
                                     reference_scale: float | None = None,
                                     min_coverage: float = 0.8) -> Optional[Dict]:
        """
        Repeated k-fold CV with fixed structure, scored on pooled out-of-fold
        predictions.

        One metric set is produced per repeat, from the full out-of-fold vector of
        that partition. The dispersion returned by :meth:`_average_cv_metrics` is
        therefore the spread across partitions, quantifying how much the estimate
        depends on which samples were grouped together, rather than the spread
        across folds.

        ``RepeatedKFold`` is deliberately not used: it emits all splits as a flat
        sequence with no repeat boundary, and pooling must occur within a repeat.
        """
        from sklearn.model_selection import KFold

        n_samples = len(y)
        cv_metrics = []

        for repeat_idx in range(n_repeats):
            kfold = KFold(n_splits=n_folds, shuffle=True,
                          random_state=self.random_state + repeat_idx)

            oof_pred = np.full(n_samples, np.nan)

            for train_idx, test_idx in kfold.split(X):
                try:
                    model, prep = descriptor.refit(X[train_idx], y[train_idx],
                                                   force_structure=True)
                    oof_pred[test_idx] = ModelFactory.predict(
                        model, prep, X[test_idx], descriptor.model_type
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue

            predicted = ~np.isnan(oof_pred)
            coverage = predicted.sum() / n_samples

            if coverage < min_coverage:
                continue

            metric_calc = RegressionMetrics(y_true=y[predicted],
                                            y_pred=oof_pred[predicted],
                                            reference=reference_scale)
            metrics = metric_calc.calculate_cv_metrics()
            metrics['coverage'] = coverage
            cv_metrics.append(metrics)

        if not cv_metrics:
            return None

        avg_metrics = self._average_cv_metrics(cv_metrics)
        avg_metrics['n_evaluations'] = len(cv_metrics)

        return avg_metrics

    def _fixed_structure_bootstrap(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   descriptor: 'ModelDescriptor',
                                   n_bootstraps: int,
                                   reference_scale:float|None=None) -> dict|None:
        """
        Bootstrap .632+ evaluation holding the discovered model structure fixed.

        **Concept:**
        In-bag and out-of-bag predictions are scored with the *same* divisor. With
        cohort-local normalisation these two arms use the in-bag and out-of-bag
        sample means respectively, so the .632+ blend combines quantities on
        different scales — at n≈15 the two means can differ by several percent.
        A fixed reference makes both arms commensurable.

        :param reference_scale: Fixed divisor for relative error metrics.
        :returns: .632+ corrected metrics, or ``None`` if every replicate failed.
        """
        n_samples = len(X)
        n_params = descriptor.get_effective_parameters()

        oob_metrics = []
        train_metrics = []

        for i in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx, assume_unique=False)

            if len(oob_idx) < 5:
                continue

            X_boot, y_boot = X[boot_idx], y[boot_idx]
            X_oob, y_oob = X[oob_idx], y[oob_idx]

            try:
                model, prep = descriptor.refit(X_boot, y_boot, force_structure=True)

                y_pred_oob = ModelFactory.predict(model, prep, X_oob, descriptor.model_type)
                metric_calc_oob = RegressionMetrics(y_true=y_oob, y_pred=y_pred_oob, reference=reference_scale)
                metrics_oob = metric_calc_oob.calculate_predictive_metrics(n_params)
                oob_metrics.append(metrics_oob)

                y_pred_train = ModelFactory.predict(model, prep, X_boot, descriptor.model_type)
                metric_calc_train = RegressionMetrics(y_true=y_boot, y_pred=y_pred_train, reference=reference_scale)
                metrics_train = metric_calc_train.calculate_predictive_metrics(n_params)
                train_metrics.append(metrics_train)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if not oob_metrics:
            return None

        return self._calculate_632_plus(train_metrics, oob_metrics, X, y, n_params)

    def _evaluate_standard(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str,
                           evaluation_method: str,
                           n_folds: int,
                           n_repeats: int,
                           n_bootstraps: int,
                           reference_scale:float|None) -> Optional[Dict]:
        """Standard evaluation for models with fixed structure."""
        if evaluation_method == 'standard-cv':
            return self.evaluator.standard_cv(X, y, model_type, n_folds=n_folds, reference_scale=reference_scale)
        elif evaluation_method == 'repeated-cv':
            return self.evaluator.repeated_cv(X, y, model_type, n_folds=n_folds, n_repeats=n_repeats, reference_scale=reference_scale)
        elif evaluation_method == 'bootstrap':
            return self.evaluator.bootstrap_632_plus(X, y, model_type, n_bootstraps=n_bootstraps, reference_scale=reference_scale)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _average_cv_metrics(self, cv_metrics: List[Dict]) -> Dict:
        """Average metrics across CV folds."""
        avg_metrics = {}
        metric_names = ['q2',  'rrmse', 'mape', 'mae', 'rmse']

        for metric in metric_names:
            values = [m[metric] for m in cv_metrics if not np.isnan(m.get(metric, np.nan))]

            if values:
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            else:
                avg_metrics[metric] = np.nan
                avg_metrics[f'{metric}_std'] = np.nan

        avg_metrics['n_evaluations'] = len(cv_metrics)
        return avg_metrics

    def _calculate_632_plus(self,
                            train_metrics: List[Dict],
                            oob_metrics: List[Dict],
                            X: np.ndarray,
                            y: np.ndarray,
                            n_params: int) -> Dict:
        """Calculate .632+ bootstrap correction."""
        train_avg = self._average_cv_metrics(train_metrics)
        oob_avg = self._average_cv_metrics(oob_metrics)

        y_perm = np.random.permutation(y)
        no_info_calc = RegressionMetrics(y, y_perm)
        no_info_metrics = no_info_calc.calculate_cv_metrics()
        no_info_mse = no_info_metrics['rmse'] ** 2

        oob_mse = oob_avg['rmse'] ** 2
        train_mse = train_avg['rmse'] ** 2

        R = (oob_mse - train_mse) / (no_info_mse - train_mse) if no_info_mse != train_mse else 0
        R = np.clip(R, 0, 1)

        w = 0.632 / (1 - 0.368 * R)

        corrected = {}
        for metric in ['q2', 'rrmse', 'mape', 'mae', 'rmse']:
            train_val = train_avg.get(metric, np.nan)
            oob_val = oob_avg.get(metric, np.nan)

            if not np.isnan(train_val) and not np.isnan(oob_val):
                corrected[f'{metric}_632plus'] = w * oob_val + (1 - w) * train_val
                corrected[f'{metric}_train'] = train_val
                corrected[f'{metric}_oob'] = oob_val
                corrected[f'{metric}_std'] = oob_avg.get(f'{metric}_std', 0)
            else:
                corrected[f'{metric}_632plus'] = np.nan
                corrected[f'{metric}_train'] = np.nan
                corrected[f'{metric}_oob'] = np.nan

        corrected['relative_overfitting'] = R
        corrected['bootstrap_weight'] = w
        corrected['n_evaluations'] = len(oob_metrics)
        corrected['gamma_rmse'] = np.sqrt(no_info_mse)
        corrected['n_samples'] = len(X)

        return corrected

    def _print_results_table(self, results_df: pd.DataFrame, evaluation_method: str):
        """Print formatted results table with method-specific information."""
        print("\n" + "=" * 160)
        print("RESULTS: Model Performance Across Datasets")

        if evaluation_method == 'repeated-cv':
            n_folds = results_df['conv_n_evaluations'].iloc[0] // 10
            n_repeats = 10
            print(f"All datasets evaluated using {n_folds}-fold CV × {n_repeats} repeats")
        elif evaluation_method == 'bootstrap':
            n_bootstraps = results_df['conv_n_evaluations'].iloc[0]
            print(f"All datasets evaluated using Bootstrap .632+ ({n_bootstraps} samples)")
        else:
            n_folds = results_df['conv_n_evaluations'].iloc[0]
            print(f"All datasets evaluated using {n_folds}-fold CV")

        print("=" * 160)

        high_variance_targets = results_df[results_df['alt_q2_std'] > 0.3]['target'].tolist()
        if high_variance_targets:
            print(f"\n⚠️  High variance on alternative dataset for: {', '.join(high_variance_targets)}")
            print("      This is expected given small sample size (n≈18)")

            if evaluation_method == 'bootstrap':
                print("      Bootstrap .632+ provides bias-corrected estimates despite high variance")
            print()

        print(f"\n{'Target':<15s} | {'Predictors':<25s} | {'Model':<15s} | "
              # f"{'Conv R²':>15s} | {'Matched R²':>15s} | {'Alt R²':>15s}")
         f"{'Conv R²':>15s} | {'Alt R²':>15s}")

        if evaluation_method == 'bootstrap':
            print(f"{'':15s} | {'':25s} | {'':15s} | "
                  f"{'(train→OOB→.632+)':>15s} | {'(train→OOB→.632+)':>15s} | "
                  f"{'(train→OOB→.632+)':>15s}")

        print("-" * 160)

        for _, row in results_df.iterrows():
            conv_str = f"{row['conv_q2']:>6.3f}±{row['conv_q2_std']:>5.3f}"
            # matched_str = f"{row['matched_q2']:>6.3f}±{row['matched_q2_std']:>5.3f}"

            if row['alt_q2_std'] > 0.3:
                alt_str = f"{row['alt_q2']:>6.3f}±{row['alt_q2_std']:>5.3f}*"
            else:
                alt_str = f"{row['alt_q2']:>6.3f}±{row['alt_q2_std']:>5.3f} "

            print(f"{row['target']:<15s} | "
                  f"{row['predictor_str']:<25s} | "
                  f"{row['model_type']:<15s} | "
                  f"{conv_str:>15s} | "
                  # f"{matched_str:>15s} | "
                  f"{alt_str:>15s}")

            if evaluation_method == 'bootstrap':
                conv_decomp = f"({row.get('conv_q2_train', 0):.2f}→{row.get('conv_rq_oob', 0):.2f}→{row['conv_q2']:.2f})"
                # matched_decomp = f"({row.get('matched_q2_train', 0):.2f}→{row.get('matched_q2_oob', 0):.2f}→{row['matched_q2']:.2f})"
                alt_decomp = f"({row.get('alt_q2_train', 0):.2f}→{row.get('alt_q2_oob', 0):.2f}→{row['alt_q2']:.2f})"
                alt_overfit = row.get('alt_q2_relative_overfitting', 0)
                overfit_marker = "⚠️" if alt_overfit > 0.5 else ""

                print(f"{'':15s} | {'':25s} | {'':15s} | "
                      f"{conv_decomp:>15s} | "
                      # f"{matched_decomp:>15s} | "
                      f"{alt_decomp:>15s} {overfit_marker}")

            print("=" * 160)
            print("* High variance (std > 0.3) - interpret with caution due to small sample")

            if evaluation_method == 'bootstrap':
                print("⚠️  = Severe overfitting (R > 0.5) - model unstable on small dataset")

            # Summary statistics
            print("\nSUMMARY:")
            print(f"  Dataset sizes (mean):")
            print(f"    Conventional:  n={results_df['conv_n'].iloc[0]:.0f}")
            # print(f"    Matched:       n={results_df['matched_n'].mean():.0f}")
            print(f"    Alternative:   n={results_df['alt_n'].mean():.0f}")

            print(f"\n  Mean performance:")
            print(f"    Conventional R²:  {results_df['conv_q2'].mean():.3f} "
                  f"(mean std: {results_df['conv_q2_std'].mean():.3f})")
            # print(f"    Matched R²:       {results_df['matched_q2'].mean():.3f} "
            #       f"(mean std: {results_df['matched_q2_std'].mean():.3f})")
            print(f"    Alternative R²:   {results_df['alt_q2'].mean():.3f} "
                  f"(mean std: {results_df['alt_q2_std'].mean():.3f})")

            if evaluation_method == 'bootstrap':
                print(f"\n  Overfitting analysis (alternative dataset):")
                avg_overfit = results_df['alt_q2_relative_overfitting'].mean()
                severe_overfit = (results_df['alt_q2_relative_overfitting'] > 0.5).sum()

                print(f"    Mean relative overfitting (R): {avg_overfit:.2f}")
                print(f"    Targets with severe overfitting (R>0.5): {severe_overfit}/{len(results_df)}")

                if avg_overfit > 0.3:
                    print(f"\n  ⚠️  High average overfitting indicates models are too complex for n=18")
                    print(f"     Bootstrap .632+ correction helps, but simpler models might be better")

            if 'conv_n_evaluations' in results_df.columns:
                print(f"\n  Evaluation completeness:")
                print(f"    Conventional:  {results_df['conv_n_evaluations'].mean():.0f} evaluations")
                # print(f"    Matched:       {results_df['matched_n_evaluations'].mean():.0f} evaluations")
                print(f"    Alternative:   {results_df['alt_n_evaluations'].mean():.0f} evaluations")

            print("=" * 160)

    # ========================================================================
    # BEST COMBINATION SELECTION
    # ========================================================================

    def _get_best_combination_for_target(self,
                                         target: str,
                                         max_predictors: Optional[int] = None,
                                         require_alternative_data: bool = False,
                                         min_alternative_samples: int = 5
                                         ) -> Tuple[Optional[Tuple], Optional[str]]:
        """
        Get best predictor combination for target.

        **CrossDatasetPerformance default:** require_alternative_data=False
        We want to test performance even on small datasets - that's the point!

        Parameters
        ----------
        target : str
            Target variable
        max_predictors : int, optional
            Maximum number of predictors
        require_alternative_data : bool, default=False
            Whether to require sufficient alternative data
        min_alternative_samples : int, default=5
            Minimum samples in alternative dataset
        """
        result = self.foundation.get_best_combination_for_target(
            target=target,
            max_predictors=max_predictors,
            filter_by_vif=True,
            vif_threshold=10.0,
            require_alternative_data=require_alternative_data,
            min_alternative_samples=min_alternative_samples
        )

        return result if result else (None, None)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _dataframe_to_dict(self, results_df: pd.DataFrame) -> Dict:
        """Convert DataFrame to nested dict."""
        results_dict = {}

        for _, row in results_df.iterrows():
            target = row['target']
            if target not in results_dict:
                results_dict[target] = {}

            predictor_str = row['predictor_str']
            results_dict[target][predictor_str] = row.to_dict()

        return results_dict

    def export_results(self, filename: str = 'cross_dataset_performance.csv'):
        """Export results to CSV."""
        if self.results_df is None:
            raise ValueError("No results to export. Run analyze_best_combinations() first.")

        self.results_df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
        print(f"  {len(self.results_df)} rows, {len(self.results_df.columns)} columns")