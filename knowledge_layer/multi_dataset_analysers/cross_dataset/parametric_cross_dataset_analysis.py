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
                 independent_variables: List[str],
                 dependent_variables: Optional[List[str]] = None,
                 parametric_selector: 'ParametricModelSelector' = None,
                 model_recommendations: Optional[Dict] = None,
                 detailed_results: Optional[Dict] = None,
                 selection_metric: str = 'adj_r2',
                 selection_strategy: str = 'simple',
                 random_state: int = 42,
                 verbose: bool = True):
        """Initialize analyzer with foundation and evaluator."""

        # Extract from selector if provided
        if parametric_selector is not None:
            model_recommendations = parametric_selector.model_recommendations
            detailed_results = parametric_selector.detailed_results
            selection_metric = parametric_selector.selection_metric
            selection_strategy = parametric_selector.model_selector.selection_strategy
            self.model_descriptors = parametric_selector.model_descriptors
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
            random_state=random_state,
            verbose=verbose
        )

        # Evaluator for multiple evaluation strategies
        self.evaluator = SmallDatasetEvaluator(
            random_state=random_state,
            verbose=verbose
        )

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
                n_bootstraps=n_bootstraps
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
            'conv_r2', 'conv_r2_std',
            'matched_r2', 'matched_r2_std',
            'alt_r2', 'alt_r2_std',
            'conv_rrmse', 'conv_rrmse_std',
            'matched_rrmse', 'matched_rrmse_std',
            'alt_rrmse', 'alt_rrmse_std',
        ]

        if evaluation_method == 'bootstrap':
            bootstrap_cols = [
                'conv_r2_oob', 'conv_r2_train', 'conv_r2_relative_overfitting',
                'matched_r2_oob', 'matched_r2_train', 'matched_r2_relative_overfitting',
                'alt_r2_oob', 'alt_r2_train', 'alt_r2_relative_overfitting',
            ]
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
                                             n_bootstraps: int) -> Optional[Dict]:
        """Evaluate on three datasets using selected method."""
        # Get matched dataset
        predictor_key = tuple(predictors)
        matched_dataset = self.foundation.get_matched_dataset(predictor_key)

        if matched_dataset is None:
            if self.verbose:
                print(f"    ⚠️  No matched dataset for {predictors}")
            return None

        # Prepare datasets
        datasets = {
            'conv': self.conventional_full,
            'matched': matched_dataset,
            'alt': self.alternative
        }

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
                    results[f'{dataset_name}_r2_{metric}'] = value
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
                             n_bootstraps: int) -> Optional[Dict]:
        """Evaluate using selected evaluation strategy."""
        # Prepare data
        data_prep = self.foundation.prepare_dataset(dataset, predictors, target)

        if data_prep is None:
            if self.verbose:
                print(f"    ⚠️  {dataset_name}: Insufficient data")
            return None

        X, y = data_prep
        n_samples = len(X)

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
                n_bootstraps=n_bootstraps
            )
        else:
            result = self._evaluate_standard(
                X, y, model_type,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps
            )

        if result is None:
            return None

        # Add sample size
        result['n'] = n_samples

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
            r2_val = result.get('r2', np.nan)
            r2_std = result.get('r2_std', 0)
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
                                       n_bootstraps: int) -> Optional[Dict]:
        """Evaluate structure-discovering model with FIXED structure."""
        if evaluation_method == 'standard-cv':
            return self._fixed_structure_standard_cv(X, y, descriptor, n_folds)
        elif evaluation_method == 'repeated-cv':
            return self._fixed_structure_repeated_cv(X, y, descriptor, n_folds, n_repeats)
        elif evaluation_method == 'bootstrap':
            return self._fixed_structure_bootstrap(X, y, descriptor, n_bootstraps)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")

    def _fixed_structure_standard_cv(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     descriptor: 'ModelDescriptor',
                                     n_folds: int) -> Optional[Dict]:
        """Standard k-fold CV with fixed structure."""
        from sklearn.model_selection import KFold

        cv_metrics = []
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model, prep = descriptor.refit(X=X_train, y=y_train, force_structure=True)
                y_pred = ModelFactory.predict(model, prep, X_test, descriptor.model_type)

                metric_calc = RegressionMetrics(y_test, y_pred)
                n_params = descriptor.get_effective_parameters()
                metrics = metric_calc.calculate_predictive_metrics(n_params)

                cv_metrics.append(metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"      Fold failed: {e}")
                continue

        if not cv_metrics:
            return None

        return self._average_cv_metrics(cv_metrics)

    def _fixed_structure_repeated_cv(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     descriptor: 'ModelDescriptor',
                                     n_folds: int,
                                     n_repeats: int) -> Optional[Dict]:
        """Repeated k-fold CV with fixed structure."""
        from sklearn.model_selection import RepeatedKFold

        cv_metrics = []
        rkfold = RepeatedKFold(
            n_splits=n_folds,
            n_repeats=n_repeats,
            random_state=self.random_state
        )

        for train_idx, test_idx in rkfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model, prep = descriptor.refit(X_train, y_train, force_structure=True)
                y_pred = ModelFactory.predict(model, prep, X_test, descriptor.model_type)

                metric_calc = RegressionMetrics(y_test, y_pred)
                n_params = descriptor.get_effective_parameters()
                metrics = metric_calc.calculate_predictive_metrics(n_params)

                cv_metrics.append(metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if not cv_metrics:
            return None

        avg_metrics = self._average_cv_metrics(cv_metrics)
        avg_metrics['n_evaluations'] = len(cv_metrics)

        return avg_metrics

    def _fixed_structure_bootstrap(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   descriptor: 'ModelDescriptor',
                                   n_bootstraps: int) -> Optional[Dict]:
        """Bootstrap .632+ evaluation with fixed structure."""
        n_samples = len(X)
        n_params = descriptor.get_effective_parameters()

        oob_metrics = []
        train_metrics = []

        for i in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.array([i for i in range(n_samples) if i not in boot_idx])

            if len(oob_idx) < 5:
                continue

            X_boot, y_boot = X[boot_idx], y[boot_idx]
            X_oob, y_oob = X[oob_idx], y[oob_idx]

            try:
                model, prep = descriptor.refit(X_boot, y_boot, force_structure=True)

                y_pred_oob = ModelFactory.predict(model, prep, X_oob, descriptor.model_type)
                metric_calc_oob = RegressionMetrics(y_oob, y_pred_oob)
                metrics_oob = metric_calc_oob.calculate_predictive_metrics(n_params)
                oob_metrics.append(metrics_oob)

                y_pred_train = ModelFactory.predict(model, prep, X_boot, descriptor.model_type)
                metric_calc_train = RegressionMetrics(y_boot, y_pred_train)
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
                           n_bootstraps: int) -> Optional[Dict]:
        """Standard evaluation for models with fixed structure."""
        if evaluation_method == 'standard-cv':
            return self.evaluator.standard_cv(X, y, model_type, n_folds=n_folds)
        elif evaluation_method == 'repeated-cv':
            return self.evaluator.repeated_cv(X, y, model_type, n_folds=n_folds, n_repeats=n_repeats)
        elif evaluation_method == 'bootstrap':
            return self.evaluator.bootstrap_632_plus(X, y, model_type, n_bootstraps=n_bootstraps)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _average_cv_metrics(self, cv_metrics: List[Dict]) -> Dict:
        """Average metrics across CV folds."""
        avg_metrics = {}
        metric_names = ['r2', 'adj_r2', 'rrmse', 'mape', 'mae', 'rmse']

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
        no_info_metrics = no_info_calc.calculate_all_metrics(n_params)
        no_info_mse = no_info_metrics['rmse'] ** 2

        oob_mse = oob_avg['rmse'] ** 2
        train_mse = train_avg['rmse'] ** 2

        R = (oob_mse - train_mse) / (no_info_mse - train_mse) if no_info_mse != train_mse else 0
        R = np.clip(R, 0, 1)

        w = 0.632 / (1 - 0.368 * R)

        corrected = {}
        for metric in ['r2', 'adj_r2', 'rrmse', 'mape', 'mae', 'rmse']:
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

        high_variance_targets = results_df[results_df['alt_r2_std'] > 0.3]['target'].tolist()
        if high_variance_targets:
            print(f"\n⚠️  High variance on alternative dataset for: {', '.join(high_variance_targets)}")
            print("      This is expected given small sample size (n≈18)")

            if evaluation_method == 'bootstrap':
                print("      Bootstrap .632+ provides bias-corrected estimates despite high variance")
            print()

        print(f"\n{'Target':<15s} | {'Predictors':<25s} | {'Model':<15s} | "
              f"{'Conv R²':>15s} | {'Matched R²':>15s} | {'Alt R²':>15s}")

        if evaluation_method == 'bootstrap':
            print(f"{'':15s} | {'':25s} | {'':15s} | "
                  f"{'(train→OOB→.632+)':>15s} | {'(train→OOB→.632+)':>15s} | "
                  f"{'(train→OOB→.632+)':>15s}")

        print("-" * 160)

        for _, row in results_df.iterrows():
            conv_str = f"{row['conv_r2']:>6.3f}±{row['conv_r2_std']:>5.3f}"
            matched_str = f"{row['matched_r2']:>6.3f}±{row['matched_r2_std']:>5.3f}"

            if row['alt_r2_std'] > 0.3:
                alt_str = f"{row['alt_r2']:>6.3f}±{row['alt_r2_std']:>5.3f}*"
            else:
                alt_str = f"{row['alt_r2']:>6.3f}±{row['alt_r2_std']:>5.3f} "

            print(f"{row['target']:<15s} | "
                  f"{row['predictor_str']:<25s} | "
                  f"{row['model_type']:<15s} | "
                  f"{conv_str:>15s} | "
                  f"{matched_str:>15s} | "
                  f"{alt_str:>15s}")

            if evaluation_method == 'bootstrap':
                conv_decomp = f"({row.get('conv_r2_train', 0):.2f}→{row.get('conv_r2_oob', 0):.2f}→{row['conv_r2']:.2f})"
                matched_decomp = f"({row.get('matched_r2_train', 0):.2f}→{row.get('matched_r2_oob', 0):.2f}→{row['matched_r2']:.2f})"
                alt_decomp = f"({row.get('alt_r2_train', 0):.2f}→{row.get('alt_r2_oob', 0):.2f}→{row['alt_r2']:.2f})"
                alt_overfit = row.get('alt_r2_relative_overfitting', 0)
                overfit_marker = "⚠️" if alt_overfit > 0.5 else ""

                print(f"{'':15s} | {'':25s} | {'':15s} | "
                      f"{conv_decomp:>15s} | "
                      f"{matched_decomp:>15s} | "
                      f"{alt_decomp:>15s} {overfit_marker}")

            print("=" * 160)
            print("* High variance (std > 0.3) - interpret with caution due to small sample")

            if evaluation_method == 'bootstrap':
                print("⚠️  = Severe overfitting (R > 0.5) - model unstable on small dataset")

            # Summary statistics
            print("\nSUMMARY:")
            print(f"  Dataset sizes (mean):")
            print(f"    Conventional:  n={results_df['conv_n'].iloc[0]:.0f}")
            print(f"    Matched:       n={results_df['matched_n'].mean():.0f}")
            print(f"    Alternative:   n={results_df['alt_n'].mean():.0f}")

            print(f"\n  Mean performance:")
            print(f"    Conventional R²:  {results_df['conv_r2'].mean():.3f} "
                  f"(mean std: {results_df['conv_r2_std'].mean():.3f})")
            print(f"    Matched R²:       {results_df['matched_r2'].mean():.3f} "
                  f"(mean std: {results_df['matched_r2_std'].mean():.3f})")
            print(f"    Alternative R²:   {results_df['alt_r2'].mean():.3f} "
                  f"(mean std: {results_df['alt_r2_std'].mean():.3f})")

            if evaluation_method == 'bootstrap':
                print(f"\n  Overfitting analysis (alternative dataset):")
                avg_overfit = results_df['alt_r2_relative_overfitting'].mean()
                severe_overfit = (results_df['alt_r2_relative_overfitting'] > 0.5).sum()

                print(f"    Mean relative overfitting (R): {avg_overfit:.2f}")
                print(f"    Targets with severe overfitting (R>0.5): {severe_overfit}/{len(results_df)}")

                if avg_overfit > 0.3:
                    print(f"\n  ⚠️  High average overfitting indicates models are too complex for n=18")
                    print(f"     Bootstrap .632+ correction helps, but simpler models might be better")

            if 'conv_n_evaluations' in results_df.columns:
                print(f"\n  Evaluation completeness:")
                print(f"    Conventional:  {results_df['conv_n_evaluations'].mean():.0f} evaluations")
                print(f"    Matched:       {results_df['matched_n_evaluations'].mean():.0f} evaluations")
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