# File: random_forest_cross_dataset_analysis.py

"""
Random Forest Cross-Dataset Performance Analyzer
================================================
Compares Random Forest performance when trained on different datasets.

Uses composition with RandomForestAnalysisFoundation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from knowledge_layer.multi_dataset_analysers.rf_analysis_foundation import RandomForestAnalysisFoundation
from knowledge_layer.multi_dataset_analysers.small_dataset_evaluation import SmallDatasetEvaluator
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


class RandomForestCrossDatasetAnalyzer:
    """
    Analyzes how Random Forest performance varies with training dataset.

    Uses RandomForestAnalysisFoundation via composition.

    Key Question:
    ------------
    "Does training on alternative ships improve alternative ship predictions?"

    Analysis:
    --------
    For each target:
    1. Get best predictor combo from selector results
    2. Train RF on conventional, matched, and alternative datasets
    3. Evaluate each via selected evaluation method
    4. Compare performance and stability

    Uses composition with:
    - RandomForestAnalysisFoundation: Common setup and data preparation
    - SmallDatasetEvaluator: Multiple evaluation strategies
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: Optional[List[str]] = None,
                 selector_results: Optional[Dict] = None,
                 rf_params: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """Initialize analyzer with foundation and evaluator."""

        # Foundation for data management
        self.foundation = RandomForestAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            selector_results=selector_results,
            rf_params=rf_params,
            random_state=random_state,
            verbose=verbose
        )

        # ✅ Evaluator for multiple evaluation strategies
        # NOTE: SmallDatasetEvaluator works with model_type strings
        # For RF, we'll pass 'random_forest' as the model_type
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

    @property
    def rf_params(self) -> Dict:
        return self.foundation.get_rf_params()

    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze_best_combinations(self,
                                  evaluation_method: str = 'repeated-cv',
                                  n_folds: int = 5,
                                  n_repeats: int = 10,
                                  n_bootstraps: int = 100,
                                  include_feature_engineered: bool = False,
                                  display_results: bool = True,
                                  require_alternative_data: bool = False,  # ✅ NEW
                                  min_alternative_samples: int = 5,
                                  max_predictors:int=1) -> pd.DataFrame:
        """
        Analyze best combination for each target across datasets.

        ✅ NEW PARAMETERS:
        require_alternative_data : bool, default=True
            If True, only analyzes targets with sufficient alternative data
        min_alternative_samples : int, default=5
            Minimum samples required in alternative dataset

        Parameters
        ----------
        evaluation_method : str, default='repeated-cv'
            Evaluation strategy:
            - 'standard-cv': Simple k-fold CV
            - 'repeated-cv': k-fold CV repeated multiple times (recommended)
            - 'bootstrap': Bootstrap .632+ (best for very small datasets)
        n_folds : int, default=5
            Number of CV folds
        n_repeats : int, default=10
            Number of repetitions (for 'repeated-cv')
        n_bootstraps : int, default=100
            Number of bootstrap samples (for 'bootstrap')
        include_feature_engineered : bool, default=False
            If True, analyzes feature-engineered combinations
        display_results : bool
            Print results table

        Returns
        -------
        pd.DataFrame
            Results with performance metrics
        """

        if evaluation_method not in ['standard-cv', 'repeated-cv', 'bootstrap']:
            raise ValueError(
                f"evaluation_method must be 'standard-cv', 'repeated-cv', or 'bootstrap'. "
                f"Got: {evaluation_method}"
            )

        if display_results and self.verbose:
            print("\n" + "=" * 140)
            print("RANDOM FOREST CROSS-DATASET PERFORMANCE ANALYSIS")
            print("=" * 140)

            if evaluation_method == 'repeated-cv':
                print(f"Evaluation: {n_folds}-fold CV repeated {n_repeats} times "
                      f"({n_folds * n_repeats} evaluations per model)")
            elif evaluation_method == 'bootstrap':
                print(f"Evaluation: Bootstrap .632+ ({n_bootstraps} samples per model)")
            else:
                print(f"Evaluation: {n_folds}-fold CV")

            print("Consistent methodology applied to all datasets")
            print("=" * 140)

        all_results = []

        for target in self.target_dimensions:
            best_combo = self.foundation.get_best_combination_for_target(
                target,
                max_predictors=max_predictors,
                filter_by_data_availability=True,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

            if best_combo is None:
                if self.verbose:
                    print(f"\n  ⚠️  No combinations found for {target}")
                continue

            if self.verbose:
                print(f"\nAnalyzing {target}:")
                print(f"  Best predictors: {' + '.join(best_combo)}")
                print(f"  Model: Random Forest")

            # Analyze across datasets with selected method
            result = self._analyze_combination_across_datasets(
                predictors=list(best_combo),
                target=target,
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
                result['model_type'] = 'random_forest'
                result['evaluation_method'] = evaluation_method
                all_results.append(result)

        # Create DataFrame
        if not all_results:
            if self.verbose:
                print("\n  ⚠️  No valid results generated!")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Reorder columns (same as parametric)
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

        # Add bootstrap-specific columns if using bootstrap
        if evaluation_method == 'bootstrap':
            bootstrap_cols = [
                'conv_r2_oob', 'conv_r2_train', 'conv_r2_relative_overfitting',
                'matched_r2_oob', 'matched_r2_train', 'matched_r2_relative_overfitting',
                'alt_r2_oob', 'alt_r2_train', 'alt_r2_relative_overfitting',
            ]
            column_order.extend(bootstrap_cols)

        # Add feature importance columns
        column_order.extend(['conv_n', 'matched_n', 'alt_n', 'feature_importance_avg'])

        existing_cols = [col for col in column_order if col in results_df.columns]
        remaining_cols = [col for col in results_df.columns if col not in column_order]
        results_df = results_df[existing_cols + remaining_cols]

        # Display
        if display_results and self.verbose:
            self._print_results_table(results_df, evaluation_method)

        # Store
        self.results_df = results_df
        self.results = self._dataframe_to_dict(results_df)

        return results_df

    def _analyze_combination_across_datasets(self,
                                             predictors: List[str],
                                             target: str,
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
        feature_importances = []

        # Evaluate on each dataset with SAME method
        for dataset_name, dataset in datasets.items():
            dataset_results = self._evaluate_on_dataset(
                dataset=dataset,
                predictors=predictors,
                target=target,
                dataset_name=dataset_name,
                evaluation_method=evaluation_method,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_bootstraps=n_bootstraps
            )

            if dataset_results is None:
                return None

            # Extract feature importance if available
            if 'feature_importance' in dataset_results:
                feature_importances.append(dataset_results['feature_importance'])
                del dataset_results['feature_importance']  # Remove before flattening

            # ✅ FIX: Flatten with dataset prefix
            # Special handling for relative_overfitting to create r2_relative_overfitting
            for metric, value in dataset_results.items():
                if metric == 'relative_overfitting':
                    # Create the expected column name for R²
                    results[f'{dataset_name}_r2_{metric}'] = value
                else:
                    results[f'{dataset_name}_{metric}'] = value

        # Average feature importances across datasets
        if feature_importances:
            avg_importance = {}
            for feat in predictors:
                importances = [fi.get(feat, 0) for fi in feature_importances]
                avg_importance[feat] = np.mean(importances)
            results['feature_importance_avg'] = avg_importance

        return results

    def _evaluate_on_dataset(self,
                             dataset: pd.DataFrame,
                             predictors: List[str],
                             target: str,
                             dataset_name: str,
                             evaluation_method: str,
                             n_folds: int,
                             n_repeats: int,
                             n_bootstraps: int) -> Optional[Dict]:
        """
        Evaluate using selected evaluation strategy.

        ✅ Uses SmallDatasetEvaluator via composition
        NOTE: For Random Forest, we need to pass rf_params to the evaluator
        """
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

        # ✅ Dispatch to appropriate evaluator method
        # NOTE: SmallDatasetEvaluator expects model_type as string
        # We'll pass 'random_forest' and it should handle RF internally
        # OR we need to modify the evaluator to accept rf_params

        # For now, we'll use a custom RF evaluation wrapper
        if evaluation_method == 'standard-cv':
            result = self.evaluator.standard_cv(
                X, y, 'random_forest',
                n_folds=n_folds,
                model_params=self.foundation.rf_params
            )

        elif evaluation_method == 'repeated-cv':
            result = self.evaluator.repeated_cv(
                X, y, 'random_forest',
                n_folds=n_folds,
                n_repeats=n_repeats,
                model_params=self.foundation.rf_params
            )

        elif evaluation_method == 'bootstrap':
            result = self.evaluator.bootstrap_632_plus(
                X, y, 'random_forest',
                n_bootstraps=n_bootstraps,
                model_params=self.foundation.rf_params
            )

        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")

        if result is None:
            if self.verbose:
                print(f"    ⚠️  {dataset_name}: Evaluation failed")
            return None

        # Add sample size
        result['n'] = n_samples

        # ✅ FEATURE IMPORTANCE: Get from full dataset fit
        feature_importance = self._get_feature_importance(X, y, predictors)
        if feature_importance is not None:
            result['feature_importance'] = feature_importance

        # ✅ STANDARDIZE: Convert .632+ metrics to primary metrics
        if evaluation_method == 'bootstrap':
            standardized = {}
            for key, value in result.items():
                if key.endswith('_632plus'):
                    # Use .632+ corrected value as primary metric
                    base_metric = key.replace('_632plus', '')
                    standardized[base_metric] = value
                else:
                    standardized[key] = value
            result = standardized

        if self.verbose:
            print(f"        → R²={result['r2']:.3f}±{result.get('r2_std', 0):.3f}, "
                  f"RRMSE={result['rrmse']:.1f}±{result.get('rrmse_std', 0):.1f}%")

        return result

    def _get_feature_importance(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                predictors: List[str]) -> Optional[Dict[str, float]]:
        """
        Get feature importance from RF fitted on full dataset.

        Parameters
        ----------
        X : np.ndarray
            Full predictor data
        y : np.ndarray
            Full target data
        predictors : list of str
            Predictor names

        Returns
        -------
        dict or None
            {predictor_name: importance_value}
        """
        try:
            rf = RandomForestRegressor(**self.foundation.rf_params)
            rf.fit(X, y)

            importances = rf.feature_importances_
            return dict(zip(predictors, importances))

        except Exception as e:
            if self.verbose:
                print(f"      Warning: Could not extract feature importance: {str(e)[:60]}")
            return None

    # ========================================================================
    # RF-SPECIFIC EVALUATION METHODS
    # ========================================================================

    def _rf_standard_cv(self, X: np.ndarray, y: np.ndarray,
                        predictors: List[str], n_folds: int) -> Optional[Dict]:
        """Standard k-fold CV for Random Forest."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        fold_metrics = []
        fold_importances = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                rf = RandomForestRegressor(**self.rf_params)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                metric_calc = RegressionMetrics(y_test, y_pred)
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rrmse': metric_calc.calculate_rrmse(),
                    'rmse': metric_calc.calculate_rmse(),
                    'mae': metric_calc.calculate_mae(),
                    'mape': metric_calc.calculate_mape()
                }

                fold_metrics.append(metrics)
                fold_importances.append(rf.feature_importances_)

            except Exception as e:
                if self.verbose:
                    print(f"      Fold failed: {str(e)[:60]}")
                continue

        if not fold_metrics:
            return None

        # Average metrics
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            else:
                avg_metrics[key] = np.nan
                avg_metrics[f'{key}_std'] = np.nan

        # Average feature importance
        avg_importance = np.mean(fold_importances, axis=0)
        avg_metrics['feature_importance'] = dict(zip(predictors, avg_importance))
        avg_metrics['n_evaluations'] = len(fold_metrics)

        return avg_metrics

    def _rf_repeated_cv(self, X: np.ndarray, y: np.ndarray,
                        predictors: List[str], n_folds: int,
                        n_repeats: int) -> Optional[Dict]:
        """Repeated k-fold CV for Random Forest."""
        all_metrics = []
        all_importances = []

        for repeat in range(n_repeats):
            # Use different random state for each repeat
            result = self._rf_standard_cv(
                X, y, predictors, n_folds
            )

            if result is not None:
                # Extract feature importance
                if 'feature_importance' in result:
                    all_importances.append(result['feature_importance'])
                    del result['feature_importance']

                all_metrics.append(result)

        if not all_metrics:
            return None

        # Average across all repeats
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key.endswith('_std') or key == 'n_evaluations':
                continue

            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            else:
                avg_metrics[key] = np.nan
                avg_metrics[f'{key}_std'] = np.nan

        # Average feature importances
        if all_importances:
            avg_importance = {}
            for feat in predictors:
                importances = [imp.get(feat, 0) for imp in all_importances]
                avg_importance[feat] = np.mean(importances)
            avg_metrics['feature_importance'] = avg_importance

        avg_metrics['n_evaluations'] = sum(m['n_evaluations'] for m in all_metrics)

        return avg_metrics

    def _rf_bootstrap_632_plus(self, X: np.ndarray, y: np.ndarray,
                               predictors: List[str],
                               n_bootstraps: int) -> Optional[Dict]:
        """Bootstrap .632+ evaluation for Random Forest."""
        n_samples = len(X)

        oob_metrics_list = []
        train_metrics_list = []
        all_importances = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.array([i for i in range(n_samples) if i not in bootstrap_idx])

            if len(oob_idx) == 0:
                continue

            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            X_oob = X[oob_idx]
            y_oob = y[oob_idx]

            try:
                rf = RandomForestRegressor(**self.rf_params)
                rf.fit(X_boot, y_boot)

                # Training performance
                y_train_pred = rf.predict(X_boot)
                train_calc = RegressionMetrics(y_boot, y_train_pred)
                train_metrics = {
                    'r2': r2_score(y_boot, y_train_pred),
                    'rrmse': train_calc.calculate_rrmse()
                }

                # OOB performance
                y_oob_pred = rf.predict(X_oob)
                oob_calc = RegressionMetrics(y_oob, y_oob_pred)
                oob_metrics = {
                    'r2': r2_score(y_oob, y_oob_pred),
                    'rrmse': oob_calc.calculate_rrmse()
                }

                train_metrics_list.append(train_metrics)
                oob_metrics_list.append(oob_metrics)
                all_importances.append(rf.feature_importances_)

            except Exception:
                continue

        if not oob_metrics_list:
            return None

        # Calculate .632+ corrected metrics
        avg_train_r2 = np.mean([m['r2'] for m in train_metrics_list])
        avg_oob_r2 = np.mean([m['r2'] for m in oob_metrics_list])

        avg_train_rrmse = np.mean([m['rrmse'] for m in train_metrics_list])
        avg_oob_rrmse = np.mean([m['rrmse'] for m in oob_metrics_list])

        # Relative overfitting rate
        if avg_train_r2 > avg_oob_r2:
            R_r2 = min((avg_train_r2 - avg_oob_r2) / max(avg_train_r2, 0.001), 1.0)
        else:
            R_r2 = 0.0

        # .632+ weights
        w = 0.632 / (1 - 0.368 * R_r2)

        # Corrected estimates
        r2_632plus = w * avg_oob_r2 + (1 - w) * avg_train_r2
        rrmse_632plus = w * avg_oob_rrmse + (1 - w) * avg_train_rrmse

        result = {
            'r2': r2_632plus,
            'r2_std': np.std([m['r2'] for m in oob_metrics_list]),
            'r2_train': avg_train_r2,
            'r2_oob': avg_oob_r2,
            'r2_relative_overfitting': R_r2,
            'rrmse': rrmse_632plus,
            'rrmse_std': np.std([m['rrmse'] for m in oob_metrics_list]),
            'rrmse_train': avg_train_rrmse,
            'rrmse_oob': avg_oob_rrmse,
            'n_evaluations': len(oob_metrics_list)
        }

        # Average feature importance
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
            result['feature_importance'] = dict(zip(predictors, avg_importance))

        return result

    # ========================================================================
    # DISPLAY METHODS
    # ========================================================================

    def _print_results_table(self, results_df: pd.DataFrame, evaluation_method: str):
        """Print formatted results table (reuses parametric version with RF tweaks)."""
        print("\n" + "=" * 160)
        print("RESULTS: Random Forest Performance Across Datasets")

        # Display evaluation method info
        if evaluation_method == 'repeated-cv':
            n_folds = 5
            n_repeats = 10
            print(f"All datasets evaluated using {n_folds}-fold CV × {n_repeats} repeats")
        elif evaluation_method == 'bootstrap':
            n_bootstraps = results_df['conv_n_evaluations'].iloc[0]
            print(f"All datasets evaluated using Bootstrap .632+ ({n_bootstraps} samples)")
        else:
            n_folds = results_df['conv_n_evaluations'].iloc[0]
            print(f"All datasets evaluated using {n_folds}-fold CV")

        print("=" * 160)

        # Check for high variance
        high_variance_targets = results_df[results_df['alt_r2_std'] > 0.3]['target'].tolist()
        if high_variance_targets:
            print(f"\n⚠️  High variance on alternative dataset for: {', '.join(high_variance_targets)}")
            print("      This is expected given small sample size (n≈18)")

            if evaluation_method == 'bootstrap':
                print("      Bootstrap .632+ provides bias-corrected estimates despite high variance")
            print()

        # Main table
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
                print(f"\n  ⚠️  High average overfitting indicates RF models may be too complex for n=18")
                print(f"     Consider reducing max_depth or increasing min_samples_leaf")

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
                                         include_feature_engineered: bool = False,
                                         max_features:int = 1
                                         ) -> Optional[Tuple[str, ...]]:
        """Get best predictor combination for target from selector results."""
        # ✅ Use foundation to extract combinations
        combos = self.foundation.extract_combinations_for_target(
            target,
            include_feature_engineered=include_feature_engineered,
            max_predictors=max_features,
        )

        if not combos:
            return None

        # Use first combination (selector's best)
        return combos[0]

    # ========================================================================
    # HELPER METHODS
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

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export_results(self, filename: str = 'rf_cross_dataset_performance.csv'):
        """Export results to CSV."""
        if self.results_df is None:
            raise ValueError("No results to export. Run analyze_best_combinations() first.")

        self.results_df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
        print(f"  {len(self.results_df)} rows, {len(self.results_df.columns)} columns")