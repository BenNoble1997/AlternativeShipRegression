"""
Random Forest Domain Shift Analyzer
====================================

Analyzes domain shift from conventional to alternative fuel ships using
Random Forest regression exclusively.

Similar to ConstrainedDomainAnalyzer but:
- Uses Random Forest only (no model selection)
- Provides feature importance analysis
- Specialized plotting for RF results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.data_split_handling_class import DataSplitHandler
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class import AnalysisPlotter,ResidualData,PredictionData,FourWayMetrics,DatasetMetrics

from config import LoggerMixin

# ============================================================================
# RANDOM FOREST DOMAIN ANALYZER
# ============================================================================


class RandomForestDomainAnalyzer(LoggerMixin):
    """
    Analyzes domain shift using Random Forest regression.

    Experimental Design:
    -------------------
    1. Split conventional data: 80% train, 20% test
    2. Train Random Forest on conventional train set
    3. Evaluate on four datasets:
       - Conv Train (training performance)
       - Conv Test (in-domain generalization)
       - Conv Matched (distribution shift control)
       - Alternative (out-of-domain + fuel type effect)

    Key Metrics:
    -----------
    - Conv Train → Conv Test gap: Normal generalization error
    - Conv Test → Conv Matched gap: Distribution shift effect
    - Conv Matched → Alternative gap: FUEL TYPE EFFECT

    Additional RF Features:
    ----------------------
    - Feature importance analysis
    - Out-of-bag error estimates
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: List[str] = None,
                 selector_results: Optional[Dict] = None,
                 rf_params: Optional[Dict] = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize Random Forest Domain Analyzer.

        **REFACTORED:** Simplified initialization - only needs selector_results
        dictionary from RandomForestSelector. Now uses DataSplitHandler for
        consistent data splitting.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset (will be split using DataSplitHandler)
        conventional_matched : pd.DataFrame
            Conventional ships matched to alternative ships (control)
        alternative_data : pd.DataFrame
            Alternative fuel ships (experimental)
        variable_dimensions : List[str]
            Column names for all dimensions
        target_dimensions : List[str], optional
            Target dimensions. If None, uses variable_dimensions.
        selector_results : dict, optional
            Dictionary from RandomForestSelector.results
            Format: {target: {combo_name: {'predictors': tuple, 'metrics': dict, ...}}}
            **CRITICAL:** This is now REQUIRED for proper operation.
        rf_params : dict, optional
            RandomForestRegressor parameters
        test_size : float
            Fraction for test split
        random_state : int
            Random seed
        verbose : bool
            Print detailed progress

        Raises
        ------
        ValueError
            If selector_results is None or empty

        Notes
        -----
        **Design Change:** The analyzer now operates as a strict pipeline:
        1. RandomForestSelector tests combinations and builds models
        2. RandomForestDomainAnalyzer tests those exact combinations on alternative fuel ships
        3. DataSplitHandler handles all train/test splitting

        This ensures:
        - Perfect alignment between selector and analyzer
        - No duplicate computation of valid combinations
        - All tested combinations use same RF configuration
        - Consistent splitting logic across all analyzers
        """

        # Setup dimensions
        if target_dimensions is None:
            target_dimensions = variable_dimensions

        # Combine and validate dimensions
        all_dimensions = list(set(variable_dimensions + target_dimensions))
        available_dimensions = [d for d in all_dimensions if d in conventional_data.columns]
        missing_dimensions = [d for d in all_dimensions if d not in conventional_data.columns]

        if missing_dimensions and verbose:
            print(f"\n  ⚠️  Warning: Some dimensions not found in training_data:")
            for dim in missing_dimensions:
                print(f"    - '{dim}'")
            print(f"  Using {len(available_dimensions)} available dimensions\n")

        if 'DWT' not in available_dimensions:
            raise ValueError("DWT must be in the training data!")

        # Store configuration
        self.variable_dimensions = [d for d in variable_dimensions if d in available_dimensions]
        self.target_dimensions = [d for d in target_dimensions if d in available_dimensions]
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.analysis_plotter = AnalysisPlotter()

        # Default RF parameters
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': random_state,
            'n_jobs': -1,
            'oob_score': True  # Enable OOB error
        }

        # Store data
        self.conventional_full = conventional_data[available_dimensions].copy()
        # self.conventional_matched = conventional_matched[available_dimensions].copy()
        self.alternative = alternative_data[available_dimensions].copy()



        # CRITICAL: Validate and store selector results
        if selector_results is None or len(selector_results) == 0:
            raise ValueError(
                "selector_results is required and cannot be empty!\n"
                "Please run RandomForestSelector.analyze_all_combinations() first,\n"
                "then pass selector.results to this analyzer."
            )

        self.selector_results = selector_results

        # Validate data quality
        self._validate_data_quality()
        self.matched_datasets = self._create_matched_sets()

        # Storage
        self.results = {}
        self.feature_importance = {}

    def _validate_data_quality(self):
        """Check data quality and warn about issues."""
        if not self.verbose:
            return

        for dataset_name, dataset in [
            ('conventional', self.conventional_full),
            # ('matched', self.conventional_matched),
            ('alternative', self.alternative)
        ]:
            nan_counts = dataset.isna().sum()
            if nan_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has missing values:")
                for col, count in nan_counts[nan_counts > 0].items():
                    pct = 100 * count / len(dataset)
                    print(f"    {col}: {count} ({pct:.1f}%)")

            if len(dataset) < 10:
                print(f"\n  ⚠️  WARNING: {dataset_name} dataset is very small ({len(dataset)} ships)")

    # ========================================================================
    # COMBINATION EXTRACTION FROM SELECTOR RESULTS (Unchanged)
    # ========================================================================

    def _extract_combinations_from_selector(self,
                                            target: str,
                                            max_predictors: Optional[int] = None,
                                            include_feature_engineered: bool = False) -> List[Tuple[str, ...]]:
        """Extract predictor combinations for a target from selector results."""
        if target not in self.selector_results:
            return []

        combos = []
        seen_predictors = set()

        for combo_name, data in self.selector_results[target].items():
            predictors = data['predictors']
            is_engineered = data.get('feature_engineered', False)

            if is_engineered and not include_feature_engineered:
                continue

            if max_predictors is not None and len(predictors) > max_predictors:
                continue

            predictor_key = tuple(predictors)
            if predictor_key in seen_predictors:
                continue

            seen_predictors.add(predictor_key)
            combos.append(predictor_key)

        combos.sort(key=lambda x: (len(x), x))
        return combos

    def _create_matched_sets(self):
        """
        Helper method for creating matched datasets for each independent variable combination
        """

        combo_list = []
        for target in self.target_dimensions:
            combo = self._extract_combinations_from_selector(target=target)
            combo_list.extend(combo)

        sorted_list = sorted(set(combo_list))

        return MatchedDataset.multiple_matched_datasets(alternative_fleet=self.alternative,conventional_fleet=self.conventional_full,
                                                        logger=self.logger,predictor_groups=sorted_list,std_reference='conventional')


    def get_combinations_summary(self) -> pd.DataFrame:
        """Get summary of available combinations from selector results."""
        summary_data = []

        for target in self.target_dimensions:
            if target not in self.selector_results:
                continue

            all_combos = self.selector_results[target]

            n_raw = sum(1 for data in all_combos.values() if not data.get('feature_engineered', False))
            n_engineered = sum(1 for data in all_combos.values() if data.get('feature_engineered', False))

            unique_predictors = set(data['predictors'] for data in all_combos.values())
            sizes = sorted(list(set(len(p) for p in unique_predictors)))

            summary_data.append({
                'target': target,
                'n_combinations': len(unique_predictors),
                'n_raw': n_raw,
                'n_engineered': n_engineered,
                'predictor_sizes': sizes
            })

        return pd.DataFrame(summary_data)

    # ========================================================================
    # MAIN ANALYSIS METHOD (Unchanged logic, uses refactored _prepare_fourway_data)
    # ========================================================================

    def analyze_all_combinations(self,
                                 max_predictors: int = 2,
                                 include_feature_engineered: bool = False,
                                 display_results: bool = True):
        """
        Analyze all combinations with four-way evaluation using RF.

        **REFACTORED:** Now extracts combinations from selector results
        and uses DataSplitHandler for consistent train/test splitting.

        Parameters
        ----------
        max_predictors : int
            Filter to combinations with at most this many predictors
        include_feature_engineered : bool
            If True, tests both raw and feature-engineered versions.
        display_results : bool, default=True
            If True, displays results during analysis.

        Returns
        -------
        dict
            results dictionary with same structure as before
        """
        if not self.selector_results:
            raise ValueError(
                "Cannot analyze combinations without selector_results.\n"
                "Run RandomForestSelector.analyze_all_combinations() first,\n"
                "then pass selector.results to this analyzer."
            )

        if display_results and self.verbose:
            print("\n" + "=" * 180)
            print("RANDOM FOREST DOMAIN SHIFT ANALYSIS (Four-Way Evaluation)")
            print("=" * 180)

        results = {}

        for target in self.target_dimensions:
            if target not in self.selector_results:
                if self.verbose:
                    print(f"\n  ⚠️  No selector results found for {target}")
                continue

            pred_combos = self._extract_combinations_from_selector(
                target=target,
                max_predictors=max_predictors,
                include_feature_engineered=include_feature_engineered
            )

            if not pred_combos:
                if self.verbose:
                    print(f"\n  ⚠️  No combinations found for {target} "
                          f"with ≤{max_predictors} predictors")
                continue

            if self.verbose:
                print(f"\nAnalyzing {target}: {len(pred_combos)} combinations from selector")

            results[target] = {}

            if display_results and self.verbose:
                print(f"\n{'=' * 180}")
                print(f"Target: {target}")
                print(f"Available predictors: {list(set(p for combo in pred_combos for p in combo))}")
                print(f"{'=' * 180}")
                print(f"{'Predictors':<25s} | "
                      f"{'Train R²':>8s} | {'Test R²':>8s} | {'Match R²':>8s} | {'Alt R²':>8s} | "
                      f"{'Train RRMSE%':>12s} | {'Test RRMSE%':>11s} | {'Match RRMSE%':>12s} | {'Alt RRMSE%':>11s} | "
                      f"{'FUEL Gap R²':>12s} | {'FUEL Gap RRMSE':>15s} | {'Top Feature':<15s}")
                print(f"{'-' * 180}")

            for pred_combo in pred_combos:
                combo_name = " + ".join(pred_combo)
                result = self.train_and_test_fourway(list(pred_combo), target)

                if result is not None:
                    results[target][combo_name] = result

                    if display_results and self.verbose:
                        feature_imp = result.get('feature_importance', {})
                        if feature_imp:
                            top_feature = max(feature_imp, key=feature_imp.get)
                        else:
                            top_feature = "N/A"

                        print(f"{combo_name:<25s} | "
                              f"{result['train_r2']:>8.3f} | "
                              f"{result['test_r2']:>8.3f} | "
                              f"{result['matched_r2']:>8.3f} | "
                              f"{result['alternative_r2']:>8.3f} | "
                              f"{result['train_rrmse']:>12.2f} | "
                              f"{result['test_rrmse']:>11.2f} | "
                              f"{result['matched_rrmse']:>12.2f} | "
                              f"{result['alternative_rrmse']:>11.2f} | "
                              f"{result['gap_fuel_type']:>+12.3f} | "
                              f"{result['gap_fuel_type_rrmse']:>+15.2f} | "
                              f"{top_feature:<15s}")

        # ✅ NEW: Convert to DataFrame before storing/returning
        results_df = self._results_to_dataframe(results)

        # Store both formats for backward compatibility
        self.results = results  # Keep dict for internal use
        self.results_df = results_df  # Add DataFrame

        if display_results and self.verbose:
            print(f"\n{'=' * 180}")
            total_tested = len(results_df) if not results_df.empty else 0
            print(f"✓ Random Forest analysis complete: {total_tested} combinations tested")

        return results_df

    def _results_to_dataframe(self, results_dict: dict) -> pd.DataFrame:
        """
        Convert results dict to DataFrame format.

        Parameters
        ----------
        results_dict : dict
            Nested dict: {target: {combo_name: metrics}}

        Returns
        -------
        pd.DataFrame
            Flattened results with standardized columns
        """
        rows = []

        for target, combos in results_dict.items():
            for combo_name, metrics in combos.items():
                # Create base row
                row = {
                    'target': target,
                    'predictor_str': combo_name,
                    'predictors': metrics.get('predictors'),
                    'n_predictors': metrics.get('n_predictors'),
                    'model_type': metrics.get('model_type', 'random_forest'),
                }

                # Add all other metrics
                for key, value in metrics.items():
                    if key not in row:
                        row[key] = value

                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Reorder columns to match parametric analyzer
        priority_cols = [
            'target', 'predictor_str', 'predictors', 'n_predictors', 'model_type',
            'train_r2', 'test_r2', 'matched_r2', 'alternative_r2',
            'train_rrmse', 'test_rrmse', 'matched_rrmse', 'alternative_rrmse',
            'train_mae', 'test_mae', 'matched_mae', 'alternative_mae',
            'train_mape', 'test_mape', 'matched_mape', 'alternative_mape',
            'train_n', 'test_n', 'matched_n', 'alternative_n',
            'gap_generalization', 'gap_distribution', 'gap_fuel_type', 'gap_total',
            'gap_generalization_rrmse', 'gap_distribution_rrmse',
            'gap_fuel_type_rrmse', 'gap_total_rrmse',
            'feature_importance', 'oob_r2'
        ]

        # Get columns in priority order, then remaining
        ordered_cols = [c for c in priority_cols if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in ordered_cols]

        return df[ordered_cols + remaining_cols]

    # ========================================================================
    # FOUR-WAY TESTING (Uses refactored _prepare_fourway_data)
    # ========================================================================

    def train_and_test_fourway(self, predictors: List[str], target: str) -> Optional[Dict]:
        """Train Random Forest on conventional train set, test on four datasets."""
        if not predictors or len(predictors) == 0:
            if self.verbose:
                print(f"  Error: No predictors provided for target {target}")
            return None

        # ✅ REFACTORED: Now uses DataSplitHandler
        data_prep = self._prepare_fourway_data(predictors, target)
        if data_prep is None:
            return None

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        datasets_to_predict = {
            'train': X_train,
            'test': data_prep['X_test'],
            'matched': data_prep['X_matched'],
            'alternative': data_prep['X_alternative']
        }

        try:
            predictions, rf_model = self._fit_and_predict_all(
                X_train, y_train, datasets_to_predict, predictors
            )
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None

        result = self._calculate_fourway_metrics(
            predictions,
            {
                'train': data_prep['y_train'],
                'test': data_prep['y_test'],
                'matched': data_prep['y_matched'],
                'alternative': data_prep['y_alternative']
            },
            predictors,
            rf_model
        )

        return result

    def _fit_and_predict_all(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             datasets_dict: Dict[str, np.ndarray],
                             predictor_names: List[str]) -> Tuple[Dict, RandomForestRegressor]:
        """Fit Random Forest once, predict on multiple datasets."""
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, y_train)

        predictions = {}
        for dataset_name, X_data in datasets_dict.items():
            predictions[dataset_name] = rf.predict(X_data)

        return predictions, rf

    def _prepare_fourway_data(self, predictors: List[str], target: str) -> Optional[Dict]:
        """
        Prepare data from all four datasets with proper cleaning.

        **REFACTORED:** Now uses DataSplitHandler.create_stratified_train_test_split()
        for consistent and maintainable train/test splitting.
        """
        # 1. Clean conventional data
        X_conv_full = self.conventional_full[predictors].values
        y_conv_full = self.conventional_full[target].values

        conv_mask = ~(np.isnan(X_conv_full).any(axis=1) | np.isnan(y_conv_full) |
                      np.isinf(X_conv_full).any(axis=1) | np.isinf(y_conv_full))

        X_conv_clean = X_conv_full[conv_mask]
        y_conv_clean = y_conv_full[conv_mask]

        if len(X_conv_clean) < 20:
            if self.verbose:
                print(f"  Warning: Insufficient clean conventional data ({len(X_conv_clean)} samples)")
            return None

        # 2. ✅ REFACTORED: Use DataSplitHandler for stratified split
        dwt_idx = predictors.index('DWT') if 'DWT' in predictors else 0

        try:
            X_train, X_test, y_train, y_test = DataSplitHandler.create_stratified_train_test_split(
                X=X_conv_clean,
                y=y_conv_clean,
                stratify_col_index=dwt_idx,
                test_size=self.test_size,
                random_state=self.random_state,
                n_bins=5
            )
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Stratified split failed ({e}), using random split")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_conv_clean, y_conv_clean,
                test_size=self.test_size,
                random_state=self.random_state
            )

        # 3. Clean matched data

        predictor_key = tuple(predictors)  # ← Add this line
        conventional_matched = self.matched_datasets[predictor_key]

        X_matched = conventional_matched[predictors].values
        y_matched = conventional_matched[target].values

        matched_mask = ~(np.isnan(X_matched).any(axis=1) | np.isnan(y_matched) |
                         np.isinf(X_matched).any(axis=1) | np.isinf(y_matched))

        X_matched = X_matched[matched_mask]
        y_matched = y_matched[matched_mask]

        # 4. Clean alternative data
        X_alt = self.alternative[predictors].values
        y_alt = self.alternative[target].values

        alt_mask = ~(np.isnan(X_alt).any(axis=1) | np.isnan(y_alt) |
                     np.isinf(X_alt).any(axis=1) | np.isinf(y_alt))

        X_alt = X_alt[alt_mask]
        y_alt = y_alt[alt_mask]

        # 5. Validate sufficient data
        min_samples = {'train': 10, 'test': 3, 'matched': 3, 'alternative': 3}
        dataset_sizes = {
            'train': len(X_train),
            'test': len(X_test),
            'matched': len(X_matched),
            'alternative': len(X_alt)
        }

        for dataset_name, min_required in min_samples.items():
            if dataset_sizes[dataset_name] < min_required:
                if self.verbose:
                    print(f"  Warning: Insufficient {dataset_name} data "
                          f"({dataset_sizes[dataset_name]} < {min_required})")
                return None

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_matched': X_matched, 'y_matched': y_matched,
            'X_alternative': X_alt, 'y_alternative': y_alt
        }

    def _calculate_fourway_metrics(self,
                                   predictions: Dict,
                                   actuals: Dict,
                                   predictors: List[str],
                                   rf_model: RandomForestRegressor) -> Dict:
        """Calculate comprehensive metrics for all four datasets."""
        result = {
            'model_type': 'random_forest',
            'predictors': predictors,
            'n_predictors': len(predictors),
        }

        for dataset_name in ['train', 'test', 'matched', 'alternative']:
            y_true = actuals[dataset_name]
            y_pred = predictions[dataset_name]

            metric_calculator = RegressionMetrics(y_true, y_pred)

            result[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            result[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            result[f'{dataset_name}_rrmse'] = metric_calculator.calculate_rrmse()
            result[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
            result[f'{dataset_name}_mape'] = metric_calculator.calculate_mape()
            result[f'{dataset_name}_n'] = len(y_true)

        # R² gaps
        result['gap_generalization'] = result['train_r2'] - result['test_r2']
        result['gap_distribution'] = result['test_r2'] - result['matched_r2']
        result['gap_fuel_type'] = result['matched_r2'] - result['alternative_r2']
        result['gap_total'] = result['test_r2'] - result['alternative_r2']

        # RRMSE gaps
        result['gap_generalization_rrmse'] = result['test_rrmse'] - result['train_rrmse']
        result['gap_distribution_rrmse'] = result['matched_rrmse'] - result['test_rrmse']
        result['gap_fuel_type_rrmse'] = result['alternative_rrmse'] - result['matched_rrmse']
        result['gap_total_rrmse'] = result['alternative_rrmse'] - result['test_rrmse']

        # Feature importance
        result['feature_importance'] = dict(zip(predictors, rf_model.feature_importances_))

        # OOB score if available
        if hasattr(rf_model, 'oob_score_'):
            result['oob_r2'] = rf_model.oob_score_

        return result

    # ========================================================================
    # DISPLAY AND PLOTTING METHODS (Unchanged)
    # ========================================================================

    def interpret_results(self,
                          significance_threshold_r2: float = 0.05,
                          significance_threshold_rrmse: float = 10.0):
        """Convenience method to display interpretation."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_interpretation(significance_threshold_r2, significance_threshold_rrmse)

    def print_initialization_summary(self):
        """Convenience method to display initialization info."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_initialization_summary()

    # ========================================================================
    # DISPLAY CONVENTIONECE FUNCTIONS
    # ========================================================================

    def print_initialization_summary(self):
        """Convenience method to display initialization info."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_initialization_summary()

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================


    def plot_domain_shift_analysis(self, save_path: str|None = None):
        """
        Plot domain shift visualization.

        **Refactored**: Prepares data, delegates plotting to DomainShiftPlotter.
        """
        if not self.results or len(self.results) == 0:
            print("No results to plot. Run analyze_all_combinations() first.")
            return

        # Prepare data for ALL targets
        targets_data = {}

        for target, models in self.results.items():
            if not models:
                continue

            # Get best model
            best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo

            # Convert analyzer metrics to plotter format
            fourway_metrics = self._convert_to_fourway_metrics(metrics)

            targets_data[target] = (
                metrics['predictors'],
                metrics['model_type'],
                fourway_metrics
            )

        if not targets_data:
            print("No valid data to plot!")
            return

        # Delegate to plotter
        fig = self.analysis_plotter.domain_shift_plotter.plot_multiple_targets(
            targets_data=targets_data,
            save_path=save_path
        )

        plt.show()

        if save_path:
            print(f"✓ Plot saved to {save_path}")

    def plot_prediction_scatter(self,
                                save_dir: str|None = None,
                                figsize: tuple = (12, 12)):
        """
        Plot prediction scatter plots.

        **Refactored**: Prepares data, delegates plotting to PredictionScatterPlotter.
        """
        if not self.results:
            print("No results to plot. Run analyze_all_combinations() first.")
            return

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        for target, models in self.results.items():
            if not models:
                continue

            best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo
            predictors = metrics['predictors']
            model_type = metrics['model_type']

            # Prepare data
            data_prep = self._prepare_fourway_data(predictors, target)
            if data_prep is None:
                continue

            # Get predictions
            predictions_dict = self._get_all_predictions(
                data_prep, predictors, model_type
            )

            if predictions_dict is None:
                continue

            # Convert to plotter format
            train_data = self._create_prediction_data(
                'train', data_prep['y_train'], predictions_dict['train'],
                metrics['train_r2'], metrics['train_rrmse'], metrics['train_n']
            )
            test_data = self._create_prediction_data(
                'test', data_prep['y_test'], predictions_dict['test'],
                metrics['test_r2'], metrics['test_rrmse'], metrics['test_n']
            )
            matched_data = self._create_prediction_data(
                'matched', data_prep['y_matched'], predictions_dict['matched'],
                metrics['matched_r2'], metrics['matched_rrmse'], metrics['matched_n']
            )
            alternative_data = self._create_prediction_data(
                'alternative', data_prep['y_alternative'], predictions_dict['alternative'],
                metrics['alternative_r2'], metrics['alternative_rrmse'], metrics['alternative_n']
            )

            # Delegate to plotter
            save_file = None
            if save_dir:
                safe_name = target.replace('/', '_').replace('\\', '_').replace(' ', '_')
                save_file = str(save_path / f'{safe_name}_scatter.png')

            self.analysis_plotter.prediction_scatter_plotter.plot_four_way_scatter(
                target_name=target,
                predictor_names=predictors,
                model_description=model_type,
                train_data=train_data,
                test_data=test_data,
                matched_data=matched_data,
                alternative_data=alternative_data,
                save_path=save_file
            )

            if save_file:
                print(f"  ✓ Saved: {Path(save_file).name}")

    def plot_residuals(self, save_dir: str|None = None):
        """
        Plot residual analysis.

        **Refactored**: Prepares data, delegates plotting to ResidualPlotter.
        """
        if not self.results:
            print("No results to plot. Run analyze_all_combinations() first.")
            return

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        for target, models in self.results.items():
            if not models:
                continue

            best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo
            predictors = metrics['predictors']
            model_type = metrics['model_type']

            # Prepare data
            data_prep = self._prepare_fourway_data(predictors, target)
            if data_prep is None:
                continue

            # Get predictions
            predictions_dict = self._get_all_predictions(
                data_prep, predictors, model_type
            )

            if predictions_dict is None:
                continue

            # Convert to plotter format
            train_residuals = ResidualData(
                y_true=data_prep['y_train'],
                y_pred=predictions_dict['train'],
                dataset_name='train'
            )
            test_residuals = ResidualData(
                y_true=data_prep['y_test'],
                y_pred=predictions_dict['test'],
                dataset_name='test'
            )
            matched_residuals = ResidualData(
                y_true=data_prep['y_matched'],
                y_pred=predictions_dict['matched'],
                dataset_name='matched'
            )
            alternative_residuals = ResidualData(
                y_true=data_prep['y_alternative'],
                y_pred=predictions_dict['alternative'],
                dataset_name='alternative'
            )

            # Delegate to plotter
            save_file = None
            if save_dir:
                safe_name = target.replace('/', '_').replace('\\', '_').replace(' ', '_')
                save_file = str(save_path / f'{safe_name}_residuals.png')

            self.analysis_plotter.residual_plotter.plot_four_way_residuals(
                target_name=target,
                predictor_names=predictors,
                model_description=model_type,
                train_data=train_residuals,
                test_data=test_residuals,
                matched_data=matched_residuals,
                alternative_data=alternative_residuals,
                save_path=save_file
            )

            if save_file:
                print(f"  ✓ Saved: {Path(save_file).name}")

        if save_dir:
            print(f"\n✓ All residual plots saved to {save_dir}")

    # ========================================================================
    # HELPER METHODS FOR DATA CONVERSION
    # ========================================================================

    def _convert_to_fourway_metrics(self, metrics: dict) -> FourWayMetrics:
        """Convert analyzer metrics dict to FourWayMetrics dataclass."""
        return FourWayMetrics(
            train=DatasetMetrics(
                r2=metrics['train_r2'],
                rrmse=metrics['train_rrmse'],
                rmse=metrics['train_rmse'],
                mae=metrics['train_mae'],
                mape=metrics['train_mape'],
                n_samples=metrics['train_n']
            ),
            test=DatasetMetrics(
                r2=metrics['test_r2'],
                rrmse=metrics['test_rrmse'],
                rmse=metrics['test_rmse'],
                mae=metrics['test_mae'],
                mape=metrics['test_mape'],
                n_samples=metrics['test_n']
            ),
            matched=DatasetMetrics(
                r2=metrics['matched_r2'],
                rrmse=metrics['matched_rrmse'],
                rmse=metrics['matched_rmse'],
                mae=metrics['matched_mae'],
                mape=metrics['matched_mape'],
                n_samples=metrics['matched_n']
            ),
            alternative=DatasetMetrics(
                r2=metrics['alternative_r2'],
                rrmse=metrics['alternative_rrmse'],
                rmse=metrics['alternative_rmse'],
                mae=metrics['alternative_mae'],
                mape=metrics['alternative_mape'],
                n_samples=metrics['alternative_n']
            ),
            gap_generalization_r2=metrics['gap_generalization'],
            gap_distribution_r2=metrics['gap_distribution'],
            gap_fuel_type_r2=metrics['gap_fuel_type'],
            gap_generalization_rrmse=metrics['gap_generalization_rrmse'],
            gap_distribution_rrmse=metrics['gap_distribution_rrmse'],
            gap_fuel_type_rrmse=metrics['gap_fuel_type_rrmse']
        )

    def _create_prediction_data(self, dataset_name: str, y_true: np.ndarray,
                                y_pred: np.ndarray, r2: float, rrmse: float,
                                n: int) -> PredictionData:
        """Create PredictionData dataclass from arrays and metrics."""
        return PredictionData(
            y_true=y_true,
            y_pred=y_pred,
            r2=r2,
            rrmse=rrmse,
            n_samples=n,
            dataset_name=dataset_name
        )

    def _get_all_predictions(self, data_prep: Dict, predictors: List[str],
                             model_type: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get predictions for all four datasets.

        **RF-specific**: Uses RandomForestRegressor instead of ModelFactory.
        """
        from sklearn.ensemble import RandomForestRegressor

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        try:
            rf = RandomForestRegressor(**self.rf_params)
            rf.fit(X_train, y_train)

            predictions = {}
            for dataset_name in ['train', 'test', 'matched', 'alternative']:
                X_data = data_prep[f'X_{dataset_name}']
                predictions[dataset_name] = rf.predict(X_data)

            return predictions

        except Exception as e:
            if self.verbose:
                print(f"  Error getting predictions: {e}")
            return None
