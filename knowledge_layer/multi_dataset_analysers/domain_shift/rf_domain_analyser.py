# File: random_forest_domain_analysis_class.py

"""
Random Forest Domain Shift Analyzer
====================================
Analyzes domain shift from conventional to alternative fuel ships using
Random Forest regression.

Uses RandomForestAnalysisFoundation via composition.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Optional
from pathlib import Path

import matplotlib.pyplot as plt
from knowledge_layer.multi_dataset_analysers.rf_analysis_foundation import RandomForestAnalysisFoundation
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class import (
    AnalysisPlotter, ResidualData, PredictionData, FourWayMetrics, DatasetMetrics
)
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility

from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.data_split_handling_class import DataSplitHandler

from config.loggermixin.logger_config import LoggerMixin


class RandomForestDomainAnalyzer(LoggerMixin):
    """
    Analyzes domain shift using Random Forest regression.

    Uses RandomForestAnalysisFoundation via composition.

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
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: Optional[List[str]] = None,
                 selector_results: Optional[Dict] = None,
                 rf_params: Optional[Dict] = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True,
                 selection_metric: str = 'rrmse',
                 target_predictor_constraints:dict[list,list[str]]=None,
                 build_year_col:str|None=None,
                 year_edges:list[str]=None,
                 build_eligibility:TemporalEligibility|None=None,
                 matching_bins:dict[str,list[str]]=None,
                 n_matches_per_ship:int=10,
                 caliper:float|None=None
                 ):
        """
        Initialize Random Forest Domain Analyzer.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset (will be split 80/20)
        alternative_data : pd.DataFrame
            Alternative fuel ships dataset
        variable_dimensions : List[str]
            List of potential predictor dimensions
        target_dimensions : List[str], optional
            List of target dimensions. If None, uses variable_dimensions.
        selector_results : dict
            Dictionary from RandomForestSelector.results
            Format: {target: {combo_name: {'predictors': tuple, 'metrics': dict, ...}}}
            **REQUIRED for proper operation**
        rf_params : dict, optional
            RandomForestRegressor parameters
        test_size : float, default=0.2
            Fraction for train/test split
        random_state : int, default=42
            Random seed
        verbose : bool, default=True
            Print detailed progress
        selection_metric : str, default='rrmse'
        Metric to use for selecting best combinations:
        - 'adj_r2': Adjusted R² (higher is better)
        - 'r2': R² (higher is better)
        - 'rrmse': Relative RMSE% (lower is better)
        - 'mape': Mean Absolute Percentage Error% (lower is better)
        - 'oob_r2': Out-of-bag R² (higher is better, RF-specific)
        """
        # ✅ Initialize foundation for data management
        self.foundation = RandomForestAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            selector_results=selector_results,
            rf_params=rf_params,
            random_state=random_state,
            verbose=verbose,
            selection_metric=selection_metric,
            target_predictor_constraints=target_predictor_constraints,
            defer_matched_sets=True,
            build_year_col=build_year_col,
            year_edges=year_edges,
            eligibility=build_eligibility,
            n_matches_per_ship=n_matches_per_ship,
            matching_bins=matching_bins,
            caliper=caliper
        )

        # Store local configuration
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.analysis_plotter = AnalysisPlotter()
        self.selection_metric = selection_metric

        # Split conventional data into train/test
        self._split_conventional_data()

        self.foundation.build_matched_sets(
            candidate_pool=self.conventional_test,
            debug=self.verbose,
        )


        # Storage
        self.results = {}
        self.feature_importance = {}
        self.results_df = None

    # ========================================================================
    # DELEGATE PROPERTIES
    # ========================================================================

    @property
    def target_dimensions(self) -> List[str]:
        """Target dimensions from foundation."""
        return self.foundation.target_dimensions

    @property
    def conventional_full(self) -> pd.DataFrame:
        """Full conventional dataset from foundation."""
        return self.foundation.conventional_full

    @property
    def alternative(self) -> pd.DataFrame:
        """Alternative dataset from foundation."""
        return self.foundation.alternative

    @property
    def selector_results(self) -> Dict:
        """Selector results from foundation."""
        return self.foundation.selector_results

    @property
    def rf_params(self) -> Dict:
        """RF parameters from foundation."""
        return self.foundation.get_rf_params()

    # ========================================================================
    # DATA SPLITTING
    # ========================================================================

    def _split_conventional_data(self):
        """
        Split the conventional dataset into train and test partitions.

        Uses a stratified split on DWT quintiles, implemented via positional
        index splitting (``train_test_split`` on ``np.arange(n)`` with
        ``stratify=pd.qcut(DWT, 5)``). This is intentionally IDENTICAL to the
        implementation in :class:`RandomForestDomainAnalyzer`: because matched
        conventional datasets are built from the test partition, both analyzers
        must produce the same partition under the same ``random_state`` for the
        parametric-vs-RF comparison to be evaluated against identical matched
        sets. Do not change one implementation without changing the other.

        Index-based splitting also avoids the failure modes of value-based
        membership reconstruction: rows containing NaN and duplicate-valued
        sister ships are assigned unambiguously by position.

        Sets ``self.conventional_train``, ``self.conventional_test``,
        ``self._train_indices``, and ``self._test_indices``. Original DataFrame
        index labels are preserved by ``.iloc`` selection on a copy.
        """
        from sklearn.model_selection import train_test_split

        conv_data = self.conventional_full.copy()
        n = len(conv_data)
        all_indices = np.arange(n)

        if 'DWT' in conv_data.columns:
            try:
                dwt_vals = conv_data['DWT'].values
                bins = pd.qcut(dwt_vals, q=5, labels=False, duplicates='drop')

                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=bins
                )

                self._train_indices = train_idx
                self._test_indices = test_idx
                self.conventional_train = conv_data.iloc[train_idx].copy()
                self.conventional_test = conv_data.iloc[test_idx].copy()

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Stratified split failed ({e}), using random split")
                self._random_split_conventional_data(conv_data)
        else:
            if self.verbose:
                print("  Warning: DWT not found, using random split")
            self._random_split_conventional_data(conv_data)

        if self.verbose:
            print(f"\nConventional data split:")
            print(f"  Train: {len(self.conventional_train)} ships ({(1 - self.test_size) * 100:.0f}%)")
            print(f"  Test:  {len(self.conventional_test)} ships ({self.test_size * 100:.0f}%)")

    def _random_split_conventional_data(self, conv_data: pd.DataFrame):
        """Fallback to random split if stratified fails."""
        from sklearn.model_selection import train_test_split

        n = len(conv_data)
        all_indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self._train_indices = train_idx
        self._test_indices = test_idx
        self.conventional_train = conv_data.iloc[train_idx].copy()
        self.conventional_test = conv_data.iloc[test_idx].copy()

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_combinations(self,
                                 max_predictors: int = 2,
                                 best_only: bool = True,  # ← PARAMETER EXISTS
                                 selection_metric: str = 'rrmse',
                                 selection_based_on_conventional: bool = True,  # ← ADD THIS
                                 include_feature_engineered: bool = False,
                                 display_results: bool = True) -> pd.DataFrame:
        """
        Analyze all combinations with four-way evaluation using RF.

        Parameters
        ----------
        max_predictors : int, default=2
            Maximum number of predictors per combination
        best_only : bool, default=True
            If True, analyze only BEST combination per target (by conventional).
            If False, analyze ALL combinations from selector.
        selection_metric : str, default='rrmse'
            Metric for selecting best: 'adj_r2', 'rrmse', 'mape'
        selection_based_on_conventional : bool, default=True
            If True, selects best by conventional performance ONLY.
            If False, filters by alternative data availability first.
        include_feature_engineered : bool, default=False
            If True, tests both raw and feature-engineered versions
        display_results : bool, default=True
            Print results table

        Returns
        -------
        pd.DataFrame
            Results with comprehensive metrics
        """
        if display_results and self.verbose:
            print("\n" + "=" * 180)
            print("RANDOM FOREST DOMAIN SHIFT ANALYSIS (Four-Way Evaluation)")
            print("=" * 180)

        all_results = []

        for target in self.target_dimensions:

            # ========================================================================
            # ✅ FIXED: Use best_only parameter properly
            # ========================================================================

            if best_only:
                # Get ONLY best combination for this target
                best_predictors = self.foundation.get_best_combination_for_target(
                    target=target,
                    max_predictors=max_predictors,
                    require_alternative_data=True,  # Check testability
                    min_alternative_samples=5,
                    selection_based_on_conventional=selection_based_on_conventional  # ← NEW!
                )

                if best_predictors is None:
                    if self.verbose:
                        print(f"\n  ⚠️  No valid combination for {target}")
                    continue

                pred_combos = [best_predictors]

                if self.verbose:
                    print(f"\nAnalyzing {target}: 1 best combination")
                    print(f"  Predictors: {best_predictors}")

            else:
                # ✅ OLD: Extract ALL combinations from selector
                pred_combos = self.foundation.extract_combinations_for_target(
                    target=target,
                    max_predictors=max_predictors,
                    include_feature_engineered=include_feature_engineered
                )

                if not pred_combos:
                    if self.verbose:
                        print(f"\n  ⚠️  No combinations found for {target}")
                    continue

                if self.verbose:
                    print(f"\nAnalyzing {target}: {len(pred_combos)} combinations")

            # Test selected combination(s)
            for pred_combo in pred_combos:
                result = self._train_and_test_fourway(list(pred_combo), target)

                if result is not None:
                    result['target'] = target
                    result['predictor_str'] = " + ".join(pred_combo)
                    all_results.append(result)

        # Create DataFrame
        if not all_results:
            if self.verbose:
                print("\n  ⚠️  No valid results generated!")
            return pd.DataFrame()

        results_df = self._results_to_dataframe_format(all_results)

        # Display results
        if display_results and self.verbose:
            self._print_results_table(results_df)

        # Store results
        self.results_df = results_df
        self.results = self._dataframe_to_dict(results_df)

        return results_df

    def _results_to_dataframe_format(self, all_results: List[Dict]) -> pd.DataFrame:
        """Convert results list to DataFrame with standardized columns."""
        results_df = pd.DataFrame(all_results)

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

        ordered_cols = [c for c in priority_cols if c in results_df.columns]
        remaining_cols = [c for c in results_df.columns if c not in ordered_cols]

        return results_df[ordered_cols + remaining_cols]

    def _print_results_table(self, results_df: pd.DataFrame):
        """Print formatted results table."""
        print(f"\n{'=' * 180}")
        print(f"{'Target':<15s} | {'Predictors':<25s} | "
              f"{'Train R²':>8s} | {'Test R²':>8s} | {'Match R²':>8s} | {'Alt R²':>8s} | "
              f"{'FUEL Gap R²':>12s} | {'FUEL Gap RRMSE':>15s}")
        print(f"{'-' * 180}")

        for _, row in results_df.iterrows():
            print(f"{row['target']:<15s} | "
                  f"{row['predictor_str']:<25s} | "
                  f"{row['train_r2']:>8.3f} | "
                  f"{row['test_r2']:>8.3f} | "
                  f"{row['matched_r2']:>8.3f} | "
                  f"{row['alternative_r2']:>8.3f} | "
                  f"{row['gap_fuel_type']:>+12.3f} | "
                  f"{row['gap_fuel_type_rrmse']:>+15.2f}")

        print(f"{'=' * 180}")
        print(f"✅ Random Forest analysis complete: {len(results_df)} combinations tested")

    def _train_and_test_fourway(self,
                                predictors: List[str],
                                target: str) -> Optional[Dict]:
        """Train RF on conventional train set, test on four datasets."""
        if not predictors or len(predictors) == 0:
            if self.verbose:
                print(f"  Error: No predictors provided for target {target}")
            return None

        # Prepare data from all four datasets
        data_prep = self._prepare_fourway_data(predictors, target)
        if data_prep is None:
            return None

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        #Get reference mean
        reference_mean = self.conventional_full[target].dropna().mean()

        datasets_to_predict = {
            'train': X_train,
            'test': data_prep['X_test'],
            'matched': data_prep['X_matched'],
            'alternative': data_prep['X_alternative']
        }

        try:
            # Fit RF model
            rf = RandomForestRegressor(**self.rf_params)
            rf.fit(X_train, y_train)

            # Predict on all datasets
            predictions = {}
            for dataset_name, X_data in datasets_to_predict.items():
                if X_data is None:
                    predictions[dataset_name] = None
                    continue
                predictions[dataset_name] = rf.predict(X_data)

        except Exception as e:
            if self.verbose:
                print(f"  RF fitting failed: {str(e)[:80]}")
            return None

        # Calculate comprehensive metrics
        result = self._calculate_fourway_metrics(
            predictions,
            {
                'train': data_prep['y_train'],
                'test': data_prep['y_test'],
                'matched': data_prep['y_matched'],
                'alternative': data_prep['y_alternative']
            },
            predictors,
            rf,
            reference_mean=reference_mean
        )

        return result

    def _prepare_fourway_data(self,
                              predictors: List[str],
                              target: str) -> Optional[Dict]:
        """
    Prepare data from all four evaluation datasets with proper cleaning.

    Train and test come from the pre-split conventional partitions; the
    matched set comes from the foundation (built from the test partition,
    so strictly out-of-sample); the alternative set may legitimately be
    unavailable for a combination with too few complete rows, in which
    case ``X_alternative``/``y_alternative`` are ``None`` and downstream
    metrics for that dataset are reported as NaN. This mirrors the
    parametric analyzer exactly so both produce structurally identical
    result rows.

    :param predictors: Predictor column names for this combination.
    :type predictors: list[str]
    :param target: Target column name.
    :type target: str
    :returns: Dict of prepared arrays, or ``None`` if conventional or
        matched data is insufficient.
    :rtype: dict or None
    """
        # 1. TRAIN DATA (from pre-split conventional_train)
        train_prep = self.foundation.prepare_dataset(
            self.conventional_train, predictors, target, min_samples=10
        )

        if train_prep is None:
            if self.verbose:
                print(f"  Warning: Insufficient training data")
            return None

        X_train, y_train = train_prep

        # 2. TEST DATA (from pre-split conventional_test)
        test_prep = self.foundation.prepare_dataset(
            self.conventional_test, predictors, target, min_samples=3
        )

        if test_prep is None:
            if self.verbose:
                print(f"  Warning: Insufficient test data")
            return None

        X_test, y_test = test_prep

        # 3. ✅ MATCHED DATA (from foundation)
        matched_dataset = self.foundation.get_matched_dataset(tuple(predictors))

        if matched_dataset is None:
            if self.verbose:
                print(f"  Warning: No matched dataset for {predictors}")
            return None

        matched_prep = self.foundation.prepare_dataset(
            matched_dataset, predictors, target, min_samples=3
        )

        if matched_prep is None:
            if self.verbose:
                print(f"  Warning: Insufficient matched data")
            return None

        X_matched, y_matched = matched_prep

        # 4. ✅ ALTERNATIVE DATA (from foundation)
        alt_prep = self.foundation.prepare_dataset(
            self.alternative, predictors, target, min_samples=3
        )

        if alt_prep is not None:
            X_alt, y_alt = alt_prep

        else:
            if self.verbose:
                print(f"  ⚠️  {predictors} → {target}: Alternative data unavailable")
            X_alt, y_alt = None, None



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
                                   rf_model: RandomForestRegressor,
                                   reference_mean:float) -> Dict:
        """Calculate comprehensive metrics for all four datasets."""
        result = {
            'model_type': 'random_forest',
            'predictors': predictors,
            'n_predictors': len(predictors),
        }

        # Metrics for each dataset
        for dataset_name in ['train', 'test', 'matched', 'alternative']:
            y_true = actuals[dataset_name]
            y_pred = predictions[dataset_name]

            if y_true is None or y_pred is None:
                result[f'{dataset_name}_r2'] = np.nan
                result[f'{dataset_name}_rmse'] = np.nan
                result[f'{dataset_name}_rrmse'] = np.nan
                result[f'{dataset_name}_mae'] = np.nan
                result[f'{dataset_name}_mape'] = np.nan
                result[f'{dataset_name}_n'] = 0
                continue

            metric_calculator = RegressionMetrics(y_true, y_pred)

            result[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            result[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            result[f'{dataset_name}_rrmse'] = metric_calculator.calculate_rrmse(y_mean=reference_mean)
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
    # HELPER METHODS
    # ========================================================================

    def _dataframe_to_dict(self, results_df: pd.DataFrame) -> Dict:
        """Convert results DataFrame to nested dict for backward compatibility."""
        results_dict = {}

        for target in results_df['target'].unique():
            target_df = results_df[results_df['target'] == target]
            results_dict[target] = {}

            for _, row in target_df.iterrows():
                predictor_str = row['predictor_str']
                metrics = row.drop(['target', 'predictor_str']).to_dict()
                results_dict[target][predictor_str] = metrics

        return results_dict

    # ========================================================================
    # CONVENIENCE METHODS (delegating to display/plotting classes)
    # ========================================================================

    def interpret_results(self, **kwargs):
        """Interpret results (wrapper for backward compatibility)."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_interpretation(**kwargs)

    def print_initialization_summary(self):
        """Print initialization summary."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_initialization_summary()

    # (Include all plotting methods from previous version - plot_domain_shift_analysis,
    # plot_prediction_scatter, plot_residuals, _convert_to_fourway_metrics,
    # _create_prediction_data, _get_all_predictions)
    # These remain unchanged from your original code

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

    def run_significance_tests(self,
                               comparison: str = 'fuel_type',
                               n_bootstrap: int = 5000,
                               confidence_level: float = 0.95,
                               ci_method: str = 'bca',
                               random_seed: Optional[int] = None,
                               debug: bool = False) -> List[Dict]:
        """
        Bootstrap significance tests for RF model RRMSE gaps.

        Identical logic to ParametricDomainAnalyzer.run_significance_tests()
        but uses RandomForestRegressor instead of ModelFactory.

        Step-by-step data flow:
        -----------------------
        1. self.results_df               → identifies (predictor, target) combinations
        2. self.foundation               → retrieves datasets (test / matched / alternative)
        3. RandomForestRegressor.fit()   → trains RF on self.conventional_train
        4. predict_fn                    → wraps rf.predict for tester
        5. TrainedModelBootstrapTester   → resamples eval sets, builds delta CI

        Dataset pairs by comparison type:
        ----------------------------------
        'distribution': conventional_test (dataset_a)  vs matched       (dataset_b)
        'fuel_type':    matched            (dataset_a)  vs alternative   (dataset_b)
        'total':        conventional_test  (dataset_a)  vs alternative   (dataset_b)

        Parameters
        ----------
        comparison : str
            Which gap to test: 'fuel_type', 'distribution', or 'total'
        n_bootstrap : int
            Number of bootstrap resamples
        confidence_level : float
            CI confidence level
        ci_method : str
            'percentile' or 'bca'
        random_seed : int, optional
            For reproducibility
        debug : bool
            Print per-combination debug output

        Returns
        -------
        list[dict]
            One dict per (predictor, target) combination tested.
        """
        from sklearn.ensemble import RandomForestRegressor
        from knowledge_layer.multi_dataset_analysers.TrainedModelBootstrapTester import (
            TrainedModelBootstrapTester
        )

        if self.results_df is None or len(self.results_df) == 0:
            raise RuntimeError("No results found. Run analyze_all_combinations() first.")

        comparison_options = ['fuel_type', 'distribution', 'total']
        if comparison not in comparison_options:
            raise ValueError(f"comparison must be one of {comparison_options}")

        comparison_label_map = {
            'fuel_type': 'Matched vs Alternative',
            'distribution': 'Conv Test vs Matched',
            'total': 'Conv Test vs Alternative',
        }
        comparison_label = comparison_label_map[comparison]

        # ── Reduce to one best combination per target ─────────────────────────
        best_rows = (
            self.results_df
            .sort_values('test_rrmse', ascending=True)
            .groupby('target', as_index=False)
            .first()
        )

        if debug:
            print(f"\n[run_significance_tests] Using {len(best_rows)} combinations "
                  f"(1 per target, selected by test_rrmse)")
            for _, r in best_rows.iterrows():
                print(f"  {r['predictor_str']:30s} → {r['target']} "
                      f"(test_rrmse={r['test_rrmse']:.3f}%)")

        tester = TrainedModelBootstrapTester(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            ci_method=ci_method,
            random_seed=random_seed,
        )

        results = []

        for _, row in best_rows.iterrows():
            target = row['target']
            predictors = list(row['predictors'])
            predictor_str = row['predictor_str']
            model_type = 'random_forest'

            if debug:
                print(f"\nTesting: {predictor_str} → {target}")

            data_prep = self._prepare_fourway_data(predictors, target)
            if data_prep is None:
                results.append(tester._insufficient_data_result(
                    target, predictor_str, model_type, comparison_label
                ))
                continue

            if comparison == 'distribution':
                X_a, y_a = data_prep['X_test'], data_prep['y_test']
                X_b, y_b = data_prep['X_matched'], data_prep['y_matched']
            elif comparison == 'fuel_type':
                X_a, y_a = data_prep['X_matched'], data_prep['y_matched']
                X_b, y_b = data_prep['X_alternative'], data_prep['y_alternative']
            else:
                X_a, y_a = data_prep['X_test'], data_prep['y_test']
                X_b, y_b = data_prep['X_alternative'], data_prep['y_alternative']

            if y_a is None or y_b is None:
                results.append(tester._insufficient_data_result(
                    target, predictor_str, model_type, comparison_label
                ))
                continue

            try:
                rf = RandomForestRegressor(**self.rf_params)
                rf.fit(data_prep['X_train'], data_prep['y_train'])
            except Exception as e:
                if debug:
                    print(f"  RF fitting failed: {e}")
                results.append(tester._insufficient_data_result(
                    target, predictor_str, model_type, comparison_label
                ))
                continue

            def predict_fn(X, _rf=rf):
                return _rf.predict(X)

            reference_mean = float(self.conventional_full[target].dropna().mean())

            result = tester.test_gap(
                predict_fn=predict_fn,
                X_a=X_a, y_a=y_a,
                X_b=X_b, y_b=y_b,
                reference_mean=reference_mean,
                comparison_label=comparison_label,
                target=target,
                predictor_str=predictor_str,
                model_type=model_type,
                debug=debug,
            )
            results.append(result)

        return results

    def export_significance_results(self,
                                    significance_results: List[Dict],
                                    output_path: str) -> None:
        """Export bootstrap significance results to Excel."""
        from knowledge_layer.multi_dataset_analysers.TrainedModelBootstrapTester import (
            TrainedModelBootstrapTester
        )
        tester = TrainedModelBootstrapTester()
        tester.export_to_excel(significance_results, output_path)


    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def export_results(self,
                       filename: str = 'rf_domain_analysis_results.csv',
                       include_all_metrics: bool = True):
        """Export results DataFrame to CSV."""
        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run analyze_all_combinations() first.")

        if include_all_metrics:
            export_df = self.results_df
        else:
            key_cols = [
                'target', 'predictor_str', 'n_predictors', 'model_type',
                'alternative_r2', 'alternative_rrmse', 'alternative_mape',
                'gap_fuel_type', 'gap_fuel_type_rrmse', 'feature_importance'
            ]
            existing_cols = [col for col in key_cols if col in self.results_df.columns]
            export_df = self.results_df[existing_cols]

        export_df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
        print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")