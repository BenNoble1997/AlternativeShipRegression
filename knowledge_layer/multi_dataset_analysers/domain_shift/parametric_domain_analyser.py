# File: parametric_regression_analyser_class.py

"""
Domain Shift Analyzer for Parametric Regression
===============================================
Analyzes domain shift from conventional to alternative fuel ships.

Uses ParametricAnalysisFoundation via composition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from config.loggermixin.logger_config import LoggerMixin
from knowledge_layer.multi_dataset_analysers.parametric_analysis_foundation import ParametricAnalysisFoundation
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class import (
    AnalysisPlotter, ResidualData, PredictionData, FourWayMetrics, DatasetMetrics
)
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility
from knowledge_layer.data_split_handling_class import DataSplitHandler
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class ParametricDomainAnalyzer(LoggerMixin):
    """
    Analyzes domain shift from conventional to alternative fuel ships.

    Uses ParametricAnalysisFoundation via composition.

    Experimental Design:
    -------------------
    1. Split conventional data: 80% train, 20% test
    2. Train models on conventional train set
    3. Evaluate on four datasets:
       - Conv Train (training performance)
       - Conv Test (in-domain generalization)
       - Conv Matched (distribution shift control)
       - Alternative (out-of-domain + fuel type effect)

    Key Metrics:
    -----------
    - Conv Train → Conv Test gap: Normal generalization error
    - Conv Test → Conv Matched gap: Distribution shift effect
    - Conv Matched → Alternative gap: FUEL TYPE EFFECT (what we care about!)
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 independent_variables: List[str],
                 dependent_variables: Optional[List[str]] = None,
                 model_recommendations: Optional[Dict] = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True,
                 selection_metric: str = 'rrmse',  # Default to BIC for consistency
                 selection_strategy: str = 'conservative',
             target_predictor_constraints: Optional[Dict[str, List[str]]] = None,
                 build_year_col:str|None = None,
                 year_edges:list=None,
                 build_eligibility:TemporalEligibility|None=None,
                 matching_bins:dict[str, list]=None,
                 n_matches_per_ship=10,
                 caliper:float|None=None):

        """
    Initialize Domain Analyzer.

    Order of operations matters here: the foundation is constructed with
    ``defer_matched_sets=True``, the conventional data is split into
    train/test, and only then are the matched datasets built — using the
    held-out test partition as the candidate pool. Matched-set evaluation
    is therefore strictly out-of-sample. Matched ships may also appear in
    the test metric; this is overlap between two evaluation views of the
    same untrained-on population, not leakage.

    :param conventional_data: Full conventional ship dataset (split 80/20
        internally).
    :type conventional_data: pandas.DataFrame
    :param alternative_data: Alternative fuel ships dataset.
    :type alternative_data: pandas.DataFrame
    :param independent_variables: Potential predictor dimensions.
    :type independent_variables: list[str]
    :param dependent_variables: Target dimensions. If ``None``, uses
        ``independent_variables``.
    :type dependent_variables: list[str] or None
    :param model_recommendations: Selector output
        ``{(predictor_tuple, target): model_type}``. Required.
    :type model_recommendations: dict or None
    :param test_size: Fraction of conventional data held out for testing
        (and used as the matched-set candidate pool).
    :type test_size: float
    :param random_state: Random seed for reproducibility. Must be shared
        with any sibling analyzer (e.g. Random Forest) so both use the
        identical split and identical matched sets.
    :type random_state: int
    :param verbose: Whether to print progress and warnings.
    :type verbose: bool
    :param selection_metric: Metric for selecting best combinations.
    :type selection_metric: str
    :param selection_strategy: Strategy for model selection.
    :type selection_strategy: str
    :param target_predictor_constraints: Allowed predictors per target.
    :type target_predictor_constraints: dict[str, list[str]] or None
    """
        # ✅ Initialize foundation for data management
        self.foundation = ParametricAnalysisFoundation(
            conventional_data=conventional_data,
            alternative_data=alternative_data,
            independent_variables=independent_variables,
            dependent_variables=dependent_variables,
            model_recommendations=model_recommendations,
            random_state=random_state,
            verbose=verbose,
            selection_metric=selection_metric,
            selection_strategy=selection_strategy,
            target_predictor_constraints=target_predictor_constraints,
            defer_matched_sets=True,
            build_year_col=build_year_col,
            year_edges=year_edges,
            build_eligibility=build_eligibility,
            n_matches_per_ship=n_matches_per_ship,
            caliper=caliper,
            matching_bins=matching_bins,
        )

        if hasattr(self, 'selector'):
            self.foundation.selector = self.selector

        # Store local configuration
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.analysis_plotter = AnalysisPlotter()
        self.selection_metric = selection_metric
        self.selection_strategy = selection_strategy

        # Split conventional data into train/test
        self._split_conventional_data()

        self.foundation.build_matched_sets(candidate_pool = self.conventional_test,
                                           debug=self.verbose)

        # Storage
        self.results = {}
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
    def model_recommendations(self) -> Dict:
        """Model recommendations from foundation."""
        return self.foundation.model_recommendations

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
        """
        Fallback random split when stratification is unavailable or fails.

        Mirrors the RF analyzer's fallback exactly (index-based, indices
        stored) so both analyzers remain split-identical under a shared seed.

        :param conv_data: Conventional dataset to split.
        :type conv_data: pandas.DataFrame
        """
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
                                 display_results: bool = True) -> pd.DataFrame:
        """
        Analyze all predictor combinations with four-way evaluation.

        Uses foundation to extract valid combinations.

        Parameters
        ----------
        max_predictors : int, default=2
            Maximum number of predictors per combination
        display_results : bool, default=True
            Print results table

        Returns
        -------
        pd.DataFrame
            Results with comprehensive metrics
        """
        if display_results and self.verbose:
            print("\n" + "=" * 180)
            print(f"DOMAIN SHIFT ANALYSIS (Max Predictors: {max_predictors})")
            print("=" * 180)

        all_results = []

        for target in self.target_dimensions:
            # ✅ Use foundation to extract combinations
            pred_combos = self.foundation.extract_combinations_for_target(
                target=target,
                max_predictors=max_predictors
            )

            if not pred_combos:
                if self.verbose:
                    print(f"\n  ⚠️  No combinations found for {target} "
                          f"with ≤{max_predictors} predictors")
                continue

            if self.verbose:
                print(f"\nAnalyzing {target}: {len(pred_combos)} combinations")

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

        results_df = pd.DataFrame(all_results)

        # Reorder columns
        column_order = [
            'target', 'predictor_str', 'predictors', 'n_predictors', 'model_type',
            'train_r2', 'test_r2', 'matched_r2', 'alternative_r2',
            'train_rrmse', 'test_rrmse', 'matched_rrmse', 'alternative_rrmse',
            'train_mae', 'test_mae', 'matched_mae', 'alternative_mae',
            'train_mape', 'test_mape', 'matched_mape', 'alternative_mape',
            'train_n', 'test_n', 'matched_n', 'alternative_n',
            'gap_generalization', 'gap_distribution', 'gap_fuel_type', 'gap_total',
            'gap_generalization_rrmse', 'gap_distribution_rrmse',
            'gap_fuel_type_rrmse', 'gap_total_rrmse'
        ]

        remaining_cols = [col for col in results_df.columns if col not in column_order]
        column_order.extend(remaining_cols)
        existing_cols = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_cols]

        # Display results
        if display_results:
            from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
            display = AnalyzerResultsDisplay(self)
            self.results_df = results_df
            display.print_analysis_results(results_df)

        # Store results
        self.results_df = results_df
        self.results = self._dataframe_to_dict(results_df)

        return results_df

    # ========================================================================
    # FOUR-WAY TESTING
    # ========================================================================

    def _train_and_test_fourway(self,
                                predictors: List[str],
                                target: str) -> Optional[Dict]:
        """
        Train on the conventional train set and evaluate on up to four datasets.

        Evaluation datasets are train, test, matched, and alternative. The
        alternative dataset may legitimately be unavailable for a given
        (predictors, target) combination when too few alternative-fuel ships
        have complete data for those columns. In that case predictions are
        skipped for the alternative set and its metrics are reported as NaN,
        rather than failing the whole combination. This preserves the
        conventional-side results so that combination selection (which is
        based on conventional test performance) remains valid.

        :param predictors: Predictor column names for this combination.
        :type predictors: list[str]
        :param target: Target column name.
        :type target: str
        :returns: Dictionary of comprehensive metrics including fuel-type
            effect isolation, or ``None`` if the model could not be fitted
            or conventional data was insufficient.
        :rtype: dict or None
        """

        # ✅ Get model type from foundation
        model_type = self.foundation.get_model_type(tuple(predictors), target)

        # Prepare data from all four datasets
        data_prep = self._prepare_fourway_data(predictors, target)
        if data_prep is None:
            return None

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        # Getting referene y_mean value
        reference_mean = self.conventional_full[target].dropna().mean()

        # ========================================================================
        # ✅ NEW: Check if we have a ModelDescriptor with discovered structure
        # ========================================================================

        descriptor = self._get_model_descriptor(tuple(predictors), target)

        try:
            if descriptor is not None and self._should_use_descriptor(model_type):
                # Use descriptor to refit with FIXED structure
                if self.verbose:
                    print(f"    Using fixed structure from selector for {model_type}")

                model, preprocessor = descriptor.refit(
                    X=X_train,
                    y=y_train,
                    force_structure=True  # ✅ KEY: Force same structure
                )
            else:
                # Standard fitting (structure may vary)
                model, preprocessor = ModelFactory.create_and_fit(
                    model_type, X_train, y_train
                )

            # Predict on all datasets that are actually available
            datasets_to_predict = {
                'train': X_train,
                'test': data_prep['X_test'],
                'matched': data_prep['X_matched'],
                'alternative': data_prep['X_alternative']
            }

            predictions = {}
            for dataset_name, X_data in datasets_to_predict.items():

                if X_data is None:
                    predictions[dataset_name] = None
                    continue

                predictions[dataset_name] = ModelFactory.predict(
                    model, preprocessor, X_data, model_type
                )

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            if self.verbose:
                print(f"  Model fitting failed: {str(e)[:80]}")
            return None

        # Calculate comprehensive metrics (unchanged)
        result = self._calculate_fourway_metrics(
            predictions,
            {
                'train': data_prep['y_train'],
                'test': data_prep['y_test'],
                'matched': data_prep['y_matched'],
                'alternative': data_prep['y_alternative']
            },
            model_type,
            predictors,
            reference_mean=reference_mean
        )

        return result

    def _prepare_fourway_data(self,
                              predictors: List[str],
                              target: str) -> Optional[Dict]:
        """
        Prepare data from all four datasets with proper cleaning.

        Uses:
        - self.conventional_train (already split)
        - self.conventional_test (already split)
        - foundation.get_matched_dataset()
        - foundation.alternative
        """
        # 1. TRAIN DATA (from pre-split conventional_train)
        train_prep = self.foundation.prepare_dataset(
            dataset=self.conventional_train, predictors=predictors, target=target, min_samples=10
        )

        if train_prep is None:
            if self.verbose:
                print(f"  Warning: Insufficient training data")
            return None

        X_train, y_train = train_prep

        # 2. TEST DATA (from pre-split conventional_test)
        test_prep = self.foundation.prepare_dataset(
            dataset=self.conventional_test, predictors=predictors, target=target, min_samples=3
        )

        if test_prep is None:
            if self.verbose:
                print(f"  Warning: Insufficient test data")
            return None

        X_test, y_test = test_prep

        # 3. ✅ MATCHED DATA (from foundation)
        matched_dataset = self.foundation.get_matched_dataset(tuple(predictors))

        if matched_dataset is None:
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
                                   model_type: str,
                                   predictors: List[str],
                                   reference_mean:float) -> Dict:
        """
        Calculate comprehensive metrics for all available datasets.

        Datasets whose actuals or predictions are ``None`` (typically the
        alternative set when it has too few ships for the current predictor
        combination) receive ``NaN`` for every metric and ``0`` for the
        sample count. Gap metrics involving an unavailable dataset naturally
        propagate as ``NaN``, which pandas sorting and aggregation handle
        gracefully (NaN rows sort last and are excluded from means).

        :param predictions: Mapping of dataset name to predicted values
            (or ``None`` if that dataset was unavailable).
        :type predictions: dict
        :param actuals: Mapping of dataset name to true values (or ``None``).
        :type actuals: dict
        :param model_type: Model type string used for fitting.
        :type model_type: str
        :param predictors: Predictor names for this combination.
        :type predictors: list[str]
        :param reference_mean: Conventional-dataset target mean used to
            normalise RRMSE for cross-dataset comparability.
        :type reference_mean: float
        :returns: Flat dictionary of per-dataset metrics and gap metrics.
        :rtype: dict
        """
        result = {
            'model_type': model_type,
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

            metrics_calc = RegressionMetrics(y_true, y_pred)

            result[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            result[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            result[f'{dataset_name}_rrmse'] = metrics_calc.calculate_rrmse(y_mean=reference_mean)
            result[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
            result[f'{dataset_name}_mape'] = metrics_calc.calculate_mape()
            result[f'{dataset_name}_n'] = len(y_true)

        # R² GAPS
        result['gap_generalization'] = result['train_r2'] - result['test_r2']
        result['gap_distribution'] = result['test_r2'] - result['matched_r2']
        result['gap_fuel_type'] = result['matched_r2'] - result['alternative_r2']
        result['gap_total'] = result['test_r2'] - result['alternative_r2']

        # RRMSE GAPS (reversed sign)
        result['gap_generalization_rrmse'] = result['test_rrmse'] - result['train_rrmse']
        result['gap_distribution_rrmse'] = result['matched_rrmse'] - result['test_rrmse']
        result['gap_fuel_type_rrmse'] = result['alternative_rrmse'] - result['matched_rrmse']
        result['gap_total_rrmse'] = result['alternative_rrmse'] - result['test_rrmse']

        return result

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _dataframe_to_dict(self, results_df: pd.DataFrame) -> Dict:
        """Convert results DataFrame to nested dict."""
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

    def print_initialization_summary(self):
        """Print initialization summary."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_initialization_summary()

    def print_analysis_results(self, **kwargs):
        """Print results table."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_analysis_results(**kwargs)

    def print_interpretation(self, **kwargs):
        """Print interpretation."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_interpretation(**kwargs)

    def print_best_combinations(self, **kwargs):
        """Print best combinations."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_best_combinations(**kwargs)

    def interpret_results(self, **kwargs):
        """Interpret results (wrapper for backward compatibility)."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_interpretation(**kwargs)

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================

    def plot_domain_shift_analysis(self, save_path: Optional[str] = None):
        """Plot domain shift visualization."""
        if not self.results or len(self.results) == 0:
            print("No results to plot. Run analyze_all_combinations() first.")
            return

        targets_data = {}

        for target, models in self.results.items():
            if not models:
                continue

            best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo

            fourway_metrics = self._convert_to_fourway_metrics(metrics)

            targets_data[target] = (
                metrics['predictors'],
                metrics['model_type'],
                fourway_metrics
            )

        if not targets_data:
            print("No valid data to plot!")
            return

        fig = self.analysis_plotter.domain_shift_plotter.plot_multiple_targets(
            targets_data=targets_data,
            save_path=save_path
        )

        plt.show()

        if save_path:
            print(f"✅ Plot saved to {save_path}")

    def plot_prediction_scatter(self,
                                save_dir: Optional[str] = None,
                                figsize: Tuple = (12, 12)):
        """Plot prediction scatter plots."""
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

            data_prep = self._prepare_fourway_data(predictors, target)
            if data_prep is None:
                continue

            predictions_dict = self._get_all_predictions(data_prep, predictors, model_type)
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
                print(f"  ✅ Saved: {Path(save_file).name}")

    def plot_residuals(self, save_dir: Optional[str] = None):
        """Plot residual analysis."""
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

            data_prep = self._prepare_fourway_data(predictors, target)
            if data_prep is None:
                continue

            predictions_dict = self._get_all_predictions(data_prep, predictors, model_type)
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
                print(f"  ✅ Saved: {Path(save_file).name}")

        if save_dir:
            print(f"\n✅ All residual plots saved to {save_dir}")

    # ========================================================================
    # HELPER METHODS FOR PLOTTING
    # ========================================================================

    def _convert_to_fourway_metrics(self, metrics: Dict) -> FourWayMetrics:
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

    def _create_prediction_data(self,
                                dataset_name: str,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                r2: float,
                                rrmse: float,
                                n: int) -> PredictionData:
        """Create PredictionData dataclass."""
        return PredictionData(
            y_true=y_true,
            y_pred=y_pred,
            r2=r2,
            rrmse=rrmse,
            n_samples=n,
            dataset_name=dataset_name
        )

    def _get_all_predictions(self,
                            data_prep: Dict,
                            predictors: List[str],
                            model_type: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit the model on the training split and predict on all available datasets.

        Datasets with ``None`` entries in ``data_prep`` (e.g. an alternative
        set that was too small for this combination) are returned as ``None``
        in the predictions dictionary rather than raising. Plotting callers
        must check for ``None`` before rendering the corresponding panel.

        :param data_prep: Output of :meth:`_prepare_fourway_data`.
        :type data_prep: dict
        :param predictors: Predictor names for this combination.
        :type predictors: list[str]
        :param model_type: Model type string used for fitting.
        :type model_type: str
        :returns: Mapping of dataset name to predictions (or ``None`` per
            dataset), or ``None`` entirely if model fitting failed.
        :rtype: dict[str, numpy.ndarray or None] or None
        """
        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        try:
            model, preprocessor = ModelFactory.create_and_fit(model_type, X_train, y_train)

            predictions = {}
            for dataset_name in ['train', 'test', 'matched', 'alternative']:
                X_data = data_prep[f'X_{dataset_name}']
                if X_data is None:
                    predictions[dataset_name] = None
                    continue

                predictions[dataset_name] = ModelFactory.predict(
                    model, preprocessor, X_data, model_type
                )

            return predictions

        except Exception as e:
            if self.verbose:
                print(f"  Error getting predictions: {e}")
            return None

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_best_combinations(self,
                              metric: str = 'test_rrmse',
                              ascending: bool = True) -> pd.DataFrame:
        """Get best predictor combination for each target."""
        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run analyze_all_combinations() first.")

        best_results = (self.results_df
                        .sort_values(metric, ascending=ascending)
                        .groupby('target')
                        .first()
                        .reset_index())

        return best_results

    def filter_results(self,
                       targets: Optional[List[str]] = None,
                       min_predictors: Optional[int] = None,
                       max_predictors: Optional[int] = None,
                       model_types: Optional[List[str]] = None,
                       min_r2: Optional[float] = None,
                       max_rrmse: Optional[float] = None) -> pd.DataFrame:
        """Filter results DataFrame based on criteria."""
        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run analyze_all_combinations() first.")

        filtered = self.results_df.copy()

        if targets is not None:
            filtered = filtered[filtered['target'].isin(targets)]

        if min_predictors is not None:
            filtered = filtered[filtered['n_predictors'] >= min_predictors]

        if max_predictors is not None:
            filtered = filtered[filtered['n_predictors'] <= max_predictors]

        if model_types is not None:
            filtered = filtered[filtered['model_type'].isin(model_types)]

        if min_r2 is not None:
            filtered = filtered[filtered['alternative_r2'] >= min_r2]

        if max_rrmse is not None:
            filtered = filtered[filtered['alternative_rrmse'] <= max_rrmse]

        return filtered

    def export_results(self,
                       filename: str = 'domain_analysis_results.csv',
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
                'gap_fuel_type', 'gap_fuel_type_rrmse'
            ]
            existing_cols = [col for col in key_cols if col in self.results_df.columns]
            export_df = self.results_df[existing_cols]

        export_df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
        print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def _get_model_descriptor(self,
                              predictors: Tuple[str, ...],
                              target: str) -> Optional['ModelDescriptor']:
        """
        Retrieve ModelDescriptor from selector's stored descriptors.

        The selector stores descriptors in:
        self.foundation.selector.model_descriptors[key][model_type]

        where key = (predictors_tuple, target)
        """
        key = (predictors, target)

        # Check if selector has model_descriptors attribute
        if not hasattr(self.foundation, 'selector'):
            return None

        selector = self.foundation.selector

        if not hasattr(selector, 'model_descriptors'):
            return None

        if key not in selector.model_descriptors:
            return None

        # Get the recommended model type for this combination
        model_type = self.foundation.get_model_type(predictors, target)

        # Get descriptor for this specific model type
        descriptors_for_key = selector.model_descriptors[key]

        if model_type not in descriptors_for_key:
            return None

        return descriptors_for_key[model_type]

    def _should_use_descriptor(self, model_type: str) -> bool:
        """
        Determine if we should use descriptor's fixed structure.

        **Use descriptor for:**
        - MFP (fixed transformations)
        - Lasso variants (fixed active features)

        **Don't use descriptor for:**
        - Linear, Ridge (structure doesn't change anyway)
        - Polynomial (structure is deterministic from degree)
        - Power, exponential, etc. (structure is fixed)

        Returns
        -------
        bool
            True if descriptor should be used for structure preservation
        """
        structure_discovering_models = [
            'mfp',
            'lasso',
            'lasso-cv',
            'polynomial-2-lasso',
            'polynomial-3-lasso',
            'polynomial-2-lasso-cv',
            'polynomial-3-lasso-cv',
        ]

        return model_type in structure_discovering_models

    def run_significance_tests(self,
                               comparison: str = 'fuel_type',
                               n_bootstrap: int = 5000,
                               confidence_level: float = 0.95,
                               ci_method: str = 'bca',
                               random_seed: Optional[int] = None,
                               debug: bool = False) -> List[Dict]:
        """
        Bootstrap significance tests for RRMSE gaps across evaluation datasets.

        Uses a fixed trained model (no retraining during bootstrap).
        Resamples the two relevant EVALUATION sets independently to build
        a distribution of the RRMSE difference.

        Step-by-step data flow:
        -----------------------
        1. self.results_df               → identifies best (predictor, target) per target
        2. self.foundation               → retrieves datasets (test / matched / alternative)
        3. ModelFactory.create_and_fit() → trains model on self.conventional_train
        4. predict_fn                    → wraps trained model for tester
        5. TrainedModelBootstrapTester   → resamples eval sets, builds delta CI
        6. Return list of result dicts

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
        # Uses test_rrmse (conventional generalisation) — consistent with how
        # analyze_all_datasets() selects best combinations. If results_df already
        # contains only one row per target (best_only=True was used correctly)
        # this is a no-op. If it contains multiple rows this picks the best one,
        # preventing redundant and misleading multiple tests per target.
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
            model_type = row['model_type']
            predictor_str = row['predictor_str']

            if debug:
                print(f"\nTesting: {predictor_str} → {target} ({model_type})")

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
            else:  # total
                X_a, y_a = data_prep['X_test'], data_prep['y_test']
                X_b, y_b = data_prep['X_alternative'], data_prep['y_alternative']

            if y_a is None or y_b is None:
                results.append(tester._insufficient_data_result(
                    target, predictor_str, model_type, comparison_label
                ))
                continue

            descriptor = self._get_model_descriptor(tuple(predictors), target)

            try:
                if descriptor is not None and self._should_use_descriptor(model_type):
                    model, preprocessor = descriptor.refit(
                        X=data_prep['X_train'],
                        y=data_prep['y_train'],
                        force_structure=True
                    )
                else:
                    model, preprocessor = ModelFactory.create_and_fit(
                        model_type,
                        data_prep['X_train'],
                        data_prep['y_train']
                    )
            except Exception as e:
                if debug:
                    print(f"  Model fitting failed: {e}")
                results.append(tester._insufficient_data_result(
                    target, predictor_str, model_type, comparison_label
                ))
                continue

            def predict_fn(X, _m=model, _p=preprocessor, _mt=model_type):
                return ModelFactory.predict(_m, _p, X, _mt)

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