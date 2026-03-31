import pandas as pd
import numpy as np
from pathlib import Path
from config import LoggerMixin
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List
import matplotlib.pyplot as plt
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.data_split_handling_class import DataSplitHandler

from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class import AnalysisPlotter,ResidualData,PredictionData,FourWayMetrics,DatasetMetrics

class ParametricDomainAnalyzer(LoggerMixin):
    """
    Analyzes domain shift from conventional to alternative fuel ships.

    **REFACTORED:** Now extracts combinations directly from model_recommendations
    rather than generating them independently. This ensures perfect alignment
    with the selector and eliminates "No recommendation" warnings.

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
                 independent_variables: list[str],
                 dependent_variables: list[str] = None,
                 model_recommendations: dict = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True,
                 selection_metric: str = 'bic',  # Default to BIC for consistency
                 selection_strategy: str = 'conservative'):

        """
        Initialize Domain Analyzer.

        **REFACTORED:** Simplified initialization - only needs model_recommendations
        dictionary from selector. No need to pass relationship_context,
        forbidden_combinations, etc. since we extract tested combinations directly.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset (will be split 80/20)
        conventional_matched : pd.DataFrame
            Conventional ships matched to alternative ships (control group)
        alternative_data : pd.DataFrame
            Alternative fuel ships (experimental group)
        independent_variables : List[str]
            List of potential predictor dimensions
        dependent_variables : List[str], optional
            List of target dimensions. If None, uses variable_dimensions.
        model_recommendations : dict
            Dictionary from PolynomialModelSelector.model_recommendations
            Format: {(predictor_tuple, target): model_type_string}
            **CRITICAL:** This is now REQUIRED for proper operation.
        test_size : float, default=0.2
            Fraction for train/test split of conventional data
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Whether to print progress and warnings

        selection_metric : str, default='bic'
            Metric to use for selecting best combinations:
            - 'adj_r2': Adjusted R² (higher is better)
            - 'r2': R² (higher is better)
            - 'aic': Akaike Information Criterion (lower is better)
            - 'bic': Bayesian Information Criterion (lower is better)
            - 'rrmse': Relative RMSE% (lower is better)
            - 'mape': Mean Absolute Percentage Error% (lower is better)
        selection_strategy : str, default='conservative'
            Strategy for model selection:
            - 'simple': Pick best by metric
            - 'conservative': Use equivalence thresholds, prefer simpler
            - 'nested_aware': Special handling for nested models

        Raises
        ------
        ValueError
            If model_recommendations is None or empty

        Notes
        -----
        **Design Change:** The analyzer now operates as a strict pipeline:
        1. PolynomialModelSelector tests combinations and selects models
        2. PolynomialDomainAnalyzer tests those exact combinations on alternative fuel ships

        This ensures:
        - No "No recommendation" warnings
        - Perfect alignment between selector and analyzer
        - No duplicate computation of valid combinations
        - All tested combinations use optimal models (no fallback to linear)
        """

        # Store basic configuration
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.analysis_plotter = AnalysisPlotter()

        if dependent_variables is None:
            dependent_variables = independent_variables

        all_dimensions = list(set(independent_variables + dependent_variables))
        available_dimensions = [d for d in all_dimensions if d in conventional_data.columns]
        missing_dimensions = [d for d in all_dimensions if d not in conventional_data.columns]

        if missing_dimensions and verbose:
            print(f"\n  ⚠️  Warning: Some dimensions not found in training_data:")
            for dim in missing_dimensions:
                print(f"    - '{dim}'")
            print(f"  Using {len(available_dimensions)} available dimensions\n")

        if 'DWT' not in available_dimensions:
            raise ValueError("DWT must be in the training data!")

        self.variable_dimensions = [d for d in independent_variables if d in available_dimensions]
        self.target_dimensions = [d for d in dependent_variables if d in available_dimensions]

        self.conventional_full = conventional_data[available_dimensions].copy()
        # self.conventional_matched = conventional_matched[available_dimensions].copy()
        self.alternative = alternative_data[available_dimensions].copy()


        if model_recommendations is None or len(model_recommendations) == 0:
            raise ValueError(
                "model_recommendations is required and cannot be empty!\n"
                "Please run PolynomialModelSelector.analyze_all_combinations() first,\n"
                "then pass selector.model_recommendations to this analyzer."
            )

        self.model_recommendations = model_recommendations
        self._validate_data_quality()

        self.matched_data_sets = self._create_matched_sets()

        self.conventional_train = None
        self.conventional_test = None
        self.results = {}
        self.results_df = None

    def print_initialization_summary(self):
        """Print initialization summary."""
        if not self.verbose:
            return

        print(f"\nDWT-Constrained Domain Analyzer initialized:")
        print(f"  Conventional full: {len(self.conventional_full)} ships")
        print(f"  Alternative: {len(self.alternative)} ships")
        print(f"  Test size: {self.test_size * 100:.0f}%")
        print(f"  Model recommendations: {len(self.model_recommendations)} combinations")

        # Show breakdown by target
        targets_in_recommendations = set(tgt for (_, tgt) in self.model_recommendations.keys())
        print(f"  Targets with recommendations: {len(targets_in_recommendations)}")
        for target in sorted(targets_in_recommendations):
            n_combos = sum(1 for (_, tgt) in self.model_recommendations.keys() if tgt == target)
            print(f"    • {target}: {n_combos} combinations")


    def _validate_data_quality(self):
        """Check data quality and warn about issues."""
        if not self.verbose:
            return

        for dataset_name, dataset in [
            ('conventional', self.conventional_full),
            # ('matched', self.conventional_matched),
            ('alternative', self.alternative)
        ]:
            # Check for NaN
            nan_counts = dataset.isna().sum()
            if nan_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has missing values:")
                for col, count in nan_counts[nan_counts > 0].items():
                    pct = 100 * count / len(dataset)
                    print(f"    {col}: {count} ({pct:.1f}%)")

            # Check for inf
            inf_mask = np.isinf(dataset.select_dtypes(include=[np.number]))
            inf_counts = inf_mask.sum()
            if inf_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has infinite values:")
                for col, count in inf_counts[inf_counts > 0].items():
                    print(f"    {col}: {count}")

            # Check minimum size
            if len(dataset) < 10:
                print(f"\n  ⚠️  WARNING: {dataset_name} dataset is very small ({len(dataset)} ships)")

    # ==================================  ======================================
    #COMBINATION EXTRACTION FROM RECOMMENDATIONS
    # ========================================================================

    def _extract_combinations_from_recommendations(self,
                                                   target: str,
                                                   max_predictors: int|None = None) -> list[tuple[str, ...]]:
        """
        Extract predictor combinations for a target from model recommendations.

        **KEY METHOD:** This replaces the old _get_available_predictors() method
        and combination generation logic. Instead of generating combinations
        independently, we extract the exact combinations that the selector tested.

        Parameters
        ----------
        target : str
            Target dimension
        max_predictors : int, optional
            Filter to combinations with at most this many predictors.
            If None, returns all combinations.

        Returns
        -------
        list of tuples
            Predictor combinations that have recommendations from the selector,
            sorted by (number of predictors, predictor names)

        Examples
        --------
        >>> # Selector tested: [('DWT',), ('DWT', 'LOA'), ('BEAM', 'DRAFT')]
        >>> combos = analyzer._extract_combinations_from_recommendations('Main_Power')
        >>> print(combos)
        [('DWT',), ('BEAM', 'DRAFT'), ('DWT', 'LOA')]

        Notes
        -----
        This ensures:
        - Perfect alignment with selector (tests exactly what selector tested)
        - No "No recommendation" warnings (all combos have recommendations)
        - No fallback to linear models (all combos have optimal model selected)
        - No duplicate computation (selector already validated combinations)
        """
        combos = []

        for (pred_combo, tgt) in self.model_recommendations.keys():
            if tgt == target:
                # Apply max_predictors filter if specified
                if max_predictors is None or len(pred_combo) <= max_predictors:
                    combos.append(pred_combo)

        # Sort by number of predictors, then alphabetically
        combos.sort(key=lambda x: (len(x), x))

        return combos

    def _create_matched_sets(self):
        """
        Helper method for creating matched datasets for each independent variable combination
        """

        matched_sets = {}
        combo_list = []
        for target in self.target_dimensions:
            combo = self._extract_combinations_from_recommendations(target=target)
            combo_list.extend(combo)

        sorted_list = sorted(set(combo_list))

        return MatchedDataset.multiple_matched_datasets(alternative_fleet=self.alternative,conventional_fleet=self.conventional_full,
                                                        logger=self.logger,predictor_groups=sorted_list,std_reference='conventional')


    def get_combinations_summary(self) -> pd.DataFrame:
        """
        Get summary of available combinations from recommendations.

        Returns
        -------
        pd.DataFrame
            Summary with columns: target, n_combinations, predictor_sizes

        Examples
        --------
        >>> summary = analyzer.get_combinations_summary()
        >>> print(summary)
           target              n_combinations  predictor_sizes
        0  DWT                 15              [1, 2]
        1  Main_Power          21              [1, 2]
        2  Speed               15              [1, 2]
        """
        summary_data = []

        for target in self.target_dimensions:
            combos = self._extract_combinations_from_recommendations(target)

            if combos:
                sizes = sorted(list(set(len(c) for c in combos)))
                summary_data.append({
                    'target': target,
                    'n_combinations': len(combos),
                    'predictor_sizes': sizes
                })

        return pd.DataFrame(summary_data)

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_combinations(self,
                                 max_predictors: int = 2,
                                 display_results: bool = True) -> pd.DataFrame:
        """
        Analyze all predictor combinations with four-way evaluation.

        **REFACTORED:** Now extracts combinations from model_recommendations
        instead of generating them independently. This ensures we test exactly
        what the selector tested, eliminating warnings and inconsistencies.

        Parameters
        ----------
        max_predictors : int, default=2
            Filter to combinations with at most this many predictors.
            Use this to limit analysis scope (e.g., only single predictors).
        verbose : bool, default=True
            If True, prints results table during analysis

        Returns
        -------
        pd.DataFrame
            Results with columns:
            - target: Target dimension
            - predictors: Predictor combination (as list)
            - predictor_str: Predictor combination (as string)
            - n_predictors: Number of predictors
            - model_type: Model type used
            - train_r2, test_r2, matched_r2, alternative_r2
            - train_rrmse, test_rrmse, matched_rrmse, alternative_rrmse
            - gap_generalization, gap_distribution, gap_fuel_type
            - gap_generalization_rrmse, gap_distribution_rrmse, gap_fuel_type_rrmse
            - (all other metrics from _calculate_fourway_metrics)

        Raises
        ------
        ValueError
            If model_recommendations is empty

        """

        # Validation
        if not self.model_recommendations:
            raise ValueError(
                "Cannot analyze combinations without model_recommendations.\n"
                "Run PolynomialModelSelector.analyze_all_combinations() first,\n"
                "then pass selector.model_recommendations to this analyzer."
            )

        # âœ… OPTIONAL DISPLAY: Header
        if display_results and self.verbose:
            print("\n" + "=" * 180)
            print(f"DOMAIN SHIFT ANALYSIS (Max Predictors: {max_predictors})")
            print("=" * 180)

        # âœ… COMPUTATION: Analyze all combinations
        all_results = []

        for target in self.target_dimensions:
            pred_combos = self._extract_combinations_from_recommendations(
                target=target,
                max_predictors=max_predictors)

            if not pred_combos:
                if self.verbose:
                    print(f"\n  ⚠️  No recommendations found for {target} "
                          f"with ≤{max_predictors} predictors")
                continue

            if self.verbose:
                print(f"\nAnalyzing {target}: {len(pred_combos)} combinations from selector")

            for pred_combo in pred_combos:
                result = self.train_and_test_fourway(list(pred_combo), target)

                if result is not None:
                    result['target'] = target
                    result['predictor_str'] = " + ".join(pred_combo)
                    all_results.append(result)

        # Create DataFrame
        if not all_results:
            if self.verbose:
                print("\n  ⚠️  Warning: No valid results generated!")
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

        # âœ… OPTIONAL DISPLAY: Results
        if display_results and self.verbose:
            from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
            display = AnalyzerResultsDisplay(self)
            # Temporarily set results_df for display
            self.results_df = results_df
            display.print_analysis_results(results_df)

        # Store results
        self.results_df = results_df
        self.results = self._dataframe_to_dict(results_df)

        return results_df

    # ========================================================================
    # FOUR-WAY TESTING (Unchanged - still works the same way)
    # ========================================================================

    def train_and_test_fourway(self, predictors, target):
        """
        Train on conventional train set, test on four datasets.

        **UNCHANGED:** This method works exactly the same way, but now
        it's guaranteed to have a model recommendation for every combination
        it's called with (no more fallback to linear).

        Returns
        -------
        dict with comprehensive metrics including fuel-type effect isolation
        """

        # Validate we have at least one predictor
        if not predictors or len(predictors) == 0:
            if self.verbose:
                print(f"  Error: No predictors provided for target {target}")
            return None

        # Get recommended model type (now guaranteed to exist!)
        model_type = self._get_model_type(predictors, target)

        # Prepare data from all four datasets
        data_prep = self._prepare_fourway_data(predictors, target)
        if data_prep is None:
            return None

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        # Fit once, predict on all datasets
        datasets_to_predict = {
            'train': X_train,
            'test': data_prep['X_test'],
            'matched': data_prep['X_matched'],
            'alternative': data_prep['X_alternative']
        }

        try:
            predictions = self._fit_and_predict_all(
                X_train, y_train, datasets_to_predict, model_type
            )
        except ValueError as e:
            if self.verbose:
                print(f"  Error: {e}")
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
            model_type,
            predictors
        )

        return result

    def _fit_and_predict_all(self, X_train, y_train, datasets_dict, model_type):
        """
        Fit model once, predict on multiple datasets using ModelFactory.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data
        datasets_dict : dict
            {dataset_name: X_array} for all datasets to predict
        model_type : str
            Model type for ModelFactory

        Returns
        -------
        dict : {dataset_name: predictions}
        """
        try:
            # Use ModelFactory for all operations
            model, preprocessor = ModelFactory.create_and_fit(model_type, X_train, y_train)

            predictions = {}
            for dataset_name, X_data in datasets_dict.items():
                predictions[dataset_name] = ModelFactory.predict(
                    model, preprocessor, X_data, model_type
                )

            return predictions

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            raise ValueError(f"Model fitting failed for {model_type}: {str(e)}")

    def _prepare_fourway_data(self, independent_variables, dependent):
        """
        Prepare data from all four datasets with proper cleaning.

        **REFACTORED:** Now uses DataSplitHandler.create_stratified_train_test_split()
        for consistent and maintainable train/test splitting.
        """
        # 1. GET AND CLEAN CONVENTIONAL DATA FIRST (before splitting)
        X_conv_full = self.conventional_full[independent_variables].values
        y_conv_full = self.conventional_full[dependent].values

        conv_mask = ~(np.isnan(X_conv_full).any(axis=1) | np.isnan(y_conv_full) |
                      np.isinf(X_conv_full).any(axis=1) | np.isinf(y_conv_full))

        X_conv_clean = X_conv_full[conv_mask]
        y_conv_clean = y_conv_full[conv_mask]

        if len(X_conv_clean) < 20:
            if self.verbose:
                print(f"  Warning: Insufficient clean conventional data ({len(X_conv_clean)} samples)")
            return None

        # 2. ✅ REFACTORED: Use DataSplitHandler for stratified split
        dwt_idx = independent_variables.index('DWT') if 'DWT' in independent_variables else 0

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

        # 3. GET AND CLEAN MATCHED CONVENTIONAL DATA (unchanged)
        predictor_key = tuple(independent_variables)
        matched_dataset = self.matched_data_sets[predictor_key]

        X_matched = matched_dataset[independent_variables].values
        y_matched = matched_dataset[dependent].values

        matched_mask = ~(np.isnan(X_matched).any(axis=1) | np.isnan(y_matched) |
                         np.isinf(X_matched).any(axis=1) | np.isinf(y_matched))

        X_matched = X_matched[matched_mask]
        y_matched = y_matched[matched_mask]

        # 4. GET AND CLEAN ALTERNATIVE DATA (unchanged)
        X_alt = self.alternative[independent_variables].values
        y_alt = self.alternative[dependent].values

        alt_mask = ~(np.isnan(X_alt).any(axis=1) | np.isnan(y_alt) |
                     np.isinf(X_alt).any(axis=1) | np.isinf(y_alt))

        X_alt = X_alt[alt_mask]
        y_alt = y_alt[alt_mask]

        # 5. VALIDATE SUFFICIENT DATA (unchanged)
        min_samples = {
            'train': 10,
            'test': 3,
            'matched': 3,
            'alternative': 3
        }

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

    def _calculate_fourway_metrics(self, predictions, actuals, model_type, predictors):
        """Calculate comprehensive metrics for all four datasets."""

        result = {
            'model_type': model_type,
            'predictors': predictors,
            'n_predictors': len(predictors),
        }

        # Metrics for each dataset
        for dataset_name in ['train', 'test', 'matched', 'alternative']:
            y_true = actuals[dataset_name]
            y_pred = predictions[dataset_name]
            metrics_calc = RegressionMetrics(y_true, y_pred)

            result[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            result[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            result[f'{dataset_name}_rrmse'] = metrics_calc.calculate_rrmse()
            result[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
            result[f'{dataset_name}_mape'] = metrics_calc.calculate_mape()
            result[f'{dataset_name}_n'] = len(y_true)

        # ===== R² GAPS =====
        result['gap_generalization'] = result['train_r2'] - result['test_r2']
        result['gap_distribution'] = result['test_r2'] - result['matched_r2']
        result['gap_fuel_type'] = result['matched_r2'] - result['alternative_r2']
        result['gap_total'] = result['test_r2'] - result['alternative_r2']

        # ===== RRMSE GAPS (reversed sign - higher RRMSE is worse) =====
        result['gap_generalization_rrmse'] = result['test_rrmse'] - result['train_rrmse']
        result['gap_distribution_rrmse'] = result['matched_rrmse'] - result['test_rrmse']
        result['gap_fuel_type_rrmse'] = result['alternative_rrmse'] - result['matched_rrmse']
        result['gap_total_rrmse'] = result['alternative_rrmse'] - result['test_rrmse']

        return result

    def _get_model_type(self, predictors, target):
        """
        Get recommended model type from ModelSelector.

        **REFACTORED:** No longer prints warnings or falls back to linear!
        Since we extract combinations from recommendations, every combination
        is guaranteed to have a recommendation.
        """
        key = (tuple(predictors), target)

        if key in self.model_recommendations:
            return self.model_recommendations[key]
        else:
            # This should NEVER happen now, but keep for safety
            raise ValueError(
                f"No recommendation found for {predictors} → {target}.\n"
                f"This should not happen when using _extract_combinations_from_recommendations().\n"
                f"Please report this as a bug."
            )

    # ========================================================================
    # HELPER METHODS (Mostly unchanged)
    # ========================================================================

    def _dataframe_to_dict(self, results_df: pd.DataFrame) -> dict:
        """
        Convert results DataFrame to nested dict format for backward compatibility.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results DataFrame

        Returns
        -------
        dict
            {target: {predictor_str: {metrics}}}
        """
        results_dict = {}

        for target in results_df['target'].unique():
            target_df = results_df[results_df['target'] == target]
            results_dict[target] = {}

            for _, row in target_df.iterrows():
                predictor_str = row['predictor_str']
                # Convert row to dict, excluding target and predictor_str
                metrics = row.drop(['target', 'predictor_str']).to_dict()
                results_dict[target][predictor_str] = metrics

        return results_dict

    # ========================================================================
    # DISPLAY METHODS
    # ========================================================================

    def print_initialization_summary(self):
        """Convenience method to display initialization info."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_initialization_summary()

    def print_analysis_results(self, **kwargs):
        """Convenience method to display results."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_analysis_results(**kwargs)

    def print_interpretation(self, **kwargs):
        """Convenience method to display interpretation."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_interpretation(**kwargs)

    def print_best_combinations(self, **kwargs):
        """Convenience method to display best combinations."""
        from knowledge_layer.multi_dataset_analysers.analyser_display import AnalyzerResultsDisplay
        display = AnalyzerResultsDisplay(self)
        display.print_best_combinations(**kwargs)


    # ========================================================================
    # UTILITY METHODS (Unchanged)
    # ========================================================================

    def get_best_combinations(self,
                              metric: str = 'alternative_rrmse',
                              ascending: bool = True) -> pd.DataFrame:
        """
        Get best predictor combination for each target.

        Parameters
        ----------
        metric : str
            Metric to optimize ('alternative_rrmse', 'alternative_r2', etc.)
        ascending : bool
            If True, lower is better (for RRMSE). If False, higher is better (for R²)

        Returns
        -------
        pd.DataFrame
            Best combination for each target
        """

        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run analyze_all_combinations() first.")

        # Get best for each target
        best_results = (self.results_df
                        .sort_values(metric, ascending=ascending)
                        .groupby('target')
                        .first()
                        .reset_index())

        return best_results

    def filter_results(self,
                       targets: List[str] = None,
                       min_predictors: int = None,
                       max_predictors: int = None,
                       model_types: List[str] = None,
                       min_r2: float = None,
                       max_rrmse: float = None) -> pd.DataFrame:
        """
        Filter results DataFrame based on criteria.

        Parameters
        ----------
        targets : List[str], optional
            Only include these targets
        min_predictors : int, optional
            Minimum number of predictors
        max_predictors : int, optional
            Maximum number of predictors
        model_types : List[str], optional
            Only include these model types
        min_r2 : float, optional
            Minimum alternative R²
        max_rrmse : float, optional
            Maximum alternative RRMSE

        Returns
        -------
        pd.DataFrame
            Filtered results
        """

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
        """
        Export results DataFrame to CSV.

        Parameters
        ----------
        filename : str
            Output filename
        include_all_metrics : bool
            If True, exports all columns. If False, exports only key metrics
        """

        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run analyze_all_combinations() first.")

        if include_all_metrics:
            export_df = self.results_df
        else:
            # Export only key columns
            key_cols = [
                'target', 'predictor_str', 'n_predictors', 'model_type',
                'alternative_r2', 'alternative_rrmse', 'alternative_mape',
                'gap_fuel_type', 'gap_fuel_type_rrmse'
            ]
            existing_cols = [col for col in key_cols if col in self.results_df.columns]
            export_df = self.results_df[existing_cols]

        export_df.to_csv(filename, index=False)
        print(f"✓ Results exported to {filename}")
        print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def interpret_results(self,
                          significance_threshold_r2=0.05,
                          significance_threshold_rrmse=10.0):
        """
        Interpret which dimensions are affected by fuel type.

        Parameters
        ----------
        significance_threshold_r2 : float
            R² gap threshold to consider effect significant (default: 0.05)
        significance_threshold_rrmse : float
            RRMSE gap threshold in % to consider effect significant (default: 10%)
        """

        print("\n" + "=" * 120)
        print("INTERPRETATION: Fuel Type Effect Analysis")
        print("=" * 120)

        summary = []

        for target, models in self.results.items():
            if not models:
                continue

            # Get best model based on lowest alternative RRMSE
            best_combo = min(models.items(),
                             key=lambda x: x[1]['alternative_rrmse'])
            combo_name, metrics = best_combo

            # Extract gaps
            fuel_gap_r2 = metrics['gap_fuel_type']
            fuel_gap_rrmse = metrics['gap_fuel_type_rrmse']
            gen_gap_r2 = metrics['gap_generalization']
            gen_gap_rrmse = metrics['gap_generalization_rrmse']
            dist_gap_r2 = metrics['gap_distribution']
            dist_gap_rrmse = metrics['gap_distribution_rrmse']

            # Classify effect magnitude
            r2_significant = abs(fuel_gap_r2) >= significance_threshold_r2
            rrmse_significant = abs(fuel_gap_rrmse) >= significance_threshold_rrmse

            if not r2_significant and not rrmse_significant:
                effect = "MINIMAL"
                icon = "✅"
            elif r2_significant and rrmse_significant:
                if abs(fuel_gap_r2) > 0.15 or abs(fuel_gap_rrmse) > 20:
                    effect = "MAJOR"
                    icon = "🚨"
                else:
                    effect = "MODERATE"
                    icon = "⚠️"
            else:
                effect = "MODERATE"
                icon = "⚠️"

            print(f"\n{icon} {target}:")
            print(f"   Best predictors: {combo_name}")
            print(f"   Model: {metrics['model_type']}")

            print(f"\n   Performance (R²):")
            print(f"     Train:       {metrics['train_r2']:.3f}")
            print(f"     Test:        {metrics['test_r2']:.3f}")
            print(f"     Matched:     {metrics['matched_r2']:.3f}")
            print(f"     Alternative: {metrics['alternative_r2']:.3f}")

            print(f"\n   Performance (RRMSE %):")
            print(f"     Train:       {metrics['train_rrmse']:.2f}%")
            print(f"     Test:        {metrics['test_rrmse']:.2f}%")
            print(f"     Matched:     {metrics['matched_rrmse']:.2f}%")
            print(f"     Alternative: {metrics['alternative_rrmse']:.2f}%")

            print(f"\n   Gaps (R²):")
            print(f"     Generalization:  {gen_gap_r2:+.3f} (train→test)")
            print(f"     Distribution:    {dist_gap_r2:+.3f} (test→matched)")
            print(f"     FUEL TYPE:       {fuel_gap_r2:+.3f} (matched→alternative) ← KEY")

            print(f"\n   Gaps (RRMSE %):")
            print(f"     Generalization:  {gen_gap_rrmse:+.2f}% (train→test)")
            print(f"     Distribution:    {dist_gap_rrmse:+.2f}% (test→matched)")
            print(f"     FUEL TYPE:       {fuel_gap_rrmse:+.2f}% (matched→alternative) ← KEY")

            print(f"\n   Effect Classification: {effect}")

            if abs(fuel_gap_rrmse) > 20:
                print(f"   ⚠️  Prediction error increases by {abs(fuel_gap_rrmse):.1f}% "
                      f"for alternative fuel ships!")

            summary.append({
                'target': target,
                'effect': effect,
                'fuel_gap_r2': fuel_gap_r2,
                'fuel_gap_rrmse': fuel_gap_rrmse,
                'alternative_rrmse': metrics['alternative_rrmse'],
                'best_predictors': combo_name
            })

        # Summary table
        print(f"\n{'=' * 140}")
        print("SUMMARY TABLE")
        print(f"{'=' * 140}")
        print(f"{'Dimension':<15s} | {'Effect':<10s} | "
              f"{'Fuel Gap R²':>12s} | {'Fuel Gap RRMSE%':>16s} | "
              f"{'Alt RRMSE%':>11s} | {'Best Predictors'}")
        print(f"{'-' * 140}")

        # Sort by RRMSE gap
        for item in sorted(summary, key=lambda x: abs(x['fuel_gap_rrmse']), reverse=True):
            print(f"{item['target']:<15s} | {item['effect']:<10s} | "
                  f"{item['fuel_gap_r2']:>+12.3f} | {item['fuel_gap_rrmse']:>+16.2f} | "
                  f"{item['alternative_rrmse']:>11.2f} | "
                  f"{item['best_predictors']}")

        print(f"\n{'=' * 140}")
        print("INTERPRETATION GUIDE:")
        print("  R² Gap: How much variance explanation is lost")
        print("  RRMSE Gap: How much prediction error increases (in %)")
        print(f"  Thresholds: R² gap > {significance_threshold_r2:.2f} OR "
              f"RRMSE gap > {significance_threshold_rrmse:.1f}% = significant fuel effect")

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

    def _get_all_predictions(self, data_prep: dict, predictors: List[str],
                             model_type: str) -> dict[str, np.ndarray]:
        """
        Get predictions for all four datasets.

        Parametric-specific: Uses ModelFactory.
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        X_train = data_prep['X_train']
        y_train = data_prep['y_train']

        try:
            model, preprocessor = ModelFactory.create_and_fit(
                model_type, X_train, y_train
            )

            predictions = {}
            for dataset_name in ['train', 'test', 'matched', 'alternative']:
                X_data = data_prep[f'X_{dataset_name}']
                predictions[dataset_name] = ModelFactory.predict(
                    model, preprocessor, X_data, model_type
                )

            return predictions

        except Exception as e:
            if self.verbose:
                print(f"  Error getting predictions: {e}")
            return None
