# multi_dataset_domain_analyzer.py (NEW FILE)
"""
Multi-Dataset Domain Analyzer
==============================

Coordinates domain shift analysis across multiple alternative fuel datasets.
Uses composition to manage multiple single-dataset analyzers.
"""

import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_regression_analyser_class import ParametricDomainAnalyzer
from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_domain_analyser import ParametricDomainAnalyzer
from knowledge_layer.multi_dataset_analysers.domain_shift.rf_domain_analyser import RandomForestDomainAnalyzer
# from knowledge_layer.multi_dataset_analysers.domain_shift.random_forest_domain_analysis_clas import RandomForestDomainAnalyzer
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class import (
    AnalysisPlotter
)
from knowledge_layer.model_selectors.parametric_formula_latex_generator_class import ParametricFormulaLatexGenerator

from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility

from knowledge_layer.multi_dataset_analysers.multi_domain_display import MultiDatasetDisplay

class MultiDatasetDomainAnalyzer:
    """
    Analyzes domain shift across multiple alternative fuel datasets.

    **Design**: Uses composition - creates separate analyzer for each fuel type,
    then aggregates and compares results.


    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_datasets: Dict[str, pd.DataFrame],
                 independent_variables: List[str],
                 dependent_variables: Optional[List[str]] = None,
                 model_recommendations: Optional[dict] = None,
                 selector=None,
                 analyzer_type: str = 'parametric',
                 rf_params: Optional[Dict] = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True,
                 selection_metric: str = 'rrmse',
                 selection_strategy: str = 'conservative',
                target_predictor_constraints: Optional[Dict[str, List[str]]] = None,
                 build_year_col:str|None=None,
                 year_edges:list[int]|None=None,
                 build_eligibility:TemporalEligibility|None=None,
                 n_matches_per_ship:int=10,
                 caliper:float|None=None,
                 matched_bins:dict[str, list]|None=None,):
        """
        Initialize multi-dataset analyzer.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_datasets : dict[str, pd.DataFrame]
            Dictionary mapping fuel type names to datasets
            Example: {'LNG': lng_df, 'Methanol': methanol_df}
        independent_variables : list[str]
            List of potential predictor dimensions
        dependent_variables : list[str], optional
            List of target dimensions
        model_recommendations : dict, optional
            Model recommendations from selector
        analyzer_type : str
            Type of analyzer: 'parametric' or 'random_forest'
        rf_params : dict, optional
            Random Forest parameters (only used if analyzer_type='random_forest')
        test_size : float
            Fraction for train/test split
        random_state : int
            Random seed
        verbose : bool
            Whether to print progress

        selection_metric : str, default='bic'
            Metric for selecting best combinations
        selection_strategy : str, default='conservative'
            Strategy for model selection

        """

        self.conventional_data = conventional_data
        self.alternative_datasets = alternative_datasets
        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables or independent_variables
        self.model_recommendations = model_recommendations
        self.selector = selector
        self.analyzer_type = analyzer_type
        self.rf_params = rf_params
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.selection_metric = selection_metric
        self.selection_strategy = selection_strategy
        self.target_predictor_constraints=target_predictor_constraints
        self.n_matches_per_ship=n_matches_per_ship
        self.caliper = caliper

        self.build_year_col=build_year_col
        self.year_edges=year_edges
        self.build_eligibility=build_eligibility
        self.matched_bins=matched_bins

        # Validate inputs
        self._validate_inputs()


        # Storage for individual analyzers
        self.analyzers: Dict[str, ParametricDomainAnalyzer | RandomForestDomainAnalyzer] = {}

        # Storage for aggregated results
        self.comparison_results: Optional[pd.DataFrame] = None

        # Composition: Use same plotter for all visualizations
        self.plotter = AnalysisPlotter()

        if self.verbose:
            self._print_initialization_summary()

        self._display = None

    @property
    def display(self):
        """Lazy initialization of display helper."""
        if self._display is None:
            self._display = MultiDatasetDisplay(self)
        return self._display

    def _validate_inputs(self):
        """Validate initialization parameters."""

        if not isinstance(self.alternative_datasets, dict):
            raise TypeError(
                f"alternative_datasets must be a dict, got {type(self.alternative_datasets)}"
            )

        if len(self.alternative_datasets) == 0:
            raise ValueError("alternative_datasets cannot be empty")

        # Check all datasets have required columns
        all_dims = list(set(self.independent_variables + self.dependent_variables))

        for fuel_type, df in self.alternative_datasets.items():
            missing = [dim for dim in all_dims if dim not in df.columns]
            if missing:
                raise ValueError(
                    f"Alternative dataset '{fuel_type}' missing columns: {missing} \n Available columns: {df.columns}"
                )

        # Warn if fuel type names are very long (plotting issues)
        long_names = [name for name in self.alternative_datasets.keys() if len(name) > 15]
        if long_names and self.verbose:
            print(f"\nâš ï¸  Warning: Long fuel type names detected: {long_names}")
            print(f"   Consider using abbreviations for better plot readability.")

        # Warn if too many datasets
        if len(self.alternative_datasets) > 4 and self.verbose:
            print(f"\nâš ï¸  Warning: {len(self.alternative_datasets)} datasets detected.")
            print(f"   Comparisons with >4 datasets may be difficult to visualize.")
            print(f"   Consider grouping similar fuel types or analyzing subsets.")

    def _print_initialization_summary(self):
        """Print initialization summary."""
        print(f"\n{'=' * 70}")
        print(f"Multi-Dataset Domain Analyzer Initialized")
        print(f"{'=' * 70}")
        print(f"Analyzer type: {self.analyzer_type}")
        print(f"Conventional data: {len(self.conventional_data)} ships")
        print(f"Alternative datasets: {len(self.alternative_datasets)}")
        for fuel_type, df in self.alternative_datasets.items():
            print(f"  â€¢ {fuel_type}: {len(df)} ships")

    def analyze_all_datasets(self,
                             max_predictors: int = 2,
                             best_only: bool = True,
                             selection_metric: str = None,
                             display_results: bool = False,
                             debug_selection: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run analysis on all alternative fuel datasets.

        **Simplified Approach**:
        1. For each fuel type, run full analysis (all combinations)
        2. Select best combination per target using TEST SET performance
        3. This ensures fair comparison - same conventional split for all fuel types

        Parameters
        ----------
        max_predictors : int, default=2
            Maximum number of predictors per combination
        best_only : bool, default=True
            If True, analyze only BEST combination per target (selected by test performance).
            If False, analyze ALL combinations from selector.
        selection_metric : str, optional
            Metric for selecting best on test data:
            - 'rrmse' or 'test_rrmse': Best generalization (recommended)
            - 'r2' or 'test_r2': Best fit on test set
            - 'mape' or 'test_mape': Best percentage error
            If None, uses analyzer default.
        display_results : bool, default=True
            Whether to display results for each dataset
        debug_selection : bool, default=False
            If True, print detailed debugging information about combination selection

        Returns
        -------
        dict[str, pd.DataFrame]
            Results DataFrames for each fuel type

        Notes
        -----
        **Key Design Decision**: Uses same combinations across all fuel types.
        Combinations are selected based on FIRST fuel type's test performance,
        then those SAME combinations are tested on all other fuel types.
        This ensures fair comparison.
        """

        # ✅ Get selection metric
        if selection_metric is None:
            selection_metric = self.selection_metric

        # ✅ Map to test metric if needed
        if not selection_metric.startswith('test_'):
            metric_map = {
                'rrmse': 'test_rrmse',
                'r2': 'test_r2',
                'adj_r2': 'test_r2',
                'mape': 'test_mape',
                'mae': 'test_mae',
                'rmse': 'test_rmse',
                'bic': 'test_rrmse',  # Use test_rrmse as proxy
                'aic': 'test_rrmse',  # Use test_rrmse as proxy
            }

            if selection_metric in metric_map:
                original_metric = selection_metric
                selection_metric = metric_map[selection_metric]
                if self.verbose:
                    print(f"\n  📊 Mapped selection metric: '{original_metric}' → '{selection_metric}'")

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Analyzing {len(self.alternative_datasets)} Alternative Fuel Datasets")
            if best_only:
                print(f"Mode: BEST ONLY (selected by {selection_metric})")
            else:
                print(f"Mode: ALL COMBINATIONS")
            print(f"{'=' * 70}\n")

        all_results = {}
        best_combinations = None  # Will be set from first dataset

        for fuel_idx, (fuel_type, alt_data) in enumerate(self.alternative_datasets.items()):

            if self.verbose:
                print(f"\n{'─' * 70}")
                print(f"Analyzing: {fuel_type}")
                print(f"{'─' * 70}")

            # Create analyzer
            analyzer = self._create_analyzer(alt_data)

            if fuel_idx == 0 and best_only:
                # ✅ FIRST DATASET: Run full analysis, then select best

                if self.verbose:
                    print(f"\n  Running full analysis on first dataset to select best combinations...")

                # Run full analysis
                if self.analyzer_type == 'parametric':
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        display_results=display_results
                    )
                elif self.analyzer_type == 'random_forest':
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        best_only=False,  # Get all combinations first
                        display_results=display_results
                    )

                # Select best combinations using TEST performance
                best_combinations = self._select_best_combinations_from_test_performance(
                    analyzer=analyzer,
                    max_predictors=max_predictors,
                    selection_metric=selection_metric,  # ✅ Already mapped to test_*
                    debug=debug_selection
                )

                if not best_combinations:
                    if self.verbose:
                        print(f"  ⚠️  No valid combinations selected!")
                    all_results[fuel_type] = pd.DataFrame()
                    continue

                # Filter results to best only
                if self.analyzer_type == 'parametric':
                    # Update analyzer's recommendations to only include best
                    analyzer.foundation.model_recommendations = best_combinations

                    # Re-run analysis with only best combinations
                    if self.verbose:
                        print(f"\n  Re-running analysis with best combinations only...")

                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        display_results=display_results
                    )

            elif best_only:
                # ✅ SUBSEQUENT DATASETS: Use same combinations as first dataset

                if best_combinations is None:
                    raise ValueError("best_combinations not set - first dataset had no valid results")

                if self.verbose:
                    print(f"\n  Using same {len(best_combinations)} combinations as first dataset...")

                if self.analyzer_type == 'parametric':
                    # Set analyzer to use same combinations
                    analyzer.foundation.model_recommendations = best_combinations

                # Run analysis
                if self.analyzer_type == 'parametric':
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        display_results=display_results
                    )
                elif self.analyzer_type == 'random_forest':
                    # For RF, filter results after analysis
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        best_only=False,
                        display_results=display_results
                    )

                    # Filter to best combinations
                    if not results_df.empty:
                        mask = results_df.apply(
                            lambda row: (tuple(row['predictors']), row['target']) in best_combinations,
                            axis=1
                        )
                        filtered_df = results_df[mask]

                        # Warn about any combinations that were expected but missing
                        expected_targets = {target for (_, target) in best_combinations.keys()}
                        found_targets = set(filtered_df['target'].unique())
                        missing_targets = expected_targets - found_targets
                        if missing_targets and self.verbose:
                            print(f"  ⚠️  {fuel_type}: No results for targets: {missing_targets} "
                                  f"(insufficient data for this fuel type)")

                        results_df = filtered_df

            else:
                # ✅ ALL COMBINATIONS: No filtering

                if self.analyzer_type == 'parametric':
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        display_results=display_results
                    )
                elif self.analyzer_type == 'random_forest':
                    results_df = analyzer.analyze_all_combinations(
                        max_predictors=max_predictors,
                        best_only=False,
                        display_results=display_results
                    )

            # Store
            self.analyzers[fuel_type] = analyzer
            all_results[fuel_type] = results_df

        # Create comparison DataFrame
        self._create_comparison_dataframe(all_results)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✅ All {len(self.alternative_datasets)} datasets analyzed")
            print(f"{'=' * 70}\n")

        return all_results

    def _select_best_combinations_from_test_performance(
            self,
            analyzer,
            max_predictors: int,
            selection_metric: str = 'test_rrmse',
            debug: bool = False
    ) -> Dict:
        """
        Select best combinations using ANALYZER'S test set performance.

        **ENHANCED DEBUG VERSION**
        """

        print(f"\n{'=' * 80}")
        print(f"DEBUG: _select_best_combinations_from_test_performance()")
        print(f"{'=' * 80}")
        print(f"  selection_metric: {selection_metric}")
        print(f"  max_predictors: {max_predictors}")
        print(f"  analyzer type: {type(analyzer).__name__}")

        # Check analyzer has results_df
        if not hasattr(analyzer, 'results_df') or analyzer.results_df is None:
            print(f"  ❌ Analyzer has no results_df!")
            raise ValueError("Analyzer must have results_df with test performance!")

        results_df = analyzer.results_df

        print(f"  ✅ Analyzer has results_df with {len(results_df)} rows")
        print(f"  Columns: {list(results_df.columns)}")
        print(f"  Targets in results_df: {sorted(results_df['target'].unique())}")

        if len(results_df) == 0:
            print(f"  ❌ results_df is empty!")
            return {}

        # Check selection_metric exists
        if selection_metric not in results_df.columns:
            print(f"  ❌ Metric '{selection_metric}' not in results_df!")
            print(f"     Available metrics: {[c for c in results_df.columns if 'rrmse' in c or 'r2' in c]}")
            return {}

        # Determine sort direction
        ascending = selection_metric in [
            'test_rrmse', 'test_mape', 'test_mae', 'test_rmse',
            'aic', 'bic', 'rrmse', 'mape', 'mae', 'rmse'
        ]

        print(f"  Sort direction: {'ascending' if ascending else 'descending'}")

        best_combos = {}

        for target in analyzer.target_dimensions:
            print(f"\n{'·' * 80}")
            print(f"Target: {target}")
            print(f"{'·' * 80}")

            # Filter to this target
            target_results = results_df[results_df['target'] == target]

            if len(target_results) == 0:
                print(f"  ❌ No results for this target")
                continue

            print(f"  Found {len(target_results)} results for this target")

            # Apply max_predictors filter
            if max_predictors is not None:
                before_filter = len(target_results)
                target_results = target_results[target_results['n_predictors'] <= max_predictors]
                after_filter = len(target_results)
                print(
                    f"  After max_predictors={max_predictors} filter: {after_filter} (removed {before_filter - after_filter})")

            if len(target_results) == 0:
                print(f"  ❌ No results after filtering")
                continue

            # Sort by TEST performance
            sorted_results = target_results.sort_values(selection_metric, ascending=ascending)

            print(f"\n  Top 5 candidates (sorted by {selection_metric}):")
            for idx, (_, row) in enumerate(sorted_results.head(5).iterrows(), 1):
                pred_str = ' + '.join(row['predictors'])
                metric_val = row[selection_metric]
                model_type = row.get('model_type', 'unknown')
                marker = "✅" if idx == 1 else "  "
                print(f"    {marker} {idx}. {pred_str:30s} ({model_type:15s}) {selection_metric}={metric_val:.4f}")

            # Select best
            best_row = sorted_results.iloc[0]

            predictors = tuple(best_row['predictors'])
            model_type = best_row['model_type']
            metric_value = best_row[selection_metric]

            # Show all metrics for selected combination
            print(f"\n  ✅ SELECTED: {' + '.join(predictors)} ({model_type})")
            print(f"     All metrics for selected combination:")
            metric_cols = [c for c in best_row.index if any(m in c for m in ['rrmse', 'r2', 'mape', 'mae'])]
            for col in sorted(metric_cols):
                val = best_row[col]
                if not np.isnan(val):
                    print(f"       {col:30s}: {val:.4f}")

            best_combos[(predictors, target)] = model_type

        print(f"\n{'=' * 80}")
        print(f"✅ Selection complete: {len(best_combos)} combinations selected")
        print(f"{'=' * 80}\n")

        return best_combos

    def _select_best_combinations(
            self,
            analyzer,
            max_predictors: int,
            selection_metric: str,
            selection_based_on_conventional: bool,
            debug: bool = False
    ) -> Dict:
        """
        Select best combinations (SIMPLIFIED).

        **NEW BEHAVIOR**: Always uses analyzer's test performance for selection.
        The 'selection_based_on_conventional' parameter is ignored.

        Parameters
        ----------
        analyzer : ParametricDomainAnalyzer or RandomForestDomainAnalyzer
            Analyzer with results from full analysis
        max_predictors : int
            Maximum predictors per combination
        selection_metric : str
            Metric for selection (e.g., 'test_rrmse')
        selection_based_on_conventional : bool
            IGNORED - kept for backward compatibility
        debug : bool, default=False
            If True, print detailed debugging

        Returns
        -------
        dict
            {(predictors_tuple, target): model_type}
        """

        if debug:
            print(f"\n{'=' * 70}")
            print(f"DEBUG: _select_best_combinations()")
            print(f"{'=' * 70}")
            print(f"  max_predictors: {max_predictors}")
            print(f"  selection_metric: {selection_metric}")
            print(f"  analyzer type: {type(analyzer).__name__}")
            print(f"  Method: Using analyzer.results_df (test set performance)")
            if selection_based_on_conventional:
                print(f"  Note: 'selection_based_on_conventional' parameter is now ignored")

        if self.analyzer_type == 'parametric':
            # ✅ Use test set performance directly
            return self._select_best_combinations_from_test_performance(
                analyzer=analyzer,
                max_predictors=max_predictors,
                selection_metric=selection_metric,
                debug=debug
            )

        elif self.analyzer_type == 'random_forest':
            # ✅ Same for RF
            return self._select_best_combinations_from_test_performance(
                analyzer=analyzer,
                max_predictors=max_predictors,
                selection_metric=selection_metric,
                debug=debug
            )

        return {}

    def _create_analyzer(self, alternative_data: pd.DataFrame):
        """Create appropriate analyzer type."""

        if self.analyzer_type == 'parametric':
            analyser= ParametricDomainAnalyzer(
                conventional_data=self.conventional_data,
                alternative_data=alternative_data,
                independent_variables=self.independent_variables,
                dependent_variables=self.dependent_variables,
                model_recommendations=self.model_recommendations,
                test_size=self.test_size,
                random_state=self.random_state,
                verbose=False,
                selection_metric=self.selection_metric,
                selection_strategy=self.selection_strategy,
                target_predictor_constraints=self.target_predictor_constraints,
                build_year_col=self.build_year_col,
                year_edges=self.year_edges,
                build_eligibility=self.build_eligibility,
                n_matches_per_ship=self.n_matches_per_ship,
                caliper=self.caliper,
                matching_bins=self.matched_bins
                # Suppress individual analyzer verbosity
            )

            if self.selector is not None:
                analyser.foundation.selector = self.selector

            analyser.foundation.detailed_results = self.selector.detailed_results



            return analyser

        elif self.analyzer_type == 'random_forest':
            return RandomForestDomainAnalyzer(
                conventional_data=self.conventional_data,
                alternative_data=alternative_data,
                variable_dimensions=self.independent_variables,
                target_dimensions=self.dependent_variables,
                selector_results=self.model_recommendations,
                rf_params=self.rf_params,
                test_size=self.test_size,
                random_state=self.random_state,
                selection_metric=self.selection_metric,
                verbose=False,
                target_predictor_constraints=self.target_predictor_constraints,
                build_year_col=self.build_year_col,
                year_edges=self.year_edges,
                build_eligibility=self.build_eligibility,
                n_matches_per_ship=self.n_matches_per_ship,
                caliper=self.caliper,
                matching_bins=self.matched_bins
            )

        else:
            raise ValueError(f"Unknown analyzer_type: {self.analyzer_type}")

    def _create_comparison_dataframe(self, all_results: Dict[str, pd.DataFrame]):
        """Create aggregated comparison DataFrame."""

        comparison_data = []

        for fuel_type, results_df in all_results.items():
            # ✅ Skip empty results
            if results_df is None or len(results_df) == 0:
                if self.verbose:
                    print(f"  ℹ️  Skipping {fuel_type}: no valid results")
                continue

            # Add fuel_type column
            df_copy = results_df.copy()
            df_copy['fuel_type'] = fuel_type
            comparison_data.append(df_copy)

        # ✅ Handle case where ALL results are empty
        if not comparison_data:
            if self.verbose:
                print("\n⚠️  Warning: No valid results from any dataset!")
            self.comparison_results = pd.DataFrame()
            return

        self.comparison_results = pd.concat(comparison_data, ignore_index=True)

        # Reorder columns for readability
        cols = ['fuel_type', 'target', 'predictor_str', 'model_type'] + \
               [col for col in self.comparison_results.columns
                if col not in ['fuel_type', 'target', 'predictor_str', 'model_type']]

        self.comparison_results = self.comparison_results[cols]

    def plot_multi_dataset_comparison(self,
                                      target: str,
                                      predictor_combo: Optional[List[str]] = None,
                                      variable_groupings: Optional[Dict[str, List[str]]] = None,
                                      threshold_metric: Optional[str] = 'median',
                                      show_average_lines: bool = False,
                                      rrmse_threshold:float|None=None,
                                      save_path: Optional[str] = None,
                                      debug:bool=False):
        """
        Plot multi-dataset comparison for a specific target.

        Parameters
        ----------
        target : str
            Target dimension to plot
        predictor_combo : list[str], optional
            Specific predictor combination to plot.
            If None, uses best combination for each fuel type.
        variable_groupings : dict, optional
            Grouping of variables by category for threshold lines.
            Example: {'Hull dimensions': ['LBP (M)', 'BEAM (M)'], 'Capacity': ['DWT']}
            If None, no threshold lines are drawn.
        threshold_metric : str, optional
            How to compute the reference threshold from conventional test RRMSE:
            'median', 'mean', or 'best' (minimum). Default: 'median'.
            Only used if variable_groupings is provided.
        show_average_lines : bool
            If True, shows dotted average lines across bars. Default False.
        save_path : str, optional
            Path to save figure
        debug : bool
            Print debug information
        """

        if not self.analyzers:
            raise ValueError("No results available. Run analyze_all_datasets() first.")

        # Gather metrics for each fuel type
        datasets_metrics = {}
        predictor_names = None
        model_description = None

        for fuel_type, analyzer in self.analyzers.items():
            if target not in analyzer.results:
                if self.verbose:
                    print(f"âš ï¸  {target} not found in {fuel_type} results")
                continue

            models = analyzer.results[target]

            if not models:
                continue

            if predictor_combo is not None:
                # Use specified combination
                combo_str = " + ".join(predictor_combo)
                if combo_str not in models:
                    if self.verbose:
                        print(f"âš ï¸  {combo_str} not found in {fuel_type} for {target}")
                    continue
                metrics = models[combo_str]
            else:
                # Use best combination
                best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
                combo_str, metrics = best_combo

            # Convert to FourWayMetrics
            fourway_metrics = analyzer._convert_to_fourway_metrics(metrics)
            datasets_metrics[fuel_type] = fourway_metrics

            # Store predictor names and model description (same for all)
            if predictor_names is None:
                predictor_names = metrics['predictors']
                model_description = metrics['model_type']

        if not datasets_metrics:
            print(f"No valid data to plot for {target}")
            return

        # ====================================================================
        # 2. Compute threshold from conventional test RRMSE (if requested)
        # ====================================================================
        # The conventional test RRMSE is the natural reference - it answers
        # "how well does this model generalise on conventional ships?"
        # We extract it from the FourWayMetrics objects we already have,
        # since test RRMSE is the same across all fuel types for a given target.


        if variable_groupings is not None and threshold_metric is not None:
            # Determine which category this target belongs to
            target_category = None
            for category, variables in variable_groupings.items():
                if target in variables:
                    target_category = category
                    break

            if target_category is not None:
                # Collect test RRMSE values for all targets in the same category
                # across all analyzers (same conventional split, so test RRMSE
                # is consistent)
                category_test_rrmse = []

                first_analyzer = next(iter(self.analyzers.values()))
                category_targets = variable_groupings[target_category]

                for cat_target in category_targets:
                    if cat_target not in first_analyzer.results:
                        continue
                    cat_models = first_analyzer.results[cat_target]
                    if not cat_models:
                        continue
                    best = min(cat_models.items(), key=lambda x: x[1]['alternative_rrmse'])
                    _, cat_metrics = best
                    cat_fourway = first_analyzer._convert_to_fourway_metrics(cat_metrics)
                    category_test_rrmse.append(cat_fourway.test.rrmse)

                if category_test_rrmse:
                    vals = np.array(category_test_rrmse)
                    if threshold_metric == 'median':
                        rrmse_threshold = float(np.median(vals))
                    elif threshold_metric == 'mean':
                        rrmse_threshold = float(np.mean(vals))
                    elif threshold_metric == 'best':
                        rrmse_threshold = float(np.min(vals))

                    if debug:
                        print(f"\nDEBUG: Threshold computation")
                        print(f"  Target: {target} → category: {target_category}")
                        print(f"  Category test RRMSE values: {[f'{v:.2f}' for v in vals]}")
                        print(f"  Threshold ({threshold_metric}): {rrmse_threshold:.2f}%")
            else:
                if debug:
                    print(f"DEBUG: Target '{target}' not found in any variable grouping category")

        # Delegate to plotter
        fig = self.plotter.domain_shift_plotter.plot_multi_dataset_comparison(
            target_name=target,
            predictor_names=predictor_names,
            model_description=model_description,
            datasets_metrics=datasets_metrics,
            save_path=save_path
        )

        plt.show()

        if save_path:
            print(f"âœ“ Multi-dataset comparison saved to {save_path}")


    def plot_fuel_type_summary(self,
                               targets: list[str] = None,
                               metric: str = 'gap_fuel_type_rrmse',
                               save_path: str = None):
        """
        Plot summary heatmap comparing fuel types across targets.

        **FIXED**: Now handles missing data and provides better error messages.

        Parameters
        ----------
        targets : list[str], optional
            Targets to include. If None, uses all targets.
        metric : str
            Metric to display in heatmap
        save_path : str, optional
            Path to save figure
        """

        if not self.analyzers:
            raise ValueError("No results available. Run analyze_all_datasets() first.")

        if targets is None:
            targets = self.dependent_variables

        # Gather data - now handles missing combinations gracefully
        targets_data = {}

        for target in targets:
            targets_data[target] = {}

            for fuel_type, analyzer in self.analyzers.items():
                # Check if this target exists in this analyzer's results
                if target not in analyzer.results:
                    if self.verbose:
                        print(f"  â„¹ï¸  {target} not found in {fuel_type} results - skipping")
                    continue

                models = analyzer.results[target]

                if not models:
                    if self.verbose:
                        print(f"  â„¹ï¸  {target} has no models in {fuel_type} results - skipping")
                    continue

                # Use best combination
                try:
                    best_combo = min(models.items(), key=lambda x: x[1]['alternative_rrmse'])
                    _, metrics = best_combo

                    fourway_metrics = analyzer._convert_to_fourway_metrics(metrics)
                    targets_data[target][fuel_type] = fourway_metrics

                except Exception as e:
                    if self.verbose:
                        print(f"  âš ï¸  Error processing {target} for {fuel_type}: {e}")
                    continue

        # Validate we have at least some data
        valid_targets = [t for t in targets_data.keys() if targets_data[t]]

        if not valid_targets:
            print("\nâŒ Error: No valid data found for any target-fuel combination!")
            print("\nDebugging info:")
            print(f"  Requested targets: {targets}")
            print(f"  Available analyzers: {list(self.analyzers.keys())}")

            for fuel_type, analyzer in self.analyzers.items():
                print(f"\n  {fuel_type} analyzer has results for:")
                if analyzer.results:
                    for target in analyzer.results.keys():
                        n_models = len(analyzer.results[target])
                        print(f"    â€¢ {target}: {n_models} models")
                else:
                    print(f"    (no results)")

            return

        # Delegate to plotter
        fig = self.plotter.domain_shift_plotter.plot_fuel_type_comparison_summary(
            targets_data=targets_data,
            metric=metric,
            save_path=save_path
        )

        plt.show()

        if save_path:
            print(f"âœ“ Fuel type summary saved to {save_path}")

    def export_comparison(self, filename: str = 'fuel_type_comparison.xlsx'):
        """Export comparison results to Excel with multiple sheets."""

        if self.comparison_results is None:
            raise ValueError("No results available. Run analyze_all_datasets() first.")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Full comparison data
            self.comparison_results.to_excel(writer, sheet_name='All_Results', index=False)

            # Sheet 2: Summary statistics
            summary_stats = self._create_summary_statistics()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        print(f"âœ“ Comparison exported to {filename}")

    def _create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics across all fuel types."""

        summary_data = []

        for fuel_type in self.alternative_datasets.keys():
            ft_data = self.comparison_results[
                self.comparison_results['fuel_type'] == fuel_type
                ]

            summary_data.append({
                'fuel_type': fuel_type,
                'n_ships': len(self.alternative_datasets[fuel_type]),
                'mean_gap_fuel_type_r2': ft_data['gap_fuel_type'].mean(),
                'std_gap_fuel_type_r2': ft_data['gap_fuel_type'].std(),
                'mean_gap_fuel_type_rrmse': ft_data['gap_fuel_type_rrmse'].mean(),
                'std_gap_fuel_type_rrmse': ft_data['gap_fuel_type_rrmse'].std(),
                'mean_alternative_r2': ft_data['alternative_r2'].mean(),
                'mean_alternative_rrmse': ft_data['alternative_rrmse'].mean()
            })

        return pd.DataFrame(summary_data)

    def print_fuel_type_comparison(self,
                                   targets: list[str]|None = None,
                                   show_rrmse:bool=True,
                                   show_r2:bool=True):
        """
        Print fuel type comparison table (best combination per target).

        Shows separate tables for RÂ² and RRMSE.


        """
        self.display.print_fuel_type_comparison(targets=targets,
                                                show_rrmse=show_rrmse,
                                                show_r2=show_r2)

    # Add new method
    def print_fuel_type_comparison_detailed(self, **kwargs):
        """
        Print detailed comparison showing all combinations.

        Shows separate tables for RÂ² and RRMSE, with all tested
        predictor combinations for each target.
        """
        self.display.print_fuel_type_comparison_detailed(**kwargs)

    def print_target_comparison(self, target: str, **kwargs):
        """Print detailed comparison for specific target."""
        self.display.print_target_comparison(target, **kwargs)


    def print_summary_statistics(self):
        """Print summary statistics."""
        self.display.print_summary_statistics()

    def generate_latex_tables(self,
                            ship_type:'ShipType',
                            output_dir: str = '.',
                            targets: Optional[List[str]] = None,
                            max_predictors: int = 2,
                            prefix: str = None,
                            best_only: bool = True,
                            show_r2_gaps: bool = True,
                            show_rrmse_gaps: bool = False,
                            landscape: bool = False,
                              suffix:str=None):
        """
        Generate and save LaTeX tables.

        Convenience method to create publication-ready tables.

        Parameters
        ----------
        **kwargs : passed to MultiDatasetLatexTableGenerator.save_tables()
            ship_type, output_dir, targets, max_predictors, etc.

        Returns
        -------
        dict
            Paths to saved table files

        Examples
        --------
        >>> multi_analyzer.generate_latex_tables(
        ...     ship_type=ShipType.TANKER,
        ...     output_dir='tables/',
        ...     best_only=True,
        ...     landscape=True  # Use for many fuel types
        ... )
        """
        from knowledge_layer.multi_dataset_analysers.multi_domain_LaTeX import MultiDatasetLatexTableGenerator

        generator = MultiDatasetLatexTableGenerator(self, verbose=True)

        generator.save_tables(ship_type=ship_type,
                                    output_dir=output_dir,
                                    targets=targets,
                                    max_predictors=max_predictors,
                                    prefix=prefix,
                                    best_only=best_only,
                                    landscape=landscape,
                              suffix=suffix)

        if self.analyzer_type == 'parametric':
            if self.selector is None:
                if self.verbose:
                    print("\n⚠️  No selector - skipping formula table")
                return

            # Get first analyzer
            first_fuel = list(self.alternative_datasets.keys())[0]
            analyzer = self.analyzers[first_fuel]

            # ✅ DEBUG: Check what's in the analyzer
            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"Formula Table Generation (using {first_fuel} analyzer)")
                print(f"{'=' * 70}")

                if hasattr(analyzer, 'results_df') and analyzer.results_df is not None:
                    print(f"Analyzer has results_df with {len(analyzer.results_df)} rows")
                    print(f"Targets in results_df: {sorted(analyzer.results_df['target'].unique())}")
                else:
                    print("⚠️  Analyzer has no results_df!")

                if hasattr(analyzer, 'results'):
                    print(f"Analyzer.results keys: {list(analyzer.results.keys())}")

                if hasattr(analyzer.foundation, 'selector'):
                    print(f"✅ Selector reference exists")
                    if hasattr(analyzer.foundation.selector, 'model_descriptors'):
                        n_descriptors = len(analyzer.foundation.selector.model_descriptors)
                        print(f"   Model descriptors: {n_descriptors}")
                        print(
                            f"   Descriptor keys (sample): {list(analyzer.foundation.selector.model_descriptors.keys())[:3]}")
                else:
                    print("❌ No selector reference!")

            # Ensure selector is set
            if not hasattr(analyzer.foundation, 'selector'):
                analyzer.foundation.selector = self.selector
            if not hasattr(analyzer.foundation, 'detailed_results'):
                analyzer.foundation.detailed_results = self.selector.detailed_results

            try:
                formula_gen = ParametricFormulaLatexGenerator(
                    domain_analyzer=analyzer,
                    verbose=True  # ← Turn on verbose for debugging
                )

                formula_gen.save_formula_table(
                    ship_type=ship_type,  # ← PASS ACTUAL PARAMETER
                    output_dir=output_dir,
                    targets=targets,
                    max_predictors=max_predictors,
                    prefix=f"{prefix}_formulas_{suffix}" if prefix else "formulas",
                    use_test_metrics=True,
                    show_metrics=True
                )

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    def run_significance_tests(self,
                               comparison: str = 'fuel_type',
                               n_bootstrap: int = 5000,
                               confidence_level: float = 0.95,
                               ci_method: str = 'bca',
                               random_seed: Optional[int] = None,
                               debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Run bootstrap significance tests across all alternative fuel datasets.

        Delegates to each per-fuel-type analyser, then aggregates results.

        Step-by-step data flow:
        -----------------------
        1. self.analyzers                → one analyser per fuel type
        2. analyser.run_significance_tests() → per-fuel bootstrap results
        3. Tag each result with fuel_type
        4. Build flat DataFrame for querying → self._significance_df

        Parameters
        ----------
        comparison : str
            Which gap to test: 'fuel_type', 'distribution', or 'total'
        n_bootstrap : int
            Bootstrap iterations per combination per fuel type
        confidence_level : float
            CI confidence level
        ci_method : str
            'percentile' or 'bca'
        random_seed : int, optional
            For reproducibility
        debug : bool
            Per-combination debug output

        Returns
        -------
        dict[str, list[dict]]
            Maps fuel_type → list of per-combination result dicts
        """
        if not self.analyzers:
            raise RuntimeError("No analysers found. Run analyze_all_datasets() first.")

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Bootstrap Significance Tests ({comparison} gap)")
            print(f"n_bootstrap={n_bootstrap}, CI={confidence_level * 100:.0f}%, "
                  f"method={ci_method}")
            print(f"{'=' * 70}\n")

        all_results = {}

        for fuel_type, analyser in self.analyzers.items():
            if self.verbose:
                print(f"  Testing: {fuel_type}...")

            fuel_results = analyser.run_significance_tests(
                comparison=comparison,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                ci_method=ci_method,
                random_seed=random_seed,
                debug=debug,
            )

            for r in fuel_results:
                r['fuel_type'] = fuel_type

            all_results[fuel_type] = fuel_results

            if self.verbose:
                n_sig = sum(1 for r in fuel_results if r.get('is_significant'))
                n_valid = sum(1 for r in fuel_results if r.get('is_significant') is not None)
                print(f"    → {n_sig}/{n_valid} combinations show significant gaps")

        # Cache for export
        self._significance_results = all_results
        self._significance_df = self._build_significance_dataframe(all_results)

        if self.verbose:
            print(f"\n✓ Significance tests complete for {len(self.analyzers)} fuel types")

        return all_results

    def _build_significance_dataframe(self,
                                      all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Flatten significance results into a single summary DataFrame.

        Data flow:
            dict[fuel_type → list[dict]]
            → tagged rows (fuel_type column added)
            → pd.DataFrame (one row per combination per fuel type)
        """
        from knowledge_layer.multi_dataset_analysers.TrainedModelBootstrapTester import (
            TrainedModelBootstrapTester
        )
        tester = TrainedModelBootstrapTester()
        rows = []

        for fuel_type, results in all_results.items():
            df = tester.results_to_dataframe(results)
            df['fuel_type'] = fuel_type
            rows.append(df)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def export_significance_results(self,
                                    output_path: str,
                                    significance_results: Optional[Dict] = None) -> None:
        """
        Export all significance results to a multi-sheet Excel file.

        One sheet per fuel type, plus a combined 'All_Results' sheet.

        Parameters
        ----------
        output_path : str
            Path for .xlsx output file
        significance_results : dict, optional
            Output of run_significance_tests(). If None, uses cached results.
        """
        from knowledge_layer.multi_dataset_analysers.TrainedModelBootstrapTester import (
            TrainedModelBootstrapTester
        )

        if significance_results is None:
            if not hasattr(self, '_significance_results'):
                raise RuntimeError(
                    "No significance results cached. Run run_significance_tests() first."
                )
            significance_results = self._significance_results

        tester = TrainedModelBootstrapTester()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for fuel_type, results in significance_results.items():
                df = tester.results_to_dataframe(results)
                safe_name = fuel_type[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
                ws = writer.sheets[safe_name]
                for col in ws.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col)
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

            if hasattr(self, '_significance_df') and len(self._significance_df) > 0:
                self._significance_df.to_excel(writer, sheet_name='All_Results', index=False)

        print(f"✓ Significance results exported to: {output_path}")
