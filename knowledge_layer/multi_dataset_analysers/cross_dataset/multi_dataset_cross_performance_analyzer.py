# multi_dataset_cross_performance.py (UPDATED)

"""
Multi-Dataset Cross-Performance Analyzer
========================================

Coordinates cross-dataset performance analysis across multiple alternative fuel datasets.
Supports BOTH parametric and Random Forest models.

Key Question:
"Does training on alternative ships improve predictions, and does this vary by fuel type?"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle

from knowledge_layer.multi_dataset_analysers.cross_dataset.parametric_cross_dataset_analysis import CrossDatasetPerformanceAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.rf_cross_dataset_analysis import RandomForestCrossDatasetAnalyzer
from knowledge_layer.multi_dataset_analysers.multi_domain_display import MultiDatasetDisplay
from knowledge_layer.multi_dataset_analysers.cross_dataset.ScaleReference import ScaleReference
from config.loggermixin.logger_config import LoggerMixin
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility

class MultiDatasetCrossPerformanceAnalyzer(LoggerMixin):
    """
    Analyzes cross-dataset performance across multiple alternative fuel types.

    **Supports both parametric and Random Forest models**.

    **Purpose**: Compare how training dataset choice affects prediction quality
    for different fuel types.

    **Three-Way Comparison Per Fuel Type**:
    - Train on Full Conventional → Test on all three datasets
    - Train on Matched Conventional → Test on all three datasets
    - Train on Alternative → Test on all three datasets

    **Multi-Fuel Comparison**:
    Compare these patterns across LNG, Methanol, Ammonia, etc.

    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_datasets: dict[str, pd.DataFrame],
                 independent_variables: list[str],
                 dependent_variables: list[str]|None = None,
                 analyzer_type: Literal['parametric', 'random_forest'] = 'parametric',
                 target_predictor_constraints: dict[str, list[str]] | None = None,
                 scale_reference:dict|None=None,
                 selection_metric:str='rrmse',
                 # Parametric-specific
                 parametric_selector=None,
                 model_recommendations: dict|None = None,
                 # RF-specific
                 rf_selector=None,
                 selector_results: dict|None = None,
                 rf_params: dict|None = None,
                 # Common
                 random_state: int = 42,
                 verbose: bool = True,
                 debug:bool=False,
                 build_year_col:str|None=None,
                 year_edges:list|None=None,
                 build_eligibility:TemporalEligibility|None=None):
        """
        Initialize multi-dataset cross-performance analyzer.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_datasets : dict[str, pd.DataFrame]
            Dictionary mapping fuel type names to datasets
        independent_variables : list[str]
            List of potential predictor dimensions
        dependent_variables : list[str], optional
            List of target dimensions
        analyzer_type : {'parametric', 'random_forest'}
            Type of analyzer to use

        **Parametric-specific parameters:**
        parametric_selector : ParametricModelSelector, optional
            Fitted selector with model_recommendations and model_descriptors
        model_recommendations : dict, optional
            Model recommendations (extracted from selector if provided)

        **Random Forest-specific parameters:**
        rf_selector : RandomForestSelector, optional
            Fitted RF selector with results
        selector_results : dict, optional
            RF selector results (extracted from rf_selector if provided)
        rf_params : dict, optional
            RandomForestRegressor parameters

        **Common parameters:**
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """

        self.conventional_data = conventional_data
        self.alternative_datasets = alternative_datasets
        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables or independent_variables
        self.analyzer_type = analyzer_type
        self.random_state = random_state
        self.verbose = verbose
        self.debug = debug
        self.selection_metric = selection_metric

        self.build_year_col = build_year_col
        self.year_edges = year_edges
        self.build_eligibility = build_eligibility

        self.set_log_level(verbose=self.verbose, debug=self.debug)

        self.target_predictor_constraints = target_predictor_constraints or {}
        if not self.target_predictor_constraints:
            self.logger.warning("No target_predictor_constraints supplied. Candidate selection will "
            "admit any recommended combination, including circular ones such as "
            "predicting DRAFT from DEPTH.")

        # ========================================================================
        # Extract from selector based on analyzer type
        # ========================================================================

        if analyzer_type == 'parametric':
            if parametric_selector is not None:
                self.selector = parametric_selector
                self.model_recommendations = parametric_selector.model_recommendations
                self.detailed_results = parametric_selector.detailed_results
            else:
                self.selector = None
                self.model_recommendations = model_recommendations
                self.detailed_results = {}

            self.rf_params = None
            self.selector_results = None

        elif analyzer_type == 'random_forest':
            if rf_selector is not None:
                self.selector = rf_selector
                self.selector_results = rf_selector.results
            else:
                self.selector = None
                self.selector_results = selector_results

            self.rf_params = rf_params or {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': random_state,
                'n_jobs': -1,
                'oob_score': True
            }

            self.model_recommendations = None
            self.detailed_results = {}

        else:
            raise ValueError(
                f"analyzer_type must be 'parametric' or 'random_forest', got: {analyzer_type}"
            )

        # Validate inputs
        self._validate_inputs()

        # Storage for individual analyzers
        self.analyzers: Dict[str, CrossDatasetPerformanceAnalyzer | RandomForestCrossDatasetAnalyzer] = {}

        # Storage for aggregated results
        self.comparison_results: Optional[pd.DataFrame] = None

        if self.verbose:
            self._print_initialization_summary()

        self._display = None

        if scale_reference is None:
            scale_reference = ScaleReference.from_frame(
                data=conventional_data,
                targets=self.dependent_variables,
                source="conventional_cohort_mean",
            )
        self.scale_reference = scale_reference

        if self.verbose:
            print(f"Reference scale: {self.scale_reference.source}")
            for target, mean in sorted(self.scale_reference.means.items()):
                print(f"    {target:<10s} {mean:>12.2f}")


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
                    f"Alternative dataset '{fuel_type}' missing columns: {missing}"
                )

        # Warn about sample sizes
        for fuel_type, df in self.alternative_datasets.items():
            if len(df) < 10:
                if self.verbose:
                    print(f"\n⚠️  Warning: {fuel_type} has only {len(df)} ships")
                    print(f"   Bootstrap evaluation recommended for robust estimates")

        # ✅ Analyzer-specific validation
        if self.analyzer_type == 'parametric':
            if self.model_recommendations is None:
                raise ValueError(
                    "For parametric analysis, either parametric_selector or "
                    "model_recommendations must be provided"
                )

        elif self.analyzer_type == 'random_forest':
            if self.selector_results is None:
                raise ValueError(
                    "For random_forest analysis, either rf_selector or "
                    "selector_results must be provided"
                )

    def _print_initialization_summary(self):
        """Print initialization summary."""
        print(f"\n{'=' * 70}")
        print(f"Multi-Dataset Cross-Performance Analyzer Initialized")
        print(f"{'=' * 70}")
        print(f"Analyzer type: {self.analyzer_type.upper()}")
        print(f"Conventional data: {len(self.conventional_data)} ships")
        print(f"Alternative datasets: {len(self.alternative_datasets)}")
        for fuel_type, df in self.alternative_datasets.items():
            print(f"  • {fuel_type}: {len(df)} ships")
        print(f"{'=' * 70}\n")

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_datasets(self,
                             evaluation_method: str = 'bootstrap',
                             n_folds: int = 5,
                             n_repeats: int = 10,
                             n_bootstraps: int = 100,
                             max_predictors: Optional[int] = None,
                             require_alternative_data: bool = False,
                             min_alternative_samples: int = 5,
                             display_results: bool = True,
                             cache_file: Optional[str] = None,
                             use_cache: bool = False) -> Dict[str, pd.DataFrame]:

        """
        Run cross-performance analysis on all alternative fuel datasets.

        **Works for both parametric and Random Forest models.**

        Parameters
        ----------
        evaluation_method : str, default='bootstrap'
            Evaluation strategy:
            - 'standard-cv': Simple k-fold CV
            - 'repeated-cv': k-fold CV repeated multiple times
            - 'bootstrap': Bootstrap .632+ 
        n_folds : int
            Number of CV folds
        n_repeats : int
            Number of repetitions (for 'repeated-cv')
        n_bootstraps : int
            Number of bootstrap samples
        max_predictors : int, optional
            Maximum number of predictors
        require_alternative_data : bool, default=False
            Whether to require sufficient alternative data.
            **Recommended False** for cross-performance - we want to test
            even on small datasets to see how performance degrades.
        min_alternative_samples : int
            Minimum samples in alternative dataset
        display_results : bool
            Display results for each dataset

        cache_file : str, optional
            Path to cache file. If None, auto-generates based on config.
        use_cache : bool, default=True
            If True, loads from cache if available and config matches.

        Returns
        -------
        dict[str, pd.DataFrame]
            Results DataFrames for each fuel type
        """

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Analyzing {len(self.alternative_datasets)} Alternative Fuel Datasets")
            print(f"Model Type: {self.analyzer_type.upper()}")
            print(f"Evaluation: {evaluation_method}")
            if evaluation_method == 'bootstrap':
                print(f"  Bootstrap samples: {n_bootstraps}")
            elif evaluation_method == 'repeated-cv':
                print(f"  CV: {n_folds}-fold × {n_repeats} repeats")
            else:
                print(f"  CV: {n_folds}-fold")
            print(f"{'=' * 70}\n")

        if cache_file is None and use_cache:
            cache_file = self._generate_cache_filename(
                evaluation_method,
                max_predictors,
                n_bootstraps
            )

            # ✅ Try to load from cache
        if use_cache and cache_file and Path(cache_file).exists():
            cached_data = self._load_from_cache(cache_file)

            if cached_data is not None:
                # Validate cache matches current config
                if self._validate_cache(cached_data, evaluation_method, max_predictors):
                    if self.verbose:
                        print(f"\n✅ Loaded results from cache: {cache_file}")
                        print(f"   Cached: {cached_data['timestamp']}")

                    # Restore state
                    self.analyzers = cached_data['analyzers']
                    self.comparison_results = cached_data['comparison_results']

                    return cached_data['all_results']
                else:
                    if self.verbose:
                        print(f"\n⚠️  Cache config mismatch - rerunning analysis")

        all_results = {}

        for fuel_type, alt_data in self.alternative_datasets.items():

            if self.verbose:
                print(f"\n{'─' * 70}")
                print(f"Analyzing: {fuel_type}")
                print(f"{'─' * 70}")

            # ✅ Create appropriate analyzer type
            if self.analyzer_type == 'parametric':
                analyzer = self._create_parametric_analyzer(alt_data)
            else:
                analyzer = self._create_rf_analyzer(alt_data)

            # ✅ Run analysis with appropriate method
            if self.analyzer_type == 'parametric':
                results_df = analyzer.analyze_best_combinations(
                    evaluation_method=evaluation_method,
                    n_folds=n_folds,
                    n_repeats=n_repeats,
                    n_bootstraps=n_bootstraps,
                    max_predictors=max_predictors,
                    require_alternative_data=require_alternative_data,
                    min_alternative_samples=min_alternative_samples,
                    display_results=display_results
                )

            else:  # random_forest
                results_df = analyzer.analyze_best_combinations(
                    evaluation_method=evaluation_method,
                    n_folds=n_folds,
                    n_repeats=n_repeats,
                    n_bootstraps=n_bootstraps,
                    require_alternative_data=require_alternative_data,
                    min_alternative_samples=min_alternative_samples,
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

        if cache_file:
            self._save_to_cache(
                cache_file,
                all_results,
                evaluation_method,
                max_predictors,
                n_bootstraps
            )

        return all_results

    def _create_parametric_analyzer(self, alternative_data: pd.DataFrame) -> CrossDatasetPerformanceAnalyzer:
        """Create parametric analyzer for one fuel type."""

        analyzer = CrossDatasetPerformanceAnalyzer(
            conventional_data=self.conventional_data,
            alternative_data=alternative_data,
            independent_variables=self.independent_variables,
            dependent_variables=self.dependent_variables,
            parametric_selector=self.selector,
            model_recommendations=self.model_recommendations,
            target_predictor_constraints = self.target_predictor_constraints,
            scale_reference=self.scale_reference,
            random_state=self.random_state,
            selection_metric=self.selection_metric,
            build_year_col=self.build_year_col,
            year_edges=self.year_edges,
            build_eligibility=self.build_eligibility,
            include_matched=False,
            verbose=True  # Suppress individual analyzer verbosity
        )

        return analyzer

    def _create_rf_analyzer(self, alternative_data: pd.DataFrame) -> RandomForestCrossDatasetAnalyzer:
        """Create RF analyzer for one fuel type."""

        analyzer = RandomForestCrossDatasetAnalyzer(
            conventional_data=self.conventional_data,
            alternative_data=alternative_data,
            dependent_variables=self.independent_variables,
            target_dimensions=self.dependent_variables,
            selector_results=self.selector_results,
            rf_params=self.rf_params,
            random_state=self.random_state,
            scale_reference=self.scale_reference,
            selection_metric=self.selection_metric,
            target_predictor_constraints=self.target_predictor_constraints,
            build_year_col=self.build_year_col,
            year_edges=self.year_edges,
            build_eligibility=self.build_eligibility,
            include_matched=False,
            verbose=True  # Suppress individual analyzer verbosity
        )

        return analyzer

    def _create_comparison_dataframe(self, all_results: Dict[str, pd.DataFrame]):
        """Create aggregated comparison DataFrame."""

        comparison_data = []

        for fuel_type, results_df in all_results.items():
            if results_df is None or len(results_df) == 0:
                if self.verbose:
                    print(f"  ℹ️  Skipping {fuel_type}: no valid results")
                continue

            # Add fuel_type column
            df_copy = results_df.copy()
            df_copy['fuel_type'] = fuel_type
            comparison_data.append(df_copy)

        if not comparison_data:
            if self.verbose:
                print("\n⚠️  Warning: No valid results from any dataset!")
            self.comparison_results = pd.DataFrame()
            return

        self.comparison_results = pd.concat(comparison_data, ignore_index=True)

        # Reorder columns
        priority_cols = [
            'fuel_type', 'target', 'predictor_str', 'model_type',
            'conv_q2', 'matched_q2', 'alt_q2',
            'conv_rrmse', 'matched_rrmse', 'alt_rrmse'
        ]

        ordered_cols = [c for c in priority_cols if c in self.comparison_results.columns]
        remaining_cols = [c for c in self.comparison_results.columns if c not in ordered_cols]

        self.comparison_results = self.comparison_results[ordered_cols + remaining_cols]

    # ========================================================================
    # COMPARISON AND VISUALIZATION METHODS (unchanged from before)
    # ========================================================================

    def print_cross_training_summary(self,
                                     targets: Optional[List[str]] = None,
                                     show_std: bool = True):
        """
        Print summary comparing cross-training performance across fuel types.

        **Works for both parametric and RF models.**

        Shows for each fuel type:
        - Performance when trained on conventional
        - Performance when trained on matched
        - Performance when trained on alternative
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run analyze_all_datasets() first.")
            return

        if targets is None:
            targets = self.dependent_variables

        print(f"\n{'=' * 120}")
        print(f"CROSS-TRAINING PERFORMANCE COMPARISON ({self.analyzer_type.upper()})")
        print(f"{'=' * 120}")
        print("\nQuestion: Does training on alternative ships improve predictions?")
        print(f"{'=' * 120}\n")

        for target in targets:
            target_data = self.comparison_results[
                self.comparison_results['target'] == target
                ]

            if len(target_data) == 0:
                continue

            print(f"\n{target}")
            print(f"{'─' * 120}")
            print(f"{'Fuel Type':<15s} | {'Predictors':<20s} | "
                  f"{'Conv R²':>12s} |  {'Alt R²':>12s} | "
                  f"{'Conv→Alt Δ':>12s}")
            print(f"{'─' * 120}")

            for _, row in target_data.iterrows():
                conv_str = f"{row['conv_q2']:.3f}"

                alt_str = f"{row['alt_q2']:.3f}"

                if show_std and 'conv_q2_std' in row:
                    conv_str += f"±{row['conv_q2_std']:.3f}"

                    alt_str += f"±{row['alt_q2_std']:.3f}"

                # Calculate drop from conventional to alternative
                delta = row['conv_q2'] - row['alt_q2']
                delta_str = f"{delta:+.3f}"

                print(f"{row['fuel_type']:<15s} | "
                      f"{row['predictor_str']:<20s} | "
                      f"{conv_str:>12s} | "

                      f"{alt_str:>12s} | "
                      f"{delta_str:>12s}")

            print(f"{'─' * 120}")

        print(f"\n{'=' * 120}")

    def plot_fuel_type_comparison(self,
                                  target: str,
                                  metric: str = 'q2',
                                  save_path: Optional[str] = None):
        """
        Plot comparison of a specific target across fuel types.

        **Works for both parametric and RF models.**

        Creates grouped bar chart showing Conv/Matched/Alt performance
        for each fuel type.
        """

        if self.comparison_results is None:
            print("No results available. Run analyze_all_datasets() first.")
            return

        target_data = self.comparison_results[
            self.comparison_results['target'] == target
            ]

        if len(target_data) == 0:
            print(f"No data for target: {target}")
            return

        # Prepare data for plotting
        fuel_types = target_data['fuel_type'].values
        conv_vals = target_data[f'conv_{metric}'].values

        alt_vals = target_data[f'alt_{metric}'].values

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(fuel_types))
        width = 0.25

        bars1 = ax.bar(x - width, conv_vals, width, label='Conventional',
                       alpha=0.8, color='#2E86AB')

        bars3 = ax.bar(x + width, alt_vals, width, label='Alternative',
                       alpha=0.8, color='#F18F01')

        ax.set_xlabel('Fuel Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')

        # ✅ Add model type to title
        ax.set_title(f'{target}: Performance Across Fuel Types ({self.analyzer_type.upper()})',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(fuel_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")

        plt.show()

    def plot_performance_heatmap(self,
                                 metric: str = 'alt_q2',
                                 targets: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        Plot heatmap of performance across fuel types and targets.

        **Works for both parametric and RF models.**
        """

        if self.comparison_results is None:
            print("No results available. Run analyze_all_datasets() first.")
            return

        if targets is None:
            targets = self.dependent_variables

        # Create pivot table
        pivot_data = self.comparison_results.pivot_table(
            index='fuel_type',
            columns='target',
            values=metric,
            aggfunc='first'
        )

        # Filter to requested targets
        available_targets = [t for t in targets if t in pivot_data.columns]
        pivot_data = pivot_data[available_targets]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1 if metric.endswith('_q2') else None)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(available_targets)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(available_targets, rotation=45, ha='right')
        ax.set_yticklabels(pivot_data.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.upper(), rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(available_targets)):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                   ha="center", va="center", color="black",
                                   fontsize=9, fontweight='bold')

        # ✅ Add model type to title
        ax.set_title(f'Performance Heatmap: {metric.upper()} ({self.analyzer_type.upper()})',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")

        plt.show()

    # ========================================================================
    # EXPORT METHODS (unchanged from before)
    # ========================================================================

    def export_comparison(self, filename: str = 'cross_performance_comparison.xlsx'):
        """Export comparison results to Excel with multiple sheets."""

        if self.comparison_results is None:
            raise ValueError("No results available. Run analyze_all_datasets() first.")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Full comparison data
            self.comparison_results.to_excel(writer, sheet_name='All_Results', index=False)

            # Sheet 2: Summary by fuel type
            summary = self._create_fuel_type_summary()
            summary.to_excel(writer, sheet_name='Fuel_Type_Summary', index=False)

            # Sheet 3: Summary by target
            target_summary = self._create_target_summary()
            target_summary.to_excel(writer, sheet_name='Target_Summary', index=False)

        print(f"✅ Comparison exported to {filename}")

    def _create_fuel_type_summary(self) -> pd.DataFrame:
        """Create summary statistics by fuel type."""

        summary_data = []

        for fuel_type in self.alternative_datasets.keys():
            ft_data = self.comparison_results[
                self.comparison_results['fuel_type'] == fuel_type
                ]

            if len(ft_data) == 0:
                continue

            summary_data.append({
                'fuel_type': fuel_type,
                'n_ships': len(self.alternative_datasets[fuel_type]),
                'n_targets_analyzed': len(ft_data),
                'mean_conv_q2': ft_data['conv_q2'].mean(),
                'mean_matched_q2': ft_data['matched_q2'].mean(),
                'mean_alt_q2': ft_data['alt_q2'].mean(),
                'mean_conv_rrmse': ft_data['conv_rrmse'].mean(),
                'mean_matched_rrmse': ft_data['matched_rrmse'].mean(),
                'mean_alt_rrmse': ft_data['alt_rrmse'].mean(),
                'mean_performance_drop': (ft_data['conv_q2'] - ft_data['alt_q2']).mean()
            })

        return pd.DataFrame(summary_data)

    def _create_target_summary(self) -> pd.DataFrame:
        """Create summary statistics by target."""

        summary_data = []

        for target in self.dependent_variables:
            target_data = self.comparison_results[
                self.comparison_results['target'] == target
                ]

            if len(target_data) == 0:
                continue

            summary_data.append({
                'target': target,
                'n_fuel_types': len(target_data),
                'mean_alt_q2': target_data['alt_q2'].mean(),
                'std_alt_q2': target_data['alt_q2'].std(),
                'min_alt_q2': target_data['alt_q2'].min(),
                'max_alt_q2': target_data['alt_q2'].max(),
                'best_fuel_type': target_data.loc[target_data['alt_q2'].idxmax(), 'fuel_type'],
                'worst_fuel_type': target_data.loc[target_data['alt_q2'].idxmin(), 'fuel_type']
            })

        return pd.DataFrame(summary_data)

    def generate_latex_tables(self,
                    ship_type: 'ShipType',
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    max_predictors: int = 2,
                    prefix: str = None,
                    best_only: bool = True,
                    include_matched: bool = False,
                    landscape: bool = False) -> Dict[str, Path]:
        """
        Generate and save LaTeX tables.

        Convenience method to create publication-ready tables.

        Parameters
        ----------
        ship_type : ShipType
            Ship type
        output_dir : str
            Output directory
        **kwargs : passed to MultiCrossPerformanceLatexTableGenerator.save_tables()
            include_matched, landscape, max_predictors, etc.

        Examples
        --------
        >>> multi_analyzer.generate_latex_tables(
        ...     ship_type=ShipType.TANKER,
        ...     output_dir='tables/',
        ...     include_matched=False,
        ...     landscape=True
        ... )
        """
        from knowledge_layer.multi_dataset_analysers.cross_dataset.cross_performance_LaTeX import \
            MultiCrossPerformanceLatexTableGenerator

        generator = MultiCrossPerformanceLatexTableGenerator(self, verbose=True)

        return generator.save_tables(
            ship_type=ship_type,
            output_dir=output_dir,
            targets=targets,
            max_predictors=max_predictors,
            prefix=prefix,
            best_only=best_only,
            landscape=landscape
        )

    def _generate_cache_filename(self,
                                 evaluation_method: str,
                                 max_predictors: Optional[int],
                                 n_bootstraps: int) -> str:
        """Generate cache filename based on analysis config."""

        fuel_types_str = "_".join(sorted(self.alternative_datasets.keys()))

        cache_name = (
            f"cross_performance_cache_"
            f"{self.analyzer_type}_"
            f"{fuel_types_str}_"
            f"{evaluation_method}_"
            f"maxp{max_predictors}_"
            f"nboot{n_bootstraps}.pkl"
        )

        return cache_name

    def _save_to_cache(self,
                       cache_file: str,
                       all_results: Dict,
                       evaluation_method: str,
                       max_predictors: Optional[int],
                       n_bootstraps: int):
        """Save results to cache file."""

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'analyzer_type': self.analyzer_type,
            'evaluation_method': evaluation_method,
            'max_predictors': max_predictors,
            'n_bootstraps': n_bootstraps,
            'fuel_types': list(self.alternative_datasets.keys()),
            'all_results': all_results,
            'comparison_results': self.comparison_results,
            'analyzers': self.analyzers  # Includes foundation data
        }

        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ Results cached to: {cache_file}")
            print(f"   Size: {size_mb:.1f} MB")

    def _load_from_cache(self, cache_file: str) -> Optional[Dict]:
        """Load results from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load cache: {e}")
            return None

    def _validate_cache(self,
                        cached_data: Dict,
                        evaluation_method: str,
                        max_predictors: Optional[int]) -> bool:
        """Validate cached data matches current config."""

        checks = [
            cached_data['analyzer_type'] == self.analyzer_type,
            cached_data['evaluation_method'] == evaluation_method,
            cached_data['max_predictors'] == max_predictors,
            set(cached_data['fuel_types']) == set(self.alternative_datasets.keys()),
        ]

        return all(checks)