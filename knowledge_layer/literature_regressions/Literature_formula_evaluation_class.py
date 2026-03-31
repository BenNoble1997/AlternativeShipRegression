import logging

import numpy as np
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from config.logger_config import LoggerMixin

import matplotlib.pyplot as plt

from knowledge_layer.literature_regressions.literature_formula_libraries_class import LiteratureFormulaLibrary
from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula
from knowledge_layer.model_plotting_utility_class import ModelPlotter
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.display_functions.literature_evaluation_display import LiteratureEvaluationDisplay
from knowledge_layer.literature_regressions.source_dataset_management_class import SourceDatasetManager


# ============================================================================
# LITERATURE FORMULA EVALUATOR (REFACTORED)
# ============================================================================

class LiteratureFormulaEvaluator(LoggerMixin):
    """
    Evaluate literature formulae on conventional vs alternative fuel datasets.

    REFACTORED: Now uses SourceDatasetManager for efficient per-source filtering.

    Key improvements:
    - Domain filtering done once per source (not per formula)
    - Clear separation: data preparation vs formula evaluation
    - Transparent reporting of data quality per source
    - Backward compatible with legacy interface

    Three-dataset comparison framework:
    - Conventional full: All conventional ships
    - Conventional matched: Conventional ships matched to alternative ships
    - Alternative: Alternative fuel ships

    This allows isolation of:
    1. Generalization error (train → test)
    2. Distribution shift (test → matched)
    3. Fuel type effect (matched → alternative)
    """

    def __init__(self,
                 conventional_data: pd.DataFrame = None,
                 alternative_data: pd.DataFrame = None,
                 dataset_manager: 'SourceDatasetManager' = None,
                 formulae: List['LiteratureFormula'] = None,
                 formulae_file: str = None,
                 shiptype: 'ShipType' = None,
                 sheet_name: str = None,
                 reference: str = "Literature",
                 target_dimensions: List[str] = None,
                 predictor_dimensions: List[str] = None,
                 verbose: bool = True,
                 display: bool = True):
        """
        Parameters
        ----------
        conventional_data : pd.DataFrame, optional
            Full conventional ship dataset (LEGACY interface)
        alternative_data : pd.DataFrame, optional
            Alternative fuel ships (LEGACY interface)
        dataset_manager : SourceDatasetManager, optional
            NEW: Pre-configured dataset manager with datasets.
            If provided, takes precedence over conventional_data/alternative_data.
            The manager will be used to filter datasets by source.
        formulae : List[LiteratureFormula], optional
            Explicit list of formulae to test
        formulae_file : str, optional
            Path to Excel/CSV file with formulae
        shiptype : ShipType, optional
            Ship type for selecting Excel sheet
        sheet_name : str, optional
            Explicit sheet name (overrides shiptype)
        reference : str
            Default reference for formulae
        target_dimensions : List[str], optional
            Only evaluate formulae with these target dimensions.
            If None, all targets are allowed.
        predictor_dimensions : List[str], optional
            Only evaluate formulae using these predictor dimensions.
            If None, all predictors are allowed.
        verbose : bool
            Print detailed progress
        display : bool
            Show initialization summary
        """

        self.verbose = verbose

        if self.verbose:
            self.logger.setLevel(logging.INFO)

        self.target_dimensions = target_dimensions
        self.predictor_dimensions = predictor_dimensions
        self.shiptype = shiptype

        # Get formulae - priority: explicit list > file > defaults
        if formulae is not None:
            self.formulae = formulae
        elif formulae_file is not None:
            self.formulae = LiteratureFormulaLibrary.from_file(
                file_path=formulae_file,
                reference=reference,
                shiptype=shiptype,
                sheet_name=sheet_name
            )
        else:
            self.formulae = LiteratureFormulaLibrary.get_default_formulae()

        # Filter formulae based on target and predictor dimensions
        self.formulae = self._filter_formulae(self.formulae)

        # NEW: Handle dataset manager vs legacy interface
        if dataset_manager is not None:
            # NEW PATH: User provided a pre-configured manager
            self.dataset_manager = dataset_manager
            self.using_manager = True

            # Extract domain info from formulae and filter datasets
            domain_map = SourceDatasetManager.extract_domains_from_library(self.formulae)
            self.dataset_manager.filter_datasets_by_source(domain_map)

        else:
            # LEGACY PATH: Create manager from raw datasets
            if conventional_data is None or alternative_data is None:
                raise ValueError(
                    "Must provide either dataset_manager or both "
                    "conventional_data and alternative_data"
                )

            if self.verbose:
                print("\nLegacy mode: Creating SourceDatasetManager from raw datasets...")

            # We need matched datasets for the legacy interface
            # Create them before initializing the manager
            self.predictor_groups = self._create_list_of_predictors()
            matched_datasets_dict = MatchedDataset.multiple_matched_datasets(
                alternative_fleet=alternative_data,
                conventional_fleet=conventional_data,
                logger=self.logger,
                predictor_groups=self.predictor_groups
            )

            # Use first matched dataset as the default matched dataset
            # (This is a simplification for backward compatibility)
            if len(matched_datasets_dict) > 0:
                first_key = list(matched_datasets_dict.keys())[0]
                matched_data = matched_datasets_dict[first_key]
            else:
                matched_data = conventional_data.copy()

            # Create manager with raw datasets
            self.dataset_manager = SourceDatasetManager(
                conventional=conventional_data,
                matched=matched_data,
                alternative=alternative_data,
                verbose=verbose
            )

            # Extract and apply domain filtering
            domain_map = SourceDatasetManager.extract_domains_from_library(self.formulae)
            self.dataset_manager.filter_datasets_by_source(domain_map)

            self.using_manager = False

        # Create matched datasets for each predictor group
        # (These are created from the filtered 'default' source datasets)
        self.predictor_groups = self._create_list_of_predictors()
        self.matched_datasets = self._create_matched_datasets()

        # Results storage
        self.results = None

        # Validate data and columns
        self._validate_data()
        self._validate_columns()

        if display and self.verbose:
            display_obj = LiteratureEvaluationDisplay(self)
            display_obj.print_initialization_summary()

            if formulae_file:
                print(f"  Source: {formulae_file}")

    def _create_list_of_predictors(self) -> list:
        """
        Create a list of unique predictor groups used by formulas.

        Returns
        -------
        list
            List of tuples, each representing a unique predictor combination
        """
        predictor_groups = []

        for formula in self.formulae:
            predictor_groups.append(tuple(formula.predictors))

        predictor_groups = list(set(predictor_groups))

        return predictor_groups

    def _create_matched_datasets(self) -> dict:
        """
        Create matched datasets for each predictor group.

        Uses the 'default' (unfiltered) source datasets to create matches.
        Source-specific filtering will be applied later during evaluation.

        Returns
        -------
        dict
            Dictionary mapping predictor tuples to matched DataFrames
        """
        # Get default source datasets for matching
        default_source = self.dataset_manager.get_datasets_for_source('default')

        return MatchedDataset.multiple_matched_datasets(
            alternative_fleet=default_source.get_dataset('alternative'),
            conventional_fleet=default_source.get_dataset('conventional'),
            logger=self.logger,
            predictor_groups=self.predictor_groups
        )

    def _filter_formulae(self, formulae: List['LiteratureFormula']) -> List['LiteratureFormula']:
        """
        Filter formulae based on target and predictor dimension constraints.

        Parameters
        ----------
        formulae : List[LiteratureFormula]
            Original list of formulae
        Returns
        -------
        List[LiteratureFormula]
            Filtered list of formulae
        """
        if not self.target_dimensions and not self.predictor_dimensions:
            # No filtering needed
            return formulae

        filtered = []
        excluded_count = 0
        excluded_reasons = {'target': 0, 'predictor': 0, 'both': 0}

        for formula in formulae:
            # Check target dimension
            target_ok = (self.target_dimensions is None or
                         formula.target in self.target_dimensions)

            # Check predictor dimensions (all predictors must be in allowed list)
            if self.predictor_dimensions is None:
                predictor_ok = True
            else:
                predictor_ok = all(pred in self.predictor_dimensions
                                   for pred in formula.predictors)

            # Include formula only if both checks pass
            if target_ok and predictor_ok:
                filtered.append(formula)
            else:
                excluded_count += 1
                if not target_ok and not predictor_ok:
                    excluded_reasons['both'] += 1
                elif not target_ok:
                    excluded_reasons['target'] += 1
                else:
                    excluded_reasons['predictor'] += 1

        # Report filtering results
        if excluded_count > 0 and self.verbose:
            display = LiteratureEvaluationDisplay(self)
            display.print_formulae_filtering_results(
                kept=len(filtered),
                excluded_count=excluded_count,
                excluded_reasons=excluded_reasons
            )

        return filtered

    def _validate_columns(self):
        """Check which formulae can be evaluated with available columns."""
        # Use default (unfiltered) datasets for validation
        default_source = self.dataset_manager.get_datasets_for_source('default')
        conventional = default_source.get_dataset('conventional')

        all_required_cols = set()
        for formula in self.formulae:
            all_required_cols.update(formula.predictors)
            all_required_cols.add(formula.target)

        available_cols = set(conventional.columns)
        missing_cols = all_required_cols - available_cols

        if missing_cols and self.verbose:

            # Count how many formulae will be affected
            affected_formulae = []
            for formula in self.formulae:
                formula_cols = set(formula.predictors + [formula.target])
                if formula_cols & missing_cols:  # Intersection
                    affected_formulae.append(formula.name)

            display = LiteratureEvaluationDisplay(self)
            display.print_column_validation_warnings(
                missing_cols=missing_cols,
                avaialable_cols=available_cols,
                affected_formulae=affected_formulae
            )

    def _validate_data(self):
        """Check data quality."""
        if self.verbose:
            display = LiteratureEvaluationDisplay(self)
            display.print_data_quality_warnings()

    def evaluate_single_formula(self, formula: 'LiteratureFormula') -> dict:
        """
        Evaluate a single formula on all three datasets.

        NEW: Retrieves pre-filtered datasets based on formula's source.

        Returns
        -------
        dict
            Metrics for all three datasets plus gap analysis
        """
        result = {
            'formula_name': formula.name,
            'formula_type': formula.formula_type,
            'predictors': formula.predictors,
            'target': formula.target,
            'reference': formula.reference,
            'bounds': formula.bounds
        }

        # NEW: Get filtered datasets for this formula's source
        source = formula.reference if formula.reference else 'default'
        source_data = self.dataset_manager.get_datasets_for_source(source)

        # Get matched dataset for this formula's predictors
        matching_parameters = tuple(formula.predictors)

        if matching_parameters not in self.matched_datasets:
            if self.verbose:
                print(f"  Warning: No matched dataset for predictors {formula.predictors}")
            matched_dataset = source_data.get_dataset('conventional')
        else:
            matched_dataset = self.matched_datasets[matching_parameters]

        # Calculate bounds from source-filtered conventional dataset
        if not formula.bounds_calculated and formula.bounds is not None:
            formula.calculate_bounds_from_data(
                source_data.get_dataset('conventional')
            )

        # Store bound limits for display
        if formula.lower_limits and formula.upper_limits:
            for pred in formula.predictors:
                if pred in formula.lower_limits:
                    result[f'bound_lower_{pred}'] = formula.lower_limits[pred]
                if pred in formula.upper_limits:
                    result[f'bound_upper_{pred}'] = formula.upper_limits[pred]

        # Get mean from the conventional dataset
        conventional_dataset = source_data.get_dataset('conventional')
        reference_mean = np.mean(conventional_dataset[formula.target].dropna())

        # NEW: Evaluate on source-filtered datasets
        for dataset_name, dataset in [
            ('conventional', source_data.get_dataset('conventional')),
            ('matched', matched_dataset),
            ('alternative', source_data.get_dataset('alternative'))
        ]:
            try:
                # Get filter info for this dataset (for reporting)
                if dataset_name in ['conventional', 'alternative']:
                    filter_info = source_data.get_filter_info(dataset_name)
                else:
                    filter_info = None  # Matched datasets have different filtering

                metrics = self._evaluate_on_dataset(
                    formula,
                    dataset,
                    dataset_name,
                    source_filter_info=filter_info,
                    reference_mean=reference_mean
                )
                result.update(metrics)
            except Exception as e:
                if self.verbose:
                    display = LiteratureEvaluationDisplay(self)
                    display.print_evaluation_error(formula.name, dataset_name, e)
                result[f'{dataset_name}_error'] = str(e)

        # Calculate gaps (similar to PolynomialDomainAnalyzer)
        if all(k in result for k in ['conventional_r2', 'matched_r2', 'alternative_r2']):
            # R² gaps
            result['gap_distribution'] = result['conventional_r2'] - result['matched_r2']
            result['gap_fuel_type'] = result['matched_r2'] - result['alternative_r2']
            result['gap_total'] = result['conventional_r2'] - result['alternative_r2']

            # RRMSE gaps (positive = worse performance)
            result['gap_distribution_rrmse'] = result['matched_rrmse'] / result['conventional_rrmse']
            result['gap_fuel_type_rrmse'] = result['alternative_rrmse'] / result['matched_rrmse']
            result['gap_total_rrmse'] = result['alternative_rrmse'] / result['conventional_rrmse']

            # MAPE gaps
            result['gap_distribution_mape'] = result['matched_mape'] - result['conventional_mape']
            result['gap_fuel_type_mape'] = result['alternative_mape'] - result['matched_mape']
            result['gap_total_mape'] = result['alternative_mape'] - result['conventional_mape']

        return result

    def _evaluate_on_dataset(self,
                             formula: 'LiteratureFormula',
                             dataset: pd.DataFrame,
                             dataset_name: str,
                             source_filter_info: 'FilteredDatasetInfo' = None,
                             reference_mean:float|None=None) -> dict:
        """
        Evaluate formula on a single dataset.

        SIMPLIFIED: Domain filtering already done by SourceDatasetManager.
        This method now only handles data cleaning, prediction, and metrics.

        Parameters
        ----------
        formula : LiteratureFormula
            Formula to evaluate
        dataset : pd.DataFrame
            PRE-FILTERED dataset (domain constraints already applied by manager)
        dataset_name : str
            Name of dataset ('conventional', 'matched', 'alternative')
        source_filter_info : FilteredDatasetInfo, optional
            Information about source-level filtering (for reporting)

        Returns
        -------
        dict
            Metrics for this dataset
        """

        # Check if required columns exist
        missing_cols = [col for col in formula.predictors + [formula.target]
                        if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Get data
        X = dataset[formula.predictors].values
        y = dataset[formula.target].values

        # Handle single predictor case
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Clean data (remove NaN/inf) - ONLY data quality cleaning
        # Domain filtering already done by SourceDatasetManager!
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 3:
            raise ValueError(f"Insufficient clean data: {len(X_clean)} samples")

        # NO DOMAIN FILTERING HERE - already done by SourceDatasetManager!
        X_final = X_clean
        y_final = y_clean

        # Make predictions
        y_pred = formula.predict(X_final)

        # Calculate metrics
        custom_metrics = RegressionMetrics(y_pred, y_final)

        metrics = {
            f'{dataset_name}_r2': r2_score(y_final, y_pred),
            f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y_final, y_pred)),
            f'{dataset_name}_rrmse': custom_metrics.calculate_rrmse(y_mean=reference_mean),
            f'{dataset_name}_mae': mean_absolute_error(y_final, y_pred),
            f'{dataset_name}_mape': custom_metrics.calculate_mape(),
            f'{dataset_name}_n': len(y_final),
        }

        # NEW: Add source filtering info to metrics
        if source_filter_info is not None:
            metrics[f'{dataset_name}_n_excluded_domain'] = source_filter_info.n_excluded
            metrics[f'{dataset_name}_pct_excluded_domain'] = source_filter_info.pct_excluded

        # Check bounds (percentile-based, separate from domain)
        if formula.bounds is not None:
            if not formula.bounds_calculated:
                formula.calculate_bounds_from_data(dataset)

            within_bounds_mask, n_out_of_bounds = formula.check_bounds(X_final)
            metrics[f'{dataset_name}_n_out_of_bounds'] = n_out_of_bounds
            metrics[f'{dataset_name}_pct_out_of_bounds'] = (
                100 * n_out_of_bounds / len(y_final) if len(y_final) > 0 else 0
            )

        return metrics

    def evaluate_all_formulae(self,
                              display_results: bool = True,
                              metrics: str | list = 'all') -> pd.DataFrame:
        """
        Evaluate all formulae on all three datasets.

        Returns
        -------
        pd.DataFrame
            Results for all formulae
        """

        results = []

        for formula in self.formulae:
            result = self.evaluate_single_formula(formula)
            if result is None:
                continue
            results.append(result)

        self.results = pd.DataFrame(results)

        # Display results if requested
        if display_results and self.verbose and len(results) > 0:
            display = LiteratureEvaluationDisplay(self)

            # Data quality report
            display.print_data_quality_report(results)

            # Performance results table
            display.print_performance_results_table(results, metrics=metrics)

        return self.results

    def interpret_results(self,
                          significance_threshold_r2: float = 0.05,
                          significance_threshold_rrmse: float = 1.5,
                          display_results: bool = True):
        """
        Interpret which formulae show significant fuel-type effects.

        Parameters
        ----------
        significance_threshold_r2 : float
            R² gap threshold for significance
        significance_threshold_rrmse : float
            RRMSE gap threshold in % for significance
        display_results : bool
            Whether to print interpretation
        """
        if self.results is None or len(self.results) == 0:
            if self.verbose:
                display = LiteratureEvaluationDisplay(self)
                display.print_no_results_warning(context="interpret")
            self.evaluate_all_formulae()

        if not display_results or not self.verbose:
            return  # Silent mode - just ensure results exist

        # All display through display class
        display = LiteratureEvaluationDisplay(self)

        display.print_interpretation_header()

        # Filter to valid results
        valid_results = self.results.dropna(subset=['gap_fuel_type', 'gap_fuel_type_rrmse'])

        # Classify formulae by fuel-type impact
        for _, row in valid_results.iterrows():
            gap_r2 = row['gap_fuel_type']
            gap_rrmse = row['gap_fuel_type_rrmse']

            # Use DisplayFormatters for classification
            effect, icon = display.fmt.classify_effect(
                gap_r2=gap_r2,
                gap_rrmse=gap_rrmse,
                threshold_r2=significance_threshold_r2,
                threshold_rrmse=significance_threshold_rrmse
            )

            display.print_formula_interpretation(
                row=row,
                effect=effect,
                icon=icon,
                significance_threshold_r2=significance_threshold_r2,
                significance_threshold_rrmse=significance_threshold_rrmse
            )

        # Summary statistics
        display.print_interpretation_summary(
            valid_results=valid_results,
            significance_threshold_r2=significance_threshold_r2,
            significance_threshold_rrmse=significance_threshold_rrmse
        )

    def export_results(self, output_path: str) -> None:
        """
        Export results to CSV file.

        Parameters
        ----------
        output_path : str
            Path for output CSV file
        """
        if self.results is None or len(self.results) == 0:
            print("No results to export. Run evaluate_all_formulae() first.")
            return

        self.results.to_csv(output_path, index=False)

        if self.verbose:
            print(f"\nResults exported to: {output_path}")

    def plot_comparison(self,
                        save_path: str = None,
                        show_values: bool = True,
                        variable_name_map: dict[str, str]|None = None,
                        alternative_name: str|None = None):
        """
        Plot three-dataset comparison of formula performance.
        Uses ModelPlotter utilities for consistency with domain analysis plots.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
        show_values : bool, default=True
            Whether to show metric values above bars
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        alternative_name : str, optional
            Name of alternative fuel dataset to include in title.
            Example: 'Dual-Fuel Ships', 'LNG Ships', 'Alternative Fuel Ships'
            Will be appended in brackets: "(Dual-Fuel Ships)"
        """
        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        # Filter valid results
        valid = self.results.dropna(subset=['gap_fuel_type', 'gap_fuel_type_rrmse'])

        if len(valid) == 0:
            print("No valid results to plot.")
            return

        # Build title with ship type and alternative dataset name
        title_parts = []
        if self.shiptype is not None:
            title_parts.append(f"{self.shiptype.string}")
        title_parts.append("Literature Formula Evaluation: Three-Dataset Comparison")

        # Add alternative dataset name if provided
        if alternative_name:
            title_parts.append(f"({alternative_name})")

        plot_title = " | ".join(title_parts)

        # Setup figure using ModelPlotter
        fig, axes = ModelPlotter.setup_figure(
            n_plots=2,
            figsize_per_plot=(9, 6),
            title=plot_title,
            max_cols=2
        )

        # Prepare data
        x = np.arange(len(valid))
        width = 0.25

        # Create formatted labels: "predictor → target\n(formula_type)"
        labels = []
        for _, row in valid.iterrows():
            # Format predictors
            if variable_name_map:
                predictors_formatted = [variable_name_map.get(p, p) for p in row['predictors']]
                predictors_str = ' + '.join(predictors_formatted)
                target_str = variable_name_map.get(row['target'], row['target'])
            else:
                predictors_str = ' + '.join(row['predictors'])
                target_str = row['target']

            # Create label: "predictor → target\n(formula_type)"
            label = f"{predictors_str} → {target_str}\n({row['formula_type']})"
            labels.append(label)

        # Plot 1: R² comparison
        ax1 = axes[0]

        bars1_conv = ax1.bar(x - width, valid['conventional_r2'], width,
                             label='Conventional', alpha=0.8, color='steelblue')
        bars1_match = ax1.bar(x, valid['matched_r2'], width,
                              label='Matched (Conv)', alpha=0.8, color='orange')
        bars1_alt = ax1.bar(x + width, valid['alternative_r2'], width,
                            label='Alternative (Fuel)', alpha=0.8, color='coral')

        # Add value labels if requested
        if show_values:
            for bars, values in [
                (bars1_conv, valid['conventional_r2']),
                (bars1_match, valid['matched_r2']),
                (bars1_alt, valid['alternative_r2'])
            ]:
                ModelPlotter.add_bar_labels(
                    ax1, bars, values.tolist(),
                    include_std=False, fontsize=7
                )

        # Format using ModelPlotter
        ModelPlotter.format_axes(
            ax1,
            xlabel='Formula',
            ylabel=ModelPlotter.get_metric_label('r2'),
            title='R² Performance: Three-Dataset Comparison',
            x_labels=labels,
            rotation=45
        )
        ax1.legend()

        # Plot 2: RRMSE comparison
        ax2 = axes[1]

        bars2_conv = ax2.bar(x - width, valid['conventional_rrmse'], width,
                             label='Conventional', alpha=0.8, color='steelblue')
        bars2_match = ax2.bar(x, valid['matched_rrmse'], width,
                              label='Matched (Conv)', alpha=0.8, color='orange')
        bars2_alt = ax2.bar(x + width, valid['alternative_rrmse'], width,
                            label='Alternative (Fuel)', alpha=0.8, color='coral')

        # Add value labels if requested
        if show_values:
            for bars, values in [
                (bars2_conv, valid['conventional_rrmse']),
                (bars2_match, valid['matched_rrmse']),
                (bars2_alt, valid['alternative_rrmse'])
            ]:
                ModelPlotter.add_bar_labels(
                    ax2, bars, values.tolist(),
                    include_std=False, fontsize=7
                )

        # Format using ModelPlotter
        ModelPlotter.format_axes(
            ax2,
            xlabel='Formula',
            ylabel=ModelPlotter.get_metric_label('rrmse'),
            title='RRMSE Performance: Three-Dataset Comparison',
            x_labels=labels,
            rotation=45
        )
        ax2.legend()

        # Save and show using ModelPlotter
        ModelPlotter.save_and_show(fig, save_path, verbose=True)

    def plot_comparison_scatter(self, save_dir: str = None, figsize: tuple = (18, 6)):
        """
        Create comparison scatter plots for each formula.

        For each formula, creates a figure with three subplots showing:
        - Conventional dataset predictions vs actual
        - Matched dataset predictions vs actual
        - Alternative dataset predictions vs actual

        Parameters
        ----------
        save_dir : str, optional
            Directory to save plots. If None, plots are displayed but not saved.
        figsize : tuple, optional
            Figure size as (width, height). Default is (18, 6).
        """
        from pathlib import Path

        if self.results is None or len(self.results) == 0:
            print("No results to plot. Run evaluate_all_formulae() first.")
            return

        # Create save directory if needed
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving plots to: {save_path}")

        # Filter to valid results
        valid = self.results.dropna(subset=['gap_fuel_type', 'gap_fuel_type_rrmse'])

        if len(valid) == 0:
            print("No valid results to plot.")
            return

        print(f"\nCreating scatter plots for {len(valid)} formulae...")

        for idx, row in valid.iterrows():
            formula_name = row['formula_name']
            target = row['target']
            source = row['reference'] if row['reference'] else 'default'

            # Find the formula object
            formula = None
            for f in self.formulae:
                if f.name == formula_name:
                    formula = f
                    break

            if formula is None:
                print(f"  Warning: Could not find formula object for {formula_name}")
                continue

            # Get source-filtered datasets
            source_data = self.dataset_manager.get_datasets_for_source(source)

            title_parts = []

            if self.shiptype is not None:
                title_parts.append(f"{self.shiptype.name}")

            title_parts.append(f"{formula_name} → {target}")

            reference = formula.reference if formula.reference else "No reference"
            title_parts.append(f"[{reference}]")

            plot_title = " | ".join(title_parts)

            # Setup figure using ModelPlotter
            fig, axes = ModelPlotter.setup_figure(
                n_plots=3,
                figsize_per_plot=(6, 6),
                title=plot_title,
                max_cols=3
            )

            matching_parameters = tuple(formula.predictors)

            if matching_parameters in self.matched_datasets:
                matched_dataset = self.matched_datasets[matching_parameters]
            else:
                matched_dataset = source_data.get_dataset('conventional')

            # Plot each dataset
            for ax_idx, (dataset_name, dataset, color) in enumerate([
                ('Conventional', source_data.get_dataset('conventional'), 'steelblue'),
                ('Matched (Conv)', matched_dataset, 'orange'),
                ('Alternative (Fuel)', source_data.get_dataset('alternative'), 'coral')
            ]):
                ax = axes[ax_idx]

                try:
                    # Get data
                    X = dataset[formula.predictors].values
                    y_true = dataset[formula.target].values

                    # Handle single predictor case
                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)

                    # Clean data (remove NaN/inf)
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_true) |
                             np.isinf(X).any(axis=1) | np.isinf(y_true))
                    X_clean = X[mask]
                    y_true_clean = y_true[mask]

                    if len(X_clean) < 3:
                        ax.text(0.5, 0.5, 'Insufficient data',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(dataset_name, fontweight='bold')
                        continue

                    # Make predictions
                    y_pred = formula.predict(X_clean)

                    # Get metrics for this dataset
                    dataset_key = {
                        'Conventional': 'conventional',
                        'Matched (Conv)': 'matched',
                        'Alternative (Fuel)': 'alternative'
                    }[dataset_name]

                    r2 = row[f'{dataset_key}_r2']
                    rrmse = row[f'{dataset_key}_rrmse']
                    n = row[f'{dataset_key}_n']

                    # Create scatter plot
                    ax.scatter(y_true_clean, y_pred, alpha=0.6, s=50,
                               color=color, edgecolors='black', linewidth=0.5)

                    # Add perfect prediction line
                    min_val = min(y_true_clean.min(), y_pred.min())
                    max_val = max(y_true_clean.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val],
                            'k--', linewidth=2, label='Perfect prediction', alpha=0.7)

                    # Set labels and title
                    ax.set_xlabel(f'Actual {target}', fontweight='bold')
                    ax.set_ylabel(f'Predicted {target}', fontweight='bold')
                    ax.set_title(f'{dataset_name}\nR²={r2:.3f}, RRMSE={rrmse:.1f}%',
                                 fontweight='bold')

                    # Add grid
                    ax.grid(True, alpha=0.3)

                    # Add legend
                    ax.legend(loc='upper left', fontsize=9)

                    # Add sample count
                    ax.text(0.98, 0.02, f'n={n}',
                            transform=ax.transAxes,
                            ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Equal aspect ratio for better visualization
                    ax.set_aspect('equal', adjustable='datalim')

                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=8, color='red')
                    ax.set_title(dataset_name, fontweight='bold')

            # Save or show using ModelPlotter utility
            if save_dir is not None:
                safe_name = formula_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                filename = save_path / f'{safe_name}_scatter.png'
                ModelPlotter.save_and_show(fig, str(filename), verbose=False)
                print(f"  ✓ Saved: {filename.name}")
            else:
                ModelPlotter.save_and_show(fig, None, verbose=False)

        if save_dir is not None:
            print(f"\n✓ All scatter plots saved to {save_path}")

    def plot_comparison_by_reference(self,
                                     save_path: str = None,
                                     show_values: bool = True,
                                     variable_name_map: dict[str, str] = None):
        """
        Plot three-dataset comparison grouped by reference.

        Creates separate bar chart figures for each unique reference source.
        This allows comparing formulas from the same source/paper.

        Parameters
        ----------
        save_path : str, optional
            Base path for saving plots (will append reference to filename)
        show_values : bool, default=True
            Whether to show metric values above bars
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        """
        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        # Filter valid results
        valid = self.results.dropna(subset=['gap_fuel_type', 'gap_fuel_type_rrmse'])

        if len(valid) == 0:
            print("No valid results to plot.")
            return

        # Handle missing references
        valid['reference'] = valid['reference'].fillna('No reference')

        # Group by reference
        references = valid['reference'].unique()

        print(f"\nCreating comparison plots for {len(references)} reference(s)...")

        for reference in references:
            # Filter formulas from this reference
            ref_formulas = valid[valid['reference'] == reference]

            if len(ref_formulas) == 0:
                continue

            print(f"\n  Reference: {reference} ({len(ref_formulas)} formulae)")

            # Build title
            title_parts = []
            if self.shiptype is not None:
                title_parts.append(f"{self.shiptype.name}")
            title_parts.append(f"Literature Formula Evaluation")
            title_parts.append(f"[{reference}]")
            plot_title = " | ".join(title_parts)

            # Setup figure using ModelPlotter
            fig, axes = ModelPlotter.setup_figure(
                n_plots=2,
                figsize_per_plot=(9, 6),
                title=plot_title,
                max_cols=2
            )

            # Prepare data
            x = np.arange(len(ref_formulas))
            width = 0.25

            # Create formatted labels: "predictor → target\n(formula_type)"
            labels = []
            for _, row in ref_formulas.iterrows():
                # Format predictors
                if variable_name_map:
                    predictors_formatted = [variable_name_map.get(p, p) for p in row['predictors']]
                    predictors_str = ' + '.join(predictors_formatted)
                    target_str = variable_name_map.get(row['target'], row['target'])
                else:
                    predictors_str = ' + '.join(row['predictors'])
                    target_str = row['target']

                # Create label: "predictor → target\n(formula_type)"
                label = f"{predictors_str} → {target_str}\n({row['formula_type']})"
                labels.append(label)

            # Plot 1: R² comparison
            ax1 = axes[0]

            bars1_conv = ax1.bar(x - width, ref_formulas['conventional_r2'], width,
                                 label='Conventional', alpha=0.8, color='steelblue')
            bars1_match = ax1.bar(x, ref_formulas['matched_r2'], width,
                                  label='Matched (Conv)', alpha=0.8, color='orange')
            bars1_alt = ax1.bar(x + width, ref_formulas['alternative_r2'], width,
                                label='Alternative (Fuel)', alpha=0.8, color='coral')

            # Add value labels if requested
            if show_values:
                for bars, values in [
                    (bars1_conv, ref_formulas['conventional_r2']),
                    (bars1_match, ref_formulas['matched_r2']),
                    (bars1_alt, ref_formulas['alternative_r2'])
                ]:
                    ModelPlotter.add_bar_labels(
                        ax1, bars, values.tolist(),
                        include_std=False, fontsize=7
                    )

            # Format using ModelPlotter
            ModelPlotter.format_axes(
                ax1,
                xlabel='Formula',
                ylabel=ModelPlotter.get_metric_label('r2'),
                title='R² Performance: Three-Dataset Comparison',
                x_labels=labels,
                rotation=45
            )
            ax1.legend()

            # Plot 2: RRMSE comparison
            ax2 = axes[1]

            bars2_conv = ax2.bar(x - width, ref_formulas['conventional_rrmse'], width,
                                 label='Conventional', alpha=0.8, color='steelblue')
            bars2_match = ax2.bar(x, ref_formulas['matched_rrmse'], width,
                                  label='Matched (Conv)', alpha=0.8, color='orange')
            bars2_alt = ax2.bar(x + width, ref_formulas['alternative_rrmse'], width,
                                label='Alternative (Fuel)', alpha=0.8, color='coral')

            # Add value labels if requested
            if show_values:
                for bars, values in [
                    (bars2_conv, ref_formulas['conventional_rrmse']),
                    (bars2_match, ref_formulas['matched_rrmse']),
                    (bars2_alt, ref_formulas['alternative_rrmse'])
                ]:
                    ModelPlotter.add_bar_labels(
                        ax2, bars, values.tolist(),
                        include_std=False, fontsize=7
                    )

            # Format using ModelPlotter
            ModelPlotter.format_axes(
                ax2,
                xlabel='Formula',
                ylabel=ModelPlotter.get_metric_label('rrmse'),
                title='RRMSE Performance: Three-Dataset Comparison',
                x_labels=labels,
                rotation=45
            )
            ax2.legend()

            # Save with reference-specific filename
            if save_path is not None:
                # Create safe filename from reference
                safe_reference = reference.replace('/', '_').replace('\\', '_').replace(' ', '_')
                safe_reference = safe_reference.replace(':', '').replace('|', '')[:50]  # Limit length

                # Insert reference into filename
                base_path = Path(save_path)
                if base_path.suffix:
                    # Has extension: insert before it
                    ref_path = base_path.with_name(f"{base_path.stem}_{safe_reference}{base_path.suffix}")
                else:
                    # No extension: append
                    ref_path = Path(f"{save_path}_{safe_reference}.png")

                ModelPlotter.save_and_show(fig, str(ref_path), verbose=False)
                print(f"    ✅ Saved: {ref_path.name}")
            else:
                ModelPlotter.save_and_show(fig, None, verbose=False)

        print(f"\n✅ Created {len(references)} comparison plot(s) by reference")

        print(f"\n✓ Created {len(references)} comparison plot(s) by reference")

    def plot_formula_fit(self, save_dir: str = None, figsize: tuple = (18, 6), n_points: int = 100):
        """
        Plot the formula's fitted relationship for each dataset.

        For each formula, creates a figure with three subplots showing:
        - Predictor vs Target with formula's fitted line
        - One subplot for each dataset (Conventional, Matched, Alternative)
        - Domain boundaries highlighted (from source filtering)

        NOTE: Only works for single-predictor formulas.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save plots.
        figsize : tuple, optional
            Figure size as (width, height).
        n_points : int, optional
            Number of points to use for the fitted line.
        """
        from pathlib import Path

        if self.results is None or len(self.results) == 0:
            print("No results to plot. Run evaluate_all_formulae() first.")
            return

        # Create save directory if needed
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving plots to: {save_path}")

        # Filter to valid results
        valid = self.results.dropna(subset=['gap_fuel_type', 'gap_fuel_type_rrmse'])

        if len(valid) == 0:
            print("No valid results to plot.")
            return

        print(f"\nCreating formula fit plots for {len(valid)} formulae...")

        for idx, row in valid.iterrows():
            formula_name = row['formula_name']
            target = row['target']
            source = row['reference'] if row['reference'] else 'default'

            # Find the formula object
            formula = None
            for f in self.formulae:
                if f.name == formula_name:
                    formula = f
                    break

            if formula is None:
                print(f"  Warning: Could not find formula object for {formula_name}")
                continue

            # Only works for single predictor formulae
            if len(formula.predictors) != 1:
                print(f"  Skipping {formula_name}: only single-predictor formulae supported")
                continue

            predictor = formula.predictors[0]

            # Get source-filtered datasets
            source_data = self.dataset_manager.get_datasets_for_source(source)
            domain_constraints = source_data.domain_constraints

            title_parts = []

            # Add ship type if available
            if self.shiptype is not None:
                title_parts.append(f"{self.shiptype.name}")

            # Add formula name
            title_parts.append(f"{predictor} → {target}")

            # Add reference
            reference = formula.reference if formula.reference else "No reference"
            title_parts.append(f"[{reference}]")

            plot_title = " | ".join(title_parts)

            # Setup figure using ModelPlotter
            fig, axes = ModelPlotter.setup_figure(
                n_plots=3,
                figsize_per_plot=(6, 6),
                title=plot_title,
                max_cols=3
            )

            matching_parameters = tuple(formula.predictors)

            if matching_parameters in self.matched_datasets:
                matched_dataset = self.matched_datasets[matching_parameters]
            else:
                matched_dataset = source_data.get_dataset('conventional')

            # Plot each dataset
            for ax_idx, (dataset_name, dataset, color) in enumerate([
                ('Conventional', source_data.get_dataset('conventional'), 'steelblue'),
                ('Matched (Conv)', matched_dataset, 'orange'),
                ('Alternative (Fuel)', source_data.get_dataset('alternative'), 'coral')
            ]):
                ax = axes[ax_idx]

                try:
                    # Get data
                    X = dataset[predictor].values
                    y_true = dataset[target].values

                    # Clean data (remove NaN/inf)
                    mask = ~(np.isnan(X) | np.isnan(y_true) | np.isinf(X) | np.isinf(y_true))
                    X_clean = X[mask]
                    y_true_clean = y_true[mask]

                    if len(X_clean) < 3:
                        ax.text(0.5, 0.5, 'Insufficient data',
                                ha='center', va='center', transform=ax.transAxes,
                                fontsize=10, color='red')
                        ax.set_title(dataset_name, fontweight='bold')
                        continue

                    # Get metrics for this dataset
                    dataset_key = {
                        'Conventional': 'conventional',
                        'Matched (Conv)': 'matched',
                        'Alternative (Fuel)': 'alternative'
                    }[dataset_name]

                    r2 = row[f'{dataset_key}_r2']
                    rrmse = row[f'{dataset_key}_rrmse']
                    n = row[f'{dataset_key}_n']

                    # Plot actual data points
                    ax.scatter(X_clean, y_true_clean, alpha=0.5, s=40,
                               color=color, edgecolors='black', linewidth=0.5,
                               label='Actual data', zorder=3)

                    # Create fitted line
                    X_min = X_clean.min()
                    X_max = X_clean.max()
                    X_range = X_max - X_min

                    # Extend range slightly for better visualization
                    X_plot_min = X_min - 0.05 * X_range
                    X_plot_max = X_max + 0.05 * X_range

                    # Handle case where all values might be zero/negative for power laws
                    if formula.formula_type in ['power', 'power_log', 'piecewise_power']:
                        X_plot_min = max(X_plot_min, X_clean[X_clean > 0].min() * 0.95 if np.any(X_clean > 0) else 0.1)

                    X_line = np.linspace(X_plot_min, X_plot_max, n_points).reshape(-1, 1)

                    # Make predictions for the line
                    try:
                        y_line = formula.predict(X_line)

                        # Plot fitted line
                        ax.plot(X_line, y_line, 'r-', linewidth=2.5,
                                label='Formula fit', alpha=0.8, zorder=4)

                        # Add bin boundary indicators for piecewise formulae
                        if formula.formula_type in ['piecewise_power', 'piecewise_linear', 'piecewise_mixed']:
                            if formula.bins is not None:
                                for i, bin_edge in enumerate(formula.bins[1:-1], 1):  # Skip first and last
                                    ax.axvline(bin_edge, color='darkred', linestyle='--',
                                               linewidth=1.5, alpha=0.6,
                                               label='Bin boundary' if i == 1 else '',
                                               zorder=2)

                                    # Add bin edge label
                                    ax.text(bin_edge, ax.get_ylim()[1] * 0.95,
                                            f'{bin_edge:,.0f}',
                                            rotation=90, va='top', ha='right',
                                            fontsize=8, color='darkred', alpha=0.7)

                    except Exception as e:
                        print(f"    Warning: Could not plot fit line for {dataset_name}: {e}")

                    # Highlight domain boundaries from source filtering
                    domain_label_added = False

                    if predictor in domain_constraints.lower_bounds:
                        lower_bound = domain_constraints.lower_bounds[predictor]
                        ax.axvline(lower_bound, color='purple', linestyle='-.',
                                   linewidth=2, alpha=0.7,
                                   label='Domain bounds' if not domain_label_added else '',
                                   zorder=5)
                        domain_label_added = True

                    if predictor in domain_constraints.upper_bounds:
                        upper_bound = domain_constraints.upper_bounds[predictor]
                        ax.axvline(upper_bound, color='purple', linestyle='-.',
                                   linewidth=2, alpha=0.7,
                                   label='Domain bounds' if not domain_label_added else '',
                                   zorder=5)

                    # Percentile bounds (if different from domain)
                    if formula.bounds is not None and formula.lower_limits:
                        lower_limit = formula.lower_limits.get(predictor)
                        upper_limit = formula.upper_limits.get(predictor)

                        if lower_limit is not None and upper_limit is not None:
                            # Add dashed lines for percentile bounds
                            ax.axvline(lower_limit, color='gray',
                                       linestyle=':', linewidth=1.5, alpha=0.5,
                                       label=f'Percentile bounds ({formula.bounds}%)',
                                       zorder=2)
                            ax.axvline(upper_limit, color='gray',
                                       linestyle=':', linewidth=1.5, alpha=0.5,
                                       zorder=2)

                    # Set labels and title
                    ax.set_xlabel(predictor, fontweight='bold', fontsize=11)
                    ax.set_ylabel(target, fontweight='bold', fontsize=11)
                    ax.set_title(f'{dataset_name}\nR²={r2:.3f}, RRMSE={rrmse:.1f}%',
                                 fontweight='bold', fontsize=10)

                    # Add grid
                    ax.grid(True, alpha=0.3, zorder=0)

                    # Add legend (smaller to fit more info)
                    ax.legend(loc='best', fontsize=8, framealpha=0.9)

                    # Add formula text
                    formula_text = formula.get_formula_text()
                    # Truncate if piecewise with many segments
                    if formula.formula_type.startswith('piecewise') and len(formula_text) > 200:
                        lines = formula_text.split('\n')
                        if len(lines) > 3:
                            formula_text = '\n'.join(lines[:2]) + f'\n... ({len(lines) - 2} more segments)'

                    ax.text(0.98, 0.02, formula_text,
                            transform=ax.transAxes,
                            ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                            fontsize=7, family='monospace')

                    # Add sample count
                    ax.text(0.02, 0.02, f'n={n}',
                            transform=ax.transAxes,
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)

                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=8, color='red')
                    ax.set_title(dataset_name, fontweight='bold')

            # Save or show using ModelPlotter utility
            if save_dir is not None:
                safe_name = formula_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                filename = save_path / f'{safe_name}_fit.png'
                ModelPlotter.save_and_show(fig, str(filename), verbose=False)
                print(f"  ✓ Saved: {filename.name}")
            else:
                ModelPlotter.save_and_show(fig, None, verbose=False)

        if save_dir is not None:
            print(f"\n✓ All formula fit plots saved to {save_path}")

    def plot_residuals(self, save_dir: str = None):
        """
        Create residual plots for each formula showing:
        - Residuals vs Predicted (check for patterns)
        - Residual histogram (check for normality)
        - Q-Q plot (check for normal distribution)
        """
        from pathlib import Path
        import scipy.stats as stats

        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving residual plots to: {save_path}")

        valid = self.results.dropna(subset=['gap_fuel_type'])

        print(f"\nCreating residual plots for {len(valid)} formulae...")

        for idx, row in valid.iterrows():
            formula_name = row['formula_name']
            target = row['target']
            source = row['reference'] if row['reference'] else 'default'

            formula = None
            for f in self.formulae:
                if f.name == formula_name:
                    formula = f
                    break

            if formula is None:
                continue

            # Get source-filtered datasets
            source_data = self.dataset_manager.get_datasets_for_source(source)

            title_parts = []
            if self.shiptype is not None:
                title_parts.append(f"{self.shiptype.name}")
            title_parts.append(f"Residual Analysis: {formula_name} → {target}")
            reference = formula.reference if formula.reference else "No reference"
            title_parts.append(f"[{reference}]")
            plot_title = " | ".join(title_parts)

            # Create figure: 3 datasets × 3 plot types = 9 subplots
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(plot_title, fontsize=14, fontweight='bold')

            matching_parameters = tuple(formula.predictors)

            if matching_parameters in self.matched_datasets:
                matched_dataset = self.matched_datasets[matching_parameters]
            else:
                matched_dataset = source_data.get_dataset('conventional')

            for row_idx, (dataset_name, dataset, color) in enumerate([
                ('Conventional', source_data.get_dataset('conventional'), 'steelblue'),
                ('Matched', matched_dataset, 'orange'),
                ('Alternative', source_data.get_dataset('alternative'), 'coral')
            ]):

                try:
                    # Get data and predictions
                    X = dataset[formula.predictors].values
                    y_true = dataset[formula.target].values

                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)

                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_true) |
                             np.isinf(X).any(axis=1) | np.isinf(y_true))
                    X_clean = X[mask]
                    y_true_clean = y_true[mask]

                    if len(X_clean) < 3:
                        continue

                    y_pred = formula.predict(X_clean)
                    residuals = y_true_clean - y_pred

                    # Plot 1: Residuals vs Predicted
                    ax1 = axes[row_idx, 0]
                    ax1.scatter(y_pred, residuals, alpha=0.5, color=color, s=30,
                                edgecolors='black', linewidth=0.5)
                    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
                    ax1.set_xlabel('Predicted', fontweight='bold')
                    ax1.set_ylabel('Residuals', fontweight='bold')
                    ax1.set_title(f'{dataset_name}\nResiduals vs Predicted', fontweight='bold')
                    ax1.grid(True, alpha=0.3)

                    # Add lowess smooth to check for patterns
                    from scipy.signal import savgol_filter
                    if len(y_pred) > 10:
                        sorted_idx = np.argsort(y_pred)
                        try:
                            smoothed = savgol_filter(residuals[sorted_idx],
                                                     min(11, len(residuals) // 2 * 2 - 1), 2)
                            ax1.plot(y_pred[sorted_idx], smoothed, 'r-', linewidth=2,
                                     label='Trend', alpha=0.7)
                            ax1.legend()
                        except:
                            pass

                    # Plot 2: Histogram of residuals
                    ax2 = axes[row_idx, 1]
                    ax2.hist(residuals, bins=20, color=color, alpha=0.7,
                             edgecolor='black', linewidth=0.5)
                    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Residuals', fontweight='bold')
                    ax2.set_ylabel('Frequency', fontweight='bold')
                    ax2.set_title(f'{dataset_name}\nResidual Distribution', fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')

                    # Add normal curve overlay
                    mu, std = residuals.mean(), residuals.std()
                    x = np.linspace(residuals.min(), residuals.max(), 100)
                    y = stats.norm.pdf(x, mu, std) * len(residuals) * (residuals.max() - residuals.min()) / 20
                    ax2.plot(x, y, 'r-', linewidth=2, label='Normal curve')
                    ax2.legend()

                    # Plot 3: Q-Q plot
                    ax3 = axes[row_idx, 2]
                    stats.probplot(residuals, dist="norm", plot=ax3)
                    ax3.get_lines()[0].set_markerfacecolor(color)
                    ax3.get_lines()[0].set_markeredgecolor('black')
                    ax3.get_lines()[0].set_markersize(6)
                    ax3.set_title(f'{dataset_name}\nQ-Q Plot', fontweight='bold')
                    ax3.grid(True, alpha=0.3)

                    # Add statistics text
                    rmse = np.sqrt(np.mean(residuals ** 2))
                    mae = np.mean(np.abs(residuals))
                    stats_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMean: {mu:.2f}\nStd: {std:.2f}'
                    ax3.text(0.05, 0.95, stats_text,
                             transform=ax3.transAxes,
                             va='top', ha='left',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             fontsize=8)

                except Exception as e:
                    print(f"    Error creating residual plots for {dataset_name}: {e}")

            plt.tight_layout()

            if save_dir is not None:
                safe_name = formula_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                filename = save_path / f'{safe_name}_residuals.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved: {filename.name}")
                plt.close()
            else:
                plt.show()

        if save_dir is not None:
            print(f"\n✓ All residual plots saved to {save_path}")

    def run_significance_tests(self,
                               comparison: str = 'fuel_type',
                               n_bootstrap: int = 5000,
                               confidence_level: float = 0.95,
                               ci_method: str = 'percentile',
                               random_seed: int|None = None,
                               debug: bool = False) -> dict:
        """
        Run bootstrap significance tests on all formula RRMSE gaps.

        Uses BootstrapSignificanceTester (via composition) to determine whether
        the observed performance differences between datasets are statistically
        meaningful rather than sampling noise.

        Step-by-step data flow:
        -----------------------
        1. self.formulae                      → list of formula objects
        2. self.dataset_manager               → retrieves filtered datasets per formula source
        3. self.matched_datasets              → provides matched conventional dataset per predictor group
        4. BootstrapSignificanceTester        → resamples both datasets, builds delta distribution
        5. CI of delta distribution           → determines significance

        Parameters
        ----------
        comparison : str
            Which gap to test:
            - 'fuel_type': matched_conventional vs alternative (isolates fuel effect)
            - 'distribution': conventional vs matched (isolates distribution shift)
            - 'total': conventional vs alternative (combined effect)
        n_bootstrap : int
            Number of bootstrap resamples
        confidence_level : float
            Confidence level for the interval (e.g. 0.95 for 95% CI)
        ci_method : str
            'percentile' or 'bca'
        random_seed : int, optional
            For reproducibility
        debug : bool
            Print step-by-step progress for each formula

        Returns
        -------
        list[dict]
            One result dict per formula. Each contains:
            - formula_name, target, predictors, reference
            - comparison_label
            - observed_rrmse_a, observed_rrmse_b, observed_delta
            - ci_lower, ci_upper, is_significant
            - n_a, n_b, n_bootstrap, confidence_level, ci_method
        """
        from knowledge_layer.literature_regressions.bootstrap_formula_tester import BootstrapSignificanceTester

        tester = BootstrapSignificanceTester(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            ci_method=ci_method,
            random_seed=random_seed
        )

        comparison_label_map = {
            'fuel_type': 'Matched vs Alternative',
            'distribution': 'Conventional vs Matched',
            'total': 'Conventional vs Alternative'
        }

        if comparison not in comparison_label_map:
            raise ValueError(f"comparison must be one of {list(comparison_label_map.keys())}")

        comparison_label = comparison_label_map[comparison]
        results = []

        for formula in self.formulae:
            if debug:
                print(f"\nTesting: {formula.name}")

            # ── Get source-filtered datasets ─────────────────────────────────
            # Data flow: formula.reference → dataset_manager → filtered datasets
            source = formula.reference if formula.reference else 'default'
            source_data = self.dataset_manager.get_datasets_for_source(source)

            # ── Get matched dataset for this formula's predictors ─────────────
            # Data flow: formula.predictors → matched_datasets dict → matched df
            matching_parameters = tuple(formula.predictors)
            if matching_parameters in self.matched_datasets:
                matched_dataset = self.matched_datasets[matching_parameters]
            else:
                matched_dataset = source_data.get_dataset('conventional')

            # ── Select dataset pair based on comparison type ──────────────────
            if comparison == 'fuel_type':
                dataset_a = matched_dataset
                dataset_b = source_data.get_dataset('alternative')
            elif comparison == 'distribution':
                dataset_a = source_data.get_dataset('conventional')
                dataset_b = matched_dataset
            else:  # total
                dataset_a = source_data.get_dataset('conventional')
                dataset_b = source_data.get_dataset('alternative')

            # ── Reference mean from conventional data ─────────────────────────
            # Using conventional mean as normaliser keeps RRMSE consistent
            # with evaluate_single_formula()
            conventional_data = source_data.get_dataset('conventional')
            target_data = conventional_data[formula.target].dropna()
            reference_mean = float(np.mean(target_data)) if len(target_data) > 0 else None

            # ── Run bootstrap test ────────────────────────────────────────────
            result = tester.test_formula_gap(
                formula=formula,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                reference_mean=reference_mean,
                comparison_label=comparison_label,
                debug=debug
            )
            results.append(result)

        return results

    def export_significance_results(self,
                                    significance_results: List[dict],
                                    output_path: str) -> None:
        """
        Export bootstrap significance results to Excel.

        Parameters
        ----------
        significance_results : list[dict]
            Output from run_significance_tests()
        output_path : str
            Path for .xlsx output file
        """
        from knowledge_layer.literature_regressions.bootstrap_formula_tester import BootstrapSignificanceTester

        # Tester instance only needed for its export method
        tester = BootstrapSignificanceTester()
        tester.export_to_excel(significance_results, output_path)