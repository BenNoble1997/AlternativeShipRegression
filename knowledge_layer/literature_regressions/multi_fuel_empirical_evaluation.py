# multi_dataset_literature_evaluator.py
"""
Multi-Dataset Literature Formula Evaluator
==========================================

Evaluates literature formulas across multiple alternative fuel datasets.
Uses composition to manage multiple single-dataset evaluators.

Key Features:
- Tests same formulas on multiple fuel types (LNG, Methanol, Ammonia, etc.)
- Aggregates results for cross-fuel comparison
- Identifies which formulas are robust vs. fuel-type-sensitive
- Creates comparative visualizations and LaTeX tables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from knowledge_layer.literature_regressions.Literature_formula_evaluation_class import LiteratureFormulaEvaluator
from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula
from knowledge_layer.literature_regressions.source_dataset_management_class import SourceDatasetManager


class MultiDatasetLiteratureEvaluator:
    """
    Evaluates literature formulas across multiple alternative fuel datasets.

    **Design**: Uses composition - creates separate evaluator for each fuel type,
    then aggregates and compares results.

    Comparison Framework:
    --------------------
    For each fuel type:
    1. Conventional full → Conventional matched → Alternative fuel
    2. Calculate fuel-type-specific gaps
    3. Compare gaps across fuel types to identify:
       - Universal formulas (work for all fuels)
       - Fuel-sensitive formulas (break down for specific fuels)
       - Consistently poor formulas (fail for all alternatives)

    """

    # ============================================================================
    # 1. CLASS-LEVEL CONSTANTS  (replace the existing three constants)
    # ============================================================================

    class MultiDatasetLiteratureEvaluator:

        # Hard defaults — used when no thresholds are supplied at construction time.
        _RRMSE_MAX_DEFAULT = 100.0
        _R2_MIN_DEFAULT = 0.0
        _R2_MAX_DEFAULT = 1.0

        # Which result columns to check for each metric type.
        # All six are checked against their respective bounds.
        _METRIC_COLUMNS = {
            'r2': ['conventional_r2', 'matched_r2', 'alternative_r2'],
            'rrmse': ['conventional_rrmse', 'matched_rrmse', 'alternative_rrmse'],
        }

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_datasets: Dict[str, pd.DataFrame],
                 formulae: List[LiteratureFormula] = None,
                 formulae_file: str = None,
                 shiptype: 'ShipType' = None,
                 sheet_name: str = None,
                 reference: str = "Literature",
                 target_dimensions: List[str] = None,
                 predictor_dimensions: List[str] = None,
                 verbose: bool = True,
                 # ── NEW: optional quality thresholds ──────────────────────
                 r2_min:    float | None = None,
                 r2_max:    float | None = None,
                 rrmse_max: float | None = None):
        """
        Initialize multi-dataset literature formula evaluator.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_datasets : dict[str, pd.DataFrame]
            Dictionary mapping fuel type names to datasets
            Example: {'LNG': lng_df, 'Methanol': methanol_df}
        formulae : List[LiteratureFormula], optional
            Explicit list of formulas to evaluate
        formulae_file : str, optional
            Path to Excel/CSV file with formulas
        shiptype : ShipType, optional
            Ship type for selecting Excel sheet
        sheet_name : str, optional
            Explicit sheet name (overrides shiptype)
        reference : str
            Default reference for formulas
        target_dimensions : List[str], optional
            Only evaluate formulas with these target dimensions
        predictor_dimensions : List[str], optional
            Only evaluate formulas using these predictor dimensions
        verbose : bool
            Print detailed progress
        display : bool
            Show initialization summary
        """

        # Store instance-level thresholds, falling back to class defaults
        self._r2_min = r2_min if r2_min is not None else 0.0
        self._r2_max = r2_max if r2_max is not None else 1.0
        self._rrmse_max = rrmse_max if rrmse_max is not None else 100.0

        self.conventional_data = conventional_data
        self.alternative_datasets = alternative_datasets
        self.formulae = formulae
        self.formulae_file = formulae_file
        self.shiptype = shiptype
        self.sheet_name = sheet_name
        self.reference = reference
        self.target_dimensions = target_dimensions
        self.predictor_dimensions = predictor_dimensions
        self.verbose = verbose

        # Validate inputs
        self._validate_inputs()

        # Storage for individual evaluators
        self.evaluators: Dict[str, LiteratureFormulaEvaluator] = {}

        # Storage for aggregated results
        self.comparison_results: Optional[pd.DataFrame] = None

        if self.verbose:
            self._print_initialization_summary()


    def _row_metrics_are_valid(self, row: pd.Series) -> bool:
        """
        Return True only if all absolute performance metrics are within
        the instance-level acceptable ranges.

        Checks the six named columns in _METRIC_COLUMNS:
          - conventional_r2, matched_r2, alternative_r2
          - conventional_rrmse, matched_rrmse, alternative_rrmse

        NaN values are treated as valid (missing data ≠ degenerate prediction).

        Gap columns are intentionally skipped — see __init__ docstring.
        """

        _METRIC_COLUMNS = {
            'r2': ['conventional_r2', 'matched_r2', 'alternative_r2'],
            'rrmse': ['conventional_rrmse', 'matched_rrmse', 'alternative_rrmse'],
        }


        for col in _METRIC_COLUMNS['r2']:
            val = row.get(col)
            if val is not None and not pd.isna(val):
                if not (self._r2_min <= val <= self._r2_max):
                    return False

        for col in _METRIC_COLUMNS['rrmse']:
            val = row.get(col)
            if val is not None and not pd.isna(val):
                if val > self._rrmse_max:
                    return False

        return True

    def _validate_inputs(self):
        """Validate initialization parameters."""

        if not isinstance(self.alternative_datasets, dict):
            raise TypeError(
                f"alternative_datasets must be a dict, got {type(self.alternative_datasets)}"
            )

        if len(self.alternative_datasets) == 0:
            raise ValueError("alternative_datasets cannot be empty")

        # Must provide either formulae or formulae_file
        if self.formulae is None and self.formulae_file is None:
            raise ValueError(
                "Must provide either 'formulae' (list of LiteratureFormula) "
                "or 'formulae_file' (path to Excel/CSV)"
            )

        # Warn if fuel type names are very long (plotting issues)
        long_names = [name for name in self.alternative_datasets.keys() if len(name) > 15]
        if long_names and self.verbose:
            print(f"\n⚠️  Warning: Long fuel type names detected: {long_names}")
            print(f"   Consider using abbreviations for better plot readability.")

        # Warn if too many datasets
        if len(self.alternative_datasets) > 6 and self.verbose:
            print(f"\n⚠️  Warning: {len(self.alternative_datasets)} datasets detected.")
            print(f"   Comparisons with >6 datasets may be difficult to visualize.")
            print(f"   Consider grouping similar fuel types or analyzing subsets.")

    def _print_initialization_summary(self):
        """Print initialization summary."""
        print(f"\n{'=' * 70}")
        print(f"Multi-Dataset Literature Formula Evaluator Initialized")
        print(f"{'=' * 70}")
        print(f"Conventional data: {len(self.conventional_data)} ships")
        print(f"Alternative datasets: {len(self.alternative_datasets)}")
        for fuel_type, df in self.alternative_datasets.items():
            print(f"  • {fuel_type}: {len(df)} ships")

        if self.formulae_file:
            print(f"Formula source: {self.formulae_file}")
        elif self.formulae:
            print(f"Formulas provided: {len(self.formulae)}")

        if self.target_dimensions:
            print(f"Target filter: {self.target_dimensions}")
        if self.predictor_dimensions:
            print(f"Predictor filter: {self.predictor_dimensions}")

    def _default_variable_groupings(self):

        variable_groupings = {
            'Hull dimensions': ['LBP (M)', 'LOA (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)'],
            'Capacity': ['DWT', 'GT', 'TEU','Np', 'Nv'],
            'Operational': ['Vs', 'MCR (KW)'],
        }

        return variable_groupings

    def evaluate_all_datasets(self,
                              display_results: bool = True,
                              metrics: str | List[str] = 'all') -> Dict[str, pd.DataFrame]:
        """
        Run formula evaluation on all alternative fuel datasets.

        Creates an evaluator for each dataset and runs evaluate_all_formulae().

        Parameters
        ----------
        display_results : bool
            Whether to display results for each dataset
        metrics : str or list
            Which metrics to display ('all', 'r2', 'rrmse', or list of metric names)

        Returns
        -------
        dict[str, pd.DataFrame]
            Results DataFrames for each fuel type
        """

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Evaluating Formulas on {len(self.alternative_datasets)} Alternative Fuel Datasets")
            print(f"{'=' * 70}\n")

        all_results = {}

        for fuel_type, alt_data in self.alternative_datasets.items():

            if self.verbose:
                print(f"\n{'─' * 70}")
                print(f"Analyzing: {fuel_type}")
                print(f"{'─' * 70}")

            # Create evaluator for this fuel type
            evaluator = self._create_evaluator(alt_data, fuel_type)

            # Run evaluation
            results_df = evaluator.evaluate_all_formulae(
                display_results=display_results and self.verbose,
                metrics=metrics
            )

            # Store evaluator and results
            self.evaluators[fuel_type] = evaluator
            all_results[fuel_type] = results_df

            variable_map = {
                'DWT': 'DWT (t)',
                'GT': 'GT (m³)',
                'LOA (M)': 'LOA (m)',
                'LBP (M)': 'LBP (m)',
                'BEAM (M)': 'Beam (m)',
                'DEPTH (M)': 'Depth (m)',
                'DRAFT (M)':'Draft (m)',
            }

            # Basic usage with values shown
            evaluator.plot_comparison(
                show_values=True,
                variable_name_map=variable_map,
                alternative_name=fuel_type,
            )

        # Create comparison DataFrame
        self._create_comparison_dataframe(all_results)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✓ All {len(self.alternative_datasets)} datasets analyzed")
            print(f"{'=' * 70}\n")

        return all_results

    def _create_evaluator(self,
                         alternative_data: pd.DataFrame,
                         fuel_type: str) -> LiteratureFormulaEvaluator:
        """
        Create LiteratureFormulaEvaluator for a specific fuel type.

        Uses SourceDatasetManager for efficient domain filtering.
        """

        # Create dataset manager for this fuel type
        # Note: We need to create matched datasets for this specific fuel type
        dataset_manager = SourceDatasetManager(
            conventional=self.conventional_data,
            matched=self.conventional_data.copy(),  # Placeholder, will be replaced
            alternative=alternative_data,
            verbose=False  # Suppress individual manager verbosity
        )

        # Create evaluator
        evaluator = LiteratureFormulaEvaluator(
            dataset_manager=dataset_manager,
            formulae=self.formulae,
            formulae_file=self.formulae_file,
            shiptype=self.shiptype,
            sheet_name=self.sheet_name,
            reference=self.reference,
            target_dimensions=self.target_dimensions,
            predictor_dimensions=self.predictor_dimensions,
            verbose=False,  # Suppress individual evaluator verbosity
            display=False  # Don't display initialization for each
        )



        return evaluator

    def _create_comparison_dataframe(self, all_results: Dict[str, pd.DataFrame]):
        """
        Aggregate per-fuel-type results and filter degenerate rows.

        Filtering criteria (configurable via __init__):
          - R²    must be in [r2_min, r2_max]   on conventional, matched AND alternative
          - RRMSE must be ≤ rrmse_max (%)        on conventional, matched AND alternative

        All six checks must pass for a row to be retained.
        """
        comparison_data = []

        for fuel_type, results_df in all_results.items():
            if results_df is None or len(results_df) == 0:
                continue
            df_copy = results_df.copy()
            df_copy['fuel_type'] = fuel_type
            comparison_data.append(df_copy)

        if not comparison_data:
            print("⚠️  Warning: No valid results from any dataset!")
            self.comparison_results = pd.DataFrame()
            return

        combined = pd.concat(comparison_data, ignore_index=True)

        # ── Filter out degenerate rows ──────────────────────────────────────
        valid_mask = combined.apply(self._row_metrics_are_valid, axis=1)
        n_removed  = (~valid_mask).sum()

        if n_removed > 0 and self.verbose:
            removed = combined.loc[~valid_mask, ['formula_name', 'fuel_type']].drop_duplicates()
            print(
                f"\n⚠️  Removed {n_removed} result row(s) outside quality thresholds "
                f"(R² ∉ [{self._r2_min}, {self._r2_max}]  or  "
                f"RRMSE > {self._rrmse_max}%):"
            )
            for _, r in removed.iterrows():
                print(f"   • {r['formula_name']}  [{r['fuel_type']}]")

        self.comparison_results = combined[valid_mask].copy()

        # ── Reorder columns for readability ────────────────────────────────
        priority_cols = [
            'fuel_type', 'formula_name', 'formula_type',
            'predictors', 'target', 'reference',
        ]
        all_cols = self.comparison_results.columns.tolist()
        ordered  = [c for c in priority_cols if c in all_cols]
        ordered += [c for c in all_cols if c not in ordered]
        self.comparison_results = self.comparison_results[ordered]

    def print_formula_robustness_summary(self,
                                        significance_threshold_rrmse: float = 10.0,
                                        significance_threshold_r2: float = 0.05):
        """
        Classify formulas by robustness across fuel types.

        Categories:
        - ROBUST: Works well for all fuel types (low gaps)
        - FUEL-SENSITIVE: Breaks down for some fuel types (high gaps)
        - CONSISTENTLY POOR: High gaps for all fuel types

        Parameters
        ----------
        significance_threshold_rrmse : float
            RRMSE gap threshold to consider significant (default: 10%)
        significance_threshold_r2 : float
            R² gap threshold to consider significant (default: 0.05)
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        print(f"\n{'=' * 100}")
        print("FORMULA ROBUSTNESS ANALYSIS")
        print(f"{'=' * 100}")

        # Analyze each formula
        formulas = self.comparison_results['formula_name'].unique()

        robust_formulas = []
        sensitive_formulas = []
        poor_formulas = []

        for formula_name in formulas:
            formula_data = self.comparison_results[
                self.comparison_results['formula_name'] == formula_name
            ]

            # Get gaps for all fuel types
            gaps_rrmse = formula_data['gap_fuel_type_rrmse'].values
            gaps_r2 = formula_data['gap_fuel_type'].values

            # Calculate statistics
            mean_gap_rrmse = np.mean(np.abs(gaps_rrmse))
            max_gap_rrmse = np.max(np.abs(gaps_rrmse))
            mean_gap_r2 = np.mean(np.abs(gaps_r2))

            # Count significant gaps
            n_significant_rrmse = np.sum(np.abs(gaps_rrmse) > significance_threshold_rrmse)
            n_significant_r2 = np.sum(np.abs(gaps_r2) > significance_threshold_r2)

            n_fuel_types = len(gaps_rrmse)

            # Classify
            if n_significant_rrmse == 0 and n_significant_r2 == 0:
                # No significant gaps for any fuel type
                robust_formulas.append({
                    'formula': formula_name,
                    'mean_gap_rrmse': mean_gap_rrmse,
                    'max_gap_rrmse': max_gap_rrmse,
                    'n_fuel_types': n_fuel_types
                })

            elif n_significant_rrmse == n_fuel_types or n_significant_r2 == n_fuel_types:
                # Significant gaps for ALL fuel types
                poor_formulas.append({
                    'formula': formula_name,
                    'mean_gap_rrmse': mean_gap_rrmse,
                    'max_gap_rrmse': max_gap_rrmse,
                    'n_fuel_types': n_fuel_types
                })

            else:
                # Significant gaps for SOME fuel types
                sensitive_formulas.append({
                    'formula': formula_name,
                    'mean_gap_rrmse': mean_gap_rrmse,
                    'max_gap_rrmse': max_gap_rrmse,
                    'n_significant': n_significant_rrmse,
                    'n_fuel_types': n_fuel_types
                })

        # Print results
        print(f"\n✅ ROBUST FORMULAS (work well for all fuel types): {len(robust_formulas)}")
        print(f"{'-' * 100}")
        if robust_formulas:
            for item in sorted(robust_formulas, key=lambda x: x['mean_gap_rrmse']):
                print(f"  {item['formula']:<50s} | "
                      f"Mean gap: {item['mean_gap_rrmse']:5.1f}% | "
                      f"Max gap: {item['max_gap_rrmse']:5.1f}%")
        else:
            print("  (None found)")

        print(f"\n⚠️  FUEL-SENSITIVE FORMULAS (break down for some fuels): {len(sensitive_formulas)}")
        print(f"{'-' * 100}")
        if sensitive_formulas:
            for item in sorted(sensitive_formulas, key=lambda x: x['n_significant'], reverse=True):
                print(f"  {item['formula']:<50s} | "
                      f"Fails for {item['n_significant']}/{item['n_fuel_types']} fuels | "
                      f"Mean gap: {item['mean_gap_rrmse']:5.1f}%")
        else:
            print("  (None found)")

        print(f"\n🚨 CONSISTENTLY POOR FORMULAS (fail for all fuels): {len(poor_formulas)}")
        print(f"{'-' * 100}")
        if poor_formulas:
            for item in sorted(poor_formulas, key=lambda x: x['mean_gap_rrmse'], reverse=True):
                print(f"  {item['formula']:<50s} | "
                      f"Mean gap: {item['mean_gap_rrmse']:5.1f}% | "
                      f"Max gap: {item['max_gap_rrmse']:5.1f}%")
        else:
            print("  (None found)")

        # Summary statistics
        print(f"\n{'=' * 100}")
        print("SUMMARY")
        print(f"{'=' * 100}")
        print(f"Total formulas analyzed: {len(formulas)}")
        print(f"  Robust:          {len(robust_formulas):3d} ({100*len(robust_formulas)/len(formulas):5.1f}%)")
        print(f"  Fuel-sensitive:  {len(sensitive_formulas):3d} ({100*len(sensitive_formulas)/len(formulas):5.1f}%)")
        print(f"  Consistently poor: {len(poor_formulas):3d} ({100*len(poor_formulas)/len(formulas):5.1f}%)")

    def plot_fuel_type_comparison(self,
                                  formula_name: str,
                                  save_path: Optional[str] = None):
        """
        Plot comparison of a specific formula across all fuel types.

        Creates bar chart showing R² and RRMSE for each fuel type.

        Parameters
        ----------
        formula_name : str
            Name of formula to plot
        save_path : str, optional
            Path to save figure
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        # Filter to this formula
        formula_data = self.comparison_results[
            self.comparison_results['formula_name'] == formula_name
        ]

        if len(formula_data) == 0:
            print(f"Formula '{formula_name}' not found in results.")
            return

        # Get target and predictors (should be same for all fuel types)
        target = formula_data['target'].iloc[0]
        predictors = formula_data['predictors'].iloc[0]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: R² comparison
        ax1 = axes[0]

        fuel_types = []
        conv_r2 = []
        matched_r2 = []
        alt_r2 = []

        for _, row in formula_data.iterrows():
            fuel_types.append(row['fuel_type'])
            conv_r2.append(row['conventional_r2'])
            matched_r2.append(row['matched_r2'])
            alt_r2.append(row['alternative_r2'])

        x = np.arange(len(fuel_types))
        width = 0.25

        ax1.bar(x - width, conv_r2, width, label='Conventional', alpha=0.8, color='steelblue')
        ax1.bar(x, matched_r2, width, label='Matched', alpha=0.8, color='orange')
        ax1.bar(x + width, alt_r2, width, label='Alternative', alpha=0.8, color='coral')

        ax1.set_xlabel('Fuel Type', fontweight='bold')
        ax1.set_ylabel('R²', fontweight='bold')
        ax1.set_title(f'R² Performance: {formula_name}', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(fuel_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: RRMSE comparison
        ax2 = axes[1]

        conv_rrmse = []
        matched_rrmse = []
        alt_rrmse = []

        for _, row in formula_data.iterrows():
            conv_rrmse.append(row['conventional_rrmse'])
            matched_rrmse.append(row['matched_rrmse'])
            alt_rrmse.append(row['alternative_rrmse'])

        ax2.bar(x - width, conv_rrmse, width, label='Conventional', alpha=0.8, color='steelblue')
        ax2.bar(x, matched_rrmse, width, label='Matched', alpha=0.8, color='orange')
        ax2.bar(x + width, alt_rrmse, width, label='Alternative', alpha=0.8, color='coral')

        ax2.set_xlabel('Fuel Type', fontweight='bold')
        ax2.set_ylabel('RRMSE (%)', fontweight='bold')
        ax2.set_title(f'RRMSE Performance: {formula_name}', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fuel_types, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add suptitle with metadata
        pred_str = ", ".join(predictors)
        fig.suptitle(f'{pred_str} → {target}', fontsize=12, y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.show()

    def plot_fuel_type_heatmap(self,
                               metric: str = 'gap_fuel_type_rrmse',
                               save_path: Optional[str] = None):
        """
        Plot heatmap showing metric values for all formulas across fuel types.

        Parameters
        ----------
        metric : str
            Metric to display ('gap_fuel_type_rrmse', 'alternative_rrmse', etc.)
        save_path : str, optional
            Path to save figure
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        # Pivot data for heatmap
        pivot_data = self.comparison_results.pivot_table(
            values=metric,
            index='formula_name',
            columns='fuel_type',
            aggfunc='mean'  # In case of duplicates
        )

        if pivot_data.empty:
            print(f"No data available for metric '{metric}'")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.4)))

        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels(pivot_data.index)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, rotation=270, labelpad=20)

        # Add values as text
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)

        ax.set_title(f'Formula Performance Heatmap: {metric}', fontweight='bold', pad=20)
        ax.set_xlabel('Fuel Type', fontweight='bold')
        ax.set_ylabel('Formula', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")

        plt.show()

    def export_comparison(self,
                         filename: str = 'multi_fuel_literature_comparison.xlsx'):
        """
        Export comparison results to Excel with multiple sheets.

        Parameters
        ----------
        filename : str
            Output filename
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Full comparison data
            self.comparison_results.to_excel(writer, sheet_name='All_Results', index=False)

            # Sheet 2: Summary by formula
            summary_by_formula = self._create_formula_summary()
            summary_by_formula.to_excel(writer, sheet_name='Summary_By_Formula', index=False)

            # Sheet 3: Summary by fuel type
            summary_by_fuel = self._create_fuel_type_summary()
            summary_by_fuel.to_excel(writer, sheet_name='Summary_By_Fuel_Type', index=False)

            # Sheet 4: Best formulas per target
            best_formulas = self._create_best_formulas_table()
            best_formulas.to_excel(writer, sheet_name='Best_Formulas', index=False)

        print(f"✓ Comparison exported to {filename}")

    def _create_formula_summary(self) -> pd.DataFrame:
        """Create summary statistics for each formula across all fuel types."""

        summary_data = []

        for formula_name in self.comparison_results['formula_name'].unique():
            formula_data = self.comparison_results[
                self.comparison_results['formula_name'] == formula_name
            ]

            summary_data.append({
                'formula_name': formula_name,
                'target': formula_data['target'].iloc[0],
                'predictors': formula_data['predictors'].iloc[0],
                '4:54 PM_fuel_types_tested': len(formula_data),
                'mean_alternative_r2': formula_data['alternative_r2'].mean(),
                'std_alternative_r2': formula_data['alternative_r2'].std(),
                'mean_alternative_rrmse': formula_data['alternative_rrmse'].mean(),
                'std_alternative_rrmse': formula_data['alternative_rrmse'].std(),
                'mean_gap_fuel_type_r2': formula_data['gap_fuel_type'].mean(),
                'mean_gap_fuel_type_rrmse': formula_data['gap_fuel_type_rrmse'].mean(),
                'max_gap_fuel_type_rrmse': formula_data['gap_fuel_type_rrmse'].abs().max()
                })

        return pd.DataFrame(summary_data)

    def _create_fuel_type_summary(self) -> pd.DataFrame:
        """Create summary statistics for each fuel type across all formulas."""

        summary_data = []

        for fuel_type in self.alternative_datasets.keys():
            fuel_data = self.comparison_results[
                self.comparison_results['fuel_type'] == fuel_type
                ]

            if len(fuel_data) == 0:
                continue

            summary_data.append({
                'fuel_type': fuel_type,
                'n_ships': len(self.alternative_datasets[fuel_type]),
                'n_formulas_tested': len(fuel_data),
                'mean_alternative_r2': fuel_data['alternative_r2'].mean(),
                'median_alternative_r2': fuel_data['alternative_r2'].median(),
                'mean_alternative_rrmse': fuel_data['alternative_rrmse'].mean(),
                'median_alternative_rrmse': fuel_data['alternative_rrmse'].median(),
                'mean_gap_fuel_type_r2': fuel_data['gap_fuel_type'].mean(),
                'mean_gap_fuel_type_rrmse': fuel_data['gap_fuel_type_rrmse'].mean(),
            })

        return pd.DataFrame(summary_data)

    def _create_best_formulas_table(self) -> pd.DataFrame:
        """Create table of best formula for each (target, fuel_type) combination."""

        best_data = []

        for target in self.comparison_results['target'].unique():
            for fuel_type in self.alternative_datasets.keys():
                target_fuel_data = self.comparison_results[
                    (self.comparison_results['target'] == target) &
                    (self.comparison_results['fuel_type'] == fuel_type)
                    ]

                if len(target_fuel_data) == 0:
                    continue

                # Best = lowest alternative_rrmse
                best_row = target_fuel_data.loc[target_fuel_data['alternative_rrmse'].idxmin()]

                best_data.append({
                    'target': target,
                    'fuel_type': fuel_type,
                    'best_formula': best_row['formula_name'],
                    'predictors': best_row['predictors'],
                    'alternative_r2': best_row['alternative_r2'],
                    'alternative_rrmse': best_row['alternative_rrmse'],
                    'gap_fuel_type_r2': best_row['gap_fuel_type'],
                    'gap_fuel_type_rrmse': best_row['gap_fuel_type_rrmse']
                })

        return pd.DataFrame(best_data)

    def get_robust_formulas(self,
                            threshold_rrmse: float = 10.0,
                            threshold_r2: float = 0.05) -> List[str]:
        """
        Get list of formulas that are robust across all fuel types.

        Parameters
        ----------
        threshold_rrmse : float
            Maximum acceptable RRMSE gap
        threshold_r2 : float
            Maximum acceptable R² gap

        Returns
        -------
        list[str]
            Formula names that pass robustness criteria
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            return []

        robust = []

        for formula_name in self.comparison_results['formula_name'].unique():
            formula_data = self.comparison_results[
                self.comparison_results['formula_name'] == formula_name
                ]

            # Check if ALL fuel types have acceptable gaps
            max_gap_rrmse = formula_data['gap_fuel_type_rrmse'].abs().max()
            max_gap_r2 = formula_data['gap_fuel_type'].abs().max()

            if max_gap_rrmse <= threshold_rrmse and max_gap_r2 <= threshold_r2:
                robust.append(formula_name)

        return robust

    @property
    def display(self):
        """Lazy initialization of display helper."""
        if not hasattr(self, '_display') or self._display is None:
            from knowledge_layer.literature_regressions.multi_empirical_display import MultiDatasetLiteratureDisplay
            self._display = MultiDatasetLiteratureDisplay(self)
        return self._display

    # Convenience methods
    def print_fuel_type_comparison(self,
                                   targets: list[str]|None=None,
                                   show_r2: bool = True,
                                   show_rrmse: bool = True,):
        """Print fuel type comparison table."""
        self.display.print_fuel_type_comparison(targets=targets,
                                                show_r2=show_r2,
                                                show_rrmse=show_rrmse)

    def print_formula_comparison(self, formula_name: str, **kwargs):
        """Print detailed comparison for specific formula."""
        self.display.print_formula_comparison(formula_name, **kwargs)

    def print_robustness_summary(self, **kwargs):
        """Print formula robustness classification."""
        self.display.print_robustness_classification(**kwargs)

    def print_fuel_type_ranking(self, **kwargs):
        """Print fuel type ranking."""
        self.display.print_fuel_type_ranking(**kwargs)

    def print_summary_statistics(self):
        """Print overall summary statistics."""
        self.display.print_summary_statistics()

    def generate_latex_tables(self,
                              ship_type: 'ShipType',
                              output_dir: str = '.',
                              targets: Optional[List[str]] = None,
                              prefix: str = None,
                              suffix:str=None,
                              show_r2_gaps: bool = True,
                              show_rrmse_gaps:bool=False,
                              show_reference: bool = True,
                              landscape: bool = False,
                              include_robustness: bool = False):
        """
        Generate and save LaTeX tables.

        Convenience method to create publication-ready tables.

        Parameters
        ----------
        ship_type : ShipType
            Ship type for tables
        output_dir : str
            Directory to save tables
        targets : List[str], optional
            Specific targets to include
        prefix : str, optional
            Filename prefix
        show_gaps : bool
            Include gap columns
        show_reference : bool
            Include reference citations
        landscape : bool
            Use landscape orientation (recommended for >3 fuel types)
        include_robustness : bool
            Generate robustness classification table

        Returns
        -------
        dict
            Paths to saved table files


        """
        from knowledge_layer.literature_regressions.empirical_LaTeX_generator import MultiDatasetLiteratureLatexGenerator

        generator = MultiDatasetLiteratureLatexGenerator(self, verbose=True)
        return generator.save_tables(
            ship_type=ship_type,
            output_dir=output_dir,
            targets=targets,
            prefix=prefix,
            suffix=suffix,
            show_r2_gaps=show_r2_gaps,
            show_rrmse_gaps=show_rrmse_gaps,
            show_reference=show_reference,
            landscape=landscape,
            include_robustness=include_robustness
        )

    def plot_two_dataset_comparison(self,
                                    fuel_type_1: str,
                                    fuel_type_2: str,
                                    metric: str = 'alternative_rrmse',
                                    targets: Optional[List[str]] = None,
                                    variable_name_map: Optional[Dict[str, str]] = None,
                                    show_values: bool = True,
                                    sort_by: str = 'dataset1',
                                    figsize: tuple[float, float] = (16, 10),
                                    save_path: Optional[str] = None):
        """
        Create side-by-side comparison of formula performance on two datasets.

        Similar to cascade plots - shows two datasets stacked vertically,
        making it easy to compare how the same formulas perform on different
        alternative fuel types.

        **Use Case:**
        Compare how literature formulas perform on LNG ships vs Methanol ships,
        showing whether formulas are fuel-type-specific or generalizable.

        Parameters
        ----------
        fuel_type_1 : str
            First fuel type (e.g., 'LNG')
        fuel_type_2 : str
            Second fuel type (e.g., 'Methanol')
        metric : str, default='alternative_rrmse'
            Metric to compare. Options:
            - 'alternative_rrmse': RRMSE on alternative fuel ships
            - 'alternative_r2': R² on alternative fuel ships
            - 'gap_fuel_type_rrmse': Fuel-type gap in RRMSE
            - 'gap_fuel_type': Fuel-type gap in R²
        targets : list[str], optional
            Specific targets to include. If None, shows all targets.
        variable_name_map : dict, optional
            Map variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        show_values : bool, default=True
            Show metric values on bars
        sort_by : str, default='dataset1'
            How to sort formulas:
            - 'dataset1': Sort by performance on first dataset
            - 'dataset2': Sort by performance on second dataset
            - 'formula_name': Alphabetically by formula name
        figsize : tuple, default=(16, 10)
            Figure size (width, height)
        save_path : str, optional
            Path to save figure

        Examples
        --------
        >>> # Compare LNG vs Methanol
        >>> evaluator.plot_two_dataset_comparison(
        ...     fuel_type_1='LNG',
        ...     fuel_type_2='Methanol',
        ...     metric='alternative_rrmse',
        ...     variable_name_map={'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        ... )

        >>> # Compare fuel-type gaps
        >>> evaluator.plot_two_dataset_comparison(
        ...     fuel_type_1='LNG',
        ...     fuel_type_2='Ammonia',
        ...     metric='gap_fuel_type_rrmse',
        ...     targets=['LOA (M)', 'Beam (M)', 'Depth (M)']
        ... )
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        # ====================================================================
        # 1. Validate Inputs
        # ====================================================================

        available_fuels = self.comparison_results['fuel_type'].unique()

        if fuel_type_1 not in available_fuels:
            raise ValueError(
                f"Fuel type '{fuel_type_1}' not found. "
                f"Available: {sorted(available_fuels)}"
            )

        if fuel_type_2 not in available_fuels:
            raise ValueError(
                f"Fuel type '{fuel_type_2}' not found. "
                f"Available: {sorted(available_fuels)}"
            )

        # ====================================================================
        # 2. Extract Data for Both Datasets
        # ====================================================================

        df1 = self.comparison_results[
            self.comparison_results['fuel_type'] == fuel_type_1
            ].copy()

        df2 = self.comparison_results[
            self.comparison_results['fuel_type'] == fuel_type_2
            ].copy()

        # Filter by targets if specified
        if targets is not None:
            df1 = df1[df1['target'].isin(targets)]
            df2 = df2[df2['target'].isin(targets)]

        if df1.empty or df2.empty:
            print("No data available after filtering.")
            return

        # Find common formulas
        common_formulas = set(df1['formula_name']) & set(df2['formula_name'])

        if not common_formulas:
            print(f"No common formulas found between {fuel_type_1} and {fuel_type_2}")
            return

        df1 = df1[df1['formula_name'].isin(common_formulas)]
        df2 = df2[df2['formula_name'].isin(common_formulas)]

        # ====================================================================
        # 3. Merge Data for Comparison
        # ====================================================================

        # Merge on formula_name and target
        merged = df1.merge(
            df2,
            on=['formula_name', 'target', 'predictors', 'formula_type', 'reference'],
            suffixes=('_1', '_2')
        )

        if merged.empty:
            print("No matching formula-target pairs found.")
            return

        # ====================================================================
        # 4. Sort Data
        # ====================================================================

        if sort_by == 'dataset1':
            merged = merged.sort_values(f'{metric}_1')
        elif sort_by == 'dataset2':
            merged = merged.sort_values(f'{metric}_2')
        elif sort_by == 'formula_name':
            merged = merged.sort_values('formula_name')

        # ====================================================================
        # 5. Create Figure with Two Subplots (Stacked Vertically)
        # ====================================================================

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Create formula labels
        labels = []
        for _, row in merged.iterrows():
            # Format predictors
            if variable_name_map:
                predictors_formatted = [
                    variable_name_map.get(p, p) for p in row['predictors']
                ]
                predictors_str = ' + '.join(predictors_formatted)
                target_str = variable_name_map.get(row['target'], row['target'])
            else:
                predictors_str = ' + '.join(row['predictors'])
                target_str = row['target']

            # Create label: "predictor → target (formula_type)"
            label = f"{predictors_str} → {target_str}\n({row['formula_type']})"
            labels.append(label)

        x = np.arange(len(labels))

        # ====================================================================
        # 6. Plot Dataset 1 (Top Panel)
        # ====================================================================

        values_1 = merged[f'{metric}_1'].values

        # Determine color scheme based on metric
        if 'gap' in metric:
            # For gaps, use diverging colors (green=good, red=bad)
            colors_1 = ['green' if v <= 0 else 'red' for v in values_1]
        else:
            # For performance metrics, use single color
            colors_1 = 'steelblue'

        bars1 = ax1.bar(x, values_1, color=colors_1, alpha=0.8,
                        edgecolor='black', linewidth=1)

        # Add value labels
        if show_values:
            for bar, val in zip(bars1, values_1):
                height = bar.get_height()

                # Format value based on metric
                if 'r2' in metric:
                    val_str = f'{val:.3f}'
                else:
                    val_str = f'{val:.1f}'

                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         val_str,
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8)

        # Format top panel
        metric_label = self._get_metric_display_name(metric)
        ax1.set_ylabel(metric_label, fontweight='bold', fontsize=11)
        ax1.set_title(f'{fuel_type_1} Dataset', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=1.5, alpha=0.5)

        # ====================================================================
        # 7. Plot Dataset 2 (Bottom Panel)
        # ====================================================================

        values_2 = merged[f'{metric}_2'].values

        if 'gap' in metric:
            colors_2 = ['green' if v <= 0 else 'red' for v in values_2]
        else:
            colors_2 = 'coral'

        bars2 = ax2.bar(x, values_2, color=colors_2, alpha=0.8,
                        edgecolor='black', linewidth=1)

        # Add value labels
        if show_values:
            for bar, val in zip(bars2, values_2):
                height = bar.get_height()

                if 'r2' in metric:
                    val_str = f'{val:.3f}'
                else:
                    val_str = f'{val:.1f}'

                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         val_str,
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8)

        # Format bottom panel
        ax2.set_ylabel(metric_label, fontweight='bold', fontsize=11)
        ax2.set_title(f'{fuel_type_2} Dataset', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Formula', fontweight='bold', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.5)

        # ====================================================================
        # 8. Add Overall Title
        # ====================================================================

        title_parts = []
        if self.shiptype:
            title_parts.append(self.shiptype.string)
        title_parts.append(f"Literature Formula Comparison: {fuel_type_1} vs {fuel_type_2}")

        fig.suptitle(' | '.join(title_parts),
                     fontsize=14, fontweight='bold', y=0.995)

        # ====================================================================
        # 9. Add Summary Statistics
        # ====================================================================

        # Calculate average performance
        avg_1 = values_1.mean()
        avg_2 = values_2.mean()

        # Add text box with summary
        summary_text = (
            f"Average {metric_label}:\n"
            f"  {fuel_type_1}: {avg_1:.2f}\n"
            f"  {fuel_type_2}: {avg_2:.2f}\n"
            f"  Difference: {avg_2 - avg_1:+.2f}"
        )

        ax1.text(0.02, 0.98, summary_text,
                 transform=ax1.transAxes,
                 fontsize=9, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightyellow', alpha=0.8),
                 family='monospace')

        plt.tight_layout()

        # ====================================================================
        # 10. Save and Show
        # ====================================================================

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Comparison plot saved to {save_path}")

        plt.show()

    def _get_metric_display_name(self, metric: str) -> str:
        """Get display name for metric."""
        display_names = {
            'alternative_rrmse': 'Alternative RRMSE (%)',
            'alternative_r2': 'Alternative R²',
            'gap_fuel_type_rrmse': 'Fuel-Type Gap RRMSE (%)',
            'gap_fuel_type': 'Fuel-Type Gap R²',
            'conventional_rrmse': 'Conventional RRMSE (%)',
            'conventional_r2': 'Conventional R²',
            'matched_rrmse': 'Matched RRMSE (%)',
            'matched_r2': 'Matched R²'
        }
        return display_names.get(metric, metric.replace('_', ' ').title())

    def plot_cascade_comparison(self,
                                targets: Optional[List[str]] = None,
                                variable_name_map: Optional[Dict[str, str]] = None,
                                show_values: bool = True,
                                show_summary_box: bool = False,
                                show_group_separators: bool = False,
                                rrmse_threshold: Optional[float] = None,
                                threshold_metric: Optional[str] = None,
                                variable_groupings: Optional[Dict[str, List[str]]] = None,
                                category_thresholds: Optional[Dict[str, float]] = None,
                                sort_by: str = 'predictor_group',
                                group_by_reference: bool = False,
                                predictor_order: Optional[List[str]] = None,
                                target_order: Optional[List[str]] = None,
                                label_content: str = 'formula_type',
                                figsize_per_plot: tuple[float, float] = (18, 5),
                                save_path: Optional[str] = None,
                                debug: bool = False,
                                suffix: str = '',
                                show_average_lines: bool = False,
                                # ── layout ──────────────────────────────────────
                                gridspec_top: float = 0.93,
                                hspace: float = 0.35,
                                # ── text ────────────────────────────────────────
                                base_font_size: int = 9,
                                tick_font_size: Optional[int] = None,
                                bar_label_font_size: Optional[int] = None,
                                # ── bars ────────────────────────────────────────
                                bar_width_scale: float = 0.75,
                                ):
        """
        Create cascade plots showing Conventional → Matched → Alternative progression.

        One subplot is created per fuel type, stacked vertically and sharing the
        x-axis.  Each subplot shows three grouped bars per formula:
        Conventional, Matched (Conv), and Alternative.

        Size and density are controlled through explicit layout, text, and bar
        parameters so the caller can tune the figure without touching internal
        logic.

        :param targets: Target variables to include. If None, all targets are shown.
        :type targets: list[str] or None
        :param variable_name_map: Map from internal variable names to display names,
            used for axis labels and bar labels.
        :type variable_name_map: dict[str, str] or None
        :param show_values: Annotate each bar with its numeric NRMSE value.
        :type show_values: bool
        :param show_summary_box: Show a text box with average NRMSE values or
            threshold explanation text in the top-left of each panel.
        :type show_summary_box: bool
        :param show_group_separators: Draw vertical lines between predictor or
            target groups.
        :type show_group_separators: bool
        :param rrmse_threshold: Fallback single global threshold line when
            ``category_thresholds`` is not used.
        :type rrmse_threshold: float or None
        :param threshold_metric: Aggregation used to auto-compute per-category
            reference lines from conventional NRMSE values:
            ``'median'``, ``'mean'``, or ``'best'``. Set to ``None`` to disable.
        :type threshold_metric: str or None
        :param variable_groupings: Category → variable-list mapping for
            per-category threshold lines.
        :type variable_groupings: dict[str, list[str]] or None
        :param category_thresholds: Explicit per-category threshold values,
            overriding auto-computed ones.
        :type category_thresholds: dict[str, float] or None
        :param sort_by: Sorting method for formulas on the x-axis.
            ``'predictor_group'``, ``'target_group'``, ``'conventional_rrmse'``,
            ``'alternative_rrmse'``, or ``'formula_name'``.
        :type sort_by: str
        :param group_by_reference: Sub-group formulas by reference source within
            each predictor or target group.
        :type group_by_reference: bool
        :param predictor_order: Custom ordering of predictor variables
            (only used with ``sort_by='predictor_group'``).
        :type predictor_order: list[str] or None
        :param target_order: Custom ordering of target variables
            (only used with ``sort_by='target_group'``).
        :type target_order: list[str] or None
        :param label_content: What to show on the second line of each x-axis tick:
            ``'formula_type'``, ``'reference'``, ``'both'``, or ``'none'``.
        :type label_content: str
        :param figsize_per_plot: Width and height in inches for a single fuel-type
            panel. Total figure height = ``figsize_per_plot[1] * n_fuel_types``.
        :type figsize_per_plot: tuple[float, float]
        :param save_path: File path to write the figure to (optional).
        :type save_path: str or None
        :param debug: Print intermediate values to stdout.
        :type debug: bool
        :param suffix: Additional string appended to the figure suptitle.
        :type suffix: str
        :param show_average_lines: Draw faint horizontal dotted lines for the
            average NRMSE of each dataset within each panel.
        :type show_average_lines: bool
        :param gridspec_top: Upper boundary of the subplot area as a fraction of
            figure height (0–1). Decrease to reserve more space for the suptitle
            and reduce the gap between title and first panel. Typical range
            0.88–0.95.
        :type gridspec_top: float
        :param hspace: Vertical spacing between stacked panels as a fraction of
            the average panel height. Decrease to tighten panels together.
        :type hspace: float
        :param base_font_size: Base font size in points. Axis labels, legends,
            and panel titles are all scaled relative to this value.
        :type base_font_size: int
        :param tick_font_size: Font size for x-axis tick labels. When ``None``,
            defaults to ``base_font_size - 1``. Set explicitly to make tick labels
            larger than the base without affecting other text.
        :type tick_font_size: int or None
        :param bar_label_font_size: Font size for numeric annotations on bars.
            When ``None``, defaults to ``base_font_size - 2``. Set explicitly to
            make bar labels larger than the base without affecting other text.
        :type bar_label_font_size: int or None
        :param bar_width_scale: Fraction of the available inter-group space used
            by the three-bar cluster (0–1). Reduce below 0.75 to add whitespace
            between formula groups.
        :type bar_width_scale: float
        """

        if debug:
            print("\n" + "=" * 80)
            print("DEBUG: Starting plot_cascade_comparison")
            print("=" * 80)

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        # ── derive font sizes from base, with optional overrides ────────────
        fs = {
            'bar_label': bar_label_font_size if bar_label_font_size is not None else base_font_size - 2,
            'tick': tick_font_size if tick_font_size is not None else base_font_size - 1,
            'axis_label': base_font_size,
            'legend': base_font_size - 1,
            'subtitle': base_font_size + 1,
            'suptitle': base_font_size + 3,
            'textbox': base_font_size - 2,
        }


        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run evaluate_all_datasets() first.")
            return

        # ====================================================================
        # 1. Get Fuel Types from Alternative Datasets Dictionary
        # ====================================================================

        fuel_types = list(self.alternative_datasets.keys())
        n_fuel_types = len(fuel_types)

        if debug:
            print(f"\nDEBUG [Section 1]: Fuel types detected")
            print(f"  - Number of fuel types: {n_fuel_types}")
            print(f"  - Fuel types: {fuel_types}")

        if self.verbose:
            print(f"\nCreating cascade plots for {n_fuel_types} fuel types: {fuel_types}")

        # ====================================================================
        # 2. Find COMMON Formulas Across All Fuel Types
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 2]: Finding common formulas")

        fuel_type_formulas = {}

        for fuel_type in fuel_types:
            df_fuel = self.comparison_results[
                self.comparison_results['fuel_type'] == fuel_type
                ].copy()

            if targets is not None:
                df_fuel = df_fuel[df_fuel['target'].isin(targets)]

            fuel_type_formulas[fuel_type] = set(df_fuel['formula_name'].unique())

            if debug:
                print(f"  - {fuel_type}: {len(fuel_type_formulas[fuel_type])} formulas")

        common_formulas = set.intersection(*fuel_type_formulas.values())

        if debug:
            print(f"  - Common formulas across all fuel types: {len(common_formulas)}")

        if not common_formulas:
            print("⚠️  No common formulas found across all fuel types!")
            return

        # ====================================================================
        # 2.5. Setup Category-Based Thresholds
        # ====================================================================
        if variable_groupings is None:
            variable_groupings = self._default_variable_groupings()

        threshold_text = None
        if category_thresholds is None and threshold_metric is not None:
            category_thresholds = self.compute_literature_reference_metrice(
                comparison_results=self.comparison_results,
                variable_groupings=variable_groupings,
                metric_type=threshold_metric,
                dataset='conventional'
            )

            if debug:
                print(f"\nDEBUG [Section 2.5]: Computed category thresholds")
                print(f"  - Metric type: {threshold_metric}")
                print(f"  - Thresholds: {category_thresholds}")

        # Create threshold explanation text
        if category_thresholds:
            threshold_text = self.create_threshold_summary_text(
                thresholds=category_thresholds,
                metric_type=threshold_metric or 'custom',
                dataset='conventional'
            )

        # ====================================================================
        # 3. Prepare Data - Use First Fuel Type for Sorting
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 3]: Preparing data for sorting")

        first_fuel = fuel_types[0]
        df_first = self.comparison_results[
            (self.comparison_results['fuel_type'] == first_fuel) &
            (self.comparison_results['formula_name'].isin(common_formulas))
            ].copy()

        if targets is not None:
            df_first = df_first[df_first['target'].isin(targets)]

        if debug:
            print(f"  - First fuel type: {first_fuel}")
            print(f"  - Rows in df_first: {len(df_first)}")

        if df_first.empty:
            print("No data available after filtering.")
            return

        # Remove duplicates
        df_first = df_first.drop_duplicates(subset=['formula_name'], keep='first')

        # ====================================================================
        # 4. ENHANCED SORTING - Group by Predictors
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 4]: Sorting formulas by '{sort_by}'")

        if sort_by == 'predictor_group':
            df_first = self._sort_by_predictor_groups(
                df_first,
                predictor_order=predictor_order,
                group_by_reference=group_by_reference,
                debug=debug
            )
        elif sort_by == 'target_group':  # NEW
            df_first = self._sort_by_target_groups(
                df_first,
                target_order=target_order,
                group_by_reference=group_by_reference,
                debug=debug
            )
        elif sort_by in ['conventional_rrmse', 'matched_rrmse', 'alternative_rrmse',
                         'gap_fuel_type_rrmse']:
            df_first = df_first.sort_values(sort_by)
        elif sort_by == 'formula_name':
            df_first = df_first.sort_values('formula_name')
        else:
            print(f"Warning: Unknown sort_by '{sort_by}', using 'predictor_group'")
            df_first = self._sort_by_predictor_groups(
                df_first,
                predictor_order=predictor_order,
                group_by_reference=group_by_reference,
                debug=debug
            )

        # ====================================================================
        # 5. Create Formula Labels and Track Group Boundaries
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 5]: Creating formula labels")
            print(f"  - label_content: {label_content}")

        valid_label_options = ['formula_type', 'reference', 'both', 'none']
        if label_content not in valid_label_options:
            print(f"⚠️  Warning: Invalid label_content '{label_content}'. "
                  f"Using 'formula_type'. Valid options: {valid_label_options}")
            label_content = 'formula_type'

        labels = []
        group_boundaries = []  # Track where predictor groups change
        last_predictor_group = None
        last_target = None
        last_reference = None

        for idx, (_, row) in enumerate(df_first.iterrows()):
            # Create label
            if variable_name_map:
                predictors_formatted = [
                    variable_name_map.get(p, p) for p in row['predictors']
                ]
                predictors_str = ' + '.join(predictors_formatted)
                target_str = variable_name_map.get(row['target'], row['target'])
            else:
                predictors_str = ' + '.join(row['predictors'])
                target_str = row['target']

            label = f"{predictors_str} → {target_str}"

            if label_content == 'formula_type':
                # Show formula type
                label += f"\n({row['formula_type']})"

            elif label_content == 'reference':
                # Show reference
                reference = row.get('reference', 'No reference')
                # Truncate long references for readability
                if len(reference) > 25:
                    reference = reference[:22] + '...'
                label += f"\n{[reference]}"

            elif label_content == 'both':
                # Show both formula type and reference
                reference = row.get('reference', 'No reference')
                # Truncate long references
                if len(reference) > 20:
                    reference = reference[:17] + '...'
                label += f"\n({row['formula_type']} | {reference})"

            elif label_content == 'none':
                # Just predictor → target, no extra info
                pass

            labels.append(label)


            # Track group boundaries (only if we'll show them)
            if show_group_separators:
                current_predictor_group = tuple(sorted(row['predictors']))
                current_target = row['target']
                current_reference = row.get('reference', 'No reference')

                if sort_by == 'predictor_group':
                    group_key = current_predictor_group
                    last_group_key = last_predictor_group
                elif sort_by == 'target_group':
                    group_key = current_target
                    last_group_key = last_target
                else:
                    # For other sort methods, don't show group separators
                    group_key = None
                    last_group_key = None

                if group_key is not None and last_group_key is not None:
                    if group_by_reference:
                        # Boundary when EITHER group OR reference changes
                        if (group_key != last_group_key or
                                current_reference != last_reference):
                            group_boundaries.append(idx)
                        last_reference = current_reference
                    else:
                        # Boundary only when group changes
                        if group_key != last_group_key:
                            group_boundaries.append(idx)

                last_predictor_group = current_predictor_group
                last_target = current_target

        formula_order = df_first['formula_name'].tolist()
        n_formulas = len(formula_order)

        if debug:
            print(f"  - Number of formulas to plot: {n_formulas}")
            print(f"  - Show group separators: {show_group_separators}")
            if show_group_separators:
                print(f"  - Group boundaries at indices: {group_boundaries}")
            print(f"  - Example labels (first 3):")
            for i, label in enumerate(labels[:3]):
                print(f"      {i + 1}. {label.replace(chr(10), ' | ')}")

                # ====================================================================
        # 6. Create Figure with Stacked Subplots
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 6]: Creating figure")
            print(f"  - Figure size: {figsize_per_plot[0]} x {figsize_per_plot[1] * n_fuel_types}")

            # Define color palette for alternative fuel types
            # Use a diverse palette that works well for colorblind users
        alt_fuel_colors = [
            '#E74C3C',  # Red
            '#1ABC9C',  # Teal
            '#2ECC71',  # Green
            '#3498DB',  # Blue
            '#F39C12',  # Orange
            '#9B59B6',  # Purple
            '#E67E22',  # Dark Orange
            '#34495E',  # Dark Gray
            '#16A085',  # Dark Teal
            '#C0392B',  # Dark Red
        ]

        # Assign colors to each fuel type
        fuel_type_colors = {}
        for idx, fuel_type in enumerate(fuel_types):
            color_idx = idx % len(alt_fuel_colors)
            fuel_type_colors[fuel_type] = alt_fuel_colors[color_idx]

        if debug:
            print(f"  - Fuel type colors:")
            for fuel, color in fuel_type_colors.items():
                print(f"    {fuel}: {color}")

        fig, axes = plt.subplots(
            n_fuel_types, 1,
            figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_fuel_types),
            sharex=True
        )

        if n_fuel_types == 1:
            axes = [axes]

        # Push panels up toward the suptitle, reducing the gap below it
        fig.subplots_adjust(top=gridspec_top, hspace=hspace)

        # Define x-axis positions
        x = np.arange(n_formulas)
        bar_w = bar_width_scale / 3

        if debug:
            print(f"  - x positions: shape={x.shape}, min={x.min()}, max={x.max()}")
            print(f"  - Bar width: {bar_w}")

        # ====================================================================
        # 7. Plot Each Fuel Type
        # ====================================================================

        if debug:
            print(f"\nDEBUG [Section 7]: Plotting each fuel type")

        for fuel_idx, (fuel_type, ax) in enumerate(zip(fuel_types, axes)):

            if debug:
                print(f"\n  --- Fuel type {fuel_idx + 1}/{n_fuel_types}: {fuel_type} ---")

            # Get data for this fuel type
            df_fuel = self.comparison_results[
                (self.comparison_results['fuel_type'] == fuel_type) &
                (self.comparison_results['formula_name'].isin(common_formulas))
                ].copy()

            if targets is not None:
                df_fuel = df_fuel[df_fuel['target'].isin(targets)]

            # Remove duplicates
            df_fuel = df_fuel.drop_duplicates(subset=['formula_name'], keep='first')

            # Reorder to match first fuel type
            df_fuel = df_fuel.set_index('formula_name').loc[formula_order].reset_index()

            if len(df_fuel) != n_formulas:
                print(f"⚠️  Warning: {fuel_type} has {len(df_fuel)} formulas, expected {n_formulas}")
                continue

            # Extract cascade data
            conv_rrmse = df_fuel['conventional_rrmse'].values
            matched_rrmse = df_fuel['matched_rrmse'].values
            alt_rrmse = df_fuel['alternative_rrmse'].values

            alt_color = fuel_type_colors[fuel_type]

            # Create grouped bars
            bars_conv = ax.bar(
                x - bar_w, conv_rrmse, bar_w,
                label='Conventional',
                color='steelblue', alpha=0.5,
                edgecolor='black', linewidth=0.8
            )

            bars_matched = ax.bar(
                x, matched_rrmse, bar_w,
                label='Matched (Conv)',
                color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.8
            )

            bars_alt = ax.bar(
                x + bar_w, alt_rrmse, bar_w,
                label=f'Alternative ({fuel_type})',
                color=alt_color, alpha=0.9,
                edgecolor='black', linewidth=0.8
            )

            # Add value labels
            if show_values:
                for bars, values in [
                    (bars_conv, conv_rrmse),
                    (bars_matched, matched_rrmse),
                    (bars_alt, alt_rrmse)
                ]:
                    for bar, val in zip(bars, values):
                        if np.isfinite(val):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.,
                                bar.get_height(),
                                f'{val:.1f}',
                                ha='center', va='bottom',
                                fontsize=fs['bar_label']   # ← was hardcoded 7
                            )

            # Add group separator lines (only if requested)
            if show_group_separators:
                for boundary_idx in group_boundaries:
                    ax.axvline(
                        x=boundary_idx - 0.5,
                        color='gray',
                        linestyle='-',
                        linewidth=2,
                        alpha=0.6,
                        zorder=5
                    )

            # Add threshold line
            if category_thresholds:
                formula_targets = df_fuel['target'].values
                self.add_category_threshold_lines(
                    ax=ax,
                    formula_targets=formula_targets,
                    formula_positions=x,
                    thresholds=category_thresholds,
                    variable_groupings=variable_groupings
                )
            elif rrmse_threshold is not None:
                # Fallback to single threshold for backward compatibility
                ax.axhline(
                    y=rrmse_threshold,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7,
                    label=f'Threshold ({rrmse_threshold}%)',
                    zorder=10
                )

            # Format panel
            ax.set_ylabel('NRMSE (%)', fontweight='bold', fontsize=fs['axis_label'])
            ax.set_title(
                f'{fuel_type} Dataset - Performance shift analysis',
                fontweight='bold', fontsize=fs['subtitle']
            )
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper left', fontsize=fs['legend'], ncol=4)

            # Set y-axis limits
            valid_values = np.concatenate([conv_rrmse, matched_rrmse, alt_rrmse])
            valid_values = valid_values[np.isfinite(valid_values)]

            if len(valid_values) > 0:
                y_min = 0
                y_max = max(valid_values.max() * 1.15,
                            rrmse_threshold * 1.2 if rrmse_threshold else valid_values.max() * 1.15)
                ax.set_ylim(y_min, y_max)


            # Add average lines
            if show_average_lines:
                avg_conv = np.nanmean(conv_rrmse)
                avg_matched = np.nanmean(matched_rrmse)
                avg_alt = np.nanmean(alt_rrmse)

                if np.isfinite(avg_conv):
                    ax.axhline(y=avg_conv, color='steelblue', linestyle=':',
                               linewidth=1, alpha=0.3)
                if np.isfinite(avg_matched):
                    ax.axhline(y=avg_matched, color='steelblue', linestyle=':',
                               linewidth=1, alpha=0.4)
                if np.isfinite(avg_alt):
                    ax.axhline(y=avg_alt, color=alt_color, linestyle=':',
                               linewidth=1, alpha=0.5)

            # Add summary box
            if show_summary_box:

                if threshold_text:
                    ax.text(
                        0.02, 0.98,
                        threshold_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        va='top',
                        ha='left',
                        bbox=dict(
                            boxstyle='round,pad=0.5',
                            facecolor='lightyellow',
                            alpha=0.8
                        ),
                        family='monospace',
                        zorder=15
                    )
                else:

                    summary_text = (
                        f"Avg NRMSE:\n"
                        f"Conv: {avg_conv:.1f}%\n"
                        f"Match: {avg_matched:.1f}%\n"
                        f"Alt: {avg_alt:.1f}%\n"
                        f"Gap: {avg_alt - avg_matched:+.1f}%"
                    )

                    ax.text(
                        0.98, 0.98, summary_text,
                        transform=ax.transAxes,
                        fontsize=8, va='top', ha='right',
                        bbox=dict(
                            boxstyle='round,pad=0.4',
                            facecolor='lightyellow',
                            alpha=0.8
                        ),
                        family='monospace'
                    )

        # ====================================================================
        # 8. Format Bottom Panel (X-axis Labels)
        # ====================================================================

        axes[-1].set_xlabel('Formula', fontweight='bold', fontsize=fs['axis_label'])
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha='right', fontsize=fs['tick'])

        # ====================================================================
        # 9. Add Overall Title
        # ====================================================================

        title_parts = []
        if self.shiptype:

            ship_str = f'Experiment 1 - {self.shiptype.string}'

        else:
            ship_str = 'Experiment 1 results'

        title_parts.append(ship_str)

        fuel_list_str = ', '.join(fuel_types)
        title_parts.append(f"Performance Shift Comparison: {fuel_list_str}")

        if suffix:
            title_parts.append(suffix)

        fig.suptitle(
            ' | '.join(title_parts),
            fontsize=fs['suptitle'],
            fontweight='bold',
            y=1.0  # sits at the very top; gridspec_top controls the gap below it
        )

        plt.tight_layout()

        # ====================================================================
        # 10. Save and Show
        # ====================================================================

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose or debug:
                print(f"✅ Cascade comparison plot saved to {save_path}")

        plt.show()

    def _sort_by_predictor_groups(self,
                                  df: pd.DataFrame,
                                  predictor_order: Optional[List[str]] = None,
                                  group_by_reference: bool = False,
                                  debug: bool = False) -> pd.DataFrame:
        """
        Sort formulas by predictor groups with optional reference sub-grouping.

        Sorting logic:
        1. Group by unique predictor combinations
        2. Order predictor groups by:
           a. Number of predictors (single-predictor first, then 2-predictor, etc.)
           b. Custom predictor_order if provided (within each predictor count)
           c. Alphabetical if not provided
        3. Within each predictor group:
           a. Sub-group by reference if group_by_reference=True
           b. Sort by conventional_rrmse

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with formula data
        predictor_order : list[str], optional
            Custom order for predictor variables
        group_by_reference : bool
            Whether to sub-group by reference within predictor groups
        debug : bool
            Print debug information

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame
        """

        if debug:
            print(f"\n  DEBUG [_sort_by_predictor_groups]:")
            print(f"    - Input rows: {len(df)}")
            print(f"    - group_by_reference: {group_by_reference}")
            print(f"    - predictor_order: {predictor_order}")

        # Create predictor group key
        df['_predictor_tuple'] = df['predictors'].apply(lambda x: tuple(sorted(x)))
        df['_n_predictors'] = df['predictors'].apply(len)

        # Create predictor group sort key
        if predictor_order is not None:
            # Custom ordering - create sort key based on predictor_order
            predictor_rank = {pred: idx for idx, pred in enumerate(predictor_order)}

            def get_sort_key(pred_tuple):
                """
                Sort key: (n_predictors, min_rank, avg_rank, pred_tuple)

                Logic:
                1. Number of predictors (1, then 2, then 3, etc.)
                2. Minimum rank among predictors (DWT=0, TEU=1, so DWT+TEU gets min=0)
                3. Average rank (for tie-breaking when min is same)
                4. Alphabetical tuple (final tie-breaker)

                Example with predictor_order=['DWT', 'TEU', 'Vs']:
                Single-predictor (n=1):
                - ['DWT'] -> (1, 0, 0.0, ('DWT',))
                - ['TEU'] -> (1, 1, 1.0, ('TEU',))
                - ['Vs'] -> (1, 2, 2.0, ('Vs',))

                Two-predictor (n=2):
                - ['DWT', 'TEU'] -> (2, 0, 0.5, ('DWT', 'TEU'))
                - ['DWT', 'Vs'] -> (2, 0, 1.0, ('DWT', 'Vs'))
                - ['TEU', 'Vs'] -> (2, 1, 1.5, ('TEU', 'Vs'))

                Three-predictor (n=3):
                - ['DWT', 'TEU', 'Vs'] -> (3, 0, 1.0, ('DWT', 'TEU', 'Vs'))
                """
                n_preds = len(pred_tuple)

                # Get ranks for all predictors
                ranks = [predictor_rank.get(p, 999) for p in pred_tuple]

                # Minimum rank (determines which "family" this belongs to)
                min_rank = min(ranks)

                # Average rank (for ordering within same family)
                avg_rank = np.mean(ranks)

                return (n_preds, min_rank, avg_rank, pred_tuple)

            df['_sort_key'] = df['_predictor_tuple'].apply(get_sort_key)

        else:
            # Alphabetical ordering
            df['_sort_key'] = df.apply(
                lambda row: (row['_n_predictors'], row['_predictor_tuple']),
                axis=1
            )

        # Sort columns
        if group_by_reference:
            # Sort by: predictor group → reference → conventional_rrmse
            sort_cols = ['_sort_key', 'reference', 'conventional_rrmse']
        else:
            # Sort by: predictor group → conventional_rrmse
            sort_cols = ['_sort_key', 'conventional_rrmse']

        df_sorted = df.sort_values(sort_cols)

        if debug:
            print(f"    - Output rows: {len(df_sorted)}")
            print(f"    - Unique predictor groups: {df_sorted['_predictor_tuple'].nunique()}")
            if group_by_reference:
                print(f"    - Unique references: {df_sorted['reference'].nunique()}")

            # Show predictor groups in order
            print(f"\n    Predictor groups in order:")
            seen_groups = set()
            for idx, (_, row) in enumerate(df_sorted.iterrows()):
                pred_tuple = row['_predictor_tuple']
                if pred_tuple not in seen_groups:
                    seen_groups.add(pred_tuple)
                    pred_list = list(pred_tuple)
                    n_formulas = len(df_sorted[df_sorted['_predictor_tuple'] == pred_tuple])
                    print(f"      {idx + 1}. {pred_list}: {n_formulas} formulas")
                    if idx >= 9:  # Show first 10 groups
                        remaining = df_sorted['_predictor_tuple'].nunique() - len(seen_groups)
                        if remaining > 0:
                            print(f"      ... and {remaining} more groups")
                        break

        # Clean up temporary columns
        df_sorted = df_sorted.drop(columns=['_predictor_tuple', '_n_predictors', '_sort_key'])

        return df_sorted

    def _sort_by_target_groups(self,
                               df: pd.DataFrame,
                               target_order: Optional[List[str]] = None,
                               group_by_reference: bool = False,
                               debug: bool = False) -> pd.DataFrame:
        """
        Sort formulas by target groups with optional reference sub-grouping.

        Sorting logic:
        1. Group by target dimension
        2. Order targets by:
           a. Custom target_order if provided
           b. Alphabetical if not provided
        3. Within each target group:
           a. Sub-group by reference if group_by_reference=True
           b. Sort by conventional_rrmse

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with formula data
        target_order : list[str], optional
            Custom order for target variables
            Example: ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)']
        group_by_reference : bool
            Whether to sub-group by reference within target groups
        debug : bool
            Print debug information

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame
        """

        if debug:
            print(f"\n  DEBUG [_sort_by_target_groups]:")
            print(f"    - Input rows: {len(df)}")
            print(f"    - group_by_reference: {group_by_reference}")
            print(f"    - target_order: {target_order}")

        # Create target sort key
        if target_order is not None:
            # Custom ordering - create rank mapping
            target_rank = {target: idx for idx, target in enumerate(target_order)}

            def get_sort_key(target):
                """
                Sort key: (rank, target)

                Targets in target_order come first in specified order,
                then any other targets alphabetically.
                """
                rank = target_rank.get(target, 999)
                return (rank, target)

            df['_sort_key'] = df['target'].apply(get_sort_key)

        else:
            # Alphabetical ordering
            df['_sort_key'] = df['target']

        # Sort columns
        if group_by_reference:
            # Sort by: target â†' reference â†' conventional_rrmse
            sort_cols = ['_sort_key', 'reference', 'conventional_rrmse']
        else:
            # Sort by: target â†' conventional_rrmse
            sort_cols = ['_sort_key', 'conventional_rrmse']

        df_sorted = df.sort_values(sort_cols)

        if debug:
            print(f"    - Output rows: {len(df_sorted)}")
            print(f"    - Unique targets: {df_sorted['target'].nunique()}")
            if group_by_reference:
                print(f"    - Unique references: {df_sorted['reference'].nunique()}")

            # Show target groups in order
            print(f"\n    Target groups in order:")
            seen_targets = set()
            for idx, (_, row) in enumerate(df_sorted.iterrows()):
                target = row['target']
                if target not in seen_targets:
                    seen_targets.add(target)
                    n_formulas = len(df_sorted[df_sorted['target'] == target])
                    print(f"      {idx + 1}. {target}: {n_formulas} formulas")
                    if idx >= 9:  # Show first 10 groups
                        remaining = df_sorted['target'].nunique() - len(seen_targets)
                        if remaining > 0:
                            print(f"      ... and {remaining} more targets")
                        break

        # Clean up temporary column
        df_sorted = df_sorted.drop(columns=['_sort_key'])

        return df_sorted

    def plot_predictor_target_fits(self,
                                   predictor: str,
                                   target: str,
                                   fuel_type: Optional[str] = None,
                                   layout: str = 'horizontal',
                                   separate_plots: bool = False,
                                   n_points: int = 100,
                                   show_data_points: bool = True,
                                   show_domain_bounds: bool = True,
                                   show_percentile_bounds: bool = True,
                                   figsize_per_plot: tuple[float, float] = (6, 6),
                                   title_prefix: Optional[str] = None,
                                   save_dir: Optional[str] = None,
                                   debug: bool = False):
        """
        Plot empirical formula fits for a specific predictor-target combination.

        Creates plots showing the fitted relationship between predictor and target
        across the three datasets (Conventional, Matched, Alternative).

        **IMPORTANT**: This plots the FORMULA's prediction, not a fitted line.
        The formula is from the literature (e.g., Watson, Schneekluth, etc.).

        Parameters
        ----------
        predictor : str
            Predictor variable name (e.g., 'DWT', 'TEU')
        target : str
            Target variable name (e.g., 'LBP (M)', 'BEAM (M)')
        fuel_type : str, optional
            Which fuel type to plot. If None, uses first fuel type in datasets.
        layout : str, default='horizontal'
            Layout of the three subplots: 'horizontal' or 'vertical'
        separate_plots : bool, default=False
            If True, creates separate figures for each dataset
        n_points : int, default=100
            Number of points for plotting the formula curve
        show_data_points : bool, default=True
            Show actual ship data as scatter points
        show_domain_bounds : bool, default=True
            Show vertical lines for domain constraints
        show_percentile_bounds : bool, default=True
            Show dashed vertical lines for percentile bounds
        figsize_per_plot : tuple, default=(6, 6)
            Size of each subplot (width, height)
        title_prefix : str, optional
            Prefix for figure title
        save_dir : str, optional
            Directory to save plots
        debug : bool, default=False
            Print debug information
        """

        if debug:
            print("\n" + "=" * 80)
            print(f"DEBUG: plot_predictor_target_fits")
            print("=" * 80)
            print(f"  Predictor: {predictor}")
            print(f"  Target: {target}")
            print(f"  Fuel type: {fuel_type}")

        # ====================================================================
        # 1. Validate and Get Evaluator
        # ====================================================================

        if fuel_type is None:
            fuel_type = list(self.alternative_datasets.keys())[0]
            if self.verbose:
                print(f"No fuel type specified, using first: {fuel_type}")

        if fuel_type not in self.alternative_datasets:
            raise ValueError(
                f"Fuel type '{fuel_type}' not found. "
                f"Available: {list(self.alternative_datasets.keys())}"
            )

        if layout not in ['horizontal', 'vertical']:
            raise ValueError(f"layout must be 'horizontal' or 'vertical', got '{layout}'")

        if fuel_type not in self.evaluators:
            raise ValueError(
                f"No evaluator found for '{fuel_type}'. "
                f"Run evaluate_all_datasets() first."
            )

        evaluator = self.evaluators[fuel_type]

        # ====================================================================
        # 2. Find Matching Formulas
        # ====================================================================

        matching_formulas = []

        for formula in evaluator.formulae:
            # Check if this formula matches predictor → target
            if (formula.target == target and
                    len(formula.predictors) == 1 and
                    formula.predictors[0] == predictor):
                matching_formulas.append(formula)

        if debug:
            print(f"\n  Found {len(matching_formulas)} matching formulas:")
            for f in matching_formulas:
                print(f"    - {f.name} ({f.formula_type}) [{f.reference}]")

        if len(matching_formulas) == 0:
            print(f"\n⚠️  No single-predictor formulas found for {predictor} → {target}")
            print(f"   Available predictor-target combinations:")

            # Show available combinations
            combinations = set()
            for formula in evaluator.formulae:
                if len(formula.predictors) == 1:
                    combinations.add((formula.predictors[0], formula.target))

            for pred, targ in sorted(combinations):
                print(f"     • {pred} → {targ}")

            return

        if self.verbose:
            print(f"\n📊 Creating fit plots for {len(matching_formulas)} formulas")
            print(f"   Predictor: {predictor}")
            print(f"   Target: {target}")
            print(f"   Fuel type: {fuel_type}")

        # ====================================================================
        # 3. Setup Save Directory
        # ====================================================================

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"   Saving to: {save_path}")

        # ====================================================================
        # 4. Get Datasets from Evaluator
        # ====================================================================

        source_data = evaluator.dataset_manager.get_datasets_for_source('default')
        domain_constraints = source_data.domain_constraints

        conventional_data = source_data.get_dataset('conventional')
        alternative_data = source_data.get_dataset('alternative')

        # Get matched dataset for this predictor
        matching_parameters = tuple([predictor])

        if matching_parameters in evaluator.matched_datasets:
            matched_data = evaluator.matched_datasets[matching_parameters]
        else:
            matched_data = conventional_data.copy()
            if self.verbose:
                print(f"   Warning: No matched dataset for {predictor}, using full conventional")

        # ====================================================================
        # 5. Create Plots for Each Formula
        # ====================================================================

        for formula_idx, formula in enumerate(matching_formulas, 1):

            if debug:
                print(f"\n  --- Formula {formula_idx}/{len(matching_formulas)}: {formula.name} ---")

            # Build title
            title_parts = []

            if title_prefix:
                title_parts.append(title_prefix)
            elif self.shiptype:
                title_parts.append(self.shiptype.string)

            title_parts.append(f"{predictor} → {target}")
            title_parts.append(f"({formula.formula_type})")
            title_parts.append(f"[{formula.reference if formula.reference else 'No reference'}]")

            main_title = " | ".join(title_parts)

            # ================================================================
            # 5a. Setup Figure Layout
            # ================================================================

            if separate_plots:
                # Create 3 separate figures
                figs = []
                axes_list = []

                for dataset_name in ['Conventional', 'Matched (Conv)', f'Alternative ({fuel_type})']:
                    fig, ax = plt.subplots(1, 1, figsize=figsize_per_plot)
                    figs.append(fig)
                    axes_list.append(ax)
                    fig.suptitle(f"{main_title}\n{dataset_name}", fontsize=11, fontweight='bold')

            else:
                # Create single figure with 3 subplots
                if layout == 'horizontal':
                    fig, axes_list = plt.subplots(
                        1, 3,
                        figsize=(figsize_per_plot[0] * 3, figsize_per_plot[1])
                    )
                else:  # vertical
                    fig, axes_list = plt.subplots(
                        3, 1,
                        figsize=(figsize_per_plot[0], figsize_per_plot[1] * 3)
                    )

                fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.995)

            # ================================================================
            # 5b. Plot Each Dataset
            # ================================================================

            for ax_idx, (dataset_name, dataset, color) in enumerate([
                ('Conventional', conventional_data, 'steelblue'),
                ('Matched (Conv)', matched_data, 'orange'),
                (f'Alternative ({fuel_type})', alternative_data, 'coral')
            ]):

                ax = axes_list[ax_idx]

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
                        ax.set_title(dataset_name, fontweight='bold', fontsize=10)
                        continue

                    # ========================================================
                    # Get Metrics for This Dataset
                    # ========================================================

                    dataset_key = {
                        'Conventional': 'conventional',
                        'Matched (Conv)': 'matched',
                        f'Alternative ({fuel_type})': 'alternative'
                    }[dataset_name]

                    # Find metrics in comparison results
                    formula_results = self.comparison_results[
                        (self.comparison_results['formula_name'] == formula.name) &
                        (self.comparison_results['fuel_type'] == fuel_type)
                        ]

                    if len(formula_results) > 0:
                        row = formula_results.iloc[0]
                        r2 = row[f'{dataset_key}_r2']
                        rrmse = row[f'{dataset_key}_rrmse']
                        n = row[f'{dataset_key}_n']
                    else:
                        # Fallback: calculate metrics
                        y_pred_metrics = formula.predict(X_clean.reshape(-1, 1))
                        from sklearn.metrics import r2_score
                        from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics

                        r2 = r2_score(y_true_clean, y_pred_metrics)
                        custom_metrics = RegressionMetrics(y_pred_metrics, y_true_clean)
                        rrmse = custom_metrics.calculate_rrmse()
                        n = len(y_true_clean)

                    # ========================================================
                    # Plot Data Points
                    # ========================================================

                    if show_data_points:
                        ax.scatter(
                            X_clean, y_true_clean,
                            alpha=0.5, s=40,
                            color=color,
                            edgecolors='black',
                            linewidth=0.5,
                            label='Actual data',
                            zorder=3
                        )

                    # ========================================================
                    # Create Formula Prediction Curve
                    # ========================================================

                    X_min = X_clean.min()
                    X_max = X_clean.max()
                    X_range = X_max - X_min

                    # Extend range slightly for visualization
                    X_plot_min = X_min - 0.05 * X_range
                    X_plot_max = X_max + 0.05 * X_range

                    # Handle power laws (need positive values)
                    if formula.formula_type in ['power', 'power_log', 'piecewise_power']:
                        X_plot_min = max(X_plot_min, X_clean[X_clean > 0].min() * 0.95
                        if np.any(X_clean > 0) else 0.1)

                    X_line = np.linspace(X_plot_min, X_plot_max, n_points).reshape(-1, 1)

                    # Make predictions using the FORMULA
                    try:
                        y_line = formula.predict(X_line)

                        # Plot formula curve
                        ax.plot(
                            X_line, y_line,
                            'r-',
                            linewidth=2.5,
                            label='Formula prediction',
                            alpha=0.8,
                            zorder=4
                        )

                        # ================================================
                        # Add Bin Boundaries for Piecewise Formulas
                        # ================================================

                        if formula.formula_type in ['piecewise_power', 'piecewise_linear',
                                                    'piecewise_mixed']:
                            if formula.bins is not None:
                                for i, bin_edge in enumerate(formula.bins[1:-1], 1):
                                    ax.axvline(
                                        bin_edge,
                                        color='darkred',
                                        linestyle='--',
                                        linewidth=1.5,
                                        alpha=0.6,
                                        label='Bin boundary' if i == 1 else '',
                                        zorder=2
                                    )

                                    # Add bin edge label
                                    y_pos = ax.get_ylim()[1] * 0.95
                                    ax.text(
                                        bin_edge, y_pos,
                                        f'{bin_edge:,.0f}',
                                        rotation=90, va='top', ha='right',
                                        fontsize=8, color='darkred', alpha=0.7
                                    )

                    except Exception as e:
                        if debug:
                            print(f"      Warning: Could not plot formula curve: {e}")

                    # ========================================================
                    # Add Domain Boundaries
                    # ========================================================

                    domain_label_added = False

                    if show_domain_bounds:
                        if predictor in domain_constraints.lower_bounds:
                            lower_bound = domain_constraints.lower_bounds[predictor]
                            ax.axvline(
                                lower_bound,
                                color='purple',
                                linestyle='-.',
                                linewidth=2,
                                alpha=0.7,
                                label='Domain bounds' if not domain_label_added else '',
                                zorder=5
                            )
                            domain_label_added = True

                        if predictor in domain_constraints.upper_bounds:
                            upper_bound = domain_constraints.upper_bounds[predictor]
                            ax.axvline(
                                upper_bound,
                                color='purple',
                                linestyle='-.',
                                linewidth=2,
                                alpha=0.7,
                                label='Domain bounds' if not domain_label_added else '',
                                zorder=5
                            )

                    # ========================================================
                    # Add Percentile Bounds
                    # ========================================================

                    if show_percentile_bounds and formula.bounds is not None:
                        if formula.lower_limits and predictor in formula.lower_limits:
                            lower_limit = formula.lower_limits[predictor]
                            upper_limit = formula.upper_limits[predictor]

                            ax.axvline(
                                lower_limit,
                                color='gray',
                                linestyle=':',
                                linewidth=1.5,
                                alpha=0.5,
                                label=f'Percentile bounds ({formula.bounds}%)',
                                zorder=2
                            )
                            ax.axvline(
                                upper_limit,
                                color='gray',
                                linestyle=':',
                                linewidth=1.5,
                                alpha=0.5,
                                zorder=2
                            )

                    # ========================================================
                    # Format Axes
                    # ========================================================

                    ax.set_xlabel(predictor, fontweight='bold', fontsize=11)
                    ax.set_ylabel(target, fontweight='bold', fontsize=11)

                    if not separate_plots:
                        ax.set_title(
                            f'{dataset_name}\nR²={r2:.3f}, RRMSE={rrmse:.1f}%',
                            fontweight='bold',
                            fontsize=10
                        )
                    else:
                        ax.set_title(
                            f'R²={r2:.3f}, RRMSE={rrmse:.1f}%',
                            fontweight='bold',
                            fontsize=10
                        )

                    ax.grid(True, alpha=0.3, zorder=0)
                    ax.legend(loc='best', fontsize=9, framealpha=0.9)

                    # Add formula text
                    formula_text = formula.get_formula_text()

                    # Truncate if too long
                    if len(formula_text) > 200:
                        lines = formula_text.split('\n')
                        if len(lines) > 3:
                            formula_text = '\n'.join(lines[:2]) + f'\n... ({len(lines) - 2} more segments)'

                    ax.text(
                        0.98, 0.02,
                        formula_text,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=7,
                        family='monospace'
                    )

                    # Add sample count
                    ax.text(
                        0.02, 0.02,
                        f'n={n}',
                        transform=ax.transAxes,
                        ha='left', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8
                    )

                except Exception as e:
                    ax.text(
                        0.5, 0.5,
                        f'Error: {str(e)[:50]}',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=8, color='red'
                    )
                    ax.set_title(dataset_name, fontweight='bold')

            # ================================================================
            # 5c. Save or Show
            # ================================================================

            plt.tight_layout()

            if save_dir is not None:
                # Create safe filename
                safe_name = formula.name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                safe_name = safe_name.replace('(', '').replace(')', '')

                if separate_plots:
                    # Save each subplot separately
                    for fig_idx, (fig, dataset_name) in enumerate(zip(
                            figs,
                            ['conventional', 'matched', 'alternative']
                    )):
                        filename = save_path / f'{safe_name}_{dataset_name}_fit.png'
                        fig.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close(fig)

                    if self.verbose:
                        print(f"  ✓ Saved: {safe_name}_*.png (3 files)")
                else:
                    # Save single figure
                    filename = save_path / f'{safe_name}_fit.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    if self.verbose:
                        print(f"  ✓ Saved: {filename.name}")
            else:
                plt.show()

        if save_dir is not None and self.verbose:
            print(f"\n✅ All fit plots saved to {save_path}")

    def compute_literature_reference_metrice(
                                             self,
                                            comparison_results: pd.DataFrame,
                                            variable_groupings: Dict[str, List[str]],
                                            metric_type: str = 'median',
                                            dataset: str = 'conventional'
                                        ) -> Dict[str, float]:

        """
            Compute reference performance metrics from literature formulas.

            Parameters
            ----------
            comparison_results : pd.DataFrame
                Results from literature formula evaluation
            variable_groupings : dict
                Grouping of variables by category
                Example: {'Hull dimensions': ['LBP (M)', 'BEAM (M)', 'DEPTH (M)']}
            metric_type : str
                Type of reference metric: 'median', 'mean', 'best' (lowest RRMSE)
            dataset : str
                Which dataset to compute reference from: 'conventional', 'matched', 'alternative'

            Returns
            -------
            dict
                Category names mapped to reference RRMSE values
                Example: {'Hull dimensions': 8.3, 'Capacity': 12.1}
            """

        reference_metrics = {}

        for category, variables in variable_groupings.items():
            # Filter to formulas targeting this variable group
            category_data = comparison_results[
                comparison_results['target'].isin(variables)
            ]

            if len(category_data) == 0:
                continue

            # Get RRMSE values for the specified dataset
            rrmse_col = f'{dataset}_rrmse'
            rrmse_values = category_data[rrmse_col].dropna()

            if len(rrmse_values) == 0:
                continue

            # Compute reference metric
            if metric_type == 'median':
                reference_metrics[category] = rrmse_values.median()
            elif metric_type == 'mean':
                reference_metrics[category] = rrmse_values.mean()
            elif metric_type == 'best':
                reference_metrics[category] = rrmse_values.min()
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")

        return reference_metrics

    def assign_formula_to_category(self,
                                    target: str,
                                    variable_groupings: Dict[str, List[str]]
                                ) -> Optional[str]:
        """
        Determine which category a formula belongs to based on its target.

        Parameters
        ----------
        target : str
            Target variable of the formula
        variable_groupings : dict
            Grouping of variables by category

        Returns
        -------
        str or None
            Category name, or None if target doesn't match any category
        """
        for category, variables in variable_groupings.items():
            if target in variables:
                return category
        return None

    def create_threshold_summary_text(self,
                                      thresholds: Dict[str, float],
                                      metric_type: str,
                                      dataset: str = 'conventional'
                                      ) -> str:
        """
        Create explanatory text for threshold lines.

        Parameters
        ----------
        thresholds : dict
            Category names mapped to threshold values
        metric_type : str
            Type of reference metric used (e.g. 'median', 'mean', 'best')
        dataset : str
            Dataset the thresholds are based on

        Returns
        -------
        str
            Formatted text explaining the reference metric lines
        """
        metric_label = metric_type.capitalize()
        lines = [f"Reference lines ({metric_label} {dataset} RRMSE):"]

        for category, value in sorted(thresholds.items()):
            lines.append(f"  {category}: {value:.1f}%")

        return '\n'.join(lines)

    def add_category_threshold_lines(self,
                                     ax,
                                     formula_targets: List[str],
                                     formula_positions: np.ndarray,
                                     thresholds: Dict[str, float],
                                     variable_groupings: Dict[str, List[str]],
                                     alpha: float = 0.7,
                                     linewidth: float = 2,
                                     linestyle: str = '--'
                                     ):
        """
        Add threshold lines to axis, with different thresholds for different variable categories.

        Draws horizontal lines at category-specific thresholds, only over the x-range
        where formulas of that category appear. Labels are shown in the summary textbox,
        not on the lines themselves.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to add lines to
        formula_targets : list[str]
            Ordered list of formula targets (same order as x-axis)
        formula_positions : np.ndarray
            X-positions of formulas
        thresholds : dict
            Category names mapped to threshold values
        variable_groupings : dict
            Grouping of variables by category
        alpha : float
            Line transparency
        linewidth : float
            Line width
        linestyle : str
            Line style

        Returns
        -------
        dict
            Mapping of category names to line objects (for legend)
        """
        threshold_lines = {}

        category_colors = {
            'Hull dimensions': '#E74C3C',
            'Capacity': '#3498DB',
            'Operational': '#2ECC71',
            'Propulsion': '#F39C12',
            'Economic': '#9B59B6',
        }

        for category, threshold in thresholds.items():
            category_mask = np.array([
                self.assign_formula_to_category(target, variable_groupings) == category
                for target in formula_targets
            ])

            if not np.any(category_mask):
                continue

            category_positions = formula_positions[category_mask]
            x_min = category_positions.min() - 0.5
            x_max = category_positions.max() + 0.5

            color = category_colors.get(category, '#95A5A6')

            line = ax.hlines(
                y=threshold,
                xmin=x_min,
                xmax=x_max,
                colors=color,
                linestyles=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=f'{category} ({threshold:.1f}%)',
                zorder=10
            )

            threshold_lines[category] = line

        return threshold_lines

    # ============================================================================
    # ADD TO: MultiDatasetLiteratureEvaluator
    # ============================================================================

    def run_significance_tests(self,
                               comparison: str = 'fuel_type',
                               n_bootstrap: int = 5000,
                               confidence_level: float = 0.95,
                               ci_method: str = 'percentile',
                               random_seed: Optional[int] = None,
                               debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Run bootstrap significance tests across all alternative fuel datasets.

        Delegates to each per-fuel-type LiteratureFormulaEvaluator, then
        aggregates results into a combined dictionary.

        Step-by-step data flow:
        -----------------------
        1. self.evaluators                    → one evaluator per fuel type (set by evaluate_all_datasets)
        2. evaluator.run_significance_tests() → bootstrap tests for that fuel type
        3. results tagged with 'fuel_type'   → aggregated into combined dict + DataFrame

        Parameters
        ----------
        comparison : str
            Which gap to test: 'fuel_type', 'distribution', or 'total'
        n_bootstrap : int
            Bootstrap iterations per formula per fuel type
        confidence_level : float
            CI confidence level
        ci_method : str
            'percentile' or 'bca'
        random_seed : int, optional
            For reproducibility
        debug : bool
            Print per-formula debug output

        Returns
        -------
        dict[str, list[dict]]
            Maps fuel_type name to list of significance test result dicts.
            Each dict contains: formula_name, target, predictors, reference,
            comparison_label, rrmse_a, rrmse_b, delta, ci_lower, ci_upper,
            is_significant, n_a, n_b
        """
        if not self.evaluators:
            raise RuntimeError(
                "No evaluators found. Run evaluate_all_datasets() first."
            )

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Bootstrap Significance Tests ({comparison} gap)")
            print(f"n_bootstrap={n_bootstrap}, CI={confidence_level * 100:.0f}%, method={ci_method}")
            print(f"{'=' * 70}\n")

        all_significance_results = {}

        for fuel_type, evaluator in self.evaluators.items():

            if self.verbose:
                print(f"  Testing: {fuel_type}...")

            # Data flow: evaluator.run_significance_tests() uses:
            #   evaluator.formulae, evaluator.dataset_manager, evaluator.matched_datasets
            fuel_results = evaluator.run_significance_tests(
                comparison=comparison,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                ci_method=ci_method,
                random_seed=random_seed,
                debug=debug
            )

            # Tag each result with the fuel type
            for r in fuel_results:
                r['fuel_type'] = fuel_type

            all_significance_results[fuel_type] = fuel_results

            if self.verbose:
                n_sig = sum(1 for r in fuel_results if r.get('is_significant'))
                n_total = sum(1 for r in fuel_results if r.get('is_significant') is not None)
                print(f"    → {n_sig}/{n_total} formulae show significant gaps")

        # Store flat combined DataFrame for easy querying
        self._significance_results = all_significance_results
        self._significance_df = self._build_significance_dataframe(all_significance_results)

        if self.verbose:
            print(f"\n✓ Significance tests complete for {len(self.evaluators)} fuel types")

        return all_significance_results

    def _build_significance_dataframe(self, all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Flatten significance results dict into a single summary DataFrame.

        Data flow:
            all_results (dict[fuel_type → list[dict]])
            → tagged rows
            → pd.DataFrame (one row per formula per fuel type)
        """
        from knowledge_layer.literature_regressions.bootstrap_formula_tester import BootstrapSignificanceTester

        tester = BootstrapSignificanceTester()
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

        One sheet per fuel type, plus a combined summary sheet.

        Parameters
        ----------
        output_path : str
            Path for .xlsx output file
        significance_results : dict, optional
            Output from run_significance_tests(). If None, uses cached results.
        """
        from knowledge_layer.literature_regressions.bootstrap_formula_tester import BootstrapSignificanceTester

        if significance_results is None:
            if not hasattr(self, '_significance_results'):
                raise RuntimeError("No significance results. Run run_significance_tests() first.")
            significance_results = self._significance_results

        tester = BootstrapSignificanceTester()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Sheet per fuel type
            for fuel_type, results in significance_results.items():
                df = tester.results_to_dataframe(results)
                safe_name = fuel_type[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=safe_name, index=False)

                ws = writer.sheets[safe_name]
                for col in ws.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col)
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

            # Combined summary sheet
            if hasattr(self, '_significance_df') and len(self._significance_df) > 0:
                self._significance_df.to_excel(writer, sheet_name='All_Results', index=False)

        print(f"✓ Significance results exported to: {output_path}")

    def generate_formula_latex_table(self,
                                     output_dir: str = '.',
                                     prefix: Optional[str] = None,
                                     targets: Optional[List[str]] = None,
                                     references: Optional[List[str]] = None,
                                     display_map: Optional[Dict[str, str]] = None,
                                     notation_map: Optional[Dict[str, str]] = None,
                                     landscape: bool = False,
                                     show_reference: bool = True,
                                     show_formula_type: bool = True,
                                     show_metrics:bool=True,
                                     debug: bool = False) -> Dict[str, Path]:
        """
        Generate and save a LaTeX table of the literature formula definitions.

        This table shows WHAT the formulae are (mathematical form), not how
        they perform. It is independent of evaluate_all_datasets().

        Step-by-step data flow:
        -----------------------
        1. self (MultiDatasetLiteratureEvaluator)
               → passed to LiteratureFormulaLatexGenerator
        2. Generator reads self.evaluators[first_key].formulae
               (or self.formulae if no evaluators yet — reads from _create_evaluator)
        3. LiteratureFormulaRenderer.render(formula) → LaTeX math string per formula
        4. Table assembled and saved to disk

        Parameters
        ----------
        output_dir : str
            Directory for output files.
        prefix : str, optional
            Filename prefix. Defaults to <ShipType.name>_literature.
        targets : list[str], optional
            Restrict to these target variables.
        references : list[str], optional
            Restrict to these references/sources.
        display_map : dict, optional
            Override/extend the default variable display map for column cells.
        notation_map : dict, optional
            Override/extend the default notation map for equation symbols.
        landscape : bool
            Use sidewaystable environment.
        show_reference : bool
            Include Reference column.
        show_formula_type : bool
            Include Formula Type column.
        debug : bool
            Print rendering details.

        Returns
        -------
        dict
            {'table': Path, 'preamble': Path}
        """
        from knowledge_layer.literature_regressions.formulae_renderer import (
            LiteratureFormulaLatexGenerator
        )

        # If evaluate_all_datasets() has not been run yet, we need at least
        # one evaluator to read formulae from. Create one silently.
        if not self.evaluators:
            if self.verbose:
                print("No evaluators found — creating one to read formulae...")
            first_fuel = next(iter(self.alternative_datasets))
            self.evaluators[first_fuel] = self._create_evaluator(
                self.alternative_datasets[first_fuel], first_fuel
            )

        generator = LiteratureFormulaLatexGenerator(
            evaluator=self,
            display_map=display_map,
            notation_map=notation_map,
            verbose=self.verbose
        )

        return generator.save_formula_table(
            output_dir=output_dir,
            prefix=prefix,
            targets=targets,
            references=references,
            landscape=landscape,
            show_reference=show_reference,
            show_formula_type=show_formula_type,
            show_metrics=show_metrics,
            debug=debug
        )