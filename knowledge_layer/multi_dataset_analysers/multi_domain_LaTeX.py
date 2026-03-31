# multi_domain_LaTeX.py
"""
LaTeX Table Generator for Multi-Dataset Domain Analysis Results
================================================================
Creates publication-ready LaTeX tables for comparing multiple alternative fuel types.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.display_functions.LaTeX_table_utils import LaTeXFormatter, LaTeXPreambleGenerator


class MultiDatasetLatexTableGenerator:
    """
    Generate LaTeX tables from MultiDatasetDomainAnalyzer results.

    **Design Pattern**: Uses composition with LaTeXFormatter to handle all
    formatting operations, keeping this class focused on table structure.
    """

    def __init__(self, multi_analyzer, verbose: bool = True):
        """
        Initialize table generator.

        Parameters
        ----------
        multi_analyzer : MultiDatasetDomainAnalyzer
            Analyzer with comparison results
        verbose : bool
            Print progress messages
        """
        self.multi_analyzer = multi_analyzer
        self.verbose = verbose

        # Composition: Create formatter instance
        self.formatter = LaTeXFormatter()

        # Validate input
        if multi_analyzer.comparison_results is None:
            raise ValueError(
                "MultiDatasetDomainAnalyzer has no results. "
                "Run analyze_all_datasets() first."
            )

        self.results_df = multi_analyzer.comparison_results.copy()

        # ✅ FIXED: Preserve order from original dictionary instead of sorting
        # Get fuel types in the order they appear in the original alternative_datasets dict
        if hasattr(multi_analyzer, 'alternative_datasets'):
            # Use the order from the original dictionary
            self.fuel_types = list(multi_analyzer.alternative_datasets.keys())
        else:
            # Fallback: maintain order of first appearance in results
            self.fuel_types = list(self.results_df['fuel_type'].unique())

        if self.verbose:
            print(f"✅ Initialized with {len(self.results_df)} results")
            print(f"   Fuel types: {', '.join(self.fuel_types)}")
            print(f"   Targets: {self.results_df['target'].nunique()}")

    # ========================================================================
    # R² TABLE GENERATION
    # ========================================================================

    def generate_r2_table(self,
                          ship_type: ShipType,
                          targets: Optional[List[str]] = None,
                          caption: str = None,
                          label: str = None,
                          max_predictors: int = 2,
                          best_only: bool = True,
                          landscape: bool = False) -> str:
        """
        Generate R² comparison table across fuel types.

        **Format**: Target | Predictors | Model | Train | Test |
                   [Match | Alt | ΔFuel] for each fuel type

        Parameters
        ----------
        ship_type : ShipType
            Ship type for caption/label
        targets : List[str], optional
            Specific targets to include
        caption, label : str, optional
            Custom caption/label
        max_predictors : int
            Maximum number of predictors
        best_only : bool
            If True, show only best combination per target
        landscape : bool
            If True, use sidewaystable environment

        Returns
        -------
        str
            LaTeX table code
        """
        # Filter data
        df = self._filter_and_prepare_data(targets, max_predictors, best_only, 'alternative_r2')

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            fuel_list = ', '.join(self.fuel_types)
            caption = (
                f"$R^{{2}}_{{adj}}$ performance for {len(df['target'].unique())} target dimensions "
                f"across multiple alternative fuel types ({fuel_list}) for {ship_type.string} vessels. "
                f"Showing best predictor combination per target (max {max_predictors} predictors)."
            )

        if label is None:
            label = f"tab_{ship_type.name}_r2_multi"

        # Build table
        latex = self._build_r2_table_structure(df, caption, label, landscape)

        return latex

    def _build_r2_table_structure(self,
                                  df: pd.DataFrame,
                                  caption: str,
                                  label: str,
                                  landscape: bool) -> str:
        """Build R² table LaTeX code."""
        lines = []

        # Table environment
        if landscape:
            lines.append(r"\begin{sidewaystable}[p]")
        else:
            lines.append(r"\begin{table}[ht]")

        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\footnotesize")

        # Column spec
        colspec = self._generate_r2_colspec()
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Headers
        header_lines = self._generate_r2_headers()
        lines.extend(header_lines)

        lines.append(r"\midrule")

        # Data rows
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            row_line = self._generate_r2_data_row(target, target_df)
            lines.append(row_line)

        # Footer
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")

        if landscape:
            lines.append(r"\end{sidewaystable}")
        else:
            lines.append(r"\end{table}")

        return "\n".join(lines)

    def _generate_r2_colspec(self) -> str:
        """
        Generate column specification for R² table.

        Format: Target(10%) | Predictors(15%) | Model(10%) | Train(6.5%) | Test(6.5%) |
                [Match(6.5%) | Alt(6.5%) | ΔFuel(6.5%)] × n_fuel_types
        """
        n_fuel = len(self.fuel_types)

        # Base columns: ~42% of width
        base = [
            r"p{0.1\textwidth}",  # Target
            r"p{0.15\textwidth}",  # Predictors (capital P for ragged-right)
            r"p{0.1\textwidth}",  # Model
            r"p{0.065\textwidth}",  # Train
            r"p{0.065\textwidth}"  # Test
        ]

        # Fuel type columns: 3 columns per fuel type
        # Remaining width ≈ 58%, distributed across fuel types
        col_width = 0.065  # 6.5% per column
        fuel_cols = [r"p{0.065\textwidth}"] * 3  # Match, Alt, ΔFuel

        all_cols = base + (fuel_cols * n_fuel)

        return "".join(all_cols)

    def _generate_r2_headers(self) -> List[str]:
        """
        Generate header rows for R² table.

        Creates:
        1. Top row with multicolumn for "Conventional" and each fuel type
        2. Bottom row with specific column names
        """
        lines = []

        # Top header row with multicolumns
        top_parts = [
            "",  # Target
            "",  # Predictors
            "",  # Model
            self.formatter.multicolumn("Conventional", 2, 'c'),
        ]

        # Add multicolumn for each fuel type (3 columns each)
        for fuel_type in self.fuel_types:
            fuel_abbr = self.formatter.abbreviate_fuel_type(fuel_type)
            top_parts.append(self.formatter.multicolumn(fuel_abbr, 3, 'c'))

        lines.append(" & ".join(top_parts) + r"\\")

        # Cmidrules
        lines.append(self.formatter.cmidrule(4, 5, trim='lr'))  # Conventional

        col_start = 6
        for _ in self.fuel_types:
            lines.append(self.formatter.cmidrule(col_start, col_start + 2, trim='lr'))
            col_start += 3

        # Bottom header row with column names
        bottom_parts = [
            "Target",
            "Predictors",
            "Model",
            "$R^2_{train}$",
            "$R^2_{test}$"
        ]

        # Add subheaders for each fuel type
        for _ in self.fuel_types:
            bottom_parts.extend([
                "$R^2_{match}$",
                "$R^2_{alt}$",
                "$\\Delta_{fuel}$"
            ])

        lines.append(" & ".join(bottom_parts) + r"\\")

        return lines

    def _generate_r2_data_row(self,
                              target: str,
                              target_df: pd.DataFrame) -> str:
        """Generate one data row for R² table."""
        parts = []

        # Target name
        parts.append(self.formatter.format_variable_name(target))

        first_row = target_df.iloc[0]

        # Predictors
        pred_str = self.formatter.format_predictor_str(
            first_row['predictors'],
            max_length=25
        )
        parts.append(pred_str)

        # Model
        model_str = self.formatter.format_model_name(first_row['model_type'])
        parts.append(model_str)

        # Train and Test (same for all fuel types)
        parts.append(self.formatter.format_r2_value(first_row['train_r2']))
        parts.append(self.formatter.format_r2_value(first_row['test_r2']))

        # Metrics for each fuel type: Match, Alt, ΔFuel
        for fuel_type in self.fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                parts.extend(["---", "---", "---"])
            else:
                fuel_data = fuel_row.iloc[0]

                # Match R²
                parts.append(self.formatter.format_r2_value(fuel_data['matched_r2']))

                # Alt R²
                parts.append(self.formatter.format_r2_value(fuel_data['alternative_r2']))

                # ✅ FIXED: Simple difference (alternative - matched)
                # Negative = performance drop, Positive = performance gain
                gap_value = fuel_data['alternative_r2'] - fuel_data['matched_r2']

                # Highlight if gap < -0.10 (10% drop in R²)
                if gap_value < -0.10:
                    gap_str = self.formatter.format_r2_gap(gap_value, use_highlight=True)
                else:
                    gap_str = self.formatter.format_r2_gap(gap_value, use_highlight=False)

                parts.append(gap_str)

        return " & ".join(parts) + r"\\"

    # ========================================================================
    # RRMSE TABLE GENERATION
    # ========================================================================

    def generate_rrmse_table(self,
                             ship_type: ShipType,
                             targets: Optional[List[str]] = None,
                             caption: str = None,
                             label: str = None,
                             max_predictors: int = 2,
                             best_only: bool = True,
                             landscape: bool = False) -> str:
        """
        Generate RRMSE comparison table across fuel types.

        **Format**: Target | Predictors | Model | Train | Test |
                   [Match | Alt] for each fuel type (NO GAPS)

        Parameters
        ----------
        ship_type : ShipType
            Ship type for caption/label
        targets : List[str], optional
            Specific targets to include
        caption, label : str, optional
            Custom caption/label
        max_predictors : int
            Maximum number of predictors
        best_only : bool
            If True, show only best combination per target
        landscape : bool
            If True, use sidewaystable environment

        Returns
        -------
        str
            LaTeX table code
        """
        # Filter data
        df = self._filter_and_prepare_data(targets, max_predictors, best_only, 'alternative_rrmse')

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            fuel_list = ', '.join(self.fuel_types)
            caption = (
                f"RRMSE (\\%) performance for {len(df['target'].unique())} target dimensions "
                f"across multiple alternative fuel types ({fuel_list}) for {ship_type.string} vessels. "
                f"Showing best predictor combination per target (max {max_predictors} predictors)."
            )

        if label is None:
            label = f"tab_{ship_type.name}_rrmse_multi"

        # Build table
        latex = self._build_rrmse_table_structure(df, caption, label, landscape)

        return latex

    def _build_rrmse_table_structure(self,
                                     df: pd.DataFrame,
                                     caption: str,
                                     label: str,
                                     landscape: bool) -> str:
        """Build RRMSE table LaTeX code."""
        lines = []

        if landscape:
            lines.append(r"\begin{sidewaystable}[p]")
        else:
            lines.append(r"\begin{table}[ht]")

        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\footnotesize")

        # Column spec
        colspec = self._generate_rrmse_colspec()
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Headers
        header_lines = self._generate_rrmse_headers()
        lines.extend(header_lines)

        lines.append(r"\midrule")

        # Data rows
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            row_line = self._generate_rrmse_data_row(target, target_df)
            lines.append(row_line)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")

        if landscape:
            lines.append(r"\end{sidewaystable}")
        else:
            lines.append(r"\end{table}")

        return "\n".join(lines)

    def _generate_rrmse_colspec(self) -> str:
        """
        Generate column specification for RRMSE table.

        Format: Target(10%) | Predictors(15%) | Model(10%) | Train(10%) | Test(10%) |
                [Match(10%) | Alt(10%)] × n_fuel_types
        """
        n_fuel = len(self.fuel_types)

        # Base columns
        base = [
            r"p{0.1\textwidth}",  # Target
            r"p{0.15\textwidth}",  # Predictors
            r"p{0.1\textwidth}",  # Model
            r"p{0.1\textwidth}",  # Train
            r"p{0.1\textwidth}"  # Test
        ]

        # Fuel type columns: 2 columns per fuel type (Match, Alt only)
        # RRMSE values need more space, so use 10% per column
        fuel_cols = [r"p{0.1\textwidth}", r"p{0.1\textwidth}"]  # Match, Alt

        all_cols = base + (fuel_cols * n_fuel)

        return "".join(all_cols)

    def _generate_rrmse_headers(self) -> List[str]:
        """Generate header rows for RRMSE table."""
        lines = []

        # Top header row with multicolumns
        top_parts = [
            "",  # Target
            "",  # Predictors
            "",  # Model
            self.formatter.multicolumn("Conventional", 2, 'c'),
        ]

        # Add multicolumn for each fuel type (2 columns each)
        for fuel_type in self.fuel_types:
            fuel_abbr = self.formatter.abbreviate_fuel_type(fuel_type)
            top_parts.append(self.formatter.multicolumn(fuel_abbr, 2, 'c'))

        lines.append(" & ".join(top_parts) + r"\\")

        # Cmidrules
        lines.append(self.formatter.cmidrule(4, 5, trim='lr'))

        col_start = 6
        for _ in self.fuel_types:
            lines.append(self.formatter.cmidrule(col_start, col_start + 1, trim='lr'))
            col_start += 2

        # Bottom header row
        bottom_parts = [
            "Target",
            "Predictors",
            "Model",
            "RRMSE$_{train}$",
            "RRMSE$_{test}$"
        ]

        # Add subheaders for each fuel type
        for _ in self.fuel_types:
            bottom_parts.extend([
                "RRMSE$_{match}$",
                "RRMSE$_{alt}$"
            ])

        lines.append(" & ".join(bottom_parts) + r"\\")

        return lines

    def _generate_rrmse_data_row(self,
                                 target: str,
                                 target_df: pd.DataFrame) -> str:
        """Generate one data row for RRMSE table."""
        parts = []

        # Target name
        formatted_target = self.formatter.format_variable_name(target)
        parts.append(formatted_target)

        first_row = target_df.iloc[0]

        # Predictors
        parts.append(self.formatter.format_predictor_str(
            first_row['predictors'],
            max_length=25
        ))

        # Model
        parts.append(self.formatter.format_model_name(first_row['model_type']))

        # Train and Test (with color)
        parts.append(self.formatter.format_rrmse_value(first_row['train_rrmse']))
        parts.append(self.formatter.format_rrmse_value(first_row['test_rrmse']))

        # Metrics for each fuel type: Match, Alt only (no gaps)
        for fuel_type in self.fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                parts.extend(["---", "---"])
            else:
                fuel_data = fuel_row.iloc[0]

                parts.append(self.formatter.format_rrmse_value(fuel_data['matched_rrmse']))
                parts.append(self.formatter.format_rrmse_value(fuel_data['alternative_rrmse']))

        return " & ".join(parts) + r"\\"

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _filter_and_prepare_data(self,
                                 targets: Optional[List[str]],
                                 max_predictors: int,
                                 best_only: bool,
                                 sort_metric: str) -> pd.DataFrame:
        """Filter and prepare data for table generation."""
        df = self.results_df.copy()

        # Filter by targets
        if targets is not None:
            df = df[df['target'].isin(targets)]

        # Filter by max_predictors
        df = df[df['n_predictors'] <= max_predictors]

        if len(df) == 0:
            return df

        # If best_only, keep best combination per target
        if best_only:
            best_combos = []

            for target in df['target'].unique():
                target_df = df[df['target'] == target]

                # Calculate average metric across fuel types for each combo
                combo_scores = {}
                for combo in target_df['predictor_str'].unique():
                    combo_df = target_df[target_df['predictor_str'] == combo]

                    # Check if this combo exists for all fuel types
                    has_all = all(
                        ft in combo_df['fuel_type'].values
                        for ft in self.fuel_types
                    )

                    if has_all:
                        avg_score = combo_df[sort_metric].mean()
                        combo_scores[combo] = avg_score

                if combo_scores:
                    # Get best combo
                    if 'rrmse' in sort_metric:
                        best_combo = min(combo_scores.items(), key=lambda x: x[1])[0]
                    else:
                        best_combo = max(combo_scores.items(), key=lambda x: x[1])[0]

                    # Keep only this combo for this target
                    best_combos.append(
                        target_df[target_df['predictor_str'] == best_combo]
                    )

            if best_combos:
                df = pd.concat(best_combos, ignore_index=True)

        return df

    # ========================================================================
    # SAVE METHODS
    # ========================================================================

    def save_tables(self,
                    ship_type: ShipType,
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    max_predictors: int = 2,
                    prefix: str = None,
                    suffix:str=None,
                    best_only: bool = True,
                    landscape: bool = False) -> Dict[str, Path]:
        """Generate and save both R² and RRMSE tables."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = "multi_fuel"


        # Generate tables
        r2_table = self.generate_r2_table(
            ship_type=ship_type,
            targets=targets,
            max_predictors=max_predictors,
            best_only=best_only,
            landscape=landscape
        )

        rrmse_table = self.generate_rrmse_table(
            ship_type=ship_type,
            targets=targets,
            max_predictors=max_predictors,
            best_only=best_only,
            landscape=landscape
        )

        # Save files

        if suffix is not None:
            r2_path = output_path / f"{ship_type.name}_{prefix}_r2_{suffix}.tex"
            rrmse_path = output_path / f"{ship_type.name}_{prefix}_rrmse_{suffix}.tex"
        else:
            r2_path = output_path / f"{ship_type.name}_{prefix}_r2.tex"
            rrmse_path = output_path / f"{ship_type.name}_{prefix}_rrmse.tex"

        r2_path.write_text(r2_table)
        rrmse_path.write_text(rrmse_table)

        # Save preamble
        preamble_path = output_path / f"{ship_type.name}_{prefix}_preamble_{suffix}.tex"
        LaTeXPreambleGenerator.save_preamble(str(preamble_path))

        if self.verbose:
            print(f"✅ Multi-dataset tables saved:")
            print(f"   R² table:    {r2_path}")
            print(f"   RRMSE table: {rrmse_path}")
            print(f"   Preamble:    {preamble_path}")

        return {
            'r2': r2_path,
            'rrmse': rrmse_path,
            'preamble': preamble_path
        }

    @staticmethod
    def print_latex_preamble():
        """Print required LaTeX preamble."""
        LaTeXPreambleGenerator.print_preamble()