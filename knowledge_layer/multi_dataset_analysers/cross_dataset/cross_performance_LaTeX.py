# multi_cross_performance_LaTeX.py
"""
LaTeX Table Generator for Multi-Dataset Cross-Performance Analysis Results
===========================================================================
Creates publication-ready LaTeX tables for comparing cross-training performance
across multiple alternative fuel types.

Key Question: "Does training on alternative ships improve predictions, and does
this vary by fuel type?"
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.display_functions.LaTeX_table_utils import LaTeXFormatter, LaTeXPreambleGenerator


class MultiCrossPerformanceLatexTableGenerator:
    """
    Generate LaTeX tables from MultiDatasetCrossPerformanceAnalyzer results.

    **Table Structure**:
    Target | Predictors | Model | Conv (R²±σ) | [Match (R²±σ) |] Alt (R²±σ)
                                                  for each fuel type

    **Key Difference from Domain Shift Tables**:
    - Shows TRAINING SET performance (what happens when trained on different datasets)
    - Optional matched column (can be excluded for cleaner tables)
    - Compares Conv→Alt performance drop across fuel types

    Examples
    --------
    >>> generator = MultiCrossPerformanceLatexTableGenerator(
    ...     multi_analyzer,
    ...     verbose=True
    ... )
    >>>
    >>> # Generate both tables
    >>> generator.save_tables(
    ...     ship_type=ShipType.TANKER,
    ...     output_dir='tables/',
    ...     include_matched=False,  # Cleaner tables
    ...     landscape=True  # For many fuel types
    ... )
    """

    def __init__(self, multi_analyzer, verbose: bool = True):
        """
        Initialize table generator.

        Parameters
        ----------
        multi_analyzer : MultiDatasetCrossPerformanceAnalyzer
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
                "MultiDatasetCrossPerformanceAnalyzer has no results. "
                "Run analyze_all_datasets() first."
            )

        self.results_df = multi_analyzer.comparison_results.copy()

        # ✅ Preserve order from original dictionary
        if hasattr(multi_analyzer, 'alternative_datasets'):
            self.fuel_types = list(multi_analyzer.alternative_datasets.keys())
        else:
            self.fuel_types = list(self.results_df['fuel_type'].unique())

        if self.verbose:
            print(f"✅ Initialized with {len(self.results_df)} results")
            print(f"   Fuel types: {', '.join(self.fuel_types)}")
            print(f"   Targets: {self.results_df['target'].nunique()}")
            print(
                f"   Evaluation method: {self.results_df['evaluation_method'].iloc[0] if 'evaluation_method' in self.results_df.columns else 'unknown'}")

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
                          include_matched: bool = False,
                          landscape: bool = False) -> str:
        """
        Generate R² cross-training performance table.

        **Shows**: Performance when models are trained on different datasets

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
        include_matched : bool, default=False
            If True, includes matched dataset column for each fuel type.
            **Recommended False** for cleaner tables - matched is less relevant
            for cross-training analysis.
        landscape : bool
            If True, use sidewaystable environment (recommended for >2 fuel types)

        Returns
        -------
        str
            LaTeX table code
        """
        # Filter data
        df = self._filter_and_prepare_data(targets, max_predictors, best_only, 'alt_r2')

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            fuel_list = ', '.join(self.fuel_types)
            eval_method = df['evaluation_method'].iloc[0] if 'evaluation_method' in df.columns else 'CV'

            caption = (
                f"Cross-training $R^{{2}}$ performance for {len(df['target'].unique())} target dimensions "
                f"across multiple alternative fuel types ({fuel_list}) for {ship_type.string} vessels. "
                f"Evaluation: {eval_method}. "
                f"Conv = trained on conventional ships, Alt = trained on alternative ships. "
                f"Showing best predictor combination per target (max {max_predictors} predictors)."
            )

        if label is None:
            label = f"tab:{ship_type.name}_cross_r2_multi"

        # Build table
        latex = self._build_r2_table_structure(df, caption, label, landscape, include_matched)

        return latex

    def _build_r2_table_structure(self,
                                  df: pd.DataFrame,
                                  caption: str,
                                  label: str,
                                  landscape: bool,
                                  include_matched: bool) -> str:
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
        colspec = self._generate_r2_colspec(include_matched)
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Headers
        header_lines = self._generate_r2_headers(include_matched)
        lines.extend(header_lines)

        lines.append(r"\midrule")

        # Data rows
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            row_line = self._generate_r2_data_row(target, target_df, include_matched)
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

    def _generate_r2_colspec(self, include_matched: bool) -> str:
        """
        Generate column specification for R² table.

        **Without matched**:
        Target(10%) | Predictors(15%) | Model(10%) | Conv(10%) |
        [Alt(10%)] × n_fuel_types

        **With matched**:
        Target(8%) | Predictors(12%) | Model(8%) | Conv(8%) |
        [Match(8%) | Alt(8%)] × n_fuel_types
        """
        n_fuel = len(self.fuel_types)

        if include_matched:
            # More columns, so tighter spacing
            base = [
                r"p{0.08\textwidth}",  # Target
                r"p{0.12\textwidth}",  # Predictors
                r"p{0.08\textwidth}",  # Model
                r"p{0.08\textwidth}",  # Conv
            ]
            fuel_cols = [r"p{0.08\textwidth}", r"p{0.08\textwidth}"]  # Match, Alt
        else:
            # Fewer columns, more space
            base = [
                r"p{0.10\textwidth}",  # Target
                r"P{0.15\textwidth}",  # Predictors
                r"p{0.10\textwidth}",  # Model
                r"p{0.10\textwidth}",  # Conv
            ]
            fuel_cols = [r"p{0.10\textwidth}"]  # Alt only

        all_cols = base + (fuel_cols * n_fuel)

        return "".join(all_cols)

    def _generate_r2_headers(self, include_matched: bool) -> List[str]:
        """Generate header rows for R² table."""
        lines = []

        # Top header row with multicolumns
        top_parts = [
            "",  # Target
            "",  # Predictors
            "",  # Model
            "Conventional",
        ]

        # Add column for each fuel type
        cols_per_fuel = 2 if include_matched else 1

        for fuel_type in self.fuel_types:
            fuel_abbr = self.formatter.abbreviate_fuel_type(fuel_type)
            if cols_per_fuel > 1:
                top_parts.append(self.formatter.multicolumn(fuel_abbr, cols_per_fuel, 'c'))
            else:
                top_parts.append(fuel_abbr)

        lines.append(" & ".join(top_parts) + r"\\")

        # Cmidrules (only if using multicolumn)
        if include_matched:
            col_start = 5
            for _ in self.fuel_types:
                lines.append(self.formatter.cmidrule(col_start, col_start + 1, trim='lr'))
                col_start += 2

        # Bottom header row with column names
        bottom_parts = [
            "Target",
            "Predictors",
            "Model",
            "$R^2_{conv}$",
        ]

        # Add subheaders for each fuel type
        for _ in self.fuel_types:
            if include_matched:
                bottom_parts.extend([
                    "$R^2_{match}$",
                    "$R^2_{alt}$"
                ])
            else:
                bottom_parts.append("$R^2_{alt}$")

        lines.append(" & ".join(bottom_parts) + r"\\")

        return lines

    def _generate_r2_data_row(self,
                              target: str,
                              target_df: pd.DataFrame,
                              include_matched: bool) -> str:
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

        # Conventional performance (same for all fuel types)
        conv_r2 = first_row['conv_r2']
        conv_std = first_row.get('conv_r2_std', 0)
        parts.append(self.formatter.format_r2_with_std(conv_r2, conv_std))

        # Metrics for each fuel type
        for fuel_type in self.fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                if include_matched:
                    parts.extend(["---", "---"])
                else:
                    parts.append("---")
            else:
                fuel_data = fuel_row.iloc[0]

                if include_matched:
                    # Matched R²
                    match_r2 = fuel_data['matched_r2']
                    match_std = fuel_data.get('matched_r2_std', 0)
                    parts.append(self.formatter.format_r2_with_std(match_r2, match_std))

                # Alternative R²
                alt_r2 = fuel_data['alt_r2']
                alt_std = fuel_data.get('alt_r2_std', 0)

                # ✅ Highlight if significant drop from conventional
                performance_drop = conv_r2 - alt_r2

                if performance_drop > 0.15:  # Drop > 15% R²
                    alt_str = self.formatter.format_r2_with_std(
                        alt_r2, alt_std, use_highlight=True
                    )
                else:
                    alt_str = self.formatter.format_r2_with_std(alt_r2, alt_std)

                parts.append(alt_str)

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
                             include_matched: bool = False,
                             landscape: bool = False) -> str:
        """
        Generate RRMSE cross-training performance table.

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
        include_matched : bool, default=False
            If True, includes matched dataset column
        landscape : bool
            If True, use sidewaystable environment

        Returns
        -------
        str
            LaTeX table code
        """
        # Filter data
        df = self._filter_and_prepare_data(targets, max_predictors, best_only, 'alt_rrmse')

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            fuel_list = ', '.join(self.fuel_types)
            eval_method = df['evaluation_method'].iloc[0] if 'evaluation_method' in df.columns else 'CV'

            caption = (
                f"Cross-training RRMSE (\\%) performance for {len(df['target'].unique())} target dimensions "
                f"across multiple alternative fuel types ({fuel_list}) for {ship_type.string} vessels. "
                f"Evaluation: {eval_method}. "
                f"Conv = trained on conventional ships, Alt = trained on alternative ships. "
                f"Showing best predictor combination per target (max {max_predictors} predictors)."
            )

        if label is None:
            label = f"tab:{ship_type.name}_cross_rrmse_multi"

        # Build table
        latex = self._build_rrmse_table_structure(df, caption, label, landscape, include_matched)

        return latex

    def _build_rrmse_table_structure(self,
                                     df: pd.DataFrame,
                                     caption: str,
                                     label: str,
                                     landscape: bool,
                                     include_matched: bool) -> str:
        """Build RRMSE table LaTeX code."""
        lines = []

        if landscape:
            lines.append(r"\begin{sidewaystable}[p]")
        else:
            lines.append(r"\begin{table}[ht]")

        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\footnotesize")

        # Column spec (same as R²)
        colspec = self._generate_r2_colspec(include_matched)
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Headers
        header_lines = self._generate_rrmse_headers(include_matched)
        lines.extend(header_lines)

        lines.append(r"\midrule")

        # Data rows
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            row_line = self._generate_rrmse_data_row(target, target_df, include_matched)
            lines.append(row_line)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")

        if landscape:
            lines.append(r"\end{sidewaystable}")
        else:
            lines.append(r"\end{table}")

        return "\n".join(lines)

    def _generate_rrmse_headers(self, include_matched: bool) -> List[str]:
        """Generate header rows for RRMSE table."""
        lines = []

        # Top header row
        top_parts = [
            "",  # Target
            "",  # Predictors
            "",  # Model
            "Conventional",
        ]

        cols_per_fuel = 2 if include_matched else 1

        for fuel_type in self.fuel_types:
            fuel_abbr = self.formatter.abbreviate_fuel_type(fuel_type)
            if cols_per_fuel > 1:
                top_parts.append(self.formatter.multicolumn(fuel_abbr, cols_per_fuel, 'c'))
            else:
                top_parts.append(fuel_abbr)

        lines.append(" & ".join(top_parts) + r"\\")

        # Cmidrules
        if include_matched:
            col_start = 5
            for _ in self.fuel_types:
                lines.append(self.formatter.cmidrule(col_start, col_start + 1, trim='lr'))
                col_start += 2

        # Bottom header row
        bottom_parts = [
            "Target",
            "Predictors",
            "Model",
            "RRMSE$_{conv}$",
        ]

        for _ in self.fuel_types:
            if include_matched:
                bottom_parts.extend([
                    "RRMSE$_{match}$",
                    "RRMSE$_{alt}$"
                ])
            else:
                bottom_parts.append("RRMSE$_{alt}$")

        lines.append(" & ".join(bottom_parts) + r"\\")

        return lines

    def _generate_rrmse_data_row(self,
                                 target: str,
                                 target_df: pd.DataFrame,
                                 include_matched: bool) -> str:
        """Generate one data row for RRMSE table."""
        parts = []

        # Target name
        parts.append(self.formatter.format_variable_name(target))

        first_row = target_df.iloc[0]

        # Predictors
        parts.append(self.formatter.format_predictor_str(
            first_row['predictors'],
            max_length=25
        ))

        # Model
        parts.append(self.formatter.format_model_name(first_row['model_type']))

        # Conventional performance
        conv_rrmse = first_row['conv_rrmse']
        conv_std = first_row.get('conv_rrmse_std', 0)
        parts.append(self.formatter.format_rrmse_with_std(conv_rrmse, conv_std))

        # Metrics for each fuel type
        for fuel_type in self.fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                if include_matched:
                    parts.extend(["---", "---"])
                else:
                    parts.append("---")
            else:
                fuel_data = fuel_row.iloc[0]

                if include_matched:
                    # Matched RRMSE
                    match_rrmse = fuel_data['matched_rrmse']
                    match_std = fuel_data.get('matched_rrmse_std', 0)
                    parts.append(self.formatter.format_rrmse_with_std(match_rrmse, match_std))

                # Alternative RRMSE
                alt_rrmse = fuel_data['alt_rrmse']
                alt_std = fuel_data.get('alt_rrmse_std', 0)

                # ✅ Highlight if significant increase from conventional
                performance_increase = alt_rrmse - conv_rrmse

                if performance_increase > 10:  # Increase > 10% RRMSE
                    alt_str = self.formatter.format_rrmse_with_std(
                        alt_rrmse, alt_std, use_highlight=True
                    )
                else:
                    alt_str = self.formatter.format_rrmse_with_std(alt_rrmse, alt_std)

                parts.append(alt_str)

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
    #
    # def save_tables(self,
    #                 ship_type: ShipType,
    #                 output_dir: str = '.',
    #                 targets: Optional[List[str]] = None,
    #                 max_predictors: int = 2,
    #                 prefix: str = None,
    #                 best_only: bool = True,
    #                 include_matched: bool = False,
    #                 landscape: bool = False) -> Dict[str, Path]:
    #     """
    #     Generate and save both R² and RRMSE tables.
    #
    #     Parameters
    #     ----------
    #     ship_type : ShipType
    #         Ship type for tables
    #     output_dir : str
    #         Directory to save files
    #     targets : List[str], optional
    #         Specific targets to include
    #     max_predictors : int
    #         Maximum number of predictors
    #     prefix : str, optional
    #         Filename prefix
    #     best_only : bool
    #         If True, show only best combination per target
    #     include_matched : bool, default=False
    #         If True, include matched dataset columns.
    #         **Recommended False** for cleaner cross-training tables.
    #     landscape : bool
    #         If True, use sidewaystable (recommended for >2 fuel types)
    #
    #     Returns
    #     -------
    #     dict
    #         Paths to saved files
    #     """
    #     output_path = Path(output_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)
    #
    #     if prefix is None:
    #         prefix = "cross_performance"
    #
    #     # Generate tables
    #     r2_table = self.generate_r2_table(
    #         ship_type=ship_type,
    #         targets=targets,
    #         max_predictors=max_predictors,
    #         best_only=best_only,
    #         include_matched=include_matched,
    #         landscape=landscape
    #     )
    #
    #     rrmse_table = self.generate_rrmse_table(
    #         ship_type=ship_type,
    #         targets=targets,
    #         max_predictors=max_predictors,
    #         best_only=best_only,
    #         include_matched=include_matched,
    #         landscape=landscape
    #     )
    #
    #     # Save files
    #     r2_path = output_path / f"{ship_type.name}_{prefix}_r2.tex"
    #     rrmse_path = output_path / f"{ship_type.name}_{prefix}_rrmse.tex"
    #
    #     r2_path.write_text(r2_table)
    #     rrmse_path.write_text(rrmse_table)
    #
    #     # Save preamble
    #     preamble_path = output_path / f"{ship_type.name}_{prefix}_preamble.tex"
    #     LaTeXPreambleGenerator.save_preamble(str(preamble_path))
    #
    #     if self.verbose:
    #         print(f"✅ Cross-performance tables saved:")
    #         print(f"   R² table:    {r2_path}")
    #         print(f"   RRMSE table: {rrmse_path}")
    #         print(f"   Preamble:    {preamble_path}")
    #
    #     return {
    #         'r2': r2_path,
    #         'rrmse': rrmse_path,
    #         'preamble': preamble_path
    #     }

    @staticmethod
    def print_latex_preamble():
        """Print required LaTeX preamble."""
        LaTeXPreambleGenerator.print_preamble()

    def generate_combined_table(self,
                                ship_type: 'ShipType',
                                targets: Optional[List[str]] = None,
                                caption: str = None,
                                label: str = None,
                                max_predictors: int = 2,
                                best_only: bool = True,
                                landscape: bool = False) -> str:
        """
        Generate combined R² and RRMSE table.

        **Shows both metrics together** under each fuel type for compact presentation.

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
            If True, use sidewaystable environment (recommended for >2 fuel types)

        Returns
        -------
        str
            LaTeX table code

        Notes
        -----
        **Table Structure:**
        Target | Predictors | Model | Conv (R²|RRMSE) | Alt1 (R²|RRMSE) | Alt2 (R²|RRMSE)

        This format is more compact than separate R² and RRMSE tables.
        """
        # Filter data
        df = self._filter_and_prepare_data(targets, max_predictors, best_only, 'alt_r2')

        if len(df) == 0:
            if self.verbose:
                print("⚠️ No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            fuel_list = ', '.join(self.fuel_types)
            eval_method = df['evaluation_method'].iloc[0] if 'evaluation_method' in df.columns else 'CV'

            caption = (
                f"Cross-training $R^{{2}}$ and RRMSE performance for {len(df['target'].unique())} target dimensions "
                f"across multiple alternative fuel types ({fuel_list}) for {ship_type.string} vessels. "
                f"Evaluation: {eval_method}. "
                f"Conv = trained on conventional ships, Alt = trained on alternative ships. "
                f"Showing best predictor combination per target (max {max_predictors} predictors)."
            )

        if label is None:
            label = f"tab:{ship_type.name}_cross_combined_multi"

        # Build table
        latex = self._build_combined_table_structure(df, caption, label, landscape)

        return latex

    def _build_combined_table_structure(self,
                                        df: pd.DataFrame,
                                        caption: str,
                                        label: str,
                                        landscape: bool) -> str:
        """Build combined R²/RRMSE table LaTeX code."""
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
        colspec = self._generate_combined_colspec()
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Headers
        header_lines = self._generate_combined_headers()
        lines.extend(header_lines)

        lines.append(r"\midrule")

        # Data rows
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            row_line = self._generate_combined_data_row(target, target_df)
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

    def _generate_combined_colspec(self) -> str:
        """
        Generate column specification for combined R²/RRMSE table.

        **Layout:**
        Target(10%) | Predictors(15%) | Model(10%) | Conv(12.5% R² + 10% RRMSE) |
        [Alt(12.5% R² + 10% RRMSE)] × n_fuel_types
        """
        n_fuel = len(self.fuel_types)

        base = [
            r"p{0.10\textwidth}",  # Target
            r"p{0.1\textwidth}",  # Predictors
            r"p{0.1\textwidth}",  # Model
        ]

        # Each dataset gets 2 columns: R² and RRMSE
        fuel_cols = [
            r"p{0.125\textwidth}",  # R²
            r">{\raggedright\arraybackslash}p{0.1\linewidth}",  # RRMSE
        ]

        # Conv + all alternative fuel types
        all_cols = base + (fuel_cols * (1 + n_fuel))

        return "".join(all_cols)

    def _generate_combined_headers(self) -> List[str]:
        """Generate header rows for combined R²/RRMSE table."""
        lines = []

        # Top header row with multicolumns for fuel types
        top_parts = [
            "",  # Target
            "",  # Predictors
            "",  # Model
            self.formatter.multicolumn("Conventional", 2, 'c'),
        ]

        # Add column for each fuel type
        for fuel_type in self.fuel_types:
            fuel_abbr = self.formatter.abbreviate_fuel_type(fuel_type)
            top_parts.append(self.formatter.multicolumn(fuel_abbr, 2, 'c'))

        lines.append(" & ".join(top_parts) + r"\\")

        # Bottom header row with metric names
        bottom_parts = [
            "Target",
            "Predictors",
            "Model",
            "$R^2_{conv}$",
            "RRMSE$_{conv}$",
        ]

        # Add subheaders for each fuel type
        for _ in self.fuel_types:
            bottom_parts.extend([
                "$R^2_{alt}$",
                "RRMSE$_{alt}$"
            ])

        lines.append(" & ".join(bottom_parts) + r"\\")

        return lines

    def _generate_combined_data_row(self,
                                    target: str,
                                    target_df: pd.DataFrame) -> str:
        """Generate one data row for combined R²/RRMSE table."""
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

        # Conventional performance (same for all fuel types)
        conv_r2 = first_row['conv_r2']
        conv_r2_std = first_row.get('conv_r2_std', 0)
        conv_rrmse = first_row['conv_rrmse']
        conv_rrmse_std = first_row.get('conv_rrmse_std', 0)

        parts.append(self.formatter.format_r2_with_std(conv_r2, conv_r2_std))
        parts.append(self.formatter.format_rrmse_with_std(conv_rrmse, conv_rrmse_std))

        # Metrics for each fuel type
        for fuel_type in self.fuel_types:
            fuel_row = target_df[target_df['fuel_type'] == fuel_type]

            if fuel_row.empty:
                parts.extend(["---", "---"])
            else:
                fuel_data = fuel_row.iloc[0]

                # Alternative R²
                alt_r2 = fuel_data['alt_r2']
                alt_r2_std = fuel_data.get('alt_r2_std', 0)
                alt_rrmse = fuel_data['alt_rrmse']
                alt_rrmse_std = fuel_data.get('alt_rrmse_std', 0)

                # ✅ Highlight if significant performance change
                performance_drop = conv_r2 - alt_r2
                performance_increase = alt_rrmse - conv_rrmse

                if performance_drop > 0.15:  # Drop > 15% R²
                    alt_r2_str = self.formatter.format_r2_with_std(
                        alt_r2, alt_r2_std, use_highlight=True
                    )
                else:
                    alt_r2_str = self.formatter.format_r2_with_std(alt_r2, alt_r2_std)

                if performance_increase > 10:  # Increase > 10% RRMSE
                    alt_rrmse_str = self.formatter.format_rrmse_with_std(
                        alt_rrmse, alt_rrmse_std, use_highlight=True
                    )
                else:
                    alt_rrmse_str = self.formatter.format_rrmse_with_std(alt_rrmse, alt_rrmse_std)

                parts.append(alt_r2_str)
                parts.append(alt_rrmse_str)

        return " & ".join(parts) + r"\\"

    def save_tables(self,
                    ship_type: 'ShipType',
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    max_predictors: int = 2,
                    prefix: str = None,
                    best_only: bool = True,
                    landscape: bool = False,
                    combined: bool = True) -> Dict[str, Path]:
        """
        Generate and save LaTeX tables.

        Parameters
        ----------
        ship_type : ShipType
            Ship type for tables
        output_dir : str
            Directory to save files
        targets : List[str], optional
            Specific targets to include
        max_predictors : int
            Maximum number of predictors
        prefix : str, optional
            Filename prefix
        best_only : bool
            If True, show only best combination per target
        landscape : bool
            If True, use sidewaystable (recommended for >2 fuel types)
        combined : bool, default=True
            If True, generates combined R²/RRMSE table (recommended).
            If False, generates separate R² and RRMSE tables.

        Returns
        -------
        dict
            Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = "cross_performance"

        saved_files = {}

        if combined:
            # Generate combined table
            combined_table = self.generate_combined_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only,
                landscape=landscape
            )

            combined_path = output_path / f"{ship_type.name}_{prefix}_combined.tex"
            combined_path.write_text(combined_table)
            saved_files['combined'] = combined_path

            if self.verbose:
                print(f"✅ Combined R²/RRMSE table saved: {combined_path}")

        else:
            # Generate separate tables
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

            r2_path = output_path / f"{ship_type.name}_{prefix}_r2.tex"
            rrmse_path = output_path / f"{ship_type.name}_{prefix}_rrmse.tex"

            r2_path.write_text(r2_table)
            rrmse_path.write_text(rrmse_table)

            saved_files['r2'] = r2_path
            saved_files['rrmse'] = rrmse_path

            if self.verbose:
                print(f"✅ Separate tables saved:")
                print(f"   R² table:    {r2_path}")
                print(f"   RRMSE table: {rrmse_path}")

        # Save preamble
        preamble_path = output_path / f"{ship_type.name}_{prefix}_preamble.tex"
        LaTeXPreambleGenerator.save_preamble(str(preamble_path))
        saved_files['preamble'] = preamble_path

        if self.verbose:
            print(f"   Preamble:    {preamble_path}")

        return saved_files