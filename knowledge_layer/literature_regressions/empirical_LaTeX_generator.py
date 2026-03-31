# empirical_LaTeX_generator.py
"""
Modified LaTeX Generator for Multi-Dataset Literature Formula Evaluation
=========================================================================

Generates publication-ready LaTeX tables showing literature formula performance
across multiple alternative fuel datasets, with matched dataset results displayed
as sub-rows under each fuel type.

Key Feature: Matched datasets are shown directly under their corresponding
alternative fuel type, making the comparison structure more intuitive.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType


class MultiDatasetLiteratureLatexGenerator:
    """
    Generate LaTeX tables from MultiDatasetLiteratureEvaluator results.

    **New Structure**: Shows matched dataset metrics as sub-rows under each
    alternative fuel type for clearer fuel-specific comparisons.

    Table Layout:
    -------------
    Target | Predictors | Model | Reference | Conv | FuelType1_Match | FuelType1_Alt | Gap | ...

    Where each fuel type gets TWO or THREE columns:
    - Match: Performance on matched conventional ships
    - Alt: Performance on alternative fuel ships
    - Gap: Difference (Alt - Match) [optional]
    """

    # Default variable name mapping for publication-ready formatting
    DEFAULT_VARIABLE_MAP = {
        'LOA (M)': '$L_{oa}$ (m)',
        'LBP (M)': '$L_{bp}$ (m)',
        'LPP (M)': '$L_{pp}$ (m)',
        'BEAM (M)': 'Beam (m)',
        'BEAM MLD (M)': 'Beam (m)',
        'DEPTH (M)': 'Depth (m)',
        'DRAFT (M)': 'Draught (m)',
        'DWT': 'DWT (t)',
        'GT': 'GT',
        'Disp': '$\\nabla$ (t)',
        'MCR (KW)': '$P_{MCR}$ (kW)',
        'AUX_POWER': '$P_{aux}$ (kW)',
        'BUNKER_CAPACITY': '$V_{bunker}$ (m$^3$)',
        'Vs': '$V_s$ (kn)',
        'C_B': '$C_B$',
        'LSW': 'LSW (t)',
        'LSW/Disp': 'LSW/$\\nabla$',
    }

    def __init__(self, multi_evaluator, verbose: bool = True,
                 variable_map: Optional[Dict[str, str]] = None):
        """
        Initialize table generator.

        Parameters
        ----------
        multi_evaluator : MultiDatasetLiteratureEvaluator
            Evaluator with comparison results
        verbose : bool
            Print progress messages
        variable_map : Dict[str, str], optional
            Custom variable name mapping for display.
            Keys are raw variable names, values are LaTeX-formatted names.
            If None, uses DEFAULT_VARIABLE_MAP.
            If provided, merges with defaults (custom takes precedence).
        """
        self.multi_evaluator = multi_evaluator
        self.verbose = verbose

        # Setup variable mapping
        self.variable_map = self.DEFAULT_VARIABLE_MAP.copy()
        if variable_map:
            self.variable_map.update(variable_map)

        # Validate input
        if multi_evaluator.comparison_results is None:
            raise ValueError(
                "MultiDatasetLiteratureEvaluator has no results. "
                "Run evaluate_all_datasets() first."
            )

        self.results_df = multi_evaluator.comparison_results.copy()

        # ✅ FIXED: Preserve order from original dictionary instead of sorting
        # Get fuel types in the order they appear in the original alternative_datasets dict
        if hasattr(multi_evaluator, 'alternative_datasets'):
            # Use the order from the original dictionary
            self.fuel_types = list(multi_evaluator.alternative_datasets.keys())
        else:
            # Fallback: maintain order of first appearance in results
            self.fuel_types = list(self.results_df['fuel_type'].unique())

        if self.verbose:
            print(f"✅ Initialized LaTeX generator")
            print(f"   Fuel types: {', '.join(self.fuel_types)}")
            print(f"   Formulas: {self.results_df['formula_name'].nunique()}")
            print(f"   Targets: {self.results_df['target'].nunique()}")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _escape_latex(self, text: str) -> str:
        """
        Escape special LaTeX characters.

        Note: This should NOT be applied to variable names that are already
        formatted with LaTeX commands (like $L_{oa}$).
        """
        if pd.isna(text):
            return "---"

        text = str(text)
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def _format_variable_name(self, variable_name: str) -> str:
        """
        Format variable name for display using mapping dictionary.

        If variable is in the map, returns the mapped value (which may contain
        LaTeX commands). Otherwise, returns the original name escaped for LaTeX.

        Parameters
        ----------
        variable_name : str
            Raw variable name (e.g., 'BEAM (M)')

        Returns
        -------
        str
            Formatted variable name (e.g., '$B$ (m)')
        """
        # Check if we have a mapping for this variable
        if variable_name in self.variable_map:
            # Return mapped value WITHOUT escaping (it may contain LaTeX)
            return self.variable_map[variable_name]
        else:
            # No mapping - escape for LaTeX
            return self._escape_latex(variable_name)

    def _format_predictor_list(self, predictors: List[str]) -> str:
        """
        Format list of predictors for display.

        Parameters
        ----------
        predictors : List[str]
            List of predictor names

        Returns
        -------
        str
            Formatted predictor string (comma-separated, mapped names)
        """
        if isinstance(predictors, str):
            predictors = [predictors]

        # Map each predictor
        mapped_preds = [self._format_variable_name(p) for p in predictors]

        # Join with comma and space
        return ", ".join(mapped_preds)

    def _clean_display_text(self, text: str) -> str:
        """
        Clean text for display by replacing underscores with spaces.

        This is used for titles, subtitles, and model type names.
        Does NOT escape LaTeX - that should be done separately if needed.
        """
        if pd.isna(text):
            return "---"
        return str(text).replace('_', ' ')

    def _abbreviate_fuel_type(self, fuel_type: str, max_length: int = 10) -> str:
        """Create abbreviated fuel type name for column headers."""
        if len(fuel_type) <= max_length:
            return fuel_type

        # Common abbreviations
        abbrev_map = {
            'Pure_alt': 'Pure alt',
            'dual_fuel': 'dual fuel',
            'Methanol': 'MeOH',
            'Ammonia': 'NH3',
            'Hydrogen': 'H2',
            'Battery': 'Batt',
        }

        if fuel_type in abbrev_map:
            return abbrev_map[fuel_type]

        # Fallback: truncate
        return fuel_type[:max_length - 1] + "."

    def _format_r2(self, r2: float) -> str:
        """
        Format R² value without conditional highlighting.

        Parameters
        ----------
        r2 : float
            R² value

        Returns
        -------
        str
            LaTeX formatted string (always 3 decimals, no color)
        """
        if pd.isna(r2):
            return "---"
        return f"{r2:.3f}"

    def _format_gap(self, gap: float) -> str:
        """
        Format gap value with sign and ALWAYS use \\rchange{}.

        Parameters
        ----------
        gap : float
            Gap value (R² gap)

        Returns
        -------
        str
            LaTeX formatted string with sign and \\rchange{}
        """
        if pd.isna(gap):
            return "---"

        # Always add explicit sign
        sign = "+" if gap >= 0 else ""
        gap_str = f"{sign}{gap:.3f}"

        # Always use \rchange{} for R² gaps
        return f"\\rchange{{{gap_str}}}"

    def _format_rrmse(self, rrmse: float) -> str:
        """
        Format RRMSE value - ALWAYS use \\rrmse{}.

        Parameters
        ----------
        rrmse : float
            RRMSE value

        Returns
        -------
        str
            LaTeX formatted string with \\rrmse{}
        """
        if pd.isna(rrmse):
            return "---"

        # Always use \rrmse{} for all RRMSE values
        return f"\\rrmse{{{rrmse:.1f}}}"

    def _get_formula_type_display(self, formula_type: str) -> str:
        """Get formula type for display - replaces underscores with backslash-underscore."""
        # Clean underscores for display, then escape for LaTeX
        cleaned = self._clean_display_text(formula_type)
        # For LaTeX, we want to show underscores as \_
        return cleaned.replace(' ', '\\_')

    # ========================================================================
    # R² TABLE GENERATION
    # ========================================================================

    def generate_r2_table(self,
                          ship_type: ShipType,
                          targets: Optional[List[str]] = None,
                          caption: str = None,
                          label: str = None,
                          show_r2_gaps: bool = True,
                          show_reference: bool = True,
                          landscape: bool = False,
                          sort_by: str = 'target') -> str:
        """
        Generate R² comparison table with matched results under each fuel type.

        **New Structure**: For each fuel type, shows:
        - Matched dataset R² (conventional ships matched to this fuel type)
        - Alternative dataset R² (actual alternative fuel ships)
        - Gap between them

        Parameters
        ----------
        ship_type : ShipType
            Ship type for table caption/label
        targets : List[str], optional
            Filter to specific targets
        caption : str, optional
            Custom table caption
        label : str, optional
            Custom table label
        show_r2_gaps : bool
            Include gap columns
        show_reference : bool
            Include reference column
        landscape : bool
            Use landscape orientation (recommended for >3 fuel types)
        sort_by : str
            Sort formulas by: 'target', 'formula_name', 'formula_type'

        Returns
        -------
        str
            LaTeX table code
        """

        # Filter data
        df = self.results_df.copy()
        if targets:
            df = df[df['target'].isin(targets)]

        if df.empty:
            return "% No data for specified targets\n"

        # Generate caption and label
        if caption is None:
            # Clean fuel type names (replace underscores with spaces)
            fuel_list = ', '.join([self._clean_display_text(ft) for ft in self.fuel_types])
            # Clean ship type name
            ship_name = self._clean_display_text(ship_type.name)
            caption = (f"$R^{{2}}_{{adj}}$ performance across multiple "
                       f"alternative fuel types ({fuel_list}) for "
                       f"{ship_name} vessels. "
                       f"Performance of literature formulas.")

        if label is None:
            label = f"tab_{ship_type.name}_r2_lit_multi"

        # Build table
        lines = []

        # Environment
        if landscape:
            lines.append("\\begin{landscape}")

        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\scriptsize")  # Smaller font for multi-column tables

        # Build column specification
        col_spec = self._build_r2_column_spec(show_reference, show_r2_gaps)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        # Header
        header_lines = self._build_r2_header(show_reference, show_r2_gaps)
        lines.extend(header_lines)

        # Data rows
        data_lines = self._build_r2_data_rows(df, show_reference, show_r2_gaps, sort_by)
        lines.extend(data_lines)

        # Close table
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")

        if landscape:
            lines.append("\\end{landscape}")

        return "\n".join(lines)

    def _build_r2_column_spec(self, show_reference: bool, show_gaps: bool) -> str:
        """
        Build column specification for R² table with FIXED widths.

        Structure:
        - Target (10%)
        - Predictors (15%)
        - Model (8%)
        - Reference (12%) [optional]
        - Conventional R² (5.5%)
        - For each fuel type:
          - Matched R² (5.5%)
          - Alternative R² (5.5%)
          - Gap (5.5%) [optional]
        """
        cols = []

        # Left columns
        cols.append("p{0.1\\textwidth}")  # Target
        cols.append("p{0.15\\textwidth}")  # Predictors
        cols.append("p{0.08\\textwidth}")  # Model

        if show_reference:
            cols.append("p{0.12\\textwidth}")  # Reference

        # Conventional R² - FIXED width
        cols.append("p{0.055\\textwidth}")

        # For each fuel type: Match, Alt, Gap (all fixed width)
        for _ in self.fuel_types:
            cols.append("p{0.055\\textwidth}")  # Match R²
            cols.append("p{0.055\\textwidth}")  # Alt R²
            if show_gaps:
                cols.append("p{0.055\\textwidth}")  # Gap

        return "".join(cols)

    def _build_r2_header(self, show_reference: bool, show_gaps: bool) -> List[str]:
        """
        Build table header with multi-column structure.

        Creates a two-row header:
        Row 1: Fuel type names spanning their columns
        Row 2: Specific column names (Match, Alt, Gap)
        """
        lines = []

        # Top rule
        lines.append("\\toprule")

        # === HEADER ROW 1: Fuel type names ===
        row1_parts = []

        # Empty cells for left columns
        n_left = 4 if show_reference else 3
        row1_parts.extend([""] * n_left)

        # Conventional (single column)
        row1_parts.append("")

        # Each fuel type spans its columns
        n_fuel_cols = 3 if show_gaps else 2
        for fuel_type in self.fuel_types:
            fuel_abbr = self._abbreviate_fuel_type(fuel_type)
            row1_parts.append(f"\\multicolumn{{{n_fuel_cols}}}{{c}}{{{fuel_abbr}}}")

        lines.append(" & ".join(row1_parts) + "\\\\")

        # === CMIDRULES: Separate fuel type columns ===
        cmidrule_lines = []

        # Start column for Conventional
        col_idx = n_left + 1

        # Conventional gets no cmidrule (single column)
        col_idx += 1

        # Each fuel type gets a cmidrule
        for _ in self.fuel_types:
            cmidrule_lines.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_fuel_cols - 1}}}")
            col_idx += n_fuel_cols

        lines.append("\n".join(cmidrule_lines))

        # === HEADER ROW 2: Column names ===
        row2_parts = []

        # Left column headers
        row2_parts.append("Target")
        row2_parts.append("Predictors")
        row2_parts.append("Model")

        if show_reference:
            row2_parts.append("Reference")

        # Conventional
        row2_parts.append("$R^2_{conv}$")

        # For each fuel type: Match, Alt, Gap
        for _ in self.fuel_types:
            row2_parts.append("$R^2_{match}$")
            row2_parts.append("$R^2_{alt}$")
            if show_gaps:
                row2_parts.append("$\\Delta_{fuel}$")

        lines.append(" & ".join(row2_parts) + "\\\\")
        lines.append("\\midrule")

        return lines

    def _build_r2_data_rows(self, df: pd.DataFrame,
                            show_reference: bool,
                            show_gaps: bool,
                            sort_by: str) -> List[str]:
        """
        Build data rows for R² table.

        Groups formulas by reference/source and adds cmidrules between sources.
        Uses multirow for reference column spanning all formulas from same source.
        """
        lines = []

        # Sort data - prioritize by reference, then by target or other criteria
        sort_cols = ['reference']
        if sort_by == 'target':
            sort_cols.append('target')
        elif sort_by == 'formula_name':
            sort_cols.append('formula_name')
        elif sort_by == 'formula_type':
            sort_cols.append('formula_type')

        # Add target as secondary sort for consistency within source
        if 'target' not in sort_cols:
            sort_cols.append('target')

        # Get unique formulas (one row per formula)
        df_copy = df.copy()
        df_copy['predictors_hashable'] = df_copy['predictors'].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        formula_cols = ['formula_name', 'target', 'predictors_hashable',
                        'formula_type', 'reference', 'conventional_r2']

        formulas_df = df_copy[formula_cols].drop_duplicates().sort_values(sort_cols)

        # Keep original predictors column for display
        formulas_df['predictors'] = formulas_df['predictors_hashable'].apply(
            lambda x: list(x) if isinstance(x, tuple) else x
        )

        prev_reference = None

        for idx, formula_row in formulas_df.iterrows():
            formula_name = formula_row['formula_name']
            target = formula_row['target']
            reference = formula_row['reference']

            # Check if we're starting a new reference group
            if prev_reference is not None and reference != prev_reference:
                # Add separator between references
                n_cols = 4 if show_reference else 3  # Left columns
                n_cols += 1  # Conv R²
                n_fuel_cols = 3 if show_gaps else 2
                n_cols += len(self.fuel_types) * n_fuel_cols

                lines.append(f"\\cmidrule{{1-{n_cols}}}")

            # Build row
            row_parts = []

            # Target - use mapping
            row_parts.append(self._format_variable_name(target))

            # Predictors (join if multiple) - use mapping
            predictors = formula_row['predictors']
            row_parts.append(self._format_predictor_list(predictors))

            # Model type - clean underscores
            row_parts.append(self._get_formula_type_display(formula_row['formula_type']))

            # Reference (with multirow handling)
            if show_reference:
                if prev_reference is None or reference != prev_reference:
                    # Starting a new reference group
                    same_ref_rows = formulas_df[
                        (formulas_df.index >= idx) &
                        (formulas_df['reference'] == reference)
                        ]
                    n_rows = len(same_ref_rows)

                    if n_rows > 1:
                        ref_text = f"\\multirow{{{n_rows}}}{{*}}{{{self._escape_latex(reference)}}}"
                    else:
                        ref_text = self._escape_latex(reference)

                    row_parts.append(ref_text)
                    prev_reference = reference
                else:
                    # Continuation of same reference - empty cell
                    row_parts.append("")

            # Conventional R²
            conv_r2 = formula_row['conventional_r2']
            row_parts.append(self._format_r2(conv_r2))

            # For each fuel type: get match, alt, gap
            for fuel_type in self.fuel_types:
                fuel_data = df[
                    (df['formula_name'] == formula_name) &
                    (df['fuel_type'] == fuel_type)
                    ]

                if fuel_data.empty:
                    # No data for this fuel type
                    row_parts.append("---")
                    row_parts.append("---")
                    if show_gaps:
                        row_parts.append("---")
                else:
                    fuel_row = fuel_data.iloc[0]

                    # Matched R²
                    match_r2 = fuel_row['matched_r2']
                    row_parts.append(self._format_r2(match_r2))

                    # Alternative R²
                    alt_r2 = fuel_row['alternative_r2']
                    row_parts.append(self._format_r2(alt_r2))

                    # Gap (if requested) - ALWAYS use \rchange{}
                    if show_gaps:
                        gap = fuel_row['gap_fuel_type']  # This is match - alt
                        # Invert sign: we want alt - match
                        gap_display = -gap if not pd.isna(gap) else gap
                        row_parts.append(self._format_gap(gap_display))

            lines.append(" & ".join(row_parts) + "\\\\")

        return lines

    # ========================================================================
    # RRMSE TABLE GENERATION
    # ========================================================================

    def generate_rrmse_table(self,
                             ship_type: ShipType,
                             targets: Optional[List[str]] = None,
                             caption: str = None,
                             label: str = None,
                             show_rrmse_gaps: bool = False,
                             show_reference: bool = True,
                             landscape: bool = False,
                             sort_by: str = 'target') -> str:
        """
        Generate RRMSE comparison table (similar structure to R² table).

        Parameters match generate_r2_table but for RRMSE metric.
        """

        # Filter data
        df = self.results_df.copy()
        if targets:
            df = df[df['target'].isin(targets)]

        if df.empty:
            return "% No data for specified targets\n"

        # Generate caption and label
        if caption is None:
            # Clean fuel type names
            fuel_list = ', '.join([self._clean_display_text(ft) for ft in self.fuel_types])
            ship_name = self._clean_display_text(ship_type.name)
            caption = (f"RRMSE (\\%) performance across multiple "
                       f"alternative fuel types ({fuel_list}) for "
                       f"{ship_name} vessels. "
                       f"Performance of literature formulas.")

        if label is None:
            label = f"tab_{ship_type.name}_rrmse_lit_multi"

        # Build table
        lines = []

        if landscape:
            lines.append("\\begin{landscape}")

        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\scriptsize")

        # Column spec with FIXED widths for RRMSE
        col_spec = self._build_rrmse_column_spec(show_reference, show_rrmse_gaps)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        # Header (modified for RRMSE)
        header_lines = self._build_rrmse_header(show_reference, show_rrmse_gaps)
        lines.extend(header_lines)

        # Data rows
        data_lines = self._build_rrmse_data_rows(df, show_reference, show_rrmse_gaps, sort_by)
        lines.extend(data_lines)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")

        if landscape:
            lines.append("\\end{landscape}")

        return "\n".join(lines)

    def _build_rrmse_column_spec(self, show_reference: bool, show_gaps: bool) -> str:
        """
        Build column specification for RRMSE table with FIXED widths.

        Uses p{0.1\textwidth} for RRMSE columns.
        """
        cols = []

        # Left columns (same as R²)
        cols.append("p{0.1\\textwidth}")  # Target
        cols.append("p{0.15\\textwidth}")  # Predictors
        cols.append("p{0.08\\textwidth}")  # Model

        if show_reference:
            cols.append("p{0.12\\textwidth}")  # Reference

        # Conventional RRMSE - FIXED width
        cols.append("p{0.1\\textwidth}")

        # For each fuel type: Match, Alt (no gap typically for RRMSE)
        for _ in self.fuel_types:
            cols.append("p{0.1\\textwidth}")  # Match RRMSE
            cols.append("p{0.1\\textwidth}")  # Alt RRMSE
            if show_gaps:
                cols.append("p{0.1\\textwidth}")  # Gap (if requested)

        return "".join(cols)

    def _build_rrmse_header(self, show_reference: bool, show_gaps: bool) -> List[str]:
        """Build header for RRMSE table (similar to R² but different column names)."""
        lines = []

        lines.append("\\toprule")

        # Row 1: Fuel type names
        row1_parts = []
        n_left = 4 if show_reference else 3
        row1_parts.extend([""] * n_left)
        row1_parts.append("")  # Conventional column

        n_fuel_cols = 3 if show_gaps else 2
        for fuel_type in self.fuel_types:
            fuel_abbr = self._abbreviate_fuel_type(fuel_type)
            row1_parts.append(f"\\multicolumn{{{n_fuel_cols}}}{{c}}{{{fuel_abbr}}}")

        lines.append(" & ".join(row1_parts) + "\\\\")

        # Cmidrules
        cmidrule_lines = []
        col_idx = n_left + 2  # Skip left cols and conventional
        for _ in self.fuel_types:
            cmidrule_lines.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_fuel_cols - 1}}}")
            col_idx += n_fuel_cols

        lines.append("\n".join(cmidrule_lines))

        # Row 2: Column names
        row2_parts = []
        row2_parts.append("Target")
        row2_parts.append("Predictors")
        row2_parts.append("Model")

        if show_reference:
            row2_parts.append("Reference")

        row2_parts.append("RRMSE$_{conv}$ (\\%)")

        for _ in self.fuel_types:
            row2_parts.append("RRMSE$_{match}$ (\\%)")
            row2_parts.append("RRMSE$_{alt}$ (\\%)")
            if show_gaps:
                row2_parts.append("$\\Delta_{fuel}$")

        lines.append(" & ".join(row2_parts) + " \\\\")
        lines.append("\\midrule")

        return lines

    def _build_rrmse_data_rows(self, df: pd.DataFrame,
                               show_reference: bool,
                               show_gaps: bool,
                               sort_by: str) -> List[str]:
        """
        Build data rows for RRMSE table - ALWAYS use \\rrmse{} formatting.

        Groups formulas by reference/source with cmidrules between sources.
        """
        lines = []

        # Sort data - prioritize by reference
        sort_cols = ['reference']
        if sort_by == 'target':
            sort_cols.append('target')
        elif sort_by == 'formula_name':
            sort_cols.append('formula_name')
        elif sort_by == 'formula_type':
            sort_cols.append('formula_type')

        if 'target' not in sort_cols:
            sort_cols.append('target')

        df_copy = df.copy()
        df_copy['predictors_hashable'] = df_copy['predictors'].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        formula_cols = ['formula_name', 'target', 'predictors_hashable',
                        'formula_type', 'reference', 'conventional_rrmse']

        formulas_df = df_copy[formula_cols].drop_duplicates().sort_values(sort_cols)

        formulas_df['predictors'] = formulas_df['predictors_hashable'].apply(
            lambda x: list(x) if isinstance(x, tuple) else x
        )

        prev_reference = None

        for idx, formula_row in formulas_df.iterrows():
            formula_name = formula_row['formula_name']
            target = formula_row['target']
            reference = formula_row['reference']

            # Separator between references
            if prev_reference is not None and reference != prev_reference:
                n_cols = 4 if show_reference else 3
                n_cols += 1
                n_fuel_cols = 3 if show_gaps else 2
                n_cols += len(self.fuel_types) * n_fuel_cols

                lines.append(f"\\cmidrule{{1-{n_cols}}}")

            # Build row
            row_parts = []

            # Target - use mapping
            row_parts.append(self._format_variable_name(target))

            # Predictors - use mapping
            predictors = formula_row['predictors']
            row_parts.append(self._format_predictor_list(predictors))

            # Model - clean underscores
            row_parts.append(self._get_formula_type_display(formula_row['formula_type']))

            # Reference (with multirow handling)
            if show_reference:
                if prev_reference is None or reference != prev_reference:
                    # Starting a new reference group
                    same_ref_rows = formulas_df[
                        (formulas_df.index >= idx) &
                        (formulas_df['reference'] == reference)
                        ]
                    n_rows = len(same_ref_rows)

                    if n_rows > 1:
                        ref_text = f"\\multirow{{{n_rows}}}{{*}}{{{self._escape_latex(reference)}}}"
                    else:
                        ref_text = self._escape_latex(reference)

                    row_parts.append(ref_text)
                    prev_reference = reference
                else:
                    # Continuation of same reference - empty cell
                    row_parts.append("")

            # Conventional RRMSE - ALWAYS use \rrmse{}
            conv_rrmse = formula_row['conventional_rrmse']
            row_parts.append(self._format_rrmse(conv_rrmse))

            # For each fuel type
            for fuel_type in self.fuel_types:
                fuel_data = df[
                    (df['formula_name'] == formula_name) &
                    (df['fuel_type'] == fuel_type)
                    ]

                if fuel_data.empty:
                    row_parts.append("---")
                    row_parts.append("---")
                    if show_gaps:
                        row_parts.append("---")
                else:
                    fuel_row = fuel_data.iloc[0]

                    # Matched RRMSE - ALWAYS use \rrmse{}
                    match_rrmse = fuel_row['matched_rrmse']
                    row_parts.append(self._format_rrmse(match_rrmse))

                    # Alternative RRMSE - ALWAYS use \rrmse{}
                    alt_rrmse = fuel_row['alternative_rrmse']
                    row_parts.append(self._format_rrmse(alt_rrmse))

                    # Gap (if requested)
                    if show_gaps:
                        gap = fuel_row['gap_fuel_type_rrmse']
                        if pd.isna(gap):
                            row_parts.append("---")
                        else:
                            row_parts.append(f"{gap:.1f}")

            lines.append(" & ".join(row_parts) + " \\\\")

        return lines


# ========================================================================
# CONVENIENCE METHODS
# ========================================================================

    def save_tables(self,
                    ship_type: ShipType,
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    prefix: str = None,
                    suffix:str=None,
                    show_r2_gaps: bool = True,
                    show_rrmse_gaps: bool = False,
                    show_reference: bool = True,
                    landscape: bool = False,
                    include_robustness: bool = False) -> Dict[str, str]:
        """
        Generate and save LaTeX tables to files.

        Parameters
        ----------
        ship_type : ShipType
            Ship type for tables
        output_dir : str
            Directory to save tables
        targets : List[str], optional
            Filter to specific targets
        prefix : str, optional
            Filename prefix (default: ship type name)
        show_r2_gaps : bool
            Include gap columns in R² table
        show_rrmse_gaps : bool
            Include gap columns in RRMSE table
        show_reference : bool
            Include reference column
        landscape : bool
            Use landscape orientation
        include_robustness : bool
            Also generate robustness classification table

        Returns
        -------
        dict
            Paths to saved files
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = ship_type.name.lower()

        saved_files = {}

        # R² table
        if self.verbose:
            print(f"\n📊 Generating R² table...")

        r2_table = self.generate_r2_table(
            ship_type=ship_type,
            targets=targets,
            show_r2_gaps=show_r2_gaps,
            show_reference=show_reference,
            landscape=landscape
        )

        r2_path = output_path / f"{prefix}_r2_multi_fuel.tex"
        with open(r2_path, 'w') as f:
            f.write(r2_table)

        saved_files['r2'] = str(r2_path)

        if self.verbose:
            print(f"   ✓ Saved: {r2_path}")

        # RRMSE table
        if self.verbose:
            print(f"\n📊 Generating RRMSE table...")

        rrmse_table = self.generate_rrmse_table(
            ship_type=ship_type,
            targets=targets,
            show_rrmse_gaps=show_rrmse_gaps,
            show_reference=show_reference,
            landscape=landscape
        )

        rrmse_path = output_path / f"{prefix}_rrmse_multi_fuel_{suffix}.tex"
        with open(rrmse_path, 'w') as f:
            f.write(rrmse_table)

        saved_files['rrmse'] = str(rrmse_path)

        if self.verbose:
            print(f"   ✓ Saved: {rrmse_path}")

        # Summary
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✅ LaTeX tables saved to: {output_dir}")
            print(f"{'=' * 70}")

        return saved_files

    def generate_preamble(self) -> str:
        """
        Generate LaTeX preamble with required packages.

        Returns
        -------
        str
            LaTeX preamble code
        """
        preamble = r"""
        % LaTeX Preamble for Multi-Dataset Literature Formula Tables
        % =============================================================
        % Required packages
        \usepackage{booktabs}      % Professional table rules
        \usepackage{multirow}      % Multi-row cells
        \usepackage{array}         % Extended column specifications
        \usepackage{pdflscape}     % Landscape pages
        \usepackage{xcolor}        % Color support
        % Custom command for highlighting changes (R² gaps)
        \newcommand{\rchange}[1]{\textcolor{red}{#1}}
        % Custom command for RRMSE values (tiered coloring)
        \newcommand{\rrmse}[1]{%
        \ifdim#1pt<5pt
        \textcolor{green!70!black}{#1}%
        \else\ifdim#1pt<10pt
        \textcolor{orange}{#1}%
        \else
        \textcolor{red}{#1}%
        \fi\fi
        }
        % Table formatting
        \renewcommand{\arraystretch}{1.2}  % Add vertical spacing
        """
        return preamble

