# parametric_formula_latex.py
"""
LaTeX Table Generator for Parametric Regression Formulas
=========================================================
Creates publication-ready tables showing discovered regression formulas.

Enhanced with:
- Predictor column
- Variable name formatting for targets/predictors
- Formula notation mapping for cleaner equations
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import re

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.display_functions.LaTeX_table_utils import LaTeXFormatter, LaTeXPreambleGenerator


class ParametricFormulaLatexGenerator:
    """
    Generate LaTeX tables showing best regression formulas from ParametricDomainAnalyzer.

    **Design Philosophy:**
    Uses ParametricDomainAnalyzer as the source of truth because:
    - Models are refitted with correct structure (via ModelDescriptor)
    - Test set metrics are available (not just training metrics)
    - Formulas match exactly what was evaluated in domain shift analysis

    Examples
    --------
    >>> # Run domain analysis
    >>> analyzer = ParametricDomainAnalyzer(
    ...     conventional_data=conv_df,
    ...     alternative_data=alt_df,
    ...     independent_variables=variables,
    ...     model_recommendations=selector.model_recommendations
    ... )
    >>> analyzer.analyze_all_combinations()
    >>>
    >>> # Generate formula table from analyzer results
    >>> generator = ParametricFormulaLatexGenerator(analyzer)
    >>> paths = generator.save_formula_table(
    ...     ship_type=ShipType.TANKER,
    ...     output_dir='tables/',
    ...     max_predictors=2
    ... )
    """

    # ========================================================================
    # NAMING MAPS
    # ========================================================================

    # Map for target/predictor column display
    VARIABLE_NAME_MAP = {
        'LOA (M)': r'$L_{OA}$ (m)',
        'LBP (M)': r'$L_{BP}$ (m)',
        'BEAM (M)': r'Beam (m)',
        'DRAFT (M)': 'Draught (m)',
        'DEPTH (M)': r'$D$ (m)',
        'DWT': r'DWT (t)',
        'GT': r'$GT$',
        'LSW': r'$\nabla_{LSW}$ (m$^3$)',
        'MCR (KW)': r'$P_{MCR}$ (kW)',
        'Vs': r'$v_s$ (kn)',
        'Main_Power': r'$P_{main}$ (kW)',
    }

    # Map for formula notation (inside equations)
    FORMULA_NOTATION_MAP = {
        'LOA (M)': r'L_{OA}',
        'LBP (M)': r'L_{BP}',
        'BEAM (M)': r'B',
        'DRAFT (M)': r'T',
        'DEPTH (M)': r'D',
        'DWT': r'DWT',
        'GT': r'GT',
        'LSW': r'\nabla_{LSW}',
        'MCR (KW)': r'P_{MCR}',
        'Vs': r'v_s',
        'Main_Power': r'P_{main}',
    }

    def __init__(self, domain_analyzer, verbose: bool = True):
        """
        Initialize formula table generator.

        Parameters
        ----------
        domain_analyzer : ParametricDomainAnalyzer
            Analyzer with results from analyze_all_combinations()
        verbose : bool
            Print progress messages
        """
        self.analyzer = domain_analyzer
        self.verbose = verbose
        self.formatter = LaTeXFormatter()

        # Validate input
        if not hasattr(domain_analyzer, 'results_df') or domain_analyzer.results_df is None:
            raise ValueError(
                "Domain analyzer has no results. "
                "Run analyzer.analyze_all_combinations() first."
            )

        if domain_analyzer.results_df.empty:
            raise ValueError(
                "Domain analyzer results are empty. "
                "Ensure analyze_all_combinations() produced valid results."
            )

        self.results_df = domain_analyzer.results_df.copy()
        self.target_dimensions = domain_analyzer.target_dimensions

        # Try to get selector for descriptor access
        self.selector = None
        if hasattr(domain_analyzer.foundation, 'selector'):
            self.selector = domain_analyzer.foundation.selector

        if self.verbose:
            print(f"✅ Initialized with {len(self.results_df)} results")
            print(f"   Targets: {len(self.target_dimensions)}")
            if self.selector and hasattr(self.selector, 'model_descriptors'):
                print(f"   Model descriptors available: {len(self.selector.model_descriptors)}")

    # ========================================================================
    # VARIABLE NAME FORMATTING
    # ========================================================================

    def _format_scientific_notation(self,
                                    formula: str,
                                    threshold: float = 1e-3,
                                    max_threshold: float = 1e4) -> str:
        """
        Convert very small or large numbers in formula to LaTeX scientific notation.

        Parameters
        ----------
        formula : str
            Formula string with numeric coefficients
        threshold : float, default=1e-3
            Numbers with |value| < threshold will be converted to scientific notation
        max_threshold : float, default=1e4
            Numbers with |value| > max_threshold will be converted to scientific notation

        Returns
        -------
        str
            Formula with scientific notation applied

        Examples
        --------
        Input:  "T = 1.5 + 0.000023 * \\Delta_{DWT}"
        Output: "T = 1.5 + 2.3 \\times 10^{-5} * \\Delta_{DWT}"

        Concept
        -------
        This method uses regex to find all numeric values in the formula string,
        then checks if they're outside the "normal" range [threshold, max_threshold].
        Small numbers (like 0.000023) and large numbers (like 45000) are converted
        to LaTeX scientific notation using \\times 10^{exponent} format.

        The conversion preserves:
        - Sign (negative numbers)
        - Position in formula
        - Surrounding operators and variables
        """
        # Pattern to match floating point numbers (including negative, scientific notation)
        # Matches: -1.23, 0.456, 1.23e-5, -4.56e+3, etc.
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

        def replace_number(match):
            """Helper to replace individual numbers."""
            num_str = match.group(0)

            try:
                value = float(num_str)
                abs_value = abs(value)

                # Check if needs scientific notation
                if abs_value == 0:
                    return num_str  # Keep zeros as-is

                if abs_value < threshold or abs_value > max_threshold:
                    # Convert to scientific notation
                    exponent = int(np.floor(np.log10(abs_value)))
                    mantissa = value / (10 ** exponent)

                    # Format mantissa (typically 2-3 significant figures)
                    if abs(mantissa - round(mantissa, 1)) < 1e-10:
                        mantissa_str = f"{mantissa:.1f}"
                    else:
                        mantissa_str = f"{mantissa:.2f}"

                    # Build LaTeX scientific notation
                    return f"{mantissa_str} \\times 10^{{{exponent}}}"
                else:
                    # Keep as-is if in normal range
                    return num_str

            except (ValueError, OverflowError):
                # If conversion fails, keep original
                return num_str

        # Apply replacement to all numbers in formula
        result = re.sub(number_pattern, replace_number, formula)

        return result

    def format_variable_for_display(self, var_name: str) -> str:
        """
        Format variable name for target/predictor columns.

        Parameters
        ----------
        var_name : str
            Raw variable name (e.g., 'LOA (M)')

        Returns
        -------
        str
            Formatted LaTeX (e.g., r'$L_{oa}$ (m)')
        """
        return self.VARIABLE_NAME_MAP.get(var_name, var_name)

    def format_variable_for_formula(self, var_name: str) -> str:
        """
        Format variable name for use inside formulas.

        Parameters
        ----------
        var_name : str
            Raw variable name (e.g., 'DRAFT (M)')

        Returns
        -------
        str
            Formula notation (e.g., r'T')
        """
        return self.FORMULA_NOTATION_MAP.get(var_name, var_name)

    def format_predictors_list(self, predictors: Tuple[str, ...]) -> str:
        """
        Format predictor combination for display.

        Parameters
        ----------
        predictors : tuple of str
            Predictor names

        Returns
        -------
        str
            Formatted string like "$L_{bp}$, $T$"
        """
        formatted = [self.format_variable_for_display(p) for p in predictors]
        return ", ".join(formatted)

    # ========================================================================
    # MAIN TABLE GENERATION
    # ========================================================================

    def generate_formula_table(self,
                               ship_type: ShipType,
                               targets: Optional[List[str]] = None,
                               max_predictors: int = 2,
                               min_r2: Optional[float] = None,
                               max_rrmse: Optional[float] = None,
                               use_test_metrics: bool = True,
                               caption: str = None,
                               label: str = None,
                               landscape: bool = False,
                               show_metrics: bool = True,
                               show_predictors: bool = True,
                               formula_notation: str = 'latex',
                               sci_notation_threshold: float = 1e-3,
                               sci_notation_max: float = 1e4) -> str:
        """
        Generate LaTeX table with best formulas for each target.

        Parameters
        ----------
        ship_type : ShipType
            Ship type for caption/label
        targets : List[str], optional
            Specific targets to include. If None, uses all targets.
        max_predictors : int, default=2
            Maximum number of predictors
        min_r2 : float, optional
            Minimum R² threshold (on test set if use_test_metrics=True)
        max_rrmse : float, optional
            Maximum RRMSE threshold (on test set if use_test_metrics=True)
        use_test_metrics : bool, default=True
            If True, show test set metrics (recommended).
            If False, show training metrics.
        caption : str, optional
            Custom caption
        label : str, optional
            Custom label
        landscape : bool, default=False
            Use sidewaystable for wide formulas
        show_metrics : bool, default=True
            Include performance metrics columns
        show_predictors : bool, default=True
            Include predictor column
        formula_notation : str, default='latex'
            Formula notation: 'latex', 'math', or 'python'
        sci_notation_threshold : float, default=1e-3
            Convert numbers with |value| < threshold to scientific notation
        sci_notation_max : float, default=1e4
            Convert numbers with |value| > max_threshold to scientific notation

        Returns
        -------
        str
            LaTeX table code
        """
        # Filter and prepare data
        df = self._prepare_formula_data(
            targets=targets,
            max_predictors=max_predictors,
            min_r2=min_r2,
            max_rrmse=max_rrmse,
            use_test_metrics=use_test_metrics,
            formula_notation=formula_notation,
            sci_notation_threshold=sci_notation_threshold,
            sci_notation_max=sci_notation_max
        )

        if df.empty:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            caption = self._generate_default_caption(
                ship_type, len(df), max_predictors, min_r2, max_rrmse, use_test_metrics
            )

        if label is None:
            label = f"tab_{ship_type.name}_formulas"

        # Build table
        latex = self._build_formula_table_structure(
            df=df,
            caption=caption,
            label=label,
            landscape=landscape,
            show_metrics=show_metrics,
            show_predictors=show_predictors,
            formula_notation=formula_notation,
            use_test_metrics=use_test_metrics
        )

        return latex

    def _prepare_formula_data(self,
                              targets: Optional[List[str]],
                              max_predictors: int,
                              min_r2: Optional[float],
                              max_rrmse: Optional[float],
                              use_test_metrics: bool,
                              formula_notation: str = 'latex',
                              sci_notation_threshold: float = 1e-3,
                              sci_notation_max: float = 1e4) -> pd.DataFrame:
        """
        Prepare data with formulas and metrics for table.

        Parameters
        ----------
        sci_notation_threshold : float, default=1e-3
            Numbers with |value| < threshold converted to scientific notation
        sci_notation_max : float, default=1e4
            Numbers with |value| > threshold converted to scientific notation

        Returns
        -------
        pd.DataFrame
            Columns: target, predictors, predictor_str, model_type,
                    formula, r2, rrmse, n_samples
        """
        # Use specified targets or all targets
        targets_to_use = targets if targets is not None else self.target_dimensions

        # Filter results
        df = self.results_df.copy()
        df = df[df['target'].isin(targets_to_use)]
        df = df[df['n_predictors'] <= max_predictors]

        if df.empty:
            return df

        # Get best combination for each target
        metric_suffix = 'test' if use_test_metrics else 'train'
        r2_col = f'{metric_suffix}_r2'
        rrmse_col = f'{metric_suffix}_rrmse'

        # Apply thresholds
        if min_r2 is not None:
            df = df[df[r2_col] >= min_r2]
        if max_rrmse is not None:
            df = df[df[rrmse_col] <= max_rrmse]

        if df.empty:
            return df

        # Get best for each target (by test RRMSE)
        best_df = (df.sort_values(rrmse_col)
                   .groupby('target')
                   .first()
                   .reset_index())

        # Extract formulas
        formula_data = []
        for _, row in best_df.iterrows():
            target = row['target']
            predictors = tuple(row['predictors'])
            model_type = row['model_type']

            # Get formula (NOW WITH SCIENTIFIC NOTATION)
            formula = self._extract_formula(
                predictors,
                target,
                model_type,
                formula_notation,
                sci_notation_threshold=sci_notation_threshold,
                sci_notation_max=sci_notation_max
            )

            if formula is None:
                if self.verbose:
                    print(f"  ⚠️  No formula for {target}: {' + '.join(predictors)} ({model_type})")
                continue

            formula_data.append({
                'target': target,
                'predictors': predictors,
                'predictor_str': ' + '.join(predictors),
                'model_type': model_type,
                'formula': formula,
                'r2': row[r2_col],
                'rrmse': row[rrmse_col],
                'n_samples': row[f'{metric_suffix}_n']
            })

        return pd.DataFrame(formula_data)

    def _extract_formula(self,
                         predictors: Tuple[str, ...],
                         target: str,
                         model_type: str,
                         notation: str = 'latex',
                         sci_notation_threshold: float = 1e-3,
                         sci_notation_max: float = 1e4) -> Optional[str]:
        """
        Extract formula from ModelDescriptor and apply notation mapping.

        Uses the analyzer's foundation to access the selector's descriptors,
        then replaces variable names with formula notation.

        Parameters
        ----------
        sci_notation_threshold : float, default=1e-3
            Numbers with |value| < threshold converted to scientific notation
        sci_notation_max : float, default=1e4
            Numbers with |value| > threshold converted to scientific notation
        """
        if self.selector is None:
            if self.verbose:
                print(f"  ⚠️  No selector available for descriptor access")
            return None

        if not hasattr(self.selector, 'model_descriptors'):
            if self.verbose:
                print(f"  ⚠️  Selector has no model_descriptors")
            return None

        key = (predictors, target)

        if key not in self.selector.model_descriptors:
            if self.verbose:
                print(f"  ⚠️  No descriptor for {predictors} → {target}")
            return None

        descriptors_for_key = self.selector.model_descriptors[key]

        if model_type not in descriptors_for_key:
            if self.verbose:
                print(
                    f"  ⚠️  No descriptor for model type {model_type} "
                    f"(available: {list(descriptors_for_key.keys())})"
                )
            return None

        descriptor = descriptors_for_key[model_type]

        # Get formula with original variable names
        try:
            formula = descriptor.get_formula_string(
                include_coefficients=True,
                notation=notation,
                only_active=True,  # For Lasso: only show non-zero terms
                sci_threshold=sci_notation_threshold,
                sci_max=sci_notation_max
            )

            # Apply formula notation mapping
            formula = self._apply_formula_notation(formula, target, predictors)



            return formula

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Failed to extract formula: {e}")
            return None

    def _apply_formula_notation(self,
                                formula: str,
                                target: str,
                                predictors: Tuple[str, ...]) -> str:
        """
        Replace variable names in formula with formula notation.

        Parameters
        ----------
        formula : str
            Formula string with original variable names
        target : str
            Target variable name
        predictors : tuple of str
            Predictor variable names

        Returns
        -------
        str
            Formula with mapped notation

        Examples
        --------
        Input:  "DRAFT (M) = 1.5 + 0.001 * DWT"
        Output: "T = 1.5 + 0.001 * \\Delta_{DWT}"
        """
        # Start with formula
        result = formula

        # Replace target (left side of equation)
        if target in self.FORMULA_NOTATION_MAP:
            result = result.replace(target, self.FORMULA_NOTATION_MAP[target])

        # Replace predictors (right side)
        # Sort by length (descending) to avoid partial replacements
        # e.g., replace "LBP (M)" before "LBP"
        all_vars = list(predictors) + [target]
        sorted_vars = sorted(
            [v for v in all_vars if v in self.FORMULA_NOTATION_MAP],
            key=len,
            reverse=True
        )

        for var in sorted_vars:
            if var in self.FORMULA_NOTATION_MAP:
                result = result.replace(var, self.FORMULA_NOTATION_MAP[var])

        return result

    # ========================================================================
    # TABLE STRUCTURE BUILDING
    # ========================================================================

    def _build_formula_table_structure(self,
                                       df: pd.DataFrame,
                                       caption: str,
                                       label: str,
                                       landscape: bool,
                                       show_metrics: bool,
                                       show_predictors: bool,
                                       formula_notation: str,
                                       use_test_metrics: bool) -> str:
        """Build the complete LaTeX table structure."""
        lines = []

        # Table environment
        if landscape:
            lines.append(r"\begin{sidewaystable}[p]")
        else:
            lines.append(r"\begin{table}[ht]")

        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\footnotesize")

        # Column specification
        colspec = self._generate_colspec(show_metrics, show_predictors)
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Header
        header_line = self._generate_header(show_metrics, show_predictors, use_test_metrics)
        lines.append(header_line)
        lines.append(r"\midrule")

        # Data rows
        for _, row in df.iterrows():
            row_line = self._generate_data_row(
                row, show_metrics, show_predictors, formula_notation
            )
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

    def _generate_colspec(self, show_metrics: bool, show_predictors: bool) -> str:
        """
        Generate column specification.

        Layout options:
        - With predictors + metrics: Target (10%) | Predictors (12%) | Model (10%) | Formula (40%) | R² (8%) | RRMSE (8%)
        - With predictors only: Target (10%) | Predictors (12%) | Model (10%) | Formula (58%)
        - Without predictors + metrics: Target (10%) | Model (10%) | Formula (50%) | R² (8%) | RRMSE (8%)
        - Without predictors: Target (10%) | Model (12%) | Formula (68%)
        """
        if show_predictors and show_metrics:
            return (
                r"p{0.10\textwidth}"  # Target
                r"p{0.12\textwidth}"  # Predictors
                r"p{0.10\textwidth}"  # Model
                r"p{0.40\textwidth}"  # Formula
                r"p{0.08\textwidth}"  # R²
                r"p{0.08\textwidth}"  # RRMSE
            )
        elif show_predictors and not show_metrics:
            return (
                r"p{0.10\textwidth}"  # Target
                r"p{0.12\textwidth}"  # Predictors
                r"p{0.10\textwidth}"  # Model
                r"p{0.58\textwidth}"  # Formula (wider)
            )
        elif not show_predictors and show_metrics:
            return (
                r"p{0.10\textwidth}"  # Target
                r"p{0.10\textwidth}"  # Model
                r"p{0.50\textwidth}"  # Formula
                r"p{0.08\textwidth}"  # R²
                r"p{0.08\textwidth}"  # RRMSE
            )
        else:  # not show_predictors and not show_metrics
            return (
                r"p{0.10\textwidth}"  # Target
                r"p{0.12\textwidth}"  # Model
                r"p{0.68\textwidth}"  # Formula (widest)
            )

    def _generate_header(self,
                         show_metrics: bool,
                         show_predictors: bool,
                         use_test_metrics: bool) -> str:
        """Generate header row."""
        parts = ["Target"]

        if show_predictors:
            parts.append("Predictors")

        parts.append("Model")
        parts.append("Formula")

        if show_metrics:
            metric_label = "test" if use_test_metrics else "train"
            parts.extend([
                f"$R^2_{{{metric_label}}}$",
                f"RRMSE$_{{{metric_label}}}$ (\\%)"
            ])

        return " & ".join(parts) + r"\\"

    def _generate_data_row(self,
                           row: pd.Series,
                           show_metrics: bool,
                           show_predictors: bool,
                           formula_notation: str) -> str:
        """Generate one data row."""
        parts = []

        # Target - format with variable name map
        target_formatted = self.format_variable_for_display(row['target'])
        parts.append(target_formatted)

        # Predictors - format with variable name map
        if show_predictors:
            predictors_formatted = self.format_predictors_list(row['predictors'])
            parts.append(predictors_formatted)

        # Model type
        model_formatted = self.formatter.format_model_name(row['model_type'])
        parts.append(model_formatted)

        # Formula - already has notation applied in _extract_formula
        formula = row['formula']
        if formula_notation == 'latex':
            # Ensure it's in math mode
            if not formula.startswith('$'):
                formula = f"${formula}$"
        parts.append(formula)

        # Metrics
        if show_metrics:
            parts.append(self.formatter.format_r2_value(row['r2']))
            parts.append(self.formatter.format_rrmse_value(row['rrmse']))

        return " & ".join(parts) + r"\\"

    # ========================================================================
    # CAPTION GENERATION
    # ========================================================================

    def _generate_default_caption(self,
                                  ship_type: ShipType,
                                  n_targets: int,
                                  max_predictors: int,
                                  min_r2: Optional[float],
                                  max_rrmse: Optional[float],
                                  use_test_metrics: bool) -> str:
        """Generate default caption."""
        metric_set = "test set" if use_test_metrics else "training set"

        caption_parts = [
            f"Best regression formulas for {n_targets} target dimensions",
            f"for {ship_type.name} vessels",
            f"(performance on {metric_set})"
        ]

        filters = []
        if max_predictors:
            filters.append(f"max {max_predictors} predictors")
        if min_r2:
            filters.append(f"$R^2 \\geq {min_r2:.2f}$")
        if max_rrmse:
            filters.append(f"RRMSE $\\leq {max_rrmse:.1f}\\%$")

        if filters:
            caption_parts.append(f"({', '.join(filters)})")

        caption = " ".join(caption_parts) + "."
        return caption

    # ========================================================================
    # SAVE METHODS
    # ========================================================================

    def save_formula_table(self,
                           ship_type: ShipType,
                           output_dir: str = '.',
                           targets: Optional[List[str]] = None,
                           max_predictors: int = 2,
                           min_r2: Optional[float] = None,
                           max_rrmse: Optional[float] = None,
                           prefix: str = None,
                           landscape: bool = False,
                           show_metrics: bool = True,
                           show_predictors: bool = True,
                           use_test_metrics: bool = True,
                           sci_notation_threshold: float = 1e-3,
                           sci_notation_max: float = 1e4) -> Dict[str, Path]:
        """
        Generate and save formula table.

        Parameters
        ----------
        ship_type : ShipType
            Ship type for naming
        output_dir : str
            Output directory
        targets : List[str], optional
            Specific targets to include
        max_predictors : int
            Maximum number of predictors
        min_r2 : float, optional
            Minimum R² threshold
        max_rrmse : float, optional
            Maximum RRMSE threshold
        prefix : str, optional
            Filename prefix
        landscape : bool
            Use landscape orientation
        show_metrics : bool
            Include performance metrics
        show_predictors : bool
            Include predictor column
        use_test_metrics : bool
            Use test set metrics (recommended)
        sci_notation_threshold : float, default=1e-3
            Convert numbers with |value| < threshold to scientific notation
        sci_notation_max : float, default=1e4
            Convert numbers with |value| > max_threshold to scientific notation

        Returns
        -------
        dict
            Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = "formulas"

        # Generate table
        latex_table = self.generate_formula_table(
            ship_type=ship_type,
            targets=targets,
            max_predictors=max_predictors,
            min_r2=min_r2,
            max_rrmse=max_rrmse,
            use_test_metrics=use_test_metrics,
            landscape=landscape,
            show_metrics=show_metrics,
            show_predictors=show_predictors,
            formula_notation='latex',
            sci_notation_threshold=sci_notation_threshold,
            sci_notation_max=sci_notation_max
        )

        if not latex_table:
            if self.verbose:
                print("⚠️  No table generated (no data matching filters)")
            return {}

        # Save table
        table_path = output_path / f"{ship_type.name}_{prefix}.tex"
        table_path.write_text(latex_table)

        # Save preamble
        preamble_path = output_path / f"{ship_type.name}_{prefix}_preamble.tex"
        LaTeXPreambleGenerator.save_preamble(str(preamble_path))

        if self.verbose:
            print(f"✅ Formula table saved:")
            print(f"   Table:    {table_path}")
            print(f"   Preamble: {preamble_path}")

        return {
            'table': table_path,
            'preamble': preamble_path
        }

    @staticmethod
    def print_latex_preamble():
        """Print required LaTeX preamble."""
        LaTeXPreambleGenerator.print_preamble()