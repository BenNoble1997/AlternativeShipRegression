# latex_formatting_utils.py
"""
LaTeX Formatting Utilities
===========================
Provides shared formatting utilities for LaTeX table generation.

Uses composition pattern - table generators create an instance of this
utility class and delegate formatting tasks to it.

**Design Philosophy**: Pure utility class with no state dependencies.
All methods are self-contained and can be called independently.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import re


class LaTeXFormatter:
    """
    Utility class for LaTeX table formatting operations.

    **Usage Pattern (Composition)**:
```python
    class MyTableGenerator:
        def __init__(self):
            self.formatter = LaTeXFormatter()

        def generate_table(self):
            escaped_text = self.formatter.escape_latex(some_text)
            formatted_r2 = self.formatter.format_r2_value(0.856)
```

    **Concept**: This class handles the "how" of formatting (escaping,
    abbreviating, color-coding) while table generators handle the "what"
    (which data to display, table structure).
    """

    # Standard thresholds - can be overridden via method parameters
    DEFAULT_RRMSE_GOOD = 5.0
    DEFAULT_RRMSE_WARNING = 10.0
    DEFAULT_R2_SIGNIFICANT = 0.05
    DEFAULT_GAP_MINOR = 10.0
    DEFAULT_GAP_MAJOR = 20.0

    # Variable name formatting dictionary
    VARIABLE_DISPLAY_NAMES = {
                             # Ship dimensions
                             'BEAM (M)': 'Beam (m)',
                             'DRAFT (M)': 'Draft (m)',
                             'LOA (M)': 'LOA (m)',
                             'LBP (M)': 'LBP (m)',
                             'DEPTH (M)': 'Depth (m)',
                             'DWT': 'DWT',

                             # Power and machinery
                             'MCR (KW)': 'MCR (kW)',
                             'AUX_POWER': 'Aux Power (kW)',

                             # Capacities
                             'BUNKER_CAPACITY': 'Bunker Capacity (m³)',}

    # ========================================================================
    # CORE LATEX UTILITIES
    # ========================================================================

    @staticmethod
    def escape_latex(text: str) -> str:
        """
        Escape special LaTeX characters.

        **Concept**: LaTeX treats certain characters as special commands
        (%, $, &, etc.). This method makes text safe for LaTeX display.

        Parameters
        ----------
        text : str
            Text to escape

        Returns
        -------
        str
            LaTeX-safe text

        'R² $>$ 90\\%'
        """
        if not isinstance(text, str):
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
            '^': r'\^{}'
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    # ========================================================================
    # METRIC FORMATTING
    # ========================================================================

    @staticmethod
    def format_r2_value(value: float, precision: int = 3) -> str:
        """
        Format R² value with specified precision.

        **Concept**: R² values are typically shown with 3 decimal places
        to provide sufficient precision for comparison.

        Parameters
        ----------
        value : float
            R² value (0 to 1)
        precision : int
            Number of decimal places

        Returns
        -------
        str
            Formatted value or '---' for NaN


        '---'
        """
        if np.isnan(value):
            return "---"
        return f"{value:.{precision}f}"

    def format_rrmse_value(self,
                           value: float,
                           use_color: bool = True,
                           precision: int = 2,
                           good_threshold: Optional[float] = None,
                           warning_threshold: Optional[float] = None) -> str:
        """
        Format RRMSE value with conditional color highlighting.

        **Concept**: RRMSE values are color-coded to quickly identify
        model quality using the \\rrmse{} LaTeX command:
        - Green: < good_threshold (excellent)
        - Orange: good_threshold to warning_threshold (acceptable)
        - Red: > warning_threshold (poor)

        Parameters
        ----------
        value : float
            RRMSE value (percentage)
        use_color : bool
            Whether to apply color highlighting
        precision : int
            Number of decimal places
        good_threshold : float, optional
            Threshold for "good" performance (default: 5.0)
        warning_threshold : float, optional
            Threshold for "warning" performance (default: 10.0)

        Returns
        -------
        str
            Formatted value with optional LaTeX color command

        Examples

        '\\rrmse{12.80}'
        """
        if np.isnan(value):
            return "---"

        if use_color:
            return f"\\rrmse{{{value:.{precision}f}}}"
        else:
            return f"{value:.{precision}f}"

    def format_r2_gap(self,
                      gap: float,
                      use_highlight: bool = True,
                      threshold: Optional[float] = None,
                      precision: int = 3,
                      invert_sign: bool = False) -> str:
        """
        Format R² gap value with optional highlighting.

        **Concept**: Gap values show performance change between datasets.
        The \\rchange{} command highlights significant changes:
        - Positive (green): improvement
        - Negative (red): degradation

        Parameters
        ----------
        gap : float
            R² gap value (can be positive or negative)
        use_highlight : bool
            Whether to apply color highlighting for significant gaps
        threshold : float, optional
            Threshold for significance (default: 0.05)
        precision : int
            Number of decimal places
        invert_sign : bool
            If True, flip sign before display (useful when gap calculation
            convention differs from display convention)

        Returns
        -------
        str
            Formatted gap with sign and optional color

        Examples

        '+0.012'
        """
        if np.isnan(gap):
            return "---"

        if threshold is None:
            threshold = self.DEFAULT_R2_SIGNIFICANT

        if invert_sign:
            gap = -gap

        sign = "+" if gap >= 0 else ""
        formatted = f"{sign}{gap:.{precision}f}"

        if use_highlight and abs(gap) >= threshold:
            return f"\\rchange{{{formatted}}}"
        else:
            return formatted

    def format_rrmse_gap(self,
                         gap: float,
                         use_highlight: bool = True,
                         threshold_minor: Optional[float] = None,
                         threshold_major: Optional[float] = None,
                         precision: int = 1) -> str:
        """
        Format RRMSE gap value with tiered highlighting.

        **Concept**: RRMSE gaps show absolute error increase. The \\fuelgap{}
        command uses tiered coloring based on severity:
        - Green: < threshold_minor (minor degradation)
        - Orange: threshold_minor to threshold_major (moderate)
        - Red: > threshold_major (severe)

        Parameters
        ----------
        gap : float
            RRMSE gap value (percentage points)
        use_highlight : bool
            Whether to apply color highlighting
        threshold_minor : float, optional
            Threshold for minor gap (default: 10.0)
        threshold_major : float, optional
            Threshold for major gap (default: 20.0)
        precision : int
            Number of decimal places

        Returns
        -------
        str
            Formatted gap with sign and tiered color

        Examples

        """
        if np.isnan(gap):
            return "---"

        if threshold_minor is None:
            threshold_minor = self.DEFAULT_GAP_MINOR
        if threshold_major is None:
            threshold_major = self.DEFAULT_GAP_MAJOR

        sign = "+" if gap >= 0 else ""
        abs_gap = abs(gap)

        if use_highlight and abs_gap >= threshold_minor:
            # Use fuelgap command which implements tiered coloring
            if gap > 0:
                return f"\\fuelgap{{{abs_gap:.{precision}f}}}"
            else:
                return f"-\\fuelgap{{{abs_gap:.{precision}f}}}"
        else:
            return f"{sign}{gap:.{precision}f}"

    # ========================================================================
    # TEXT ABBREVIATION
    # ========================================================================

    def abbreviate_fuel_type(self,
                             fuel_type: str,
                             max_length: int = 10,
                             custom_abbrev: Optional[Dict[str, str]] = None) -> str:
        """
        Abbreviate fuel type name for table headers.

        **Concept**: Fuel type names can be long (e.g., "pure_alt_fuel"),
        making tables wide. We use standard abbreviations while maintaining
        clarity.

        Parameters
        ----------
        fuel_type : str
            Full fuel type name
        max_length : int
            Maximum length for abbreviation
        custom_abbrev : Dict[str, str], optional
            Custom abbreviation mappings (overrides defaults)

        Returns
        -------
        str
            Abbreviated, LaTeX-escaped fuel type name

        """
        if len(fuel_type) <= max_length:
            return self.escape_latex(fuel_type)

        # Standard abbreviations
        default_abbrev = {
            'Pure_alt': 'Pure',
            'dual_fuel': 'Dual',
            'pure_alt_fuel': 'Pure',
            'alternative': 'Alt',
            'methanol': 'MeOH',
            'ammonia': 'NH$_3$',  # LaTeX subscript
            'hydrogen': 'H$_2$',
            'lng': 'LNG',
            'lpg': 'LPG'
        }

        # Merge with custom abbreviations if provided
        abbrev_map = default_abbrev.copy()
        if custom_abbrev:
            abbrev_map.update(custom_abbrev)

        if fuel_type in abbrev_map:
            return abbrev_map[fuel_type]

        # Generic truncation with ellipsis
        return self.escape_latex(fuel_type[:max_length - 2] + "..")

    def abbreviate_reference(self,
                             reference: str,
                             max_length: int = 15,
                             style: str = 'author_year') -> str:
        """
        Abbreviate reference citation for display.

        **Concept**: Academic citations can be long. We extract key
        information (author + year) to save space while maintaining
        traceability.

        Parameters
        ----------
        reference : str
            Full reference string
        max_length : int
            Maximum length (used as fallback for truncation)
        style : str
            Citation style: 'author_year', 'first_author_year', or 'truncate'

        Returns
        -------
        str
            Abbreviated citation (e.g., "Smith (2020)")

        """
        if pd.isna(reference) or reference == 'No reference':
            return "---"

        ref = str(reference)

        if style == 'author_year' or style == 'first_author_year':
            # Extract year if present
            year_match = re.search(r'\((\d{4})\)', ref)
            if year_match:
                year = year_match.group(1)
                # Get first author
                author = ref.split('(')[0].strip()
                if ',' in author:
                    author = author.split(',')[0]
                if ' et al' in author:
                    author = author.replace(' et al', '')

                return self.escape_latex(f"{author} ({year})")

        # Fallback: truncate
        if len(ref) > max_length:
            ref = ref[:max_length - 3] + "..."

        return self.escape_latex(ref)

    def format_predictor_str(self,
                             predictors: List[str]|str,
                             max_length: int = 20,
                             separator: str = " + ",
                             format_names:bool=True,
                             custom_mapping:dict[str,str]|None=None) -> str:
        """
        Format list of predictors for display.

        **Concept**: Predictor combinations need to be shown compactly.
        We join predictors with a separator and truncate if necessary.

        Parameters
        ----------
        predictors : List[str]
            List of predictor names
        max_length : int
            Maximum total string length
        separator : str
            Separator between predictors (default: " + ")

        Returns
        -------
        str
            Formatted predictor string

        """
        if isinstance(predictors, str):
            predictors = [predictors]

        if format_names:
            predictors = [self.format_variable_name(p, custom_mapping) for p in predictors]



        pred_str = separator.join(predictors)
        if len(pred_str) > max_length:
            pred_str = pred_str[:max_length - 3] + "..."
        return self.escape_latex(pred_str)

    def format_formula_name(self,
                            formula_name: str,
                            max_len: int = 30) -> str:
        """
        Format formula name for display.

        **Concept**: Formula names from literature can be verbose.
        We truncate long names to keep tables compact.

        Parameters
        ----------
        formula_name : str
            Full formula name
        max_len : int
            Maximum length

        Returns
        -------
        str
            Truncated and escaped formula name

        """
        if len(formula_name) <= max_len:
            return self.escape_latex(formula_name)

        return self.escape_latex(formula_name[:max_len - 3] + "...")

    def format_model_name(self, model_type: str) -> str:
        """
        Format model type name for display.

        **Concept**: Model type strings can be abbreviated to save space
        while remaining clear.

        Parameters
        ----------
        model_type : str
            Full model type string

        Returns
        -------
        str
            Abbreviated model name


        'Poly-2'
        """
        abbrev_map = {
            'random_forest': 'RF',
            'polynomial-2': 'Poly-2',
            'polynomial-3': 'Poly-3',
            'polynomial_2': 'Poly-2',
            'polynomial_3': 'Poly-3',
            'linear': 'Linear',
            'ridge': 'Ridge',
            'lasso': 'Lasso'
        }

        if model_type in abbrev_map:
            return abbrev_map[model_type]

        return self.escape_latex(model_type[:10])

    # ========================================================================
    # CAPTION AND LABEL GENERATION
    # ========================================================================

    def generate_caption(self,
                         metric_name: str,
                         targets: Optional[List[str]],
                         ship_type,
                         fuel_types: Optional[List[str]] = None,
                         max_target_display: int = 3,
                         additional_info: str = "") -> str:
        """
        Generate standard caption for tables.

        **Concept**: Captions should clearly describe table contents.
        We include metric type, targets, fuel types, and ship type in a
        consistent, readable format.

        Parameters
        ----------
        metric_name : str
            Name of metric (e.g., "R²", "RRMSE (\%)")
        targets : List[str], optional
            Target dimensions shown
        ship_type : ShipType or str
            Ship type for context
        fuel_types : List[str], optional
            List of fuel types compared
        max_target_display : int
            Maximum targets to list explicitly
        additional_info : str
            Extra information to append to caption

        Returns
        -------
        str
            Formatted caption (NOT LaTeX-escaped - do that separately)


        'R² performance for DWT, LOA across multiple alternative fuel types...'
        """
        # Build target description
        if targets and len(targets) <= max_target_display:
            target_str = ", ".join(targets)
            target_desc = f"for {target_str}"
        elif targets:
            target_desc = f"for {len(targets)} target dimensions"
        else:
            target_desc = ""

        # Build fuel type description
        if fuel_types and len(fuel_types) > 0:
            fuel_list = ", ".join(fuel_types)
            fuel_desc = f"across multiple alternative fuel types ({fuel_list})"
        else:
            fuel_desc = "comparing conventional vs alternative fuel datasets"

        # Assemble caption
        caption_parts = [
            f"{metric_name} performance",
            target_desc,
            fuel_desc,
            f"for {ship_type.string} vessels."
        ]

        # Filter out empty parts
        caption_parts = [p for p in caption_parts if p]
        caption = " ".join(caption_parts)

        if additional_info:
            caption += f" {additional_info}"

        return caption

    def generate_label(self,
                       label_type: str,
                       ship_type,
                       targets: Optional[List[str]] = None,
                       suffix: str = "") -> str:
        """
        Generate LaTeX label for table reference.

        **Concept**: Labels allow cross-referencing tables in LaTeX.
        We use consistent naming: tab:{ship_type}_{type}_{target}_{suffix}

        Parameters
        ----------
        label_type : str
            Type of table (e.g., "r2", "rrmse", "robustness")
        ship_type : ShipType or str
            Ship type
        targets : List[str], optional
            Target dimensions (used if single target)
        suffix : str
            Additional suffix for uniqueness

        Returns
        -------
        str
            LaTeX label (e.g., "tab:TANKER_r2_DWT_multi_fuel")

        """
        # Build label components
        components = ["tab", str(ship_type), label_type]

        # Add target if single target
        if targets and len(targets) == 1:
            target_safe = self._sanitize_for_label(targets[0])
            components.append(target_safe)

        # Add suffix if provided
        if suffix:
            components.append(suffix)

        return "_".join(components)

    @staticmethod
    def _sanitize_for_label(text: str) -> str:
        """
        Sanitize text for use in LaTeX labels.

        **Concept**: LaTeX labels have restrictions on allowed characters.
        Remove spaces, parentheses, and special characters.

        Parameters
        ----------
        text : str
            Text to sanitize

        Returns
        -------
        str
            Label-safe text
        """
        return (text
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('.', '')
                .replace(',', '')
                .replace('/', '_')
                .replace('\\', '_'))

    # ========================================================================
    # TABLE STRUCTURE HELPERS
    # ========================================================================

    @staticmethod
    def multicolumn(label: str,
                    n_cols: int,
                    alignment: str = 'c') -> str:
        """
        Generate LaTeX multicolumn header.

        **Concept**: Multi-column headers group related columns.
        LaTeX's \\multicolumn command spans multiple columns.

        Parameters
        ----------
        label : str
            Header label (will NOT be escaped - do that beforehand)
        n_cols : int
            Number of columns to span
        alignment : str
            Column alignment ('c', 'l', 'r')

        Returns
        -------
        str
            LaTeX multicolumn command


        """
        return f"\\multicolumn{{{n_cols}}}{{{alignment}}}{{{label}}}"

    @staticmethod
    def cmidrule(start_col: int, end_col: int, trim: str = 'lr') -> str:
        """
        Generate LaTeX cmidrule for column groups.

        **Concept**: Cmidrules add horizontal lines under specific column
        groups, helping visually separate sections in complex tables.

        Parameters
        ----------
        start_col : int
            Starting column number (1-indexed)
        end_col : int
            Ending column number (1-indexed)
        trim : str
            Trim options ('lr', 'l', 'r', or '')

        Returns
        -------
        str
            LaTeX cmidrule command

        """
        trim_str = f"({trim})" if trim else ""
        return f"\\cmidrule{trim_str}{{{start_col}-{end_col}}}"

    @staticmethod
    def generate_colspec(column_alignments: List[str]) -> str:
        """
        Generate LaTeX tabular column specification.

        **Concept**: The column spec defines table structure (e.g., 'lrrr'
        means left-aligned, then three right-aligned columns).

        Parameters
        ----------
        column_alignments : List[str]
            List of alignments ('l', 'c', 'r') for each column

        Returns
        -------
        str
            Column specification string

        """
        return "".join(column_alignments)

    def format_variable_name(self,
                             variable_name: str,
                             custom_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Format variable name for display using mapping dictionary.

        **Concept**: Research variable names are often in ALL_CAPS for
        consistency in code, but should be properly formatted for publication.
        This method uses a lookup dictionary to convert variable names to
        their display-ready versions.

        Parameters
        ----------
        variable_name : str
            Raw variable name (e.g., 'BEAM (M)')
        custom_mapping : Dict[str, str], optional
            Additional/override mappings beyond defaults

        Returns
        -------
        str
            Formatted variable name (e.g., 'Beam (m)')

        'Custom Variable'
        """
        # Merge default and custom mappings
        mapping = self.VARIABLE_DISPLAY_NAMES.copy()
        if custom_mapping:
            mapping.update(custom_mapping)

        # Return mapped name if exists, otherwise return original
        return mapping.get(variable_name, variable_name)

    def parse_formula_name(self, formula_name: str) -> Dict[str, any]:
        """
        Parse literature formula name into components.

        **Concept**: Literature formula names follow a pattern:
        'predictor1_predictor2_to_target_model_type'

        Strategy:
        1. Find and extract model type suffix
        2. Split remaining on '_to_' to separate predictors from target
        3. Parse predictors and target, handling units in parentheses

        Parameters
        ----------
        formula_name : str
            Full formula name (e.g., 'DWT_to_LBP_(M)_power')

        Returns
        -------
        dict
            Dictionary with keys: 'predictors', 'target', 'model_type'

        Examples
        --------
        >>> formatter = LaTeXFormatter()
        >>> formatter.parse_formula_name('DWT_to_LBP_(M)_power')
        {'predictors': ['DWT'], 'target': 'LBP (M)', 'model_type': 'power'}
        >>> formatter.parse_formula_name('BEAM_(M)_LOA_(M)_to_DWT_cubic_polynomial')
        {'predictors': ['BEAM (M)', 'LOA (M)'], 'target': 'DWT', 'model_type': 'cubic_polynomial'}
        """
        # Default values in case parsing fails
        result = {
            'predictors': [formula_name],
            'target': 'Unknown',
            'model_type': 'Unknown'
        }

        # Check if formula contains '_to_' separator
        if '_to_' not in formula_name:
            return result

        # Step 1: Extract model type from the end
        # Check longest patterns first to avoid partial matches
        model_types = [
            'cubic_polynomial',
            'piecewise_power',
            'cubic_poly',
            'polynomial',
            'logarithmic',
            'exponential',
            'piecewise',
            'power',
            'linear',
            'log',
        ]

        model_type = 'Unknown'
        remaining = formula_name

        for mt in model_types:
            if formula_name.endswith('_' + mt):
                model_type = mt
                # Remove the model type and its preceding underscore
                remaining = formula_name[:-len(mt) - 1]
                break

        # If no model type found, work with original formula_name
        if model_type == 'Unknown':
            remaining = formula_name

        # Step 2: Split on '_to_'
        if '_to_' not in remaining:
            return result

        parts = remaining.split('_to_')
        if len(parts) != 2:
            return result

        predictors_part = parts[0]
        target_part = parts[1]

        # Step 3: Parse predictors and target
        # Strategy: Look for pattern X_(Y) where Y is in parentheses
        # These should be kept together as "X (Y)"

        predictors = self._split_variables(predictors_part)
        target = self._clean_variable_name(target_part)

        result = {
            'predictors': predictors,
            'target': target,
            'model_type': model_type
        }

        return result

    def _split_variables(self, var_string: str) -> List[str]:
        """
        Split a string into variable names, keeping units with their variables.

        **Concept**: 'BEAM_(M)_LOA_(M)_DWT' becomes ['BEAM (M)', 'LOA (M)', 'DWT']

        Strategy: Look for pattern VAR_(UNIT) and keep them together.

        Parameters
        ----------
        var_string : str
            String containing one or more variables

        Returns
        -------
        List[str]
            List of cleaned variable names
        """
        import re

        # Pattern to match: word characters followed by optional _(stuff in parens)
        # This captures BEAM_(M) as one unit, or DWT as one unit
        pattern = r'([A-Za-z0-9]+(?:_\([^)]+\))?)'

        matches = re.findall(pattern, var_string)

        # Clean each match
        variables = [self._clean_variable_name(m) for m in matches if m]

        return variables

    def _clean_variable_name(self, var_name: str) -> str:
        """
        Clean a single variable name by converting _(X) to (X).

        Parameters
        ----------
        var_name : str
            Raw variable name like 'BEAM_(M)' or 'DWT'

        Returns
        -------
        str
            Cleaned name like 'BEAM (M)' or 'DWT'
        """
        import re

        # Replace _(X) with space (X)
        # e.g., BEAM_(M) -> BEAM (M)
        cleaned = re.sub(r'_(\([^)]+\))', r' \1', var_name)

        # Handle special cases like C_B, LSW/Disp
        # Keep underscores for these compound identifiers
        special_compounds = {
            'C_B': 'C_B',  # Block coefficient - keep underscore
            'LSW': 'LSW',
            'Lpp': 'Lpp',
        }

        for compound, replacement in special_compounds.items():
            if compound in cleaned:
                return cleaned

        # For everything else, replace remaining underscores with spaces
        cleaned = cleaned.replace('_', ' ')

        return cleaned.strip()

    def format_r2_with_std(self,
                           value: float,
                           std: float = 0.0,
                           precision: int = 3,
                           use_highlight: bool = False,
                           highlight_threshold: float = 0.7) -> str:
        """
        Format R² value with standard deviation.

        **Concept**: Shows both mean performance and uncertainty from
        cross-validation or bootstrap evaluation.

        Parameters
        ----------
        value : float
            R² value (0 to 1)
        std : float, default=0.0
            Standard deviation from evaluation
        precision : int
            Number of decimal places
        use_highlight : bool
            Whether to highlight poor performance
        highlight_threshold : float
            Threshold below which to highlight (poor R²)

        Returns
        -------
        str
            Formatted value with ± std

        Examples
        --------
        >>> formatter.format_r2_with_std(0.856, 0.032)
        '0.856±0.032'

        >>> formatter.format_r2_with_std(0.456, 0.085, use_highlight=True)
        '\\textcolor{red}{0.456±0.085}'
        """
        if np.isnan(value):
            return "---"

        # Format base value
        value_str = f"{value:.{precision}f}"

        # Add std if non-zero
        if std > 0 and not np.isnan(std):
            std_str = f"{std:.{precision}f}"
            formatted = f"{value_str}±{std_str}"
        else:
            formatted = value_str

        # Highlight if requested and below threshold
        if use_highlight and value < highlight_threshold:
            return f"\\textcolor{{red}}{{{formatted}}}"

        return formatted

    def format_rrmse_with_std(self,
                              rrmse: float,
                              std: float,
                              use_highlight: bool = False) -> str:
        """
        Format RRMSE value with standard deviation using \\rrmse command.

        The \\rrmse{} command provides consistent formatting for RRMSE values.

        Parameters
        ----------
        rrmse : float
            RRMSE value (in percentage)
        std : float
            Standard deviation
        use_highlight : bool
            If True, uses bold formatting for poor performance

        Returns
        -------
        str
            LaTeX formatted string like "\\rrmse{24.6}±8.3"

        Examples
        --------
        >>> formatter.format_rrmse_with_std(24.6, 8.3)
        '\\rrmse{24.6}±8.3'
        >>> formatter.format_rrmse_with_std(45.2, 12.1, use_highlight=True)
        '\\textbf{\\rrmse{45.2}±12.1}'
        """
        if np.isnan(rrmse) or np.isnan(std):
            return "---"

        # Format with 1 decimal place
        formatted = f"\\rrmse{{{rrmse:.1f}}}±{std:.1f}"

        if use_highlight:
            formatted = f"\\textbf{{{formatted}}}"

        return formatted

class LaTeXPreambleGenerator:
    """
    Generates LaTeX preamble with required packages and commands.

    **Design**: Separate class for preamble generation allows easy
    customization and versioning of LaTeX requirements.
    """

    STANDARD_PREAMBLE = r"""
% Required packages for domain analysis tables
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{rotating}  % For sideways tables

% Define color formatting commands
\newcommand{\rchange}[1]{%
  \ifnum\pdfstrcmp{\detokenize{#1}}{+}=0
    \textcolor{green!70!black}{#1}%
  \else
    \ifnum\pdfstrcmp{\unexpanded\expandafter{\romannumeral-`\q#1}}{-}=0
      \textcolor{red}{#1}%
    \else
      #1%
    \fi
  \fi
}

\newcommand{\rrmse}[1]{%
  \ifdim#1pt<5pt
    \textcolor{green!70!black}{#1}%
  \else\ifdim#1pt<10pt
    \textcolor{orange}{#1}%
  \else
    \textcolor{red}{#1}%
  \fi\fi
}

% Gap highlighting for performance differences
\newcommand{\fuelgap}[1]{%
  \ifdim#1pt<10pt
    \textcolor{green!70!black}{#1}%
  \else\ifdim#1pt<20pt
    \textcolor{orange}{#1}%
  \else
    \textcolor{red}{#1}%
  \fi\fi
}
"""

    @classmethod
    def get_preamble(cls,
                     include_rotating: bool = True,
                     custom_commands: Optional[str] = None) -> str:
        """
        Get LaTeX preamble with optional customization.

        Parameters
        ----------
        include_rotating : bool
            Include rotating package for sidewaystable
        custom_commands : str, optional
            Additional LaTeX commands to include

        Returns
        -------
        str
            LaTeX preamble code
        """
        preamble = cls.STANDARD_PREAMBLE

        if not include_rotating:
            preamble = preamble.replace(r"\usepackage{rotating}  % For sideways tables", "")

        if custom_commands:
            preamble += "\n" + custom_commands

        return preamble

    @classmethod
    def print_preamble(cls):
        """Print preamble to console."""
        print("\n" + "=" * 70)
        print("REQUIRED LATEX PREAMBLE")
        print("=" * 70)
        print("\nAdd this to your LaTeX document preamble:")
        print(cls.get_preamble())
        print("=" * 70 + "\n")

    @classmethod
    def save_preamble(cls, filepath: str):
        """Save preamble to file."""
        from pathlib import Path
        Path(filepath).write_text(cls.get_preamble())
        print(f"✅ Preamble saved to {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demonstrate usage of LaTeX formatting utilities."""

    # Create formatter instance
    formatter = LaTeXFormatter()

    print("=" * 70)
    print("LaTeX Formatting Utilities - Demo")
    print("=" * 70)

    # Text escaping
    print("\n1. Text Escaping:")
    text = "R² performance: 85% with α=0.05"
    print(f"   Original: {text}")
    print(f"   Escaped:  {formatter.escape_latex(text)}")

    # Metric formatting
    print("\n2. Metric Formatting:")
    print(f"   R² value: {formatter.format_r2_value(0.85632)}")
    print(f"   RRMSE (good): {formatter.format_rrmse_value(4.5)}")
    print(f"   RRMSE (warning): {formatter.format_rrmse_value(7.8)}")
    print(f"   RRMSE (poor): {formatter.format_rrmse_value(12.3)}")

    # Gap formatting
    print("\n3. Gap Formatting:")
    print(f"   R² gap (significant): {formatter.format_r2_gap(-0.052)}")
    print(f"   R² gap (minor): {formatter.format_r2_gap(0.012)}")
    print(f"   RRMSE gap (severe): {formatter.format_rrmse_gap(25.3)}")
    print(f"   RRMSE gap (moderate): {formatter.format_rrmse_gap(15.1)}")

    # Abbreviations
    print("\n4. Abbreviations:")
    print(f"   Fuel type: {formatter.abbreviate_fuel_type('pure_alt_fuel')}")
    print(f"   Fuel type: {formatter.abbreviate_fuel_type('ammonia')}")
    print(f"   Reference: {formatter.abbreviate_reference('Smith et al. (2020)')}")
    print(f"   Model: {formatter.format_model_name('random_forest')}")

    # Complex formatting
    print("\n5. Complex Formatting:")
    predictors = ['DWT', 'LOA', 'BEAM']
    print(f"   Predictors: {formatter.format_predictor_str(predictors)}")

    # Caption and label generation
    print("\n6. Caption and Label Generation:")
    caption = formatter.generate_caption(
        metric_name="R²",
        targets=["DWT", "LOA"],
        ship_type="TANKER",
        fuel_types=["LNG", "Methanol", "Ammonia"]
    )
    print(f"   Caption: {caption[:60]}...")

    label = formatter.generate_label("r2", "TANKER", ["DWT"], "multi_fuel")
    print(f"   Label: {label}")

    # Preamble
    print("\n7. LaTeX Preamble:")
    LaTeXPreambleGenerator.print_preamble()