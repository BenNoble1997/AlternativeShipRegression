"""
LaTeX Table Generator for Random Forest Analysis Results
=========================================================
Specialized table generator for RandomForestSelector and RandomForestDomainAnalyzer.

Handles nested dictionary structure from RF classes.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType


class RandomForestLatexTableGenerator:
    """
    Generate LaTeX tables from Random Forest selector or analyzer results.

    **Designed for Random Forest Classes:**
    - RandomForestSelector (single dataset)
    - RandomForestDomainAnalyzer (domain shift)

    **Key Difference from Parametric Generator:**
    RF classes use nested dictionary structure instead of DataFrame,
    so this class handles data extraction differently.
    """

    # LaTeX preamble (same as parametric)
    REQUIRED_PACKAGES = r"""
% Required packages for domain analysis tables
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{biblatex}  % For \textcite (optional)

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
"""

    # Thresholds
    RRMSE_GOOD_THRESHOLD = 5.0
    RRMSE_WARNING_THRESHOLD = 10.0
    R2_SIGNIFICANT_GAP = 0.05

    def __init__(self,
                 selector_or_analyzer,
                 verbose: bool = True):
        """
        Initialize table generator from RF selector or analyzer.

        Parameters
        ----------
        selector_or_analyzer : RandomForestSelector or RandomForestDomainAnalyzer
            Either class instance
        verbose : bool
            Print progress messages
        """
        self.verbose = verbose

        # Detect input type
        self.input_type = self._detect_input_type(selector_or_analyzer)

        if self.input_type == 'selector':
            self.results = selector_or_analyzer.results  # Nested dict
            self.mode = 'selector'
        elif self.input_type == 'analyzer':
            self.results = selector_or_analyzer.results  # Nested dict
            self.mode = 'analyzer'
        else:
            raise ValueError(
                "Input must be RandomForestSelector or RandomForestDomainAnalyzer"
            )

        if self.verbose:
            total_combos = sum(len(combos) for combos in self.results.values())
            print(f"✅ Initialized in {self.mode} mode with {total_combos} combinations")

    def _detect_input_type(self, obj) -> str:
        """Detect whether input is selector or analyzer."""
        class_name = obj.__class__.__name__

        if class_name == 'RandomForestSelector':
            return 'selector'
        elif class_name == 'RandomForestDomainAnalyzer':
            return 'analyzer'
        else:
            # Try duck typing - analyzer has 'alternative' attribute
            if hasattr(obj, 'results'):
                if hasattr(obj, 'alternative'):
                    return 'analyzer'
                else:
                    return 'selector'
            return 'unknown'

    # ========================================================================
    # DATA CONVERSION: Nested Dict → DataFrame
    # ========================================================================

    def _convert_to_dataframe(self,
                              targets: Optional[List[str]] = None,
                              max_predictors: int = 2,
                              best_only: bool = False,
                              sort_by: str = None) -> pd.DataFrame:
        """
        Convert nested dict structure to DataFrame for table generation.

        This is the KEY method that bridges RF data structure to table format.
        """
        rows = []

        # Determine which targets to process
        if targets is None:
            targets_to_process = list(self.results.keys())
        else:
            targets_to_process = [t for t in targets if t in self.results]

        for target in targets_to_process:
            combos = self.results[target]

            for combo_name, data in combos.items():
                predictors = data.get('predictors', [])

                # Filter by max_predictors
                if len(predictors) > max_predictors:
                    continue

                # Build row
                row = {
                    'target': target,
                    'predictor_str': combo_name,
                    'predictors': tuple(predictors),
                    'n_predictors': len(predictors),
                }

                # Add metrics based on mode
                # CRITICAL FIX: Metrics are stored DIRECTLY in data, not under 'metrics' key
                if self.mode == 'selector':
                    # Selector: single dataset metrics
                    row.update({
                        'r2': data.get('r2', np.nan),
                        'adj_r2': data.get('adj_r2', np.nan),
                        'rrmse': data.get('rrmse', np.nan),
                        'mape': data.get('mape', np.nan),
                        'std_r2': data.get('std_r2', np.nan),
                        'std_adj_r2': data.get('std_adj_r2', np.nan),
                        'std_rrmse': data.get('std_rrmse', np.nan),
                        'n_samples': data.get('n_samples', np.nan),
                        'max_vif': data.get('max_vif', np.nan),
                        'combination_type': data.get('combination_type', 'other'),
                        'feature_engineered': data.get('feature_engineered', False),
                    })
                else:  # analyzer
                    # Analyzer: four-way metrics
                    row.update({
                        'train_r2': data.get('train_r2', np.nan),
                        'test_r2': data.get('test_r2', np.nan),
                        'matched_r2': data.get('matched_r2', np.nan),
                        'alternative_r2': data.get('alternative_r2', np.nan),
                        'train_rrmse': data.get('train_rrmse', np.nan),
                        'test_rrmse': data.get('test_rrmse', np.nan),
                        'matched_rrmse': data.get('matched_rrmse', np.nan),
                        'alternative_rrmse': data.get('alternative_rrmse', np.nan),
                        'gap_distribution': data.get('gap_distribution', np.nan),
                        'gap_fuel_type': data.get('gap_fuel_type', np.nan),
                        'gap_distribution_rrmse': data.get('gap_distribution_rrmse', np.nan),
                        'gap_fuel_type_rrmse': data.get('gap_fuel_type_rrmse', np.nan),
                        'combination_type': data.get('combination_type', 'other'),
                        'model_type': data.get('model_type', 'random_forest'),
                    })

                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Sort if requested
        if sort_by:
            ascending = False if 'r2' in sort_by else True
            df = df.sort_values(['target', sort_by], ascending=[True, ascending])

        # Best only: keep first row per target (already sorted)
        if best_only:
            df = df.groupby('target').first().reset_index()

        return df

    # ========================================================================
    # ANALYZER MODE: Domain Shift Tables
    # ========================================================================

    def generate_r2_table(self,
                          ship_type: ShipType,
                          alt_type: str = 'pure_alt',
                          targets: Optional[List[str]] = None,
                          caption: str = None,
                          label: str = None,
                          max_predictors: int = 2,
                          sort_by: str = 'alternative_r2',
                          highlight_threshold: float = 0.05,
                          best_only: bool = False) -> str:
        """
        Generate R² table for analyzer results (domain shift).

        Parameters
        ----------
        ship_type : ShipType
            Ship type for caption/label
        alt_type : str
            Alternative fuel type description
        targets : List[str], optional
            Specific targets
        caption, label : str, optional
            Custom caption/label
        max_predictors : int
            Maximum predictors
        sort_by : str
            Sorting metric
        highlight_threshold : float
            Threshold for highlighting gaps
        best_only : bool
            If True, show only best combination per target

        Returns
        -------
        str
            LaTeX table code
        """
        if self.mode != 'analyzer':
            raise ValueError(
                "This method requires analyzer mode (RandomForestDomainAnalyzer)"
            )

        # Convert to DataFrame
        df = self._convert_to_dataframe(
            targets=targets,
            max_predictors=max_predictors,
            best_only=best_only,
            sort_by=sort_by
        )

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            caption = self._generate_r2_caption(targets, max_predictors, ship_type, alt_type, best_only)
        else:
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets, 'r2_rf', ship_type)

        # Build table
        latex = self._build_r2_table_structure(df, caption, label, highlight_threshold, best_only)

        return latex

    def _build_r2_table_structure(self,
                                  df: pd.DataFrame,
                                  caption: str,
                                  label: str,
                                  highlight_threshold: float,
                                  best_only: bool) -> str:
        """Build R² table LaTeX code."""
        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\small")

        # Column specification
        colspec = (
            r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Target
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Predictors
            r">{\raggedleft\arraybackslash}p{0.075\linewidth}"  # Conv R²
            r">{\raggedleft\arraybackslash}p{0.075\linewidth}"  # Match R²
            r">{\raggedleft\arraybackslash}p{0.075\linewidth}"  # Alt R²
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Dist gap
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Fuel gap
        )

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")

        # Headers
        headers = [
            "Target",
            "Predictors",
            "Conv $R^2$",
            "Match $R^2$",
            "Alt $R^2$",
            "Distribution Gap",
            "Fuel Gap"
        ]
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # Data rows
        if best_only:
            # Simple iteration
            for _, row in df.iterrows():
                parts = [
                    self._escape_latex(row['target']),
                    self._escape_latex(row['predictor_str']),
                    f"{row['test_r2']:.3f}",
                    f"{row['matched_r2']:.3f}",
                    f"{row['alternative_r2']:.3f}",
                    self._format_r2_gap(row['gap_distribution'], False),
                    self._format_r2_gap(row['gap_fuel_type'], True, highlight_threshold)
                ]
                lines.append(" & ".join(parts) + r"\\")
        else:
            # Grouped by target with multirow
            for target in df['target'].unique():
                target_df = df[df['target'] == target]
                n_rows = len(target_df)

                for idx, (_, row) in enumerate(target_df.iterrows()):
                    parts = []

                    # Target (multirow on first)
                    if idx == 0:
                        if n_rows > 1:
                            parts.append(f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}")
                        else:
                            parts.append(self._escape_latex(target))
                    else:
                        parts.append("")

                    # Rest of columns
                    parts.extend([
                        self._escape_latex(row['predictor_str']),
                        f"{row['test_r2']:.3f}",
                        f"{row['matched_r2']:.3f}",
                        f"{row['alternative_r2']:.3f}",
                        self._format_r2_gap(row['gap_distribution'], False),
                        self._format_r2_gap(row['gap_fuel_type'], True, highlight_threshold)
                    ])

                    lines.append(" & ".join(parts) + r"\\")

                if target != df['target'].unique()[-1]:
                    lines.append(r"\cline{1-7}")

        # Footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_rrmse_table(self,
                             ship_type: ShipType,
                             alt_type: str = 'pure_alt',
                             targets: Optional[List[str]] = None,
                             caption: str = None,
                             label: str = None,
                             max_predictors: int = 2,
                             sort_by: str = 'alternative_rrmse',
                             best_only: bool = False) -> str:
        """Generate RRMSE table for analyzer results."""
        if self.mode != 'analyzer':
            raise ValueError(
                "This method requires analyzer mode"
            )

        # Convert to DataFrame
        df = self._convert_to_dataframe(
            targets=targets,
            max_predictors=max_predictors,
            best_only=best_only,
            sort_by=sort_by
        )

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            caption = self._generate_rrmse_caption(targets, max_predictors, ship_type, alt_type, best_only)
        else:
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets, 'rrmse_rf', ship_type)

        # Build table
        latex = self._build_rrmse_table_structure(df, caption, label, best_only)

        return latex

    def _build_rrmse_table_structure(self,
                                     df: pd.DataFrame,
                                     caption: str,
                                     label: str,
                                     best_only: bool) -> str:
        """Build RRMSE table LaTeX code."""
        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\small")

        # Column specification
        colspec = (
            r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Target
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Predictors
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Conv RRMSE
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Match RRMSE
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Alt RRMSE
        )

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")

        # Headers
        headers = [
            "Target",
            "Predictors",
            "Conv RRMSE (\\%)",
            "Match RRMSE (\\%)",
            "Alt RRMSE (\\%)"
        ]
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # Data rows
        if best_only:
            # Simple iteration
            for _, row in df.iterrows():
                parts = [
                    self._escape_latex(row['target']),
                    self._escape_latex(row['predictor_str']),
                    self._format_rrmse_value(row['test_rrmse']),
                    self._format_rrmse_value(row['matched_rrmse']),
                    self._format_rrmse_value(row['alternative_rrmse'])
                ]
                lines.append(" & ".join(parts) + r"\\")
        else:
            # Grouped by target
            for target in df['target'].unique():
                target_df = df[df['target'] == target]
                n_rows = len(target_df)

                for idx, (_, row) in enumerate(target_df.iterrows()):
                    parts = []

                    # Target (multirow on first)
                    if idx == 0:
                        if n_rows > 1:
                            parts.append(f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}")
                        else:
                            parts.append(self._escape_latex(target))
                    else:
                        parts.append("")

                    # Rest of columns
                    parts.extend([
                        self._escape_latex(row['predictor_str']),
                        self._format_rrmse_value(row['test_rrmse']),
                        self._format_rrmse_value(row['matched_rrmse']),
                        self._format_rrmse_value(row['alternative_rrmse'])
                    ])

                    lines.append(" & ".join(parts) + r"\\")

                if target != df['target'].unique()[-1]:
                    lines.append(r"\hline")

        # Footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    # ========================================================================
    # SELECTOR MODE: Single Dataset Tables
    # ========================================================================

    def generate_selector_r2_table(self,
                                   ship_type: ShipType,
                                   targets: Optional[List[str]] = None,
                                   caption: str = None,
                                   label: str = None,
                                   max_predictors: int = 2,
                                   sort_by: str = 'adj_r2',
                                   include_std: bool = True,
                                   best_only: bool = False) -> str:
        """Generate R² table for selector results."""
        if self.mode != 'selector':
            raise ValueError("This method requires selector mode")

        # Convert to DataFrame
        df = self._convert_to_dataframe(
            targets=targets,
            max_predictors=max_predictors,
            best_only=best_only,
            sort_by=sort_by
        )

        if len(df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Generate caption/label
        if caption is None:
            caption = self._generate_selector_r2_caption(targets, max_predictors, best_only)
        else:
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets, 'selector_r2_rf', ship_type)

        # Build table
        latex = self._build_selector_r2_table(df, caption, label, include_std, best_only)

        return latex

    def _build_selector_r2_table(self,
                                 df: pd.DataFrame,
                                 caption: str,
                                 label: str,
                                 include_std: bool,
                                 best_only: bool) -> str:
        """Build selector R² table."""
        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(r"\small")

        # Column specification
        if include_std:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # R²
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # Adj R²
                r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Std(Adj R²)
            )
            headers = ["Predictors", "Target", "$R^2$", "Adj $R^2$", "Std(Adj $R^2$)"]
        else:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.18\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.16\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.14\linewidth}"  # R²
                r">{\raggedleft\arraybackslash}p{0.14\linewidth}"  # Adj R²
            )
            headers = ["Predictors", "Target", "$R^2$", "Adj $R^2$"]

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # Data rows
        if best_only:
            for _, row in df.iterrows():
                parts = [
                    self._escape_latex(row['predictor_str']),
                    self._escape_latex(row['target']),
                    f"{row['r2']:.3f}",
                    f"{row['adj_r2']:.3f}"
                ]
                if include_std:
                    std_val = row.get('std_adj_r2', np.nan)
                    parts.append(f"±{std_val:.3f}" if not np.isnan(std_val) else "---")

                lines.append(" & ".join(parts) + r"\\")
        else:
            for target in df['target'].unique():
                target_df = df[df['target'] == target]
                n_rows = len(target_df)

                for idx, (_, row) in enumerate(target_df.iterrows()):
                    parts = [self._escape_latex(row['predictor_str'])]

                    # Target (multirow on first)
                    if idx == 0:
                        if n_rows > 1:
                            parts.append(f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}")
                        else:
                            parts.append(self._escape_latex(target))
                    else:
                        parts.append("")

                    # Metrics
                    parts.extend([
                        f"{row['r2']:.3f}",
                        f"{row['adj_r2']:.3f}"
                    ])

                    if include_std:
                        std_val = row.get('std_adj_r2', np.nan)
                        parts.append(f"±{std_val:.3f}" if not np.isnan(std_val) else "---")

                    lines.append(" & ".join(parts) + r"\\")

                if target != df['target'].unique()[-1]:
                    lines.append(r"\hline")

        # Footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\^{}'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    def _format_r2_gap(self, gap: float, use_highlight: bool, threshold: float = 0.05) -> str:
        """Format R² gap with optional highlighting."""
        sign = "+" if gap >= 0 else ""
        formatted = f"{sign}{gap:.3f}"
        if use_highlight and abs(gap) >= threshold:
            return f"\\rchange{{{formatted}}}"
        return formatted

    def _format_rrmse_value(self, value: float) -> str:
        """Format RRMSE with conditional color."""
        return f"\\rrmse{{{value:.2f}}}"

    def _generate_r2_caption(self, targets, max_predictors, ship_type, alt_type, best_only):
        """Generate R² caption."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"$R^2$ metrics (Random Forest) for predicting {target_str} comparing conventional vs {alt_type}ed {ship_type} dataset."
        else:
            caption = f"$R^2$ metrics (Random Forest) showing domain shift effects of conventional vs {alt_type}ed {ship_type} dataset."

        if best_only:
            caption += " Showing best predictor combination per target."
        else:
            caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"

        return caption

    def _generate_rrmse_caption(self, targets, max_predictors, ship_type, alt_type, best_only):
        """Generate RRMSE caption."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"RRMSE (\\%) metrics (Random Forest) for predicting {target_str} comparing conventional vs {alt_type}ed {ship_type} dataset."
        else:
            caption = f"RRMSE (\\%) metrics (Random Forest) showing domain shift of conventional vs {alt_type}ed {ship_type} dataset."

        if best_only:
            caption += " Showing best predictor combination per target."
        else:
            caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"

        return caption

    def _generate_selector_r2_caption(self, targets, max_predictors, best_only):
        """Generate selector R² caption."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"Random Forest $R^2$ metrics for predicting {target_str}"
        else:
            caption = "Random Forest $R^2$ metrics for model selection"

        if best_only:
            caption += " (best per target)"
        else:
            caption += f" (max {max_predictors} predictors)"

        return caption

    def _generate_label(self, targets, metric_type, ship_type):
        """Generate LaTeX label."""
        if targets and len(targets) == 1:
            target_safe = targets[0].replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
            return f"tab:{ship_type}_{metric_type}_{target_safe}"
        return f"tab:{ship_type}_{metric_type}_metrics"

    # ========================================================================
    # SAVE METHODS
    # ========================================================================

    def save_tables(self,
                    ship_type: ShipType,
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    max_predictors: int = 2,
                    prefix: str = None,
                    best_only: bool = False,
                    alt_type:str='alt fuel') -> Dict[str, Path]:
        """Generate and save tables."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = f"{self.mode}_rf"

        if self.mode == 'selector':
            r2_table = self.generate_selector_r2_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only
            )
            rrmse_table = ""  # Selector RRMSE not implemented yet
        else:
            r2_table = self.generate_r2_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only
            )
            rrmse_table = self.generate_rrmse_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only
            )

            # Save files
        r2_path = output_path / f"{ship_type}_{prefix}_RF_{alt_type}_r2.tex"
        rrmse_path = output_path / f"{ship_type}_{prefix}_RF_{alt_type}_rrmse.tex"
        preamble_path = output_path / f"{ship_type}_{prefix}_RF_preamble.tex"

        r2_path.write_text(r2_table)
        if rrmse_table:
            rrmse_path.write_text(rrmse_table)
        preamble_path.write_text(self.REQUIRED_PACKAGES)

        if self.verbose:
            print(f"✅ Random Forest {self.mode} tables saved:")
            print(f"   R² table:     {r2_path}")
            if rrmse_table:
                print(f"   RRMSE table:  {rrmse_path}")
            print(f"   Preamble:     {preamble_path}")

        return {
            'r2': r2_path,
            'rrmse': rrmse_path if rrmse_table else None,
            'preamble': preamble_path
        }

    @staticmethod
    def print_latex_preamble():
        """Print required LaTeX preamble."""
        print("\n" + "=" * 70)
        print("REQUIRED LATEX PREAMBLE (RANDOM FOREST TABLES)")
        print("=" * 70)
        print("\nAdd this to your LaTeX document preamble:")
        print(RandomForestLatexTableGenerator.REQUIRED_PACKAGES)
        print("=" * 70 + "\n")