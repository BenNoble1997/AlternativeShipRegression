"""
LaTeX Table Generator for Selector and Domain Analysis Results
===============================================================
Creates publication-ready LaTeX tables for ship design analysis.

Handles TWO types of input:
1. ParametricModelSelector results (single dataset performance)
2. ParametricDomainAnalyzer results (four-way domain shift analysis)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType


class ParametricDomainAnalysisLatexTableGenerator:
    """
    Generate LaTeX tables from selector or analyzer results.

    **Two Input Modes:**

    Mode 1: Selector Results (Single Dataset)
    ------------------------------------------
    - Shows performance on conventional ships only
    - Columns: Model Type | Predictors | Target | R² | RRMSE | AIC | BIC
    - Use for: Documenting model selection process

    Mode 2: Analyzer Results (Domain Shift)
    ----------------------------------------
    - Shows performance across four datasets
    - Columns: Model Type | Predictors | Target | Conv/Match/Alt metrics + gaps
    - Use for: Demonstrating fuel type effects

    Concepts Explained
    ------------------

    **R² Gap Formatting:**
    - Distribution gap (Conv → Matched): Effect of different distributions
    - Fuel gap (Matched → Alternative): Pure fuel type effect
    - Gaps highlighted using \rchange{} (red if negative, green if positive)

    **RRMSE Formatting:**
    - Conditional coloring based on error magnitude
    - Green: < 5%, Orange: 5-10%, Red: > 10%
    """

    # LaTeX preamble requirements
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
        Initialize table generator from selector or analyzer.

        Parameters
        ----------
        selector_or_analyzer : ParametricModelSelector or ParametricDomainAnalyzer
            Either:
            - ParametricModelSelector instance (for single dataset tables)
            - ParametricDomainAnalyzer instance (for domain shift tables)
        verbose : bool, default=True
            Whether to print progress messages

        Examples
        --------
        >>> # From selector
        >>> selector = ParametricModelSelector(...)
        >>> selector.analyze_all_combinations()
        >>> generator = ParametricDomainAnalysisLatexTableGenerator(selector)
        >>>
        >>> # From analyzer
        >>> analyzer = ParametricDomainAnalyzer(...)
        >>> analyzer.analyze_all_combinations()
        >>> generator = ParametricDomainAnalysisLatexTableGenerator(analyzer)
        """
        self.verbose = verbose

        # Detect input type and extract data
        self.input_type = self._detect_input_type(selector_or_analyzer)

        if self.input_type == 'selector':
            self.results_df = self._extract_selector_data(selector_or_analyzer)
            self.mode = 'selector'
        elif self.input_type == 'analyzer':
            self.results_df = self._extract_analyzer_data(selector_or_analyzer)
            self.mode = 'analyzer'
        else:
            raise ValueError(
                "Input must be ParametricModelSelector or ParametricDomainAnalyzer instance"
            )

        if self.verbose:
            print(f"✅ Initialized in {self.mode} mode with {len(self.results_df)} results")

    def _detect_input_type(self, obj) -> str:
        """Detect whether input is selector or analyzer."""
        class_name = obj.__class__.__name__

        if class_name == 'ParametricModelSelector':
            return 'selector'
        elif class_name == 'ParametricDomainAnalyzer':
            return 'analyzer'
        else:
            # Try duck typing
            if hasattr(obj, 'model_recommendations') and hasattr(obj, 'detailed_results'):
                if hasattr(obj, 'alternative'):
                    return 'analyzer'
                else:
                    return 'selector'

            return 'unknown'

    def _extract_selector_data(self, selector) -> pd.DataFrame:
        """
        Extract data from ParametricModelSelector.

        Creates DataFrame with columns:
        - target, predictor_str, predictors, n_predictors, model_type
        - r2, adj_r2, rrmse, mape, rmse, mae
        - aic, bic (if available)
        - max_vif, heteroscedasticity
        """
        if not hasattr(selector, 'model_recommendations'):
            raise ValueError(
                "Selector has no results. Run analyze_all_combinations() first."
            )

        rows = []

        for (predictors, target), model_type in selector.model_recommendations.items():
            key = (predictors, target)

            if key not in selector.detailed_results:
                continue

            model_results = selector.detailed_results[key]

            if model_type not in model_results:
                continue

            metrics = model_results[model_type]

            row = {
                'target': target,
                'predictor_str': ' + '.join(predictors),
                'predictors': predictors,
                'n_predictors': len(predictors),
                'model_type': model_type,

                # Core metrics
                'r2': metrics.get('r2', np.nan),
                'adj_r2': metrics.get('adj_r2', np.nan),
                'rrmse': metrics.get('rrmse', np.nan),
                'mape': metrics.get('mape', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),

                # Information criteria
                'aic': metrics.get('aic', np.nan),
                'bic': metrics.get('bic', np.nan),

                # Diagnostics
                'max_vif': metrics.get('max_vif', np.nan),
                'heteroscedasticity': metrics.get('heteroscedasticity', 'unknown'),

                # Standard deviations (from CV)
                'std_r2': metrics.get('std_r2', np.nan),
                'std_adj_r2': metrics.get('std_adj_r2', np.nan),
                'std_rrmse': metrics.get('std_rrmse', np.nan),

                # Sample info
                'n_samples': metrics.get('n_samples', np.nan),
                'n_features': metrics.get('n_features', np.nan)
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def _extract_analyzer_data(self, analyzer) -> pd.DataFrame:
        """
        Extract data from ParametricDomainAnalyzer.

        Uses the results_df attribute directly.
        """
        if not hasattr(analyzer, 'results_df'):
            raise ValueError(
                "Analyzer has no results. Run analyze_all_combinations() first."
            )

        if analyzer.results_df is None or len(analyzer.results_df) == 0:
            raise ValueError(
                "Analyzer results_df is empty. Run analyze_all_combinations() first."
            )

        return analyzer.results_df.copy()

    # ========================================================================
    # SELECTOR MODE: Single Dataset Tables
    # ========================================================================

    def generate_selector_r2_table(self,
                                   ship_type:ShipType,
                                   targets: Optional[List[str]] = None,
                                   caption: str = None,
                                   label: str = None,
                                   max_predictors: int = 2,
                                   sort_by: str = 'adj_r2',
                                   include_std: bool = True,
                                   ) -> str:
        """
        Generate R² table for selector results (single dataset).

        Columns: Model Type | Predictors | Target | R² | Adj R² | Std(R²)

        Parameters
        ----------
        targets : List[str], optional
            Specific targets to include
        caption : str, optional
            Custom caption
        label : str, optional
            LaTeX label
        max_predictors : int, default=2
            Maximum number of predictors
        sort_by : str, default='adj_r2'
            Column to sort by
        include_std : bool, default=True
            Whether to include standard deviation column (from CV)

        Returns
        -------
        str
            LaTeX table code
        """
        if self.mode != 'selector':
            raise ValueError(
                "This method requires selector mode. "
                "Initialize with ParametricModelSelector instance."
            )

        # Filter data
        filtered_df = self._filter_dataframe(targets, max_predictors)

        if len(filtered_df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Sort
        filtered_df = filtered_df.sort_values(
            ['target', sort_by],
            ascending=[True, False]
        )

        # Generate caption and label
        if caption is None:
            caption = self._generate_selector_r2_caption(targets, max_predictors)
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets=targets,metric_type= 'selector_r2',ship_type=ship_type)

        # Build table
        latex = self._build_selector_r2_table(filtered_df, caption, label, include_std)

        return latex

    def _build_selector_r2_table(self,
                                 df: pd.DataFrame,
                                 caption: str,
                                 label: str,
                                 include_std: bool) -> str:
        """Build selector R² table."""

        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")

        # Column specification
        if include_std:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Model type
                r">{\raggedright\arraybackslash}p{0.16\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.08\linewidth}"  # R²
                r">{\raggedleft\arraybackslash}p{0.08\linewidth}"  # Adj R²
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # Std(Adj R²)
                r">{\raggedleft\arraybackslash}p{0.08\linewidth}"  # AIC
                r">{\raggedleft\arraybackslash}p{0.08\linewidth}"  # BIC
            )
            headers = [
                "Model Type",
                "Predictors",
                "Target",
                "$R^2$",
                "Adj $R^2$",
                "Std(Adj $R^2$)",
                "AIC",
                "BIC"
            ]
        else:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Model type
                r">{\raggedright\arraybackslash}p{0.18\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # R²
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # Adj R²
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # AIC
                r">{\raggedleft\arraybackslash}p{0.10\linewidth}"  # BIC
            )
            headers = [
                "Model Type",
                "Predictors",
                "Target",
                "$R^2$",
                "Adj $R^2$",
                "AIC",
                "BIC"
            ]

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # Group by target
        current_target = None
        target_rows = []

        for idx, row in df.iterrows():
            if row['target'] != current_target:
                if target_rows:
                    lines.extend(self._format_selector_target_group_r2(
                        current_target, target_rows, include_std
                    ))
                    lines.append(r"\hline")

                current_target = row['target']
                target_rows = []

            target_rows.append(row)

        # Flush last group
        if target_rows:
            lines.extend(self._format_selector_target_group_r2(
                current_target, target_rows, include_std
            ))

        # Table footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def _format_selector_target_group_r2(self,
                                         target: str,
                                         rows: List[pd.Series],
                                         include_std: bool) -> List[str]:
        """Format target group for selector R² table."""

        lines = []
        n_rows = len(rows)

        for i, row in enumerate(rows):
            parts = []

            # Model type
            parts.append(self._escape_latex(row['model_type']))

            # Predictors
            parts.append(self._escape_latex(row['predictor_str']))

            # Target (multirow on first row)
            if i == 0:
                if n_rows > 1:
                    target_cell = f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}"
                else:
                    target_cell = self._escape_latex(target)
                parts.append(target_cell)
            else:
                parts.append("")

            # R²
            parts.append(f"{row['r2']:.3f}")

            # Adj R²
            parts.append(f"{row['adj_r2']:.3f}")

            # Std(Adj R²) if requested
            if include_std:
                std_val = row.get('std_adj_r2', np.nan)
                if not np.isnan(std_val):
                    parts.append(f"±{std_val:.3f}")
                else:
                    parts.append("---")

            # AIC
            aic = row.get('aic', np.nan)
            if not np.isnan(aic):
                parts.append(f"{aic:.1f}")
            else:
                parts.append("---")

            # BIC
            bic = row.get('bic', np.nan)
            if not np.isnan(bic):
                parts.append(f"{bic:.1f}")
            else:
                parts.append("---")

            lines.append(" & ".join(parts) + r"\\")

        return lines

    def generate_selector_rrmse_table(self,
                                      ship_type:ShipType,
                                      targets: Optional[List[str]] = None,
                                      caption: str = None,
                                      label: str = None,
                                      max_predictors: int = 2,
                                      sort_by: str = 'rrmse',
                                      include_std: bool = True) -> str:
        """
        Generate RRMSE table for selector results.

        Columns: Model Type | Predictors | Target | RRMSE | Std(RRMSE) | MAPE
        """
        if self.mode != 'selector':
            raise ValueError(
                "This method requires selector mode."
            )

        # Filter data
        filtered_df = self._filter_dataframe(targets, max_predictors)

        if len(filtered_df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Sort
        filtered_df = filtered_df.sort_values(
            ['target', sort_by],
            ascending=[True, True]
        )

        # Generate caption and label
        if caption is None:
            caption = self._generate_selector_rrmse_caption(targets, max_predictors)
            caption = self._escape_latex(caption)
        else:
            caption = self._escape_latex(caption)
        if label is None:
            label = self._generate_label(targets=targets, metric_type='selector_rrmse',ship_type=ship_type)

        # Build table
        latex = self._build_selector_rrmse_table(filtered_df, caption, label, include_std)

        return latex

    def _build_selector_rrmse_table(self,
                                    df: pd.DataFrame,
                                    caption: str,
                                    label: str,
                                    include_std: bool) -> str:
        """Build selector RRMSE table."""

        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")

        # Column specification
        if include_std:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Model type
                r">{\raggedright\arraybackslash}p{0.18\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # RRMSE
                r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Std(RRMSE)
                r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # MAPE
            )
            headers = [
                "Model Type",
                "Predictors",
                "Target",
                "RRMSE (\\%)",
                "Std(RRMSE)",
                "MAPE (\\%)"
            ]
        else:
            colspec = (
                r">{\raggedright\arraybackslash}p{0.16\linewidth}"  # Model type
                r">{\raggedright\arraybackslash}p{0.20\linewidth}"  # Predictors
                r">{\raggedright\arraybackslash}p{0.16\linewidth}"  # Target
                r">{\raggedleft\arraybackslash}p{0.14\linewidth}"  # RRMSE
                r">{\raggedleft\arraybackslash}p{0.14\linewidth}"  # MAPE
            )
            headers = [
                "Model Type",
                "Predictors",
                "Target",
                "RRMSE (\\%)",
                "MAPE (\\%)"
            ]

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # Group by target
        current_target = None
        target_rows = []

        for idx, row in df.iterrows():
            if row['target'] != current_target:
                if target_rows:
                    lines.extend(self._format_selector_target_group_rrmse(
                        current_target, target_rows, include_std
                    ))
                    lines.append(r"\hline")

                current_target = row['target']
                target_rows = []

            target_rows.append(row)

        # Flush last group
        if target_rows:
            lines.extend(self._format_selector_target_group_rrmse(
                current_target, target_rows, include_std
            ))

        # Table footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def _format_selector_target_group_rrmse(self,
                                            target: str,
                                            rows: List[pd.Series],
                                            include_std: bool) -> List[str]:
        """Format target group for selector RRMSE table."""

        lines = []
        n_rows = len(rows)

        for i, row in enumerate(rows):
            parts = []

            # Model type
            parts.append(self._escape_latex(row['model_type']))

            # Predictors
            parts.append(self._escape_latex(row['predictor_str']))

            # Target (multirow on first row)
            if i == 0:
                if n_rows > 1:
                    target_cell = f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}"
                else:
                    target_cell = self._escape_latex(target)
                parts.append(target_cell)
            else:
                parts.append("")

            # RRMSE (with conditional formatting)
            parts.append(self._format_rrmse_value(row['rrmse']))

            # Std(RRMSE) if requested
            if include_std:
                std_val = row.get('std_rrmse', np.nan)
                if not np.isnan(std_val):
                    parts.append(f"±{std_val:.2f}")
                else:
                    parts.append("---")

            # MAPE
            mape = row.get('mape', np.nan)
            if not np.isnan(mape):
                parts.append(f"{mape:.2f}")
            else:
                parts.append("---")

            lines.append(" & ".join(parts) + r"\\")

        return lines

    # ========================================================================
    # ANALYZER MODE: Domain Shift Tables (keep existing methods)
    # ========================================================================

    def generate_r2_table(self,
                          ship_type:ShipType,
                          alt_type:str = 'pure_alt',
                          targets: Optional[List[str]] = None,
                          caption: str = None,
                          label: str = None,
                          max_predictors: int = 2,
                          sort_by: str = 'alternative_r2',
                          highlight_threshold: float = 0.05,
                          best_only:bool=False) -> str:
        """
        Generate R² table for analyzer results (domain shift).

        Only available in analyzer mode.
        """
        if self.mode != 'analyzer':
            raise ValueError(
                "This method requires analyzer mode. "
                "Use generate_selector_r2_table() for selector mode."
            )

        # Filter data
        filtered_df = self._filter_dataframe(targets, max_predictors)

        if len(filtered_df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Sort
        filtered_df = filtered_df.sort_values(
            ['target', sort_by],
            ascending=[True, False if 'r2' in sort_by else True]
        )

        # ← ADD THIS BLOCK
        # If best_only, keep only the best combination per target
        if best_only:
            filtered_df = filtered_df.groupby('target').first().reset_index()
            if self.verbose:
                print(f"  Best-only mode: {len(filtered_df)} targets")

        # Generate caption and label
        if caption is None:
            caption = self._generate_r2_caption(targets, max_predictors, ship_type,alt_type)
            caption = self._escape_latex(caption)
        else:
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets=targets,metric_type= 'r2',ship_type=ship_type)

        # Build table
        latex = self._build_r2_table_structure(filtered_df, caption, label, highlight_threshold,best_only)

        return latex

    def _build_r2_table_structure(self,
                                  df: pd.DataFrame,
                                  caption: str,
                                  label: str,
                                  highlight_threshold: float,
                                  best_only:bool=False) -> str:
        """Build complete R² table LaTeX code."""

        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")

        # Column specification
        colspec = (
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Target
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Predictor
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Model type
            r">{\raggedleft\arraybackslash}p{0.06\linewidth}"  # Conv R²
            r">{\raggedleft\arraybackslash}p{0.06\linewidth}"  # Match R²
            r">{\raggedleft\arraybackslash}p{0.06\linewidth}"  # Alt R²
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Dist gap
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Fuel gap
        )

        lines.append(r"\small")

        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")

        # Column headers
        headers = [
            "Target",
            "Predictors",
            "Model Type",
            "Conv $R^2$",
            "Match $R^2$",
            "Alt $R^2$",
            "Distribution Gap",
            "Fuel Gap"
        ]
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        # ← MODIFY THIS SECTION
        if best_only:
            # Simple iteration - no grouping needed
            for idx, row in df.iterrows():
                parts = []

                # Target (no multirow)
                parts.append(self._escape_latex(row['target']))

                # Predictors
                parts.append(self._escape_latex(row['predictor_str']))

                # Model type
                parts.append(self._escape_latex(row['model_type']))

                # R² values
                parts.append(f"{row['test_r2']:.3f}")
                parts.append(f"{row['matched_r2']:.3f}")
                parts.append(f"{row['alternative_r2']:.3f}")

                # Distribution gap
                dist_gap = row['gap_distribution']
                parts.append(self._format_r2_gap(dist_gap, use_highlight=False))

                # Fuel gap (highlighted)
                fuel_gap = row['gap_fuel_type']
                parts.append(self._format_r2_gap(
                    fuel_gap,
                    use_highlight=True,
                    threshold=highlight_threshold
                ))

                lines.append(" & ".join(parts) + r"\\")

        else:

            # Group by target
            current_target = None
            target_rows = []

            for idx, row in df.iterrows():
                if row['target'] != current_target:
                    if target_rows:
                        lines.extend(self._format_target_group_r2(
                            current_target, target_rows, highlight_threshold
                        ))
                        lines.append(r"\cline{1-8}")

                    current_target = row['target']
                    target_rows = []

                target_rows.append(row)

            # Flush last group
            if target_rows:
                lines.extend(self._format_target_group_r2(
                    current_target, target_rows, highlight_threshold
                ))

        # Table footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def _format_target_group_r2(self,
                                target: str,
                                rows: List[pd.Series],
                                highlight_threshold: float) -> List[str]:
        """Format rows for a single target with multirow."""

        lines = []
        n_rows = len(rows)

        for i, row in enumerate(rows):
            parts = []


            # Target (multirow on first row)
            if i == 0:
                if n_rows > 1:
                    target_cell = f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}"
                else:
                    target_cell = self._escape_latex(target)
                parts.append(target_cell)
            else:
                parts.append("")

            # Predictors
            parts.append(self._escape_latex(row['predictor_str']))

            # Model type
            parts.append(self._escape_latex(row['model_type']))

            # R² values
            parts.append(f"{row['test_r2']:.3f}")
            parts.append(f"{row['matched_r2']:.3f}")
            parts.append(f"{row['alternative_r2']:.3f}")

            # Distribution gap
            dist_gap = row['gap_distribution']
            parts.append(self._format_r2_gap(dist_gap, use_highlight=False))

            # Fuel gap (highlighted)
            fuel_gap = row['gap_fuel_type']
            parts.append(self._format_r2_gap(
                fuel_gap,
                use_highlight=True,
                threshold=highlight_threshold
            ))

            lines.append(" & ".join(parts) + r"\\")

        return lines

    def _format_r2_gap(self,
                       gap: float,
                       use_highlight: bool = True,
                       threshold: float = 0.05) -> str:
        """Format R² gap value with optional highlighting."""
        sign = "+" if gap >= 0 else ""
        formatted = f"{sign}{gap:.3f}"

        if use_highlight and abs(gap) >= threshold:
            return f"\\rchange{{{formatted}}}"
        else:
            return formatted

    def generate_rrmse_table(self,
                             ship_type:ShipType,
                             alt_type:str='Alt_pure',
                             targets: Optional[List[str]] = None,
                             caption: str = None,
                             label: str = None,
                             max_predictors: int = 2,
                             sort_by: str = 'alternative_rrmse',
                             best_only: bool = False) -> str:
        """
        Generate RRMSE table for analyzer results.

        Only available in analyzer mode.
        """
        if self.mode != 'analyzer':
            raise ValueError(
                "This method requires analyzer mode. "
                "Use generate_selector_rrmse_table() for selector mode."
            )

        # Filter data
        filtered_df = self._filter_dataframe(targets, max_predictors)

        if len(filtered_df) == 0:
            if self.verbose:
                print("⚠️  No data matching filters")
            return ""

        # Sort
        filtered_df = filtered_df.sort_values(
            ['target', sort_by],
            ascending=[True, True]
        )

        # If best_only, keep only the best combination per target
        if best_only:
            filtered_df = filtered_df.groupby('target').first().reset_index()
            if self.verbose:
                print(f"  Best-only mode: {len(filtered_df)} targets")

        # Generate caption and label
        if caption is None:
            caption = self._generate_rrmse_caption(targets, max_predictors, ship_type,alt_type)
            caption = self._escape_latex(caption)
        else:
            caption = self._escape_latex(caption)

        if label is None:
            label = self._generate_label(targets, 'rrmse',ship_type=ship_type)

            # Build table
        latex = self._build_rrmse_table_structure(filtered_df, caption, label,best_only)

        return latex


    def _build_rrmse_table_structure(self,
                                     df: pd.DataFrame,
                                     caption: str,
                                     label: str,
                                     best_only:bool=False) -> str:
        """Build complete RRMSE table LaTeX code."""

        lines = []

        # Table header
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")

        # Column specification
        colspec = (
            r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Target
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"  # Predictors
            r">{\raggedright\arraybackslash}p{0.12\linewidth}"  # Model type
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Conv RRMSE
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Match RRMSE
            r">{\raggedleft\arraybackslash}p{0.12\linewidth}"  # Alt RRMSE
        )

        lines.append(r'\small')
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")

        # Column headers
        headers = [
            "Target",
            "Predictors",
            "Model Type",
            "Conv RRMSE (\\%)",
            "Match RRMSE (\\%)",
            "Alt RRMSE (\\%)"
        ]
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        if best_only:
            # Simple iteration - no grouping needed
            for idx, row in df.iterrows():
                parts = []

                # Target (no multirow)
                parts.append(self._escape_latex(row['target']))

                # Predictors
                parts.append(self._escape_latex(row['predictor_str']))

                # Model type
                parts.append(self._escape_latex(row['model_type']))

                # RRMSE values
                parts.append(self._format_rrmse_value(row['test_rrmse']))
                parts.append(self._format_rrmse_value(row['matched_rrmse']))
                parts.append(self._format_rrmse_value(row['alternative_rrmse']))

                lines.append(" & ".join(parts) + r"\\")

        else:

            # Group by target
            current_target = None
            target_rows = []

            for idx, row in df.iterrows():
                if row['target'] != current_target:
                    if target_rows:
                        lines.extend(self._format_target_group_rrmse(
                            current_target, target_rows
                        ))
                        lines.append(r"\hline")

                    current_target = row['target']
                    target_rows = []

                target_rows.append(row)

            # Flush last group
            if target_rows:
                lines.extend(self._format_target_group_rrmse(
                    current_target, target_rows
                ))

        # Table footer
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines)


    def _format_target_group_rrmse(self,
                                   target: str,
                                   rows: List[pd.Series]) -> List[str]:
        """Format rows for a single target with multirow."""

        lines = []
        n_rows = len(rows)

        for i, row in enumerate(rows):
            parts = []


            # Target (multirow on first row)
            if i == 0:
                if n_rows > 1:
                    target_cell = f"\\multirow{{{n_rows}}}{{=}}{{\\raggedright {self._escape_latex(target)}}}"
                else:
                    target_cell = self._escape_latex(target)
                parts.append(target_cell)
            else:
                parts.append("")

            # Predictors
            parts.append(self._escape_latex(row['predictor_str']))

            # Model type
            parts.append(self._escape_latex(row['model_type']))

            # RRMSE values (with conditional formatting)
            parts.append(self._format_rrmse_value(row['test_rrmse']))
            parts.append(self._format_rrmse_value(row['matched_rrmse']))
            parts.append(self._format_rrmse_value(row['alternative_rrmse']))

            lines.append(" & ".join(parts) + r"\\")

        return lines


    def _format_rrmse_value(self, value: float) -> str:
        """Format RRMSE value with conditional color."""
        return f"\\rrmse{{{value:.2f}}}"


    # ========================================================================
    # UNIFIED SAVE METHOD
    # ========================================================================

    def save_tables(self,
                    ship_type:ShipType,
                    output_dir: str = '.',
                    targets: Optional[List[str]] = None,
                    max_predictors: int = 2,
                    prefix: str = None,
                    best_only:bool=True,
                    alt_type:str='Pure alt fuel') -> Dict[str, Path]:
        """
        Generate and save tables appropriate for the mode.

        Selector mode: Saves R² and RRMSE tables for single dataset
        Analyzer mode: Saves R² and RRMSE tables for domain shift

        Parameters
        ----------
        output_dir : str, default='.'
            Directory to save tables
        targets : List[str], optional
            Specific targets to include
        max_predictors : int, default=2
            Maximum number of predictors
        prefix : str, optional
            Prefix for filenames. If None, uses mode name.

        Returns
        -------
        Dict[str, Path]
            Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = self.mode

        if self.mode == 'selector':
            r2_table = self.generate_selector_r2_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors
            )
            rrmse_table = self.generate_selector_rrmse_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors
            )
        else:  # analyzer
            r2_table = self.generate_r2_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only,
                alt_type=alt_type
            )
            rrmse_table = self.generate_rrmse_table(
                ship_type=ship_type,
                targets=targets,
                max_predictors=max_predictors,
                best_only=best_only,
                alt_type=alt_type
            )



        # Save files
        r2_path = output_path / f"{ship_type}_{prefix}_parametric_r2.tex"
        rrmse_path = output_path / f"{ship_type}_{prefix}_parametric_rrmse.tex"
        preamble_path = output_path / f"{ship_type}_{prefix}_parametric_preamble.tex"

        r2_path.write_text(r2_table)
        rrmse_path.write_text(rrmse_table)
        preamble_path.write_text(self.REQUIRED_PACKAGES)

        if self.verbose:
            print(f"✅ {self.mode.capitalize()} tables saved:")
            print(f"   R² table:     {r2_path}")
            print(f"   RRMSE table:  {rrmse_path}")
            print(f"   Preamble:     {preamble_path}")

        return {
            'r2': r2_path,
            'rrmse': rrmse_path,
            'preamble': preamble_path
        }


    # ========================================================================
    # UTILITY METHODS (shared)
    # ========================================================================

    def _filter_dataframe(self,
                          targets: Optional[List[str]],
                          max_predictors: int) -> pd.DataFrame:
        """Filter DataFrame by targets and predictor count."""
        df = self.results_df.copy()

        if targets is not None:
            df = df[df['target'].isin(targets)]

        df = df[df['n_predictors'] <= max_predictors]

        return df


    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
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


    def _generate_r2_caption(self, targets, max_predictors, ship_type,alt_type):
        """Generate caption for analyzer R² table."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"$R^2$ metrics for predicting {target_str} and comparing the performance of a conventional vs a {alt_type}ed {ship_type} dataset."
        else:
            caption = f"$R^2$ metrics showing domain shift effects of a conventional vs a {alt_type}ed {ship_type} dataset."

        caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"
        return caption


    def _generate_rrmse_caption(self, targets, max_predictors, ship_type,alt_type):
        """Generate caption for analyzer RRMSE table."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"RRMSE (\\%) metrics for predicting {target_str} and comparing the performance of a conventional vs a {alt_type}ed {ship_type} dataset."
        else:
            caption = f"RRMSE (\\%) metrics showing domain shift effects on a conventional vs a {alt_type}ed {ship_type} dataset."

        caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"
        return caption


    def _generate_selector_r2_caption(self, targets, max_predictors):
        """Generate caption for selector R² table."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"$R^2$ and information criteria for predicting {target_str}"
        else:
            caption = "$R^2$ and information criteria for model selection"

        caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"
        return caption


    def _generate_selector_rrmse_caption(self, targets, max_predictors):
        """Generate caption for selector RRMSE table."""
        if targets and len(targets) <= 3:
            target_str = ", ".join(targets)
            caption = f"RRMSE (\\%) and MAPE (\\%) metrics for predicting {target_str}"
        else:
            caption = "RRMSE (\\%) and MAPE (\\%) metrics for model selection"

        caption += f" (max {max_predictors} predictor{'s' if max_predictors > 1 else ''})"
        return caption


    def _generate_label(self, targets, metric_type,ship_type):
        """Generate LaTeX label."""
        if targets and len(targets) == 1:
            target_safe = targets[0].replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
            label = f"tab:{ship_type}_{metric_type}_{target_safe}"
        else:
            label = f"tab:{ship_type}_{metric_type}_metrics"

        return label


    @staticmethod
    def print_latex_preamble():
        """Print required LaTeX preamble."""
        print("\n" + "=" * 70)
        print("REQUIRED LATEX PREAMBLE")
        print("=" * 70)
        print("\nAdd this to your LaTeX document preamble:")
        print(ParametricDomainAnalysisLatexTableGenerator.REQUIRED_PACKAGES)
        print("=" * 70 + "\n")