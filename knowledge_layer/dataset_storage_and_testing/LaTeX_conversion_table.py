"""
LaTeX table conversion utilities for ship design hypothesis testing results.

This module provides functions to convert DataFrames from various statistical
analyses into well-formatted LaTeX tables suitable for academic papers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict


def escape_latex_special_chars(text: str) -> str:
    """
    Escape special LaTeX characters in text.

    Parameters
    ----------
    text : str
        Text that may contain special characters

    Returns
    -------
    str
        Text with escaped special characters
    """
    # Characters that need escaping in LaTeX
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }

    for char, escaped in replacements.items():
        text = text.replace(char, escaped)

    return text


def format_p_value_latex(p_value: float, threshold: float = 0.05) -> str:
    """
    Format p-value with appropriate precision and boldface if significant.

    Parameters
    ----------
    p_value : float
        The p-value to format
    threshold : float, default=0.05
        Significance threshold for bolding

    Returns
    -------
    str
        LaTeX-formatted p-value string
    """
    if pd.isna(p_value):
        return 'N/A'

    # Format based on magnitude
    if p_value < 0.001:
        formatted = r'$< 0.001$'
    elif p_value < 0.01:
        formatted = f'${p_value:.3f}$'
    else:
        formatted = f'${p_value:.3f}$'

    # Bold if significant
    if p_value < threshold:
        formatted = r'\textbf{' + formatted + '}'

    return formatted


def progressive_comparison_to_latex(
    df: pd.DataFrame,
    caption: str = "Progressive comparison analysis results",
    label: str = "tab:progressive_comparison",
    output_file: Optional[str] = None
) -> str:
    """
    Convert progressive comparison results to LaTeX table.

    The progressive comparison shows how p-values change across three
    comparison levels: Raw (all ships), Overlap (similar size range),
    and Matched (precisely matched pairs). This helps distinguish true
    fuel type effects from confounding by ship size distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Output from progressive_comparison_analysis()
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    output_file : str, optional
        If provided, save to this file

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    # Table header
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{' + caption + '}')
    lines.append(r'    \label{' + label + '}')
    lines.append(r'    \small')

    # Determine number of columns
    n_cols = 8  # Feature, Test, Raw_p, Raw_sig, Overlap_p, Overlap_sig, Matched_p, Matched_sig

    lines.append(r'    \begin{tabular}{ll' + 'c' * 6 + '}')
    lines.append(r'        \toprule')

    # Column headers with multicolumn for grouped headers
    lines.append(r'        & & \multicolumn{2}{c}{\textbf{Raw}} & \multicolumn{2}{c}{\textbf{Overlap}} & \multicolumn{2}{c}{\textbf{Matched}} \\')
    lines.append(r'        \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}')
    lines.append(r'        \textbf{Feature} & \textbf{Test} & $p$-value & Sig. & $p$-value & Sig. & $p$-value & Sig. \\')
    lines.append(r'        \midrule')

    # Group by feature for cleaner presentation
    features = df['Feature'].unique()

    for i, feature in enumerate(features):
        feature_data = df[df['Feature'] == feature]

        for j, (idx, row) in enumerate(feature_data.iterrows()):
            # Only print feature name on first row of each feature group
            if j == 0:
                feat_str = escape_latex_special_chars(feature)
            else:
                feat_str = ''

            test_str = escape_latex_special_chars(row['Test'])

            # Format each comparison level
            row_parts = [feat_str, test_str]

            for level in ['Raw', 'Overlap', 'Matched']:
                p_col = f'{level}_p'
                sig_col = f'{level}_sig'

                # Format p-value
                p_str = '\pvalue{'+format_p_value_latex(row[p_col])+'}'

                # Significance indicator
                sig_str = r'\sig' if row[sig_col] else r'\notsig'

                row_parts.extend([p_str, sig_str])

            lines.append('        ' + ' & '.join(row_parts) + r' \\')

        # Add midrule between features (except after last)
        if i < len(features) - 1:
            lines.append(r'        \midrule')

    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Bold $p$-values indicate significance at $\alpha = 0.05$. ')
    lines.append(r'        $\checkmark$ = significant, $-$ = not significant.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    # Save if requested
    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"LaTeX table saved to: {output_file}")

    return latex_code


def sensitivity_analysis_to_latex(
    df: pd.DataFrame,
    matching_features: List[str],
    caption: str = "Matching sensitivity analysis results",
    label: str = "tab:sensitivity_analysis",
    output_file: Optional[str] = None
) -> str:
    """
    Convert sensitivity analysis results to LaTeX table.

    Sensitivity analysis tests whether findings are robust to different
    matching strategies (e.g., matching on DWT vs. LBP vs. BEAM). Consistent
    results across strategies indicate robust findings.

    Parameters
    ----------
    df : pd.DataFrame
        Output from matching_sensitivity_analysis()
    matching_features : List[str]
        List of matching features used
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    output_file : str, optional
        If provided, save to this file

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{' + caption + '}')
    lines.append(r'    \label{' + label + '}')
    lines.append(r'    \small')

    # Create column specification
    n_match_cols = len(matching_features) * 2  # p-value + sig for each
    col_spec = 'll' + 'cc' * len(matching_features)

    lines.append(r'    \begin{tabular}{' + col_spec + '}')
    lines.append(r'        \toprule')

    # Multi-column headers for each matching strategy
    header_parts = [r'        & ']
    for feat in matching_features:
        escaped_feat = escape_latex_special_chars(feat)
        header_parts.append(r' & \multicolumn{2}{c}{\textbf{Match: ' + escaped_feat + r'}}')
    header_parts.append(r' \\')
    lines.append(''.join(header_parts))

    # Add cmidrules
    cmidrule_parts = []
    for i, _ in enumerate(matching_features):
        start_col = 3 + i * 2
        end_col = start_col + 1
        cmidrule_parts.append(r'\cmidrule(lr){' + f'{start_col}-{end_col}' + '}')
    lines.append('        ' + ' '.join(cmidrule_parts))

    # Sub-headers
    subheader_parts = [r'\textbf{Feature}', r'\textbf{Test}']
    for _ in matching_features:
        subheader_parts.extend([r'$p$-value', r'Sig.'])
    lines.append('        ' + ' & '.join(subheader_parts) + r' \\')
    lines.append(r'        \midrule')

    # Data rows
    features = df['Feature'].unique()

    for i, feature in enumerate(features):
        feature_data = df[df['Feature'] == feature]

        for j, (idx, row) in enumerate(feature_data.iterrows()):
            # Feature name only on first row
            if j == 0:
                feat_str = escape_latex_special_chars(feature)
            else:
                feat_str = ''

            test_str = escape_latex_special_chars(row['Test'])

            row_parts = [feat_str, test_str]

            # Add data for each matching strategy
            for match_feat in matching_features:
                p_col = f'Match_{match_feat}_p'
                sig_col = f'Match_{match_feat}_sig'

                p_str = format_p_value_latex(row[p_col])
                sig_str = r'$\checkmark$' if row[sig_col] else r'$-$'

                row_parts.extend([p_str, sig_str])

            lines.append('        ' + ' & '.join(row_parts) + r' \\')

        if i < len(features) - 1:
            lines.append(r'        \midrule')

    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Bold $p$-values indicate significance at $\alpha = 0.05$. ')
    lines.append(r'        Results show robustness across different matching strategies.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"LaTeX table saved to: {output_file}")

    return latex_code


def pattern_analysis_to_latex(
    df: pd.DataFrame,
    caption: str = "Pattern analysis summary",
    label: str = "tab:pattern_analysis",
    output_file: Optional[str] = None
) -> str:
    """
    Convert pattern analysis results to LaTeX table.

    Pattern analysis categorizes how statistical significance changes
    across progressive comparison levels, helping identify whether
    differences are due to fuel type or size distribution effects.

    Parameters
    ----------
    df : pd.DataFrame
        Output from analyze_progressive_patterns()
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    output_file : str, optional
        If provided, save to this file

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{' + caption + '}')
    lines.append(r'    \label{' + label + '}')
    lines.append(r'    \small')
    lines.append(r'    \begin{tabular}{llcccp{5cm}}')
    lines.append(r'        \toprule')
    lines.append(r'        \textbf{Feature} & \textbf{Test} & \textbf{Raw} & \textbf{Overlap} & \textbf{Matched} & \textbf{Interpretation} \\')
    lines.append(r'        \midrule')

    # Group by pattern type for organized presentation
    patterns = df['Pattern'].unique()

    for pattern in patterns:
        # Add pattern header
        pattern_display = escape_latex_special_chars(pattern.replace('_', ' '))
        lines.append(r'        \multicolumn{6}{l}{\textit{' + pattern_display + r'}} \\')

        pattern_data = df[df['Pattern'] == pattern]

        for idx, row in pattern_data.iterrows():
            feat_str = escape_latex_special_chars(row['Feature'])
            test_str = escape_latex_special_chars(row['Test'])

            # Get p-values (already formatted as strings in the input df)
            raw_p = row['Raw_p']
            overlap_p = row['Overlap_p']
            matched_p = row['Matched_p']

            # Interpretation (wrap text)
            interp = escape_latex_special_chars(row['Interpretation'])

            row_parts = [feat_str, test_str, raw_p, overlap_p, matched_p, interp]
            lines.append('        ' + ' & '.join(row_parts) + r' \\')

        lines.append(r'        \midrule')

    # Remove last midrule and add bottomrule
    lines[-1] = r'        \bottomrule'

    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Patterns help distinguish true fuel effects from size distribution artifacts.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"LaTeX table saved to: {output_file}")

    return latex_code


def robustness_summary_to_latex(
    df: pd.DataFrame,
    caption: str = "Robustness summary across matching strategies",
    label: str = "tab:robustness_summary",
    output_file: Optional[str] = None
) -> str:
    """
    Convert robustness summary to LaTeX table.

    The robustness summary shows how many matching strategies resulted
    in significant findings for each feature-test combination. Higher
    consistency indicates more robust findings.

    Parameters
    ----------
    df : pd.DataFrame
        Output from analyze_sensitivity_robustness()
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    output_file : str, optional
        If provided, save to this file

    Returns
    -------
    str
        LaTeX table code
    """

    # First, inspect the DataFrame to understand its structure
    print("Robustness DataFrame columns:", df.columns.tolist())
    print("Robustness DataFrame head:")
    print(df.head())

    # Try to identify column names flexibly
    # Common variations: N_Significant/N_Sig/Num_Significant, N_Total/Total, Robustness/Assessment
    col_map = {}

    # Map standard names to actual column names
    for col in df.columns:
        col_lower = col.lower().replace('_', '').replace(' ', '')

        if 'feature' in col_lower:
            col_map['feature'] = col
        elif 'test' in col_lower:
            col_map['test'] = col
        elif 'nsignificant' in col_lower or 'numsig' in col_lower or 'nsig' in col_lower:
            col_map['n_sig'] = col
        elif 'ntotal' in col_lower or 'total' in col_lower:
            col_map['n_total'] = col
        elif 'robustness' in col_lower or 'assessment' in col_lower:
            col_map['robustness'] = col

    # If we couldn't map all columns, provide helpful error
    required = ['feature', 'test', 'n_sig', 'n_total', 'robustness']
    missing = [r for r in required if r not in col_map]

    if missing:
        print(f"\nWARNING: Could not identify columns for: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        print("Skipping robustness summary table generation.")
        return ""

    lines = []

    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{' + caption + '}')
    lines.append(r'    \label{' + label + '}')
    lines.append(r'    \small')
    lines.append(r'    \begin{tabular}{llccp{6cm}}')
    lines.append(r'        \toprule')
    lines.append(r'        \textbf{Feature} & \textbf{Test} & \textbf{Significant} & \textbf{Total} & \textbf{Robustness} \\')
    lines.append(r'        \midrule')

    for idx, row in df.iterrows():
        feat_str = escape_latex_special_chars(str(row[col_map['feature']]))
        test_str = escape_latex_special_chars(str(row[col_map['test']]))
        n_sig = str(row[col_map['n_sig']])
        n_total = str(row[col_map['n_total']])

        # Robustness assessment
        robustness = escape_latex_special_chars(str(row[col_map['robustness']]))

        row_parts = [feat_str, test_str, n_sig, n_total, robustness]
        lines.append('        ' + ' & '.join(row_parts) + r' \\')

    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Higher consistency across matching strategies indicates more robust findings.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"LaTeX table saved to: {output_file}")

    return latex_code


def export_all_tables(
    progressive_df: Optional[pd.DataFrame] = None,
    sensitivity_df: Optional[pd.DataFrame] = None,
    pattern_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    matching_features: Optional[List[str]] = None,
    output_dir: str = "latex_tables",
    prefix: str = ""
) -> Dict[str, str]:
    """
    Export all analysis results to LaTeX tables in one go.

    Parameters
    ----------
    progressive_df : pd.DataFrame, optional
        Progressive comparison results
    sensitivity_df : pd.DataFrame, optional
        Sensitivity analysis results
    pattern_df : pd.DataFrame, optional
        Pattern analysis results
    robustness_df : pd.DataFrame, optional
        Robustness summary results
    matching_features : List[str], optional
        Required if sensitivity_df is provided
    output_dir : str, default="latex_tables"
        Directory to save tables
    prefix : str, default=""
        Prefix for filenames (e.g., ship type)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping table type to LaTeX code
    """

    # Create output directory with absolute path
    output_path = Path(output_dir).resolve()
    output_path.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("EXPORTING LATEX TABLES")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("=" * 80)

    results = {}

    if progressive_df is not None:
        filename = output_path / f"{prefix}_progressive_comparison.tex"
        print(f"Generating progressive comparison table...")
        latex = progressive_comparison_to_latex(
            progressive_df,
            output_file=str(filename)
        )
        results['progressive'] = latex

    if sensitivity_df is not None:
        if matching_features is None:
            raise ValueError("matching_features required for sensitivity table")
        filename = output_path / f"{prefix}_sensitivity_analysis.tex"
        print(f"Generating sensitivity analysis table...")
        latex = sensitivity_analysis_to_latex(
            sensitivity_df,
            matching_features,
            output_file=str(filename)
        )
        results['sensitivity'] = latex

    if pattern_df is not None:
        filename = output_path / f"{prefix}_pattern_analysis.tex"
        print(f"Generating pattern analysis table...")
        latex = pattern_analysis_to_latex(
            pattern_df,
            output_file=str(filename)
        )
        results['pattern'] = latex

    if robustness_df is not None:
        filename = output_path / f"{prefix}_robustness_summary.tex"
        print(f"Generating robustness summary table...")
        latex = robustness_summary_to_latex(
            robustness_df,
            output_file=str(filename)
        )
        if latex:  # Only add if not empty (in case of column mapping failure)
            results['robustness'] = latex

    print("=" * 80)
    print(f"All tables exported to: {output_path}")
    print(f"Generated {len(results)} LaTeX table(s)")
    print("=" * 80)

    return results