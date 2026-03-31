"""
Functions to create consolidated comparison tables across multiple ship types.

This module extends the hypothesis testing workflow to automatically generate
tables that compare results across all ship types in a single view.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from parameters.shipshape.ShipShapeBasic_class import ShipType
from distribution_comparison_class import DistributionComparison


def collect_mann_whitney_results(
    fleet,
    ship_type: ShipType,
    features: List[str],
        conventional_type:str='matched',
    matching_parameter: str = 'DWT',
    n_matches_per_ship: int = 10,
    use_ratios: bool = False
) -> pd.DataFrame:
    """
    Run Mann-Whitney tests for a single ship type and return structured results.

    This collects p-values for both dual-fuel and pure alternative fuel ships
    compared to matched conventional ships.

    RATIO ANALYSIS CONCEPT:
    When use_ratios=True, instead of testing absolute values (e.g., LBP), we test
    the ratio relative to the matching parameter (e.g., LBP/DWT). This helps:
    - Control for size effects: A larger ship naturally has larger dimensions
    - Isolate design differences: LBP/DWT ratio reflects slenderness, independent of size
    - Reveal fuel-specific design choices: Alternative fuel ships might have different
      proportions (e.g., wider beam for tank space) even at the same DWT

    Example: If alternative fuel ships have LBP=150m and DWT=50,000t (ratio=0.003)
    while conventional ships have LBP=140m and DWT=50,000t (ratio=0.0028), the
    ratio analysis detects this 7% proportional difference.

    Parameters
    ----------
    fleet : FleetData
        Fleet data object with processed datasets
    ship_type : ShipType
        Type of ship being analyzed
    features : List[str]
        Features to test (e.g., ['LBP (M)', 'BEAM (M)', ...])
    matching_parameter : str
        Parameter to use for matching (default: 'DWT')
    n_matches_per_ship : int
        Number of matches per alternative ship
    use_ratios : bool
        If True, test feature/matching_parameter ratios instead of absolute values

    conventional_type: str
        Specify which dataset to use for testing datasets.

    Returns
    -------
    pd.DataFrame
        Results with columns: Feature, ShipType, DualFuel_p, PureAlt_p
    """
    from distribution_comparison_class import DistributionComparison

    results = []

    # Get datasets
    conventional_df = fleet.processed_data['pure_conventional']

    for fuel_type in ['dual_fuel', 'pure_alt_fuel']:
        try:
            alternative_df = fleet.processed_data[fuel_type]

            if conventional_type == 'matched':

                # Create matched dataset
                conventional = fleet.add_matched_dataset(
                    alternative_df=fuel_type,
                    conventional_df='pure_conventional',
                    matching_features=[matching_parameter],
                    n_matches_per_ship=n_matches_per_ship,
                    store_dataset=False,
                    name=f'matched_{matching_parameter}_{fuel_type}'
                )

            elif conventional_type == 'overlap':

                name = matching_parameter + 'overlap'

                conventional = fleet.add_overlap_dataset(alternative_df=fuel_type,
                                                         conventional_df='pure_conventional',
                                                         parameter=matching_parameter,
                                                         name=name)

            else:
                conventional = fleet.processed_data['pure_conventional'].copy()

            # Create ratio columns if requested
            if use_ratios:
                conventional = conventional.copy()
                alternative_df = alternative_df.copy()

                # Check if matching parameter exists in both datasets
                if matching_parameter not in conventional.columns:
                    print(f"Warning: {matching_parameter} not in matched conventional dataset")
                    continue
                if matching_parameter not in alternative_df.columns:
                    print(f"Warning: {matching_parameter} not in alternative dataset")
                    continue

            # Run comparison
            comparison = DistributionComparison(
                conventional_data=conventional,
                alternative_data=alternative_df
            )

            # Test each feature
            for feature in features:
                # Determine the actual column to test
                if use_ratios:
                    # Create ratio column name
                    ratio_col = f'{feature}/{matching_parameter}'

                    # Check if feature exists and create ratio
                    if feature in conventional.columns and feature in alternative_df.columns:
                        # Create ratio columns in both datasets
                        conventional[ratio_col] = (
                            conventional[feature] / conventional[matching_parameter]
                        )
                        alternative_df[ratio_col] = (
                            alternative_df[feature] / alternative_df[matching_parameter]
                        )

                        test_column = ratio_col
                    else:
                        # Feature not available
                        results.append({
                            'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                            'ShipType': ship_type.name,
                            'FuelType': fuel_type,
                            'p_value': np.nan
                        })
                        continue
                else:
                    test_column = feature

                # Check if test column exists
                if test_column in conventional.columns and test_column in alternative_df.columns:
                    # Run Mann-Whitney test
                    mw_result = comparison.mann_whitney_test(test_column)

                    results.append({
                        'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                        'ShipType': ship_type.name,
                        'FuelType': fuel_type,
                        'p_value': mw_result['p_value']
                    })
                else:
                    # Feature not available for this ship type
                    results.append({
                        'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                        'ShipType': ship_type.name,
                        'FuelType': fuel_type,
                        'p_value': np.nan
                    })

        except Exception as e:
            print(f"Error processing {ship_type.name} - {fuel_type}: {e}")
            import traceback
            traceback.print_exc()
            # Add NaN results for all features
            for feature in features:
                results.append({
                    'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                    'ShipType': ship_type.name,
                    'FuelType': fuel_type,
                    'p_value': np.nan
                })

    return pd.DataFrame(results)


def collect_bootstrap_results(
    fleet,
    ship_type: ShipType,
    features: List[str],
    ci_method:str='bca',
    conventional_type:str='matched',
    matching_parameter: str = 'DWT',
    n_matches_per_ship: int = 10,
    n_bootstrap: int = 10000,
    use_ratios: bool = False
) -> pd.DataFrame:
    """
    Run bootstrap CI tests for a single ship type and return structured results.

    This tests whether the difference in mean and median is significant
    using bootstrap confidence intervals.

    RATIO ANALYSIS CONCEPT:
    When use_ratios=True, we test ratios (e.g., LBP/DWT) instead of absolute values.
    This is particularly powerful for bootstrap tests because:
    - Mean ratio difference: Tests if average proportions differ (e.g., alternative
      fuel ships are systematically 5% wider relative to DWT)
    - Median ratio difference: Tests if typical proportions differ (robust to outliers)

    Example interpretation: If mean LBP/DWT for alternative fuel is 0.0031 and for
    conventional is 0.0029, the bootstrap test tells you if this 7% proportional
    difference is statistically significant or just sampling noise.

    Parameters
    ----------
    fleet : FleetData
        Fleet data object with processed datasets
    ship_type : ShipType
        Type of ship being analyzed
    features : List[str]
        Features to test
    matching_parameter : str
        Parameter to use for matching
    n_matches_per_ship : int
        Number of matches per alternative ship
    n_bootstrap : int
        Number of bootstrap samples
    use_ratios : bool
        If True, test feature/matching_parameter ratios instead of absolute values

    ci_method : str
        Method to use for determining confidence intervals. Can be percentile, bca or t test

    conventional_type : str
        Specify whether full dataset, overlap, or matched dataset is used

    Returns
    -------
    pd.DataFrame
        Results with columns: Feature, ShipType, FuelType, mean_sig, median_sig
    """


    results = []

    for fuel_type in ['dual_fuel', 'pure_alt_fuel']:
        try:
            alternative_df = fleet.processed_data[fuel_type]

            if conventional_type == 'matched':

                # Create matched dataset
                conventional_df = fleet.add_matched_dataset(
                    alternative_df=fuel_type,
                    conventional_df='pure_conventional',
                    matching_features=[matching_parameter],
                    n_matches_per_ship=n_matches_per_ship,
                    store_dataset=False,
                    name=f'matched_{matching_parameter}_{fuel_type}'
                )

            elif conventional_type == 'overlap':

                name = matching_parameter+'overlap'

                conventional_df = fleet.add_overlap_dataset(alternative_df=fuel_type,
                                                         conventional_df='pure_conventional',
                                                         parameter=matching_parameter,
                                                            name=name)

            else:
                conventional_df = fleet.processed_data['pure_conventional'].copy()


            # Create ratio columns if requested
            if use_ratios:
                conventional_df = conventional_df.copy()
                alternative_df = alternative_df.copy()

                # Check if matching parameter exists
                if matching_parameter not in conventional_df.columns:
                    print(f"Warning: {matching_parameter} not in matched conventional dataset")
                    continue
                if matching_parameter not in alternative_df.columns:
                    print(f"Warning: {matching_parameter} not in alternative dataset")
                    continue

            # Run comparison
            comparison = DistributionComparison(
                conventional_data=conventional_df,
                alternative_data=alternative_df
            )

            # Test each feature
            for feature in features:
                # Determine the actual column to test
                if use_ratios:
                    # Create ratio column name
                    ratio_col = f'{feature}/{matching_parameter}'

                    # Check if feature exists and create ratio
                    if feature in conventional_df.columns and feature in alternative_df.columns:
                        # Create ratio columns
                        conventional_df[ratio_col] = (
                            conventional_df[feature] / conventional_df[matching_parameter]
                        )
                        alternative_df[ratio_col] = (
                            alternative_df[feature] / alternative_df[matching_parameter]
                        )

                        test_column = ratio_col
                    else:
                        # Feature not available
                        results.append({
                            'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                            'ShipType': ship_type.name,
                            'FuelType': fuel_type,
                            'mean_sig': False,
                            'median_sig': False
                        })
                        continue
                else:
                    test_column = feature

                # Check if test column exists
                if test_column in conventional_df.columns and test_column in alternative_df.columns:
                    # Run bootstrap test
                    boot_result = comparison.bootstrap_difference_ci(
                        feature=test_column,
                        n_bootstrap=n_bootstrap,
                        ci_method=ci_method
                    )

                    results.append({
                        'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                        'ShipType': ship_type.name,
                        'FuelType': fuel_type,
                        'mean_sig': boot_result['mean_significant'],
                        'median_sig': boot_result['median_significant']
                    })
                else:
                    # Feature not available
                    results.append({
                        'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                        'ShipType': ship_type.name,
                        'FuelType': fuel_type,
                        'mean_sig': False,
                        'median_sig': False
                    })

        except Exception as e:
            print(f"Error processing {ship_type.name} - {fuel_type}: {e}")
            import traceback
            traceback.print_exc()
            for feature in features:
                results.append({
                    'Feature': feature if not use_ratios else f'{feature}/{matching_parameter}',
                    'ShipType': ship_type.name,
                    'FuelType': fuel_type,
                    'mean_sig': False,
                    'median_sig': False
                })

    return pd.DataFrame(results)


def create_mann_whitney_comparison_table(
    all_results: pd.DataFrame,
    all_features: List[str],
    ship_types: List[ShipType],
    output_file: Optional[str] = None,
    variable_name_map: Optional[Dict[str, str]] = None,
    conventional_type:str = 'matched'
) -> str:
    """
    Create LaTeX table comparing Mann-Whitney results across all ship types.

    Layout matches your manual table:
    - Rows: Features
    - Columns: Ship types (with sub-columns for dual/pure alternative)

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined results from all ship types (from collect_mann_whitney_results)
    all_features : List[str]
        All possible features across all ship types
    ship_types : List[ShipType]
        Ship types to include
    output_file : str, optional
        Path to save LaTeX file
    variable_name_map : dict, optional
        Mapping from original feature names to display names for LaTeX table

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    # Start table
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{p-value analysis results when comparing ship datasets to a'+f'{conventional_type}'+r' matched conventional dataset using Mann Whitney U test}')
    lines.append(r'    \label{tab:progressive_comparison_all_ships_'+f'{conventional_type}'+r'matched}')
    lines.append(r'    \scriptsize')

    # Build column specification dynamically
    n_ship_types = len(ship_types)
    # First column (feature name) is wider, then pairs of columns for each ship type
    col_spec = r'>{\raggedright\arraybackslash}p{0.15\linewidth}'
    for _ in ship_types:
        col_spec += r'>{\centering\arraybackslash}p{0.1\linewidth}>{\centering\arraybackslash}p{0.1\linewidth}'

    lines.append(r'    \begin{tabular}{' + col_spec + '}')
    lines.append(r'        \toprule')

    # Multi-column header row (ship type names)
    header_parts = ['']
    for ship_type in ship_types:
        ship_name = ship_type.name.replace('_', ' ').title()
        header_parts.append(r'\multicolumn{2}{c}{\textbf{' + ship_name + '}}')
    lines.append('        ' + ' & '.join(header_parts) + r'\\')

    # Add cmidrules between ship types
    cmidrule_parts = []
    for i in range(len(ship_types)):
        start_col = 2 + i * 2
        end_col = start_col + 1
        cmidrule_parts.append(r'\cmidrule(lr){' + f'{start_col}-{end_col}' + '}')
    lines.append('        ' + ' '.join(cmidrule_parts))

    # Sub-header row (Dual fueled / Pure Alternative)
    subheader_parts = [r'\textbf{Feature}']
    for _ in ship_types:
        subheader_parts.extend(['Dual fueled', 'Pure Alternative'])
    lines.append('        ' + '& '.join(subheader_parts) + r'\\')
    lines.append(r'        \midrule')

    # Data rows - one per feature
    for feature in all_features:
        # Get display name for the feature
        if variable_name_map is not None and feature in variable_name_map:
            feature_display = variable_name_map[feature]
        else:
            feature_display = feature

        # Use display name in row, but ORIGINAL name for searching
        row_parts = [feature_display]

        for ship_type in ship_types:
            # Get results for this ship type and feature
            for fuel_type in ['dual_fuel', 'pure_alt_fuel']:
                # Search using ORIGINAL feature name
                mask = (
                    (all_results['Feature'] == feature) &
                    (all_results['ShipType'] == ship_type.name) &
                    (all_results['FuelType'] == fuel_type)
                )

                if mask.any():
                    p_val = all_results[mask]['p_value'].values[0]

                    if pd.isna(p_val):
                        cell_str = 'N/A'
                    else:
                        # Format p-value with bold if significant
                        if p_val < 0.001:
                            cell_str = r'\pvalue{$< 0.001$}'
                        else:
                            cell_str = r'\pvalue{' + f'{p_val:.3f}' + '}'
                else:
                    cell_str = 'N/A'

                row_parts.append(cell_str)

        lines.append('        ' + ' & '.join(row_parts) + r' \\')
        lines.append('')  # Blank line for readability

    # Footer
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Bold $p$-value s indicate significance at $\alpha = 0.05$.')
    lines.append(r'        \sig = significant, \notsig = not significant.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"Mann-Whitney comparison table saved to: {output_file}")

    return latex_code


def create_bootstrap_mean_comparison_table(
    all_results: pd.DataFrame,
    all_features: List[str],
    ship_types: List[ShipType],
    conventional_type:str = 'matched',
    output_file: Optional[str] = None,
    variable_name_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Create LaTeX table comparing bootstrap CI results for MEAN differences.

    This table shows whether the difference in means between alternative and
    conventional ships is statistically significant using bootstrap confidence
    intervals. This tests whether the average value differs between fuel types.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined results from all ship types (from collect_bootstrap_results)
    all_features : List[str]
        All possible features across all ship types
    ship_types : List[ShipType]
        Ship types to include
    output_file : str, optional
        Path to save LaTeX file
    variable_name_map : dict, optional
        Mapping from original feature names to display names for LaTeX table

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    # Start table
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{Bootstrap confidence interval test for difference in \textbf{mean} between alternative and'+f'{conventional_type}'+r' conventional ship datasets}')
    lines.append(r'    \label{tab:bootstrap_mean_comparison_all_ships_'+f'{conventional_type}')
    lines.append(r'    \scriptsize')

    # Column specification
    col_spec = r'>{\raggedright\arraybackslash}p{0.15\linewidth}'
    for _ in ship_types:
        col_spec += r'>{\centering\arraybackslash}p{0.1\linewidth}>{\centering\arraybackslash}p{0.1\linewidth}'

    lines.append(r'    \begin{tabular}{' + col_spec + '}')
    lines.append(r'        \toprule')

    # Multi-column header
    header_parts = ['']
    for ship_type in ship_types:
        ship_name = ship_type.name.replace('_', ' ').title()
        header_parts.append(r'\multicolumn{2}{c}{\textbf{' + ship_name + '}}')
    lines.append('        ' + ' & '.join(header_parts) + r'\\')

    # Cmidrules
    cmidrule_parts = []
    for i in range(len(ship_types)):
        start_col = 2 + i * 2
        end_col = start_col + 1
        cmidrule_parts.append(r'\cmidrule(lr){' + f'{start_col}-{end_col}' + '}')
    lines.append('        ' + ' '.join(cmidrule_parts))

    # Sub-header
    subheader_parts = [r'\textbf{Feature}']
    for _ in ship_types:
        subheader_parts.extend(['Dual fueled', 'Pure Alternative'])
    lines.append('        ' + '& '.join(subheader_parts) + r'\\')
    lines.append(r'        \midrule')

    # Data rows
    for feature in all_features:
        # Get display name for the feature
        if variable_name_map is not None and feature in variable_name_map:
            feature_display = variable_name_map[feature]
        else:
            feature_display = feature

        # Use display name in row, but ORIGINAL name for searching
        row_parts = [feature_display]

        for ship_type in ship_types:
            for fuel_type in ['dual_fuel', 'pure_alt_fuel']:
                # Search using ORIGINAL feature name
                mask = (
                    (all_results['Feature'] == feature) &
                    (all_results['ShipType'] == ship_type.name) &
                    (all_results['FuelType'] == fuel_type)
                )

                if mask.any():
                    mean_sig = all_results[mask]['mean_sig'].values[0]

                    # Significance for mean only
                    if mean_sig:
                        cell_str = r'\sig'
                    else:
                        cell_str = r'\notsig'
                else:
                    cell_str = 'N/A'

                row_parts.append(cell_str)

        lines.append('        ' + ' & '.join(row_parts) + r' \\')
        lines.append('')  # Blank line for readability

    # Footer
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: \sig = significant difference in mean at $\alpha = 0.05$, '
                 r'\notsig = no significant difference. '
                 r'Bootstrap confidence intervals with 10,000 resamples.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"Bootstrap mean comparison table saved to: {output_file}")

    return latex_code


def create_bootstrap_median_comparison_table(
    all_results: pd.DataFrame,
    all_features: List[str],
    ship_types: List[ShipType],
        conventional_type:str='matched',
    output_file: Optional[str] = None,
    variable_name_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Create LaTeX table comparing bootstrap CI results for MEDIAN differences.

    This table shows whether the difference in medians between alternative and
    conventional ships is statistically significant. The median is more robust
    to outliers than the mean, making it useful for skewed distributions.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined results from all ship types (from collect_bootstrap_results)
    all_features : List[str]
        All possible features across all ship types
    ship_types : List[ShipType]
        Ship types to include
    output_file : str, optional
        Path to save LaTeX file
    variable_name_map : dict, optional
        Mapping from original feature names to display names for LaTeX table

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    # Start table
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{Bootstrap confidence interval test for difference in \textbf{median} between alternative and'+f'{conventional_type}'+r' conventional ship datasets}')
    lines.append(r'    \label{tab:bootstrap_median_comparison_all_ships_'+f'{conventional_type}')
    lines.append(r'    \scriptsize')

    # Column specification
    col_spec = r'>{\raggedright\arraybackslash}p{0.15\linewidth}'
    for _ in ship_types:
        col_spec += r'>{\centering\arraybackslash}p{0.1\linewidth}>{\centering\arraybackslash}p{0.1\linewidth}'

    lines.append(r'    \begin{tabular}{' + col_spec + '}')
    lines.append(r'        \toprule')

    # Multi-column header
    header_parts = ['']
    for ship_type in ship_types:
        ship_name = ship_type.name.replace('_', ' ').title()
        header_parts.append(r'\multicolumn{2}{c}{\textbf{' + ship_name + '}}')
    lines.append('        ' + ' & '.join(header_parts) + r'\\')

    # Cmidrules
    cmidrule_parts = []
    for i in range(len(ship_types)):
        start_col = 2 + i * 2
        end_col = start_col + 1
        cmidrule_parts.append(r'\cmidrule(lr){' + f'{start_col}-{end_col}' + '}')
    lines.append('        ' + ' '.join(cmidrule_parts))

    # Sub-header
    subheader_parts = [r'\textbf{Feature}']
    for _ in ship_types:
        subheader_parts.extend(['Dual fueled', 'Pure Alternative'])
    lines.append('        ' + '& '.join(subheader_parts) + r'\\')
    lines.append(r'        \midrule')

    # Data rows
    for feature in all_features:
        # Get display name for the feature (used in LaTeX output)
        if variable_name_map is not None and feature in variable_name_map:
            feature_display = variable_name_map[feature]
        else:
            feature_display = feature

        # CRITICAL: Search using the ORIGINAL feature name, not the display name
        row_parts = [feature_display]

        for ship_type in ship_types:
            for fuel_type in ['dual_fuel', 'pure_alt_fuel']:
                # Use ORIGINAL feature name for searching in results DataFrame
                mask = (
                    (all_results['Feature'] == feature) &  # Use original 'feature', not 'feature_display'
                    (all_results['ShipType'] == ship_type.name) &
                    (all_results['FuelType'] == fuel_type)
                )

                if mask.any():
                    median_sig = all_results[mask]['median_sig'].values[0]

                    # Significance for median only
                    if median_sig:
                        cell_str = r'\sig'
                    else:
                        cell_str = r'\notsig'
                else:
                    cell_str = 'N/A'

                row_parts.append(cell_str)

        lines.append('        ' + ' & '.join(row_parts) + r' \\')
        lines.append('')  # Blank line for readability

    # Footer
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: \sig = significant difference in median at $\alpha = 0.05$, '
                 r'\notsig = no significant difference. '
                 r'Bootstrap confidence intervals with 10,000 resamples.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    latex_code = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(latex_code)
        print(f"Bootstrap median comparison table saved to: {output_file}")

    return latex_code


# Add to your hypothesis_testing.py at the bottom, before if __name__ == '__main__':

def run_multi_ship_comparison(
    ship_types: List[ShipType],
    get_variables_func,
    getting_alt_fuel_ships_func,
        bootstrap:bool=True,
        mann_whitney:bool=True,
    conventional_type:str = 'matched',
    output_dir: str = 'multi_ship_tables',
    matching_parameter: str = 'DWT',
    n_matches_per_ship: int = 10,
    use_ratios: bool = False,
        suffix:str= '',
        variable_name_map: dict[str,str]|None=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run analysis for all ship types and create consolidated comparison tables.

    This is the main workflow function that:
    1. Loops through all ship types
    2. Collects Mann-Whitney and bootstrap results
    3. Generates consolidated LaTeX tables

    Parameters
    ----------
    ship_types : List[ShipType]
        Ship types to analyze
    get_variables_func : callable
        Function to get variables for each ship type
    getting_alt_fuel_ships_func : callable
        Function to load fleet data for each ship type
    output_dir : str
        Directory to save output tables
    matching_parameter : str
        Parameter to use for matching (default: 'DWT')
    n_matches_per_ship : int
        Number of matches per alternative ship
    use_ratios : bool
        If True, test feature/matching_parameter ratios instead of absolute values.
        This controls for size effects and isolates proportional design differences.
        Example: Testing LBP/DWT instead of LBP reveals slenderness differences
        independent of ship size.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (mann_whitney_results, bootstrap_results)
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("MULTI-SHIP TYPE COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(ship_types)} ship types")
    print(f"Output directory: {output_path}")
    print(f"Matching parameter: {matching_parameter}")
    print(f"Use ratios: {use_ratios}")
    if use_ratios:
        print(f"  → Testing feature/{matching_parameter} ratios instead of absolute values")
    print("=" * 80)

    # Collect all features across all ship types
    all_features_set = set()

    # Storage for results
    all_mw_results = []
    all_boot_results = []

    # Process each ship type
    for ship_type in ship_types:
        print(f"\n{'=' * 80}")
        print(f"Processing: {ship_type.name}")
        print(f"{'=' * 80}")

        try:
            # Get variables and fleet for this ship type
            bins, features, filepath, save_dir = get_variables_func(shiptype=ship_type)
            fleet = getting_alt_fuel_ships_func(filepath=filepath)

            # Add to feature set
            all_features_set.update(features)

            # Collect Mann-Whitney results

            if mann_whitney:
                print(f"Running Mann-Whitney tests...")
                mw_results = collect_mann_whitney_results(
                    fleet=fleet,
                    ship_type=ship_type,
                    features=features,
                    matching_parameter=matching_parameter,
                    n_matches_per_ship=n_matches_per_ship,
                    use_ratios=use_ratios,
                    conventional_type=conventional_type,
                )
                all_mw_results.append(mw_results)

            # Collect Bootstrap results

            if bootstrap:
                print(f"Running Bootstrap CI tests...")
                boot_results = collect_bootstrap_results(
                    fleet=fleet,
                    ship_type=ship_type,
                    features=features,
                    matching_parameter=matching_parameter,
                    n_matches_per_ship=n_matches_per_ship,
                    use_ratios=use_ratios,
                    conventional_type=conventional_type,
                )
                all_boot_results.append(boot_results)

            print(f"✓ Completed {ship_type.name}")

        except Exception as e:
            print(f"✗ Error processing {ship_type.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all results
    if mann_whitney:
        combined_mw = pd.concat(all_mw_results, ignore_index=True)

    if bootstrap:
        combined_boot = pd.concat(all_boot_results, ignore_index=True)

    # Save raw combined results
    end = '_ratios' if use_ratios else '_absolute'

    if mann_whitney:
        combined_mw.to_csv(output_path / f'combined_mann_whitney_results{end}_{conventional_type}_{suffix}.csv', index=False)

    if bootstrap:
        combined_boot.to_csv(output_path / f'combined_bootstrap_results{end}_{conventional_type}_{suffix}.csv', index=False)

    # Create ordered feature list (common features first, then ship-specific)
    common_features = ['LBP (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)', 'MCR (KW)']
    ship_specific = sorted(all_features_set - set(common_features))
    all_features = common_features + ship_specific

    # If using ratios, append the ratio notation to feature names for display
    if use_ratios:
        all_features = [f'{feat}/{matching_parameter}' for feat in all_features]

    # Generate comparison tables
    print(f"\n{'=' * 80}")
    print("Generating comparison tables...")
    print(f"{'=' * 80}")

    if mann_whitney:

        mw_table = create_mann_whitney_comparison_table(
            all_results=combined_mw,
            all_features=all_features,
            ship_types=ship_types,
            conventional_type=conventional_type,
            output_file=str(output_path / f'mann_whitney_comparison_table{end}_{conventional_type}_{suffix}.tex'),
            variable_name_map=variable_name_map  # Can be passed as parameter if needed
        )

    if bootstrap:

        boot_mean_table = create_bootstrap_mean_comparison_table(
            all_results=combined_boot,
            all_features=all_features,
            ship_types=ship_types,
            conventional_type=conventional_type,
            output_file=str(output_path / f'bootstrap_mean_comparison_table{end}_{conventional_type}_{suffix}.tex'),
            variable_name_map=variable_name_map  # Can be passed as parameter if needed
        )

        boot_median_table = create_bootstrap_median_comparison_table(
            all_results=combined_boot,
            all_features=all_features,
            ship_types=ship_types,
            conventional_type=conventional_type,
            output_file=str(output_path / f'bootstrap_median_comparison_table{end}_{conventional_type}_{suffix}.tex'),
            variable_name_map=variable_name_map  # Can be passed as parameter if needed
        )

    print(f"\n{'=' * 80}")
    print("MULTI-SHIP COMPARISON COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_path}")

    if mann_whitney:
        print(f"  - combined_mann_whitney_results{end}_{conventional_type}_{suffix}.csv")
        print(f"  - mann_whitney_comparison_table{end}_{conventional_type}_{suffix}.tex")

    if bootstrap:
        print(f"  - combined_bootstrap_results{end}_{conventional_type}_{suffix}.csv")
        print(f"  - bootstrap_mean_comparison_table{end}_{conventional_type}_{suffix}.tex")
        print(f"  - bootstrap_median_comparison_table{end}_{conventional_type}_{suffix}.tex")
    print(f"{'=' * 80}")

    if mann_whitney and bootstrap:
        return combined_mw, combined_boot

    elif mann_whitney:
        return combined_mw

    else:
        return combined_boot