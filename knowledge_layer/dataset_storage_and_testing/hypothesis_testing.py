import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from alt_fuel_datasets import getting_alt_fuel_ships, get_variables
from parameters.shipshape.ShipShapeBasic_class import ShipType
from typing import Optional, List

from distribution_comparison_class import DistributionComparison

from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from knowledge_layer.dataset_storage_and_testing.sensitivity_test import matching_sensitivity_analysis,format_sensitivity_table,analyze_sensitivity_robustness
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.fleet_data_class.Plotting_fleet_class import FleetPlottingClass
from LaTeX_conversion_table import export_all_tables
from knowledge_layer.dataset_storage_and_testing.multi_dataset_stat_analysis import run_multi_ship_comparison
from knowledge_layer.dataset_storage_and_testing.stat_test_debug import (
    run_diagnostic_analysis,
    run_multi_ship_diagnostic
)

from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_heatmap"
    ,["#91bfdb", "#ffffbf","#fc8d59" ]
    # ,["#fc8d59", "#ffffbf", "#91bfdb"]
)

def create_plots(fleet:FleetData,
                 conventional_df:str,
                 alternative_df:str,
                 matching_param:str,
                 parameters:list[str],
                 normal_q_plot:bool=True,
                 log_q_plot:bool=False,
                 dataset_q_plot=True):

    plotter = FleetPlottingClass()

    matching_list = [matching_param]

    overlap_dataset = fleet.add_overlap_dataset(alternative_df=alternative_df,
                                                conventional_df=conventional_df,
                                                parameter=matching_param,
                                                save_dataset=False,
                                                name=f'overlapping by {matching_param}')

    print(f'length of overlap dataset is {len(overlap_dataset)}')

    matched_dataset = fleet.add_matched_dataset(alternative_df=alternative_df,
                                                conventional_df=conventional_df,
                                                matching_features=matching_list,
                                                store_dataset=False,
                                                name=f'matching by {matching_param}')

    print(f'length of matched dataset is {len(matched_dataset)}')

    conventional_dataset = fleet.processed_data[conventional_df]
    alternative_dataset = fleet.processed_data[alternative_df]

    plot_dfs ={conventional_df:conventional_dataset,
               f'overlap_dataset_by_{matching_param}':overlap_dataset,
               f'matched_dataset_by_{matching_param}':matched_dataset,}

    for name,dataset in plot_dfs.items():

        print('='*80)
        print(f'creating plots for alternative fuel vs {name}')
        print('=' * 80)

        plotter.plotting_histograms(parameters=parameters,
                                    conventional_df=dataset,
                                    alternative_df=alternative_dataset,
                                    name = name)


        # fleet.histogram_plots(parameters=parameters,
        #                     conventional_df=conventional_df,
        #                     alternative_df=alternative_df,
        #                     print_stats=True)

        if normal_q_plot:

            plotter.plotting_qq_plots(conventional_df=dataset,alternative_df=alternative_dataset,
                                      parameters=parameters,name=name, print_stats=True)


        if log_q_plot:

            plotter.plotting_qq_plots(conventional_df=dataset,alternative_df=alternative_dataset,name=name,
                                      parameters=parameters,print_stats=True, use_log_transform=True)

        if dataset_q_plot:

            plotter.plotting_qq_plots(conventional_df=dataset,alternative_df=alternative_dataset,name=name,
                                      parameters=parameters,print_stats=True, use_log_transform=False, comparison_type='datasets')

def statistical_comparisons(
                conventional_dataset:pd.DataFrame,
                alternative_dataset:pd.DataFrame,
                conventional_name:str,
                alternative_name:str,
                parameters:list[str],
                ship_type:ShipType,
                hotelling_t_test:bool=True,
                m_descrepancy:bool=True,
                K_S_test:bool=True,
                M_W_test:bool=True,
                welch_test:bool=True,):

    conventional_df = conventional_dataset.copy()
    alternative_df = alternative_dataset.copy()

    comparison = DistributionComparison(conventional_data=conventional_df,
                                        alternative_data=alternative_df)

    print("=" * 80)
    print(f"{parameters} statistical tests using {conventional_name} vs {alternative_name}")
    print("=" * 80)

    if hotelling_t_test:

        print("=" * 80)
        print("HOTELLING'S T² TEST (All Features)")
        print("=" * 80)

        hotelling_result = comparison.hotelling_t2_test(parameters)
        print(hotelling_result['interpretation'])
        print()

    if m_descrepancy:

        print("=" * 80)
        print("MAXIMUM MEAN DISCREPANCY (All Features)")
        print("=" * 80)
        mmd_result = comparison.maximum_mean_discrepancy(parameters, kernel_bandwidth=1.0)
        print(mmd_result['interpretation'])
        print()

    # Run all tests and create summary table
    print("=" * 80)
    print("SUMMARY TABLE (All Univariate Tests)")
    print("=" * 80)
    summary = comparison.run_all_tests(features=parameters,
                                       K_S_test=K_S_test,
                                       M_W_test=M_W_test,
                                       welch_test=welch_test)
    print(summary.to_string(index=False))

    # Create visualizations
    shiptype_name = ship_type.name
    save_fig = f'{conventional_name}_{alternative_name}_{shiptype_name}_distribution_comparison_ver2.png'

    fig = comparison.visualize_distribution_comparison('LBP (M)')
    plt.savefig(save_fig, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization saved as {save_fig}")

def specific_parameter_comparisons(
                conventional_dataset:pd.DataFrame,
                alternative_dataset:pd.DataFrame,
                conventional_name:str,
                alternative_name:str,
                parameters:str,
                ship_type:ShipType,
                K_S_test:bool=False,
                M_W_test:bool=True,
                welch_test:bool=True,):

    conventional_df = conventional_dataset.copy()
    alternative_df = alternative_dataset.copy()

    comparison = DistributionComparison(conventional_data=conventional_df,
                                        alternative_data=alternative_df, )


    print("=" * 80)
    print(f"{parameters} statistical tests using {conventional_name} vs {alternative_name}")
    print("=" * 80)

    if K_S_test:

        # Run individual tests
        print("=" * 80)
        print(f"KOLMOGOROV-SMIRNOV TEST ({parameters})")
        print("=" * 80)
        ks_result = comparison.kolmogorov_smirnov_test(parameters)
        print(ks_result['interpretation'])
        print()

    if M_W_test:

        print("=" * 80)
        print(f"MANN-WHITNEY U TEST ({parameters})")
        print("=" * 80)
        mw_result = comparison.mann_whitney_test(parameters)
        print(mw_result['interpretation'])
        print()

    if welch_test:

        print("=" * 80)
        print(f"WELCH'S T-TEST ({parameters})")
        print("=" * 80)
        t_result = comparison.welch_t_test(parameters)
        print(t_result['interpretation'])
        print()

def significance_sensitivity(feature:list[str],
                             conventional_fleet:pd.DataFrame,
                             alternative_fleet:pd.DataFrame,
                             K_S_test:bool=True,
                             M_W_test:bool=True,):

    conventional_df = conventional_fleet.copy()
    alternative_df = alternative_fleet.copy()



def individual_and_total_comparisons(fleet:FleetData,
                                     conventional_dataset:str,
                                     alternative_dataset:str,
                                     parameters:list[str],
                                     matching_parameter:str,
                                     ship_type:ShipType,
                                     bootstrap_difference_ci:bool=True,
                                        K_S_test:bool=True,
                                        M_W_test:bool=True,
                                        welch_test:bool=True,
                                     hotelling_t_test:bool=True,
                                    m_descrepancy:bool=True
                                     ):
    alternative_df = fleet.processed_data[alternative_dataset]
    conventional_df = fleet.processed_data[conventional_dataset]

    overlap_df = fleet.add_overlap_dataset(alternative_df=alternative_dataset,
                                                conventional_df=conventional_dataset,
                                                name='overlap_dwt',
                                                parameter=matching_parameter,
                                                save_dataset=False)

    matched_df = fleet.add_matched_dataset(alternative_df=alternative_dataset,
                                           conventional_df=conventional_dataset,
                                           name='matched_dwt',
                                           matching_features=[matching_parameter],
                                           store_dataset=False)


    conventional_dict = {f'raw conventional set':conventional_df,
                         f'overal conventional set({matching_parameter})':overlap_df,
                         f'matched conventional set ({matching_parameter})':matched_df,}

    for name, dataset in conventional_dict.items():


        if bootstrap_difference_ci:

            print(f"=" * 80)
            print(f"\nBootstrap difference ({name}) vs {alternative_dataset}")
            print(f"=" * 80)


            comparison = DistributionComparison(conventional_data=dataset,
                                                alternative_data=alternative_df, )


            for parameter in parameters:
                comparison.bootstrap_difference_ci(feature=parameter)


        for parameter in parameters:

            specific_parameter_comparisons(
                                       conventional_dataset=dataset,
                                       alternative_dataset=alternative_df,
                                        conventional_name=name,
                                        alternative_name=alternative_dataset,
                                       parameters=parameter,
                                       ship_type=ship_type,
                                        K_S_test=K_S_test,
                                        M_W_test=M_W_test,
                                        welch_test=welch_test)

        statistical_comparisons(conventional_dataset=dataset,
                                alternative_dataset=alternative_df,
                                conventional_name=name,
                                alternative_name=alternative_dataset,
                                parameters=parameters,
                                ship_type=ship_type,
                                K_S_test=K_S_test,
                                M_W_test=M_W_test,
                                welch_test=welch_test,
                                hotelling_t_test=hotelling_t_test,
                                m_descrepancy=m_descrepancy)


def progressive_comparison_analysis(
        fleet,
        conventional_dataset: str,
        alternative_dataset: str,
        test_features: List[str],
        matching_parameter: str = 'DWT',
        ship_type: Optional[ShipType] = None,
        n_matches_per_ship: int = 10,
        K_S_test: bool = False,
        M_W_test: bool = True,
        welch_test: bool = False,
        print_results: bool = True,
        save_tables: bool = True,
        output_dir: Optional[str] = None,
        variable_name_map: Optional[dict] = None,
        use_ratios: bool = False,  # NEW PARAMETER
        debug: bool = False  # NEW: Enable debug prints
) -> pd.DataFrame:
    """
    Progressive comparison analysis across three levels of control.

    NEW: Now supports ratio analysis via use_ratios parameter.

    RATIO ANALYSIS CONCEPT:
    When use_ratios=True, instead of testing absolute values (e.g., LBP), we test
    the ratio relative to the matching parameter (e.g., LBP/DWT). This:
    - Controls for size effects: Larger ships naturally have larger dimensions
    - Isolates design differences: LBP/DWT reflects slenderness independent of size
    - Reveals fuel-specific design choices: Alternative fuel ships might have different
      proportions (e.g., wider beam for tank space) even at the same DWT

    Example: If alternative fuel ships have LBP=150m and DWT=50,000t (ratio=0.003)
    while conventional ships have LBP=140m and DWT=50,000t (ratio=0.0028), the
    ratio analysis detects this 7% proportional difference.

    Parameters
    ----------
    ...existing parameters...
    use_ratios : bool, default=False
        If True, test feature/matching_parameter ratios instead of absolute values.
        This controls for size effects and isolates proportional design differences.
    debug : bool, default=False
        If True, print detailed information about ratio creation and testing.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        - Feature (will show "LBP (M)/DWT" format if use_ratios=True)
        - Test, Raw_p, Raw_sig, Overlap_p, Overlap_sig, Matched_p, Matched_sig
    """
    from distribution_comparison_class import DistributionComparison

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: Progressive Comparison Configuration")
        print("=" * 80)
        print(f"use_ratios: {use_ratios}")
        print(f"matching_parameter: {matching_parameter}")
        print(f"test_features: {test_features}")
        print("=" * 80)

    if print_results:
        print("=" * 80)
        print("PROGRESSIVE COMPARISON ANALYSIS")
        if use_ratios:
            print(f"MODE: RATIO ANALYSIS (testing feature/{matching_parameter})")
        else:
            print("MODE: ABSOLUTE VALUE ANALYSIS")
        print("=" * 80)
        print(f"Testing {len(test_features)} features across 3 comparison levels")
        print(f"Alternative dataset: {alternative_dataset}")
        print(f"Conventional dataset: {conventional_dataset}")
        print(f"Matching parameter: {matching_parameter}")
        print("=" * 80)

    # Get alternative dataset (stays constant)
    alternative_df = fleet.processed_data[alternative_dataset].copy()

    # Create the three comparison datasets
    if print_results:
        print(f"\n{'=' * 80}")
        print("Creating comparison datasets...")
        print(f"{'=' * 80}")

    # 1. Raw conventional
    raw_conventional = fleet.processed_data[conventional_dataset].copy()
    if print_results:
        print(f"✓ Raw conventional: n={len(raw_conventional)}")

    # 2. Overlap dataset
    overlap_conventional = fleet.add_overlap_dataset(
        alternative_df=alternative_dataset,
        conventional_df=conventional_dataset,
        parameter=matching_parameter,
        save_dataset=False,
        name=f'overlap_{matching_parameter}'
    ).copy()
    if print_results:
        print(f"✓ Overlap conventional: n={len(overlap_conventional)}")

    # 3. Matched dataset
    matched_conventional = fleet.add_matched_dataset(
        alternative_df=alternative_dataset,
        conventional_df=conventional_dataset,
        matching_features=[matching_parameter],
        n_matches_per_ship=n_matches_per_ship,
        store_dataset=False,
        name=f'matched_{matching_parameter}'
    ).copy()
    if print_results:
        print(f"✓ Matched conventional: n={len(matched_conventional)}")

    # NEW: Create ratio columns if requested
    if use_ratios:
        if debug:
            print("\n" + "=" * 80)
            print("DEBUG: Creating Ratio Columns")
            print("=" * 80)

        # Check matching parameter exists in all datasets
        for dataset_name, dataset in [
            ('alternative', alternative_df),
            ('raw_conventional', raw_conventional),
            ('overlap_conventional', overlap_conventional),
            ('matched_conventional', matched_conventional)
        ]:
            if matching_parameter not in dataset.columns:
                raise ValueError(
                    f"Matching parameter '{matching_parameter}' not found in {dataset_name} dataset"
                )

            if debug:
                print(f"\n{dataset_name}:")
                print(f"  Has {matching_parameter}: ✓")
                print(f"  {matching_parameter} range: [{dataset[matching_parameter].min():.1f}, "
                      f"{dataset[matching_parameter].max():.1f}]")

        # Create ratio columns in all datasets
        datasets_to_modify = [
            ('alternative', alternative_df),
            ('raw_conventional', raw_conventional),
            ('overlap_conventional', overlap_conventional),
            ('matched_conventional', matched_conventional)
        ]

        for dataset_name, dataset in datasets_to_modify:
            if debug:
                print(f"\nCreating ratios in {dataset_name}:")

            for feature in test_features:
                if feature in dataset.columns:
                    ratio_col = f'{feature}/{matching_parameter}'
                    dataset[ratio_col] = dataset[feature] / dataset[matching_parameter]

                    if debug:
                        valid_ratios = dataset[ratio_col].dropna()
                        print(f"  {ratio_col}:")
                        print(f"    Created: ✓ (n={len(valid_ratios)})")
                        print(f"    Range: [{valid_ratios.min():.6f}, {valid_ratios.max():.6f}]")
                        print(f"    Mean: {valid_ratios.mean():.6f}")
                else:
                    if debug:
                        print(f"  {feature}: Not available in {dataset_name}")

        # Update test_features to use ratio column names
        original_features = test_features.copy()
        test_features = [f'{feat}/{matching_parameter}' for feat in test_features]

        if debug:
            print("\nUpdated test_features:")
            for orig, ratio in zip(original_features, test_features):
                print(f"  {orig} → {ratio}")

    # Storage for results
    all_results = []

    # Dictionary of datasets to compare
    comparison_levels = {
        'Raw': raw_conventional,
        'Overlap': overlap_conventional,
        'Matched': matched_conventional
    }

    # Run tests for each comparison level
    for level_name, conv_df in comparison_levels.items():
        if print_results:
            print(f"\n{'=' * 80}")
            print(f"Running tests: {level_name} comparison")
            print(f"{'=' * 80}")

        if debug:
            print(f"\nDEBUG: {level_name} level")
            print(f"  Conventional dataset size: {len(conv_df)}")
            print(f"  Alternative dataset size: {len(alternative_df)}")

        # Create comparison object
        comparison = DistributionComparison(
            conventional_data=conv_df,
            alternative_data=alternative_df
        )

        # Run all tests
        try:
            # Filter test_features to only those available in both datasets
            available_features = [
                f for f in test_features
                if f in conv_df.columns and f in alternative_df.columns
            ]

            if debug:
                print(f"  Available features for testing: {len(available_features)}/{len(test_features)}")
                if len(available_features) < len(test_features):
                    missing = set(test_features) - set(available_features)
                    print(f"  Missing features: {missing}")

            test_results = comparison.run_all_tests(
                features=available_features,
                K_S_test=K_S_test,
                M_W_test=M_W_test,
                welch_test=welch_test
            )

            # Add level information
            test_results['Comparison_Level'] = level_name
            all_results.append(test_results)

            if print_results:
                print(f"✓ Tests complete for {level_name} level")

        except Exception as e:
            print(f"✗ Error with {level_name} level: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        raise ValueError("No successful comparisons completed")

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Pivot to create the desired output format
    if print_results:
        print(f"\n{'=' * 80}")
        print("Reformatting results...")
        print(f"{'=' * 80}")

    output_rows = []

    # Get unique feature-test combinations
    for feature in test_features:
        for test_type in combined_results['Test'].unique():

            row = {
                'Feature': feature,
                'Test': test_type
            }

            # For each comparison level, get p-value and significance
            for level in ['Raw', 'Overlap', 'Matched']:
                mask = (
                        (combined_results['Feature'] == feature) &
                        (combined_results['Test'] == test_type) &
                        (combined_results['Comparison_Level'] == level)
                )

                if mask.any():
                    result_row = combined_results[mask].iloc[0]
                    p_value = result_row['P-value']
                    significant = result_row['Significant']

                    row[f'{level}_p'] = p_value
                    row[f'{level}_sig'] = significant
                else:
                    row[f'{level}_p'] = np.nan
                    row[f'{level}_sig'] = False

            output_rows.append(row)

    # Create DataFrame
    output_df = pd.DataFrame(output_rows)

    # Print summary if requested
    if print_results:
        print(f"\n{'=' * 80}")
        print("PROGRESSIVE COMPARISON SUMMARY")
        print(f"{'=' * 80}")

        for idx, row in output_df.iterrows():
            feature = row['Feature']
            test = row['Test']

            sig_raw = row['Raw_sig']
            sig_overlap = row['Overlap_sig']
            sig_matched = row['Matched_sig']

            p_raw = row['Raw_p']
            p_overlap = row['Overlap_p']
            p_matched = row['Matched_p']

            # Determine pattern
            if sig_raw and sig_overlap and sig_matched:
                pattern = "PERSISTENT (sig at all levels)"
            elif sig_raw and sig_overlap and not sig_matched:
                pattern = "ELIMINATED BY MATCHING"
            elif sig_raw and not sig_overlap and not sig_matched:
                pattern = "SIZE RANGE EFFECT"
            elif not sig_raw and not sig_overlap and not sig_matched:
                pattern = "NOT SIGNIFICANT"
            else:
                pattern = "INCONSISTENT"

            print(f"\n{feature} - {test}:")
            print(f"  Raw: p={p_raw:.4f} {'✓' if sig_raw else '✗'}")
            print(f"  Overlap: p={p_overlap:.4f} {'✓' if sig_overlap else '✗'}")
            print(f"  Matched: p={p_matched:.4f} {'✓' if sig_matched else '✗'}")
            print(f"  Pattern: {pattern}")

    # Save tables if requested
    if save_tables and output_dir:
        # Create output directory with ship type subdirectory
        if ship_type:
            output_path = Path(output_dir) / ship_type.name
        else:
            output_path = Path(output_dir)

        output_path.mkdir(exist_ok=True, parents=True)

        # Generate LaTeX table for each test type
        suffix = '_ratios' if use_ratios else '_absolute'

        for test_type in output_df['Test'].unique():
            test_data = output_df[output_df['Test'] == test_type]

            # Create LaTeX table
            latex_table = create_progressive_latex_table(
                results_df=test_data,
                test_name=test_type,
                ship_type=ship_type,
                alternative_dataset=alternative_dataset,
                variable_name_map=variable_name_map,
                use_ratios=use_ratios  # NEW: Pass flag to LaTeX generator
            )

            # Save to file
            if ship_type:
                filename = f'progressive_{test_type.replace(" ", "_")}_{alternative_dataset}_{ship_type.name}{suffix}.tex'
            else:
                filename = f'progressive_{test_type.replace(" ", "_")}_{alternative_dataset}{suffix}.tex'

            filepath = output_path / filename
            filepath.write_text(latex_table)

            if print_results:
                print(f"✓ Saved LaTeX table: {filepath}")

    return output_df


def create_progressive_latex_table(
        results_df: pd.DataFrame,
        test_name: str,
        ship_type: Optional[ShipType] = None,
        alternative_dataset: str = '',
        variable_name_map: Optional[dict] = None,
        use_ratios: bool = False  # NEW PARAMETER
) -> str:
    """
    Create LaTeX table for progressive comparison results.

    NEW: Handles both absolute values and ratio analysis.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for a single test type
    test_name : str
        Name of the test (e.g., 'Mann-Whitney')
    ship_type : ShipType, optional
        Ship type for caption
    alternative_dataset : str
        Name of alternative dataset for caption
    variable_name_map : dict, optional
        Mapping from feature names to LaTeX display names
    use_ratios : bool, default=False
        Whether this table shows ratio analysis results

    Returns
    -------
    str
        LaTeX table code
    """

    lines = []

    # Determine ship type name for caption
    if ship_type:
        ship_name = ship_type.name.replace('_', ' ').lower()
    else:
        ship_name = ''

    # Determine fuel type for caption
    if 'dual' in alternative_dataset.lower():
        fuel_desc = 'dual-fuelled'
    elif 'pure' in alternative_dataset.lower():
        fuel_desc = 'alternatively fuelled'
    else:
        fuel_desc = 'alternative fuel'

    # Start table
    lines.append(r'\subsection{' + test_name + ' test results}')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')

    # Caption - different for ratio vs absolute
    if use_ratios:
        caption = (f"$p$-value analysis results when comparing proportional design features "
                   f"(feature/DWT ratios) between {fuel_desc} {ship_name} and conventionally "
                   f"powered {ship_name} sets")
    else:
        caption = (f"$p$-value analysis results when comparing absolute feature values between "
                   f"{fuel_desc} {ship_name} and conventionally powered {ship_name} sets")

    lines.append(r'    \caption{' + caption + '}')

    # Label
    label_safe = alternative_dataset.replace('_', '_')
    suffix = '_ratios' if use_ratios else '_absolute'
    if ship_type:
        label = f"tab:progressive_comparison_{label_safe}_{ship_type.name}{suffix}"
    else:
        label = f"tab:progressive_comparison_{label_safe}{suffix}"
    lines.append(r'    \label{' + label + '}')

    lines.append(r'    \small')

    # Column specification
    col_spec = (r'>{\raggedright\arraybackslash}p{0.15\linewidth}'
                r'>{\centering\arraybackslash}p{0.1\linewidth}'
                r'>{\centering\arraybackslash}p{0.05\linewidth}'
                r'>{\centering\arraybackslash}p{0.1\linewidth}'
                r'>{\centering\arraybackslash}p{0.05\linewidth}'
                r'>{\centering\arraybackslash}p{0.1\linewidth}'
                r'>{\centering\arraybackslash}p{0.05\linewidth}')

    lines.append(r'    \begin{tabular}{' + col_spec + '}')
    lines.append(r'        \toprule')

    # Header row
    lines.append(
        r'        & \multicolumn{2}{c}{\textbf{Raw}} & \multicolumn{2}{c}{\textbf{Overlap}} & \multicolumn{2}{c}{\textbf{Matched}} \\')
    lines.append(r'        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}')
    lines.append(r'        \textbf{Feature}& $p$-value & Sig. & $p$-value & Sig. & $p$-value & Sig. \\')
    lines.append(r'        \midrule')

    # Data rows
    for idx, row in results_df.iterrows():
        feature = row['Feature']

        # Get display name
        if variable_name_map and feature in variable_name_map:
            feature_display = variable_name_map[feature]
        else:
            feature_display = feature

        row_parts = [feature_display]

        # For each level: Raw, Overlap, Matched
        for level in ['Raw', 'Overlap', 'Matched']:
            p_val = row[f'{level}_p']
            is_sig = row[f'{level}_sig']

            # Format p-value
            if pd.isna(p_val):
                p_str = 'N/A'
            elif p_val < 0.001:
                p_str = r'$< 0.001$'
            else:
                p_str = f'${p_val:.3f}$'

            # Bold if significant
            if is_sig:
                p_str = r'\textbf{' + p_str + '}'

            # Significance indicator
            sig_str = r'\sig' if is_sig else r'\notsig'

            row_parts.extend([p_str, sig_str])

        lines.append('        ' + ' & '.join(row_parts) + r' \\')

    # Footer
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    \begin{tablenotes}')
    lines.append(r'        \small')
    lines.append(r'        \item Note: Bold $p$-values indicate significance at $\alpha = 0.05$.')
    lines.append(r'        \sig = significant, \notsig = not significant.')
    if use_ratios:
        lines.append(
            r'        \item Ratio analysis controls for ship size effects by testing proportional differences.')
    lines.append(r'    \end{tablenotes}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def run_progressive_for_all_datasets(
        fleet,
        ship_type: ShipType,
        test_features: List[str],
        matching_parameter: str = 'DWT',
        output_dir: str = 'progressive_analysis',
        variable_name_map: Optional[dict] = None,
        print_results: bool = True,
        use_ratios: bool = False,  # NEW PARAMETER
        debug: bool = False  # NEW PARAMETER
) -> dict:
    """
    Run progressive analysis for both dual-fuel and pure alternative datasets.

    NEW: Supports ratio analysis via use_ratios parameter.

    Parameters
    ----------
    ...existing parameters...
    use_ratios : bool, default=False
        If True, test feature/matching_parameter ratios instead of absolute values
    debug : bool, default=False
        If True, print detailed debugging information

    Returns
    -------
    dict
        Dictionary with keys 'dual_fuel' and 'pure_alt_fuel' containing results
    """

    all_results = {}

    for alt_dataset in ['dual_fuel', 'pure_alt_fuel']:
        if alt_dataset not in fleet.processed_data:
            print(f"Warning: {alt_dataset} not found in fleet data")
            continue

        if print_results:
            print("\n" + "=" * 100)
            print(f"PROCESSING: {alt_dataset}")
            if use_ratios:
                print(f"MODE: Ratio Analysis (testing feature/{matching_parameter})")
            else:
                print("MODE: Absolute Value Analysis")
            print("=" * 100)

        try:
            results = progressive_comparison_analysis(
                fleet=fleet,
                conventional_dataset='pure_conventional',
                alternative_dataset=alt_dataset,
                test_features=test_features,
                matching_parameter=matching_parameter,
                ship_type=ship_type,
                n_matches_per_ship=10,
                K_S_test=False,
                M_W_test=True,
                welch_test=False,
                print_results=print_results,
                save_tables=True,
                output_dir=output_dir,
                variable_name_map=variable_name_map,
                use_ratios=use_ratios,  # NEW: Pass through
                debug=debug  # NEW: Pass through
            )

            all_results[alt_dataset] = results

            # Also save CSV
            output_path = Path(output_dir) / ship_type.name
            suffix = '_ratios' if use_ratios else '_absolute'
            csv_file = output_path / f'progressive_{alt_dataset}{suffix}.csv'
            results.to_csv(csv_file, index=False)

            if print_results:
                print(f"✓ Saved CSV: {csv_file}")

        except Exception as e:
            print(f"✗ Error processing {alt_dataset}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def format_progressive_comparison_table(progressive_df: pd.DataFrame) -> None:
    """
    Create a nicely formatted display of progressive comparison results.

    Parameters
    ----------
    progressive_df : pd.DataFrame
        Output from progressive_comparison_analysis
    """

    print("\n" + "=" * 100)
    print("FORMATTED PROGRESSIVE COMPARISON RESULTS")
    print("=" * 100)
    print("Significance threshold: α = 0.05")
    print("=" * 100)

    # Format for display
    for idx, row in progressive_df.iterrows():
        if idx % 2 == 0:  # New feature
            print(f"\n{'-' * 100}")

        # Feature and test
        display_parts = [f"{row['Feature']:<15}", f"{row['Test']:<12}"]

        # P-values and significance for each level
        for level in ['Raw', 'Overlap', 'Matched']:
            p_col = f'{level}_p'
            sig_col = f'{level}_sig'

            p_val = row[p_col]
            is_sig = row[sig_col]

            if pd.isna(p_val):
                p_str = "N/A    "
                sig_str = " "
            else:
                p_str = f"{p_val:6.4f}"
                sig_str = "✓" if is_sig else "✗"

            display_parts.append(p_str)
            display_parts.append(sig_str)

        print("  ".join(display_parts))

    print("=" * 100)
    print("\nLegend: ✓ = Significant (p < 0.05), ✗ = Not significant")
    print("=" * 100)


def plot_progressive_comparison_heatmap(
        progressive_df: pd.DataFrame,
        fuel_type: str,
        save_path: str = None
):
    """
    Create heatmap showing p-values across progressive comparison levels.

    Parameters
    ----------
    progressive_df : pd.DataFrame
        Output from progressive_comparison_analysis
    fuel_type : str
        Name for the fuel type (for title and filename)
    save_path : str, optional
        Path to save the figure
    """

    features = progressive_df['Feature'].unique()
    tests = progressive_df['Test'].unique()
    comparison_levels = ['Raw', 'Overlap', 'Matched']

    fig, axes = plt.subplots(1, len(tests), figsize=(15, 8))
    if len(tests) == 1:
        axes = [axes]

    for idx, test in enumerate(tests):
        # Filter for this test
        test_data = progressive_df[progressive_df['Test'] == test]

        # Create matrix
        matrix = np.zeros((len(features), len(comparison_levels)))
        for i, feature in enumerate(features):
            for j, level in enumerate(comparison_levels):
                p_col = f'{level}_p'
                value = test_data[test_data['Feature'] == feature][p_col].values
                if len(value) > 0:
                    matrix[i, j] = value[0]

        # Plot heatmap
        im = axes[idx].imshow(matrix, cmap=custom_cmap, aspect='auto', vmin=0, vmax=0.1)

        # Set ticks and labels
        axes[idx].set_xticks(np.arange(len(comparison_levels)))
        axes[idx].set_yticks(np.arange(len(features)))
        axes[idx].set_xticklabels(comparison_levels)
        axes[idx].set_yticklabels(features)

        # Rotate the tick labels for better readability
        plt.setp(axes[idx].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(features)):
            for j in range(len(comparison_levels)):
                text = axes[idx].text(j, i, f'{matrix[i, j]:.3f}',
                                      ha="center", va="center", color="black", fontsize=9)

        axes[idx].set_title(f'{test} Test\n{fuel_type}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Comparison Level', fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel('Feature', fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('p-value', rotation=270, labelpad=15)

    plt.suptitle(f'Progressive Comparison Analysis: {fuel_type}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path is None:
        save_path = f'progressive_comparison_heatmap_{fuel_type}.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {save_path}")
    plt.show()


def analyze_progressive_patterns(progressive_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze patterns in how significance changes across comparison levels.

    Parameters
    ----------
    progressive_df : pd.DataFrame
        Output from progressive_comparison_analysis

    Returns
    -------
    pd.DataFrame
        Summary of patterns
    """

    pattern_results = []

    for idx, row in progressive_df.iterrows():
        feature = row['Feature']
        test = row['Test']

        # Get significance at each level
        sig_raw = row['Raw_sig']
        sig_overlap = row['Overlap_sig']
        sig_matched = row['Matched_sig']

        # Get p-values
        p_raw = row['Raw_p']
        p_overlap = row['Overlap_p']
        p_matched = row['Matched_p']

        # Calculate p-value progression
        p_increase_raw_to_overlap = p_overlap - p_raw if not pd.isna(p_overlap) and not pd.isna(p_raw) else np.nan
        p_increase_overlap_to_matched = p_matched - p_overlap if not pd.isna(p_matched) and not pd.isna(
            p_overlap) else np.nan
        p_increase_total = p_matched - p_raw if not pd.isna(p_matched) and not pd.isna(p_raw) else np.nan

        # Classify pattern
        if sig_raw and sig_overlap and sig_matched:
            pattern_type = "PERSISTENT"
            interpretation = "Difference robust to all controls - real fuel type effect"
        elif sig_raw and sig_overlap and not sig_matched:
            pattern_type = "MATCHED_ELIMINATED"
            interpretation = "Size distribution explains difference"
        elif sig_raw and not sig_overlap and not sig_matched:
            pattern_type = "OVERLAP_ELIMINATED"
            interpretation = "Size range difference (not distribution)"
        elif not sig_raw and not sig_overlap and not sig_matched:
            pattern_type = "NEVER_SIGNIFICANT"
            interpretation = "No difference at any level"
        else:
            pattern_type = "INCONSISTENT"
            interpretation = "Unexpected pattern - investigate"

        pattern_results.append({
            'Feature': feature,
            'Test': test,
            'Raw_p': f"{p_raw:.4f}" if not pd.isna(p_raw) else "N/A",
            'Overlap_p': f"{p_overlap:.4f}" if not pd.isna(p_overlap) else "N/A",
            'Matched_p': f"{p_matched:.4f}" if not pd.isna(p_matched) else "N/A",
            'P_Increase_Total': f"{p_increase_total:.4f}" if not pd.isna(p_increase_total) else "N/A",
            'Pattern': pattern_type,
            'Interpretation': interpretation
        })

    pattern_df = pd.DataFrame(pattern_results)

    # Print summary
    print("\n" + "=" * 100)
    print("PATTERN ANALYSIS SUMMARY")
    print("=" * 100)
    print("\nPattern Distribution:")
    print(pattern_df['Pattern'].value_counts())
    print("\n" + "=" * 100)

    print(pattern_df.to_string(index=False))

    return pattern_df


def sensitivity_analysis_workflow(
        fleet: FleetData,
        conventional_dataset: str,
        alternative_dataset: str,
        ship_type: ShipType,
        test_features: list[str]=['LBP (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)'],
        matching_features:list[str] = ['DWT', 'LBP (M)', 'BEAM (M)'],
        K_S_test:bool=True,
        M_W_test:bool=True,
        welch_test:bool=False
):
    """
    Complete sensitivity analysis workflow.
    """


    # Define matching strategies to compare
    # Use features that make sense for matching


    # Run sensitivity analysis
    sensitivity_results = matching_sensitivity_analysis(
        fleet=fleet,
        conventional_dataset=conventional_dataset,
        alternative_dataset=alternative_dataset,
        test_features=test_features,
        matching_features=matching_features,
        n_matches_per_ship=10,
        K_S_test=K_S_test,
        M_W_test=M_W_test,
        welch_test=welch_test
    )

    # Save raw results
    output_file = f'sensitivity_analysis_{alternative_dataset}_{ship_type.name}_v2.csv'
    sensitivity_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Format for display
    format_sensitivity_table(sensitivity_results, matching_features)

    # Analyze robustness
    robustness_summary = analyze_sensitivity_robustness(
        sensitivity_results,
        matching_features
    )

    # Save robustness summary
    robustness_file = f'robustness_summary_{alternative_dataset}_{ship_type.name}_v2.csv'
    robustness_summary.to_csv(robustness_file, index=False)
    print(f"\nRobustness summary saved to: {robustness_file}")

    return sensitivity_results, robustness_summary


def plot_sensitivity_heatmap(sensitivity_df,
                             matching_features,
                             fuel_type):
    """
    Create heatmap showing p-values across matching strategies.
    """
    # Pivot for heatmap
    features = sensitivity_df['Feature'].unique()
    tests = sensitivity_df['Test'].unique()


    # Adjust figure width based on number of tests
    if len(tests) == 1:
        fig_width = 8  # Narrower for single test
    else:
        fig_width = 5 * len(tests)  # Scale with number of tests

    fig, axes = plt.subplots(1, len(tests), figsize=(fig_width, 6))
    if len(tests) == 1:
        axes = [axes]

    for idx, test in enumerate(tests):
        # Filter for this test
        test_data = sensitivity_df[sensitivity_df['Test'] == test]

        # Create matrix
        matrix = np.zeros((len(features), len(matching_features)))
        for i, feature in enumerate(features):
            for j, match_feat in enumerate(matching_features):
                p_col = f'Match_{match_feat}_p'
                value = test_data[test_data['Feature'] == feature][p_col].values
                if len(value) > 0:
                    matrix[i, j] = value[0]

        # Plot
        sns.heatmap(matrix,
                    annot=True,
                    fmt='.3f',
                    xticklabels=matching_features,
                    yticklabels=features,
                    cmap=custom_cmap,  # Red = low p (significant)
                    vmin=0,
                    vmax=0.1,
                    cbar_kws={'label': 'p-value'},
                    ax=axes[idx])

        axes[idx].set_title(f'{test} Test - {fuel_type}')
        axes[idx].set_xlabel('Matching Feature')
        axes[idx].set_ylabel('Tested Feature')

    plt.tight_layout()
    plt.savefig(f'sensitivity_heatmap_{fuel_type}_ver2.png', dpi=300, bbox_inches='tight')
    plt.show()



def main(make_plots: bool = False,
         block1: bool = False,
         progressive_comparison: bool = False,
         matching_sensitivity: bool = False,
         multi_ship_comparison: bool = True,
         run_diagnostics:bool=False):  # NEW PARAMETER

    # Initialize all variables to None
    progressive_pure = None
    sens_pure = None
    patterns_pure = None
    robust_pure = None

    progressive_dual = None
    sens_dual = None
    patterns_dual = None
    robust_dual = None

    # # Bulk carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    # ship_type=ShipType.BULK_CARRIER

    # # Ro-Pax carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX')
    # ship_type=ShipType.RO_PAX
    #
    # Container ship filepath
    types_list = [ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.CONTAINER_SHIP,ShipType.RO_PAX]

    for ship_type in types_list:
        bins, features, filepath, save_dir = get_variables(shiptype=ship_type)

        if run_diagnostics:
            print("\n" + "=" * 80)
            print("RUNNING DIAGNOSTIC ANALYSIS")
            print("=" * 80)

            # Option 1: Single ship type diagnostic
            results = run_diagnostic_analysis(
                ship_type=ShipType.BULK_CARRIER,
                get_variables_func=get_variables,
                getting_alt_fuel_ships_func=getting_alt_fuel_ships,
                output_dir='diagnostic_bulk_carrier'
            )

    #
    # # Bulk carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    # ship_type=ShipType.TANKER

        fleet = getting_alt_fuel_ships(filepath=filepath,)

        matching_features = ['DWT', 'LBP (M)', 'BEAM (M)']

        if ship_type == ShipType.CONTAINER_SHIP:
            matching_features.append('TEU')

        alt_datasets = ['dual_fuel','pure_alt_fuel']

        for alt_dataset in alt_datasets:

            if make_plots:
                create_plots(fleet=fleet,
                             conventional_df='pure_conventional',
                             alternative_df=alt_dataset,
                             parameters=features,
                             matching_param='DWT')


            if block1:

                individual_and_total_comparisons(fleet=fleet,
                                                     conventional_dataset='pure_conventional',
                                                     alternative_dataset=alt_dataset,
                                                     matching_parameter='DWT',
                                                     parameters=features,
                                                     ship_type=ship_type,
                                                     hotelling_t_test=False,
                                                     m_descrepancy=False,
                                                     welch_test=False,
                                                     K_S_test=False,)


            if matching_sensitivity:
                print("\n" + "=" * 80)
                print("RUNNING SENSITIVITY ANALYSIS")
                print("=" * 80)

                try:
                    # For pure alternative fuel ships
                    sens_pure, robust_pure = sensitivity_analysis_workflow(
                        fleet=fleet,
                        conventional_dataset='pure_conventional',
                        alternative_dataset=alt_dataset,
                        ship_type=ship_type,
                        test_features=features,
                        matching_features=matching_features,
                        M_W_test=True,
                        K_S_test=False,
                        welch_test=False
                    )

                    plot_sensitivity_heatmap(sensitivity_df=sens_pure,
                                            matching_features=matching_features,
                                            fuel_type='Pure_Alt_Fuel')
                except Exception as e:
                    print(f"ERROR in pure alt fuel sensitivity analysis: {e}")
                    import traceback
                    traceback.print_exc()

            if progressive_comparison:
                print("\n" + "=" * 80)
                print("RUNNING PROGRESSIVE COMPARISON ANALYSIS")
                print("=" * 80)

                variable_name_map = {
                    'LBP (M)': r'$L_{BP}$ (m)',
                    'LBP (M)/DWT': r'$L_{BP}/\text{DWT}$',  # NEW: Add ratio mappings
                    'BEAM (M)': 'Beam (m)',
                    'BEAM (M)/DWT': r'$B/\text{DWT}$',
                    'DRAFT (M)': 'Draft (m)',
                    'DRAFT (M)/DWT': r'$T/\text{DWT}$',
                    # ... etc
                }

                output_dir = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\experiment0'

                # Run ABSOLUTE value analysis
                print("\n" + "-" * 80)
                print("ABSOLUTE VALUE ANALYSIS")
                print("-" * 80)
                absolute_results = run_progressive_for_all_datasets(
                    fleet=fleet,
                    ship_type=ship_type,
                    test_features=features,
                    matching_parameter='DWT',
                    output_dir=output_dir,
                    variable_name_map=variable_name_map,
                    print_results=True,
                    use_ratios=False,  # Absolute values
                    debug=False
                )

                # Run RATIO analysis
                print("\n" + "-" * 80)
                print("RATIO ANALYSIS")
                print("-" * 80)
                ratio_results = run_progressive_for_all_datasets(
                    fleet=fleet,
                    ship_type=ship_type,
                    test_features=features,
                    matching_parameter='DWT',
                    output_dir=output_dir,
                    variable_name_map=variable_name_map,
                    print_results=True,
                    use_ratios=True,  # Test ratios
                    debug=True  # Enable debugging for first run
                )


        # ONLY export tables if we have data to export
        # Check for pure alt fuel results
        if any([progressive_pure is not None, sens_pure is not None,
                patterns_pure is not None, robust_pure is not None]):
            print("\n" + "=" * 80)
            print("EXPORTING PURE ALT FUEL TABLES")
            print("=" * 80)
            try:
                export_all_tables(
                    progressive_df=progressive_pure,
                    sensitivity_df=sens_pure,
                    pattern_df=patterns_pure,
                    robustness_df=robust_pure,
                    matching_features=matching_features,
                    output_dir=f"latex_tables_{ship_type.name}_pure_alt",
                    prefix=f"{alt_datasets}_fuel_{ship_type.name}"
                )
            except Exception as e:
                print(f"ERROR exporting pure alt fuel tables: {e}")
                import traceback
                traceback.print_exc()

        # Check for dual fuel results
        if any([progressive_dual is not None, sens_dual is not None,
                patterns_dual is not None, robust_dual is not None]):
            print("\n" + "=" * 80)
            print("EXPORTING DUAL FUEL TABLES")
            print("=" * 80)
            try:
                export_all_tables(
                    progressive_df=progressive_dual,
                    sensitivity_df=sens_dual,
                    pattern_df=patterns_dual,
                    robustness_df=robust_dual,
                    matching_features=matching_features,
                    output_dir=f"latex_tables_{ship_type.name}_dual",
                    prefix=f"dual_fuel_{ship_type.name}"
                )
            except Exception as e:
                print(f"ERROR exporting dual fuel tables: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        if multi_ship_comparison:
            print("\n" + "=" * 80)
            print("RUNNING MULTI-SHIP TYPE COMPARISON")
            print("=" * 80)


            ship_types_to_compare = [
                ShipType.BULK_CARRIER,
                ShipType.TANKER,
                ShipType.CONTAINER_SHIP,
                ShipType.RO_PAX
            ]

            output_dir=r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\experiment0'

            variable_name_map = {'LOA (M)':'$L_{OA} (m)$',
                                 'LBP (M)':'$L_{BP} (m)$',
                                 'BEAM (M)':'Beam (m)',
                                 'DRAFT (M)':'Draft (m)',
                                 'DEPTH (M)':'Depth (m)',
                                 'DWT':'DWT (t)'}

            results = run_multi_ship_comparison(
                ship_types=ship_types_to_compare,
                get_variables_func=get_variables,
                getting_alt_fuel_ships_func=getting_alt_fuel_ships,
                output_dir=output_dir,
                variable_name_map=variable_name_map,
                matching_parameter='DWT',
                n_matches_per_ship=10,
                use_ratios=False,
                conventional_type='overlap',
                bootstrap=False
            )



if __name__ == '__main__':
    main(make_plots=False,
         block1=False,
         matching_sensitivity=False,
         progressive_comparison=False,
         multi_ship_comparison=True,
         run_diagnostics=False)