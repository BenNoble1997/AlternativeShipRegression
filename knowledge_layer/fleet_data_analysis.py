from Operational_profile.data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
import math
import numpy as np
from scipy.stats import gaussian_kde

from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_regression_analyser_class import ParametricDomainAnalyzer
from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.random_forest_domain_analysis_clas import RandomForestDomainAnalyzer
from knowledge_layer.multi_dataset_analysers.domain_shift.model_comparison_analyser_class import ModelComparisonAnalyzer

from knowledge_layer.old_static_handlers.DWT_regression_analyser import (DWTConstrainedModelSelector)
from knowledge_layer.old_static_handlers.speed_impact_analyser import SpeedImpactAnalyzer
from knowledge_layer.old_static_handlers.feature_engineering_analysis import FeatureEngineeringAnalyzer
from knowledge_layer.old_static_handlers.linearvsrandomforest import RandomForestComparator
# from randomforestdomainshift import RandomForestDomainAnalyzer
# from randomforestdomainshift_class_v2 import RandomForestDomainAnalyzer,LinearVsRFDomainAnalyzer,BestLinearModelSelector

from parameters.shipshape.ShipShapeBasic_class import ShipType
from literature_regressions.Literature_formula_evaluation_class import LiteratureFormulaEvaluator
from literature_regressions.literature_formula_libraries_class import LiteratureFormulaLibrary



def wrap_title(title: str, max_chars: int = 40) -> str:
    """
    Wrap plot titles if they exceed a certain character length.
    Tries to break at natural points like 'vs' or before second set of variables.

    Args:
        title: The title string to wrap
        max_chars: Maximum characters before wrapping (default 40)

    Returns:
        Title with newline character if wrapping is needed
    """
    if len(title) <= max_chars:
        return title

    # Try to find good break points
    # Priority: 'vs', ' - ', ', ', ' '

    # First, try breaking at 'vs' if it exists and is roughly in the middle
    if ' vs ' in title:
        parts = title.split(' vs ')
        if len(parts) == 2:
            # Check if breaking here makes sense
            if len(parts[0]) <= max_chars and len(parts[1]) <= max_chars:
                return f"{parts[0]} vs\n{parts[1]}"

    # Try breaking at ' - '
    if ' - ' in title:
        parts = title.split(' - ', 1)
        if len(parts[0]) <= max_chars:
            return f"{parts[0]} -\n{parts[1]}"

    # Try breaking at comma
    if ', ' in title:
        parts = title.split(', ', 1)
        if len(parts[0]) <= max_chars:
            return f"{parts[0]},\n{parts[1]}"

    # Last resort: break at space closest to middle
    words = title.split()
    mid_point = len(title) // 2
    best_break = 0
    min_distance = float('inf')

    current_pos = 0
    for i, word in enumerate(words[:-1]):  # Don't break after last word
        current_pos += len(word) + 1  # +1 for space
        distance = abs(current_pos - mid_point)
        if distance < min_distance:
            min_distance = distance
            best_break = i + 1

    if best_break > 0:
        first_line = ' '.join(words[:best_break])
        second_line = ' '.join(words[best_break:])
        return f"{first_line}\n{second_line}"

    # If all else fails, return original
    return title



def load_fleet_data(filepath: str|Path) -> FleetData:

    fleet=FleetData(verbose=True)

    rename_dict = {'MMSI (MARITIME MOBILE SERVICE IDENTITY)':'MMSI',
                   'NUMBER OF SCREWS':'SCREWS',
                   'BEAM MLD (M)':'BEAM (M)',
                   'DEPTH MOULDED (M)':'DEPTH (M)',}

    fleet.import_data(file_path=filepath,
                      column_renames=rename_dict, )

    fleet.list_datasets()

    return fleet


# def test_contains_filter(fleet: FleetData, column: str, search_string: str):
#     """
#     Test if your CONTAINS filter is working by showing before/after unique values.
#     """
#
#     df = fleet.raw_data['raw1']
#
#     def get_unique_values(df: pd.DataFrame,
#                           column: str,
#                           include_nulls: bool = False,
#                           sort_results: bool = True) -> list:
#         """
#         Get all unique values from a specific column in a DataFrame.
#
#         :param df: The DataFrame to analyze
#         :param column: The column name to get unique values from
#         :param include_nulls: Whether to include NaN/None values in results
#         :param sort_results: Whether to sort the results alphabetically
#         :return: List of unique values
#         """
#
#
#         if column not in df.columns:
#             raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
#
#         # Get unique values
#         unique_vals = df[column].unique()
#
#         # Handle nulls
#         if not include_nulls:
#             # Filter out NaN, None, and pd.NA
#             unique_vals = [val for val in unique_vals if pd.notna(val)]
#
#         # Convert to list
#         result = list(unique_vals)
#
#         # Sort if requested
#         if sort_results:
#             try:
#                 result.sort()
#             except TypeError:
#                 # Handle case where values can't be compared (mixed types)
#                 result.sort(key=str)
#
#         return result
#
#     print(f"=== Testing CONTAINS filter for '{search_string}' in column '{column}' ===")
#
#     # Show original unique values
#     original_uniques = get_unique_values(df, column)
#     print(f"\nOriginal unique values ({len(original_uniques)}):")
#     for val in original_uniques:
#         marker = " ✓" if search_string in str(val) else ""
#         print(f"  - {val}{marker}")
#
#     # Apply your filter
#
#
#     contains_filter = fleet.create_filter(column, search_string, ComparisonOperator.CONTAINS)
#     filtered_df = fleet.process_tools.filter.apply_multiple_filters(df, contains_filter)
#
#     # Show filtered unique values
#     filtered_uniques = get_unique_values(filtered_df, column)
#     print(f"\nAfter CONTAINS filter ({len(filtered_uniques)}):")
#     for val in filtered_uniques:
#         print(f"  - {val}")
#
#     # Summary
#     expected_count = len([val for val in original_uniques if search_string in str(val)])
#     actual_count = len(filtered_uniques)
#
#     print(f"\n=== Results ===")
#     print(f"Expected unique values with '{search_string}': {expected_count}")
#     print(f"Actual unique values after filter: {actual_count}")
#     print(f"Filter working correctly: {expected_count == actual_count}")
#
#     return filtered_df


def getting_alt_fuel_ships(filepath: str|Path,
                           ship_type:str='BULK_CARRIER',):

    fleet_data = load_fleet_data(filepath)


    ratio1 = ('MAIN BUNKER CAPACITY (M3)','DWT')
    ratio2 = ('LOA (M)','DWT')
    ratio3 = ('TOTAL MAIN POWER (KW)','DWT')
    ratio4 = ('TOTAL MAIN POWER (KW)','MAIN BUNKER CAPACITY (M3)')
    ratio5 = ('LOA (M)','BEAM (M)')
    ratio6 = ('BEAM (M)','DRAFT (M)')


    ratio_dict = {'bunker capacity / DWT': ratio1,
                  'LOA (M) / DWT': ratio2,
                  'main engine kW / DWT': ratio3,
                  'main engine kW / bunker capacity': ratio4,
                  'Length/Beam': ratio5,
                  'Beam/Draft': ratio6}

    fleet_data.create_ratio_columns(datafile='raw1',
                                    columns=ratio_dict,
                                    name = 'with_ratios'
                                    )

    fleet_data.create_cb_column(datafile='with_ratios',
                                beam_column='BEAM (M)'
                                    )

    fuel_type_columns = ['MAIN ENGINE FUEL TYPE', 'AUXILIARY ENGINE FUEL TYPE']
    alt_fuels = ['LNG', 'Methanol', 'AMMONIA', 'Hydrogen', 'biofuel', 'cng']  # Fixed typo
    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    # First, create base dataset with fuel columns split


    fleet_data.processing_fleet_data(
        datafile='with_ratios',
        name='processed_with_ratios',
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
        powers_columns=['DWT'],
        power_value=3,
        products_columns=['LBP (M)','BEAM (M)','DEPTH (M)']
    )


    fleet_data.processing_fleet_data(
        datafile='processed_with_ratios',
        name='base_with_split_fuels',
        fuel_columns=fuel_type_columns,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
    )

    # Filter 1: Ships with ANY alternative fuel (includes dual-fuel)
    alt_fuel_filter = fleet_data.create_filter(
        column=fuel_type_columns,
        value=alt_fuels,
        operator=ComparisonOperator.CONTAINS,
        require_all=False
    )

    fleet_data.processing_fleet_data(
        datafile='base_with_split_fuels',
        name='any_alt_fuel',
        filter_by=alt_fuel_filter,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
    )

    # Filter 2: Ships with ONLY alternative fuels (no conventional fuels)
    # This requires checking that NO fuel columns contain conventional fuels
    conventional_fuel_filter = fleet_data.create_filter(
        column=['MAIN ENGINE FUEL TYPE_1', 'MAIN ENGINE FUEL TYPE_2',
                'AUXILIARY ENGINE FUEL TYPE_1', 'AUXILIARY ENGINE FUEL TYPE_2'],
        value=conventional_fuels,
        operator=ComparisonOperator.CONTAINS,
        require_all=False,
        negate=True  # Keep rows that DON'T contain conventional fuels
    )

    fleet_data.processing_fleet_data(
        datafile='any_alt_fuel',
        name='pure_alt_fuel',
        filter_by=conventional_fuel_filter,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
    )

    # Filter 3: Dual-fuel ships (have both alternative AND conventional)
    # Start with ships that have alt fuels, then keep only those WITH conventional
    conventional_fuel_filter_positive = fleet_data.create_filter(
        column=['MAIN ENGINE FUEL TYPE_1', 'MAIN ENGINE FUEL TYPE_2',
                'AUXILIARY ENGINE FUEL TYPE_1', 'AUXILIARY ENGINE FUEL TYPE_2'],
        value=conventional_fuels,
        operator=ComparisonOperator.CONTAINS,
        require_all=False,
        negate=False  # Keep rows that DO contain conventional fuels
    )

    fleet_data.processing_fleet_data(
        datafile='any_alt_fuel',
        name='dual_fuel',
        filter_by=conventional_fuel_filter_positive,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
    )

    # Filter 4: Pure conventional ships (no alternative fuels at all)
    fleet_data.processing_fleet_data(
        datafile='base_with_split_fuels',
        name='pure_conventional',
        filter_by=alt_fuel_filter,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],  # Same filter but we'll negate it...
    )
    # Actually, need to create negated version:
    no_alt_fuel_filter = fleet_data.create_filter(
        column=fuel_type_columns,
        value=alt_fuels,
        operator=ComparisonOperator.CONTAINS,
        require_all=False,
        negate=True  # Keep rows that DON'T contain alt fuels
    )

    fleet_data.processing_fleet_data(
        datafile='base_with_split_fuels',
        name='pure_conventional',
        filter_by=no_alt_fuel_filter,
        duplicates = ['LOA (M)','LBP (M)','BEAM (M)','DWT','DRAFT (M)'],
    )

    # Now remove dual-fuel ships from pure_conventional using reference filter
    fleet_data.filter_by_dataset(
        datafile='pure_conventional',
        column_name='MMSI',
        reference_values='dual_fuel',
    )

    # Print summary
    print("\n=== FLEET CATEGORIZATION SUMMARY ===")
    print(f"Total ships (after deduplication): {len(fleet_data.processed_data['base_with_split_fuels'])}")
    print(f"Ships with ANY alternative fuel: {len(fleet_data.processed_data['any_alt_fuel'])}")
    print(f"  - Pure alternative fuel only: {len(fleet_data.processed_data['pure_alt_fuel'])}")
    print(f"  - Dual-fuel (alt + conventional): {len(fleet_data.processed_data['dual_fuel'])}")
    print(f"Pure conventional fuel only: {len(fleet_data.processed_data['pure_conventional'])}")


    fleet_data.add_matched_dataset(alternative_df='pure_alt_fuel',
                                   conventional_df='pure_conventional',
                                   matching_features={'DWT':3.5, 'SPEED (KNOTS)':1},
                                   name='matched_dwt3.5_speed',
                                   n_matches_per_ship=10, )



    fleet_data.add_matched_dataset(alternative_df='pure_alt_fuel',
                                   conventional_df='pure_conventional',
                                   matching_features={'DWT':3.5, 'SPEED (KNOTS)':1},
                                   name='matched_dwt3.5_speed_type',
                                   n_matches_per_ship=10,
                                   binning_features=['TYPE'])

    fleet_data.add_matched_dataset(alternative_df='dual_fuel',
                                   conventional_df='pure_conventional',
                                   matching_features={'DWT': 3.5, 'SPEED (KNOTS)': 1},
                                   name='matched_dualfuel_dwt3.5',
                                   n_matches_per_ship=10,
                                   binning_features=['TYPE'])




    fleet_data.list_datasets()

    print(fleet_data.processed_data['base_with_split_fuels'].columns)

    return fleet_data


def plotting_box_plots(fleet: FleetData,
                       parameters: List[str],
                       name: str = 'box_plot',
                       conventional_df: str = 'pure_conventional',
                       alternative_df: str = 'pure_alt_fuel',
                       show_outliers: bool = True,
                       print_stats: bool = False,
                       alphas: List[float] = None,
                       use_different_colors: bool = True,
                       show_median_lines: bool = True,
                       show_mean_lines: bool = False,
                       shade_between: bool = True):  # New parameters

    conventional_fleet = fleet.processed_data[conventional_df]
    alternative_fleet = fleet.processed_data[alternative_df]

    # Validate alphas list
    if alphas is not None and len(alphas) != 2:
        print(
            f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets (2). Using default transparency.")
        alphas = None

    if alphas is None:
        alphas = [0.6, 0.8]

    # Set colors based on use_different_colors flag
    if use_different_colors:
        box_colors = ['blue', 'orange']  # Default distinct colors
    else:
        box_colors = ['blue', 'blue']  # Same color for both

    def get_stats(df: pd.Series, name: str):
        print(f'Stats for {name}: {df.describe()}')

    n_plots = len(parameters)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]

        conventional_data = conventional_fleet[param].dropna()
        alternative_data = alternative_fleet[param].dropna()

        if len(conventional_data) == 0 and len(alternative_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(wrap_title(f'{param} - No Data'))
            continue

        plot_data = []
        labels = []
        box_alphas = []
        plot_colors = []
        positions = []  # Track box positions

        current_pos = 1
        if len(conventional_data) > 0:
            plot_data.append(conventional_data)
            labels.append(conventional_df)
            box_alphas.append(alphas[0])
            plot_colors.append(box_colors[0])
            positions.append(current_pos)
            current_pos += 1

        if len(alternative_data) > 0:
            plot_data.append(alternative_data)
            labels.append(alternative_df)
            box_alphas.append(alphas[1])
            plot_colors.append(box_colors[1])
            positions.append(current_pos)

        box_plot = ax.boxplot(plot_data,
                              positions=positions,
                              tick_labels=labels,
                              showfliers=show_outliers,
                              patch_artist=True)

        # Set transparency and color for each box
        for patch, alpha, color in zip(box_plot['boxes'], box_alphas, plot_colors):
            patch.set_alpha(alpha)
            patch.set_facecolor(color)

        # Calculate statistics for both datasets
        stats = {}
        for j, (data, label) in enumerate(zip(plot_data, labels)):
            stats[label] = {
                'median': data.median(),
                'mean': data.mean(),
                'position': positions[j]
            }

        # Draw median/mean lines and shading if both datasets exist
        if len(plot_data) == 2:
            conv_label = labels[0]
            alt_label = labels[1]

            # Get x-axis limits for lines
            xlim = ax.get_xlim()

            # Draw median lines
            if show_median_lines:
                conv_median = stats[conv_label]['median']
                alt_median = stats[alt_label]['median']

                # Draw horizontal dotted lines
                ax.axhline(y=conv_median, color=plot_colors[0],
                           linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'{conv_label} median')
                ax.axhline(y=alt_median, color=plot_colors[1],
                           linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'{alt_label} median')

                # Shade between median lines
                if shade_between:
                    # Determine if alternative is higher or lower
                    if alt_median > conv_median:
                        shade_color = 'green'
                        shade_alpha = 0.15
                    else:
                        shade_color = 'red'
                        shade_alpha = 0.15

                    ax.fill_between(xlim, conv_median, alt_median,
                                    color=shade_color, alpha=shade_alpha,
                                    zorder=0)  # Behind everything else

            # Draw mean lines
            if show_mean_lines:
                conv_mean = stats[conv_label]['mean']
                alt_mean = stats[alt_label]['mean']

                # Draw horizontal dotted lines (different style from median)
                ax.axhline(y=conv_mean, color=plot_colors[0],
                           linestyle=':', linewidth=2, alpha=0.7,
                           label=f'{conv_label} mean')
                ax.axhline(y=alt_mean, color=plot_colors[1],
                           linestyle=':', linewidth=2, alpha=0.7,
                           label=f'{alt_label} mean')

                # Optional: shade between mean lines too (if median not shown)
                if shade_between and not show_median_lines:
                    if alt_mean > conv_mean:
                        shade_color = 'green'
                        shade_alpha = 0.15
                    else:
                        shade_color = 'red'
                        shade_alpha = 0.15

                    ax.fill_between(xlim, conv_mean, alt_mean,
                                    color=shade_color, alpha=shade_alpha,
                                    zorder=0)

            # Add small legend for the lines (positioned to not overlap with outlier box)
            if show_median_lines or show_mean_lines:
                ax.legend(loc='lower right', fontsize=7, framealpha=0.8)

        ax.set_title(wrap_title(f'{param} Distribution'))

        # Calculate outlier statistics
        outlier_text_lines = []

        for j, (data, label) in enumerate(zip(plot_data, labels)):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            lower_outliers = data[data < lower_fence]
            upper_outliers = data[data > upper_fence]

            outlier_info = []
            if len(lower_outliers) > 0:
                outlier_info.append(f'Low: {len(lower_outliers)}')
            if len(upper_outliers) > 0:
                outlier_info.append(f'High: {len(upper_outliers)}')

            total_outliers = len(lower_outliers) + len(upper_outliers)
            total_count = len(data)
            pct = (total_outliers / total_count) * 100 if total_count > 0 else 0

            if total_outliers > 0:
                detail_text = f' - {", ".join(outlier_info)}'
            else:
                detail_text = ''

            outlier_text_lines.append(f'{label}: {total_outliers} outliers ({pct:.1f}%){detail_text}')

        outlier_text = '\n'.join(outlier_text_lines)
        visibility_note = "" if show_outliers else " (hidden)"
        full_text = f"Outliers{visibility_note}:\n{outlier_text}"

        ax.text(0.02, 0.98, full_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8)

        if print_stats:
            if len(conventional_data) > 0:
                get_stats(conventional_data, f'{param} for conventional fuel')
            if len(alternative_data) > 0:
                get_stats(alternative_data, f'{param} for alternative fuel')


            print(f'Plot made for {param} (Conv: {len(conventional_data)}, Alt: {len(alternative_data)})')

    for i in range(n_plots, rows * cols):
        axes[i].axis('off')

    fig.suptitle(f'{name} Distribution')
    plt.tight_layout()
    plt.show()




def making_scatter_plots(fleet: FleetData,
                         dataframe_names: List[str],
                         parameters: List[str],
                         x_axis: str | List[str],
                         show_outliers: bool = True,
                         name: str = '',
                         print_stats: bool = False,
                         colors: List[str] = None,
                         alphas: List[float] = None,
                         use_different_colors: bool = True):  # Add color toggle
    """
    Create scatter plots comparing multiple ship datasets.

    Args:
        fleet: FleetData object with processed datasets
        dataframe_names: List of dataframe names to plot
        parameters: List of y-axis parameter names to plot
        x_axis: Single x-axis parameter name or list of x-axis parameters
        show_outliers: Whether to include outliers in plots
        name: Name for the overall plot title
        print_stats: Whether to print statistical summaries to console
        colors: Optional list of colors for each dataset (auto-generated if None)
        alphas: Optional list of transparency values (0-1) for each dataset
        use_different_colors: Whether to use different colors for each dataset (default True)
    """

    # Handle colors based on use_different_colors flag
    if use_different_colors:
        if colors is None:
            import matplotlib.cm as cm
            colors = [cm.tab10(i % 10) for i in range(len(dataframe_names))]
    else:
        colors = ['blue'] * len(dataframe_names)  # All same color

    # Validate and set alphas
    if alphas is not None and len(alphas) != len(dataframe_names):
        print(
            f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({len(dataframe_names)}). Using default transparency.")
        alphas = None

    if alphas is None:
        alphas = [0.6] * len(dataframe_names)

    x_axes = [x_axis] if isinstance(x_axis, str) else x_axis
    n_plots = len(parameters)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    def get_stats(df: pd.Series, name: str):
        if print_stats:
            print(f'Stats for {name}: {df.describe()}')

    for x_param in x_axes:
        print(f"\n=== Creating plots with {x_param} as X-axis ===")

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, y_param in enumerate(parameters):
            ax = axes[i]
            has_data = False

            for j, df_name in enumerate(dataframe_names):
                df = fleet.processed_data[df_name]
                df_clean = df[[x_param, y_param]].dropna()

                if len(df_clean) > 0:
                    has_data = True
                    ax.scatter(df_clean[x_param], df_clean[y_param],
                               alpha=alphas[j],
                               label=df_name,
                               color=colors[j],
                               s=20)

                    if print_stats:
                        get_stats(df_clean[y_param], f'{y_param} for {df_name}')
                        print(f'  {df_name}: {len(df_clean)} points')

            if not has_data:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title(wrap_title(f'{y_param} vs {x_param} - No Data'))
                continue

            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_title(wrap_title(f'{y_param} vs {x_param}'))
            ax.legend()
            ax.grid(True, alpha=0.3)

        for i in range(n_plots, rows * cols):
            axes[i].axis('off')

        fig.suptitle(f'{name} Parameter Relationships with {x_param}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()


def making_3d_scatter_plots(fleet: FleetData,
                            dataframe_names: List[str],
                            parameters: List[Tuple[str, str, str]],
                            show_outliers: bool = True,
                            name: str = '',
                            print_stats: bool = True,
                            colors: List[str] = None,
                            alphas: List[float] = None,
                            use_different_colors: bool = True):  # Add color toggle
    """
    Create 3D scatter plots comparing multiple ship datasets.

    Args:
        fleet: FleetData object with processed datasets
        dataframe_names: List of dataframe names to plot
        parameters: List of tuples (x_param, y_param, z_param) for each 3D plot
        show_outliers: Whether to include outliers in plots
        name: Name for the overall plot title
        print_stats: Whether to print statistical summaries to console
        colors: Optional list of colors for each dataset (auto-generated if None)
        alphas: Optional list of transparency values (0-1) for each dataset
        use_different_colors: Whether to use different colors for each dataset (default True)
    """

    # Handle colors based on use_different_colors flag
    if use_different_colors:
        if colors is None:
            import matplotlib.cm as cm
            colors = [cm.tab10(i % 10) for i in range(len(dataframe_names))]
    else:
        colors = ['blue'] * len(dataframe_names)  # All same color

    # Validate and set alphas
    if alphas is not None and len(alphas) != len(dataframe_names):
        print(
            f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({len(dataframe_names)}). Using default transparency.")
        alphas = None

    if alphas is None:
        alphas = [0.6] * len(dataframe_names)

    def get_stats(df: pd.Series, name: str):
        if print_stats:
            print(f'Stats for {name}: {df.describe()}')

    n_plots = len(parameters)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    fig = plt.figure(figsize=(7 * cols, 7 * rows))

    for i, (x_param, y_param, z_param) in enumerate(parameters):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        has_data = False

        for j, df_name in enumerate(dataframe_names):
            df = fleet.processed_data[df_name]
            df_clean = df[[x_param, y_param, z_param]].dropna()

            if len(df_clean) > 0:
                has_data = True
                ax.scatter(df_clean[x_param], df_clean[y_param], df_clean[z_param],
                           alpha=alphas[j],
                           label=df_name,
                           color=colors[j],
                           s=20)

                if print_stats:
                    get_stats(df_clean[z_param], f'{z_param} for {df_name}')
                    print(f'  {df_name}: {len(df_clean)} points')

        if not has_data:
            ax.text2D(0.5, 0.5, 'No data available',
                      ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_title(f'{z_param} vs {y_param} vs {x_param} - No Data'))
            continue

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_zlabel(z_param)
        ax.set_title(wrap_title(f'{z_param} vs {y_param} vs {x_param}'))
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{name} 3D Parameter Relationships', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def making_parallel_coordinates(fleet: FleetData,
                                dataframe_names: List[str],
                                parameters: List[str],
                                name: str = 'Parallel Coordinates',
                                colors: List[str] = None,
                                alphas: List[float] = None,
                                use_different_colors: bool = True,
                                normalize: bool = True,
                                print_stats: bool = True):
    """
    Create parallel coordinate plots comparing multiple ship datasets.

    Args:
        fleet: FleetData object with processed datasets
        dataframe_names: List of dataframe names to plot
        parameters: List of parameters to include as axes in the plot
        name: Title for the plot
        colors: Optional list of colors for each dataset
        alphas: Optional list of transparency values (0-1) for each dataset
        use_different_colors: Whether to use different colors for each dataset (default True)
        normalize: Whether to normalize data to 0-1 scale for better visualization (default True)
        print_stats: Whether to print statistics to console
    """

    # Handle colors
    if use_different_colors:
        if colors is None:
            import matplotlib.cm as cm
            colors = [cm.tab10(i % 10) for i in range(len(dataframe_names))]
    else:
        colors = ['blue'] * len(dataframe_names)

    # Validate and set alphas
    if alphas is not None and len(alphas) != len(dataframe_names):
        print(
            f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({len(dataframe_names)}). Using default transparency.")
        alphas = None

    if alphas is None:
        alphas = [0.3] * len(dataframe_names)  # Lower default alpha for parallel coords

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine global min/max for normalization if needed
    if normalize:
        global_mins = {}
        global_maxs = {}

        for param in parameters:
            all_values = []
            for df_name in dataframe_names:
                df = fleet.processed_data[df_name]
                if param in df.columns:
                    all_values.extend(df[param].dropna().values)

            if all_values:
                global_mins[param] = min(all_values)
                global_maxs[param] = max(all_values)
            else:
                global_mins[param] = 0
                global_maxs[param] = 1

    # Plot each dataset
    for idx, df_name in enumerate(dataframe_names):
        df = fleet.processed_data[df_name]

        # Select only the parameters that exist in the dataframe
        available_params = [p for p in parameters if p in df.columns]

        if not available_params:
            print(f"Warning: No parameters found in {df_name}")
            continue

        # Get clean data (drop rows with any NaN in selected parameters)
        plot_data = df[available_params].dropna()

        if len(plot_data) == 0:
            print(f"Warning: No complete data rows for {df_name}")
            continue

        if print_stats:
            print(f"\n{df_name}: {len(plot_data)} complete rows")
            print(f"Parameters: {available_params}")

        # Normalize data if requested
        if normalize:
            normalized_data = plot_data.copy()
            for param in available_params:
                min_val = global_mins[param]
                max_val = global_maxs[param]
                if max_val - min_val > 0:
                    normalized_data[param] = (plot_data[param] - min_val) / (max_val - min_val)
                else:
                    normalized_data[param] = 0.5  # If all values same, put in middle
            plot_data = normalized_data

        # Plot each row as a line
        for _, row in plot_data.iterrows():
            ax.plot(range(len(available_params)), row.values,
                    color=colors[idx], alpha=alphas[idx], linewidth=0.5)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], linewidth=2,
                              label=dataframe_names[i])
                       for i in range(len(dataframe_names))]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set up axes
    ax.set_xticks(range(len(parameters)))
    ax.set_xticklabels([wrap_title(p, max_chars=15) for p in parameters], rotation=45, ha='right')
    ax.set_xlim(-0.5, len(parameters) - 0.5)

    if normalize:
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Normalized Value (0-1)')

        # Add actual value ranges as text
        for i, param in enumerate(parameters):
            if param in global_mins and param in global_maxs:
                min_val = global_mins[param]
                max_val = global_maxs[param]
                # Add small text showing actual range
                ax.text(i, -0.1, f'[{min_val:.1f}-{max_val:.1f}]',
                        ha='center', fontsize=7, rotation=45)
    else:
        ax.set_ylabel('Value')

    ax.set_xlabel('Parameters')
    ax.set_title(f'{name} - Parallel Coordinates Plot')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def making_parallel_coordinates_separate(fleet: FleetData,
                                         dataframe_names: List[str],
                                         parameters: List[str],
                                         name: str = 'Parallel Coordinates',
                                         colors: List[str] = None,
                                         alphas: List[float] = None,
                                         use_different_colors: bool = True,
                                         normalize: bool = True,
                                         print_stats: bool = True):
    """
    Create separate parallel coordinate plots for each dataset (one subplot per dataset).

    Args:
        fleet: FleetData object with processed datasets
        dataframe_names: List of dataframe names to plot
        parameters: List of parameters to include as axes in the plot
        name: Title for the overall figure
        colors: Optional list of colors for each dataset
        alphas: Optional list of transparency values (0-1) for each dataset
        use_different_colors: Whether to use different colors for each dataset (default True)
        normalize: Whether to normalize data to 0-1 scale for better visualization (default True)
        print_stats: Whether to print statistics to console
    """

    # Handle colors
    if use_different_colors:
        if colors is None:
            import matplotlib.cm as cm
            colors = [cm.tab10(i % 10) for i in range(len(dataframe_names))]
    else:
        colors = ['blue'] * len(dataframe_names)

    # Validate and set alphas
    if alphas is not None and len(alphas) != len(dataframe_names):
        print(
            f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({len(dataframe_names)}). Using default transparency.")
        alphas = None

    if alphas is None:
        alphas = [0.3] * len(dataframe_names)

    # Calculate layout
    n_plots = len(dataframe_names)
    cols = min(2, n_plots)  # Max 2 columns
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 6 * rows))

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine global min/max for normalization if needed
    if normalize:
        global_mins = {}
        global_maxs = {}

        for param in parameters:
            all_values = []
            for df_name in dataframe_names:
                df = fleet.processed_data[df_name]
                if param in df.columns:
                    all_values.extend(df[param].dropna().values)

            if all_values:
                global_mins[param] = min(all_values)
                global_maxs[param] = max(all_values)
            else:
                global_mins[param] = 0
                global_maxs[param] = 1

    # Plot each dataset in its own subplot
    for idx, df_name in enumerate(dataframe_names):
        ax = axes[idx]
        df = fleet.processed_data[df_name]

        # Select only the parameters that exist in the dataframe
        available_params = [p for p in parameters if p in df.columns]

        if not available_params:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{df_name} - No Data')
            continue

        # Get clean data
        plot_data = df[available_params].dropna()

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No complete rows', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{df_name} - No Data')
            continue

        if print_stats:
            print(f"\n{df_name}: {len(plot_data)} complete rows")

        # Normalize data if requested
        if normalize:
            normalized_data = plot_data.copy()
            for param in available_params:
                min_val = global_mins[param]
                max_val = global_maxs[param]
                if max_val - min_val > 0:
                    normalized_data[param] = (plot_data[param] - min_val) / (max_val - min_val)
                else:
                    normalized_data[param] = 0.5
            plot_data = normalized_data

        # Plot each row as a line
        for _, row in plot_data.iterrows():
            ax.plot(range(len(available_params)), row.values,
                    color=colors[idx], alpha=alphas[idx], linewidth=0.5)

        # Set up axes
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels([wrap_title(p, max_chars=15) for p in available_params],
                           rotation=45, ha='right')
        ax.set_xlim(-0.5, len(available_params) - 0.5)

        if normalize:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Normalized Value (0-1)')
        else:
            ax.set_ylabel('Value')

        ax.set_xlabel('Parameters')
        ax.set_title(f'{df_name} ({len(plot_data)} ships)')
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{name} - Parallel Coordinates Plots', fontsize=16, y=0.99)
    plt.tight_layout()
    plt.show()


def split_and_plot(fleet: FleetData,
                   plot_function: callable,
                   split_column: str,
                   n_bins: int,
                   bin_labels: List[str] = None,
                   **plot_kwargs):
    """
    Wrapper function that splits data into bins and creates subplots for each bin.

    :param fleet: FleetData object with processed datasets
    :param plot_function: The plotting function to use (making_scatter_plots, making_parallel_coordinates, etc.)
    :param split_column: Column name to split data by
    :param n_bins: Number of bins to split data into
    :param bin_labels: Optional custom labels for bins (e.g., ['Small', 'Medium', 'Large'])
    :param plot_kwargs: All other arguments to pass to the plotting function
    """

    # Get the dataframe names from plot_kwargs
    if 'dataframe_names' in plot_kwargs:
        dataframe_names = plot_kwargs['dataframe_names']
    elif 'conventional_df' in plot_kwargs and 'alternative_df' in plot_kwargs:
        # For box plots or other functions using conventional/alternative
        dataframe_names = [plot_kwargs['conventional_df'], plot_kwargs['alternative_df']]
    else:
        raise ValueError("Could not determine dataframe names from plot_kwargs")

    # Collect all data for the split column to determine bin edges
    all_split_data = []
    for df_name in dataframe_names:
        df = fleet.processed_data[df_name]
        if split_column in df.columns:
            all_split_data.extend(df[split_column].dropna().values)

    if not all_split_data:
        print(f"Error: No data found in column '{split_column}'")
        return

    # Create bins using equal-width for comparing ships of similar sizes
    bin_edges = np.linspace(min(all_split_data), max(all_split_data), n_bins + 1)

    # Ensure unique bin edges (in case of repeated values)
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    if actual_n_bins < n_bins:
        print(f"Warning: Only {actual_n_bins} unique bins possible (requested {n_bins})")

    # Generate bin labels if not provided
    if bin_labels is None:
        bin_labels = [f'{split_column} [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]'
                      for i in range(actual_n_bins)]

    print(f"\nSplitting data by {split_column} into {actual_n_bins} bins:")
    for i, label in enumerate(bin_labels):
        print(f"  Bin {i + 1}: {label}")

    # IMPORTANT: Save the original name BEFORE the loop
    original_name = plot_kwargs.get('name', '')

    # Create a temporary FleetData object for each bin
    for bin_idx in range(actual_n_bins):
        print(f"\n{'=' * 60}")
        print(f"Processing Bin {bin_idx + 1}/{actual_n_bins}: {bin_labels[bin_idx]}")
        print(f"{'=' * 60}")

        # Create filtered dataframes for this bin
        bin_fleet = FleetData()  # Create new FleetData instance
        bin_fleet.processed_data = {}

        lower_bound = bin_edges[bin_idx]
        upper_bound = bin_edges[bin_idx + 1]

        # Include upper bound for last bin
        if bin_idx == actual_n_bins - 1:
            mask_func = lambda x: (x >= lower_bound) & (x <= upper_bound)
        else:
            mask_func = lambda x: (x >= lower_bound) & (x < upper_bound)

        bin_empty = True
        for df_name in dataframe_names:
            df = fleet.processed_data[df_name]

            if split_column in df.columns:
                # Filter data for this bin
                mask = mask_func(df[split_column])
                bin_df = df[mask].copy()

                if len(bin_df) > 0:
                    bin_fleet.processed_data[df_name] = bin_df
                    bin_empty = False
                    print(f"  {df_name}: {len(bin_df)} ships")
                else:
                    # Add empty dataframe to maintain structure
                    bin_fleet.processed_data[df_name] = df.iloc[0:0].copy()
                    print(f"  {df_name}: 0 ships")
            else:
                # If split column doesn't exist, include all data
                bin_fleet.processed_data[df_name] = df.copy()

        if bin_empty:
            print(f"  Skipping empty bin")
            continue

        # Update the plot name to include bin info using the SAVED original_name
        plot_kwargs['name'] = f"{original_name} - {bin_labels[bin_idx]}"

        # Call the plotting function with the binned data
        plot_function(fleet=bin_fleet, **plot_kwargs)


def making_butterfly_plots(fleet: FleetData,
                           parameters: List[str],
                           show_outliers: bool = True,
                           print_stats: bool = True,
                           bins: int = 30):
    """
    Create butterfly plots comparing conventional vs alternative fuel ships.

    Args:
        fleet: FleetData object with processed datasets
        parameters: List of parameter names to plot
        show_outliers: Whether to include outliers in plots
        print_stats: Whether to print statistical summaries to console
        bins: Number of bins for histograms
    """

    conventional_fleet = fleet.processed_data['processed3']
    alternative_fleet = fleet.processed_data['alt_fuels']

    def get_stats(df: pd.Series, name: str):
        if print_stats:
            print(f'Stats for {name}: {df.describe()}')

    # Create subplots for each parameter
    n_plots = len(parameters)

    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    # Create subplots with calculated layout
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]

        # Clean data - remove NaN values
        conventional_data = conventional_fleet[param].dropna()
        alternative_data = alternative_fleet[param].dropna()

        # Check if we have data to plot
        if len(conventional_data) == 0 and len(alternative_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'{param} - No Data')
            continue

        # Handle outliers if requested
        if not show_outliers:
            for data_name, data in [('conventional', conventional_data), ('alternative', alternative_data)]:
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_fence = Q1 - 1.5 * IQR
                    upper_fence = Q3 + 1.5 * IQR

                    if data_name == 'conventional':
                        conventional_data = data[(data >= lower_fence) & (data <= upper_fence)]
                    else:
                        alternative_data = data[(data >= lower_fence) & (data <= upper_fence)]

        # Find common range for both datasets
        if len(conventional_data) > 0 and len(alternative_data) > 0:
            min_val = min(conventional_data.min(), alternative_data.min())
            max_val = max(conventional_data.max(), alternative_data.max())
        elif len(conventional_data) > 0:
            min_val, max_val = conventional_data.min(), conventional_data.max()
        else:
            min_val, max_val = alternative_data.min(), alternative_data.max()

        # Create histogram bins
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate histograms
        if len(conventional_data) > 0:
            conv_counts, _ = np.histogram(conventional_data, bins=bin_edges)
            # Plot conventional on left side (negative x-values)
            ax.barh(bin_edges[:-1], -conv_counts, height=(max_val - min_val) / bins,
                    alpha=0.7, color='blue', label='Conventional')

        if len(alternative_data) > 0:
            alt_counts, _ = np.histogram(alternative_data, bins=bin_edges)
            # Plot alternative on right side (positive x-values)
            ax.barh(bin_edges[:-1], alt_counts, height=(max_val - min_val) / bins,
                    alpha=0.7, color='red', label='Alternative')

        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.8)

        # Set labels and title
        ax.set_ylabel(param)
        ax.set_xlabel('Frequency (Conventional ← | → Alternative)')
        ax.set_title(f'{param} Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistical annotations
        if len(conventional_data) > 0 and len(alternative_data) > 0:
            conv_median = conventional_data.median()
            alt_median = alternative_data.median()

            # Add median lines
            max_freq = max(max(conv_counts) if len(conventional_data) > 0 else 0,
                           max(alt_counts) if len(alternative_data) > 0 else 0)

            ax.plot([-max_freq * 0.1, 0], [conv_median, conv_median],
                    color='darkblue', linewidth=2, alpha=0.8)
            ax.plot([0, max_freq * 0.1], [alt_median, alt_median],
                    color='darkred', linewidth=2, alpha=0.8)

            # Add text box with comparison stats
            stats_text = f'Medians:\nConv: {conv_median:.0f}\nAlt: {alt_median:.0f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8)

        # Print stats
        if len(conventional_data) > 0:
            get_stats(conventional_data, f'{param} for conventional fuel')
        if len(alternative_data) > 0:
            get_stats(alternative_data, f'{param} for alternative fuel')

        if print_stats:
            print(f'Butterfly plot made for {param} (Conv: {len(conventional_data)}, Alt: {len(alternative_data)})')

    # Hide unused subplots
    for i in range(n_plots, rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def making_density_plots(fleet: FleetData,
                         conventional_df:str,
                         alternative_df:str,
                         parameters: List[str],
                         show_outliers: bool = True,
                         print_stats: bool = True,
                         fill_alpha: float = 0.3):
    """
    Create density plots comparing conventional vs alternative fuel ships.

    Args:
        fleet: FleetData object with processed datasets
        parameters: List of parameter names to plot
        show_outliers: Whether to include outliers in density calculation
        print_stats: Whether to print statistical summaries to console
        fill_alpha: Transparency for filled density areas
    """

    conventional_fleet = fleet.processed_data[conventional_df]
    alternative_fleet = fleet.processed_data[alternative_df]

    def get_stats(df: pd.Series, name: str):
        if print_stats:
            print(f'Stats for {name}: {df.describe()}')

    # Calculate layout dynamically
    # Create subplots for each parameter
    n_plots = len(parameters)

    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    # Create subplots with calculated layout
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]

        # Clean data - remove NaN values
        conventional_data = conventional_fleet[param].dropna()
        alternative_data = alternative_fleet[param].dropna()

        # Check if we have data to plot
        if len(conventional_data) == 0 and len(alternative_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'{param} - No Data')
            continue

        # Handle outliers if requested
        outlier_info = []
        if not show_outliers:
            for data_name, data in [('Conventional', conventional_data), ('Alternative', alternative_data)]:
                if len(data) > 0:
                    original_count = len(data)
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_fence = Q1 - 1.5 * IQR
                    upper_fence = Q3 + 1.5 * IQR

                    filtered_data = data[(data >= lower_fence) & (data <= upper_fence)]
                    outliers_removed = original_count - len(filtered_data)

                    if data_name == 'Conventional':
                        conventional_data = filtered_data
                    else:
                        alternative_data = filtered_data

                    if outliers_removed > 0:
                        outlier_info.append(f'{data_name}: {outliers_removed} outliers removed')

        # Check again after outlier removal
        if len(conventional_data) == 0 and len(alternative_data) == 0:
            ax.text(0.5, 0.5, 'No data after outlier removal', ha='center', va='center')
            ax.set_title(f'{param} - No Data')
            continue

        # Find common range for density estimation
        all_data = []
        if len(conventional_data) > 0:
            all_data.extend(conventional_data)
        if len(alternative_data) > 0:
            all_data.extend(alternative_data)

        min_val = min(all_data)
        max_val = max(all_data)
        x_range = np.linspace(min_val, max_val, 300)

        # Plot conventional data density
        if len(conventional_data) >= 2:  # Need at least 2 points for KDE
            try:
                conv_kde = gaussian_kde(conventional_data)
                conv_density = conv_kde(x_range)

                ax.plot(x_range, conv_density, color='blue', linewidth=2,
                        label=f'Conventional (n={len(conventional_data)})')
                ax.fill_between(x_range, conv_density, alpha=fill_alpha, color='blue')

                # Add median line
                conv_median = conventional_data.median()
                max_density = conv_density.max()
                ax.axvline(conv_median, color='darkblue', linestyle='--', alpha=0.8,
                           label=f'Conv. Median: {conv_median:.0f}')

            except Exception as e:
                print(f"Warning: Could not create density plot for conventional {param}: {e}")

        # Plot alternative fuel data density
        if len(alternative_data) >= 2:  # Need at least 2 points for KDE
            try:
                alt_kde = gaussian_kde(alternative_data)
                alt_density = alt_kde(x_range)

                ax.plot(x_range, alt_density, color='red', linewidth=2,
                        label=f'Alternative (n={len(alternative_data)})')
                ax.fill_between(x_range, alt_density, alpha=fill_alpha, color='red')

                # Add median line
                alt_median = alternative_data.median()
                ax.axvline(alt_median, color='darkred', linestyle='--', alpha=0.8,
                           label=f'Alt. Median: {alt_median:.0f}')

            except Exception as e:
                print(f"Warning: Could not create density plot for alternative {param}: {e}")

        # Handle case where one dataset has < 2 points
        if len(conventional_data) == 1:
            ax.axvline(conventional_data.iloc[0], color='blue', linewidth=3, alpha=0.7,
                       label=f'Conventional (single point)')
        if len(alternative_data) == 1:
            ax.axvline(alternative_data.iloc[0], color='red', linewidth=3, alpha=0.7,
                       label=f'Alternative (single point)')

        # Set labels and formatting
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.set_title(f'{param} Density Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add outlier information if any
        if outlier_info:
            outlier_text = '\n'.join(outlier_info)
            ax.text(0.02, 0.98, outlier_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8)

        # Print stats
        if len(conventional_data) > 0:
            get_stats(conventional_data, f'{param} for conventional fuel')
        if len(alternative_data) > 0:
            get_stats(alternative_data, f'{param} for alternative fuel')

        if print_stats:
            print(f'Density plot made for {param} (Conv: {len(conventional_data)}, Alt: {len(alternative_data)})')

    # Hide unused subplots
    for i in range(n_plots, rows * cols):
        axes[i].axis('off')

    fig.suptitle(f'{param} Density Distribution')
    plt.tight_layout()
    plt.show()


def create_matched_conventional_dataset(fleet: FleetData,
                                        conventional_df:str,
                                        alternative_df:str,
                                        matching_features: List[str] = ['DWT', 'SPEED (KNOTS)'],
                                        n_matches_per_ship: int = 10,):
    """
    Create a dataset of conventional ships matched to alternative fuel ships.

    Returns:
        matched_conventional: DataFrame of matched conventional ships (deduplicated)
        match_details: List of dictionaries with matching information
    """

    conventional_fleet = fleet.processed_data[conventional_df]
    alternative_fleet = fleet.processed_data[alternative_df]

    # Clean data - only keep rows with all matching features
    conv_clean = conventional_fleet[matching_features].dropna()
    alt_clean = alternative_fleet[matching_features].dropna()

    all_matched_indices = []
    match_details = []

    for alt_idx, alt_ship in alt_clean.iterrows():
        # Calculate similarity scores for all conventional ships
        similarity_scores = []

        for conv_idx, conv_ship in conv_clean.iterrows():
            score = 0
            valid_comparison = True

            # Calculate relative differences for each feature
            for feature in matching_features:
                alt_val = alt_ship[feature]
                conv_val = conv_ship[feature]

                if alt_val == 0:  # Skip if alternative ship has zero value
                    valid_comparison = False
                    break

                relative_diff = ((conv_val - alt_val) / alt_val) ** 2
                score += relative_diff

            if valid_comparison:
                final_score = np.sqrt(score)
                similarity_scores.append((conv_idx, final_score))

        # Sort by similarity score (lower = more similar) and take top N
        similarity_scores.sort(key=lambda x: x[1])
        top_matches = similarity_scores[:n_matches_per_ship]

        # Store the indices and details
        for conv_idx, score in top_matches:
            all_matched_indices.append(conv_idx)
            match_details.append({
                'alt_ship_idx': alt_idx,
                'conv_ship_idx': conv_idx,
                'similarity_score': score,
                'alt_ship_name': alternative_fleet.loc[
                    alt_idx, 'NAME'] if 'NAME' in alternative_fleet.columns else f'Alt_{alt_idx}',
                'conv_ship_name': conventional_fleet.loc[
                    conv_idx, 'NAME'] if 'NAME' in conventional_fleet.columns else f'Conv_{conv_idx}'
            })

    # Create matched dataset and remove duplicates
    matched_indices_unique = list(set(all_matched_indices))
    matched_conventional = conventional_fleet.loc[matched_indices_unique].copy()

    print(f"Found {len(all_matched_indices)} total matches")
    print(f"After removing duplicates: {len(matched_indices_unique)} unique conventional ships")
    print(f"Matching {len(alt_clean)} alternative fuel ships")

    return matched_conventional, match_details



def analysis_function(fleet:FleetData,
                      ship_type:str,
                      parameters:List[str],
                      conventional_df:str,
                      alternative_df:str,
                      split_column:str='DWT',
                      bins:int=4,
                      scatter_diagrams:bool=False):


    # filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier\Listing_2025091710537 2016-2025 status in service or Idle.csv')



    parameters1 = ['bunker capacity / DWT', 'LOA (M) / DWT', 'main engine kW / DWT',
       'main engine kW / bunker capacity']

    parameters2 = ['ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)', 'LOA (M)','BEAM (M)','Length/Beam','Beam/Draft']

    conventional = 'conventional'
    alternative = 'alt_fuels'
    matched_dwt_speed = 'matched_dwt_speed'
    matched_dwt = 'matched_dwt'
    matched_dwt_weighted = 'matched_dwt_weighted'
    matched_dwt_speed = 'matched_dwt3.5_speed'

    #
    plotting_box_plots(fleet=fleet,
                       parameters=parameters,
                       conventional_df=conventional_df,
                       alternative_df=alternative_df,
                       name=f'{ship_type} Box plot - conventional vs alternative full',
                       show_outliers=False)

    split_and_plot(fleet=fleet,
                   plot_function=plotting_box_plots,
                   split_column = 'DWT',
                   n_bins=4,
                parameters=parameters1,
                conventional_df=conventional,
                name='conventional fuel ships',
                alternative_df=alternative,
                   show_outliers=False)

    split_and_plot(fleet=fleet,
                   plot_function=plotting_box_plots,
                   split_column = split_column,
                   n_bins=bins,
                parameters=parameters,
                conventional_df=conventional_df,
                       print_stats=False,
                alternative_df=alternative_df,
                   show_outliers=False)


    # plotting_box_plots(fleet=fleet,
    #                    parameters=parameters2,
    #                    conventional_df=matched_dwt,
    #                    name = 'matched conventional fuel ships',
    #                    alternative_df=alternative,
    #                    print_stats=True,
    #                    show_outliers=False)


    if scatter_diagrams:
        x_axis = ['DWT','LOA (M)']


        making_scatter_plots(fleet=fleet,
                           dataframe_names=[conventional_df, alternative_df],
                             parameters=parameters,
                             x_axis=x_axis,
                             print_stats=True,
                             show_outliers=False,
                             name=ship_type,
                             alphas = [0.5,1])

        split_and_plot(fleet=fleet,
                       plot_function=making_scatter_plots,
                       split_column = split_column,
                       n_bins=bins,
                       dataframe_names=[conventional_df, alternative_df],
                       parameters=parameters,
                       x_axis=x_axis,
                       print_stats=True,
                       show_outliers=False,
                       name=ship_type,
                       alphas=[0.5, 1]
                       )

    # making_3d_scatter_plots(fleet=fleet,
    #                         dataframe_names=[matched_dwt, alternative],
    #                         parameters=[('DWT','BEAM MLD (M)', 'LOA (M)')],
    #                         name = ship_type,
    #                         alphas = [0.5,1]
    #                         )

    # making_parallel_coordinates(fleet=fleet,
    #                             dataframe_names=[conventional, alternative],
    #                             parameters=parameters1,
    #                             )
    #
    # making_parallel_coordinates(fleet=fleet,
    #                             dataframe_names=[matched_dwt_speed, alternative],
    #                             parameters=parameters1,
    #                             )
    #
    # split_and_plot(plot_function=making_parallel_coordinates,
    #                fleet=fleet,
    #                split_column='DWT',
    #                n_bins=4,
    #                dataframe_names=[matched_dwt_speed, alternative],
    #                parameters=parameters1,
    #                )
    #



    # making_butterfly_plots(fleet=fleet, parameters=parameters)
    #
    # making_density_plots(fleet=fleet,
    #                    conventional_df=conventional,
    #                    alternative_df=alternative,
    #                      parameters=parameters,
    #                      )
    #
    # making_density_plots(fleet=fleet,
    #                    conventional_df=conventional,
    #                    alternative_df=alternative,
    #                      parameters=parameters2,
    #                      )


    # test_contains_filter(fleet=fleet,
    #                      column = 'MAIN ENGINE FUEL TYPE',
    #                      search_string = 'LNG',)

def full_ship_analysis(filepath:Path,
                       ship_type:str,
                       scatter_plots:bool=False):

    conventional = 'pure_conventional'
    alternative = 'pure_alt_fuel'
    matched_dwt_speed = 'matched_dwt_speed'
    matched_dwt = 'matched_dwt'
    matched_dwt_speed = 'matched_dwt_speed_weighted'

    parameters1 = ['bunker capacity / DWT', 'LOA (M) / DWT', 'main engine kW / DWT',
                   'main engine kW / bunker capacity']

    parameters2 = ['ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)', 'LOA (M)', 'BEAM MLD (M)','C_B', 'Length/Beam',
                   'Beam/Draft']

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type)

    analysis_function(
        fleet=fleet,
        ship_type=ship_type,
        parameters=parameters1,
        conventional_df=conventional,
        alternative_df=alternative,
    scatter_diagrams=scatter_plots)

    analysis_function(
        fleet=fleet,
        ship_type=ship_type,
        parameters=parameters2,
        conventional_df=conventional,
        alternative_df=alternative,
    scatter_diagrams=scatter_plots)

    analysis_function(
        fleet=fleet,
        ship_type=ship_type,
        parameters=parameters1,
        conventional_df=matched_dwt_speed,
        alternative_df=alternative,
    scatter_diagrams=scatter_plots)

    analysis_function(
        fleet=fleet,
        ship_type=ship_type,
        parameters=parameters2,
        conventional_df=matched_dwt_speed,
        alternative_df=alternative,
    scatter_diagrams=scatter_plots)

def poly_regression_selector(filepath:Path,
                        conventional_df:pd.DataFrame,
                        reccomendations:Path = None,
                        variable_dimensions:List[str]=None,
                        target_dimensions:List[str]=None,
                        cv_method:str|None=None,
                        ranking_att ='rrmse',
                        fixed_dimensions:List[str]=['DWT']):


    if variable_dimensions is None:
        variable_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)']

    if target_dimensions is None:
        target_dimensions = variable_dimensions

    # STAGE 1: Find best models (DWT always included)
    print("STAGE 1: DWT-CONSTRAINED MODEL SELECTION")
    print("=" * 100)
    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variable_dimensions,
                                            target_dimensions=target_dimensions,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='rrmse')

    model_dict = poly_selector.analyze_all_combinations(max_additional_predictors=2, cv_method=cv_method)

    # Export and visualize
    if reccomendations is not None:
        poly_selector.export_recommendations(reccomendations)
    poly_selector.plot_metrics_comparison(fixed_parameters=fixed_dimensions,
                                     cv_method=cv_method,
                                     metric=ranking_att,
                                     )

    poly_selector.plot_cv_stability(fixed_parameters=fixed_dimensions)

    print("\n\nSTAGE 2: DWT-CONSTRAINED random forest regression selection")
    print("=" * 100)

    rf_selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variable_dimensions,
        target_dimensions=target_dimensions,
        fixed_dimensions=['DWT'],
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        selection_metric='rrmse',
        verbose=True
    )

    # Analyze all combinations
    rf_results = rf_selector.analyze_all_combinations(
        max_additional_predictors=2,
        cv_method='stratified',
        n_folds=10,
        random_state=42
    )

    # Print summary
    rf_selector.print_summary()

    # ============================================================================
    # 2.2 PLOTTING
    # ============================================================================

    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='rrmse',
    )

    # Plot feature importance
    rf_selector.plot_feature_importance(
    )


def DomainShiftAnalysis(filepath:Path,
                        conventional_df:pd.DataFrame,
                        conventional_matched_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        ship_type:str='BULK CARRIER',
                        reccomendations:Path = None,
                        variable_dimensions:List[str]=None,
                        target_dimensions:List[str]=None,
                        cv_method:str|None=None,
                        ranking_att ='rrmse',
                        fixed_dimensions:List[str]=['DWT'],
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,):


    if variable_dimensions is None:
        variable_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)']

    if target_dimensions is None:
        target_dimensions = variable_dimensions

    # STAGE 1: Find best models (DWT always included)
    print("STAGE 1: POLYNOMIAL REGRESSION MODEL SELECTION")
    print("=" * 100)
    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variable_dimensions,
                                            target_dimensions=target_dimensions,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='adj_r2',
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization)

    model_dict = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

    # Export and visualize
    if reccomendations is not None:
        poly_selector.export_recommendations(reccomendations)

    poly_selector.plot_metrics_comparison(
                                     cv_method=cv_method,
                                     metric=ranking_att,
                                     )

    poly_selector.plot_metrics_comparison(
                                     cv_method=cv_method,
                                     )

    poly_selector.plot_cv_stability()

    print("\n\nSTAGE 2: POLYNOMIAL DOMAIN SHIFT ANALYSIS")
    print("=" * 100)

    poly_analyzer = ParametricDomainAnalyzer(
        conventional_data=conventional_df,  # Will be split 80/20
        conventional_matched=conventional_matched_df,
        alternative_data=alternative_df,
        independent_variables=variable_dimensions,
        dependent_variables=target_dimensions,
        model_recommendations=model_dict,  # Use best models from step 1
        test_size=0.2,
        random_state=42
    )

    # Step 3: Analyze
    results = poly_analyzer.analyze_all_combinations(max_predictors=2)

    # Step 4: Interpret
    poly_analyzer.interpret_results(significance_threshold_r2=0.05)

    poly_analyzer.plot_domain_shift_analysis()

    poly_analyzer.plot_prediction_scatter()

    poly_analyzer.plot_residuals()

    # ============================================================================
    # 3: Random forest regression
    # ============================================================================


    print("\n\nSTAGE 3:  random forest regression selection")
    print("=" * 100)

    rf_selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variable_dimensions,
        target_dimensions=target_dimensions,
        fixed_dimensions=['DWT'],
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        selection_metric='adj_r2',
        verbose=True
    )

    # Analyze all combinations
    rf_results = rf_selector.analyze_all_combinations(
        max_predictors=2,
        cv_method='stratified',
        n_folds=10,
        random_state=42
    )

    # Print summary
    rf_selector.print_summary()


    # ============================================================================
    # 3.2 PLOTTING
    # ============================================================================



    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='rrmse',
    )

    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='adj_r2',
    )

    # Plot feature importance
    rf_selector.plot_feature_importance(
    )


    # ============================================================================
    # 4. CREATE RANDOM FOREST DOMAIN ANALYZER
    # ============================================================================

    print("\n\nSTAGE 5: random forest domain shift analysis")
    print("=" * 100)

    rf_analyzer = RandomForestDomainAnalyzer(
        conventional_data=conventional_df,
        conventional_matched=conventional_matched_df,
        alternative_data=alternative_df,
        variable_dimensions=variable_dimensions,
        target_dimensions=target_dimensions,
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        },
        test_size=0.2,
        random_state=42,
        verbose=True
    )

    # ============================================================================
    # 4.1. ANALYZE ALL COMBINATIONS
    # ============================================================================

    rf_results = rf_analyzer.analyze_all_combinations(max_predictors=2)

    # ============================================================================
    # 4.2. INTERPRET RESULTS
    # ============================================================================

    rf_analyzer.interpret_results(
        significance_threshold_r2=0.05,
        significance_threshold_rrmse=10.0
    )

    # ============================================================================
    # 4.3. VISUALIZE
    # ============================================================================

    # Domain shift plots
    rf_analyzer.plot_domain_shift_analysis(
    )

    # Feature importance
    rf_analyzer.plot_feature_importance_analysis(
    )

    # Create scatter plots
    rf_analyzer.plot_prediction_scatter()

    # Create residual plots
    rf_analyzer.plot_residuals()

    # ============================================================================
    # 4.4. GET SUMMARY
    # ============================================================================

    summary = rf_analyzer.get_summary()
    for target, info in summary.items():
        print(f"\n{target}:")
        print(f"  Best predictors: {info['predictors']}")
        print(f"  Alternative RRMSE: {info['alternative_rrmse']:.2f}%")
        print(f"  Fuel gap (RRMSE): {info['fuel_gap_rrmse']:+.2f}%")
    #
    # # ============================================================================
    # # 5. CREATE copmarison plots
    # # ============================================================================

    comparer = ModelComparisonAnalyzer(
        polynomial_selector=poly_selector,
        rf_selector=rf_selector,
        polynomial_domain_analyzer=poly_analyzer,
        rf_domain_analyzer=rf_analyzer,
        verbose=True
    )

    # ============================================================================
    # COMPREHENSIVE ANALYSIS
    # ============================================================================

    # Print summary
    comparer.print_summary()

    # Create all plots
    comparer.plot_selector_comparison(
        metric='rrmse',
    )

    comparer.plot_domain_shift_comparison(
    )

    comparer.plot_comprehensive_comparison(
    )

    # # Export results
    # comparer.export_all_comparisons(base_filename='roro_model_comparison')

def formulae_assessment(conventional_df:pd.DataFrame,
                        conventional_matched_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        target_dimensions:List[str]=None,
                        predictor_dimensions:List[str]=None,
                        shiptype:ShipType = None):

    formula_excel = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature_formulae.xlsx'


    print("\n" + "=" * 80)
    print("PHASE 2: Literature Formula Analysis")
    print("=" * 80)

    # Load and inspect formulae
    formulae = LiteratureFormulaLibrary.from_file(file_path=formula_excel,
                                                  reference='Papanikolaou',
                                                  shiptype=shiptype)

    print(f"\nLoaded {len(formulae)} formulae:")
    for f in formulae:
        print(f"  {f.name}: {f.predictors} → {f.target}")

    # Evaluate (use matched set for fair comparison with ML)
    lit_evaluator = LiteratureFormulaEvaluator(
        conventional_data=conventional_df,
        conventional_matched=conventional_matched_df,
        alternative_data=alternative_df,
        target_dimensions=target_dimensions,
        predictor_dimensions=predictor_dimensions,
        formulae_file=formula_excel,
        reference='Papanikolaou'
    )

    lit_results = lit_evaluator.evaluate_all_formulae()

    lit_evaluator.interpret_results()

    lit_evaluator.plot_comparison()

    lit_evaluator.plot_comparison_scatter()

    lit_evaluator.plot_formula_fit()

    lit_evaluator.plot_residuals()



def dwt_regression_analysis(filepath:Path,
                    conventional_df:pd.DataFrame,
                    conventional_matched_df:pd.DataFrame,
                     alternative_df:pd.DataFrame,
                    ship_type:str='BULK CARRIER',
                    reccomendations:Path = None,
                    dimensions:List[str]=None):

    if dimensions is None:
        dimensions = ['DWT', 'LOA (M)','LBP (M)', 'DRAFT (M)', 'BEAM (M)']

    # STAGE 1: Find best models (DWT always included)
    print("STAGE 1: DWT-CONSTRAINED MODEL SELECTION")
    print("=" * 100)
    selector = DWTConstrainedModelSelector(conventional_df, dimensions, verbose=True)
    model_dict = selector.analyze_all_combinations(max_additional_predictors=2)

    # Export and visualize
    if reccomendations is not None:
        selector.export_recommendations(reccomendations)
    selector.plot_metrics_comparison()


    # # STAGE 2: Use best models in domain analysis
    # print("\n\nSTAGE 2: DWT-CONSTRAINED DOMAIN SHIFT ANALYSIS")
    # print("=" * 100)
    # analyzer = DWTConstrainedDomainAnalyzer(
    #     conventional_train=conventional_df,
    #     conventional_matched=conventional_matched_df,
    #     alternative_data=alternative_df,
    #     dimensions=dimensions,
    #     model_selector=selector
    # )
    #
    # results = analyzer.analyze_all_combinations(max_additional_predictors=2)
    # analyzer.interpret_results()

def speed_regression_analysis(filepath:Path,
                    conventional_df:pd.DataFrame,
                    conventional_matched_df:pd.DataFrame,
                     alternative_df:pd.DataFrame,
                        ship_type:str='BULK CARRIER',
                        reccomendations:Path = None):


    dimensions = ['DWT','LOA (M)','LBP (M)','DRAFT (M)','BEAM (M)']

    analyzer = SpeedImpactAnalyzer(
        data=conventional_df,
        dimensions=dimensions,
        speed_column='SPEED (KNOTS)',
        verbose=True
    )

    # Analyze with and without speed
    analyzer.analyze_with_and_without_speed(max_additional_predictors=2)

    # Create summary
    summary = analyzer.create_comparison_summary()

    # Export results
    analyzer.export_results(reccomendations)

    # Visualize
    analyzer.plot_speed_impact()
    analyzer.plot_metric_correlations()

    print("\n" + "=" * 140)
    print("SPEED ANALYSIS COMPLETE")


def feature_regression_analysis(filepath:Path,
                    conventional_df:pd.DataFrame,
                    conventional_matched_df:pd.DataFrame,
                     alternative_df:pd.DataFrame,
                        ship_type:str='BULK CARRIER',
                        reccomendations:Path = None):
    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type)

    conventional_df = fleet.processed_data['pure_conventional']
    conventional_matched_df = fleet.processed_data['matched_dwt3.5_speed']
    alternative_df = fleet.processed_data['pure_alt_fuel']

    dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'LDT', 'DRAFT (M)', 'BEAM (M)', 'bunker capacity / DWT']


    engineered_features = {
        'C_B': 'C_B',  # Column 'CB' contains block coefficient
        'Froude': 'F_n',  # Column 'FN' contains Froude number
        'L_B_ratio': 'LB_Ratio',  # Column 'LB_Ratio' contains L/B ratio
        'B_T_ratio': 'BT_Ratio'  # Column 'BT_Ratio' contains B/T ratio
    }


    analyzer = FeatureEngineeringAnalyzer(
        data=conventional_df,
        base_dimensions=['DWT', 'Length', 'Beam', 'Draft'],
        engineered_features=engineered_features,
        verbose=True
    )

    # Run analysis (test adding all features together - fast mode)
    analyzer.analyze_feature_impact(
        max_base_predictors=3,
        test_all_feature_combinations=False  # Set True for exhaustive testing
    )

    # Create summary
    summary = analyzer.create_summary_dataframe()

    # Identify most valuable features
    feature_value = analyzer.identify_most_valuable_features()

    # Export results
    analyzer.export_results(reccomendations)

    # Visualize
    analyzer.plot_feature_impact()

    print("\n" + "=" * 100)
    print("FEATURE ANALYSIS COMPLETE")

def random_forest_test(filepath:Path,
                    conventional_df:pd.DataFrame,
                    conventional_matched_df:pd.DataFrame,
                     alternative_df:pd.DataFrame,
                        ship_type:str='BULK CARRIER',
                        reccomendations:Path = None):


    dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'bunker capacity / DWT']

    comparator = RandomForestComparator(
        training_data=conventional_df,
        dimensions=dimensions,
        verbose=True
    )

    # Compare with different RF configurations
    print("\n" + "=" * 140)
    print("Testing Random Forest with 100 trees, no depth limit")
    print("=" * 140)
    comparator.compare_all_combinations(
        max_additional_predictors=2,
        rf_n_estimators=100,
        rf_max_depth=None
    )

    # Create summary
    comparator.create_summary_dataframe()
    comparator.print_summary_statistics()

    # Export and visualize
    comparator.export_results(reccomendations)
    comparator.plot_comparison()


# def random_forest_domain_shift(filepath:Path,
#                     conventional_df:pd.DataFrame,
#                     conventional_matched_df:pd.DataFrame,
#                      alternative_df:pd.DataFrame,
#                                dimensions:List[str],
#                                predicting_dimensions:List[str],
#                         ship_type:str='BULK CARRIER',
#                         reccomendations:Path = None):
#
#
#
#
#
#
#     # Run domain shift analysis
#     analyzer = RandomForestDomainAnalyzer(
#         conventional_train=conventional_df,
#         conventional_matched=conventional_matched_df,
#         alternative_data=alternative_df,
#         predictor_dimensions=dimensions,
#         target_dimensions=predicting_dimensions,
#         verbose=True
#     )
#
#     # Analyze with RF config
#     results = analyzer.analyze_all_combinations(
#         max_additional_predictors=2,
#         rf_n_estimators=100,
#         rf_max_depth=10
#     )
#
#     # Print summary
#     analyzer.create_summary_dataframe()
#     analyzer.print_summary_statistics()
#
#     # Export and visualize
#     analyzer.export_results(reccomendations)
#     analyzer.plot_comparison()
#
#     print("\n" + "=" * 160)
#     print("DOMAIN SHIFT ANALYSIS COMPLETE")

# def rf_vs_lr_methods(filepath:Path,
#                     conventional_df:pd.DataFrame,
#                     conventional_matched_df:pd.DataFrame,
#                      alternative_df:pd.DataFrame,
#                     reccomendations:Path = None,
#                     predictor_dimensions:List[str] = None,
#                     target_dimensions:List[str] = None,):
#
#
#     dimensions = ['DWT', 'LOA (M)','LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)']
#
#     # STAGE 1: Find best models (DWT always included)
#     print("STAGE 1: DWT-CONSTRAINED MODEL SELECTION")
#     print("=" * 100)
#     selector = BestLinearModelSelector(training_data=conventional_df,
#                                        predictor_dimensions=predictor_dimensions,
#                                        target_dimensions=target_dimensions,)
#
#
#     model_dict = selector.analyze_all_combinations(max_additional_predictors=2)
#
#
#     # STAGE 2: Use best models in domain analysis
#     print("\n\nSTAGE 2: DWT-CONSTRAINED DOMAIN SHIFT ANALYSIS")
#     print("=" * 100)
#     analyzer = LinearVsRFDomainAnalyzer(
#         conventional_train=conventional_df,
#         conventional_matched=conventional_matched_df,
#         alternative_data=alternative_df,
#         predictor_dimensions=predictor_dimensions,
#         target_dimensions=target_dimensions,
#         linear_model_selector=selector
#     )
#
#
#     results = analyzer.analyze_all_combinations(max_additional_predictors=2)
#
#     analyzer.create_summary_dataframe()
#
#     analyzer.print_summary_statistics()
#
#     analyzer.export_results(reccomendations)

def run_regression_analysis(filepath:Path|str,
                            shiptype:ShipType=ShipType.BULK_CARRIER,
                            target_dimensions:List[str]=['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)'],
                            variable_dimensions:List[str]|None = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'SPEED (KNOTS)'],
                            dual_fuel_analysis:bool = False,
                            reccomendations_path:Path = None):

    if variable_dimensions is None:
        variable_dimensions = target_dimensions

    ship_type_name = shiptype.name

    print(f'='*60)
    print(('='*20)+ f'RUNNING FULL ANALYSIS FOR {shiptype}'+('='*20))
    print(f'=' * 60)

    print(f'='*60)
    print(('='*20)+ f'STEP 1: Loading data'+('='*20))
    print(f'=' * 60)

    ship_type = 'BULK CARRIER'

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type)

    conventional_df = fleet.processed_data['pure_conventional']
    conventional_matched_df = fleet.processed_data['matched_dwt3.5_speed']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']
    dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']


    formulae_assessment(conventional_df=conventional_df,
                        conventional_matched_df=conventional_matched_df,
                        alternative_df=alternative_df,
                        target_dimensions=target_dimensions,
                        predictor_dimensions=variable_dimensions,
                        shiptype=shiptype)

    DomainShiftAnalysis(filepath=filepath,
                        ship_type=ship_type_name,
                        conventional_df=conventional_df,
                        conventional_matched_df=conventional_matched_df,
                        alternative_df=alternative_df,
                        reccomendations=reccomendations_path,
                        variable_dimensions=variable_dimensions,
                        target_dimensions=target_dimensions,
                        cv_method='stratified',
                        include_regularized=False,
                        use_cv_regularization=False
                        )

    if dual_fuel_analysis:
        formulae_assessment(conventional_df=conventional_df,
                            conventional_matched_df=conventional_matched_df,
                            alternative_df=alternative_df,
                            target_dimensions=target_dimensions,
                            predictor_dimensions=target_dimensions,
                            shiptype=ShipType.BULK_CARRIER)

        DomainShiftAnalysis(filepath=filepath,
                            ship_type=ship_type,
                            conventional_df=conventional_df,
                            conventional_matched_df=dual_fuel_matched_df,
                            alternative_df=dual_fuel_df,
                            reccomendations=reccomendations_path,
                            variable_dimensions=variable_dimensions,
                            target_dimensions=target_dimensions,
                            cv_method='stratified',
                            include_regularized=False,
                            use_cv_regularization=False
                            )

def main():

    # #RORO filepath
    # filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\RORO')
    # ship_type='RORO'
    #
    # fleet = getting_alt_fuel_ships(filepath=filepath,
    #                                ship_type=ship_type)

    # # Bulk carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    # ship_type='BULK CARRIER'
    #
    # run_regression_analysis(filepath=filepath,
    #                         shiptype=ship_type)

    # fleet = getting_alt_fuel_ships(filepath=filepath,
    #                                ship_type=ship_type)
    #
    # # #Container ship
    filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    ship_type=ShipType.CONTAINER_SHIP

    run_regression_analysis(filepath=filepath,
                            shiptype=ship_type)


    #
    # fleet = getting_alt_fuel_ships(filepath=filepath,
    #                                ship_type=ship_type)

    # conventional_df = fleet.processed_data['pure_conventional']
    # conventional_matched_df = fleet.processed_data['matched_dwt3.5_speed']
    # alternative_df = fleet.processed_data['pure_alt_fuel']
    # dual_fuel_df = fleet.processed_data['dual_fuel']
    # dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']




    parameters1 = ['bunker capacity / DWT','C_B', 'LOA (M) / DWT', 'main engine kW / DWT',
                   'main engine kW / bunker capacity']

    parameters2 = ['ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)', 'LOA (M)', 'BEAM (M)', 'Length/Beam',
                   'Beam/Draft']

    target_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)']

    # formulae_assessment(conventional_df=conventional_df,
    #                     conventional_matched_df=conventional_matched_df,
    #                     alternative_df=alternative_df,
    #                     target_dimensions=target_dimensions,
    #                     predictor_dimensions=target_dimensions,
    #                     shiptype=ShipType.BULK_CARRIER)


    # print('===BULK CARRIER DATA SET ANALYSIS===')

    # # full_ship_analysis(filepath=filepath,
    # #                    ship_type='Bulk carrier')
    # regression_analysis(filepath=filepath,
    #                     ship_type='Bulk carrier')

    # analysis_function(
    #     filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier'),
    # ship_type='Bulk Carrier',
    # parameters=parameters1,
    # conventional_df=conventional,
    # alternative_df=alternative)



    dwt_reccomendations_path = Path(r'/knowledge_layer/old_static_handlers/DWT_reccomendations.csv')
    speed_reccomendations_path = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\speed_reccomendations.csv')
    feature_reccomendations_path = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\feature_reccomendations.csv')

    random_forest_path = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\random_forest_reccomendations.csv')

    # variable_dimensions = ['DWT','SPEED (KNOTS)', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)','C_B',
    #    'Length/Beam', 'Beam/Draft','C_B','GT']
    # target_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)','C_B','TOTAL MAIN POWER (KW)','MAIN BUNKER CAPACITY (M3)']

    variable_dimensions = ['DWT','SPEED (KNOTS)', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)']


    target_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)']


    # dwt_regression_analysis(filepath=filepath,
    #                         ship_type='RORO',
    #                         reccomendations=dwt_reccomendations_path,
    #                         dimensions=training_dimensions)

    # regression_selector(filepath=filepath,
    #                         ship_type='RORO',
    #                         reccomendations=dwt_reccomendations_path,
    #                         dimensions=training_dimensions)
    #
    # regression_selector(filepath=filepath,
    #                         ship_type='RORO',
    #                     conventional_df=conventional_df,
    #                     conventional_matched_df=conventional_matched_df,
    #                     alternative_df=alternative_df,
    #                         reccomendations=dwt_reccomendations_path,
    #                         dimensions=training_dimensions,
    #                     cv_method='grouped')

    # DomainShiftAnalysis(filepath=filepath,
    #                     ship_type='Bulk carrier',
    #                     conventional_df=conventional_df,
    #                     conventional_matched_df=conventional_matched_df,
    #                     alternative_df=alternative_df,
    #                     reccomendations=dwt_reccomendations_path,
    #                     variable_dimensions=variable_dimensions,
    #                     target_dimensions=target_dimensions,
    #                     cv_method='stratified',
    #                     include_regularized=False,
    #                     use_cv_regularization=False
    #                     )
    #
    # DomainShiftAnalysis(filepath=filepath,
    #                     ship_type='RORO',
    #                     conventional_df=conventional_df,
    #                     conventional_matched_df=conventional_matched_df,
    #                     alternative_df=alternative_df,
    #                     reccomendations=dwt_reccomendations_path,
    #                     variable_dimensions=variable_dimensions,
    #                     target_dimensions=target_dimensions,
    #                     cv_method='stratified',
    #                     )

    # print('='*60)
    # print("SAME ANALYSIS WITH DUAL FUEL DATA SET")
    # print('=' * 60)
    #
    #
    # DomainShiftAnalysis(filepath=filepath,
    #                     ship_type='RORO',
    #                     conventional_df=conventional_df,
    #                     conventional_matched_df=dual_fuel_matched_df,
    #                     alternative_df=dual_fuel_df,
    #                     reccomendations=dwt_reccomendations_path,
    #                     variable_dimensions=variable_dimensions,
    #                     target_dimensions=target_dimensions,
    #                     cv_method='stratified')

    # poly_regression_selector(filepath=filepath,
    #                     conventional_df=conventional_df,
    #                     reccomendations=dwt_reccomendations_path,
    #                     variable_dimensions=variable_dimensions,
    #                     target_dimensions=target_dimensions,
    #                     cv_method='stratified')
    #
    # poly_regression_selector(filepath=filepath,
    #                     conventional_df=alternative_df,
    #                     reccomendations=dwt_reccomendations_path,
    #                     variable_dimensions=variable_dimensions,
    #                     target_dimensions=target_dimensions,
    #                     cv_method='stratified')

    # random_forest_test(filepath=filepath,
    #                    ship_type='RORO',
    #                    reccomendations=speed_reccomendations_path)

    # dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)', ]
    #
    # random_forest_domain_shift(filepath=filepath,
    #                            ship_type='RORO',
    #                            reccomendations=random_forest_path,
    #                            dimensions=dimensions)
    prediction_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'SCREWS', 'SPEED (KNOTS)', 'C_B','main engine kW / DWT',
       'main engine kW / bunker capacity', 'Length/Beam', 'Beam/Draft','TOTAL AUX POWER (KW)','TOTAL MAIN POWER (KW)']
    training_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)','SCREWS','TOTAL AUX POWER (KW)','TOTAL MAIN POWER (KW)']



    # random_forest_domain_shift(filepath=filepath,
    #                            ship_type='RORO',
    #                            reccomendations=random_forest_path,
    #                            dimensions=training_dimensions,)

    # random_forest_domain_shift(filepath=filepath,
    #                            ship_type='RORO',
    #                            reccomendations=random_forest_path,
    #                            dimensions=prediction_dimensions,
    #                            predicting_dimensions=training_dimensions,)
    
    # rf_vs_lr_methods(filepath=filepath,
    #                  conventional_df=conventional_df,
    #                  conventional_matched_df=conventional_matched_df,
    #                  alternative_df=alternative_df,
    #                  reccomendations=random_forest_path,
    #                  target_dimensions=training_dimensions,
    #                  predictor_dimensions=prediction_dimensions,
    #                  )

    # full_ship_analysis(filepath=filepath,
    #                    ship_type='RORO',)

    # speed_regression_analysis(filepath=filepath,
    #                           ship_type='RORO',
    #                           reccomendations=speed_reccomendations_path)
    #
    #
    # print('===Container ship dataset analysis===')

    # full_ship_analysis(filepath=filepath,
    #                    ship_type='Container ship')


if __name__ == '__main__':
    main()