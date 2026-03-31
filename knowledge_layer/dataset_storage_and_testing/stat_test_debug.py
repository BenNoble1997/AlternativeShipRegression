"""
Comprehensive debugging analysis for statistical testing workflow.

This module provides detailed diagnostics to understand:
- Why certain p-values differ unexpectedly
- Data availability patterns
- Sample size issues
- Matching quality
- Distribution characteristics

Run this BEFORE your main hypothesis testing to identify potential issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from parameters.shipshape.ShipShapeBasic_class import ShipType


class StatisticalTestDebugger:
    """
    Comprehensive debugging toolkit for statistical testing workflow.

    This class provides methods to diagnose issues with:
    - Missing data patterns
    - Sample size discrepancies
    - Distribution characteristics
    - Matching quality
    - Feature correlations
    """

    def __init__(self, fleet, ship_type: ShipType):
        """
        Initialize debugger for a specific ship type.

        Parameters
        ----------
        fleet : FleetData
            Fleet data object with processed datasets
        ship_type : ShipType
            Ship type to analyze
        """
        self.fleet = fleet
        self.ship_type = ship_type
        self.conv_df = fleet.processed_data.get('pure_conventional')
        self.dual_df = fleet.processed_data.get('dual_fuel')
        self.pure_df = fleet.processed_data.get('pure_alt_fuel')

    def run_full_diagnostic(self,
                            features: List[str],
                            matching_parameter: str = 'DWT',
                            output_dir: Optional[str] = None) -> Dict:
        """
        Run complete diagnostic analysis.

        This is the main function - call this to get a comprehensive report.

        Parameters
        ----------
        features : List[str]
            Features to analyze (e.g., ['LBP (M)', 'LOA (M)', 'BEAM (M)'])
        matching_parameter : str
            Parameter used for matching (default: 'DWT')
        output_dir : str, optional
            Directory to save diagnostic plots and reports

        Returns
        -------
        Dict
            Dictionary containing all diagnostic results
        """

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
        else:
            output_path = None

        print("=" * 100)
        print(f"COMPREHENSIVE DIAGNOSTIC ANALYSIS: {self.ship_type.name}")
        print("=" * 100)

        results = {}

        # 1. Data availability analysis
        print("\n" + "=" * 100)
        print("1. DATA AVAILABILITY ANALYSIS")
        print("=" * 100)
        results['data_availability'] = self.analyze_data_availability(features)

        # 2. Feature correlation analysis
        print("\n" + "=" * 100)
        print("2. FEATURE CORRELATION ANALYSIS")
        print("=" * 100)
        results['correlations'] = self.analyze_feature_correlations(
            features, output_path
        )

        # 3. Distribution comparison
        print("\n" + "=" * 100)
        print("3. DISTRIBUTION CHARACTERISTICS")
        print("=" * 100)
        results['distributions'] = self.analyze_distributions(
            features, output_path
        )

        # 4. Matching quality analysis
        print("\n" + "=" * 100)
        print("4. MATCHING QUALITY ANALYSIS")
        print("=" * 100)
        results['matching'] = self.analyze_matching_quality(
            features, matching_parameter, output_path
        )

        # 5. Sample size analysis
        print("\n" + "=" * 100)
        print("5. SAMPLE SIZE ANALYSIS")
        print("=" * 100)
        results['sample_sizes'] = self.analyze_sample_sizes(
            features, matching_parameter
        )

        # 6. Specific LBP vs LOA investigation
        if 'LBP (M)' in features and 'LOA (M)' in features:
            print("\n" + "=" * 100)
            print("6. LBP vs LOA SPECIFIC ANALYSIS")
            print("=" * 100)
            results['lbp_loa'] = self.analyze_lbp_loa_discrepancy(output_path)

        # Generate summary report
        print("\n" + "=" * 100)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 100)
        self.generate_summary_report(results, features)

        # Save detailed report to file if output directory specified
        if output_path:
            self.save_detailed_report(results, features, output_path)

        return results

    def analyze_data_availability(self, features: List[str]) -> pd.DataFrame:
        """
        Analyze data availability patterns for each feature.

        This checks:
        - How many ships have data for each feature
        - Which ships have multiple features (to understand test sample overlap)
        """

        availability_data = []

        for fuel_type, df_name in [('Conventional', self.conv_df),
                                   ('Dual Fuel', self.dual_df),
                                   ('Pure Alternative', self.pure_df)]:
            if df_name is None:
                continue

            row = {'Fuel Type': fuel_type, 'Total Ships': len(df_name)}

            for feature in features:
                if feature in df_name.columns:
                    n_available = df_name[feature].notna().sum()
                    pct_available = (n_available / len(df_name)) * 100
                    row[f'{feature}_n'] = n_available
                    row[f'{feature}_%'] = pct_available
                else:
                    row[f'{feature}_n'] = 0
                    row[f'{feature}_%'] = 0.0

            availability_data.append(row)

        availability_df = pd.DataFrame(availability_data)

        # Print formatted table
        print("\nData Availability by Feature and Fuel Type:")
        print("-" * 100)
        for idx, row in availability_df.iterrows():
            print(f"\n{row['Fuel Type']} (Total: {row['Total Ships']} ships):")
            for feature in features:
                n_col = f'{feature}_n'
                pct_col = f'{feature}_%'
                if n_col in row:
                    print(f"  {feature:20s}: {row[n_col]:4.0f} ships ({row[pct_col]:5.1f}%)")

        return availability_df

    def analyze_feature_correlations(self,
                                     features: List[str],
                                     output_path: Optional[Path] = None) -> Dict:
        """
        Analyze correlations between features to understand relationships.

        This helps explain why similar features (like LBP and LOA) might
        give different test results.
        """

        correlation_results = {}

        for fuel_type, df in [('Conventional', self.conv_df),
                              ('Dual Fuel', self.dual_df),
                              ('Pure Alternative', self.pure_df)]:
            if df is None:
                continue

            # Get subset with valid data for all features
            valid_features = [f for f in features if f in df.columns]
            subset = df[valid_features].dropna()

            if len(subset) < 3:
                print(f"\n{fuel_type}: Insufficient data for correlation analysis")
                continue

            # Compute correlation matrix
            corr_matrix = subset.corr()
            correlation_results[fuel_type] = corr_matrix

            print(f"\n{fuel_type} Correlation Matrix:")
            print("-" * 80)
            print(corr_matrix.round(3))

            # Highlight strong correlations
            print(f"\nStrong correlations (|r| > 0.9) in {fuel_type}:")
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        feat1 = corr_matrix.columns[i]
                        feat2 = corr_matrix.columns[j]
                        print(f"  {feat1} ↔ {feat2}: r = {corr_val:.3f}")

            # Create correlation heatmap
            if output_path:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                            cmap='coolwarm', center=0, vmin=-1, vmax=1,
                            square=True, ax=ax)
                ax.set_title(f'{fuel_type} - Feature Correlations\n{self.ship_type.name}')
                plt.tight_layout()
                plt.savefig(output_path / f'correlation_{fuel_type.replace(" ", "_")}.png',
                            dpi=150)
                plt.close()

        return correlation_results

    def analyze_distributions(self,
                              features: List[str],
                              output_path: Optional[Path] = None) -> Dict:
        """
        Analyze distribution characteristics for each feature.

        This checks:
        - Mean, median, std dev
        - Skewness, kurtosis
        - Presence of outliers
        - Normality
        """

        from scipy import stats

        dist_results = {}

        for feature in features:
            print(f"\n{feature}:")
            print("-" * 80)

            feature_stats = {}

            for fuel_type, df in [('Conventional', self.conv_df),
                                  ('Dual Fuel', self.dual_df),
                                  ('Pure Alternative', self.pure_df)]:
                if df is None or feature not in df.columns:
                    continue

                data = df[feature].dropna()

                if len(data) < 3:
                    continue

                # Calculate statistics
                stats_dict = {
                    'n': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                }

                # Normality test
                if len(data) >= 8:
                    _, p_shapiro = stats.shapiro(data[:5000])  # Shapiro-Wilk limited to 5000
                    stats_dict['shapiro_p'] = p_shapiro
                    stats_dict['is_normal'] = p_shapiro > 0.05

                feature_stats[fuel_type] = stats_dict

                # Print summary
                print(f"  {fuel_type}:")
                print(f"    n={stats_dict['n']}, mean={stats_dict['mean']:.2f}, "
                      f"median={stats_dict['median']:.2f}, std={stats_dict['std']:.2f}")
                print(f"    skewness={stats_dict['skewness']:.3f}, "
                      f"kurtosis={stats_dict['kurtosis']:.3f}")
                if 'shapiro_p' in stats_dict:
                    print(f"    Shapiro-Wilk p={stats_dict['shapiro_p']:.3f} "
                          f"(normal: {stats_dict['is_normal']})")

            dist_results[feature] = feature_stats

            # Create distribution plots
            if output_path and len(feature_stats) > 0:
                self._plot_distributions(feature, feature_stats, output_path)

        return dist_results

    def _plot_distributions(self, feature: str, stats_dict: Dict, output_path: Path):
        """Create distribution comparison plots."""

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Histogram comparison
        ax = axes[0]
        for fuel_type, df in [('Conventional', self.conv_df),
                              ('Dual Fuel', self.dual_df),
                              ('Pure Alternative', self.pure_df)]:
            if df is None or feature not in df.columns:
                continue

            data = df[feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, label=fuel_type, density=True)

        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Comparison: {feature}\n{self.ship_type.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Box plot comparison
        ax = axes[1]
        data_for_boxplot = []
        labels = []
        for fuel_type, df in [('Conventional', self.conv_df),
                              ('Dual Fuel', self.dual_df),
                              ('Pure Alternative', self.pure_df)]:
            if df is None or feature not in df.columns:
                continue

            data = df[feature].dropna()
            if len(data) > 0:
                data_for_boxplot.append(data)
                labels.append(fuel_type)

        if data_for_boxplot:
            ax.boxplot(data_for_boxplot, labels=labels)
            ax.set_ylabel(feature)
            ax.set_title(f'Box Plot Comparison: {feature}')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        safe_feature_name = feature.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(output_path / f'distribution_{safe_feature_name}.png', dpi=150)
        plt.close()

    def analyze_matching_quality(self,
                                 features: List[str],
                                 matching_parameter: str = 'DWT',
                                 output_path: Optional[Path] = None) -> Dict:
        """
        Analyze quality of matching procedure.

        This checks:
        - How well matched ships compare on the matching parameter
        - Whether different features get matched to different ships
        - Balance of matched samples
        """

        matching_results = {}

        for fuel_type_name, fuel_type_key in [('Dual Fuel', 'dual_fuel'),
                                              ('Pure Alternative', 'pure_alt_fuel')]:

            if fuel_type_key not in self.fleet.processed_data:
                continue

            print(f"\n{fuel_type_name} Matching Analysis:")
            print("-" * 80)

            try:
                # Create matched dataset
                matched_conv = self.fleet.add_matched_dataset(
                    alternative_df=fuel_type_key,
                    conventional_df='pure_conventional',
                    matching_features=[matching_parameter],
                    n_matches_per_ship=10,
                    store_dataset=False,
                    name=f'debug_match_{fuel_type_key}'
                )

                alt_df = self.fleet.processed_data[fuel_type_key]

                # Check matching parameter distribution
                if matching_parameter in matched_conv.columns and matching_parameter in alt_df.columns:
                    match_param_conv = matched_conv[matching_parameter].dropna()
                    match_param_alt = alt_df[matching_parameter].dropna()

                    print(f"  {matching_parameter} distribution:")
                    print(f"    Conventional matched: mean={match_param_conv.mean():.1f}, "
                          f"std={match_param_conv.std():.1f}")
                    print(f"    Alternative: mean={match_param_alt.mean():.1f}, "
                          f"std={match_param_alt.std():.1f}")

                # Check sample sizes for each feature
                print(f"\n  Sample sizes available for testing:")
                feature_samples = {}
                for feature in features:
                    if feature in matched_conv.columns and feature in alt_df.columns:
                        n_conv = matched_conv[feature].notna().sum()
                        n_alt = alt_df[feature].notna().sum()
                        feature_samples[feature] = {'n_conv': n_conv, 'n_alt': n_alt}
                        print(f"    {feature:20s}: Conv={n_conv:4d}, Alt={n_alt:4d}")

                matching_results[fuel_type_name] = {
                    'matched_conv_size': len(matched_conv),
                    'alt_size': len(alt_df),
                    'feature_samples': feature_samples
                }

            except Exception as e:
                print(f"  Error in matching analysis: {e}")
                matching_results[fuel_type_name] = {'error': str(e)}

        return matching_results

    def analyze_sample_sizes(self,
                             features: List[str],
                             matching_parameter: str = 'DWT') -> pd.DataFrame:
        """
        Create comprehensive sample size comparison table.

        This shows exactly how many observations go into each test.
        """

        sample_data = []

        for feature in features:
            for fuel_type_name, fuel_type_key in [('Dual Fuel', 'dual_fuel'),
                                                  ('Pure Alternative', 'pure_alt_fuel')]:

                if fuel_type_key not in self.fleet.processed_data:
                    continue

                try:
                    # Create matched dataset
                    matched_conv = self.fleet.add_matched_dataset(
                        alternative_df=fuel_type_key,
                        conventional_df='pure_conventional',
                        matching_features=[matching_parameter],
                        n_matches_per_ship=10,
                        store_dataset=False,
                        name=f'sample_size_check_{fuel_type_key}_{feature}'
                    )

                    alt_df = self.fleet.processed_data[fuel_type_key]

                    # Count valid observations
                    if feature in matched_conv.columns and feature in alt_df.columns:
                        n_conv = matched_conv[feature].notna().sum()
                        n_alt = alt_df[feature].notna().sum()
                        n_total = n_conv + n_alt

                        sample_data.append({
                            'Feature': feature,
                            'Fuel Type': fuel_type_name,
                            'N_Conventional': n_conv,
                            'N_Alternative': n_alt,
                            'N_Total': n_total
                        })

                except Exception as e:
                    sample_data.append({
                        'Feature': feature,
                        'Fuel Type': fuel_type_name,
                        'N_Conventional': 0,
                        'N_Alternative': 0,
                        'N_Total': 0,
                        'Error': str(e)
                    })

        sample_df = pd.DataFrame(sample_data)

        print("\nSample Sizes for Statistical Tests:")
        print("-" * 80)
        print(sample_df.to_string(index=False))

        return sample_df

    def analyze_lbp_loa_discrepancy(self, output_path: Optional[Path] = None) -> Dict:
        """
        Specific analysis of LBP vs LOA discrepancy.

        This investigates why these highly correlated features might give
        different statistical test results.
        """

        results = {}

        print("\nLBP vs LOA Detailed Analysis:")
        print("-" * 80)

        for fuel_type, df in [('Conventional', self.conv_df),
                              ('Dual Fuel', self.dual_df),
                              ('Pure Alternative', self.pure_df)]:
            if df is None:
                continue

            # Check if both exist
            has_lbp = 'LBP (M)' in df.columns
            has_loa = 'LOA (M)' in df.columns

            if not (has_lbp and has_loa):
                print(f"\n{fuel_type}: Missing LBP or LOA data")
                continue

            # Analyze ships with both measurements
            both_valid = df['LBP (M)'].notna() & df['LOA (M)'].notna()
            lbp_only = df['LBP (M)'].notna() & df['LOA (M)'].isna()
            loa_only = df['LOA (M)'].notna() & df['LBP (M)'].isna()

            print(f"\n{fuel_type} (Total: {len(df)} ships):")
            print(f"  Have both LBP and LOA: {both_valid.sum()} ({both_valid.sum() / len(df) * 100:.1f}%)")
            print(f"  Have only LBP: {lbp_only.sum()} ({lbp_only.sum() / len(df) * 100:.1f}%)")
            print(f"  Have only LOA: {loa_only.sum()} ({loa_only.sum() / len(df) * 100:.1f}%)")

            # Analyze overhang (LOA - LBP) where both exist
            if both_valid.sum() > 0:
                overhang = df.loc[both_valid, 'LOA (M)'] - df.loc[both_valid, 'LBP (M)']
                correlation = df.loc[both_valid, ['LBP (M)', 'LOA (M)']].corr().iloc[0, 1]

                print(f"\n  Where both exist (n={both_valid.sum()}):")
                print(f"    Correlation: r = {correlation:.4f}")
                print(f"    Overhang (LOA-LBP): mean={overhang.mean():.2f}m, "
                      f"std={overhang.std():.2f}m")
                print(f"    Overhang range: [{overhang.min():.2f}m, {overhang.max():.2f}m]")

                results[fuel_type] = {
                    'n_both': both_valid.sum(),
                    'n_lbp_only': lbp_only.sum(),
                    'n_loa_only': loa_only.sum(),
                    'correlation': correlation,
                    'overhang_mean': overhang.mean(),
                    'overhang_std': overhang.std()
                }

                # Create overhang distribution plot
                if output_path:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(overhang, bins=30, edgecolor='black', alpha=0.7)
                    ax.axvline(overhang.mean(), color='red', linestyle='--',
                               linewidth=2, label=f'Mean: {overhang.mean():.2f}m')
                    ax.set_xlabel('Overhang (LOA - LBP) [m]')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Overhang Distribution: {fuel_type}\n{self.ship_type.name}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_path / f'overhang_{fuel_type.replace(" ", "_")}.png',
                                dpi=150)
                    plt.close()

        return results

    def generate_summary_report(self, results: Dict, features: List[str]):
        """Generate summary of key findings."""

        print("\nKEY FINDINGS:")
        print("-" * 80)

        # Check for data availability issues
        if 'data_availability' in results:
            avail_df = results['data_availability']
            for feature in features:
                n_col = f'{feature}_n'
                if n_col in avail_df.columns:
                    min_avail = avail_df[n_col].min()
                    if min_avail < 10:
                        print(f"⚠ WARNING: {feature} has very low sample size ({min_avail}) "
                              f"in at least one dataset")

        # Check for correlation issues
        if 'correlations' in results:
            for fuel_type, corr_matrix in results['correlations'].items():
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.95:
                            feat1 = corr_matrix.columns[i]
                            feat2 = corr_matrix.columns[j]
                            print(f"ℹ INFO: {feat1} and {feat2} are highly correlated "
                                  f"(r={corr_val:.3f}) in {fuel_type}")

        # Check for LBP/LOA issues
        if 'lbp_loa' in results:
            print("\nLBP vs LOA Analysis:")
            for fuel_type, data in results['lbp_loa'].items():
                if 'n_lbp_only' in data and data['n_lbp_only'] > 0:
                    print(f"  {fuel_type}: {data['n_lbp_only']} ships have only LBP")
                if 'n_loa_only' in data and data['n_loa_only'] > 0:
                    print(f"  {fuel_type}: {data['n_loa_only']} ships have only LOA")

        print("\n" + "=" * 80)

    def save_detailed_report(self, results: Dict, features: List[str], output_path: Path):
        """Save detailed text report to file."""

        report_file = output_path / f'diagnostic_report_{self.ship_type.name}.txt'

        with open(report_file, 'w') as f:
            f.write(f"DIAGNOSTIC REPORT: {self.ship_type.name}\n")
            f.write("=" * 100 + "\n\n")

            # Write data availability
            if 'data_availability' in results:
                f.write("DATA AVAILABILITY:\n")
                f.write("-" * 80 + "\n")
                f.write(results['data_availability'].to_string())
                f.write("\n\n")

            # Write sample sizes
            if 'sample_sizes' in results:
                f.write("SAMPLE SIZES FOR TESTS:\n")
                f.write("-" * 80 + "\n")
                f.write(results['sample_sizes'].to_string())
                f.write("\n\n")

            # Write correlations
            if 'correlations' in results:
                f.write("FEATURE CORRELATIONS:\n")
                f.write("-" * 80 + "\n")
                for fuel_type, corr_matrix in results['correlations'].items():
                    f.write(f"\n{fuel_type}:\n")
                    f.write(corr_matrix.to_string())
                    f.write("\n")
                f.write("\n")

        print(f"\nDetailed report saved to: {report_file}")


def run_diagnostic_analysis(
        ship_type: ShipType,
        get_variables_func,
        getting_alt_fuel_ships_func,
        output_dir: Optional[str] = None
) -> Dict:
    """
    Convenience function to run full diagnostic for a ship type.

    Parameters
    ----------
    ship_type : ShipType
        Ship type to analyze
    get_variables_func : callable
        Function to get variables (from your workflow)
    getting_alt_fuel_ships_func : callable
        Function to load fleet data (from your workflow)
    output_dir : str, optional
        Directory to save diagnostic outputs

    Returns
    -------
    Dict
        Diagnostic results

    """

    # Get data for this ship type
    bins, features, filepath, save_dir = get_variables_func(shiptype=ship_type)
    fleet = getting_alt_fuel_ships_func(filepath=filepath)

    # Create debugger
    debugger = StatisticalTestDebugger(fleet, ship_type)

    # Run full diagnostic
    results = debugger.run_full_diagnostic(
        features=features,
        matching_parameter='DWT',
        output_dir=output_dir
    )

    return results


def run_multi_ship_diagnostic(
        ship_types: List[ShipType],
        get_variables_func,
        getting_alt_fuel_ships_func,
        output_base_dir: str = 'diagnostic_outputs'
):
    """
    Run diagnostic analysis for multiple ship types.

    Parameters
    ----------
    ship_types : List[ShipType]
        Ship types to analyze
    get_variables_func : callable
        Function to get variables
    getting_alt_fuel_ships_func : callable
        Function to load fleet data
    output_base_dir : str
        Base directory for outputs

    Example
    -------
    >>> ship_types = [ShipType.BULK_CARRIER, ShipType.TANKER,
    ...               ShipType.CONTAINER_SHIP, ShipType.RO_PAX]
    >>> run_multi_ship_diagnostic(
    ...     ship_types=ship_types,
    ...     get_variables_func=get_variables,
    ...     getting_alt_fuel_ships_func=getting_alt_fuel_ships
    ... )
    """

    all_results = {}

    for ship_type in ship_types:
        print("\n" + "=" * 100)
        print(f"PROCESSING: {ship_type.name}")
        print("=" * 100)

        output_dir = Path(output_base_dir) / ship_type.name

        try:
            results = run_diagnostic_analysis(
                ship_type=ship_type,
                get_variables_func=get_variables_func,
                getting_alt_fuel_ships_func=getting_alt_fuel_ships_func,
                output_dir=str(output_dir)
            )

            all_results[ship_type.name] = results
            print(f"✓ Completed diagnostic for {ship_type.name}")

        except Exception as e:
            print(f"✗ Error processing {ship_type.name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ship_type.name] = {'error': str(e)}

    # Create comparison summary across ship types
    print("\n" + "=" * 100)
    print("CROSS-SHIP TYPE COMPARISON")
    print("=" * 100)

    # Compare data availability across ship types
    print("\nData Availability Comparison:")
    for ship_type_name, results in all_results.items():
        if 'data_availability' in results:
            print(f"\n{ship_type_name}:")
            avail_df = results['data_availability']
            for idx, row in avail_df.iterrows():
                print(f"  {row['Fuel Type']}: {row['Total Ships']} ships")

    return all_results