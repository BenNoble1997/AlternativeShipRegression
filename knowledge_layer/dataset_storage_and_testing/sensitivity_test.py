import pandas as pd
import numpy as np

from distribution_comparison_class import DistributionComparison
from typing import List
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData


def matching_sensitivity_analysis(
        fleet: FleetData,
        conventional_dataset: str,
        alternative_dataset: str,
        test_features: List[str],
        matching_features: List[str],
        n_matches_per_ship: int = 10,
        K_S_test: bool = True,
        M_W_test: bool = True,
        welch_test: bool = False,
        alpha: float = 0.05
) -> pd.DataFrame:
    """
    Sensitivity analysis: Test how results change with different matching features.

    This tests robustness by creating matched datasets using different features
    and checking if statistical conclusions remain consistent.

    Parameters
    ----------
    fleet : FleetData
        Fleet data object containing all datasets
    conventional_dataset : str
        Name of conventional dataset
    alternative_dataset : str
        Name of alternative fuel dataset
    test_features : List[str]
        Features to test for differences (e.g., ['LBP (M)', 'BEAM (M)', 'DRAFT (M)'])
    matching_features : List[str]
        Features to use for matching (e.g., ['DWT', 'TEU', 'GT', 'SPEED (KN)'])
        Analysis will create a matched dataset for each of these
    n_matches_per_ship : int, default=10
        Number of matches per alternative ship
    K_S_test : bool, default=True
        Whether to run Kolmogorov-Smirnov test
    M_W_test : bool, default=True
        Whether to run Mann-Whitney test
    welch_test : bool, default=False
        Whether to run Welch's t-test
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - Feature: The feature being tested
        - Test: Type of test (K-S, M-W, Welch)
        - Match_<feature>_p: p-value when matching on <feature>
        - Match_<feature>_sig: significance boolean when matching on <feature>

    Examples
    --------
    >>> results = matching_sensitivity_analysis(
    ...     fleet=fleet,
    ...     conventional_dataset='pure_conventional',
    ...     alternative_dataset='pure_alt_fuel',
    ...     test_features=['LBP (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)'],
    ...     matching_features=['DWT', 'TEU', 'GT', 'SPEED (KN)']
    ... )
    >>> print(results)
    """

    print("=" * 80)
    print("MATCHING SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Testing {len(test_features)} features across {len(matching_features)} matching strategies")
    print(f"Alternative dataset: {alternative_dataset}")
    print(f"Conventional dataset: {conventional_dataset}")
    print(f"Matching strategies: {', '.join(matching_features)}")
    print("=" * 80)

    # Get alternative dataset (stays constant)
    alternative_df = fleet.processed_data[alternative_dataset]

    # Storage for results
    all_results = []

    # Loop through each matching feature
    for match_feature in matching_features:
        print(f"\n{'=' * 80}")
        print(f"Creating matched dataset using: {match_feature}")
        print(f"{'=' * 80}")

        try:
            # Create matched dataset for this feature
            matched_df = fleet.add_matched_dataset(
                alternative_df=alternative_dataset,
                conventional_df=conventional_dataset,
                name=f'matched_{match_feature}',
                matching_features=[match_feature],
                n_matches_per_ship=n_matches_per_ship,
                store_dataset=False  # Don't store, just use temporarily
            )

            print(f"✓ Successfully created matched dataset (n={len(matched_df)})")

            # Create comparison object
            comparison = DistributionComparison(
                conventional_data=matched_df,
                alternative_data=alternative_df
            )

            # Run all tests for this matching strategy
            print(f"Running statistical tests...")
            test_results = comparison.run_all_tests(
                features=test_features,
                K_S_test=K_S_test,
                M_W_test=M_W_test,
                welch_test=welch_test
            )

            # Add matching feature information to results
            test_results['Matching_Feature'] = match_feature
            all_results.append(test_results)

            print(f"✓ Tests complete for {match_feature}")

        except Exception as e:
            print(f"✗ Error creating matched dataset for {match_feature}: {str(e)}")
            print(f"  Skipping this matching feature...")
            continue

    if not all_results:
        raise ValueError("No successful matches created. Check your matching features.")

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Pivot to create the desired output format
    print(f"\n{'=' * 80}")
    print("Reformatting results...")
    print(f"{'=' * 80}")

    # Create output table
    output_rows = []

    # Get unique feature-test combinations
    for feature in test_features:
        for test_type in combined_results['Test'].unique():

            row = {
                'Feature': feature,
                'Test': test_type
            }

            # For each matching feature, get p-value and significance
            for match_feature in matching_features:
                # Find the row in results for this combination
                mask = (
                        (combined_results['Feature'] == feature) &
                        (combined_results['Test'] == test_type) &
                        (combined_results['Matching_Feature'] == match_feature)
                )

                if mask.any():
                    result_row = combined_results[mask].iloc[0]
                    p_value = result_row['P-value']
                    significant = result_row['Significant']

                    # Add to output
                    row[f'Match_{match_feature}_p'] = p_value
                    row[f'Match_{match_feature}_sig'] = significant
                else:
                    # Missing data
                    row[f'Match_{match_feature}_p'] = np.nan
                    row[f'Match_{match_feature}_sig'] = False

            output_rows.append(row)

    # Create DataFrame
    output_df = pd.DataFrame(output_rows)

    # Calculate summary statistics
    print(f"\n{'=' * 80}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'=' * 80}")

    # Count how many matching strategies show significant results for each feature-test
    for idx, row in output_df.iterrows():
        feature = row['Feature']
        test = row['Test']

        # Count significant results across matching strategies
        sig_cols = [col for col in output_df.columns if col.endswith('_sig')]
        n_significant = sum(row[col] for col in sig_cols)
        n_total = len(matching_features)

        # Get p-value range
        p_cols = [col for col in output_df.columns if col.endswith('_p')]
        p_values = [row[col] for col in p_cols if not pd.isna(row[col])]

        if p_values:
            p_min = min(p_values)
            p_max = max(p_values)
            p_range = p_max - p_min

            consistency = "ROBUST" if n_significant == 0 or n_significant == n_total else "INCONSISTENT"

            print(f"\n{feature} - {test}:")
            print(f"  Significant in {n_significant}/{n_total} matching strategies")
            print(f"  p-value range: [{p_min:.4f}, {p_max:.4f}] (range: {p_range:.4f})")
            print(f"  Consistency: {consistency}")

    return output_df


def format_sensitivity_table(sensitivity_df: pd.DataFrame,
                             matching_features: List[str],
                             alpha: float = 0.05) -> None:
    """
    Create a nicely formatted display of sensitivity analysis results.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Output from matching_sensitivity_analysis
    matching_features : List[str]
        List of matching features used
    alpha : float, default=0.05
        Significance threshold
    """

    print("\n" + "=" * 100)
    print("FORMATTED SENSITIVITY ANALYSIS RESULTS")
    print("=" * 100)
    print(f"Significance threshold: α = {alpha}")
    print("=" * 100)

    # Create header
    header_parts = ['Feature', 'Test']
    for match_feat in matching_features:
        header_parts.append(f'{match_feat}_p')
        header_parts.append(f'{match_feat}_sig')

    # Format for display
    for idx, row in sensitivity_df.iterrows():
        if idx % 2 == 0:  # New feature
            print(f"\n{'-' * 100}")

        # Feature and test
        display_parts = [f"{row['Feature']:<12}", f"{row['Test']:<4}"]

        # P-values and significance for each matching strategy
        for match_feat in matching_features:
            p_col = f'Match_{match_feat}_p'
            sig_col = f'Match_{match_feat}_sig'

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


def analyze_sensitivity_robustness(sensitivity_df: pd.DataFrame,
                                   matching_features: List[str]) -> pd.DataFrame:
    """
    Analyze how robust findings are across matching strategies.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Output from matching_sensitivity_analysis
    matching_features : List[str]
        List of matching features used

    Returns
    -------
    pd.DataFrame
        Summary showing consistency of results
    """

    robustness_results = []

    for idx, row in sensitivity_df.iterrows():
        feature = row['Feature']
        test = row['Test']

        # Get significance results across all matching strategies
        sig_cols = [f'Match_{mf}_sig' for mf in matching_features]
        sig_results = [row[col] for col in sig_cols if col in row.index]

        # Get p-values across all matching strategies
        p_cols = [f'Match_{mf}_p' for mf in matching_features]
        p_values = [row[col] for col in p_cols if col in row.index and not pd.isna(row[col])]

        if p_values:
            # Calculate metrics
            n_significant = sum(sig_results)
            n_total = len(sig_results)
            pct_significant = (n_significant / n_total) * 100

            p_min = min(p_values)
            p_max = max(p_values)
            p_mean = np.mean(p_values)
            p_std = np.std(p_values)
            p_range = p_max - p_min

            # Determine consistency category
            if n_significant == 0:
                consistency = "CONSISTENTLY NOT SIGNIFICANT"
            elif n_significant == n_total:
                consistency = "CONSISTENTLY SIGNIFICANT"
            elif n_significant < n_total / 2:
                consistency = "MOSTLY NOT SIGNIFICANT"
            else:
                consistency = "MOSTLY SIGNIFICANT"

            # Robustness score (inverse of p-value range)
            # Lower range = more robust
            robustness_score = 1 - min(p_range, 1.0)  # Normalized to [0, 1]

            robustness_results.append({
                'Feature': feature,
                'Test': test,
                'N_Significant': f"{n_significant}/{n_total}",
                'Pct_Significant': f"{pct_significant:.1f}%",
                'P_Mean': f"{p_mean:.4f}",
                'P_StdDev': f"{p_std:.4f}",
                'P_Range': f"{p_range:.4f}",
                'Consistency': consistency,
                'Robustness_Score': f"{robustness_score:.3f}"
            })

    robustness_df = pd.DataFrame(robustness_results)

    # Print summary
    print("\n" + "=" * 100)
    print("ROBUSTNESS SUMMARY")
    print("=" * 100)
    print("\nInterpretation:")
    print("  • High robustness score (>0.9): Results very consistent across matching strategies")
    print("  • Medium robustness score (0.7-0.9): Some variation but generally consistent")
    print("  • Low robustness score (<0.7): Results depend on matching choice")
    print("=" * 100)

    print(robustness_df.to_string(index=False))

    return robustness_df