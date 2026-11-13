"""
Filter Comparison Script
========================
Compares the effect of percentile-based vs IQR-based outlier removal
on the Clarksons fleet dataset.

Both filters are applied independently to the same original dataset,
so that neither method's removals influence the other's bounds.

Usage
-----
Set the ship type and run. Results are printed as a summary table
and optionally saved to Excel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from alt_fuel_datasets import pre_processing
from parameters.shipshape.ShipShapeBasic_class import ShipType
from Operational_profile.data_processing.filter.special_filters import SpecialFilter


# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

SHIP_TYPES = [
    ShipType.BULK_CARRIER,
    ShipType.TANKER,
    ShipType.CONTAINER_SHIP,
    ShipType.RO_PAX,
]

PERCENTILE = 0.95                          # Current method: keeps central 95%
IQR_WINDOW = 1.5                           # IQR multiplier (Tukey standard)
SOURCE_DATASET = 'processed_date'   # Dataset name after ratio/age processing

FILTER_COLUMNS = [
    'DWT',
    'LBP (M)',
    'BEAM (M)',
    'DEPTH (M)',
    'DRAFT (M)',
    'LOA (M)',
]

SAVE_TO_EXCEL = True
OUTPUT_PATH = Path('filter_comparison_results.xlsx')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_summary(df: pd.DataFrame, columns: List[str], label: str) -> pd.DataFrame:
    """
    Compute per-column min, max, and count for a filtered dataset.

    :param df: Filtered DataFrame.
    :param columns: Columns to summarise.
    :param label: Label for this filter method (used as column prefix).
    :returns: DataFrame with one row per column.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        valid = valid[valid > 0]
        rows.append({
            'column': col,
            f'{label}_n': len(valid),
            f'{label}_min': valid.min() if len(valid) > 0 else np.nan,
            f'{label}_max': valid.max() if len(valid) > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index('column')


def compute_bounds_summary(
        sf: SpecialFilter,
        df: pd.DataFrame,
        columns: List[str],
        percentile: float,
        iqr_window: float
) -> pd.DataFrame:
    """
    Compute and display the actual bounds each method would apply.

    Useful for understanding *where* each method draws the line,
    independent of how many points fall outside.

    :param sf: SpecialFilter instance.
    :param df: Original (unfiltered) DataFrame.
    :param columns: Columns to inspect.
    :param percentile: Percentile fraction (e.g. 0.95).
    :param iqr_window: IQR multiplier (e.g. 1.5).
    :returns: DataFrame with lower/upper bounds for each method and column.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue

        pct_bounds = sf.get_percentile_bounds(df, col, percentile)
        iqr_bounds = sf.IQR_bounds(df, col, iqr_window)

        tail = (1 - percentile) / 2
        lower_pct_label = f'{tail:.1%}'
        upper_pct_label = f'{1 - tail:.1%}'

        rows.append({
            'column': col,
            f'pct_lower ({lower_pct_label})': pct_bounds[0] if pct_bounds else np.nan,
            f'pct_upper ({upper_pct_label})': pct_bounds[1] if pct_bounds else np.nan,
            f'iqr_lower (Q1 - {iqr_window}*IQR)': iqr_bounds[0] if iqr_bounds else np.nan,
            f'iqr_upper (Q3 + {iqr_window}*IQR)': iqr_bounds[1] if iqr_bounds else np.nan,
        })

    return pd.DataFrame(rows).set_index('column')


def build_removal_table(
        original: pd.DataFrame,
        pct_filtered: pd.DataFrame,
        iqr_filtered: pd.DataFrame,
        columns: List[str]
) -> pd.DataFrame:
    """
    Build a combined summary table comparing both filter methods.

    Shows, for each column:
    - Original n, min, max
    - After percentile filter: n retained, n removed, % removed, min, max
    - After IQR filter: same

    :param original: Original unfiltered DataFrame.
    :param pct_filtered: DataFrame after percentile filtering.
    :param iqr_filtered: DataFrame after IQR filtering.
    :param columns: Columns to include.
    :returns: Wide-format summary DataFrame.
    """
    orig_summary = compute_summary(original, columns, 'original')
    pct_summary  = compute_summary(pct_filtered, columns, 'percentile')
    iqr_summary  = compute_summary(iqr_filtered, columns, 'iqr')

    combined = orig_summary.join(pct_summary).join(iqr_summary)

    # Add derived columns: removed count and % removed
    combined['pct_n_removed'] = combined['original_n'] - combined['percentile_n']
    combined['pct_%_removed'] = (combined['pct_n_removed'] / combined['original_n'] * 100).round(2)

    combined['iqr_n_removed'] = combined['original_n'] - combined['iqr_n']
    combined['iqr_%_removed'] = (combined['iqr_n_removed'] / combined['original_n'] * 100).round(2)

    # Reorder for readability
    col_order = [
        'original_n', 'original_min', 'original_max',
        'percentile_n', 'pct_n_removed', 'pct_%_removed', 'percentile_min', 'percentile_max',
        'iqr_n',       'iqr_n_removed', 'iqr_%_removed', 'iqr_min',        'iqr_max',
    ]
    return combined[[c for c in col_order if c in combined.columns]]


# =============================================================================
# MAIN
# =============================================================================

def run_comparison_single(
        ship_type: ShipType,
        percentile: float,
        iqr_window: float,
        filter_columns: List[str],
        source_dataset: str,
        print_debug: bool = False,
) -> dict:
    """
    Run the filter comparison for a single ship type and return results.

    Step-by-step:
    1.  Load fleet data using existing pipeline (getting_ratios).
    2.  Extract the pre-processed dataset (before outlier removal).
    3.  Apply percentile filter independently to a copy.
    4.  Apply IQR filter independently to the same original copy.
    5.  Compute per-column bounds for both methods.
    6.  Build comparison table: n, min, max, removed count/% per method.

    :param ship_type: ShipType enum value.
    :param percentile: Percentile fraction for the percentile method.
    :param iqr_window: IQR multiplier for the IQR method.
    :param filter_columns: Columns to compare.
    :param source_dataset: Name of the dataset to load from the FleetData object.
    :param print_debug: Print intermediate steps.
    :returns: Dict with keys 'removal_table', 'bounds_table', 'pct_filtered',
              'iqr_filtered', 'original'.
    """
    _, _, filepath, _ = get_variables(ship_type)

    # ------------------------------------------------------------------
    # STEP 1: Load data up to the point BEFORE percentile filtering
    # ------------------------------------------------------------------
    print(f"\n  Loading data from: {filepath}")
    fleet_data = pre_processing(filepath=filepath, shiptype=ship_type.name)

    if source_dataset not in fleet_data.processed_data:
        available = list(fleet_data.processed_data.keys())
        raise KeyError(
            f"Dataset '{source_dataset}' not found. Available: {available}\n"
            f"Check SOURCE_DATASET at the top of this script."
        )

    original_df = fleet_data.processed_data[source_dataset].copy()
    print(f"  Original dataset: {len(original_df):,} rows")

    # Validate columns — skip any that are absent for this ship type
    # (e.g. RoPax has Np/Nv instead of TEU)
    active_cols = [c for c in filter_columns if c in original_df.columns]
    skipped = [c for c in filter_columns if c not in original_df.columns]
    if skipped:
        print(f"  WARNING: Skipping missing columns: {skipped}")

    if print_debug:
        print(f"  Active filter columns: {active_cols}")
        print(f"  Percentile: {percentile} → excludes {(1-percentile)/2:.1%} each tail")
        print(f"  IQR window: {iqr_window}")

    # ------------------------------------------------------------------
    # STEP 2: Instantiate SpecialFilter
    # ------------------------------------------------------------------
    sf = SpecialFilter(verbose=False)

    # ------------------------------------------------------------------
    # STEP 3: Percentile filter on a clean copy
    # ------------------------------------------------------------------
    pct_filtered = sf.multiple_percentile_filter(
        data=original_df.copy(),
        columns=active_cols,
        percentile=percentile
    )
    print(f"  Percentile filter: {len(pct_filtered):,} retained "
          f"({len(pct_filtered)/len(original_df):.1%})")

    # ------------------------------------------------------------------
    # STEP 4: IQR filter on a clean copy of the ORIGINAL
    # ------------------------------------------------------------------
    iqr_filtered = sf.multiple_IQR_filter(
        data=original_df.copy(),
        columns=active_cols,
        window=iqr_window
    )
    print(f"  IQR filter:        {len(iqr_filtered):,} retained "
          f"({len(iqr_filtered)/len(original_df):.1%})")

    # ------------------------------------------------------------------
    # STEP 5: Bounds table
    # ------------------------------------------------------------------
    bounds_table = compute_bounds_summary(
        sf=sf,
        df=original_df,
        columns=active_cols,
        percentile=percentile,
        iqr_window=iqr_window
    )

    # ------------------------------------------------------------------
    # STEP 6: Removal comparison table
    # ------------------------------------------------------------------
    removal_table = build_removal_table(
        original=original_df,
        pct_filtered=pct_filtered,
        iqr_filtered=iqr_filtered,
        columns=active_cols
    )

    return {
        'removal_table': removal_table,
        'bounds_table': bounds_table,
        'pct_filtered': pct_filtered,
        'iqr_filtered': iqr_filtered,
        'original': original_df,
        'n_original': len(original_df),
        'n_pct': len(pct_filtered),
        'n_iqr': len(iqr_filtered),
    }


def run_all_ship_types(
        ship_types: List[ShipType] = None,
        percentile: float = PERCENTILE,
        iqr_window: float = IQR_WINDOW,
        filter_columns: List[str] = None,
        source_dataset: str = SOURCE_DATASET,
        save_excel: bool = SAVE_TO_EXCEL,
        output_path: Path = OUTPUT_PATH,
        print_debug: bool = False,
) -> dict[str, dict]:
    """
    Run filter comparison across all specified ship types and consolidate results.

    Each ship type is processed independently. Results are stored per ship type
    and also written to a single Excel file with one sheet pair per ship type.

    :param ship_types: List of ShipType values to process.
    :param percentile: Percentile fraction for the percentile method.
    :param iqr_window: IQR multiplier for the IQR method.
    :param filter_columns: Columns to compare. Defaults to module-level FILTER_COLUMNS.
    :param source_dataset: Dataset name within FleetData to use as the baseline.
    :param save_excel: Whether to save results to Excel.
    :param output_path: Path for Excel output.
    :param print_debug: Print intermediate debug steps.
    :returns: Dict mapping ship type name → result dict from run_comparison_single.
    """
    if ship_types is None:
        ship_types = SHIP_TYPES
    if filter_columns is None:
        filter_columns = FILTER_COLUMNS

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.2f}'.format)

    all_results = {}
    overall_rows = []   # For the cross-ship-type summary

    for ship_type in ship_types:
        print(f"\n{'='*60}")
        print(f"SHIP TYPE: {ship_type.name}")
        print(f"{'='*60}")

        try:
            result = run_comparison_single(
                ship_type=ship_type,
                percentile=percentile,
                iqr_window=iqr_window,
                filter_columns=filter_columns,
                source_dataset=source_dataset,
                print_debug=print_debug,
            )
        except Exception as e:
            print(f"  ERROR processing {ship_type.name}: {e}")
            continue

        all_results[ship_type.name] = result

        # Print per-ship-type tables
        print(f"\n  --- Bounds ---")
        print(result['bounds_table'].to_string())
        print(f"\n  --- Removal Comparison ---")
        print(result['removal_table'].to_string())

        # Accumulate overall summary row
        n_orig = result['n_original']
        n_pct  = result['n_pct']
        n_iqr  = result['n_iqr']
        overall_rows.append({
            'ship_type':       ship_type.name,
            'n_original':      n_orig,
            'pct_n_retained':  n_pct,
            'pct_n_removed':   n_orig - n_pct,
            'pct_%_removed':   round((n_orig - n_pct) / n_orig * 100, 2),
            'iqr_n_retained':  n_iqr,
            'iqr_n_removed':   n_orig - n_iqr,
            'iqr_%_removed':   round((n_orig - n_iqr) / n_orig * 100, 2),
        })

    # ------------------------------------------------------------------
    # Cross-ship-type summary
    # ------------------------------------------------------------------
    overall_df = pd.DataFrame(overall_rows).set_index('ship_type')

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY — ALL SHIP TYPES")
    print(f"{'='*60}")
    print(overall_df.to_string())

    # ------------------------------------------------------------------
    # Save to Excel — one sheet pair per ship type + summary sheet
    # ------------------------------------------------------------------
    if save_excel and all_results:
        print(f"\nSaving results to {output_path}...")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Summary sheet first
            overall_df.to_excel(writer, sheet_name='Overall_Summary')

            for ship_name, result in all_results.items():
                # Sheet names are limited to 31 chars in Excel
                prefix = ship_name[:12]

                result['removal_table'].to_excel(
                    writer, sheet_name=f'{prefix}_Removal'
                )
                result['bounds_table'].to_excel(
                    writer, sheet_name=f'{prefix}_Bounds'
                )

        print("  Saved.")

    return all_results


if __name__ == '__main__':
    from alt_fuel_datasets import get_variables

    all_results = run_all_ship_types(
        percentile=PERCENTILE,
        iqr_window=IQR_WINDOW,
        print_debug=True,
    )