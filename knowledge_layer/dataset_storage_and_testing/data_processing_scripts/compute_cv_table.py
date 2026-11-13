import pandas as pd
import numpy as np
from pathlib import Path

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships, get_variables
from parameters.shipshape.ShipShapeBasic_class import ShipType




def create_ship_datasets(dataset:str):
    types_list = [ShipType.BULK_CARRIER, ShipType.TANKER, ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    max_build_year = 1990

    fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['Biofuel']
    alt_fuels = {'Alt_fuel': alt_fuel,
                 'Biofuel': bio_fuels}

    datasets = {}
    for shiptype in types_list:
        bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)
        fleet = getting_alt_fuel_ships(filepath=filepath, shiptype=shiptype, max_build_year=max_build_year,fuel_type_columns=fuel_type_columns,
                                       conventional_fuels=conventional_fuels,alt_fuels=alt_fuels)
        datasets[shiptype.string] = fleet.processed_data[dataset]
    return datasets


def _compute_cv_for_df(df: pd.DataFrame, target_variables: list[str], round_decimals: int) -> dict:
    """Compute CV for each target variable in a single dataframe."""
    row = {}
    for var in target_variables:
        if var not in df.columns:
            row[var] = np.nan
            continue
        data = df[var].dropna()
        if len(data) < 2 or data.mean() == 0:
            row[var] = np.nan
            continue
        row[var] = round((data.std(ddof=1) / data.mean()) * 100, round_decimals)
    return row


def compute_cv_table(
    ship_datasets: dict[str, pd.DataFrame],
    target_variables: list[str],
    round_decimals: int = 1,
) -> pd.DataFrame:
    """
    Compute CV for target variables across ship types (whole-fleet, no binning).
    """
    rows = []
    for ship_type, df in ship_datasets.items():
        row = {'Ship Type': ship_type, 'DWT Bin': 'All', 'n': len(df)}
        row.update(_compute_cv_for_df(df, target_variables, round_decimals))
        rows.append(row)

    result = pd.DataFrame(rows).set_index(['Ship Type', 'DWT Bin'])
    col_order = ['n'] + [v for v in target_variables if v in result.columns]
    return result[col_order]


def compute_cv_table_binned(
    ship_datasets: dict[str, pd.DataFrame],
    bins_per_shiptype: dict[str, list],
    target_variables: list[str],
    round_decimals: int = 1,
    min_bin_samples: int = 10,
) -> pd.DataFrame:
    """
    Compute CV for target variables, broken down by DWT bin for specified ship types.
    Ship types without bins fall back to whole-fleet CV.

    Parameters
    ----------
    ship_datasets : dict[str, pd.DataFrame]
        Keys are ship type names, values are conventional datasets
    bins_per_shiptype : dict[str, list]
        Keys are ship type names, values are DWT bin edges (same format as get_variables)
        Ship types not in this dict get whole-fleet CV only
    target_variables : list[str]
        Column names to compute CV for
    round_decimals : int
        Decimal places for rounding
    min_bin_samples : int
        Minimum ships in a bin to compute CV (bins below this are skipped)

    Returns
    -------
    pd.DataFrame
        MultiIndex table (Ship Type, DWT Bin) with CV values
    """
    rows = []

    for ship_type, df in ship_datasets.items():

        if ship_type not in bins_per_shiptype:
            # No binning for this ship type - whole fleet only
            row = {'Ship Type': ship_type, 'DWT Bin': 'All', 'n': len(df)}
            row.update(_compute_cv_for_df(df, target_variables, round_decimals))
            rows.append(row)
            continue

        dwt_bins = bins_per_shiptype[ship_type]

        if 'DWT' not in df.columns:
            print(f"Warning: DWT column not found for {ship_type}, skipping binning")
            continue

        # Assign each ship to a DWT bin
        bin_labels = [
            f'{int(dwt_bins[i]/1000)}k–{int(dwt_bins[i+1]/1000)}k'
            if dwt_bins[i+1] != np.inf
            else f'{int(dwt_bins[i]/1000)}k+'
            for i in range(len(dwt_bins) - 1)
        ]

        df = df.copy()
        df['_dwt_bin'] = pd.cut(
            df['DWT'],
            bins=dwt_bins,
            labels=bin_labels,
            right=True
        )

        for bin_label in bin_labels:
            bin_df = df[df['_dwt_bin'] == bin_label]

            if len(bin_df) < min_bin_samples:
                continue

            row = {'Ship Type': ship_type, 'DWT Bin': bin_label, 'n': len(bin_df)}
            row.update(_compute_cv_for_df(bin_df, target_variables, round_decimals))
            rows.append(row)

    result = pd.DataFrame(rows).set_index(['Ship Type', 'DWT Bin'])
    col_order = ['n'] + [v for v in target_variables if v in result.columns]
    return result[col_order]


def print_cv_table(cv_table: pd.DataFrame, title: str = "Coefficient of Variation (%)"):
    """Print formatted CV table."""
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    print(cv_table.to_string())
    print(f"{'=' * 90}\n")


if __name__ == "__main__":

    datanames=['pure_conventional','dual_fuel_Alt_fuel_alt_main','pure_Biofuel']
    target_vars = ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'DWT']

    cv_datasets = {}
    for dataset in datanames:


        datasets = create_ship_datasets(dataset=dataset)

        # Whole-fleet CV (all ship types)
        cv_whole = compute_cv_table(datasets, target_vars)

        cv_datasets[dataset] = cv_whole

    for dataset,cv in cv_datasets.items():

        print(f'CV of dataset {dataset}:')

        print_cv_table(cv, "Coefficient of Variation (%) — Whole Fleet")

        # # Binned CV (bulk carriers and tankers only)
        # bins_for_binned_types = {
        #     'Bulk carrier': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf],
        #     'Tanker':       [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf],
        # }
        #
        # cv_binned = compute_cv_table_binned(
        #     datasets,
        #     bins_per_shiptype=bins_for_binned_types,
        #     target_variables=target_vars,
        # )
        # print_cv_table(cv_binned, "Coefficient of Variation (%) — By DWT Bin")