from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

from Operational_profile.data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (pre_processing, getting_alt_fuel_ships, get_variables)

def all_combinations(items:list[str]):

    return list(combo for r in range(1, len(items)+1) for combo in combinations (items,r))

def create_combo_dict(combinations:list[str],
                      renames:dict|None=None):

    combo_dict = {}

    for combo in combinations:
        name = ""
        if renames is None:

            for item in combo:
                name += f"_{item}"
            combo_dict[name] = combo

        else:
            for item in combo:
                name += f"_{renames[item]}"
            combo_dict[name] = combo

    return combo_dict

def create_summary_row(fleet: "FleetData",
                         shiptype:str,
                         fuel_keys:list[str]):

    base_row = {"Ship type":shiptype,
           "Removed outliers":len(fleet.processed_data["processed_outliers"])-len(fleet.raw_data["raw1"]),
            "missing dimensions":len(fleet.processed_data["processed_missing_data"])-len(fleet.processed_data["processed_outliers"]),
    }

    for key in fuel_keys:

        base_row[f"missing dimensions {key}"] = len(fleet.processed_data[f"processed_missing_{key}"]) - len(fleet.raw_data["raw1"])

    return base_row



def load_fleet_data(filepath: str|Path) -> FleetData:

    fleet=FleetData(verbose=True)

    rename_dict = {'MMSI (MARITIME MOBILE SERVICE IDENTITY)':'MMSI',
                   'NUMBER OF SCREWS':'SCREWS',
                   'BEAM MLD (M)':'BEAM (M)',
                   'DEPTH MOULDED (M)':'DEPTH (M)',
                   'ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)':'TOTAL MAIN POWER (KW)',
                   'ENGINE DERIVED TOTAL MAIN ENGINE MECHANICAL KW': 'MCR (KW)',
                    'TOTAL MAIN POWER (KW)':'MCR (KW)',
                   'AUXILIARY DERIVED TOTAL MECHANICAL GENERATED (KW)':'TOTAL AUX POWER (KW)',
                   'AUXILIARY ENGINE FUEL TYPE':'AUX_FUEL',
                   'MAIN ENGINE FUEL TYPE':"MAIN_FUEL",
                   'MAIN BUNKER FUEL TYPE':"BUNKER_FUEL",
                   'SPEED (KNOTS)':'Vs',
                   'SPEED (KN)':'Vs',
                   'VS':'Vs',
                   'LANE LENGTH (M)':'LM',
                   'PASSENGERS':'Np',
                   'VEHICLE CAPACITY':'Nv',
                   'NP':'Np',
                   'NV':'Nv',
                   'LDT':'LSW',
                   'MAIN BUNKER CAPACITY (M3)':'BUNKER (M3)'}

    fleet.import_data(file_path=filepath,
                      column_renames=rename_dict, )

    fleet.list_datasets()



    # export_data = fleet.raw_data['raw1']
    #
    # filtered_df = export_data[~export_data['TYPE'].str.contains('maran', case=False, na=False)]
    #
    # filtered_df.to_excel(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')


    return fleet

def adding_matched_datasets(fleet_data: FleetData,
                            alternative_df:str,
                            conventional_df:str,
                            alt_set_designation:str='pure'):


    fleet_data.add_matched_dataset(alternative_df=alternative_df,
                                   conventional_df=conventional_df,
                                   matching_features=['DWT', 'Vs'],
                                   feature_weights={'DWT': 3.5, 'Vs': 1},
                                   name=f'{alt_set_designation}_matched_dwt_speed',
                                   n_matches_per_ship=10, )

    fleet_data.add_matched_dataset(alternative_df=alternative_df,
                                   conventional_df=conventional_df,
                                   matching_features=['DWT'],
                                   name=f'{alt_set_designation}_matched_dwt',
                                   n_matches_per_ship=10, )

def fuel_investigation_process(filepath: str | Path,
                   shiptype: str,
                   fuel_type_columns: list[str],
                    max_nan_columns:list[str],
                   max_build_year: int = 1990,
                   outlier_method: str = 'percentile',
                   iqr_window: float = 1.5,
                   percentile: float = 0.95):
    """
    Pre-process fleet data from raw import through to a cleaned base dataset.

    :param filepath: Path to the data directory or file.
    :type filepath: str or Path
    :param shiptype: Ship type identifier string (unused directly here, passed for context).
    :type shiptype: str
    :param fuel_type_columns: Column names containing fuel type information to split.
    :type fuel_type_columns: list[str]
    :param max_build_year: Minimum build year for ships to be retained (inclusive).
    :type max_build_year: int
    :param outlier_method: Outlier removal strategy. Either ``'percentile'`` (removes the
        outermost tail fraction of each column's distribution symmetrically) or ``'iqr'``
        (removes values outside ``Q1 - window*IQR`` and ``Q3 + window*IQR``). For skewed
        distributions such as DWT and LOA, the IQR method tends to be more conservative
        because it anchors on the interquartile range rather than absolute tail quantiles.
    :type outlier_method: str
    :param iqr_window: Multiplier for the IQR bounds. Only used when
        ``outlier_method='iqr'``. A value of 1.5 is the standard Tukey fence;
        3.0 targets only extreme outliers.
    :type iqr_window: float
    :param percentile: Central fraction of each distribution to retain. Only used when
        ``outlier_method='percentile'``. A value of 0.95 keeps the middle 95%
        (excludes 2.5% on each tail).
    :type percentile: float
    :returns: Processed :class:`FleetData` object with named intermediate datasets.
    :rtype: FleetData
    :raises ValueError: If ``outlier_method`` is not ``'percentile'`` or ``'iqr'``.
    """
    if outlier_method not in ('percentile', 'iqr'):
        raise ValueError(
            f"outlier_method must be 'percentile' or 'iqr', got '{outlier_method}'"
        )

    fleet_data = load_fleet_data(filepath)

    date_filter = fleet_data.create_filter(
        column='BUILT',
        value=max_build_year,
        operator=ComparisonOperator.GREATER_EQUAL,
        require_all=False
    )

    percentile_filter_cols = ['DWT', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)']
    IQR_cols = ['DWT', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)']

    # Resolve outlier kwargs based on chosen method
    if outlier_method == 'percentile':
        outlier_kwargs = dict(
            percentile_filter_cols=percentile_filter_cols,
            percentile=percentile,
            IQR_filter=None,
        )
    else:
        outlier_kwargs = dict(
            IQR_filter=IQR_cols,
            percentile_filter_cols=None,
            percentile=None,
        )

    fleet_data.processing_fleet_data(
        datafile='raw1',
        name='processed_outliers',
        duplicates=None,
        **outlier_kwargs
    )


    fleet_data.processing_fleet_data(
        datafile='processed_outliers',
        name='processed_missing_data',
        filter_by=date_filter,
        duplicates=None,
        max_nan_columns = max_nan_columns,
        max_nan_no = 1
    )



    combo = all_combinations(fuel_type_columns)


    renames = {'MAIN_FUEL': "MAIN",
               'AUX_FUEL': 'AUX',
               "BUNKER_FUEL": "BUNKER"}

    combo_dict = create_combo_dict(combo, renames)

    for key,items in combo_dict.items():

        nan_columns = max_nan_columns + list(items)

        fleet_data.processing_fleet_data(
            datafile='processed_outliers',
            name=f'processed_missing_{key}',
            filter_by=date_filter,
            duplicates=None,
            max_nan_columns = nan_columns,
            max_nan_no = 1
        )




    fleet_data.list_datasets()

    return fleet_data

if __name__ == '__main__':
    fuel_cols = ['MAIN_FUEL', 'AUX_FUEL', "BUNKER_FUEL"]
    max_nan_columns = ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)', 'DWT']
    #
    # combo = all_combinations(fuel_cols)
    #
    # print(combo)
    #
    # renames = {'MAIN_FUEL':"MAIN",
    #            'AUX_FUEL':'AUX',
    #            "BUNKER_FUEL":"BUNKER"}
    #
    # combo_dict = create_combo_dict(combo, renames)
    #
    # print(combo_dict)

    types_list = [ShipType.BULK_CARRIER,ShipType.TANKER, ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    # Create fleet_datasets

    summary_table = []

    for shiptype in types_list:
        bins, variables, _, filepath, save_dir = get_variables(shiptype)

        fleet = fuel_investigation_process(filepath=filepath, shiptype=shiptype, fuel_type_columns=fuel_cols, max_nan_columns=max_nan_columns)

        combo = all_combinations(fuel_cols)

        renames = {'MAIN_FUEL': "MAIN",
                   'AUX_FUEL': 'AUX',
                   "BUNKER_FUEL": "BUNKER"}

        combo_dict = create_combo_dict(combo, renames)

        combo_names = list(combo_dict.keys())

        summary_row = create_summary_row(shiptype=shiptype,
                           fleet=fleet,
                           fuel_keys=combo_names)

        summary_table.append(summary_row)

    summary_df = pd.DataFrame(summary_table)

    summary_df.to_excel(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\output\missing_data_summary.xlsx")

