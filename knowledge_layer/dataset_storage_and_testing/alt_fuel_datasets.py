from pathlib import Path
import numpy as np
from Operational_profile.data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from parameters.shipshape.ShipShapeBasic_class import ShipType

def load_fleet_data(filepath: str|Path) -> FleetData:

    fleet=FleetData(verbose=True)

    rename_dict = {'MMSI (MARITIME MOBILE SERVICE IDENTITY)':'MMSI',
                   'NUMBER OF SCREWS':'SCREWS',
                   'BEAM MLD (M)':'BEAM (M)',
                   'DEPTH MOULDED (M)':'DEPTH (M)',
                   'ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)':'TOTAL MAIN POWER (KW)',
                   'ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)': 'MCR (KW)',
                    'TOTAL MAIN POWER (KW)':'MCR (KW)',
                   'AUXILIARY DERIVED TOTAL MECHANICAL GENERATED (KW)':'TOTAL AUX POWER (KW)',
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

def pre_processing(filepath: str | Path,
                   ship_type: str = 'BULK_CARRIER',
                   fuel_type_columns: list[str] = ['MAIN ENGINE FUEL TYPE', 'AUXILIARY ENGINE FUEL TYPE'],
                   max_build_year: int = 1990,
                   outlier_method: str = 'percentile',
                   iqr_window: float = 1.5,
                   percentile: float = 0.95):
    """
    Pre-process fleet data from raw import through to a cleaned base dataset.

    :param filepath: Path to the data directory or file.
    :type filepath: str or Path
    :param ship_type: Ship type identifier string (unused directly here, passed for context).
    :type ship_type: str
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

    max_nan_columns = ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)', 'DWT']
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
        name='processed_date',
        filter_by=date_filter,
        duplicates=None,
    )

    fleet_data.processing_fleet_data(
        datafile='processed_date',
        name='processed_outliers',
        filter_by=date_filter,
        duplicates=None,
        powers_columns=['DWT'],
        power_value=3,
        products_columns=['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)'],
        max_nan_columns=max_nan_columns,
        max_nan_no=1,
        **outlier_kwargs
    )

    fleet_data.processing_fleet_data(
        datafile='processed_outliers',
        name='processed_sister_ships',
        duplicates=['LOA (M)', 'LBP (M)', 'BEAM (M)', 'DWT', 'DRAFT (M)'],
    )

    renames = {
        'LBP (M)*BEAM (M)*DEPTH (M)': 'LppBD',
        'BEAM (M)*DEPTH (M)*LOA (M)': 'LBD',
        'BEAM (M)*DRAFT (M)*LOA (M)': 'LBT',
    }

    fleet_data.processing_fleet_data(
        datafile='processed_sister_ships',
        name='base_with_split_fuels',
        fuel_columns=fuel_type_columns,
        duplicates=['LOA (M)', 'LBP (M)', 'BEAM (M)', 'DWT', 'DRAFT (M)'],
        products_columns=['LBP (M)'],
        column_renames=renames
    )

    print(f"\n=== Added ratio columns and applied age filter to datasets "
          f"(outlier method: {outlier_method}) ===")
    fleet_data.list_datasets()

    return fleet_data


def getting_alt_fuel_ships(filepath: str | Path,
                           ship_type: str = 'BULK_CARRIER',
                           fuel_type_columns: list[str] = ['MAIN ENGINE FUEL TYPE', 'AUXILIARY ENGINE FUEL TYPE'],
                           alt_fuels: list[str] = ['LNG', 'Methanol', 'AMMONIA', 'Hydrogen', 'biofuel', 'cng'],
                           conventional_fuels: list[str] = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO'],
                           outlier_method: str = 'percentile',
                           iqr_window: float = 1.5,
                           percentile: float = 0.95,
                           create_plots: bool = False,
                           plot_target: str = 'LBP (M)',
                           plot_predictor: str = 'DWT'):
    """
    Load, pre-process, and categorise ships by fuel type.

    :param filepath: Path to the raw data directory or file.
    :type filepath: str or Path
    :param ship_type: Ship type label, used for display and routing in :func:`get_variables`.
    :type ship_type: str
    :param fuel_type_columns: Raw column names containing fuel type strings.
    :type fuel_type_columns: list[str]
    :param alt_fuels: Fuel type substrings that classify a ship as alternative-fuelled.
    :type alt_fuels: list[str]
    :param conventional_fuels: Fuel type substrings that classify a ship as conventionally fuelled.
    :type conventional_fuels: list[str]
    :param outlier_method: Outlier removal strategy passed to :func:`pre_processing`.
        ``'percentile'`` removes symmetric tails; ``'iqr'`` uses the Tukey fence.
    :type outlier_method: str
    :param iqr_window: IQR fence multiplier. Only used when ``outlier_method='iqr'``.
    :type iqr_window: float
    :param percentile: Tail fraction to retain. Only used when ``outlier_method='percentile'``.
    :type percentile: float
    :param create_plots: Whether to produce diagnostic scatter plots.
    :type create_plots: bool
    :param plot_target: Y-axis variable for the matching importance plot.
    :type plot_target: str
    :param plot_predictor: X-axis variable for the matching importance plot.
    :type plot_predictor: str
    :returns: :class:`FleetData` instance with all fuel-type subsets stored as named datasets.
    :rtype: FleetData
    """
    fleet_data = pre_processing(
        filepath=filepath,
        ship_type=ship_type,
        outlier_method=outlier_method,
        iqr_window=iqr_window,
        percentile=percentile,
    )
    # fleet_data = load_fleet_data(filepath)
    #


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


    fleet_data.list_datasets()

    fleet_data.plot_matching_importance(conventional_df='pure_conventional',
                                        alternative_df='pure_alt_fuel',
                                        target_variable=plot_target,
                                        predictor_variable=plot_predictor,
                                        name=f'{ship_type} matching dataset comparison')

    print(fleet_data.processed_data['base_with_split_fuels'].columns)

    return fleet_data

def get_variables(shiptype:'ShipType'):

    print(f'USING DATA FOR {shiptype} SHIP')

    constant_vars = ['BEAM (M)','DEPTH (M)','DRAFT (M)','LBP (M)','LOA (M)','DWT']

    if shiptype == ShipType.TANKER:

        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Tanker')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
        variables = constant_vars + ['MCR (KW)']


    elif shiptype == ShipType.BULK_CARRIER:
        print('bulk carrier filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Bulk_carrier')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
        variables = constant_vars+['MCR (KW)']
        # variables = ['DWT', 'LBP (M)', 'DRAFT (M)']

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('container ship filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Containership')
        bins = None
        variables = constant_vars + ['TEU','Vs','MCR (KW)']

    else:
        print('ropax filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\RO_Pax')
        bins = None
        variables = constant_vars + [ 'Np', 'Nv','Vs']

    return bins,variables, filepath, save_dir

