

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from knowledge_layer.dataset_storage_and_testing.data_availability_table_generator import (
    build_availability_counts,
    generate_availability_table,
    extract_fleet_datasets)

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships,get_variables
from parameters.shipshape.ShipShapeBasic_class import ShipType

types_list = [ShipType.BULK_CARRIER,ShipType.CONTAINER_SHIP,ShipType.TANKER, ShipType.RO_PAX]

fleets = {}

for shiptype in types_list:
    bins, variables, _, filepath, save_dir,data = get_variables(shiptype)

    max_build_year = 1990

    fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['Biofuel']
    alt_fuels = {'Alt_fuel': alt_fuel,
                 'Biofuel': bio_fuels}

    fleets[shiptype.string] = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=shiptype.string,
                                       fuel_type_columns=fuel_type_columns,
                                       alt_fuels=alt_fuels,
                                       conventional_fuels=conventional_fuels,
                                       max_build_year=max_build_year,
                                                     outliers=False,)

save_dir = Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\output")

variable_labels = {'DWT': "DWT",
             'LOA (M)':"$L_{OA}$ (m)", 'LBP (M)':"$L_{BP}$ (m)",
             'BEAM (M)':"Beam (m)", 'DRAFT (M)': "Draught (m)",
             'DEPTH (M)': "Depth (m)", 'MCR (KW)':"P$_{MCR}$ (kW)", 'TEU':"TEU",
             'Vs':"$V_{s}$", 'Nv':"$N_{v}$", 'Np':"N_{p}", 'LM':"LM"
}

variables = ['DWT', 'TEU',
       'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'DEPTH (M)',
             'BEAM (M)', 'Vs', 'Nv', 'Np', 'LM', 'MCR (KW)']


datasets = extract_fleet_datasets(
    fleets,
    dataset_keys={'Conventional':'pure_conventional',
    "dual fuel LNG": "dual_fuel_LNG", "Bio fuel": "pure_Biofuel"},
)

latex = generate_availability_table(
    datasets=datasets,
    variables=variables,           # from get_variables(), unioned across types
    low_data_limit=10,
    variable_labels=variable_labels,
    caption="Availability of data points for key ship parameters across alternative-fuel datasets."
            " Variables with fewer than 10 observations are highlighted in red to indicate"
            " insufficient data density for reliable statistical modelling.", save_path=save_dir / 'data_availability_outliers_bio.tex',
)


