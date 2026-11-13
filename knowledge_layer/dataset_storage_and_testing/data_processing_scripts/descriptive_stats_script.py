"""
Generate the appendix descriptive-statistics tables, one per ship type.

Mirrors ``data_availability.py``: the same fleets, the same dataset keys and
the same variable labels, so the availability table and these tables describe
exactly the same cohorts. The datasets given in ``dataset_keys`` must match
those used in the cross-dataset performance tables, or the statistics will not
explain the performance figures they are meant to contextualise.
"""

import logging
from pathlib import Path

from knowledge_layer.dataset_storage_and_testing.data_processing_scripts.stats_table_generator import (
    generate_descriptive_tables_per_ship_type,
)
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (
    getting_alt_fuel_ships, get_variables,
)
from knowledge_layer.dataset_storage_and_testing.data_availability_table_generator import (
    extract_fleet_datasets,
)
from parameters.shipshape.ShipShapeBasic_class import ShipType

logging.basicConfig(level=logging.INFO)

types_list = [ShipType.BULK_CARRIER, ShipType.CONTAINER_SHIP,
              ShipType.TANKER, ShipType.RO_PAX]

fleets = {}

for shiptype in types_list:
    bins, variables, _, filepath, save_dir, data = get_variables(shiptype)

    max_build_year = 1990

    fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['Biofuel']
    alt_fuels = {'Alt_fuel': alt_fuel,
                 'Biofuel': bio_fuels}

    fleets[shiptype.string] = getting_alt_fuel_ships(
        filepath=filepath,
        shiptype=shiptype.string,
        fuel_type_columns=fuel_type_columns,
        alt_fuels=alt_fuels,
        conventional_fuels=conventional_fuels,
        max_build_year=max_build_year,
        outliers=False,
    )

save_dir = Path(
    r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer"
    r"\dataset_storage_and_testing\data_processing_scripts\output\sister_maker"
)

variable_labels = {
    'DWT': "DWT",
    'LOA (M)': "$L_{OA}$ (m)", 'LBP (M)': "$L_{BP}$ (m)",
    'BEAM (M)': "Beam (m)", 'DRAFT (M)': "Draught (m)",
    'DEPTH (M)': "Depth (m)", 'MCR (KW)': "P$_{MCR}$ (kW)", 'TEU': "TEU",
    'Vs': "$V_{s}$", 'Nv': "$N_{v}$", 'Np': "$N_{p}$", 'LM': "LM",
}

# Restricted to the variables that appear as a target or a predictor in the
# cross-dataset performance tables. Including variables that no cohort reports
# would produce whole blocks of red cells that carry no information.
variables = ['DWT', 'TEU', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'DEPTH (M)',
             'BEAM (M)', 'MCR (KW)']

datasets = extract_fleet_datasets(
    fleets,
    dataset_keys={
        'Conventional': 'pure_conventional',
        'Dual fuel LNG': 'dual_fuel_LNG',
        'Dual fuel Methanol': 'dual_fuel_Methanol',
    },
)

caption = (
    "Descriptive statistics for the {ship_type} datasets. "
    "$n$ is the number of vessels reporting each variable, counted "
    "independently per variable. CV is the within-cohort coefficient of "
    "variation $\\sigma_d/\\bar{{y}}_d$; $S$ is the cohort standard deviation "
    "expressed on the conventional-fleet scale, $\\sigma_d/\\bar{{y}}_{{conv}}$, "
    "which is the same scale used to normalise NRMSE in the cross-dataset "
    "performance tables. Cohorts with fewer than ten reporting vessels are "
    "highlighted; variables reported by no vessel are shown as ---."
)

tables = generate_descriptive_tables_per_ship_type(
    datasets=datasets,
    variables=variables,
    reference_dataset='Conventional',
    low_data_limit=10,
    variable_labels=variable_labels,
    caption_template=caption,
    label_template="tab:descriptive_stats_{slug}",
    save_dir=save_dir,
    filename_template="descriptive_stats_{slug}.tex",
)

for ship_type in tables:
    print(f"Wrote descriptive statistics table for {ship_type}")