from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Operational_profile.data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from parameters.shipshape.ShipShapeBasic_class import ShipType

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (pre_processing, getting_alt_fuel_ships, get_variables)

def create_fleet_data(types_list:list,
                    #   max_build_year:int,
                    # fuel_type_columns:list[str],
                    #   conventional_fuels:list[str],
                    #   alt_fuels:dict[str,list[str]]
                      ):

        fleet_datasets = {}

        for shiptype in types_list:
            bins,variables,pred_constraints, filepath, save_dir, datasets = get_variables(shiptype)

            max_build_year = 1990

            fuel_type_columns = ['MAIN_FUEL', 'AUX_FUEL', "BUNKER_FUEL"]
            conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

            alt_fuel = ['LNG', 'Methanol','Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
            bio_fuels = ['Biofuel']
            alt_fuels = {'Alt_fuel': alt_fuel,
                         'Biofuel': bio_fuels}


            fleet = getting_alt_fuel_ships(filepath=filepath,
                                           shiptype=shiptype.string,
                                           fuel_type_columns=fuel_type_columns,
                                           alt_fuels=alt_fuels,
                                           conventional_fuels=conventional_fuels,
                                           max_build_year=max_build_year,)

            fleet_datasets[shiptype.string] = fleet

        return fleet_datasets

def quick_plot(fleet:"FleetData"):

    conventional_df = fleet.processed_data[f'pure_conventional']

    sns.set_theme()

    sns.relplot(data=conventional_df, x="DWT", y="LOA (M)", hue="TYPE", col="TYPE")

    plt.show()

def create_pre_processing_row(shiptype:str,
                         fleet:"FleetData",
                              differences:bool=True):

    if differences:
        # removed_outiers = len(fleet.processed_data[f'processed_outliers'])-len(fleet.raw_data[f'raw1'])
        removed_missing_data =  len(fleet.processed_data[f'processed_missing_data']) - len(fleet.raw_data[f'raw1'])
        removed_sister_ships =  len(fleet.processed_data[f'processed_sister_ships_hull']) -len(fleet.processed_data[f'processed_missing_data'])

        row = {"Ship type":shiptype,
               "Raw dataset":len(fleet.raw_data[f'raw1']),
               # "Remove outliers":removed_outiers,
               # "Remove outliers pct": removed_outiers/len(fleet.raw_data[f'raw1']),
               "Remove missing data":removed_missing_data,
               "removed missing data":removed_missing_data/len(fleet.raw_data[f'raw1']),
               "Remove sister ships":removed_sister_ships,
               "Remove sister ships pct ": removed_sister_ships/len(fleet.raw_data[f'raw1']),
               "processed_dataset":len(fleet.processed_data['processed_sister_ships_hull'])
               }

    else:

        row = {"Ship type": shiptype,
               "Raw dataset": len(fleet.raw_data['raw1']),
               "Remove missing data": len(fleet.processed_data[f'processed_missing_data']),
               "Remove outliers": len(fleet.processed_data['processed_outliers']),
               "Remove sister ships": len(fleet.processed_data['processed_sister_ships_hull']),
               "processed_dataset":len(fleet.processed_data['processed_sister_ships_hull'])
               }

    return row

def alternative_fuels(shiptype:str,
                         fleet:"FleetData",
                              differences:bool=True):



    row = {"Ship type": shiptype,
           "Pure conservative": len(fleet.processed_data['pure_conventional']),
           "Dual fuel LNG": len(fleet.processed_data['dual_fuel_LNG']),
           "Pure LNG": len(fleet.processed_data['pure_LNG']),
           "Dual fuel Methanol":len(fleet.processed_data['dual_fuel_Methanol']),
           "Pure Methanol": len(fleet.processed_data['pure_Methanol']),
           }

    return row


if __name__ == '__main__':

    # types_list = [ShipType.BULK_CARRIER,ShipType.CONTAINER_SHIP,ShipType.TANKER]
    suffix_main = "sister_ship_maker"
    # types_list = [ShipType.CONTAINER_SHIP]
    types_list = [ShipType.BULK_CARRIER,ShipType.TANKER, ShipType.CONTAINER_SHIP, ShipType.RO_PAX]
    # types_list = [ShipType.RO_PAX]

    # Create fleet_datasets

    fleet_datasets = create_fleet_data(types_list=types_list)

    pre_process_result = []
    fuels_result = []

    for type,fleet in fleet_datasets.items():

        quick_plot(fleet)

        path_dir = Path(
            r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\demo_datasets") / type

        path_dir.mkdir(parents=True, exist_ok=True)

        fleet.processed_data["base_with_split_fuels"].to_excel(path_dir/f"pre_processed_data_{suffix_main}.xlsx")

        for str in ["conv_main", "alt_main"]:
            suffix = f"{type}_dual_fuel_{str}_{suffix_main}.xlsx"

            path = path_dir / suffix

            fleet.processed_data[f'dual_fuel_Alt_fuel_{str}'].to_excel(path)

        pre_process_row = create_pre_processing_row(shiptype=type,fleet=fleet)
        fuel_row = alternative_fuels(shiptype=type,fleet=fleet)

        pre_process_result.append(pre_process_row)
        fuels_result.append(fuel_row)

    pre_processing_table = pd.DataFrame(pre_process_result)

    fuels_table = pd.DataFrame(fuels_result)

    save_dir = Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts")

    save_path = save_dir / f"pre_processing_{suffix_main}.xlsx"
    fuels_df_path = save_dir / f"fuels_{suffix_main}.xlsx"

    pre_processing_table.to_excel(save_path)
    fuels_table.to_excel(fuels_df_path)




