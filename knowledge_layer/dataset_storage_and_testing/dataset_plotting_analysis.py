from pathlib import Path
from parameters.shipshape.ShipShapeBasic_class import ShipType
from alt_fuel_datasets import pre_processing, getting_alt_fuel_ships
from knowledge_layer.fleet_data_class.Plotting_fleet_class import FleetPlottingClass

def get_datasets():
    bulk_carrier_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    bulkcarrier = ShipType.BULK_CARRIER

    tanker_filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    tanker = ShipType.TANKER

    container_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    container = ShipType.CONTAINER_SHIP

    ropax_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
    ropax = ShipType.RO_PAX

    datafiles = {bulkcarrier.string: bulk_carrier_filepath,
                 tanker.string: tanker_filepath,
                 container.string: container_filepath,
                 ropax.string: ropax_filepath}

    shipdata_dict = {}

    for shiptype,filepath in datafiles.items():

        fleet = pre_processing(filepath=filepath, shiptype=shiptype)
        dataset = fleet.processed_data['base_with_split_fuels']
        shipdata_dict[shiptype] = dataset
    return shipdata_dict

    #
    # # # Bulk carrier filepath
    # bulk_carrier_filepath = Path(
    #     r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    # bulkcarrier = ShipType.BULK_CARRIER
    # bulk_fleet = getting_ratios(filepath=bulk_carrier_filepath, ship_type=bulkcarrier)
    # bulker_df = bulk_fleet.processed_data['base_with_split_fuels']
    #
    # # Tanker
    # tanker_filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    # tanker = ShipType.TANKER
    # tanker_fleet = getting_ratios(filepath=tanker_filepath, ship_type=tanker)
    # tanker_df = tanker_fleet.processed_data['base_with_split_fuels']
    #
    # # # #Container ship
    # container_filepath = Path(
    #     r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    # container = ShipType.CONTAINER_SHIP
    # container_fleet = getting_ratios(filepath=container_filepath, ship_type=container)
    # container_df = container_fleet.processed_data['base_with_split_fuels']
    #
    # # #RO PAx
    # ropax_filepath = Path(
    #     r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
    # ropax = ShipType.RO_PAX
    # ropax_fleet = getting_ratios(filepath=ropax_filepath, ship_type=ropax)
    # ropax_df = ropax_fleet.processed_data['base_with_split_fuels']
    #
    # return {bulkcarrier.name:bulker_df,
    #         tanker.name:tanker_df,
    #         container.name:container_df,
    #         ropax.name:ropax_df}

def get_database_tuples():

    # # Bulk carrier filepath
    bulk_carrier_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    bulkcarrier = ShipType.BULK_CARRIER


    tanker_filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    tanker = ShipType.TANKER

    container_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    container = ShipType.CONTAINER_SHIP

    ropax_filepath = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
    ropax = ShipType.RO_PAX

    datafiles = {bulkcarrier.name:bulk_carrier_filepath,
                 tanker.name:tanker_filepath,
                 container.name:container_filepath,
                 ropax.name:ropax_filepath}

    dual_fuel_dict = {}
    alt_fuel_dict = {}

    for shiptype,filepath in datafiles.items():


        fleet = getting_alt_fuel_ships(filepath=filepath, shiptype=shiptype)

        conventional = fleet.processed_data['pure_conventional']
        alt_fuel  = fleet.processed_data['pure_alt_fuel']
        dual_fuel =  fleet.processed_data['dual_fuel']

        alt_fuel_tuple = (conventional, alt_fuel)
        dual_fuel_tuple = (conventional, dual_fuel)

        alt_fuel_dict[shiptype] = alt_fuel_tuple
        dual_fuel_dict[shiptype] = dual_fuel_tuple

    return alt_fuel_dict, dual_fuel_dict



def main():

    ship_datasets = get_datasets()

    print(ship_datasets.keys())

    plotter = FleetPlottingClass(debug=False, verbose=True)

    # variables = ['DWT', 'LBP (M)','LOA (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)', 'LBD','LSW','Vs','MCR (KW)']

    variables = ['DWT', 'LBP (M)', 'BEAM (M)','DEPTH (M)','MCR (KW)']

    variable_map = {'LBP (M)':'LBP (m)',
                    'BEAM (M)':'Beam (m)',
                    'DEPTH (M)':'Depth (m)',
                    'MCR (KW)':'MCR (kW)'}

    # plotter.plotting_simple_multi_histograms(
    #     dataframes=ship_datasets,
    #     y_variables=variables,
    #
    #     name='Ship Type Comparison',
    #     n_hist_bins=30,
    #     density=True,  # Normalize to probability density
    #     show_kde=True,  # Show smooth density curves
    #     show_statistics=True,  # Show mean/median lines
    #     print_stats=True,  # Print statistics to console
    #     overlay_style='overlay'  # Can also use 'side_by_side'
    # )

    plotter.plotting_separate_histograms(
        dataframes=ship_datasets,
        y_variables=variables,
        variable_name_map=variable_map,
        name='Ship Type Comparison',
        n_hist_bins=30,
        density=True,  # Normalize to probability density
        show_kde=True,  # Show smooth density curves
        show_statistics=True,  # Show mean/median lines
        print_stats=True,  # Print statistics to console
        share_x_axis=False,  # Same x-axis range for easy comparison
        share_y_axis=False,  # Different y-axis (auto-scaled per subplot)
        single_figure=True,
        transpose=False,
        cell_width=5.0,
        cell_height=3.5,
        gridspec_top=0.95,
        base_font_size=12,
        stats_font_size=12,  # keep the text box legible on small cells
        suptitle_font_size=12,
    )


    # plotter.plotting_conditional_histograms(
    #     dataframes=ship_datasets,
    #     x_variables='DWT',  # Condition on deadweight tonnage
    #     y_variables=variables,
    #     name='Ship Dimensions by Size',
    #     n_x_bins=4,  # Split DWT into 4 ranges (quartiles)
    #     n_hist_bins=20,
    #     x_bin_labels=['Small', 'Medium', 'Large', 'X-Large'],  # Optional custom labels
    #     density=True,
    #     show_kde=True,
    #     show_statistics=True,
    #     print_stats=True,
    #     overlay_style='overlay'
    # )

    # # Check if data is normally distributed
    # plotter.plotting_separate_qqplots(
    #     dataframes=ship_datasets,
    #     y_variables=['LBP (M)', 'BEAM (M)', 'DRAFT (M)','DEPTH (M)','DWT', 'LBD'],
    #     distribution='norm',
    #     single_figure=True
    # )

    plotter.plotting_separate_qqplots(
        dataframes=ship_datasets,
        y_variables=variables,
        distribution='norm',
        single_figure=True,
        standardize=True
    )

    # # Compare alternative fuels vs conventional
    # plotter.plotting_separate_qqplots(
    #     dataframes=ship_datasets,
    #     y_variables=['LBP (M)', 'BEAM (M)', 'DRAFT (M)','DEPTH (M)','DWT', 'LBD'],
    #     reference_dataset='Conventional',
    #     single_figure=True
    # )

    alt_fuel_dict, dual_fuel_dict = get_database_tuples()

    plotter.plotting_bootstrap_ci_comparison(
    dataframes = alt_fuel_dict,
    y_variables = ['LBP (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)', 'LBD','LBT','LSW','MCR (KW)'],
    statistic = 'both',
    confidence_level = 0.95,
    single_figure = True
)

    plotter.plotting_bootstrap_ci_comparison(
        dataframes=dual_fuel_dict,
        y_variables=['LBP (M)', 'BEAM (M)', 'DRAFT (M)', 'DEPTH (M)', 'LBD','LBT','LSW','MCR (KW)'],
        statistic='both',
        confidence_level=0.95,
        single_figure=True
    )


if __name__ == '__main__':
    main()
