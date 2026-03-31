
import pandas as pd
import numpy as np
from pathlib import Path


from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships


def get_variables(shiptype:ShipType):

    print(f'USING DATA FOR {shiptype} SHIP')

    constant_vars = ['DWT','LBP (M)','BEAM (M)','DRAFT (M)']

    if shiptype == ShipType.TANKER:

        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\parametric_regression\Tanker')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
        variables = constant_vars


    elif shiptype == ShipType.BULK_CARRIER:
        print('bulk carrier filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\parametric_regression\Bulk_carrier')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
        variables = constant_vars
        # variables = ['DWT', 'LBP (M)', 'DRAFT (M)']

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('container ship filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\parametric_regression\Containership')
        bins = None
        variables = constant_vars + ['TEU']

    else:
        print('ropax filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\parametric_regression\RO_Pax')
        bins = None
        variables = constant_vars + ['LM']

    return bins,variables, filepath, save_dir

def parametric_selector(shiptype:ShipType,
                        fixed_dimensions:list[str]=['DWT'],
                        cv_method:str|None = 'stratified',
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,
                        save_dir:Path|str|None = None) -> pd.DataFrame:
    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    print(ship_type)

    bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']



    print(('=' * 20) + f'STEP 2: Parametric regressions' + ('=' * 20))
    print(f'=' * 60)

    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variables,
                                            target_dimensions=['DWT','LBP (M)'],
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='bic',
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization,
                                            fit_mfp=True)

    poly_results = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

    export_path=save_dir/f'{shiptype.name}_poly.pkl'

    poly_selector.save_results(filepath=export_path)

    poly_selector.plot_metrics_comparison()

    poly_selector.print_formulas_for_target('LBP (M)', top_n=5)

    poly_selector.print_best_formulas()


    imported_selector = ParametricModelSelector.from_saved_results(results_filepath=export_path,
        training_data=conventional_df,
        variable_dimensions=variables
    )

    imported_selector.print_formulas_for_target('LBP (M)', top_n=5)

    imported_selector.print_best_formulas()


def check_best_combinations(selector,
                           save_dir):
    """Test best combinations selection with different parameters"""

    print("\n" + "=" * 80)
    print("TEST 6: BEST COMBINATIONS SELECTION")
    print("=" * 80)

    try:
        # Test 1: Basic best combinations
        print("\n--- Testing get_best_combinations ---")
        best = selector.get_best_combinations()
        print(f"✅ Found best for {len(best)} targets")

        # Test 2: With VIF filtering
        print("\n--- Testing with VIF filtering ---")
        best_vif = selector.get_best_combinations(max_vif=5.0)
        print(f"✅ VIF filtering: {len(best_vif)} targets")

        # Test 3: Export best combinations
        print("\n--- Testing export_best_combinations ---")
        df = selector.export_best_combinations(
            filename=save_dir / 'test_best.csv',
            max_vif=10.0
        )
        print(f"✅ Exported {len(df)} best combinations")

        # Test 4: Print summary
        print("\n--- Testing print_best_combinations_summary ---")
        selector.print_best_combinations_summary()
        print("✅ Print summary working")

    except Exception as e:
        print(f"❌ Best combinations test failed: {e}")


def check_plotting_methods(selector):
    """Test all plotting capabilities"""

    print("\n" + "=" * 80)
    print("TEST 5: PLOTTING METHODS")
    print("=" * 80)

    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode to prevent display

    try:
        # Test 1: Metrics comparison
        print("\n--- Testing plot_metrics_comparison ---")
        selector.plot_metrics_comparison(
            max_predictors=2,
            metric='adj_r2',
            save_path=None  # Don't save
        )
        plt.close('all')
        print("✅ plot_metrics_comparison: Working")

        # Test 2: CV stability (if CV was used)
        if 'std_r2' in list(selector.detailed_results.values())[0][
            list(list(selector.detailed_results.values())[0].keys())[0]]:
            print("\n--- Testing plot_cv_stability ---")
            selector.plot_cv_stability(max_predictors=2)
            plt.close('all')
            print("✅ plot_cv_stability: Working")

        # Test 3: Single predictor regressions
        print("\n--- Testing plot_single_predictor_regressions ---")
        target = selector.target_dimensions[0]
        selector.plot_single_predictor_regressions(
            target=target,
            top_n=3,
            save_path=None
        )
        plt.close('all')
        print("✅ plot_single_predictor_regressions: Working")

    except Exception as e:
        print(f"❌ Plotting failed: {e}")
    finally:
        plt.close('all')


def check_selection_metrics(conventional_df, variables):
    """Test different selection metrics"""

    print("\n" + "=" * 80)
    print("TEST 3: SELECTION METRICS")
    print("=" * 80)

    metrics = ['bic', 'aic', 'adj_r2', 'rrmse']

    for metric in metrics:
        print(f"\n--- Testing selection_metric='{metric}' ---")

        selector = ParametricModelSelector(
            conventional_df,
            variable_dimensions=variables,
            target_dimensions=['LBP (M)'],
            selection_metric=metric,
            verbose=False
        )

        selector.analyze_all_combinations(
            max_predictors=1,
            cv_method=None,  # Faster
            display_results=False
        )

        # Check that metric appears in results
        for key, models in selector.detailed_results.items():
            for model_name, model_metrics in models.items():
                assert metric in model_metrics, f"Metric {metric} not in results"
                break
            break

        print(f"✅ {metric}: Selection working correctly")


def check_cv_methods(conventional_df, variables):
    """Test different cross-validation strategies"""

    print("\n" + "=" * 80)
    print("TEST 2: CROSS-VALIDATION METHODS")
    print("=" * 80)

    cv_methods = [None, 'standard', 'stratified']

    for cv_method in cv_methods:
        print(f"\n--- Testing cv_method='{cv_method}' ---")

        selector = ParametricModelSelector(
            conventional_df,
            variable_dimensions=variables,
            target_dimensions=['DWT', 'LBP (M)'],
            selection_metric='bic',
            verbose=False  # Reduce output
        )

        selector.analyze_all_combinations(
            max_predictors=1,  # Keep it fast
            cv_method=cv_method,
            display_results=False
        )

        # Check results exist
        assert len(selector.model_recommendations) > 0, f"No results for {cv_method}"
        print(f"✅ {cv_method}: {len(selector.model_recommendations)} combinations fitted")



def main():
    shiptype = ShipType.BULK_CARRIER

    # parametric_selector(shiptype=shiptype,)

    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    print(ship_type)

    bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']

    export_path = save_dir / f'{shiptype.name}_poly.pkl'

    selector = ParametricModelSelector.from_saved_results(results_filepath=export_path,
                                                                   training_data=conventional_df,
                                                                   variable_dimensions=variables
                                                                   )

    check_selection_metrics(conventional_df, variables)

    check_selection_metrics(conventional_df, variables)

    check_plotting_methods(selector)

    check_best_combinations(selector,save_dir=save_dir)




if __name__ == '__main__':
    main()