
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
#from knowledge_layer.coefficient_and_importance_analysis.parametric_coefficient_analysier import CoefficientComparisonAnalyzer

from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import get_variables, getting_alt_fuel_ships

from parameters.shipshape.ShipShapeBasic_class import ShipType



def selectors(shiptype: ShipType,
              conventional_df: pd.DataFrame,
              fixed_dimensions: List[str] = ['DWT'],
              cv_method: str | None = None,
              include_regularized: bool = True,
              use_cv_regularization: bool = True,
              save_dir: Path | str | None = None,
              selection_criteria:str='bic',
              outliers:str='percentile',
              exclude_models=['ridge','ridge-cv', 'lasso-cv','polynomial-2-ridge','polynomial-3-ridge'],
              suffix:str='_v4') -> tuple:
    """
    Run both Parametric and Random Forest model selection for a ship type.

    Optimized for overnight batch processing with timestamps and error handling.

    Parameters
    ----------
    shiptype : ShipType
        Type of ship to analyze
    conventional_df : pd.DataFrame
        Conventional ship data
    fixed_dimensions : List[str]
        Fixed dimensions (for reference only)
    cv_method : str or None
        Cross-validation method (None for faster runtime)
    include_regularized : bool
        Include Ridge/Lasso models
    use_cv_regularization : bool
        Use CV for regularization parameter tuning
    save_dir : Path or None
        Directory to save results

    Returns
    -------
    tuple
        (poly_results, poly_selector, rf_results, rf_selector)
    """
    import time
    from datetime import datetime, timedelta

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    start_time = time.time()
    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} RUNNING FULL ANALYSIS FOR {shiptype} {"=" * 30}')
    print(f'{"=" * 100}')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    ship_type = shiptype.name
    bins, variables,_, filepath, save_dir,_ = get_variables(shiptype=shiptype)

    print(f'\nDataset: {len(conventional_df)} ships')
    print(f'Variables: {len(variables)} dimensions')
    print(f'CV method: {cv_method if cv_method else "None (faster)"}')

    # ========================================================================
    # STEP 1: PARAMETRIC MODEL SELECTION
    # ========================================================================

    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} STEP 1: PARAMETRIC REGRESSION {"=" * 30}')
    print(f'{"=" * 100}')

    step1_start = time.time()

    try:
        poly_selector = ParametricModelSelector(
            conventional_df,
            variable_dimensions=variables,
            target_dimensions=variables,
            verbose=True,  # ✅ CHANGED: True for progress tracking
            fixed_dimensions=fixed_dimensions,
            selection_metric=selection_criteria,
            selection_strategy='conservative',  # ✅ ADDED: Conservative selection
            assumptions='flexible',  # ✅ ADDED: Moderate assumption checking
            vif_threshold=10.0,  # ✅ ADDED: VIF filtering
            include_regularized=include_regularized,
            use_cv_regularization=use_cv_regularization,
            fit_mfp=True
        )

        print(f'\n⏱️  Parametric selector initialized')
        print(f'   Selection: BIC with conservative strategy')
        print(f'   Assumptions: moderate (filters by VIF and heteroscedasticity)')
        print(f'   Models: Linear, Polynomial, MFP' +
              (' + Ridge/Lasso' if include_regularized else ''))

        poly_results = poly_selector.analyze_all_combinations(
            max_predictors=2,
            cv_method=cv_method,
            exclude_models=exclude_models,
            display_results=True  # ✅ CHANGED: Show results
        )

        step1_time = time.time() - step1_start
        print(f'\n✅ Parametric analysis complete')
        print(f'   Time: {timedelta(seconds=int(step1_time))}')
        print(f'   Combinations tested: {len(poly_selector.model_recommendations)}')

    except Exception as e:
        print(f'\n❌ Parametric analysis FAILED: {e}')
        import traceback
        traceback.print_exc()
        poly_results = None
        poly_selector = None
        step1_time = time.time() - step1_start

    # ========================================================================
    # STEP 2: RANDOM FOREST MODEL SELECTION
    # ========================================================================

    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} STEP 2: RANDOM FOREST REGRESSION {"=" * 30}')
    print(f'{"=" * 100}')

    step2_start = time.time()

    try:
        rf_selector = RandomForestSelector(
            training_data=conventional_df,
            variable_dimensions=variables,
            target_dimensions=variables,
            fixed_dimensions=['DWT'],
            rf_params={
                'n_estimators': 100,
                'max_depth': 10,
                'max_features': 0.5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1  # Use all CPU cores
            },
            selection_metric='rrmse',
            verbose=True  # ✅ CHANGED: True for progress tracking
        )

        print(f'\n⏱️  Random Forest selector initialized')
        print(f'   Selection: RRMSE (optimal for non-parametric models)')
        print(f'   RF params: 100 trees, depth=10, max_features=0.5')
        print(f'   Feature engineering: Disabled (raw features only)')

        rf_results = rf_selector.analyze_all_combinations(
            max_predictors=2,
            cv_method='stratified',  # Better than 'standard' for ship data
            n_folds=10,
            random_state=42,
            use_feature_engineering=False,  # ✅ Raw features only (faster + simpler)
            display_results=True  # ✅ CHANGED: Show results
        )

        step2_time = time.time() - step2_start
        print(f'\n✅ Random Forest analysis complete')
        print(f'   Time: {timedelta(seconds=int(step2_time))}')

        # Count total combinations
        total_combos = sum(len(combos) for combos in rf_selector.results.values())
        print(f'   Combinations tested: {total_combos}')

    except Exception as e:
        print(f'\n❌ Random Forest analysis FAILED: {e}')
        import traceback
        traceback.print_exc()
        rf_results = None
        rf_selector = None
        step2_time = time.time() - step2_start

    # ========================================================================
    # STEP 3: SAVE RESULTS
    # ========================================================================

    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} STEP 3: SAVING RESULTS {"=" * 30}')
    print(f'{"=" * 100}')

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save Parametric results
        if poly_selector is not None:
            try:
                poly_path = save_dir / f'{shiptype.name}_parametric_selector{suffix}.pkl'
                poly_selector.save_results(poly_path)
                print(f'✅ Parametric results saved: {poly_path.name}')

                # Also save Excel for easy inspection
                poly_excel = save_dir / f'{shiptype.name}_parametric_results_{timestamp}.xlsx'
                poly_selector.save_results(poly_excel, format='excel')
                print(f'   Excel report: {poly_excel.name}')

            except Exception as e:
                print(f'❌ Failed to save parametric results: {e}')

        # Save Random Forest results
        if rf_selector is not None:
            try:
                rf_path = save_dir / f'{shiptype.name}_rf_selector{suffix}.pkl'
                rf_selector.save_results(rf_path)
                print(f'✅ Random Forest results saved: {rf_path.name}')

                # Also save Excel for easy inspection
                rf_excel = save_dir / f'{shiptype.name}_rf_results_{timestamp}.xlsx'
                rf_selector.save_results(rf_excel, format='excel')
                print(f'   Excel report: {rf_excel.name}')

            except Exception as e:
                print(f'❌ Failed to save RF results: {e}')

    # ========================================================================
    # SUMMARY
    # ========================================================================

    total_time = time.time() - start_time

    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} ANALYSIS COMPLETE {"=" * 30}')
    print(f'{"=" * 100}')
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Total runtime: {timedelta(seconds=int(total_time))}')

    if poly_selector is not None:
        print(f'\nParametric:')
        print(f'  Time: {timedelta(seconds=int(step1_time))}')
        print(f'  Combinations: {len(poly_selector.model_recommendations)}')
        print(f'  Targets: {len(poly_selector.target_dimensions)}')

    if rf_selector is not None:
        print(f'\nRandom Forest:')
        print(f'  Time: {timedelta(seconds=int(step2_time))}')
        total_rf = sum(len(combos) for combos in rf_selector.results.values())
        print(f'  Combinations: {total_rf}')
        print(f'  Targets: {len(rf_selector.target_dimensions)}')

    print(f'\n{"=" * 100}\n')

    return poly_results, poly_selector, rf_results, rf_selector


# ============================================================================
# BATCH PROCESSING FUNCTION FOR OVERNIGHT RUNS
# ============================================================================

def batch_selectors(shiptypes: List[ShipType],
                    data_dict: dict[ShipType, pd.DataFrame],
                    cv_method: str | None = "stratified",
                    save_base_dir: Path | str = Path('results'),
                    log_file: str | None = None,
                    selection_criteria:str='bic',
                    suffix:str='_v4',
                    outliers:str='percentile'):
    """
    Run selector analysis for multiple ship types with comprehensive logging.
    """
    import sys
    from datetime import datetime, timedelta
    import time

    save_base_dir = Path(save_base_dir)
    save_base_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if log_file:
        log_path = save_base_dir / log_file
        # ✅ FIX: Specify UTF-8 encoding
        log_f = open(log_path, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = log_f

    batch_start = time.time()

    # ✅ FIX: Replace Unicode block characters with simple ASCII
    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} BATCH ANALYSIS START {"=" * 30}')
    print(f'{"=" * 100}')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Ship types: {[st.name for st in shiptypes]}')
    print(f'CV method: {cv_method if cv_method else "None (fastest)"}')
    print(f'Save directory: {save_base_dir}')

    results_summary = {}

    for i, shiptype in enumerate(shiptypes, 1):
        print(f'\n{"#" * 100}')
        print(f'{"#" * 30} SHIP TYPE {i}/{len(shiptypes)}: {shiptype.name} {"#" * 30}')
        print(f'{"#" * 100}')

        if shiptype not in data_dict:
            print(f'WARNING: No data for {shiptype.name}, skipping...')
            continue

        conventional_df = data_dict[shiptype]

        try:
            poly_results, poly_selector, rf_results, rf_selector = selectors(
                shiptype=shiptype,
                conventional_df=conventional_df,
                fixed_dimensions=['DWT'],
                cv_method=cv_method,
                include_regularized=True,
                use_cv_regularization=True,
                save_dir=save_base_dir / shiptype.name/suffix,
                selection_criteria=selection_criteria,
                suffix=suffix,
                outliers=outliers,
            )

            results_summary[shiptype.name] = {
                'status': 'SUCCESS',
                'parametric_combinations': len(poly_selector.model_recommendations) if poly_selector else 0,
                'rf_combinations': sum(len(c) for c in rf_selector.results.values()) if rf_selector else 0,
                'error': None
            }

        except Exception as e:
            print(f'\nCRITICAL ERROR for {shiptype.name}: {e}')
            import traceback
            traceback.print_exc()

            results_summary[shiptype.name] = {
                'status': 'FAILED',
                'parametric_combinations': 0,
                'rf_combinations': 0,
                'error': str(e)
            }

    # Final summary
    batch_time = time.time() - batch_start

    print(f'\n{"=" * 100}')
    print(f'{"=" * 30} BATCH ANALYSIS COMPLETE {"=" * 30}')
    print(f'{"=" * 100}')
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Total runtime: {timedelta(seconds=int(batch_time))}')

    print(f'\n{"=" * 100}')
    print(f'SUMMARY')
    print(f'{"=" * 100}')

    for ship_name, summary in results_summary.items():
        status_icon = '[OK]' if summary['status'] == 'SUCCESS' else '[FAIL]'
        print(f'\n{status_icon} {ship_name}: {summary["status"]}')
        if summary['status'] == 'SUCCESS':
            print(f'   Parametric: {summary["parametric_combinations"]} combinations')
            print(f'   Random Forest: {summary["rf_combinations"]} combinations')
        else:
            print(f'   Error: {summary["error"]}')

    successful = sum(1 for s in results_summary.values() if s['status'] == 'SUCCESS')
    print(f'\n{"=" * 100}')
    print(f'Success rate: {successful}/{len(shiptypes)} ship types')
    print(f'{"=" * 100}\n')

    # Cleanup logging
    if log_file:
        sys.stdout = original_stdout
        log_f.close()
        print(f'Log saved to: {log_path}')

    return results_summary


# ============================================================================
# USAGE EXAMPLE FOR OVERNIGHT RUNS
# ============================================================================



def run_overnight_analysis(save_base_dir:Path,
                           selection_criteria:str='bic',
                           outliers:str='percentile',

                           suffix:str='_no_outliers'):
    """
    Example overnight batch processing script.

    Run this before going to sleep!
    """
    from parameters.shipshape.ShipShapeBasic_class import ShipType

    # Define ship types to analyze
    shiptypes = [
        ShipType.TANKER,
        ShipType.BULK_CARRIER,
        ShipType.CONTAINER_SHIP,
        ShipType.RO_PAX,  # Add if needed
    ]

    # Load data for each ship type
    print("Loading datasets...")
    data_dict = {}

    fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
    max_build_year= 1990
    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['Biofuel']

    alt_fuels = {'Alt_fuel': alt_fuel,
                 'Biofuel': bio_fuels}

    for shiptype in shiptypes:
        try:
            _, variables,_, filepath, _,_ = get_variables(shiptype=shiptype)
            fleet = getting_alt_fuel_ships(filepath=filepath, shiptype=shiptype.string, outlier_method=outliers,
                                           fuel_type_columns=fuel_type_columns,
                                            max_build_year=max_build_year, conventional_fuels=conventional_fuels,
                                            alt_fuels=alt_fuels, outliers=False)
            data_dict[shiptype] = fleet.processed_data['pure_conventional']
            print(f"✅ {shiptype.name}: {len(data_dict[shiptype])} ships")
        except Exception as e:
            print(f"❌ Failed to load {shiptype.name}: {e}")

    # Run batch analysis
    results_summary = batch_selectors(
        shiptypes=shiptypes,
        data_dict=data_dict,
        cv_method=None,  # ✅ None = fastest (good for overnight runs)
        save_base_dir=save_base_dir,
        log_file=f'batch_analysis_{selection_criteria}.log',
        selection_criteria=selection_criteria,
        outliers=outliers,
        suffix=suffix,
    )

    return results_summary


if __name__ == '__main__':
    # Run the overnight analysis

    save_base_dir = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\results\new')

    run_overnight_analysis(save_base_dir=save_base_dir,
                           selection_criteria='aic',
                           outliers='percentile',
                           suffix='_sister_ship_new_preproc_3')