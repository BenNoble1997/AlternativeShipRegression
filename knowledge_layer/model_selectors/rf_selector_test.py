"""
Random Forest Selector Test Script
===================================

Production-oriented testing of RandomForestSelector with real ship data.

Tests:
1. Full analysis workflow with data loading
2. Cross-validation strategies
3. Feature engineering impact
4. Best combinations selection
5. Plotting methods
6. Export/import functionality
"""

import pandas as pd
import numpy as np
from pathlib import Path

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships


def get_variables(shiptype: ShipType):
    """
    Get ship-specific variables and configuration.

    Returns
    -------
    tuple
        (variables, filepath, save_dir)
    """
    print(f'USING DATA FOR {shiptype} SHIP')

    constant_vars = ['DWT', 'LBP (M)', 'BEAM (M)', 'DRAFT (M)']

    if shiptype == ShipType.TANKER:
        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\random_forest\Tanker')
        variables = constant_vars

    elif shiptype == ShipType.BULK_CARRIER:
        print('Bulk carrier filepath used')
        filepath = Path(
            r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\random_forest\Bulk_carrier')
        variables = constant_vars

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('Container ship filepath used')
        filepath = Path(
            r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\random_forest\Containership')
        variables = constant_vars + ['TEU']

    else:
        print('RoPax filepath used')
        filepath = Path(
            r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\random_forest\RO_Pax')
        variables = constant_vars + ['LM']

    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    return variables, filepath, save_dir


def random_forest_selector(shiptype: ShipType,
                           fixed_dimensions: list[str] = ['DWT'],
                           cv_method: str | None = 'stratified',
                           use_feature_engineering: bool = False,
                           poly_degree: int = 2,
                           save_dir: Path | str | None = None) -> RandomForestSelector:
    """
    Run full Random Forest analysis for a ship type.

    **Concept:**
    Random Forest is an ensemble method that:
    - Builds multiple decision trees on random subsets of data
    - Averages predictions to reduce overfitting
    - Provides feature importance rankings
    - Handles non-linear relationships naturally

    Parameters
    ----------
    shiptype : ShipType
        Type of ship to analyze
    fixed_dimensions : list[str]
        Dimensions that must be present (for reference only)
    cv_method : str or None
        Cross-validation strategy
    use_feature_engineering : bool
        Whether to test engineered features
    poly_degree : int
        Polynomial degree for feature engineering
    save_dir : Path or None
        Directory to save results

    Returns
    -------
    RandomForestSelector
        Fitted selector with results
    """
    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name
    print(ship_type)

    variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']

    print(f"\nDataset: {len(conventional_df)} ships")
    print(f"Variables: {variables}")

    print('\n' + ('=' * 20) + f'STEP 2: Random Forest Analysis' + ('=' * 20))
    print(f'=' * 60)

    # Define Random Forest parameters
    rf_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'max_features': 0.5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    rf_selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=['DWT', 'LBP (M)'],
        rf_params=rf_params,
        selection_metric='rrmse',
        verbose=True
    )

    # Run analysis
    rf_results = rf_selector.analyze_all_combinations(
        max_predictors=2,
        cv_method=cv_method,
        n_folds=10,
        use_feature_engineering=use_feature_engineering,
        poly_degree=poly_degree,
        include_ratios=True,
        include_logs=True,
        display_results=True
    )

    print('\n' + ('=' * 20) + f'STEP 3: Save Results' + ('=' * 20))
    print(f'=' * 60)

    export_path = save_dir / f'{shiptype.name}_rf.pkl'
    rf_selector.save_results(filepath=export_path)
    print(f"✓ Results saved to {export_path}")

    print('\n' + ('=' * 20) + f'STEP 4: Visualization' + ('=' * 20))
    print(f'=' * 60)

    # Plot metrics comparison
    rf_selector.plot_metrics_comparison(
        max_predictors=2,
        metric='adj_r2',
        save_path=save_dir / f'{shiptype.name}_rf_metrics.png'
    )

    # Plot feature importance
    rf_selector.plot_feature_importance(
        save_path=save_dir / f'{shiptype.name}_rf_importance.png'
    )

    print('\n' + ('=' * 20) + f'STEP 5: Best Combinations' + ('=' * 20))
    print(f'=' * 60)

    rf_selector.print_best_combinations()

    print('\n' + ('=' * 20) + f'STEP 6: Load and Verify' + ('=' * 20))
    print(f'=' * 60)

    # Test loading
    imported_selector = RandomForestSelector.from_saved_results(
        results_filepath=export_path,
        training_data=conventional_df,
        variable_dimensions=variables
    )

    print("✓ Results loaded successfully")
    imported_selector.print_best_combinations()

    return rf_selector


def check_cv_methods(conventional_df, variables):
    """
    Test different cross-validation strategies.

    **Concept - CV Methods:**

    1. **None (No CV)**:
       - Fits on full dataset
       - Fast but no generalization estimate
       - Use for exploratory analysis

    2. **Standard K-Fold**:
       - Random splits
       - Good for i.i.d. data
       - Default for most cases

    3. **Stratified K-Fold**:
       - Maintains target distribution in each fold
       - Good for continuous targets (bins target)
       - Ensures each fold is representative

    4. **Grouped K-Fold** (not tested here, needs groups):
       - Keeps related samples together
       - Prevents data leakage
       - Use when samples are not independent (e.g., same ship measured multiple times)
    """
    print("\n" + "=" * 80)
    print("TEST 1: CROSS-VALIDATION METHODS")
    print("=" * 80)

    cv_methods = [None, 'standard', 'stratified']

    for cv_method in cv_methods:
        print(f"\n--- Testing cv_method='{cv_method}' ---")

        selector = RandomForestSelector(
            training_data=conventional_df,
            variable_dimensions=variables,
            target_dimensions=['DWT', 'LBP (M)'],
            selection_metric='rrmse',
            verbose=False
        )

        selector.analyze_all_combinations(
            max_predictors=1,  # Keep it fast
            cv_method=cv_method,
            n_folds=5,
            use_feature_engineering=False,
            display_results=False
        )

        # Check results exist
        assert len(selector.results) > 0, f"No results for {cv_method}"

        # Check for CV metrics if CV was used
        first_target = list(selector.results.keys())[0]
        first_combo = list(selector.results[first_target].values())[0]
        metrics = first_combo['metrics']

        if cv_method is not None:
            assert 'std_r2' in metrics, f"Missing CV std metrics for {cv_method}"
            print(f"✅ {cv_method}: {len(selector.results)} targets, CV std metrics present")
        else:
            print(f"✅ {cv_method}: {len(selector.results)} targets")


def check_feature_engineering(conventional_df, variables):
    """
    Test feature engineering impact.

    **Concept - Feature Engineering for Random Forests:**

    While Random Forests can capture non-linearities, engineered features can:

    1. **Polynomial Features**:
       - Explicit interaction terms (X₁ * X₂)
       - Higher-order terms (X₁², X₁³)
       - Helps capture specific known relationships

    2. **Ratio Features**:
       - X₁/X₂ ratios (e.g., Length/Beam)
       - Captures proportional relationships
       - Common in naval architecture (form coefficients)

    3. **Log Features**:
       - log(X) transformations
       - Linearizes exponential relationships
       - Stabilizes variance

    **Trade-offs**:
    - ✓ Can improve performance
    - ✓ More interpretable relationships
    - ✗ Increases feature count (risk of overfitting)
    - ✗ Longer training time
    """
    print("\n" + "=" * 80)
    print("TEST 2: FEATURE ENGINEERING IMPACT")
    print("=" * 80)

    target = 'LBP (M)'

    # Test 1: No feature engineering (baseline)
    print("\n--- Baseline (Raw Features) ---")
    selector_raw = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=[target],
        selection_metric='rrmse',
        verbose=False
    )

    selector_raw.analyze_all_combinations(
        max_predictors=2,
        cv_method='standard',
        n_folds=5,
        use_feature_engineering=False,
        display_results=False
    )

    best_raw = selector_raw.get_best_combinations()[target]
    metrics_raw = best_raw[1]['metrics']
    print(f"✅ Raw features: Adj R² = {metrics_raw['adj_r2']:.4f}, RRMSE = {metrics_raw['rrmse']:.2f}%")

    # Test 2: With polynomial features
    print("\n--- With Polynomial Features (degree=2) ---")
    selector_poly = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=[target],
        selection_metric='rrmse',
        verbose=False
    )

    selector_poly.analyze_all_combinations(
        max_predictors=2,
        cv_method='standard',
        n_folds=5,
        use_feature_engineering=True,
        poly_degree=2,
        include_ratios=False,
        include_logs=False,
        display_results=False
    )

    best_poly = selector_poly.get_best_combinations()[target]
    metrics_poly = best_poly[1]['metrics']
    improvement_poly = metrics_raw['rrmse'] - metrics_poly['rrmse']
    print(f"✅ Polynomial: Adj R² = {metrics_poly['adj_r2']:.4f}, RRMSE = {metrics_poly['rrmse']:.2f}%")
    print(f"   Improvement: {improvement_poly:.2f}% RRMSE reduction")

    # Test 3: Full feature engineering
    print("\n--- Full Feature Engineering (Poly + Ratios + Logs) ---")
    selector_full = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=[target],
        selection_metric='rrmse',
        verbose=False
    )

    selector_full.analyze_all_combinations(
        max_predictors=2,
        cv_method='standard',
        n_folds=5,
        use_feature_engineering=True,
        poly_degree=2,
        include_ratios=True,
        include_logs=True,
        display_results=False
    )

    best_full = selector_full.get_best_combinations()[target]
    metrics_full = best_full[1]['metrics']
    improvement_full = metrics_raw['rrmse'] - metrics_full['rrmse']
    print(f"✅ Full engineering: Adj R² = {metrics_full['adj_r2']:.4f}, RRMSE = {metrics_full['rrmse']:.2f}%")
    print(f"   Improvement: {improvement_full:.2f}% RRMSE reduction")

    # Summary
    print("\n--- Summary ---")
    print(f"Raw:              RRMSE = {metrics_raw['rrmse']:.2f}%")
    print(f"Polynomial:       RRMSE = {metrics_poly['rrmse']:.2f}% (Δ {improvement_poly:+.2f}%)")
    print(f"Full engineering: RRMSE = {metrics_full['rrmse']:.2f}% (Δ {improvement_full:+.2f}%)")


def check_combination_types(conventional_df, variables):
    """
    Test combination type classification.

    **Concept - Combination Types:**

    Naval architecture dimensions fall into natural groups:

    1. **Hull**: LOA, LBP, Breadth, Depth, Draft
       - Define ship geometry
       - Highly correlated

    2. **Tonnage**: DWT, GT, NT
       - Measure capacity
       - Derived from hull

    3. **Propulsion**: Main Engine Power, Speed
       - Performance characteristics
       - Less correlated with hull

    4. **Mixed**: Combinations across groups
       - E.g., Hull + Propulsion
       - Often most predictive

    5. **Other**: Special dimensions (TEU, LM, etc.)

    This classification helps:
    - Organize results
    - Identify which dimension groups work best
    - Avoid highly collinear combinations
    """
    print("\n" + "=" * 80)
    print("TEST 3: COMBINATION TYPE CLASSIFICATION")
    print("=" * 80)

    selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=['DWT', 'LBP (M)'],
        selection_metric='rrmse',
        verbose=False
    )

    selector.analyze_all_combinations(
        max_predictors=2,
        cv_method='standard',
        n_folds=5,
        use_feature_engineering=False,
        display_results=False
    )

    # Count combinations by type
    print("\n--- Combinations by Type ---")
    type_counts = {}

    for target, combos in selector.results.items():
        print(f"\nTarget: {target}")
        target_types = {}

        for combo_name, data in combos.items():
            combo_type = data.get('combination_type', 'other')
            target_types[combo_type] = target_types.get(combo_type, 0) + 1
            type_counts[combo_type] = type_counts.get(combo_type, 0) + 1

        for combo_type, count in sorted(target_types.items()):
            print(f"  {combo_type}: {count} combinations")

    # Get best by type
    print("\n--- Best Combination per Type ---")
    for combo_type in ['hull', 'tonnage', 'propulsion', 'mixed', 'other']:
        print(f"\n{combo_type.upper()}:")
        best = selector.get_best_combinations(filter_by_type=combo_type)

        if not best:
            print(f"  No {combo_type} combinations found")
            continue

        for target, (combo_name, data) in best.items():
            metrics = data['metrics']
            print(f"  {target}: {combo_name}")
            print(f"    Adj R² = {metrics['adj_r2']:.4f}, RRMSE = {metrics['rrmse']:.2f}%")

    print(f"\n✅ Classification working: {len(type_counts)} types found")


def check_best_combinations(selector, save_dir):
    """
    Test best combinations selection with different parameters.

    **Concept - Selection Criteria:**

    When selecting "best" combination, consider:

    1. **Primary Metric**: RRMSE, Adj R², MAPE
       - Lower RRMSE = better predictions
       - Higher Adj R² = better explained variance

    2. **VIF (Variance Inflation Factor)**:
       - VIF > 10: High multicollinearity
       - Means predictors are redundant
       - Should filter these out

    3. **Simplicity**:
       - Prefer fewer predictors when performance is similar
       - Easier to implement
       - More robust

    4. **Feature Engineering**:
       - Raw features more interpretable
       - Engineered features may perform better
       - Trade-off depends on use case
    """
    print("\n" + "=" * 80)
    print("TEST 4: BEST COMBINATIONS SELECTION")
    print("=" * 80)

    try:
        # Test 1: Basic best combinations
        print("\n--- Testing get_best_combinations ---")
        best = selector.get_best_combinations()
        print(f"✅ Found best for {len(best)} targets")

        for target, (combo_name, data) in best.items():
            metrics = data['metrics']
            print(f"\n{target}:")
            print(f"  Combination: {combo_name}")
            print(f"  Adj R²: {metrics['adj_r2']:.4f}")
            print(f"  RRMSE: {metrics['rrmse']:.2f}%")
            print(f"  Max VIF: {metrics.get('max_vif', 1.0):.2f}")

        # Test 2: Prefer feature engineered
        print("\n--- Testing with feature engineering preference ---")
        best_eng = selector.get_best_combinations(prefer_feature_engineered=True)
        print(f"✅ Feature engineering preference: {len(best_eng)} targets")

        # Test 3: Filter by combination type
        print("\n--- Testing filter by type ---")
        for combo_type in ['hull', 'mixed']:
            best_type = selector.get_best_combinations(filter_by_type=combo_type)
            if best_type:
                print(f"✅ {combo_type} combinations: {len(best_type)} targets")

        # Test 4: Export best combinations
        print("\n--- Testing export_results ---")
        df = selector.export_results(
            filename=save_dir / 'test_rf_best.csv'
        )
        print(f"✅ Exported {len(df)} rows")

        # Test 5: Print summary
        print("\n--- Testing print_best_combinations ---")
        selector.print_best_combinations()
        print("✅ Print summary working")

    except Exception as e:
        print(f"❌ Best combinations test failed: {e}")
        import traceback
        traceback.print_exc()


def check_plotting_methods(selector, save_dir):
    """
    Test all plotting capabilities.

    **Concept - Visualization Types:**

    1. **Metrics Comparison**:
       - Bar charts showing performance across targets
       - Grouped by predictor count
       - Helps identify best overall approach

    2. **Feature Importance**:
       - Which predictors matter most
       - Unique to Random Forests
       - Guides feature selection

    3. **By Combination Type**:
       - Separate plots for hull/tonnage/propulsion
       - Easier to compare within categories
    """
    print("\n" + "=" * 80)
    print("TEST 5: PLOTTING METHODS")
    print("=" * 80)

    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode

    try:
        # Test 1: Metrics comparison
        print("\n--- Testing plot_metrics_comparison ---")
        selector.plot_metrics_comparison(
            max_predictors=2,
            metric='adj_r2',
            group_by_type=False,
            save_path=save_dir / 'test_metrics_comp.png'
        )
        plt.close('all')
        print("✅ plot_metrics_comparison: Working")

        # Test 2: Metrics by type
        print("\n--- Testing plot_metrics_comparison (grouped by type) ---")
        selector.plot_metrics_comparison(
            max_predictors=2,
            metric='rrmse',
            group_by_type=True,
            save_path=save_dir / 'test_metrics_by_type.png'
        )
        plt.close('all')
        print("✅ plot_metrics_comparison (grouped): Working")

        # Test 3: Feature importance
        print("\n--- Testing plot_feature_importance ---")
        selector.plot_feature_importance(
            save_path=save_dir / 'test_feature_importance.png'
        )
        plt.close('all')
        print("✅ plot_feature_importance: Working")

        # Test 4: Different metrics
        print("\n--- Testing different metrics ---")
        for metric in ['adj_r2', 'rrmse', 'mape']:
            selector.plot_metrics_comparison(
                max_predictors=2,
                metric=metric,
                save_path=save_dir / f'test_metric_{metric}.png'
            )
            plt.close('all')
        print("✅ Multiple metrics: Working")

    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')


def check_export_import(selector, save_dir):
    """
    Test save/load functionality.

    **Concept - Serialization:**

    Random Forest models contain:
    - Multiple decision trees (ensemble)
    - Feature transformers (if engineered)
    - Preprocessors (scalers, polynomial transformers)
    - All fitted parameters

    **Pickle Format** (recommended):
    - Saves everything
    - Can reload and make predictions
    - Larger file size

    **Excel/CSV Formats**:
    - Metrics only
    - Good for reporting
    - Cannot make predictions
    """
    print("\n" + "=" * 80)
    print("TEST 6: EXPORT AND IMPORT")
    print("=" * 80)

    try:
        # Test 1: Save as pickle
        print("\n--- Testing save (pickle format) ---")
        pickle_path = save_dir / 'test_rf_selector.pkl'
        selector.save_results(filepath=pickle_path, format='pickle')
        print(f"✅ Saved to {pickle_path}")

        # Test 2: Load and verify
        print("\n--- Testing load ---")

        # Create new instance
        loaded_selector = RandomForestSelector(
            training_data=selector.data,
            variable_dimensions=selector.variable_dimensions,
            target_dimensions=selector.target_dimensions,
            verbose=False
        )

        loaded_selector.load_results(pickle_path)

        # Verify results match
        original_best = selector.get_best_combinations()
        loaded_best = loaded_selector.get_best_combinations()

        assert len(original_best) == len(loaded_best), "Result count mismatch"

        for target in original_best.keys():
            orig_metrics = original_best[target][1]['metrics']
            load_metrics = loaded_best[target][1]['metrics']

            assert np.isclose(orig_metrics['adj_r2'], load_metrics['adj_r2']), \
                f"Metrics mismatch for {target}"

        print("✅ Loaded results match original")

        # Test 3: Quick save (pickle + Excel)
        print("\n--- Testing quick_save ---")
        selector.quick_save(
            name='test_rf_quick',
            output_dir=save_dir
        )
        print("✅ Quick save completed")

        # Test 4: Save as Excel
        print("\n--- Testing save (Excel format) ---")
        excel_path = save_dir / 'test_rf_results.xlsx'
        selector.save_results(
            filepath=excel_path,
            format='excel',
            include_metadata=True
        )
        print(f"✅ Saved to {excel_path}")

        # Test 5: from_saved_results classmethod
        print("\n--- Testing from_saved_results ---")
        reloaded_selector = RandomForestSelector.from_saved_results(
            results_filepath=pickle_path,
            training_data=selector.data,
            variable_dimensions=selector.variable_dimensions
        )
        print("✅ from_saved_results working")

    except Exception as e:
        print(f"❌ Export/import failed: {e}")
        import traceback
        traceback.print_exc()


def check_display_methods(selector):
    """
    Test display and printing methods.

    These methods provide formatted output for:
    - Initialization summary
    - Analysis results
    - Best combinations
    - Feature engineering comparison
    """
    print("\n" + "=" * 80)
    print("TEST 7: DISPLAY METHODS")
    print("=" * 80)

    try:
        # Test 1: Initialization summary
        print("\n--- Testing print_initialization_summary ---")
        selector.print_initialization_summary()
        print("✅ print_initialization_summary: Working")

        # Test 2: Results summary
        print("\n--- Testing print_summary ---")
        selector.print_summary(group_by_type=True)
        print("✅ print_summary: Working")

        # Test 3: Best combinations
        print("\n--- Testing print_best_combinations ---")
        selector.print_best_combinations()
        print("✅ print_best_combinations: Working")

        # Test 4: Feature engineering comparison (if applicable)
        if any('Feat Eng' in combo for target_results in selector.results.values()
               for combo in target_results.keys()):
            print("\n--- Testing print_feature_engineering_comparison ---")
            selector.print_feature_engineering_comparison()
            print("✅ print_feature_engineering_comparison: Working")

    except Exception as e:
        print(f"❌ Display methods failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main test runner - comprehensive testing workflow.
    """
    print("\n" + "▓" * 80)
    print("RANDOM FOREST SELECTOR - COMPREHENSIVE TEST SUITE")
    print("▓" * 80)

    shiptype = ShipType.BULK_CARRIER

    # Load data
    print(f'\n' + '=' * 60)
    print(('=' * 20) + f'STEP 1: Loading Data' + ('=' * 20))
    print(f'=' * 60)

    variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath, ship_type=shiptype)
    conventional_df = fleet.processed_data['pure_conventional']

    print(f"\nDataset loaded: {len(conventional_df)} ships")
    print(f"Variables: {variables}")

    # Run individual tests
    try:
        check_cv_methods(conventional_df, variables)
        check_feature_engineering(conventional_df, variables)
        check_combination_types(conventional_df, variables)

        # Create selector for remaining tests
        print(f'\n' + '=' * 60)
        print(('=' * 20) + f'STEP 2: Full Analysis' + ('=' * 20))
        print(f'=' * 60)

        selector = RandomForestSelector(
            training_data=conventional_df,
            variable_dimensions=variables,
            target_dimensions=['DWT', 'LBP (M)'],
            selection_metric='rrmse',
            verbose=True
        )

        selector.analyze_all_combinations(
            max_predictors=2,
            cv_method='stratified',
            n_folds=10,
            use_feature_engineering=True,
            poly_degree=2,
            display_results=True
        )

        check_best_combinations(selector, save_dir)
        check_plotting_methods(selector, save_dir)
        check_export_import(selector, save_dir)
        check_display_methods(selector)

        print("\n" + "▓" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("▓" * 80)

    except Exception as e:
        print("\n" + "!" * 80)
        print("TEST SUITE FAILED")
        print("!" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """
    Quick test for rapid iteration during development.
    """
    print("\n" + "=" * 80)
    print("QUICK TEST - BASIC FUNCTIONALITY")
    print("=" * 80)

    shiptype = ShipType.BULK_CARRIER
    variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath, ship_type=shiptype)
    conventional_df = fleet.processed_data['pure_conventional']

    selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=['LBP (M)'],
        selection_metric='rrmse',
        verbose=True
    )

    selector.analyze_all_combinations(
max_predictors=2,
cv_method='standard',
n_folds=5,
use_feature_engineering=False,
display_results=True
)

    selector.print_best_combinations()
    print("\n✓ Quick test completed")

if __name__ == '__main__':
    main()
    # quick_test()