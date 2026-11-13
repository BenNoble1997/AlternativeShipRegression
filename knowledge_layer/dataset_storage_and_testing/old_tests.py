if matching_sensitivity:
    print("\n" + "=" * 80)
    print("RUNNING SENSITIVITY ANALYSIS")
    print("=" * 80)

    try:
        # For pure alternative fuel ships
        sens_pure, robust_pure = sensitivity_analysis_workflow(
            fleet=fleet,
            conventional_dataset='pure_conventional',
            alternative_dataset=alt_dataset,
            ship_type=ship_type,
            test_features=features,
            matching_features=matching_features,
            M_W_test=True,
            K_S_test=False,
            welch_test=False
        )

        plot_sensitivity_heatmap(sensitivity_df=sens_pure,
                                 matching_features=matching_features,
                                 fuel_type='Pure_Alt_Fuel')
    except Exception as e:
        print(f"ERROR in pure alt fuel sensitivity analysis: {e}")
        import traceback

        traceback.print_exc()

if progressive_comparison:
    print("\n" + "=" * 80)
    print("RUNNING PROGRESSIVE COMPARISON ANALYSIS")
    print("=" * 80)

    variable_name_map = {
        'LBP (M)': r'$L_{BP}$ (m)',
        'LBP (M)/DWT': r'$L_{BP}/\text{DWT}$',  # NEW: Add ratio mappings
        'BEAM (M)': 'Beam (m)',
        'BEAM (M)/DWT': r'$B/\text{DWT}$',
        'DRAFT (M)': 'Draft (m)',
        'DRAFT (M)/DWT': r'$T/\text{DWT}$',
        # ... etc
    }

    output_dir = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\experiment0'

    # Run ABSOLUTE value analysis
    print("\n" + "-" * 80)
    print("ABSOLUTE VALUE ANALYSIS")
    print("-" * 80)
    absolute_results = run_progressive_for_all_datasets(
        fleet=fleet,
        ship_type=ship_type,
        test_features=features,
        matching_parameter='DWT',
        output_dir=output_dir,
        variable_name_map=variable_name_map,
        print_results=True,
        use_ratios=False,  # Absolute values
        debug=False
    )

    # Run RATIO analysis
    print("\n" + "-" * 80)
    print("RATIO ANALYSIS")
    print("-" * 80)
    ratio_results = run_progressive_for_all_datasets(
        fleet=fleet,
        ship_type=ship_type,
        test_features=features,
        matching_parameter='DWT',
        output_dir=output_dir,
        variable_name_map=variable_name_map,
        print_results=True,
        use_ratios=True,  # Test ratios
        debug=True  # Enable debugging for first run
    )

# ONLY export tables if we have data to export
# Check for pure alt fuel results
if any([progressive_pure is not None, sens_pure is not None,
        patterns_pure is not None, robust_pure is not None]):
    print("\n" + "=" * 80)
    print("EXPORTING PURE ALT FUEL TABLES")
    print("=" * 80)
    try:
        export_all_tables(
            progressive_df=progressive_pure,
            sensitivity_df=sens_pure,
            pattern_df=patterns_pure,
            robustness_df=robust_pure,
            matching_features=matching_features,
            output_dir=f"latex_tables_{ship_type.name}_pure_alt",
            prefix=f"{alt_datasets}_fuel_{ship_type.name}"
        )
    except Exception as e:
        print(f"ERROR exporting pure alt fuel tables: {e}")
        import traceback

        traceback.print_exc()

# Check for dual fuel results
if any([progressive_dual is not None, sens_dual is not None,
        patterns_dual is not None, robust_dual is not None]):
    print("\n" + "=" * 80)
    print("EXPORTING DUAL FUEL TABLES")
    print("=" * 80)
    try:
        export_all_tables(
            progressive_df=progressive_dual,
            sensitivity_df=sens_dual,
            pattern_df=patterns_dual,
            robustness_df=robust_dual,
            matching_features=matching_features,
            output_dir=f"latex_tables_{ship_type.name}_dual",
            prefix=f"dual_fuel_{ship_type.name}"
        )
    except Exception as e:
        print(f"ERROR exporting dual fuel tables: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)