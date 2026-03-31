
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_regression_analyser_class import ParametricDomainAnalyzer
#from knowledge_layer.coefficient_and_importance_analysis.parametric_coefficient_analysier import CoefficientComparisonAnalyzer
from knowledge_layer.multi_dataset_analysers.coefficient_comparison.parametric_coefficient_comparison import CoefficientComparisonAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.parametric_cross_dataset_analysis import CrossDatasetPerformanceAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.multi_dataset_cross_performance_analyzer import MultiDatasetCrossPerformanceAnalyzer

from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.random_forest_domain_analysis_clas import RandomForestDomainAnalyzer
from knowledge_layer.multi_dataset_analysers.rf_feature_importance.rf_feature_importance_Full import RFFullFeatureImportanceAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.rf_cross_dataset_analysis import RandomForestCrossDatasetAnalyzer

from knowledge_layer.multi_dataset_analysers.domain_shift.model_comparison_analyser_class import ModelComparisonAnalyzer

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships
from knowledge_layer.fleet_data_class.data_checking import LatexAvailabilityTable

from knowledge_layer.multi_dataset_analysers.ParametricDomainShiftLaTeX import ParametricDomainAnalysisLatexTableGenerator
from knowledge_layer.multi_dataset_analysers.RFDomainShiftLaTeX import RandomForestLatexTableGenerator

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.literature_regressions.Literature_formula_evaluation_class import LiteratureFormulaEvaluator
from knowledge_layer.literature_regressions.literature_formula_libraries_class import LiteratureFormulaLibrary
from knowledge_layer.multi_dataset_analysers.multi_domain_shift_analysis import MultiDatasetDomainAnalyzer
from knowledge_layer.literature_regressions.multi_fuel_empirical_evaluation import MultiDatasetLiteratureEvaluator

def get_variable_name_maps():
    # variable_map = {
    #     'DWT': 'DWT (t)',
    #     'GT': 'GT (m³)',
    #     'LOA (M)': 'LOA (m)',
    #     'LBP (M)': 'LBP (m)',
    #     'BEAM (M)': 'Beam (m)',
    #     'DEPTH (M)': 'Depth (m)',
    #     'DRAFT (M)': 'Draught (m)',
    #     'TEU': 'TEU',
    #     'Vs': 'Vs (kn)'
    # }

    variable_map = {
        'DWT': 'DWT',
        'GT': 'GT',
        'LOA (M)': r'$L_{OA}$',
        'LOA(M)': r'$L_{OA}$',
        'LBP (M)': r'$L_{BP}$',
        'LBP(M)': r'$L_{BP}$',
        'BEAM (M)': r'$B$',
        'DEPTH (M)': r'$D$',
        'DEPTH(M)': r'$D$',
        'DRAFT (M)': r'$T$',
        'DRAFT(M)': r'$T$',
        'TEU': 'TEU',
        'Vs': r'$V_s$',
        'MCR (KW)': r'$P_{MCR}$'
    }

    return variable_map

def get_variables(shiptype:ShipType):

    print(f'USING DATA FOR {shiptype} SHIP')

    constant_vars = ['BEAM (M)','DEPTH (M)','DRAFT (M)','LBP (M)','LOA (M)','DWT']

    default_pred_constraints = {'BEAM (M)':['DWT'],
                                    'DEPTH (M)':['DWT'],
                                    'DRAFT (M)':['DWT'],
                                    'LBP (M)':['DWT'],
                                    'LOA (M)':['DWT'],
                                    'DWT':['BEAM (M)','DEPTH (M)','DRAFT (M)','LBP (M)','LOA (M)']}


    if shiptype == ShipType.TANKER:

        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Tanker')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
        variables = constant_vars + ['MCR (KW)']
        pred_constraints = default_pred_constraints


    elif shiptype == ShipType.BULK_CARRIER:
        print('bulk carrier filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Bulk_carrier')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
        variables = constant_vars+['MCR (KW)']
        pred_constraints = default_pred_constraints

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('container ship filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Containership')
        bins = None
        variables = constant_vars + ['TEU','Vs','MCR (KW)']

        pred_constraints = {'BEAM (M)': ['DWT','TEU','Vs'],
                                    'DEPTH (M)': ['DWT','TEU','Vs'],
                                    'DRAFT (M)': ['DWT','TEU','Vs'],
                                    'LBP (M)': ['DWT','TEU','Vs'],
                                    'LOA (M)': ['DWT','TEU','Vs'],
                                    'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)','TEU'],
                                    'TEU':['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)','DWT']}

    else:
        print('ropax filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\RO_Pax')
        bins = None
        variables = constant_vars + [ 'Np', 'Nv','Vs']

        pred_constraints = {'BEAM (M)': ['DWT','Np', 'Nv','Vs'],
                            'DEPTH (M)': ['DWT','Np', 'Nv','Vs'],
                            'DRAFT (M)': ['DWT','Np', 'Nv','Vs'],
                            'LBP (M)': ['DWT','Np', 'Nv','Vs'],
                            'LOA (M)': ['DWT','Np', 'Nv','Vs'],
                            'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'Np', 'Nv','Vs'],
                            'Np': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT', 'Nv','Vs'],
                            'Nv': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT', 'Np','Vs']}

    return bins,variables,pred_constraints, filepath, save_dir

def rf_predictors():

    return {
            'n_estimators': 200,
            'max_depth': None,
            'max_features':"sqrt",
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }


def formulae_assessment(conventional_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        target_dimensions:List[str]=None,
                        predictor_dimensions:List[str]=None,
                        shiptype:ShipType = None,
                        residuals: bool = False,
                        plots:bool=False,
                        save_dir:Path|str|None = None) -> pd.DataFrame:

    formula_excel = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature_formulae.xlsx'

    literature_direct = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature formulae'

    print("\n" + "=" * 80)
    print("PHASE 2: Literature Formula Analysis")
    print("=" * 80)

    # Load and inspect formulae
    # formulae = LiteratureFormulaLibrary.from_file(file_path=formula_excel,
    #                                               reference='Papanikolaou',
    #                                               shiptype=shiptype)

    formulae = LiteratureFormulaLibrary.from_directory(
        directory_path=literature_direct,
        shiptype=shiptype,
        reference="Literature 2024"
    )

    print(f"\nLoaded {len(formulae)} formulae:")
    for f in formulae:
        print(f"  {f.name}: {f.predictors} → {f.target} using {f.get_formula_text()}")

    # Evaluate (use matched set for fair comparison with ML)
    lit_evaluator = LiteratureFormulaEvaluator(
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        target_dimensions=target_dimensions,
        predictor_dimensions=predictor_dimensions,
        formulae_file=formula_excel,
        reference='Papanikolaou',
        shiptype=shiptype
    )

    lit_evaluator.evaluate_all_formulae(metrics='r2')

    lit_evaluator.evaluate_all_formulae(metrics='rrmse')


    lit_evaluator.interpret_results()

    if plots:

        lit_evaluator.plot_comparison()

        lit_evaluator.plot_comparison_by_reference()

        lit_evaluator.plot_comparison_scatter()

        lit_evaluator.plot_formula_fit()

    if residuals:
        lit_evaluator.plot_residuals()

def selectors(shiptype:ShipType,
                            conventional_df:pd.DataFrame,
                        fixed_dimensions:List[str]=['DWT'],
                        cv_method:str|None = None,
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,
                        save_dir:Path|str|None = None) -> pd.DataFrame:

    rf_param = rf_predictors()

    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    print(ship_type)

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    print(('=' * 20) + f'STEP 2: Parametric regressions' + ('=' * 20))
    print(f'=' * 60)

    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variables,
                                            target_dimensions=variables,
                                            verbose=False,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='bic',
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization,
                                            fit_mfp=True)

    poly_results = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method,display_results=False)


    print(f"\n\nSTAGE:  random forest regression selection")
    print("=" * 100)

    rf_selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variables,
        target_dimensions=variables,
        fixed_dimensions=['DWT'],
        rf_params=rf_param,
        selection_metric='rrmse',
        verbose=False
    )

    # Analyze all combinations
    rf_results = rf_selector.analyze_all_combinations(
        max_predictors=2,
        cv_method='stratified',
        n_folds=10,
        random_state=42,
        use_feature_engineering=False,
        group_by_type=True
    )

    # Print summary
    rf_selector.print_summary()

    if save_dir is not None:
        poly_selector.save_results(save_dir/f'{shiptype.name}_parametric_selector_v5.pkl')
        rf_selector.save_results(save_dir/f'{shiptype.name}_rf_selector_v5.pkl')


    return poly_results, poly_selector,rf_results, rf_selector


def analysers(poly_results,
              rf_results,
              variables,
              conventional_df,
              alternative_df,
              display:bool=False):
    """
    Create domain analyzers with automatic variable filtering.

    Ensures only variables present in BOTH datasets are used for analysis.
    """

    rf_param= rf_predictors()

    print("\n\n" + "=" * 70)
    print("Step 3: ANALYZER INITIALIZATION (Domain Shift)")
    print("=" * 70)

    rf = rf_results()

    # ✅ Filter variables to those available in alternative dataset
    available_in_conv = set(conventional_df.columns)
    available_in_alt = set(alternative_df.columns)

    common_variables = [v for v in variables
                        if v in available_in_conv and v in available_in_alt]

    # Report filtering
    if len(common_variables) < len(variables):
        missing = set(variables) - set(common_variables)
        print(f"\n⚠️  Variable Filtering:")
        print(f"   Requested: {len(variables)} variables")
        print(f"   Available in both datasets: {len(common_variables)} variables")
        print(f"   Missing from alternative dataset: {sorted(missing)}")
        print(f"   Using: {common_variables}")

    # ✅ CRITICAL: Filter model_recommendations to only include common variables
    filtered_poly_results = {}
    for (predictors, target), model_type in poly_results.items():
        # Check if ALL predictors and target are in common_variables
        if all(pred in common_variables for pred in predictors) and target in common_variables:
            filtered_poly_results[(predictors, target)] = model_type

    if len(filtered_poly_results) < len(poly_results):
        skipped = len(poly_results) - len(filtered_poly_results)
        print(f"   Filtered {skipped} combinations containing unavailable variables")

    # ✅ Filter RF results similarly
    filtered_rf_results = {}
    for target, combos in rf_results.items():
        if target not in common_variables:
            continue

        filtered_rf_results[target] = {}
        for combo_name, data in combos.items():
            predictors = data['predictors']
            if all(pred in common_variables for pred in predictors):
                filtered_rf_results[target][combo_name] = data

    print(f"   Filtered poly recommendations: {len(filtered_poly_results)} combinations")
    print(f"   Filtered RF results: {sum(len(v) for v in filtered_rf_results.values())} combinations")

    # ✅ Create analyzers with FILTERED results
    print("\nCreating Parametric Domain Analyzer...")
    poly_analyzer = ParametricDomainAnalyzer(
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        independent_variables=common_variables,
        dependent_variables=common_variables,
        model_recommendations=filtered_poly_results,
        test_size=0.2,
        random_state=42,
        verbose=display  # ← Turn off verbose for cleaner output
    )

    print("\nCreating Random Forest Domain Analyzer...")
    rf_analyzer = RandomForestDomainAnalyzer(
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        variable_dimensions=common_variables,
        target_dimensions=common_variables,
        selector_results=filtered_rf_results,
        rf_params=rf_param,
        test_size=0.2,
        random_state=42,
        verbose=display  # ← Turn off verbose
    )

    # ✅ RUN THE ANALYSIS!
    print("\nRunning Parametric Analysis...")
    poly_results_df = poly_analyzer.analyze_all_combinations(
        max_predictors=2,
        display_results=display
    )

    print("\nRunning Random Forest Analysis...")
    rf_results_df = rf_analyzer.analyze_all_combinations(
        max_predictors=2,
        display_results=display
    )

    print(f"\n✅ Analysis Complete:")
    print(f"   Poly: {len(poly_analyzer.results)} targets analyzed")
    print(f"   RF: {len(rf_analyzer.results)} targets analyzed")

    print("=" * 70)
    return poly_analyzer, rf_analyzer


def demonstrate_analysis_plots(analyzer: ParametricDomainAnalyzer|RandomForestDomainAnalyzer,):

    print("\n\n" + "=" * 70)
    print("Step 1: ANALYZER RESULTS (Domain Shift)")
    print("=" * 70)

    analyzer.plot_domain_shift_analysis()

    print("\n\n" + "=" * 70)
    print("Step 2: plotting prediction scatter")
    print("=" * 70)

    analyzer.plot_prediction_scatter()

    print("\n\n" + "=" * 70)
    print("Step 2: plotting residuals")
    print("=" * 70)

    analyzer.plot_residuals()




def pickle_importer(shiptype:ShipType,):

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']


    poly_selector_pklpath = save_dir/f'{shiptype}_parametric_selector.pkl'
    rf_selector_pklpath = save_dir/f'{shiptype}_rf_selector.pkl'

    poly_selector = ParametricModelSelector.from_saved_results(
        results_filepath=poly_selector_pklpath,
        training_data=conventional_df,
        variable_dimensions=variables
    )

    poly_result = poly_selector.model_recommendations

    poly_selector.show_selection_results()

    rf_selector = RandomForestSelector.from_saved_results(results_filepath=rf_selector_pklpath,
                                                          training_data=conventional_df,
                                                          variable_dimensions=variables)

    rf_result = rf_selector.results
    rf_selector.show_selection_results()

    poly_analyzer, rf_analyzer = analysers(poly_results=poly_result,
                                           rf_results=rf_result,
                                           conventional_df=conventional_df,
                                           alternative_df=alternative_df,
                                           variables=variables, )

    return poly_analyzer, rf_analyzer



def experiment2_LaTeX(poly_selector,
                      poly_analyzer,
                      rf_selector,
                      rf_analyzer,
                      shiptype,
                      save_dir,
                      alt_type):

    # bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)

    #Poly tables

    generator_analyzer = ParametricDomainAnalysisLatexTableGenerator(poly_analyzer)

    print("\nGenerating analyzer R² table...")
    r2_table_analyzer = generator_analyzer.generate_r2_table(max_predictors=2, ship_type=shiptype)
    print("\nPreview (first 15 lines):")
    for line in r2_table_analyzer.split('\n')[:15]:
        print(line)

    rrmse_table_analyzer = generator_analyzer.generate_rrmse_table(max_predictors=2, ship_type=shiptype)

    print("\nPreview (first 15 lines):")
    for line in rrmse_table_analyzer.split('\n')[:15]:
        print(line)

    generator_analyzer.save_tables(ship_type=shiptype,
                                   output_dir=save_dir,
                                   alt_type=alt_type)

    #
    #RF LaTeX tables
    #
    rf_generator = RandomForestLatexTableGenerator(rf_analyzer)

    rf_generator.save_tables(ship_type=shiptype,
                             best_only=True,
                             output_dir=save_dir,
                             alt_type=alt_type)


def experiment2_plots(poly_selector:ParametricModelSelector,
                      poly_analyzer:ParametricDomainAnalyzer,
                      rf_selector:RandomForestSelector,
                      rf_analyzer:RandomForestDomainAnalyzer,
                      shiptype:ShipType,
                      save_dir: str | Path,
                      cv_method = None,):


    poly_rrmse_comparison = save_dir/'rrmse_comparison.png'
    poly_selector.plot_metrics_comparison(
        cv_method=cv_method,
        metric='rrmse',
        title=f'RRMSE (%) comparison of parametric models for {shiptype}',
        group_by_type=False,
        save_path=poly_rrmse_comparison
    )


    poly_r2_comparison = save_dir/'poly_r2_comparison.png'
    poly_selector.plot_metrics_comparison(
        cv_method=cv_method,
        metric='adj_r2',
        title=f'adj r^2 comparison of parametric models for {shiptype}',
        group_by_type=False,
        save_path=poly_r2_comparison
    )

    poly_analyzer.interpret_results(significance_threshold_r2=0.05)

    poly_domain_shift = save_dir/'poly_domain_shift.png'

    poly_analyzer.plot_domain_shift_analysis(save_path=poly_domain_shift)



    rf_rrmse_comparison = save_dir/'rf_rrmse_comparison.png'
    rf_selector.plot_metrics_comparison(
        metric='rrmse',
        save_path=rf_rrmse_comparison,

        group_by_type=False,
    )

    # Plot r2 comparison
    rf_r2_comparison = save_dir/'rf_r2_comparison.png'
    rf_selector.plot_metrics_comparison(
        metric='adj_r2',
        save_path=rf_r2_comparison,
        group_by_type=False,
    )

    rf_analyzer.interpret_results(
        significance_threshold_r2=0.05,
        significance_threshold_rrmse=10.0
    )

    # ============================================================================
    # 4.3. VISUALIZE
    # ============================================================================

    # Domain shift plots
    rf_domain_shift = save_dir/'rf_domain_shift.png'
    rf_analyzer.plot_domain_shift_analysis(save_path=rf_domain_shift)


    # Feature importance
    # feature_importance = save_dir/'feature_importance.png'
    # rf_analyzer.plot_feature_importance_analysis(save_path=feature_importance)
    #
    # print(f"\n\nSTAGE: Comparing random forest and polynomial regression")
    # print("=" * 100)

    comparer = ModelComparisonAnalyzer(
        polynomial_selector=poly_selector,
        rf_selector=rf_selector,
        polynomial_domain_analyzer=poly_analyzer,
        rf_domain_analyzer=rf_analyzer,
        verbose=True
    )

    # ============================================================================
    # COMPREHENSIVE ANALYSIS
    # ============================================================================

    # Print summary
    comparer.print_summary()

    # Create all plots
    comparer.plot_selector_comparison(
        metric='rrmse',
    )
    comparer.plot_selector_comparison(
        metric='r2',
    )
    comparison_plot = save_dir/'parametric_vs_RF_comparison.png'
    comparer.plot_domain_shift_comparison(save_path=comparison_plot)


def experiment3_analysis(
                         shiptype: ShipType,
                        alt_dataset:str=None,
                         coefficients:bool=True,
                         parametric_cross_dataset:bool=False,
                         rf_cross_dataset:bool=False,
                         rf_feature_importance:bool=True,
ignore_variables:list|None=['LOA (M)'],):

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)


    rf_param=rf_predictors()


    fleet = getting_alt_fuel_ships(filepath=filepath,ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']

    poly_selector_pklpath = save_dir/f'{shiptype.name}_parametric_selector_v5.pkl'
    rf_selector_pklpath = save_dir/f'{shiptype.name}_rf_selector_v5.pkl'


    poly_selector = ParametricModelSelector.from_saved_results(
        results_filepath=poly_selector_pklpath,
        training_data=conventional_df,
        variable_dimensions=variables
    )

    poly_result = poly_selector.model_recommendations



    rf_selector = RandomForestSelector.from_saved_results(results_filepath=rf_selector_pklpath,
                                                          training_data=conventional_df,
                                                          variable_dimensions=variables)

    rf_result = rf_selector.results


    alternative_sets = {'Dual fuel':dual_fuel_df,
                        'Pure alternative':alternative_df}

    for fuel,data in alternative_sets.items():


        if coefficients:

            print(f"\n\nSTAGE : POLYNOMIAL COEFFICIENT COMPARISON ANALYSIS")
            print("=" * 100)

            name_map = get_variable_name_maps()
            coeff_analyzer = CoefficientComparisonAnalyzer(
                parametric_selector=poly_selector,
                conventional_data=conventional_df,
                alternative_data=data,
                verbose=True
            )

            # Analyze all targets
            results = coeff_analyzer.analyze_all_targets(
                min_alternative_samples=10,
                display_results=False,
                max_predictors=1
            )

            effect_sizes = coeff_analyzer.decompose_coefficient_changes(
                effect_size_threshold=0.2  # Small effect threshold
            )

            # ============================================================================
            # STEP 3: Visualizations
            # ============================================================================

            # ✅ FIX: Plot coefficient decomposition for each target individually
            print("\nGenerating coefficient decomposition plots...")

            save_path = save_dir/'experiment_3'/'parametric'/f'{shiptype.name}_{fuel}_coeff_decomposition_v6.png'

            formula_mappings = name_map

            variable_name_map = {
        'DWT': 'DWT',
        'GT': 'GT',
        'LOA (M)': 'LOA (m)',
        'LBP (M)': 'LBP (m)',
        'BEAM (M)': 'Beam (m)',
        'DEPTH (M)': r'Depth (m)',
        'DRAFT (M)': r'Draught (m)',
        'TEU': 'TEU',
        'Vs': r'Speed (knots)',
        'MCR (KW)': r'MCR (kW)'
    }


            coeff_analyzer.plot_coefficient_changes(show_distribution=False,
                                                    show_fuel=True,
                                                    save_path=save_path,
                                                    title_prefix=shiptype.string,
                                                    title_suffix=fuel,
                                                    min_r2=0.6,
                                                    variable_name_map = variable_name_map,
                                                    formula_name_map =formula_mappings,
                                                    figsize_per_plot =(5.5,4))


        # ============================================================================
        # Cross-Dataset Analysis
        # ============================================================================

        if parametric_cross_dataset:

            print("\n\nSTAGE: CROSS-DATASET PARAMETRIC PERFORMANCE ANALYSIS")
            print("=" * 100)



            param_cross_analysis = CrossDatasetPerformanceAnalyzer(conventional_data=conventional_df,
                                                                   alternative_data=alternative_df,
                                                                   independent_variables=variables,
                                                                   dependent_variables=variables,
                                                                   parametric_selector=poly_selector,
                                                                   model_recommendations=poly_selector.model_recommendations,
                                                                   verbose=True)




            print("\n\nSTAGE: CROSS-DATASET PARAMETRIC PERFORMANCE ANALYSIS - (1 PREDICTOR)")
            print("=" * 100)

            results_bootstrap = param_cross_analysis.analyze_best_combinations(
                evaluation_method='bootstrap',
                n_bootstraps=100,
                max_predictors=1,
                require_alternative_data=True,

            )

            print("\n\nSTAGE: CROSS-DATASET PARAMETRIC PERFORMANCE ANALYSIS (2 PREDICTORS)")
            print("=" * 100)

            results_bootstrap = param_cross_analysis.analyze_best_combinations(
                evaluation_method='bootstrap',
                n_bootstraps=100,
                max_predictors=2,
                require_alternative_data=True,

            )

        if rf_cross_dataset:
            print("\n\nSTAGE: CROSS-DATASET RANDOM FOREST PERFORMANCE ANALYSIS")
            print("=" * 100)

            rf_cross_dataset = RandomForestCrossDatasetAnalyzer(conventional_data=conventional_df,
                                                                alternative_data=alternative_df,
                                                                variable_dimensions=variables,
                                                                target_dimensions=variables,
                                                                selector_results=rf_selector.results,
                                                                rf_params=rf_param)

            results = rf_cross_dataset.analyze_best_combinations(evaluation_method='bootstrap',)


        # ============================================================================
        # Random Forest Importance Analysis
        # ============================================================================

        if rf_feature_importance:

            print("\n\nSTAGE: RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
            print("=" * 100)

            if ignore_variables is not None:
                new_variables = []

                for variable in variables:
                    if variable not in ignore_variables:
                        new_variables.append(variable)
                variables = new_variables



            analyzer = RFFullFeatureImportanceAnalyzer(
                conventional_data=conventional_df,  # All conventional ships
                alternative_data=data,  # All alternative fuel ships
                variable_dimensions=variables,
                verbose=True,
                rf_params=rf_param
            )

            results = analyzer.analyze_all_targets(display_results=True)

            # Get significant changes
            changes = analyzer.get_importance_changes(
                threshold=0.05,
                effect='fuel',
                top_n=10  # Top 10 per target
            )

            analyzer.get_importance_changes()

            # Decompose effects
            decomp = analyzer.decompose_importance_changes(top_n_per_target=15)

            save_path = save_dir / 'experiment_3' / 'random_forest' / f'{shiptype.name}_{fuel}_HM_feature_importance_v5.png'

            name_map = {
                'DWT': 'DWT (t)',
                'GT': 'GT (m³)',
                'LOA (M)': r'LOA (m)',
                'LBP (M)': r'LBP (m)',
                'BEAM (M)': 'Beam (m)',
                'DRAFT (M)': 'Draught (m)',
                'DEPTH (M)': 'Depth (m)',
                'Vs': r'$V_s$ (kn)',
                'MCR (KW)': r'MCR (kW)'
            }

            variable_map = {
                'DWT': 'DWT',
                'GT': 'GT',
                'LOA (M)': r'$L_{OA}$',
                'LBP (M)': r'$L_{BP}$',
                'BEAM (M)': r'$B$',
                'DEPTH (M)': r'$D$',
                'DRAFT (M)': r'$T$',
                'TEU': 'TEU',
                'Vs': r'$V_s$',
                'MCR (KW)': r'$P_{MCR}$'
            }

            analyzer.plot_importance_heatmap(separate_figures=False,
                                             top_n=None,
                                             min_r2=0.6,
                                             save_path=save_path,
                                             figsize=(7,7),
                                             base_font_size=10,
                                             tick_font_size=12,
                                             annotation_font_size=12,
                                             title_prefix=shiptype.string,
                                             title_suffix=fuel,
                                             alternative_dataset_name=fuel,
                                             annotate=True,
                                             variable_name_map=name_map,
                                             annotation_format='.3f',
                                             n_cols=4,
                                             gridspec_top=0.9,
                                             debug=True)

            # analyzer.plot_importance_change_heatmap(separate_figures=False,
            #                                         title_prefix=shiptype.string,
            #                                         title_suffix = fuel
            #                                         )
            #
            # save_path = save_dir / 'experiment_3' / 'random_forest' / f'{shiptype.name}_{fuel}_HM_importance_across_v5.png'
            #
            #
            # analyzer.plot_importance_across_targets(save_path=save_path,
            #                                  title_prefix=shiptype.string,
            #                                  title_suffix=fuel,
            #                                         separate_figures=False,
            #                                         figsize=(14,5))




def LaTeX_Table_results(shiptype:ShipType,
                        save_dir:Path = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results'),
                        cv_method:str|None=None,
                        ranking_att ='rrmse',
                        fixed_dimensions:List[str]=['DWT'],
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,
                        parametric:bool=True,
                        rf:bool=True):

    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type,
                                   outlier_method='percentile')


def demonstrate_multi_analyser(
        conventional_df: pd.DataFrame,
        alternative_df: dict[str, pd.DataFrame],
        variables: list[str],
        shiptype: 'ShipType',
        target_predictor_constraints=None,
        poly_selector: ParametricModelSelector = None,
        rf_selector: RandomForestSelector = None,
        analysis: bool = True,
        save_dir: str = None,
        prefix: str = '',
        selection_metric: str = 'rrmse',
        debug: bool = False,
        save_amend: str = None,
        save_analysis:bool=True,
        significance_test:bool=False,
        save_significance:bool=False,
) -> MultiDatasetDomainAnalyzer:
    """
    Demonstrate multi-dataset domain analysis.

    Returns
    -------
    MultiDatasetDomainAnalyzer
        The analyzer with all fuel type results
    """

    # ========================================================================
    # SETUP: Determine analyzer type and selector
    # ========================================================================

    if poly_selector is not None:
        analyzer_str = 'parametric'
        selector = poly_selector
        results = selector.model_recommendations
    elif rf_selector is not None:
        analyzer_str = 'random_forest'
        selector = rf_selector
        results = selector.results
    else:
        raise ValueError("Must provide either poly_selector or rf_selector")

    print("\n\n" + "=" * 70)
    print(f"Step 1: Multi-Dataset Analysis for {analyzer_str.upper()}")
    print("=" * 70)

    rf_param=rf_predictors()

    # ========================================================================
    # CREATE MULTI-DATASET ANALYZER
    # ========================================================================

    multi_dataset_analyzer = MultiDatasetDomainAnalyzer(
        conventional_data=conventional_df,
        alternative_datasets=alternative_df,
        independent_variables=variables,
        model_recommendations=results,
        selector=selector,
        analyzer_type=analyzer_str,
        selection_metric=selection_metric,
        verbose=False,
        rf_params=rf_param,
        target_predictor_constraints=target_predictor_constraints
    )

    # ========================================================================
    # RUN ANALYSIS
    # ========================================================================

    all_results = multi_dataset_analyzer.analyze_all_datasets(
        max_predictors=2,
        display_results=True,
        debug_selection=debug
    )

    # ========================================================================
    # DISPLAY AND PLOT RESULTS
    # ========================================================================

    if analysis:
        print("\n" + "=" * 70)
        print("Step 2: Display Summary Statistics")
        print("=" * 70)

        multi_dataset_analyzer.print_summary_statistics()

        print("\n" + "=" * 70)
        print("Step 3: Fuel Type Comparison Table")
        print("=" * 70)

        multi_dataset_analyzer.print_fuel_type_comparison()

        # Get available targets from first analyzer
        available_targets = list(
            list(multi_dataset_analyzer.analyzers.values())[0].results.keys()
        )

        if available_targets:
            first_target = list(available_targets)[0]

            print("\n" + "=" * 70)
            print(f"Step 4: Multi-Dataset Comparison for {first_target}")
            print("=" * 70)

            # Plot multi-dataset comparison for specific target
            save_path = None
            if save_dir is not None:
                save_path = save_dir / f'{prefix}_multi_dataset_{first_target.replace(" ", "_")}_{save_amend}.png'

            multi_dataset_analyzer.plot_multi_dataset_comparison(
                target=first_target,
                save_path=save_path
            )
        else:
            print("⚠️  No targets with results found!")

        print("\n" + "=" * 70)
        print("Step 5: Fuel Type Summary Heatmap")
        print("=" * 70)

        # Plot summary heatmap
        save_path = None
        if save_dir is not None:
            save_path = save_dir / f'{prefix}_fuel_type_summary_{save_amend}.png'

        multi_dataset_analyzer.plot_fuel_type_summary(
            metric='gap_fuel_type_rrmse',
            save_path=save_path
        )

        print("\n" + "=" * 70)
        print("Step 6: Individual Fuel Type Domain Shift Plots")
        print("=" * 70)

        # Plot domain shift for each fuel type
        for fuel_type, analyzer in multi_dataset_analyzer.analyzers.items():
            print(f"\n📈 Plotting domain shift for {fuel_type}")

            save_path = None
            if save_dir is not None:
                save_path = save_dir / f'{prefix}_{fuel_type}_domain_shift_{save_amend}.png'

            # analyzer.plot_domain_shift_analysis(save_path=save_path)
    if significance_test:
        multi_dataset_analyzer.run_significance_tests(ci_method='bca',
                                                      n_bootstrap=50000,
                                                      debug=True,
                                                      random_seed=42)

    # ========================================================================
    # GENERATE LATEX TABLES
    # ========================================================================

    if save_dir is not None:
        print("\n" + "=" * 70)
        print("Step 7: Generate LaTeX Tables")
        print("=" * 70)

        if save_analysis:

            multi_dataset_analyzer.generate_latex_tables(
                ship_type=shiptype,
                output_dir=save_dir,
                targets=variables,
                prefix=prefix,
                suffix=save_amend
            )

        if save_significance:

            multi_dataset_analyzer.export_significance_results(output_path=save_dir/'experiment2'/f'{prefix}_significance_test_{shiptype.string}_{save_amend}.xlsx')


    return multi_dataset_analyzer


def demonstrate_analyser_comparison(
        shiptype: 'ShipType',
        poly_multi_analyzer: MultiDatasetDomainAnalyzer,
        rf_multi_analyzer: MultiDatasetDomainAnalyzer,
        poly_selector: ParametricModelSelector,
        rf_selector: RandomForestSelector,
        save_dir: str = None,
        save_amend: str = None,
        selection_metric: str = 'test_rrmse'
):
    """
    Compare parametric vs RF analyzers using the SAME analyzers created by MultiDatasetDomainAnalyzer.

    Parameters
    ----------
    shiptype : ShipType
        Ship type being analyzed
    poly_multi_analyzer : MultiDatasetDomainAnalyzer
        Multi-dataset analyzer that used parametric regression
    rf_multi_analyzer : MultiDatasetDomainAnalyzer
        Multi-dataset analyzer that used Random Forest
    poly_selector : ParametricModelSelector
        Parametric model selector
    rf_selector : RandomForestSelector
        Random Forest selector
    save_dir : str, optional
        Directory to save plots
    save_amend : str, optional
        Suffix for saved files
    selection_metric : str, default='test_rrmse'
        Metric used for comparison
    """

    print("\n\n" + "=" * 100)
    print("PARAMETRIC vs RANDOM FOREST COMPARISON")
    print("=" * 100)
    print(f"  Using analyzers created by MultiDatasetDomainAnalyzer")
    print(f"  Selection metric: {selection_metric}")
    print(f"  Fuel types: {list(poly_multi_analyzer.analyzers.keys())}")

    # Get fuel type names
    fuel_types = list(poly_multi_analyzer.analyzers.keys())

    # Verify both analyzers have the same fuel types
    if set(fuel_types) != set(rf_multi_analyzer.analyzers.keys()):
        raise ValueError(
            f"Fuel type mismatch!\n"
            f"  Parametric: {fuel_types}\n"
            f"  RF: {list(rf_multi_analyzer.analyzers.keys())}"
        )

    # ========================================================================
    # COMPARE FOR EACH FUEL TYPE
    # ========================================================================

    for fuel_type in fuel_types:
        print("\n" + "=" * 100)
        print(f"Comparing: {fuel_type}")
        print("=" * 100)

        # Extract the SAME analyzers that were created by MultiDatasetDomainAnalyzer
        poly_analyzer = poly_multi_analyzer.analyzers[fuel_type]
        rf_analyzer = rf_multi_analyzer.analyzers[fuel_type]

        # Verify they analyzed the same alternative dataset
        print(f"\n  Parametric analyzer:")
        print(f"    Alternative ships: {len(poly_analyzer.alternative)}")
        print(f"    Targets analyzed: {list(poly_analyzer.results.keys())}")

        print(f"\n  RF analyzer:")
        print(f"    Alternative ships: {len(rf_analyzer.alternative)}")
        print(f"    Targets analyzed: {list(rf_analyzer.results.keys())}")

        # ========================================================================
        # CREATE COMPARISON ANALYZER
        # ========================================================================

        comparer = ModelComparisonAnalyzer(
            polynomial_selector=poly_selector,
            rf_selector=rf_selector,
            polynomial_domain_analyzer=poly_analyzer,
            rf_domain_analyzer=rf_analyzer,
            verbose=True
        )

        # ========================================================================
        # GENERATE COMPARISON DATAFRAME
        # ========================================================================

        print(f"\n  Generating comparison dataframe...")
        comparison_df = comparer.compare_domain_shift(
            selection_metric=selection_metric,
            verbose_selection=True  # Show detailed selection process
        )

        print(f"\n  ✅ Comparison complete: {len(comparison_df)} targets compared")

        # ========================================================================
        # PLOT COMPARISON
        # ========================================================================

        if save_dir is not None:
            comparison_dir = save_dir / 'experiment2'
            comparison_dir.mkdir(parents=True, exist_ok=True)

            comparison_plot = comparison_dir / f'{fuel_type}_parametric_vs_RF_comparison_{save_amend}.png'

            print(f"\n  📊 Generating comparison plot...")
            comparer.plot_domain_shift_comparison(figsize=(16,8),
                save_path=comparison_plot,
                shiptype=shiptype,
                dataset=fuel_type,
                rrmse_threshold=10,
                threshold_metric=None,
              gridspec_top=0.93,  # 0.88–0.95 typical; lower = more suptitle room
              base_font_size=9,
              tick_font_size=10,  # x-axis tick labels, set independently
              bar_label_font_size=9,
            variable_name_map=get_variable_name_maps()
              # numeric annotations on bars, set independently
              )


            print(f"  ✅ Plot saved: {comparison_plot.name}")

            # Also save comparison data
            csv_path = comparison_dir / f'{fuel_type}_comparison_data_{save_amend}.csv'
            comparison_df.to_csv(csv_path, index=False)
            print(f"  ✅ Data saved: {csv_path.name}")
        else:
            # Just show plot without saving
            comparer.plot_domain_shift_comparison(
                shiptype=shiptype,
                dataset=fuel_type,
            )

    print("\n" + "=" * 100)
    print("✅ All comparisons complete!")
    print("=" * 100)


def demonstrating_functionality(
        shiptype: ShipType,
        selection_metric: str = 'rrmse',
        parametric: bool = True,
        rf: bool = True,
        analysis: bool = False,
        latex: bool = True,
        analyser_comparison: bool = True,
        debug: bool = False,
        save_amend: str = 'ver_5',
        target_predictor_default:bool=False
):
    """
    Demonstrate complete analysis workflow.

    Parameters
    ----------
    shiptype : ShipType
        Type of ship to analyze
    selection_metric : str, default='rrmse'
        Metric for selecting best combinations
    parametric : bool, default=True
        Run parametric analysis
    rf : bool, default=True
        Run Random Forest analysis
    analysis : bool, default=True
        Display analysis results and plots
    latex : bool, default=True
        Generate LaTeX tables
    analyser_comparison : bool, default=True
        Compare parametric vs RF
    debug : bool, default=False
        Show debug output
    save_amend : str, default='ver_2'
        Suffix for saved files
    """

    # ========================================================================
    # SETUP
    # ========================================================================

    bins, variables,target_predictor_constraints, filepath, save_dir = get_variables(shiptype=shiptype)

    if target_predictor_default:
        target_predictor_constraints = None

    rf_params=rf_predictors()

    if not latex:
        save_dir = None

    print("\n" + "=" * 100)
    print(f"DEMONSTRATING FUNCTIONALITY: {shiptype.name}")
    print("=" * 100)
    print(f"  Variables: {variables}")
    print(f"  Selection metric: {selection_metric}")
    print(f"  Save directory: {save_dir}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    fleet = getting_alt_fuel_ships(filepath=filepath, ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']

    print(f"\n  Conventional ships: {len(conventional_df)}")
    print(f"  Pure alternative ships: {len(alternative_df)}")
    print(f"  Dual fuel ships: {len(dual_fuel_df)}")

    alternative_sets = {
        'dual_fuel': dual_fuel_df,
        'Pure_alt': alternative_df
    }

    # ========================================================================
    # LOAD SELECTORS
    # ========================================================================

    poly_selector_pklpath = save_dir / f'{shiptype.name}_parametric_selector_v5.pkl'
    rf_selector_pklpath = save_dir / f'{shiptype.name}_rf_selector_v5.pkl'

    poly_selector = None
    rf_selector = None

    if parametric:
        print("\n" + "=" * 100)
        print("LOADING PARAMETRIC SELECTOR")
        print("=" * 100)

        poly_selector = ParametricModelSelector.from_saved_results(
            results_filepath=poly_selector_pklpath,
            training_data=conventional_df,
            variable_dimensions=variables
        )

        poly_result = poly_selector.model_recommendations
        poly_selector.show_selection_results()

    if rf:
        print("\n" + "=" * 100)
        print("LOADING RANDOM FOREST SELECTOR")
        print("=" * 100)

        rf_selector = RandomForestSelector.from_saved_results(
            results_filepath=rf_selector_pklpath,
            training_data=conventional_df,
            variable_dimensions=variables,
            rf_params=rf_params
        )

        rf_result = rf_selector.results
        rf_selector.show_selection_results()

    # ========================================================================
    # RUN MULTI-DATASET ANALYSES
    # ========================================================================

    poly_multi_analyzer = None
    rf_multi_analyzer = None

    if parametric and analysis:
        poly_multi_analyzer = demonstrate_multi_analyser(
            conventional_df=conventional_df,
            alternative_df=alternative_sets,
            variables=variables,
            poly_selector=poly_selector,
            save_dir=save_dir,
            analysis=analysis,
            shiptype=shiptype,
            prefix='parametric',
            selection_metric=selection_metric,
            debug=debug,
            save_amend=save_amend,
            target_predictor_constraints=target_predictor_constraints
        )

    if rf and analysis:
        rf_multi_analyzer = demonstrate_multi_analyser(
            conventional_df=conventional_df,
            alternative_df=alternative_sets,
            variables=variables,
            rf_selector=rf_selector,
            analysis=analysis,
            save_dir=save_dir,
            shiptype=shiptype,
            prefix='RF',
            selection_metric=selection_metric,
            debug=debug,
            save_amend=save_amend,
            target_predictor_constraints=target_predictor_constraints
        )

    # ========================================================================
    # COMPARE PARAMETRIC VS RF
    # ========================================================================

    if analyser_comparison and poly_multi_analyzer is not None and rf_multi_analyzer is not None:
        demonstrate_analyser_comparison(
            shiptype=shiptype,
            poly_multi_analyzer=poly_multi_analyzer,
            rf_multi_analyzer=rf_multi_analyzer,
            poly_selector=poly_selector,
            rf_selector=rf_selector,
            save_dir=save_dir,
            save_amend=save_amend,
            selection_metric='test_rrmse'  # Use test performance for comparison
        )

    elif analyser_comparison:
        print("\n⚠️  Cannot run analyser_comparison: need both parametric and rf to be True")

    print("\n" + "=" * 100)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 100)


def multi_dataset_empirical_tester(shiptype:ShipType,
                                   latex:bool=False,
                                   confidence_interval:bool=False):
    formula_excel = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature_formulae.xlsx'

    literature_direct = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature formulae'

    print("\n" + "=" * 80)
    print("PHASE 2: Literature Formula Analysis")
    print("=" * 80)

    # Load and inspect formulae
    # formulae = LiteratureFormulaLibrary.from_file(file_path=formula_excel,
    #                                               reference='Papanikolaou',
    #                                               shiptype=shiptype)


    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,outlier_method='percentile')

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']

    alternative_sets = {'Dual fuel':dual_fuel_df,
                        'Pure alternative':alternative_df}


    formulae = LiteratureFormulaLibrary.from_directory(
        directory_path=literature_direct,
        shiptype=shiptype,
        reference="Literature 2024"
    )

    variable_map = get_variable_name_maps()



    # Create evaluator
    multi_evaluator = MultiDatasetLiteratureEvaluator(
        conventional_data=conventional_df,
        alternative_datasets=alternative_sets,
        target_dimensions=variables,
        formulae = formulae,
        shiptype=shiptype
    )

    # Run evaluation
    results = multi_evaluator.evaluate_all_datasets(display_results=True)

    # Analysis
    multi_evaluator.print_fuel_type_comparison()

    save_path = save_dir/'experiment1'/f'{shiptype.string}_multi_empirical_v4.png'

    multi_evaluator.plot_cascade_comparison(variable_name_map=variable_map,
                                            rrmse_threshold=10.00,
                                            debug=True,
                                            sort_by='target_group',
                                            predictor_order=['DWT','TEU',],
                                            target_order=['BEAM (M)','DEPTH (M)','DRAFT (M)','LBP (M)','LOA (M)','DWT','TEU','Vs','MCR (KW)'],
                                            save_path=save_path,
                                            group_by_reference=True,
                                            label_content='reference',
                                            figsize_per_plot=(16, 4),  # narrower/shorter per panel
                                            gridspec_top=0.92,  # close the suptitle gap
                                            hspace=0.40,  # tighten panels
                                            base_font_size=9,
                                            tick_font_size=9,  # x-labels slightly larger than base
                                            bar_label_font_size=8,  # bar numbers slightly smaller
                                            bar_width_scale=0.85,  # narrower clusters, more breathing room
                                            show_values=True,
                                            )


    multi_evaluator.plot_predictor_target_fits(
        predictor='DWT',
        target='LBP (M)',
        fuel_type='Pure alternative',
        show_domain_bounds=False,
        show_percentile_bounds=False,
        layout='horizontal'
    )

    if latex:
        multi_evaluator.generate_latex_tables(ship_type=shiptype,
                                              output_dir=save_dir,
                                              suffix='v2')

    if confidence_interval:


        sig_results = multi_evaluator.run_significance_tests(
        comparison='fuel_type',   # tests matched vs alternative gap
        n_bootstrap=5000,
        ci_method='bca',   # or 'bca' for more accuracy
    )
        multi_evaluator.export_significance_results(output_path=save_dir/'significance_tests.xlsx')

    multi_evaluator.generate_formula_latex_table(output_dir=save_dir/'experiment1',
                                                 prefix=f'{shiptype.string}',
                                                 show_formula_type=False)



def demo_formula_writing(shiptype:ShipType):
    poly_selector_pklpath = f'{shiptype}_poly_selector.pkl'
    rf_selector_pklpath = f'{shiptype}_rf_selector.pkl'

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    alt_fuel_sets = ['pure_alt_fuel', 'dual_fuel']

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']
    # alternative_df = fleet.processed_data['pure_alt_fuel']


    get_selectors(shiptype=shiptype,
                  conventional_df=conventional_df,
                  variables=variables,
                  save_dir=save_dir,)

def get_selectors(shiptype:ShipType,
                  save_dir:Path=None,
                  latex:bool=True,
                  parametric:bool=True,
                  rf:bool=False,
                  param_metric_comparison:bool=False,
                  target_plots:bool=False,
                  plot_targets:str|None=None,
                  plot_predictors:str|None=None,
                  ):

    if save_dir is None:
        bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath, ship_type=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']


    poly_selector_pklpath = save_dir/f'{shiptype.name}_parametric_selector_v5.pkl'
    rf_selector_pklpath = save_dir/f'{shiptype.name}_rf_selector_v5.pkl'

    variable_map = get_variable_name_maps()

    formula_map = {
        'DWT': 'DWT',
        'GT': 'GT',
        'LOA (M)': 'LOA',
        'LBP (M)': 'LBP',
        'BEAM (M)': 'B',
        'DEPTH (M)': 'D',
        'DRAFT (M)': 'T',
        'TEU': 'TEU',
        'Vs': 'Vs'
    }


    if parametric:

        selector = ParametricModelSelector.from_saved_results(
            results_filepath=poly_selector_pklpath,
            training_data=conventional_df,
            variable_dimensions=variables
        )

        poly_result = selector.model_recommendations

        selector.show_selection_results()



        # Use with ship type and formatted names
        selector.plot_model_selection(
            predictor_combo=(plot_predictors,),
            target=plot_targets,
            ship_type=shiptype.string,
            metric='aic',
            variable_name_map=variable_map,
            show_hetero=True
        )

        if param_metric_comparison:
            for variable in variables:


                vars= ['DWT','TEU','Vs','MCR (KW)']

                selector.plot_metrics_comparison(
                    max_predictors=2,
                    ship_type=shiptype.string,
                    variable_name_map=variable_map,
                    targets=[variable],
                    include_only_vars=vars,
                    group_by_type=False,
                    metric='rrmse'
                )

                selector.plot_metrics_comparison(
                    max_predictors=2,
                    ship_type=shiptype.string,
                    variable_name_map=variable_map,
                    targets=[variable],
                    include_only_vars=variables,
                    group_by_type=False,
                    metric='adj_r2'
                )

        if target_plots:

            selector.plot_predictor_functions(
                target=plot_targets,
                predictor=plot_predictors,
                show_metrics=True,
                separate_plots=True,
                show_model_type=True,
                highlight_recommended=False,
                title_prefix = shiptype.string,
                variable_name_map=variable_map,
                formula_notation_map=formula_map,
                debug=True
            )


    if rf:

        selector = RandomForestSelector.from_saved_results(results_filepath=rf_selector_pklpath,
                                                              training_data=conventional_df,
                                                              variable_dimensions=variables)

        rf_result = selector.results
        selector.show_selection_results()



        selector.plot_combination_comparison(target='LBP (M)',
                                             metric='r2_adj',
                                             ship_type=shiptype.string,
                                             variable_name_map=variable_map)





    return selector



def cross_dataset_multi(shiptype:ShipType,
                        latex:bool=False,
                        parametric:bool=False,
                        rf:bool=False):

    bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)

    if not latex:
        save_dir=None

    fleet = getting_alt_fuel_ships(filepath=filepath,ship_type=shiptype)

    rf_param = rf_predictors()

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']

    selector = get_selectors(shiptype=shiptype,
                              latex=latex,
                              conventional_df=conventional_df,
                              variables=variables,
                              parametric=parametric,
                              rf=rf,
                             save_dir=save_dir)

    alternative_sets = {'dual_fuel':dual_fuel_df,
                        'Pure_alt':alternative_df}

    if parametric:
        analyzer_type = 'parametric'
        cache = save_dir/f'cache/{shiptype.name}_cross_param.pkl'

        analyzer = MultiDatasetCrossPerformanceAnalyzer(
            conventional_data=conventional_df,
            alternative_datasets=alternative_sets,
            independent_variables=variables,
            parametric_selector=selector,
            analyzer_type=analyzer_type,
        )

        results = analyzer.analyze_all_datasets(
            evaluation_method='bootstrap',
            n_bootstraps=100,
            max_predictors=1,
            display_results=True
        )



    if rf:

        analyzer_type = 'random_forest'

        cache = save_dir/f'cache/{shiptype.name}_cross_rf.pkl'
        # Multi-dataset cross-performance
        analyzer = MultiDatasetCrossPerformanceAnalyzer(
            conventional_data=conventional_df,
            alternative_datasets=alternative_sets,
            independent_variables=variables,
            rf_selector=selector,
            analyzer_type=analyzer_type,
            rf_params=rf_param
        )

        results = analyzer.analyze_all_datasets(
            evaluation_method='bootstrap',
            n_folds=5,
            n_repeats=10,
            max_predictors=1,
            display_results=True

        )

    if latex:
        output_dir = save_dir /"experiment_3"/analyzer_type
        prefix = f"cross_dataset_{analyzer_type}_v2"

        analyzer.generate_latex_tables(ship_type=shiptype,
                                       output_dir=output_dir,
                                       targets=variables,
                                       max_predictors=1,
                                       prefix=prefix
                                       )

def check_dataset(variables: list[str],
                  output_dir:Path):

    types_list = [ShipType.BULK_CARRIER, ShipType.TANKER, ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    fleet_data = {}

    for shiptype in types_list:

        bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)


        fleet = getting_alt_fuel_ships(filepath=filepath, ship_type=shiptype.string)

        conventional_df = fleet.processed_data['pure_conventional']
        alternative_df = fleet.processed_data['pure_alt_fuel']
        dual_fuel_df = fleet.processed_data['dual_fuel']

        alternative_sets = {'Dual fuel': dual_fuel_df,
                            'Pure alternative': alternative_df}

        fleet_data[shiptype.string] = alternative_sets

    # Step 3: Create table generator
    table_generator = LatexAvailabilityTable(
        ship_type_data=fleet_data,
        variables=variables,
        caption="Table showing availability of different data points within the alternative ship datasets.",
        label="tab:data_availability_check",
        metrics=['count']
    )

    # Step 4: Preview as DataFrame (optional)
    summary_df = table_generator.get_summary_dataframe()
    print(summary_df)

    # Step 5: Generate and save LaTeX table
    latex_table = table_generator.generate_table()
    print(latex_table)

    # Save to file
    table_generator.save_table(output_dir/'data_availability_table.tex')





if __name__ == '__main__':

    shiptype = ShipType.BULK_CARRIER
    # shiptype = ShipType.TANKER
    # shiptype = ShipType.CONTAINER_SHIP
    shiptype = ShipType.RO_PAX
    #
    types_list = [ShipType.CONTAINER_SHIP,ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.RO_PAX]
    # types_list = [ShipType.TANKER,ShipType.RO_PAX]
    # # types_list = [ShipType.CONTAINER_SHIP,ShipType.BULK_CARRIER,ShipType.TANKER]
    types_list = [ShipType.RO_PAX]

    # types_list = [ShipType.TANKER,ShipType.CONTAINER_SHIP,ShipType.RO_PAX]

    variables = ['DWT','LBP (M)','LOA (M)','BEAM (M)','DRAFT (M)','DEPTH (M)','MCR (KW)','LSW','Vs','TEU','LM', 'Np', 'Nv']
    output_dir = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results'

    # check_dataset(variables=variables,
    #               output_dir=Path(output_dir))
    #
    # # multi_dataset_empirical_tester(shiptype=shiptype)
    # #
    # demo_formula_writing(shiptype=shiptype,)

    # demo_formula_writing(shiptype=shiptype,)

        # multi_dataset_empirical_tester(shiptype=shiptype)

    for shiptype in types_list:
        # experiment3_analysis(shiptype=shiptype,
        #                       rf_cross_dataset=False,
        #                       rf_feature_importance=True,
        #                       parametric_cross_dataset=False,
        #                      coefficients=True)

        # get_selectors(shiptype=shiptype,
        #               target_plots=True,
        #               plot_targets='BEAM (M)',
        #               plot_predictors='DWT')

        # demonstrating_functionality(shiptype=shiptype,
        #                         analysis=True,
        #                             parametric=True,
        #                             rf=True,
        #                             analyser_comparison=True,
        #                             debug=False,
        #                             save_amend='single_var3',
        #                             target_predictor_default=False)


        # multi_dataset_empirical_tester(shiptype=shiptype)

        experiment3_analysis(shiptype=shiptype,
                             rf_cross_dataset=False,
                             rf_feature_importance=False,
                             parametric_cross_dataset=False,
                             coefficients=True)

        # demo_formula_writing(shiptype=shiptype, )


    # types_list = [ShipType.BULK_CARRIER]

    # types_list = [ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    # for type in types_list:
    #     # demonstrating_functionality(shiptype=type,
    #     #                         analysis=True,
    #     #                             rf=True,
    #     #                             analyser_comparison=True,
    #     #                             debug=True,
    #     #                             save_amend='ver_5')
    #     experiment3_analysis(shiptype=type,
    #                          rf_cross_dataset=False,
    #                          rf_feature_importance=False,
    #                          parametric_cross_dataset=False,
    #                          coefficients=True)

    # for shiptype in types_list:
    #
    #     # cross_dataset_multi(shiptype=shiptype,
    #     #                     parametric=False,
    #     #                     rf=True,
    #     #                     latex=True)
    # #
    #     cross_dataset_multi(shiptype=shiptype,
    #                         parametric=True,
    #                         rf=False,
    #                         latex=True)
    #



    # for shiptype in types_list:
    #     experiment3_analysis(shiptype=shiptype,
    #                          rf_cross_dataset=False,
    #                          rf_feature_importance=True,
    #                          parametric_cross_dataset=False,
    #                          coefficients=False)






    #
    # for shiptype in types_list:
    #
    #     multi_dataset_empirical_tester(shiptype=shiptype)


    # main(shiptype=shiptype,
    #      export_selector=False,
    #      experiment1=False,
    #      experiment2=False,
    #      experiment3=True)


    # for types in types_list:
    #     main(shiptype=types,
    #          export_selector=False,
    #          experiment1=True,
    #          experiment2=False,
    #          experiment3=False
    #          )