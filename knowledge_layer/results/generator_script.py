
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility
from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
#from knowledge_layer.coefficient_and_importance_analysis.parametric_coefficient_analysier import CoefficientComparisonAnalyzer
from knowledge_layer.multi_dataset_analysers.coefficient_comparison.parametric_coefficient_comparison import CoefficientComparisonAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.parametric_cross_dataset_analysis import CrossDatasetPerformanceAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.multi_dataset_cross_performance_analyzer import MultiDatasetCrossPerformanceAnalyzer

from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector
from knowledge_layer.multi_dataset_analysers.rf_feature_importance.rf_feature_importance_Full import RFFullFeatureImportanceAnalyzer
from knowledge_layer.multi_dataset_analysers.cross_dataset.rf_cross_dataset_analysis import RandomForestCrossDatasetAnalyzer

from knowledge_layer.multi_dataset_analysers.domain_shift.model_comparison_analyser_class import ModelComparisonAnalyzer

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships, get_variables
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


def rf_predictors():

    return {
            'n_estimators': 1000,
            'max_depth': None,
            'max_features':"sqrt",
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        'oob_score':True
        }

def selecting_datasets(fleet:'FleetData',
                       datasets:dict[str,str]):

    """
    selecting the datasets to be evaluated from the fleet data class and giving them a name to be used

    :param fleet: The Fleet Data class being used
    :param datasets: The datasets to be evaluated. need to be at 2

    :return: dict with {name_of_dataset: dataframe}

    """

    result = {}

    for name, df in datasets.items():

        result[name] = fleet.processed_data[df]
        print(f"  {name} ships: {len(fleet.processed_data[df])}")

    return result


def formulae_assessment(conventional_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        target_dimensions:List[str]=None,
                        predictor_dimensions:List[str]=None,
                        shiptype:ShipType = None,
                        residuals: bool = False,
                        plots:bool=False,
                        save_dir:Path|str|None = None) -> pd.DataFrame:

    formula_excel = r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\literature_regressions\literature_formulae.xlsx'

    literature_direct = r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\literature_regressions\literature formulae'

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
        poly_selector.save_results(save_dir/f'{shiptype.name}_parametric_selector_with_outliers_lbp.pkl')
        rf_selector.save_results(save_dir/f'{shiptype.name}_rf_selector_with_outliers_lbp.pkl')


    return poly_results, poly_selector,rf_results, rf_selector


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


def experiment3_analysis(
                         shiptype: ShipType,
                        fleet:'FleetData',
                        datasets:dict,
                        bins:int,
                        variables:list,
                        save_dir:Path,
                        target_predictor_constraints,
                        selector_suffix:str,
                         coefficients:bool=True,
                        print_coef_results:bool=False,
                        coef_curve_plots:bool=True,
                        coef_bar_plots:bool=True,
        bootstrap:bool=False,
                         parametric_cross_dataset:bool=False,
                         rf_cross_dataset:bool=False,
                         rf_feature_importance:bool=True,
                        ignore_variables:list|None=['LOA (M)'],
                        norm_method:str='scale',
                        normalise:bool=True,
                        suffix:str='v1',
                        debug:bool=False,
                        build_year_col:str|None=None,
                        year_edges:list|None=None,
                        build_eligibility:TemporalEligibility|None=None,
                        n_matches_per_ship=10,
                        caliper:float|None=None,
                        matching_bins:dict[str,list]|None=None,):




    rf_param=rf_predictors()


    conventional_df = fleet.processed_data['pure_conventional']

    alternative_sets = selecting_datasets(fleet=fleet, datasets=datasets)

    poly_selector_pklpath = save_dir/f'{shiptype.name}_parametric_selector{selector_suffix}.pkl'
    rf_selector_pklpath = save_dir/f'{shiptype.name}_rf_selector{selector_suffix}.pkl'

    save_dir = save_dir / 'experiment_3'

    save_dir.mkdir(parents=True, exist_ok=True)


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




    for fuel,data in alternative_sets.items():


        if coefficients:

            print(f"\n\nSTAGE : POLYNOMIAL COEFFICIENT COMPARISON ANALYSIS")
            print("=" * 100)

            name_map = get_variable_name_maps()
            coeff_analyzer = CoefficientComparisonAnalyzer(
                parametric_selector=poly_selector,
                conventional_data=conventional_df,
                alternative_data=data,
                verbose=True,
                target_predictor_constraints=target_predictor_constraints,
                normalization_method=norm_method,
                normalize=normalise,
                build_year_col=build_year_col,
                year_edges=year_edges,
                build_eligibility=build_eligibility,
                n_matches_per_ship=n_matches_per_ship,
                caliper=caliper,
                matching_bins=matching_bins,
            )

            # Analyze all targets
            results = coeff_analyzer.analyze_all_targets(
                min_alternative_samples=10,
                display_results=print_coef_results,
                max_predictors=1,
                debug=debug
            )

            coeff_analyzer.print_exclusion_manifest(n_requested=len(variables))

            if bootstrap:

                bootstrap = coeff_analyzer.bootstrap_all_targets(n_boot=1000, seed=42,)

                table = coeff_analyzer.coefficient_uncertainty_table()
                save_root = save_dir / 'parametric'/"bootstrap"

                save_root.mkdir(parents=True, exist_ok=True)

                save_path = save_root / f'{shiptype.name}_{fuel}_bootstrap_confidence_{suffix}.xlsx'

                table.to_excel(save_path)

            if print_coef_results:

                effect_sizes = coeff_analyzer.decompose_coefficient_changes(
                    effect_size_threshold=0.2  # Small effect threshold
                )

            # =============================================================================
            # THE PLOTTER
            # =============================================================================
            # Constructed lazily off the analyser. It owns a FunctionalFormCurveExtractor,
            # which is where the model evaluation happens; the plotter itself only draws.



            # ============================================================================
            # STEP 3: Visualizations
            # ============================================================================



            # ✅ FIX: Plot coefficient decomposition for each target individually
            print("\nGenerating coefficient decomposition plots...")

            save_root = save_dir/'parametric'

            save_root.mkdir(parents=True, exist_ok=True)

            save_path = save_root / f'{shiptype.name}_{fuel}_coeff_decomposition_{suffix}.png'



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

            FORMULA_NAMES = {
                'DWT': 'DWT',
                'GT': 'GT',
                'LOA (M)': 'L',
                'BEAM (M)': 'B',
                'DRAUGHT (M)': 'T',
                'MCR (KW)': 'MCR',
                'SPEED (KTS)': 'V',
            }

            def _slug(name: str) -> str:
                """
                Convert a variable name into a filesystem-safe fragment.

                :param name: Internal variable name, e.g. ``'MCR (KW)'``.
                :type name: str
                :returns: Lowercased name with punctuation and spaces collapsed to
                    underscores, e.g. ``'mcr_kw'``.
                :rtype: str
                """
                keep = ''.join(c if c.isalnum() else '_' for c in name.lower())
                return '_'.join(part for part in keep.split('_') if part)

            if coef_bar_plots:
                coeff_analyzer.plot_coefficient_changes(show_distribution=False,
                                                        show_fuel=True,
                                                        save_path=save_path,
                                                        title_prefix=shiptype.string,
                                                        title_suffix=fuel,
                                                        min_r2=0.6,
                                                        variable_name_map = variable_name_map,
                                                        formula_name_map =formula_mappings,
                                                        figsize_per_plot =(6,3),
                                                        title_bold_spacing = 0.060,
                                                        title_italic_spacing = 0.04,
                                                        title_gap_below = 0.015,)

            if coef_curve_plots:
                ff = coeff_analyzer.functional_form
                #
                # for target in variables:
                #     ff.plot_functional_form(
                #         target,
                #         layout='panels',
                #         show_points=False,
                #         reference_dataset='conventional_full',
                #         variable_name_map=variable_name_map,
                #         formula_name_map=formula_mappings,
                #         save_path=save_root /'line_plots'/ f'{shiptype.name}_{fuel}_form_panels_{_slug(target)}.pdf',
                #         debug=debug,
                #     )

                # Any target whose selected model turned out to be multi-predictor is
                # refused by the gate rather than cut into a slice. Report the count.
                ff.extractor.print_skip_manifest(n_requested=len(variables))

                save_path = save_root /suffix

                save_path.mkdir(parents=True, exist_ok=True)

                # ff.plot_functional_form(
                #     'MCR (KW)',
                #     layout='overlay',
                #     show_points=True,
                #     point_alpha=0.25,
                #     ylim_mode='data',
                #     variable_name_map=variable_name_map,
                #     formula_name_map=formula_mappings,
                #     title='Installed power vs deadweight — bulk carriers',
                #     save_path=save_path/ f'{shiptype.name}_{fuel}_overlay_mcr.pdf',
                #     debug=debug,
                # )

                # =============================================================================
                # 8. FIGURE — ALL TARGETS ON ONE SHEET
                # =============================================================================
                # One row of three cohort panels per target. Curves are built once per target
                # and reused, so this costs no more model evaluation than the loop above.

                save_path = save_root /suffix

                save_path.mkdir(parents=True, exist_ok=True)

                DISPLAY_SCALES = {
                    'DWT': 1e3,  # → "DWT (t) ×10³"
                    'TEU': 1e3,
                    # 'MCR (KW)': (1e3, 'MCR (MW)'),  # → "MCR (MW)", a real unit change
                }

                # ff.plot_functional_forms_all_targets(
                #     targets=variables,
                #     layout='panels',
                #     variable_name_map=variable_name_map,
                #     formula_name_map=formula_mappings,
                #     suptitle=f'{shiptype.name.replace("_", " ").title()} — '
                #              f'fitted form by fuel cohort',
                #     figsize_per_panel=(4.0, 3.2),
                #     show_formula = True,
                #     save_path=save_path / f'{shiptype.name}_{fuel}_form_all_targets_panels.png',
                #     debug=debug,
                # )

                # Compact alternative: one axis per target, tiled two across.

                if shiptype == ShipType.RO_PAX:
                    N_COLS = 3
                else:
                    N_COLS = 4

                ff.plot_functional_forms_all_targets(
                    targets=variables,
                    layout='overlay',
                    n_cols=N_COLS,
                    show_formula=True,
                    min_r2=0.6,
                    display_scale_map=DISPLAY_SCALES,
                    variable_name_map=variable_name_map,
                    formula_name_map=formula_mappings,
                    figsize_per_panel=(6, 3.65),
                    suptitle_height_in=0.25,
                    r2_mode = 'box',
                    font_scale=1.4,
                    r2_decimals=3,
                    title=f'{shiptype.string} selected model plotted for {fuel} '
                          f'and corresponding matched datasets',
                    save_path=save_path / f'{shiptype.name}_{fuel}_form_all_targets_overlay_{suffix}.png',
                    show_plot=True
                )

                save_path = save_root /suffix

                save_path.mkdir(parents=True, exist_ok=True)

                # for target in variables:
                #
                #     ff.plot_functional_form(
                #         target,
                #         layout='overlay',
                #         x_range_mode='union',
                #         xlim_mode='grid',
                #         ylim_mode='curves',
                #         mark_support=True,
                #         variable_name_map=variable_name_map,
                #         formula_name_map=formula_mappings,
                #         title='Extrapolation beyond each cohort\'s observed range',
                #         save_path=save_path / f'{shiptype.name}_{fuel}_form_extrapolation_{target}.pdf',
                #     )

        # ============================================================================
        # Cross-Dataset Analysis
        # ============================================================================

        if parametric_cross_dataset:

            print("\n\nSTAGE: CROSS-DATASET PARAMETRIC PERFORMANCE ANALYSIS")
            print("=" * 100)



            param_cross_analysis = CrossDatasetPerformanceAnalyzer(conventional_data=conventional_df,
                                                                   alternative_data=data,
                                                                   independent_variables=variables,
                                                                   dependent_variables=variables,
                                                                   parametric_selector=poly_selector,
                                                                   model_recommendations=poly_selector.model_recommendations,
                                                                   verbose=True,
                                                                   build_year_col=build_year_col,
                                                                   year_edges=year_edges,
                                                                   build_eligibility=build_eligibility,)




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
                                                                alternative_data=data,
                                                                dependent_variables=variables,
                                                                target_dimensions=variables,
                                                                selector_results=rf_selector.results,
                                                                rf_params=rf_param,
                                                                build_year_col=build_year_col,
                                                                year_edges=year_edges,
                                                                build_eligibility=build_eligibility,)

            results = rf_cross_dataset.analyze_best_combinations(evaluation_method='bootstrap',)


        # ============================================================================
        # Random Forest Importance Analysis
        # ============================================================================

        if rf_feature_importance:

            if ignore_variables is not None:
                importance_variables = [v for v in variables if v not in ignore_variables]
            else:
                importance_variables = variables

            analyzer = RFFullFeatureImportanceAnalyzer(
                conventional_data=conventional_df,
                alternative_data=data,
                variable_dimensions=importance_variables,  # local name
                verbose=True,
                rf_params=rf_param,
                build_year_col=build_year_col,
                year_edges=year_edges,
                build_eligibility=build_eligibility,
                matching_bins=matching_bins,
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

            save_path = save_dir / 'random_forest' / f'{shiptype.name}_{fuel}_HM_feature_importance_{suffix}.png'

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
                                             figsize=(6.25,5.5),
                                             base_font_size=11,
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

            save_path=save_dir / 'random_forest' / f'{shiptype.name}_{fuel}_importance_summary.xlsx'

            analyzer.export_importance_summary(filepath=save_path,
                                               min_r2=0.6,
                                               debug=True)


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
        build_year_col:str|None=None,
        year_edges:list|None=None,
        build_eligibility:TemporalEligibility|None=None,
        n_matches_per_ship:int=10,
        caliper:float|None=None,
        matched_bins:dict[str,list]|None=None,
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
        target_predictor_constraints=target_predictor_constraints,
        build_year_col=build_year_col,
        year_edges=year_edges,
        build_eligibility=build_eligibility,
        n_matches_per_ship=n_matches_per_ship,
        caliper=caliper,
         matched_bins=matched_bins,
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
        selection_metric: str = 'train_rrmse',
        threshold_metric:str|None="median",
        rrmse_threshold:float=None,
        variable_groupings: dict=None,
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
            verbose_selection=True,
            # Show detailed selection process
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
            comparer.plot_domain_shift_comparison(figsize=(16,7),
                save_path=comparison_plot,
                shiptype=shiptype,
                dataset=fuel_type,
                rrmse_threshold=rrmse_threshold,
                threshold_metric=threshold_metric,
              gridspec_top=0.93,  # 0.88–0.95 typical; lower = more suptitle room
              base_font_size=9,
              tick_font_size=10,  # x-axis tick labels, set independently
              bar_label_font_size=9,
            variable_name_map=get_variable_name_maps(),
                                                  variable_groupings=variable_groupings,
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
        fleet:'FleetData',
        datasets:dict,
        bins:int,
        variables:list[str],
        target_predictor_constraints:str,
        save_dir:Path|str,
        selector_suffix:str,
        min_year:int=1990,
        selection_metric: str = 'rrmse',
        parametric: bool = True,
        rf: bool = True,
        analysis: bool = False,
        latex: bool = True,
        analyser_comparison: bool = True,
        debug: bool = False,
        save_amend: str = 'ver_5',
        target_predictor_default:bool=False,
        confidence_interval:bool=True,
        build_year_col:str|None=None,
        year_edges:list|None=None,
        build_eligibility:TemporalEligibility|None=None,
        n_matches_per_ship:int=10,
        caliper:float|None=None,
        matched_bins:dict[str,list]|None=None,
        variable_grouping:dict[str,list]|None=None,
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


    conventional_df = fleet.processed_data['pure_conventional']


    print(f"\n  Conventional ships: {len(conventional_df)}")

    alternative_sets = selecting_datasets(fleet, datasets)

    # ========================================================================
    # LOAD SELECTORS
    # ========================================================================

    poly_selector_pklpath = save_dir / f'{shiptype.name}_parametric_selector{selector_suffix}.pkl'
    rf_selector_pklpath = save_dir / f'{shiptype.name}_rf_selector{selector_suffix}.pkl'

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

    save_dir = save_dir/"experiment2"

    if parametric and analysis:
        poly_multi_analyzer = demonstrate_multi_analyser(
            conventional_df=conventional_df,
            alternative_df=alternative_sets,
            variables=variables,
            poly_selector=poly_selector,
            save_dir=save_dir/save_amend/f'param_{min_year}',
            analysis=analysis,
            shiptype=shiptype,
            prefix='parametric',
            selection_metric=selection_metric,
            debug=debug,
            save_amend=save_amend,
            target_predictor_constraints=target_predictor_constraints,
            significance_test=confidence_interval,
            save_significance=confidence_interval,
            build_year_col = build_year_col,
            year_edges=year_edges,
            build_eligibility=build_eligibility,
            n_matches_per_ship=n_matches_per_ship,
            caliper=caliper,
            matched_bins=bins
        )

    if rf and analysis:
        rf_multi_analyzer = demonstrate_multi_analyser(
            conventional_df=conventional_df,
            alternative_df=alternative_sets,
            variables=variables,
            rf_selector=rf_selector,
            analysis=analysis,
            save_dir=save_dir/save_amend/f'rf_{min_year}',
            shiptype=shiptype,
            prefix='RF',
            selection_metric=selection_metric,
            debug=debug,
            save_amend=save_amend,
            target_predictor_constraints=target_predictor_constraints,
            significance_test=confidence_interval,
            save_significance=confidence_interval,
            build_year_col = build_year_col,
            year_edges=year_edges,
            build_eligibility=build_eligibility,
            n_matches_per_ship=n_matches_per_ship,
            caliper=caliper,
            matched_bins=bins
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
            save_dir=save_dir/save_amend/f'comparison_{min_year}',
            save_amend=save_amend,
            selection_metric='train_rrmse',
            variable_groupings=variable_grouping

        )

    elif analyser_comparison:
        print("\n⚠️  Cannot run analyser_comparison: need both parametric and rf to be True")

    print("\n" + "=" * 100)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 100)


def multi_dataset_empirical_tester(shiptype:ShipType,
                                   fleet:"FleetData",
                                   datasets:dict,
                                   bins:int,
                                   variables:list,
                                   save_dir:Path,
                                   min_year:int=1990,
                                   latex:bool=False,
                                   confidence_interval:bool=False,
                                   build_year_col:str|None=None,
                                   build_eligibility:TemporalEligibility|None=None,
                                   n_matches_per_ship:int=None,
                                   suffix:str = "v3",
                                   caliper:float|None=None,
                                   matched_bins:dict[str,list]|None=None,
                                   threshold_metric:str="median",
                                   rrmse_threshold:float|None=None):
    formula_excel = r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\literature_regressions\literature_formulae.xlsx'

    literature_direct = r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\literature_regressions\literature formulae'

    print("\n" + "=" * 80)
    print("PHASE 2: Literature Formula Analysis")
    print("=" * 80)

    # Load and inspect formulae
    # formulae = LiteratureFormulaLibrary.from_file(file_path=formula_excel,
    #                                               reference='Papanikolaou',
    #                                               shiptype=shiptype)



    conventional_df = fleet.processed_data['pure_conventional']

    alternative_sets = selecting_datasets(fleet=fleet,datasets=datasets)


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
        shiptype=shiptype,
        built_year_col=build_year_col,
        build_eligiblity=build_eligibility,
        n_matches_per_ship=n_matches_per_ship,
        caliper=caliper,
        matching_bins=matched_bins,
    )

    # Run evaluation
    results = multi_evaluator.evaluate_all_datasets(display_results=True)

    # Analysis
    multi_evaluator.print_fuel_type_comparison()

    save_base = Path(save_dir/'experiment1'/suffix/f'{min_year}')

    save_path = save_base / f'{shiptype.string}_multi_empirical_{suffix}.png'

    multi_evaluator.plot_cascade_comparison(variable_name_map=variable_map,
                                            rrmse_threshold=rrmse_threshold,
                                            threshold_metric=threshold_metric,
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
                                            exclude_variables=['Vs']
                                            )

    for name in alternative_sets.keys():

        multi_evaluator.plot_predictor_target_fits(
            predictor='DWT',
            target='LBP (M)',
            fuel_type=name,
            show_domain_bounds=False,
            show_percentile_bounds=False,
            layout='horizontal'
        )

    if latex:
        multi_evaluator.generate_latex_tables(ship_type=shiptype,
                                              output_dir=save_base,
                                              suffix=suffix)

    if confidence_interval:


        sig_results = multi_evaluator.run_significance_tests(
        comparison='fuel_type',   # tests matched vs alternative gap
        n_bootstrap=5000,
        ci_method='bca',   # or 'bca' for more accuracy
    )
        multi_evaluator.export_significance_results(output_path=save_base/f'significance_tests_{suffix}.xlsx')

    multi_evaluator.generate_formula_latex_table(output_dir=save_base,
                                                 prefix=f'{shiptype.string}',
                                                 show_formula_type=False)



def demo_formula_writing(shiptype:ShipType,
                         fleet:"FleetData",
                         bins:int,
                         variables:list,
                         save_dir:Path):


    poly_selector_pklpath = f'{shiptype}_poly_selector_with_outliers_lbp.pkl'
    rf_selector_pklpath = f'{shiptype}_rf_selector_with_outliers_lbp.pkl'


    conventional_df = fleet.processed_data['pure_conventional']
    # alternative_df = fleet.processed_data['pure_alt_fuel']


    get_selectors(shiptype=shiptype,
                  variables=variables,
                  save_dir=save_dir,
                  fleet=fleet,
                  bins=bins)

def get_selectors(shiptype:ShipType,
                  fleet:"FleetData",
                  save_dir:Path,
                  bins:int,
                  variables:list[str],
                  conventional_dataset:pd.DataFrame|None=None, #ADDED FOR PLOTTING PURPOSES ONLY
                  latex:bool=True,
                  parametric:bool=True,
                  rf:bool=False,
                  param_metric_comparison:bool=False,
                  target_plots:bool=False,
                  plot_targets:str|None=None,
                  plot_predictors:str|None=None,
                  create_plots:bool=False,
                  show_hetero:bool=False,
                  plt_show:bool=True,
                  selector_suffix:str="_with_outliers_cv",
                  max_predictors:int=2
                  ):


    conventional_df = fleet.processed_data['pure_conventional']


    poly_selector_pklpath = save_dir/f'{shiptype.name}_parametric_selector{selector_suffix}.pkl'
    rf_selector_pklpath = save_dir/f'{shiptype.name}_rf_selector{selector_suffix}.pkl'

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

        if create_plots:


            # Use with ship type and formatted names

            selector.plot_model_selection(
                predictor_combo=(plot_predictors,),
                target=plot_targets,
                ship_type=shiptype.string,
                figsize=(10,6),
                metric='aic',
                exclude=["exponential","lasso", "power","power-log-multivariate","exponential-log-multivariate"],
                name_map={
                    "linear": "M1",
                    "power-offset": "M2",
                    "polynomial-2": "M3",
                    "polynomial-2-lasso": "M3†",
                    "polynomial-3": "M4",
                     "polynomial-3-lasso":"M4†",
                    "exponential-offset":"M5",
                    "exponential":"M5",
                    "logarithmic":"M6",
                    "logarithmic-power":"M7",
                    "power-log-multivariate":"M8",
                    "exponential-log-multivariate":"M9",
                    "mfp":"M10"
                },
            legend_loc='upper left',  # or 'upper right', 'lower left', 'lower right', 'best', 'outside right'
            info_box_loc=None,
                variable_name_map=variable_map,
                show_hetero=show_hetero,
                plt_show=plt_show
            )

            model_name_map = {
                    "linear": "M1",
                    "power-offset": "M2",
                    "polynomial-2": "M3",
                    "polynomial-2-lasso": "M3†",
                    "polynomial-3": "M4",
                     "polynomial-3-lasso":"M4†",
                    "exponential-offset":"M5",
                "exponential":"M5",
                    "logarithmic":"M6",
                    "logarithmic-power":"M7",
                    "power-log-multivariate":"M8",
                    "exponential-log-multivariate":"M9",
                    "mfp":"M10"
                }

            df, holdout = selector.get_holdout_metrics(dataset=conventional_dataset,metric='rrmse', test_size=0.2, random_state=42, debug=True)
            # print(df)
            # holdout=None

            selector.plot_metrics_comparison(
                targets=[plot_targets],  # one target -> one subplot
                metric='rrmse',
                figsize=(8,6),
                max_predictors=max_predictors,
                cv_method=None,
                # include_only_vars=['LBP', 'DWT'],
                ship_type=shiptype.string,
                variable_name_map=variable_map,
                model_name_map=model_name_map,
                holdout=holdout,
            )

            #
            # if param_metric_comparison:
            #     for variable in variables:
            #
            #
            #         vars= ['DWT','TEU','Vs','MCR (KW)']
            #
            #
            #         selector.plot_metrics_comparison(
            #             max_predictors=2,
            #             ship_type=shiptype.string,
            #             variable_name_map=variable_map,
            #             targets=[variable],
            #             include_only_vars=vars,
            #             group_by_type=False,
            #             metric='rrmse'
            #         )
            #
            #         selector.plot_metrics_comparison(
            #             max_predictors=2,
            #             ship_type=shiptype.string,
            #             variable_name_map=variable_map,
            #             targets=[variable],
            #             include_only_vars=variables,
            #             group_by_type=False,
            #             metric='adj_r2'
            #         )
            #
            # if target_plots:
            #
            #     selector.plot_predictor_functions(
            #         target=plot_targets,
            #         predictor=plot_predictors,
            #         show_metrics=True,
            #         separate_plots=True,
            #         show_model_type=True,
            #         highlight_recommended=False,
            #         title_prefix = shiptype.string,
            #         variable_name_map=variable_map,
            #         formula_notation_map=formula_map,
            #         debug=True
            #     )


    if rf:

        selector = RandomForestSelector.from_saved_results(results_filepath=rf_selector_pklpath,
                                                              training_data=conventional_df,
                                                              variable_dimensions=variables)

        rf_result = selector.results
        selector.show_selection_results()

        if create_plots:

            selector.plot_combination_comparison(target='LBP (M)',
                                                 metric='r2_adj',
                                                 ship_type=shiptype.string,
                                                 variable_name_map=variable_map)





    return selector



def cross_dataset_multi(shiptype:ShipType,
                        bins:int,
                        variables:list,
                        save_dir:Path,
                        target_predictor_constraints,
                        fleet:'FleetData',
                        datasets:dict,
                        suffix:str='',
                        latex:bool=False,
                        parametric:bool=False,
                        rf:bool=False,
                        build_year_col:str|None=None,
                        year_edge:list[str]|None=None,
                        build_eligibility:TemporalEligibility|None=None,
                        selector_suffix:str='',
                        evaluation_method:str="bootstrap",
                        n_folds:int|None=None,
                        n_repeats:int|None=None,
                        n_bootstraps:int=100):


    if not latex:
        save_dir=None

    rf_param = rf_predictors()

    conventional_df = fleet.processed_data['pure_conventional']
    # alternative_df = fleet.processed_data['pure_alt_fuel']
    # dual_fuel_df = fleet.processed_data['dual_fuel']

    selector = get_selectors(shiptype=shiptype,
                              latex=latex,
                              variables=variables,
                              parametric=parametric,
                              rf=rf,
                             save_dir=save_dir, fleet=fleet, bins=bins, selector_suffix=selector_suffix)

    alternative_sets = selecting_datasets(fleet=fleet,datasets=datasets,)

    if parametric:
        analyzer_type = 'parametric'
        cache = save_dir/f'cache/{shiptype.name}_cross_param.pkl'

        analyzer = MultiDatasetCrossPerformanceAnalyzer(
            conventional_data=conventional_df,
            alternative_datasets=alternative_sets,
            independent_variables=variables,
            parametric_selector=selector,
            analyzer_type=analyzer_type,
            target_predictor_constraints=target_predictor_constraints,
            build_year_col=build_year_col,
            year_edges=year_edge,
            build_eligibility=build_eligibility,
        )
        'bootstrap'
        results = analyzer.analyze_all_datasets(
            evaluation_method=evaluation_method,
            n_bootstraps=n_bootstraps,
            n_folds=n_folds,
            n_repeats=n_repeats,
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
            target_predictor_constraints=target_predictor_constraints,
            rf_params=rf_param,
            build_year_col=build_year_col,
            year_edges=year_edge,
            build_eligibility=build_eligibility,
        )

        results = analyzer.analyze_all_datasets(
            evaluation_method=evaluation_method,
            n_folds=n_folds,
            n_repeats=n_repeats,
            n_bootstraps=n_bootstraps,
            max_predictors=1,
            display_results=True

        )

    if latex:


        output_dir = save_dir / "experiment_3" / analyzer_type /suffix

        if evaluation_method == "bootstrap":
            prefix = f"cross_dataset_{analyzer_type}_{evaluation_method}_{suffix}"

        else:
            prefix = f"cross_dataset_{analyzer_type}_{suffix}"

        prefix = f"cross_dataset_{analyzer_type}_{evaluation_method}_{n_folds}_{suffix}"

        analyzer.generate_latex_tables(ship_type=shiptype,
                                       output_dir=output_dir,
                                       targets=variables,
                                       max_predictors=1,
                                       prefix=prefix
                                       )

def check_dataset(output_dir:Path):

    types_list = [ShipType.BULK_CARRIER, ShipType.TANKER, ShipType.CONTAINER_SHIP]

    fleet_data = {}

    min_year = 1990

    fuel_type_columns = ['MAIN ENGINE FUEL TYPE', 'AUXILIARY ENGINE FUEL TYPE', "MAIN BUNKER FUEL TYPE"]
    alt_fuels = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['biofuel']

    alt_fuels = {'Alternative fuel': alt_fuels,
                 'Bio fuels': bio_fuels}

    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    for shiptype in types_list:

        bins, variables,_, filepath, save_dir = get_variables(shiptype=shiptype)


        fleet = getting_alt_fuel_ships(filepath=filepath, shiptype=shiptype.string, fuel_type_columns=fuel_type_columns,
                                       max_build_year=1990,alt_fuels=alt_fuels,conventional_fuels=conventional_fuels,)

        conventional_df = fleet.processed_data['pure_conventional']
        alternative_df = fleet.processed_data['pure_Bio fuels']
        dual_fuel_df = fleet.processed_data['dual_fuel_Alternative fuel']

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


# Processed datasets:
#   - processed_date: 7020 rows, 41 columns
#   - processed_outliers: 5819 rows, 69 columns
#   - processed_sister_ships: 3226 rows, 69 columns
#   - base_with_split_fuels: 3226 rows, 79 columns
#   - any_Alt_fuel: 74 rows, 79 columns
#   - pure_Alt_fuel: 1 rows, 79 columns
#   - dual_fuel_Alt_fuel: 73 rows, 79 columns
#   - pure_conventional: 3207 rows, 79 columns
#   - dual_fuel_Alt_fuel_alt_main: 73 rows, 79 columns
#   - dual_fuel_Alt_fuel_conv_main: 0 rows, 79 columns
#   - any_Biofuel: 19 rows, 79 columns
#   - pure_Biofuel: 17 rows, 79 columns
#   - dual_fuel_Biofuel: 2 rows, 79 columns
#   - dual_fuel_Biofuel_alt_main: 2 rows, 79 columns
#   - dual_fuel_Biofuel_conv_main: 0 rows, 79 columns


if __name__ == '__main__':

    shiptype = ShipType.BULK_CARRIER
    # shiptype = ShipType.TANKER
    # shiptype = ShipType.CONTAINER_SHIP
    # shiptype = ShipType.RO_PAX
    # #
    types_list = [ShipType.BULK_CARRIER,ShipType.CONTAINER_SHIP, ShipType.RO_PAX,ShipType.TANKER]
    # types_list = [ShipType.TANKER, ShipType.BULK_CARRIER]
    # types_list = [ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.RO_PAX]
    # types_list = [ShipType.CONTAINER_SHIP,ShipType.TANKER]
    # # types_list = [ShipType.BULK_CARRIER, ShipType.RO_PAX,ShipType.TANKER]
    types_list = [ShipType.TANKER]
    # types_list = []
    # # # types_list = [ShipType.BULK_CARRIER]
    types_list = [ShipType.CONTAINER_SHIP]
    # types_list = [ShipType.RO_PAX]



    min_year = 1990
    match_number = 5
    caliper=None

    fuel_type_columns = ['MAIN_FUEL', 'AUX_FUEL', "BUNKER_FUEL"]
    alt_fuels = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['biofuel']

    alt_fuels = {'Alt_fuel': alt_fuels,
                 'Biofuel': bio_fuels}

    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    datasets = {"Dual fuel":"dual_fuel_Alt_fuel",
                "Biofuel":"pure_Biofuel"}

    # selector_suffix = '_sister_ship_hull'
    # selector_suffix = "_with_outliers_basic"
    selector_suffix = '_sister_ship_new_preproc_3'

    build_eligibility  = TemporalEligibility(anchor_column="BUILT", max_lag=5, max_lead=5)


    from knowledge_layer.dataset_storage_and_testing.hypothesis_testing import main
    #
    #
    #
    alt_datasets = ["dual_fuel_LNG", "dual_fuel_Methanol"]
    #
    # "dual fuel LNG": "dual_fuel_LNG", "Bio fuel": "pure_Biofuel"

    year = 1990

    # main(fuel_type_columns=fuel_type_columns,
    #      make_plots=False,
    #      block1=False,
    #      matching_sensitivity=False,
    #      progressive_comparison=False,
    #      multi_ship_comparison=True,
    #      run_diagnostics=False,
    #      alt_datasets=alt_datasets,
    #      match_type='full',
    #      min_date=year)

    #
    version = "refined3"
    for shiptype in types_list:

        bins, variables, target_predictor_constraints, filepath, save_dir, datasets = get_variables(shiptype=shiptype)

        if shiptype == ShipType.RO_PAX:
            variables.remove('Np')
            variables.remove('Nv')



        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=shiptype.string,
                                       fuel_type_columns=fuel_type_columns,
                                       max_build_year=min_year,
                                       alt_fuels=alt_fuels,
                                       sister_ships=True,
                                       outliers=False,
                                       conventional_fuels=conventional_fuels)

        dimensions = ["BEAM (M)", "LOA (M)", "LBP (M)", "DRAFT (M)"]
        # dimensions = ["DWT"]
        # dimensions = ["MCR (KW)"]
        dimensions = ["DWT"]

        conventional_dataset = pd.read_excel(r"C:\Users\bnoble\OneDrive - Delft University of Technology\Desktop\CONTAINERSHIPCONVENTIONALUSED.xlsx")



        for dimension in dimensions:
            get_selectors(
                shiptype=shiptype,
                fleet=fleet,
                save_dir=save_dir,
                bins=bins,
                variables=variables,
                target_plots=True,
                plot_targets=dimension,
                plot_predictors="LBP (M)",
                create_plots=True,
                param_metric_comparison=True,
                selector_suffix = selector_suffix,
                 max_predictors=2,
                conventional_dataset=conventional_dataset,
            )
    #
    #     # for variable in variables:
    #     #
    #     #     predictors = [v for v in variables if v != variable]
    #     #
    #     #     print(predictors)
    #     #
    #     #     for pred in predictors:
    #     #         get_selectors(
    #     #             shiptype=shiptype,
    #     #             fleet=fleet,
    #     #             save_dir=save_dir,
    #     #             bins=bins,
    #     #             variables=variables,
    #     #             target_plots=True,
    #     #             plot_targets=variable,
    #     #             plot_predictors=pred,
    #     #             create_plots=True,
    #     #             param_metric_comparison=True
    #     #         )
    # # # #
    # #
    #     # multi_dataset_empirical_tester(shiptype=shiptype, fleet=fleet,datasets=datasets, variables=variables,
    #     #                                save_dir=save_dir, bins=bins, min_year=min_year, latex=True, confidence_interval=True,
    #     #                                suffix=f"with_sisterships_{version}",
    #     #                                build_eligibility=build_eligibility, n_matches_per_ship=match_number, caliper=caliper,
    #     #                                matched_bins = bins)
    #
    #     # multi_dataset_empirical_tester(shiptype=shiptype, fleet=fleet,datasets=datasets, variables=variables,
    #     #                                save_dir=save_dir, bins=bins, min_year=min_year, latex=True, confidence_interval=True,
    #     #                                suffix=f"with_sisterships_{version}_excluded_capacity",
    #     #                                build_eligibility=build_eligibility, n_matches_per_ship=match_number, caliper=caliper,
    #     #                                matched_bins = bins)
    # # #     # # # #
    # # #     # # # #
    # # #     # # # # demo_formula_writing(shiptype=shiptype,fleet=fleet, bins=bins, variables=variables,save_dir=save_dir)
    # # #     # # #
        demonstrating_functionality(shiptype=shiptype, fleet=fleet, datasets=datasets, variables=variables,
                                    target_predictor_constraints=target_predictor_constraints, save_dir=save_dir,bins=bins,
                                analysis=True,min_year=min_year,confidence_interval=True,
                                    parametric=True,rf=True,analyser_comparison=True,
                                    debug=False,
                                    save_amend=f"with_sisterships_test_Results",
                                    target_predictor_default=False,
                                    build_eligibility=build_eligibility,
                                    selector_suffix=selector_suffix,
                                    n_matches_per_ship=match_number, caliper=caliper,
                                    matched_bins = bins,
                                    variable_grouping={'Hull dimensions': ['LBP (M)', 'LOA (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)']},)
    # # # # # #     #
    # # #
    # # #
    # #     experiment3_analysis(shiptype=shiptype,
    # #                          fleet=fleet, datasets=datasets, variables=variables, bins=bins,save_dir=save_dir,
    # #                           rf_cross_dataset=False,
    # #                           rf_feature_importance=True,
    # #                           parametric_cross_dataset=False,
    # #                          coefficients=True,
    # #                          coef_bar_plots=False,
    # #                          bootstrap=False,
    # #                          target_predictor_constraints=target_predictor_constraints,
    # #                          suffix=f"with_sisterships_{version}",
    # #                          debug=False,
    # #                          normalise=False,
    # #                          build_eligibility=build_eligibility,
    # #                          selector_suffix=selector_suffix,
    # #                          n_matches_per_ship=match_number, caliper=caliper, matching_bins=bins)
    # # #
    #     n_folds = 5
    #     n_repeats = 100
    # #
    # #
    # #
    # #
    #     # cross_dataset_multi(shiptype=shiptype,fleet=fleet,
    #     #                     datasets=datasets, bins=bins, variables=variables,
    #     #                     save_dir=save_dir,
    #     #                     parametric=True,
    #     #                     target_predictor_constraints=target_predictor_constraints,
    #     #                     rf=False,
    #     #                     latex=True,
    #     #                     suffix=f"with_sisterships_{version}",
    #     #                     build_eligibility=build_eligibility,
    #     #                     selector_suffix=selector_suffix,
    #     #                     evaluation_method="repeated-cv",
    #     #                     n_repeats=100,
    #     #                     n_folds=5,
    #     #                     )
    #     #
    #
    #     cross_dataset_multi(shiptype=shiptype,fleet=fleet,
    #                         datasets=datasets, bins=bins, variables=variables,
    #                         save_dir=save_dir,
    #                         parametric=False,
    #                         target_predictor_constraints=target_predictor_constraints,
    #                         rf=True,
    #                         suffix=f"with_sisterships_{version}",
    #                         latex=True,
    #                         build_eligibility=build_eligibility,
    #                                 selector_suffix=selector_suffix,
    #                                 evaluation_method="repeated-cv",
    #                                 n_repeats=n_repeats,
    #                                 n_folds=n_folds)
