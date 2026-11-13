
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_regression_analyser_class import ParametricDomainAnalyzer
from knowledge_layer.coefficient_and_importance_analysis.parametric_coefficient_analysier import CoefficientComparisonAnalyzer

from knowledge_layer.model_selectors.random_forest_selector_class import RandomForestSelector
from knowledge_layer.multi_dataset_analysers.domain_shift.random_forest_domain_analysis_clas import RandomForestDomainAnalyzer
from knowledge_layer.random_forest_regressions.random_forest_feature_importance import RFImportanceComparisonAnalyzer

from knowledge_layer.multi_dataset_analysers.domain_shift.model_comparison_analyser_class import ModelComparisonAnalyzer
from knowledge_layer.random_forest_regressions.RandomForestFullPredictor_class import RandomForestFullPredictor
from knowledge_layer.parametric_regression.parametric_training_comparison import TrainingDomainComparator

from dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships

from knowledge_layer.multi_dataset_analysers.ParametricDomainShiftLaTeX import ParametricDomainAnalysisLatexTableGenerator
from knowledge_layer.multi_dataset_analysers.RFDomainShiftLaTeX import RandomForestLatexTableGenerator

from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.literature_regressions.Literature_formula_evaluation_class import LiteratureFormulaEvaluator
from knowledge_layer.literature_regressions.literature_formula_libraries_class import LiteratureFormulaLibrary


def plot_dimension_by_decade_seaborn(df, year_col, dim_col):
    """
    Creates:
      1. Stacked Seaborn boxplots of dimension by decade
      2. Ridge-style KDE plot by decade

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    year_col : str
        Name of the column containing years.
    dim_col : str
        Name of the numeric dimension column to plot.
    """

    # Filter valid rows
    df = df[[year_col, dim_col]].dropna()

    # Convert years → decades
    df["Decade"] = (df[year_col] // 10) * 10
    df["Decade"] = df["Decade"].astype(int)

    # Order decades
    decades = sorted(df["Decade"].unique())

    # --------------------------------------------
    # 1) Boxplot (stacked vertically)
    # --------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="Decade",
        y=dim_col,
        order=decades,
        palette="Blues"
    )
    plt.title(f"{dim_col} distribution by decade")
    plt.xlabel("Decade")
    plt.ylabel(dim_col)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # 2) Ridge-like plot: KDE FacetGrid
    # --------------------------------------------
    # This makes a “stacked” KDE ridge plot look
    g = sns.FacetGrid(
        df,
        row="Decade",
        hue="Decade",
        aspect=4,
        height=1.2,
        palette="Blues",
        row_order=decades
    )

    g.map(sns.kdeplot, dim_col, fill=True, alpha=0.8)
    g.map(plt.axhline, y=0, lw=1, color="black")
    g.set_titles("{row_name}s")
    g.set(yticks=[], ylabel="")
    g.fig.subplots_adjust(hspace=-0.6)
    g.fig.suptitle(f"{dim_col} density by decade (ridge-style)", y=1.02)

    plt.show()





def get_variables(shiptype:ShipType):

    print(f'USING DATA FOR {shiptype} SHIP')

    if shiptype == ShipType.TANKER:

        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Tanker')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
        variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']


    elif shiptype == ShipType.BULK_CARRIER:
        print('bulk carrier filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Bulk_carrier')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
        variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']
        # variables = ['DWT', 'LBP (M)', 'DRAFT (M)']

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('container ship filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\Containership')
        bins = None
        variables = ['DWT', 'TEU', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']

    else:
        print('ropax filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
        save_dir = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\RO_Pax')
        bins = None
        variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD', 'LM', 'Np', 'Nv','Vs','MCR (KW)']

    return bins,variables, filepath, save_dir

def poly_regression_selector(filepath:Path,
                        conventional_df:pd.DataFrame,
                        reccomendations:Path = None,
                        variable_dimensions:List[str]=None,
                        target_dimensions:List[str]=None,
                        cv_method:str|None=None,
                        ranking_att ='rrmse',
                        fixed_dimensions:List[str]=['DWT']):


    if variable_dimensions is None:
        variable_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)']

    if target_dimensions is None:
        target_dimensions = variable_dimensions

    # STAGE 1: Find best models (DWT always included)
    print("STAGE 1: DWT-CONSTRAINED MODEL SELECTION")
    print("=" * 100)
    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variable_dimensions,
                                            target_dimensions=target_dimensions,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='rrmse')

    model_dict = poly_selector.analyze_all_combinations(max_additional_predictors=2, cv_method=cv_method)

    # Export and visualize
    if reccomendations is not None:
        poly_selector.export_recommendations(reccomendations)
    poly_selector.plot_metrics_comparison(fixed_parameters=fixed_dimensions,
                                     cv_method=cv_method,
                                     metric=ranking_att,
                                     )

    poly_selector.plot_cv_stability(fixed_parameters=fixed_dimensions)



    print("\n\nSTAGE 2: DWT-CONSTRAINED random forest regression selection")
    print("=" * 100)

    rf_selector = RandomForestSelector(
        training_data=conventional_df,
        variable_dimensions=variable_dimensions,
        target_dimensions=target_dimensions,
        fixed_dimensions=['DWT'],
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        selection_metric='rrmse',
        verbose=True
    )

    # Analyze all combinations
    rf_results = rf_selector.analyze_all_combinations(
        max_additional_predictors=2,
        cv_method='stratified',
        n_folds=10,
        random_state=42
    )

    # Print summary
    rf_selector.print_summary()

    # ============================================================================
    # 2.2 PLOTTING
    # ============================================================================

    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='rrmse',
    )

    # Plot feature importance
    rf_selector.plot_feature_importance(
    )


def DomainShiftAnalysis(filepath:Path,
                        conventional_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        reccomendations:Path = None,
                        variable_dimensions:List[str]=None,
                        target_dimensions:List[str]=None,
                        cv_method:str|None=None,
                        ranking_att ='rrmse',
                        fixed_dimensions:List[str]=['DWT'],
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,
                        polynomial:bool=True,
                        rf_regression:bool=True,
                        compare_shift:bool=True,
                        compare_coeffLbool=True):


    if variable_dimensions is None:
        variable_dimensions = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM MLD (M)']

    if target_dimensions is None:
        target_dimensions = variable_dimensions

    stages=1

    if polynomial:



        # STAGE 1: Find best models (DWT always included)
        print(f"STAGE {stages}: POLYNOMIAL REGRESSION MODEL SELECTION")
        print("=" * 100)
        poly_selector = ParametricModelSelector(conventional_df,
                                                variable_dimensions=variable_dimensions,
                                                target_dimensions=target_dimensions,
                                                verbose=True,
                                                fixed_dimensions=fixed_dimensions,
                                                selection_metric='bic',
                                                include_regularized=include_regularized,
                                                use_cv_regularization=use_cv_regularization)

        model_dict = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

        # Export and visualize
        if reccomendations is not None:
            poly_selector.export_recommendations(reccomendations)

        poly_selector.plot_metrics_comparison(
                                         cv_method=cv_method,
                                         metric=ranking_att,
            group_by_type=True
                                         )

        poly_selector.plot_metrics_comparison(
                                         cv_method=cv_method,
            group_by_type=True
                                         )

        poly_selector.plot_cv_stability()

        poly_selector.plot_all_single_predictor_regressions(top_n_per_target = 6)

        stages+=1

        print(f"\n\nSTAGE {stages}: POLYNOMIAL DOMAIN SHIFT ANALYSIS")
        print("=" * 100)

        poly_analyzer = ParametricDomainAnalyzer(
            conventional_data=conventional_df,
            alternative_data=alternative_df,
            independent_variables=variable_dimensions,
            dependent_variables=target_dimensions,
            model_recommendations=model_dict,  # Use best models from step 1
            test_size=0.2,
            random_state=42
        )

        # Step 3: Analyze
        results = poly_analyzer.analyze_all_combinations(max_predictors=2)

        # Step 4: Interpret
        poly_analyzer.interpret_results(significance_threshold_r2=0.05)

        poly_analyzer.plot_domain_shift_analysis()

        poly_analyzer.plot_prediction_scatter()

        poly_analyzer.plot_residuals()

        poly_analyzer.plot_single_predictor_fits()

        stages += 1
        print(f"\n\nSTAGE {stages}: POLYNOMIAL TRAINING SHIFT ANALYSIS")
        print("=" * 100)

        coeff_analyzer = CoefficientComparisonAnalyzer(
            parametric_selector=poly_selector,
            conventional_data=conventional_df,
            alternative_data=alternative_df,
            verbose=True
        )

        # Analyze all targets
        results = coeff_analyzer.analyze_all_targets(
            min_alternative_samples=10,
            display_results=True
        )

        effect_sizes = coeff_analyzer.interpret_coefficient_changes(
            effect_size_threshold=0.2  # Small effect threshold
        )

        # ============================================================================
        # STEP 3: Visualizations
        # ============================================================================

        # Plot coefficient changes



        coeff_analyzer.plot_coefficient_comparison(save_path='coefficient_changes.png')

        # Plot cross-evaluation
        coeff_analyzer.plot_cross_evaluation(save_path='model_transfer.png')



        #
        # comparator = TrainingDomainComparator(domain_analyzer=poly_analyzer,
        #                                       model_selector=poly_selector)
        #
        # # Run comparison
        # results = comparator.compare_training_domains(
        #     max_predictors=2,
        #     training_sources=['conventional_full', 'conventional_matched', 'alternative']
        # )
        #
        #
        # # Specific target - average across all predictor combinations for SPEED
        #
        # for target in target_dimensions:
        #     comparator.plot_performance_heatmap(target=target, metric='r2')
        #
        #
        # comparator.plot_prediction_lines(target='SPEED (KN)', predictor='DWT')
        # comparator.plot_coefficient_comparison(target='GT', predictors=['DWT', 'LOA (M)'])

    # ============================================================================
    # 3: Random forest regression
    # ============================================================================

    if rf_regression:

        stages+=1
        print(f"\n\nSTAGE {stages}:  random forest regression selection")
        print("=" * 100)

        rf_selector = RandomForestSelector(
            training_data=conventional_df,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            fixed_dimensions=['DWT'],
            rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'max_features':0.5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
            selection_metric='adj_r2',
            verbose=True
        )

        # Analyze all combinations
        rf_results = rf_selector.analyze_all_combinations(
            max_predictors=2,
            cv_method='stratified',
            n_folds=10,
            random_state=42,
            use_feature_engineering=True,
            group_by_type=True
        )

        # Print summary
        rf_selector.print_summary()


        # ============================================================================
        # 3.2 PLOTTING
        # ============================================================================



        # Plot RRMSE comparison
        rf_selector.plot_metrics_comparison(
            metric='rrmse',
            group_by_type=True,
        )

        # Plot RRMSE comparison
        rf_selector.plot_metrics_comparison(
            metric='adj_r2',
            group_by_type=True,
        )

        # Plot feature importance
        rf_selector.plot_feature_importance(
        )

        # ============================================================================
        # Full prediction analysis
        # ============================================================================

        stages+=1

        print(f"\n\nSTAGE {stages}: random forest domain shift analysis")
        print("=" * 100)

        predictor = RandomForestFullPredictor(
            data=conventional_df,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'max_features':0.5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
            verbose=True)

        predictor.train_all_targets()

        comparison = predictor.compare_with_selector(rf_selector.results)

        print(comparison)

        # Plot performance comparison (bar chart)
        predictor.plot_performance_comparison(metric='r2')

        # Plot feature importance for all targets (subplots)
        predictor.plot_feature_importance_comparison()

        # Bonus: Heatmap view
        predictor.plot_feature_importance_heatmap()

        #Compare alternative fuel results

        print(f"\n\nSTAGE {stages}: random forest domain shift analysis for alternatively fuelled ships")
        print("=" * 100)

        predictor = RandomForestFullPredictor(
            data=alternative_df,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            rf_params={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            verbose=True)

        predictor.train_all_targets()


        # Plot performance comparison (bar chart)
        predictor.plot_performance_comparison(metric='r2')

        # Plot feature importance for all targets (subplots)
        predictor.plot_feature_importance_comparison()

        # Bonus: Heatmap view
        predictor.plot_feature_importance_heatmap()


        # ============================================================================
        #CREATE RANDOM FOREST DOMAIN ANALYZER
        # ============================================================================
        stages+=1

        print(f"\n\nSTAGE {stages}: random forest domain shift analysis")
        print("=" * 100)

        rf_analyzer = RandomForestDomainAnalyzer(
            conventional_data=conventional_df,
            alternative_data=alternative_df,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            selector_results=rf_results,
            rf_params={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features':0.5,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            },
            test_size=0.2,
            random_state=42,
            verbose=True
        )

        # ============================================================================
        # 4.1. ANALYZE ALL COMBINATIONS
        # ============================================================================

        rf_results = rf_analyzer.analyze_all_combinations(max_predictors=2)

        # ============================================================================
        # 4.2. INTERPRET RESULTS
        # ============================================================================

        rf_analyzer.interpret_results(
            significance_threshold_r2=0.05,
            significance_threshold_rrmse=10.0
        )

        # ============================================================================
        # 4.3. VISUALIZE
        # ============================================================================

        # Domain shift plots
        rf_analyzer.plot_domain_shift_analysis(
        )

        # Feature importance
        rf_analyzer.plot_feature_importance_analysis(
        )

        # Create scatter plots
        rf_analyzer.plot_prediction_scatter()

        # Create residual plots
        rf_analyzer.plot_residuals()

        # ============================================================================
        # 4.4. GET SUMMARY
        # ============================================================================

        summary = rf_analyzer.get_summary()
        for target, info in summary.items():
            print(f"\n{target}:")
            print(f"  Best predictors: {info['predictors']}")
            print(f"  Alternative RRMSE: {info['alternative_rrmse']:.2f}%")
            print(f"  Fuel gap (RRMSE): {info['fuel_gap_rrmse']:+.2f}%")
        #
    # # ============================================================================
    # # 5. CREATE copmarison plots
    # # ============================================================================


    # ============================================================================
    # STEP 2: Importance Comparison Analysis
    # ============================================================================
    importance_analyzer = RFImportanceComparisonAnalyzer(
        rf_selector=rf_selector,
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        calculate_permutation_importance=False,  # Set True for more reliable results (slower)
        verbose=True
    )

    # Analyze all targets
    results = importance_analyzer.analyze_all_targets(
        min_alternative_samples=10,
        display_results=True
    )

    # Get importance changes
    changes = importance_analyzer.get_importance_changes(
        threshold=0.05,  # 5 percentage points
        sort_by='absolute'
    )

    # ============================================================================
    # STEP 3: Visualizations
    # ============================================================================

    # Plot importance comparison (side-by-side bars)
    importance_analyzer.plot_importance_comparison(
        save_path='importance_comparison.png'
    )

    # Plot importance changes (horizontal bars showing gains/losses)
    importance_analyzer.plot_importance_changes(
        threshold=0.05,
        save_path='importance_changes.png'
    )

    # Plot cross-evaluation
    importance_analyzer.plot_cross_evaluation(
        save_path='rf_cross_evaluation.png'
    )

    if polynomial and rf_regression and compare_shift:
        stages+=1

        print(f"\n\nSTAGE {stages}: Comparing random forest and polynomial regression")
        print("=" * 100)

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


        comparer.plot_domain_shift_comparison()

def selectors_and_analysers(shiptype:ShipType,
                            conventional_df:pd.DataFrame,
                            alternative_df:pd.DataFrame,
                        fixed_dimensions:List[str]=['DWT'],
                        cv_method:str|None = None,
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,

                        parametric:bool=True,
                        rf:bool=True):
    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    print(ship_type)

    bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)



    print(('=' * 20) + f'STEP 2: Parametric regressions' + ('=' * 20))
    print(f'=' * 60)

    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variables,
                                            target_dimensions=variables,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='bic',
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization,
                                            fit_mfp=True)

    model_dict = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

    print("\n\n" + "=" * 70)
    print("Step 3: ANALYZER RESULTS (Domain Shift)")
    print("=" * 70)

    poly_analyzer = ParametricDomainAnalyzer(
        conventional_data=conventional_df,  # Will be split 80/20
        alternative_data=alternative_df,
        independent_variables=variables,
        dependent_variables=variables,
        model_recommendations=model_dict,  # Use best models from step 1
        test_size=0.2,
        random_state=42
    )

    # Step 3: Analyze
    results = poly_analyzer.analyze_all_combinations(max_predictors=2)


    print(f"\n\nSTAGE:  random forest regression selection")
    print("=" * 100)

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
            'n_jobs': -1
        },
        selection_metric='rrmse',
        verbose=True
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

    rf_selector.plot_metrics_comparison(
        metric='rrmse',
        save_path=save_dir,

        group_by_type=False,
    )

    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='adj_r2',
        save_path=save_dir,
        group_by_type=False,
    )

    print(f"\n\nSTAGE: random forest domain shift analysis")
    print("=" * 100)

    rf_analyzer = RandomForestDomainAnalyzer(
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        variable_dimensions=variables,
        target_dimensions=variables,
        selector_results=rf_results,
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        },
        test_size=0.2,
        random_state=42,
        verbose=True
    )

    # ============================================================================
    # 4.1. ANALYZE ALL COMBINATIONS
    # ============================================================================

    rf_results = rf_analyzer.analyze_all_combinations(max_predictors=2)

    return poly_selector, poly_analyzer, rf_selector, rf_analyzer

def selectors(shiptype:ShipType,
                            conventional_df:pd.DataFrame,
                        fixed_dimensions:List[str]=['DWT'],
                        cv_method:str|None = None,
                        include_regularized:bool=True,
                        use_cv_regularization:bool=True,
                        parametric:bool=True,
                        rf:bool=True):
    print(f'=' * 60)
    print(('=' * 20) + f'RUNNING FULL ANALYSIS FOR {shiptype}' + ('=' * 20))

    print(('=' * 20) + f'STEP 1: Loading data' + ('=' * 20))
    print(f'=' * 60)

    ship_type = shiptype.name

    print(ship_type)

    bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)

    print(('=' * 20) + f'STEP 2: Parametric regressions' + ('=' * 20))
    print(f'=' * 60)

    poly_selector = ParametricModelSelector(conventional_df,
                                            variable_dimensions=variables,
                                            target_dimensions=variables,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric='bic',
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization,
                                            fit_mfp=True)

    poly_results = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

    print(f"\n\nSTAGE:  random forest regression selection")
    print("=" * 100)

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
            'n_jobs': -1
        },
        selection_metric='rrmse',
        verbose=True
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

    rf_selector.plot_metrics_comparison(
        metric='rrmse',
        save_path=save_dir,

        group_by_type=False,
    )

    # Plot RRMSE comparison
    rf_selector.plot_metrics_comparison(
        metric='adj_r2',
        save_path=save_dir,
        group_by_type=False,
    )

    return poly_results, poly_selector,rf_results, rf_selector


def analysers(poly_results,
             rf_results,
             variables,
             conventional_df,
             alternative_df):
    print("\n\n" + "=" * 70)
    print("Step 3: ANALYZER RESULTS (Domain Shift)")
    print("=" * 70)

    poly_analyzer = ParametricDomainAnalyzer(
        conventional_data=conventional_df,  # Will be split 80/20
        alternative_data=alternative_df,
        independent_variables=variables,
        dependent_variables=variables,
        model_recommendations=poly_results,  # Use best models from step 1
        test_size=0.2,
        random_state=42
    )

    # Step 3: Analyze
    results = poly_analyzer.analyze_all_combinations(max_predictors=2)


    print(f"\n\nSTAGE: random forest domain shift analysis")
    print("=" * 100)

    rf_analyzer = RandomForestDomainAnalyzer(
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        variable_dimensions=variables,
        target_dimensions=variables,
        selector_results=rf_results,
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        },
        test_size=0.2,
        random_state=42,
        verbose=True
    )



    # ============================================================================
    # 4.1. ANALYZE ALL COMBINATIONS
    # ============================================================================

    rf_results = rf_analyzer.analyze_all_combinations(max_predictors=2)

    return  poly_analyzer, rf_analyzer



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
    feature_importance = save_dir/'feature_importance.png'
    rf_analyzer.plot_feature_importance_analysis(save_path=feature_importance)

    print(f"\n\nSTAGE: Comparing random forest and polynomial regression")
    print("=" * 100)

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

def experiment3_analysis(poly_selector:ParametricModelSelector,
                      poly_analyzer:ParametricDomainAnalyzer,
                      rf_selector:RandomForestSelector,
                      rf_analyzer:RandomForestDomainAnalyzer,
                      shiptype:ShipType,
                      save_dir: str | Path,
                         conventional_df:pd.DataFrame,
                         alternative_df:pd.DataFrame,
                      cv_method = None,):

    print(f"\n\nSTAGE : POLYNOMIAL TRAINING SHIFT ANALYSIS")
    print("=" * 100)

    coeff_analyzer = CoefficientComparisonAnalyzer(
        parametric_selector=poly_selector,
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        verbose=True
    )

    # Analyze all targets
    results = coeff_analyzer.analyze_all_targets(
        min_alternative_samples=10,
        display_results=True
    )

    effect_sizes = coeff_analyzer.interpret_coefficient_changes(
        effect_size_threshold=0.2  # Small effect threshold
    )

    # ============================================================================
    # STEP 3: Visualizations
    # ============================================================================

    # Plot coefficient changes

    coeff_analyzer.plot_coefficient_comparison(save_path=save_dir/'coefficient_changes.png')

    # Plot cross-evaluation
    coeff_analyzer.plot_cross_evaluation(save_path=save_dir/'model_transfer.png')


    importance_analyzer = RFImportanceComparisonAnalyzer(
        rf_selector=rf_selector,
        conventional_data=conventional_df,
        alternative_data=alternative_df,
        calculate_permutation_importance=False,  # Set True for more reliable results (slower)
        verbose=True
    )

    # Analyze all targets
    results = importance_analyzer.analyze_all_targets(
        min_alternative_samples=10,
        display_results=True
    )

    # Get importance changes
    changes = importance_analyzer.get_importance_changes(
        threshold=0.05,  # 5 percentage points
        sort_by='absolute'
    )

    # ============================================================================
    # STEP 3: Visualizations
    # ============================================================================

    # Plot importance comparison (side-by-side bars)
    importance_analyzer.plot_importance_comparison(
        save_path=save_dir/'importance_comparison.png'
    )

    # Plot importance changes (horizontal bars showing gains/losses)
    importance_analyzer.plot_importance_changes(
        threshold=0.05,
        save_path=save_dir/'importance_changes.png'
    )

    # Plot cross-evaluation
    importance_analyzer.plot_cross_evaluation(
        save_path=save_dir/'rf_cross_evaluation.png'
    )


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

    bins,variables, filepath, save_dir = get_variables(shiptype=shiptype)

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=ship_type)


    #
    # if shiptype == ShipType.TANKER:
    #     filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    #     bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
    #     variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']
    #
    #
    # elif shiptype == ShipType.BULK_CARRIER:
    #     filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    #     bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
    #     # variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']
    #     variables = ['DWT', 'LBP (M)', 'DRAFT (M)']
    #
    # elif shiptype == ShipType.CONTAINER_SHIP:
    #     filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    #     bins = None
    #     variables = ['DWT', 'TEU', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','Vs','MCR (KW)']
    #
    # else:
    #     filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
    #     bins = None
    #     variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD', 'LM', 'Np', 'Nv','Vs','MCR (KW)']

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=ship_type)


    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']

    if parametric:

        poly_selector = ParametricModelSelector(conventional_df,
                                                variable_dimensions=variables,
                                                target_dimensions=variables,
                                                verbose=True,
                                                fixed_dimensions=fixed_dimensions,
                                                selection_metric='bic',
                                                include_regularized=include_regularized,
                                                use_cv_regularization=use_cv_regularization,
                                                fit_mfp=True)

        model_dict = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)



        poly_selector.plot_metrics_comparison(
            cv_method=cv_method,
            metric='rrmse',
            title=f'RRMSE (%) comparison of parametric models for {shiptype}',
            group_by_type=False,
            save_path = save_dir
        )

        poly_selector.plot_metrics_comparison(
            cv_method=cv_method,
            metric='adj_r2',
            title=f'adj r^2 comparison of parametric models for {shiptype}',
            group_by_type=False,
            save_path = save_dir
        )

        generator_selector = ParametricDomainAnalysisLatexTableGenerator(poly_selector)

        print("\nGenerating selector R² table...")
        r2_table = generator_selector.generate_selector_r2_table(max_predictors=2,ship_type=ship_type)
        print("\nPreview (first 15 lines):")
        for line in r2_table.split('\n')[:15]:
            print(line)

        print("\nGenerating selector R² table...")
        rrmse_table = generator_selector.generate_selector_rrmse_table(max_predictors=2,
                                                                       ship_type=ship_type)
        print("\nPreview (first 15 lines):")
        for line in rrmse_table.split('\n')[:15]:
            print(line)

        generator_selector.save_tables(ship_type=ship_type,
                                       output_dir= save_dir,
                                       best_only=True)

        print("\n\n" + "=" * 70)
        print("MODE 2: ANALYZER RESULTS (Domain Shift)")
        print("=" * 70)

        poly_analyzer = ParametricDomainAnalyzer(
            conventional_data=conventional_df,  # Will be split 80/20
            alternative_data=alternative_df,
            independent_variables=variables,
            dependent_variables=variables,
            model_recommendations=model_dict,  # Use best models from step 1
            test_size=0.2,
            random_state=42
        )

        # Step 3: Analyze
        results = poly_analyzer.analyze_all_combinations(max_predictors=2)



        generator_analyzer = ParametricDomainAnalysisLatexTableGenerator(poly_analyzer)

        print("\nGenerating analyzer R² table...")
        r2_table_analyzer = generator_analyzer.generate_r2_table(max_predictors=2, ship_type=ship_type)
        print("\nPreview (first 15 lines):")
        for line in r2_table_analyzer.split('\n')[:15]:
            print(line)

        rrmse_table_analyzer = generator_analyzer.generate_rrmse_table(max_predictors=2, ship_type=ship_type)

        print("\nPreview (first 15 lines):")
        for line in rrmse_table_analyzer.split('\n')[:15]:
            print(line)

        generator_analyzer.save_tables(ship_type=ship_type,
                                       output_dir=save_dir)

    if rf:
        print(f"\n\nSTAGE:  random forest regression selection")
        print("=" * 100)

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
                'n_jobs': -1
            },
            selection_metric='rrmse',
            verbose=True
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

        rf_selector.plot_metrics_comparison(
            metric='rrmse',
            save_path=save_dir,

            group_by_type=False,
        )

        # Plot RRMSE comparison
        rf_selector.plot_metrics_comparison(
            metric='adj_r2',
            save_path=save_dir,
            group_by_type=False,
        )

        print(f"\n\nSTAGE: random forest domain shift analysis")
        print("=" * 100)

        rf_analyzer = RandomForestDomainAnalyzer(
            conventional_data=conventional_df,
            alternative_data=alternative_df,
            variable_dimensions=variables,
            target_dimensions=variables,
            selector_results=rf_results,
            rf_params={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 0.5,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            },
            test_size=0.2,
            random_state=42,
            verbose=True
        )

        # ============================================================================
        # 4.1. ANALYZE ALL COMBINATIONS
        # ============================================================================

        rf_results = rf_analyzer.analyze_all_combinations(max_predictors=2)

        rf_generator = RandomForestLatexTableGenerator(rf_analyzer)

        rf_generator.save_tables(ship_type=ship_type,
                                 best_only=True,
                                 output_dir=save_dir)



def Polynomial_class_analysis(
                              shiptype:ShipType,
                              conventional_df:pd.DataFrame,
                              alternative_df:pd.DataFrame,
                        enable_piecewise:bool=True,
                        fit_mfp:bool=True,
                        piecewise_bins:dict[str,list[str|int]]|None=None,
                        reccomendations: Path = None,
                        variable_dimensions: List[str] = None,
                        target_dimensions: List[str] = None,
                        cv_method: str | None ='stratified',
                        selection_metric:str = 'bic',
                        ranking_att='rrmse',
                        fixed_dimensions: List[str] = ['DWT'],
                        include_regularized: bool = True,
                        use_cv_regularization: bool = True,
                        domain_shift:bool = True,
                        training_shift:bool = False,):

    if variable_dimensions is None:
        variable_dimensions = ['DWT','LBP (M)', 'DRAFT (M)', 'BEAM (M)','DEPTH (M)']

    if target_dimensions is None:
        target_dimensions = variable_dimensions

    stages = 1

    if variable_dimensions is None:
        variable_dimensions = target_dimensions

    ship_type_name = shiptype.name

    # STAGE 1: Find best models (DWT always included)
    print(f"STAGE {stages}: POLYNOMIAL REGRESSION MODEL SELECTION")
    print("=" * 100)
    poly_selector = ParametricModelSelector(training_data=conventional_df,
                                            variable_dimensions=variable_dimensions,
                                            target_dimensions=target_dimensions,
                                            verbose=True,
                                            fixed_dimensions=fixed_dimensions,
                                            selection_metric=selection_metric,
                                            include_regularized=include_regularized,
                                            use_cv_regularization=use_cv_regularization,
                                            enable_piecewise=enable_piecewise,
                                            piecewise_bins=piecewise_bins,
                                            fit_mfp=fit_mfp,
                                            max_piecewise_predictors=1,
                                            min_samples_per_bin=10)

    if enable_piecewise is None and piecewise_bins is not None:
        poly_selector.print_piecewise_summary()

    model_dict = poly_selector.analyze_all_combinations(max_predictors=2, cv_method=cv_method)

    # Export and visualize
    if reccomendations is not None:
        poly_selector.export_recommendations(reccomendations)

    poly_selector.plot_metrics_comparison(
        cv_method=cv_method,
        metric=ranking_att,
        group_by_type=True
    )

    poly_selector.plot_metrics_comparison(
        cv_method=cv_method,
        group_by_type=True
    )

    poly_selector.plot_cv_stability()

    poly_selector.plot_all_single_predictor_regressions(top_n_per_target=6)

    #DEBUG

    poly_selector.compare_selection_criteria(('BEAM (M)', 'DRAFT (M)'), 'DWT')

    # And check what polynomial-2-ridge looks like
    key = (('BEAM (M)', 'DRAFT (M)'), 'DWT')
    if key in poly_selector.detailed_results:
        results = poly_selector.detailed_results[key]
        print("\nPolynomial-2 variants for BEAM + DRAFT → DWT:")
        for model in ['polynomial-2', 'polynomial-2-ridge', 'polynomial-2-lasso']:
            if model in results:
                m = results[model]
                print(f"  {model:20s}: Adj R²={m['adj_r2']:.4f}, "
                      f"BIC={m['bic']:8.1f}, VIF={m.get('max_vif', 1.0):.1f}")

    stages += 1

    if domain_shift:

        print(f"\n\nSTAGE {stages}: POLYNOMIAL DOMAIN SHIFT ANALYSIS")
        print("=" * 100)

        poly_analyzer = ParametricDomainAnalyzer(
            conventional_data=conventional_df,  # Will be split 80/20
            alternative_data=alternative_df,
            independent_variables=variable_dimensions,
            dependent_variables=target_dimensions,
            model_recommendations=model_dict,  # Use best models from step 1
            test_size=0.2,
            random_state=42
        )

        # Step 3: Analyze
        results = poly_analyzer.analyze_all_combinations(max_predictors=2)

        # Step 4: Interpret
        poly_analyzer.interpret_results(significance_threshold_r2=0.05)

        poly_analyzer.plot_domain_shift_analysis()

        poly_analyzer.plot_prediction_scatter()

        poly_analyzer.plot_residuals()

        poly_analyzer.plot_single_predictor_fits()

    if training_shift:

        stages += 1
        print(f"\n\nSTAGE {stages}: POLYNOMIAL TRAINING SHIFT ANALYSIS")
        print("=" * 100)

        comparator = TrainingDomainComparator(domain_analyzer=poly_analyzer,
                                              model_selector=poly_selector)

        # Run comparison
        results = comparator.compare_training_domains(
            max_predictors=2,
            training_sources=['conventional_full', 'conventional_matched', 'alternative']
        )

        # Specific target - average across all predictor combinations for SPEED

        for target in target_dimensions:
            comparator.plot_performance_heatmap(target=target, metric='r2')

        comparator.plot_prediction_lines(target='SPEED (KN)', predictor='DWT')
        comparator.plot_coefficient_comparison(target='GT', predictors=['DWT', 'LOA (M)'])



        # comparer.plot_comprehensive_comparison(
        # )

        # # Export results
        # comparer.export_all_comparisons(base_filename='roro_model_comparison')

def formulae_assessment(conventional_df:pd.DataFrame,
                        alternative_df:pd.DataFrame,
                        target_dimensions:List[str]=None,
                        predictor_dimensions:List[str]=None,
                        shiptype:ShipType = None,
                        residuals: bool = False,
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

    lit_evaluator.plot_comparison()

    lit_evaluator.plot_comparison_by_reference()

    lit_evaluator.plot_comparison_scatter()

    lit_evaluator.plot_formula_fit()

    if residuals:
        lit_evaluator.plot_residuals()

def run_regression_class_testers(filepath:Path|str,
                            shiptype:ShipType=ShipType.BULK_CARRIER,
                            polynomial_shift:bool=True,
                            random_forest_shift:bool=True,
                            target_dimensions:List[str]=['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)'],
                            variable_dimensions:List[str]|None = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'SPEED (KN)']):

    if variable_dimensions is None:
        variable_dimensions = target_dimensions

    ship_type_name = shiptype.name

    print(f'='*60)
    print(('='*20)+ f'RUNNING FULL ANALYSIS FOR {shiptype}'+('='*20))
    print(f'=' * 60)

    print(f'='*60)
    print(('='*20)+ f'STEP 1: Loading data'+('='*20))
    print(f'=' * 60)

    ship_type = shiptype.name

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=ship_type)

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']
    dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']



def  run_regression_analysis(filepath:Path|str,
                            shiptype:ShipType=ShipType.BULK_CARRIER,
                            textbook:bool = False,
                            polynomial_shift:bool=True,
                            random_forest_shift:bool=True,
                            target_dimensions:List[str]=['DWT', 'LOA (M)',],
                            variable_dimensions:List[str]|None = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'SPEED (KN)'],
                            dual_fuel_analysis:bool = False,
                            reccomendations_path:Path = None, ):

    if variable_dimensions is None:
        variable_dimensions = target_dimensions

    ship_type_name = shiptype.name

    bins, variables, filepath = get_variables(shiptype=shiptype,)

    print(f'='*60)
    print(('='*20)+ f'RUNNING FULL ANALYSIS FOR {shiptype}'+('='*20))
    print(f'=' * 60)

    print(f'='*60)
    print(('='*20)+ f'STEP 1: Loading data'+('='*20))
    print(f'=' * 60)



    ship_type = shiptype.name

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=ship_type)

    conventional_df = fleet.processed_data['pure_conventional']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']


    if textbook:
        formulae_assessment(conventional_df=conventional_df,
                            alternative_df=alternative_df,
                            target_dimensions=variables,
                            predictor_dimensions=variables,
                            shiptype=shiptype)

    DomainShiftAnalysis(filepath=filepath,
                        conventional_df=conventional_df,
                        alternative_df=alternative_df,
                        reccomendations=reccomendations_path,
                        variable_dimensions=variables,
                        target_dimensions=variables,
                        cv_method='stratified',
                        include_regularized=False,
                        use_cv_regularization=False,
                        polynomial=polynomial_shift,
                        rf_regression=random_forest_shift
                        )

    if dual_fuel_analysis:
        formulae_assessment(conventional_df=conventional_df,
                            alternative_df=alternative_df,
                            target_dimensions=variables,
                            predictor_dimensions=variables,
                            shiptype=ShipType.BULK_CARRIER)

        DomainShiftAnalysis(filepath=filepath,
                            ship_type=ship_type,
                            conventional_df=conventional_df,
                            alternative_df=dual_fuel_df,
                            reccomendations=reccomendations_path,
                            variable_dimensions=variables,
                            target_dimensions=variables,
                            cv_method='stratified',
                            include_regularized=False,
                            use_cv_regularization=False
                            )

def create_and_store_results(shiptype,
                             experiment1:bool=True,
                             experiment2:bool=True,
                             experiment3:bool=True,):
    bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)

    alt_fuel_sets = ['pure_alt_fuel', 'dual_fuel']

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=shiptype)

    conventional_df = fleet.processed_data['pure_conventional']
    # alternative_df = fleet.processed_data['pure_alt_fuel']

    poly_result, poly_selector,rf_result, rf_selector = selectors(conventional_df=conventional_df,shiptype=shiptype,)

    for alt_fuel_set in alt_fuel_sets:
        alternative_df = fleet.processed_data[alt_fuel_set]

        group_dir = save_dir / alt_fuel_set

        group_dir.mkdir(parents=True, exist_ok=True)

        if experiment1:
            print('Empirical formulae performance assessment')

            formulae_assessment(conventional_df=conventional_df,
                                alternative_df=alternative_df,
                                target_dimensions=variables,
                                predictor_dimensions=variables,
                                shiptype=shiptype)

        if experiment3 or experiment2:

            poly_analyzer,rf_analyzer = analysers(poly_results=poly_result,
                                                  rf_results=rf_result,
                                                  conventional_df=conventional_df,
                                                  alternative_df=alternative_df,
                                                  variables=variables,)

            # poly_selector, poly_analyzer, rf_selector, rf_analyzer = selectors_and_analysers(shiptype=shiptype,
            #                                                                              alternative_df=alternative_df,
            #                                                                              conventional_df=conventional_df,
            #                                                                          rf=True)
        if experiment2:

            experiment2_LaTeX(poly_selector=poly_selector, poly_analyzer=poly_analyzer,
                              rf_selector=rf_selector, rf_analyzer=rf_analyzer, shiptype=shiptype, save_dir=group_dir,
                              alt_type=alt_fuel_set)

            experiment2_plots(poly_selector=poly_selector, poly_analyzer=poly_analyzer, shiptype=shiptype,
                              rf_selector=rf_selector, rf_analyzer=rf_analyzer, save_dir=group_dir)

        if experiment3:
            experiment3_analysis(poly_selector=poly_selector, poly_analyzer=poly_analyzer,
                             rf_selector=rf_selector, rf_analyzer=rf_analyzer, shiptype=shiptype,
                             save_dir=group_dir, conventional_df=conventional_df, alternative_df=alternative_df)



def main(block1:bool=True,
         block2:bool=True,
         block3:bool=True,
         block4=True,
         block5:bool=True):
    #
    # # Bulk carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    # ship_type=ShipType.BULK_CARRIER

    #Tanker
    filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    ship_type = ShipType.TANKER
    #
    #
    # # #Container ship
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    # ship_type=ShipType.CONTAINER_SHIP



    # #RO PAx
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')
    # ship_type=ShipType.RO_PAX

    # kpi_dependent_variables = ['DWT', 'LDT', 'MAIN BUNKER CAPACITY (M3)', 'TOTAL MAIN POWER (KW)', 'TOTAL AUX POWER (KW)', 'TEU', 'SPEED (KN)']
    # kpi_independent_variables = ['DWT', 'LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)','SPEED (KN)', 'SCREWS','MAIN BUNKER CAPACITY (M3)', 'TOTAL MAIN POWER (KW)', 'TOTAL AUX POWER (KW)']

    SD_dependent_variables =  ['DWT', 'LOA (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)']
    SD_independent_variables = ['DWT', 'LOA (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)']

    # run_regression_analysis(filepath=filepath,
    #                         shiptype=ship_type,
    #                         textbook=True,
    #                         polynomial=True,
    #                         random_forest=True,
    #                         target_dimensions=SD_dependent_variables,
    #                         variable_dimensions=SD_independent_variables,
    #                         )




    # kpi_dependent_variables = ['DWT', 'LOA (M)', 'DRAFT (M)']
    # kpi_independent_variables = ['DWT', 'LOA (M)', 'DRAFT (M)', 'BEAM (M)','SPEED (KN)','MAIN BUNKER CAPACITY (M3)']

    # kpi_dependent_variables = ['DWT', 'LOA (M)','LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)','TEU']
    # kpi_independent_variables = ['DWT', 'LOA (M)','LBP (M)','DEPTH (M)','DRAFT (M)', 'BEAM (M)','Vs','TEU']

    kpi_dependent_variables = ['DWT', 'LOA (M)','DEPTH (M)']
    kpi_independent_variables = ['DWT', 'LOA (M)','LBP (M)','Vs','BEAM (M)']

    if block1:

        ship_type=ShipType.BULK_CARRIER

        bins,variables,filepath = get_variables(shiptype=ship_type,)

        run_regression_analysis(filepath=filepath,
                                shiptype=ship_type,
                                textbook=False,
                                polynomial_shift=True,
                                random_forest_shift=True,
                                target_dimensions=variables,
                                variable_dimensions=variables,
                                )



    if block2:
        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=ship_type)

        conventional_df = fleet.processed_data['pure_conventional']
        # conventional_matched_df = fleet.processed_data['matched_dwt_speed']
        alternative_df = fleet.processed_data['pure_alt_fuel']
        dual_fuel_df = fleet.processed_data['dual_fuel']
        # dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']

        # Polynomial_class_analysis(shiptype=ship_type,
        #                           conventional_df=conventional_df,
        #                           conventional_matched_df=conventional_matched_df,
        #                           alternative_df=alternative_df,
        #                           domain_shift=False,
        #                           training_shift=False,
        #                           selection_metric='adj_r2')
        if ship_type==ShipType.TANKER:
            bins = {'DWT': [10000,25000,55000,75000,120000,170000,250000,330000, np.inf]}
            variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD']

        elif ship_type==ShipType.BULK_CARRIER:
            bins = {'DWT':[10000,25000,55000,75000,120000,250000,330000,np.inf]}
            variables = ['DWT', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD']

        elif ship_type==ShipType.CONTAINER_SHIP:
            bins = None
            variables=['DWT','TEU','LBP (M)', 'DRAFT (M)', 'BEAM (M)','DEPTH (M)','LBD']

        else:
            bins = None
            variables = ['DWT', 'TEU', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)', 'LBD','LM','Np','Nv']

        piecewise = False

        Polynomial_class_analysis(shiptype=ship_type,
                                  conventional_df=conventional_df,
                                  # conventional_matched_df=conventional_matched_df,
                                  alternative_df=alternative_df,
                                  enable_piecewise=piecewise,
                                  variable_dimensions=variables,
                                  piecewise_bins=bins,
                                  domain_shift=True,
                                  training_shift=False,
                                  selection_metric='bic')
    if block3:
        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=ship_type)

        conventional_df = fleet.processed_data['pure_conventional']
        conventional_matched_df = fleet.processed_data['matched_dwt_speed']
        alternative_df = fleet.processed_data['pure_alt_fuel']
        dual_fuel_df = fleet.processed_data['dual_fuel']

        kpi_dependent_variables = ['LOA (M)','LBP (M)','DRAFT (M)','DEPTH (M)', 'BEAM (M)','LppBD','LBD','LBT', 'LM','Nv','Np']

        # kpi_dependent_variables = ['DWT','TEU']

        kpi_independent_variables = ['DWT', 'LOA (M)', 'DRAFT (M)', 'BEAM (M)', 'Vs','LBP (M)','TEU',
                                     'MAIN BUNKER CAPACITY (M3)','LM','Nv','Np','TEU']


        formulae_assessment(conventional_df=conventional_df,
                            conventional_matched_df=conventional_matched_df,
                            alternative_df=alternative_df,
                            target_dimensions=kpi_dependent_variables,
                            predictor_dimensions=kpi_independent_variables,
                            shiptype=ship_type,
                            residuals=False)


        kpi_dependent_variables = ['DWT','TEU']
        kpi_independent_variables = ['DWT', 'LOA (M)', 'DRAFT (M)', 'BEAM (M)', 'Vs','LBP (M)','TEU',
                                     'MAIN BUNKER CAPACITY (M3)']


        # formulae_assessment(conventional_df=conventional_df,
        #                     conventional_matched_df=conventional_matched_df,
        #                     alternative_df=alternative_df,
        #                     target_dimensions=kpi_dependent_variables,
        #                     predictor_dimensions=kpi_independent_variables,
        #                     shiptype=ship_type,
        #                     residuals=True)

    if block4:
        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=ship_type)

        conventional_df = fleet.processed_data['pure_conventional']

        alternative_df = fleet.processed_data['pure_alt_fuel']
        dual_fuel_df = fleet.processed_data['dual_fuel']
        dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']

        plot_dimension_by_decade_seaborn(df=conventional_df,
                                         year_col='BUILT',
                                         dim_col='DWT')

        plot_dimension_by_decade_seaborn(df=alternative_df,
                                         year_col='BUILT',
                                         dim_col='DWT')

    if block5:
        shiptype = ShipType.BULK_CARRIER
        # shiptype = ShipType.TANKER
        # shiptype = ShipType.CONTAINER_SHIP
        # shiptype = ShipType.RO_PAX

        fleet_list = [ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.CONTAINER_SHIP,ShipType.RO_PAX]

        create_and_store_results(shiptype=shiptype,
                                 experiment1=False,
                                 experiment2=False,
                                 experiment3=True)
    #     bins, variables, filepath, save_dir = get_variables(shiptype=shiptype)
    #
    #     alt_fuel_sets = ['pure_alt_fuel','dual_fuel']
    #
    #     fleet = getting_alt_fuel_ships(filepath=filepath,
    #                                    ship_type=shiptype)
    #
    #     conventional_df = fleet.processed_data['pure_conventional']
    #     # alternative_df = fleet.processed_data['pure_alt_fuel']
    #
    #     for alt_fuel_set in alt_fuel_sets:
    #         alternative_df = fleet.processed_data[alt_fuel_set]
    #
    #         group_dir = save_dir/alt_fuel_set
    #
    #         group_dir.mkdir(parents=True, exist_ok=True)
    #
    #         poly_selector, poly_analyzer, rf_selector, rf_analyzer = selectors_and_analysers(shiptype=shiptype,
    #                                                                                      conventional_df=conventional_df,
    #                                                                                      alternative_df=alternative_df,                                                                        parametric=True,
    #                                                                                      rf=True)
    #         print('Empirical formulae performance assessment')
    #
    #         formulae_assessment(conventional_df=conventional_df,
    #                         alternative_df=alternative_df,
    #                         target_dimensions=variables,
    #                         predictor_dimensions=variables,
    #                         shiptype=shiptype)
    #
    #         experiment2_LaTeX(poly_selector=poly_selector,poly_analyzer=poly_analyzer,
    #                       rf_selector=rf_selector,rf_analyzer=rf_analyzer,shiptype=shiptype,save_dir=group_dir,
    #                           alt_type=alt_fuel_set)
    #
    #         experiment2_plots(poly_selector=poly_selector,poly_analyzer=poly_analyzer,shiptype=shiptype,
    #                       rf_selector=rf_selector,rf_analyzer=rf_analyzer,save_dir=group_dir)
    #
    #         experiment3_analysis(poly_selector=poly_selector,poly_analyzer=poly_analyzer,
    #                          rf_selector=rf_selector,rf_analyzer=rf_analyzer,shiptype=shiptype,
    #                          save_dir=group_dir,conventional_df=conventional_df,alternative_df=alternative_df)

        # LaTeX_Table_results(shiptype=shiptype,parametric=False,rf=True)


if __name__ == '__main__':
    # main(block1=False,
    #      block2=True)
    main(block1=False,
         block2=False,
         block3=False,
         block4=False,
         block5=True)



