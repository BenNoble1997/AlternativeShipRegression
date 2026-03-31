"""
Parametric regression selector

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict
from tqdm import tqdm

from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
from knowledge_layer.parametric_regression.parametric_model_description_class import ModelDescriptor
from knowledge_layer.parametric_regression.print_parametric_formulas import FormulaPrinter

from knowledge_layer.data_split_handling_class import DataSplitHandler
from knowledge_layer.model_selectors.best_model_selector import BestModelSelector

from knowledge_layer.model_plotting_utility_class import MetricsComparisonPlotter, ModelPlotter
from knowledge_layer.model_selectors.ModelSelectionPlotter import ModelSelectionPlotter
from knowledge_layer.variable_combination_manager_class import VariableCombinationManager
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.model_selectors.Selector_save_class import ResultsExporter
warnings.filterwarnings('ignore')


# ============================================================================
# POLYNOMIAL MODEL SELECTOR WITH PROGRESS BARS
# ============================================================================

class ParametricModelSelector:
    """
    Polynomial model selector with separated processing and display.
    """

    def __init__(self,
                 training_data: pd.DataFrame,
                 variable_dimensions: list[str],
                 target_dimensions: list[str] | None = None,
                 fixed_dimensions: list[str] | None = None,
                 relationship_context: dict[str, list[str]] | None = None,
                 forbidden_combinations: list[tuple[str, ...]] | None = None,
                 #Model selection criterua
                 selection_metric='bic',
                 selection_strategy='conservative',
                 lrt_alpha: float = 0.05,
                 assumptions:str='moderate',
                 vif_threshold:int = 10,
                 include_regularized=True,
                 use_cv_regularization=True,
                 # NEW PARAMETERS for piecewise:
                 enable_piecewise=False,
                 piecewise_bins=None,
                 max_piecewise_predictors=1,
                 min_samples_per_bin=10,
                 fit_mfp:bool=False,

                 verbose=True):
        """
        Polynomial model selector with separated processing and display.

        Parameters
        ----------
        training_data : pd.DataFrame
            Training dataset containing all dimensions
        variable_dimensions : List[str]
            List of potential predictor dimensions
        target_dimensions : List[str] | None, optional
            List of target dimensions. If None, all variable_dimensions are targets.
        fixed_dimensions : List[str] | None, optional
            Dimensions that must be present in every combination (for reference only).
            Note: Currently stored but not enforced in combinations.
        relationship_context : dict[str, List[str]] | None, optional
            Dictionary mapping dimensions to lists of dimensions they should not
            be combined with (e.g., {'Volume': ['Length', 'Width']})
        forbidden_combinations : List[Tuple[str, ...]] | None, optional
            List of specific predictor combinations that should not be tested
        selection_metric : str, default='adj_r2'
                        Metric to use for selecting best model:
                        - 'adj_r2': Adjusted R² (higher is better) - balances fit and complexity
                        - 'r2': R² (higher is better) - raw fit quality
                        - 'aic': Akaike Information Criterion (lower is better) - good for prediction
                        - 'bic': Bayesian Information Criterion (lower is better) - prefers simpler models
                        - 'rrmse': Relative RMSE% (lower is better)
                        - 'mape': Mean Absolute Percentage Error% (lower is better)
        include_regularized : bool, default=True
            Whether to include regularized models (Ridge/Lasso) in analysis.
            If False, these models are automatically added to exclusion list.
        use_cv_regularization : bool, default=True
            If True, uses Ridge-CV/Lasso-CV with automatic alpha selection.
            If False, uses Ridge/Lasso with default alpha values.

        enable_piecewise : bool, default=False
        Whether to test piecewise variants of models alongside global models.
        If True, both 'power' and 'power-piecewise' will be tested.

        piecewise_bins : dict, optional
            Bin specification for piecewise models.
            Format: {variable_name: [bin_edges]}
            Example: {'DWT': [0, 1000, 5000, 10000, 50000, np.inf]}
            Required if enable_piecewise=True.

        max_piecewise_predictors : int, default=1
            Maximum number of predictors for piecewise models.
            Typically 1 since piecewise models bin on a single variable.
            Set to 0 to disable piecewise models entirely.

        min_samples_per_bin : int, default=10
            Minimum samples required per bin for piecewise fitting.
            Bins with fewer samples will be skipped.

        verbose : bool, default=True
            Whether to print progress and results

        Attributes
        ----------
        model_recommendations : dict
            Maps (predictor_combo, target) -> best_model_name
        detailed_results : dict
            Maps (predictor_combo, target) -> {model_name: metrics_dict}

        Notes
        -----

        **Piecewise Models:**
        Piecewise models are common in naval architecture where relationships
        change at different ship sizes. For example, a formula for predicting
        length might use different coefficients for:
        - Small ships (DWT < 5000)
        - Medium ships (5000 ≤ DWT < 20000)
        - Large ships (DWT ≥ 20000)

        Model Selection:
        - All applicable models are tested by default
        - Models are automatically filtered based on data constraints:
          * Non-linear models (exponential, power, etc.) require single predictor
          * Some models require positive X values (logarithmic)
          * Some models require positive Y values (exponential, power)
        - Users can exclude specific models via exclude_models parameter in
          fit_all_models() or analyze_all_combinations()
        - Regularized models can be disabled via include_regularized=False

        Available Models:
        - Linear: 'linear'
        - Regularized: 'ridge', 'lasso', 'ridge-cv', 'lasso-cv'
        - Polynomial: 'polynomial-2', 'polynomial-3'
        - Non-linear (single predictor): 'exponential', 'logarithmic', 'power',
          'exponential-offset', 'power-offset', 'logarithmic-power'"""

        if 'DWT' not in variable_dimensions:
            raise ValueError("DWT must be in variable_dimensions list!")

        self.combination_manager = VariableCombinationManager(
            training_data=training_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            fixed_dimensions=[],
            relationship_context=relationship_context,
            forbidden_combinations=forbidden_combinations,
            verbose=verbose
        )

        if 'DWT' not in self.combination_manager.available_dimensions:
            raise ValueError("DWT must be in the training data!")

        if enable_piecewise and piecewise_bins is None:
            raise ValueError(
                "enable_piecewise=True requires piecewise_bins to be specified. "
                "Example: piecewise_bins={'DWT': [0, 5000, 10000, np.inf]}"
            )

        if enable_piecewise and max_piecewise_predictors < 1:
            if verbose:
                print("ℹ️  enable_piecewise=True but max_piecewise_predictors=0, "
                      "no piecewise models will be tested")

            # Store new attributes
        self.enable_piecewise = enable_piecewise
        self.piecewise_bins = piecewise_bins
        self.max_piecewise_predictors = max_piecewise_predictors
        self.min_samples_per_bin = min_samples_per_bin

        self.data = training_data[self.combination_manager.available_dimensions].copy()
        self.variable_dimensions = self.combination_manager.get_variable_dimensions()
        self.target_dimensions = self.combination_manager.get_target_dimensions()
        self.fixed_dimensions = fixed_dimensions or ['DWT']
        self.relationship_context = self.combination_manager.relationship_context
        self.forbidden_combinations = self.combination_manager.forbidden_combinations
        self.selection_metric = selection_metric
        self.include_regularized = include_regularized
        self.use_cv_regularization = use_cv_regularization
        self.fit_mfp = fit_mfp
        self.verbose = verbose

        self.model_selector = BestModelSelector(
            selection_metric=selection_metric,
            selection_strategy=selection_strategy,
            lrt_alpha=lrt_alpha,
            assumptions=assumptions,
            vif_threshold=vif_threshold,
            verbose=False
        )
        # Storage
        self.model_recommendations = {}
        self.model_descriptors = {}
        self.detailed_results = {}


    # ========================================================================
    # MAIN ANALYSIS METHOD (REFACTORED to use DataSplitHandler)
    # ========================================================================
    def analyze_all_combinations(self,
                                 max_predictors=2,
                                 exclude_models=None,
                                 cv_method='standard',
                                 n_folds=10,
                                 groups=None,
                                 random_state=42,
                                 display_results=True):
        """
        Analyze all predictor combinations with separated processing and display.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors per combination
        exclude_models : list of str, optional
            Model types to exclude from testing.
            Examples: ['power-offset', 'polynomial-3', 'exponential-offset']
            See ModelFactory.get_available_models() for full list.
        cv_method : str
            'standard', 'stratified', 'grouped', or None
        n_folds : int
            Number of CV folds
        groups : array-like, optional
            Group labels for grouped CV
        random_state : int
            Random seed
        display_results : bool
            Whether to display results table (default: True, respects verbose)

        Returns
        -------
        dict
            model_recommendations dictionary

        """


        # ✅ REFACTORED: Use DataSplitHandler for validation
        DataSplitHandler.validate_cv_inputs(cv_method, groups)

        # Optional DISPLAY: Header
        if display_results and self.verbose:
            from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
            display = SelectorResultsDisplay(self)
            display.print_analysis_header(
                max_predictors=max_predictors,
                cv_method=cv_method,
                n_folds=n_folds,
                exclude_models=exclude_models
            )

        if self.selection_metric in ['aic', 'bic'] and cv_method is not None:
            print("Note: AIC/BIC calculated on full training set, not via CV")

        # COMPUTATION ONLY (unchanged)
        for target in self.target_dimensions:
            self._analyze_single_target_silent(
                target=target,
                max_predictors=max_predictors,
                exclude_models=exclude_models,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=groups,
                random_state=random_state
            )

        self._create_descriptors_for_recommendations()

        # Optional DISPLAY: Results
        if display_results and self.verbose:
            self.show_selection_results(cv_method=cv_method)

        return self.model_recommendations

    def show_selection_results(self,
                               cv_method='standard',):
        """
        Function for displaying the results of the selection procedure.
        """

        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_results(cv_method=cv_method)
        display.print_completion_summary()

    def create_missing_descriptors(self, force: bool = False):
        """
        Create ModelDescriptors for recommended models.

        **Use Case:** Run this after loading results from pickle to ensure
        all recommended models have descriptors available for formula generation.

        Parameters
        ----------
        force : bool, default=False
            If True, recreates all descriptors even if they exist.
            If False, only creates missing descriptors.


        """
        if not self.model_recommendations:
            raise ValueError(
                "No model recommendations available. "
                "Run analyze_all_combinations() first."
            )

        if force:
            if self.verbose:
                print("Force=True: Recreating all descriptors...")
            # Clear existing descriptors
            self.model_descriptors = {}

        # Delegate to internal method
        self._create_descriptors_for_recommendations()

        if self.verbose:
            print("\n✅ Descriptor creation complete!")
            print(f"   Total recommendations: {len(self.model_recommendations)}")
            print(f"   Descriptors available: {len(self.model_descriptors)}")


    # ========================================================================
    # TARGET-LEVEL PROCESSING (REFACTORED)
    # ========================================================================

    def _analyze_single_target_silent(self,
                                      target: str,
                                      max_predictors: int,
                                      exclude_models: list[str]|None,
                                      cv_method: str,
                                      n_folds: int,
                                      groups: np.ndarray|None,
                                      random_state: int):
        """Process all combinations for a target with progress bar (no printing)."""

        valid_combos = self.combination_manager.generate_combinations(
            target=target,
            min_predictors=1,
            max_predictors=max_predictors
        )

        if not valid_combos:
            if self.verbose:
                print(f"  ⚠️  Skipping {target}: No available predictors after exclusions")
            return

        total_combos = len(valid_combos)
        if total_combos == 0:
            return

        desc = f"Target: {target[:25]:25s}"
        pbar = tqdm(
            total=total_combos,
            desc=desc,
            disable=not self.verbose,
            unit='combo',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for pred_combo in valid_combos:
            combo_type = self.combination_manager.classify_combination_type(pred_combo)

            self._process_single_combination_silent(
                pred_combo=pred_combo,
                target=target,
                exclude_models=exclude_models,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=groups,
                random_state=random_state,
                combination_type=combo_type
            )
            pbar.update(1)

        pbar.close()

    def _process_single_combination_silent(self,
                                           pred_combo: tuple[str, ...],
                                           target: str,
                                           exclude_models: list[str]|None,
                                           cv_method: str,
                                           n_folds: int,
                                           groups: np.ndarray|None,
                                           random_state: int,
                                           combination_type: str = 'other'):
        """Process a single combination without printing."""

        if not pred_combo or len(pred_combo) == 0:
            return

        # ✅ REFACTORED: Use DataSplitHandler to prepare data
        try:
            X_clean, y_clean, pred_groups = DataSplitHandler.prepare_combination_data(
                data=self.data,
                pred_combo=pred_combo,
                target=target,
                cv_method=cv_method,
                groups=groups
            )
        except ValueError:
            return

        # Fit all applicable models (now with exclude_models parameter)
        try:
            model_results = self.fit_all_models(
                X=X_clean,
                y=y_clean,
                predictor_combo=pred_combo,
                exclude_models=exclude_models,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=pred_groups,
                random_state=random_state,
                target=target
            )

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            if self.verbose and len(pred_combo) == 1:
                print(f"    Single-pred {' + '.join(pred_combo)} → {target}: Fitting failed - {str(e)[:80]}")
            return

        # Store results (NO PRINTING)
        if model_results:
            best_name, best_metrics = self._select_best_model(model_results)

            key = (tuple(pred_combo), target)
            self.model_recommendations[key] = best_name

            # Calculate VIF for this combination (once for all models)
            if len(pred_combo) > 1:
                X = X_clean
                _, vif_info = self.combination_manager._check_vif_combination(pred_combo)
                max_vif = vif_info['max_vif']
                vif_values = vif_info['vif_df'].set_index('feature')['vif'].to_dict()
            else:
                max_vif = 1.0
                vif_values = {pred_combo[0]: 1.0}


            for model_name in model_results:
                model_results[model_name]['combination_type'] = combination_type
                model_results[model_name]['max_vif'] = max_vif
                model_results[model_name]['vif_values'] = vif_values


            self.detailed_results[key] = model_results


    # ========================================================================
    # MODEL FITTING (REFACTORED to use DataSplitHandler)
    # ========================================================================
    def fit_all_models(self,
                       X,
                       y,
                       predictor_combo: tuple|list[str],
                       target:str,
                       exclude_models=None, cv_method=None,
                       n_folds=10, groups=None, random_state=42):
        """
        Fit all applicable models to the data.

        This method now dynamically discovers which models can be applied
        to the given data using ModelFactory.get_applicable_models().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictor variables
        y : array-like, shape (n_samples,)
            Target variable
        exclude_models : list of str, optional
            Model types to exclude from testing.
            Examples: ['power-offset', 'polynomial-3']
        cv_method : str, optional
            Cross-validation method: 'standard', 'stratified', 'grouped', or None
        n_folds : int, default=10
            Number of cross-validation folds
        groups : array-like, optional
            Group labels for grouped CV
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary mapping model names to their metrics:
            {
                'linear': {'r2': 0.85, 'adj_r2': 0.84, ...},
                'polynomial-2': {'r2': 0.92, 'adj_r2': 0.91, ...},
                ...
            }

        Notes
        -----
        - Models are automatically filtered based on data constraints
        - Non-linear models (exponential, power, etc.) only tested for single predictors
        - Models requiring positive values only tested when data satisfies requirements
        - User can exclude specific models via exclude_models parameter
        - Regularized models (Ridge/Lasso) can be disabled via include_regularized attribute

        """
        EXPECTED_MODEL_ERRORS = (ValueError, RuntimeError, RuntimeWarning)
        results = {}

        # ====================================================================
        # Data Preparation
        # ====================================================================

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Clean data
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if groups is not None:
            groups = groups[mask]

        # Check minimum samples
        if len(X) < 5:
            if self.verbose:
                print(f"  ⚠️  Insufficient data: {len(X)} samples (minimum 5 required)")
            return results

        n_features = X.shape[1]

        # ====================================================================
        # Determine Which Models to Test
        # ====================================================================

        # Initialize exclusion list
        if exclude_models is None:
            exclude_models = []
        else:
            exclude_models = list(exclude_models)  # Make a copy to avoid modifying user's list

        # Add regularized models to exclusion list if disabled
        if not self.include_regularized:
            regularized_models = ['ridge', 'lasso', 'ridge-cv', 'lasso-cv',
                # Regularized polynomial variants
                'polynomial-2-ridge', 'polynomial-2-lasso',
                'polynomial-3-ridge', 'polynomial-3-lasso'
            ]
            exclude_models.extend(regularized_models)

        # Get models that are applicable to this data
        applicable_models = ModelFactory.get_applicable_models(
            X=X, y=y,
            exclude_models=exclude_models,
            clean_first=False,  # We already cleaned above
            min_samples=5
        )

        predictor_names = list(predictor_combo)

        # NEW: Add piecewise variants if enabled and appropriate
        if self.enable_piecewise and n_features <= self.max_piecewise_predictors:
            piecewise_models = self._get_applicable_piecewise_models(
                X, y, exclude_models, predictor_names
            )
            applicable_models.extend(piecewise_models)

        if self.fit_mfp:
            applicable_models.append('mfp')

        # ====================================================================
        # Verbose Output: Only Warn if No Applicable Models
        # ====================================================================

        if self.verbose and len(applicable_models) == 0:
            print(f"  ⚠️  No applicable models found for this data")

            # Show why models were excluded (helpful for debugging)
            inapplicable = ModelFactory.get_inapplicable_models_with_reasons(
                X=X, y=y, exclude_models=exclude_models, clean_first=False
            )
            if inapplicable:
                print(f"  Reasons:")
                for model_type, reason in list(inapplicable.items())[:5]:  # Show first 5
                    print(f"    • {model_type}: {reason}")
                if len(inapplicable) > 5:
                    print(f"    • ... and {len(inapplicable) - 5} more")

        # ====================================================================
        # Setup Cross-Validation (✅ REFACTORED to use DataSplitHandler)
        # ====================================================================

        cv_config = DataSplitHandler.setup_cv_config(
            X=X,
            y=y,
            cv_method=cv_method,
            n_folds=n_folds,
            groups=groups,
            random_state=random_state
        )

        # ====================================================================
        # Fit Each Applicable Model
        # ====================================================================

        for model_type in applicable_models:

            try:

                if model_type.endswith('-piecewise'):

                    results[model_type] = self._fit_piecewise_model(
                        X, y, model_type, cv_config, predictor_names
                    )

                else:
                    # Determine actual feature count for this model type

                    # Fit the model (with or without CV)
                    results[model_type] = self._fit_single_model(
                        X=X, y=y,predictor_combo=predictor_combo,
                        model_type=model_type, cv_config=cv_config,
                        target=target
                    )

            except EXPECTED_MODEL_ERRORS as e:
                # Even if ModelFactory says it's applicable, fitting can still fail
                # (e.g., convergence issues, numerical instability)
                if self.verbose:
                    print(f"  ⚠️  {model_type} fitting failed: {str(e)[:60]}")
                continue

        # ====================================================================
        # Return Results
        # ====================================================================

        return results

    def _fit_single_model(self, X, y,predictor_combo,target, model_type, cv_config):
        """Fit single model using ModelFactory."""

        # ✅ Determine actual feature count based on model type
        # actual_n_features = ModelFactory.get_actual_n_features(model_type=model_type, X=X)

        is_structure_discovering = self._is_structure_discovering_model(model_type)

        if cv_config['method'] is None:
            model, preprocessor = ModelFactory.create_and_fit(model_type=model_type, X=X, y=y)
            y_pred = ModelFactory.predict(model, preprocessor, X, model_type)

            actual_n_features = ModelFactory.get_n_parameters(
                model_type=model_type, X=X, fitted_model=model  # ✅ Pass fitted model
            )

            metric_calculator = RegressionMetrics(y_pred, y)
            metrics = metric_calculator.calculate_all_metrics(actual_n_features)
            het_scedast_result= metric_calculator.comprehensive_heteroscedasticity_analysis(X=X,feature_names=predictor_combo)
            metrics['spearman'] = het_scedast_result['spearman']['overall']['heteroscedastic']
            metrics['breusch_pagan'] = het_scedast_result['breusch_pagan']['heteroscedastic']
            metrics['white'] = het_scedast_result['white']['heteroscedastic']
            metrics['heteroscedasticity'] = het_scedast_result['summary']['conclusion']

            descriptor = ModelDescriptor.from_fitted_model(
                model=model, preprocessor=preprocessor, model_type=model_type,
                independent_vars=list(predictor_combo),dependent_var= target,  # Need target from outer scope
                metrics=metrics
            )

            key = (tuple(predictor_combo), target)
            self.model_descriptors[key] = {model_type: descriptor}


            if hasattr(model, 'selected_alpha_'):
                metrics['selected_alpha'] = model.selected_alpha_
            elif hasattr(model, 'alpha'):
                metrics['selected_alpha'] = model.alpha

            return metrics

        if is_structure_discovering:
            return self._fit_with_cv_fixed_structure(
                X, y, predictor_combo, model_type, cv_config,target
            )

        else:
            return self._fit_with_cv_standard(
                X, y, predictor_combo, model_type, cv_config,target
            )


    def _is_structure_discovering_model(self, model_type: str) -> bool:
        """
        Check if model discovers structure during fitting.

        Returns
        -------
        bool
            True if model selects features or transformations
        """
        structure_discovering = [
            'mfp',  # Discovers transformations
            'polynomial-2-lasso',  # Selects polynomial terms
            'polynomial-3-lasso',
            'polynomial-2-lasso-cv',
            'polynomial-3-lasso-cv',
        ]

        return model_type in structure_discovering

    def _fit_with_cv_fixed_structure(self, X, y, predictor_combo,
                                     model_type, cv_config,target):
        """
        CV for structure-discovering models (MFP, polynomial-lasso).

        **Two-Phase Approach:**
        1. Discover structure on full training data
        2. Test with FIXED structure across CV folds

        **Concept:**
        We want to know: "How well does THIS specific model generalize?"
        Not: "How well do different random models generalize?"
        """

        # ========================================================================
        # PHASE 1: Discover structure on full training data
        # ========================================================================

        if self.verbose:
            print(f"  Phase 1: Discovering structure for {model_type}...")

        # Fit on full data
        model_full, prep_full = ModelFactory.create_and_fit(model_type=model_type, X=X, y=y)

        # Extract structure via descriptor
        descriptor = ModelDescriptor.from_fitted_model(
            model=model_full, preprocessor=prep_full, model_type=model_type,
            independent_vars=list(predictor_combo), dependent_var=target  # Need target from outer scope
        )

        # Get actual parameter count from discovered structure
        actual_n_features = descriptor.get_effective_parameters()

        if self.verbose:
            print(f"    Discovered structure: {actual_n_features} effective parameters")
            if model_type == 'mfp':
                # Show count of transformations instead of details
                n_transforms = sum(1 for t in descriptor.transformations.values() if t[0] != 'excluded')
                print(f"    {n_transforms} variables with transformations")
            elif 'lasso' in model_type:
                if descriptor.active_features:
                    print(f"    Active features: {len(descriptor.active_features)}/{len(descriptor.coefficients)}")

        # ========================================================================
        # PHASE 2: CV with FIXED structure
        # ========================================================================

        if self.verbose:
            print(f"  Phase 2: Testing stability with fixed structure...")

        cv_metrics = []
        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args, **split_kwargs)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # ✅ Refit with FIXED structure from Phase 1
                model, prep = descriptor.refit(
                    X=X_train,y= y_train,
                    force_structure=True  # KEY: Force same structure
                )

                y_pred = ModelFactory.predict(model, prep, X_test, model_type)

                # Calculate metrics with FIXED parameter count
                metric_calculator = RegressionMetrics(y_pred, y_test)
                metrics = metric_calculator.calculate_predictive_metrics(actual_n_features)
                cv_metrics.append(metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"    Fold {fold_idx + 1} failed: {e}")
                continue

        if not cv_metrics:
            raise ValueError(f"All CV folds failed for {model_type}")

        # Average CV metrics
        avg_metrics = self._average_metrics(cv_metrics)

        # ========================================================================
        # PHASE 3: Information criteria on full data
        # ========================================================================

        # Refit on full data for AIC/BIC
        y_pred_full = ModelFactory.predict(model_full, prep_full, X, model_type)

        metric_calculator_full = RegressionMetrics(y, y_pred_full)
        ic_metrics = metric_calculator_full.calculate_information_criteria(actual_n_features)


        het_scedast_result = metric_calculator_full.comprehensive_heteroscedasticity_analysis(X=X,
                                                                                         feature_names=predictor_combo)
        ic_metrics['spearman'] = het_scedast_result['spearman']['overall']['heteroscedastic']
        ic_metrics['breusch_pagan'] = het_scedast_result['breusch_pagan']['heteroscedastic']
        ic_metrics['white'] = het_scedast_result['white']['heteroscedastic']
        ic_metrics['heteroscedasticity'] = het_scedast_result['summary']['conclusion']

        # Combine
        final_metrics = {**avg_metrics, **ic_metrics}

        # ✅ Store descriptor for this model
        key = (tuple(predictor_combo), target)
        if key not in self.model_descriptors:
            self.model_descriptors[key] = {}
        self.model_descriptors[key][model_type] = descriptor

        return final_metrics

    def _fit_with_cv_standard(self, X, y, predictor_combo, model_type, cv_config,target):
        """
        Standard CV for fixed-structure models (linear, polynomial, etc.).

        **Concept:**
        Structure doesn't change across folds, so standard CV is appropriate.
        """
        cv_metrics = []
        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for train_idx, test_idx in splitter.split(*split_args, **split_kwargs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model, prep = ModelFactory.create_and_fit(model_type, X_train, y_train)
                y_pred = ModelFactory.predict(model, prep, X_test, model_type)

                # For polynomial-lasso, get actual parameters from fitted model
                if 'lasso' in model_type and 'polynomial' in model_type:
                    actual_n_features = ModelFactory.get_n_parameters(
                        model_type, X_train, fitted_model=model
                    )
                else:
                    actual_n_features = ModelFactory.get_n_parameters(
                        model_type, X_train
                    )

                metric_calculator = RegressionMetrics(y_pred, y_test)
                metrics = metric_calculator.calculate_predictive_metrics(actual_n_features)
                cv_metrics.append(metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if not cv_metrics:
            raise ValueError(f"All CV folds failed for {model_type}")

        avg_metrics = self._average_metrics(cv_metrics)

        # Fit on full data for AIC/BIC and descriptor storage
        model_full, prep_full = ModelFactory.create_and_fit(model_type, X, y)
        y_pred_full = ModelFactory.predict(model_full, prep_full, X, model_type)

        actual_n_features = ModelFactory.get_n_parameters(
            model_type, X, fitted_model=model_full
        )

        metric_calculator_full = RegressionMetrics(y, y_pred_full)
        ic_metrics = metric_calculator_full.calculate_information_criteria(actual_n_features)

        het_scedast_result = metric_calculator_full.comprehensive_heteroscedasticity_analysis(X=X,
                                                                                              feature_names=predictor_combo)
        ic_metrics['spearman'] = het_scedast_result['spearman']['overall']['heteroscedastic']
        ic_metrics['breusch_pagan'] = het_scedast_result['breusch_pagan']['heteroscedastic']
        ic_metrics['white'] = het_scedast_result['white']['heteroscedastic']
        ic_metrics['heteroscedasticity'] = het_scedast_result['summary']['conclusion']

        # ✅ Create and store descriptor
        descriptor = ModelDescriptor.from_fitted_model(
            model_full, prep_full, model_type,
            list(predictor_combo), target,
            metrics=ic_metrics
        )

        key = (tuple(predictor_combo), target)
        if key not in self.model_descriptors:
            self.model_descriptors[key] = {}
        self.model_descriptors[key][model_type] = descriptor

        final_metrics = {**avg_metrics, **ic_metrics}
        return final_metrics


    def _fit_with_cv(self, X, y,predictor_combo, actual_n_features, model_type, cv_config):
        """Fit model with cross-validation using correct feature count."""
        cv_metrics = []
        selected_alphas = []

        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for train_idx,test_idx in splitter.split(*split_args, **split_kwargs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model, preprocessor = ModelFactory.create_and_fit(model_type, X_train, y_train)
                y_pred = ModelFactory.predict(model, preprocessor, X_test, model_type)

                #  Use pre-computed actual feature count
                metric_calculator = RegressionMetrics(y_pred, y_test)
                metrics = metric_calculator.calculate_predictive_metrics(actual_n_features)
                cv_metrics.append(metrics)

                if hasattr(model, 'selected_alpha_'):
                    selected_alphas.append(model.selected_alpha_)
                elif hasattr(model, 'alpha'):
                    selected_alphas.append(model.alpha)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if not cv_metrics:
            raise ValueError(f"All CV folds failed for model type: {model_type}")

        avg_metrics = self._average_metrics(cv_metrics)

        # PART 2: Separate full-data fit for AIC/BIC
        model_full, preprocessor_full = ModelFactory.create_and_fit(model_type, X, y)
        y_pred_full = ModelFactory.predict(model_full, preprocessor_full, X, model_type)

        metric_calculator_full = RegressionMetrics(y, y_pred_full)
        ic_metrics = metric_calculator_full.calculate_information_criteria(actual_n_features)
        het_scedast_result = metric_calculator_full.comprehensive_heteroscedasticity_analysis(X=X,
                                                                                         feature_names=predictor_combo)

        # Combine
        final_metrics = {**avg_metrics, **ic_metrics}


        final_metrics['spearman'] = het_scedast_result['spearman']['overall']['heteroscedastic']
        final_metrics['breusch_pagan'] = het_scedast_result['breusch_pagan']['heteroscedastic']
        final_metrics['white'] = het_scedast_result['white']['heteroscedastic']
        final_metrics['heteroscedasticity'] = het_scedast_result['summary']['conclusion']

        if selected_alphas:
            final_metrics['selected_alpha'] = np.mean(selected_alphas)
            final_metrics['std_alpha'] = np.std(selected_alphas)

        return final_metrics

    def _get_applicable_piecewise_models(self, X, y, exclude_models, predictor_names):
        """
        Determine which piecewise model variants are applicable.

        Parameters
        ----------
        X : np.ndarray
            Predictor data
        y : np.ndarray
            Target data
        exclude_models : list
            Models to exclude
        predictor_names : list of str
            Names of predictors

        Returns
        -------
        list of str
            Applicable piecewise model types
        """
        if predictor_names is None:
            # Can't do piecewise without knowing variable names
            return []

        n_features = X.shape[1]

        # Only single predictor piecewise supported currently
        if n_features != 1:
            return []

        # Check if binning variable is in our bins dict
        binning_var = predictor_names[0]
        if binning_var not in self.piecewise_bins:
            return []

        # Get base models that can be piecewise
        base_models = ModelFactory.PIECEWISE_ALLOWED_BASE_MODELS

        # Check applicability of each
        applicable_piecewise = []
        for base_model in base_models:
            piecewise_model = f"{base_model}-piecewise"

            # Skip if explicitly excluded
            if piecewise_model in exclude_models:
                continue

            # Check if base model is applicable
            is_applicable, _ = ModelFactory.is_model_applicable(
                base_model, X, y, clean_first=False, min_samples=5
            )

            if is_applicable:
                applicable_piecewise.append(piecewise_model)

        return applicable_piecewise

    def _fit_piecewise_model(self, X, y, model_type, cv_config, predictor_names):
        """
        Fit a piecewise model with optional cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Predictor data
        y : np.ndarray
            Target data
        model_type : str
            Piecewise model type (e.g., 'power-piecewise')
        cv_config : dict
            CV configuration from DataSplitHandler
        predictor_names : list of str
            Names of predictors

        Returns
        -------
        dict
            Metrics for the piecewise model
        """
        binning_var = predictor_names[0]  # Single predictor assumption

        if cv_config['method'] is None:
            # No CV - fit once on full data
            model, preprocessor = ModelFactory.create_and_fit(
                model_type, X, y,
                bins=self.piecewise_bins,
                binning_variable=binning_var
            )

            y_pred = ModelFactory.predict(model, preprocessor, X, model_type)

            # Get actual n_features (total parameters across bins)
            actual_n_features = model.get_n_parameters()

            metric_calculator = RegressionMetrics(y, y_pred)
            metrics = metric_calculator.calculate_all_metrics(actual_n_features)

            # Add piecewise-specific info
            metrics['piecewise_diagnostics'] = model.get_bin_diagnostics()

            return metrics

        else:
            # With CV
            return self._fit_piecewise_with_cv(
                X, y, model_type, cv_config, binning_var
            )

    def _fit_piecewise_with_cv(self, X, y, model_type, cv_config, binning_var):
        """
        Fit piecewise model with cross-validation.

        Similar to _fit_with_cv() but handles piecewise specifics.
        """
        cv_metrics = []

        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for train_idx, test_idx in splitter.split(*split_args, **split_kwargs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model, preprocessor = ModelFactory.create_and_fit(
                    model_type, X_train, y_train,
                    bins=self.piecewise_bins,
                    binning_variable=binning_var
                )

                y_pred = ModelFactory.predict(model, preprocessor, X_test, model_type)

                actual_n_features = model.get_n_parameters()

                metric_calculator = RegressionMetrics(y_test, y_pred)
                metrics = metric_calculator.calculate_predictive_metrics(actual_n_features)
                cv_metrics.append(metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if not cv_metrics:
            raise ValueError(f"All CV folds failed for model type: {model_type}")

        avg_metrics = self._average_metrics(cv_metrics)

        # Fit on full data for AIC/BIC
        model_full, _ = ModelFactory.create_and_fit(
            model_type, X, y,
            bins=self.piecewise_bins,
            binning_variable=binning_var
        )

        y_pred_full = ModelFactory.predict(model_full, None, X, model_type)
        actual_n_features = model_full.get_n_parameters()

        metric_calculator_full = RegressionMetrics(y, y_pred_full)
        ic_metrics = metric_calculator_full.calculate_information_criteria(actual_n_features)

        final_metrics = {**avg_metrics, **ic_metrics}

        # Add piecewise diagnostics
        final_metrics['piecewise_diagnostics'] = model_full.get_bin_diagnostics()

        return final_metrics

    def _select_best_model(self, model_results):
        """Delegate to BestModelSelector."""
        return self.model_selector.select_best_model(model_results)

    def _average_metrics(self, cv_metrics: list[dict]) -> dict:
        """Average metrics across CV folds."""
        avg_metrics = {}
        std_metrics = {}

        for key in cv_metrics[0].keys():
            if key in ['n_samples', 'n_features']:
                avg_metrics[key] = cv_metrics[0][key]
            else:
                values = [m[key] for m in cv_metrics if not np.isnan(m[key])]
                if values:
                    avg_metrics[key] = np.mean(values)
                    std_metrics[key] = np.std(values)
                else:
                    avg_metrics[key] = np.nan
                    std_metrics[key] = np.nan

        avg_metrics['std_r2'] = std_metrics.get('r2', np.nan)
        avg_metrics['std_adj_r2'] = std_metrics.get('adj_r2', np.nan)
        avg_metrics['std_rrmse'] = std_metrics.get('rrmse', np.nan)

        return avg_metrics

    """
    Add to ParametricModelSelector class
    """

    def get_best_combinations(self,
                              selection_metric: str|None = None,
                              max_vif: float = 10.0,
                              penalize_heteroscedasticity: bool = True,
                              heteroscedasticity_penalty: float = 0.05,
                              prefer_simpler: bool = True,
                              equivalence_threshold: float|None = None) -> dict[
        str, tuple[tuple[str, ...], str, dict]]:
        """
        Get best predictor combination for each target using BestModelSelector logic.

        **Delegates to BestModelSelector.select_best_combination()** for consistency.

        Parameters
        ----------
        selection_metric : str, optional
            Metric to optimize. If None, uses selector's default.
        max_vif : float, default=10.0
            Maximum acceptable VIF
        penalize_heteroscedasticity : bool, default=True
            Whether to penalize heteroscedastic models
        heteroscedasticity_penalty : float, default=0.05
            Penalty magnitude (0.05 = 5%)
        prefer_simpler : bool, default=True
            Prefer fewer predictors when equivalent
        equivalence_threshold : float, optional
            Threshold for equivalence (None = use defaults)

        Returns
        -------
        dict
            {target: (predictors_tuple, model_type, metrics_dict)}

        """

        metric = selection_metric or self.selection_metric

        if not self.model_recommendations:
            raise ValueError(
                "No recommendations available. Run analyze_all_combinations() first."
            )

        best_combinations = {}

        for target in self.target_dimensions:
            if self.verbose:
                print(f"\nSelecting best combination for {target}...")

            # Get all candidates for this target
            candidates = self._get_candidate_combinations(target)

            if not candidates:
                if self.verbose:
                    print(f"  ⚠️  No candidates for {target}")
                continue

            if self.verbose:
                print(f"  Evaluating {len(candidates)} combinations")

            # Delegate to BestModelSelector
            best = self.model_selector.select_best_combination(
                candidates=candidates,
                selection_metric=metric,
                max_vif=max_vif,
                penalize_heteroscedasticity=penalize_heteroscedasticity,
                heteroscedasticity_penalty=heteroscedasticity_penalty,
                prefer_simpler=prefer_simpler,
                equivalence_threshold=equivalence_threshold
            )

            if best is not None:
                best_combinations[target] = (
                    best['predictors'],
                    best['model_type'],
                    best['metrics']
                )

                if self.verbose:
                    print(f"  ✓ Selected: {' + '.join(best['predictors'])} → {best['model_type']}")
                    print(f"    {metric.upper()}: {best['metrics'][metric]:.4f}")
                    print(f"    Max VIF: {best['metrics'].get('max_vif', 1.0):.2f}")
                    print(f"    Het: {best['metrics'].get('heteroscedasticity', 'unknown')}")

        return best_combinations

    def _get_candidate_combinations(self, target: str) -> list[dict]:
        """
        Get all candidate combinations for a target.

        **Format expected by BestModelSelector:**
        Each dict must contain:
        - 'predictors': tuple of predictor names
        - 'model_type': recommended model type
        - 'metrics': full metrics dictionary
        - 'n_predictors': number of predictors

        Returns
        -------
        list of dict
            Candidate combinations in BestModelSelector format
        """
        candidates = []

        for (predictors, tgt), model_type in self.model_recommendations.items():
            if tgt != target:
                continue

            key = (predictors, tgt)

            if key not in self.detailed_results:
                continue

            model_results = self.detailed_results[key]

            if model_type not in model_results:
                continue

            metrics = model_results[model_type]

            candidates.append({
                'predictors': predictors,
                'model_type': model_type,
                'metrics': metrics,
                'n_predictors': len(predictors)
            })

        return candidates

    # Keep the export and print methods as before
    def export_best_combinations(self,
                                 filename: str = 'best_combinations.csv',
                                 **kwargs) -> pd.DataFrame:
        """Export best combinations to CSV."""

        best_combos = self.get_best_combinations(**kwargs)

        export_data = []

        for target, (predictors, model_type, metrics) in best_combos.items():
            row = {
                'target': target,
                'predictors': ' + '.join(predictors),
                'n_predictors': len(predictors),
                'model_type': model_type,
                'r2': metrics['r2'],
                'adj_r2': metrics['adj_r2'],
                'rrmse': metrics['rrmse'],
                'mape': metrics['mape'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'aic': metrics.get('aic', np.nan),
                'bic': metrics.get('bic', np.nan),
                'max_vif': metrics.get('max_vif', 1.0),
                'heteroscedasticity': metrics.get('heteroscedasticity', 'unknown'),
                'spearman_het': metrics.get('spearman', False),
                'breusch_pagan_het': metrics.get('breusch_pagan', False),
                'white_het': metrics.get('white', False),
                'n_samples': metrics['n_samples'],
                'n_features': metrics['n_features']
            }

            # Add individual VIF values
            vif_values = metrics.get('vif_values', {})
            for pred in predictors:
                row[f'vif_{pred}'] = vif_values.get(pred, np.nan)

            export_data.append(row)

        best_df = pd.DataFrame(export_data)
        best_df.to_csv(filename, index=False)

        if self.verbose:
            print(f"\n✓ Best combinations exported to {filename}")
            print(f"  {len(best_df)} targets")

        return best_df

    def print_best_combinations_summary(self, **kwargs):
        """Print formatted summary of best combinations."""

        best_combos = self.get_best_combinations(**kwargs)

        print("\n" + "=" * 140)
        print("BEST COMBINATIONS PER TARGET (VIF and Heteroscedasticity Filtered)")
        print("=" * 140)

        print(f"\n{'Target':<15s} | {'Predictors':<25s} | {'Model':<15s} | "
              f"{'Adj R²':>8s} | {'RRMSE%':>8s} | {'Max VIF':>8s} | {'Het':>12s}")
        print("-" * 140)

        for target, (predictors, model_type, metrics) in best_combos.items():
            preds_str = ' + '.join(predictors)
            het_status = metrics.get('heteroscedasticity', 'unknown')

            # ✅ FIXED: Check for 'Strong'/'Weak' instead of 'Yes'
            if het_status == 'No':
                het_emoji = "✓"
            elif het_status in ['Strong', 'Weak']:
                het_emoji = "⚠️"
            else:
                het_emoji = "?"

            print(f"{target:<15s} | {preds_str:<25s} | {model_type:<15s} | "
                  f"{metrics['adj_r2']:>8.4f} | {metrics['rrmse']:>8.2f} | "
                  f"{metrics.get('max_vif', 1.0):>8.2f} | {het_emoji} {het_status:<10s}")

        print("=" * 140)

        # Summary statistics
        total = len(best_combos)

        # ✅ FIXED: Count 'Strong'/'Weak' instead of 'Yes'
        het_count = sum(1 for _, (_, _, m) in best_combos.items()
                        if m.get('heteroscedasticity') in ['Strong', 'Weak'])

        high_vif_count = sum(1 for _, (_, _, m) in best_combos.items()
                             if m.get('max_vif', 1.0) > 5.0)

        print(f"\nSummary:")
        print(f"  Total targets:                    {total}")
        print(f"  With heteroscedasticity:          {het_count} ({het_count / total * 100:.1f}%)")
        print(f"  With VIF > 5:                     {high_vif_count} ({high_vif_count / total * 100:.1f}%)")

        avg_r2 = np.mean([m['adj_r2'] for _, (_, _, m) in best_combos.items()])
        avg_rrmse = np.mean([m['rrmse'] for _, (_, _, m) in best_combos.items()])
        avg_vif = np.mean([m.get('max_vif', 1.0) for _, (_, _, m) in best_combos.items()])

        print(f"\nAverage Performance:")
        print(f"  Adj R²:    {avg_r2:.4f}")
        print(f"  RRMSE:     {avg_rrmse:.2f}%")
        print(f"  Max VIF:   {avg_vif:.2f}")


    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def export_recommendations(self, filename='model_recommendations.csv'):
        """Export recommendations with all metrics."""
        export_data = []
        for (predictors, target), model in self.model_recommendations.items():
            model_scores = self.detailed_results[(predictors, target)][model]
            export_data.append({
                'Predictors': ' + '.join(predictors),
                'Target': target,
                'Best_Model': model,
                'R2': model_scores['r2'],
                'Adj_R2': model_scores['adj_r2'],
                'RRMSE_%': model_scores['rrmse'],
                'MAPE_%': model_scores['mape'],
                'RMSE': model_scores['rmse'],
                'MAE': model_scores['mae'],
                # ✅ Add Spearman diagnostics
                'Spearman_Rho': model_scores.get('spearman_rho', np.nan),
                'Heteroscedasticity': model_scores.get('heteroscedasticity', 'unknown'),
                # VIF diagnostics
                'Max_VIF': model_scores.get('max_vif', np.nan),
                'Selection_Metric': self.selection_metric,
                'N_Features': model_scores['n_features'],
                'N_Samples': model_scores['n_samples']
            })

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"\n✓ Enhanced recommendations exported to {filename}")
        return df

    def print_initialization_summary(self):
        """Convenience method to display initialization info."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_initialization_summary()

    def print_results(self, **kwargs):
        """Convenience method to display results."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_results(**kwargs)

    def print_best_combinations(self):
        """Convenience method to display best combinations summary."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_best_combinations_summary()


    def plot_metrics_comparison(self,
                                max_predictors: int = 2,
                                title: str | None = None,
                                cv_method: str = None,
                                metric: str = 'rrmse',
                                ship_type: str|None = None,
                                variable_name_map: dict[str,str]|None = None,
                                targets: list[str] = None,
                                include_only_vars: list[str]|None = None,
                                group_by_type: bool = False,
                                figsize_per_plot: tuple = (8, 6),
                                save_path: str = None):
        """
        Plot comparison of best models for all target dimensions.

        **Enhanced with ship type and variable name formatting.**

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors to include in plot
        title : str, optional
            Custom title. If None, uses default.
        cv_method : str, optional
            CV method used (for title)
        metric : str
            Metric to plot ('adj_r2', 'rrmse', 'mape', 'r2')
        ship_type : str, optional
            Ship type to include in title, e.g., 'Container Ships'
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        targets : list of str, optional
            List of specific targets to plot. If None, plots all targets.
            Example: ['LOA (M)', 'Beam (M)']
        include_only_vars : list of str, optional
            Only include combinations that use ONLY variables from this list.
            All predictors in a combination must be in this list.
            Example: ['DWT', 'GT', 'LOA (M)'] will exclude combinations
            containing 'Beam (M)' or other variables.
        group_by_type : bool
            If True, creates separate plots for each combination type
        figsize_per_plot : tuple
            Size for each individual subplot
        save_path : str, optional
            Path to save the figure

        """
        if group_by_type:
            self._plot_metrics_by_type(
                max_predictors, metric, cv_method,
                figsize_per_plot, save_path,
                ship_type=ship_type,
                variable_name_map=variable_name_map,
                targets=targets,
                include_only_vars=include_only_vars
            )
        else:
            results_by_target = self._prepare_plot_data(
                max_predictors=max_predictors,
                metric=metric,
                variable_name_map=variable_name_map,
                targets=targets,
                include_only_vars=include_only_vars
            )

            if not results_by_target:
                print("No results to plot after filtering!")
                return

            if title is None:
                title = 'Model Performance Comparison Across Dimensions'

            MetricsComparisonPlotter.plot_metrics_comparison(
                results_by_target=results_by_target,
                metric=metric,
                cv_method=cv_method,
                ship_type=ship_type,
                variable_name_map=variable_name_map,
                figsize_per_plot=figsize_per_plot,
                title=title,
                save_path=save_path,
                verbose=self.verbose
            )

    def _plot_metrics_by_type(self,
                              max_predictors: int,
                              metric: str,
                              cv_method: str,
                              figsize_per_plot: tuple,
                              save_path: str = None,
                              ship_type: str|None = None,
                              variable_name_map: dict[str,str]|None = None,
                              targets: list[str]|None = None,
                              include_only_vars: list[str]|None = None):
        """Create separate full figures for each combination type."""
        # Prepare data grouped by type
        data_by_type = {}

        # Filter targets if specified
        targets_to_plot = targets if targets else self.target_dimensions

        for (pred_combo, target), model_name in self.model_recommendations.items():
            # Filter by target
            if target not in targets_to_plot:
                continue

            # Filter by max_predictors
            if len(pred_combo) > max_predictors:
                continue

            # Filter by include_only_vars
            if include_only_vars is not None:
                if not all(var in include_only_vars for var in pred_combo):
                    continue

            model_results = self.detailed_results[(pred_combo, target)]
            metrics = model_results[model_name]
            combo_type = metrics.get('combination_type', 'other')

            if combo_type not in data_by_type:
                data_by_type[combo_type] = {}

            if target not in data_by_type[combo_type]:
                data_by_type[combo_type][target] = []

            # Format predictor names if mapping provided
            if variable_name_map:
                predictors_formatted = [variable_name_map.get(p, p) for p in pred_combo]
                predictors_str = " + ".join(predictors_formatted)
            else:
                predictors_str = " + ".join(pred_combo)

            label = f"{predictors_str}\n({model_name})"

            data_by_type[combo_type][target].append({
                'label': label,
                'metric_value': metrics[metric],
                'std_value': metrics.get(f'std_{metric}', 0),
                'n_predictors': len(pred_combo),
                'model_name': model_name
            })

        if not data_by_type:
            print("No data to plot after filtering")
            return

        # Create a separate figure for each combination type
        type_order = ['hull', 'tonnage', 'propulsion', 'mixed', 'other']

        for combo_type in type_order:
            if combo_type not in data_by_type:
                continue

            type_data = data_by_type[combo_type]

            # Build title for this type
            type_title = f"Model Performance - {combo_type.upper()} Combinations"

            # Use the shared plotter for consistency
            MetricsComparisonPlotter.plot_metrics_comparison(
                results_by_target=type_data,
                metric=metric,
                cv_method=cv_method,
                ship_type=ship_type,
                variable_name_map=variable_name_map,
                figsize_per_plot=figsize_per_plot,
                title=type_title,
                save_path=save_path.replace('.png', f'_{combo_type}.png') if save_path else None,
                verbose=self.verbose
            )

    def _prepare_plot_data(self,
                           max_predictors: int,
                           metric: str,
                           filter_by_type: str|None = None,
                           variable_name_map: dict[str,str]|None = None,
                           targets: list[str]|None = None,
                           include_only_vars: list[str]|None = None) -> dict[str, list[dict]]:
        """
        Transform internal data structure to format expected by plotter.

        **Key Concept**: Don't regenerate combinations - use what was actually fitted!
        This ensures plots match analysis results exactly, respecting all filters.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors
        metric : str
            Metric to extract
        filter_by_type : str, optional
            Filter by combination type
        variable_name_map : dict, optional
            Variable name formatting
        targets : list of str, optional
            Specific targets to include. If None, uses all targets.
        include_only_vars : list of str, optional
            Only include combinations where ALL predictors are in this list.

        Returns
        -------
        dict
            {target: [result_dicts]}
        """
        results_by_target = {}

        # Determine which targets to plot
        if targets is not None:
            # Validate that specified targets exist
            invalid_targets = [t for t in targets if t not in self.target_dimensions]
            if invalid_targets:
                if self.verbose:
                    print(f"Warning: These targets not found: {invalid_targets}")
            targets_to_plot = [t for t in targets if t in self.target_dimensions]
        else:
            targets_to_plot = self.target_dimensions

        if not targets_to_plot:
            if self.verbose:
                print("No valid target dimensions to plot!")
            return results_by_target

        # ✅ Iterate through fitted combinations directly
        for target in targets_to_plot:
            target_results = []

            # Get all combinations that were actually fitted for this target
            for (pred_combo, tgt), best_model_name in self.model_recommendations.items():
                if tgt != target:
                    continue

                # Filter by number of predictors
                if len(pred_combo) > max_predictors:
                    continue

                # ✅ NEW: Filter by include_only_vars
                if include_only_vars is not None:
                    # Check if ALL predictors in combo are in include_only_vars list
                    if not all(var in include_only_vars for var in pred_combo):
                        continue

                # Get results
                if (pred_combo, target) not in self.detailed_results:
                    continue

                model_results = self.detailed_results[(pred_combo, target)]

                if best_model_name not in model_results:
                    continue

                metrics = model_results[best_model_name]

                # Filter by type if requested
                if filter_by_type and metrics.get('combination_type') != filter_by_type:
                    continue

                metric_value = metrics.get(metric, np.nan)
                std_value = metrics.get(f'std_{metric}', 0)

                # Format predictor names if mapping provided
                if variable_name_map:
                    predictors_formatted = [variable_name_map.get(p, p) for p in pred_combo]
                    predictors_str = " + ".join(predictors_formatted)
                else:
                    predictors_str = " + ".join(pred_combo)

                label = f"{predictors_str}\n({best_model_name})"

                target_results.append({
                    'label': label,
                    'metric_value': metric_value,
                    'std_value': std_value,
                    'n_predictors': len(pred_combo),
                    'model_name': best_model_name,
                    'predictors': pred_combo
                })

            # Sort by number of predictors, then by metric value
            target_results.sort(key=lambda x: (x['n_predictors'], -x['metric_value']))

            if target_results:
                results_by_target[target] = target_results

        return results_by_target


    def plot_cv_stability(self,
                          max_predictors: int = 2,
                          figsize_per_plot: tuple = (10, 6),
                          save_path: str = None):
        """
        Plot CV standard deviation vs mean performance for all dimensions.
        Shows model stability across CV folds.
        """
        import math
        import matplotlib.pyplot as plt

        target_dimensions = self.target_dimensions

        n_plots = len(target_dimensions)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)

        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        fig.suptitle('Cross-Validation Stability Analysis',
                     fontsize=16, fontweight='bold')

        for idx, target in enumerate(target_dimensions):
            ax = axes[idx]
            self._plot_cv_stability_single(ax, target, max_predictors)

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ CV stability plot saved to {save_path}")

        plt.show()

    def _plot_cv_stability_single(self, ax, target: str, max_predictors: int):
        """Plot CV stability for single dimension."""
        from itertools import combinations

        stability_data = []

        available_predictors = [d for d in self.variable_dimensions if d != target]

        if target in self.relationship_context:
            excluded_dims = self.relationship_context[target]
            available_predictors = [d for d in available_predictors if d not in excluded_dims]

        for n_predictors in range(1, min(max_predictors + 1, len(available_predictors) + 1)):

            pred_combos = list(combinations(available_predictors, n_predictors))

            for pred_combo in pred_combos:
                key = (tuple(pred_combo), target)

                if key in self.model_recommendations and key in self.detailed_results:
                    best_model = self.model_recommendations[key]
                    metrics = self.detailed_results[key][best_model]

                    mean_r2 = metrics.get('adj_r2', np.nan)
                    std_r2 = metrics.get('std_adj_r2', 0)

                    if not np.isnan(mean_r2) and std_r2 > 0:
                        stability_data.append({
                            'predictors': ' + '.join(pred_combo),
                            'model': best_model,
                            'mean_r2': mean_r2,
                            'std_r2': std_r2,
                            'n_predictors': len(pred_combo)
                        })

        if not stability_data:
            ax.text(0.5, 0.5, f'No CV results for {target}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target, fontweight='bold')
            return

        means = [d['mean_r2'] for d in stability_data]
        stds = [d['std_r2'] for d in stability_data]
        colors = plt.cm.viridis([d['n_predictors'] / max(max_predictors, 1)
                                 for d in stability_data])

        scatter = ax.scatter(means, stds, c=colors, s=100, alpha=0.6, edgecolors='black')

        for d in stability_data:
            ax.annotate(d['model'][:3], (d['mean_r2'], d['std_r2']),
                        fontsize=7, ha='center')

        ax.set_xlabel('Mean Adjusted R²', fontsize=10)
        ax.set_ylabel('Std Dev of Adjusted R²', fontsize=10)
        ax.set_title(f'Target: {target}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.axhspan(0, ax.get_ylim()[1] * 0.3, alpha=0.1, color='green',
                   label='Low variance zone')

        ax.legend(fontsize=8)

    def plot_single_predictor_regressions(self,
                                          target: str,
                                          top_n: int|None = None,
                                          figsize_per_plot: tuple[float, float] = (6, 5),
                                          sort_by: str = 'adj_r2',
                                          color_by: str|None = 'auto',
                                          save_path: str|None = None):
        """
        Plot scatter plots with regression lines for single-predictor models.

        Each subplot shows one predictor vs the target, with an optional third
        dimension represented by color.

        Parameters
        ----------
        target : str
            Target dimension to plot
        top_n : int, optional
            Limit to top N predictors. If None, shows all single-variable models.
        figsize_per_plot : tuple
            Size for each subplot (width, height)
        sort_by : str
            Metric to sort by ('adj_r2', 'rrmse', 'mape'). Default: 'adj_r2'
        color_by : str or None, optional
            Dimension to use for color scale.
            - 'auto': automatically selects from priority list
            - None: no color dimension (2D plot)
            - str: specific dimension name
            Default: 'auto'
        save_path : str, optional
            Path to save the figure
        """

        # Determine color dimension
        color_dim = self._determine_color_dimension(color_by, target)

        if color_by == 'auto' and color_dim is None:
            if self.verbose:
                print(f"⚠️  Warning: No suitable color dimension found. Creating 2D plots.")

        # Filter for single-predictor models for this target
        single_pred_results = []

        for (pred_combo, tgt), model_name in self.model_recommendations.items():
            if tgt != target or len(pred_combo) != 1:
                continue

            predictor = pred_combo[0]
            metrics = self.detailed_results[(pred_combo, tgt)][model_name]

            single_pred_results.append({
                'predictor': predictor,
                'model_name': model_name,
                'metrics': metrics,
                'pred_combo': pred_combo
            })

        if not single_pred_results:
            print(f"No single-predictor regression results found for target '{target}'")
            return

        # Sort results
        if sort_by == 'adj_r2':
            single_pred_results.sort(key=lambda x: x['metrics']['adj_r2'], reverse=True)
        elif sort_by == 'rrmse':
            single_pred_results.sort(key=lambda x: x['metrics']['rrmse'])
        elif sort_by == 'mape':
            single_pred_results.sort(key=lambda x: x['metrics']['mape'])

        # Limit to top_n if specified
        if top_n is not None:
            single_pred_results = single_pred_results[:top_n]
            title_suffix = f" (Top {top_n})"
        else:
            title_suffix = ""

        n_plots = len(single_pred_results)

        if n_plots == 0:
            print(f"No results to plot for target '{target}'")
            return

        # Calculate layout
        rows, cols = ModelPlotter.calculate_grid_layout(n_plots, max_cols=3)

        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        # Handle single plot case
        if n_plots == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        fig.suptitle(f'Single-Predictor Regressions for {target}{title_suffix}',
                     fontsize=14, fontweight='bold', y=0.995)

        # Create shared colorbar if using color dimension
        if color_dim:
            # Get color data for normalization
            color_data = self.data[color_dim].values
            color_data_clean = color_data[~np.isnan(color_data)]

            if len(color_data_clean) > 0:
                vmin, vmax = color_data_clean.min(), color_data_clean.max()
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])

                # Add colorbar to figure
                cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                                    pad=0.02, aspect=30)
                cbar.set_label(color_dim, fontsize=11, fontweight='bold')
            else:
                color_dim = None  # Fallback if no valid color data

        # Plot each predictor
        for idx, result in enumerate(single_pred_results):
            ax = axes[idx]
            self._plot_single_regression(ax, result, target, color_dim)

        # Hide unused subplots
        ModelPlotter.hide_extra_subplots(axes, n_plots)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Scatter plot saved to {save_path}")

        plt.show()

    def _plot_single_regression(self, ax, result: dict, target: str, color_dim: str|None = None):
        """
        Plot scatter and regression line for a single predictor-target pair.

        Parameters
        ----------
        ax : matplotlib Axes
            Subplot axes
        result : dict
            Dictionary containing predictor, model_name, and metrics
        target : str
            Target variable name
        color_dim : str, optional
            Dimension to use for color scale
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        predictor = result['predictor']
        model_name = result['model_name']
        metrics = result['metrics']

        # Get data
        X = self.data[[predictor]].values
        y = self.data[target].values

        # Get color data if specified
        if color_dim:
            color_data = self.data[color_dim].values
        else:
            color_data = None

        # Clean data
        if color_dim:
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(color_data))
            X_clean = X[mask]
            y_clean = y[mask]
            color_clean = color_data[mask]
        else:
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            color_clean = None

        if len(X_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{predictor}\n({model_name})', fontweight='bold')
            return

        # Fit model to get predictions
        try:
            model, preprocessor = ModelFactory.create_and_fit(model_name, X_clean, y_clean)

            # Create smooth line for predictions
            X_range = np.linspace(X_clean.min(), X_clean.max(), 200).reshape(-1, 1)
            y_pred_line = ModelFactory.predict(model, preprocessor, X_range, model_name)

            # Get predictions for actual points (for residuals if needed)
            y_pred_points = ModelFactory.predict(model, preprocessor, X_clean, model_name)

        except Exception as e:
            ax.text(0.5, 0.5, f'Model fitting failed\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{predictor}\n({model_name})', fontweight='bold')
            return

        # Plot scatter points
        if color_dim and color_clean is not None:
            scatter = ax.scatter(X_clean, y_clean, c=color_clean,
                                 cmap='viridis', alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5,
                                 label='Data points')
        else:
            scatter = ax.scatter(X_clean, y_clean, alpha=0.5, s=30,
                                 color='steelblue', edgecolors='black', linewidth=0.5,
                                 label='Data points')

        # Plot regression line
        ax.plot(X_range, y_pred_line, 'r-', linewidth=2,
                label=f'{model_name}', alpha=0.8)

        # Format axes
        ax.set_xlabel(predictor, fontsize=9)
        ax.set_ylabel(target, fontsize=9)

        # Title with metrics
        title = f'{predictor}\n({model_name})'
        subtitle = f'Adj R² = {metrics["adj_r2"]:.3f}, RRMSE = {metrics["rrmse"]:.1f}%'
        ax.set_title(f'{title}\n{subtitle}', fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    def _determine_color_dimension(self, color_by: str|None, target: str) ->str|None:
        """
        Determine which dimension to use for color scale.

        Parameters
        ----------
        color_by : str or None
            User specification: 'auto', None, or specific dimension name
        target : str
            Target dimension (to avoid using as color)

        Returns
        -------
        str or None
            Dimension to use for color, or None if no suitable dimension
        """
        # If explicitly set to None, return None
        if color_by is None:
            return None

        # If specific dimension requested
        if color_by != 'auto':
            if color_by in self.data.columns and color_by != target:
                return color_by
            else:
                if self.verbose:
                    print(f"⚠️  Warning: Requested color dimension '{color_by}' not available or is the target.")
                return None

        # Auto-select from priority list
        priority_list = ['DWT', 'GT', 'TEU', 'LOA (M)', 'LBP (M)']

        for dim in priority_list:
            if dim in self.data.columns and dim != target:
                # Check if dimension has valid data
                if not self.data[dim].isna().all():
                    return dim

        # No suitable dimension found
        return None

    def plot_all_single_predictor_regressions(self,
                                              top_n_per_target: int|None = None,
                                              figsize_per_plot: tuple[float, float] = (6, 5),
                                              sort_by: str = 'adj_r2',
                                              color_by: str|None = 'auto',
                                              save_path_template: str|None = None):
        """
        Plot scatter plots for all targets.

        Parameters
        ----------
        top_n_per_target : int, optional
            Limit to top N predictors per target
        figsize_per_plot : tuple
            Size for each subplot
        sort_by : str
            Metric to sort by ('adj_r2', 'rrmse', 'mape')
        color_by : str or None, optional
            Dimension to use for color scale. Default: 'auto'
        save_path_template : str, optional
            Template for save paths. Use {target} as placeholder.
            Example: 'plots/regression_{target}.png'
        """
        for target in self.target_dimensions:
            if save_path_template:
                save_path = save_path_template.format(target=target.replace(' ', '_'))
            else:
                save_path = None

            # Call the single target method
            self.plot_single_predictor_regressions(
                target=target,
                top_n=top_n_per_target,
                figsize_per_plot=figsize_per_plot,
                sort_by=sort_by,
                color_by=color_by,
                save_path=save_path
            )

    # ========================================================================
    # CONVENIENCE METHODS FOR MODEL DISCOVERY
    # ========================================================================

    @staticmethod
    def get_available_models():
        """
        Get list of all available model types.

        Returns
        -------
        list of str
            All model types that can be created by ModelFactory

        """
        return ModelFactory.get_available_models()

    @staticmethod
    def get_model_descriptions():
        """
        Get descriptions of all available model types.

        Returns
        -------
        dict : {model_type: description}
            Human-readable descriptions of each model

        Examples
        --------
        >>> descriptions = ParametricModelSelector.get_model_descriptions()
        >>> for model_type, desc in descriptions.items():
        ...     print(f"{model_type}: {desc}")
        """
        models = ModelFactory.get_available_models()
        return {m: ModelFactory.get_model_description(m) for m in models}

    def show_applicable_models_for_combination(self, pred_combo, target):
        """
        Show which models are applicable for a specific predictor-target combination.

        Useful for understanding why certain models are or aren't being tested.

        Parameters
        ----------
        pred_combo : tuple of str
            Predictor combination, e.g., ('DWT', 'LOA')
        target : str
            Target dimension

        """
        # Get data for this combination
        try:
            X = self.data[list(pred_combo)].values
            y = self.data[target].values
        except KeyError as e:
            print(f"❌ Error: {e}")
            return

        # Get applicable and inapplicable models
        applicable = ModelFactory.get_applicable_models(X, y)
        inapplicable = ModelFactory.get_inapplicable_models_with_reasons(X, y)

        # Print results
        predictors_str = " + ".join(pred_combo)
        print(f"\nPredictor: {predictors_str} → Target: {target}")
        print("─" * 60)

        # Applicable models
        print(f"✓ Applicable models ({len(applicable)}):")
        if applicable:
            # Group by category for readability
            linear_models = [m for m in applicable if m in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']]
            poly_models = [m for m in applicable if m.startswith('polynomial')]
            nonlinear_models = [m for m in applicable if m not in linear_models and m not in poly_models]

            if linear_models:
                print(f"  • Linear/Regularized: {', '.join(linear_models)}")
            if poly_models:
                print(f"  • Polynomial: {', '.join(poly_models)}")
            if nonlinear_models:
                print(f"  • Non-linear: {', '.join(nonlinear_models)}")
        else:
            print("  (none)")

        print()

        # Inapplicable models
        print(f"✗ Inapplicable models ({len(inapplicable)}):")
        if inapplicable:
            for model_type, reason in inapplicable.items():
                print(f"  • {model_type}: {reason}")
        else:
            print("  (none)")

        print()

    def summarize_model_testing_strategy(self, max_predictors=2, exclude_models=None):
        """
        Summarize what models will be tested across all combinations.

        Provides an overview of the analysis strategy before running it.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors per combination
        exclude_models : list of str, optional
            Models to exclude

        """
        print("\nModel Testing Strategy Summary")
        print("─" * 60)

        # Basic info
        n_targets = len(self.target_dimensions)
        print(f"Targets: {n_targets} ({', '.join(self.target_dimensions)})")
        print(f"Max predictors: {max_predictors}")

        # Estimate combinations
        n_variable_dims = len(self.variable_dimensions)
        from math import comb
        estimated_combos = sum(comb(n_variable_dims - 1, k) for k in range(1, max_predictors + 1))
        estimated_combos *= n_targets
        print(f"Estimated combinations: ~{estimated_combos}")

        # Model info
        all_models = ModelFactory.get_available_models()

        if exclude_models:
            all_models = [m for m in all_models if m not in exclude_models]

        if not self.include_regularized:
            all_models = [m for m in all_models if m not in ['ridge', 'lasso', 'ridge-cv', 'lasso-cv']]

        print(f"\nAvailable model types: {len(all_models)}")

        # Categorize models
        linear_models = [m for m in all_models if m in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']]
        poly_models = [m for m in all_models if m.startswith('polynomial')]
        nonlinear_models = [m for m in all_models if m not in linear_models and m not in poly_models]

        print("\nModel categories:")
        print(f"  • Linear/Regularized: {len(linear_models)} types")
        if linear_models:
            print(f"    ({', '.join(linear_models)})")

        print(f"  • Polynomial: {len(poly_models)} types")
        if poly_models:
            print(f"    ({', '.join(poly_models)})")

        print(f"  • Non-linear: {len(nonlinear_models)} types (single predictor only)")
        if nonlinear_models:
            print(f"    ({', '.join(nonlinear_models)})")

        # Estimate total fits
        # Multi-predictor combos get ~8 models (no non-linear)
        # Single-predictor combos get ~14 models (all applicable)
        estimated_fits_low = estimated_combos * 6
        estimated_fits_high = estimated_combos * 12
        print(f"\nEstimated total model fits: {estimated_fits_low}-{estimated_fits_high}")

        if exclude_models:
            print(f"\nExcluded models ({len(exclude_models)}):")
            excluded_str = ", ".join(exclude_models[:10])
            if len(exclude_models) > 10:
                excluded_str += f" ... (+{len(exclude_models) - 10} more)"
            print(f"  {excluded_str}")

        print()

    def compare_selection_criteria(self, pred_combo, target):
        """
        Compare how different selection criteria would rank models for a specific combination.

        Useful for understanding why certain models are selected and how criteria differ.

        Parameters
        ----------
        pred_combo : tuple of str
            Predictor combination, e.g., ('DWT', 'LOA')
        target : str
            Target dimension

        Examples
        --------
        >>> selector.compare_selection_criteria(('DWT',), 'LOA (M)')
        """
        key = (tuple(pred_combo), target)

        if key not in self.detailed_results:
            print(f"❌ No results found for {' + '.join(pred_combo)} → {target}")
            return

        model_results = self.detailed_results[key]

        predictors_str = " + ".join(pred_combo)
        print(f"\n{'=' * 60}")
        print(f"Comparison of Selection Criteria")
        print(f"Predictor: {predictors_str} → Target: {target}")
        print(f"{'=' * 60}\n")

        # Rank by each metric
        rankings = {
            'adj_r2': sorted(model_results.items(), key=lambda x: x[1]['adj_r2'], reverse=True),
            'aic': sorted([(k, v) for k, v in model_results.items() if not np.isnan(v.get('aic', np.nan))],
                          key=lambda x: x[1]['aic']),
            'bic': sorted([(k, v) for k, v in model_results.items() if not np.isnan(v.get('bic', np.nan))],
                          key=lambda x: x[1]['bic']),
            'rrmse': sorted([(k, v) for k, v in model_results.items() if not np.isnan(v.get('rrmse', np.nan))],
                            key=lambda x: x[1]['rrmse'])
        }

        print("Model Rankings by Different Criteria:\n")

        # Print Adj R² rankings
        print("By Adj R² (higher is better):")
        for rank, (model_name, metrics) in enumerate(rankings['adj_r2'][:5], 1):
            marker = "👑" if rank == 1 else "  "
            print(f"{marker} {rank}. {model_name:20s}: {metrics['adj_r2']:.4f}")

        print("\nBy AIC (lower is better):")
        adj_r2_best = rankings['adj_r2'][0][0] if rankings['adj_r2'] else None
        for rank, (model_name, metrics) in enumerate(rankings['aic'][:5], 1):
            marker = "👑" if rank == 1 else "  "
            same = " ✓ SAME as Adj R²" if model_name == adj_r2_best else ""
            print(f"{marker} {rank}. {model_name:20s}: {metrics['aic']:8.1f}{same}")

        print("\nBy BIC (lower is better):")
        for rank, (model_name, metrics) in enumerate(rankings['bic'][:5], 1):
            marker = "👑" if rank == 1 else "  "
            same = " ✓ SAME as Adj R²" if model_name == adj_r2_best else ""
            print(f"{marker} {rank}. {model_name:20s}: {metrics['bic']:8.1f}{same}")

        print("\nBy RRMSE% (lower is better):")
        for rank, (model_name, metrics) in enumerate(rankings['rrmse'][:5], 1):
            marker = "👑" if rank == 1 else "  "
            same = " ✓ SAME as Adj R²" if model_name == adj_r2_best else ""
            print(f"{marker} {rank}. {model_name:20s}: {metrics['rrmse']:8.2f}{same}")

        # Check if all criteria agree
        print(f"\n{'-' * 60}")
        aic_best = rankings['aic'][0][0] if rankings['aic'] else None
        bic_best = rankings['bic'][0][0] if rankings['bic'] else None
        rrmse_best = rankings['rrmse'][0][0] if rankings['rrmse'] else None

        if adj_r2_best == aic_best == bic_best:
            print(f"✓ Adj R², AIC, and BIC all agree: {adj_r2_best} is best")
        else:
            print(f"⚠️  Criteria disagree:")
            print(f"   Adj R² prefers: {adj_r2_best}")
            print(f"   AIC prefers   : {aic_best}")
            print(f"   BIC prefers   : {bic_best}")
            print(f"   RRMSE prefers : {rrmse_best}")

    def print_piecewise_summary(self):
        """Print summary of piecewise model configuration."""
        if not self.enable_piecewise:
            print("ℹ️  Piecewise models: Disabled")
            return

        print("\n" + "=" * 70)
        print("PIECEWISE MODEL CONFIGURATION")
        print("=" * 70)

        print(f"\nEnabled: Yes")
        print(f"Max predictors for piecewise: {self.max_piecewise_predictors}")
        print(f"Min samples per bin: {self.min_samples_per_bin}")

        print(f"\nBinning specification:")
        for var, edges in self.piecewise_bins.items():
            print(f"  {var}: {edges}")
            n_bins = len(edges) - 1
            print(f"    → {n_bins} bins")

        print(f"\nAllowed base models:")
        for model in ModelFactory.PIECEWISE_ALLOWED_BASE_MODELS:
            print(f"  • {model} → {model}-piecewise")

        print("=" * 70 + "\n")

    # ========================================================================
    # RESULTS EXPORT/IMPORT (Thin wrappers to ResultsExporter)
    # ========================================================================

    def save_results(self,
                     filepath: str,
                     format: str = 'auto',
                     include_metadata: bool = True,
                     compress: bool = False) -> None:
        """
        Save analysis results.

        Parameters
        ----------
        filepath : str
            Path to save results
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto'
        include_metadata : bool
            Whether to save configuration metadata
        compress : bool
            Whether to compress pickle files


        """

        exporter = ResultsExporter(self)
        exporter.save(filepath, format=format,
                      include_metadata=include_metadata,
                      compress=compress)

    def load_results(self,
                     filepath: str,
                     format: str = 'auto') -> None:
        """
        Load previously saved results.

        Parameters
        ----------
        filepath : str
            Path to saved results
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto'

        Examples
        --------
        >>> selector.load_results('results/models.pkl')
        """

        exporter = ResultsExporter(self,skip_validation=True)
        exporter.load(filepath=filepath, format=format)

    def quick_save(self,
                   name: str = 'results',
                   output_dir: str = 'results') -> None:
        """
        Quick save with both pickle and Excel.

        Parameters
        ----------
        name : str
            Base name for files
        output_dir : str
            Directory to save in

        Examples
        --------
        >>> selector.quick_save('ship_models_v1')
        """

        exporter = ResultsExporter(self)
        exporter.quick_save(name, output_dir)

    @classmethod
    def from_saved_results(cls,
                           results_filepath: str,
                           training_data: pd.DataFrame,
                           create_descriptors:bool=True,
                           **kwargs):
        """
        Create selector and load previous results.

        Parameters
        ----------
        results_filepath : str
            Path to saved results
        training_data : pd.DataFrame
            Training data (needed for analysis/plotting)
        **kwargs
            Arguments for selector initialization

        Returns
        -------
        selector instance with loaded results

        Examples
        --------
        >>> selector = ParametricModelSelector.from_saved_results(
        ...     'results/models.pkl',
        ...     training_data=data,
        ...     variable_dimensions=['DWT', 'GT', ...]
        ... )
        """
        instance = cls(training_data=training_data, **kwargs)
        instance.load_results(filepath=results_filepath)

        if create_descriptors:
            if instance.verbose:
                print("\nCreating missing descriptors for formula generation...")
            instance.create_missing_descriptors()

        return instance

    def print_formulas_for_target(self,
                                  target: str,
                                  max_predictors: int = 2,
                                  notation: str = 'math',
                                  sort_by: str = 'adj_r2',
                                  top_n: int|None = None) -> None:

        return FormulaPrinter.print_formulas_for_target(target=target,
                                                        model_recommendations=self.model_recommendations,
                                                        model_descriptors=self.model_descriptors,
                                                        detailed_results=self.detailed_results,
                                                        max_predictors=max_predictors,
                                                        notation=notation,
                                                        sort_by=sort_by,
                                                        top_n=top_n
                                                        )

    def print_all_formulas(self,
                           max_predictors: int = 2,
                           notation: str = 'math',
                           sort_by: str = 'adj_r2',
                           top_n_per_target: int|None = None,
                           save_to_file: str|None = None) -> None:

        return FormulaPrinter.print_all_formulas(target_dimensions=self.target_dimensions,
                                                 model_descriptors=self.model_descriptors,
                                                 model_reccomendations=self.model_recommendations,
                                                 detailed_results=self.detailed_results,
                                                 max_predictors=max_predictors,
                                                 notation=notation,
                                                 sort_by=sort_by,
                                                 top_n_per_target=top_n_per_target,
                                                 save_to_file=save_to_file)

    def print_best_formulas(self,
                            notation: str = 'math',
                            include_metrics: bool = True) -> None:
        """
        Print formulas for the best model for each target.

        Shows only the recommended model for each target variable.

        Parameters
        ----------
        notation : str, default='math'
            Formula notation style
        include_metrics : bool, default=True
            Whether to show performance metrics

        Examples
        --------
        >>> # Show best formulas
        >>> selector.print_best_formulas()

        >>> # LaTeX notation without metrics
        >>> selector.print_best_formulas(
        ...     notation='latex',
        ...     include_metrics=False
        ... )
        """

        return FormulaPrinter.print_best_formulas(target_dimensions=self.target_dimensions,
                                                 model_descriptors=self.model_descriptors,
                                                 model_recommendations=self.model_recommendations,
                                                 detailed_results=self.detailed_results,
                                                selection_metric=self.selection_metric,
                                                  notation=notation,
                                                  include_metrics=include_metrics,)

    def export_formulas_to_csv(self,
                               filename: str = 'model_formulas.csv',
                               max_predictors: int = 2,
                               notation: str = 'math') -> pd.DataFrame:

        return FormulaPrinter.export_formulas_to_csv(
                                                     max_predictors=max_predictors,
                                                 model_descriptors=self.model_descriptors,
                                                 model_recommendations=self.model_recommendations,
                                                 detailed_results=self.detailed_results,
                                                filename=filename,
                                                notation=notation,)

    def _create_descriptors_for_recommendations(self):
        """
        Create descriptors for all recommended models.

        Called after selection is complete to ensure formula generation works.
        This ensures every recommended model has a ModelDescriptor available.

        **Use Cases:**
        - Automatically called at end of analyze_all_combinations()
        - Can be called manually on loaded results: selector.create_missing_descriptors()
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("Creating descriptors for recommended models...")
            print("=" * 70)

        n_created = 0
        n_skipped = 0
        n_failed = 0

        for key, best_model_type in self.model_recommendations.items():
            predictor_combo, target = key

            # Check if descriptor already exists
            if key in self.model_descriptors:
                if best_model_type in self.model_descriptors[key]:
                    n_skipped += 1
                    continue  # Already have it

            # ✅ FIX: Convert tuple to list for DataFrame indexing
            predictor_list = list(predictor_combo)

            # Need to create descriptor
            # Get data
            try:
                data_subset = self.data[predictor_list + [target]].dropna()
            except KeyError as e:
                n_failed += 1
                if self.verbose:
                    print(f"  ⚠️  Column not found for {predictor_combo} → {target}: {e}")
                continue

            if len(data_subset) < 20:
                n_failed += 1
                if self.verbose:
                    print(f"  ⚠️  Insufficient data for {predictor_combo} → {target} (n={len(data_subset)})")
                continue

            # ✅ FIX: Use list for indexing
            X = data_subset[predictor_list].values
            y = data_subset[target].values

            # Clean
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]

            if len(X_clean) < 20:
                n_failed += 1
                if self.verbose:
                    print(f"  ⚠️  Insufficient clean data for {predictor_combo} → {target} (n={len(X_clean)})")
                continue

            # Get metrics
            if key not in self.detailed_results:
                n_failed += 1
                if self.verbose:
                    print(f"  ⚠️  No detailed results for {predictor_combo} → {target}")
                continue

            if best_model_type not in self.detailed_results[key]:
                n_failed += 1
                if self.verbose:
                    print(f"  ⚠️  Model type {best_model_type} not in results for {predictor_combo} → {target}")
                continue

            metrics = self.detailed_results[key][best_model_type]

            # Create descriptor
            try:
                from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
                from knowledge_layer.parametric_regression.parametric_model_description_class import ModelDescriptor

                model, preprocessor = ModelFactory.create_and_fit(
                    best_model_type, X_clean, y_clean
                )

                descriptor = ModelDescriptor.from_fitted_model(
                    model=model,
                    preprocessor=preprocessor,
                    model_type=best_model_type,
                    independent_vars=predictor_list,  # ✅ Use list
                    dependent_var=target,
                    metrics=metrics
                )

                if key not in self.model_descriptors:
                    self.model_descriptors[key] = {}

                self.model_descriptors[key][best_model_type] = descriptor
                n_created += 1

            except Exception as e:
                if self.verbose:
                    print(
                        f"  ⚠️  Failed to create descriptor for {predictor_combo} → {target} ({best_model_type}): {e}")
                n_failed += 1

        if self.verbose:
            print(f"\n✅ Descriptor creation summary:")
            print(f"   Created: {n_created}")
            print(f"   Skipped (already exist): {n_skipped}")
            if n_failed > 0:
                print(f"   Failed: {n_failed}")
            print("=" * 70 + "\n")


    def plot_model_selection(self,
                             predictor_combo: tuple[str, ...],
                             target: str,
                             metric: str|None = None,
                             ship_type: str|None = None,
                             variable_name_map: dict[str,str]|None = None,
                             highlight_selected: bool = True,
                             show_values: bool = True,
                             figsize: tuple[float, float] = (10, 6),
                             save_path: str|None = None,
                             show_hetero:bool=True):
        """
        Plot comparison of all models tested for a specific combination.

        Shows why a particular model was selected by displaying all tested
        models and their performance on the selection metric.

        Parameters
        ----------
        predictor_combo : tuple of str
            Predictor combination, e.g., ('DWT',) or ('DWT', 'GT')
        target : str
            Target variable
        metric : str, optional
            Metric to display. If None, uses selector's selection_metric.
        ship_type : str, optional
            Ship type to include in title, e.g., 'Container Ships'
        variable_name_map : dict, optional
            Dictionary mapping variable names to formatted display names.
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        highlight_selected : bool, default=True
            Highlight the selected model in different color
        show_values : bool, default=True
            Show metric values on bars
        figsize : tuple, default=(10, 6)
            Figure size
        save_path : str, optional
            Path to save figure

        """

        ModelSelectionPlotter.plot_model_selection_comparison(
            selector=self,
            predictor_combo=predictor_combo,
            target=target,
            metric=metric,
            ship_type=ship_type,
            variable_name_map=variable_name_map,
            highlight_selected=highlight_selected,
            show_values=show_values,
            figsize=figsize,
            save_path=save_path,
            show_heteroscedasticity=show_hetero
        )


    def plot_predictor_functions(
            self,
            target: str,
            predictor: str|None = None,
            show_metrics: bool = True,
            separate_plots: bool = True,
            title_prefix: str|None = None,
            variable_name_map: dict[str, str]|None = None,
            formula_notation_map: dict[str, str]|None = None,
            figsize: tuple[float, float] = (7, 5),
            n_line_points: int = 200,
            show_model_type: bool = False,
            highlight_recommended: bool = True,
            save_path_template: str|None = None,
            debug: bool = False,
    ) -> None:
        """
        Plot fitted regression curves for all single-predictor models of a target,
        with the fitted formula written as the figure title.

        For each predictor (or a specific one), retrieves every model that was tested
        from ``detailed_results`` and overlays the fitted curve on the scatter data.
        The figure title shows the actual formula with coefficients rather than the
        model type name.

        Formula strings are sourced from ``model_descriptors`` where available
        (recommended models), and from a fresh ``ModelDescriptor`` built after
        refitting for non-recommended models.

        Variable names appearing inside the formula string can be replaced with
        shorter mathematical symbols via *formula_notation_map* (e.g.
        ``{'LOA (M)': 'L_OA', 'DWT': 'DWT'}``).  Axis labels use the separate
        *variable_name_map*.

        :param target: Target dimension to plot (must be in ``target_dimensions``).
        :type target: str
        :param predictor: Optional predictor to restrict the plot to.  If ``None``,
            all single-predictor combinations for *target* are shown.
        :type predictor: str or None
        :param show_metrics: If ``True``, annotates each plot with R², Adj R² and
            RRMSE from ``detailed_results``.
        :type show_metrics: bool
        :param separate_plots: If ``True`` (default), each model gets its own
            ``matplotlib`` figure.  If ``False``, all models for a given predictor
            are arranged as subplots within a single figure.
        :type separate_plots: bool
        :param title_prefix: Optional string prepended to each figure title,
            separated by a pipe character.  Intended for the ship type, e.g.
            ``'Bulk Carrier'`` produces ``'Bulk Carrier | DWT → LBP'``.
        :type title_prefix: str or None
        :param variable_name_map: Optional mapping from raw column names to
            formatted display strings used on axis labels
            (e.g. ``{'DWT': 'DWT (t)', 'LOA (M)': '$L_{OA}$ (m)'}``).
        :type variable_name_map: dict or None
        :param formula_notation_map: Optional mapping from raw column names to
            short symbols used *inside* the formula string
            (e.g. ``{'LOA (M)': 'L_OA', 'DWT': 'DWT'}``).
            Applied as plain string substitution after the formula is retrieved
            from the descriptor, so symbols should match the notation style used
            by ``ModelDescriptor.get_formula_string``.
            If ``None``, raw column names are kept in the formula.
        :type formula_notation_map: dict or None
        :param figsize: Size ``(width, height)`` for each individual figure or
            subplot.
        :type figsize: tuple[float, float]
        :param n_line_points: Number of evenly spaced points used to draw the
            smooth prediction curve.
        :type n_line_points: int
        :param show_model_type: If ``True``, prepends the model type name on its
            own line above the formula in the axes title, e.g.
            ``'power\nL_OA = 3.21·DWT^0.412'``.  Defaults to ``False``.
        :type show_model_type: bool
            is drawn with a distinct marker in the title and a slightly thicker
            line.
        :type highlight_recommended: bool
        :param save_path_template: Optional path template for saving figures.
            Use ``{predictor}`` and ``{model}`` as placeholders.
            Example: ``'plots/{predictor}_{model}.png'``.
        :type save_path_template: str or None
        :param debug: If ``True``, prints step-by-step diagnostic information
            about which combinations are found, which models are attempted, and
            any fitting or descriptor failures.
        :type debug: bool
        :raises ValueError: If *target* is not in ``target_dimensions``, or if
            *predictor* is supplied but no single-predictor combination for it
            exists in ``detailed_results``.
        """

        # =========================================================================
        # 1. Validate inputs
        # =========================================================================

        if target not in self.target_dimensions:
            raise ValueError(
                f"Target '{target}' not in target_dimensions. "
                f"Available: {self.target_dimensions}"
            )

        if debug:
            print(f"\n[DEBUG] plot_predictor_functions called")
            print(f"  target={target!r}, predictor={predictor!r}")
            print(f"  separate_plots={separate_plots}, show_metrics={show_metrics}")
            print(f"  formula_notation_map={formula_notation_map}")

        # =========================================================================
        # 2. Collect relevant (predictor, model_type, metrics) triples
        # =========================================================================

        # Structure: { predictor_name: [ {model_type, metrics, is_recommended}, ... ] }
        predictor_model_map: Dict[str, List[Dict]] = {}

        for (pred_combo, tgt), model_results in self.detailed_results.items():

            if tgt != target or len(pred_combo) != 1:
                continue

            pred_name = pred_combo[0]

            if predictor is not None and pred_name != predictor:
                continue

            recommended = self.model_recommendations.get((pred_combo, tgt))

            if debug:
                print(f"  [DEBUG] Found combo ({pred_name},) → {target}: "
                      f"{list(model_results.keys())} | recommended={recommended}")

            entries = []
            for model_type, metrics in model_results.items():
                entries.append({
                    'model_type': model_type,
                    'metrics': metrics,
                    'is_recommended': (model_type == recommended),
                    'pred_combo': pred_combo,
                })

            # Recommended first, then alphabetical
            entries.sort(key=lambda e: (not e['is_recommended'], e['model_type']))
            predictor_model_map[pred_name] = entries

        if not predictor_model_map:
            msg = (f"No single-predictor combinations found for target='{target}'"
                   + (f", predictor='{predictor}'" if predictor else "") + ".")
            raise ValueError(msg)

        if debug:
            print(f"  [DEBUG] Predictors to plot: {list(predictor_model_map.keys())}")

        # =========================================================================
        # 3. Formula retrieval helper
        #    Tries model_descriptors first (exact stored coefficients), then
        #    builds a fresh descriptor from a refit for non-recommended models.
        # =========================================================================

        def _get_formula(pred_name: str, model_type: str, entry: dict,
                         X: np.ndarray, y: np.ndarray) ->str|None:
            """
            Retrieve the formula string for a fitted model.

            Preference order:
            1. ``self.model_descriptors`` (pre-built, exact)
            2. Refit + build ``ModelDescriptor`` on the fly
            3. Fall back to bare model type name

            Variable names are substituted using *formula_notation_map* if
            provided, via simple string replacement on the raw formula.
            """
            pred_combo = entry['pred_combo']
            key = (pred_combo, target)

            formula: str|None = None

            # --- Attempt 1: stored descriptor ------------------------------------
            if (key in self.model_descriptors
                    and model_type in self.model_descriptors[key]):
                try:
                    descriptor = self.model_descriptors[key][model_type]

                    # Refit to ensure variable names are wired into coefficient keys.
                    # The stored descriptor may have generic indexed keys (e.g. 'x0')
                    # rather than actual variable names — refit resolves this.
                    descriptor.refit(X, y, force_structure=True)

                    formula = descriptor.get_formula_string(
                        include_coefficients=True,
                        notation='math',
                        only_active=True,
                    )

                except Exception as exc:
                    if debug:
                        print(f"    [DEBUG] Refit descriptor failed: {exc}")
                    formula = None

            # --- Attempt 2: refit + build descriptor on the fly ------------------
            if formula is None:
                try:
                    model, preprocessor = ModelFactory.create_and_fit(model_type, X, y)
                    descriptor = ModelDescriptor.from_fitted_model(
                        model=model,
                        preprocessor=preprocessor,
                        model_type=model_type,
                        independent_vars=[pred_name],
                        dependent_var=target,
                    )
                    formula = descriptor.get_formula_string(
                        include_coefficients=True,
                        notation='math',
                        only_active=True,
                    )
                    if debug:
                        print(f"    [DEBUG] Formula from refit descriptor: {formula}")
                except Exception as exc:
                    if debug:
                        print(f"    [DEBUG] Refit descriptor failed: {exc}")
                    formula = None

            # --- Attempt 3: bare model type name as last resort ------------------
            if formula is None:
                formula = model_type
                if debug:
                    print(f"    [DEBUG] Falling back to model type name: {formula}")

            # --- Apply formula notation substitutions ----------------------------
            if formula_notation_map and formula != model_type:
                for raw_name, symbol in formula_notation_map.items():
                    formula = formula.replace(raw_name, symbol)

            return formula

        # =========================================================================
        # 4. Helper: render scatter + fitted curve onto a single Axes
        # =========================================================================

        # Colour palette: light steel blue data, vivid orange curve
        _SCATTER_COLOR = '#7FB3D3'  # muted blue — data points
        _CURVE_COLOR = '#E8762B'  # warm orange — fitted curve
        _SCATTER_EDGE = '#2C6E9B'  # darker blue edge for contrast
        _REC_CURVE_COLOR = '#C0392B'  # deeper red for recommended model

        def _draw_on_ax(
                ax: plt.Axes,
                X: np.ndarray,
                y: np.ndarray,
                pred_name: str,
                entry: dict,
                debug: bool,
        ) -> None:
            """
            Render scatter data and the fitted curve for one model onto *ax*.
            The formula string is set as the axes title.
            """
            model_type = entry['model_type']
            metrics = entry['metrics']
            is_rec = entry['is_recommended']

            # ---- Scatter --------------------------------------------------------
            ax.scatter(
                X.ravel(), y,
                alpha=0.55, s=30,
                color=_SCATTER_COLOR,
                edgecolors=_SCATTER_EDGE, linewidth=0.5,
                zorder=2,
                label='Data',
            )

            # ---- Fit model and generate prediction curve ------------------------
            try:
                model, preprocessor = ModelFactory.create_and_fit(model_type, X, y)

                X_line = np.linspace(X.min(), X.max(), n_line_points).reshape(-1, 1)
                y_line = ModelFactory.predict(model, preprocessor, X_line, model_type)

                curve_color = _REC_CURVE_COLOR if (is_rec and highlight_recommended) else _CURVE_COLOR
                lw = 2.8 if (is_rec and highlight_recommended) else 2.0

                rec_label = ' ★' if (is_rec and highlight_recommended) else ''
                ax.plot(
                    X_line.ravel(), y_line,
                    color=curve_color, linewidth=lw, linestyle='-',
                    zorder=3, label=f'Fit{rec_label}',
                )

                if debug:
                    print(f"    [DEBUG] Plotted {model_type}: "
                          f"y_line=[{y_line.min():.2f}, {y_line.max():.2f}]")

            except Exception as exc:
                ax.text(0.5, 0.5, f'Fit failed:\n{type(exc).__name__}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='red')
                if debug:
                    print(f"    [DEBUG] Fitting FAILED for {model_type}: {exc}")

            # ---- Axis labels ----------------------------------------------------
            pred_label = (variable_name_map or {}).get(pred_name, pred_name)
            tgt_label = (variable_name_map or {}).get(target, target)

            ax.set_xlabel(pred_label, fontsize=10)
            ax.set_ylabel(tgt_label, fontsize=10)

            # ---- Formula as axes title ------------------------------------------
            formula_str = _get_formula(pred_name, model_type, entry, X, y)

            rec_marker = ' ★' if (is_rec and highlight_recommended) else ''
            title_str = f'{model_type}\n{formula_str}{rec_marker}' if show_model_type else f'{formula_str}{rec_marker}'
            ax.set_title(
                f'{title_str}',
                fontsize=9,
                fontweight='normal',
                pad=8,
                wrap=True,
            )

            # ---- Optional metrics annotation ------------------------------------
            if show_metrics:
                r2 = metrics.get('r2', float('nan'))
                adj_r2 = metrics.get('adj_r2', float('nan'))
                rrmse = metrics.get('rrmse', float('nan'))
                txt = f'$R^2$={r2:.3f}   Adj $R^2$={adj_r2:.3f}\nNRMSE={rrmse:.2f}%'
                ax.text(
                    0.03, 0.97, txt,
                    transform=ax.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.35',
                              facecolor='white', alpha=0.80,
                              edgecolor='lightgrey'),
                )

            ax.grid(True, alpha=0.22, linestyle='--')
            ax.legend(fontsize=8, loc='lower right')

        # =========================================================================
        # 5. Main plotting loop — one predictor at a time
        # =========================================================================

        for pred_name, entries in predictor_model_map.items():

            n_models = len(entries)
            pred_label = (variable_name_map or {}).get(pred_name, pred_name)
            tgt_label = (variable_name_map or {}).get(target, target)

            if debug:
                print(f"\n[DEBUG] Plotting predictor='{pred_name}' "
                      f"({n_models} models): {[e['model_type'] for e in entries]}")

            # Retrieve clean data once per predictor
            cols = [pred_name, target]
            subset = self.data[cols].dropna()
            X = subset[pred_name].values.reshape(-1, 1)
            y = subset[target].values

            if debug:
                print(f"  [DEBUG] Data shape after dropna: X={X.shape}, y={y.shape}")

            # ------------------------------------------------------------------
            # 5a.  separate_plots=True  →  one figure per model
            # ------------------------------------------------------------------
            if separate_plots:
                for entry in entries:
                    fig, ax = plt.subplots(figsize=figsize)

                    # Suptitle carries the predictor → target context;
                    # individual ax title carries the formula.
                    rec_marker = ' ★ (recommended)' if (entry['is_recommended'] and highlight_recommended) else ''
                    base_title = f'{pred_label}  →  {tgt_label}{rec_marker}'
                    suptitle = f'{title_prefix} | {base_title}' if title_prefix else base_title
                    fig.suptitle(suptitle, fontsize=11, fontweight='bold')

                    _draw_on_ax(ax, X, y, pred_name, entry, debug)
                    plt.tight_layout()

                    if save_path_template:
                        safe_model = entry['model_type'].replace('-', '_')
                        safe_pred = pred_name.replace(' ', '_').replace('(', '').replace(')', '')
                        path = save_path_template.format(predictor=safe_pred, model=safe_model)
                        plt.savefig(path, dpi=150, bbox_inches='tight')
                        if self.verbose:
                            print(f"  Saved: {path}")

                    plt.show()

            # ------------------------------------------------------------------
            # 5b.  separate_plots=False  →  all models as subplots in one figure
            # ------------------------------------------------------------------
            else:
                import math as _math
                cols_n = min(3, n_models)
                rows_n = _math.ceil(n_models / cols_n)

                fig, axes = plt.subplots(
                    rows_n, cols_n,
                    figsize=(figsize[0] * cols_n, figsize[1] * rows_n),
                )

                axes_flat = (
                    [axes] if n_models == 1
                    else np.array(axes).flatten().tolist()
                )

                base_title = f'All models:  {pred_label}  →  {tgt_label}'
                suptitle = f'{title_prefix} | {base_title}' if title_prefix else base_title
                fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)

                for ax, entry in zip(axes_flat, entries):
                    _draw_on_ax(ax, X, y, pred_name, entry, debug)

                for ax in axes_flat[n_models:]:
                    ax.set_visible(False)

                plt.tight_layout()

                if save_path_template:
                    safe_pred = pred_name.replace(' ', '_').replace('(', '').replace(')', '')
                    path = save_path_template.format(predictor=safe_pred, model='all')
                    plt.savefig(path, dpi=150, bbox_inches='tight')
                    if self.verbose:
                        print(f"  Saved: {path}")

                plt.show()