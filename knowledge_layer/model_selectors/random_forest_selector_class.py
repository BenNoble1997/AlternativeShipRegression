"""
Random Forest Regression Selector
==================================

Focused selector for Random Forest regression with comprehensive evaluation.

Key features:
- Random Forest regression only (no model type comparison)
- Optional hyperparameter tuning
- Same CV strategies as ConstrainedModelSelector
- Feature importance analysis
- Similar plotting capabilities
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import warnings
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from knowledge_layer.model_plotting_utility_class import MetricsComparisonPlotter, FeatureImportancePlotter
from knowledge_layer.variable_combination_manager_class import VariableCombinationManager
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.data_split_handling_class import DataSplitHandler
from knowledge_layer.model_selectors.Selector_save_class import ResultsExporter

from  knowledge_layer.model_selectors.predictor_combination_comparison_plotter import PredictorCombinationPlotter
warnings.filterwarnings('ignore')


class RandomForestSelector:
    """Random Forest selector with combination type classification."""

    def __init__(self,
                 training_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: List[str] | None = None,
                 fixed_dimensions: List[str] | None = None,
                 relationship_context: dict[str, List[str]] | None = None,
                 forbidden_combinations: List[Tuple[str, ...]] | None = None,
                 rf_params: dict = None,
                 selection_metric: str = 'rrmse',
                 verbose: bool = True):

        if 'DWT' not in variable_dimensions:
            raise ValueError("DWT must be in variable_dimensions list!")

        # REFACTORED: Use VariableCombinationManager for dimension handling
        self.combination_manager = VariableCombinationManager(
            training_data=training_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            fixed_dimensions=[],  # Don't force dimensions - generate all combinations
            relationship_context=relationship_context,
            forbidden_combinations=forbidden_combinations,
            verbose=verbose
        )

        # Additional check for DWT
        if 'DWT' not in self.combination_manager.available_dimensions:
            raise ValueError("DWT must be in the training data!")

        # Store references for convenience
        self.data = training_data[self.combination_manager.available_dimensions].copy()
        self.variable_dimensions = self.combination_manager.get_variable_dimensions()
        self.target_dimensions = self.combination_manager.get_target_dimensions()
        self.fixed_dimensions = fixed_dimensions or ['DWT']  # Store for reference only
        self.relationship_context = self.combination_manager.relationship_context
        self.forbidden_combinations = self.combination_manager.forbidden_combinations
        self.selection_metric = selection_metric
        self.verbose = verbose

        # Default Random Forest parameters
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'max_features':0.5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        # Storage
        self.results = {}
        self.feature_importance = {}
        self.preprocessors = {}

    def _classify_combination_type(self, predictors: Tuple[str, ...]) -> str:
        """Classify predictor combination by dimension category."""
        return self.combination_manager.classify_combination_type(predictors)

    # ========================================================================
    # MAIN ANALYSIS METHOD (REFACTORED to use DataSplitHandler)
    # ========================================================================

    def analyze_all_combinations(self,
                                 max_predictors: int = 2,
                                 cv_method: str = 'stratified',
                                 n_folds: int = 10,
                                 groups: Optional[np.ndarray] = None,
                                 random_state: int = 42,
                                 display_results: bool = True,
                                 group_by_type: bool = False,
                                 use_feature_engineering: bool = False,
                                 poly_degree: int = 2,
                                 include_ratios: bool = True,
                                 include_logs: bool = True,
                                 scale_raw_features: bool = False):
        """
        Analyze all predictor combinations with automatic type classification.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors per combination
        cv_method : str
            'standard', 'stratified', 'grouped', or None
        n_folds : int
            Number of CV folds
        groups : array-like, optional
            Group labels for grouped CV
        random_state : int
            Random seed
        display_results : bool
            Whether to display results table
        group_by_type : bool
            Whether to group display by combination type (hull/tonnage/propulsion/mixed/other)
        use_feature_engineering : bool
            If True, fit both raw and engineered features
        poly_degree : int
            Degree for polynomial features
        include_ratios : bool
            Whether to include ratio features
        include_logs : bool
            Whether to include log features
        scale_raw_features : bool
            Whether to scale raw features

        Returns
        -------
        dict
            results dictionary
        """
        # ✅ REFACTORED: Use DataSplitHandler for validation
        DataSplitHandler.validate_cv_inputs(cv_method, groups)

        if display_results and self.verbose:
            from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
            display = SelectorResultsDisplay(self)
            display.print_analysis_header(
                max_predictors=max_predictors,
                cv_method=cv_method,
                n_folds=n_folds,
                use_feature_engineering=use_feature_engineering,
                poly_degree=poly_degree
            )

        for target in self.target_dimensions:
            self._analyze_single_target_silent(
                target=target,
                max_predictors=max_predictors,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=groups,
                random_state=random_state,
                use_feature_engineering=use_feature_engineering,
                poly_degree=poly_degree,
                include_ratios=include_ratios,
                include_logs=include_logs,
                scale_raw_features=scale_raw_features
            )

        if display_results and self.verbose:
            self.show_selection_results(group_by_type=group_by_type,
                                        cv_method=cv_method)

        return self.results

    def show_selection_results(self,
                               group_by_type: bool = False,
                               cv_method: str = 'stratified',):

        """
        Function for displaying results of analyse_all_combinations
        """

        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_results(
            group_by_type=group_by_type,
            cv_method=cv_method
        )
        display.print_completion_summary()

    # ========================================================================
    # TARGET-LEVEL PROCESSING (Updated to pass combination type)
    # ========================================================================

    def _analyze_single_target_silent(self,
                                      target: str,
                                      max_predictors: int,
                                      cv_method: str,
                                      n_folds: int,
                                      groups: Optional[np.ndarray],
                                      random_state: int,
                                      use_feature_engineering: bool = False,
                                      poly_degree: int = 2,
                                      include_ratios: bool = True,
                                      include_logs: bool = True,
                                      scale_raw_features: bool = False):
        """Process all combinations for a target."""
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
        if use_feature_engineering:
            total_combos *= 2

        if total_combos == 0:
            return

        target_results = {}

        desc = f"Target: {target[:25]:25s}"
        pbar = tqdm(
            total=total_combos,
            desc=desc,
            disable=not self.verbose,
            unit='combo',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        self._process_all_combinations_silent(
            target=target,
            target_results=target_results,
            valid_combos=valid_combos,
            cv_method=cv_method,
            n_folds=n_folds,
            groups=groups,
            random_state=random_state,
            progress_bar=pbar,
            use_feature_engineering=use_feature_engineering,
            poly_degree=poly_degree,
            include_ratios=include_ratios,
            include_logs=include_logs,
            scale_raw_features=scale_raw_features
        )

        pbar.close()
        self.results[target] = target_results

    def _count_total_combinations(self, target: str, max_predictors: int):
        """Count total combinations to be tested."""
        combos = self.combination_manager.generate_combinations(
            target=target,
            min_predictors=1,
            max_predictors=max_predictors
        )
        return len(combos)

    def _get_available_predictors(self, target: str) -> List[str]:
        """Determine which predictors are available for a specific target."""
        # Get all single-predictor combinations (will respect relationship context)
        combos = self.combination_manager.generate_combinations(
            target=target,
            min_predictors=1,
            max_predictors=1
        )
        # Extract predictor names from single-predictor tuples
        return [combo[0] for combo in combos]

    # ========================================================================
    # COMBINATION-LEVEL PROCESSING (Updated to store combination type)
    # ========================================================================

    def _process_all_combinations_silent(self,
                                         target: str,
                                         target_results: Dict,
                                         valid_combos: List[Tuple[str, ...]],
                                         cv_method: str,
                                         n_folds: int,
                                         groups: Optional[np.ndarray],
                                         random_state: int,
                                         progress_bar: tqdm,
                                         use_feature_engineering: bool = False,
                                         poly_degree: int = 2,
                                         include_ratios: bool = True,
                                         include_logs: bool = True,
                                         scale_raw_features: bool = False):
        """Process all combinations with type classification."""
        for pred_combo in valid_combos:

            combo_type = self._classify_combination_type(pred_combo)

            # Fit with RAW features
            self._process_single_combination_silent(
                pred_combo=pred_combo,
                target=target,
                target_results=target_results,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=groups,
                random_state=random_state,
                use_feature_engineering=False,
                scale_features=scale_raw_features,
                combination_type=combo_type
            )
            progress_bar.update(1)

            # Fit with ENGINEERED features
            if use_feature_engineering:
                self._process_single_combination_silent(
                    pred_combo=pred_combo,
                    target=target,
                    target_results=target_results,
                    cv_method=cv_method,
                    n_folds=n_folds,
                    groups=groups,
                    random_state=random_state,
                    use_feature_engineering=True,
                    poly_degree=poly_degree,
                    include_ratios=include_ratios,
                    include_logs=include_logs,
                    scale_features=True,
                    combination_type=combo_type
                )
                progress_bar.update(1)

    def _process_single_combination_silent(self,
                                           pred_combo: Tuple[str, ...],
                                           target: str,
                                           target_results: Dict,
                                           cv_method: str,
                                           n_folds: int,
                                           groups: Optional[np.ndarray],
                                           random_state: int,
                                           use_feature_engineering: bool = False,
                                           poly_degree: int = 2,
                                           include_ratios: bool = True,
                                           include_logs: bool = True,
                                           scale_features: bool = False,
                                           combination_type: str = 'other'):
        """Process a single combination and store with type."""
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

        if use_feature_engineering:
            try:
                X_clean, engineered_names, preprocessors = self.engineer_features(
                    X_clean,
                    list(pred_combo),
                    poly_degree=poly_degree,
                    include_ratios=include_ratios,
                    include_logs=include_logs,
                    scale=True
                )
                combo_key_eng = (tuple(pred_combo), target, 'engineered')
                self.preprocessors[combo_key_eng] = preprocessors
            except Exception as e:
                if self.verbose:
                    print(f"  Feature engineering failed for {pred_combo}: {e}")
                return
        else:
            engineered_names = list(pred_combo)

            if scale_features:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_clean = scaler.fit_transform(X_clean)
                combo_key_raw = (tuple(pred_combo), target, 'raw')
                self.preprocessors[combo_key_raw] = {'scaler': scaler, 'poly': None}

        try:
            rf_results = self.fit_random_forest(
                X_clean, y_clean,
                predictor_names=engineered_names,
                cv_method=cv_method,
                n_folds=n_folds,
                groups=pred_groups,
                random_state=random_state
            )
        except Exception:
            return

        # Calculate VIF for this combination (once for all models)
        if len(pred_combo) > 1:
            _, vif_info = self.combination_manager._check_vif_combination(pred_combo)
            max_vif = vif_info['max_vif']
            vif_values = vif_info['vif_df'].set_index('feature')['vif'].to_dict()
        else:
            max_vif = 1.0
            vif_values = {pred_combo[0]: 1.0}

        # Add VIF info to metrics
        rf_results['max_vif'] = max_vif
        rf_results['vif_values'] = vif_values

        if use_feature_engineering:
            combo_key = " + ".join(pred_combo) + " (Feat Eng)"
        else:
            combo_key = " + ".join(pred_combo)

        # Store combination_type in results
        target_results[combo_key] = {
            'predictors': pred_combo,
            'metrics': rf_results,
            'feature_engineered': use_feature_engineering,
            'combination_type': combination_type
        }

    # ========================================================================
    # FEATURE ENGINEERING METHODS (Unchanged)
    # ========================================================================

    @staticmethod
    def _generate_ratio_features(X: np.ndarray,
                                 feature_names: List[str],
                                 epsilon: float = 1e-8) -> Tuple[np.ndarray, List[str]]:
        """Generate all pairwise ratios from features."""
        n_features = X.shape[1]

        if n_features < 2:
            return np.empty((X.shape[0], 0)), []

        ratios = []
        ratio_names = []

        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    ratio = X[:, i] / (X[:, j] + epsilon)
                    ratios.append(ratio)
                    ratio_names.append(f'{feature_names[i]}/{feature_names[j]}')

        if ratios:
            return np.column_stack(ratios), ratio_names
        else:
            return np.empty((X.shape[0], 0)), []

    @staticmethod
    def _generate_log_features(X: np.ndarray,
                               feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Generate log-transformed features."""
        X_log = np.log1p(np.abs(X))
        log_names = [f'log({name})' for name in feature_names]
        return X_log, log_names

    def engineer_features(self,
                          X: np.ndarray,
                          feature_names: List[str],
                          poly_degree: int = 2,
                          include_ratios: bool = True,
                          include_logs: bool = True,
                          scale: bool = True) -> Tuple[np.ndarray, List[str], Dict]:
        """Apply feature engineering transformations."""
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        preprocessors = {}

        # 1. Polynomial features
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            poly_names = poly.get_feature_names_out(feature_names).tolist()
            preprocessors['poly'] = poly
        else:
            X_poly = X.copy()
            poly_names = list(feature_names)
            preprocessors['poly'] = None

        X_engineered = X_poly
        engineered_names = poly_names

        # 2. Ratios
        if include_ratios:
            X_ratios, ratio_names = self._generate_ratio_features(X, feature_names)
            if X_ratios.shape[1] > 0:
                X_engineered = np.column_stack([X_engineered, X_ratios])
                engineered_names.extend(ratio_names)

        # 3. Logs
        if include_logs:
            X_log, log_names = self._generate_log_features(X, feature_names)
            X_engineered = np.column_stack([X_engineered, X_log])
            engineered_names.extend(log_names)

        # 4. Scale
        if scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_engineered)
            preprocessors['scaler'] = scaler
            return X_scaled, engineered_names, preprocessors
        else:
            preprocessors['scaler'] = None
            return X_engineered, engineered_names, preprocessors

    # ========================================================================
    # RANDOM FOREST FITTING (REFACTORED to use DataSplitHandler)
    # ========================================================================

    def fit_random_forest(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          predictor_names: List[str],
                          cv_method: Optional[str] = None,
                          n_folds: int = 10,
                          groups: Optional[np.ndarray] = None,
                          random_state: int = 42) -> Dict:
        """Fit Random Forest with optional cross-validation."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))
        X = X[mask]
        y = y[mask]

        if groups is not None:
            groups = groups[mask]

        if len(X) < 10:
            raise ValueError(f"Insufficient clean data: {len(X)} samples")

        n_features = X.shape[1]

        # ✅ REFACTORED: Use DataSplitHandler for CV setup
        cv_config = DataSplitHandler.setup_cv_config(
            X=X,
            y=y,
            cv_method=cv_method,
            n_folds=n_folds,
            groups=groups,
            random_state=random_state
        )

        if cv_config['method'] is None:
            return self._fit_no_cv(X, y, n_features, predictor_names)
        else:
            return self._fit_with_cv(X, y, n_features, predictor_names, cv_config)

    def _fit_no_cv(self, X, y, n_features, predictor_names):
        """Fit without cross-validation."""
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X, y)
        y_pred = rf.predict(X)

        metric_calc = RegressionMetrics(y, y_pred)
        metrics = metric_calc.calculate_all_metrics(n_features)
        metrics['feature_importance'] = dict(zip(predictor_names, rf.feature_importances_))

        return metrics

    def _fit_with_cv(self, X, y, n_features, predictor_names, cv_config):
        """Fit with cross-validation."""
        cv_metrics = []
        cv_feature_importance = []

        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for fold_idx, (train_idx, test_idx) in enumerate(
                splitter.split(*split_args, **split_kwargs)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                rf = RandomForestRegressor(**self.rf_params)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                calculate_metrics = RegressionMetrics(y_test, y_pred)
                metrics = calculate_metrics.calculate_all_metrics(n_features)
                cv_metrics.append(metrics)

                cv_feature_importance.append(rf.feature_importances_)

            except Exception:
                continue

        if not cv_metrics:
            raise ValueError("All CV folds failed")

        avg_metrics = self._average_metrics(cv_metrics)
        avg_importance = np.mean(cv_feature_importance, axis=0)
        avg_metrics['feature_importance'] = dict(zip(predictor_names, avg_importance))

        return avg_metrics

    def _average_metrics(self, cv_metrics: List[dict]) -> dict:
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

    # ========================================================================
    # UTILITY METHODS (Updated to handle combination types)
    # ========================================================================

    def get_best_combinations(self,
                              prefer_feature_engineered: bool = True,
                              filter_by_type: Optional[str] = None) -> Dict[str, Tuple[str, Dict]]:
        """
        Get best predictor combination for each target.

        Parameters
        ----------
        prefer_feature_engineered : bool
            If True and performance is similar, prefer feature-engineered models
        filter_by_type : str, optional
            Filter to only combinations of specific type ('hull', 'tonnage', 'propulsion', 'mixed', 'other')

        Returns
        -------
        dict
            {target: (combo_name, data_dict)}
        """
        best_combos = {}

        for target, combos in self.results.items():
            if not combos:
                continue

            # Filter by type if requested
            if filter_by_type is not None:
                combos = {
                    k: v for k, v in combos.items()
                    if v.get('combination_type') == filter_by_type
                }
                if not combos:
                    continue

            if self.selection_metric == 'adj_r2':
                best_combo = max(
                    combos.items(),
                    key=lambda x: x[1]['metrics']['adj_r2']
                )
            elif self.selection_metric in ['rrmse', 'mape']:
                best_combo = min(
                    combos.items(),
                    key=lambda x: x[1]['metrics'][self.selection_metric]
                )
            else:
                best_combo = max(
                    combos.items(),
                    key=lambda x: x[1]['metrics']['adj_r2']
                )

            best_combos[target] = best_combo

        return best_combos

    def export_results(self, filename: str = 'random_forest_results.csv'):
        """Export all results to CSV with combination types."""
        export_data = []

        for target, combos in self.results.items():
            for combo_name, data in combos.items():
                metrics = data['metrics']

                row = {
                    'Target': target,
                    'Predictors': combo_name,
                    'N_Predictors': len(data['predictors']),
                    'Combination_Type': data.get('combination_type', 'other'),
                    'Feature_Engineered': data.get('feature_engineered', False),
                    'R2': metrics['r2'],
                    'Adj_R2': metrics['adj_r2'],
                    'RRMSE_%': metrics['rrmse'],
                    'MAPE_%': metrics['mape'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'N_Samples': metrics['n_samples'],
                    'Max_VIF': metrics.get('max_vif', 1.0),
                }

                # Add individual VIF values
                vif_values = metrics.get('vif_values', {})
                for feat, vif in vif_values.items():
                    row[f'VIF_{feat}'] = vif

                feature_imp = metrics.get('feature_importance', {})
                for feat, imp in feature_imp.items():
                    row[f'Importance_{feat}'] = imp

                export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"\n✓ Results exported to {filename}")
        return df

    # ========================================================================
    # PLOTTING METHODS (Updated to handle combination types)
    # ========================================================================

    def plot_metrics_comparison(self,
                                max_predictors: int = 2,
                                metric: str = 'rrmse',
                                group_by_type: bool = False,
                                figsize_per_plot: tuple = (10, 6),
                                save_path: str = None):
        """
        Plot comparison using shared plotter.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors to include
        metric : str
            Metric to plot
        group_by_type : bool
            If True, creates separate plots for each combination type
        figsize_per_plot : tuple
            Size for each subplot
        save_path : str, optional
            Path to save figure
        """
        if group_by_type:
            self._plot_metrics_by_type(max_predictors, metric, figsize_per_plot, save_path)
        else:
            results_by_target = self._prepare_plot_data(
                max_predictors=max_predictors,
                metric=metric
            )

            MetricsComparisonPlotter.plot_metrics_comparison(
                results_by_target=results_by_target,
                metric=metric,
                cv_method='stratified',
                figsize_per_plot=figsize_per_plot,
                title="Random Forest Performance",
                save_path=save_path,
                verbose=self.verbose
            )

    def _plot_metrics_by_type(self,
                              max_predictors: int,
                              metric: str,
                              figsize_per_plot: tuple,
                              save_path: str = None):
        """Create separate full figures for each combination type."""
        # Prepare data grouped by type
        data_by_type = {}
        for target, combos in self.results.items():
            for combo_name, data in combos.items():
                if len(data['predictors']) > max_predictors:
                    continue

                combo_type = data.get('combination_type', 'other')
                if combo_type not in data_by_type:
                    data_by_type[combo_type] = {}

                if target not in data_by_type[combo_type]:
                    data_by_type[combo_type][target] = []

                metrics = data['metrics']
                data_by_type[combo_type][target].append({
                    'label': combo_name,
                    'metric_value': metrics[metric],
                    'std_value': metrics.get(f'std_{metric}', 0),
                    'n_predictors': len(data['predictors'])
                })

        if not data_by_type:
            print("No data to plot")
            return

        # Create a separate figure for each combination type
        type_order = ['hull', 'tonnage', 'propulsion', 'mixed', 'other']

        for combo_type in type_order:
            if combo_type not in data_by_type:
                continue

            type_data = data_by_type[combo_type]

            # ✅ FIX: Handle Path object correctly
            if save_path is not None:
                # Convert to Path if string
                from pathlib import Path
                save_path_obj = Path(save_path)
                # Create new path with type suffix
                type_save_path = save_path_obj.parent / f"{save_path_obj.stem}_{combo_type}{save_path_obj.suffix}"
            else:
                type_save_path = None

            # Use the shared plotter for consistency
            MetricsComparisonPlotter.plot_metrics_comparison(
                results_by_target=type_data,
                metric=metric,
                cv_method='stratified',
                figsize_per_plot=figsize_per_plot,
                title=f"Random Forest Performance - {combo_type.upper()} Combinations",
                save_path=type_save_path,
                verbose=self.verbose
            )

    def _prepare_plot_data(self,
                           max_predictors: int,
                           metric: str,
                           filter_by_type: Optional[str] = None) -> dict[str, List[dict]]:
        """Transform internal data structure to plotter format."""
        results_by_target = {}

        for target, combos in self.results.items():
            target_results = []

            for combo_name, data in combos.items():
                if len(data['predictors']) > max_predictors:
                    continue

                # Filter by type if requested
                if filter_by_type and data.get('combination_type') != filter_by_type:
                    continue

                metrics = data['metrics']
                target_results.append({
                    'label': combo_name,
                    'metric_value': metrics[metric],
                    'std_value': metrics.get(f'std_{metric}', 0),
                    'n_predictors': len(data['predictors'])
                })

            target_results.sort(key=lambda x: (x['n_predictors'], -x['metric_value']))

            if target_results:
                results_by_target[target] = target_results

        return results_by_target

    def plot_feature_importance(self,
                                figsize_per_plot: tuple = (10, 6),
                                save_path: str = None):
        """Plot feature importance using shared plotter."""
        best_combos = self.get_best_combinations()

        importance_by_target = {
            target: data['metrics'].get('feature_importance', {})
            for target, (_, data) in best_combos.items()
        }

        FeatureImportancePlotter.plot_feature_importance(
            importance_by_target=importance_by_target,
            figsize_per_plot=figsize_per_plot,
            save_path=save_path,
            verbose=self.verbose
        )

    # ========================================================================
    # DISPLAY RESULTS (Convenience methods)
    # ========================================================================

    def print_initialization_summary(self):
        """Convenience method to display initialization info."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_initialization_summary()

    def print_summary(self, **kwargs):
        """Convenience method to display results."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_results(**kwargs)

    def print_best_combinations(self, **kwargs):
        """Convenience method to display best combinations."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_best_combinations_summary(**kwargs)

    def print_feature_engineering_comparison(self, **kwargs):
        """Convenience method to display feature engineering impact."""
        from knowledge_layer.model_selectors.selector_display_class import SelectorResultsDisplay
        display = SelectorResultsDisplay(self)
        display.print_feature_engineering_comparison(**kwargs)

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

        """

        exporter = ResultsExporter(self,skip_validation=True)
        exporter.load(filepath, format=format)

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

        """

        exporter = ResultsExporter(self)
        exporter.quick_save(name, output_dir)

    @classmethod
    def from_saved_results(cls,
                           results_filepath: str,
                           training_data: pd.DataFrame,
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

        """
        instance = cls(training_data=training_data, **kwargs)
        instance.load_results(results_filepath)
        return instance

    def plot_combination_comparison(self,
                                    target: str,
                                    metric: Optional[str] = None,
                                    ship_type: Optional[str] = None,
                                    variable_name_map: Optional[Dict[str, str]] = None,
                                    highlight_best: bool = True,
                                    show_values: bool = True,
                                    show_vif: bool = True,
                                    max_predictors: Optional[int] = None,
                                    filter_feature_engineered: Optional[bool] = None,
                                    sort_by_n_predictors: bool = True,
                                    figsize: Tuple[float, float] = (12, 8),
                                    save_path: Optional[str] = None):
        """
        Plot comparison of all predictor combinations for a specific target.

        Shows which predictor combinations perform best and helps identify
        whether adding more predictors or using feature engineering improves performance.

        Parameters
        ----------
        target : str
            Target variable to analyze
        metric : str, optional
            Metric to display (if None, uses selector's selection_metric)
        ship_type : str, optional
            Ship type for title, e.g., 'Container Ships'
        variable_name_map : dict, optional
            Map variable names to formatted display names
            Example: {'DWT': 'DWT (t)', 'LOA (M)': 'L_OA (m)'}
        highlight_best : bool, default=True
            Highlight best-performing combination
        show_values : bool, default=True
            Show metric values on plot
        show_vif : bool, default=True
            Show VIF warnings for multicollinearity
        max_predictors : int, optional
            Limit to combinations with at most this many predictors
        filter_feature_engineered : bool, optional
            Show only raw (False) or engineered (True) features
        sort_by_n_predictors : bool, default=True
            Group by number of predictors
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, optional
            Path to save figure

        Examples
        --------
        >>> # Basic usage
        >>> selector.plot_combination_comparison(target='LOA (M)')

        >>> # With formatting and ship type
        >>> variable_map = {
        ...     'DWT': 'DWT (t)',
        ...     'LOA (M)': 'L_OA (m)',
        ...     'GT': 'GT (m³)'
        ... }
        >>> selector.plot_combination_comparison(
        ...     target='LOA (M)',
        ...     ship_type='Container Ships',
        ...     variable_name_map=variable_map,
        ...     max_predictors=2
        ... )

        >>> # Compare only raw features
        >>> selector.plot_combination_comparison(
        ...     target='Beam (M)',
        ...     filter_feature_engineered=False,
        ...     metric='rrmse'
        ... )
        """

        PredictorCombinationPlotter.plot_combination_comparison(
            selector=self,
            target=target,
            metric=metric,
            ship_type=ship_type,
            variable_name_map=variable_name_map,
            highlight_best=highlight_best,
            show_values=show_values,
            show_vif=show_vif,
            max_predictors=max_predictors,
            filter_feature_engineered=filter_feature_engineered,
            sort_by_n_predictors=sort_by_n_predictors,
            figsize=figsize,
            save_path=save_path
        )

    def plot_all_combination_comparisons(self,
                                         metric: Optional[str] = None,
                                         ship_type: Optional[str] = None,
                                         variable_name_map: Optional[Dict[str, str]] = None,
                                         max_predictors: Optional[int] = None,
                                         filter_feature_engineered: Optional[bool] = None,
                                         save_path_template: Optional[str] = None):
        """
        Plot combination comparisons for all targets.

        Creates a separate plot for each target variable showing all
        tested predictor combinations and their performance.

        Parameters
        ----------
        metric : str, optional
            Metric to display
        ship_type : str, optional
            Ship type for titles
        variable_name_map : dict, optional
            Variable name formatting
        max_predictors : int, optional
            Max predictors to include
        filter_feature_engineered : bool, optional
            Filter by feature engineering status
        save_path_template : str, optional
            Template for save paths. Use {target} placeholder.
            Example: 'plots/rf_combinations_{target}.png'

        Examples
        --------
        >>> # Plot all targets
        >>> selector.plot_all_combination_comparisons(
        ...     variable_name_map=variable_map,
        ...     save_path_template='plots/rf_{target}.png'
        ... )
        """


        PredictorCombinationPlotter.plot_multiple_targets(
            selector=self,
            targets=self.target_dimensions,
            metric=metric,
            ship_type=ship_type,
            variable_name_map=variable_name_map,
            max_predictors=max_predictors,
            filter_feature_engineered=filter_feature_engineered,
            save_path_template=save_path_template
        )