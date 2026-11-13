"""
Coefficient Comparison Analyzer (REFACTORED with Display Class)
================================================================

Now much cleaner - all display logic delegated to CoefficientComparisonDisplay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings

from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.multi_dataset_analysers.coefficient_comparison.coefficient_comparison_display import CoefficientComparisonDisplay
from config import LoggerMixin

warnings.filterwarnings('ignore')


class CoefficientComparisonAnalyzer(LoggerMixin):
    """
    Analyze coefficient changes using matched datasets.

    **Display Separated:** All printing delegated to CoefficientComparisonDisplay
    """

    def __init__(self,
                 parametric_selector,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 normalize: bool = True,
                 normalization_method: str = 'zscore',
                 verbose: bool = True):
        """Initialize coefficient comparison analyzer."""

        self.selector = parametric_selector
        self.conventional_data = conventional_data
        self.alternative_data = alternative_data
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.verbose = verbose

        # Storage
        self.normalization_params = {}
        self.comparison_results = None
        self.fitted_models = {
            'conventional_full': {},
            'conventional_matched': {},
            'alternative': {}
        }

        # Validate inputs
        self._validate_inputs()

        # Create matched datasets
        if verbose:
            print("\n" + "=" * 120)
            print("CREATING MATCHED DATASETS FOR COEFFICIENT COMPARISON")
            print("=" * 120)

        self.matched_datasets = self._create_matched_datasets()

        # ✅ Initialize display class
        self.display = CoefficientComparisonDisplay(self)

        # ✅ Use display for initialization summary
        if verbose:
            self.display.print_initialization_summary()

    def _validate_inputs(self):
        """Validate inputs (computation only, no printing)."""
        if not hasattr(self.selector, 'model_recommendations'):
            raise ValueError(
                "Selector must have model_recommendations.\n"
                "Run selector.analyze_all_combinations() first."
            )

        if not self.selector.model_recommendations:
            raise ValueError("Selector has no model recommendations (empty dict)")

        conv_cols = set(self.conventional_data.columns)
        alt_cols = set(self.alternative_data.columns)

        common_cols = conv_cols.intersection(alt_cols)
        if not common_cols:
            raise ValueError("Conventional and alternative datasets have no common columns")

        if self.verbose:
            print(f"\n  Validation:")
            print(f"    Common dimensions:  {len(common_cols)}")
            print(f"    Model recommendations: {len(self.selector.model_recommendations)}")

    def _create_matched_datasets(self):
        """Create matched datasets (computation only)."""
        predictor_combos = set()
        for (predictors, _) in self.selector.model_recommendations.keys():
            predictor_combos.add(predictors)

        predictor_combos = sorted(list(predictor_combos))

        if self.verbose:
            print(f"  Creating matched datasets for {len(predictor_combos)} predictor combinations...")

        matched_sets = MatchedDataset.multiple_matched_datasets(
            alternative_fleet=self.alternative_data,
            conventional_fleet=self.conventional_data,
            logger=self.logger,
            predictor_groups=predictor_combos,
            std_reference='conventional'
        )

        if self.verbose:
            print(f"  ✓ Created {len(matched_sets)} matched datasets")

        return matched_sets

    def _normalize_data(self, X, y, fit_params=False, params=None):
        """
        Normalize data using specified method.

        Parameters
        ----------
        X : np.ndarray
            Predictor data
        y : np.ndarray
            Target data
        fit_params : bool
            If True, calculate and return normalization parameters
        params : dict, optional
            Pre-calculated normalization parameters to apply

        Returns
        -------
        X_norm : np.ndarray
            Normalized predictors
        y_norm : np.ndarray
            Normalized target
        params : dict or None
            Normalization parameters (if fit_params=True)
        """
        if not self.normalize:
            # No normalization
            return X, y, None

        if fit_params:
            # Calculate normalization parameters
            params = {
                'X': {},
                'y': {}
            }

            if self.normalization_method == 'zscore':
                # Z-score normalization: (x - mean) / std
                params['X']['mean'] = np.mean(X, axis=0)
                params['X']['std'] = np.std(X, axis=0)

                params['y']['mean'] = np.mean(y)
                params['y']['std'] = np.std(y)

            elif self.normalization_method == 'minmax':
                # Min-max normalization: (x - min) / (max - min)
                params['X']['min'] = np.min(X, axis=0)
                params['X']['max'] = np.max(X, axis=0)

                params['y']['min'] = np.min(y)
                params['y']['max'] = np.max(y)

            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        elif params is None:
            raise ValueError("Must provide params when fit_params=False")

        # Apply normalization
        if self.normalization_method == 'zscore':
            X_norm = (X - params['X']['mean']) / params['X']['std']
            y_norm = (y - params['y']['mean']) / params['y']['std']

        elif self.normalization_method == 'minmax':
            X_norm = (X - params['X']['min']) / (params['X']['max'] - params['X']['min'])
            y_norm = (y - params['y']['min']) / (params['y']['max'] - params['y']['min'])

        return X_norm, y_norm, params if fit_params else None

    def _denormalize_predictions(self, y_pred_norm, y_params):
        """
        Denormalize predictions back to original scale.

        Parameters
        ----------
        y_pred_norm : np.ndarray
            Normalized predictions
        y_params : dict
            Normalization parameters for y

        Returns
        -------
        y_pred : np.ndarray
            Predictions in original scale
        """
        if not self.normalize:
            return y_pred_norm

        if self.normalization_method == 'zscore':
            y_pred = y_pred_norm * y_params['std'] + y_params['mean']

        elif self.normalization_method == 'minmax':
            y_pred = y_pred_norm * (y_params['max'] - y_params['min']) + y_params['min']

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        return y_pred

    def _extract_coefficients(self, model, preprocessor, model_type, predictor_names):
        """
        Extract coefficients from fitted model.

        Parameters
        ----------
        model : fitted model
            Model object
        preprocessor : object or None
            Preprocessor (e.g., PolynomialFeatures for polynomial models)
        model_type : str
            Model type string
        predictor_names : list of str
            Names of predictors

        Returns
        -------
        dict
            Coefficient dictionary {feature_name: coefficient_value}
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        # Use ModelFactory to extract coefficients
        # This handles all model types including polynomial, linear, etc.
        coeffs = ModelFactory.get_model_coefficients(
            model=model,
            preprocessor=preprocessor,
            feature_names=predictor_names,
            model_type=model_type
        )

        # Add intercept if available
        if hasattr(model, 'intercept_'):
            coeffs['intercept'] = model.intercept_

        return coeffs

    def _calculate_metrics(self, y_true, y_pred, n_features):
        """
        Calculate regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        n_features : int
            Number of features in model

        Returns
        -------
        dict
            Dictionary of metrics
        """
        from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics

        metric_calculator = RegressionMetrics(y_pred, y_true)
        metrics = metric_calculator.calculate_all_metrics(n_features)

        return metrics

    def _prepare_data(self, data, predictor_names, target_name):
        """
        Prepare predictor and target data from dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing predictors and target
        predictor_names : list of str
            Names of predictor columns
        target_name : str
            Name of target column

        Returns
        -------
        tuple or None
            (X, y) arrays, or None if insufficient data
        """
        try:
            # Extract data
            X = data[list(predictor_names)].values
            y = data[target_name].values

            # Remove NaN/Inf
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                     np.isinf(X).any(axis=1) | np.isinf(y))

            X_clean = X[mask]
            y_clean = y[mask]

            # Check minimum samples
            if len(X_clean) < 5:
                if self.verbose:
                    print(f"  ⚠️  Insufficient clean data: {len(X_clean)} samples")
                return None

            return X_clean, y_clean

        except KeyError as e:
            if self.verbose:
                print(f"  ⚠️  Missing columns: {e}")
            return None

    def _compare_coefficients_threeway(self, coeffs_full, coeffs_matched, coeffs_alt, model_type):
        """
        Compare coefficients across three datasets.

        For polynomial models, compares all polynomial terms (x0, x0^2, x0 x1, etc.)
        Not just the original predictor names.

        Parameters
        ----------
        coeffs_full : dict
            Coefficients from full conventional model
        coeffs_matched : dict
            Coefficients from matched conventional model
        coeffs_alt : dict
            Coefficients from alternative model
        model_type : str
            Model type

        Returns
        -------
        dict
            Comparison results with distribution and fuel effects
        """
        comparison = {
            'distribution_effect': {},  # Full → Matched
            'fuel_effect': {},  # Matched → Alt
            'total_effect': {},  # Full → Alt
            'interpretation': ''
        }

        # Find ALL coefficient keys from any of the three models
        all_keys = set(coeffs_full.keys()) | set(coeffs_matched.keys()) | set(coeffs_alt.keys())

        # Only keep keys that appear in ALL three models
        common_keys = set(coeffs_full.keys()).intersection(
            set(coeffs_matched.keys()).intersection(set(coeffs_alt.keys()))
        )

        if not common_keys:
            comparison['interpretation'] = "No common coefficients found across all three models"
            return comparison

        for key in common_keys:
            val_full = coeffs_full[key]
            val_matched = coeffs_matched[key]
            val_alt = coeffs_alt[key]

            # Distribution effect (Full → Matched)
            dist_change = val_matched - val_full
            comparison['distribution_effect'][key] = {
                'from': val_full,
                'to': val_matched,
                'absolute_change': dist_change,
                'relative_change': (dist_change / val_full * 100) if abs(val_full) > 1e-10 else np.inf
            }

            # Fuel effect (Matched → Alt)
            fuel_change = val_alt - val_matched
            comparison['fuel_effect'][key] = {
                'from': val_matched,
                'to': val_alt,
                'absolute_change': fuel_change,
                'relative_change': (fuel_change / val_matched * 100) if abs(val_matched) > 1e-10 else np.inf
            }

            # Total effect (Full → Alt)
            total_change = val_alt - val_full
            comparison['total_effect'][key] = {
                'from': val_full,
                'to': val_alt,
                'absolute_change': total_change,
                'relative_change': (total_change / val_full * 100) if abs(val_full) > 1e-10 else np.inf
            }

        # Calculate fuel contribution percentage
        total_abs_change = sum(abs(comparison['total_effect'][k]['absolute_change'])
                               for k in common_keys)
        fuel_abs_change = sum(abs(comparison['fuel_effect'][k]['absolute_change'])
                              for k in common_keys)

        if total_abs_change > 1e-10:
            fuel_contribution = 100 * fuel_abs_change / total_abs_change
            comparison['interpretation'] = (
                f"Fuel effect accounts for {fuel_contribution:.1f}% of total coefficient changes"
            )
        else:
            comparison['interpretation'] = "Coefficients show minimal change across datasets"

        return comparison

    def _get_best_combination_for_target(self, target: str) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
        """
        Get best predictor combination and model type for a target.

        Uses the selector's get_best_combinations() method.

        Parameters
        ----------
        target : str
            Target dimension

        Returns
        -------
        best_combo : tuple of str or None
            Best predictor combination
        best_model_type : str or None
            Best model type for that combination
        """
        # Get best combinations for all targets
        try:
            best_combos = self.selector.get_best_combinations(
                selection_metric=self.selector.selection_metric,
                max_vif=10.0,
                penalize_heteroscedasticity=True,
                heteroscedasticity_penalty=0.05,
                prefer_simpler=True
            )

            if target in best_combos:
                predictors, model_type, metrics = best_combos[target]
                return predictors, model_type
            else:
                return None, None

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Could not get best combination for {target}: {e}")
            return None, None

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_targets(self,
                           min_alternative_samples: int = 10,
                           min_matched_samples: int = 10,
                           display_results: bool = True) -> pd.DataFrame:
        """
        Analyze all targets with three-way comparison.

        ✅ DISPLAY SEPARATED: Uses display class for all printing
        """

        # ✅ OPTIONAL DISPLAY: Header
        if display_results and self.verbose:
            self.display.print_analysis_header()

        # ✅ COMPUTATION: Analyze all targets
        all_results = []

        for target in self.selector.target_dimensions:
            # ✅ OPTIONAL DISPLAY: Target header
            if display_results and self.verbose:
                best_combo, best_model_type = self._get_best_combination_for_target(target)
                if best_combo:
                    self.display.print_target_header(target, best_combo, best_model_type)

            result = self._analyze_single_target_threeway(
                target=target,
                min_alternative_samples=min_alternative_samples,
                min_matched_samples=min_matched_samples,
                display_results=display_results
            )

            if result is not None:
                all_results.append(result)

        # Create results DataFrame
        if not all_results:
            print("\n⚠️  No valid results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        self.comparison_results = results_df

        # ✅ OPTIONAL DISPLAY: Summary
        if display_results and self.verbose:
            self.display.print_summary_table(results_df)

        return results_df

    def _analyze_single_target_threeway(self,
                                        target: str,
                                        min_alternative_samples: int,
                                        min_matched_samples: int,
                                        display_results: bool) -> Optional[Dict]:
        """Analyze single target (computation only, minimal printing)."""

        best_combo, best_model_type = self._get_best_combination_for_target(target)

        if best_combo is None:
            if self.verbose:
                print(f"  ⚠️  No recommendations found for {target}")
            return None

        # Prepare data (unchanged - all computation)
        conv_full_data = self._prepare_data(self.conventional_data, best_combo, target)
        if conv_full_data is None:
            return None
        X_conv_full, y_conv_full = conv_full_data

        predictor_key = tuple(best_combo)
        if predictor_key not in self.matched_datasets:
            return None

        matched_dataset = self.matched_datasets[predictor_key]
        conv_matched_data = self._prepare_data(matched_dataset, best_combo, target)
        if conv_matched_data is None:
            return None
        X_conv_matched, y_conv_matched = conv_matched_data

        alt_data = self._prepare_data(self.alternative_data, best_combo, target)
        if alt_data is None:
            return None
        X_alt, y_alt = alt_data

        # Check minimum samples
        if len(X_alt) < min_alternative_samples or len(X_conv_matched) < min_matched_samples:
            return None

        # Normalize data (computation)
        X_conv_full_norm, y_conv_full_norm, norm_params = self._normalize_data(
            X_conv_full, y_conv_full, fit_params=True
        )

        X_conv_matched_norm, y_conv_matched_norm, _ = self._normalize_data(
            X_conv_matched, y_conv_matched, fit_params=False, params=norm_params
        )

        X_alt_norm, y_alt_norm, _ = self._normalize_data(
            X_alt, y_alt, fit_params=False, params=norm_params
        )

        key = (tuple(best_combo), target)
        self.normalization_params[key] = norm_params

        # Fit models (computation)
        try:
            conv_full_model, conv_full_prep = ModelFactory.create_and_fit(
                best_model_type, X_conv_full_norm, y_conv_full_norm
            )
            y_pred_conv_full_norm = ModelFactory.predict(
                conv_full_model, conv_full_prep, X_conv_full_norm, best_model_type
            )

            conv_matched_model, conv_matched_prep = ModelFactory.create_and_fit(
                best_model_type, X_conv_matched_norm, y_conv_matched_norm
            )
            y_pred_conv_matched_norm = ModelFactory.predict(
                conv_matched_model, conv_matched_prep, X_conv_matched_norm, best_model_type
            )

            alt_model, alt_prep = ModelFactory.create_and_fit(
                best_model_type, X_alt_norm, y_alt_norm
            )
            y_pred_alt_norm = ModelFactory.predict(
                alt_model, alt_prep, X_alt_norm, best_model_type
            )

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Model fitting failed: {e}")
            return None

        # Store models
        self.fitted_models['conventional_full'][key] = (conv_full_model, conv_full_prep)
        self.fitted_models['conventional_matched'][key] = (conv_matched_model, conv_matched_prep)
        self.fitted_models['alternative'][key] = (alt_model, alt_prep)

        # ✅ FIXED: Extract coefficients with correct arguments
        conv_full_coeffs = self._extract_coefficients(
            conv_full_model, conv_full_prep, best_model_type, list(best_combo)
        )
        conv_matched_coeffs = self._extract_coefficients(
            conv_matched_model, conv_matched_prep, best_model_type, list(best_combo)
        )
        alt_coeffs = self._extract_coefficients(
            alt_model, alt_prep, best_model_type, list(best_combo)
        )

        # Calculate metrics (computation)
        y_pred_conv_full = self._denormalize_predictions(y_pred_conv_full_norm, norm_params['y'])
        y_pred_conv_matched = self._denormalize_predictions(y_pred_conv_matched_norm, norm_params['y'])
        y_pred_alt = self._denormalize_predictions(y_pred_alt_norm, norm_params['y'])

        conv_full_metrics = self._calculate_metrics(y_conv_full, y_pred_conv_full, len(best_combo))
        conv_matched_metrics = self._calculate_metrics(y_conv_matched, y_pred_conv_matched, len(best_combo))
        alt_metrics = self._calculate_metrics(y_alt, y_pred_alt, len(best_combo))

        # Cross-evaluations (computation)
        cross_evals = {}
        try:
            y_matched_by_full_norm = ModelFactory.predict(
                conv_full_model, conv_full_prep, X_conv_matched_norm, best_model_type
            )
            y_matched_by_full = self._denormalize_predictions(y_matched_by_full_norm, norm_params['y'])
            cross_evals['full_on_matched'] = self._calculate_metrics(
                y_conv_matched, y_matched_by_full, len(best_combo)
            )

            y_alt_by_full_norm = ModelFactory.predict(
                conv_full_model, conv_full_prep, X_alt_norm, best_model_type
            )
            y_alt_by_full = self._denormalize_predictions(y_alt_by_full_norm, norm_params['y'])
            cross_evals['full_on_alt'] = self._calculate_metrics(
                y_alt, y_alt_by_full, len(best_combo)
            )

            y_alt_by_matched_norm = ModelFactory.predict(
                conv_matched_model, conv_matched_prep, X_alt_norm, best_model_type
            )
            y_alt_by_matched = self._denormalize_predictions(y_alt_by_matched_norm, norm_params['y'])
            cross_evals['matched_on_alt'] = self._calculate_metrics(
                y_alt, y_alt_by_matched, len(best_combo)
            )

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Cross-evaluation failed: {e}")

        # Coefficient comparison (computation)
        coeff_comparison = self._compare_coefficients_threeway(
            conv_full_coeffs,
            conv_matched_coeffs,
            alt_coeffs,
            best_model_type
        )

        # ✅ OPTIONAL DISPLAY: Results
        if display_results and self.verbose:
            self.display.print_single_target_results(
                target, best_combo, best_model_type,
                conv_full_coeffs, conv_matched_coeffs, alt_coeffs,
                coeff_comparison,
                conv_full_metrics, conv_matched_metrics, alt_metrics,
                cross_evals
            )

        # Build result dictionary (computation)
        result = {
            'target': target,
            'predictors': best_combo,  # ✅ Keep as tuple for later use
            'n_predictors': len(best_combo),
            'model_type': best_model_type,
            'normalized': self.normalize,
            'normalization_method': self.normalization_method if self.normalize else None,

            'n_conventional_full': len(X_conv_full),
            'n_conventional_matched': len(X_conv_matched),
            'n_alternative': len(X_alt),
            'conv_full_r2': conv_full_metrics['r2'],
            'conv_full_rrmse': conv_full_metrics['rrmse'],
            'conv_matched_r2': conv_matched_metrics['r2'],
            'conv_matched_rrmse': conv_matched_metrics['rrmse'],
            'alt_r2': alt_metrics['r2'],
            'alt_rrmse': alt_metrics['rrmse'],

            'cross_full_on_matched_r2': cross_evals.get('full_on_matched', {}).get('r2', np.nan),
            'cross_full_on_matched_rrmse': cross_evals.get('full_on_matched', {}).get('rrmse', np.nan),
            'cross_full_on_alt_r2': cross_evals.get('full_on_alt', {}).get('r2', np.nan),
            'cross_full_on_alt_rrmse': cross_evals.get('full_on_alt', {}).get('rrmse', np.nan),
            'cross_matched_on_alt_r2': cross_evals.get('matched_on_alt', {}).get('r2', np.nan),
            'cross_matched_on_alt_rrmse': cross_evals.get('matched_on_alt', {}).get('rrmse', np.nan),

            'coefficients_full': conv_full_coeffs,
            'coefficients_matched': conv_matched_coeffs,
            'coefficients_alternative': alt_coeffs,

            'coefficient_comparison': coeff_comparison,
        }

        return result

    # ========================================================================
    # DECOMPOSITION METHOD
    # ========================================================================

    def decompose_coefficient_changes(self,
                                      effect_size_threshold: float = 0.2) -> pd.DataFrame:
        """
        Decompose coefficient changes.

        ✅ DISPLAY SEPARATED: Uses display class for printing
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results available. Run analyze_all_targets() first.")
            return pd.DataFrame()

        if not self.normalize:
            print("⚠️  WARNING: Normalization not enabled!")

        decompositions = []

        for _, row in self.comparison_results.iterrows():
            coeff_comp = row['coefficient_comparison']

            coef_keys = [k for k in coeff_comp['fuel_effect'].keys()
                         if k != 'interpretation']

            for coef_name in coef_keys:
                dist_effect = coeff_comp['distribution_effect'][coef_name]
                fuel_effect = coeff_comp['fuel_effect'][coef_name]
                total_effect = coeff_comp['total_effect'][coef_name]

                total_abs = abs(total_effect['absolute_change'])
                if total_abs > 1e-10:
                    dist_pct = 100 * abs(dist_effect['absolute_change']) / total_abs
                    fuel_pct = 100 * abs(fuel_effect['absolute_change']) / total_abs
                else:
                    dist_pct = np.nan
                    fuel_pct = np.nan

                fuel_abs = abs(fuel_effect['absolute_change'])
                if fuel_abs < effect_size_threshold:
                    effect_size = "Negligible"
                    emoji = "✓"
                elif fuel_abs < 0.5:
                    effect_size = "Small"
                    emoji = "⚠️"
                elif fuel_abs < 0.8:
                    effect_size = "Medium"
                    emoji = "🔶"
                else:
                    effect_size = "Large"
                    emoji = "🚨"

                decompositions.append({
                    'target': row['target'],
                    'model_type': row['model_type'],
                    'coefficient': coef_name,
                    'full': dist_effect['from'],
                    'matched': dist_effect['to'],
                    'alternative': fuel_effect['to'],
                    'distribution_effect': dist_effect['absolute_change'],
                    'fuel_effect': fuel_effect['absolute_change'],
                    'total_effect': total_effect['absolute_change'],
                    'distribution_pct': dist_pct,
                    'fuel_pct': fuel_pct,
                    'fuel_effect_size': effect_size,
                    'fuel_significant': fuel_abs >= effect_size_threshold,
                    'emoji': emoji
                })

        decomp_df = pd.DataFrame(decompositions)

        # ✅ USE DISPLAY CLASS
        if self.verbose:
            self.display.print_decomposition_summary(decomp_df, effect_size_threshold)

        return decomp_df

    # ========================================================================
    # CONVENIENCE DISPLAY METHODS
    # ========================================================================

    def print_matching_quality(self):
        """Print matching quality assessment."""
        self.display.print_matching_quality(self.matched_datasets)

    def plot_coefficient_decomposition(self,
                                       target: Optional[str] = None,
                                       figsize: Tuple[int, int] = (12, 10),
                                       save_path: Optional[str] = None):
        """
        Plot two-way coefficient decomposition.

        Creates visualization showing:
        - Full → Matched → Alternative progression
        - Distribution vs Fuel effect contributions

        Parameters
        ----------
        target : str, optional
            Specific target to plot. If None, plots all.
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        results_df = self.comparison_results

        if target is not None:
            results_df = results_df[results_df['target'] == target]
            if len(results_df) == 0:
                print(f"No results found for target '{target}'")
                return

        n_targets = len(results_df)
        fig, axes = plt.subplots(n_targets, 2, figsize=figsize)

        if n_targets == 1:
            axes = axes.reshape(1, -1)

        title_suffix = " (Standardized)" if self.normalize else ""
        fig.suptitle(f'Coefficient Decomposition{title_suffix}',
                     fontsize=16, fontweight='bold')

        for idx, (_, row) in enumerate(results_df.iterrows()):

            # Get model info for equation formatting
            target_name = row['target']
            model_type = row['model_type']
            coeffs_alt = row['coefficients_alternative']

            # ✅ Generate equation string
            equation_str = self._format_model_equation(coeffs_alt, model_type, target_name)

            # ================================================================
            # LEFT: Coefficient progression (waterfall-style)
            # ================================================================
            ax_coef = axes[idx, 0]

            coeffs_full = row['coefficients_full']
            coeffs_matched = row['coefficients_matched']
            coeffs_alt = row['coefficients_alternative']

            common_keys = sorted(set(coeffs_full.keys()).intersection(
                set(coeffs_matched.keys()).intersection(set(coeffs_alt.keys()))
            ))

            if common_keys:
                # Plot lines showing progression
                for key in common_keys:
                    vals = [coeffs_full[key], coeffs_matched[key], coeffs_alt[key]]
                    ax_coef.plot([0, 1, 2], vals, 'o-', linewidth=2,
                                 markersize=8, label=key, alpha=0.7)

                ax_coef.set_xticks([0, 1, 2])
                ax_coef.set_xticklabels(['Full', 'Matched', 'Alt'])
                ax_coef.set_ylabel('Coefficient Value' + (' (std)' if self.normalize else ''),
                                   fontweight='bold')

                # ✅ Updated title with equation
                title_text = f"{target_name}\nCoefficient Progression"
                if len(equation_str) < 60:  # Only show if not too long
                    title_text += f"\n{equation_str}"
                else:
                    title_text += f"\n{equation_str[:57]}..."

                ax_coef.set_title(title_text, fontweight='bold', fontsize=9)
                ax_coef.legend(fontsize=8, loc='best')
                ax_coef.grid(True, alpha=0.3)
                ax_coef.axhline(y=0, color='black', linewidth=1, linestyle='--')

            # ================================================================
            # RIGHT: Effect decomposition (stacked bars)
            # ================================================================
            ax_decomp = axes[idx, 1]

            coeff_comp = row['coefficient_comparison']

            if common_keys:
                dist_effects = []
                fuel_effects = []
                labels = []

                for key in common_keys:
                    if key in coeff_comp['distribution_effect']:
                        dist_effects.append(
                            coeff_comp['distribution_effect'][key]['absolute_change']
                        )
                        fuel_effects.append(
                            coeff_comp['fuel_effect'][key]['absolute_change']
                        )
                        labels.append(key)

                x_pos = np.arange(len(labels))

                # Stack bars
                ax_decomp.barh(x_pos, dist_effects, 0.8,
                               label='Distribution Effect',
                               color='orange', alpha=0.7)
                ax_decomp.barh(x_pos, fuel_effects, 0.8, left=dist_effects,
                               label='Fuel Effect',
                               color='red', alpha=0.7)

                ax_decomp.set_yticks(x_pos)
                ax_decomp.set_yticklabels(labels)
                ax_decomp.set_xlabel('Change' + (' (std)' if self.normalize else ''),
                                     fontweight='bold')
                ax_decomp.set_title(f"{target_name}\nEffect Decomposition", fontweight='bold')
                ax_decomp.legend(fontsize=8)
                ax_decomp.axvline(x=0, color='black', linewidth=2)
                ax_decomp.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Decomposition plot saved to {save_path}")

        plt.show()

    def plot_cross_evaluation_threeway(self,
                                       figsize: Tuple[int, int] = (16, 6),
                                       save_path: Optional[str] = None):
        """
        Plot three-way cross-evaluation showing all model transfers.

        Shows matrix of: Which model (Full/Matched/Alt) on which data.
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        results_df = self.comparison_results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        targets = results_df['target'].values
        x = np.arange(len(targets))
        width = 0.25

        # ====================================================================
        # Plot 1: R² comparison
        # ====================================================================

        # In-sample performance
        full_r2 = results_df['conv_full_r2'].values
        matched_r2 = results_df['conv_matched_r2'].values
        alt_r2 = results_df['alt_r2'].values

        # Cross-evaluations
        full_on_alt = results_df['cross_full_on_alt_r2'].values
        matched_on_alt = results_df['cross_matched_on_alt_r2'].values

        ax1.bar(x - width, full_r2, width, label='Full on Full', color='royalblue', alpha=0.8)
        ax1.bar(x, matched_r2, width, label='Matched on Matched', color='steelblue', alpha=0.8)
        ax1.bar(x + width, alt_r2, width, label='Alt on Alt', color='forestgreen', alpha=0.8)

        # Cross-evals as markers
        ax1.scatter(x - width / 2, full_on_alt, marker='x', s=100,
                    color='red', label='Full on Alt', zorder=5)
        ax1.scatter(x + width / 2, matched_on_alt, marker='+', s=100,
                    color='darkred', label='Matched on Alt', zorder=5)

        ax1.set_ylabel('R²', fontweight='bold', fontsize=12)
        ax1.set_title('Three-Way Model Transfer (R²)', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(bottom=0)

        # ====================================================================
        # Plot 2: RRMSE comparison
        # ====================================================================

        full_rrmse = results_df['conv_full_rrmse'].values
        matched_rrmse = results_df['conv_matched_rrmse'].values
        alt_rrmse = results_df['alt_rrmse'].values

        full_on_alt_rrmse = results_df['cross_full_on_alt_rrmse'].values
        matched_on_alt_rrmse = results_df['cross_matched_on_alt_rrmse'].values

        ax2.bar(x - width, full_rrmse, width, label='Full on Full', color='royalblue', alpha=0.8)
        ax2.bar(x, matched_rrmse, width, label='Matched on Matched', color='steelblue', alpha=0.8)
        ax2.bar(x + width, alt_rrmse, width, label='Alt on Alt', color='forestgreen', alpha=0.8)

        ax2.scatter(x - width / 2, full_on_alt_rrmse, marker='x', s=100,
                    color='red', label='Full on Alt', zorder=5)
        ax2.scatter(x + width / 2, matched_on_alt_rrmse, marker='+', s=100,
                    color='darkred', label='Matched on Alt', zorder=5)

        ax2.set_ylabel('RRMSE (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Three-Way Model Transfer (RRMSE)', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(targets, rotation=45, ha='right')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Three-Way Cross-Evaluation: Model Transfer Analysis',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Cross-evaluation plot saved to {save_path}")

        plt.show()

    # ========================================================================
    # EXPORT METHODS
    # ========================================================================

    def export_results(self, filename: str = 'coefficient_comparison_threeway.csv'):
        """Export three-way comparison results to CSV."""

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to export. Run analyze_all_targets() first.")
            return

        export_df = self.comparison_results.copy()

        # ✅ Convert tuple predictors to string before other conversions
        export_df['predictors'] = export_df['predictors'].apply(
            lambda x: ' + '.join(x) if isinstance(x, tuple) else x
        )

        # Convert dict columns to JSON strings
        export_df['coefficients_full'] = export_df['coefficients_full'].apply(str)
        export_df['coefficients_matched'] = export_df['coefficients_matched'].apply(str)
        export_df['coefficients_alternative'] = export_df['coefficients_alternative'].apply(str)
        export_df['coefficient_comparison'] = export_df['coefficient_comparison'].apply(str)

        export_df.to_csv(filename, index=False)

        if self.verbose:
            print(f"✓ Three-way results exported to {filename}")
            print(f"  {len(export_df)} rows, {len(export_df.columns)} columns")

    def _format_model_equation(self, coefficients: Dict, model_type: str, target: str) -> str:
        """
        Format model equation as a string for display.

        Parameters
        ----------
        coefficients : dict
            Coefficient dictionary from get_model_coefficients
        model_type : str
            Model type (e.g., 'linear', 'polynomial-2', 'power', 'mfp')
        target : str
            Target variable name

        Returns
        -------
        str
            Formatted equation string
        """

        # Handle MFP models specially
        if model_type == 'mfp':
            # MFP coefficients are already in transformed form
            terms = []
            for key, value in coefficients.items():
                if key == 'intercept':
                    continue
                if value is not None:
                    if self.normalize:
                        terms.append(f"{value:.3f}·{key}")
                    else:
                        terms.append(f"{value:.2f}·{key}")

            intercept = coefficients.get('intercept', 0)
            if self.normalize:
                equation = f"{target} = {intercept:.3f}"
            else:
                equation = f"{target} = {intercept:.2f}"

            if terms:
                equation += " + " + " + ".join(terms)

            return equation

        # Handle piecewise models
        if model_type.endswith('-piecewise'):
            return f"{target} = Piecewise {model_type.replace('-piecewise', '')} (see bins)"

        # Handle non-linear single-predictor models
        if model_type in ['exponential', 'logarithmic', 'power',
                          'exponential-offset', 'power-offset', 'logarithmic-power']:
            params = list(coefficients.values())

            if model_type == 'exponential':
                return f"{target} = {params[0]:.3f}·exp({params[1]:.3f}·x)"
            elif model_type == 'logarithmic':
                return f"{target} = {params[0]:.3f}·ln(x) + {params[1]:.3f}"
            elif model_type == 'power':
                return f"{target} = {params[0]:.3f}·x^{params[1]:.3f}"
            elif model_type == 'exponential-offset':
                return f"{target} = {params[0]:.3f}·exp({params[1]:.3f}·x) + {params[2]:.3f}"
            elif model_type == 'power-offset':
                return f"{target} = {params[0]:.3f}·x^{params[1]:.3f} + {params[2]:.3f}"
            elif model_type == 'logarithmic-power':
                return f"{target} = {params[0]:.3f}·ln(x)^{params[1]:.3f}"

        # Handle multivariate log-transformed models
        if model_type == 'power-log-multivariate':
            a = coefficients['a (scale)']
            terms = []
            for key, value in coefficients.items():
                if 'exponent' in key:
                    var_name = key.split('(')[1].split(' exponent')[0]
                    terms.append(f"{var_name}^{value:.3f}")
            return f"{target} = {a:.3f}·" + "·".join(terms)

        if model_type == 'exponential-log-multivariate':
            a = coefficients['a (scale)']
            terms = []
            for key, value in coefficients.items():
                if 'coefficient' in key:
                    var_name = key.split('(')[1].split(' coefficient')[0]
                    terms.append(f"{value:.3f}·{var_name}")
            return f"{target} = {a:.3f}·exp(" + " + ".join(terms) + ")"

        # Handle linear/polynomial models (most common case)
        terms = []
        intercept = None

        for key, value in coefficients.items():
            if key == 'intercept' or key == '1':  # PolynomialFeatures adds '1' for intercept
                intercept = value
            else:
                # Format coefficient
                if self.normalize:
                    coef_str = f"{value:.3f}"
                else:
                    coef_str = f"{value:.2f}"

                # Clean up polynomial feature names
                # e.g., "x0 x1" -> "x₀·x₁", "x0^2" -> "x₀²"
                feature_str = key.replace(' ', '·').replace('^2', '²').replace('^3', '³')

                terms.append(f"{coef_str}·{feature_str}")

        # Build equation
        if intercept is not None:
            if self.normalize:
                equation = f"{target} = {intercept:.3f}"
            else:
                equation = f"{target} = {intercept:.2f}"
        else:
            equation = f"{target} = 0"

        if terms:
            equation += " + " + " + ".join(terms)

        # Clean up negative signs (turn "+ -" into "- ")
        equation = equation.replace("+ -", "- ")

        return equation

    def plot_coefficient_changes(self,
                                 figsize: Tuple[int, int] = (12, 10),
                                 show_effects: str = 'both',
                                 save_path: Optional[str] = None):
        """
        Plot coefficient changes as horizontal bar charts.

        Creates subplots for each target showing how coefficients change
        from Full → Matched → Alternative.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        show_effects : str
            Which effects to show:
            - 'both': Show both distribution and fuel effects (stacked)
            - 'distribution': Show only distribution effect
            - 'fuel': Show only fuel effect
            - 'total': Show only total effect
        save_path : str, optional
            Path to save the figure

        Examples
        --------
        >>> analyzer.plot_coefficient_changes(show_effects='both')
        >>> analyzer.plot_coefficient_changes(show_effects='total', save_path='coef_changes.png')
        """

        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        results_df = self.comparison_results
        n_targets = len(results_df)

        # Create subplots
        fig, axes = plt.subplots(n_targets, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        title_suffix = " (Standardized Coefficients)" if self.normalize else ""
        fig.suptitle(f'Coefficient Changes{title_suffix}',
                     fontsize=16, fontweight='bold', y=0.995)

        for idx, (_, row) in enumerate(results_df.iterrows()):
            ax = axes[idx]

            target = row['target']
            predictors = row['predictors']
            model_type = row['model_type']
            coeff_comp = row['coefficient_comparison']

            # Extract coefficient changes
            coef_data = self._prepare_coefficient_change_data(coeff_comp, predictors)

            if not coef_data:
                ax.text(0.5, 0.5, f'No coefficient data for {target}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target} - Coefficient Changes', fontweight='bold')
                continue

            # Plot based on show_effects
            if show_effects == 'both':
                self._plot_stacked_changes(ax, coef_data, target, model_type)
            elif show_effects in ['distribution', 'fuel', 'total']:
                self._plot_single_effect(ax, coef_data, target, model_type, show_effects)
            else:
                raise ValueError(f"Invalid show_effects: {show_effects}. "
                                 f"Must be 'both', 'distribution', 'fuel', or 'total'")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Coefficient changes plot saved to {save_path}")

        plt.show()

    def _prepare_coefficient_change_data(self, coeff_comp: Dict, predictors) -> list[Dict]:
        """
        Prepare coefficient change data for plotting.

        Parameters
        ----------
        coeff_comp : dict
            Coefficient comparison dictionary
        predictors : tuple of str or str
            Predictor names for labeling

        Returns
        -------
        list of dict
            List of coefficient data with labels and changes
        """
        coef_data = []

        # Get all coefficient keys from total_effect
        if 'total_effect' not in coeff_comp:
            return []

        coef_keys = list(coeff_comp['total_effect'].keys())

        # ✅ Handle both tuple and string formats
        if isinstance(predictors, str):
            predictor_list = [p.strip() for p in predictors.split('+')]
        else:
            predictor_list = list(predictors)

        # Sort: intercept first, then by order they appear
        def sort_key(key):
            key_lower = str(key).lower()

            # Intercept always first
            if 'intercept' in key_lower or 'const' in key_lower or key == '1':
                return (0, 0, '')

            # Try to extract numeric index if present
            import re
            numbers = re.findall(r'\d+', str(key))
            if numbers:
                return (1, int(numbers[0]), key)

            # Otherwise sort alphabetically
            return (2, 0, key)

        sorted_keys = sorted(coef_keys, key=sort_key)

        # Create labels
        for key in sorted_keys:
            key_lower = str(key).lower()

            # Check if intercept
            if 'intercept' in key_lower or 'const' in key_lower or key == '1':
                label = 'intercept'
            else:
                # Use the actual key name (polynomial feature name)
                label = str(key)

            # Get the changes
            dist_change = coeff_comp['distribution_effect'][key]['absolute_change']
            fuel_change = coeff_comp['fuel_effect'][key]['absolute_change']
            total_change = coeff_comp['total_effect'][key]['absolute_change']

            coef_data.append({
                'label': label,
                'key': key,
                'distribution': dist_change,
                'fuel': fuel_change,
                'total': total_change
            })

        return coef_data

    def _plot_stacked_changes(self, ax, coef_data: list[dict], target: str, model_type: str):
        """
        Plot stacked bar chart showing distribution and fuel effects.

        Parameters
        ----------
        ax : matplotlib Axes
            Subplot axes
        coef_data : list of dict
            Coefficient change data
        target : str
            Target variable name
        model_type : str
            Model type
        """
        labels = [d['label'] for d in coef_data]
        dist_changes = np.array([d['distribution'] for d in coef_data])
        fuel_changes = np.array([d['fuel'] for d in coef_data])

        y_pos = np.arange(len(labels))

        # Create stacked horizontal bars
        # Distribution effect (green) - from 0 to distribution change
        bars_dist = ax.barh(y_pos, dist_changes, 0.6,
                            label='Distribution Effect',
                            color='green', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Fuel effect (red) - stacked on distribution
        bars_fuel = ax.barh(y_pos, fuel_changes, 0.6, left=dist_changes,
                            label='Fuel Effect',
                            color='red', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add value labels at the end of bars (total change)
        for i, (pos, dist, fuel) in enumerate(zip(y_pos, dist_changes, fuel_changes)):
            total_change = dist + fuel
            # Position label at end of bar
            label_x = total_change
            ha = 'left' if total_change >= 0 else 'right'

            # Calculate offset as percentage of x-range
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            offset = 0.02 * x_range if x_range > 0 else 0.1
            label_x += offset if total_change >= 0 else -offset

            ax.text(label_x, pos, f'{total_change:+.3f}',
                    ha=ha, va='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
        ax.set_xlabel('Change (std dev)' if self.normalize else 'Coefficient Change',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{target} - Coefficient Changes',
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_single_effect(self, ax, coef_data: list[dict],
                            target: str, model_type: str, effect: str):
        """
        Plot single effect (distribution, fuel, or total).

        Parameters
        ----------
        ax : matplotlib Axes
            Subplot axes
        coef_data : list of dict
            Coefficient change data
        target : str
            Target variable name
        model_type : str
            Model type
        effect : str
            Which effect to plot ('distribution', 'fuel', or 'total')
        """
        labels = [d['label'] for d in coef_data]
        changes = np.array([d[effect] for d in coef_data])

        y_pos = np.arange(len(labels))

        # Color mapping
        colors = {
            'distribution': 'green',
            'fuel': 'red',
            'total': 'steelblue'
        }

        titles = {
            'distribution': 'Distribution Effect',
            'fuel': 'Fuel Effect',
            'total': 'Total Effect'
        }

        # Create horizontal bars
        bars = ax.barh(y_pos, changes, 0.6,
                       color=colors[effect], alpha=0.7,
                       edgecolor='black', linewidth=0.5)

        # Add value labels at the end of bars
        for i, (pos, change) in enumerate(zip(y_pos, changes)):
            label_x = change
            ha = 'left' if change >= 0 else 'right'
            offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            label_x += offset if change >= 0 else -offset

            ax.text(label_x, pos, f'{change:+.3f}',
                    ha=ha, va='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
        ax.set_xlabel('Change (std dev)' if self.normalize else 'Coefficient Change',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{target} - {titles[effect]}',
                     fontweight='bold', fontsize=12, pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)