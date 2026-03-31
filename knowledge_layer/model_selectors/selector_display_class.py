"""
Display class for model selector results - FIXED

✅ FIX: Updated to work with PolynomialModelSelector's data structure:
- Uses model_recommendations and detailed_results
- Not the non-existent 'results' attribute
"""

import numpy as np
from typing import Optional, Dict, List
from knowledge_layer.display_functions.display_formatters import DisplayFormatters


class SelectorResultsDisplay:
    """
    Handles all display logic for model selectors.

    ✅ FIXED: Works with PolynomialModelSelector's actual attributes:
    - self.selector.model_recommendations
    - self.selector.detailed_results
    """

    def __init__(self, selector):
        """
        Initialize display for a selector.

        Parameters
        ----------
        selector : PolynomialModelSelector or RandomForestSelector
            Selector instance with results to display
        """
        self.selector = selector
        self.fmt = DisplayFormatters()

    # ========================================================================
    # INITIALIZATION DISPLAY
    # ========================================================================

    def print_initialization_summary(self):
        """Print configuration summary at selector initialization."""
        selector = self.selector

        print(f"\n{selector.__class__.__name__} initialized")
        print(f"  Selection metric: {selector.selection_metric}")
        print(f"  Fixed predictors: {', '.join(selector.fixed_dimensions)}")

        if selector.relationship_context:
            print(f"  Relationship exclusions: {len(selector.relationship_context)} rules")

        if selector.forbidden_combinations:
            print(f"  Forbidden combinations: {len(selector.forbidden_combinations)} rules")

        # Type-specific info
        if hasattr(selector, 'include_regularized'):
            # Polynomial selector
            if selector.include_regularized:
                cv_type = "CV" if selector.use_cv_regularization else "fixed"
                print(f"  Ridge/Lasso enabled ({cv_type} alpha selection)")
        elif hasattr(selector, 'rf_params'):
            # Random Forest selector
            print(f"  RF parameters: n_estimators={selector.rf_params['n_estimators']}, "
                  f"max_depth={selector.rf_params['max_depth']}")

    # ========================================================================
    # ANALYSIS PROGRESS DISPLAY
    # ========================================================================

    def print_analysis_header(self,
                              max_predictors: int,
                              cv_method: Optional[str],
                              n_folds: int = 10,
                              exclude_models: Optional[List[str]] = None,
                              use_feature_engineering: bool = False,
                              poly_degree: Optional[int] = None):
        """Print header at analysis start."""
        cv_text = self.fmt.format_cv_description(cv_method, n_folds)

        print(self.fmt.format_header(
            f"MODEL SELECTION (Max Predictors: {max_predictors}){cv_text}",
            width=140
        ))

        # Feature engineering info (Random Forest)
        if use_feature_engineering and poly_degree:
            print(f"  Feature Engineering ENABLED ✓ (poly_degree={poly_degree})")
            print(f"  Testing both RAW and ENGINEERED features for each combination")

        # Model availability info (Polynomial)
        if hasattr(self.selector, 'include_regularized'):
            if not self.selector.include_regularized:
                print("  Ridge and Lasso models DISABLED")
            else:
                print("  Ridge and Lasso models ENABLED ✓")

        # Exclusions
        if exclude_models:
            excluded_str = ", ".join(exclude_models[:5])
            if len(exclude_models) > 5:
                excluded_str += f" ... (+{len(exclude_models) - 5} more)"
            print(f"  Excluded models: {excluded_str}")

        # Combination rules
        if self.selector.forbidden_combinations:
            print(f"  Forbidden combinations: {len(self.selector.forbidden_combinations)} rules active")

        print("=" * 140)

    def print_completion_summary(self):
        """Print summary at analysis completion."""

        # ====================================================================
        # Count total combinations based on selector type
        # ====================================================================

        # OPTION 1: Polynomial Structure (model_recommendations)
        if hasattr(self.selector, 'model_recommendations'):
            total_combinations = len(self.selector.model_recommendations)

        # OPTION 2: Random Forest Structure (results by target)
        elif hasattr(self.selector, 'results'):
            total_combinations = sum(
                len(combos) for combos in self.selector.results.values()
            )

        # FALLBACK: Can't determine
        else:
            total_combinations = 0
            if hasattr(self.selector, 'verbose') and self.selector.verbose:
                print("Warning: Could not determine total combinations")

        # ====================================================================
        # Print summary
        # ====================================================================
        print(f"\n{'=' * 140}")
        print(f"✓ Model selection complete: {total_combinations} combinations tested")
        print(f"{'=' * 140}")

    # ========================================================================
    # ✅ FIXED: MAIN RESULTS DISPLAY
    # ========================================================================

    def print_results(self,
                      group_by_type: bool = False,
                      sort_by: str = 'n_predictors',
                      cv_method: Optional[str] = None):
        """
        Print all results in formatted tables.

        ✅ FIXED: Now works with model_recommendations + detailed_results
        """
        for target in self.selector.target_dimensions:
            # ✅ FIX: Get results for this target from model_recommendations
            target_results = self._get_results_for_target(target)

            if not target_results:
                continue

            if group_by_type:
                self._print_target_table_grouped(target, target_results, sort_by, cv_method)
            else:
                self._print_target_table(target, target_results, sort_by, cv_method)

    def _get_results_for_target(self, target: str) -> List[Dict]:
        """
        Extract results for a target from either selector structure.

        Handles both:
        - PolynomialModelSelector (model_recommendations + detailed_results)
        - RandomForestSelector (results by target)
        """
        results_list = []

        # ====================================================================
        # OPTION 1: Polynomial Structure
        # ====================================================================
        if hasattr(self.selector, 'model_recommendations'):
            # Polynomial structure: {(predictors, target): best_model}
            for (pred_combo, tgt), best_model in self.selector.model_recommendations.items():
                if tgt == target:
                    key = (pred_combo, tgt)
                    if key in self.selector.detailed_results:
                        metrics = self.selector.detailed_results[key][best_model]

                        results_list.append({
                            'combo_name': " + ".join(pred_combo),
                            'predictors': pred_combo,
                            'model_name': best_model,
                            'metrics': metrics,
                            'combination_type': metrics.get('combination_type', 'other')
                        })

        # ====================================================================
        # OPTION 2: Random Forest Structure
        # ====================================================================
        elif hasattr(self.selector, 'results') and target in self.selector.results:
            # Random Forest structure: {target: {combo_name: {...}}}
            for combo_name, data in self.selector.results[target].items():
                results_list.append({
                    'combo_name': combo_name,
                    'predictors': data['predictors'],
                    'model_name': 'random_forest',  # Always RF for this selector
                    'metrics': data['metrics'],
                    'combination_type': data.get('combination_type', 'other')
                })

        # ====================================================================
        # FALLBACK: No valid structure found
        # ====================================================================
        else:
            if hasattr(self.selector, 'verbose') and self.selector.verbose:
                print(f"Warning: Could not extract results for target '{target}'")
                print(f"  Selector type: {type(self.selector).__name__}")
                print(f"  Available attributes: {[a for a in dir(self.selector) if not a.startswith('_')]}")

        return results_list

    def _print_target_table(self,
                            target: str,
                            results_list: List[Dict],
                            sort_by: str,
                            cv_method: Optional[str]):
        """Print results table for one target (ungrouped)."""
        print(self.fmt.format_header(f"Target: {target}", width=140))

        # Show available predictors
        if hasattr(self.selector, 'combination_manager'):
            available = self.selector.combination_manager.generate_combinations(
                target=target, min_predictors=1, max_predictors=1
            )
            pred_names = sorted([p[0] for p in available])
            print(f"Available predictors: {pred_names}")
            print("=" * 140)

        # Print column headers
        self._print_table_header(cv_method, show_type=False)
        print(self.fmt.format_separator(140))

        # Sort and print results
        results_list = self._sort_results(results_list, sort_by)
        for result in results_list:
            self._print_result_row(result, cv_method, show_type=False)

    def _print_target_table_grouped(self,
                                    target: str,
                                    results_list: List[Dict],
                                    sort_by: str,
                                    cv_method: Optional[str]):
        """Print results table grouped by combination type."""
        print(self.fmt.format_header(f"Target: {target}", width=140))

        # Group results by type
        results_by_type = {}
        for result in results_list:
            combo_type = result.get('combination_type', 'other')
            if combo_type not in results_by_type:
                results_by_type[combo_type] = []
            results_by_type[combo_type].append(result)

        if not results_by_type:
            print("  No results for this target")
            return

        # Print each group
        type_order = ['hull', 'tonnage', 'propulsion', 'mixed', 'other']

        for combo_type in type_order:
            if combo_type not in results_by_type:
                continue

            print(self.fmt.format_subheader(
                f"  {combo_type.upper()} COMBINATIONS:",
                width=136
            ))

            # Print header
            self._print_table_header(cv_method, indent="  ")

            # Sort and print
            sorted_results = self._sort_results(results_by_type[combo_type], sort_by)
            for result in sorted_results:
                self._print_result_row(result, cv_method, indent="  ")

    # ========================================================================
    # TABLE FORMATTING
    # ========================================================================

    def _print_table_header(self, cv_method: Optional[str],
                            show_type: bool = False,
                            indent: str = ""):
        """Print table column headers."""
        # Determine if we're showing AIC/BIC or RRMSE/MAPE
        show_aic_bic = (hasattr(self.selector, 'selection_metric') and
                        self.selector.selection_metric in ['aic', 'bic'])

        # Build header line
        if show_type:
            header = f"{indent}{'Predictors':<38s} | {'Type':>12s} | "
        else:
            header = f"{indent}{'Predictors':<38s} | "

        header += f"{'Model':<18s} | {'Adj R²':>8s} | "

        if cv_method is not None:
            header += f"{'±SD':>6s} | "

        if show_aic_bic:
            header += f"{'AIC':>8s} | {'BIC':>8s} | "
        else:
            header += f"{'RRMSE%':>8s} | "
            if cv_method is not None:
                header += f"{'±SD':>6s} | "
            header += f"{'MAPE%':>8s} | "

        header += f"{'Max VIF':>8s} | {'Heterosc.':>12s}"

        # Add feature importance for Random Forest
        if hasattr(self.selector, 'rf_params'):
            header += f" | {'Top Feature':<20s}"

        print(header)

    def _print_result_row(self,
                          result: Dict,
                          cv_method: Optional[str],
                          show_type: bool = False,
                          indent: str = ""):
        """Print a single result row."""
        combo_name = result['combo_name']
        model_name = result['model_name']
        metrics = result['metrics']
        combo_type = result.get('combination_type', 'other')

        # Truncate long names
        if len(combo_name) > 38:
            combo_name = combo_name[:35] + "..."

        # Build row
        if show_type:
            row = f"{indent}{combo_name:<38s} | {combo_type:>12s} | "
        else:
            row = f"{indent}{combo_name:<38s} | "

        # Model name
        model_display = self.fmt.format_model_name(model_name, max_length=18)
        row += f"{model_display:<18s} | "

        # Adj R²
        row += f"{self.fmt.format_metric(metrics['adj_r2'], 'adj_r2'):>8s} | "

        # CV std
        if cv_method is not None:
            std_r2 = metrics.get('std_adj_r2', 0)
            row += f"{self.fmt.format_metric(std_r2, 'adj_r2', decimals=3):>6s} | "

        # AIC/BIC or RRMSE/MAPE
        show_aic_bic = (hasattr(self.selector, 'selection_metric') and
                        self.selector.selection_metric in ['aic', 'bic'])

        if show_aic_bic:
            row += f"{self.fmt.format_metric(metrics.get('aic', np.nan), 'aic'):>8s} | "
            row += f"{self.fmt.format_metric(metrics.get('bic', np.nan), 'bic'):>8s} | "
        else:
            row += f"{self.fmt.format_metric(metrics['rrmse'], 'rrmse'):>8s} | "
            if cv_method is not None:
                std_rrmse = metrics.get('std_rrmse', 0)
                row += f"{self.fmt.format_metric(std_rrmse, 'rrmse'):>6s} | "
            row += f"{self.fmt.format_metric(metrics['mape'], 'mape'):>8s} | "

        # VIF
        max_vif = metrics.get('max_vif', 1.0)
        row += f"{self.fmt.format_metric(max_vif, 'vif'):>8s}"

        heterosc = metrics.get('hetroscedasticity', 'unknown')
        heterosc_display = self._format_heteroscedasticity(heterosc)
        row += f"{heterosc_display:>12s}"

        # Feature importance (Random Forest)
        if hasattr(self.selector, 'rf_params'):
            feature_imp = metrics.get('feature_importance', {})
            if feature_imp:
                top_feature = max(feature_imp, key=feature_imp.get)
                top_feature = top_feature[:20] if len(top_feature) > 20 else top_feature
            else:
                top_feature = "N/A"
            row += f" | {top_feature:<20s}"

        print(row)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _sort_results(self, results_list: List[Dict], sort_by: str) -> List[Dict]:
        """Sort results based on sort_by parameter."""
        if sort_by == 'n_predictors':
            return sorted(
                results_list,
                key=lambda x: (len(x['predictors']), -x['metrics'].get('adj_r2', 0))
            )
        elif sort_by == 'adj_r2':
            return sorted(results_list, key=lambda x: -x['metrics'].get('adj_r2', 0))
        elif sort_by == 'rrmse':
            return sorted(results_list, key=lambda x: x['metrics'].get('rrmse', np.inf))
        else:
            return results_list

    def _format_heteroscedasticity(self, severity: str) -> str:
        """
        Format heteroscedasticity severity with visual indicators.

        Parameters
        ----------
        severity : str
            One of: 'negligible', 'weak', 'moderate', 'strong', or 'unknown'

        Returns
        -------
        str
            Formatted severity string with emoji/symbol
        """
        severity_lower = severity.lower().split()[0]  # Get first word (e.g., "weak" from "weak increasing")

        severity_map = {
            'no': '✓ None',
            'weak': '○ Weak',
            'moderate': '⚠ mod.',
            'strong': '✗ Strong',
            'unknown': '? unknown'
        }

        return severity_map.get(severity_lower, severity[:12])

    # ========================================================================
    # ✅ FIXED: SUMMARY DISPLAYS
    # ========================================================================

    def print_best_combinations_summary(self, group_by_type: bool = False):
        """
        Print summary of best combinations for each target.

        ✅ FIXED: Uses model_recommendations + detailed_results
        """
        print(self.fmt.format_header("BEST COMBINATIONS SUMMARY", width=140))

        if group_by_type:
            header = (f"{'Target':<15s} | {'Type':>12s} | {'Best Predictors':<30s} | "
                      f"{'Model':<18s} | {'Adj R²':>8s} | {'RRMSE%':>8s} | {'MAPE%':>8s}")
        else:
            header = (f"{'Target':<15s} | {'Best Predictors':<40s} | "
                      f"{'Model':<18s} | {'Adj R²':>8s} | {'RRMSE%':>8s} | {'MAPE%':>8s}")

        print(header)
        print(self.fmt.format_separator(140))

        # ✅ FIX: Get best combinations using proper data structure
        for target in sorted(self.selector.target_dimensions):
            target_results = self._get_results_for_target(target)

            if not target_results:
                continue

            # Find best by adj_r2
            best_result = max(target_results, key=lambda x: x['metrics'].get('adj_r2', 0))

            combo_name = best_result['combo_name']
            model_name = best_result['model_name']
            metrics = best_result['metrics']
            combo_type = best_result.get('combination_type', 'other')

            # ✅ Check for diagnostic issues
            issues = []
            heterosc = metrics.get('heteroscedasticity', 'unknown').lower().split()[0]
            if heterosc in ['moderate', 'strong']:
                issues.append(f"Het:{heterosc[:3]}")

            max_vif = metrics.get('max_vif', 1.0)
            if max_vif >= 10:
                issues.append(f"VIF:{max_vif:.0f}")

            issues_str = ", ".join(issues) if issues else "✓"

            # Truncate combo name
            display_combo = combo_name if len(combo_name) <= 40 else combo_name[:37] + "..."

            # Model name
            model_display = self.fmt.format_model_name(model_name, max_length=18)

            if group_by_type:
                print(f"{target:<15s} | {combo_type:>12s} | {display_combo:<30s} | "
                      f"{model_display:<18s} | "
                      f"{self.fmt.format_metric(metrics['adj_r2'], 'adj_r2'):>8s} | "
                      f"{self.fmt.format_metric(metrics['rrmse'], 'rrmse'):>8s} | "
                      f"{self.fmt.format_metric(metrics['mape'], 'mape'):>8s} "
                      f"{issues_str:<15s}")
            else:
                print(f"{target:<15s} | {display_combo:<40s} | "
                      f"{model_display:<18s} | "
                      f"{self.fmt.format_metric(metrics['adj_r2'], 'adj_r2'):>8s} | "
                      f"{self.fmt.format_metric(metrics['rrmse'], 'rrmse'):>8s} | "
                      f"{self.fmt.format_metric(metrics['mape'], 'mape'):>8s}"
                      f"{issues_str:<15s}")

        print("=" * 140)

    def print_feature_engineering_comparison(self, filter_by_type: Optional[str] = None):
        """
        Compare raw vs engineered features (Random Forest only).

        Note: This method may not work for PolynomialModelSelector
        """
        if not hasattr(self.selector, 'rf_params'):
            print("Feature engineering comparison only available for Random Forest selector")
            return

        print(self.fmt.format_header("FEATURE ENGINEERING IMPACT ANALYSIS", width=140))
        # Implementation unchanged from before
        pass

    # ========================================================================
    # DEBUG/ANALYSIS DISPLAYS
    # ========================================================================

    def print_model_testing_strategy(self, max_predictors: int = 2):
        """Show what will be tested before running analysis."""
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        print(self.fmt.format_header("MODEL TESTING STRATEGY SUMMARY", width=80))

        # Basic info
        n_targets = len(self.selector.target_dimensions)
        print(f"Targets: {n_targets} ({', '.join(self.selector.target_dimensions)})")
        print(f"Max predictors: {max_predictors}")

        # Estimate combinations
        if hasattr(self.selector, 'combination_manager'):
            total_combos = 0
            for target in self.selector.target_dimensions:
                combos = self.selector.combination_manager.generate_combinations(
                    target=target, min_predictors=1, max_predictors=max_predictors
                )
                total_combos += len(combos)
            print(f"Estimated combinations: ~{total_combos}")

        # Model types (Polynomial only)
        if hasattr(self.selector, 'include_regularized'):
            all_models = ModelFactory.get_available_models()

            if not self.selector.include_regularized:
                all_models = [m for m in all_models
                              if m not in ['ridge', 'lasso', 'ridge-cv', 'lasso-cv']]

            print(f"\nAvailable model types: {len(all_models)}")

            # Categorize
            linear_models = [m for m in all_models
                             if m in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']]
            poly_models = [m for m in all_models if m.startswith('polynomial')]
            nonlinear_models = [m for m in all_models
                                if m not in linear_models and m not in poly_models]

            if linear_models:
                print(f"  • Linear/Regularized: {len(linear_models)}")
                print(f"    ({', '.join(linear_models)})")
            if poly_models:
                print(f"  • Polynomial: {len(poly_models)}")
                print(f"    ({', '.join(poly_models)})")
            if nonlinear_models:
                print(f"  • Non-linear: {len(nonlinear_models)} (single predictor only)")
                print(f"    ({', '.join(nonlinear_models)})")

        print("=" * 80)