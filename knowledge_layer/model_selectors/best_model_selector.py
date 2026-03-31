"""
Best Model Selector
===================
Handles model selection logic using various criteria and strategies.

Supports:
- Simple metric-based selection (adj_r2, AIC, BIC, RRMSE, MAPE)
- Tie-breaking when models are equivalent
- Nested model considerations (Linear → Poly-2 → Poly-3)
- Optional sequential testing for nested models
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats


class BestModelSelector:
    """
    Handles model selection across different regression model types.

    This class encapsulates the logic for selecting the best model from
    a set of fitted models, supporting multiple selection criteria and
    strategies for handling nested model structures.

    Concepts Explained
    ------------------

    **Nested Models:**
    Models are nested when one is a special case of another by setting
    parameters to zero. Linear ⊂ Polynomial-2 ⊂ Polynomial-3.

    **Information Criteria (AIC/BIC):**
    - Lower is better
    - Automatically penalize complexity
    - ΔAIC/BIC < 2: Models essentially equivalent
    - ΔAIC/BIC 2-6: Positive evidence
    - ΔAIC/BIC > 10: Very strong evidence

    **Selection Strategies:**
    - 'simple': Just pick best by chosen metric
    - 'conservative': Use tie-breaking threshold (ΔBIC < 2 → pick simpler)
    - 'nested': Special handling for Linear/Poly hierarchy
    """

    # Define which models are nested
    NESTED_MODEL_ORDER = [
        'linear',
        'polynomial-2-lasso',  # ✅ Regularized poly-2 with selection
        'polynomial-2',
        'polynomial-3-lasso'
        'polynomial-3'
    ]

    # Equivalence thresholds
    BIC_EQUIVALENCE_THRESHOLD = 2.0
    AIC_EQUIVALENCE_THRESHOLD = 2.0

    # LRT significance level
    DEFAULT_LRT_ALPHA = 0.05

    def __init__(self,
                 selection_metric: str = 'adj_r2',
                 selection_strategy: str = 'simple',
                 assumptions:str='flexible',
                 lrt_alpha: float = 0.05,
                 vif_threshold: float = 10.0,
                 verbose: bool = True):
        """
        Initialize the model selector.

        Parameters
        ----------
        selection_metric : str, default='adj_r2'
            Primary metric to use for selection:
            - 'adj_r2': Adjusted R² (higher is better)
            - 'r2': R² (higher is better)
            - 'aic': Akaike Information Criterion (lower is better)
            - 'bic': Bayesian Information Criterion (lower is better)
            - 'rrmse': Relative RMSE% (lower is better)
            - 'mape': Mean Absolute Percentage Error% (lower is better)

        selection_strategy : str, default='simple'
            Strategy for model selection:
            - 'simple': Pick best by metric, no special handling
            - 'conservative': Use equivalence thresholds, prefer simpler
            - 'nested_aware': Special handling for nested models
            - 'sequential_lrt': Sequential likelihood ratio tests for nested models
            - 'lrt_then_compare': LRT for nested, then compare to others by metric

        assumptions : str, default='flexible'
            Specify the extent to which selector must adhere to assumptions of homoscedasticity and multicolinearity
                - None, assumptions disregarded.
                - flexible - allow some tolerence of breaking assumptions
                - strict - Allow no tolerance of breaking assumptions

        lrt_alpha : float, default=0.05
            Significance level for likelihood ratio tests

        vif_threshold: float = 10.0,
            Maximum acceptable VIF for multicolinearity

        verbose : bool, default=True
            Whether to print informational messages
        """
        self.selection_metric = selection_metric
        self.selection_strategy = selection_strategy
        self.lrt_alpha = lrt_alpha
        self.assumptions = assumptions
        self.vif_threshold = vif_threshold
        self.verbose = verbose

        # Validate inputs
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_metrics = ['adj_r2', 'r2', 'aic', 'bic', 'rrmse', 'mape']
        if self.selection_metric not in valid_metrics:
            raise ValueError(
                f"Invalid selection_metric: {self.selection_metric}. "
                f"Must be one of {valid_metrics}"
            )

        valid_strategies = ['simple', 'conservative', 'nested_aware',
                            'sequential_lrt', 'lrt_then_compare']
        if self.selection_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid selection_strategy: {self.selection_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        if not 0 < self.lrt_alpha < 1:
            raise ValueError(
                f"Invalid lrt_alpha: {self.lrt_alpha}. "
                f"Must be between 0 and 1"
            )

    def _apply_assumption(self, model_results: dict[str, dict]) -> dict[str, dict]:
        """
        Filter models based on assumption adherence level.

        Checks:
        - Heteroscedasticity (Spearman, Breusch-Pagan, White tests)
        - Multicollinearity (VIF)

        Parameters
        ----------
        model_results : dict
            Dictionary of {model_name: metrics_dict}

        Returns
        -------
        dict
            Filtered dictionary containing only models meeting assumption criteria
        """

        if self.assumptions is None or self.assumptions == 'flexible':
            return model_results

        # ========================================================================
        # Categorize models by assumption violations
        # ========================================================================

        strict_compliant = {}  # No violations at all
        moderate_compliant = {}  # Minor violations allowed

        for model_name, metrics in model_results.items():
            # Get diagnostic flags
            spearman_het = metrics.get('spearman', False)
            bp_het = metrics.get('breusch_pagan', False)
            white_het = metrics.get('white', False)

            # Count violations
            het_violations = sum([spearman_het, bp_het, white_het])

            # Strict: NO violations
            if het_violations == 0:
                strict_compliant[model_name] = metrics

            # Moderate: Allow minor violations
            # - Heteroscedasticity detected by at most 1 test (not 2+)
            # - OR VIF slightly elevated but not extreme (VIF < 15)
            moderate_ok = (het_violations <= 1)
            if moderate_ok:
                moderate_compliant[model_name] = metrics

        # ========================================================================
        # Apply filtering based on assumption level
        # ========================================================================

        if self.assumptions == 'strict':
            if len(strict_compliant) >= 1:
                if self.verbose:
                    n_removed = len(model_results) - len(strict_compliant)
                    if n_removed > 0:
                        print(f"  ℹ️  Strict assumptions: {n_removed} models excluded")
                        self._print_violation_summary(model_results, strict_compliant)
                return strict_compliant

            elif len(moderate_compliant) >= 1:
                if self.verbose:
                    print(f"  ⚠️  No models meet strict assumptions, using moderate")
                    n_removed = len(model_results) - len(moderate_compliant)
                    if n_removed > 0:
                        print(f"      {n_removed} models excluded")
                return moderate_compliant

            else:
                if self.verbose:
                    print(f"  ⚠️  No models meet any assumptions, using all models")
                return model_results

        elif self.assumptions == 'moderate':
            if len(moderate_compliant) >= 1:
                if self.verbose:
                    n_removed = len(model_results) - len(moderate_compliant)
                    if n_removed > 0:
                        print(f"  ℹ️  Moderate assumptions: {n_removed} models excluded")
                        self._print_violation_summary(model_results, moderate_compliant)
                return moderate_compliant

            else:
                if self.verbose:
                    print(f"  ⚠️  No models meet moderate assumptions, using all")
                return model_results

        return model_results

    def _print_violation_summary(self,
                                 all_models: dict[str, dict],
                                 compliant_models: dict[str, dict]):
        """Print summary of why models were excluded."""
        excluded = {k: v for k, v in all_models.items() if k not in compliant_models}

        if not excluded:
            return

        print(f"      Excluded models:")
        for model_name, metrics in excluded.items():
            violations = []

            if metrics.get('spearman', False):
                violations.append("Spearman")
            if metrics.get('breusch_pagan', False):
                violations.append("BP")
            if metrics.get('white', False):
                violations.append("White")

            max_vif = metrics.get('max_vif', 1.0)
            if max_vif > self.vif_threshold:
                violations.append(f"VIF={max_vif:.1f}")

            violation_str = ", ".join(violations) if violations else "unknown"
            print(f"        • {model_name}: {violation_str}")

        # metrics['spearman'] = het_scedast_result['spearman']['overall']['heteroscedastic']
        # metrics['breusch_pagan'] = het_scedast_result['breusch_pagan']['heteroscedastic']
        # metrics['white'] = het_scedast_result['white']['heteroscedastic']
    def select_best_model(self,
                         model_results: dict[str, dict]) -> Tuple[str, dict]:
        """
        Select the best model from results dictionary.

        Parameters
        ----------
        model_results : dict
            Dictionary mapping model names to their metrics:
            {
                'linear': {'r2': 0.85, 'adj_r2': 0.84, 'bic': 1234.5, ...},
                'polynomial-2': {'r2': 0.92, 'adj_r2': 0.91, 'bic': 1230.1, ...},
                ...
            }

        Returns
        -------
        best_model_name : str
            Name of selected model
        best_model_metrics : dict
            Metrics dictionary for selected model

        """
        if not model_results:
            return None, None

        model_results = self._apply_assumption(model_results)

        # Route to appropriate selection method
        if self.selection_strategy == 'simple':
            return self._select_simple(model_results)
        elif self.selection_strategy == 'conservative':
            return self._select_conservative(model_results)
        elif self.selection_strategy == 'nested_aware':
            return self._select_nested_aware(model_results)

    # ========================================================================
    # SELECTION STRATEGIES
    # ========================================================================

    def _select_simple(self,
                      model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Simple selection: just pick best by metric.

        This is the straightforward approach - fit all models,
        calculate metric, pick the best one.
        """
        # Metrics where HIGHER is better
        if self.selection_metric in ['adj_r2', 'r2']:
            return self._select_maximize(model_results, self.selection_metric)

        # Metrics where LOWER is better
        elif self.selection_metric in ['aic', 'bic', 'rrmse', 'mape']:
            return self._select_minimize(model_results, self.selection_metric)

    def _select_conservative(self,
                            model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Conservative selection with tie-breaking.

        When models are essentially equivalent (within threshold),
        prefer the simpler model.

        Concept
        -------
        If two models have BIC within 2 points of each other, they're
        statistically equivalent. In this case, prefer the simpler model
        (fewer parameters) to avoid overfitting.
        """
        if self.selection_metric == 'bic':
            return self._select_bic_conservative(model_results)
        elif self.selection_metric == 'aic':
            return self._select_aic_conservative(model_results)
        else:
            # For other metrics, fall back to simple selection
            return self._select_simple(model_results)

    def _select_nested_aware(self,
                            model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Nested-aware selection: special handling for Linear/Poly hierarchy.

        Strategy:
        1. Separate nested models (Linear/Poly-2/Poly-3) from others
        2. Select best nested model conservatively (with threshold)
        3. Compare nested winner to all non-nested models
        4. Pick overall best

        Concept
        -------
        Nested models have special structure. We want to be conservative
        about adding complexity within the nested family, but still allow
        a completely different model (exponential, power) to win if it's
        clearly better.
        """
        # Separate nested from non-nested models
        nested_models = {}
        non_nested_models = {}

        for name, metrics in model_results.items():
            if name in self.NESTED_MODEL_ORDER:
                nested_models[name] = metrics
            else:
                non_nested_models[name] = metrics

        # Phase 1: Select among nested models conservatively
        nested_winner_name = None
        nested_winner_metrics = None

        if nested_models:
            if self.selection_metric in ['bic', 'aic']:
                nested_winner_name, nested_winner_metrics = \
                    self._select_nested_conservative(nested_models)
            else:
                # For other metrics, just pick best nested
                nested_winner_name, nested_winner_metrics = \
                    self._select_simple(nested_models)

        # Phase 2: Compare to all other models
        all_candidates = {}
        if nested_winner_name:
            all_candidates[nested_winner_name] = nested_winner_metrics
        all_candidates.update(non_nested_models)

        # Select overall best
        if self.selection_metric in ['bic', 'aic']:
            return self._select_minimize(all_candidates, self.selection_metric)
        else:
            return self._select_simple(all_candidates)

    def _select_sequential_lrt(self, model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Sequential Likelihood Ratio Test selection.

        Only considers nested models. Uses formal hypothesis testing
        to decide when to add complexity.
        """
        nested_models = {
            name: metrics for name, metrics in model_results.items()
            if name in self.NESTED_MODEL_ORDER
        }

        if not nested_models:
            if self.verbose:
                print("⚠️  No nested models available for LRT, using fallback")
            return self._select_fallback(model_results)

        return self._sequential_lrt_nested(nested_models)

    def _select_lrt_then_compare(self, model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Two-phase selection: LRT for nested, then compare to others.

        This combines hypothesis testing for nested models with
        information criteria for comparing across model families.
        """
        nested_models = {
            name: metrics for name, metrics in model_results.items()
            if name in self.NESTED_MODEL_ORDER
        }

        non_nested_models = {
            name: metrics for name, metrics in model_results.items()
            if name not in self.NESTED_MODEL_ORDER
        }

        # Phase 1: LRT for nested models
        if nested_models:
            nested_winner_name, nested_winner_metrics = \
                self._sequential_lrt_nested(nested_models)
        else:
            nested_winner_name = None
            nested_winner_metrics = None

        # Phase 2: Compare to non-nested models
        all_candidates = {}
        if nested_winner_name:
            all_candidates[nested_winner_name] = nested_winner_metrics
        all_candidates.update(non_nested_models)

        return self._select_simple(all_candidates)

    # ========================================================================
    # LIKELIHOOD RATIO TEST METHODS
    # ========================================================================

    def _sequential_lrt_nested(self, nested_models: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Perform sequential LRT through nested model hierarchy.

        Tests: Linear vs Poly-2, then Poly-2 vs Poly-3 (if Poly-2 wins)
        """
        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"Sequential LRT (α = {self.lrt_alpha:.3f})")
            print(f"{'─' * 60}")

        selected_name = None
        selected_metrics = None

        for model_name in self.NESTED_MODEL_ORDER:
            if model_name not in nested_models:
                continue

            if selected_name is None:
                # First available model becomes baseline
                selected_name = model_name
                selected_metrics = nested_models[model_name]

                if self.verbose:
                    print(f"Starting with: {selected_name}")
                continue

            # Test current selection vs next level
            candidate_name = model_name
            candidate_metrics = nested_models[model_name]

            # Perform LRT
            is_significant, lr_stat, p_value, df = self._likelihood_ratio_test(
                simple_metrics=selected_metrics,
                complex_metrics=candidate_metrics,
                simple_name=selected_name,
                complex_name=candidate_name
            )

            if self.verbose:
                print(f"\nTest: {selected_name} vs {candidate_name}")
                print(f"  LR: {lr_stat:.3f}, df: {df}, p: {p_value:.4f}")

            if is_significant:
                if self.verbose:
                    print(f"  → Significant! Accept {candidate_name}")
                selected_name = candidate_name
                selected_metrics = candidate_metrics
            else:
                if self.verbose:
                    print(f"  → Not significant. Keep {selected_name}")
                break  # Stop testing more complex models

        if self.verbose:
            print(f"{'─' * 60}\n")

        return selected_name, selected_metrics

    def _likelihood_ratio_test(self,
                               simple_metrics: Dict,
                               complex_metrics: Dict,
                               simple_name: str,
                               complex_name: str) -> Tuple[bool, float, float, int]:
        """
        Perform likelihood ratio test between two nested models.

        Returns
        -------
        is_significant : bool
        lr_statistic : float
        p_value : float
        df : int
        """
        # Extract log-likelihoods
        ll_simple = simple_metrics.get('log_likelihood')
        ll_complex = complex_metrics.get('log_likelihood')

        # Fallback: calculate from AIC if log_likelihood not available
        if ll_simple is None and 'aic' in simple_metrics:
            k_simple = simple_metrics['n_features'] + 1
            ll_simple = (2 * k_simple - simple_metrics['aic']) / 2

        if ll_complex is None and 'aic' in complex_metrics:
            k_complex = complex_metrics['n_features'] + 1
            ll_complex = (2 * k_complex - complex_metrics['aic']) / 2

        if ll_simple is None or ll_complex is None:
            raise ValueError(
                f"Cannot perform LRT: log_likelihood not available for "
                f"{simple_name} or {complex_name}. "
                f"Ensure models calculate 'log_likelihood' or 'aic'."
            )

        # Calculate degrees of freedom
        k_simple = simple_metrics['n_features'] + 1
        k_complex = complex_metrics['n_features'] + 1
        df = k_complex - k_simple

        if df <= 0:
            raise ValueError(
                f"Invalid df = {df} for LRT. "
                f"Complex model must have more parameters."
            )

        # Calculate LR statistic
        lr_statistic = -2 * (ll_simple - ll_complex)

        # Get p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df)

        # Decision
        is_significant = p_value < self.lrt_alpha

        return is_significant, lr_statistic, p_value, df

    # ========================================================================
    # HELPER METHODS: Specific Selection Logic
    # ========================================================================

    def _select_maximize(self,
                        model_results: Dict[str, Dict],
                        metric: str) -> Tuple[str, Dict]:
        """Select model with maximum value of metric (for adj_r2, r2)."""
        valid_models = {
            k: v for k, v in model_results.items()
            if not np.isnan(v.get(metric, np.nan))
        }

        if not valid_models:
            if self.verbose:
                print(f"⚠️  Warning: No valid {metric} values, using fallback")
            return self._select_fallback(model_results)

        best_name = max(valid_models, key=lambda k: valid_models[k][metric])
        return best_name, valid_models[best_name]

    def _select_minimize(self,
                        model_results: Dict[str, Dict],
                        metric: str) -> Tuple[str, Dict]:
        """Select model with minimum value of metric (for AIC, BIC, RRMSE, MAPE)."""
        valid_models = {
            k: v for k, v in model_results.items()
            if not np.isnan(v.get(metric, np.nan)) and not np.isinf(v.get(metric, np.inf))
        }

        if not valid_models:
            if self.verbose:
                print(f"⚠️  Warning: No valid {metric} values, using fallback")
            return self._select_fallback(model_results)

        best_name = min(valid_models, key=lambda k: valid_models[k][metric])
        return best_name, valid_models[best_name]

    def _select_bic_conservative(self,
                                 model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Select by BIC with conservative tie-breaking.

        If models are within ΔBIC < 2, prefer simpler model.
        """
        valid_models = {
            k: v for k, v in model_results.items()
            if not np.isnan(v.get('bic', np.nan)) and not np.isinf(v.get('bic', np.inf))
        }

        if not valid_models:
            return self._select_fallback(model_results)

        # Sort by BIC (lower is better)
        sorted_models = sorted(valid_models.items(),
                              key=lambda x: x[1]['bic'])

        best_name, best_metrics = sorted_models[0]
        best_bic = best_metrics['bic']

        # Find all models within threshold
        equivalent_models = []
        for name, metrics in sorted_models:
            delta_bic = metrics['bic'] - best_bic
            if delta_bic <= self.BIC_EQUIVALENCE_THRESHOLD:
                equivalent_models.append((name, metrics, delta_bic))

        # If multiple models are equivalent, pick simpler one
        if len(equivalent_models) > 1:
            selected = self._pick_simplest_model(equivalent_models)

            if self.verbose:
                equiv_names = [name for name, _, _ in equivalent_models]
                print(f"ℹ️  Models {equiv_names} have equivalent BIC (ΔBIC ≤ 2)")
                print(f"    Selected simpler model: {selected[0]}")

            return selected[0], selected[1]

        return best_name, best_metrics

    def _select_aic_conservative(self,
                                 model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """Select by AIC with conservative tie-breaking."""
        # Similar logic to BIC
        valid_models = {
            k: v for k, v in model_results.items()
            if not np.isnan(v.get('aic', np.nan)) and not np.isinf(v.get('aic', np.inf))
        }

        if not valid_models:
            return self._select_fallback(model_results)

        sorted_models = sorted(valid_models.items(),
                              key=lambda x: x[1]['aic'])

        best_name, best_metrics = sorted_models[0]
        best_aic = best_metrics['aic']

        equivalent_models = []
        for name, metrics in sorted_models:
            delta_aic = metrics['aic'] - best_aic
            if delta_aic <= self.AIC_EQUIVALENCE_THRESHOLD:
                equivalent_models.append((name, metrics, delta_aic))

        if len(equivalent_models) > 1:
            selected = self._pick_simplest_model(equivalent_models)

            if self.verbose:
                equiv_names = [name for name, _, _ in equivalent_models]
                print(f"ℹ️  Models {equiv_names} have equivalent AIC (ΔAIC ≤ 2)")
                print(f"    Selected simpler model: {selected[0]}")

            return selected[0], selected[1]

        return best_name, best_metrics

    def _select_nested_conservative(self,
                                    nested_models: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Conservative selection within nested model family.

        Uses BIC/AIC with strict threshold: only accept more complex
        model if improvement exceeds threshold.
        """
        if self.selection_metric == 'bic':
            metric = 'bic'
            threshold = self.BIC_EQUIVALENCE_THRESHOLD
        else:
            metric = 'aic'
            threshold = self.AIC_EQUIVALENCE_THRESHOLD

        # Start with simplest model
        selected_name = None
        selected_metrics = None

        for model_name in self.NESTED_MODEL_ORDER:
            if model_name not in nested_models:
                continue

            if selected_name is None:
                # First available model
                selected_name = model_name
                selected_metrics = nested_models[model_name]
            else:
                # Compare to current selection
                current_value = selected_metrics[metric]
                candidate_value = nested_models[model_name][metric]

                improvement = current_value - candidate_value  # Lower is better

                if improvement > threshold:
                    # Significant improvement, accept more complex model
                    if self.verbose:
                        print(f"ℹ️  {model_name} significantly better than "
                              f"{selected_name} (Δ{metric.upper()} = {improvement:.2f})")
                    selected_name = model_name
                    selected_metrics = nested_models[model_name]
                else:
                    # Not significant, keep simpler model
                    if self.verbose:
                        print(f"ℹ️  {model_name} not significantly better than "
                              f"{selected_name} (Δ{metric.upper()} = {improvement:.2f}), "
                              f"keeping simpler model")

        return selected_name, selected_metrics

    def _pick_simplest_model(self,
                            equivalent_models: List[Tuple[str, Dict, float]]) -> Tuple[str, Dict]:
        """
        Pick simplest model from list of equivalent models.

        Simplicity determined by:
        1. Position in NESTED_MODEL_ORDER (if applicable)
        2. Number of features (n_features)
        3. Alphabetical order (tiebreaker)
        """
        # Try nested order first
        for nested_name in self.NESTED_MODEL_ORDER:
            for name, metrics, delta in equivalent_models:
                if name == nested_name:
                    return (name, metrics)

        # Otherwise, use n_features
        simplest = min(equivalent_models,
                      key=lambda x: (x[1].get('n_features', float('inf')), x[0]))
        return (simplest[0], simplest[1])

    def _select_fallback(self,
                        model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Fallback selection when primary metric not available.

        Tries adj_r2, then r2, then first available model.
        """
        if self.verbose:
            print(f"⚠️  Falling back to alternative metric")

        # Try adj_r2
        valid = {k: v for k, v in model_results.items()
                if not np.isnan(v.get('adj_r2', np.nan))}
        if valid:
            best = max(valid, key=lambda k: valid[k]['adj_r2'])
            return best, valid[best]

        # Try r2
        valid = {k: v for k, v in model_results.items()
                if not np.isnan(v.get('r2', np.nan))}
        if valid:
            best = max(valid, key=lambda k: valid[k]['r2'])
            return best, valid[best]

        # Last resort: first model
        first_name = list(model_results.keys())[0]
        return first_name, model_results[first_name]

    # ========================================================================
    # DIAGNOSTIC AND REPORTING METHODS
    # ========================================================================

    def compare_all_criteria(self,
                             model_results: Dict[str, Dict]) -> Dict[str, Tuple[str, Dict]]:
        """
        Compare selections across all available criteria.

        Useful for understanding how different metrics would rank models.

        Returns
        -------
        dict
            {metric_name: (best_model_name, best_model_metrics)}
        """
        original_metric = self.selection_metric
        original_strategy = self.selection_strategy
        original_verbose = self.verbose

        self.verbose = False  # Suppress messages during comparison

        comparisons = {}

        for metric in ['adj_r2', 'r2', 'aic', 'bic', 'rrmse', 'mape']:
            self.selection_metric = metric
            try:
                comparisons[metric] = self.select_best_model(model_results)
            except:
                comparisons[metric] = (None, None)

        # Restore original settings
        self.selection_metric = original_metric
        self.selection_strategy = original_strategy
        self.verbose = original_verbose

        return comparisons

    def print_comparison_table(self, model_results: Dict[str, Dict]):
        """
        Print formatted comparison of model selection criteria.

        Shows which model would be selected by each criterion.
        """
        comparisons = self.compare_all_criteria(model_results)

        print("\n" + "="*70)
        print("MODEL SELECTION COMPARISON")
        print("="*70)

        print(f"\nCurrent settings:")
        print(f"  Metric:   {self.selection_metric}")
        print(f"  Strategy: {self.selection_strategy}")

        print(f"\nComparison across metrics:")
        print("-"*70)

        for metric, (name, _) in comparisons.items():
            if name:
                marker = "→" if metric == self.selection_metric else " "
                print(f"  {marker} {metric:10s}: {name}")
            else:
                print(f"    {metric:10s}: (not available)")

        # Check for consensus
        selected_models = [name for name, _ in comparisons.values() if name]
        if selected_models and len(set(selected_models)) == 1:
            print(f"\n✓ All metrics agree: {selected_models[0]} is best")
        else:
            print(f"\n⚠ Metrics disagree - selection depends on criterion")

        print("="*70 + "\n")

    """
    Add these methods to the BestModelSelector class
    """

    # ========================================================================
    # COMBINATION SELECTION (NEW)
    # ========================================================================

    def select_best_combination(self,
                                candidates: List[Dict],
                                selection_metric: str,
                                max_vif: float = 10.0,
                                penalize_heteroscedasticity: bool = True,
                                heteroscedasticity_penalty: float = 0.05,
                                prefer_simpler: bool = True,
                                equivalence_threshold: Optional[float] = None) -> Optional[Dict]:
        """
        Select best combination from a list of candidates.

        **Selection Process:**
        1. Filter by VIF (remove high multicollinearity)
        2. Apply heteroscedasticity penalties
        3. Select best by metric
        4. Tie-breaking (prefer simpler if equivalent)

        This is the combination-level equivalent of select_best_model().

        Parameters
        ----------
        candidates : list of dict
            List of candidate combinations, each containing:
            - 'predictors': tuple of predictor names
            - 'model_type': model type string
            - 'metrics': dict of performance metrics
            - 'n_predictors': number of predictors
        selection_metric : str
            Metric to optimize: 'adj_r2', 'bic', 'aic', 'rrmse', 'mape'
        max_vif : float, default=10.0
            Maximum acceptable VIF for any predictor
        penalize_heteroscedasticity : bool, default=True
            Whether to penalize heteroscedastic models
        heteroscedasticity_penalty : float, default=0.05
            Penalty magnitude (5% by default)
        prefer_simpler : bool, default=True
            When equivalent, prefer fewer predictors
        equivalence_threshold : float, optional
            Threshold for considering combinations equivalent.
            If None, uses metric-specific defaults:
            - BIC/AIC: 2.0
            - RRMSE/MAPE: 2.0 (percentage points)
            - R²: 0.02

        Returns
        -------
        dict or None
            Selected candidate dict, or None if all filtered out

        Examples
        --------
        >>> selector = BestModelSelector(
        ...     selection_metric='bic',
        ...     assumptions='moderate'
        ... )
        >>> best = selector.select_best_combination(
        ...     candidates=candidate_list,
        ...     selection_metric='bic',
        ...     max_vif=10.0
        ... )
        """

        if not candidates:
            return None

        # ========================================================================
        # STEP 1: Filter by VIF
        # ========================================================================

        valid_candidates = self._filter_combinations_by_vif(candidates, max_vif)

        if not valid_candidates:
            if self.verbose:
                print(f"  ⚠️  All {len(candidates)} combinations exceed VIF threshold ({max_vif})")
                print(f"     Using all candidates as fallback")
            valid_candidates = candidates
        elif self.verbose and len(valid_candidates) < len(candidates):
            n_filtered = len(candidates) - len(valid_candidates)
            print(f"  Filtered {n_filtered} combinations by VIF (>{max_vif})")

        # ========================================================================
        # STEP 2: Apply heteroscedasticity penalties
        # ========================================================================

        if penalize_heteroscedasticity:
            penalized_candidates = self._apply_heteroscedasticity_penalties_to_combinations(
                valid_candidates,
                selection_metric,
                heteroscedasticity_penalty
            )
        else:
            penalized_candidates = valid_candidates

        # ========================================================================
        # STEP 3: Select best by metric with tie-breaking
        # ========================================================================

        if equivalence_threshold is None:
            equivalence_threshold = self._get_default_equivalence_threshold(selection_metric)

        best = self._select_best_combination_by_metric(
            penalized_candidates,
            selection_metric,
            prefer_simpler,
            equivalence_threshold
        )

        return best

    def _filter_combinations_by_vif(self,
                                    candidates: List[Dict],
                                    max_vif: float) -> List[Dict]:
        """
        Filter combinations by VIF threshold.

        Parameters
        ----------
        candidates : list of dict
            Candidate combinations
        max_vif : float
            Maximum acceptable VIF

        Returns
        -------
        list of dict
            Candidates with VIF ≤ max_vif
        """
        valid = []

        for candidate in candidates:
            metrics = candidate['metrics']
            vif = metrics.get('max_vif', 1.0)

            if vif <= max_vif:
                valid.append(candidate)
            elif self.verbose:
                preds_str = ' + '.join(candidate['predictors'])
                print(f"    ✗ {preds_str}: VIF={vif:.2f} > {max_vif}")

        return valid

    def _apply_heteroscedasticity_penalties_to_combinations(self,
                                                            candidates: List[Dict],
                                                            metric: str,
                                                            penalty: float) -> List[Dict]:
        """
        Apply penalties to combinations with heteroscedasticity.

        **Rationale:**
        - Heteroscedasticity → Biased standard errors
        - Inefficient estimates → Poor predictions
        - Violated assumptions → Invalid inference

        Parameters
        ----------
        candidates : list of dict
            Candidate combinations
        metric : str
            Selection metric
        penalty : float
            Penalty magnitude

        Returns
        -------
        list of dict
            Candidates with penalized metrics (originals preserved)
        """
        penalized = []

        for candidate in candidates:
            metrics = candidate['metrics'].copy()

            # Check if heteroscedastic
            het_detected = self._detect_heteroscedasticity(metrics)

            if het_detected:
                # Store original
                metrics[f'_original_{metric}'] = metrics[metric]

                # Apply penalty
                if metric in ['r2', 'adj_r2']:
                    metrics[metric] = metrics[metric] - penalty
                elif metric in ['rrmse', 'mape', 'rmse', 'mae']:
                    metrics[metric] = metrics[metric] * (1 + penalty)
                elif metric in ['aic', 'bic']:
                    metrics[metric] = metrics[metric] * (1 + penalty)

                if self.verbose:
                    preds_str = ' + '.join(candidate['predictors'])
                    original = metrics[f'_original_{metric}']
                    penalized_val = metrics[metric]
                    print(f"    ⚠️  {preds_str}: heteroscedastic, "
                          f"{metric} {original:.4f} → {penalized_val:.4f}")

            penalized_candidate = candidate.copy()
            penalized_candidate['metrics'] = metrics
            penalized_candidate['het_penalty_applied'] = het_detected
            penalized.append(penalized_candidate)

        return penalized

    def _detect_heteroscedasticity(self, metrics: Dict) -> bool:
        """
        Detect heteroscedasticity from diagnostic tests.

        **Strategy:** Conservative approach
        - Flag if 2+ tests detect heteroscedasticity, OR
        - Spearman test detects it (most reliable)

        Parameters
        ----------
        metrics : dict
            Model metrics with diagnostic test results

        Returns
        -------
        bool
            True if heteroscedasticity detected
        """
        spearman = metrics.get('spearman', False)
        bp = metrics.get('breusch_pagan', False)
        white = metrics.get('white', False)

        n_positive = sum([spearman, bp, white])

        # 2+ tests OR Spearman (most reliable for our use case)
        return n_positive >= 2 or spearman

    def _select_best_combination_by_metric(self,
                                           candidates: List[Dict],
                                           metric: str,
                                           prefer_simpler: bool,
                                           equivalence_threshold: float) -> Optional[Dict]:
        """
        Select best combination with tie-breaking.

        **Tie-breaking:**
        - If multiple combinations within threshold → prefer simpler
        - Mimics nested model selection logic

        Parameters
        ----------
        candidates : list of dict
            Candidates (filtered and penalized)
        metric : str
            Selection metric
        prefer_simpler : bool
            Prefer fewer predictors in ties
        equivalence_threshold : float
            Threshold for equivalence

        Returns
        -------
        dict or None
            Selected candidate
        """

        if not candidates:
            return None

        # Determine sort direction
        ascending = metric not in ['r2', 'adj_r2']

        # Sort candidates
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x['metrics'][metric],
            reverse=not ascending
        )

        best = sorted_candidates[0]
        best_value = best['metrics'][metric]

        # Find equivalent candidates
        equivalent = [best]

        for candidate in sorted_candidates[1:]:
            value = candidate['metrics'][metric]
            delta = abs(value - best_value)

            is_equivalent = delta < equivalence_threshold

            if is_equivalent:
                equivalent.append(candidate)
            else:
                break  # Sorted, so done

        # Tie-breaking
        if len(equivalent) > 1 and prefer_simpler:
            # Sort by: (1) n_predictors, (2) metric value
            equivalent.sort(
                key=lambda x: (
                    x['n_predictors'],
                    x['metrics'][metric] * (1 if ascending else -1)
                )
            )
            selected = equivalent[0]

            if self.verbose:
                equiv_strs = [' + '.join(c['predictors']) for c in equivalent]
                print(f"  ℹ️  Equivalent: {', '.join(equiv_strs)}")
                print(f"     Selected: {' + '.join(selected['predictors'])} (simpler)")
        else:
            selected = equivalent[0]

        return selected

    def _get_default_equivalence_threshold(self, metric: str) -> float:
        """
        Get default equivalence threshold for metric.

        Parameters
        ----------
        metric : str
            Selection metric

        Returns
        -------
        float
            Default threshold
        """
        thresholds = {
            'bic': 2.0,
            'aic': 2.0,
            'rrmse': 2.0,
            'mape': 2.0,
            'rmse': 2.0,
            'mae': 2.0,
            'r2': 0.02,
            'adj_r2': 0.02
        }

        return thresholds.get(metric, 2.0)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_selector():
    """Show how to use the BestModelSelector class."""

    # Example model results
    model_results = {
        'linear': {
            'r2': 0.850, 'adj_r2': 0.848,
            'aic': 5100.2, 'bic': 5110.5,
            'rrmse': 12.3, 'mape': 10.1,
            'n_features': 2, 'n_samples': 500
        },
        'polynomial-2': {
            'r2': 0.920, 'adj_r2': 0.915,
            'aic': 5098.1, 'bic': 5109.8,
            'rrmse': 9.1, 'mape': 7.5,
            'n_features': 5, 'n_samples': 500
        },
        'polynomial-3': {
            'r2': 0.925, 'adj_r2': 0.917,
            'aic': 5100.0, 'bic': 5115.2,
            'rrmse': 8.9, 'mape': 7.3,
            'n_features': 9, 'n_samples': 500
        },
        'exponential': {
            'r2': 0.910, 'adj_r2': 0.908,
            'aic': 5105.0, 'bic': 5112.0,
            'rrmse': 9.5, 'mape': 7.8,
            'n_features': 2, 'n_samples': 500
        }
    }

    print("\n" + "█" * 70)
    print("DEMONSTRATION: BestModelSelector Class")
    print("█" * 70)

    # Example 1: Simple selection by BIC
    print("\n1️⃣  Simple BIC Selection")
    print("-" * 70)
    selector1 = BestModelSelector(selection_metric='bic',
                                  selection_strategy='simple',
                                  verbose=True)
    best_name, best_metrics = selector1.select_best_model(model_results)
    print(f"Selected: {best_name} (BIC = {best_metrics['bic']:.1f})")

    # Example 2: Conservative BIC selection
    print("\n2️⃣  Conservative BIC Selection (with tie-breaking)")
    print("-" * 70)
    selector2 = BestModelSelector(selection_metric='bic',
                                  selection_strategy='conservative',
                                  verbose=True)
    best_name, best_metrics = selector2.select_best_model(model_results)
    print(f"Selected: {best_name} (BIC = {best_metrics['bic']:.1f})")

    # Example 3: Nested-aware selection
    print("\n3️⃣  Nested-Aware BIC Selection")
    print("-" * 70)
    selector3 = BestModelSelector(selection_metric='bic',
                                  selection_strategy='nested_aware',
                                  verbose=True)
    best_name, best_metrics = selector3.select_best_model(model_results)
    print(f"Selected: {best_name} (BIC = {best_metrics['bic']:.1f})")

    # Example 4: Comparison table
    print("\n4️⃣  Comparison Across All Criteria")
    print("-" * 70)
    selector3.print_comparison_table(model_results)




#
# if __name__ == "__main__":
#     # demonstrate_selector()