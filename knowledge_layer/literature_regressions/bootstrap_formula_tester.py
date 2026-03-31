"""
Bootstrap Significance Tester for Formula Performance Gaps
==========================================================

Tests whether the observed RRMSE difference between two datasets
(e.g., matched conventional vs alternative fuel) is statistically
significant using bootstrap confidence intervals.

Composition role:
    Used by LiteratureFormulaEvaluator and MultiDatasetLiteratureEvaluator
    to enrich standard evaluation results with statistical significance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula


class BootstrapSignificanceTester:
    """
    Tests statistical significance of RRMSE differences between datasets
    using bootstrap confidence intervals.

    Concept:
    --------
    Given a formula F and two datasets A, B, we want to know:
    "Is the observed RRMSE(F, B) - RRMSE(F, A) real, or sampling noise?"

    Method:
    -------
    1. Compute observed difference: delta = RRMSE(F, B) - RRMSE(F, A)
    2. Resample A with replacement -> A*; compute RRMSE(F, A*)
    3. Resample B with replacement -> B*; compute RRMSE(F, B*)
    4. Compute bootstrap difference: delta* = RRMSE(F, B*) - RRMSE(F, A*)
    5. Repeat n_bootstrap times to build distribution of delta*
    6. Compute CI of delta* distribution
    7. If CI excludes zero -> significant difference

    Note on small samples:
    ----------------------
    With n≈12-18 alternative fuel ships, bootstrap distributions will be
    wide, reflecting genuine uncertainty. This is correct behaviour -
    small samples should produce wide CIs. The BCa method corrects for
    bias and skewness, which is important when distributions are asymmetric
    (common with small samples and ratio metrics like RRMSE).

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (default: 5000)
    confidence_level : float
        Confidence level for intervals (default: 0.95)
    ci_method : str
        Method for CI: 'percentile' (fast) or 'bca' (more accurate)
    random_seed : int, optional
        Seed for reproducibility
    """

    def __init__(self,
                 n_bootstrap: int = 5000,
                 confidence_level: float = 0.95,
                 ci_method: str = 'percentile',
                 random_seed: Optional[int] = None):

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.ci_method = ci_method
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self._validate_inputs()

    def _validate_inputs(self):
        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {self.n_bootstrap}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level}")
        if self.ci_method not in ['percentile', 'bca']:
            raise ValueError(f"ci_method must be 'percentile' or 'bca', got {self.ci_method}")

    # =========================================================================
    # Core Testing Methods
    # =========================================================================

    def test_formula_gap(self,
                         formula: 'LiteratureFormula',
                         dataset_a: pd.DataFrame,
                         dataset_b: pd.DataFrame,
                         reference_mean: Optional[float] = None,
                         comparison_label: str = 'A vs B',
                         debug: bool = False) -> Dict:
        """
        Test whether the RRMSE difference between two datasets is significant.

        Parameters
        ----------
        formula : LiteratureFormula
            Formula to evaluate on both datasets
        dataset_a : pd.DataFrame
            First dataset (typically: matched conventional)
        dataset_b : pd.DataFrame
            Second dataset (typically: alternative fuel)
        reference_mean : float, optional
            Mean of target variable used to normalise RRMSE.
            If None, computed from dataset_a (conventional reference).
        comparison_label : str
            Human-readable label for this comparison (for reporting)
        debug : bool
            Print step-by-step progress

        Returns
        -------
        dict
            Keys:
            - formula_name, target, reference, comparison_label
            - observed_rrmse_a, observed_rrmse_b, observed_delta
            - ci_lower, ci_upper, is_significant
            - bootstrap_deltas (array of bootstrap differences)
            - n_a, n_b, n_bootstrap, confidence_level, ci_method
        """
        if debug:
            print(f"\n{'='*60}")
            print(f"Bootstrap test: {formula.name}")
            print(f"Comparison: {comparison_label}")
            print(f"n_bootstrap: {self.n_bootstrap}, method: {self.ci_method}")

        # ── Step 1: Extract and clean data ──────────────────────────────────
        X_a, y_a = self._extract_clean_data(formula, dataset_a)
        X_b, y_b = self._extract_clean_data(formula, dataset_b)

        if X_a is None or X_b is None:
            return self._insufficient_data_result(formula, comparison_label)

        if debug:
            print(f"\n[Step 1] Data extracted")
            print(f"  Dataset A: n={len(y_a)}")
            print(f"  Dataset B: n={len(y_b)}")

        # ── Step 2: Reference mean for RRMSE normalisation ─────────────────
        if reference_mean is None:
            reference_mean = np.mean(y_a)

        if debug:
            print(f"[Step 2] Reference mean for RRMSE: {reference_mean:.4f}")

        # ── Step 3: Observed RRMSE values ───────────────────────────────────
        observed_rrmse_a = self._compute_rrmse(formula, X_a, y_a, reference_mean)
        observed_rrmse_b = self._compute_rrmse(formula, X_b, y_b, reference_mean)
        observed_delta = observed_rrmse_b - observed_rrmse_a

        if debug:
            print(f"[Step 3] Observed RRMSE")
            print(f"  RRMSE_A: {observed_rrmse_a:.4f}%")
            print(f"  RRMSE_B: {observed_rrmse_b:.4f}%")
            print(f"  Delta (B - A): {observed_delta:.4f}%")

        # ── Step 4: Bootstrap resampling ────────────────────────────────────
        if debug:
            print(f"\n[Step 4] Running {self.n_bootstrap} bootstrap iterations...")

        bootstrap_deltas = self._run_bootstrap(
            formula, X_a, y_a, X_b, y_b, reference_mean, debug=debug
        )

        if debug:
            print(f"  Bootstrap delta distribution:")
            print(f"    Mean: {np.mean(bootstrap_deltas):.4f}%")
            print(f"    Std:  {np.std(bootstrap_deltas):.4f}%")
            print(f"    Min:  {np.min(bootstrap_deltas):.4f}%")
            print(f"    Max:  {np.max(bootstrap_deltas):.4f}%")

        # ── Step 5: Confidence interval ─────────────────────────────────────
        if self.ci_method == 'bca':
            ci_lower, ci_upper = self._bca_ci(
                bootstrap_deltas=bootstrap_deltas,
                observed_delta=observed_delta,
                formula=formula,
                X_a=X_a, y_a=y_a,
                X_b=X_b, y_b=y_b,
                reference_mean=reference_mean,
                debug=debug
            )
        else:
            ci_lower, ci_upper = self._percentile_ci(bootstrap_deltas)

        if debug:
            print(f"\n[Step 5] {self.ci_method.upper()} Confidence Interval ({self.confidence_level*100:.0f}%)")
            print(f"  CI: [{ci_lower:.4f}%, {ci_upper:.4f}%]")

        # ── Step 6: Significance decision ───────────────────────────────────
        # Significant if CI does not contain zero
        is_significant = (ci_lower > 0) or (ci_upper < 0)

        if debug:
            print(f"\n[Step 6] Significance")
            print(f"  CI excludes zero: {is_significant}")
            direction = "B > A (worse)" if observed_delta > 0 else "B < A (better)"
            print(f"  Direction: {direction}")

        return {
            'formula_name': formula.name,
            'target': formula.target,
            'predictors': formula.predictors,
            'reference': formula.reference,
            'comparison_label': comparison_label,
            'observed_rrmse_a': observed_rrmse_a,
            'observed_rrmse_b': observed_rrmse_b,
            'observed_delta': observed_delta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': is_significant,
            'bootstrap_deltas': bootstrap_deltas,
            'n_a': len(y_a),
            'n_b': len(y_b),
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'ci_method': self.ci_method,
        }

    def test_all_formulae(self,
                          formulae: List['LiteratureFormula'],
                          dataset_a: pd.DataFrame,
                          dataset_b: pd.DataFrame,
                          reference_mean_map: Optional[Dict[str, float]] = None,
                          comparison_label: str = 'Matched vs Alternative',
                          debug: bool = False) -> List[Dict]:
        """
        Run significance tests for a list of formulae.

        Parameters
        ----------
        formulae : List[LiteratureFormula]
            All formulae to test
        dataset_a : pd.DataFrame
            First dataset (typically matched conventional)
        dataset_b : pd.DataFrame
            Second dataset (typically alternative fuel)
        reference_mean_map : dict, optional
            Maps target variable name to reference mean.
            If None, computed from dataset_a per formula.
        comparison_label : str
            Label for this comparison pair
        debug : bool
            Print per-formula debug info

        Returns
        -------
        list[dict]
            One result dict per formula
        """
        results = []

        for i, formula in enumerate(formulae):
            if debug:
                print(f"\nFormula {i+1}/{len(formulae)}: {formula.name}")

            # Get reference mean for this formula's target
            ref_mean = None
            if reference_mean_map and formula.target in reference_mean_map:
                ref_mean = reference_mean_map[formula.target]

            result = self.test_formula_gap(
                formula=formula,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                reference_mean=ref_mean,
                comparison_label=comparison_label,
                debug=debug
            )
            results.append(result)

        return results

    # =========================================================================
    # Bootstrap Core
    # =========================================================================

    def _run_bootstrap(self,
                       formula: 'LiteratureFormula',
                       X_a: np.ndarray,
                       y_a: np.ndarray,
                       X_b: np.ndarray,
                       y_b: np.ndarray,
                       reference_mean: float,
                       debug: bool = False) -> np.ndarray:
        """
        Run bootstrap resampling and compute RRMSE difference for each iteration.

        Each iteration independently resamples dataset A and B with replacement,
        then computes RRMSE(B*) - RRMSE(A*). This preserves the within-dataset
        correlation structure while building a distribution of the difference.
        """
        n_a = len(y_a)
        n_b = len(y_b)
        bootstrap_deltas = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            # Resample each dataset independently with replacement
            idx_a = np.random.choice(n_a, size=n_a, replace=True)
            idx_b = np.random.choice(n_b, size=n_b, replace=True)

            X_a_boot = X_a[idx_a]
            y_a_boot = y_a[idx_a]
            X_b_boot = X_b[idx_b]
            y_b_boot = y_b[idx_b]

            rrmse_a = self._compute_rrmse(formula, X_a_boot, y_a_boot, reference_mean)
            rrmse_b = self._compute_rrmse(formula, X_b_boot, y_b_boot, reference_mean)

            bootstrap_deltas[i] = rrmse_b - rrmse_a

            if debug and (i + 1) % 1000 == 0:
                print(f"  Iteration {i+1}/{self.n_bootstrap} complete")

        return bootstrap_deltas

    # =========================================================================
    # CI Methods
    # =========================================================================

    def _percentile_ci(self, bootstrap_deltas: np.ndarray) -> Tuple[float, float]:
        """
        Percentile CI: take alpha/2 and 1-alpha/2 quantiles of bootstrap distribution.

        Simple and robust. Sufficient for most use cases.
        """
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_deltas, 100 * alpha / 2)
        upper = np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2))
        return lower, upper

    def _bca_ci(self,
                bootstrap_deltas: np.ndarray,
                observed_delta: float,
                formula: 'LiteratureFormula',
                X_a: np.ndarray, y_a: np.ndarray,
                X_b: np.ndarray, y_b: np.ndarray,
                reference_mean: float,
                debug: bool = False) -> Tuple[float, float]:
        """
        Bias-Corrected and Accelerated (BCa) bootstrap CI.

        Corrects for:
        - Bias: bootstrap distribution is shifted relative to observed statistic
        - Acceleration: spread of bootstrap distribution changes with parameter value

        This is more accurate than the percentile method when:
        - Bootstrap distribution is skewed (common with small samples)
        - Observed statistic is near the boundary of its range

        The acceleration 'a' is estimated via jackknife on both datasets.
        """
        from scipy import stats

        # ── Bias correction z0 ──────────────────────────────────────────────
        # Proportion of bootstrap samples below the observed delta
        p_below = np.mean(bootstrap_deltas < observed_delta)
        # Clamp to avoid ppf(0) or ppf(1)
        p_below = np.clip(p_below, 1e-10, 1 - 1e-10)
        z0 = stats.norm.ppf(p_below)

        if debug:
            print(f"\n  [BCa] Bias correction z0: {z0:.4f}")

        # ── Acceleration 'a' via jackknife ──────────────────────────────────
        # Empirical influence values measure how each observation affects the statistic
        n_a, n_b = len(y_a), len(y_b)
        L_all = []

        # Influence from dataset A (remove each observation, recompute delta)
        for i in range(n_a):
            X_jack = np.delete(X_a, i, axis=0)
            y_jack = np.delete(y_a, i)
            rrmse_jack_a = self._compute_rrmse(formula, X_jack, y_jack, reference_mean)
            rrmse_b_full = self._compute_rrmse(formula, X_b, y_b, reference_mean)
            delta_jack = rrmse_b_full - rrmse_jack_a
            L_i = (n_a - 1) * (observed_delta - delta_jack)
            L_all.append(L_i)

        # Influence from dataset B
        for i in range(n_b):
            X_jack = np.delete(X_b, i, axis=0)
            y_jack = np.delete(y_b, i)
            rrmse_a_full = self._compute_rrmse(formula, X_a, y_a, reference_mean)
            rrmse_jack_b = self._compute_rrmse(formula, X_jack, y_jack, reference_mean)
            delta_jack = rrmse_jack_b - rrmse_a_full
            L_i = (n_b - 1) * (observed_delta - delta_jack)
            L_all.append(L_i)

        L_all = np.array(L_all)
        L_mean = np.mean(L_all)
        deviations = L_mean - L_all

        numerator = np.sum(deviations ** 3)
        denominator = 6 * (np.sum(deviations ** 2)) ** 1.5
        a = numerator / denominator if denominator != 0 else 0.0

        if debug:
            print(f"  [BCa] Acceleration a: {a:.6f}")

        # ── Adjusted quantiles ───────────────────────────────────────────────
        alpha = 1 - self.confidence_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        def adjusted_quantile(z_alpha):
            """BCa adjusted quantile formula."""
            numerator = z0 + z_alpha
            denom = 1 - a * (z0 + z_alpha)
            if abs(denom) < 1e-10:
                return z_alpha  # fallback to unadjusted
            return stats.norm.cdf(z0 + numerator / denom)

        p_lower = adjusted_quantile(z_alpha_lower)
        p_upper = adjusted_quantile(z_alpha_upper)

        # Clamp to valid range
        p_lower = np.clip(p_lower, 0.001, 0.999)
        p_upper = np.clip(p_upper, 0.001, 0.999)

        ci_lower = np.percentile(bootstrap_deltas, 100 * p_lower)
        ci_upper = np.percentile(bootstrap_deltas, 100 * p_upper)

        if debug:
            print(f"  [BCa] Adjusted quantiles: [{p_lower:.4f}, {p_upper:.4f}]")

        return ci_lower, ci_upper

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_clean_data(self,
                            formula: 'LiteratureFormula',
                            dataset: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract predictor and target arrays from dataset, removing NaN/inf.

        Returns (None, None) if insufficient clean data.
        """
        min_samples = 3

        missing = [c for c in formula.predictors + [formula.target] if c not in dataset.columns]
        if missing:
            return None, None

        X = dataset[formula.predictors].values
        y = dataset[formula.target].values

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))

        X_clean = X[mask]
        y_clean = y[mask]

        if len(y_clean) < min_samples:
            return None, None

        return X_clean, y_clean

    def _compute_rrmse(self,
                       formula: 'LiteratureFormula',
                       X: np.ndarray,
                       y: np.ndarray,
                       reference_mean: float) -> float:
        """
        Compute RRMSE of formula predictions on given data.

        RRMSE = RMSE / reference_mean * 100 (%)
        """
        try:
            y_pred = formula.predict(X)
            metrics = RegressionMetrics(y_pred, y)
            return metrics.calculate_rrmse(y_mean=reference_mean)
        except Exception:
            return np.nan

    def _insufficient_data_result(self,
                                  formula: 'LiteratureFormula',
                                  comparison_label: str) -> Dict:
        """Return a standardised result dict when data is insufficient."""
        return {
            'formula_name': formula.name,
            'target': formula.target,
            'predictors': formula.predictors,
            'reference': formula.reference,
            'comparison_label': comparison_label,
            'observed_rrmse_a': np.nan,
            'observed_rrmse_b': np.nan,
            'observed_delta': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'is_significant': None,
            'bootstrap_deltas': np.array([]),
            'n_a': 0,
            'n_b': 0,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'ci_method': self.ci_method,
            'error': 'Insufficient data'
        }

    # =========================================================================
    # Export
    # =========================================================================

    def results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert list of result dicts to a clean summary DataFrame.

        Drops the raw bootstrap_deltas array (too large for tabular export).

        Parameters
        ----------
        results : list[dict]
            Output from test_all_formulae()

        Returns
        -------
        pd.DataFrame
            Summary table with one row per formula
        """
        rows = []
        for r in results:
            rows.append({
                'formula_name': r['formula_name'],
                'target': r['target'],
                'predictors': ' + '.join(r['predictors']),
                'reference': r['reference'],
                'comparison': r['comparison_label'],
                'rrmse_a': r['observed_rrmse_a'],
                'rrmse_b': r['observed_rrmse_b'],
                'delta_rrmse': r['observed_delta'],
                'ci_lower': r['ci_lower'],
                'ci_upper': r['ci_upper'],
                'is_significant': r['is_significant'],
                'n_a': r['n_a'],
                'n_b': r['n_b'],
                'ci_method': r['ci_method'],
                'confidence_level': r['confidence_level'],
                'n_bootstrap': r['n_bootstrap'],
            })
        return pd.DataFrame(rows)

    def export_to_excel(self,
                        results: List[Dict],
                        output_path: str,
                        sheet_name: str = 'Bootstrap_Significance') -> None:
        """
        Export significance test results to an Excel file.

        Parameters
        ----------
        results : list[dict]
            Output from test_all_formulae()
        output_path : str
            Path for the output .xlsx file
        sheet_name : str
            Sheet name within the workbook
        """
        df = self.results_to_dataframe(results)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-format column widths
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        print(f"✓ Significance test results saved to: {output_path}")