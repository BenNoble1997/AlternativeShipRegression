"""
Bootstrap Significance Tester for Trained Model Performance Gaps
================================================================

Tests whether observed RRMSE differences between evaluation datasets
are statistically significant for TRAINED models (parametric / RF).

Key distinction from literature formula tester:
- Literature: Fixed formulas — resample evaluation data only.
- Trained models: Model is fixed (already trained on conventional train set).
  We resample the EVALUATION datasets to test whether the gap in
  evaluation performance is real or sampling noise.

The model is NOT retrained during bootstrap — we're asking:
    "Is the performance difference we see on these two evaluation sets
     too large to be explained by the sampling variability of those sets?"

Dataset pairs by comparison type:
    'distribution': conventional_test vs matched
    'fuel_type':    matched vs alternative
    'total':        conventional_test vs alternative
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


class TrainedModelBootstrapTester:
    """
    Bootstrap significance testing for trained model evaluation gaps.

    The model is trained once on the conventional training set and then
    evaluated on two different test sets. Bootstrap resampling of each
    test set builds a distribution of the RRMSE difference, from which
    a confidence interval is computed.

    If the CI excludes zero, the gap is statistically significant.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (default: 5000)
    confidence_level : float
        CI confidence level (default: 0.95)
    ci_method : str
        'percentile' or 'bca'
    random_seed : int, optional
        For reproducibility
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
            raise ValueError(f"confidence_level must be in (0,1), got {self.confidence_level}")
        if self.ci_method not in ['percentile', 'bca']:
            raise ValueError(f"ci_method must be 'percentile' or 'bca', got {self.ci_method}")

    # =========================================================================
    # Core Public API
    # =========================================================================

    def test_gap(self,
                 predict_fn: Callable[[np.ndarray], np.ndarray],
                 X_a: np.ndarray,
                 y_a: np.ndarray,
                 X_b: np.ndarray,
                 y_b: np.ndarray,
                 reference_mean: float,
                 comparison_label: str = 'A vs B',
                 target: str = '',
                 predictor_str: str = '',
                 model_type: str = '',
                 debug: bool = False) -> Dict:
        """
        Test whether the RRMSE difference between two evaluation datasets
        is statistically significant for a fixed trained model.

        Parameters
        ----------
        predict_fn : callable
            Function that takes X (np.ndarray) and returns y_pred (np.ndarray).
            This is the already-trained model — it is NOT retrained during bootstrap.
        X_a : np.ndarray
            Features for dataset A (e.g. conventional test, or matched)
        y_a : np.ndarray
            Targets for dataset A
        X_b : np.ndarray
            Features for dataset B (e.g. matched, or alternative)
        y_b : np.ndarray
            Targets for dataset B
        reference_mean : float
            Mean of target from conventional data — used to normalise RRMSE
            consistently with the main analysis
        comparison_label : str
            Human-readable description of the comparison
        target : str
            Target variable name (for reporting)
        predictor_str : str
            Predictor combination string (for reporting)
        model_type : str
            Model type string (for reporting)
        debug : bool
            Print step-by-step progress

        Returns
        -------
        dict
            Keys:
            - target, predictor_str, model_type, comparison_label
            - observed_rrmse_a, observed_rrmse_b, observed_delta
            - ci_lower, ci_upper, is_significant
            - n_a, n_b, n_bootstrap, confidence_level, ci_method
        """
        if debug:
            print(f"\n{'='*60}")
            print(f"Bootstrap test: {predictor_str} → {target}")
            print(f"Comparison: {comparison_label}")
            print(f"n_a={len(y_a)}, n_b={len(y_b)}")

        # ── Step 1: Clean data ───────────────────────────────────────────────
        X_a, y_a = self._clean(X_a, y_a)
        X_b, y_b = self._clean(X_b, y_b)

        if X_a is None or X_b is None or len(y_a) < 3 or len(y_b) < 3:
            return self._insufficient_data_result(
                target, predictor_str, model_type, comparison_label
            )

        if debug:
            print(f"[Step 1] Clean data: n_a={len(y_a)}, n_b={len(y_b)}")

        # ── Step 2: Observed RRMSE values ────────────────────────────────────
        observed_rrmse_a = self._compute_rrmse(predict_fn, X_a, y_a, reference_mean)
        observed_rrmse_b = self._compute_rrmse(predict_fn, X_b, y_b, reference_mean)
        observed_delta = observed_rrmse_b - observed_rrmse_a

        if debug:
            print(f"[Step 2] Observed: RRMSE_A={observed_rrmse_a:.3f}%, "
                  f"RRMSE_B={observed_rrmse_b:.3f}%, delta={observed_delta:+.3f}%")

        # ── Step 3: Bootstrap resampling ─────────────────────────────────────
        bootstrap_deltas = self._run_bootstrap(
            predict_fn, X_a, y_a, X_b, y_b, reference_mean, debug=debug
        )

        if debug:
            print(f"[Step 3] Bootstrap delta: mean={np.mean(bootstrap_deltas):.3f}%, "
                  f"std={np.std(bootstrap_deltas):.3f}%")

        # ── Step 4: Confidence interval ──────────────────────────────────────
        if self.ci_method == 'bca':
            ci_lower, ci_upper = self._bca_ci(
                bootstrap_deltas=bootstrap_deltas,
                observed_delta=observed_delta,
                predict_fn=predict_fn,
                X_a=X_a, y_a=y_a,
                X_b=X_b, y_b=y_b,
                reference_mean=reference_mean,
                debug=debug
            )
        else:
            ci_lower, ci_upper = self._percentile_ci(bootstrap_deltas)

        is_significant = (ci_lower > 0) or (ci_upper < 0)

        if debug:
            print(f"[Step 4] {self.ci_method.upper()} CI: "
                  f"[{ci_lower:.3f}%, {ci_upper:.3f}%]")
            print(f"[Step 5] Significant: {is_significant} "
                  f"({'CI excludes zero' if is_significant else 'CI contains zero'})")

        return {
            'target': target,
            'predictor_str': predictor_str,
            'model_type': model_type,
            'comparison_label': comparison_label,
            'observed_rrmse_a': observed_rrmse_a,
            'observed_rrmse_b': observed_rrmse_b,
            'observed_delta': observed_delta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': is_significant,
            'n_a': len(y_a),
            'n_b': len(y_b),
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'ci_method': self.ci_method,
        }

    # =========================================================================
    # Bootstrap Core
    # =========================================================================

    # def _run_bootstrap(self,
    #                    predict_fn: Callable,
    #                    X_a: np.ndarray,
    #                    y_a: np.ndarray,
    #                    X_b: np.ndarray,
    #                    y_b: np.ndarray,
    #                    reference_mean: float,
    #                    debug: bool = False) -> np.ndarray:
    #
    #     n_a, n_b = len(y_a), len(y_b)
    #     deltas = np.zeros(self.n_bootstrap)
    #
    #     if debug:
    #         start_time = time.perf_counter()
    #         block_start = start_time
    #
    #     for i in range(self.n_bootstrap):
    #         idx_a = np.random.choice(n_a, size=n_a, replace=True)
    #         idx_b = np.random.choice(n_b, size=n_b, replace=True)
    #
    #         rrmse_a = self._compute_rrmse(
    #             predict_fn, X_a[idx_a], y_a[idx_a], reference_mean
    #         )
    #         rrmse_b = self._compute_rrmse(
    #             predict_fn, X_b[idx_b], y_b[idx_b], reference_mean
    #         )
    #         deltas[i] = rrmse_b - rrmse_a
    #
    #         if debug and (i + 1) % 1000 == 0:
    #             now = time.perf_counter()
    #             block_time = now - block_start
    #             total_time = now - start_time
    #             print(
    #                 f"Iteration {i + 1}/{self.n_bootstrap} | "
    #                 f"Last 1000: {block_time:.2f}s | "
    #                 f"Total: {total_time:.2f}s"
    #             )
    #             block_start = now
    #
    #     return deltas

    # =========================================================================
    # CI Methods
    # =========================================================================

    def _percentile_ci(self, bootstrap_deltas: np.ndarray) -> Tuple[float, float]:
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_deltas, 100 * alpha / 2)
        upper = np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2))
        return lower, upper

    # def _bca_ci(self,
    #             bootstrap_deltas: np.ndarray,
    #             observed_delta: float,
    #             predict_fn: Callable,
    #             X_a: np.ndarray, y_a: np.ndarray,
    #             X_b: np.ndarray, y_b: np.ndarray,
    #             reference_mean: float,
    #             debug: bool = False) -> Tuple[float, float]:
    #     """
    #     BCa CI using jackknife on both evaluation datasets.
    #
    #     Influence values capture how much each observation in either
    #     evaluation set affects the observed RRMSE difference.
    #     """
    #     from scipy import stats
    #
    #     # Bias correction
    #     p_below = np.clip(np.mean(bootstrap_deltas < observed_delta), 1e-10, 1 - 1e-10)
    #     z0 = stats.norm.ppf(p_below)
    #
    #     # Acceleration via jackknife
    #     n_a, n_b = len(y_a), len(y_b)
    #     L_all = []
    #
    #     rrmse_b_full = self._compute_rrmse(predict_fn, X_b, y_b, reference_mean)
    #
    #     for i in range(n_a):
    #         mask = np.ones(n_a, dtype=bool)
    #         mask[i] = False
    #         rrmse_a_jack = self._compute_rrmse(
    #             predict_fn, X_a[mask], y_a[mask], reference_mean
    #         )
    #         delta_jack = rrmse_b_full - rrmse_a_jack
    #         L_all.append((n_a - 1) * (observed_delta - delta_jack))
    #
    #     rrmse_a_full = self._compute_rrmse(predict_fn, X_a, y_a, reference_mean)
    #
    #     for i in range(n_b):
    #         mask = np.ones(n_b, dtype=bool)
    #         mask[i] = False
    #         rrmse_b_jack = self._compute_rrmse(
    #             predict_fn, X_b[mask], y_b[mask], reference_mean
    #         )
    #         delta_jack = rrmse_b_jack - rrmse_a_full
    #         L_all.append((n_b - 1) * (observed_delta - delta_jack))
    #
    #     L_all = np.array(L_all)
    #     L_mean = np.mean(L_all)
    #     dev = L_mean - L_all
    #     numerator = np.sum(dev ** 3)
    #     denominator = 6 * (np.sum(dev ** 2)) ** 1.5
    #     a = numerator / denominator if abs(denominator) > 1e-10 else 0.0
    #
    #     if debug:
    #         print(f"  [BCa] z0={z0:.4f}, a={a:.6f}")
    #
    #     alpha = 1 - self.confidence_level
    #     z_lower = stats.norm.ppf(alpha / 2)
    #     z_upper = stats.norm.ppf(1 - alpha / 2)
    #
    #     def adjusted_q(z_alpha):
    #         num = z0 + z_alpha
    #         den = 1 - a * (z0 + z_alpha)
    #         if abs(den) < 1e-10:
    #             return stats.norm.cdf(z_alpha)
    #         return stats.norm.cdf(z0 + num / den)
    #
    #     p_lower = np.clip(adjusted_q(z_lower), 0.001, 0.999)
    #     p_upper = np.clip(adjusted_q(z_upper), 0.001, 0.999)
    #
    #     return (
    #         np.percentile(bootstrap_deltas, 100 * p_lower),
    #         np.percentile(bootstrap_deltas, 100 * p_upper)
    #     )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_rrmse(self,
                       predict_fn: Callable,
                       X: np.ndarray,
                       y: np.ndarray,
                       reference_mean: float) -> float:
        try:
            y_pred = np.asarray(predict_fn(X), dtype=float)
            if not np.isfinite(y_pred).all():
                return np.nan
            return RegressionMetrics(y, y_pred).calculate_rrmse(y_mean=reference_mean)
        except Exception:
            return np.nan

    def _clean(self,
               X: np.ndarray,
               y: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Remove NaN/inf rows from X and y."""
        if X is None or y is None:
            return None, None
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))
        X_c, y_c = X[mask], y[mask]
        return (X_c, y_c) if len(y_c) >= 3 else (None, None)

    def _insufficient_data_result(self,
                                   target: str,
                                   predictor_str: str,
                                   model_type: str,
                                   comparison_label: str) -> Dict:
        return {
            'target': target,
            'predictor_str': predictor_str,
            'model_type': model_type,
            'comparison_label': comparison_label,
            'observed_rrmse_a': np.nan,
            'observed_rrmse_b': np.nan,
            'observed_delta': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'is_significant': None,
            'n_a': 0,
            'n_b': 0,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'ci_method': self.ci_method,
            'error': 'Insufficient data',
        }

    def _run_bootstrap(self,
                       predict_fn: Callable,
                       X_a: np.ndarray,
                       y_a: np.ndarray,
                       X_b: np.ndarray,
                       y_b: np.ndarray,
                       reference_mean: float,
                       debug: bool = False) -> np.ndarray:
        """
        Vectorised bootstrap: precompute per-observation squared errors once,
        then resample those errors entirely in NumPy.

        Rows whose predictions are non-finite (NaN/inf) are dropped from the
        squared-error tables before resampling, and a warning is emitted.
        If fewer than 3 finite rows remain in either set, an all-NaN delta
        array is returned so the caller can flag the result as invalid.

        :param predict_fn: Trained model prediction function.
        :type predict_fn: Callable
        :param X_a: Features for evaluation set A.
        :type X_a: np.ndarray
        :param y_a: Targets for evaluation set A.
        :type y_a: np.ndarray
        :param X_b: Features for evaluation set B.
        :type X_b: np.ndarray
        :param y_b: Targets for evaluation set B.
        :type y_b: np.ndarray
        :param reference_mean: Conventional target mean for RRMSE normalisation.
        :type reference_mean: float
        :param debug: Print step-by-step progress.
        :type debug: bool
        :returns: Bootstrap distribution of the RRMSE delta (B − A).
        :rtype: np.ndarray
        """
        try:
            y_pred_a = predict_fn(X_a)
            y_pred_b = predict_fn(X_b)
        except Exception:
            return self._run_bootstrap_loop(
                predict_fn, X_a, y_a, X_b, y_b, reference_mean, debug
            )

        sq_err_a = (y_a - np.asarray(y_pred_a, dtype=float)) ** 2
        sq_err_b = (y_b - np.asarray(y_pred_b, dtype=float)) ** 2

        # ── Guard: drop rows with non-finite predictions ────────────────────────
        finite_a = np.isfinite(sq_err_a)
        finite_b = np.isfinite(sq_err_b)

        if not finite_a.all() or not finite_b.all():
            n_bad = int((~finite_a).sum() + (~finite_b).sum())
            print(f"  ⚠️  {n_bad} non-finite prediction(s) dropped from bootstrap "
                  f"(A: {(~finite_a).sum()}, B: {(~finite_b).sum()})")
            sq_err_a = sq_err_a[finite_a]
            sq_err_b = sq_err_b[finite_b]

        n_a, n_b = len(sq_err_a), len(sq_err_b)
        if n_a < 3 or n_b < 3:
            return np.full(self.n_bootstrap, np.nan)

        if debug:
            obs_rrmse_a = np.sqrt(sq_err_a.mean()) / reference_mean * 100
            obs_rrmse_b = np.sqrt(sq_err_b.mean()) / reference_mean * 100
            print(f"  [Vectorised] Precomputed sq errors: "
                  f"RRMSE_A={obs_rrmse_a:.3f}%, RRMSE_B={obs_rrmse_b:.3f}%")

        idx_a_all = np.random.choice(n_a, size=(self.n_bootstrap, n_a), replace=True)
        idx_b_all = np.random.choice(n_b, size=(self.n_bootstrap, n_b), replace=True)

        mse_a = sq_err_a[idx_a_all].mean(axis=1)
        mse_b = sq_err_b[idx_b_all].mean(axis=1)

        rrmse_a = np.sqrt(mse_a) / reference_mean * 100
        rrmse_b = np.sqrt(mse_b) / reference_mean * 100
        deltas = rrmse_b - rrmse_a

        if debug:
            print(f"  [Vectorised] Bootstrap delta: "
                  f"mean={deltas.mean():.3f}%, std={deltas.std():.3f}%")

        return deltas

    def _run_bootstrap_loop(self,
                            predict_fn: Callable,
                            X_a: np.ndarray,
                            y_a: np.ndarray,
                            X_b: np.ndarray,
                            y_b: np.ndarray,
                            reference_mean: float,
                            debug: bool = False) -> np.ndarray:
        """
        Original loop-based fallback. Used only if vectorised precomputation fails.
        Kept here so the vectorised method has a safe fallback path.
        """
        n_a, n_b = len(y_a), len(y_b)
        deltas = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            idx_a = np.random.choice(n_a, size=n_a, replace=True)
            idx_b = np.random.choice(n_b, size=n_b, replace=True)

            rrmse_a = self._compute_rrmse(predict_fn, X_a[idx_a], y_a[idx_a], reference_mean)
            rrmse_b = self._compute_rrmse(predict_fn, X_b[idx_b], y_b[idx_b], reference_mean)
            deltas[i] = rrmse_b - rrmse_a

            if debug and (i + 1) % 1000 == 0:
                print(f"  Iteration {i + 1}/{self.n_bootstrap}")

        return deltas

    def _bca_ci(self,
                bootstrap_deltas: np.ndarray,
                observed_delta: float,
                predict_fn: Callable,
                X_a: np.ndarray, y_a: np.ndarray,
                X_b: np.ndarray, y_b: np.ndarray,
                reference_mean: float,
                debug: bool = False) -> Tuple[float, float]:
        """
        Vectorised BCa CI using precomputed squared errors for the jackknife.

        Why the jackknife can also be vectorised
        ----------------------------------------
        The jackknife leave-one-out RRMSE for dataset A, dropping observation i, is:

            rrmse_a_jack[i] = sqrt( mean(sq_err_a[ all j != i ]) ) / ref_mean * 100

        Since sq_err_a is precomputed and fixed, this is a masked mean — no predict
        calls needed. For dataset B the same logic applies.

        The influence value for observation i in dataset A is:
            L_i = (n_a - 1) * (observed_delta - delta_jack[i])

        where delta_jack[i] = rrmse_b_full - rrmse_a_jack[i]
        (the full-dataset RRMSE for the unchanged dataset stays constant).
        """
        from scipy import stats

        n_a, n_b = len(y_a), len(y_b)

        # ── Precompute squared errors ───────────────────────────────────────────
        y_pred_a = predict_fn(X_a)
        y_pred_b = predict_fn(X_b)

        sq_err_a = (y_a - y_pred_a) ** 2  # shape (n_a,)
        sq_err_b = (y_b - y_pred_b) ** 2  # shape (n_b,)

        # Full-dataset RRMSE values (used as the fixed half in each jackknife step)
        rrmse_a_full = np.sqrt(sq_err_a.mean()) / reference_mean * 100
        rrmse_b_full = np.sqrt(sq_err_b.mean()) / reference_mean * 100

        # ── Bias correction (z0) ───────────────────────────────────────────────
        p_below = np.clip(np.mean(bootstrap_deltas < observed_delta), 1e-10, 1 - 1e-10)
        z0 = stats.norm.ppf(p_below)

        # ── Vectorised jackknife for dataset A ─────────────────────────────────
        # Leave out observation i: mean of all sq_err_a EXCEPT index i
        # sum_a - sq_err_a[i]
        # ─────────────────── then divide by (n_a - 1)
        sum_a = sq_err_a.sum()
        # loo_mse_a[i] = (sum_a - sq_err_a[i]) / (n_a - 1)   shape (n_a,)
        loo_mse_a = (sum_a - sq_err_a) / (n_a - 1)
        loo_rrmse_a = np.sqrt(loo_mse_a) / reference_mean * 100  # shape (n_a,)

        # delta_jack[i] = rrmse_b_full - loo_rrmse_a[i]
        delta_jack_a = rrmse_b_full - loo_rrmse_a  # shape (n_a,)
        L_a = (n_a - 1) * (observed_delta - delta_jack_a)  # shape (n_a,)

        # ── Vectorised jackknife for dataset B ─────────────────────────────────
        sum_b = sq_err_b.sum()
        loo_mse_b = (sum_b - sq_err_b) / (n_b - 1)
        loo_rrmse_b = np.sqrt(loo_mse_b) / reference_mean * 100  # shape (n_b,)

        # delta_jack[i] = loo_rrmse_b[i] - rrmse_a_full
        delta_jack_b = loo_rrmse_b - rrmse_a_full  # shape (n_b,)
        L_b = (n_b - 1) * (observed_delta - delta_jack_b)  # shape (n_b,)

        # ── Acceleration (a) from combined influence values ────────────────────
        L_all = np.concatenate([L_a, L_b])
        L_mean = L_all.mean()
        dev = L_mean - L_all
        numerator = np.sum(dev ** 3)
        denominator = 6 * (np.sum(dev ** 2)) ** 1.5
        a = numerator / denominator if abs(denominator) > 1e-10 else 0.0

        if debug:
            print(f"  [BCa vectorised] z0={z0:.4f}, a={a:.6f}")

        # ── Adjusted quantiles ─────────────────────────────────────────────────
        alpha = 1 - self.confidence_level
        z_lower = stats.norm.ppf(alpha / 2)
        z_upper = stats.norm.ppf(1 - alpha / 2)

        def adjusted_q(z_alpha):
            num = z0 + z_alpha
            den = 1 - a * (z0 + z_alpha)
            if abs(den) < 1e-10:
                return stats.norm.cdf(z_alpha)
            return stats.norm.cdf(z0 + num / den)

        p_lower = np.clip(adjusted_q(z_lower), 0.001, 0.999)
        p_upper = np.clip(adjusted_q(z_upper), 0.001, 0.999)

        return (
            np.percentile(bootstrap_deltas, 100 * p_lower),
            np.percentile(bootstrap_deltas, 100 * p_upper)
        )

    # =========================================================================
    # Export
    # =========================================================================

    def results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert list of result dicts to summary DataFrame."""
        rows = []
        for r in results:
            rows.append({
                'target': r.get('target', ''),
                'predictors': r.get('predictor_str', ''),
                'model_type': r.get('model_type', ''),
                'comparison': r.get('comparison_label', ''),
                'rrmse_a': r.get('observed_rrmse_a'),
                'rrmse_b': r.get('observed_rrmse_b'),
                'delta_rrmse': r.get('observed_delta'),
                'ci_lower': r.get('ci_lower'),
                'ci_upper': r.get('ci_upper'),
                'is_significant': r.get('is_significant'),
                'n_a': r.get('n_a'),
                'n_b': r.get('n_b'),
                'ci_method': r.get('ci_method'),
                'confidence_level': r.get('confidence_level'),
                'n_bootstrap': r.get('n_bootstrap'),
            })
        return pd.DataFrame(rows)

    def export_to_excel(self,
                        results: List[Dict],
                        output_path: str,
                        sheet_name: str = 'Bootstrap_Significance') -> None:
        """Export significance results to Excel."""
        df = self.results_to_dataframe(results)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        print(f"✓ Significance results saved to: {output_path}")