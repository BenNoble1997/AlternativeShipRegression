"""
Bootstrap uncertainty for structure-preserving coefficient comparison.
=====================================================================

Concept
-------
``CoefficientComparisonAnalyzer`` already fits one model per cohort with the
structure held fixed, so the three coefficient vectors are directly
comparable.  What it does not report is how well each coefficient is
*determined* by its cohort.  With n=16 alternative-fuel vessels a coefficient
may be essentially arbitrary, and a point estimate gives no way to tell.

The bootstrap answers this by refitting the same fixed structure on B
resamples of each cohort.  The spread of the resulting coefficient values is
the sampling uncertainty of that coefficient.  Because the effects
(distribution, fuel, total) are differences between cohorts, computing them
per replicate propagates the uncertainty into the effects themselves, which is
what licenses a statement such as "the exponent on DWT is lower for methanol
vessels" or its refusal.

Design decisions, and why
-------------------------
1. **Structure is frozen across replicates.**  Each replicate refits the
   *same* descriptor (same MFP transforms, same active Lasso features) rather
   than re-running selection.  Two reasons: the coefficient dictionaries then
   share keys across replicates so they can be stacked, and the resulting
   interval isolates estimation uncertainty from model-selection uncertainty.
   Re-selecting per replicate measures a different and larger quantity — worth
   doing as a separate robustness check, but it is not what a coefficient
   confidence interval means.

2. **Normalisation parameters are frozen.**  They are fit once on the full
   conventional cohort, exactly as the point estimate does.  Refitting them
   per replicate would make coefficients from different replicates live on
   slightly different scales and so not be poolable.

3. **Cohorts are resampled independently.**  Each cohort is resampled to its
   own size, so the conventional cohort yields tight intervals and the
   alternative cohort wide ones, which is the honest picture.  Note the
   caveats in ``effect_draws``.

4. **Percentile intervals, not mean +/- sd.**  At small n the replicate
   distribution is heavy-tailed: a minority of degenerate resamples produce
   enormous coefficients that dominate a standard deviation.  Percentiles are
   bounded by what the replicates actually produced and do not imply symmetry.

5. **Failed refits are counted, not silently dropped.**  A resample of 16
   vessels contains about 10 distinct ships, occasionally fewer, which can
   leave the design matrix near-singular.  The failure rate and the
   distinct-observation count per replicate are recorded because they are the
   mechanism behind the wide intervals and should be reported alongside them.

Usage
-----
Mix into the analyzer and apply the one patch described in ``PATCH`` below::

    class CoefficientComparisonAnalyzer(CoefficientBootstrapMixin, LoggerMixin):
        ...

    analyzer.analyze_all_targets()
    boot = analyzer.bootstrap_all_targets(n_boot=1000, seed=42)
    table = analyzer.coefficient_uncertainty_table()

PATCH
-----
``_analyze_single_target_threeway`` discards the cleaned design matrices once
it has fitted.  The bootstrap needs them.  Insert the following immediately
after the ``X_alt_clean, y_alt_clean, mask_alt = ...`` unpacking in Step 3::

    self._cache_fit_inputs(
        key=key,
        target=target,
        predictors=best_combo,
        model_type=best_model_type,
        base_descriptor=base_descriptor,
        arrays={
            'conventional_full':    (X_conv_full_clean,    y_conv_full_clean),
            'conventional_matched': (X_conv_matched_clean, y_conv_matched_clean),
            'alternative':          (X_alt_clean,          y_alt_clean),
        },
    )

Nothing else in the existing class changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


COHORTS = ('conventional_full', 'conventional_matched', 'alternative')

#: Effect name -> (from_cohort, to_cohort).  Mirrors the point-estimate
#: decomposition in ``_compare_coefficients_threeway``.
EFFECTS = {
    'distribution_effect': ('conventional_full', 'conventional_matched'),
    'fuel_effect': ('conventional_matched', 'alternative'),
    'total_effect': ('conventional_full', 'alternative'),
}


# ============================================================================
# Containers
# ============================================================================

@dataclass
class CohortBootstrap:
    """Bootstrap replicate coefficients for one cohort of one target.

    :param draws: One row per successful replicate, one column per
        coefficient.  Column names match ``_extract_clean_coefficients``
        output plus ``intercept`` where the model has one.
    :param n_unique: Number of distinct observations in each successful
        replicate's resample.  Diagnostic for why intervals are wide.
    """

    cohort: str
    n_obs: int
    n_boot_requested: int
    draws: pd.DataFrame
    n_unique: np.ndarray
    point: Dict[str, float] = field(default_factory=dict)

    @property
    def n_boot_successful(self) -> int:
        return len(self.draws)

    @property
    def failure_rate(self) -> float:
        """Fraction of replicates whose refit failed or returned non-finite
        coefficients.  A high value is itself a reportable finding."""
        if self.n_boot_requested == 0:
            return np.nan
        return 1.0 - self.n_boot_successful / self.n_boot_requested

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """Percentile summary, one row per coefficient.

        :param level: Central interval mass, e.g. 0.95 for a 2.5-97.5 band.
        """
        if self.draws.empty:
            return pd.DataFrame()

        alpha = (1.0 - level) / 2.0
        q = self.draws.quantile([alpha, 0.25, 0.5, 0.75, 1.0 - alpha])

        out = pd.DataFrame({
            'coefficient': self.draws.columns,
            'point': [self.point.get(c, np.nan) for c in self.draws.columns],
            'median': q.loc[0.5].to_numpy(),
            'q_lo': q.loc[alpha].to_numpy(),
            'q_hi': q.loc[1.0 - alpha].to_numpy(),
            'iqr': (q.loc[0.75] - q.loc[0.25]).to_numpy(),
        })
        out['cohort'] = self.cohort
        out['n_obs'] = self.n_obs
        out['n_boot'] = self.n_boot_successful
        out['failure_rate'] = self.failure_rate
        # Width relative to the point estimate: the direct analogue of the
        # "sd exceeds the mean" flag used in the performance tables.
        with np.errstate(divide='ignore', invalid='ignore'):
            out['rel_width'] = np.abs(out['q_hi'] - out['q_lo']) / np.abs(out['point'])
        out['unstable'] = out['rel_width'] > 1.0
        return out


@dataclass
class TargetBootstrap:
    """Bootstrap results for all three cohorts of one target."""

    target: str
    predictors: tuple
    model_type: str
    cohorts: Dict[str, CohortBootstrap]

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        frames = [cb.summary(level) for cb in self.cohorts.values()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, 'target', self.target)
        out.insert(1, 'model_type', self.model_type)
        return out

    def effect_draws(self, effect: str) -> pd.DataFrame:
        """Per-replicate distribution of a between-cohort coefficient change.

        Replicate *b* of the source cohort is paired with replicate *b* of the
        target cohort.  Since the cohorts are resampled independently this is
        a valid draw from the difference distribution.

        Two caveats worth stating in the paper:

        * ``conventional_matched`` is a subset of ``conventional_full``, so
          resampling them independently ignores their positive correlation.
          The distribution effect interval is therefore conservative (too
          wide), not anti-conservative.
        * The matched cohort's membership was determined by matching against
          the observed alternative vessels.  Resampling does not re-run the
          matching, so uncertainty in the matching itself is excluded.

        :param effect: One of ``distribution_effect``, ``fuel_effect``,
            ``total_effect``.
        """
        if effect not in EFFECTS:
            raise ValueError(f"Unknown effect {effect!r}; expected one of {list(EFFECTS)}")

        src, dst = EFFECTS[effect]
        a = self.cohorts.get(src)
        b = self.cohorts.get(dst)
        if a is None or b is None or a.draws.empty or b.draws.empty:
            return pd.DataFrame()

        shared = [c for c in a.draws.columns if c in b.draws.columns]
        if not shared:
            return pd.DataFrame()

        # Truncate to the common number of successful replicates.  Failures
        # are not aligned across cohorts, so this pairs replicate ranks rather
        # than resample identities -- harmless, since the cohorts are
        # independent and the draws are exchangeable within each cohort.
        m = min(len(a.draws), len(b.draws))
        return (b.draws[shared].iloc[:m].reset_index(drop=True)
                - a.draws[shared].iloc[:m].reset_index(drop=True))

    def effect_summary(self, level: float = 0.95) -> pd.DataFrame:
        """Percentile intervals for all three effects, with a zero-exclusion
        flag.

        ``excludes_zero`` is the operative column: it is the test of whether a
        coefficient change is distinguishable from sampling noise, and so
        whether the change is worth interpreting at all.
        """
        alpha = (1.0 - level) / 2.0
        rows = []

        for effect in EFFECTS:
            d = self.effect_draws(effect)
            if d.empty:
                continue
            q = d.quantile([alpha, 0.5, 1.0 - alpha])
            for coef in d.columns:
                lo, med, hi = q.loc[alpha, coef], q.loc[0.5, coef], q.loc[1.0 - alpha, coef]
                rows.append({
                    'target': self.target,
                    'model_type': self.model_type,
                    'effect': effect,
                    'coefficient': coef,
                    'median': med,
                    'q_lo': lo,
                    'q_hi': hi,
                    'excludes_zero': bool(lo > 0 or hi < 0),
                    'n_boot': len(d),
                })

        return pd.DataFrame(rows)


# ============================================================================
# Mixin
# ============================================================================

class CoefficientBootstrapMixin:
    """Adds bootstrap coefficient intervals to ``CoefficientComparisonAnalyzer``.

    Relies on three methods of the host class: ``_copy_descriptor``,
    ``_extract_clean_coefficients``, and the ``_cache_fit_inputs`` hook
    defined here.
    """

    # -- cache populated by the patched _analyze_single_target_threeway ------

    def _cache_fit_inputs(self, key, target, predictors, model_type,
                          base_descriptor, arrays) -> None:
        """Stash the cleaned arrays and base descriptor for later resampling.

        Called from ``_analyze_single_target_threeway``.  Storing rather than
        recomputing guarantees the bootstrap sees exactly the rows the point
        estimate saw, including the nonlinear-model cleaning mask.
        """
        if not hasattr(self, '_fit_inputs'):
            self._fit_inputs: Dict[Any, dict] = {}

        self._fit_inputs[key] = {
            'target': target,
            'predictors': tuple(predictors),
            'model_type': model_type,
            'base_descriptor': base_descriptor,
            'arrays': {k: (np.asarray(X), np.asarray(y)) for k, (X, y) in arrays.items()},
        }

    # -- core resampling loop ------------------------------------------------

    def _bootstrap_cohort(self, base_descriptor, X, y, cohort, n_boot, rng,
                          point_coeffs=None) -> CohortBootstrap:
        """Refit the frozen structure on ``n_boot`` resamples of one cohort.

        Each iteration draws ``n`` row indices with replacement from the
        cohort's ``n`` rows, clones the descriptor so no state leaks between
        replicates, and refits with ``force_structure=True``.
        """
        n = len(y)
        records: list[dict] = []
        uniques: list[int] = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                desc_b = self._copy_descriptor(base_descriptor)
                desc_b.refit(X[idx], y[idx], force_structure=True)
                coeffs = self._extract_clean_coefficients(desc_b)
                if getattr(desc_b, 'intercept', None) is not None:
                    coeffs = {**coeffs, 'intercept': desc_b.intercept}
            except Exception:
                # Degenerate resample: near-singular design, non-convergent
                # nonlinear fit, or too few distinct rows.  Counted as a
                # failure rather than imputed.
                continue

            if not coeffs or not all(np.isfinite(v) for v in coeffs.values()):
                continue

            records.append(coeffs)
            uniques.append(int(np.unique(idx).size))

        draws = pd.DataFrame.from_records(records) if records else pd.DataFrame()

        # Guard against structural drift: a replicate that somehow produced a
        # different feature set would introduce NaNs and silently distort the
        # percentiles.  Keep only coefficients present in every replicate.
        if not draws.empty:
            complete = draws.columns[draws.notna().all(axis=0)]
            draws = draws[complete]

        return CohortBootstrap(
            cohort=cohort,
            n_obs=n,
            n_boot_requested=n_boot,
            draws=draws,
            n_unique=np.asarray(uniques, dtype=int),
            point=dict(point_coeffs or {}),
        )

    # -- public API ----------------------------------------------------------

    def bootstrap_target_coefficients(self, target: str, n_boot: int = 1000,
                                      seed: Optional[int] = None,
                                      cohorts: Sequence[str] = COHORTS,
                                      ) -> Optional[TargetBootstrap]:
        """Bootstrap the coefficients of one target across all three cohorts.

        Requires ``analyze_all_targets`` to have run first, since the cached
        arrays and the selected structure come from that pass.

        :param n_boot: Replicates per cohort.  1000 is comfortable for stable
            2.5/97.5 percentiles; 200 is enough for a rough interval.
        :param seed: Seed for reproducibility.  Record it in the paper.
        """
        cache = getattr(self, '_fit_inputs', {})
        key = next((k for k in cache if cache[k]['target'] == target), None)
        if key is None:
            if getattr(self, 'verbose', False):
                print(f"  No cached fit inputs for {target}; run analyze_all_targets first.")
            return None

        entry = cache[key]
        rng = np.random.default_rng(seed)

        point_by_cohort = {
            'conventional_full': 'coefficients_full',
            'conventional_matched': 'coefficients_matched',
            'alternative': 'coefficients_alternative',
        }
        results_df = getattr(self, 'comparison_results', None)
        point_row = None
        if results_df is not None and not results_df.empty:
            hit = results_df[results_df['target'] == target]
            if not hit.empty:
                point_row = hit.iloc[0]

        out: Dict[str, CohortBootstrap] = {}
        for cohort in cohorts:
            arrays = entry['arrays'].get(cohort)
            if arrays is None:
                continue
            X, y = arrays
            pts = {}
            if point_row is not None:
                pts = point_row.get(point_by_cohort[cohort], {}) or {}
            out[cohort] = self._bootstrap_cohort(
                base_descriptor=entry['base_descriptor'],
                X=X, y=y, cohort=cohort, n_boot=n_boot, rng=rng,
                point_coeffs=pts,
            )

        return TargetBootstrap(
            target=target,
            predictors=entry['predictors'],
            model_type=entry['model_type'],
            cohorts=out,
        )

    def bootstrap_all_targets(self, n_boot: int = 1000, seed: Optional[int] = None,
                              ) -> Dict[str, TargetBootstrap]:
        """Bootstrap every target that ``analyze_all_targets`` fitted.

        Each target gets its own seeded generator derived from ``seed`` so
        results are reproducible and independent of iteration order.
        """
        cache = getattr(self, '_fit_inputs', {})
        seeds = np.random.SeedSequence(seed).spawn(max(len(cache), 1))

        results: Dict[str, TargetBootstrap] = {}
        for (key, entry), s in zip(cache.items(), seeds):
            target = entry['target']
            if getattr(self, 'verbose', False):
                print(f"  Bootstrapping {target} ({entry['model_type']}), B={n_boot}")
            tb = self.bootstrap_target_coefficients(
                target=target, n_boot=n_boot,
                seed=int(s.generate_state(1)[0]),
            )
            if tb is not None:
                results[target] = tb

        self.bootstrap_results = results
        return results

    # -- reporting -----------------------------------------------------------

    def coefficient_uncertainty_table(self, level: float = 0.95) -> pd.DataFrame:
        """Tidy per-coefficient intervals for every target and cohort."""
        boots = getattr(self, 'bootstrap_results', {})
        frames = [tb.summary(level) for tb in boots.values()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def effect_uncertainty_table(self, level: float = 0.95) -> pd.DataFrame:
        """Tidy intervals for the distribution, fuel and total effects.

        This is the table that supports or refuses the claim that
        alternative-fuel vessels differ dimensionally: read the
        ``excludes_zero`` column.
        """
        boots = getattr(self, 'bootstrap_results', {})
        frames = [tb.effect_summary(level) for tb in boots.values()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def bootstrap_diagnostics(self) -> pd.DataFrame:
        """Per-cohort replicate failure rate and distinct-observation counts.

        Report this alongside wide intervals.  It distinguishes "the fit is
        usually fine but a few resamples were degenerate" from "the fit never
        worked", which the intervals alone cannot.
        """
        rows = []
        for target, tb in getattr(self, 'bootstrap_results', {}).items():
            for cohort, cb in tb.cohorts.items():
                rows.append({
                    'target': target,
                    'model_type': tb.model_type,
                    'cohort': cohort,
                    'n_obs': cb.n_obs,
                    'n_boot_requested': cb.n_boot_requested,
                    'n_boot_successful': cb.n_boot_successful,
                    'failure_rate': cb.failure_rate,
                    'mean_distinct_obs': float(np.mean(cb.n_unique)) if cb.n_unique.size else np.nan,
                    'min_distinct_obs': int(np.min(cb.n_unique)) if cb.n_unique.size else -1,
                })
        return pd.DataFrame(rows)