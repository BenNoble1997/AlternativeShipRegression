"""
Availability Subsample Auditor
==============================
Diagnoses whether conditioning a cohort on the availability of a *candidate*
variable (e.g. service speed ``Vs``) yields a representative subsample of the
full cohort, or a biased slice.

Motivation
----------
A minimum-count rule (n >= 10) is a *precision* gate: it asks whether the
retained sample is large enough for a stable estimate. It says nothing about
*representativeness* — whether the ships that happen to report the candidate
variable resemble the ships that do not. A precise estimate computed on an
unrepresentative subsample still estimates the wrong quantity.

This auditor answers the representativeness question directly. Given a cohort
and a candidate variable, it splits the cohort into "candidate present" vs
"candidate missing", then compares the two halves on the *commonly-available*
dimensions (variables recorded for essentially every ship, e.g. DWT, LOA,
Beam, MCR). If the two halves are indistinguishable on those dimensions, the
subsample is representative and the candidate may be admitted; if they differ
materially, admitting the candidate silently switches the population being
described.

Design
------
Standalone and composition-friendly: it depends only on a cohort DataFrame and
column names, not on any analyzer or foundation. It is intended to be called
uniformly across ship types so a single admission rule can be applied and
reported consistently.

Two complementary instruments are computed per commonly-available dimension:

- **Standardised Mean Difference (SMD)** — a unitless, sample-size-independent
  effect size for the imbalance between the two halves. Interpreted with the
  usual matching-diagnostic thresholds (|SMD| < 0.1 negligible, < 0.25 small,
  >= 0.25 notable). Because it does not depend on n, it remains informative
  when the "present" half is small.
- **Mann-Whitney U** — a non-parametric test of stochastic dominance, giving a
  p-value. It is underpowered at small n, so it is treated as *supporting*
  evidence rather than the arbiter; the representativeness verdict is driven by
  the SMD.

:ivar smd_negligible: |SMD| below this is treated as negligible imbalance.
:vartype smd_negligible: float
:ivar smd_small: |SMD| below this (but above ``smd_negligible``) is small.
:vartype smd_small: float
:ivar alpha: Significance level for the (supporting) Mann-Whitney test.
:vartype alpha: float
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats

from config.logger_config import LoggerMixin


class AvailabilitySubsampleAuditor(LoggerMixin):
    """
    Audit whether splitting a cohort on a candidate variable's availability
    produces a representative subsample.

    Designed to be used via composition, not inheritance.

    :param smd_negligible: |SMD| threshold below which imbalance on a dimension
        is considered negligible.
    :type smd_negligible: float
    :param smd_small: |SMD| threshold below which imbalance is considered small
        (values at or above this are flagged as notable).
    :type smd_small: float
    :param alpha: Significance level for the supporting Mann-Whitney U test.
    :type alpha: float
    :param verbose: Whether to emit informational logging via ``LoggerMixin``.
    :type verbose: bool
    """

    def __init__(self,
                 smd_negligible: float = 0.10,
                 smd_small: float = 0.25,
                 alpha: float = 0.05,
                 verbose: bool = True):
        self.smd_negligible = smd_negligible
        self.smd_small = smd_small
        self.alpha = alpha
        self.verbose = verbose

    # =========================================================================
    # Public API
    # =========================================================================

    def audit_variable(self,
                       cohort: pd.DataFrame,
                       candidate: str,
                       common_variables: List[str],
                       cohort_label: str = '',
                       debug: bool = False) -> Optional[Dict]:
        """
        Audit one candidate variable within one cohort.

        Splits ``cohort`` into ships for which ``candidate`` is present
        (non-NaN) and ships for which it is missing, then compares the two
        halves on each of ``common_variables`` using SMD and Mann-Whitney U.

        Step-by-step data flow:

        1. Split ``cohort`` on availability of ``candidate`` -> present / missing
        2. For each dimension in ``common_variables``:
           compute per-dimension SMD and Mann-Whitney p (on rows where the
           dimension itself is non-NaN in both halves)
        3. Aggregate: max |SMD| and count of notable imbalances across dimensions
        4. Verdict: ``representative`` iff every dimension's |SMD| < ``smd_small``
        5. Return a flat result dict (per-dimension detail nested under
           ``dimensions``)

        :param cohort: The full cohort for one ship type / fuel category.
        :type cohort: pd.DataFrame
        :param candidate: Column name whose availability defines the split
            (e.g. ``'Vs'``).
        :type candidate: str
        :param common_variables: Commonly-available dimensions on which to
            compare the two halves (e.g. ``['DWT', 'LOA (M)', 'BEAM (M)',
            'MCR (KW)']``). The candidate itself is ignored if present here.
        :type common_variables: list[str]
        :param cohort_label: Human-readable cohort description for reporting.
        :type cohort_label: str
        :param debug: Print step-by-step progress.
        :type debug: bool
        :returns: Result dict, or ``None`` if the candidate is absent from the
            cohort or one half is empty (nothing to compare).
        :rtype: dict or None
        """
        if candidate not in cohort.columns:
            if debug:
                print(f"[audit] candidate '{candidate}' not a column in cohort "
                      f"'{cohort_label}' — skipping")
            return None

        # -- Step 1: split on availability -----------------------------------
        present_mask = cohort[candidate].notna()
        present = cohort[present_mask]
        missing = cohort[~present_mask]

        n_present, n_missing = len(present), len(missing)

        if debug:
            print(f"\n[audit] {cohort_label} | candidate='{candidate}': "
                  f"n_present={n_present}, n_missing={n_missing}")

        # If either half is empty there is nothing to compare: the split is
        # degenerate (candidate is either universal or absent). Representative
        # by construction when nothing is dropped; undefined when all dropped.
        if n_present == 0 or n_missing == 0:
            if debug:
                print(f"  degenerate split (one half empty) — "
                      f"representativeness undefined, returning None")
            return None

        # -- Step 2: per-dimension comparison --------------------------------
        dimensions = []
        for dim in common_variables:
            if dim == candidate or dim not in cohort.columns:
                continue

            dim_result = self._compare_dimension(
                present[dim], missing[dim], dim, debug=debug
            )
            if dim_result is not None:
                dimensions.append(dim_result)

        if not dimensions:
            if debug:
                print(f"  no comparable common dimensions — returning None")
            return None

        # -- Step 3: aggregate -----------------------------------------------
        abs_smds = [d['abs_smd'] for d in dimensions if np.isfinite(d['abs_smd'])]
        max_abs_smd = max(abs_smds) if abs_smds else np.nan
        n_notable = sum(
            1 for d in dimensions
            if np.isfinite(d['abs_smd']) and d['abs_smd'] >= self.smd_small
        )
        n_significant = sum(
            1 for d in dimensions
            if d['mw_pvalue'] is not None and d['mw_pvalue'] < self.alpha
        )

        # -- Step 4: verdict (driven by SMD, not the p-values) ---------------
        representative = (n_notable == 0)

        if debug:
            verdict = 'REPRESENTATIVE' if representative else 'BIASED'
            print(f"  max|SMD|={max_abs_smd:.3f}, notable={n_notable}, "
                  f"mw_sig={n_significant}/{len(dimensions)} -> {verdict}")

        # -- Step 5: flat result ---------------------------------------------
        return {
            'cohort_label': cohort_label,
            'candidate': candidate,
            'n_present': n_present,
            'n_missing': n_missing,
            'fraction_present': n_present / (n_present + n_missing),
            'max_abs_smd': max_abs_smd,
            'n_dimensions': len(dimensions),
            'n_notable_smd': n_notable,
            'n_significant_mw': n_significant,
            'representative': representative,
            'dimensions': dimensions,
        }

    def audit_matrix(self,
                     cohorts: Dict[str, pd.DataFrame],
                     candidates: List[str],
                     common_variables: List[str],
                     debug: bool = False) -> pd.DataFrame:
        """
        Audit several candidate variables across several cohorts at once.

        Applies :meth:`audit_variable` for every (cohort, candidate) pair and
        collects the flat summary rows into a single DataFrame suitable for a
        supplementary table. Per-dimension detail is not expanded here; call
        :meth:`audit_variable` directly for that.

        :param cohorts: Mapping of cohort label -> cohort DataFrame
            (e.g. ``{'Container / Dual fuel': df1, 'Container / Bio': df2}``).
        :type cohorts: dict[str, pd.DataFrame]
        :param candidates: Candidate variables to audit (e.g. ``['Vs', 'LSW']``).
        :type candidates: list[str]
        :param common_variables: Commonly-available comparison dimensions.
        :type common_variables: list[str]
        :param debug: Print step-by-step progress.
        :type debug: bool
        :returns: One row per (cohort, candidate) pair that produced a verdict;
            degenerate / absent pairs are omitted.
        :rtype: pd.DataFrame
        """
        rows = []
        for cohort_label, cohort in cohorts.items():
            for candidate in candidates:
                result = self.audit_variable(
                    cohort=cohort,
                    candidate=candidate,
                    common_variables=common_variables,
                    cohort_label=cohort_label,
                    debug=debug,
                )
                if result is None:
                    continue
                rows.append({
                    'cohort': result['cohort_label'],
                    'candidate': result['candidate'],
                    'n_present': result['n_present'],
                    'n_missing': result['n_missing'],
                    'fraction_present': result['fraction_present'],
                    'max_abs_smd': result['max_abs_smd'],
                    'n_notable_smd': result['n_notable_smd'],
                    'n_significant_mw': result['n_significant_mw'],
                    'representative': result['representative'],
                })

        return pd.DataFrame(rows)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compare_dimension(self,
                           present_values: pd.Series,
                           missing_values: pd.Series,
                           dim_name: str,
                           debug: bool = False) -> Optional[Dict]:
        """
        Compare one commonly-available dimension across the two halves.

        Rows where the dimension itself is NaN are dropped from each half
        before comparison (a ship can report the candidate yet still be missing
        a comparison dimension). Requires at least two finite values per half.

        :param present_values: Dimension values for the "candidate present" half.
        :type present_values: pd.Series
        :param missing_values: Dimension values for the "candidate missing" half.
        :type missing_values: pd.Series
        :param dim_name: Name of the dimension (for reporting).
        :type dim_name: str
        :param debug: Print per-dimension detail.
        :type debug: bool
        :returns: Per-dimension result dict, or ``None`` if either half has
            fewer than two finite values.
        :rtype: dict or None
        """
        a = pd.to_numeric(present_values, errors='coerce').to_numpy()
        b = pd.to_numeric(missing_values, errors='coerce').to_numpy()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            if debug:
                print(f"    {dim_name}: insufficient finite values "
                      f"(present={len(a)}, missing={len(b)}) — skipped")
            return None

        smd = self._standardised_mean_difference(a, b)

        # Mann-Whitney U (two-sided). Guard the degenerate all-identical case,
        # which raises in scipy; treat it as "no detectable difference".
        try:
            mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative='two-sided')
            mw_stat, mw_p = float(mw_stat), float(mw_p)
        except ValueError:
            mw_stat, mw_p = np.nan, None

        if debug:
            p_str = f"{mw_p:.4f}" if mw_p is not None else "n/a"
            print(f"    {dim_name}: SMD={smd:+.3f} "
                  f"(present mean={a.mean():.3g}, missing mean={b.mean():.3g}), "
                  f"MW p={p_str}")

        return {
            'dimension': dim_name,
            'n_present': int(len(a)),
            'n_missing': int(len(b)),
            'mean_present': float(a.mean()),
            'mean_missing': float(b.mean()),
            'smd': float(smd),
            'abs_smd': float(abs(smd)),
            'mw_stat': mw_stat,
            'mw_pvalue': mw_p,
            'notable': bool(abs(smd) >= self.smd_small),
        }

    @staticmethod
    def _standardised_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
        """
        Standardised mean difference between two samples.

        Defined as ``(mean(a) - mean(b)) / s_pooled`` where ``s_pooled`` is the
        pooled standard deviation using the sample variances of the two groups:

            s_pooled = sqrt( (var(a) + var(b)) / 2 )

        This is the balance-diagnostic convention (equal weighting of the two
        group variances), independent of the group sizes so that it stays
        interpretable when one half is small. Returns ``0.0`` when both means
        coincide and ``inf`` when the pooled SD is zero but the means differ.

        :param a: First sample (finite values only).
        :type a: np.ndarray
        :param b: Second sample (finite values only).
        :type b: np.ndarray
        :returns: Standardised mean difference.
        :rtype: float
        """
        mean_diff = a.mean() - b.mean()
        # Sample variance (ddof=1) matches the balance-diagnostic convention.
        var_a = a.var(ddof=1)
        var_b = b.var(ddof=1)
        s_pooled = np.sqrt((var_a + var_b) / 2.0)

        if s_pooled == 0:
            return 0.0 if mean_diff == 0 else np.inf
        return mean_diff / s_pooled