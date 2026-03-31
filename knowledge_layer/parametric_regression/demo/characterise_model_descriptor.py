"""
Characterisation Suite — ModelDescriptor
==========================================

Purpose
-------
Records the *exact current behaviour* of ModelDescriptor before the
planned refactor.  Running this script produces a JSON snapshot of every
observable output.  After the refactor, running it again and diffing the
two JSON files is your pass/fail signal.

Usage
-----
Before the refactor:
    python characterise_model_descriptor.py
    # Produces: descriptor_characterisation_BEFORE.json

After the refactor:
    python characterise_model_descriptor.py
    # Produces: descriptor_characterisation_AFTER.json

Then diff:
    python -c "
    import json
    before = json.load(open('descriptor_characterisation_BEFORE.json'))
    after  = json.load(open('descriptor_characterisation_AFTER.json'))
    diffs  = {k: (before[k], after[k])
              for k in before if before[k] != after.get(k)}
    print(f'{len(diffs)} differences found.')
    for k, (b, a) in diffs.items():
        print(f'  {k}')
        print(f'    BEFORE: {b}')
        print(f'    AFTER : {a}')
    "

Coverage
--------
Every code path through get_formula_string() is exercised:

    Model type          Notation    include_coefficients    only_active
    ──────────────────────────────────────────────────────────────────
    mfp (fp1)           math        True / False
    mfp (fp2 distinct)  latex       True / False
    mfp (fp2 repeated)  python      True / False
    mfp (excluded var)  math        True
    linear              all three   True
    ridge               math        True
    lasso               math        True / False (only_active)
    polynomial-2        math        True
    polynomial-2-lasso  all three   True / False
    power               all three   True / False
    logarithmic         math        True / False
    exponential         math        True / False
    power-offset        math        True / False
    power-log-multi     math        True
    exponential-log-m   math        True

Structural properties captured for each descriptor:
    - effective_df
    - regularization_type, regularization_alpha
    - active_features
    - coefficient keys (not values — values change with random seed)
    - intercept sign (positive / negative / None)
    - transformations structure (types and powers, not numeric coefs)

NOTE: Coefficient *values* are deliberately excluded from the snapshot
because they vary with random seed.  What we are testing is that the
*structure* and *rendering logic* are preserved after the refactor.
"""

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np

from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
from knowledge_layer.parametric_regression.parametric_model_description_class import ModelDescriptor

from knowledge_layer.parametric_regression.parametric_formula_render import (
    ParametricFormulaRenderer,
)

# ── Reproducible data ───────────────────────────────────────────────────────
np.random.seed(0)

NOTATIONS = ['math', 'latex', 'python']


# ============================================================================
# Helpers
# ============================================================================

def make_snapshot() -> dict:
    """Run all characterisation cases and return a flat snapshot dict."""
    snap = {}

    # ── Data generators ─────────────────────────────────────────────────────
    n = 150
    DWT  = np.random.uniform(5_000, 120_000, n)
    GT   = np.random.uniform(3_000,  80_000, n)
    LOA  = 50 + 4.2 * np.sqrt(DWT) - 0.7 * np.log(GT) + np.random.normal(0, 5, n)
    LOA  = np.clip(LOA, 1, None)

    X2   = np.column_stack([DWT, GT])      # two predictors
    X1   = DWT.reshape(-1, 1)              # one predictor
    X3   = np.column_stack([DWT, GT,       # three predictors
                            np.random.uniform(1, 10, n)])

    # ── Section 1: MFP model family ─────────────────────────────────────────
    _snap_mfp(snap, X2, LOA, ['DWT', 'GT'], 'LOA')

    # ── Section 2: Linear family ────────────────────────────────────────────
    for mt in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']:
        _snap_linear(snap, mt, X2, LOA, ['DWT', 'GT'], 'LOA')

    # ── Section 3: Polynomial family ────────────────────────────────────────
    for mt in ['polynomial-2', 'polynomial-3',
               'polynomial-2-lasso', 'polynomial-2-ridge']:
        _snap_polynomial(snap, mt, X2, LOA, ['DWT', 'GT'], 'LOA')

    # ── Section 4: Univariate nonlinear ─────────────────────────────────────
    for mt in ['power', 'logarithmic', 'exponential',
               'power-offset', 'exponential-offset', 'logarithmic-power']:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _snap_univariate(snap, mt, X1, LOA, ['DWT'], 'LOA')

    # ── Section 5: Log-transform multivariate ───────────────────────────────
    LOA_pos = np.clip(LOA, 1, None)
    DWT_pos = np.clip(DWT, 1, None)
    GT_pos  = np.clip(GT,  1, None)
    X2_pos  = np.column_stack([DWT_pos, GT_pos])

    for mt in ['power-log-multivariate', 'exponential-log-multivariate']:
        _snap_log_multivariate(snap, mt, X2_pos, LOA_pos, ['DWT', 'GT'], 'LOA')

    # ── Section 6: Serialisation round-trip ─────────────────────────────────
    _snap_serialisation(snap, X2, LOA, ['DWT', 'GT'], 'LOA')

    # ── Section 7: refit() structural preservation ──────────────────────────
    _snap_refit(snap, X2, LOA, ['DWT', 'GT'], 'LOA')

    # ── Section 8: Effective DF table ───────────────────────────────────────
    _snap_effective_df(snap, X2, LOA, ['DWT', 'GT'])

    # ── Section 9: Formatting utilities (pure functions) ────────────────────
    _snap_formatting_utilities(snap)

    return snap


# ============================================================================
# Section helpers
# ============================================================================

def _fit_and_describe(model_type, X, y, var_names, target, **kwargs):
    """Fit model_type and return a ModelDescriptor."""
    model, prep = ModelFactory.create_and_fit(model_type, X, y, **kwargs)
    desc = ModelDescriptor.from_fitted_model(
        model, prep, model_type, var_names, target
    )
    return desc


def _record_descriptor_structure(snap, prefix, desc):
    """
    Record structural (non-numeric) properties of a descriptor.
    Coefficient *values* are excluded; only keys and sign are recorded.
    """
    snap[f'{prefix}.model_type']          = desc.model_type
    snap[f'{prefix}.dependent_var']       = desc.dependent_var
    snap[f'{prefix}.independent_vars']    = desc.independent_vars
    snap[f'{prefix}.effective_df']        = desc.get_effective_parameters()
    snap[f'{prefix}.regularization_type'] = desc.regularization_type
    snap[f'{prefix}.active_features']     = sorted(desc.active_features or [])
    snap[f'{prefix}.coef_keys']           = sorted(desc.coefficients.keys()
                                                    if desc.coefficients else [])
    snap[f'{prefix}.intercept_sign']      = (
        'positive' if (desc.intercept or 0) > 0
        else 'negative' if (desc.intercept or 0) < 0
        else 'zero_or_none'
    )
    # MFP: capture transform types and powers (not coefficient values)
    if desc.transformations:
        snap[f'{prefix}.transform_types'] = {
            str(k): v[0] for k, v in desc.transformations.items()
        }
        snap[f'{prefix}.transform_powers'] = {
            str(k): v[1] for k, v in desc.transformations.items()
        }


def _record_all_formulas(snap, prefix, desc):
    """Record get_formula_string() output for all notation × include_coefficients combos."""
    for notation in NOTATIONS:
        for include in [True, False]:
            key = f'{prefix}.formula.{notation}.include={include}'
            snap[key] = desc.get_formula_string(
                include_coefficients=include,
                notation=notation,
            )
        # only_active=False (Lasso only — harmless for others)
        key_inactive = f'{prefix}.formula.{notation}.only_active=False'
        snap[key_inactive] = desc.get_formula_string(
            include_coefficients=True,
            notation=notation,
            only_active=False,
        )

    # Coefficient names
    snap[f'{prefix}.coef_names.active']   = desc.get_coefficient_names(only_active=True)
    snap[f'{prefix}.coef_names.all']      = desc.get_coefficient_names(only_active=False)


# ── Section 1: MFP ──────────────────────────────────────────────────────────

def _snap_mfp(snap, X, y, var_names, target):
    """
    Exercise MFP with three different transformation outcomes by
    constructing descriptors directly with known transformations.

    We use from_fitted_model() for a real fit (fp1 case) and construct
    synthetic descriptors for fp2 and excluded-variable cases, since
    whether MFP selects fp2 depends on data characteristics.
    """
    prefix = 'mfp_fp1_real'
    desc = _fit_and_describe('mfp', X, y, var_names, target,
                             feature_names=var_names)
    _record_descriptor_structure(snap, prefix, desc)
    _record_all_formulas(snap, prefix, desc)

    # ── Synthetic FP2 descriptor (distinct powers) ──
    prefix = 'mfp_fp2_distinct_synthetic'
    desc_fp2 = ModelDescriptor(
        model_type='mfp',
        independent_vars=['DWT', 'GT'],
        dependent_var='LOA',
        transformations={
            0: ('fp2', [0.5, 2]),    # DWT → √DWT + DWT²
            1: ('fp1', [-1]),        # GT  → 1/GT
        },
        coefficients={
            '√DWT': 3.14,
            'DWT^2': -0.00012,
            '1/GT': -500.0,
        },
        intercept=45.0,
    )
    _record_descriptor_structure(snap, prefix, desc_fp2)
    _record_all_formulas(snap, prefix, desc_fp2)

    # ── Synthetic FP2 descriptor (repeated power) ──
    prefix = 'mfp_fp2_repeated_synthetic'
    desc_fp2r = ModelDescriptor(
        model_type='mfp',
        independent_vars=['DWT', 'GT'],
        dependent_var='LOA',
        transformations={
            0: ('fp2', [0.5]),       # DWT → √DWT + √DWT·ln(DWT)  (repeated)
            1: ('linear', [1]),      # GT  → GT
        },
        coefficients={
            '√DWT': 2.5,
            '√DWT·ln(DWT)': -0.03,
            'GT': 0.0001,
        },
        intercept=30.0,
    )
    _record_descriptor_structure(snap, prefix, desc_fp2r)
    _record_all_formulas(snap, prefix, desc_fp2r)

    # ── Synthetic descriptor with one variable excluded ──
    prefix = 'mfp_with_excluded_synthetic'
    desc_excl = ModelDescriptor(
        model_type='mfp',
        independent_vars=['DWT', 'GT', 'Speed'],
        dependent_var='LOA',
        transformations={
            0: ('fp1', [0.5]),
            1: ('excluded', []),
            2: ('linear', [1]),
        },
        coefficients={
            '√DWT': 4.1,
            'Speed': 2.3,
        },
        intercept=10.0,
    )
    _record_descriptor_structure(snap, prefix, desc_excl)
    _record_all_formulas(snap, prefix, desc_excl)


# ── Section 2: Linear family ────────────────────────────────────────────────

def _snap_linear(snap, model_type, X, y, var_names, target):
    prefix = f'linear_family.{model_type}'
    desc = _fit_and_describe(model_type, X, y, var_names, target)
    _record_descriptor_structure(snap, prefix, desc)
    _record_all_formulas(snap, prefix, desc)


# ── Section 3: Polynomial family ────────────────────────────────────────────

def _snap_polynomial(snap, model_type, X, y, var_names, target):
    prefix = f'polynomial_family.{model_type}'
    ok, _ = ModelFactory.is_model_applicable(model_type, X, y)
    if not ok:
        snap[f'{prefix}._skipped'] = 'not applicable'
        return
    desc = _fit_and_describe(model_type, X, y, var_names, target)
    _record_descriptor_structure(snap, prefix, desc)
    _record_all_formulas(snap, prefix, desc)
    # Extra: polynomial-degree stored correctly
    snap[f'{prefix}.polynomial_degree'] = desc.polynomial_degree


# ── Section 4: Univariate nonlinear ─────────────────────────────────────────

def _snap_univariate(snap, model_type, X, y, var_names, target):
    prefix = f'univariate.{model_type}'
    ok, _ = ModelFactory.is_model_applicable(model_type, X, y)
    if not ok:
        snap[f'{prefix}._skipped'] = 'not applicable'
        return
    desc = _fit_and_describe(model_type, X, y, var_names, target)
    _record_descriptor_structure(snap, prefix, desc)
    _record_all_formulas(snap, prefix, desc)
    # Check popt stored in metadata
    snap[f'{prefix}.has_popt_metadata'] = 'popt' in (desc.metadata or {})
    snap[f'{prefix}.coef_keys_sorted']  = sorted(desc.coefficients.keys()
                                                  if desc.coefficients else [])


# ── Section 5: Log-transform multivariate ───────────────────────────────────

def _snap_log_multivariate(snap, model_type, X, y, var_names, target):
    prefix = f'log_multivariate.{model_type}'
    ok, _ = ModelFactory.is_model_applicable(model_type, X, y)
    if not ok:
        snap[f'{prefix}._skipped'] = 'not applicable'
        return
    desc = _fit_and_describe(model_type, X, y, var_names, target)
    _record_descriptor_structure(snap, prefix, desc)
    _record_all_formulas(snap, prefix, desc)
    snap[f'{prefix}.has_log_intercept_metadata'] = (
        'log_intercept' in (desc.metadata or {})
    )


# ── Section 6: Serialisation round-trip ─────────────────────────────────────

def _snap_serialisation(snap, X, y, var_names, target):
    """Verify that to_json/from_json preserves structural properties."""
    desc = _fit_and_describe('mfp', X, y, var_names, target,
                             feature_names=var_names)
    formula_before = desc.get_formula_string(notation='math')
    coef_keys_before = sorted(desc.coefficients.keys()
                              if desc.coefficients else [])
    transforms_before = {str(k): v[0]
                         for k, v in (desc.transformations or {}).items()}

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'test_descriptor.json')
        desc.to_json(path)
        desc_loaded = ModelDescriptor.from_json(path)

    formula_after    = desc_loaded.get_formula_string(notation='math')
    coef_keys_after  = sorted(desc_loaded.coefficients.keys()
                               if desc_loaded.coefficients else [])
    transforms_after = {str(k): v[0]
                        for k, v in (desc_loaded.transformations or {}).items()}

    snap['serialisation.formula_preserved']    = (formula_before == formula_after)
    snap['serialisation.coef_keys_preserved']  = (coef_keys_before == coef_keys_after)
    snap['serialisation.transforms_preserved'] = (transforms_before == transforms_after)
    snap['serialisation.effective_df_preserved'] = (
        desc.get_effective_parameters() == desc_loaded.get_effective_parameters()
    )
    snap['serialisation.formula_before'] = formula_before
    snap['serialisation.formula_after']  = formula_after


# ── Section 7: refit() structural preservation ──────────────────────────────

def _snap_refit(snap, X, y, var_names, target):
    """
    Verify that refit() preserves the transformation structure and
    produces a formula with the same shape (same terms, different values).
    """
    n_alt = 30
    DWT_a = np.random.uniform(5_000, 120_000, n_alt)
    GT_a  = np.random.uniform(3_000,  80_000, n_alt)
    LOA_a = 60 + 5.0 * np.sqrt(DWT_a) - 0.6 * np.log(GT_a) + np.random.normal(0, 6, n_alt)
    X_alt = np.column_stack([DWT_a, GT_a])

    desc = _fit_and_describe('mfp', X, y, var_names, target,
                             feature_names=var_names)

    transforms_before = {str(k): v for k, v in (desc.transformations or {}).items()}
    coef_keys_before  = sorted(desc.coefficients.keys()
                               if desc.coefficients else [])
    formula_structure_before = desc.get_formula_string(include_coefficients=False)

    desc.refit(X_alt, LOA_a, force_structure=True)

    transforms_after = {str(k): v for k, v in (desc.transformations or {}).items()}
    coef_keys_after  = sorted(desc.coefficients.keys()
                               if desc.coefficients else [])
    formula_structure_after = desc.get_formula_string(include_coefficients=False)

    snap['refit.transforms_unchanged']       = (transforms_before == transforms_after)
    snap['refit.coef_keys_unchanged']        = (coef_keys_before  == coef_keys_after)
    snap['refit.formula_structure_unchanged']= (formula_structure_before ==
                                                formula_structure_after)
    snap['refit.formula_structure']          = formula_structure_before

    # Lasso refit
    desc_lasso = _fit_and_describe('polynomial-2-lasso', X, y, var_names, target)
    alpha_before    = desc_lasso.regularization_alpha
    active_before   = sorted(desc_lasso.active_features or [])

    desc_lasso.refit(X_alt, LOA_a, force_structure=True)
    alpha_after = desc_lasso.regularization_alpha

    snap['refit.lasso_alpha_unchanged'] = (alpha_before == alpha_after)
    snap['refit.lasso_alpha']           = alpha_before


# ── Section 8: Effective DF table ───────────────────────────────────────────

def _snap_effective_df(snap, X, y, var_names):
    """Record naive vs actual parameter counts for all model types."""
    model_types = [
        'linear', 'ridge', 'lasso',
        'polynomial-2', 'polynomial-2-lasso', 'polynomial-2-ridge',
        'mfp',
    ]
    for mt in model_types:
        ok, _ = ModelFactory.is_model_applicable(mt, X, y)
        if not ok:
            snap[f'effective_df.{mt}.naive']  = None
            snap[f'effective_df.{mt}.actual'] = None
            continue
        kwargs = {'feature_names': var_names} if mt == 'mfp' else {}
        model, prep = ModelFactory.create_and_fit(mt, X, y, **kwargs)
        naive  = ModelFactory.get_n_parameters(mt, X)
        actual = ModelFactory.get_n_parameters(mt, X, fitted_model=model)
        snap[f'effective_df.{mt}.naive']  = naive
        snap[f'effective_df.{mt}.actual'] = actual
        snap[f'effective_df.{mt}.differ'] = (naive != actual)


# ── Section 9: Formatting utilities ─────────────────────────────────────────

def _snap_formatting_utilities(snap):
    """
    Capture the output of the pure formatting methods directly.
    These are the methods that will move to ParametricFormulaRenderer,
    so they need the most careful characterisation.
    """
    # We need a descriptor instance to call these methods.
    # Use a minimal synthetic one.
    desc = ModelDescriptor(
        model_type='mfp',
        independent_vars=['DWT'],
        dependent_var='LOA',
        transformations={0: ('fp1', [0.5])},
        coefficients={'√DWT': 4.2},
        intercept=50.0,
    )

    # _format_coefficient — test all branches
    cases = {
        'normal_positive':     1.234,
        'normal_negative':    -1.234,
        'small_positive':      0.000123,
        'small_negative':     -0.000456,
        'large_positive':      45_000.0,
        'large_negative':     -45_000.0,
        'zero':                0.0,
        'boundary_low':        1e-3,      # exactly at threshold — not sci
        'boundary_high':       1e4,       # exactly at threshold — not sci
        'below_boundary_low':  9.99e-4,   # just below — sci
        'above_boundary_high': 1.001e4,   # just above — sci
    }
    for label, val in cases.items():
        for notation in NOTATIONS:
            snap[f'format_coefficient.{label}.{notation}'] = (
                desc._format_coefficient(val, notation)
            )

    # _format_power — test all special cases and fraction detection
    power_cases = {
        'zero':         (0,      'DWT'),
        'half':         (0.5,    'DWT'),
        'neg_half':     (-0.5,   'DWT'),
        'neg_one':      (-1,     'DWT'),
        'one':          (1,      'DWT'),
        'two':          (2,      'DWT'),
        'neg_two':      (-2,     'DWT'),
        'third':        (1/3,    'DWT'),
        'two_thirds':   (2/3,    'DWT'),
        'neg_third':    (-1/3,   'DWT'),
        'three_halves': (1.5,    'DWT'),
        'decimal':      (0.789,  'DWT'),
    }
    for label, (power, var) in power_cases.items():
        for notation in NOTATIONS:
            snap[f'format_power.{label}.{notation}'] = (
                desc._format_power(var, power, notation)
            )

    # _format_power_name — must match keys used in coefficients dict
    power_name_cases = {
        'zero':      (0,    'DWT'),
        'half':      (0.5,  'DWT'),
        'neg_half':  (-0.5, 'DWT'),
        'neg_one':   (-1,   'GT'),
        'one':       (1,    'DWT'),
        'two':       (2,    'DWT'),
        'third':     (1/3,  'DWT'),
        'decimal':   (0.789,'DWT'),
    }
    for label, (power, var) in power_name_cases.items():
        snap[f'format_power_name.{label}'] = desc._format_power_name(var, power)

    # _simplify_power — fraction detection
    simplify_cases = {
        'half':         0.5,
        'neg_half':    -0.5,
        'third':        1/3,
        'two_thirds':   2/3,
        'neg_third':   -1/3,
        'three_halves': 1.5,
        'quarter':      0.25,
        'three_quarters': 0.75,
        'decimal_nonfrac': 0.789,
        'integer_2':    2.0,
        'integer_neg3': -3.0,
        'tiny':         0.0001,
    }
    for label, power in simplify_cases.items():
        snap[f'simplify_power.{label}'] = desc._simplify_power(power)

    # _get_fp2_second_term_name
    fp2_cases = {
        'power_zero':    ('DWT', 0),
        'power_half':    ('DWT', 0.5),
        'power_neg_one': ('GT',  -1),
        'power_two':     ('DWT', 2),
    }
    for label, (var, power) in fp2_cases.items():
        snap[f'fp2_second_term_name.{label}'] = desc._get_fp2_second_term_name(var, power)


# ============================================================================
# Entry point
# ============================================================================

def _determine_output_path() -> str:
    """Choose BEFORE or AFTER filename based on whether BEFORE already exists."""
    before = Path('descriptor_characterisation_BEFORE.json')
    after  = Path('descriptor_characterisation_AFTER.json')
    if not before.exists():
        return str(before)
    return str(after)


def _diff_snapshots(before_path: str, after_path: str):
    """Print a human-readable diff between two snapshots."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    all_keys = sorted(set(before) | set(after))
    diffs    = []
    missing  = []

    for k in all_keys:
        if k not in before:
            missing.append(f'  NEW KEY   : {k} = {after[k]}')
        elif k not in after:
            missing.append(f'  LOST KEY  : {k} (was: {before[k]})')
        elif before[k] != after[k]:
            diffs.append((k, before[k], after[k]))

    print(f"\n{'=' * 70}")
    print(f"  DIFF: {before_path}  →  {after_path}")
    print(f"{'=' * 70}")
    print(f"  Total keys  : {len(all_keys)}")
    print(f"  Differences : {len(diffs)}")
    print(f"  Added/removed keys: {len(missing)}")

    if missing:
        print("\n  Key changes:")
        for m in missing:
            print(m)

    if diffs:
        print(f"\n  Value differences:")
        for k, b, a in diffs:
            print(f"\n  KEY   : {k}")
            print(f"  BEFORE: {b!r}")
            print(f"  AFTER : {a!r}")
    else:
        print("\n  ✓ No value differences — refactor is behaviour-preserving.")


def _promote_after_to_before():
    """
    Replace BEFORE with AFTER, then delete AFTER.

    Use this when all differences between BEFORE and AFTER have been
    reviewed and confirmed as intentional improvements.  The AFTER
    snapshot becomes the new baseline for future refactors.
    """
    before = Path('descriptor_characterisation_BEFORE.json')
    after  = Path('descriptor_characterisation_AFTER.json')

    if not after.exists():
        print("No AFTER snapshot found — nothing to promote.")
        return

    if before.exists():
        before.unlink()

    after.rename(before)
    print(f"✅ AFTER promoted to BEFORE.  New baseline: {before}")


if __name__ == '__main__':
    import sys

    # ── Special modes ────────────────────────────────────────────────────────
    # python characterise_model_descriptor.py --promote
    #     Promotes the AFTER snapshot as the new BEFORE baseline.
    #     Use after reviewing all diffs and confirming they are correct.
    if '--promote' in sys.argv:
        _promote_after_to_before()
        sys.exit(0)

    # ── Normal mode: capture snapshot ────────────────────────────────────────
    output_path = _determine_output_path()
    print(f"Running characterisation suite...")

    snapshot = make_snapshot()

    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"  {len(snapshot)} observations captured.")
    print(f"  Saved to: {output_path}")

    # If both files now exist, auto-diff
    before = 'descriptor_characterisation_BEFORE.json'
    after  = 'descriptor_characterisation_AFTER.json'
    if Path(before).exists() and Path(after).exists():
        _diff_snapshots(before, after)
    else:
        print(f"\n  Run again after the refactor to produce the AFTER snapshot,")
        print(f"  then the diff will be printed automatically.")