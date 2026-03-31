"""
Demo 02 — ModelDescriptor
==========================

Demonstrates ModelDescriptor as the persistent bridge between the fitting
layer (ModelFactory) and the analysis layer (cross-dataset comparisons).

Covers
------
1. Creating a descriptor from a fitted model via from_fitted_model().
2. Formula reconstruction in math, LaTeX, and Python notation.
3. Effective degrees of freedom — how Lasso and MFP report sparse counts.
4. Structure-preserving refitting via refit() for:
     - MFP  (fixed_transforms pathway)
     - Lasso (fixed alpha pathway)
5. Serialisation round-trip via to_json() / from_json().
6. End-to-end cross-dataset coefficient comparison workflow — the primary
   research use case.

Running this script requires no external data files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from parametric_model_factory import ModelFactory
from parametric_model_description_class import ModelDescriptor

np.random.seed(42)


# ============================================================================
# Helpers
# ============================================================================

def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


# ============================================================================
# Shared synthetic datasets
# ============================================================================

# --- Conventional ships (used as training data for discovery) ---
n_conv = 150
DWT_c  = np.random.uniform(5_000, 100_000, n_conv)
GT_c   = np.random.uniform(3_000,  80_000, n_conv)
LOA_c  = (
    50
    + 4.2 * np.sqrt(DWT_c)
    - 0.8 * np.log(GT_c)
    + np.random.normal(0, 5, n_conv)
)
X_conv = np.column_stack([DWT_c, GT_c])
y_conv = LOA_c
PRED_NAMES = ['DWT', 'GT']
TARGET     = 'LOA'

# --- Alternative-fuel ships (smaller, different coefficients) ---
n_alt  = 20
DWT_a  = np.random.uniform(5_000, 100_000, n_alt)
GT_a   = np.random.uniform(3_000,  80_000, n_alt)
LOA_a  = (
    60                         # Higher intercept
    + 5.0 * np.sqrt(DWT_a)    # Stronger DWT effect
    - 0.6 * np.log(GT_a)
    + np.random.normal(0, 6, n_alt)
)
X_alt = np.column_stack([DWT_a, GT_a])
y_alt = LOA_a


# ============================================================================
# 1. Creating a descriptor from a fitted model
# ============================================================================

section("1 — Creating a Descriptor from a Fitted Model")

print("""
ModelDescriptor.from_fitted_model() inspects the model object and
preprocessor and extracts all structural information: transformations,
active features, regularisation parameters, and coefficients.

It handles every model type differently:
  - MFP       → copies transforms_ and maps coef_ to descriptive names
  - Lasso     → records active features and regularization_alpha
  - Nonlinear → unpacks popt into {'a': ..., 'b': ..., 'c': ...}
  - Linear    → zips coef_ with variable names directly
""")

# Fit an MFP model
model_mfp, prep_mfp = ModelFactory.create_and_fit(
    'mfp', X_conv, y_conv, verbose=False,
    feature_names=PRED_NAMES,
)
y_pred_mfp = ModelFactory.predict(model_mfp, prep_mfp, X_conv, 'mfp')

desc_mfp = ModelDescriptor.from_fitted_model(
    model=model_mfp,
    preprocessor=prep_mfp,
    model_type='mfp',
    independent_vars=PRED_NAMES,
    dependent_var=TARGET,
    metrics={'r2': r2(y_conv, y_pred_mfp)},
)

print("--- MFP Descriptor ---")
print(desc_mfp)
print(f"\nTransformations: {desc_mfp.transformations}")
print(f"Coefficients   : {desc_mfp.coefficients}")
print(f"Intercept      : {desc_mfp.intercept:.4f}")
print(f"Effective DF   : {desc_mfp.get_effective_parameters()}")

# Fit a polynomial-2-lasso model
model_lasso, prep_lasso = ModelFactory.create_and_fit('polynomial-2-lasso', X_conv, y_conv)
y_pred_lasso = ModelFactory.predict(model_lasso, prep_lasso, X_conv, 'polynomial-2-lasso')

desc_lasso = ModelDescriptor.from_fitted_model(
    model=model_lasso,
    preprocessor=prep_lasso,
    model_type='polynomial-2-lasso',
    independent_vars=PRED_NAMES,
    dependent_var=TARGET,
    metrics={'r2': r2(y_conv, y_pred_lasso)},
)

print("\n--- Polynomial-2-Lasso Descriptor ---")
print(desc_lasso)
print(f"\nActive features ({len(desc_lasso.active_features or [])}):"
      f" {desc_lasso.active_features}")
print(f"Regularisation alpha : {desc_lasso.regularization_alpha}")


# ============================================================================
# 2. Formula reconstruction
# ============================================================================

section("2 — Formula Reconstruction")

print("""
get_formula_string() reconstructs the model equation from the stored
coefficients and transformations.  Three notation styles are supported:
  'math'   — Unicode mathematical notation (for console / reports)
  'latex'  — LaTeX source (for thesis / paper)
  'python' — valid Python expression (for code generation)

For MFP, the formula reflects the actual discovered transformations.
For Lasso, zero-coefficient terms are suppressed by default (only_active=True).
""")

for model_type, desc, label in [
    ('mfp',               desc_mfp,   'MFP'),
    ('polynomial-2-lasso', desc_lasso, 'Poly-2-Lasso'),
]:
    print(f"\n--- {label} ---")
    for notation in ['math', 'latex', 'python']:
        formula = desc.get_formula_string(
            include_coefficients=True,
            notation=notation,
        )
        print(f"  [{notation:6s}]  {formula}")

    print(f"  [β-symbols]  "
          f"{desc.get_formula_string(include_coefficients=False, notation='math')}")

# Fit and display a univariate nonlinear model
X_1d = np.random.uniform(1, 50, (80, 1))
y_1d = 3.5 * X_1d.flatten() ** 0.65 + np.random.normal(0, 3, 80)

model_pow, prep_pow = ModelFactory.create_and_fit('power', X_1d, y_1d)
desc_pow = ModelDescriptor.from_fitted_model(
    model_pow, prep_pow, 'power', ['DWT'], 'MCR'
)

print("\n--- Power model ---")
for notation in ['math', 'latex', 'python']:
    print(f"  [{notation:6s}]  {desc_pow.get_formula_string(notation=notation)}")


# ============================================================================
# 3. Effective degrees of freedom
# ============================================================================

section("3 — Effective Degrees of Freedom")

print("""
Effective DF drives AIC / BIC penalisation.  Using the wrong count biases
model comparison:
  - Ridge   : all features present (shrunk but non-zero)  → n_features + 1
  - Lasso   : only non-zero features count                 → sparse DF
  - MFP     : depends on which transformations were chosen → actual terms
  - Polynomial : all expansion terms                       → comb(n+d, d)

The table below shows the difference between naive and actual counts.
""")

X_df = np.random.uniform(1, 10, (100, 3))
y_df = 2 * X_df[:, 0] + 0.1 * X_df[:, 1] + 5

print(f"  {'Model':25s}  {'naive DF':>10}  {'actual DF':>10}")
print("  " + "-" * 48)

for mt in ['linear', 'ridge', 'lasso', 'polynomial-2', 'polynomial-2-lasso', 'mfp']:
    ok, _ = ModelFactory.is_model_applicable(mt, X_df, y_df)
    if not ok:
        continue
    # Pass feature_names for MFP so fitted_model has named transforms,
    # which is required for get_n_parameters to count correctly.
    kwargs = {'feature_names': ['x1', 'x2', 'x3']} if mt == 'mfp' else {}
    model, prep = ModelFactory.create_and_fit(mt, X_df, y_df, **kwargs)
    naive  = ModelFactory.get_n_parameters(mt, X_df)
    actual = ModelFactory.get_n_parameters(mt, X_df, fitted_model=model)
    print(f"  {mt:25s}  {naive:10}  {actual:10}")


# ============================================================================
# 4. Structure-preserving refit — MFP
# ============================================================================

section("4a — Structure-Preserving Refit: MFP")

print("""
descriptor.refit(X_new, y_new, force_structure=True) re-estimates
coefficients on a new dataset while holding the transformation structure fixed.

For MFP this means passing fixed_transforms to MFPRegressor, bypassing
the backward-elimination and function-selection algorithm entirely.

Why this matters: if MFP is free to discover a different structure on the
alternative-fuel dataset, any coefficient difference is ambiguous — it could
reflect a genuine population shift OR simply a different structural choice.
Fixing the structure isolates the coefficient difference.
""")

# Refit MFP on alternative-fuel data using conventional structure
model_mfp_alt, prep_mfp_alt = desc_mfp.refit(X_alt, y_alt, force_structure=True)
y_pred_alt = ModelFactory.predict(model_mfp_alt, prep_mfp_alt, X_alt, 'mfp')

print("Conventional transformations applied to alternative-fuel data:")
print(f"  Stored transforms: {desc_mfp.transformations}")
print(f"\nCoefficients after refit on alternative data:")
print(f"  Intercept : {desc_mfp.intercept:.4f}  (updated in-place by refit)")
for name, val in (desc_mfp.coefficients or {}).items():
    print(f"  {name:15s}: {val:.4f}")
print(f"\nAlternative R² = {r2(y_alt, y_pred_alt):.4f}")

# Show that the original conventional coefficients are gone (descriptor mutated)
print("""
Note: refit() mutates the descriptor's coefficients in-place.  If you need
to preserve the original fit, create a copy before calling refit().
""")


# ============================================================================
# 4b. Structure-preserving refit — Lasso
# ============================================================================

section("4b — Structure-Preserving Refit: Lasso")

print("""
For Lasso, force_structure=True fixes regularization_alpha at the value
stored during the original fit.  This applies the same sparsity pressure
to the new dataset; different features may still be selected, but the
comparison is on equal footing.

If force_structure=False (default for non-MFP), ModelFactory.create_and_fit
is called fresh, potentially choosing a different alpha.
""")

alpha_used = desc_lasso.regularization_alpha
print(f"Stored Lasso alpha: {alpha_used}")

model_lasso_alt, prep_lasso_alt = desc_lasso.refit(X_alt, y_alt, force_structure=True)
y_pred_lasso_alt = ModelFactory.predict(
    model_lasso_alt, prep_lasso_alt, X_alt, 'polynomial-2-lasso'
)

print(f"Active features after refit: {desc_lasso.active_features}")
print(f"Alternative R² = {r2(y_alt, y_pred_lasso_alt):.4f}")


# ============================================================================
# 5. Serialisation round-trip
# ============================================================================

section("5 — Serialisation Round-Trip (JSON)")

print("""
to_json() / from_json() persist the full descriptor — including
transformations, coefficients, metrics, and metadata — as a JSON file.

This enables:
  - Saving models between sessions without re-fitting.
  - Sharing model structures across team members.
  - Auditing exactly which transformations were applied.

numpy arrays (active_feature_mask) are converted to lists and restored
automatically.
""")

# Re-fit the MFP so the descriptor holds the conventional coefficients again
model_mfp_fresh, _ = ModelFactory.create_and_fit('mfp', X_conv, y_conv, verbose=False)
y_pred_fresh = ModelFactory.predict(model_mfp_fresh, None, X_conv, 'mfp')
desc_to_save = ModelDescriptor.from_fitted_model(
    model_mfp_fresh, None, 'mfp', PRED_NAMES, TARGET,
    metrics={'r2': r2(y_conv, y_pred_fresh)},
)

with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / 'mfp_descriptor.json')
    desc_to_save.to_json(path)

    # Inspect raw JSON
    with open(path) as fh:
        raw = json.load(fh)
    print("Raw JSON keys:", list(raw.keys()))
    print("Stored metrics:", raw.get('metrics'))
    print("Stored transformations:", raw.get('transformations'))

    # Restore
    desc_loaded = ModelDescriptor.from_json(path)

print("\nDescriptor after round-trip:")
print(desc_loaded)
print(f"\nFormula preserved: {desc_loaded.get_formula_string()}")


# ============================================================================
# 6. End-to-end cross-dataset coefficient comparison
# ============================================================================

section("6 — End-to-End Cross-Dataset Coefficient Comparison")

print("""
This section assembles the full research workflow:

  Step 1 — Fit on conventional ships, capture descriptor.
  Step 2 — Refit with same structure on alternative-fuel ships.
  Step 3 — Print coefficient table for both datasets.
  Step 4 — Report performance on each dataset.

The coefficient differences are interpretable as genuine population shifts
because the transformation structure is held constant.
""")

# Step 1
model_step1, _ = ModelFactory.create_and_fit(
    'mfp', X_conv, y_conv, verbose=False,
    feature_names=PRED_NAMES,
)
y_pred_step1 = ModelFactory.predict(model_step1, None, X_conv, 'mfp')
desc_step1 = ModelDescriptor.from_fitted_model(
    model_step1, None, 'mfp', PRED_NAMES, TARGET,
    metrics={'r2': r2(y_conv, y_pred_step1)},
)

# Capture conventional coefficients before refit mutates desc
conv_intercept = desc_step1.intercept
conv_coefs = dict(desc_step1.coefficients or {})

print("Step 1: Conventional fit complete.")
print(f"  Formula: {desc_step1.get_formula_string()}")
print(f"  R²     : {desc_step1.metrics.get('r2', 'n/a'):.4f}")

# Step 2
model_step2, _ = desc_step1.refit(X_alt, y_alt, force_structure=True)
y_pred_step2 = ModelFactory.predict(model_step2, None, X_alt, 'mfp')

alt_intercept = desc_step1.intercept         # updated by refit
alt_coefs = dict(desc_step1.coefficients or {})

print("\nStep 2: Alternative-fuel refit complete.")
print(f"  Structure applied: {desc_step1.transformations}")
print(f"  R²               : {r2(y_alt, y_pred_step2):.4f}")

# Step 3 — comparison table
print("\nStep 3: Coefficient comparison")
print(f"\n  {'Term':20s}  {'Conventional':>14}  {'Alternative':>14}  {'Δ':>10}")
print("  " + "-" * 62)

all_terms = ['Intercept'] + list(conv_coefs.keys())
for term in all_terms:
    c_val = conv_intercept if term == 'Intercept' else conv_coefs.get(term)
    a_val = alt_intercept  if term == 'Intercept' else alt_coefs.get(term)
    if c_val is None or a_val is None:
        continue
    delta = a_val - c_val
    print(f"  {term:20s}  {c_val:14.4f}  {a_val:14.4f}  {delta:+10.4f}")

# Step 4 — performance summary
print(f"\nStep 4: Performance summary")
print(f"  Conventional n={n_conv:4d}  R² = {r2(y_conv, y_pred_step1):.4f}")
print(f"  Alternative  n={n_alt:4d}   R² = {r2(y_alt,  y_pred_step2):.4f}")

print("""
Interpretation:
  The positive Δ on the intercept and √DWT coefficient suggests that
  alternative-fuel ships sit on a higher LOA–DWT curve than conventional
  ships — consistent with the extra volume required for alternative fuel
  storage.  The ln(GT) coefficient shift is smaller, suggesting GT–LOA
  geometry is more conserved across fuel types.

  (With real data these differences would be tested formally via bootstrap
  confidence intervals on the coefficient differences.)
""")