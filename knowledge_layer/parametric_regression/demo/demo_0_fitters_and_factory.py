"""
Demo 01 — Fitters and ModelFactory
====================================

Demonstrates the parametric regression fitting layer in isolation.

Covers
------
1. ModelFactory applicability checking — understanding which models are
   valid for a given dataset before attempting any fits.
2. Fitting and predicting with each major model family (univariate
   nonlinear, multivariate linear, polynomial, MFP).
3. Correct parameter counting for sparse models (Lasso, MFP) and why it
   matters for AIC / BIC.
4. MFP transformation discovery — showing that the algorithm correctly
   identifies √x and ln(x) transformations from data.
5. MFP fixed-structure refitting — showing that the same transformation
   structure can be applied to a second dataset.

Running this script requires no external data files; all datasets are
generated synthetically.
"""

import numpy as np
from parametric_model_factory import ModelFactory

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
# 1. Applicability checking
# ============================================================================

section("1 — Applicability Checking")

print("""
ModelFactory.is_model_applicable() lets you audit which models are valid
for a dataset *before* attempting any fits.  Each fitter encodes its own
constraints: sample-size requirements, positivity constraints, and
single-predictor requirements.  Checking upfront prevents runtime errors
and makes the analysis strategy explicit.
""")

n = 60
X_multi   = np.random.uniform(1, 100, (n, 3))          # 3 predictors, all positive
X_single  = X_multi[:, [0]]                             # 1 predictor
y_pos     = 5 * np.sqrt(X_multi[:, 0]) + np.random.normal(0, 2, n)
y_general = y_pos - 10                                  # Contains negatives

print("--- Dataset: 3 predictors, positive X, positive y ---")
applicable = ModelFactory.get_applicable_models(X_multi, y_pos)
print(f"Applicable models ({len(applicable)}): {applicable}")

print("\n--- Dataset: 3 predictors, positive X, y contains negatives ---")
applicable_neg = ModelFactory.get_applicable_models(X_multi, y_general)
print(f"Applicable models ({len(applicable_neg)}): {applicable_neg}")

print("\n--- Why certain models are inapplicable (3-predictor, negative y) ---")
reasons = ModelFactory.get_inapplicable_models_with_reasons(X_multi, y_general)
for model, reason in reasons.items():
    print(f"  {model:35s}: {reason}")


# ============================================================================
# 2. Univariate nonlinear models
# ============================================================================

section("2 — Univariate Nonlinear Models")

print("""
UnivariateNonlinearFitter fits parametric curves via scipy.optimize.curve_fit
(Levenberg-Marquardt).  Each model is a (func_name, popt) tuple — the function
is reconstructed from the name and parameter array at prediction time.
Requires exactly one predictor.
""")

x = np.random.uniform(1, 100, 120).reshape(-1, 1)
y_power = 3.5 * x.flatten() ** 0.7 + np.random.normal(0, 5, 120)
y_log   = 20 * np.log(x.flatten()) - 15 + np.random.normal(0, 3, 120)

for model_type, y_target, label in [
    ('power',       y_power, 'y = 3.5·x^0.7  (true exponent ≈ 0.7)'),
    ('logarithmic', y_log,   'y = 20·ln(x) − 15'),
]:
    model, prep = ModelFactory.create_and_fit(model_type, x, y_target)
    y_pred = ModelFactory.predict(model, prep, x, model_type)
    n_params = ModelFactory.get_n_parameters(model_type, x)

    func_name, popt = model
    print(f"  {model_type:20s} | {label}")
    print(f"    popt   : {popt.round(4)}")
    print(f"    R²     : {r2(y_target, y_pred):.4f}")
    print(f"    n_params: {n_params}")
    print()


# ============================================================================
# 3. Multivariate linear and regularised models
# ============================================================================

section("3 — Multivariate Linear and Regularised Models")

print("""
MultivariateLinearFitter handles plain linear regression, Ridge (L2) and
Lasso (L1) regularisation, and two log-linearisable models.

Key point on parameter counting:
  - Ridge keeps all features but shrinks them → n_features + 1 params
  - Lasso drives some coefficients to exactly zero → sparse, fewer params
  - For AIC/BIC, the Lasso count should be the number of NON-ZERO coefs.
  - Providing fitted_model to get_n_parameters() returns the sparse count.
""")

n = 100
X3 = np.random.uniform(1, 10, (n, 3))
y_lin = 2 * X3[:, 0] + 0.1 * X3[:, 1] + 5  # X2 barely relevant, X3 irrelevant

print("  model          | coefs kept | n_params (sparse) | R²")
print("  " + "-" * 58)

for model_type in ['linear', 'ridge', 'lasso', 'lasso-cv']:
    model, prep = ModelFactory.create_and_fit(model_type, X3, y_lin)
    y_pred = ModelFactory.predict(model, prep, X3, model_type)

    n_params_default = ModelFactory.get_n_parameters(model_type, X3)
    n_params_actual  = ModelFactory.get_n_parameters(model_type, X3, fitted_model=model)

    if hasattr(model, 'coef_'):
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))
    else:
        n_nonzero = '—'

    print(f"  {model_type:15s}| {str(n_nonzero):10} | "
          f"{n_params_default:5} → {n_params_actual:5}     | {r2(y_lin, y_pred):.4f}")

print("""
Notice that for Lasso the default n_params (without fitted_model) returns the
maximum possible count, while the actual sparse count is smaller.  This matters
when comparing AIC / BIC values across model types.
""")

# Log-transform multivariate
print("--- Log-linearisable models ---")
X_pos = np.random.uniform(1, 10, (n, 2))
y_power_multi = 2.0 * X_pos[:, 0] ** 1.5 * X_pos[:, 1] ** 0.8

for model_type in ['power-log-multivariate', 'exponential-log-multivariate']:
    ok, reason = ModelFactory.is_model_applicable(model_type, X_pos, y_power_multi)
    if ok:
        model, prep = ModelFactory.create_and_fit(model_type, X_pos, y_power_multi)
        y_pred = ModelFactory.predict(model, prep, X_pos, model_type)
        print(f"  {model_type:35s}: R² = {r2(y_power_multi, y_pred):.4f}")
    else:
        print(f"  {model_type:35s}: not applicable — {reason}")


# ============================================================================
# 4. Polynomial models
# ============================================================================

section("4 — Polynomial Models")

print("""
PolynomialFitter uses sklearn PolynomialFeatures to expand the input matrix
to all monomials up to the specified degree.  The preprocessor is returned
alongside the model so that prediction can replicate the same expansion.

The combinatorial growth in feature count is the central constraint:
  3 predictors, degree 2 → 10 features
  3 predictors, degree 3 → 20 features
Without regularisation, more features than samples leads to overfitting.
""")

n_poly = 80
X_poly = np.random.uniform(0, 5, (n_poly, 2))
y_quad = 2 * X_poly[:, 0] ** 2 - 1.5 * X_poly[:, 0] * X_poly[:, 1] + 3

print("  model                  | n_features | R² (in-sample)")
print("  " + "-" * 52)

for model_type in ['polynomial-2', 'polynomial-2-lasso', 'polynomial-3']:
    ok, reason = ModelFactory.is_model_applicable(model_type, X_poly, y_quad)
    if not ok:
        print(f"  {model_type:25s}: not applicable — {reason}")
        continue

    model, prep = ModelFactory.create_and_fit(model_type, X_poly, y_quad)
    y_pred = ModelFactory.predict(model, prep, X_poly, model_type)
    n_f = ModelFactory.get_n_parameters(model_type, X_poly, fitted_model=model)

    print(f"  {model_type:25s}| {n_f:10} | {r2(y_quad, y_pred):.4f}")


# ============================================================================
# 5. MFP — transformation discovery
# ============================================================================

section("5 — MFP Transformation Discovery")

print("""
MFPFitter wraps MFPRegressor, which runs the Royston & Sauerbrei (2008)
algorithm: cyclic Backward Elimination → Function Selection until convergence.

For each retained variable the closed-test hierarchy selects:
  FP2 > FP1 > Linear > Excluded

The test below generates data from a known model and checks that MFP
recovers the correct transformations.
""")

n_mfp = 200
x1 = np.random.uniform(1, 50, n_mfp)
x2 = np.random.uniform(1, 20, n_mfp)
x3 = np.random.uniform(1, 10, n_mfp)   # Noise variable — should be excluded

y_mfp = (
    4.0 * np.sqrt(x1)       # FP1, power 0.5
    - 0.8 * np.log(x2)      # FP1, power 0 (ln)
    + np.random.normal(0, 1.5, n_mfp)
)

X_mfp = np.column_stack([x1, x2, x3])

print("True model: y = 4·√x1 − 0.8·ln(x2) + noise")
print("x3 is a noise variable and should be excluded by backward elimination.")
print()

model_mfp, _ = ModelFactory.create_and_fit(
    'mfp', X_mfp, y_mfp,
    verbose=True,
)

y_pred_mfp = ModelFactory.predict(model_mfp, None, X_mfp, 'mfp')

print("\n--- Discovered transformations ---")
for var, transform in model_mfp.get_selected_transformations().items():
    print(f"  {var}: {transform}")

print(f"\n--- Fitted coefficients ---")
print(f"  Intercept : {model_mfp.intercept_:.3f}  (true ≈ 0)")
for i, c in enumerate(model_mfp.coef_):
    print(f"  β{i + 1}        : {c:.3f}")

n_max    = ModelFactory.get_n_parameters('mfp', X_mfp)
n_actual = ModelFactory.get_n_parameters('mfp', X_mfp, fitted_model=model_mfp)

print(f"\n--- Parameter count ---")
print(f"  Worst-case (before fitting) : {n_max}   (all FP2 = 2 × n_features + 1)")
print(f"  Actual (after fitting)      : {n_actual}  (reflects exclusions & FP1 selections)")
print(f"\n--- Performance ---")
print(f"  R² = {r2(y_mfp, y_pred_mfp):.4f}")
print(f"  RMSE = {np.sqrt(np.mean((y_mfp - y_pred_mfp) ** 2)):.4f}")

print("""
Expected:
  ✓ x1 → √x1 (FP1, power 0.5)
  ✓ x2 → ln(x2) (FP1, power 0)
  ✓ x3 → EXCLUDED
""")


# ============================================================================
# 6. MFP — fixed-structure refitting (cross-dataset workflow)
# ============================================================================

section("6 — MFP Fixed-Structure Refitting")

print("""
The fixed-structure pathway is the key operation for cross-dataset
coefficient comparison.

Workflow:
  1. Discover transformations on dataset A (conventional ships).
  2. Refit with the SAME transformations on dataset B (alternative ships).
  3. Compare only the coefficients — any difference is a genuine
     population difference, not a structural artefact.

Without structure-fixing, MFP might choose different transformations on
dataset B, making coefficient comparison meaningless.
""")

# Dataset A — conventional ships (larger sample, used for discovery)
n_conv = 150
x1_conv = np.random.uniform(1, 50, n_conv)
x2_conv = np.random.uniform(1, 20, n_conv)
y_conv = (
    4.0 * np.sqrt(x1_conv)
    - 0.8 * np.log(x2_conv)
    + 3.0                    # Intercept
    + np.random.normal(0, 1.5, n_conv)
)
X_conv = np.column_stack([x1_conv, x2_conv])

# Dataset B — alternative ships (smaller sample, different coefficients)
n_alt = 25
x1_alt = np.random.uniform(1, 50, n_alt)
x2_alt = np.random.uniform(1, 20, n_alt)
y_alt = (
    5.5 * np.sqrt(x1_alt)   # Larger coefficient — detectable shift
    - 0.6 * np.log(x2_alt)
    + 5.0                    # Different intercept
    + np.random.normal(0, 2.0, n_alt)
)
X_alt = np.column_stack([x1_alt, x2_alt])

# Step 1: discover on conventional
model_conv, _ = ModelFactory.create_and_fit('mfp', X_conv, y_conv, verbose=False)
transforms_discovered = model_conv.transforms_

print("Transformations discovered on conventional data:")
for var, t_str in model_conv.get_selected_transformations().items():
    print(f"  {var}: {t_str}")

# Step 2: refit on alternative with SAME structure
model_alt, _ = ModelFactory.create_and_fit(
    'mfp', X_alt, y_alt,
    fixed_transforms=transforms_discovered,
    verbose=False,
)

# Step 3: compare coefficients
print("\nCoefficient comparison (same transformation structure):")
print(f"  {'Term':15s}  {'Conventional':>14}  {'Alternative':>14}")
print("  " + "-" * 46)
print(f"  {'Intercept':15s}  {model_conv.intercept_:14.4f}  {model_alt.intercept_:14.4f}")

names = model_conv.get_selected_transformations()
for i, (var, t_str) in enumerate(names.items()):
    if i < len(model_conv.coef_) and i < len(model_alt.coef_):
        print(f"  {t_str:15s}  {model_conv.coef_[i]:14.4f}  {model_alt.coef_[i]:14.4f}")

# Performance on each dataset
y_pred_conv = ModelFactory.predict(model_conv, None, X_conv, 'mfp')
y_pred_alt  = ModelFactory.predict(model_alt,  None, X_alt,  'mfp')
print(f"\nPerformance:")
print(f"  Conventional  R² = {r2(y_conv, y_pred_conv):.4f}")
print(f"  Alternative   R² = {r2(y_alt,  y_pred_alt):.4f}")

print("""
Notice that the alternative dataset uses the same transformation structure
(√x1 and ln(x2)) but with different coefficients, reflecting the genuine
population difference between fuel types.
""")