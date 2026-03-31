"""
Demo 03 — Model Visualisation
==============================

Plots fitted parametric models alongside their derived formulas.
Each figure is laid out as:
  - Left panel  : scatter of data with fitted curve overlaid
  - Right panel : residuals vs fitted values

Figures produced
----------------
1. Univariate nonlinear gallery — one subplot per model family fitted to the
   same DWT → LOA dataset, with formulas in the title.
2. MFP fit with variable annotations — shows which transformation was applied
   to each predictor and the resulting formula.
3. Cross-dataset coefficient comparison — two overlaid curves (conventional
   and alternative) showing the same structural form with different
   coefficients, annotated with the formula and Δcoefficients.
4. Effective-DF comparison bar chart — naive vs actual parameter count for
   each model type, to illustrate the AIC/BIC implication of sparse models.

Running this script requires no external data files.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from parametric_model_factory import ModelFactory
from parametric_model_description_class import ModelDescriptor

np.random.seed(42)

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'figure.dpi':       120,
})

CONV_COLOR = '#2C7BB6'   # Blue  — conventional
ALT_COLOR  = '#D7191C'   # Red   — alternative
FIT_COLOR  = '#1a1a2e'   # Near-black — fitted curve
RES_COLOR  = '#555555'


# ============================================================================
# Synthetic ship-like data
# ============================================================================

n_conv = 200
DWT_c  = np.random.uniform(5_000, 150_000, n_conv)
# True: LOA = 50 + 4.5·√DWT  (power-law relationship common in NA formulae)
LOA_c  = 50 + 4.5 * np.sqrt(DWT_c) + np.random.normal(0, 8, n_conv)
# Ensure positive (required for some models)
LOA_c  = np.clip(LOA_c, 1, None)

n_alt = 25
DWT_a  = np.random.uniform(5_000, 150_000, n_alt)
LOA_a  = 62 + 5.2 * np.sqrt(DWT_a) + np.random.normal(0, 10, n_alt)
LOA_a  = np.clip(LOA_a, 1, None)

X_conv_1d = DWT_c.reshape(-1, 1)
X_alt_1d  = DWT_a.reshape(-1, 1)

# Multi-predictor data for MFP section
GT_c   = np.random.uniform(3_000, 100_000, n_conv)
LOA_mp = 40 + 4.0 * np.sqrt(DWT_c) - 0.7 * np.log(GT_c) + np.random.normal(0, 6, n_conv)
X_mp   = np.column_stack([DWT_c, GT_c])


# ============================================================================
# Helpers
# ============================================================================

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smooth_x(x_data, n_pts=300):
    """Return a fine grid spanning the data range for smooth curve plotting."""
    return np.linspace(x_data.min(), x_data.max(), n_pts).reshape(-1, 1)


def annotate_formula(ax, formula: str, r2_val: float,
                     x_frac=0.04, y_frac=0.95, fontsize=9):
    """Add formula and R² as a text box in axes-fraction coordinates."""
    text = f"{formula}\n$R^2 = {r2_val:.4f}$"
    ax.text(
        x_frac, y_frac, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9),
        family='monospace',
    )


def add_residual_panel(ax, y_true, y_pred, color=RES_COLOR):
    """Plot residuals vs fitted values on a supplied axes."""
    residuals = y_true - y_pred
    ax.axhline(0, color='black', lw=1.0, zorder=3)
    ax.scatter(y_pred, residuals, s=12, alpha=0.5, color=color, zorder=2)
    ax.set_xlabel('Fitted values (m)', fontsize=9)
    ax.set_ylabel('Residual (m)',       fontsize=9)
    ax.set_title('Residuals vs Fitted', fontsize=10)


# ============================================================================
# Figure 1 — Univariate nonlinear model gallery
# ============================================================================

univariate_models = [
    ('power',            'Power:  $y = a \\cdot DWT^b$'),
    ('logarithmic',      'Logarithmic:  $y = a \\ln(DWT) + b$'),
    ('exponential',      'Exponential:  $y = a \\cdot e^{b \\cdot DWT}$'),
    ('power-offset',     'Power + offset:  $y = a \\cdot DWT^b + c$'),
]

# Filter to models applicable to the positive DWT/LOA data
applicable = ModelFactory.get_applicable_models(X_conv_1d, LOA_c)
univariate_models = [(m, l) for m, l in univariate_models if m in applicable]

n_models = len(univariate_models)
fig1, axes1 = plt.subplots(
    n_models, 2,
    figsize=(12, 3.2 * n_models),
    squeeze=False,
)
fig1.suptitle(
    'Figure 1 — Univariate Nonlinear Models: DWT → LOA',
    fontsize=13, fontweight='bold', y=1.01,
)

x_smooth = smooth_x(DWT_c)

for row, (model_type, label) in enumerate(univariate_models):
    ax_fit = axes1[row, 0]
    ax_res = axes1[row, 1]

    print(f"[Fig 1] Fitting {model_type} ...", end=' ')
    with warnings.catch_warnings():
        # Exponential is a deliberately poor structural choice for √DWT data;
        # the optimiser may overflow before converging. Suppress the warning
        # — the fit still succeeds and the poor R² is itself informative.
        warnings.simplefilter('ignore', RuntimeWarning)
        model, prep = ModelFactory.create_and_fit(model_type, X_conv_1d, LOA_c)
    y_pred       = ModelFactory.predict(model, prep, X_conv_1d, model_type)
    y_smooth     = ModelFactory.predict(model, prep, x_smooth,  model_type)

    desc = ModelDescriptor.from_fitted_model(
        model, prep, model_type, ['DWT'], 'LOA',
        metrics={'r2': r2_score(LOA_c, y_pred)},
    )
    formula_math = desc.get_formula_string(notation='math')
    r2_val       = r2_score(LOA_c, y_pred)
    print(f"R²={r2_val:.4f}")

    # Fit panel
    ax_fit.scatter(DWT_c, LOA_c, s=8, alpha=0.3, color=CONV_COLOR, zorder=2,
                   label='Data')
    ax_fit.plot(x_smooth.flatten(), y_smooth, lw=2, color=FIT_COLOR, zorder=3,
                label='Fit')
    ax_fit.set_xlabel('DWT (t)',  fontsize=9)
    ax_fit.set_ylabel('LOA (m)', fontsize=9)
    ax_fit.set_title(label,      fontsize=10)
    ax_fit.legend(fontsize=8, framealpha=0.7)
    annotate_formula(ax_fit, formula_math, r2_val)

    # Residual panel
    add_residual_panel(ax_res, LOA_c, y_pred)

fig1.tight_layout()
fig1.savefig('demo_fig1_univariate_gallery.png', bbox_inches='tight', dpi=150)
print("Saved: demo_fig1_univariate_gallery.png\n")
plt.show()


# ============================================================================
# Figure 2 — MFP fit with transformation annotations
# ============================================================================

print("[Fig 2] Fitting MFP (DWT + GT → LOA) ...")

model_mfp, _ = ModelFactory.create_and_fit(
    'mfp', X_mp, LOA_mp,
    verbose=False,
    feature_names=['DWT', 'GT'],
)
y_pred_mfp = ModelFactory.predict(model_mfp, None, X_mp, 'mfp')
desc_mfp   = ModelDescriptor.from_fitted_model(
    model_mfp, None, 'mfp', ['DWT', 'GT'], 'LOA',
    metrics={'r2': r2_score(LOA_mp, y_pred_mfp)},
)

formula_math  = desc_mfp.get_formula_string(notation='math')
formula_latex = desc_mfp.get_formula_string(notation='latex')
r2_mfp        = r2_score(LOA_mp, y_pred_mfp)
print(f"  Formula (math) : {formula_math}")
print(f"  Formula (LaTeX): {formula_latex}")
print(f"  R² = {r2_mfp:.4f}")

transforms = model_mfp.get_selected_transformations()

fig2 = plt.figure(figsize=(14, 5))
gs   = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

ax_dwt = fig2.add_subplot(gs[0])
ax_gt  = fig2.add_subplot(gs[1])
ax_res = fig2.add_subplot(gs[2])

fig2.suptitle(
    'Figure 2 — MFP: DWT + GT → LOA  |  Discovered transformations',
    fontsize=12, fontweight='bold',
)

# Panel 1: LOA vs DWT (other variable at median, or only predictor if GT eliminated)
dwt_grid = smooth_x(DWT_c)
gt_med   = np.full_like(dwt_grid, np.median(GT_c))
X_dwt_grid = np.column_stack([dwt_grid.flatten(), gt_med.flatten()])
y_dwt_curve = ModelFactory.predict(model_mfp, None, X_dwt_grid, 'mfp')

ax_dwt.scatter(DWT_c, LOA_mp, s=8, alpha=0.25, color=CONV_COLOR)
ax_dwt.plot(dwt_grid.flatten(), y_dwt_curve, lw=2.5, color=FIT_COLOR)
ax_dwt.set_xlabel('DWT (t)', fontsize=9)
ax_dwt.set_ylabel('LOA (m)', fontsize=9)
dwt_transform = transforms.get('DWT', 'DWT (excluded)')
ax_dwt.set_title(f'DWT → {dwt_transform}', fontsize=10)
annotate_formula(ax_dwt, f"GT fixed at median\n({np.median(GT_c):,.0f} t)", r2_mfp,
                 fontsize=8)

# Panel 2: LOA vs GT.
# If MFP eliminated GT (as can happen with this synthetic data), the curve
# will be flat — which is itself the correct and informative result.
gt_grid  = smooth_x(GT_c)
dwt_med  = np.full_like(gt_grid, np.median(DWT_c))
X_gt_grid = np.column_stack([dwt_med.flatten(), gt_grid.flatten()])
y_gt_curve = ModelFactory.predict(model_mfp, None, X_gt_grid, 'mfp')

gt_transform = transforms.get('GT', None)
gt_eliminated = gt_transform is None or gt_transform == 'EXCLUDED'

ax_gt.scatter(GT_c, LOA_mp, s=8, alpha=0.25, color=CONV_COLOR)
ax_gt.plot(gt_grid.flatten(), y_gt_curve, lw=2.5, color=FIT_COLOR,
           linestyle='--' if gt_eliminated else '-')
ax_gt.set_xlabel('GT (t)', fontsize=9)
ax_gt.set_ylabel('LOA (m)', fontsize=9)
title_suffix = 'EXCLUDED by BE' if gt_eliminated else gt_transform
ax_gt.set_title(f'GT → {title_suffix}', fontsize=10,
                color='#c0392b' if gt_eliminated else 'black')
if gt_eliminated:
    ax_gt.text(
        0.5, 0.5,
        'GT eliminated by\nBackward Elimination\n(p > α₁ = 0.05)',
        transform=ax_gt.transAxes,
        fontsize=9, ha='center', va='center', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0',
                  edgecolor='#ffaaaa', alpha=0.9),
    )
else:
    annotate_formula(ax_gt, f"DWT fixed at median\n({np.median(DWT_c):,.0f} t)",
                     r2_mfp, fontsize=8)

# Formula box spanning both panels (placed on ax_gt)
ax_gt.text(
    0.5, -0.22, formula_math,
    transform=ax_gt.transAxes,
    fontsize=9, ha='center', va='top', family='monospace',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
              edgecolor='#aabbdd', alpha=0.95),
)

# Panel 3: residuals
add_residual_panel(ax_res, LOA_mp, y_pred_mfp, color=CONV_COLOR)

fig2.savefig('demo_fig2_mfp_fit.png', bbox_inches='tight', dpi=150)
print("Saved: demo_fig2_mfp_fit.png\n")
plt.show()


# ============================================================================
# Figure 3 — Cross-dataset coefficient comparison
# ============================================================================

print("[Fig 3] Fitting MFP on conventional, refitting on alternative ...")

# Conventional fit
model_conv, _ = ModelFactory.create_and_fit(
    'mfp', X_conv_1d, LOA_c,
    verbose=False, feature_names=['DWT'],
)
y_pred_conv = ModelFactory.predict(model_conv, None, X_conv_1d, 'mfp')
desc_conv   = ModelDescriptor.from_fitted_model(
    model_conv, None, 'mfp', ['DWT'], 'LOA',
    metrics={'r2': r2_score(LOA_c, y_pred_conv)},
)
conv_intercept = desc_conv.intercept
conv_coefs     = dict(desc_conv.coefficients or {})

# Alternative refit (fixed structure)
model_alt_refitted, _ = desc_conv.refit(X_alt_1d, LOA_a, force_structure=True)
y_pred_alt  = ModelFactory.predict(model_alt_refitted, None, X_alt_1d, 'mfp')
alt_intercept = desc_conv.intercept          # mutated in-place by refit
alt_coefs     = dict(desc_conv.coefficients or {})

print(f"  Conv  formula: {ModelDescriptor.from_fitted_model(model_conv, None, 'mfp', ['DWT'], 'LOA').get_formula_string()}")
print(f"  Alt   formula: intercept={alt_intercept:.3f}, coefs={alt_coefs}")

# Smooth curves for both coefficient sets
x_plot = smooth_x(np.concatenate([DWT_c, DWT_a]))
coef_key = list(conv_coefs.keys())[0]   # e.g. '√DWT'

y_conv_curve = conv_intercept + conv_coefs[coef_key] * np.sqrt(x_plot.flatten())
y_alt_curve  = alt_intercept  + alt_coefs[coef_key]  * np.sqrt(x_plot.flatten())

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    'Figure 3 — Cross-Dataset Coefficient Comparison: DWT → LOA',
    fontsize=12, fontweight='bold',
)

ax_curves = axes3[0]
ax_coefs  = axes3[1]

# Panel 1: overlaid curves
ax_curves.scatter(DWT_c, LOA_c, s=8,  alpha=0.25, color=CONV_COLOR,
                  label='Conventional data', zorder=2)
ax_curves.scatter(DWT_a, LOA_a, s=35, alpha=0.75, color=ALT_COLOR,
                  marker='D', label='Alternative-fuel data', zorder=3)
ax_curves.plot(x_plot.flatten(), y_conv_curve, lw=2.5, color=CONV_COLOR,
               label='Conventional fit', zorder=4)
ax_curves.plot(x_plot.flatten(), y_alt_curve,  lw=2.5, color=ALT_COLOR,
               linestyle='--', label='Alternative-fuel fit', zorder=4)

ax_curves.set_xlabel('DWT (t)', fontsize=10)
ax_curves.set_ylabel('LOA (m)', fontsize=10)
ax_curves.set_title('Fitted curves — same structure, different coefficients',
                    fontsize=10)
ax_curves.legend(fontsize=9, framealpha=0.8)

# Formula annotations
ax_curves.text(
    0.03, 0.97,
    f"Conv : {desc_conv.get_formula_string(notation='math')}\n"
    f"Alt  : LOA = {alt_intercept:.2f} {alt_coefs[coef_key]:+.3f}·{coef_key}",
    transform=ax_curves.transAxes,
    fontsize=8.5, va='top', ha='left', family='monospace',
    bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
              edgecolor='#cccccc', alpha=0.92),
)

# Panel 2: coefficient bar comparison
terms    = ['Intercept', coef_key]
c_vals   = [conv_intercept, conv_coefs[coef_key]]
a_vals   = [alt_intercept,  alt_coefs[coef_key]]
x_pos    = np.arange(len(terms))
bar_w    = 0.35

bars_c = ax_coefs.bar(x_pos - bar_w / 2, c_vals, bar_w,
                       color=CONV_COLOR, alpha=0.85, label='Conventional')
bars_a = ax_coefs.bar(x_pos + bar_w / 2, a_vals, bar_w,
                       color=ALT_COLOR,  alpha=0.85, label='Alternative-fuel')

# Annotate Δ above the taller bar
for i, (cv, av) in enumerate(zip(c_vals, a_vals)):
    delta = av - cv
    y_top = max(abs(cv), abs(av)) * (1.0 if min(cv, av) >= 0 else -1)
    sign  = '+' if delta >= 0 else ''
    ax_coefs.text(
        i, max(cv, av) + abs(max(cv, av)) * 0.04,
        f'Δ = {sign}{delta:.3f}',
        ha='center', va='bottom', fontsize=9, fontweight='bold',
    )

ax_coefs.set_xticks(x_pos)
ax_coefs.set_xticklabels(terms, fontsize=10)
ax_coefs.set_ylabel('Coefficient value', fontsize=10)
ax_coefs.set_title('Coefficient comparison\n(same structure, re-estimated)',
                    fontsize=10)
ax_coefs.legend(fontsize=9)
ax_coefs.axhline(0, color='black', lw=0.8)

fig3.tight_layout()
fig3.savefig('demo_fig3_coefficient_comparison.png', bbox_inches='tight', dpi=150)
print("Saved: demo_fig3_coefficient_comparison.png\n")
plt.show()


# ============================================================================
# Figure 4 — Effective-DF comparison bar chart
# ============================================================================

print("[Fig 4] Computing naive vs actual parameter counts ...")

X_df = np.random.uniform(1, 10, (100, 3))
y_df = 2 * X_df[:, 0] + 0.1 * X_df[:, 1] + 5
model_names   = ['linear', 'ridge', 'lasso', 'polynomial-2',
                 'polynomial-2-lasso', 'mfp']
naive_counts  = []
actual_counts = []

for mt in model_names:
    ok, _ = ModelFactory.is_model_applicable(mt, X_df, y_df)
    if not ok:
        naive_counts.append(0); actual_counts.append(0)
        continue
    kwargs = {'feature_names': ['x1', 'x2', 'x3']} if mt == 'mfp' else {}
    m, p = ModelFactory.create_and_fit(mt, X_df, y_df, **kwargs)
    naive_counts.append( ModelFactory.get_n_parameters(mt, X_df))
    actual_counts.append(ModelFactory.get_n_parameters(mt, X_df, fitted_model=m))

fig4, ax4 = plt.subplots(figsize=(11, 5))
fig4.suptitle(
    'Figure 4 — Naive vs Actual Parameter Count\n'
    '(difference matters for AIC / BIC penalisation of sparse models)',
    fontsize=12, fontweight='bold',
)

x_pos = np.arange(len(model_names))
bar_w = 0.38
bars_n = ax4.bar(x_pos - bar_w / 2, naive_counts,  bar_w,
                  color='#7f7f7f', alpha=0.75, label='Naive (max possible)')
bars_a = ax4.bar(x_pos + bar_w / 2, actual_counts, bar_w,
                  color=CONV_COLOR,  alpha=0.85, label='Actual (effective DF)')

# Annotate each pair with the difference
for i, (n, a) in enumerate(zip(naive_counts, actual_counts)):
    diff = n - a
    if diff > 0:
        ax4.text(i, max(n, a) + 0.2, f'−{diff}',
                 ha='center', va='bottom', fontsize=9,
                 color='#c0392b', fontweight='bold')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
ax4.set_ylabel('Number of parameters', fontsize=10)
ax4.set_ylim(0, max(naive_counts) * 1.25)
ax4.legend(fontsize=10)

# Shade the models where they differ
for i, (n, a) in enumerate(zip(naive_counts, actual_counts)):
    if n != a:
        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.07, color='red')

ax4.text(
    0.99, 0.97,
    "Red shading: sparse models where\nnaive DF over-penalises AIC/BIC",
    transform=ax4.transAxes,
    fontsize=8.5, va='top', ha='right',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0',
              edgecolor='#ffaaaa', alpha=0.9),
)

fig4.tight_layout()
fig4.savefig('demo_fig4_effective_df.png', bbox_inches='tight', dpi=150)
print("Saved: demo_fig4_effective_df.png\n")
plt.show()

print("All figures saved successfully.")