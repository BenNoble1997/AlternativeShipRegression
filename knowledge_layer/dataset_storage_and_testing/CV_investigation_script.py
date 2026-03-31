"""
Metric Visualisation: CV, R², RRMSE under Domain Shift
=======================================================

Three plots, each making one sharp methodological point:

  Plot 1 — CV drives R²: identical model, three dataset spreads
            → R² changes; RRMSE barely moves
  Plot 2 — Systematic bias: R² and RRMSE diverge for a single variable
            → the most deceptive domain shift failure mode
  Plot 3 — ΔR² vs ΔRRMSE scatter across ship variables & fuel types
            → domain shift map; quadrants map onto thesis narrative

All dummy data is ship-like in scale (bulk carrier proportions).
Each function accepts debug=True for step-by-step console output.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Colour palette ────────────────────────────────────────────────────────────
C_CONV  = '#2166ac'   # conventional  — blue
C_MATCH = '#4dac26'   # matched       — green
C_DF    = '#d73027'   # dual fuel     — red
C_ALT   = '#f46d43'   # pure alt fuel — orange
C_GOOD  = '#1a9850'
C_WARN  = '#fdae61'
C_CRIT  = '#d73027'

# ── Helpers ───────────────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _rrmse(y_true: np.ndarray, y_pred: np.ndarray, norm_mean: float) -> float:
    """RRMSE normalised by a fixed reference mean (always the conv dataset mean)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return (rmse / norm_mean) * 100.0

def _cv(values: np.ndarray) -> float:
    return (np.std(values) / np.mean(values)) * 100.0




def _annotate(ax, txt: str, loc: str = 'tl'):
    """Place a metric annotation box."""
    x = 0.04 if 'l' in loc else 0.96
    y = 0.96 if 't' in loc else 0.04
    ha = 'left' if 'l' in loc else 'right'
    va = 'top'  if 't' in loc else 'bottom'
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=9.5,
            va=va, ha=ha,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.92))


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — CV drives R²
# ═════════════════════════════════════════════════════════════════════════════


def simple_example():


    # -------------------------
    # Settings
    # -------------------------
    np.random.seed(42)

    mean = 100
    sample_size = 200
    cv_values = [0.05, 0.1, 0.25, 0.5, 1.0]

    for cv in cv_values:
        std = cv * mean
        data = np.random.normal(loc=mean, scale=std, size=sample_size)

        # Create an index for scatter (just to spread points horizontally)
        x = np.arange(sample_size)

        plt.figure()
        plt.scatter(x, data)
        plt.axhline(mean, linestyle="--")
        plt.title(f"Scatter: Mean = {mean}, CV = {cv} (Std = {std:.1f})")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — CV drives R²
# ═════════════════════════════════════════════════════════════════════════════

def plot1_conditional_scatter(debug: bool = False) -> plt.Figure:
    """
    Concept
    -------
    R² reflects the conditional scatter of the target around the fitted
    relationship — not the marginal spread of the target variable.

    We hold the DWT range constant (same ships, same predictor range)
    and vary how consistently LOA follows the LOA~f(DWT) relationship:

      Conventional  : tight conditional scatter  → high R²
                      ships are designed to a well-established formula
      Dual Fuel     : moderate scatter           → moderate R²
                      incremental modifications loosen the relationship
      Pure Alt Fuel : large conditional scatter  → low R²
                      fundamental redesign; same DWT, very different LOA

    Two panels per scenario:
      Left  — scatter plot of LOA vs DWT with fitted curve
      Right — residual plot (actual − predicted) vs DWT
              THIS is what R² is actually measuring

    RRMSE (fixed conventional normaliser) is also shown to contrast
    how it responds differently to conditional scatter vs R².

    Debug output: conditional std, R², RRMSE, residual stats per scenario.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Plot 1 — R² Reflects Conditional Scatter Around the Fitted Relationship\n'
        'DWT range held constant; only the LOA~DWT relationship consistency changes',
        fontsize=12, fontweight='bold'
    )

    # DWT range fixed across ALL scenarios — same predictor space
    DWT_RANGE  = (20_000, 200_000)
    N          = 60
    NORM_MEAN  = None   # set from conventional scenario below

    # True relationship: LOA ≈ 2.5 · DWT^0.33  (bulk carrier approximation)
    def true_loa(dwt): return 2.5 * dwt ** 0.33

    dwt = RNG.uniform(*DWT_RANGE, size=N)
    dwt_sorted = np.sort(dwt)

    # Scenarios: vary conditional noise std around the true relationship
    scenarios = [
        {
            'label':      'Conventional Ships',
            'subtitle':   'Tight conditional scatter\n— established design practice',
            'noise_std':  3.5,    # metres conditional std
            'color':      C_CONV,
        },
        {
            'label':      'Dual Fuel Ships',
            'subtitle':   'Moderate conditional scatter\n— incremental modification',
            'noise_std':  9.0,
            'color':      C_DF,
        },
        {
            'label':      'Pure Alternative Fuel Ships',
            'subtitle':   'Large conditional scatter\n— fundamental redesign',
            'noise_std':  18.0,
            'color':      C_ALT,
        },
    ]

    if debug:
        print(f"\n{'═'*65}")
        print("PLOT 1: Conditional scatter drives R²")
        print(f"  DWT range (fixed) : {DWT_RANGE[0]:,} – {DWT_RANGE[1]:,} t")
        print(f"  N per scenario    : {N}")
        print(f"{'─'*65}")

    # Build grid: 3 rows (scenarios) × 2 cols (scatter | residuals)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                          left=0.07, right=0.97, top=0.88, bottom=0.07)

    for row, sc in enumerate(scenarios):
        noise     = RNG.normal(0, sc['noise_std'], size=N)
        loa_obs   = true_loa(dwt) + noise
        loa_pred  = true_loa(dwt)          # model trained on conv — formula known
        residuals = loa_obs - loa_pred

        # Fix RRMSE normaliser to conventional mean on first pass
        if NORM_MEAN is None:
            NORM_MEAN = float(np.mean(loa_obs))

        r2_val    = _r2(loa_obs, loa_pred)
        rrmse_val = _rrmse(loa_obs, loa_pred, NORM_MEAN)
        res_std   = float(np.std(residuals))

        if debug:
            print(f"\n  Scenario : {sc['label']}")
            print(f"  Cond. noise std : {sc['noise_std']:.1f} m")
            print(f"  Residual std    : {res_std:.2f} m  "
                  f"(should ≈ noise std)")
            print(f"  R²              : {r2_val:.3f}  ← driven by residual std")
            print(f"  RRMSE           : {rrmse_val:.2f}%  "
                  f"(normalised by {NORM_MEAN:.1f} m)")

        # ── Left panel: LOA vs DWT ────────────────────────────────────────
        ax_scatter = fig.add_subplot(gs[row, 0])
        ax_scatter.scatter(dwt / 1_000, loa_obs,
                           alpha=0.45, s=20, color=sc['color'])
        ax_scatter.plot(dwt_sorted / 1_000, true_loa(dwt_sorted),
                        color='black', lw=2, label='Fitted model (conv)')

        # ±1 conditional std band — this is what R² is measuring
        ax_scatter.fill_between(
            dwt_sorted / 1_000,
            true_loa(dwt_sorted) - sc['noise_std'],
            true_loa(dwt_sorted) + sc['noise_std'],
            color=sc['color'], alpha=0.15,
            label=f'±1 conditional std ({sc["noise_std"]:.0f} m)'
        )

        ax_scatter.set_xlabel('DWT (×10³ t)', fontsize=9)
        ax_scatter.set_ylabel('Length OA (m)', fontsize=9)
        ax_scatter.set_title(
            f'{sc["label"]}\n{sc["subtitle"]}',
            fontsize=9.5, fontweight='bold', color=sc['color']
        )
        _annotate(ax_scatter,
                  f'R²    = {r2_val:.3f}\nRRMSE = {rrmse_val:.2f}%')
        ax_scatter.legend(fontsize=7.5, loc='upper left')
        ax_scatter.grid(alpha=0.3)

        # ── Right panel: residuals vs DWT ─────────────────────────────────
        ax_res = fig.add_subplot(gs[row, 1])
        ax_res.scatter(dwt / 1_000, residuals,
                       alpha=0.45, s=20, color=sc['color'])
        ax_res.axhline(0,  color='black', lw=1.5, ls='--')
        ax_res.axhline( sc['noise_std'], color=sc['color'],
                        lw=1, ls=':', alpha=0.7, label=f'+1σ = {sc["noise_std"]:.0f} m')
        ax_res.axhline(-sc['noise_std'], color=sc['color'],
                        lw=1, ls=':', alpha=0.7, label=f'−1σ')

        ax_res.set_xlabel('DWT (×10³ t)', fontsize=9)
        ax_res.set_ylabel('Residual: actual − predicted (m)', fontsize=9)
        ax_res.set_title(
            'Residuals vs DWT\n'
            r'R² measures how large this scatter is relative to total LOA variance',
            fontsize=9, fontweight='bold'
        )
        ax_res.legend(fontsize=7.5)
        ax_res.grid(alpha=0.3)

        if debug:
            print(f"  Residual range  : [{residuals.min():.1f}, "
                  f"{residuals.max():.1f}] m")

    if debug:
        print(f"\n  ► Key takeaway:")
        print(f"    R² falls because the conditional scatter around LOA~f(DWT)")
        print(f"    grows — not because the predictor range changed.")
        print(f"    RRMSE also rises, but on a consistent scale.")
        print(f"    A low R² on alt fuel ships means the LOA~DWT relationship")
        print(f"    is less consistent — ships of the same DWT vary more in LOA.")
        print(f"{'═'*65}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — R² and RRMSE diverge (four scatter panels)
# ═════════════════════════════════════════════════════════════════════════════

def plot2_divergence(debug: bool = False) -> plt.Figure:
    """
    Concept
    -------
    R² and RRMSE measure fundamentally different things and can diverge
    under domain shift.  Four predicted-vs-actual panels cover the
    four (good/poor) × (R², RRMSE) combinations for ship length:

      A — Good R², Good RRMSE : clean transfer
      B — Good R², Poor RRMSE : systematic offset bias
                                R² cannot detect a constant mean shift
      C — Poor R², Good RRMSE : alternative ships are homogeneous
                                (tight cluster → small R² denominator)
      D — Poor R², Poor RRMSE : fundamental relationship breakdown

    Case B is the most deceptive domain shift failure.

    Debug output: CV of actuals, mean bias, R², RRMSE, mechanism.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        'Plot 2 — How R² and RRMSE Can Diverge Under Domain Shift\n'
        'Predicting ship Length OA (m);  each panel = a different failure mode',
        fontsize=12, fontweight='bold'
    )

    N         = 60
    MEAN_LEN  = 180.0    # typical bulk carrier LOA, metres
    NORM_MEAN = MEAN_LEN # fixed conventional reference

    base = RNG.normal(MEAN_LEN, 20.0, size=N)   # CV ≈ 11 %

    # Case definitions ─────────────────────────────────────────────────────
    # Each entry: actual array, predicted array, label, colour, axes position
    tight_actual = RNG.normal(MEAN_LEN, 4.0, size=N)  # CV ≈ 2 %  for case C

    cases = [
        {
            'title':   'A — Clean Transfer\n(Good R², Good RRMSE)',
            'actual':  base,
            'pred':    base + RNG.normal(0, 3, N),
            'color':   C_GOOD,
            'ax':      axes[0, 0],
            'mech':    'Small random noise only — model transfers well',
        },
        {
            'title':   'B — Systematic Offset Bias\n(Good R², Poor RRMSE)',
            'actual':  base,
            'pred':    base + 18 + RNG.normal(0, 3, N),  # +18 m constant shift
            'color':   C_WARN,
            'ax':      axes[0, 1],
            'mech':    'Constant +18 m offset — R² insensitive to mean shift',
        },
        {
            'title':   'C — Homogeneous Alt Dataset\n(Poor R², Good RRMSE)',
            'actual':  tight_actual,
            'pred':    tight_actual + RNG.normal(0, 3, N),
            'color':   C_WARN,
            'ax':      axes[1, 0],
            'mech':    'Alt ships cluster tightly (CV ≈ 2 %) → tiny R² denominator',
        },
        {
            'title':   'D — Fundamental Breakdown\n(Poor R², Poor RRMSE)',
            'actual':  base,
            'pred':    base * RNG.uniform(0.70, 1.30, N),  # ±30 % multiplicative scatter
            'color':   C_CRIT,
            'ax':      axes[1, 1],
            'mech':    'Multiplicative scatter — design relationship restructured',
        },
    ]

    if debug:
        print(f"\n{'═'*60}")
        print("PLOT 2: R² and RRMSE divergence")
        print(f"  Conv reference mean : {NORM_MEAN:.1f} m")
        print(f"{'─'*60}")

    for cfg in cases:
        actual, pred = cfg['actual'], cfg['pred']
        ax = cfg['ax']

        r2_val    = _r2(actual, pred)
        rrmse_val = _rrmse(actual, pred, NORM_MEAN)
        bias      = float(np.mean(pred - actual))
        cv_val    = _cv(actual)

        if debug:
            print(f"\n  {cfg['title'].replace(chr(10),' ')}")
            print(f"    CV(actual) : {cv_val:.1f}%")
            print(f"    Mean bias  : {bias:+.2f} m")
            print(f"    R²         : {r2_val:.3f}")
            print(f"    RRMSE      : {rrmse_val:.2f}%")
            print(f"    Mechanism  : {cfg['mech']}")

        lo = min(actual.min(), pred.min()) - 8
        hi = max(actual.max(), pred.max()) + 8
        ax.scatter(actual, pred, alpha=0.45, s=22, color=cfg['color'])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect prediction')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('Actual Length OA (m)', fontsize=10)
        ax.set_ylabel('Predicted Length OA (m)', fontsize=10)
        ax.set_title(cfg['title'], fontsize=10, fontweight='bold',
                     color=cfg['color'])
        _annotate(ax, f'CV     = {cv_val:.1f}%\nR²     = {r2_val:.3f}\nRRMSE = {rrmse_val:.2f}%')
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if debug:
        print(f"\n  ► Case B is the key risk: R² looks acceptable but")
        print(f"    every prediction is ~18 m too high — impractical for design.")
        print(f"{'═'*60}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — ΔR² vs ΔRRMSE domain shift map
# ═════════════════════════════════════════════════════════════════════════════

def plot3_delta_scatter(debug: bool = False) -> plt.Figure:
    """
    Concept
    -------
    Plotting ΔR² (R² drop) against ΔRRMSE (RRMSE increase) for every
    target variable × fuel type in a single scatter gives an interpretable
    'domain shift map'.  The quadrant each point falls in maps directly
    onto the thesis narrative:

      Bottom-left : small ΔR², small ΔRRMSE → transferable relationship
      Top-left    : small ΔR², large ΔRRMSE → systematic bias / offset
      Bottom-right: large ΔR², small ΔRRMSE → homogeneous alt dataset
      Top-right   : large ΔR², large ΔRRMSE → fundamental restructuring

    Dual fuel points (circles) should cluster closer to origin than
    pure alternative (triangles), illustrating incremental vs fundamental.

    Debug output: per-variable Δ values and quadrant assignment.
    """
    # Dummy Δ values — designed to span all quadrants realistically
    # (delta_r2_drop, delta_rrmse_increase_pct)
    variables = {
        'Length OA':      {'df': (0.02, 1.2),  'alt': (0.05, 3.1)},
        'Beam':           {'df': (0.03, 1.8),  'alt': (0.08, 4.5)},
        'Draft':          {'df': (0.04, 2.5),  'alt': (0.12, 7.2)},
        'Displacement':   {'df': (0.06, 4.1),  'alt': (0.22, 12.8)},
        'GT':             {'df': (0.05, 3.3),  'alt': (0.18,  9.5)},
        'ME Power':       {'df': (0.14, 6.8),  'alt': (0.38, 18.4)},
        'Fuel Tank Vol':  {'df': (0.10, 8.2),  'alt': (0.42, 22.1)},
        'Cargo Capacity': {'df': (0.09, 5.5),  'alt': (0.31, 15.3)},
    }

    # Threshold lines — where to draw quadrant boundaries
    R2_THRESH    = 0.10   # 10 pp R² drop
    RRMSE_THRESH = 8.0    # 8 % RRMSE increase

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(
        'Plot 3 — Domain Shift Map: ΔR² vs ΔRRMSE per Variable\n'
        'Each point = one ship variable × fuel type;  '
        'quadrant = failure mode',
        fontsize=12, fontweight='bold'
    )

    # ── Quadrant shading ──────────────────────────────────────────────────────
    XLIM = (-0.02, 0.50);  YLIM = (-1.5, 28)
    ax.set_xlim(*XLIM);    ax.set_ylim(*YLIM)

    def _shade(x0, x1, y0, y1, color):
        ax.fill_between([x0, x1], y0, y1, color=color, alpha=0.08, zorder=0)

    _shade(XLIM[0], R2_THRESH,    YLIM[0], RRMSE_THRESH, C_GOOD)   # BL — clean
    _shade(XLIM[0], R2_THRESH,    RRMSE_THRESH, YLIM[1], C_WARN)   # TL — bias
    _shade(R2_THRESH, XLIM[1],    YLIM[0], RRMSE_THRESH, C_WARN)   # BR — homogeneous
    _shade(R2_THRESH, XLIM[1],    RRMSE_THRESH, YLIM[1], C_CRIT)   # TR — breakdown

    ax.axvline(R2_THRESH,    color='#888888', lw=1.2, ls='--')
    ax.axhline(RRMSE_THRESH, color='#888888', lw=1.2, ls='--')

    # Quadrant text labels
    kw = dict(fontsize=9, alpha=0.75, va='center')
    ax.text(0.005, 3.5,  'Clean\nTransfer',       color=C_GOOD, **kw)
    ax.text(0.005, 22.0, 'Systematic\nBias',       color=C_WARN, **kw)
    ax.text(0.20,  3.5,  'Homogeneous\nAlt Data',  color=C_WARN, **kw)
    ax.text(0.20,  22.0, 'Fundamental\nBreakdown', color=C_CRIT, **kw)

    # ── Data points ───────────────────────────────────────────────────────────
    if debug:
        print(f"\n{'═'*65}")
        print("PLOT 3: ΔR² vs ΔRRMSE domain shift map")
        print(f"  Thresholds: ΔR² > {R2_THRESH}  |  ΔRRMSE > {RRMSE_THRESH}%")
        print(f"{'─'*65}")
        print(f"  {'Variable':<18} {'Type':<18} {'ΔR²':>6} {'ΔRRMSE':>8}  Quadrant")
        print(f"  {'─'*18} {'─'*18} {'─'*6} {'─'*8}  {'─'*22}")

    for var, fdata in variables.items():
        for ftype, (dr2, drrmse) in fdata.items():
            color  = C_DF  if ftype == 'df' else C_ALT
            marker = 'o'   if ftype == 'df' else '^'
            label  = 'Dual Fuel' if ftype == 'df' else 'Pure Alternative'

            ax.scatter(dr2, drrmse, marker=marker, color=color,
                       s=95, zorder=5, edgecolors='white', lw=0.8)

            # Offset label slightly so points aren't obscured
            x_off = 6 if dr2 < 0.35 else -6
            ha    = 'left' if x_off > 0 else 'right'
            ax.annotate(var, (dr2, drrmse),
                        textcoords='offset points', xytext=(x_off, 5),
                        fontsize=8, color=color, ha=ha)

            if debug:
                q = _quadrant(dr2, drrmse, R2_THRESH, RRMSE_THRESH)
                print(f"  {var:<18} {label:<18} {dr2:>6.3f} {drrmse:>8.2f}  {q}")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DF,
               markersize=9, label='Dual Fuel', markeredgecolor='#aaaaaa'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_ALT,
               markersize=9, label='Pure Alternative', markeredgecolor='#aaaaaa'),
        mpatches.Patch(facecolor=C_GOOD, alpha=0.3,  label='Clean transfer'),
        mpatches.Patch(facecolor=C_WARN, alpha=0.3,  label='Bias / homogeneous'),
        mpatches.Patch(facecolor=C_CRIT, alpha=0.3,  label='Fundamental breakdown'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9)

    ax.set_xlabel('ΔR²  (R² drop:  conventional − alternative,  higher = worse)',
                  fontsize=11)
    ax.set_ylabel('ΔRRMSE  (RRMSE increase,  higher = worse,  %)',
                  fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if debug:
        print(f"\n  ► Geometric vars (Length, Beam, Draft) cluster bottom-left")
        print(f"    → supports transferability claim")
        print(f"  ► ME Power, Fuel Tank Vol top-right")
        print(f"    → supports fundamental restructuring claim")
        print(f"  ► Dual fuel consistently closer to origin than pure alt")
        print(f"    → supports incremental vs fundamental distinction")
        print(f"{'═'*65}")
    return fig


def _quadrant(dr2, drrmse, r2_t, rrmse_t) -> str:
    r2_bad, rrmse_bad = dr2 > r2_t, drrmse > rrmse_t
    if not r2_bad and not rrmse_bad: return "Clean Transfer"
    if not r2_bad and rrmse_bad:     return "Systematic Bias"
    if r2_bad     and not rrmse_bad: return "Homogeneous Alt Data"
    return "Fundamental Breakdown"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_all(debug: bool = False):
    """
    Generate all three diagnostic plots.

    Parameters
    ----------
    debug : bool
        Print step-by-step breakdown for each plot stage.
    """
    print("Generating Plot 1: Conditional scatter drives R²...")
    fig1 = plot1_conditional_scatter(debug=debug)

    print("Generating Plot 2: R² and RRMSE divergence...")
    fig2 = plot2_divergence(debug=debug)

    print("Generating Plot 3: ΔR² vs ΔRRMSE domain shift map...")
    fig3 = plot3_delta_scatter(debug=debug)

    plt.show()
    return fig1, fig2, fig3


if __name__ == '__main__':
    run_all(debug=True)
    # simple_example()