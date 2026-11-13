"""
Functional-Form Comparison Plots
================================

Renders the fitted response curve of one model structure estimated on three
cohorts — Full Conventional, Matched Conventional, Alternative Fuel — so that
a change in the *shape* of the design relationship can be read directly,
rather than inferred from a table of coefficient deltas.

Relationship to the coefficient plots
-------------------------------------
:class:`~coefficient_plotter.CoefficientChangePlotter` answers "which
parameters moved, and by how much?".  This module answers "does the fitted
function actually go anywhere different?".  The two can disagree in both
directions: correlated predictors can produce large offsetting coefficient
shifts with a near-identical curve, and a small shift in a fractional power
can bend a curve substantially at the ends of the range.  Both views are
needed.

The module contains rendering only.  All model evaluation is delegated to
:class:`~functional_form_curves.FunctionalFormCurveExtractor` by composition.
"""


import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from knowledge_layer.multi_dataset_analysers.coefficient_comparison.function_form_curves import (
    CohortCurve,
    FunctionalFormCurveExtractor,
    TargetCurveSet,
)


@dataclass(frozen=True)
class FontSizes:
    """
    Point sizes for every piece of text a functional-form figure draws.

    Why a container rather than loose arguments
    -------------------------------------------
    Legibility is a property of the figure as a whole, not of any one
    label.  Tuning it means moving several sizes together and re-rendering,
    which is awkward when each size is a separate keyword buried among
    thirty others.  Gathering them into one object gives two things a flat
    signature cannot: a single ``font_scale`` that moves every size at once
    while preserving the ratios between them, and a named set that can be
    defined once and reused across every figure in a document.

    The container is frozen so that a set shared between figures cannot be
    mutated by one of them; :meth:`scaled` returns a new instance.

    Typical use::

        base = FontSizes(axis_label=11, tick_label=9)

        plotter.plot_functional_forms_all_targets(
            layout='overlay', fonts=base, font_scale=1.25,
        )

    A plain ``dict`` is accepted anywhere a ``FontSizes`` is, so a one-off
    override needs no import::

        plotter.plot_functional_form('dwt', fonts={'tick_label': 7})

    :ivar figure_title: Figure-level heading, and the suptitle of a
        multi-target grid.
    :vartype figure_title: float
    :ivar axis_title: Per-axis heading: the cohort name under
        ``'panels'``, the target name under ``'overlay'``.
    :vartype axis_title: float
    :ivar axis_label: The x and y axis labels.
    :vartype axis_label: float
    :ivar tick_label: Tick numerals.  Usually the first size to reduce,
        since ticks are the densest text on a tiled figure.
    :vartype tick_label: float
    :ivar legend: Legend entries, including the R² text appended to them.
    :vartype legend: float
    :ivar formula: The italic β-notation formula line.
    :vartype formula: float
    :ivar annotation: Free-standing text drawn inside the axes: the R² box
        and the "not evaluable" placeholder.
    :vartype annotation: float
    """

    figure_title: float = 13.0
    axis_title: float = 10.0
    axis_label: float = 10.0
    tick_label: float = 9.0
    legend: float = 9.0
    formula: float = 9.0
    annotation: float = 9.0

    def scaled(self, factor: float) -> 'FontSizes':
        """
        Multiply every size by a common factor.

        Scaling the set rather than each size individually is what keeps a
        figure's typographic hierarchy intact: the title stays larger than
        the ticks by the same ratio at any overall size.

        :param factor: Strictly positive multiplier.
        :type factor: float
        :returns: A new instance; the original is unchanged.
        :rtype: FontSizes
        :raises ValueError: If ``factor`` is not finite and positive, which
            would otherwise produce zero-height or inverted text.
        """
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"font_scale must be finite and positive; got {factor!r}."
            )

        return replace(self, **{
            f.name: getattr(self, f.name) * factor for f in fields(self)
        })

    @classmethod
    def resolve(
            cls,
            spec: 'FontSizes | Mapping[str, float] | None' = None,
            scale: float = 1.0,
    ) -> 'FontSizes':
        """
        Normalise a user-supplied font specification into an instance.

        Accepts ``None`` (the defaults), an existing instance, or a mapping
        of field name to size.  An unrecognised key raises rather than
        being ignored, on the same reasoning as the keyword validation in
        :meth:`~FunctionalFormPlotter.plot_functional_forms_all_targets`: a
        misspelled size that silently does nothing costs more time to
        diagnose than an exception does to read.

        :param spec: The specification to normalise.
        :type spec: FontSizes or dict, optional
        :param scale: Multiplier applied to the resolved set.
        :type scale: float
        :returns: A resolved, scaled instance.
        :rtype: FontSizes
        :raises TypeError: On an unusable type or an unknown field name.
        """
        if spec is None:
            base = cls()
        elif isinstance(spec, cls):
            base = spec
        elif isinstance(spec, Mapping):
            known = {f.name for f in fields(cls)}
            unknown = set(spec) - known

            if unknown:
                raise TypeError(
                    f"Unknown font size(s) {sorted(unknown)}. "
                    f"Available: {sorted(known)}."
                )

            base = replace(cls(), **dict(spec))
        else:
            raise TypeError(
                f"'fonts' must be a FontSizes, a dict of size overrides, or "
                f"None; got {type(spec).__name__}."
            )

        return base if scale == 1.0 else base.scaled(scale)


class FunctionalFormPlotter:
    """
    Draw fitted-function comparisons across fuel cohorts.

    Two layouts
    -----------
    ``layout='panels'`` (default)
        One axis per cohort, side by side, sharing both axes.  Reading three
        curves against a common frame makes a change in curvature easy to
        see and keeps the scatter legible when it is switched on.

    ``layout='overlay'``
        All three curves on one axis.  Better for detecting a small vertical
        offset, worse once observation points are drawn.

    Axis definition
    ---------------
    Both layouts take their limits from one nominated cohort — the full
    conventional fleet by default.  Fixing the frame to a single cohort is
    what makes the panels comparable: if each panel were scaled to its own
    data, a steeper-looking alternative-fuel curve might reflect nothing but
    a narrower axis.

    Extrapolation
    -------------
    A curve is drawn solid only across the range its own cohort actually
    covers, and dotted at reduced opacity outside it.  With a
    dual-fuel cohort of 20–150 ships against a conventional fleet of
    thousands, much of a shared axis is extrapolation for the alternative
    model, and an unbroken line would overstate what the data supports.

    Composition
    -----------
    The plotter holds a
    :class:`~functional_form_curves.FunctionalFormCurveExtractor`, which in
    turn holds the analyser.  Construct with either an ``extractor`` or an
    ``analyzer``; the latter builds the extractor for you.
    """

    #: Per-cohort colour, line style and marker.  Override via the
    #: ``cohort_styles`` constructor argument to match a house palette.
    DEFAULT_COHORT_STYLES: dict[str, dict[str, Any]] = {
        'conventional_full': {
            'color': '#3B6EA5', 'linestyle': '-', 'marker': 'o',
        },
        'conventional_matched': {
            'color': '#D08C34', 'linestyle': '--', 'marker': 's',
        },
        'alternative': {
            'color': '#4C9A6B', 'linestyle': '-.', 'marker': '^',
        },
    }

    #: Keywords consumed by the curve extractor. Split out so that
    #: :meth:`plot_functional_forms_all_targets` can build each target's
    #: curves once and reuse them for drawing.
    _EXTRACT_KEYS: frozenset[str] = frozenset({
        'reference_dataset', 'n_grid', 'x_range_mode', 'range_padding',
    })

    #: Keywords consumed by :meth:`_render_curve_set_onto`. ``reference_dataset``
    #: appears in both sets: the extractor uses it to choose the evaluation
    #: range, the renderer to choose the axis frame.
    _RENDER_KEYS: frozenset[str] = frozenset({
        'show_points', 'reference_dataset', 'xlim_mode', 'ylim_mode',
        'ylim_padding', 'mark_support', 'linewidth', 'point_size',
        'point_alpha', 'variable_name_map', 'formula_name_map',
        'display_scale_map', 'show_formula', 'formula_fontsize',
        'max_xticks', 'max_yticks', 'xtick_rotation',
        'legend_loc', 'share_axes',
    })

    #: Anchor positions for the R² box, as
    #: ``(x, y, horizontalalignment, verticalalignment)`` in axes
    #: coordinates.  Named rather than free-form so a typo is caught at
    #: call time instead of silently placing the box off the axes.
    _R2_BOX_ANCHORS: dict[str, tuple[float, float, str, str]] = {
        'lower right': (0.97, 0.03, 'right', 'bottom'),
        'lower left': (0.03, 0.03, 'left', 'bottom'),
        'upper right': (0.97, 0.97, 'right', 'top'),
        'upper left': (0.03, 0.97, 'left', 'top'),
    }

    #: Weight given to legend overlap when ``r2_loc='auto'`` scores the
    #: four corners.  Set well above 1 so that avoiding the legend
    #: outranks avoiding the curves: a box over a line is still readable,
    #: a box under the legend is not there at all.
    _R2_LEGEND_PENALTY: float = 8.0

    #: Digit-to-superscript translation used to build ``×10ⁿ`` axis suffixes.
    _SUPERSCRIPTS = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')

    def __init__(
            self,
            analyzer=None,
            extractor: Optional[FunctionalFormCurveExtractor] = None,
            cohort_styles: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """
        Bind the plotter to a curve source.

        :param analyzer: A completed
            :class:`~CoefficientComparisonAnalyzer`.  Used to construct an
            extractor when one is not supplied.
        :param extractor: An existing extractor, when several plotters
            should share one.
        :type extractor: FunctionalFormCurveExtractor, optional
        :param cohort_styles: Style overrides, merged over
            :attr:`DEFAULT_COHORT_STYLES` per cohort key.
        :type cohort_styles: dict, optional

        :raises ValueError: If neither ``analyzer`` nor ``extractor`` is
            given.
        """
        if extractor is None:
            if analyzer is None:
                raise ValueError(
                    "FunctionalFormPlotter needs either an 'analyzer' or an "
                    "'extractor'; it does not evaluate models itself."
                )
            extractor = FunctionalFormCurveExtractor(analyzer)

        self.extractor = extractor
        self.analyzer = extractor.analyzer

        self.cohort_styles = {
            cohort: {**style, **(cohort_styles or {}).get(cohort, {})}
            for cohort, style in self.DEFAULT_COHORT_STYLES.items()
        }

    # -------------------------------------------------------------------
    # Public API — single target
    # -------------------------------------------------------------------
    def plot_functional_form(
            self,
            target: str,
            *,
            layout: str = 'panels',
            show_points: bool = False,
            reference_dataset: str = 'conventional_full',
            n_grid: int = 200,
            x_range_mode: str = 'reference',
            range_padding: float = 0.0,
            share_axes: bool = True,
            xlim_mode: str = 'reference',
            ylim_mode: str = 'reference',
            ylim_padding: float = 0.05,
            mark_support: bool = True,
            linewidth: float = 2.0,
            point_size: float = 12.0,
            point_alpha: float = 0.30,
            display_scale_map: Optional[dict[str, Any]] = None,
            max_xticks: Optional[int] = None,
            max_yticks: Optional[int] = None,
            xtick_rotation: float = 0.0,
            show_r2: bool = True,
            r2_cohorts: 'str | Sequence[str]' = 'alternative',
            r2_mode: str = 'legend',
            r2_decimals: int = 2,
            r2_loc: str = 'auto',
            figsize_per_panel: tuple[float, float] = (4.5, 4.0),
            overlay_width_factor: float = 1.4,
            wspace: Optional[float] = None,
            hspace: Optional[float] = None,
            fonts: 'FontSizes | dict[str, float] | None' = None,
            font_scale: float = 1.0,
            variable_name_map: Optional[dict[str, str]] = None,
            formula_name_map: Optional[dict[str, str]] = None,
            show_formula: bool = True,
            formula_fontsize: Optional[float] = None,
            title: Optional[str] = None,
            legend_loc: str = 'best',
            save_path: str | Path | None = None,
            dpi: int = 300,
            show_plot: bool = False,
            curve_set: Optional[TargetCurveSet] = None,
            debug: bool = False,
    ):
        """
        Plot one target's fitted function across the three cohorts.

        Returns ``None`` for a target whose selected model is not
        single-predictor; the extractor reports the skip and records it.

        :param target: Target variable name.
        :type target: str

        :param layout: ``'panels'`` for one axis per cohort in a row, or
            ``'overlay'`` for all three curves on a single axis.
        :type layout: str

        :param show_points: Draw the observed data behind the curves.
            Defaults to ``False``, so the figure reads as a comparison of
            fitted forms rather than three separate goodness-of-fit plots.
        :type show_points: bool

        :param reference_dataset: Cohort that defines the axis frame and,
            under ``x_range_mode='reference'``, the evaluation range.
        :type reference_dataset: str

        :param n_grid: Number of points along the curve.
        :type n_grid: int

        :param x_range_mode: ``'reference'`` to evaluate only over the
            reference cohort's range, ``'union'`` to cover every cohort.
        :type x_range_mode: str

        :param range_padding: Fractional padding on the evaluation range.
        :type range_padding: float

        :param share_axes: Share x and y across panels.  Switching this off
            defeats the purpose of the comparison and is offered only for
            diagnosis.
        :type share_axes: bool

        :param xlim_mode: ``'reference'`` to clip the x-axis to the
            reference cohort's observed range, or ``'grid'`` to show the
            whole evaluation range.
        :type xlim_mode: str

        :param ylim_mode: ``'reference'`` (frame set by the reference
            cohort), ``'curves'`` (contain all three curves) or ``'data'``
            (contain curves and all observations).
        :type ylim_mode: str

        :param ylim_padding: Fractional padding added to the y-range.
        :type ylim_padding: float

        :param mark_support: Draw each curve dotted and faded outside the
            range its own cohort covers.
        :type mark_support: bool

        :param linewidth: Curve line width.
        :type linewidth: float

        :param point_size: Scatter marker area.
        :type point_size: float

        :param point_alpha: Scatter marker opacity.
        :type point_alpha: float

        :param show_r2: Annotate the fitted R² of each selected cohort.
            The value is the in-sample R² recorded by the analyser for the
            model fitted on that cohort, in original units — the same
            number the results table carries, not one recomputed from the
            plotted curve.
        :type show_r2: bool

        :param r2_cohorts: Which cohorts are annotated: ``'alternative'``
            (default), ``'all'``, a single cohort key, or an explicit
            sequence of keys.
        :type r2_cohorts: str or sequence[str]

        :param r2_mode: Where the value is written.  ``'legend'`` appends
            it to the cohort's legend entry under ``'overlay'`` and to the
            panel heading under ``'panels'``; ``'box'`` draws a separate
            framed annotation inside the axes; ``'both'`` does each;
            ``'none'`` suppresses it, as ``show_r2=False`` does.
        :type r2_mode: str

        :param r2_decimals: Decimal places in the printed R².
        :type r2_decimals: int

        :param r2_loc: Corner for the box.  ``'auto'`` (default) scores
            the four corners for legend overlap and data density and takes
            the clearest, so the box and the legend do not land on top of
            each other; a fixed ``'lower right'``, ``'lower left'``,
            ``'upper right'`` or ``'upper left'`` overrides that.
        :type r2_loc: str

        :param figsize_per_panel: ``(width, height)`` in inches per axis.
            The first lever to reach for when labels collide: enlarging
            the panel is usually better than shrinking the text, since it
            keeps the figure's type size consistent with the surrounding
            document.
        :type figsize_per_panel: tuple[float, float]

        :param overlay_width_factor: Extra width given to a single-target
            ``'overlay'`` figure, which carries a three-entry legend inside
            one axis where ``'panels'`` spreads the same information across
            three.
        :type overlay_width_factor: float

        :param wspace: Horizontal gap between panels as a fraction of the
            average panel width, applied after ``tight_layout``.  ``None``
            leaves the automatic spacing.
        :type wspace: float, optional

        :param hspace: Vertical gap between panels, on the same terms.
        :type hspace: float, optional

        :param fonts: Point sizes for every text element: a
            :class:`FontSizes`, a dict of overrides such as
            ``{'tick_label': 8}``, or ``None`` for the defaults.
        :type fonts: FontSizes or dict, optional

        :param font_scale: Multiplier applied to the whole font set, for
            scaling a figure up or down in one step without disturbing the
            ratios between title, label and tick sizes.
        :type font_scale: float

        :param variable_name_map: Internal → display names for axis labels
            and the title.
        :type variable_name_map: dict[str, str], optional

        :param formula_name_map: Internal → display names inside the
            formula line.  Falls back to ``variable_name_map``.
        :type formula_name_map: dict[str, str], optional

        :param formula_fontsize: Explicit point size for the formula line,
            overriding :attr:`FontSizes.formula`.  Retained for
            compatibility with existing call sites; new code should set
            the size through ``fonts``.
        :type formula_fontsize: float, optional

        :param show_formula: Show the shared β-notation formula under the
            title.
        :type show_formula: bool

        :param title: Replace the generated figure title.
        :type title: str, optional

        :param legend_loc: Matplotlib legend location.
        :type legend_loc: str

        :param save_path: Destination file; parent directories are created.
        :type save_path: str or pathlib.Path, optional

        :param dpi: Resolution used when saving.
        :type dpi: int

        :param show_plot: Call ``plt.show()`` at the end.
        :type show_plot: bool

        :param curve_set: A pre-built curve set, bypassing extraction.
            Use when the same curves feed several figures.
        :type curve_set: TargetCurveSet, optional

        :param debug: Print each stage — extraction, limit resolution,
            per-cohort drawing, saving — to stdout.
        :type debug: bool

        :param display_scale_map: Mapping from variable name to a display
            divisor, or to a ``(divisor, label)`` pair.  Values are divided
            before drawing and the axis label gains a ``×10ⁿ`` suffix, or
            the supplied label replaces it outright.  Affects the figure
            only; the underlying curve arrays are untouched.
            Example: ``{'DWT': 1e3, 'MCR (KW)': (1e3, 'MCR (MW)')}``.
        :type display_scale_map: dict, optional

        :param max_xticks: Upper bound on the number of x-axis ticks.
        :type max_xticks: int, optional

        :param max_yticks: Upper bound on the number of y-axis ticks.
        :type max_yticks: int, optional

        :param xtick_rotation: Rotation in degrees for x tick labels.
        :type xtick_rotation: float

        :returns: The figure, or ``None`` when the target could not be
            plotted.
        :rtype: matplotlib.figure.Figure, optional
        """
        if debug:
            print("\n" + "=" * 78)
            print(f"🔍 DEBUG: plot_functional_form — {target!r} (layout={layout})")
            print("=" * 78)

        # ── Stage 0: resolve presentation settings ───────────────────────
        # Done before anything is drawn so that an invalid font key or
        # cohort name fails immediately, rather than after the models have
        # been evaluated.
        fonts = FontSizes.resolve(fonts, font_scale)
        r2_targets = self._resolve_r2_cohorts(show_r2, r2_cohorts, r2_mode)
        r2_in_labels = r2_mode in ('legend', 'both')
        r2_in_box = r2_mode in ('box', 'both')

        if debug:
            print(f"   fonts      : {fonts}")
            print(f"   R² cohorts : {sorted(r2_targets) or 'none'} "
                  f"(mode={r2_mode})")

        # ── Stage 1: obtain curves ───────────────────────────────────────
        if curve_set is None:
            curve_set = self.extractor.build_curve_set(
                target,
                reference_dataset=reference_dataset,
                n_grid=n_grid,
                x_range_mode=x_range_mode,
                range_padding=range_padding,
                include_points=show_points or ylim_mode == 'data',
                debug=debug,
            )

        if curve_set is None:
            # Either no analysed row, or refused by the single-predictor
            # gate — which has already printed its own warning.
            return None

        if not curve_set.valid_curves:
            print(f"⚠️  No cohort could be evaluated for {target!r}: "
                  f"{curve_set.message or 'all cohorts failed'}")
            return None

        # ── Stage 1b: display rescaling, before anything reads the arrays ─
        curve_set = self._apply_display_scaling(
            curve_set, display_scale_map, debug=debug,
        )

        # ── Stage 2: figure skeleton ─────────────────────────────────────
        if layout == 'panels':
            n_panels = len(curve_set.curves)
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
                sharex=share_axes, sharey=share_axes,
            )
            axes = np.atleast_1d(axes)
        elif layout == 'overlay':
            fig, ax = plt.subplots(
                figsize=(figsize_per_panel[0] * overlay_width_factor,
                         figsize_per_panel[1]),
            )
            axes = np.array([ax])
        else:
            raise ValueError(
                f"Unknown layout {layout!r}; expected 'panels' or 'overlay'."
            )

        if debug:
            print(f"   axes       : {len(axes)} "
                  f"({'one per cohort' if layout == 'panels' else 'shared'})")

        # ── Stage 3: axis limits, from the reference cohort ──────────────
        xlim, ylim = self._resolve_limits(
            curve_set=curve_set,
            reference_dataset=reference_dataset,
            xlim_mode=xlim_mode,
            ylim_mode=ylim_mode,
            ylim_padding=ylim_padding,
            include_points=show_points,
            debug=debug,
        )

        # ── Stage 4: draw ────────────────────────────────────────────────
        for idx, curve in enumerate(curve_set.curves):
            ax = axes[idx] if layout == 'panels' else axes[0]

            caption = self._cohort_caption(
                curve, r2_targets, r2_decimals, include_r2=r2_in_labels,
            )

            self._draw_cohort(
                ax=ax,
                curve=curve,
                show_points=show_points,
                mark_support=mark_support,
                linewidth=linewidth,
                point_size=point_size,
                point_alpha=point_alpha,
                label=(f'{curve.label} ({caption})'
                       if layout == 'overlay' else None),
                fonts=fonts,
                debug=debug,
            )

            if layout == 'panels':
                ax.set_title(
                    f'{curve.label}  ({caption})',
                    fontsize=fonts.axis_title, fontweight='bold',
                )

        # ── Stage 5: labels, limits, ticks, legend ───────────────────────
        x_label = self._axis_label(
            curve_set.predictor, variable_name_map, display_scale_map)
        y_label = self._axis_label(
            curve_set.target, variable_name_map, display_scale_map)

        for idx, ax in enumerate(axes):
            ax.set_xlabel(x_label, fontsize=fonts.axis_label)
            if idx == 0 or not share_axes:
                ax.set_ylabel(y_label, fontsize=fonts.axis_label)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            self._apply_tick_policy(
                ax, max_xticks, max_yticks, xtick_rotation,
                tick_labelsize=fonts.tick_label)

        # The legend is built first so that ``r2_loc='auto'`` has
        # something to measure; the box is what moves out of the way, not
        # the legend, because the legend's own 'best' placement already
        # accounts for the data.
        legend = None

        if layout == 'overlay':
            legend = axes[0].legend(loc=legend_loc, fontsize=fonts.legend,
                                    framealpha=0.9)

        # The box is drawn per axis under 'panels', where each axis holds
        # one cohort, and once on the shared axis under 'overlay'.
        if r2_in_box:
            if layout == 'panels':
                for idx, curve in enumerate(curve_set.curves):
                    self._add_r2_box(
                        ax=axes[idx],
                        curve_set=replace(curve_set, curves=[curve]),
                        r2_targets=r2_targets & {curve.cohort},
                        r2_decimals=r2_decimals,
                        fontsize=fonts.annotation,
                        loc=r2_loc,
                        debug=debug,
                    )
            else:
                self._add_r2_box(
                    ax=axes[0], curve_set=curve_set,
                    r2_targets=r2_targets, r2_decimals=r2_decimals,
                    fontsize=fonts.annotation, loc=r2_loc,
                    legend=legend, debug=debug,
                )

        # ── Stage 6: title block ─────────────────────────────────────────
        self._add_title_block(
            fig=fig,
            curve_set=curve_set,
            title=title,
            variable_name_map=variable_name_map,
            formula_name_map=formula_name_map or variable_name_map,
            show_formula=show_formula,
            fonts=(fonts if formula_fontsize is None
                   else replace(fonts, formula=formula_fontsize)),
            debug=debug,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.91))

        # Applied after tight_layout, which sets its own spacing and would
        # otherwise discard these.
        if wspace is not None or hspace is not None:
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

        # ── Stage 7: output ──────────────────────────────────────────────
        self._finalise(fig, save_path, dpi, show_plot, debug)

        return fig

    # -------------------------------------------------------------------
    # Public API — many targets
    # -------------------------------------------------------------------
    def plot_functional_forms_all_targets(
            self,
            targets: Optional[Sequence[str]] = None,
            *,
            layout: str = 'panels',
            n_cols: int = 2,
            figsize_per_panel: tuple[float, float] = (4.5, 4.0),
            wspace: Optional[float] = None,
            hspace: Optional[float] = None,
            fonts: 'FontSizes | dict[str, float] | None' = None,
            font_scale: float = 1.0,
            show_r2: bool = True,
            r2_cohorts: 'str | Sequence[str]' = 'alternative',
            r2_mode: str = 'legend',
            r2_decimals: int = 2,
            r2_loc: str = 'auto',
            min_r2: Optional[float] = 0.6,
            filter_dataset: str = 'alternative',
            min_n_samples: Optional[int] = None,
            annotate_filter: bool = True,
            save_path: str | Path | None = None,
            dpi: int = 300,
            show_plot: bool = False,
            suptitle: Optional[str] = None,
            title: Optional[str] = None,
            suptitle_height_in: float = 0.5,
            debug: bool = False,
            **plot_kwargs,
    ):
        """
        Plot several targets on one figure.

        Under ``layout='panels'`` each target occupies a full row of three
        cohort panels, since the row is only readable when its three axes
        stay adjacent.  Under ``layout='overlay'`` each target is a single
        axis and the targets are tiled into ``n_cols`` columns.

        Targets are dropped for two reasons, both recorded rather than
        silent: a selected model that is not single-predictor, and a fitted
        model that fails the performance gate.  Call
        :meth:`~FunctionalFormCurveExtractor.print_skip_manifest` on
        ``self.extractor`` afterwards for the accounting a caption needs.

        Unknown keywords
        ----------------
        Every entry of ``plot_kwargs`` must be accepted by either the
        extractor or the renderer; anything else raises :exc:`TypeError`.
        A keyword aimed at the wrong method — or simply mistyped — would
        otherwise produce a figure that silently ignored it.

        :param targets: Targets to plot.  Defaults to every analysed target.
        :type targets: sequence[str], optional
        :param layout: ``'panels'`` or ``'overlay'``.
        :type layout: str
        :param n_cols: Columns in the tile grid; ``'overlay'`` only.
        :type n_cols: int
        :param figsize_per_panel: ``(width, height)`` in inches per axis.
            On a tiled figure this is the main legibility control: the
            whole grid scales with it, so text keeps its size relative to
            the page instead of shrinking as targets are added.
        :type figsize_per_panel: tuple[float, float]

        :param wspace: Horizontal gap between tiles as a fraction of the
            average tile width, applied after ``tight_layout``.  ``None``
            leaves the automatic spacing.
        :type wspace: float, optional

        :param hspace: Vertical gap between tile rows, on the same terms.
            Usually the setting to raise first under ``'overlay'``, where
            each tile carries a heading and a formula line above its axes.
        :type hspace: float, optional

        :param fonts: Point sizes for every text element: a
            :class:`FontSizes`, a dict of overrides, or ``None``.
        :type fonts: FontSizes or dict, optional

        :param font_scale: Multiplier applied to the whole font set.
        :type font_scale: float

        :param show_r2: Annotate each selected cohort's fitted R².  See
            :meth:`plot_functional_form` for the exact quantity.
        :type show_r2: bool

        :param r2_cohorts: Cohorts to annotate; ``'alternative'`` by
            default.
        :type r2_cohorts: str or sequence[str]

        :param r2_mode: ``'legend'``, ``'box'``, ``'both'`` or ``'none'``.
        :type r2_mode: str

        :param r2_decimals: Decimal places in the printed R².
        :type r2_decimals: int

        :param r2_loc: Corner for the box form of the annotation, or
            ``'auto'`` (default) to pick the corner clearest of the legend
            and the curves, per tile.
        :type r2_loc: str

        :param min_r2: Minimum R² for a target to be plotted.  ``None``
            (the default) plots every target regardless of fit quality.
            See :meth:`_apply_performance_filter` for why this gate needs a
            caption sentence.
        :type min_r2: float, optional
        :param filter_dataset: Which cohort's R² is tested: ``'alternative'``
            (default, the most restrictive), ``'conventional_full'``,
            ``'conventional_matched'``, ``'all'`` or ``'any'``.
        :type filter_dataset: str
        :param min_n_samples: Minimum alternative-cohort sample size for a
            target to be plotted.
        :type min_n_samples: int, optional
        :param annotate_filter: Append the gate and the shown/requested
            counts to the figure heading.  Leaving this on means a figure
            can never be read without its filtering being visible.
        :type annotate_filter: bool

        :param save_path: Destination file.
        :type save_path: str or pathlib.Path, optional
        :param dpi: Resolution used when saving.
        :type dpi: int
        :param show_plot: Call ``plt.show()`` at the end.
        :type show_plot: bool
        :param suptitle: Figure-level title above the target rows.
        :type suptitle: str, optional
        :param title: Alias for ``suptitle``, accepted so the parameter name
            matches :meth:`plot_functional_form`.  Supplying both raises
            :exc:`TypeError` rather than silently preferring one.
        :type title: str, optional
        :param suptitle_height_in: Vertical space reserved for the title, in
            **inches**.
        :type suptitle_height_in: float
        :param debug: Print per-target progress and layout decisions.
        :type debug: bool
        :param plot_kwargs: Any keyword accepted by
            :meth:`plot_functional_form` except the figure-management ones,
            applied to every target.

        :returns: The figure, or ``None`` when nothing could be plotted.
        :rtype: matplotlib.figure.Figure, optional

        :raises TypeError: If both ``title`` and ``suptitle`` are given, or
            if ``plot_kwargs`` contains a keyword neither the extractor nor
            the renderer accepts.
        """
        # ── Stage 0: resolve the title alias and validate keywords ───────
        if title is not None and suptitle is not None:
            raise TypeError(
                "Got both 'title' and 'suptitle'; they name the same "
                "figure-level heading. Pass only one."
            )

        heading = suptitle if suptitle is not None else title

        fonts = FontSizes.resolve(fonts, font_scale)

        unknown = set(plot_kwargs) - self._EXTRACT_KEYS - self._RENDER_KEYS

        if unknown:
            accepted = sorted(self._EXTRACT_KEYS | self._RENDER_KEYS)
            raise TypeError(
                f"Unexpected keyword argument(s) "
                f"{sorted(unknown)} for plot_functional_forms_all_targets. "
                f"Figure-level options are named parameters (layout, n_cols, "
                f"figsize_per_panel, wspace, hspace, fonts, font_scale, "
                f"show_r2, r2_cohorts, r2_mode, r2_decimals, r2_loc, "
                f"min_r2, filter_dataset, min_n_samples, "
                f"annotate_filter, save_path, dpi, show_plot, suptitle, "
                f"title, suptitle_height_in, debug); per-target options are "
                f"{accepted}."
            )

        results = self.analyzer.comparison_results

        if results is None or len(results) == 0:
            print("No results to plot. Run analyze_all_targets() first.")
            return None

        if targets is None:
            targets = list(results['target'])

        n_requested = len(targets)

        # ── Stage 0b: performance gate ───────────────────────────────────
        # Applied before curve construction, so no model is evaluated for a
        # target that will not be drawn.
        targets, filter_note = self._apply_performance_filter(
            targets=targets,
            min_r2=min_r2,
            filter_dataset=filter_dataset,
            min_n_samples=min_n_samples,
            debug=debug,
        )

        if not targets:
            print(f"⚠️  No targets meet the performance criteria "
                  f"(min_r2={min_r2}, dataset={filter_dataset!r}, "
                  f"min_n_samples={min_n_samples}).")
            return None

        extract_kwargs = {
            k: v for k, v in plot_kwargs.items() if k in self._EXTRACT_KEYS
        }
        render_kwargs = {
            k: v for k, v in plot_kwargs.items() if k in self._RENDER_KEYS
        }

        # Presentation settings are named parameters here rather than
        # members of _RENDER_KEYS, so they are forwarded explicitly. Fonts
        # are passed already resolved: resolving once means every tile in
        # the grid is guaranteed the same set.
        render_kwargs.update(
            fonts=fonts,
            show_r2=show_r2,
            r2_cohorts=r2_cohorts,
            r2_mode=r2_mode,
            r2_decimals=r2_decimals,
            r2_loc=r2_loc,
        )

        # ── Stage 1: build every surviving target's curves once ──────────
        curve_sets: list[TargetCurveSet] = []

        for target in targets:
            cs = self.extractor.build_curve_set(
                target,
                include_points=plot_kwargs.get('show_points', False),
                debug=debug,
                **extract_kwargs,
            )
            if cs is not None and cs.valid_curves:
                curve_sets.append(cs)
            elif debug and cs is not None:
                print(f"   ⚠️  skipping {target!r}: no evaluable curves")

        if not curve_sets:
            print("⚠️  No targets produced evaluable curves.")
            return None

        # ── Stage 2: grid geometry ───────────────────────────────────────
        # Axes cannot be moved between figures in matplotlib, so the shared
        # drawing routine is pointed at sub-axes created here rather than
        # each target being rendered to its own figure and transplanted.
        n_targets = len(curve_sets)

        if layout == 'panels':
            n_rows, n_panel_cols = n_targets, 3
        elif layout == 'overlay':
            n_panel_cols = min(n_cols, n_targets)
            n_rows = int(np.ceil(n_targets / n_panel_cols))
        else:
            raise ValueError(
                f"Unknown layout {layout!r}; expected 'panels' or 'overlay'."
            )

        # The heading is assembled before the figure so its line count is
        # known, and the reserved band grows with it.
        heading_lines: list[str] = []

        if heading:
            heading_lines.append(heading)

        if annotate_filter:
            if filter_note:
                heading_lines.append(filter_note)
            # elif n_targets < n_requested:
            #     # Dropped by the single-predictor gate rather than by R².
            #     heading_lines.append(
            #         f'{n_targets}/{n_requested} targets shown')

        heading_text = '\n'.join(heading_lines)

        reserved_in = (
            suptitle_height_in * len(heading_lines) if heading_lines else 0.0
        )

        fig_width = figsize_per_panel[0] * n_panel_cols
        fig_height = figsize_per_panel[1] * n_rows + reserved_in

        if debug:
            print(f"\n🔍 DEBUG: plot_functional_forms_all_targets — "
                  f"{n_requested} requested, {n_targets} drawn, "
                  f"{len(self.extractor.skipped_targets)} skipped, "
                  f"grid {n_rows}×{n_panel_cols}, "
                  f"figure {fig_width:.1f}×{fig_height:.1f} in")

        fig, axes = plt.subplots(
            n_rows, n_panel_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        # ── Stage 3: draw each target ────────────────────────────────────
        # Rescaling is applied per target here rather than inside the
        # renderer, so the limits the renderer resolves are already in
        # display units.
        display_scale_map = render_kwargs.get('display_scale_map')

        for row_idx, cs in enumerate(curve_sets):
            if layout == 'panels':
                row_axes = list(axes[row_idx])
            else:
                r, c = divmod(row_idx, n_panel_cols)
                row_axes = [axes[r][c]]

            self._render_curve_set_onto(
                axes_list=row_axes,
                curve_set=self._apply_display_scaling(
                    cs, display_scale_map, debug=debug),
                layout=layout,
                debug=debug,
                **render_kwargs,
            )

        # Blank any unused tiles. Only reachable under 'overlay', where the
        # grid is n_cols-wide and the target count need not fill it.
        if layout == 'overlay':
            for flat_idx in range(n_targets, n_rows * n_panel_cols):
                r, c = divmod(flat_idx, n_panel_cols)
                axes[r][c].set_visible(False)

        # ── Stage 4: heading ─────────────────────────────────────────────
        # Under 'overlay' the top row of tiles also carries a formula line
        # drawn just above its axes, which tight_layout does not account
        # for, so the reserved band has to clear both.
        if heading_text:
            reserved = reserved_in / fig_height
            fig.suptitle(
                heading_text, fontsize=fonts.figure_title, fontweight='bold',
                y=1.0 - reserved * 0.20,
            )
            rect = (0.0, 0.0, 1.0, 1.0 - reserved)
        else:
            rect = (0.0, 0.0, 1.0, 1.0)

        fig.tight_layout(rect=rect)

        # After tight_layout, which would otherwise overwrite these.
        if wspace is not None or hspace is not None:
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

        if debug and heading_text:
            print(f"   suptitle   : {heading_text!r} "
                  f"(reserved {reserved_in:.2f} in)")

        self._finalise(fig, save_path, dpi, show_plot, debug)

        return fig

    # -------------------------------------------------------------------
    # Rendering helpers
    # -------------------------------------------------------------------
    def _render_curve_set_onto(
            self,
            axes_list: list,
            curve_set: TargetCurveSet,
            layout: str,
            *,
            show_points: bool = False,
            reference_dataset: str = 'conventional_full',
            xlim_mode: str = 'reference',
            ylim_mode: str = 'reference',
            ylim_padding: float = 0.05,
            mark_support: bool = True,
            linewidth: float = 2.0,
            point_size: float = 12.0,
            point_alpha: float = 0.30,
            variable_name_map: Optional[dict[str, str]] = None,
            formula_name_map: Optional[dict[str, str]] = None,
            show_formula: bool = True,
            formula_fontsize: Optional[float] = None,
            display_scale_map: Optional[dict[str, Any]] = None,
            max_xticks: Optional[int] = None,
            max_yticks: Optional[int] = None,
            xtick_rotation: float = 0.0,
            fonts: 'FontSizes | dict[str, float] | None' = None,
            font_scale: float = 1.0,
            show_r2: bool = True,
            r2_cohorts: 'str | Sequence[str]' = 'alternative',
            r2_mode: str = 'legend',
            r2_decimals: int = 2,
            r2_loc: str = 'auto',
            legend_loc: str = 'best',
            share_axes: bool = True,
            debug: bool = False,
            **_ignored,
    ) -> None:
        """
        Draw one curve set onto pre-existing axes.

        Factored out so the single-target and multi-target entry points
        share one drawing path; only figure creation differs between them.

        Tile headings
        -------------
        Under ``'overlay'`` each tile carries its own two-line heading: the
        target name in bold, and the β-notation formula in italic beneath
        it.  On a sheet of several targets there is no figure-level title
        that could carry the formula, and a reader cannot interpret a gap
        between two curves without knowing what functional form produced
        them — a power law and a fractional polynomial diverge for quite
        different reasons.  The formula is drawn as a separate text object
        rather than as a second line of ``set_title`` so it can be italic
        and smaller while the target name stays bold.

        Under ``'panels'`` the three cohort axes carry cohort names instead,
        and the formula is left to the figure-level title block.

        :param axes_list: Target axes — three under ``'panels'``, one under
            ``'overlay'``.
        :type axes_list: list[matplotlib.axes.Axes]
        :param curve_set: Curves to draw.
        :type curve_set: TargetCurveSet
        :param layout: ``'panels'`` or ``'overlay'``.
        :type layout: str
        :param show_points: Draw observations.
        :type show_points: bool
        :param reference_dataset: Cohort defining the axis frame.
        :type reference_dataset: str
        :param xlim_mode: ``'reference'`` or ``'grid'``.
        :type xlim_mode: str
        :param ylim_mode: ``'reference'``, ``'curves'`` or ``'data'``.
        :type ylim_mode: str
        :param ylim_padding: Fractional y padding.
        :type ylim_padding: float
        :param mark_support: Fade out-of-support curve segments.
        :type mark_support: bool
        :param linewidth: Curve line width.
        :type linewidth: float
        :param point_size: Scatter marker area.
        :type point_size: float
        :param point_alpha: Scatter marker opacity.
        :type point_alpha: float
        :param variable_name_map: Internal → display names for axis labels
            and the tile heading.
        :type variable_name_map: dict[str, str], optional
        :param formula_name_map: Internal → display names inside the
            formula line.  Falls back to ``variable_name_map``.
        :type formula_name_map: dict[str, str], optional
        :param show_formula: Draw the formula beneath the tile heading.
            ``'overlay'`` only.
        :type show_formula: bool
        :param fonts: Resolved font sizes, or a spec to resolve.  The
            multi-target entry point passes an already-resolved set so
            every tile in a grid shares it.
        :type fonts: FontSizes or dict, optional

        :param font_scale: Multiplier applied when ``fonts`` still needs
            resolving.  Ignored for an already-scaled set.
        :type font_scale: float

        :param show_r2: Annotate each selected cohort's fitted R².
        :type show_r2: bool

        :param r2_cohorts: Cohorts to annotate.
        :type r2_cohorts: str or sequence[str]

        :param r2_mode: ``'legend'``, ``'box'``, ``'both'`` or ``'none'``.
        :type r2_mode: str

        :param r2_decimals: Decimal places in the printed R².
        :type r2_decimals: int

        :param r2_loc: Corner for the box form of the annotation.
        :type r2_loc: str

        :param formula_fontsize: Point size of the formula line, overriding
            :attr:`FontSizes.formula` when given.  The tile
            heading's padding is derived from it, so a larger formula opens
            up its own space rather than colliding with the heading.
        :type formula_fontsize: float
        :param legend_loc: Legend location for overlay mode.
        :type legend_loc: str
        :param share_axes: Suppress repeated y-labels across panels.
        :type share_axes: bool
        :param debug: Print per-cohort drawing details and the heading text.
        :type debug: bool

        :param display_scale_map: Mapping from variable name to a display
            divisor, or to a ``(divisor, label)`` pair.  Values are divided
            before drawing and the axis label gains a ``×10ⁿ`` suffix, or
            the supplied label replaces it outright.  Affects the figure
            only; the underlying curve arrays are untouched.
            Example: ``{'DWT': 1e3, 'MCR (KW)': (1e3, 'MCR (MW)')}``.
        :type display_scale_map: dict, optional

        :param max_xticks: Upper bound on the number of x-axis ticks.
        :type max_xticks: int, optional

        :param max_yticks: Upper bound on the number of y-axis ticks.
        :type max_yticks: int, optional

        :param xtick_rotation: Rotation in degrees for x tick labels.
        :type xtick_rotation: float

        :returns: ``None``.  Mutates the supplied axes.
        """
        fonts = (fonts if isinstance(fonts, FontSizes)
                 else FontSizes.resolve(fonts, font_scale))

        if formula_fontsize is not None:
            fonts = replace(fonts, formula=formula_fontsize)

        r2_targets = self._resolve_r2_cohorts(show_r2, r2_cohorts, r2_mode)
        r2_in_labels = r2_mode in ('legend', 'both')
        r2_in_box = r2_mode in ('box', 'both')

        xlim, ylim = self._resolve_limits(
            curve_set=curve_set,
            reference_dataset=reference_dataset,
            xlim_mode=xlim_mode,
            ylim_mode=ylim_mode,
            ylim_padding=ylim_padding,
            include_points=show_points,
            debug=debug,
        )

        for idx, curve in enumerate(curve_set.curves):
            ax = axes_list[idx] if layout == 'panels' else axes_list[0]

            caption = self._cohort_caption(
                curve, r2_targets, r2_decimals, include_r2=r2_in_labels,
            )

            self._draw_cohort(
                ax=ax, curve=curve, show_points=show_points,
                mark_support=mark_support, linewidth=linewidth,
                point_size=point_size, point_alpha=point_alpha,
                label=(f'{curve.label} ({caption})'
                       if layout == 'overlay' else None),
                fonts=fonts, debug=debug,
            )

            if layout == 'panels':
                ax.set_title(f'{curve.label}  ({caption})',
                             fontsize=fonts.axis_title, fontweight='bold')

                if r2_in_box:
                    self._add_r2_box(
                        ax=ax,
                        curve_set=replace(curve_set, curves=[curve]),
                        r2_targets=r2_targets & {curve.cohort},
                        r2_decimals=r2_decimals,
                        fontsize=fonts.annotation,
                        loc=r2_loc,
                        debug=debug,
                    )

        name_map = variable_name_map or {}

        x_label = self._axis_label(
            curve_set.predictor, variable_name_map, display_scale_map)
        y_label = self._axis_label(
            curve_set.target, variable_name_map, display_scale_map)

        for idx, ax in enumerate(axes_list):
            ax.set_xlabel(x_label, fontsize=fonts.axis_label)
            if idx == 0 or not share_axes:
                ax.set_ylabel(y_label, fontsize=fonts.axis_label)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            self._apply_tick_policy(
                ax, max_xticks, max_yticks, xtick_rotation,
                tick_labelsize=fonts.tick_label)

        if layout == 'overlay':
            ax = axes_list[0]

            formula = (
                self._format_formula(
                    curve_set, formula_name_map or variable_name_map,
                )
                if show_formula else ''
            )

            # Reserve vertical space above the axes for the formula line,
            # then place the bold heading above it. Padding is in points and
            # scales with the formula's own size so the two never collide.
            heading_pad = (fonts.formula * 1.6) if formula else 4.0
            ax.set_title(y_label, fontsize=fonts.axis_title,
                         fontweight='bold', pad=heading_pad)

            if formula:
                ax.text(
                    0.5, 1.01, formula,
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    fontsize=fonts.formula, fontstyle='italic',
                    color='#333333',
                )

            legend = ax.legend(loc=legend_loc, fontsize=fonts.legend,
                               framealpha=0.9)

            if r2_in_box:
                self._add_r2_box(
                    ax=ax, curve_set=curve_set, r2_targets=r2_targets,
                    r2_decimals=r2_decimals, fontsize=fonts.annotation,
                    loc=r2_loc, legend=legend, debug=debug,
                )

            if debug:
                print(f"   tile heading: {y_label}"
                      + (f" / {formula}" if formula else ""))

    def _draw_cohort(
            self,
            ax,
            curve: CohortCurve,
            show_points: bool,
            mark_support: bool,
            linewidth: float,
            point_size: float,
            point_alpha: float,
            label: Optional[str] = None,
            fonts: Optional[FontSizes] = None,
            debug: bool = False,
    ) -> None:
        """
        Draw a single cohort's scatter and curve onto one axis.

        The curve is split at its cohort's support boundary: solid within
        the range the cohort observed, dotted and faded beyond it.  The two
        segments are drawn from one array with ``NaN`` masking rather than
        by slicing, so a curve with interior undefined regions still breaks
        in the right places.

        :param ax: Axis to draw on.
        :type ax: matplotlib.axes.Axes
        :param curve: Cohort curve.
        :type curve: CohortCurve
        :param show_points: Draw the observations behind the curve.
        :type show_points: bool
        :param mark_support: Fade the extrapolated segment.
        :type mark_support: bool
        :param linewidth: Curve line width.
        :type linewidth: float
        :param point_size: Scatter marker area.
        :type point_size: float
        :param point_alpha: Scatter marker opacity.
        :type point_alpha: float
        :param label: Legend label for this cohort's line, or ``None`` to
            keep it out of the legend.  Composed by the caller rather than
            here, because what belongs in it — sample size, R², both —
            is a figure-level decision that this method has no view of.
        :type label: str, optional
        :param fonts: Resolved font sizes; only :attr:`FontSizes.annotation`
            is used, for the "not evaluable" placeholder.  Defaults apply
            when ``None``.
        :type fonts: FontSizes, optional
        :param debug: Print what was drawn.
        :type debug: bool

        :returns: ``None``.
        """
        fonts = fonts or FontSizes()

        style = self.cohort_styles.get(
            curve.cohort, {'color': 'grey', 'linestyle': '-', 'marker': 'o'},
        )

        if not curve.valid:
            ax.text(0.5, 0.5, f'{curve.label}\nnot evaluable',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=fonts.annotation, color='grey')
            if debug:
                print(f"   [{curve.cohort}] skipped: {curve.message}")
            return

        if show_points and curve.x_points is not None:
            ax.scatter(
                curve.x_points, curve.y_points,
                s=point_size, alpha=point_alpha,
                color=style['color'], marker=style['marker'],
                edgecolors='none', zorder=1,
            )

        inside = self._support_mask(curve, mark_support)

        y_inside = np.where(inside, curve.y_curve, np.nan)
        y_outside = np.where(inside, np.nan, curve.y_curve)

        if mark_support and np.any(~inside):
            ax.plot(
                curve.x_grid, y_outside,
                color=style['color'], linestyle=':',
                linewidth=linewidth * 0.9, alpha=0.45, zorder=2,
            )

        ax.plot(
            curve.x_grid, y_inside,
            color=style['color'], linestyle=style['linestyle'],
            linewidth=linewidth, label=label, zorder=3,
        )

        if debug:
            n_out = int(np.sum(~inside))
            print(f"   [{curve.cohort}] drawn: "
                  f"{len(curve.x_grid) - n_out} in-support points, "
                  f"{n_out} extrapolated, "
                  f"{curve.n_undefined} undefined")

    def _resolve_r2_cohorts(
            self,
            show_r2: bool,
            r2_cohorts: 'str | Sequence[str]',
            r2_mode: str,
    ) -> frozenset[str]:
        """
        Decide which cohorts get their R² printed.

        Defaults to the alternative-fuel cohort alone.  That is the number
        a transferability figure is being read for: the conventional fits
        are the reference structure, and their R² is a property of the
        baseline rather than of the comparison.  Printing all three is
        available, but it triples the text on a tile that already carries a
        heading, a formula and a three-entry legend.

        :param show_r2: Master switch.  ``False`` disables the annotation
            regardless of the other two arguments.
        :type show_r2: bool
        :param r2_cohorts: ``'alternative'``, ``'all'``, a single cohort
            key, or an explicit sequence of keys.
        :type r2_cohorts: str or sequence[str]
        :param r2_mode: ``'legend'``, ``'box'``, ``'both'`` or ``'none'``.
        :type r2_mode: str

        :returns: The cohort keys to annotate; empty when disabled.
        :rtype: frozenset[str]
        :raises ValueError: On an unknown cohort key or mode.
        """
        valid_modes = {'legend', 'box', 'both', 'none'}

        if r2_mode not in valid_modes:
            raise ValueError(
                f"Unknown r2_mode {r2_mode!r}; expected one of "
                f"{sorted(valid_modes)}."
            )

        if not show_r2 or r2_mode == 'none':
            return frozenset()

        known = set(FunctionalFormCurveExtractor.COHORT_LABELS)

        if isinstance(r2_cohorts, str):
            requested = known if r2_cohorts == 'all' else {r2_cohorts}
        else:
            requested = set(r2_cohorts)

        unknown = requested - known

        if unknown:
            raise ValueError(
                f"Unknown cohort key(s) {sorted(unknown)} in r2_cohorts. "
                f"Expected 'all' or a subset of {sorted(known)}."
            )

        return frozenset(requested)

    def _format_r2(
            self,
            value: Optional[float],
            r2_decimals: int = 2,
    ) -> Optional[str]:
        """
        Render one R² value, or nothing when there is none to render.

        Returning ``None`` for a missing or non-finite value keeps the
        caller from printing ``R² = nan`` onto a figure, where a reader
        would take it for a computed result rather than an absence.

        :param value: The stored R², possibly ``None`` or ``NaN``.
        :type value: float, optional
        :param r2_decimals: Decimal places.
        :type r2_decimals: int
        :returns: Formatted text, or ``None``.
        :rtype: str, optional
        """
        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        return f'R² = {value:.{r2_decimals}f}'

    def _cohort_caption(
            self,
            curve: CohortCurve,
            r2_targets: frozenset[str],
            r2_decimals: int,
            include_r2: bool,
    ) -> str:
        """
        Build the parenthesised detail that follows a cohort's name.

        Sample size and fit quality belong together: an R² of 0.8 on
        twenty ships and on two thousand are different claims, and putting
        them in one bracket stops a reader taking the second number
        without the first.

        :param curve: The cohort being labelled.
        :type curve: CohortCurve
        :param r2_targets: Cohorts whose R² is to be shown.
        :type r2_targets: frozenset[str]
        :param r2_decimals: Decimal places for the R².
        :type r2_decimals: int
        :param include_r2: ``False`` in ``'box'`` mode, where the R² is
            drawn separately and would otherwise appear twice.
        :type include_r2: bool

        :returns: Caption text without surrounding brackets.
        :rtype: str
        """
        parts = [f'n = {curve.n_fit}']

        if include_r2 and curve.cohort in r2_targets:
            r2_text = self._format_r2(curve.r2, r2_decimals)
            if r2_text:
                parts.append(r2_text)

        return ', '.join(parts)

    @staticmethod
    def _get_renderer(fig):
        """
        Obtain a renderer for measuring text, whatever the backend.

        Text extents are only knowable once something can measure them, and
        vector backends expose no renderer until a draw has happened.  The
        Agg fallback measures against a raster canvas instead; the metrics
        are the same, since both use the same font machinery.  The original
        canvas is restored so that a later ``savefig`` still writes the
        format the caller asked for.

        :param fig: Figure whose renderer is wanted.
        :type fig: matplotlib.figure.Figure
        :returns: A renderer, or ``None`` when none can be obtained.
        """
        try:
            return fig.canvas.get_renderer()
        except AttributeError:
            pass

        try:
            fig.canvas.draw()
            return fig.canvas.get_renderer()
        except Exception:
            pass

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            original = fig.canvas
            renderer = FigureCanvasAgg(fig).get_renderer()
            fig.canvas = original
            return renderer
        except Exception:
            return None

    @staticmethod
    def _bbox_to_axes(ax, bbox) -> tuple[float, float, float, float]:
        """
        Convert a display-space bounding box into axes fractions.

        Axes fractions are the right frame for this comparison because the
        anchors are expressed in them, and because they survive the
        ``tight_layout`` call that follows: resizing the axes moves the
        legend and the box together.

        :param ax: Axis defining the coordinate system.
        :type ax: matplotlib.axes.Axes
        :param bbox: Bounding box in display coordinates.
        :type bbox: matplotlib.transforms.Bbox
        :returns: ``(x0, y0, x1, y1)`` in axes fractions.
        :rtype: tuple[float, float, float, float]
        """
        inverse = ax.transAxes.inverted()
        (x0, y0), (x1, y1) = inverse.transform(
            [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])

        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def _rect_overlap(a, b) -> float:
        """
        Area of the intersection of two axes-fraction rectangles.

        :param a: ``(x0, y0, x1, y1)``.
        :type a: tuple
        :param b: ``(x0, y0, x1, y1)``.
        :type b: tuple
        :returns: Overlapping area; ``0.0`` when disjoint.
        :rtype: float
        """
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])

        return dx * dy if dx > 0 and dy > 0 else 0.0

    def _data_fraction_in(self, ax, rect) -> float:
        """
        Share of the plotted points falling inside a rectangle.

        Used as the tie-breaker between corners the legend leaves free, on
        the same principle as matplotlib's own ``loc='best'``: of the
        corners that are available, prefer the emptiest.

        Lines and scatter collections are both counted, and the curves are
        subsampled — the answer only has to rank four candidates, not be
        exact.

        :param ax: Axis whose artists are inspected.
        :type ax: matplotlib.axes.Axes
        :param rect: ``(x0, y0, x1, y1)`` in axes fractions.
        :type rect: tuple
        :returns: Fraction in ``[0, 1]``; ``0.0`` when the axis is empty.
        :rtype: float
        """
        chunks: list[np.ndarray] = []

        for line in ax.lines:
            data = line.get_xydata()
            if data is not None and len(data):
                chunks.append(np.asarray(data)[::3])

        for collection in ax.collections:
            try:
                offsets = np.asarray(collection.get_offsets())
            except Exception:
                continue
            if offsets.ndim == 2 and len(offsets):
                chunks.append(offsets[::5])

        if not chunks:
            return 0.0

        points = np.vstack(chunks).astype(float)
        points = points[np.isfinite(points).all(axis=1)]

        if not len(points):
            return 0.0

        # Straight through the data transform rather than a manual
        # limit-based rescale, so log axes are handled correctly.
        display = ax.transData.transform(points)
        fractions = ax.transAxes.inverted().transform(display)

        inside = (
            (fractions[:, 0] >= rect[0]) & (fractions[:, 0] <= rect[2]) &
            (fractions[:, 1] >= rect[1]) & (fractions[:, 1] <= rect[3])
        )

        return float(np.mean(inside))

    def _resolve_box_anchor(
            self,
            ax,
            text,
            legend=None,
            debug: bool = False,
    ) -> str:
        """
        Choose the corner for an already-drawn annotation box.

        The box is placed provisionally, measured, and then moved.  Sizing
        it by measurement rather than by estimate is what makes the overlap
        test meaningful: a three-cohort box is roughly three times the
        height of a one-cohort box, and a guess would have to be
        pessimistic enough to reject corners that are in fact free.

        Each of the four corners is scored as
        ``penalty × legend_overlap + data_fraction``, both normalised by the
        box's own area, and the lowest wins.  Ties fall to the order of
        :attr:`_R2_BOX_ANCHORS`, so a figure with an empty lower right and
        no legend keeps the default placement.

        A legend at ``loc='best'`` has no position until the figure is
        drawn, so one draw is forced in that case only.

        :param ax: Axis holding the annotation.
        :type ax: matplotlib.axes.Axes
        :param text: The already-created annotation, used for its extent.
        :type text: matplotlib.text.Text
        :param legend: Legend to keep clear of, or ``None`` when the axis
            has none — as under ``layout='panels'``, where the corner is
            then chosen on data occupancy alone.
        :type legend: matplotlib.legend.Legend, optional
        :param debug: Print the score of every candidate corner.
        :type debug: bool

        :returns: A key of :attr:`_R2_BOX_ANCHORS`.  Falls back to
            ``'lower right'`` when nothing can be measured, since a
            mispositioned annotation is not worth failing a figure over.
        :rtype: str
        """
        fig = ax.figure

        # loc code 0 is 'best', which is resolved during the draw.
        if legend is not None and getattr(legend, '_loc', None) == 0:
            try:
                fig.canvas.draw()
            except Exception:
                pass

        renderer = self._get_renderer(fig)

        if renderer is None:
            if debug:
                print("   R² box: no renderer; keeping 'lower right'")
            return 'lower right'

        try:
            tx0, ty0, tx1, ty1 = self._bbox_to_axes(
                ax, text.get_window_extent(renderer))
        except Exception:
            return 'lower right'

        # get_window_extent measures the glyphs, not the rounded frame
        # around them, so the padding is added back in axes fractions.
        axes_box = ax.get_window_extent(renderer)
        pad_points = 0.35 * text.get_fontsize() * 2.0
        pad_display = pad_points * fig.dpi / 72.0

        pad_x = pad_display / max(axes_box.width, 1e-9)
        pad_y = pad_display / max(axes_box.height, 1e-9)

        width = (tx1 - tx0) + pad_x
        height = (ty1 - ty0) + pad_y
        area = max(width * height, 1e-9)

        legend_rect = None

        if legend is not None:
            try:
                legend_rect = self._bbox_to_axes(
                    ax, legend.get_window_extent(renderer))
            except Exception:
                legend_rect = None

        best_key, best_score = 'lower right', None

        for key, (ax_x, ax_y, ha, va) in self._R2_BOX_ANCHORS.items():
            x0 = ax_x - width if ha == 'right' else ax_x
            y0 = ax_y - height if va == 'top' else ax_y
            rect = (x0, y0, x0 + width, y0 + height)

            overlap = (0.0 if legend_rect is None
                       else self._rect_overlap(rect, legend_rect) / area)
            occupancy = self._data_fraction_in(ax, rect)
            score = self._R2_LEGEND_PENALTY * overlap + occupancy

            if debug:
                print(f"     {key:<12s} legend={overlap:.2f} "
                      f"data={occupancy:.2f} score={score:.3f}")

            if best_score is None or score < best_score - 1e-9:
                best_key, best_score = key, score

        return best_key

    def _add_r2_box(
            self,
            ax,
            curve_set: TargetCurveSet,
            r2_targets: frozenset[str],
            r2_decimals: int,
            fontsize: float,
            loc: str = 'auto',
            legend=None,
            debug: bool = False,
    ) -> None:
        """
        Draw the R² values as a standalone box inside the axes.

        The alternative to appending them to the legend.  Preferred when
        the legend is already crowded, or when the figure is being read for
        fit quality rather than for cohort identity — a boxed number is
        found faster than one buried at the end of a legend entry.

        Nothing is drawn when no selected cohort has a usable value, so a
        target the analyser could not score leaves no empty frame behind.

        :param ax: Axis to annotate.
        :type ax: matplotlib.axes.Axes
        :param curve_set: Curves on that axis.
        :type curve_set: TargetCurveSet
        :param r2_targets: Cohorts whose R² is to be shown.
        :type r2_targets: frozenset[str]
        :param r2_decimals: Decimal places.
        :type r2_decimals: int
        :param fontsize: Point size of the box text.
        :type fontsize: float
        :param loc: One of the keys of :attr:`_R2_BOX_ANCHORS`, or
            ``'auto'`` to choose a corner the legend is not using.
        :type loc: str
        :param legend: Legend to keep clear of under ``loc='auto'``.
            Ignored for a fixed corner.
        :type legend: matplotlib.legend.Legend, optional
        :param debug: Print the assembled lines and the chosen corner.
        :type debug: bool

        :returns: ``None``.
        :raises ValueError: On an unknown ``loc``.
        """
        if not r2_targets:
            return

        if loc != 'auto' and loc not in self._R2_BOX_ANCHORS:
            raise ValueError(
                f"Unknown r2_loc {loc!r}; expected 'auto' or one of "
                f"{sorted(self._R2_BOX_ANCHORS)}."
            )

        lines: list[str] = []

        for curve in curve_set.curves:
            if curve.cohort not in r2_targets:
                continue

            r2_text = self._format_r2(curve.r2, r2_decimals)

            if r2_text is None:
                continue

            # The cohort name is only worth the width when more than one
            # value is shown; with a single value it is already implied by
            # the figure's subject.
            lines.append(
                f'{curve.label}: {r2_text}' if len(r2_targets) > 1
                else r2_text
            )

        if not lines:
            return

        # Placed provisionally, then measured and moved if 'auto'.  The
        # text has to exist before its size is knowable, and its size is
        # what decides whether a corner is big enough to be free.
        resolved = 'lower right' if loc == 'auto' else loc
        x, y, ha, va = self._R2_BOX_ANCHORS[resolved]

        text = ax.text(
            x, y, '\n'.join(lines),
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=fontsize, color='#222222',
            bbox={
                'boxstyle': 'round,pad=0.35',
                'facecolor': 'white',
                'edgecolor': '#CCCCCC',
                'linewidth': 0.6,
                'alpha': 0.88,
            },
            zorder=5,
        )

        if loc == 'auto':
            resolved = self._resolve_box_anchor(
                ax, text, legend=legend, debug=debug)
            x, y, ha, va = self._R2_BOX_ANCHORS[resolved]
            text.set_position((x, y))
            text.set_ha(ha)
            text.set_va(va)

        if debug:
            print(f"   R² box ({resolved}"
                  f"{', auto' if loc == 'auto' else ''}): "
                  f"{' | '.join(lines)}")

    def _support_mask(
            self,
            curve: CohortCurve,
            mark_support: bool,
    ) -> np.ndarray:
        """
        Boolean mask of grid points within a cohort's observed range.

        The mask is dilated by one index on each side so the solid and
        dotted segments meet rather than leaving a one-pixel gap at the
        support boundary.

        :param curve: Cohort curve carrying ``support``.
        :type curve: CohortCurve
        :param mark_support: When ``False``, every point counts as inside.
        :type mark_support: bool
        :returns: Mask aligned with ``curve.x_grid``.
        :rtype: numpy.ndarray
        """
        if not mark_support or curve.support is None:
            return np.ones_like(curve.x_grid, dtype=bool)

        low, high = curve.support
        inside = (curve.x_grid >= low) & (curve.x_grid <= high)

        dilated = inside.copy()
        dilated[:-1] |= inside[1:]
        dilated[1:] |= inside[:-1]

        return dilated

    # -------------------------------------------------------------------
    # Limits and titles
    # -------------------------------------------------------------------

    def _resolve_limits(
            self,
            curve_set: TargetCurveSet,
            reference_dataset: str,
            xlim_mode: str,
            ylim_mode: str,
            ylim_padding: float,
            include_points: bool,
            debug: bool = False,
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        """
        Resolve the shared axis frame.

        Anchoring the frame to one cohort is what makes the three panels
        comparable.  Under ``ylim_mode='reference'`` the frame is set by the
        reference cohort alone, so a curve that departs far from
        conventional practice runs off the top of the panel — which is
        itself the finding.  ``'curves'`` widens the frame until all three
        fit, trading that signal for legibility.

        :param curve_set: Curves being plotted.
        :type curve_set: TargetCurveSet
        :param reference_dataset: Cohort defining the frame.
        :type reference_dataset: str
        :param xlim_mode: ``'reference'`` or ``'grid'``.
        :type xlim_mode: str
        :param ylim_mode: ``'reference'``, ``'curves'`` or ``'data'``.
        :type ylim_mode: str
        :param ylim_padding: Fractional padding on the y-range.
        :type ylim_padding: float
        :param include_points: Whether observations will be drawn, which
            matters only under ``ylim_mode='data'``.
        :type include_points: bool
        :param debug: Print the resolved limits.
        :type debug: bool

        :returns: ``(xlim, ylim)``; either may be ``None`` when it could
            not be determined.
        :rtype: tuple
        :raises ValueError: On an unrecognised mode.
        """
        reference = curve_set.get(reference_dataset)

        if reference is None or not reference.valid:
            reference = curve_set.valid_curves[0]
            if debug:
                print(f"   ⚠️  reference cohort {reference_dataset!r} "
                      f"unavailable; framing on {reference.cohort!r}")

        # ── x limits ─────────────────────────────────────────────────────
        if xlim_mode == 'grid':
            xlim = (float(np.min(reference.x_grid)),
                    float(np.max(reference.x_grid)))
        elif xlim_mode == 'reference':
            xlim = reference.support or (
                float(np.min(reference.x_grid)), float(np.max(reference.x_grid))
            )
        else:
            raise ValueError(
                f"Unknown xlim_mode {xlim_mode!r}; expected 'reference' or 'grid'."
            )

        # ── y limits ─────────────────────────────────────────────────────
        pools: list[np.ndarray] = []

        if ylim_mode == 'reference':
            pools.append(reference.y_curve)
            if include_points and reference.y_points is not None:
                pools.append(reference.y_points)
        elif ylim_mode == 'curves':
            pools.extend(c.y_curve for c in curve_set.valid_curves)
        elif ylim_mode == 'data':
            pools.extend(c.y_curve for c in curve_set.valid_curves)
            pools.extend(
                c.y_points for c in curve_set.valid_curves
                if c.y_points is not None
            )
        else:
            raise ValueError(
                f"Unknown ylim_mode {ylim_mode!r}; expected 'reference', "
                f"'curves' or 'data'."
            )

        stacked = np.concatenate([np.asarray(p, dtype=float).ravel() for p in pools])
        finite = stacked[np.isfinite(stacked)]

        if finite.size == 0:
            ylim = None
        else:
            low, high = float(finite.min()), float(finite.max())
            span = high - low or max(abs(high), 1.0)
            ylim = (low - span * ylim_padding, high + span * ylim_padding)

        if debug:
            print(f"   xlim       : {xlim} (mode={xlim_mode})")
            print(f"   ylim       : {ylim} (mode={ylim_mode})")

        return xlim, ylim

    def _add_title_block(
            self,
            fig,
            curve_set: TargetCurveSet,
            title: Optional[str],
            variable_name_map: Optional[dict[str, str]],
            formula_name_map: Optional[dict[str, str]],
            show_formula: bool,
            fonts: Optional[FontSizes] = None,
            debug: bool = False,
    ) -> None:
        """
        Write the figure title and formula line.

        With a single predictor no slice note is needed: the curve *is* the
        model, so the formula alone tells the reader everything about how
        the line was produced.

        :param fig: Figure to annotate.
        :type fig: matplotlib.figure.Figure
        :param curve_set: Curves being plotted.
        :type curve_set: TargetCurveSet
        :param title: Explicit title replacing the generated one.
        :type title: str, optional
        :param variable_name_map: Internal → display names for the title.
        :type variable_name_map: dict[str, str], optional
        :param formula_name_map: Internal → display names for the formula.
        :type formula_name_map: dict[str, str], optional
        :param show_formula: Include the formula line.
        :type show_formula: bool
        :param fonts: Resolved font sizes; the heading uses
            :attr:`FontSizes.figure_title` and the formula line
            :attr:`FontSizes.formula`.
        :type fonts: FontSizes, optional
        :param debug: Print the assembled lines.
        :type debug: bool

        :returns: ``None``.
        """
        fonts = fonts or FontSizes()
        name_map = variable_name_map or {}
        target_display = name_map.get(curve_set.target, curve_set.target)
        predictor_display = name_map.get(curve_set.predictor,
                                         curve_set.predictor)

        lines: list[tuple[str, str]] = []

        if title:
            lines.append((title, 'bold'))
        else:
            lines.append((
                f'{target_display} vs {predictor_display} — '
                f'{curve_set.model_type} fitted on three cohorts',
                'bold',
            ))

        if show_formula:
            formula = self._format_formula(curve_set, formula_name_map)
            if formula:
                lines.append((formula, 'italic'))

        # Line spacing is driven by the point size rather than fixed, so
        # that scaling the fonts up does not overlap the two lines.
        y = 0.985
        for text, style in lines:
            size = fonts.figure_title if style == 'bold' else fonts.formula
            fig.text(
                0.5, y, text,
                ha='center', va='top',
                fontsize=size,
                fontweight='bold' if style == 'bold' else 'normal',
                fontstyle='italic' if style == 'italic' else 'normal',
            )
            y -= 0.0032 * size + 0.004

        if debug:
            for text, style in lines:
                print(f"   title[{style}]: {text}")

    def _format_formula(
            self,
            curve_set: TargetCurveSet,
            formula_name_map: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Render the shared β-notation formula with display names substituted.

        Both the figure-level title block and the per-tile overlay titles
        need this string, so the substitution lives here rather than being
        written out at each call site.

        Replacement runs longest-key-first so that nested names are not
        partially rewritten by a shorter key — without it, a map containing
        both ``'LOA (M)'`` and ``'LOA'`` would corrupt the first.

        :param curve_set: Curve set carrying the raw formula string.
        :type curve_set: TargetCurveSet
        :param formula_name_map: Internal → display names to substitute.
        :type formula_name_map: dict[str, str], optional

        :returns: Formula with display names, or ``''`` when the descriptor
            produced no formula.
        :rtype: str
        """
        formula = curve_set.formula

        if not formula:
            return ''

        for internal in sorted(formula_name_map or {}, key=len, reverse=True):
            formula = formula.replace(internal, formula_name_map[internal])

        return formula

    def _resolve_display_scale(
            self,
            name: str,
            display_scale_map: Optional[dict[str, Any]] = None,
    ) -> tuple[float, Optional[str]]:
        """
        Resolve the display divisor and optional label override for a variable.

        Two spec forms are accepted, so the common case stays terse while
        full control remains available:

        ``{'DWT': 1e3}``
            Divide plotted values by 1000; the axis label gains an
            automatically generated ``×10³`` suffix.

        ``{'MCR (KW)': (1e3, 'MCR (MW)')}``
            Divide by 1000 and replace the axis label outright.  Use this
            wherever the rescaling corresponds to a real unit change, since
            ``MCR (MW)`` reads better than ``MCR (kW) ×10³``.

        :param name: Internal variable name.
        :type name: str
        :param display_scale_map: Mapping from variable name to a divisor,
            or to a ``(divisor, label)`` pair.
        :type display_scale_map: dict, optional

        :returns: ``(divisor, label_override)``; ``(1.0, None)`` when the
            variable is absent from the map.
        :rtype: tuple[float, str | None]

        :raises ValueError: If a divisor is zero or non-finite, which would
            blank the axis rather than rescale it.
        """
        if not display_scale_map or name not in display_scale_map:
            return 1.0, None

        spec = display_scale_map[name]

        if isinstance(spec, (tuple, list)):
            divisor, label = float(spec[0]), str(spec[1])
        else:
            divisor, label = float(spec), None

        if divisor == 0 or not math.isfinite(divisor):
            raise ValueError(
                f"Display scale for {name!r} must be a finite non-zero "
                f"divisor, got {divisor!r}."
            )

        return divisor, label

    def _scale_suffix(self, divisor: float) -> str:
        """
        Build the axis-label suffix describing a display divisor.

        Follows the convention that the suffix states what a tick value
        must be multiplied by to recover the physical quantity, so a curve
        drawn as ``DWT / 1000`` is labelled ``×10³``.

        Exact powers of ten render as ``×10ⁿ`` with a Unicode superscript;
        anything else falls back to the literal factor.

        :param divisor: Value the plotted quantity was divided by.
        :type divisor: float
        :returns: Suffix string, empty when no rescaling was applied.
        :rtype: str
        """
        if divisor == 1.0:
            return ''

        exponent = math.log10(abs(divisor))

        if abs(exponent - round(exponent)) < 1e-9:
            return '×10' + str(int(round(exponent))).translate(self._SUPERSCRIPTS)

        return f'×{divisor:g}'

    def _axis_label(
            self,
            name: str,
            variable_name_map: Optional[dict[str, str]] = None,
            display_scale_map: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Compose an axis label from the name map and the display scale.

        Precedence runs: an explicit label in ``display_scale_map`` wins
        outright; otherwise the ``variable_name_map`` entry is used and the
        generated ``×10ⁿ`` suffix appended.  Keeping the two maps separate
        means a caller can change units on a figure without editing the
        name map every other figure shares.

        :param name: Internal variable name.
        :type name: str
        :param variable_name_map: Internal → display names.
        :type variable_name_map: dict[str, str], optional
        :param display_scale_map: Display divisors, per
            :meth:`_resolve_display_scale`.
        :type display_scale_map: dict, optional

        :returns: Axis label.
        :rtype: str
        """
        divisor, override = self._resolve_display_scale(name, display_scale_map)

        if override is not None:
            return override

        base = (variable_name_map or {}).get(name, name)
        suffix = self._scale_suffix(divisor)

        return f'{base} {suffix}' if suffix else base

    def _apply_display_scaling(
            self,
            curve_set: TargetCurveSet,
            display_scale_map: Optional[dict[str, Any]] = None,
            debug: bool = False,
    ) -> TargetCurveSet:
        """
        Return a copy of a curve set with plotted values rescaled.

        Rescaling is applied here, to a **copy**, rather than in the
        extractor.  The extractor's arrays are the analysis output — they
        feed divergence tables and numerical checks — and dividing them by
        a thousand to tidy an axis would corrupt every consumer that is not
        a figure.  Doing it at draw time keeps the change where it belongs:
        in presentation.

        Every x-derived field is rescaled together, including
        :attr:`CohortCurve.support`.  Missing one would misplace the
        solid/dotted boundary by the scale factor, which is the kind of
        error that produces a plausible-looking figure.

        :param curve_set: Curves in physical units.
        :type curve_set: TargetCurveSet
        :param display_scale_map: Display divisors, per
            :meth:`_resolve_display_scale`.
        :type display_scale_map: dict, optional
        :param debug: Report the resolved divisors.
        :type debug: bool

        :returns: A rescaled copy, or the original when no scaling applies.
        :rtype: TargetCurveSet
        """
        x_divisor, _ = self._resolve_display_scale(
            curve_set.predictor, display_scale_map)
        y_divisor, _ = self._resolve_display_scale(
            curve_set.target, display_scale_map)

        if x_divisor == 1.0 and y_divisor == 1.0:
            return curve_set

        if debug:
            print(f"   scaling    : x ÷{x_divisor:g}, y ÷{y_divisor:g}")

        scaled = [
            replace(
                curve,
                x_grid=curve.x_grid / x_divisor,
                y_curve=curve.y_curve / y_divisor,
                x_points=(None if curve.x_points is None
                          else curve.x_points / x_divisor),
                y_points=(None if curve.y_points is None
                          else curve.y_points / y_divisor),
                support=(None if curve.support is None
                         else (curve.support[0] / x_divisor,
                               curve.support[1] / x_divisor)),
            )
            for curve in curve_set.curves
        ]

        return replace(curve_set, curves=scaled)

    def _apply_tick_policy(
            self,
            ax,
            max_xticks: Optional[int] = None,
            max_yticks: Optional[int] = None,
            xtick_rotation: float = 0.0,
            tick_labelsize: Optional[float] = None,
    ) -> None:
        """
        Thin out, resize and rotate tick labels on one axis.

        Complements rescaling rather than replacing it.  Rescaling shortens
        each label; thinning reduces how many there are.  A five-digit DWT
        tick set needs both on a narrow tile.

        :param ax: Axis to adjust.
        :type ax: matplotlib.axes.Axes
        :param max_xticks: Upper bound on x-axis tick count.  ``None``
            leaves matplotlib's choice alone.
        :type max_xticks: int, optional
        :param max_yticks: Upper bound on y-axis tick count.
        :type max_yticks: int, optional
        :param xtick_rotation: Rotation in degrees for x tick labels.
            Labels are right-anchored when rotated so they trail away from
            their tick rather than centring under it.
        :type xtick_rotation: float
        :param tick_labelsize: Point size for the tick numerals.  Applied
            after the locator, since changing the count and changing the
            size are independent remedies for the same crowding.
        :type tick_labelsize: float, optional

        :returns: ``None``.
        """
        if max_xticks is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))

        if max_yticks is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))

        if tick_labelsize is not None:
            ax.tick_params(axis='both', which='major',
                           labelsize=tick_labelsize)

        if xtick_rotation:
            for label in ax.get_xticklabels():
                label.set_rotation(xtick_rotation)
                label.set_horizontalalignment('right')

    def _apply_performance_filter(
            self,
            targets: Sequence[str],
            min_r2: Optional[float] = None,
            filter_dataset: str = 'alternative',
            min_n_samples: Optional[int] = None,
            debug: bool = False,
    ) -> tuple[list[str], Optional[str]]:
        """
        Drop targets whose fitted models are too poor to interpret.

        Rationale
        ---------
        A curve drawn from a model that explains almost nothing is still a
        curve; the eye reads it as a design relationship regardless.  Gating
        on R² keeps the figure to models whose shape carries information.

        Two cautions apply, and both belong in the caption rather than the
        code:

        A low alternative-cohort R² is itself a transferability result.
        Filtering on it removes exactly the cases where the conventional
        structure fails to carry over, which is why every drop is recorded
        in the extractor's skip manifest instead of vanishing.

        R² is a model-adequacy measure, not an error measure.  Through the
        identity :math:`R^2 = 1 - (\\mathrm{NRMSE}/\\mathrm{CV})^2`, a cohort
        with low dispersion can post a poor R² while predicting as
        accurately as the conventional fleet.  A target failing this gate
        has not necessarily been predicted badly — it may simply be a
        narrow cohort.

        Delegation
        ----------
        The comparison itself is delegated to
        ``analyzer._filter_results_by_performance``, so the R² column names
        and the ``'all'``/``'any'`` semantics are defined once and shared
        with the coefficient plots.

        :param targets: Candidate target names, before filtering.
        :type targets: sequence[str]
        :param min_r2: Minimum R² for a target to be plotted.  ``None``
            disables the gate.
        :type min_r2: float, optional
        :param filter_dataset: Which cohort's R² is compared against
            ``min_r2``: ``'alternative'``, ``'conventional_full'``,
            ``'conventional_matched'``, ``'all'`` or ``'any'``.
        :type filter_dataset: str
        :param min_n_samples: Minimum alternative-cohort sample size.
        :type min_n_samples: int, optional
        :param debug: Print the before and after counts.
        :type debug: bool

        :returns: ``(kept_targets, note)``, where ``note`` summarises the
            gate for the figure heading, or ``None`` when nothing was
            dropped.
        :rtype: tuple[list[str], str | None]
        """
        targets = list(targets)

        if min_r2 is None and min_n_samples is None:
            return targets, None

        results = self.analyzer.comparison_results
        candidates = results[results['target'].isin(targets)]

        kept_df = self.analyzer._filter_results_by_performance(
            results_df=candidates,
            min_r2=min_r2,
            dataset=filter_dataset,
            min_n_samples=min_n_samples,
            verbose=debug,
        )

        kept = set(kept_df['target'])
        ordered = [t for t in targets if t in kept]

        # Every drop is recorded, so a caption can reconcile the number of
        # targets requested against the number drawn.
        r2_column = {
            'alternative': 'alt_r2',
            'conventional_full': 'conv_full_r2',
            'conventional_matched': 'conv_matched_r2',
        }.get(filter_dataset)

        for _, row in candidates.iterrows():
            if row['target'] in kept:
                continue

            details: dict[str, Any] = {'threshold_r2': min_r2,
                                       'filter_dataset': filter_dataset}

            if r2_column is not None and r2_column in row:
                details['observed_r2'] = float(row[r2_column])
            if 'n_alternative' in row:
                details['n_alternative'] = int(row['n_alternative'])

            self.extractor._record_skip(
                target=row['target'],
                predictors=tuple(row['predictors']),
                model_type=row['model_type'],
                reason='below_performance_threshold',
                **details,
            )

        n_dropped = len(targets) - len(ordered)

        if debug:
            print(f"   performance gate: {len(targets)} → {len(ordered)} "
                  f"targets (min_r2={min_r2}, dataset={filter_dataset!r}, "
                  f"min_n={min_n_samples})")

        if n_dropped == 0:
            return ordered, None

        parts = []
        if min_r2 is not None:
            parts.append(f'R²({filter_dataset}) ≥ {min_r2:.2f}')
        if min_n_samples is not None:
            parts.append(f'n ≥ {min_n_samples}')

        note = ""

        return ordered, note
    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def _finalise(
            self,
            fig,
            save_path: str | Path | None,
            dpi: int,
            show_plot: bool,
            debug: bool = False,
    ) -> None:
        """
        Save and optionally display a completed figure.

        :param fig: Figure to write.
        :type fig: matplotlib.figure.Figure
        :param save_path: Destination file; parents are created.
        :type save_path: str or pathlib.Path, optional
        :param dpi: Resolution.
        :type dpi: int
        :param show_plot: Call ``plt.show()``.
        :type show_plot: bool
        :param debug: Print the destination.
        :type debug: bool

        :returns: ``None``.
        """
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

            if debug or getattr(self.analyzer, 'verbose', False):
                print(f"✅ Functional-form plot saved to {save_path}")

        if show_plot:
            plt.show()