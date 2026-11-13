"""
Functional-Form Curve Extraction
================================

Turns the three structure-preserving fits held by
:class:`~CoefficientComparisonAnalyzer` (Full Conventional, Matched
Conventional, Alternative Fuel) into *plottable curves* in original
physical units.

Separation of concerns
----------------------
This module contains **no matplotlib**.  It answers the question
"what does the fitted function look like?" as arrays; rendering those
arrays is the job of
:class:`~functional_form_plotter.FunctionalFormPlotter`.

Keeping the two apart means the same numeric output can be sent to a
figure, to a LaTeX table, or to a numerical check that two curves agree
to within some tolerance, without any of those consumers inheriting
plotting state.
"""


from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class CohortCurve:
    """
    One cohort's fitted response curve, in original physical units.

    An instance is produced per (target, cohort) pair.  ``y_curve`` is
    aligned element-wise with ``x_grid`` and may contain ``NaN`` where the
    model is undefined (for example a log-domain form evaluated at a
    non-positive normalised predictor value); the ``NaN`` is deliberate, so
    a broken region shows as a gap rather than silently shifting the curve.

    :ivar cohort: Internal cohort key, one of ``'conventional_full'``,
        ``'conventional_matched'``, ``'alternative'``.
    :vartype cohort: str
    :ivar label: Human-readable cohort name used in titles and legends.
    :vartype label: str
    :ivar x_grid: Predictor values in original units, shape ``(n_grid,)``.
    :vartype x_grid: numpy.ndarray
    :ivar y_curve: Model response in original units, shape ``(n_grid,)``.
    :vartype y_curve: numpy.ndarray
    :ivar x_points: Observed predictor values for this cohort, or ``None``
        when scatter data was not requested.
    :vartype x_points: numpy.ndarray, optional
    :ivar y_points: Observed target values for this cohort, or ``None``.
    :vartype y_points: numpy.ndarray, optional
    :ivar n_fit: Number of rows the model was fitted on, after the same
        cleaning the analyser applied.
    :vartype n_fit: int
    :ivar support: ``(min, max)`` of the predictor actually observed in
        this cohort.  Any part of ``x_grid`` outside this interval is
        extrapolation *for this cohort*.
    :vartype support: tuple[float, float], optional
    :ivar r2: In-sample coefficient of determination of *this cohort's*
        fit, taken from the analysed result row rather than recomputed, so
        the number on a figure is the same number in the results table.
        ``None`` when the analyser recorded none.

        Two readings this value does **not** support.  It is not a
        cross-cohort accuracy comparison: through
        :math:`R^2 = 1 - (\\mathrm{NRMSE}/\\mathrm{CV})^2` a narrow cohort
        can post a poor :math:`R^2` while predicting as tightly as a broad
        one, so a low alternative-fuel value may be dispersion rather than
        misfit.  And it is not a statement about the curve as drawn: it
        measures the fit on the cohort's own observations, whereas the
        curve is evaluated over a shared grid that mostly lies outside a
        small cohort's support.
    :vartype r2: float, optional
    :ivar rrmse: Relative RMSE of the same fit, on the same terms.  Carried
        alongside :attr:`r2` because it is the scale-free error measure the
        :math:`R^2` caveat above points to.
    :vartype rrmse: float, optional
    :ivar valid: ``False`` when the cohort could not be evaluated; the
        remaining fields are then placeholders.
    :vartype valid: bool
    :ivar message: Diagnostic string explaining a failure or a partial
        result.
    :vartype message: str
    """

    cohort: str
    label: str
    x_grid: np.ndarray
    y_curve: np.ndarray
    x_points: Optional[np.ndarray] = None
    y_points: Optional[np.ndarray] = None
    n_fit: int = 0
    support: Optional[tuple[float, float]] = None
    r2: Optional[float] = None
    rrmse: Optional[float] = None
    valid: bool = True
    message: str = ''

    @property
    def n_undefined(self) -> int:
        """
        Number of grid points at which the model returned a non-finite value.

        :returns: Count of ``NaN`` entries in :attr:`y_curve`.
        :rtype: int
        """
        return int(np.sum(~np.isfinite(self.y_curve)))


@dataclass
class TargetCurveSet:
    """
    All cohort curves for one target, plus the metadata needed to label them.

    This is the single object handed to the plotter.  Everything the figure
    needs — axis ranges, formula string, sample counts — is resolved here,
    so the plotter performs no model evaluation of its own.

    Single-predictor only
    ---------------------
    A set is built only for models with exactly one predictor, so the
    x-axis variable is unambiguous and the curve is the model rather than a
    cut through it.  Multi-predictor targets are refused upstream by
    :meth:`~FunctionalFormCurveExtractor.build_curve_set`.

    :ivar target: Internal target variable name.
    :vartype target: str
    :ivar predictor: The model's single predictor, plotted on the x-axis.
    :vartype predictor: str
    :ivar model_type: Model-type identifier, e.g. ``'mfp'``,
        ``'power-log-multivariate'``.
    :vartype model_type: str
    :ivar reference_dataset: Cohort whose observed range defined the grid.
    :vartype reference_dataset: str
    :ivar normalized: Whether the models were fitted on normalised data.
    :vartype normalized: bool
    :ivar normalization_method: Method actually used, or ``None``.
    :vartype normalization_method: str, optional
    :ivar curves: One :class:`CohortCurve` per cohort, in display order.
    :vartype curves: list[CohortCurve]
    :ivar formula: β-notation formula string for the shared structure.
    :vartype formula: str
    :ivar message: Diagnostic string for target-level failures.
    :vartype message: str
    """

    target: str
    predictor: str
    model_type: str
    reference_dataset: str
    normalized: bool
    normalization_method: Optional[str]
    curves: list[CohortCurve] = field(default_factory=list)
    formula: str = ''
    message: str = ''

    @property
    def valid_curves(self) -> list[CohortCurve]:
        """
        Curves that were successfully evaluated.

        :returns: Subset of :attr:`curves` with ``valid`` set.
        :rtype: list[CohortCurve]
        """
        return [c for c in self.curves if c.valid]

    def get(self, cohort: str) -> Optional[CohortCurve]:
        """
        Look up one cohort's curve by key.

        :param cohort: Cohort key.
        :type cohort: str
        :returns: The matching curve, or ``None``.
        :rtype: CohortCurve, optional
        """
        for curve in self.curves:
            if curve.cohort == cohort:
                return curve
        return None


# =============================================================================
# Extractor
# =============================================================================

class FunctionalFormCurveExtractor:
    """
    Evaluate the three refitted models over a shared predictor grid.

    Composes :class:`~CoefficientComparisonAnalyzer` — it holds a reference
    rather than subclassing, because it needs only three things from the
    analyser (the refitted descriptors, the normalisation parameters, and
    the cohort datasets) and has no business inheriting its analysis API.

    Why a shared grid matters
    -------------------------
    Each cohort's model is a different function.  Plotting each over its own
    data range would confound two things a transferability study must keep
    apart: a change in the *function* and a change in the *domain the
    function was fitted on*.  Evaluating all three over one grid, taken from
    a nominated reference cohort, isolates the first.  Where a cohort's own
    data does not cover that grid, :attr:`CohortCurve.support` records it so
    the extrapolated region can be marked rather than quietly drawn.

    Multi-predictor models
    ----------------------
    A model in more than one predictor has no single curve; it has a
    surface.  Two reductions are offered:

    ``slice_mode='fixed'`` (default)
        Sweep one predictor and hold the others at a fixed point — the
        reference cohort's median by default.  This is the *conditional
        slice*: "how does the target respond to length, for an otherwise
        typical conventional ship?"  The fixed point is taken from the
        reference cohort and reused for all three curves; taking each
        cohort's own median would mix a coefficient change with a change in
        where the function is being read, which is the same denominator
        trap that inflates cross-cohort NRMSE comparisons.

    ``slice_mode='marginal'``
        Average the model over the reference cohort's observed joint
        distribution of the other predictors — a partial-dependence curve.
        Free of an arbitrary evaluation point, but smoother and slower, and
        still anchored to the reference cohort's covariate distribution for
        the same commensurability reason.

    Recovering a predictor from a descriptor
    ----------------------------------------
    :class:`~ModelDescriptor` stores structure and coefficients but exposes
    no ``predict``.  A predictor is recovered by calling
    :meth:`~ModelDescriptor.refit` with ``force_structure=True`` on a **deep
    copy** of the stored descriptor, using that cohort's own fitting data.
    Because the structure is forced and the estimator is deterministic, this
    reproduces the stored coefficients exactly; the copy is required because
    ``refit`` mutates the descriptor in place.
    """

    #: Display order and human-readable labels for the three cohorts.
    COHORT_LABELS: dict[str, str] = {
        'conventional_full': 'Full conventional',
        'conventional_matched': 'Matched conventional',
        'alternative': 'Alternative fuel',
    }

    #: Result-row columns holding each cohort's own fit quality.  Mapped
    #: here rather than in the plotter so that the column names stay a
    #: property of the analyser's schema, and a renderer never has to know
    #: how the analyser spells its metrics.
    _METRIC_COLUMNS: dict[str, tuple[str, str]] = {
        'conventional_full': ('conv_full_r2', 'conv_full_rrmse'),
        'conventional_matched': ('conv_matched_r2', 'conv_matched_rrmse'),
        'alternative': ('alt_r2', 'alt_rrmse'),
    }

    #: Model types whose fitting domain requires strictly positive inputs.
    #: Mirrors the guard in ``_clean_data_for_nonlinear``; grid points that
    #: violate it are marked undefined rather than fed to the model.
    _POSITIVE_DOMAIN_HINTS: frozenset[str] = frozenset({
        'power', 'logarithmic', 'logarithmic-power', 'mfp',
    })

    def __init__(self, analyzer):
        """
        Bind the extractor to a completed comparison analysis.

        :param analyzer: A :class:`~CoefficientComparisonAnalyzer` on which
            :meth:`~CoefficientComparisonAnalyzer.analyze_all_targets` has
            already been run, so that ``model_descriptors`` and
            ``normalization_params`` are populated.
        """
        self.analyzer = analyzer

        #: Targets refused before any curve was built, each a dict carrying
        #: ``'target'``, ``'predictors'``, ``'model_type'`` and ``'reason'``.
        #: Kept rather than discarded so a figure caption can reconcile the
        #: number of targets requested against the number drawn — the same
        #: reasoning as the analyser's exclusion manifest.
        self.skipped_targets: list[dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def build_curve_set(
            self,
            target: str,
            *,
            reference_dataset: str = 'conventional_full',
            n_grid: int = 200,
            x_range_mode: str = 'reference',
            range_padding: float = 0.0,
            include_points: bool = True,
            debug: bool = False,
    ) -> Optional[TargetCurveSet]:
        """
        Build the three cohort curves for one single-predictor target.

        Targets whose selected combination carries more than one predictor
        are skipped with a warning and recorded in :attr:`skipped_targets`.
        Drawing them would require reducing a fitted surface to a curve by
        holding the other predictors somewhere, and every available choice
        of "somewhere" mixes a change in the relationship with a change in
        where the relationship is being read.

        :param target: Target variable name, as it appears in
            ``analyzer.comparison_results['target']``.
        :type target: str
        :param reference_dataset: Cohort defining the grid range.  One of
            the keys of :attr:`COHORT_LABELS`.
        :type reference_dataset: str
        :param n_grid: Number of grid points along the predictor.
        :type n_grid: int
        :param x_range_mode: ``'reference'`` to span only the reference
            cohort's observed range, or ``'union'`` to span every cohort's
            range.  ``'union'`` shows where the alternative fleet sits
            outside conventional experience, at the cost of extrapolating
            the conventional curve.
        :type x_range_mode: str
        :param range_padding: Fractional padding added to each end of the
            grid range, e.g. ``0.05`` for 5%.
        :type range_padding: float
        :param include_points: Extract the observed scatter data as well as
            the curves.  The plotter decides whether to draw it.
        :type include_points: bool
        :param debug: Print each stage — row lookup, normalisation
            resolution, grid construction, per-cohort evaluation — to
            stdout.
        :type debug: bool

        :returns: The assembled curve set, or ``None`` when the target has
            no analysed row or is not single-predictor.
        :rtype: TargetCurveSet, optional
        """
        if debug:
            print("\n" + "=" * 78)
            print(f"🔍 DEBUG: build_curve_set — target={target!r}")
            print("=" * 78)

        # ── Stage 1: locate the analysed row ─────────────────────────────
        row = self._get_result_row(target)
        if row is None:
            if debug:
                print("   ✗ no row in analyzer.comparison_results")
            return None

        predictors = tuple(row['predictors'])
        model_type = row['model_type']
        key = (predictors, target)

        if debug:
            print(f"   predictors : {predictors}")
            print(f"   model_type : {model_type}")

        # ── Stage 2: single-predictor gate ───────────────────────────────
        if len(predictors) != 1:
            self._record_skip(
                target=target,
                predictors=predictors,
                model_type=model_type,
                reason='multi_predictor_model',
            )
            print(
                f"⚠️  Skipping {target!r}: selected model uses "
                f"{len(predictors)} predictors {predictors}. The "
                f"functional-form comparison plots a single curve and is "
                f"defined for one-predictor models only."
            )
            return None

        predictor = predictors[0]

        curve_set = TargetCurveSet(
            target=target,
            predictor=predictor,
            model_type=model_type,
            reference_dataset=reference_dataset,
            normalized=bool(row.get('normalized', False)),
            normalization_method=row.get('normalization_method'),
            formula=self._get_formula(key, debug=debug),
        )

        # ── Stage 3: normalisation contract ──────────────────────────────
        norm_params, norm_method = self._resolve_normalization(key, row)

        if debug:
            print(f"   normalise  : {curve_set.normalized} "
                  f"(method={norm_method!r}, params="
                  f"{'present' if norm_params else 'none'})")

        # ── Stage 4: per-cohort fitting data, in original units ──────────
        cohort_data: dict[str, dict[str, Any]] = {}

        for cohort in self.COHORT_LABELS:
            cohort_data[cohort] = self._cohort_arrays(
                cohort=cohort,
                predictors=predictors,
                target=target,
                model_type=model_type,
                norm_params=norm_params,
                norm_method=norm_method,
                normalized=curve_set.normalized,
                debug=debug,
            )

        if cohort_data.get(reference_dataset, {}).get('X_orig') is None:
            curve_set.message = (
                f"Reference cohort {reference_dataset!r} has no usable data; "
                f"cannot define an axis."
            )
            if debug:
                print(f"   ✗ {curve_set.message}")
            return curve_set

        # ── Stage 5: the shared grid ─────────────────────────────────────
        x_grid = self._build_grid(
            cohort_data=cohort_data,
            reference_dataset=reference_dataset,
            x_range_mode=x_range_mode,
            n_grid=n_grid,
            range_padding=range_padding,
            debug=debug,
        )

        # ── Stage 6: evaluate every cohort on that grid ──────────────────
        # Fit quality is read from the analysed row, not recomputed here.
        # Recomputing it against ``y_curve`` would answer a different
        # question — how well the model tracks its own cohort *on the
        # shared grid* — and would quietly disagree with the results table.
        for cohort, label in self.COHORT_LABELS.items():
            r2, rrmse = self._cohort_metrics(row, cohort, debug=debug)

            curve_set.curves.append(self._evaluate_cohort(
                cohort=cohort,
                label=label,
                key=key,
                arrays=cohort_data[cohort],
                x_grid=x_grid,
                model_type=model_type,
                norm_params=norm_params,
                norm_method=norm_method,
                normalized=curve_set.normalized,
                include_points=include_points,
                r2=r2,
                rrmse=rrmse,
                debug=debug,
            ))

        if debug:
            print("   ── summary ──")
            for c in curve_set.curves:
                status = 'ok' if c.valid else f'FAILED ({c.message})'
                r2_text = 'n/a' if c.r2 is None else f'{c.r2:.3f}'
                print(f"     {c.label:<24s} n={c.n_fit:<5d} "
                      f"R²={r2_text:<7s} "
                      f"undefined={c.n_undefined:<4d} {status}")

        return curve_set

    def build_all_curve_sets(
            self,
            targets: Optional[Sequence[str]] = None,
            *,
            debug: bool = False,
            **kwargs,
    ) -> list[TargetCurveSet]:
        """
        Build curve sets for many targets in one pass.

        Multi-predictor targets are skipped rather than raising, so a mixed
        selection still produces a figure for the targets that qualify.
        Consult :meth:`print_skip_manifest` afterwards for the count.

        :param targets: Target names to process.  Defaults to every target
            in ``analyzer.comparison_results``, in results order.
        :type targets: sequence[str], optional
        :param debug: Forwarded to :meth:`build_curve_set`, plus a
            per-target progress line.
        :type debug: bool
        :param kwargs: Any other keyword accepted by
            :meth:`build_curve_set`.

        :returns: One curve set per target that could be built.
        :rtype: list[TargetCurveSet]
        """
        results = self.analyzer.comparison_results

        if results is None or len(results) == 0:
            if debug:
                print("🔍 DEBUG: build_all_curve_sets — no comparison results")
            return []

        if targets is None:
            targets = list(results['target'])

        sets: list[TargetCurveSet] = []

        for target in targets:
            if debug:
                print(f"\n🔍 DEBUG: build_all_curve_sets — {target}")
            curve_set = self.build_curve_set(target, debug=debug, **kwargs)
            if curve_set is not None:
                sets.append(curve_set)

        if debug:
            print(f"\n🔍 DEBUG: built {len(sets)}/{len(targets)} curve sets "
                  f"({len(self.skipped_targets)} skipped)")

        return sets

    # -------------------------------------------------------------------
    # Stage 1–3 helpers: lookup and contracts
    # -------------------------------------------------------------------

    def _record_skip(
            self,
            target: str,
            predictors: tuple[str, ...],
            model_type: str,
            reason: str,
            **details,
    ) -> None:
        """
        Record a target refused before any curve was built.

        Silent skipping would make a figure showing four of six targets
        indistinguishable from one where only four were ever requested.

        :param target: Target variable name.
        :type target: str
        :param predictors: The predictor combination that was selected.
        :type predictors: tuple[str, ...]
        :param model_type: Selected model type.
        :type model_type: str
        :param reason: Short machine-readable reason code.
        :type reason: str
        :param details: Additional diagnostic fields merged into the record.

        :returns: ``None``.  Appends to :attr:`skipped_targets`.
        """
        record = {
            'target': target,
            'predictors': predictors,
            'n_predictors': len(predictors),
            'model_type': model_type,
            'reason': reason,
        }
        record.update(details)
        self.skipped_targets.append(record)

    def print_skip_manifest(self, n_requested: int | None = None) -> None:
        """
        Print every target refused by the single-predictor gate.

        :param n_requested: Number of targets originally requested.  When
            supplied, an accounting line reconciles requested, plotted and
            skipped counts for use in a figure caption.
        :type n_requested: int, optional

        :returns: ``None``.  Prints to stdout.
        """
        print("\n" + "=" * 78)
        print("FUNCTIONAL-FORM SKIP MANIFEST")
        print("=" * 78)

        if n_requested is not None:
            n_skipped = len(self.skipped_targets)
            print(f"Requested: {n_requested} | "
                  f"Plotted: {n_requested - n_skipped} | "
                  f"Skipped: {n_skipped}")
            print("-" * 78)

        if not self.skipped_targets:
            print("  (none)")
        else:
            for record in self.skipped_targets:
                print(f"\n  • {record['target']}  "
                      f"[{record.get('model_type', 'unknown model')}]")
                print(f"    reason     : {record['reason']}")
                print(f"    predictors : {record['n_predictors']} — "
                      f"{record['predictors']}")

        print("\n" + "=" * 78)

    def _get_result_row(self, target: str) -> Optional[pd.Series]:
        """
        Return the comparison-results row for a target.

        :param target: Target variable name.
        :type target: str
        :returns: The row, or ``None`` when the target was not analysed.
        :rtype: pandas.Series, optional
        """
        results = self.analyzer.comparison_results

        if results is None or len(results) == 0:
            return None

        matches = results[results['target'] == target]

        if len(matches) == 0:
            return None

        return matches.iloc[0]

    def _cohort_metrics(
            self,
            row: pd.Series,
            cohort: str,
            debug: bool = False,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Read one cohort's fit quality out of the analysed result row.

        Missing or non-finite entries return ``None`` rather than
        ``NaN``.  The distinction matters downstream: ``None`` means the
        analyser recorded nothing and the annotation should be omitted,
        whereas a ``NaN`` reaching a format string would print the string
        ``'nan'`` onto the figure as though it were a result.

        :param row: The target's row of ``analyzer.comparison_results``.
        :type row: pandas.Series
        :param cohort: Cohort key, one of :attr:`COHORT_LABELS`.
        :type cohort: str
        :param debug: Print the resolved values.
        :type debug: bool

        :returns: ``(r2, rrmse)``, either element ``None`` when unavailable.
        :rtype: tuple[float | None, float | None]
        """
        columns = self._METRIC_COLUMNS.get(cohort)

        if columns is None:
            return None, None

        def _read(column: str) -> Optional[float]:
            if column not in row:
                return None
            value = row[column]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        r2, rrmse = _read(columns[0]), _read(columns[1])

        if debug:
            print(f"   [{cohort}] metrics: "
                  f"R²={'n/a' if r2 is None else f'{r2:.4f}'}, "
                  f"rRMSE={'n/a' if rrmse is None else f'{rrmse:.4f}'}")

        return r2, rrmse

    def _resolve_swept_index(
            self,
            predictors: tuple[str, ...],
            swept_predictor: Optional[str],
    ) -> int:
        """
        Resolve which predictor column is swept along the x-axis.

        :param predictors: Predictor names in model-column order.
        :type predictors: tuple[str, ...]
        :param swept_predictor: Requested predictor name, or ``None`` to
            take the first.
        :type swept_predictor: str, optional
        :returns: Column index into the design matrix.
        :rtype: int
        :raises ValueError: If the requested predictor is not in the model.
        """
        if swept_predictor is None:
            return 0

        if swept_predictor not in predictors:
            raise ValueError(
                f"Requested swept predictor {swept_predictor!r} is not in the "
                f"fitted model, whose predictors are {predictors}. The x-axis "
                f"must be one of the model's own inputs."
            )

        return predictors.index(swept_predictor)

    def _resolve_normalization(
            self,
            key: tuple,
            row: pd.Series,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Recover the normalisation contract used when the models were fitted.

        The method varies by model form — multiplicative forms use
        ``'scale'`` rather than ``'zscore'`` — so it is read back from the
        analysis rather than assumed.  Three sources are tried in
        decreasing order of authority: the analyser's per-key record, the
        results row, then the analyser-level default.

        :param key: ``(predictors, target)`` key.
        :type key: tuple
        :param row: The target's comparison-results row.
        :type row: pandas.Series
        :returns: ``(normalization_params, normalization_method)``; both
            ``None`` when the analysis ran unnormalised.
        :rtype: tuple[dict | None, str | None]
        """
        if not bool(row.get('normalized', False)):
            return None, None

        params = self.analyzer.normalization_params.get(key)

        method = (
            self.analyzer.normalization_methods.get(key)
            or row.get('normalization_method')
            or self.analyzer.normalization_method
        )

        return params, method

    def _get_formula(self, key: tuple, debug: bool = False) -> str:
        """
        Render the shared β-notation formula from the full-conventional fit.

        All three cohorts share one structure by construction, so a single
        formula labels the whole figure.

        :param key: ``(predictors, target)`` key.
        :type key: tuple
        :param debug: Report a rendering failure to stdout.
        :type debug: bool
        :returns: Formula string, or ``''`` when unavailable.
        :rtype: str
        """
        descriptor = self.analyzer.model_descriptors['conventional_full'].get(key)

        if descriptor is None:
            return ''

        try:
            return descriptor.get_formula_string(
                include_coefficients=False, notation='math',
            )
        except Exception as exc:
            if debug:
                print(f"   ⚠️  formula rendering failed: {exc}")
            return ''

    # -------------------------------------------------------------------
    # Stage 4 helpers: cohort data
    # -------------------------------------------------------------------

    def _resolve_dataset(
            self,
            cohort: str,
            predictors: tuple[str, ...],
    ) -> Optional[pd.DataFrame]:
        """
        Return the raw DataFrame backing one cohort.

        The matched cohort is predictor-specific, since matching is
        performed per predictor combination.

        :param cohort: Cohort key.
        :type cohort: str
        :param predictors: Predictor combination, needed to select the
            matched set.
        :type predictors: tuple[str, ...]
        :returns: The cohort's DataFrame, or ``None``.
        :rtype: pandas.DataFrame, optional
        :raises ValueError: On an unrecognised cohort key.
        """
        foundation = self.analyzer.foundation

        if cohort == 'conventional_full':
            return foundation.conventional_full
        if cohort == 'alternative':
            return foundation.alternative
        if cohort == 'conventional_matched':
            return foundation.get_matched_dataset(predictors)

        raise ValueError(
            f"Unknown cohort {cohort!r}; expected one of "
            f"{sorted(self.COHORT_LABELS)}."
        )

    def _cohort_arrays(
            self,
            cohort: str,
            predictors: tuple[str, ...],
            target: str,
            model_type: str,
            norm_params: Optional[dict],
            norm_method: Optional[str],
            normalized: bool,
            debug: bool = False,
    ) -> dict[str, Any]:
        """
        Reproduce one cohort's fitting arrays, in both scales.

        The same preparation and cleaning the analyser applied is repeated
        here, so the normalised arrays returned are exactly the ones the
        stored descriptor was fitted on.  That is what makes the
        forced-structure refit in :meth:`_recover_predictor` an identity
        rather than a re-estimation on different rows.

        :param cohort: Cohort key.
        :type cohort: str
        :param predictors: Predictor names.
        :type predictors: tuple[str, ...]
        :param target: Target name.
        :type target: str
        :param model_type: Model-type identifier, controlling cleaning.
        :type model_type: str
        :param norm_params: Normalisation parameters fitted on the full
            conventional cohort.
        :type norm_params: dict, optional
        :param norm_method: Normalisation method.
        :type norm_method: str, optional
        :param normalized: Whether normalisation was in force.
        :type normalized: bool
        :param debug: Print row counts per stage.
        :type debug: bool

        :returns: Dict with ``'X_orig'``, ``'y_orig'``, ``'X_norm'``,
            ``'y_norm'``, ``'n'`` and ``'message'``.  Array entries are
            ``None`` when the cohort is unusable.
        :rtype: dict
        """
        blank = {'X_orig': None, 'y_orig': None, 'X_norm': None,
                 'y_norm': None, 'n': 0, 'message': ''}

        dataset = self._resolve_dataset(cohort, predictors)

        if dataset is None or len(dataset) == 0:
            blank['message'] = 'cohort dataset unavailable'
            if debug:
                print(f"   [{cohort}] ✗ dataset unavailable")
            return blank

        prep = self.analyzer.foundation.prepare_dataset(
            dataset, list(predictors), target,
        )

        if prep is None:
            blank['message'] = 'insufficient complete rows'
            if debug:
                print(f"   [{cohort}] ✗ insufficient complete rows")
            return blank

        X_orig, y_orig = prep

        # Normalised copies, produced by the foundation itself so the
        # transform is never reimplemented here.
        if normalized and norm_params is not None:
            X_norm, y_norm, _ = self.analyzer.foundation._normalize_data(
                X_orig, y_orig, method=norm_method,
                fit_params=False, params=norm_params,
            )
        else:
            X_norm, y_norm = X_orig, y_orig

        # Same cleaning gate the analyser applied before fitting.
        X_clean, y_clean, mask, _report = self.analyzer._clean_data_for_nonlinear(
            X_norm, y_norm, model_type, self.COHORT_LABELS[cohort],
            target=target, debug=False,
        )

        if X_clean is None:
            blank['message'] = 'no rows survive nonlinear cleaning'
            if debug:
                print(f"   [{cohort}] ✗ no rows survive cleaning")
            return blank

        result = {
            'X_orig': X_orig[mask],
            'y_orig': y_orig[mask],
            'X_norm': X_clean,
            'y_norm': y_clean,
            'n': int(len(y_clean)),
            'message': '',
        }

        if debug:
            print(f"   [{cohort}] rows {len(y_orig)} → {result['n']} after cleaning")

        return result

    # -------------------------------------------------------------------
    # Stage 5 helpers: grid and slice point
    # -------------------------------------------------------------------
    def _build_grid(
            self,
            cohort_data: dict[str, dict[str, Any]],
            reference_dataset: str,
            x_range_mode: str,
            n_grid: int,
            range_padding: float,
            debug: bool = False,
    ) -> np.ndarray:
        """
        Construct the shared evaluation grid for the predictor.

        Evaluating all three models over one grid is what isolates a change
        in the *function* from a change in the *domain it was fitted on*.
        Where a cohort's own data does not cover the grid,
        :attr:`CohortCurve.support` records it so the extrapolated region
        can be marked rather than quietly drawn.

        :param cohort_data: Output of :meth:`_cohort_arrays` per cohort.
        :type cohort_data: dict
        :param reference_dataset: Cohort defining the range under
            ``x_range_mode='reference'``.
        :type reference_dataset: str
        :param x_range_mode: ``'reference'`` or ``'union'``.
        :type x_range_mode: str
        :param n_grid: Number of grid points.
        :type n_grid: int
        :param range_padding: Fractional padding at each end.
        :type range_padding: float
        :param debug: Print the resolved range.
        :type debug: bool

        :returns: Monotonically increasing grid, shape ``(n_grid,)``.
        :rtype: numpy.ndarray
        :raises ValueError: On an unrecognised ``x_range_mode``.
        """
        if x_range_mode == 'reference':
            sources = [reference_dataset]
        elif x_range_mode == 'union':
            sources = list(self.COHORT_LABELS)
        else:
            raise ValueError(
                f"Unknown x_range_mode {x_range_mode!r}; "
                f"expected 'reference' or 'union'."
            )

        lows, highs = [], []

        for cohort in sources:
            X = cohort_data.get(cohort, {}).get('X_orig')
            if X is None:
                continue
            lows.append(float(np.min(X[:, 0])))
            highs.append(float(np.max(X[:, 0])))

        low, high = min(lows), max(highs)

        if range_padding:
            span = high - low
            low -= span * range_padding
            high += span * range_padding

        if debug:
            print(f"   grid       : [{low:.4g}, {high:.4g}] "
                  f"× {n_grid} points (mode={x_range_mode})")

        return np.linspace(low, high, n_grid)

    def _resolve_slice_values(
            self,
            X_ref: np.ndarray,
            predictors: tuple[str, ...],
            swept_index: int,
            slice_statistic: str,
            slice_overrides: Optional[dict[str, float]],
            debug: bool = False,
    ) -> dict[str, float]:
        """
        Choose the fixed values at which non-swept predictors are held.

        Taken from the reference cohort and reused for every curve.  A
        per-cohort slice point would make each curve a reading of its model
        at a different ship, so a visible gap between curves could not be
        attributed to the coefficients.

        :param X_ref: Reference cohort design matrix in original units.
        :type X_ref: numpy.ndarray
        :param predictors: Predictor names, aligned with the columns of
            ``X_ref``.
        :type predictors: tuple[str, ...]
        :param swept_index: Column index that is swept and therefore not
            fixed.
        :type swept_index: int
        :param slice_statistic: ``'median'`` or ``'mean'``.
        :type slice_statistic: str
        :param slice_overrides: Explicit per-variable values that take
            precedence over the statistic.
        :type slice_overrides: dict[str, float], optional
        :param debug: Print the resolved slice point.
        :type debug: bool

        :returns: Mapping from predictor name to held value.  Empty for a
            single-predictor model.
        :rtype: dict[str, float]
        :raises ValueError: On an unrecognised ``slice_statistic``.
        """
        if slice_statistic == 'median':
            stat = np.median
        elif slice_statistic == 'mean':
            stat = np.mean
        else:
            raise ValueError(
                f"Unknown slice_statistic {slice_statistic!r}; "
                f"expected 'median' or 'mean'."
            )

        overrides = slice_overrides or {}
        values: dict[str, float] = {}

        for idx, name in enumerate(predictors):
            if idx == swept_index:
                continue
            values[name] = float(overrides.get(name, stat(X_ref[:, idx])))

        if debug and values:
            pretty = ', '.join(f'{k}={v:.4g}' for k, v in values.items())
            print(f"   slice at   : {pretty} ({slice_statistic} of reference)")

        return values

    # -------------------------------------------------------------------
    # Stage 6 helpers: evaluation
    # -------------------------------------------------------------------

    def _evaluate_cohort(
            self,
            cohort: str,
            label: str,
            key: tuple,
            arrays: dict[str, Any],
            x_grid: np.ndarray,
            model_type: str,
            norm_params: Optional[dict],
            norm_method: Optional[str],
            normalized: bool,
            include_points: bool,
            r2: Optional[float] = None,
            rrmse: Optional[float] = None,
            debug: bool = False,
    ) -> CohortCurve:
        """
        Evaluate one cohort's fitted model over the shared grid.

        With a single predictor the design matrix is the grid itself,
        reshaped to a column; there is no covariate to hold and no
        reduction to justify.

        :param cohort: Cohort key.
        :type cohort: str
        :param label: Human-readable cohort name.
        :type label: str
        :param key: ``(predictors, target)`` key for descriptor lookup.
        :type key: tuple
        :param arrays: Output of :meth:`_cohort_arrays` for this cohort.
        :type arrays: dict
        :param x_grid: Shared grid in original units.
        :type x_grid: numpy.ndarray
        :param model_type: Model-type identifier.
        :type model_type: str
        :param norm_params: Normalisation parameters.
        :type norm_params: dict, optional
        :param norm_method: Normalisation method.
        :type norm_method: str, optional
        :param normalized: Whether normalisation was in force.
        :type normalized: bool
        :param include_points: Attach the observed scatter data.
        :type include_points: bool
        :param r2: This cohort's in-sample R², carried through to the
            curve for annotation.  Attached even to a failed curve, so a
            caller can still report why a cohort with a known fit quality
            could not be drawn.
        :type r2: float, optional
        :param rrmse: This cohort's relative RMSE, on the same terms.
        :type rrmse: float, optional
        :param debug: Print per-cohort progress.
        :type debug: bool

        :returns: The cohort's curve; ``valid`` is ``False`` on failure.
        :rtype: CohortCurve
        """
        nan_curve = np.full_like(x_grid, np.nan, dtype=float)

        def _fail(message: str) -> CohortCurve:
            if debug:
                print(f"   [{cohort}] ✗ {message}")
            return CohortCurve(
                cohort=cohort, label=label, x_grid=x_grid,
                y_curve=nan_curve, r2=r2, rrmse=rrmse,
                valid=False, message=message,
            )

        if arrays['X_norm'] is None:
            return _fail(arrays['message'] or 'no data')

        descriptor = self.analyzer.model_descriptors[cohort].get(key)

        if descriptor is None:
            return _fail('no refitted descriptor stored for this cohort')

        try:
            model, preprocessor = self._recover_predictor(
                descriptor, arrays['X_norm'], arrays['y_norm'],
            )
        except Exception as exc:
            return _fail(f'structure-preserving refit failed: {exc}')

        y_curve = self._predict_original_scale(
            model=model,
            preprocessor=preprocessor,
            X_design=x_grid.reshape(-1, 1),
            model_type=model_type,
            norm_params=norm_params,
            norm_method=norm_method,
            normalized=normalized,
        )

        X_orig = arrays['X_orig']
        support = (float(np.min(X_orig[:, 0])), float(np.max(X_orig[:, 0])))

        curve = CohortCurve(
            cohort=cohort,
            label=label,
            x_grid=x_grid,
            y_curve=y_curve,
            x_points=X_orig[:, 0] if include_points else None,
            y_points=arrays['y_orig'] if include_points else None,
            n_fit=arrays['n'],
            support=support,
            r2=r2,
            rrmse=rrmse,
            valid=True,
        )

        if curve.n_undefined:
            curve.message = (
                f'{curve.n_undefined}/{len(x_grid)} grid points outside the '
                f'model domain'
            )

        if debug:
            print(f"   [{cohort}] evaluated: n_fit={curve.n_fit}, "
                  f"support=[{support[0]:.4g}, {support[1]:.4g}], "
                  f"undefined={curve.n_undefined}")

        return curve

    def _recover_predictor(
            self,
            descriptor,
            X_norm: np.ndarray,
            y_norm: np.ndarray,
    ) -> tuple[Any, Any]:
        """
        Recover a callable predictor from a stored descriptor.

        Deep-copies the descriptor first, because
        :meth:`~ModelDescriptor.refit` writes the new fit back into the
        descriptor it is called on; without the copy this would overwrite
        the analyser's stored coefficients and silently corrupt every
        downstream coefficient plot.

        :param descriptor: Stored :class:`~ModelDescriptor` for this cohort.
        :param X_norm: The cohort's normalised design matrix, exactly as
            fitted.
        :type X_norm: numpy.ndarray
        :param y_norm: The cohort's normalised target vector.
        :type y_norm: numpy.ndarray
        :returns: ``(model, preprocessor)`` accepted by
            :meth:`~ModelFactory.predict`.
        :rtype: tuple
        """
        copy = self.analyzer._copy_descriptor(descriptor)
        return copy.refit(X_norm, y_norm, force_structure=True)

    def _fixed_design(
            self,
            x_grid: np.ndarray,
            swept_index: int,
            slice_values: dict[str, float],
            predictors: tuple[str, ...],
    ) -> np.ndarray:
        """
        Build the conditional-slice design matrix in original units.

        :param x_grid: Swept-predictor grid.
        :type x_grid: numpy.ndarray
        :param swept_index: Column index of the swept predictor.
        :type swept_index: int
        :param slice_values: Held values for the other predictors.
        :type slice_values: dict[str, float]
        :param predictors: Predictor names, aligned with columns.
        :type predictors: tuple[str, ...]
        :returns: Design matrix, shape ``(n_grid, n_predictors)``.
        :rtype: numpy.ndarray
        """
        design = np.empty((len(x_grid), len(predictors)), dtype=float)

        for idx, name in enumerate(predictors):
            design[:, idx] = (
                x_grid if idx == swept_index else slice_values[name]
            )

        return design

    def _marginal_designs(
            self,
            x_grid: np.ndarray,
            swept_index: int,
            X_reference: np.ndarray,
            marginal_max_rows: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Build one design matrix per reference row for partial dependence.

        Each block holds the covariates of a single reference ship and
        sweeps the chosen predictor across the whole grid; averaging the
        blocks' predictions integrates the model over the reference
        cohort's observed joint covariate distribution.

        :param x_grid: Swept-predictor grid.
        :type x_grid: numpy.ndarray
        :param swept_index: Column index of the swept predictor.
        :type swept_index: int
        :param X_reference: Reference cohort design matrix, original units.
        :type X_reference: numpy.ndarray
        :param marginal_max_rows: Maximum number of rows to average over;
            rows are subsampled evenly when the cohort is larger.
        :type marginal_max_rows: int
        :returns: ``(design_blocks, weights)``, with equal weights.
        :rtype: tuple[list[numpy.ndarray], list[float]]
        """
        n_rows = len(X_reference)

        if n_rows > marginal_max_rows:
            idx = np.linspace(0, n_rows - 1, marginal_max_rows).astype(int)
            rows = X_reference[idx]
        else:
            rows = X_reference

        blocks = []

        for row in rows:
            block = np.tile(row.astype(float), (len(x_grid), 1))
            block[:, swept_index] = x_grid
            blocks.append(block)

        return blocks, [1.0] * len(blocks)

    def _predict_original_scale(
            self,
            model,
            preprocessor,
            X_design: np.ndarray,
            model_type: str,
            norm_params: Optional[dict],
            norm_method: Optional[str],
            normalized: bool,
    ) -> np.ndarray:
        """
        Run one design block through the full normalisation round trip.

        The models were fitted on normalised data whose parameters were
        estimated on the full conventional cohort, so a grid in physical
        units must be pushed through the *same* transform before it can be
        shown to the model, and the prediction pulled back afterwards.
        Because all three cohorts share those parameters, the resulting
        curves are directly commensurable.

        Grid points at which the model is undefined — most commonly a
        z-scored predictor turning negative inside a log-domain form — are
        returned as ``NaN`` rather than dropped, so the output stays aligned
        with the grid and the gap is visible in the figure.

        :param model: Fitted model object.
        :param preprocessor: Fitted preprocessor, or ``None``.
        :param X_design: Design matrix in original units.
        :type X_design: numpy.ndarray
        :param model_type: Model-type identifier.
        :type model_type: str
        :param norm_params: Normalisation parameters.
        :type norm_params: dict, optional
        :param norm_method: Normalisation method.
        :type norm_method: str, optional
        :param normalized: Whether normalisation was in force.
        :type normalized: bool
        :returns: Predictions in original units, shape ``(n_rows,)``.
        :rtype: numpy.ndarray
        """
        n_rows = len(X_design)

        if normalized and norm_params is not None:
            X_norm, _, _ = self.analyzer.foundation._normalize_data(
                X_design, np.ones(n_rows), method=norm_method,
                fit_params=False, params=norm_params,
            )
        else:
            X_norm = X_design

        valid = np.isfinite(X_norm).all(axis=1)

        if self._requires_positive_domain(model_type):
            valid &= (X_norm > 0).all(axis=1)

        y_out = np.full(n_rows, np.nan, dtype=float)

        if not valid.any():
            return y_out

        try:
            y_norm = ModelFactory.predict(
                model, preprocessor, X_norm[valid], model_type,
            )
        except Exception:
            return y_out

        y_norm = np.asarray(y_norm, dtype=float).ravel()

        if normalized and norm_params is not None:
            y_pred = self.analyzer.foundation.denormalize_predictions(
                y_norm, norm_params, norm_method,
            )
        else:
            y_pred = y_norm

        y_out[valid] = y_pred

        # Belt and braces: a fractional power on a nominally valid input can
        # still overflow, and a NaN drawn as zero would be a false curve.
        y_out[~np.isfinite(y_out)] = np.nan

        return y_out

    def _requires_positive_domain(self, model_type: str) -> bool:
        """
        Decide whether a model type is undefined at non-positive inputs.

        :param model_type: Model-type identifier.
        :type model_type: str
        :returns: ``True`` when strictly positive predictors are required.
        :rtype: bool
        """
        return (
            'log' in model_type
            or model_type in self._POSITIVE_DOMAIN_HINTS
        )