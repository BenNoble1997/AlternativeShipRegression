"""
Significance Table LaTeX Generator (interval reporting)
=======================================================
Converts bootstrap significance results (as exported to Excel by
``TrainedModelBootstrapTester.export_to_excel``) into a publication-ready
LaTeX table in which each alternative-fuel cohort contributes a signed
``delta NRMSE`` together with its bootstrap confidence interval.

Rationale
---------
An earlier version of this generator reported the signed delta and marked
significance with a superscript asterisk, deferring the interval itself to
the caption. That convention compresses a continuous quantity into a binary
one: it tells the reader whether the interval excludes zero, but not how
narrowly, nor how precisely the delta is estimated. With small alternative
cohorts the width of the interval is frequently the more informative
quantity, and reviewers reasonably ask for it.

The interval is therefore rendered in the cell alongside the point estimate.
The asterisk becomes redundant, since an interval excluding zero is visible
on inspection, and is removed. Emphasis is retained: cells whose interval
excludes zero are emboldened, so the table can still be scanned for the
significant results without reading every bound.

Two consistency checks are run at generation time. Because the interval is
now printed, any disagreement between the ``is_significant`` flag and the
bounds becomes visible to the reader, and because the export may arrive in a
European locale where a mis-parsed decimal separator silently corrupts a
numeric column, that disagreement is a realistic failure mode rather than a
theoretical one. The generator therefore verifies that the flag agrees with
the bounds, and that each point estimate lies inside its own interval.

Design
------
Composition-friendly and standalone: it depends only on the exported results
frame plus optional per-target model metadata, not on any analyzer. The three
responsibilities are separated so each can be reused or overridden:

- :class:`SignificanceResultsLoader`   -- read and normalise the Excel export
- :class:`IntervalCellFormatter`       -- format one delta-with-interval cell
- :class:`SignificanceTableGenerator`  -- assemble the LaTeX table

The loader is deliberately tolerant of European-locale exports, in which
decimal separators are commas and numeric columns may arrive as text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import warnings


# =============================================================================
# Loading
# =============================================================================

class SignificanceResultsLoader:
    """
    Load and normalise a bootstrap significance export.

    Handles the two ways the export may arrive: with genuine numeric dtypes,
    or with European-locale text (comma decimal separators) that pandas reads
    as ``object``. Booleans may likewise arrive as the strings ``'TRUE'`` /
    ``'FALSE'``.

    :param numeric_columns: Columns coerced to float on load.
    :type numeric_columns: Sequence[str]
    :param boolean_columns: Columns coerced to nullable boolean on load.
    :type boolean_columns: Sequence[str]
    :param verbose: Whether to print a short load summary.
    :type verbose: bool
    """

    DEFAULT_NUMERIC = (
        'rrmse_a', 'rrmse_b', 'delta_rrmse',
        'ci_lower', 'ci_upper',
        'n_a', 'n_b', 'confidence_level', 'n_bootstrap',
    )
    DEFAULT_BOOLEAN = ('is_significant',)

    def __init__(self,
                 numeric_columns: Optional[Sequence[str]] = None,
                 boolean_columns: Optional[Sequence[str]] = None,
                 verbose: bool = True):
        self.numeric_columns = tuple(numeric_columns or self.DEFAULT_NUMERIC)
        self.boolean_columns = tuple(boolean_columns or self.DEFAULT_BOOLEAN)
        self.verbose = verbose

    def load(self,
             source: "str | Path | pd.DataFrame",
             sheet_name: "str | int" = 0,
             debug: bool = False) -> pd.DataFrame:
        """
        Load results from an Excel path or accept an in-memory frame.

        :param source: Path to the ``.xlsx`` export, or a DataFrame already in
            the exported shape.
        :type source: str or Path or pd.DataFrame
        :param sheet_name: Sheet to read when ``source`` is a path.
        :type sheet_name: str or int
        :param debug: Print per-column coercion detail.
        :type debug: bool
        :returns: Normalised results frame.
        :rtype: pd.DataFrame
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            if debug:
                print(f"[load] using in-memory frame: {df.shape}")
        else:
            df = pd.read_excel(source, sheet_name=sheet_name)
            if debug:
                print(f"[load] read {source} sheet={sheet_name}: {df.shape}")

        for col in self.numeric_columns:
            if col in df.columns:
                before = df[col].dtype
                df[col] = self._to_float(df[col])
                if debug:
                    n_nan = int(df[col].isna().sum())
                    print(f"  numeric '{col}': {before} -> float64 "
                          f"({n_nan} NaN)")

        for col in self.boolean_columns:
            if col in df.columns:
                before = df[col].dtype
                df[col] = self._to_bool(df[col])
                if debug:
                    print(f"  boolean '{col}': {before} -> boolean")

        if self.verbose:
            n_sig = int(df['is_significant'].fillna(False).sum()) \
                if 'is_significant' in df.columns else 0
            print(f"Loaded {len(df)} significance results "
                  f"({n_sig} with CI excluding zero)")

        return df

    @staticmethod
    def _to_float(series: pd.Series) -> pd.Series:
        """
        Coerce a column to float, tolerating comma decimal separators.

        Values already numeric pass through. Text values have thousands
        separators (spaces) stripped and a comma decimal separator replaced
        with a point before conversion. Unconvertible values become ``NaN``.

        :param series: Column to coerce.
        :type series: pd.Series
        :returns: Float column.
        :rtype: pd.Series
        """
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)

        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace('\u00a0', '', regex=False)   # non-breaking space
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        return pd.to_numeric(cleaned, errors='coerce')

    @staticmethod
    def _to_bool(series: pd.Series) -> pd.Series:
        """
        Coerce a column to nullable boolean.

        Accepts genuine booleans, and the strings ``TRUE``/``FALSE`` in any
        case. Anything else -- including blanks and the ``None`` written for
        an invalidated test -- becomes ``pd.NA``, so that "test not run" stays
        distinguishable from "not significant".

        :param series: Column to coerce.
        :type series: pd.Series
        :returns: Nullable boolean column.
        :rtype: pd.Series
        """
        if pd.api.types.is_bool_dtype(series):
            return series.astype('boolean')

        mapped = (
            series.astype(str)
            .str.strip()
            .str.upper()
            .map({'TRUE': True, 'FALSE': False, '1': True, '0': False,
                  '1.0': True, '0.0': False})
        )
        return mapped.astype('boolean')


# =============================================================================
# Cell formatting
# =============================================================================

class IntervalCellFormatter:
    """
    Format a single signed-delta cell with its bootstrap confidence interval.

    A cell shows the signed change in NRMSE followed by the interval bounds.
    When the interval excludes zero the point estimate is emboldened, so the
    table remains scannable without an asterisk convention.

    Bold weight in maths
    --------------------
    ``\\mathbf`` changes the alphabet for digits and letters but leaves signs,
    commas and brackets at normal weight. That is acceptable, and portable,
    when only the point estimate is emboldened. To embolden the interval as
    well, set ``bold_command='bm'`` and load the ``bm`` package, since
    ``\\mathbf`` would otherwise leave the brackets visibly lighter than their
    contents.

    :param delta_decimals: Decimal places shown for the point estimate.
    :type delta_decimals: int
    :param ci_decimals: Decimal places shown for the interval bounds.
    :type ci_decimals: int
    :param bold_scope: What to embolden when the interval excludes zero:
        ``'delta'`` (point estimate only), ``'all'`` (estimate and interval),
        or ``'none'``.
    :type bold_scope: str
    :param bold_command: Maths bold command, ``'mathbf'`` or ``'bm'``. Use
        ``'bm'`` with ``bold_scope='all'``.
    :type bold_command: str
    :param layout: ``'inline'`` places the interval after the estimate on one
        line. ``'stacked'`` places it beneath in a smaller size, which needs
        the ``makecell`` package and roughly halves the column width.
    :type layout: str
    :param explicit_delta_sign: Whether the point estimate carries an explicit
        ``+`` when positive. Bounds never do, by convention.
    :type explicit_delta_sign: bool
    :param missing_placeholder: Cell content when no result exists for a
        (row, cohort) pair.
    :type missing_placeholder: str
    :param invalid_placeholder: Cell content when a result exists but its
        significance is undetermined (``is_significant`` is null), e.g. a test
        invalidated by non-finite predictions.
    :type invalid_placeholder: str
    """

    _BOLD_SCOPES = ('delta', 'all', 'none')
    _LAYOUTS = ('inline', 'stacked')
    _BOLD_COMMANDS = ('mathbf', 'bm')

    def __init__(self,
                 delta_decimals: int = 2,
                 ci_decimals: int = 2,
                 bold_scope: str = 'delta',
                 bold_command: str = 'mathbf',
                 layout: str = 'inline',
                 explicit_delta_sign: bool = True,
                 missing_placeholder: str = '---',
                 invalid_placeholder: str = 'n.a.'):

        if bold_scope not in self._BOLD_SCOPES:
            raise ValueError(f"bold_scope must be one of {self._BOLD_SCOPES}")
        if layout not in self._LAYOUTS:
            raise ValueError(f"layout must be one of {self._LAYOUTS}")
        if bold_command not in self._BOLD_COMMANDS:
            raise ValueError(
                f"bold_command must be one of {self._BOLD_COMMANDS}")

        if bold_scope == 'all' and bold_command == 'mathbf':
            warnings.warn(
                "bold_scope='all' with bold_command='mathbf' leaves brackets "
                "and signs at normal weight. Use bold_command='bm' (and load "
                "the bm package) for a uniform bold cell.",
                stacklevel=2,
            )

        self.delta_decimals = delta_decimals
        self.ci_decimals = ci_decimals
        self.bold_scope = bold_scope
        self.bold_command = bold_command
        self.layout = layout
        self.explicit_delta_sign = explicit_delta_sign
        self.missing_placeholder = missing_placeholder
        self.invalid_placeholder = invalid_placeholder

    # -- public -------------------------------------------------------------

    def format_cell(self,
                    delta: Optional[float],
                    ci_lower: Optional[float],
                    ci_upper: Optional[float],
                    is_significant: Optional[bool]) -> str:
        """
        Render one delta-with-interval cell as LaTeX.

        Four states are distinguished:

        - no result for this pair -> ``missing_placeholder``
        - result present, significance undetermined -> ``invalid_placeholder``
        - result present, interval missing -> point estimate alone
        - result present -> estimate and interval, emboldened per
          ``bold_scope`` when the interval excludes zero

        :param delta: Signed change in NRMSE (percentage points), or ``None``.
        :type delta: float or None
        :param ci_lower: Lower interval bound, or ``None``.
        :type ci_lower: float or None
        :param ci_upper: Upper interval bound, or ``None``.
        :type ci_upper: float or None
        :param is_significant: Whether the interval excludes zero; ``None`` or
            ``pd.NA`` when undetermined.
        :type is_significant: bool or None
        :returns: LaTeX for the cell (already math-wrapped).
        :rtype: str
        """
        if not self._is_finite(delta):
            return self.missing_placeholder

        if is_significant is None or pd.isna(is_significant):
            return self.invalid_placeholder

        significant = bool(is_significant)
        emphasise = significant and self.bold_scope != 'none'

        delta_body = self._format_number(
            delta, self.delta_decimals, signed=self.explicit_delta_sign)

        if emphasise:
            delta_body = self._bold(delta_body)

        # An interval may legitimately be absent (e.g. a degenerate resample);
        # fall back to the point estimate rather than emitting empty brackets.
        if not (self._is_finite(ci_lower) and self._is_finite(ci_upper)):
            return f"${delta_body}$"

        interval_body = self._format_interval(ci_lower, ci_upper)

        if emphasise and self.bold_scope == 'all':
            interval_body = self._bold(interval_body)

        if self.layout == 'stacked':
            return (rf"\makecell[r]{{${delta_body}$\\"
                    rf"\scriptsize ${interval_body}$}}")

        return rf"${delta_body}_{{{interval_body}}}$"

    # -- internals ----------------------------------------------------------

    def _format_interval(self, ci_lower: float, ci_upper: float) -> str:
        """
        Render the interval bounds as a bracketed pair.

        Bounds are written without an explicit positive sign, which is the
        usual convention and keeps the cell narrower; the point estimate
        already carries the sign that matters for interpretation.

        :param ci_lower: Lower bound.
        :type ci_lower: float
        :param ci_upper: Upper bound.
        :type ci_upper: float
        :returns: LaTeX maths fragment, without surrounding ``$``.
        :rtype: str
        """
        low = self._format_number(ci_lower, self.ci_decimals, signed=False)
        high = self._format_number(ci_upper, self.ci_decimals, signed=False)
        return rf"[{low},\,{high}]"

    def _bold(self, body: str) -> str:
        """
        Wrap a maths fragment in the configured bold command.

        :param body: Maths fragment, without surrounding ``$``.
        :type body: str
        :returns: Emboldened fragment.
        :rtype: str
        """
        return rf"\{self.bold_command}{{{body}}}"

    @staticmethod
    def _format_number(value: float, decimals: int, signed: bool) -> str:
        """
        Format a number for maths mode, using a proper minus sign.

        Python renders a negative as an ASCII hyphen. Inside ``$...$`` TeX
        already sets that as a minus, so no substitution is needed; the method
        exists to keep sign handling in one place and to guard against a
        negative zero appearing when a small value rounds down.

        :param value: Value to format.
        :type value: float
        :param decimals: Decimal places.
        :type decimals: int
        :param signed: Whether to force an explicit ``+`` on positives.
        :type signed: bool
        :returns: Formatted number.
        :rtype: str
        """
        rendered = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"

        # -0.00 is an artefact of rounding, not a meaningful sign.
        if rendered.lstrip('+-').strip('0.') == '':
            rendered = f"{0.0:+.{decimals}f}" if signed else f"{0.0:.{decimals}f}"

        return rendered

    @staticmethod
    def _is_finite(value) -> bool:
        """
        Whether a value is present and finite.

        :param value: Candidate value.
        :type value: object
        :returns: True when the value is a finite number.
        :rtype: bool
        """
        if value is None:
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False


# =============================================================================
# Table assembly
# =============================================================================

class SignificanceTableGenerator:
    """
    Assemble a LaTeX table of model specifications and per-cohort deltas.

    Composes a :class:`SignificanceResultsLoader` and an
    :class:`IntervalCellFormatter`; either may be supplied to override the
    defaults.

    Step-by-step data flow:

    1. ``loader.load()``            -> normalised results frame
    2. ``_validate_row_key()``      -> the key identifies one row per cohort
    3. ``_validate_intervals()``    -> flags agree with bounds; deltas inside
    4. ``_pivot_by_fuel()``         -> one row per target, one entry per
                                       (row, cohort) carrying delta and bounds
    5. ``model_metadata`` merge     -> predictors / model / R2 / NRMSE columns
    6. ``formatter.format_cell()``  -> LaTeX for each delta cell
    7. ``_render()``                -> complete ``table*`` environment

    :param loader: Results loader (default constructed if omitted).
    :type loader: SignificanceResultsLoader or None
    :param formatter: Cell formatter (default constructed if omitted).
    :type formatter: IntervalCellFormatter or None
    :param target_labels: Mapping of raw target name to LaTeX label, e.g.
        ``{'BEAM (M)': r'\\gls{b}'}``. Unmapped targets are escaped verbatim.
    :type target_labels: dict[str, str] or None
    :param verbose: Whether to print progress.
    :type verbose: bool
    """

    METADATA_SPEC = (
        ('predictors', 'Predictors',      'text'),
        ('model',      'Model',           'text'),
        ('reference',  'Ref.',            'text'),
        ('r2_adj',     r'$R^{2}_{adj}$',  'r2'),
        ('nrmse_test', r'NRMSE$_{test}$', 'nrmse'),
    )

    def __init__(self,
                 loader: Optional[SignificanceResultsLoader] = None,
                 formatter: Optional[IntervalCellFormatter] = None,
                 target_labels: Optional[Dict[str, str]] = None,
                 verbose: bool = True):
        self.loader = loader or SignificanceResultsLoader(verbose=verbose)
        self.formatter = formatter or IntervalCellFormatter()
        self.target_labels = target_labels or {}
        self.verbose = verbose

    # -- public -------------------------------------------------------------

    def generate(self,
                 source: "str | Path | pd.DataFrame",
                 caption: str,
                 label: str,
                 fuel_order: Optional[List[str]] = None,
                 fuel_labels: Optional[Dict[str, str]] = None,
                 row_order: Optional[List] = None,
                 row_key: "str | Sequence[str]" = 'target',
                 row_header: str = 'Target',
                 model_metadata: Optional[pd.DataFrame] = None,
                 sheet_name: "str | int" = 0,
                 use_table_star: bool = True,
                 nrmse_macro: Optional[str] = r'\NRMSE',
                 show_cohort_sizes: bool = True,
                 tabcolsep: str = '4pt',
                 strict_intervals: bool = False,
                 debug: bool = False) -> str:
        """
        Generate the complete LaTeX table.

        The unit of a table row is given by ``row_key``. For selected-model
        tables this is ``'target'``. For empirical-formula tables it must
        identify each formula uniquely; where two sources publish the same
        functional form the name alone is insufficient, and a composite key
        such as ``('formula_name', 'reference')`` should be supplied.

        When ``row_key`` names several columns, a row is identified by the
        tuple of their values, and ``row_order``, ``target_labels`` and the
        ``model_metadata`` index are all keyed on that same tuple.

        :param source: Excel path or in-memory results frame.
        :type source: str or Path or pd.DataFrame
        :param caption: Table caption (LaTeX; not escaped).
        :type caption: str
        :param label: Table label.
        :type label: str
        :param fuel_order: Cohort column order. Defaults to order of first
            appearance in the results.
        :type fuel_order: list[str] or None
        :param fuel_labels: Display names for cohorts, e.g.
            ``{'LNG': 'LNG dual fuel'}``. Unmapped cohorts are escaped
            verbatim.
        :type fuel_labels: dict[str, str] or None
        :param row_order: Row order, given as ``row_key`` values (tuples when
            ``row_key`` is composite). Defaults to order of first appearance.
        :type row_order: list or None
        :param row_key: Column, or sequence of columns, whose value(s)
            identify a table row. Must be unique within each fuel cohort.
        :type row_key: str or Sequence[str]
        :param row_header: Header text for the first column.
        :type row_header: str
        :param model_metadata: Optional specification columns, indexed by
            ``row_key`` value (or tuple), with any of ``predictors``,
            ``model``, ``r2_adj``, ``nrmse_test``.
        :type model_metadata: pd.DataFrame or None
        :param sheet_name: Sheet to read when ``source`` is a path.
        :type sheet_name: str or int
        :param use_table_star: Emit ``table*`` rather than ``table``.
        :type use_table_star: bool
        :param nrmse_macro: Macro wrapped around the test NRMSE value, or
            ``None`` for a plain number.
        :type nrmse_macro: str or None
        :param show_cohort_sizes: Append ``($n=\\,$...)`` to each cohort
            header, read from the ``n_b`` column.
        :type show_cohort_sizes: bool
        :param tabcolsep: Inter-column padding. Interval cells are wide, so a
            smaller value than the LaTeX default is usually needed.
        :type tabcolsep: str
        :param strict_intervals: Raise rather than warn when a significance
            flag disagrees with its bounds, or a delta falls outside its own
            interval.
        :type strict_intervals: bool
        :param debug: Print step-by-step progress.
        :type debug: bool
        :returns: Complete LaTeX table environment.
        :rtype: str
        :raises ValueError: If required columns are absent, if the ``row_key``
            values are not unique within a fuel cohort, or if
            ``strict_intervals`` is set and an interval inconsistency is found.
        """
        # -- Step 1: load ----------------------------------------------------
        df = self.loader.load(source, sheet_name=sheet_name, debug=debug)

        key_columns = self._as_key_columns(row_key)

        required = set(key_columns) | {'fuel_type', 'delta_rrmse',
                                       'ci_lower', 'ci_upper',
                                       'is_significant'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"results are missing required column(s): {sorted(missing)}"
            )

        # -- Step 2: verify the key actually identifies a row ----------------
        self._validate_row_key(df, key_columns, debug=debug)

        # -- Step 3: verify flags and bounds tell the same story -------------
        self._validate_intervals(df, key_columns,
                                 strict=strict_intervals, debug=debug)

        # -- Step 4: pivot ---------------------------------------------------
        fuels = fuel_order or list(dict.fromkeys(df['fuel_type']))
        row_ids = row_order or list(dict.fromkeys(
            self._row_ids(df, key_columns)
        ))

        if debug:
            print(f"[generate] {len(row_ids)} rows x {len(fuels)} cohorts "
                  f"(row_key={key_columns})")

        lookup = self._pivot_by_fuel(df, key_columns=key_columns, debug=debug)
        metadata_columns = self._resolve_metadata_columns(model_metadata)

        # -- Steps 5-6: build rows -------------------------------------------
        rows = []
        for row_id in row_ids:
            cells = [self._row_label(row_id)]
            cells.extend(self._metadata_cells(row_id, model_metadata,
                                              nrmse_macro, metadata_columns))
            for fuel in fuels:
                entry = lookup.get((row_id, fuel))
                if entry is None:
                    cells.append(self.formatter.missing_placeholder)
                    if debug:
                        print(f"  no result: {row_id} / {fuel}")
                else:
                    cells.append(self.formatter.format_cell(
                        entry['delta_rrmse'],
                        entry['ci_lower'],
                        entry['ci_upper'],
                        entry['is_significant'],
                    ))
            rows.append(cells)

        # -- Step 7: render --------------------------------------------------
        latex = self._render(rows, fuels, caption, label,
                             use_table_star=use_table_star,
                             metadata_columns=metadata_columns,
                             row_header=row_header,
                             confidence_level=self._confidence_level(df),
                             cohort_sizes=(self._cohort_sizes(df)
                                           if show_cohort_sizes else None),
                             fuel_labels=fuel_labels or {},
                             tabcolsep=tabcolsep,
                             debug=debug)

        if self.verbose:
            print(f"Generated table '{label}' "
                  f"({len(rows)} rows, {len(fuels)} cohort columns)")

        return latex

    def generate_to_file(self,
                         output_path: "str | Path",
                         **kwargs) -> str:
        """
        Generate the table and write it to a ``.tex`` file.

        :param output_path: Destination path; parent directories are created.
        :type output_path: str or Path
        :param kwargs: Passed through to :meth:`generate`.
        :returns: The generated LaTeX.
        :rtype: str
        """
        latex = self.generate(**kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding='utf-8')
        if self.verbose:
            print(f"Saved to: {output_path}")
        return latex

    # -- validation ---------------------------------------------------------

    def _validate_row_key(self,
                          df: pd.DataFrame,
                          key_columns: List[str],
                          debug: bool = False) -> None:
        """
        Verify that the row key uniquely identifies a row within each cohort.

        Silently collapsing duplicates would discard results without warning,
        so a non-unique key is treated as a configuration error. The most
        common causes are keying an empirical-formula table on ``'target'``
        when several formulae share a target, or on ``'formula_name'`` alone
        when two sources publish the same functional form.

        The error message names the additional columns that would disambiguate
        the offending rows, so the fix is directly actionable.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :param key_columns: Columns forming the row identifier.
        :type key_columns: list[str]
        :param debug: Print the per-cohort uniqueness check.
        :type debug: bool
        :raises ValueError: If any (row identifier, cohort) pair occurs more
            than once.
        """
        counts = df.groupby(['fuel_type'] + key_columns).size()
        duplicated = counts[counts > 1]

        if debug:
            print(f"[validate] key {key_columns}: {len(counts)} pairs, "
                  f"{len(duplicated)} duplicated")

        if len(duplicated) == 0:
            return

        # Identify which other columns vary within the colliding groups, and
        # would therefore disambiguate them if added to the key.
        collided = df.groupby(key_columns).filter(lambda g: len(g) > len(
            g['fuel_type'].unique()
        ))
        candidates = []
        if not collided.empty:
            for col in df.columns:
                if col in key_columns or col == 'fuel_type':
                    continue
                varies = collided.groupby(key_columns)[col].nunique()
                if (varies > 1).any():
                    candidates.append(col)

        examples = ', '.join(
            f"{key[1:] if len(key) > 2 else key[1]!r} in {key[0]!r} ({n}x)"
            for key, n in duplicated.head(3).items()
        )
        hint = (f" Adding {candidates[:3]} to row_key would disambiguate them."
                if candidates else
                " No other column distinguishes these rows; they may be "
                "genuine duplicates in the export.")

        raise ValueError(
            f"row_key={key_columns} is not unique within a fuel cohort: "
            f"{len(duplicated)} duplicated pair(s), e.g. {examples}. "
            f"Rows would be silently overwritten.{hint}"
        )

    def _validate_intervals(self,
                            df: pd.DataFrame,
                            key_columns: List[str],
                            strict: bool = False,
                            debug: bool = False) -> None:
        """
        Check that the significance flags and the interval bounds agree.

        Two conditions are tested:

        1. ``is_significant`` matches ``ci_lower > 0 or ci_upper < 0``. While
           significance was marked only by an asterisk a disagreement was
           invisible; once the bounds are printed, a reader can check it, so
           it must be checked here first.
        2. The point estimate lies within its own interval. A violation is
           almost always a parsing fault rather than a statistical one, since
           a comma decimal separator surviving into one column but not another
           produces exactly this signature.

        Rows with a null flag are skipped: an undetermined test is a known
        state, not an inconsistency.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :param key_columns: Columns forming the row identifier, used only to
            make the diagnostic message locatable.
        :type key_columns: list[str]
        :param strict: Raise instead of warning.
        :type strict: bool
        :param debug: Print the counts checked.
        :type debug: bool
        :raises ValueError: If ``strict`` and any check fails.
        """
        testable = df[df['is_significant'].notna()
                      & df['ci_lower'].notna()
                      & df['ci_upper'].notna()
                      & df['delta_rrmse'].notna()]

        if testable.empty:
            if debug:
                print("[validate] no rows with complete interval data")
            return

        excludes_zero = (testable['ci_lower'] > 0) | (testable['ci_upper'] < 0)
        flag_mismatch = testable[
            excludes_zero != testable['is_significant'].astype(bool)]

        outside = testable[
            (testable['delta_rrmse'] < testable['ci_lower'])
            | (testable['delta_rrmse'] > testable['ci_upper'])]

        if debug:
            print(f"[validate] {len(testable)} rows checked: "
                  f"{len(flag_mismatch)} flag mismatch, "
                  f"{len(outside)} delta outside interval")

        problems = []
        if not flag_mismatch.empty:
            problems.append(
                f"{len(flag_mismatch)} row(s) where is_significant disagrees "
                f"with the bounds, e.g. "
                f"{self._describe_rows(flag_mismatch, key_columns)}"
            )
        if not outside.empty:
            problems.append(
                f"{len(outside)} row(s) where delta_rrmse falls outside its "
                f"own interval, e.g. "
                f"{self._describe_rows(outside, key_columns)}"
            )

        if not problems:
            return

        message = ("interval consistency check failed: "
                   + '; '.join(problems)
                   + ". Since the bounds are now printed in the table, a "
                     "reader can see this. Check the export for a locale or "
                     "column-alignment fault before publishing.")

        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)

    @staticmethod
    def _describe_rows(subset: pd.DataFrame,
                       key_columns: List[str],
                       limit: int = 3) -> str:
        """
        Summarise offending rows for a diagnostic message.

        :param subset: Rows that failed a check.
        :type subset: pd.DataFrame
        :param key_columns: Columns forming the row identifier.
        :type key_columns: list[str]
        :param limit: Maximum number of rows described.
        :type limit: int
        :returns: Human-readable description.
        :rtype: str
        """
        parts = []
        for _, row in subset.head(limit).iterrows():
            ident = ' / '.join(str(row[c]) for c in key_columns)
            parts.append(
                f"{ident} in {row['fuel_type']!r} "
                f"(delta={row['delta_rrmse']:.3f}, "
                f"CI=[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
                f"flag={row['is_significant']})"
            )
        return '; '.join(parts)

    # -- data shaping -------------------------------------------------------

    def _pivot_by_fuel(self,
                       df: pd.DataFrame,
                       key_columns: List[str],
                       debug: bool = False) -> Dict[tuple, Dict]:
        """
        Index results by ``(row identifier, fuel_type)``.

        A dict keyed on the pair is used rather than a pivoted frame so that
        the delta, its bounds and its significance flag stay associated, and
        so that a missing pair is an explicit absence rather than a ``NaN``
        that could be confused with an invalidated test. Uniqueness of the key
        is enforced upstream by :meth:`_validate_row_key`.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :param key_columns: Columns forming the row identifier.
        :type key_columns: list[str]
        :param debug: Print the number of indexed pairs.
        :type debug: bool
        :returns: Mapping of ``(row_id, fuel_type)`` to a dict with
            ``delta_rrmse``, ``ci_lower``, ``ci_upper`` and ``is_significant``.
        :rtype: dict
        """
        row_ids = self._row_ids(df, key_columns)

        lookup: Dict[tuple, Dict] = {}
        for row_id, (_, row) in zip(row_ids, df.iterrows()):
            lookup[(row_id, row['fuel_type'])] = {
                'delta_rrmse': row['delta_rrmse'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'is_significant': row['is_significant'],
            }

        if debug:
            print(f"  indexed {len(lookup)} (row, cohort) pairs")

        return lookup

    @staticmethod
    def _confidence_level(df: pd.DataFrame) -> Optional[float]:
        """
        Read the confidence level from the results, if unambiguous.

        The level is a property of the analysis rather than of any one row, so
        it is taken from the data instead of being restated by the caller.
        A frame mixing levels returns ``None``, and the header then omits the
        percentage rather than asserting a wrong one.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :returns: Confidence level as a fraction, or ``None``.
        :rtype: float or None
        """
        if 'confidence_level' not in df.columns:
            return None

        levels = df['confidence_level'].dropna().unique()
        if len(levels) != 1:
            return None

        return float(levels[0])

    @staticmethod
    def _cohort_sizes(df: pd.DataFrame) -> Dict[str, int]:
        """
        Read the alternative-cohort size for each fuel type.

        ``n_b`` is the alternative-fuel sample in each comparison. Cohorts
        whose size varies across rows are omitted rather than averaged, since
        a single number would misdescribe them.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :returns: Mapping of fuel type to cohort size.
        :rtype: dict[str, int]
        """
        if 'n_b' not in df.columns:
            return {}

        sizes = {}
        for fuel, group in df.groupby('fuel_type'):
            values = group['n_b'].dropna().unique()
            if len(values) == 1:
                sizes[str(fuel)] = int(values[0])

        return sizes

    def _row_label(self, row_id) -> str:
        """
        Map a row identifier to its LaTeX label.

        The identifier is a scalar for a single key column and a tuple for a
        composite key; ``target_labels`` may be keyed on either. An unmapped
        tuple falls back to its components joined by a space, each escaped, so
        an absent label degrades to something readable rather than raising.

        :param row_id: Row identifier (scalar or tuple).
        :type row_id: object
        :returns: LaTeX label.
        :rtype: str
        """
        if row_id in self.target_labels:
            return self.target_labels[row_id]

        if isinstance(row_id, tuple):
            return ' '.join(self._escape(str(part)) for part in row_id)

        return self._escape(str(row_id))

    @staticmethod
    def _as_key_columns(row_key: "str | Sequence[str]") -> List[str]:
        """
        Normalise ``row_key`` to a list of column names.

        A bare string is wrapped so that single and composite keys share one
        code path downstream.

        :param row_key: Column name, or sequence of column names.
        :type row_key: str or Sequence[str]
        :returns: Column names as a list.
        :rtype: list[str]
        """
        if isinstance(row_key, str):
            return [row_key]
        return list(row_key)

    @staticmethod
    def _row_ids(df: pd.DataFrame, key_columns: List[str]) -> List:
        """
        Extract the row identifier for every row of the results frame.

        A single key column yields scalar identifiers; a composite key yields
        tuples, so that identifiers remain hashable and usable as dict keys.

        :param df: Normalised results frame.
        :type df: pd.DataFrame
        :param key_columns: Columns forming the row identifier.
        :type key_columns: list[str]
        :returns: One identifier per row, in frame order.
        :rtype: list
        """
        if len(key_columns) == 1:
            return df[key_columns[0]].tolist()
        return list(df[key_columns].itertuples(index=False, name=None))

    # -- metadata -----------------------------------------------------------

    def _metadata_cells(self,
                        row_id,
                        model_metadata: Optional[pd.DataFrame],
                        nrmse_macro: Optional[str],
                        metadata_columns: Optional[Sequence[str]] = None
                        ) -> List[str]:
        """
        Build the specification cells for one row.

        Only those recognised columns actually present in ``model_metadata``
        are emitted, so a metadata frame carrying just ``predictors`` and
        ``reference`` produces two cells rather than five with three
        placeholders.

        :param row_id: Row identifier; a scalar for a single key column, or a
            tuple for a composite key. Must match the metadata index.
        :type row_id: object
        :param model_metadata: Per-row metadata frame, or ``None``.
        :type model_metadata: pd.DataFrame or None
        :param nrmse_macro: Macro wrapped around the NRMSE value, or ``None``.
        :type nrmse_macro: str or None
        :param metadata_columns: Resolved column list from
            :meth:`_resolve_metadata_columns`; recomputed if omitted.
        :type metadata_columns: Sequence[str] or None
        :returns: One cell per resolved metadata column (empty when no
            metadata was supplied).
        :rtype: list[str]
        """
        if model_metadata is None:
            return []

        if metadata_columns is None:
            metadata_columns = self._resolve_metadata_columns(model_metadata)

        placeholder = self.formatter.missing_placeholder

        if row_id not in model_metadata.index:
            return [placeholder] * len(metadata_columns)

        meta = model_metadata.loc[row_id]

        # A duplicated index entry yields a frame rather than a Series; take
        # the first row so a stray duplicate degrades gracefully instead of
        # raising deep inside the formatting below.
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        formatters = {name: kind for name, _, kind in self.METADATA_SPEC}

        cells = []
        for col in metadata_columns:
            value = meta.get(col)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append(placeholder)
                continue

            kind = formatters[col]
            if kind == 'r2':
                cells.append(f"{float(value):.3f}")
            elif kind == 'nrmse':
                cells.append(f"{nrmse_macro}{{{float(value):.2f}}}"
                             if nrmse_macro else f"{float(value):.2f}")
            else:
                cells.append(self._escape(str(value)))

        return cells

    def _resolve_metadata_columns(self,
                                  model_metadata: Optional[pd.DataFrame]
                                  ) -> List[str]:
        """
        Determine which recognised metadata columns are present.

        Columns not listed in :attr:`METADATA_SPEC` are ignored, so a metadata
        frame carrying extra bookkeeping columns renders only the intended
        specification fields.

        :param model_metadata: Per-row metadata frame, or ``None``.
        :type model_metadata: pd.DataFrame or None
        :returns: Recognised column names present in the frame, in render
            order; empty when no metadata was supplied.
        :rtype: list[str]
        """
        if model_metadata is None:
            return []
        return [name for name, _, _ in self.METADATA_SPEC
                if name in model_metadata.columns]

    def _metadata_headers(self,
                          metadata_columns: Sequence[str]) -> List[str]:
        """
        Header labels for the resolved metadata columns.

        :param metadata_columns: Resolved column names, as returned by
            :meth:`_resolve_metadata_columns`.
        :type metadata_columns: Sequence[str]
        :returns: LaTeX header labels, in the same order.
        :rtype: list[str]
        """
        labels = {name: header for name, header, _ in self.METADATA_SPEC}
        return [labels[col] for col in metadata_columns]

    # -- rendering ----------------------------------------------------------

    def _render(self,
                rows: List[List[str]],
                fuels: List[str],
                caption: str,
                label: str,
                use_table_star: bool,
                metadata_columns: Sequence[str],
                row_header: str = 'Target',
                confidence_level: Optional[float] = None,
                cohort_sizes: Optional[Dict[str, int]] = None,
                fuel_labels: Optional[Dict[str, str]] = None,
                tabcolsep: str = '4pt',
                debug: bool = False) -> str:
        """
        Assemble the LaTeX table environment from prepared rows.

        The column specification is derived from the resolved metadata
        columns: text fields are left-aligned, numeric fields left-aligned
        with a right-aligned habit for the deltas. Interval cells are wider
        than plain numbers and of uniform width, so they are centred rather
        than right-aligned; right alignment would leave the point estimates
        ragged because the bounds vary in digit count.

        :param rows: Prepared cell content, one list per table row.
        :type rows: list[list[str]]
        :param fuels: Cohort names, in column order.
        :type fuels: list[str]
        :param caption: Table caption.
        :type caption: str
        :param label: Table label.
        :type label: str
        :param use_table_star: Emit ``table*`` rather than ``table``.
        :type use_table_star: bool
        :param metadata_columns: Resolved metadata column names.
        :type metadata_columns: Sequence[str]
        :param row_header: Header text for the first column.
        :type row_header: str
        :param confidence_level: Level as a fraction, for the spanning header.
        :type confidence_level: float or None
        :param cohort_sizes: Fuel type to cohort size, for the column headers.
        :type cohort_sizes: dict[str, int] or None
        :param fuel_labels: Display names for cohorts.
        :type fuel_labels: dict[str, str] or None
        :param tabcolsep: Inter-column padding.
        :type tabcolsep: str
        :param debug: Print the resolved column specification.
        :type debug: bool
        :returns: Complete LaTeX table.
        :rtype: str
        """
        env = 'table*' if use_table_star else 'table'
        fuel_labels = fuel_labels or {}
        cohort_sizes = cohort_sizes or {}

        alignment = {name: ('r' if kind in ('r2', 'nrmse') else 'l')
                     for name, _, kind in self.METADATA_SPEC}
        meta_spec = ''.join(alignment[col] for col in metadata_columns)

        col_spec = 'l' + meta_spec + 'c' * len(fuels)
        n_lead = 1 + len(metadata_columns)

        if debug:
            print(f"[render] column spec '{col_spec}' "
                  f"({n_lead} leading + {len(fuels)} cohort)")

        lines = [
            rf'\begin{{{env}}}[t]',
            r'\centering',
            rf'\caption{{{caption}}}',
            r'\footnotesize',
            rf'\setlength{{\tabcolsep}}{{{tabcolsep}}}',
            rf'\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
        ]

        if confidence_level is not None:
            ci_label = (rf'$\Delta_{{NRMSE}}$ (\%) with '
                        rf'{confidence_level * 100:g}\% bootstrap CI')
        else:
            ci_label = r'$\Delta_{NRMSE}$ (\%) with bootstrap CI'

        span = rf'\multicolumn{{{len(fuels)}}}{{c}}{{{ci_label}}}'
        lines.append(' & '.join([''] * n_lead) + f' & {span}\\\\')
        lines.append(rf'\cmidrule(lr){{{n_lead + 1}-{n_lead + len(fuels)}}}')

        header = ([row_header]
                  + self._metadata_headers(metadata_columns)
                  + [self._fuel_header(f, fuel_labels, cohort_sizes)
                     for f in fuels])
        lines.append(' & '.join(header) + r'\\')
        lines.append(r'\midrule')

        for row in rows:
            lines.append(' & '.join(row) + r'\\')

        lines += [
            r'\bottomrule',
            r'\end{tabular}',
            rf'\label{{{label}}}',
            rf'\end{{{env}}}',
        ]

        return '\n'.join(lines)

    def _fuel_header(self,
                     fuel: str,
                     fuel_labels: Dict[str, str],
                     cohort_sizes: Dict[str, int]) -> str:
        """
        Build the header cell for one cohort column.

        A display label is used when supplied and the raw name escaped
        otherwise. The cohort size is appended when known, which puts the
        sample behind each interval next to the interval itself rather than
        in the caption.

        :param fuel: Raw cohort name.
        :type fuel: str
        :param fuel_labels: Display names for cohorts.
        :type fuel_labels: dict[str, str]
        :param cohort_sizes: Fuel type to cohort size.
        :type cohort_sizes: dict[str, int]
        :returns: LaTeX header cell.
        :rtype: str
        """
        label = fuel_labels.get(fuel, self._escape(str(fuel)))

        size = cohort_sizes.get(str(fuel))
        if size is None:
            return label

        return rf'{label} ($n={size}$)'

    @staticmethod
    def _escape(text: str) -> str:
        """
        Escape LaTeX special characters in plain text.

        :param text: Raw text.
        :type text: str
        :returns: Escaped text safe for a table cell.
        :rtype: str
        """
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text