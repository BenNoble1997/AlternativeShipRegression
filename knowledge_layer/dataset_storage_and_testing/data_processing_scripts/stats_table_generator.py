"""
Generate LaTeX descriptive-statistics tables for fleet datasets.

Produces, for each ship type, a table giving the sample size, central tendency
and spread of every design variable within each fuel-type dataset. The table
supplies the context that cross-dataset performance tables cannot carry: it
shows how dispersed each cohort is and where it sits relative to the
conventional fleet, which is what determines whether an :math:`R^2` figure is
low because the model is poor or because the cohort is narrow.

Two dispersion measures are reported, because they answer different questions:

``CV``
    :math:`\\sigma_d / \\bar{y}_d`, the ordinary coefficient of variation.
    Unitless, so comparable across variables. Reports how homogeneous a cohort
    is in its own terms.

``S``
    :math:`\\sigma_d / \\bar{y}_{ref}`, the cohort's standard deviation
    expressed on the *reference* cohort's scale. This is the quantity that
    links a performance table's two metric columns when NRMSE is normalised by
    a single reference mean across all cohorts: for a target variable,
    :math:`1 - R^2 \\approx (\\mathrm{NRMSE}/S)^2`. It differs from ``CV`` by
    the factor :math:`\\bar{y}_d / \\bar{y}_{ref}`, so it carries both the
    spread of the cohort and its position relative to the reference fleet.

``S`` is interpretable only where the variable acts as a regression target.
Where it acts as a predictor, read the min and max columns instead, since the
relevant question there is the span over which coefficients are identified,
not a normalised dispersion.

Requires the following in the LaTeX preamble::

    \\usepackage{booktabs}
    \\usepackage[table]{xcolor}
    \\definecolor{myred}{RGB}{237,106,90}
    \\definecolor{myorange}{RGB}{255,203,119}
    \\newcommand{\\lowdata}[1]{\\cellcolor{myorange}#1}

The module separates computation from display, mirroring
``data_availability_table_generator``:

* :func:`build_descriptive_stats` produces a tidy ``DataFrame`` of statistics
  that can be inspected, tested, or reused independently.
* :func:`render_descriptive_latex` converts that frame into the final LaTeX
  string, applying the formatting and flagging rules.
* :func:`generate_descriptive_table` chains the two and optionally writes to
  disk.
* :func:`generate_descriptive_tables_per_ship_type` produces one table per
  ship type, which is the layout intended for an appendix.

Standard deviations use ``ddof=1``. Statistics are computed on the non-null
observations of each variable independently, so ``n`` may differ between rows
of the same cohort; this matches how the models are fitted, since each
target-predictor pair drops its own incomplete rows.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Statistic columns in report order, mapped to their default LaTeX headers.
DEFAULT_STAT_HEADERS = {
    "n": r"$n$",
    "mean": "Mean",
    "std": "SD",
    "min": "Min",
    "max": "Max",
    "cv": "CV",
    "s_ref": "$S$",
}


def build_descriptive_stats(datasets: dict[str, dict[str, pd.DataFrame]],
                            variables: list[str],
                            reference_dataset: str | None = None,
                            ) -> pd.DataFrame:
    """
    Compute per-variable descriptive statistics across grouped datasets.

    A variable absent from a dataset's columns is treated identically to one
    present but entirely null: both yield ``n = 0`` and null statistics, since
    in either case nothing is available for modelling.

    :param datasets: Nested mapping ``{ship_type: {dataset_label: DataFrame}}``.
        Insertion order is preserved and determines row order in the output.
    :type datasets: dict[str, dict[str, pd.DataFrame]]
    :param variables: Column names to summarise, in the row order desired for
        the final table.
    :type variables: list[str]
    :param reference_dataset: Dataset label whose mean normalises the ``s_ref``
        column, typically the conventional cohort. Must be present in every
        ship type group. If ``None``, the first dataset label of each group is
        used and ``s_ref`` is computed within that ship type.
    :type reference_dataset: str or None
    :returns: Tidy frame with one row per ``(ship_type, variable, dataset)``
        and columns ``n``, ``mean``, ``std``, ``min``, ``max``, ``cv``,
        ``s_ref``.
    :rtype: pd.DataFrame
    :raises ValueError: If ``datasets`` or ``variables`` is empty, or if
        ``reference_dataset`` is missing from a ship type group.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one ship type group")
    if not variables:
        raise ValueError("variables must contain at least one column name")

    records: list[dict] = []

    for ship_type, group in datasets.items():
        if not group:
            raise ValueError(f"ship type '{ship_type}' has no datasets assigned")

        reference = reference_dataset or next(iter(group))
        if reference not in group:
            raise ValueError(
                f"reference dataset '{reference}' not found for ship type "
                f"'{ship_type}'. Available: {list(group)}"
            )

        # Reference means are resolved per ship type, so S is always expressed
        # on the scale of the fleet the models were normalised against.
        ref_df = group[reference]
        ref_means: dict[str, float] = {}
        for variable in variables:
            if variable in ref_df.columns:
                values = ref_df[variable].dropna()
                ref_means[variable] = float(values.mean()) if len(values) else np.nan
            else:
                ref_means[variable] = np.nan

        for dataset_label, df in group.items():
            for variable in variables:
                if variable not in df.columns:
                    logger.debug(
                        "Variable '%s' not present in %s / %s; recording n=0",
                        variable, ship_type, dataset_label,
                    )
                    values = pd.Series(dtype=float)
                else:
                    values = pd.to_numeric(df[variable], errors="coerce").dropna()

                n = int(len(values))
                mean = float(values.mean()) if n else np.nan
                # ddof=1 needs at least two observations; a single-vessel
                # cohort has no defined spread and is reported as null rather
                # than as zero, which would falsely imply homogeneity.
                std = float(values.std(ddof=1)) if n > 1 else np.nan

                ref_mean = ref_means.get(variable, np.nan)

                records.append({
                    "ship_type": ship_type,
                    "variable": variable,
                    "dataset": dataset_label,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "min": float(values.min()) if n else np.nan,
                    "max": float(values.max()) if n else np.nan,
                    "cv": std / abs(mean) if (n > 1 and mean not in (0, np.nan)
                                              and np.isfinite(mean) and mean != 0)
                          else np.nan,
                    "s_ref": std / abs(ref_mean) if (n > 1 and np.isfinite(ref_mean)
                                                     and ref_mean != 0)
                             else np.nan,
                })

    result = pd.DataFrame.from_records(records)

    logger.info(
        "Built descriptive stats: %d ship types x %d variables x %d cohorts",
        result["ship_type"].nunique(), result["variable"].nunique(),
        result["dataset"].nunique(),
    )

    return result


def resolve_decimals(mean: float) -> int:
    """
    Choose a decimal count from the magnitude of a variable's mean.

    Applied once per variable rather than per cell, so that every figure in a
    variable's block is rendered to the same precision and the columns align
    on the decimal point. Beam and draught then read to 0.1 m while DWT reads
    to the tonne, without either being over- or under-specified.

    :param mean: Reference magnitude, normally the largest cohort mean for the
        variable.
    :type mean: float
    :returns: Number of decimal places.
    :rtype: int
    """
    if not np.isfinite(mean) or mean == 0:
        return 2

    magnitude = abs(mean)
    if magnitude >= 1000:
        return 0
    if magnitude >= 10:
        return 1
    if magnitude >= 1:
        return 2
    return 3


def format_number(value: float, decimals: int,
                  thousands_sep: str = r"\,") -> str:
    """
    Render one numeric cell.

    :param value: Value to format. Null and non-finite values render as an
        em dash.
    :type value: float
    :param decimals: Decimal places.
    :type decimals: int
    :param thousands_sep: Separator inserted between thousands groups.
        Defaults to a plain TeX thin space, which needs no package.
    :type thousands_sep: str
    :returns: LaTeX cell content.
    :rtype: str
    """
    if value is None or not np.isfinite(value):
        return "---"

    text = f"{value:,.{decimals}f}"
    if thousands_sep != ",":
        text = text.replace(",", thousands_sep)
    return text


def render_descriptive_latex(stats: pd.DataFrame,
                             ship_type: str | None = None,
                             low_data_limit: int = 10,
                             variable_labels: dict[str, str] | None = None,
                             statistics: list[str] | None = None,
                             stat_headers: dict[str, str] | None = None,
                             caption: str = "",
                             label: str = "tab:descriptive_stats",
                             thousands_sep: str = r"\,",
                             font_size: str = "footnotesize",
                             table_env: str = "table",
                             dispersion_decimals: int = 2) -> str:
    """
    Render descriptive statistics as a full LaTeX table environment.

    Rows are grouped by variable, with one sub-row per cohort. The variable
    name is printed once per block and the blocks are separated by
    ``\\addlinespace``, which avoids a ``multirow`` dependency.

    :param stats: Output of :func:`build_descriptive_stats`.
    :type stats: pd.DataFrame
    :param ship_type: Restrict the table to one ship type. Required when
        ``stats`` covers several, since a single table mixing ship types would
        have no meaningful reference scale.
    :type ship_type: str or None
    :param low_data_limit: Cohorts with ``n`` strictly below this (but above
        zero) have their sample size wrapped in ``\\lowdata``. Rows with
        ``n = 0`` are rendered in ``myred``.
    :type low_data_limit: int
    :param variable_labels: Optional mapping from raw column names to display
        names. Unmapped variables use their raw name.
    :type variable_labels: dict[str, str] or None
    :param statistics: Statistic columns to include, in order. Defaults to all
        of ``n``, ``mean``, ``std``, ``min``, ``max``, ``cv``, ``s_ref``. Drop
        ``s_ref`` for a narrower table where the variable is only ever a
        predictor.
    :type statistics: list[str] or None
    :param stat_headers: Optional override of the column headers.
    :type stat_headers: dict[str, str] or None
    :param caption: Table caption text.
    :type caption: str
    :param label: LaTeX cross-reference label.
    :type label: str
    :param thousands_sep: Separator between thousands groups.
    :type thousands_sep: str
    :param font_size: LaTeX font size command name, without backslash.
    :type font_size: str
    :param table_env: ``table`` for a single-column float, ``table*`` to span
        both columns of a two-column layout.
    :type table_env: str
    :param dispersion_decimals: Decimal places for the ``cv`` and ``s_ref``
        columns, which are unitless and so do not follow the per-variable
        precision rule.
    :type dispersion_decimals: int
    :returns: Complete LaTeX table as a single string.
    :rtype: str
    :raises ValueError: If ``low_data_limit`` is not positive, if ``ship_type``
        is ambiguous or absent, or if an unknown statistic is requested.
    """
    if low_data_limit <= 0:
        raise ValueError(f"low_data_limit must be positive, got {low_data_limit}")

    statistics = list(statistics or DEFAULT_STAT_HEADERS)
    unknown = [s for s in statistics if s not in DEFAULT_STAT_HEADERS]
    if unknown:
        raise ValueError(
            f"unknown statistic(s) {unknown}; expected from "
            f"{list(DEFAULT_STAT_HEADERS)}"
        )

    headers = {**DEFAULT_STAT_HEADERS, **(stat_headers or {})}
    variable_labels = variable_labels or {}

    available = list(dict.fromkeys(stats["ship_type"]))
    if ship_type is None:
        if len(available) != 1:
            raise ValueError(
                "stats covers several ship types; pass ship_type to select one. "
                f"Available: {available}"
            )
        ship_type = available[0]
    elif ship_type not in available:
        raise ValueError(
            f"ship type '{ship_type}' not present in stats. Available: {available}"
        )

    frame = stats[stats["ship_type"] == ship_type]

    # --- Column specification -------------------------------------------
    column_spec = "ll" + "r" * len(statistics)

    header_cells = ["Variable", "Dataset"] + [headers[s] for s in statistics]
    header_row = " & ".join(header_cells) + r"\\"

    # --- Data rows --------------------------------------------------------
    variables = list(dict.fromkeys(frame["variable"]))
    body: list[str] = []

    for block_index, variable in enumerate(variables):
        block = frame[frame["variable"] == variable]

        # Precision is set once per variable from the largest cohort mean, so
        # every figure in the block is directly comparable by eye.
        means = block["mean"].dropna()
        decimals = resolve_decimals(means.abs().max() if len(means) else np.nan)

        if block_index:
            body.append(r"        \addlinespace")

        for row_index, (_, row) in enumerate(block.iterrows()):
            name = variable_labels.get(variable, variable) if row_index == 0 else ""
            n = int(row["n"])

            if n == 0:
                cells = [r"\cellcolor{myred}---"] * len(statistics)
            else:
                cells = []
                for stat in statistics:
                    if stat == "n":
                        cells.append(rf"\lowdata{{{n}}}" if n < low_data_limit else str(n))
                    elif stat in ("cv", "s_ref"):
                        cells.append(format_number(row[stat], dispersion_decimals,
                                                   thousands_sep))
                    else:
                        cells.append(format_number(row[stat], decimals, thousands_sep))

            body.append(f"        {name} & {row['dataset']} & "
                        + " & ".join(cells) + r"\\")

    # --- Assemble ---------------------------------------------------------
    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \{font_size}",
        rf"    \begin{{tabular}}{{{column_spec}}}",
        r"        \toprule",
        f"        {header_row}",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        rf"    \label{{{label}}}",
        rf"\end{{{table_env}}}",
    ]

    logger.info(
        "Rendered descriptive table for %s: %d variables, %d statistics",
        ship_type, len(variables), len(statistics),
    )

    return "\n".join(lines)


def generate_descriptive_table(datasets: dict[str, dict[str, pd.DataFrame]],
                               variables: list[str],
                               ship_type: str | None = None,
                               reference_dataset: str | None = None,
                               low_data_limit: int = 10,
                               variable_labels: dict[str, str] | None = None,
                               statistics: list[str] | None = None,
                               caption: str = "",
                               label: str = "tab:descriptive_stats",
                               save_path: str | Path | None = None,
                               **render_kwargs) -> str:
    """
    Build statistics and render one LaTeX table in a single call.

    :param datasets: Nested mapping ``{ship_type: {dataset_label: DataFrame}}``.
    :type datasets: dict[str, dict[str, pd.DataFrame]]
    :param variables: Column names to report, in row order.
    :type variables: list[str]
    :param ship_type: Ship type to render. Required if ``datasets`` holds more
        than one.
    :type ship_type: str or None
    :param reference_dataset: Dataset label whose mean normalises ``s_ref``.
    :type reference_dataset: str or None
    :param low_data_limit: Threshold below which sample sizes are flagged.
    :type low_data_limit: int
    :param variable_labels: Optional raw-name to display-name mapping.
    :type variable_labels: dict[str, str] or None
    :param statistics: Statistic columns to include, in order.
    :type statistics: list[str] or None
    :param caption: Table caption text.
    :type caption: str
    :param label: LaTeX cross-reference label.
    :type label: str
    :param save_path: If given, the table is also written to this file.
    :type save_path: str, Path, or None
    :param render_kwargs: Forwarded to :func:`render_descriptive_latex`.
    :returns: The LaTeX table string.
    :rtype: str
    """
    stats = build_descriptive_stats(datasets=datasets,
                                    variables=variables,
                                    reference_dataset=reference_dataset)

    latex = render_descriptive_latex(stats=stats,
                                     ship_type=ship_type,
                                     low_data_limit=low_data_limit,
                                     variable_labels=variable_labels,
                                     statistics=statistics,
                                     caption=caption,
                                     label=label,
                                     **render_kwargs)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(latex, encoding="utf-8")
        logger.info("Saved descriptive table to %s", save_path)

    return latex


def generate_descriptive_tables_per_ship_type(
        datasets: dict[str, dict[str, pd.DataFrame]],
        variables: list[str],
        reference_dataset: str | None = None,
        low_data_limit: int = 10,
        variable_labels: dict[str, str] | None = None,
        statistics: list[str] | None = None,
        caption_template: str = "",
        label_template: str = "tab:descriptive_stats_{slug}",
        save_dir: str | Path | None = None,
        filename_template: str = "descriptive_stats_{slug}.tex",
        **render_kwargs) -> dict[str, str]:
    """
    Render one table per ship type, which is the intended appendix layout.

    Statistics are computed once across all ship types so the reference means
    are resolved consistently, then split for rendering.

    :param caption_template: Caption text, formatted with ``{ship_type}``.
    :type caption_template: str
    :param label_template: Label text, formatted with ``{slug}``, a lowercased
        ship type with non-alphanumeric characters replaced by underscores.
    :type label_template: str
    :param save_dir: If given, each table is written into this directory.
    :type save_dir: str, Path, or None
    :param filename_template: Filename, formatted with ``{slug}``.
    :type filename_template: str
    :returns: Mapping from ship type to its LaTeX table string.
    :rtype: dict[str, str]
    """
    stats = build_descriptive_stats(datasets=datasets,
                                    variables=variables,
                                    reference_dataset=reference_dataset)

    tables: dict[str, str] = {}

    for ship_type in dict.fromkeys(stats["ship_type"]):
        slug = "".join(c if c.isalnum() else "_" for c in ship_type).lower()

        latex = render_descriptive_latex(
            stats=stats,
            ship_type=ship_type,
            low_data_limit=low_data_limit,
            variable_labels=variable_labels,
            statistics=statistics,
            caption=caption_template.format(ship_type=ship_type),
            label=label_template.format(slug=slug),
            **render_kwargs,
        )

        if save_dir is not None:
            path = Path(save_dir) / filename_template.format(slug=slug)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(latex, encoding="utf-8")
            logger.info("Saved descriptive table for %s to %s", ship_type, path)

        tables[ship_type] = latex

    return tables