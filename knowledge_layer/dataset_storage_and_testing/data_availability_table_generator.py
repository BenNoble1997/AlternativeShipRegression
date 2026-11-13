"""
Generate LaTeX data-availability tables for fleet datasets.

Produces a table showing, for each design variable, the number of ships with
reported data across multiple ship types and fuel-type datasets. Counts below
a configurable threshold are wrapped in ``\\lowdata{...}``; variables with no
data at all are rendered as ``\\cellcolor{myred}---``.

Requires the following in the LaTeX preamble::

    \\usepackage{booktabs}
    \\usepackage[table]{xcolor}
    \\definecolor{myred}{RGB}{237,106,90}
    \\definecolor{myorange}{RGB}{255,203,119}
    \\newcommand{\\lowdata}[1]{\\cellcolor{myorange}#1}

The module separates computation from display:

* :func:`build_availability_counts` produces a plain ``DataFrame`` of counts
  (variables as rows, ``(ship type, dataset)`` MultiIndex columns) that can be
  inspected, tested, or reused independently.
* :func:`render_availability_latex` converts that counts frame into the final
  LaTeX string, applying the threshold formatting rules.
* :func:`generate_availability_table` is a convenience wrapper that chains the
  two and optionally writes the result to disk.
* :func:`extract_fleet_datasets` adapts a collection of ``FleetData`` objects
  into the nested-DataFrame structure the builder expects.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_availability_counts(datasets: dict[str, dict[str, pd.DataFrame]],
                              variables: list[str]) -> pd.DataFrame:
    """
    Count non-null observations per variable across grouped datasets.

    A variable that is absent from a dataset's columns is treated identically
    to a variable that is present but entirely null: both yield a count of 0,
    since in either case no observations are available for modelling.

    :param datasets: Nested mapping ``{ship_type: {dataset_label: DataFrame}}``.
        The outer keys become the ``\\multicolumn`` group headers, the inner
        keys the per-column sub-headers. Insertion order is preserved.
    :type datasets: dict[str, dict[str, pd.DataFrame]]
    :param variables: Column names to count, in the row order desired for the
        final table.
    :type variables: list[str]
    :returns: DataFrame with ``variables`` as the index and a two-level
        ``MultiIndex`` ``(ship_type, dataset_label)`` as columns, holding
        integer counts.
    :rtype: pd.DataFrame
    :raises ValueError: If ``datasets`` or ``variables`` is empty.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one ship type group")
    if not variables:
        raise ValueError("variables must contain at least one column name")

    counts: dict[tuple[str, str], list[int]] = {}

    for ship_type, group in datasets.items():
        if not group:
            raise ValueError(
                f"ship type '{ship_type}' has no datasets assigned"
            )
        for dataset_label, df in group.items():
            column_counts = []
            for variable in variables:
                if variable in df.columns:
                    column_counts.append(int(df[variable].notna().sum()))
                else:
                    logger.debug(
                        "Variable '%s' not present in %s / %s; recording 0",
                        variable, ship_type, dataset_label,
                    )
                    column_counts.append(0)
            counts[(ship_type, dataset_label)] = column_counts

    result = pd.DataFrame(
        counts,
        index=pd.Index(variables, name="variable"),
    )
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["ship_type", "dataset"]
    )

    logger.info(
        "Built availability counts: %d variables x %d dataset columns",
        len(result), len(result.columns),
    )

    return result


def format_count_cell(count: int, low_data_limit: int) -> str:
    """
    Format a single count according to the availability thresholds.

    :param count: Number of non-null observations.
    :type count: int
    :param low_data_limit: Counts strictly below this (but above zero) are
        wrapped in ``\\lowdata{...}``.
    :type low_data_limit: int
    :returns: LaTeX cell content.
    :rtype: str
    :raises ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    if count == 0:
        return r"\cellcolor{myred}---"
    if count < low_data_limit:
        return rf"\lowdata{{{count}}}"
    return str(count)


def render_availability_latex(counts: pd.DataFrame,
                              low_data_limit: int = 10,
                              variable_labels: dict[str, str] | None = None,
                              caption: str = "",
                              label: str = "tab:data_availability_check",
                              first_column_header: str = "Design variable",
                              first_column_width: float = 0.10,
                              data_column_width: float = 0.08,
                              font_size: str = "scriptsize") -> str:
    """
    Render a counts DataFrame as a full LaTeX ``table*`` environment.

    :param counts: Output of :func:`build_availability_counts` - variables as
        the index, ``(ship_type, dataset)`` MultiIndex columns.
    :type counts: pd.DataFrame
    :param low_data_limit: Counts strictly below this threshold are flagged
        with ``\\lowdata``.
    :type low_data_limit: int
    :param variable_labels: Optional mapping from raw column names to display
        names, e.g. ``{'LBP (M)': 'LBP (m)'}``. Unmapped variables use their
        raw name.
    :type variable_labels: dict[str, str] or None
    :param caption: Table caption text.
    :type caption: str
    :param label: LaTeX label for cross-referencing.
    :type label: str
    :param first_column_header: Header text for the variable-name column.
    :type first_column_header: str
    :param first_column_width: Width of the first column as a fraction of
        ``\\linewidth``.
    :type first_column_width: float
    :param data_column_width: Width of each data column as a fraction of
        ``\\linewidth``.
    :type data_column_width: float
    :param font_size: LaTeX font size command name (without backslash).
    :type font_size: str
    :returns: Complete LaTeX table as a single string.
    :rtype: str
    :raises ValueError: If ``low_data_limit`` is not positive or ``counts``
        lacks the expected two-level column MultiIndex.
    """
    if low_data_limit <= 0:
        raise ValueError(
            f"low_data_limit must be positive, got {low_data_limit}"
        )
    if counts.columns.nlevels != 2:
        raise ValueError(
            "counts must have a two-level (ship_type, dataset) column "
            f"MultiIndex, got {counts.columns.nlevels} level(s)"
        )

    variable_labels = variable_labels or {}
    n_data_columns = len(counts.columns)

    # --- Column specification -------------------------------------------
    first_col = (rf">{{\raggedright\arraybackslash}}"
                 rf"p{{{first_column_width:.2f}\linewidth}}")
    data_col = (rf">{{\centering\arraybackslash}}"
                rf"p{{{data_column_width:.2f}\linewidth}}")
    column_spec = first_col + data_col * n_data_columns

    # --- Group header row and cmidrules ---------------------------------
    # Ship types in first-seen order; each spans its number of datasets.
    ship_types = list(dict.fromkeys(counts.columns.get_level_values(0)))

    group_cells = []
    cmidrules = []
    cursor = 2  # column 1 is the variable-name column
    for ship_type in ship_types:
        span = len(counts[ship_type].columns)
        group_cells.append(
            rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{ship_type}}}}}"
        )
        cmidrules.append(
            rf"\cmidrule(lr){{{cursor}-{cursor + span - 1}}}"
        )
        cursor += span

    group_header = " & " + " & ".join(group_cells) + r"\\"
    cmidrule_line = " ".join(cmidrules)

    # --- Sub-header row --------------------------------------------------
    dataset_headers = [dataset for _, dataset in counts.columns]
    sub_header = (rf"\textbf{{{first_column_header}}}& "
                  + "& ".join(dataset_headers) + r"\\")

    # --- Data rows --------------------------------------------------------
    data_rows = []
    for variable, row in counts.iterrows():
        display_name = variable_labels.get(variable, variable)
        cells = [format_count_cell(int(count), low_data_limit)
                 for count in row]
        data_rows.append(f" {display_name}&   " + "&   ".join(cells) + r"\\")

    # --- Assemble ---------------------------------------------------------
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \{font_size}",
        rf"    \begin{{tabular}}{{{column_spec}}}",
        r"        \toprule",
        f"        {group_header}",
        f"        {cmidrule_line}",
        f"       {sub_header}",
        r"        \midrule",
        *data_rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        rf"    \label{{{label}}}",
        r"\end{table*}",
    ]

    logger.info(
        "Rendered LaTeX table: %d variables, %d ship type groups, "
        "low-data limit %d", len(counts), len(ship_types), low_data_limit,
    )

    return "\n".join(lines)


def extract_fleet_datasets(fleets: dict[str, "FleetData"],
                           dataset_keys: dict[str, str]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Adapt FleetData objects into the nested structure the builder expects.

    :param fleets: Mapping from ship type display name to its ``FleetData``
        instance, e.g. ``{'Bulk Carrier': bulk_fleet, 'Tanker': tanker_fleet}``.
    :type fleets: dict[str, FleetData]
    :param dataset_keys: Mapping from column display label to the key under
        which the dataset is stored in ``processed_data``, e.g.
        ``{'Dual fuelled': 'dual_fuel_Alt_fuel', 'Pure Alternative': 'pure_Alt_fuel'}``.
        The same keys are looked up in every fleet.
    :type dataset_keys: dict[str, str]
    :returns: Nested mapping ``{ship_type: {dataset_label: DataFrame}}``.
    :rtype: dict[str, dict[str, pd.DataFrame]]
    :raises KeyError: If a dataset key is missing from a fleet's
        ``processed_data``.
    """
    datasets: dict[str, dict[str, pd.DataFrame]] = {}

    for ship_type, fleet in fleets.items():
        group: dict[str, pd.DataFrame] = {}
        for display_label, storage_key in dataset_keys.items():
            if storage_key not in fleet.processed_data:
                raise KeyError(
                    f"Dataset '{storage_key}' not found in processed_data "
                    f"for ship type '{ship_type}'. Available: "
                    f"{list(fleet.processed_data.keys())}"
                )
            group[display_label] = fleet.processed_data[storage_key]
        datasets[ship_type] = group

    return datasets


def generate_availability_table(datasets: dict[str, dict[str, pd.DataFrame]],
                                variables: list[str],
                                low_data_limit: int = 10,
                                variable_labels: dict[str, str] | None = None,
                                caption: str = "",
                                label: str = "tab:data_availability_check",
                                save_path: str | Path | None = None) -> str:
    """
    Build counts and render the LaTeX availability table in one call.

    :param datasets: Nested mapping ``{ship_type: {dataset_label: DataFrame}}``.
    :type datasets: dict[str, dict[str, pd.DataFrame]]
    :param variables: Column names to report, in row order.
    :type variables: list[str]
    :param low_data_limit: Threshold below which counts are flagged.
    :type low_data_limit: int
    :param variable_labels: Optional raw-name to display-name mapping.
    :type variable_labels: dict[str, str] or None
    :param caption: Table caption text.
    :type caption: str
    :param label: LaTeX cross-reference label.
    :type label: str
    :param save_path: If given, the table is also written to this file.
    :type save_path: str, Path, or None
    :returns: The LaTeX table string.
    :rtype: str
    """
    counts = build_availability_counts(datasets=datasets,
                                       variables=variables)

    latex = render_availability_latex(counts=counts,
                                      low_data_limit=low_data_limit,
                                      variable_labels=variable_labels,
                                      caption=caption,
                                      label=label)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(latex, encoding="utf-8")
        logger.info("Saved availability table to %s", save_path)

    return latex