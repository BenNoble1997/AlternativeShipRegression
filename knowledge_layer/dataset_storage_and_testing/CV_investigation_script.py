"""
Coefficient-of-variation profiling of fuel cohorts, by ship type.

Reports the dispersion of each experiment variable within each fuel cohort, so
that R-squared changes in the cross-dataset tables can be attributed to
variance compression rather than to predictive degradation.

For a model evaluated on a cohort, R2 = 1 - MSE / var(y) and
NRMSE = RMSE / mean(y). Writing CV = sd(y) / mean(y) gives

    R2 = 1 - (NRMSE / CV)^2

so R2 is the relative error benchmarked against the cohort's own spread. A
cohort more homogeneous than the conventional fleet depresses R2 even when
absolute error is unchanged, which is what the relative table quantifies.

Two tables are produced:

- ``CV`` - coefficient of variation in percent, per ship type, cohort and
  variable.
- ``CV relative`` - the same divided by the conventional cohort's CV for the
  same ship type and variable. Below 1 means more homogeneous than the
  conventional fleet.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (
    getting_alt_fuel_ships, get_variables)
from parameters.shipshape.ShipShapeBasic_class import ShipType


_FONT = 'Arial'
_HEADER_FILL = PatternFill('solid', fgColor='DDDDDD')
_COMPRESSED_FILL = PatternFill('solid', fgColor='FFF2CC')


# ---------------------------------------------------------------------------
# Stage 1: build every fleet once
# ---------------------------------------------------------------------------

def build_fleets(types_list: list,
                 max_build_year: int = 1990,
                 fuel_type_columns: list[str] | None = None,
                 conventional_fuels: list[str] | None = None,
                 alt_fuels: dict[str, list[str]] | None = None,
                 debug: bool = False) -> tuple[dict, dict[str, list[str]]]:
    """
    Import every ship type's fleet once and capture its experiment variables.

    The original script rebuilt the fleets inside the dataset loop, so a run
    over three cohort names imported each ship type three times. Every cohort
    is a view on the same imported fleet, so one pass suffices.

    :param types_list: ShipType members to import.
    :type types_list: list
    :param max_build_year: Earliest build year retained.
    :type max_build_year: int
    :param fuel_type_columns: Columns searched for fuel strings.
    :type fuel_type_columns: list[str] or None
    :param conventional_fuels: Tokens identifying conventional fuels.
    :type conventional_fuels: list[str] or None
    :param alt_fuels: Mapping of cohort name to its fuel tokens.
    :type alt_fuels: dict[str, list[str]] or None
    :param debug: Print each ship type as it is imported, with its variables.
    :type debug: bool
    :returns: ``(fleets, variables_by_shiptype)``, both keyed by the ship type
        string.
    :rtype: tuple[dict, dict[str, list[str]]]
    """
    if fuel_type_columns is None:
        fuel_type_columns = ['MAIN_FUEL', 'BUNKER_FUEL']

    if conventional_fuels is None:
        conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    if alt_fuels is None:
        alt_fuels = {
            'Alt_fuel': ['LNG', 'Methanol', 'Ethanol', 'Ammonia',
                         'Hydrogen', 'cng', 'ethane'],
            'Biofuel': ['Biofuel'],
        }

    fleets = {}
    variables_by_shiptype = {}

    for shiptype in types_list:
        bins, variables, _, filepath, save_dir, data = get_variables(shiptype=shiptype)

        fleet = getting_alt_fuel_ships(
            filepath=filepath,
            shiptype=shiptype,
            max_build_year=max_build_year,
            fuel_type_columns=fuel_type_columns,
            conventional_fuels=conventional_fuels,
            alt_fuels=alt_fuels,
            outliers=False
        )

        fleets[shiptype.string] = fleet
        variables_by_shiptype[shiptype.string] = list(variables)

        if debug:
            print(f"[fleets] {shiptype.string}: {len(variables)} experiment "
                  f"variables -> {list(variables)}")

    return fleets, variables_by_shiptype


def collect_cohorts(fleets: dict,
                    dataset_names: dict[str, str],
                    debug: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Extract the named cohorts from every fleet.

    Missing or empty datasets are skipped rather than raising, since not every
    ship type populates every cohort.

    :param fleets: Mapping of ship type string to fleet object.
    :type fleets: dict
    :param dataset_names: Mapping of display label to ``processed_data`` key,
        e.g. ``{'Conventional': 'pure_conventional'}``.
    :type dataset_names: dict[str, str]
    :param debug: Print each cohort resolved and each one skipped.
    :type debug: bool
    :returns: Nested mapping ``{ship_type: {cohort_label: DataFrame}}``.
    :rtype: dict[str, dict[str, pd.DataFrame]]
    """
    cohorts = {}

    for ship_type, fleet in fleets.items():
        resolved = {}

        for label, dataset_name in dataset_names.items():
            frame = fleet.processed_data.get(dataset_name)

            if frame is None or len(frame) == 0:
                if debug:
                    print(f"[cohorts] {ship_type}: '{dataset_name}' missing "
                          f"or empty - skipped")
                continue

            resolved[label] = frame

            if debug:
                print(f"[cohorts] {ship_type} / {label}: {len(frame)} ships")

        if resolved:
            cohorts[ship_type] = resolved

    return cohorts


# ---------------------------------------------------------------------------
# Stage 2: CV tables
# ---------------------------------------------------------------------------

def _compute_cv_for_df(df: pd.DataFrame,
                       target_variables: list[str],
                       round_decimals: int,
                       min_samples: int = 2) -> dict:
    """
    Compute CV as a percentage for each variable in a single dataframe.

    Carried over from the original script, with ``min_samples`` exposed and
    infinities coerced so that a single non-finite value cannot poison a
    standard deviation.

    :param df: Cohort dataframe.
    :type df: pd.DataFrame
    :param target_variables: Columns to profile.
    :type target_variables: list[str]
    :param round_decimals: Decimal places for the rounded percentage.
    :type round_decimals: int
    :param min_samples: Minimum non-null rows required to report a CV.
    :type min_samples: int
    :returns: Mapping of variable name to CV percent, NaN where unavailable.
    :rtype: dict
    """
    row = {}

    for var in target_variables:
        if var not in df.columns:
            row[var] = np.nan
            continue

        data = pd.to_numeric(df[var], errors='coerce')
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < min_samples or data.mean() == 0:
            row[var] = np.nan
            continue

        row[var] = round((data.std(ddof=1) / data.mean()) * 100, round_decimals)

    return row


def compute_cv_matrix(cohorts: dict[str, dict[str, pd.DataFrame]],
                      variables_by_shiptype: dict[str, list[str]],
                      round_decimals: int = 1,
                      min_samples: int = 2,
                      debug: bool = False) -> pd.DataFrame:
    """
    Build the CV table indexed by ship type and cohort.

    Each ship type is profiled on its own experiment variables, so a variable
    used only for containers is blank on the RoPax rows even where the column
    happens to exist. Columns are the union across ship types, in order of
    first appearance.

    Sample sizes are reported per cohort as ``n ships``, and per variable as
    ``n <variable>``, because a cohort's usable n varies by variable wherever
    reporting is incomplete.

    :param cohorts: Output of :func:`collect_cohorts`.
    :type cohorts: dict[str, dict[str, pd.DataFrame]]
    :param variables_by_shiptype: Experiment variables per ship type.
    :type variables_by_shiptype: dict[str, list[str]]
    :param round_decimals: Decimal places for the CV percentages.
    :type round_decimals: int
    :param min_samples: Minimum non-null rows required to report a CV.
    :type min_samples: int
    :param debug: Print each row as it is computed.
    :type debug: bool
    :returns: MultiIndex (Ship Type, Dataset) table of CV percentages.
    :rtype: pd.DataFrame
    """
    column_order: list[str] = []

    for ship_type in cohorts:
        for variable in variables_by_shiptype.get(ship_type, []):
            if variable not in column_order:
                column_order.append(variable)

    rows = []

    for ship_type, cohort_map in cohorts.items():
        variables = variables_by_shiptype.get(ship_type, [])

        if debug and not variables:
            print(f"[cv] {ship_type}: no experiment variables declared")

        for cohort_label, frame in cohort_map.items():
            row = {'Ship Type': ship_type,
                   'Dataset': cohort_label,
                   'n ships': len(frame)}

            row.update(_compute_cv_for_df(
                frame, variables, round_decimals, min_samples))

            for variable in variables:
                if variable in frame.columns:
                    counts = pd.to_numeric(frame[variable], errors='coerce')
                    row[f'n {variable}'] = int(counts.notna().sum())

            rows.append(row)

            if debug:
                reported = sum(1 for v in variables
                               if not pd.isna(row.get(v, np.nan)))
                print(f"[cv] {ship_type} / {cohort_label}: {len(frame)} ships, "
                      f"{reported}/{len(variables)} variables reported")

    table = pd.DataFrame(rows).set_index(['Ship Type', 'Dataset'])

    ordered = (['n ships']
               + column_order
               + [f'n {v}' for v in column_order if f'n {v}' in table.columns])

    return table[[c for c in ordered if c in table.columns]]


def compute_relative_cv(cv_table: pd.DataFrame,
                        reference_dataset: str = 'Conventional',
                        round_decimals: int = 3,
                        debug: bool = False) -> pd.DataFrame:
    """
    Express every CV as a ratio to the conventional CV for the same ship type.

    A value below one means the cohort is more homogeneous than the
    conventional fleet for that variable, and therefore that its R-squared is
    depressed relative to the conventional baseline even at unchanged relative
    error. The reference rows are retained, showing 1.0 throughout, so the
    anchor is visible in the table.

    :param cv_table: Output of :func:`compute_cv_matrix`.
    :type cv_table: pd.DataFrame
    :param reference_dataset: Dataset label used as the denominator.
    :type reference_dataset: str
    :param round_decimals: Decimal places for the ratios.
    :type round_decimals: int
    :param debug: Print any ship type lacking the reference cohort.
    :type debug: bool
    :returns: Table of ratios with the same index as ``cv_table``.
    :rtype: pd.DataFrame
    :raises ValueError: If no ship type carries the reference dataset.
    """
    variable_columns = [c for c in cv_table.columns
                        if c != 'n ships' and not str(c).startswith('n ')]

    datasets = cv_table.index.get_level_values('Dataset')

    if reference_dataset not in set(datasets):
        raise ValueError(
            f"reference_dataset '{reference_dataset}' not present. Datasets "
            f"available: {sorted(set(datasets))}."
        )

    rows = []

    for ship_type in cv_table.index.get_level_values('Ship Type').unique():
        block = cv_table.loc[ship_type]

        if reference_dataset not in block.index:
            if debug:
                print(f"[relative] {ship_type}: no '{reference_dataset}' "
                      f"cohort - ratios omitted")
            continue

        reference = block.loc[reference_dataset, variable_columns]

        for cohort_label in block.index:
            values = block.loc[cohort_label, variable_columns]
            ratios = (values / reference).replace([np.inf, -np.inf], np.nan)

            row = {'Ship Type': ship_type,
                   'Dataset': cohort_label,
                   'n ships': block.loc[cohort_label, 'n ships']}
            row.update(ratios.round(round_decimals).to_dict())
            rows.append(row)

    relative = pd.DataFrame(rows).set_index(['Ship Type', 'Dataset'])

    return relative[['n ships'] + variable_columns]


# ---------------------------------------------------------------------------
# Stage 3: output
# ---------------------------------------------------------------------------

def write_cv_workbook(output_path: Path | str,
                      cv_table: pd.DataFrame,
                      relative_table: pd.DataFrame,
                      highlight_below: float = 0.5,
                      debug: bool = False) -> Path:
    """
    Write both tables to a two-sheet workbook.

    Values are written rather than formulas, so the file reads straight back
    with :func:`pandas.read_excel` without recalculation.

    :param output_path: Destination ``.xlsx`` path.
    :type output_path: Path or str
    :param cv_table: Output of :func:`compute_cv_matrix`.
    :type cv_table: pd.DataFrame
    :param relative_table: Output of :func:`compute_relative_cv`.
    :type relative_table: pd.DataFrame
    :param highlight_below: Ratio cells below this are shaded on the relative
        sheet. Set to 0 to disable shading.
    :type highlight_below: float
    :param debug: Print each sheet as it is written.
    :type debug: bool
    :returns: Path to the written workbook.
    :rtype: Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        cv_table.to_excel(writer, sheet_name='CV')
        _format_sheet(writer.sheets['CV'], cv_table,
                      number_format='0.0', highlight_below=0.0)

        relative_table.to_excel(writer, sheet_name='CV relative')
        _format_sheet(writer.sheets['CV relative'], relative_table,
                      number_format='0.000',
                      highlight_below=highlight_below)

        if debug:
            print(f"[excel] CV: {cv_table.shape[0]} rows x "
                  f"{cv_table.shape[1]} columns")
            print(f"[excel] CV relative: {relative_table.shape[0]} rows x "
                  f"{relative_table.shape[1]} columns")

    if debug:
        print(f"[excel] saved: {output_path}")

    return output_path


def _format_sheet(sheet,
                  table: pd.DataFrame,
                  number_format: str,
                  highlight_below: float) -> None:
    """
    Apply fonts, number formats, widths and a frozen header to one sheet.

    Count columns keep an integer format regardless of the sheet's default,
    since they are sample sizes rather than measurements.

    :param sheet: Worksheet to format.
    :type sheet: openpyxl.worksheet.worksheet.Worksheet
    :param table: Frame written to the sheet, used for column names and shape.
    :type table: pd.DataFrame
    :param number_format: Excel number format for measurement columns.
    :type number_format: str
    :param highlight_below: Shade numeric cells below this value; 0 disables.
    :type highlight_below: float
    """
    n_index = table.index.nlevels
    n_rows = len(table)

    for col_idx in range(1, n_index + len(table.columns) + 1):
        header = sheet.cell(row=1, column=col_idx)
        header.font = Font(name=_FONT, bold=True, size=10)
        header.fill = _HEADER_FILL
        header.alignment = Alignment(horizontal='center', wrap_text=True)

    for offset, column in enumerate(table.columns):
        col_idx = n_index + 1 + offset
        is_count = str(column) == 'n ships' or str(column).startswith('n ')
        fmt = '#,##0' if is_count else number_format

        for row_idx in range(2, n_rows + 2):
            cell = sheet.cell(row=row_idx, column=col_idx)
            cell.font = Font(name=_FONT, size=10)
            cell.number_format = fmt

            if (highlight_below > 0 and not is_count
                    and isinstance(cell.value, (int, float))
                    and cell.value < highlight_below):
                cell.fill = _COMPRESSED_FILL

    for col_idx in range(1, n_index + len(table.columns) + 1):
        letter = get_column_letter(col_idx)
        header_value = sheet.cell(row=1, column=col_idx).value
        width = len(str(header_value)) + 2 if header_value else 10
        sheet.column_dimensions[letter].width = min(max(width, 10), 18)

    sheet.freeze_panes = sheet.cell(row=2, column=n_index + 1).coordinate


def print_cv_table(cv_table: pd.DataFrame,
                   title: str = "Coefficient of Variation (%)") -> None:
    """
    Print a formatted table to the console.

    :param cv_table: Table to print.
    :type cv_table: pd.DataFrame
    :param title: Heading shown above the table.
    :type title: str
    """
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    print(cv_table.to_string())
    print(f"{'=' * 110}\n")


# ---------------------------------------------------------------------------
# Carried over unchanged from the original script
# ---------------------------------------------------------------------------

def compute_cv_table_binned(
    ship_datasets: dict[str, pd.DataFrame],
    bins_per_shiptype: dict[str, list],
    target_variables: list[str],
    round_decimals: int = 1,
    min_bin_samples: int = 10,
) -> pd.DataFrame:
    """
    Compute CV broken down by DWT bin, falling back to whole-fleet CV.

    :param ship_datasets: Mapping of ship type name to dataframe.
    :type ship_datasets: dict[str, pd.DataFrame]
    :param bins_per_shiptype: Mapping of ship type name to DWT bin edges.
    :type bins_per_shiptype: dict[str, list]
    :param target_variables: Columns to compute CV for.
    :type target_variables: list[str]
    :param round_decimals: Decimal places for rounding.
    :type round_decimals: int
    :param min_bin_samples: Minimum ships in a bin to compute CV.
    :type min_bin_samples: int
    :returns: MultiIndex (Ship Type, DWT Bin) table of CV values.
    :rtype: pd.DataFrame
    """
    rows = []

    for ship_type, df in ship_datasets.items():

        if ship_type not in bins_per_shiptype:
            row = {'Ship Type': ship_type, 'DWT Bin': 'All', 'n': len(df)}
            row.update(_compute_cv_for_df(df, target_variables, round_decimals))
            rows.append(row)
            continue

        dwt_bins = bins_per_shiptype[ship_type]

        if 'DWT' not in df.columns:
            print(f"Warning: DWT column not found for {ship_type}, skipping binning")
            continue

        bin_labels = [
            f'{int(dwt_bins[i]/1000)}k-{int(dwt_bins[i+1]/1000)}k'
            if dwt_bins[i+1] != np.inf
            else f'{int(dwt_bins[i]/1000)}k+'
            for i in range(len(dwt_bins) - 1)
        ]

        df = df.copy()
        df['_dwt_bin'] = pd.cut(df['DWT'], bins=dwt_bins,
                                labels=bin_labels, right=True)

        for bin_label in bin_labels:
            bin_df = df[df['_dwt_bin'] == bin_label]

            if len(bin_df) < min_bin_samples:
                continue

            row = {'Ship Type': ship_type, 'DWT Bin': bin_label,
                   'n': len(bin_df)}
            row.update(_compute_cv_for_df(bin_df, target_variables,
                                          round_decimals))
            rows.append(row)

    result = pd.DataFrame(rows).set_index(['Ship Type', 'DWT Bin'])
    col_order = ['n'] + [v for v in target_variables if v in result.columns]
    return result[col_order]


# ---------------------------------------------------------------------------

if __name__ == "__main__":

    types_list = [ShipType.BULK_CARRIER, ShipType.TANKER,
                  ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    dataset_names = {
        'Conventional': 'pure_conventional',
        'LNG': 'dual_fuel_LNG',
        'Methanol': 'dual_fuel_Methanol',
    }

    output_path = Path(
        r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer"
        r"\dataset_storage_and_testing\data_processing_scripts\output"
        r"\cohort_cv.xlsx"
    )

    fleets, variables_by_shiptype = build_fleets(types_list, debug=True)
    cohorts = collect_cohorts(fleets, dataset_names, debug=True)

    cv_table = compute_cv_matrix(cohorts, variables_by_shiptype, debug=True)
    relative_table = compute_relative_cv(cv_table,
                                         reference_dataset='Conventional',
                                         debug=True)

    print_cv_table(cv_table, "Coefficient of Variation (%)")
    print_cv_table(relative_table,
                   "CV relative to conventional (<1 = more homogeneous)")

    write_cv_workbook(output_path, cv_table, relative_table, debug=True)