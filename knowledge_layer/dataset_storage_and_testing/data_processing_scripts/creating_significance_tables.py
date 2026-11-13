"""
Generate significance tables for experiments 1 and 2.

Sheet loading
-------------
The significance exports are not uniform. Some ship types have fewer
alternative-fuel cohorts than others (bulk carriers and RoPax have no methanol
fleet), and different export runs have written the results in different
layouts. Selecting a sheet by position, as in ``sheet_name=2``, therefore fails
on some files and, worse, could silently select the wrong sheet on others.

``load_significance_results`` selects by content instead. It recognises the two
layouts the exporter produces:

- **pooled**: one sheet holds every cohort, distinguished by a ``fuel_type``
  column. The sheet is identified by having the required result columns.
- **per-cohort**: each cohort has its own sheet, named for the fuel. The
  sheets named in ``fuel_order`` are read and stacked, and a ``fuel_type``
  column is added if the export omitted it.

In both cases the function returns the cohorts actually present, which is then
passed to the generator as ``fuel_order``. Passing a cohort that has no rows
would otherwise produce a column of ``---`` placeholders, which reads as a data
failure rather than as a cohort that does not exist for this ship type.
"""

from pathlib import Path

import pandas as pd

from knowledge_layer.dataset_storage_and_testing.significance_table_generator import (
    SignificanceTableGenerator,
)
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (
    getting_alt_fuel_ships,
    get_variables,
)
from parameters.shipshape.ShipShapeBasic_class import ShipType


# =============================================================================
# Configuration
# =============================================================================

RUN_EXPERIMENT_1 = False
RUN_EXPERIMENT_2 = True

# Print the sheet inventory of every workbook before generating anything.
# Leave on until the layouts are confirmed; it costs one pass over the files.
INSPECT_WORKBOOKS = True

RESULTS_ROOT = Path(
    r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\new"
)

# Columns that mark a sheet as a significance export.
REQUIRED_COLUMNS = frozenset(
    {'delta_rrmse', 'ci_lower', 'ci_upper', 'is_significant'}
)

TARGET_LABELS = {
    'BEAM (M)': r'\gls{b}',  'DEPTH (M)': r'\gls{d}',  'DRAFT (M)': r'\gls{t}',
    'DWT': r'\gls{dwt}',     'LBP (M)': r'\gls{lbp}',  'LOA (M)': r'\gls{loa}',
    'Np': r'\gls{np}',       'Nv': r'\gls{nv}',        'Vs': r'\gls{vs}',
}

# The exporter writes cohort names inconsistently ('dual fuel LNG' but
# 'Dual fuel Methanol'). Those strings become column headers verbatim, so they
# are mapped to display labels here rather than left to print as stored.
FUEL_LABELS = {
    'dual fuel LNG': 'LNG dual fuel',
    'Dual fuel Methanol': 'Methanol dual fuel',
}

# Directories rather than files: the filename is the same across ship types,
# so it is stated once below.
EXPERIMENT_1_DIRECTORIES = {
    ShipType.BULK_CARRIER:
        RESULTS_ROOT / 'Bulk_carrier' / 'experiment1' / '1990',
    ShipType.TANKER:
        RESULTS_ROOT / 'Tanker' / 'experiment1' / '1990',
    ShipType.CONTAINER_SHIP:
        RESULTS_ROOT / 'Containership' / 'experiment1' / '1990',
    ShipType.RO_PAX:
        RESULTS_ROOT / 'RO_Pax' / 'experiment1' / '1990',
}

# Each results directory holds several historical exports side by side
# ('_with_outliers', '_matched_5', '_v2', '_v3', ...), which differ in
# preprocessing rather than in shape. Any of them would load cleanly and
# produce a plausible table, so the export is named exactly rather than
# matched by pattern: the failure mode of guessing here is a stale table that
# looks correct, not an error.
EXPERIMENT_1_FILENAME = 'significance_tests_new_datasets.xlsx'

# Experiment 2 stores the parametric and random forest results in separate
# workbooks under separate directories. The directory stem and the filename
# stem differ between the two, so both are parameterised.
EXPERIMENT_2_SPECS = {
    ShipType.BULK_CARRIER: ('Bulk_carrier', 'Bulk carrier'),
    ShipType.TANKER:       ('Tanker',       'Tanker'),
    ShipType.CONTAINER_SHIP: ('Containership', 'Container ship'),
    ShipType.RO_PAX:       ('RO_Pax',       'RoPax'),
}

MODEL_KINDS = {
    'param': ('param_1990', 'parametric_significance_test', 'parametric'),
    'rf':    ('rf_1990',    'RF_significance_test',         'random forest'),
}


def experiment_2_path(ship_directory: str,
                      file_stem: str,
                      model_kind: str) -> Path:
    """
    Build the experiment 2 workbook path for one ship type and model family.

    Building the path rather than writing out eight literals removes the class
    of error where a random forest entry points at a parametric file, which is
    silent: the table generates successfully and reports the wrong results.

    :param ship_directory: Directory stem for the ship type, e.g.
        ``'Bulk_carrier'``.
    :type ship_directory: str
    :param file_stem: Ship name as it appears in the filename, e.g.
        ``'Bulk carrier'``.
    :type file_stem: str
    :param model_kind: ``'param'`` or ``'rf'``.
    :type model_kind: str
    :returns: Full path to the workbook.
    :rtype: Path
    """
    subdirectory, filename_prefix, _ = MODEL_KINDS[model_kind]

    return (RESULTS_ROOT / ship_directory / subdirectory / 'experiment2'
            / f'{filename_prefix}_{file_stem}_new_datasets.xlsx')


# =============================================================================
# Sheet loading
# =============================================================================

def resolve_export(directory: "str | Path", filename: str) -> Path:
    """
    Resolve a named export inside a results directory.

    The filename is stated by the caller rather than matched by pattern,
    because these directories accumulate exports from earlier runs that differ
    only in preprocessing. A pattern that matched several of them would either
    fail or, worse, silently select one, and every candidate loads without
    complaint.

    When the named file is absent the error lists the other exports present,
    which is usually enough to identify a rename or a moved run without
    opening a file browser.

    :param directory: Directory holding the export.
    :type directory: str or Path
    :param filename: Exact filename expected.
    :type filename: str
    :returns: Path to the export.
    :rtype: Path
    :raises FileNotFoundError: If the directory or the named file is absent.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")

    path = directory / filename
    if path.is_file():
        return path

    alternatives = sorted(p.name for p in directory.glob('*.xlsx'))
    raise FileNotFoundError(
        f"{directory}: {filename!r} not found. "
        f"Workbooks present: {alternatives or '(none)'}"
    )


def inspect_workbook(path: "str | Path") -> None:
    """
    Print the sheet inventory of one workbook.

    Shows, per sheet, its shape and the distinct values of ``fuel_type`` and
    ``comparison`` where present. This is what distinguishes a pooled export
    (every sheet has ``fuel_type``, sheets differ by ``comparison``) from a
    per-cohort export (sheets are named for fuels and may lack ``fuel_type``).

    :param path: Workbook to inspect.
    :type path: str or Path
    """
    path = Path(path)

    if not path.exists():
        print(f"  MISSING: {path}")
        return

    workbook = pd.ExcelFile(path)
    print(f"  {path.name}: {len(workbook.sheet_names)} sheet(s)")

    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)

        fuels = (sorted(str(v) for v in frame['fuel_type'].dropna().unique())
                 if 'fuel_type' in frame.columns else None)
        comparisons = (
            sorted(str(v) for v in frame['comparison'].dropna().unique())
            if 'comparison' in frame.columns else None)
        has_results = REQUIRED_COLUMNS.issubset(frame.columns)

        print(f"    [{sheet!r}] {frame.shape[0]}x{frame.shape[1]} "
              f"results={has_results} fuel_type={fuels} "
              f"comparison={comparisons}")


def load_significance_results(path: "str | Path",
                              fuel_order,
                              verbose: bool = True):
    """
    Load a significance export, whatever sheet layout it uses.

    Two layouts are recognised, tried in order:

    1. A single sheet carrying the result columns together with a
       ``fuel_type`` column. This is the pooled layout; the first matching
       sheet is returned whole.
    2. One sheet per cohort, named for the fuel. Sheets whose name matches an
       entry of ``fuel_order`` are read and concatenated, and ``fuel_type`` is
       filled in from the sheet name where the export omitted it.

    Matching of sheet names to fuel names ignores case and surrounding
    whitespace, since exporters are inconsistent about both.

    :param path: Workbook to read.
    :type path: str or Path
    :param fuel_order: Cohorts wanted, in column order.
    :type fuel_order: Sequence
    :param verbose: Print which layout was detected and which cohorts found.
    :type verbose: bool
    :returns: The results frame, and the cohorts actually present in it, in
        the requested order.
    :rtype: tuple[pd.DataFrame, list[str]]
    :raises FileNotFoundError: If the workbook does not exist.
    :raises ValueError: If no sheet carries the required result columns, or if
        none of the requested cohorts is present.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"significance export not found: {path}")

    wanted = [str(f) for f in fuel_order]
    workbook = pd.ExcelFile(path)

    # -- Layout 1: one sheet, all cohorts ---------------------------------
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)

        if not REQUIRED_COLUMNS.issubset(frame.columns):
            continue
        if 'fuel_type' not in frame.columns:
            continue

        # Match case-insensitively but return the spelling used in the data,
        # since the generator looks cohorts up against the raw fuel_type
        # values. Returning the requested spelling would silently produce an
        # empty column wherever the two differ in case.
        available = {str(v).strip().lower(): str(v)
                     for v in frame['fuel_type'].dropna().unique()}
        present = [available[f.strip().lower()] for f in wanted
                   if f.strip().lower() in available]

        if not present:
            raise ValueError(
                f"{path.name}: sheet {sheet!r} holds cohorts "
                f"{sorted(available.values())}, none of which are in the "
                f"requested order {wanted}"
            )

        if verbose:
            skipped = [f for f in wanted if f.strip().lower() not in available]
            print(f"  {path.name}: pooled layout, sheet {sheet!r}, "
                  f"{len(frame)} rows, cohorts {present}"
                  + (f", absent {skipped}" if skipped else ""))

        return frame, present

    # -- Layout 2: one sheet per cohort ------------------------------------
    lookup = {str(s).strip().lower(): s for s in workbook.sheet_names}

    frames, present = [], []
    for fuel in wanted:
        sheet = lookup.get(fuel.strip().lower())
        if sheet is None:
            continue

        frame = workbook.parse(sheet)
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            continue

        if 'fuel_type' not in frame.columns:
            frame = frame.assign(fuel_type=fuel)

        frames.append(frame)
        present.append(fuel)

    if frames:
        if verbose:
            skipped = [f for f in wanted if f not in present]
            print(f"  {path.name}: per-cohort layout, cohorts {present}"
                  + (f", absent {skipped}" if skipped else ""))

        return pd.concat(frames, ignore_index=True), present

    raise ValueError(
        f"{path.name}: no sheet carries the result columns "
        f"{sorted(REQUIRED_COLUMNS)} for any of {wanted}. "
        f"Sheets present: {workbook.sheet_names}. "
        f"Run inspect_workbook() on this file to see what it contains."
    )


# =============================================================================
# Captions
# =============================================================================

def experiment_1_caption(ship_label: str) -> str:
    """
    Caption for an empirical-formula significance table.

    :param ship_label: Ship type as it should read in prose.
    :type ship_label: str
    :returns: LaTeX caption.
    :rtype: str
    """
    return (
        rf'Bootstrap significance results for published empirical formulae, '
        rf'\textbf{{{ship_label}}}. '
        r'$\Delta_{NRMSE}$ is the change in NRMSE from the matched '
        r'conventional dataset to the alternative-fuel dataset; positive '
        r'values denote degradation and negative values improved performance. '
        r'Brackets give BCa bootstrap confidence intervals, and \textbf{bold} '
        r'denotes an interval excluding zero.'
    )


def experiment_2_caption(ship_label: str, model_label: str) -> str:
    """
    Caption for a selected-model significance table.

    :param ship_label: Ship type as it should read in prose.
    :type ship_label: str
    :param model_label: Model family, e.g. ``'random forest'``.
    :type model_label: str
    :returns: LaTeX caption.
    :rtype: str
    """
    return (
        rf'Selected {model_label} models for designing '
        rf'\textbf{{{ship_label}}}. '
        r'$\Delta_{NRMSE}$ is the change in NRMSE from the matched '
        r'conventional dataset to the alternative-fuel dataset; positive '
        r'values denote degradation and negative values improved performance. '
        r'Brackets give BCa bootstrap confidence intervals, and \textbf{bold} '
        r'denotes an interval excluding zero.'
    )


# =============================================================================
# Experiment 1
# =============================================================================

def run_experiment_1(inspect: bool = False) -> None:
    """
    Generate one empirical-formula significance table per ship type.

    Rows are keyed on ``('formula_name', 'reference')`` because several
    formulae share a target, and two sources publish the same functional form,
    so neither column alone identifies a row.

    :param inspect: Print the sheet inventory of each workbook first.
    :type inspect: bool
    """
    if inspect:
        print("\nExperiment 1 workbook inventory")
        for ship_type, directory in EXPERIMENT_1_DIRECTORIES.items():
            print(f"{ship_type.name}:")
            try:
                inspect_workbook(
                    resolve_export(directory, EXPERIMENT_1_FILENAME))
            except FileNotFoundError as error:
                print(f"  {error}")

    generator = SignificanceTableGenerator(target_labels=TARGET_LABELS)

    print("\nExperiment 1 tables")
    failures = []

    for ship_type, directory in EXPERIMENT_1_DIRECTORIES.items():

        # One unreadable export should not prevent the other ship types from
        # regenerating; the failures are collected and reported at the end so
        # they cannot be lost in the scrollback.
        try:
            _, _, _, _, save_dir, alt_fuel_data = get_variables(
                shiptype=ship_type)

            path = resolve_export(directory, EXPERIMENT_1_FILENAME)

            results, present_fuels = load_significance_results(
                path, fuel_order=list(alt_fuel_data.keys())
            )

            output_path = (save_dir / 'experiment1' / 'significance'
                           / f'{ship_type.name}_significance_table.tex')

            generator.generate_to_file(
                output_path=output_path,
                source=results,
                caption=experiment_1_caption(ship_type.string),
                label=f'tab_{ship_type.name}_empirical_significance',
                row_key=('formula_name', 'reference'),
                row_header='Formula',
                fuel_order=present_fuels,
                fuel_labels=FUEL_LABELS,
                strict_intervals=True,
            )

        except (FileNotFoundError, ValueError) as error:
            print(f"  SKIPPED {ship_type.name}: {error}")
            failures.append((ship_type.name, error))

    if failures:
        print(f"\n{len(failures)} ship type(s) not generated: "
              f"{[name for name, _ in failures]}")


# =============================================================================
# Experiment 2
# =============================================================================

def run_experiment_2(inspect: bool = False) -> None:
    """
    Generate parametric and random forest significance tables per ship type.

    Rows are keyed on ``'target'``, which is sufficient here because model
    selection yields one model per target.

    :param inspect: Print the sheet inventory of each workbook first.
    :type inspect: bool
    """
    if inspect:
        print("\nExperiment 2 workbook inventory")
        for ship_type, (ship_dir, file_stem) in EXPERIMENT_2_SPECS.items():
            print(f"{ship_type.name}:")
            for model_kind in MODEL_KINDS:
                inspect_workbook(
                    experiment_2_path(ship_dir, file_stem, model_kind))

    generator = SignificanceTableGenerator(target_labels=TARGET_LABELS)

    print("\nExperiment 2 tables")
    failures = []

    for ship_type, (ship_dir, file_stem) in EXPERIMENT_2_SPECS.items():

        for model_kind, (_, _, model_label) in MODEL_KINDS.items():

            try:
                _, _, _, _, save_dir, alt_fuel_data = get_variables(
                    shiptype=ship_type)

                path = experiment_2_path(ship_dir, file_stem, model_kind)

                results, present_fuels = load_significance_results(
                    path, fuel_order=list(alt_fuel_data.keys())
                )

                output_path = (
                    save_dir / 'experiment2' / 'significance'
                    / f'{ship_type.name}_{model_kind}_significance_table.tex'
                )

                generator.generate_to_file(
                    output_path=output_path,
                    source=results,
                    caption=experiment_2_caption(ship_type.string,
                                                 model_label),
                    label=f'tab_{ship_type.name}_{model_kind}_significance',
                    row_key='target',
                    row_header='Target',
                    fuel_order=present_fuels,
                    fuel_labels=FUEL_LABELS,
                    strict_intervals=True,
                )

            except (FileNotFoundError, ValueError) as error:
                print(f"  SKIPPED {ship_type.name}/{model_kind}: {error}")
                failures.append((f'{ship_type.name}/{model_kind}', error))

    if failures:
        print(f"\n{len(failures)} table(s) not generated: "
              f"{[name for name, _ in failures]}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    if RUN_EXPERIMENT_1:
        run_experiment_1(inspect=INSPECT_WORKBOOKS)

    if RUN_EXPERIMENT_2:
        run_experiment_2(inspect=INSPECT_WORKBOOKS)