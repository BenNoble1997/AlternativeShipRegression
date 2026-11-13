"""
Reference-scale reporting for denominator-fixed error metrics.

Computes the per-target reference scales used to normalise NRMSE, together with
each fuel cohort's own mean and the ratio between them, and emits both an Excel
workbook and pasteable LaTeX.

**Concept:**
Relative error metrics divide RMSE by a measure of target magnitude. When that
divisor is taken from the evaluation cohort itself, values computed on different
cohorts are not commensurable: a cohort of larger vessels receives a smaller
relative error for an identical absolute error. Fixing the divisor at the
conventional-cohort mean makes the metric a linear rescaling of RMSE, comparable
across cohorts.

The ratio column quantifies exactly how much the old cohort-local convention was
flattering each cohort. A ratio of 2.0 means reported errors were half what they
should have been relative to a common scale.

The reference is built via ``ScaleReference.from_frame`` rather than being
recomputed here, so the reported values cannot drift from the ones the analysis
pipeline actually used.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (
    getting_alt_fuel_ships,
    get_variables,
)
from knowledge_layer.multi_dataset_analysers.cross_dataset.ScaleReference import (
    ScaleReference,
)
from parameters.shipshape.ShipShapeBasic_class import ShipType


# ============================================================================
# CONFIGURATION
# ============================================================================

TYPES_LIST = [
    ShipType.BULK_CARRIER,
    ShipType.CONTAINER_SHIP,
    ShipType.TANKER,
    ShipType.RO_PAX,
]

#: Cohort label -> key in ``fleet.processed_data``. The first entry is the
#: reference cohort and must remain first.
COHORT_KEYS = {
    'Conventional': 'pure_conventional',
    'Dual fuel': 'dual_fuel_Alt_fuel_alt_main',
    'Biofuel': 'pure_Biofuel',
}

REFERENCE_COHORT = 'Conventional'

FUEL_CONFIG = {
    'max_build_year': 1990,
    'fuel_type_columns': ['MAIN_FUEL', 'BUNKER_FUEL'],
    'conventional_fuels': ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO'],
    'alt_fuels': {
        'Alt_fuel': ['LNG', 'Methanol', 'Ethanol', 'Ammonia',
                     'Hydrogen', 'cng', 'ethane'],
        'Biofuel': ['Biofuel'],
    },
}

TARGET_LABELS = {
    'BEAM (M)': r'\gls{b}', 'DEPTH (M)': r'\gls{d}', 'DRAFT (M)': r'\gls{t}',
    'DWT': r'\gls{dwt}', 'LBP (M)': r'\gls{lbp}', 'LOA (M)': r'\gls{loa}',
    'MCR (KW)': r'\gls{mcr}', 'TEU': r'\gls{teu}', 'Vs': r'\gls{vs}',
    'Np': r'\gls{np}', 'Nv': r'\gls{nv}', 'LSW': r'\gls{lsw}',
}


@dataclass(frozen=True)
class TargetFormat:
    """
    Display convention for one target dimension.

    Reference scales span five orders of magnitude across targets, so a single
    numeric format is unreadable. Each target is rescaled into a natural unit
    before printing.

    :param divisor: Factor applied before display (e.g. 1000 to show kW as MW).
    :param decimals: Decimal places after rescaling.
    :param unit: Unit label, empty for dimensionless counts.
    :param siunitx: siunitx unit macro, empty for dimensionless counts.
    """

    divisor: float = 1.0
    decimals: int = 1
    unit: str = ''
    siunitx: str = ''


TARGET_FORMATS = {
    'BEAM (M)': TargetFormat(1.0, 1, 'm', r'\metre'),
    'DEPTH (M)': TargetFormat(1.0, 1, 'm', r'\metre'),
    'DRAFT (M)': TargetFormat(1.0, 2, 'm', r'\metre'),
    'LBP (M)': TargetFormat(1.0, 1, 'm', r'\metre'),
    'LOA (M)': TargetFormat(1.0, 1, 'm', r'\metre'),
    'DWT': TargetFormat(1000.0, 1, 'kt', r'\kilo\tonne'),
    'LSW': TargetFormat(1000.0, 1, 'kt', r'\kilo\tonne'),
    'MCR (KW)': TargetFormat(1000.0, 1, 'MW', r'\mega\watt'),
    'TEU': TargetFormat(1.0, 0, '', ''),
    'Np': TargetFormat(1.0, 0, '', ''),
    'Nv': TargetFormat(1.0, 0, '', ''),
    'Vs': TargetFormat(1.0, 1, 'kn', r'\knot'),
}

DEFAULT_FORMAT = TargetFormat(1.0, 2, '', '')


# ============================================================================
# LOADING
# ============================================================================

def load_fleets(types_list: list[ShipType] | None = None,
                debug: bool = False) -> dict[str, object]:
    """
    Load fuel-classified fleets for each ship type.

    :param types_list: Ship types to load. Defaults to ``TYPES_LIST``.
    :param debug: Print per-ship-type progress and cohort sizes.
    :returns: Mapping of ship-type string to the loaded fleet object.
    """
    types_list = TYPES_LIST if types_list is None else types_list
    fleets: dict[str, object] = {}

    for shiptype in types_list:
        _, _, _, filepath, _ = get_variables(shiptype=shiptype)

        if debug:
            print(f"[load] {shiptype.string}: reading {filepath}")

        fleets[shiptype.string] = getting_alt_fuel_ships(
            filepath=filepath,
            shiptype=shiptype.string,
            fuel_type_columns=FUEL_CONFIG['fuel_type_columns'],
            alt_fuels=FUEL_CONFIG['alt_fuels'],
            conventional_fuels=FUEL_CONFIG['conventional_fuels'],
            max_build_year=FUEL_CONFIG['max_build_year'],
        )

        if debug:
            for label, key in COHORT_KEYS.items():
                cohort = fleets[shiptype.string].processed_data.get(key)
                size = 'absent' if cohort is None else f"n={len(cohort)}"
                print(f"[load]   {label:<14s} {size}")

    return fleets


# ============================================================================
# COMPUTATION
# ============================================================================

def complete_case_mean(cohort: pd.DataFrame,
                       target: str) -> tuple[float, int]:
    """
    Mean of one target over its finite values.

    Complete cases are taken on the target column only, never on target plus
    predictors. A per-combination divisor would give the same target different
    scales in different rows of the same table, breaking within-table
    comparability.

    :param cohort: Cohort frame.
    :param target: Target column name.
    :returns: ``(mean, n)``, or ``(nan, 0)`` if the target is absent or empty.
    """
    if target not in cohort.columns:
        return float('nan'), 0

    values = pd.to_numeric(cohort[target], errors='coerce').to_numpy()
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return float('nan'), 0

    return float(np.mean(finite)), int(finite.size)


def build_scale_reference_frame(fleets: dict[str, object],
                                targets_by_type: dict[str, list[str]],
                                debug: bool = False) -> pd.DataFrame:
    """
    Assemble reference scales, cohort means and their ratios.

    The reference column is produced by ``ScaleReference.from_frame`` so it is
    guaranteed identical to the divisor used during evaluation. Cohort means are
    computed here for reporting only.

    :param fleets: Mapping of ship-type string to loaded fleet object.
    :param targets_by_type: Mapping of ship-type string to its target list.
    :param debug: Print per-target reference values and ratios.
    :returns: Long-form frame, one row per (ship type, target, cohort).
    """
    records = []

    for ship_label, fleet in fleets.items():
        targets = targets_by_type.get(ship_label, [])
        reference_key = COHORT_KEYS[REFERENCE_COHORT]
        reference_frame = fleet.processed_data.get(reference_key)

        if reference_frame is None:
            print(f"[scale] {ship_label}: reference cohort "
                  f"'{reference_key}' absent - skipped")
            continue

        available = [t for t in targets if t in reference_frame.columns]
        missing = sorted(set(targets) - set(available))

        if missing:
            print(f"[scale] {ship_label}: targets absent from reference "
                  f"cohort, skipped: {missing}")

        if not available:
            continue

        reference = ScaleReference.from_frame(
            data=reference_frame,
            targets=available,
            source='conventional_cohort_mean',
        )

        if debug:
            print(f"\n[scale] {ship_label} reference ({reference.source}):")

        for target in available:
            reference_mean = reference.for_target(target)
            _, n_reference = complete_case_mean(reference_frame, target)

            if debug:
                print(f"[scale]   {target:<12s} {reference_mean:>12.2f} "
                      f"(n={n_reference})")

            for cohort_label, cohort_key in COHORT_KEYS.items():
                cohort = fleet.processed_data.get(cohort_key)

                if cohort is None:
                    continue

                cohort_mean, n_cohort = complete_case_mean(cohort, target)
                ratio = (cohort_mean / reference_mean
                         if np.isfinite(cohort_mean) and reference_mean else np.nan)

                records.append({
                    'ship_type': ship_label,
                    'target': target,
                    'reference_source': reference.source,
                    'reference_mean': reference_mean,
                    'n_reference': n_reference,
                    'cohort': cohort_label,
                    'cohort_mean': cohort_mean,
                    'n_cohort': n_cohort,
                    'ratio_to_reference': ratio,
                })

                if debug and cohort_label != REFERENCE_COHORT:
                    print(f"[scale]     {cohort_label:<12s} "
                          f"mean={cohort_mean:>12.2f} (n={n_cohort:>4d}) "
                          f"ratio={ratio:.3f}")

    return pd.DataFrame(records)


def check_dimensional_consistency(frame: pd.DataFrame,
                                  linear_targets: tuple[str, ...] = (
                                      'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)',
                                      'LBP (M)', 'LOA (M)'),
                                  capacity_targets: tuple[str, ...] = (
                                      'DWT', 'TEU'),
                                  debug: bool = False) -> pd.DataFrame:
    """
    Check linear-dimension ratios against the cube root of capacity ratios.

    Linear dimensions scale as the cube root of displaced volume, so for a
    cohort whose capacity ratio is :math:`r`, linear ratios should sit near
    :math:`r^{1/3}`. Large departures indicate either a genuine change in hull
    proportions or an error in cohort construction, and are worth resolving
    before the ratios are published.

    :param frame: Long-form output of :func:`build_scale_reference_frame`.
    :param linear_targets: Targets expected to scale with the cube root.
    :param capacity_targets: Targets treated as capacity measures.
    :param debug: Print each comparison.
    :returns: One row per (ship type, cohort) with observed and expected ratios.
    """
    records = []
    off_reference = frame[frame['cohort'] != REFERENCE_COHORT]

    for (ship_label, cohort_label), group in off_reference.groupby(
            ['ship_type', 'cohort'], sort=False):

        capacity = group[group['target'].isin(capacity_targets)]
        linear = group[group['target'].isin(linear_targets)]

        capacity_ratios = capacity['ratio_to_reference'].dropna()
        linear_ratios = linear['ratio_to_reference'].dropna()

        if capacity_ratios.empty or linear_ratios.empty:
            continue

        capacity_ratio = float(capacity_ratios.mean())
        expected = capacity_ratio ** (1.0 / 3.0)
        observed = float(linear_ratios.mean())

        records.append({
            'ship_type': ship_label,
            'cohort': cohort_label,
            'capacity_ratio': capacity_ratio,
            'expected_linear_ratio': expected,
            'observed_linear_ratio': observed,
            'discrepancy': observed - expected,
        })

        if debug:
            print(f"[cuberoot] {ship_label} / {cohort_label}: "
                  f"capacity {capacity_ratio:.3f} -> expected linear "
                  f"{expected:.3f}, observed {observed:.3f} "
                  f"(delta {observed - expected:+.3f})")

    return pd.DataFrame(records)


# ============================================================================
# LATEX
# ============================================================================

def format_reference_value(target: str,
                           value: float,
                           use_siunitx: bool = False) -> str:
    """
    Render one reference scale in its natural unit.

    :param target: Target column name, used to select the display convention.
    :param value: Reference mean in the target's native units.
    :param use_siunitx: Emit ``\\SI{}{}`` rather than a plain thin-spaced unit.
    :returns: LaTeX-ready cell content.
    """
    if not np.isfinite(value):
        return '--'

    fmt = TARGET_FORMATS.get(target, DEFAULT_FORMAT)
    scaled = value / fmt.divisor
    number = f"{scaled:,.{fmt.decimals}f}"

    if not fmt.unit:
        return number

    if use_siunitx:
        return rf"\SI{{{scaled:.{fmt.decimals}f}}}{{{fmt.siunitx}}}"

    return rf"{number}\,{fmt.unit}"


def generate_reference_column(frame: pd.DataFrame,
                              ship_type: str,
                              target_order: list[str],
                              use_siunitx: bool = False,
                              debug: bool = False) -> str:
    """
    Emit the reference-scale cells for one ship type, in table row order.

    A column cannot be appended as a block: each cell must be inserted into its
    row. The output is therefore one cell per line, ordered to match
    ``target_order``, alongside the header cell and the revised ``\\cmidrule``
    indices that inserting a column requires.

    :param frame: Long-form output of :func:`build_scale_reference_frame`.
    :param ship_type: Ship-type string to emit.
    :param target_order: Targets in the order they appear as table rows.
    :param use_siunitx: Emit ``\\SI{}{}`` rather than plain units.
    :param debug: Print each cell as it is generated.
    :returns: Comment-annotated LaTeX fragment.
    """
    subset = frame[(frame['ship_type'] == ship_type) &
                   (frame['cohort'] == REFERENCE_COHORT)]
    lookup = dict(zip(subset['target'], subset['reference_mean']))

    lines = [
        f"% Reference scales for {ship_type}",
        r"% 1. Add one p{} column to \begin{tabular}{...} after the Target column",
        r"% 2. Insert the header cell:  Target & $\bar{y}_{\mathrm{ref}}$ & Predictors & ...",
        r"% 3. Shift every \cmidrule index up by one: {4-5}->{5-6}, {6-7}->{7-8}, {8-9}->{9-10}",
        r"% 4. Insert one cell below into each row, immediately after the target",
        "",
    ]

    for target in target_order:
        if target not in lookup:
            if debug:
                print(f"[latex] {ship_type}: no reference for '{target}'")
            lines.append(f"% {target}: NO REFERENCE FOUND")
            continue

        label = TARGET_LABELS.get(target, target)
        cell = format_reference_value(target, lookup[target], use_siunitx)
        lines.append(f"{cell}    % {label}  ({target})")

        if debug:
            print(f"[latex] {ship_type}: {target:<12s} -> {cell}")

    return "\n".join(lines)


def generate_appendix_table(frame: pd.DataFrame,
                            target_order_by_type: dict[str, list[str]],
                            label: str = 'tab:scale_reference',
                            use_siunitx: bool = False,
                            debug: bool = False) -> str:
    """
    Emit the full appendix table of reference scales, cohort means and ratios.

    This is the table that makes the normalisation auditable: any pre-correction
    NRMSE can be reconstructed by multiplying the reported value by the ratio for
    its cohort.

    :param frame: Long-form output of :func:`build_scale_reference_frame`.
    :param target_order_by_type: Mapping of ship-type string to row order.
    :param label: LaTeX label for the table.
    :param use_siunitx: Emit ``\\SI{}{}`` rather than plain units.
    :param debug: Print each row as it is generated.
    :returns: Complete ``table*`` environment.
    """
    off_reference = [c for c in COHORT_KEYS if c != REFERENCE_COHORT]
    n_columns = 3 + 2 * len(off_reference)

    header_groups = "".join(
        rf" & \multicolumn{{2}}{{c}}{{{cohort}}}" for cohort in off_reference
    )
    midrules = "".join(
        rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}"
        for i in range(len(off_reference))
    )
    subheader = "".join(" & Mean & Ratio" for _ in off_reference)

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Reference scales used to normalise \gls{nrmse}, with cohort "
        r"means and their ratios to the reference. The reference is the "
        r"conventional-cohort mean of each target, applied identically to every "
        r"cohort and both model families, so that reported \gls{nrmse} is a "
        r"linear rescaling of \gls{rmse} and is comparable across cohorts. "
        r"Ratios above unity indicate a cohort whose vessels exceed the "
        r"conventional fleet average; under cohort-local normalisation such a "
        r"cohort receives a proportionally smaller relative error for identical "
        r"absolute error.}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{ll{'r' * (n_columns - 2)}}}",
        r"\toprule",
        rf" & &{header_groups}\\",
        midrules,
        rf"Ship type & Target & $\bar{{y}}_{{\mathrm{{ref}}}}$\tnote{{a}}"
        rf"{subheader}\\",
        r"\midrule",
    ]

    for ship_index, (ship_label, targets) in enumerate(
            target_order_by_type.items()):

        subset = frame[frame['ship_type'] == ship_label]

        if subset.empty:
            continue

        if ship_index > 0:
            lines.append(r"\midrule")

        emitted = 0

        for target in targets:
            rows = subset[subset['target'] == target]

            if rows.empty:
                continue

            reference_rows = rows[rows['cohort'] == REFERENCE_COHORT]

            if reference_rows.empty:
                continue

            reference_mean = float(reference_rows['reference_mean'].iloc[0])
            cells = [
                ship_label if emitted == 0 else '',
                TARGET_LABELS.get(target, target),
                format_reference_value(target, reference_mean, use_siunitx),
            ]

            for cohort_label in off_reference:
                cohort_rows = rows[rows['cohort'] == cohort_label]

                if cohort_rows.empty:
                    cells.extend(['--', '--'])
                    continue

                cohort_mean = float(cohort_rows['cohort_mean'].iloc[0])
                ratio = float(cohort_rows['ratio_to_reference'].iloc[0])
                cells.append(
                    format_reference_value(target, cohort_mean, use_siunitx))
                cells.append(
                    f"{ratio:.2f}" if np.isfinite(ratio) else '--')

            lines.append(" & ".join(cells) + r"\\")
            emitted += 1

            if debug:
                print(f"[appendix] {ship_label} / {target}: emitted")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item[a] Conventional-cohort mean, computed on complete cases for the "
        r"target only, so that a single scale applies to every predictor "
        r"combination for that target.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_scale_reference_report(output_dir: Path,
                               types_list: list[ShipType] | None = None,
                               use_siunitx: bool = False,
                               debug: bool = True) -> pd.DataFrame:
    """
    Compute reference scales, write the workbook, and emit the LaTeX fragments.

    :param output_dir: Directory for the workbook and ``.tex`` files.
    :param types_list: Ship types to process. Defaults to ``TYPES_LIST``.
    :param use_siunitx: Emit ``\\SI{}{}`` rather than plain units.
    :param debug: Print per-stage progress.
    :returns: The long-form reference frame.
    """
    types_list = TYPES_LIST if types_list is None else types_list
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fleets = load_fleets(types_list=types_list, debug=debug)

    targets_by_type = {}
    for shiptype in types_list:
        _, variables, _, _, _ = get_variables(shiptype=shiptype)
        targets_by_type[shiptype.string] = list(variables)

    frame = build_scale_reference_frame(
        fleets=fleets,
        targets_by_type=targets_by_type,
        debug=debug,
    )

    if frame.empty:
        raise ValueError(
            "No reference scales produced. Check that processed_data contains "
            f"'{COHORT_KEYS[REFERENCE_COHORT]}' and that get_variables returns "
            "target names matching the cohort columns."
        )

    consistency = check_dimensional_consistency(frame, debug=debug)

    wide = frame.pivot_table(
        index=['ship_type', 'target', 'reference_mean'],
        columns='cohort',
        values=['cohort_mean', 'n_cohort', 'ratio_to_reference'],
    ).reset_index()

    workbook = output_dir / 'scale_reference.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name='long', index=False)
        wide.to_excel(writer, sheet_name='wide')
        consistency.to_excel(writer, sheet_name='cube_root_check', index=False)

    if debug:
        print(f"\n[write] workbook -> {workbook}")

    for ship_label, targets in targets_by_type.items():
        fragment = generate_reference_column(
            frame=frame,
            ship_type=ship_label,
            target_order=targets,
            use_siunitx=use_siunitx,
            debug=debug,
        )
        safe = ship_label.replace(' ', '_')
        path = output_dir / f'reference_column_{safe}.tex'
        path.write_text(fragment, encoding='utf-8')

        if debug:
            print(f"[write] column   -> {path}")

    appendix = generate_appendix_table(
        frame=frame,
        target_order_by_type=targets_by_type,
        use_siunitx=use_siunitx,
        debug=debug,
    )
    appendix_path = output_dir / 'scale_reference_appendix.tex'
    appendix_path.write_text(appendix, encoding='utf-8')

    if debug:
        print(f"[write] appendix -> {appendix_path}")

    return frame


if __name__ == "__main__":
    OUTPUT_DIR = Path(
        r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer"
        r"\dataset_storage_and_testing\data_processing_scripts\output"
    )

    run_scale_reference_report(
        output_dir=OUTPUT_DIR,
        use_siunitx=False,
        debug=True,
    )