"""
Diagnostic plots for two matching design decisions.

The script answers two questions visually, without summary statistics:

1. Does incorporating build year into the matching change which conventional
   ships are selected, and does it matter whether build year enters as a
   continuous matching feature or as an exact-matching bin?

2. How does tightening the caliper change the matched cohort, and at what
   width do alternative ships start losing their partners entirely?

Both are answered with the same display: DWT on the abscissa against each
target variable in turn, with the matched conventional ships and the actual
alternative ships overlaid in two colours. If the matched cloud envelops the
alternative points, the control cohort spans the design region the models
will be asked to predict. Where it does not, the matched set is extrapolating.
"""
from __future__ import annotations

import logging
from math import ceil, sqrt
from pathlib import Path
from typing import Any
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import (
    get_variables,
    getting_alt_fuel_ships,
)
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from parameters.shipshape.ShipShapeBasic_class import ShipType

logger = logging.getLogger(__name__)

CONV_COLOUR = '#2c7fb8'
ALT_COLOUR = '#d95f02'
X_VARIABLE = 'DWT'

# Open markers in distinct shapes for the matched variants, so overlapping
# cohorts stay separable. Solid fill is reserved for the alternative cohort.
MATCHED_PALETTE = ('#2c7fb8', '#41ae76', '#8856a7', '#e6ab02')
MATCHED_MARKERS = ('o', 's', '^', 'D')


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------




def report_coverage(alternative: pd.DataFrame,
                    conventional: pd.DataFrame,
                    columns: list[str]) -> None:
    """
    Log non-null coverage for candidate matching features.

    Matching drops any ship missing a matching feature, so a feature with poor
    coverage in the alternative cohort costs sample size directly. Worth
    checking before committing to speed or build year as a matching variable.

    :param columns: Candidate matching features to report on.
    """
    for col in columns:
        if col not in alternative.columns:
            logger.warning(f'{col} absent from alternative cohort')
            continue
        alt_pct = alternative[col].notna().mean() * 100
        conv_pct = conventional[col].notna().mean() * 100
        logger.info(f'{col} coverage: alternative {alt_pct:.1f}%, conventional {conv_pct:.1f}%')


def add_build_year_bin(frames: list[pd.DataFrame],
                       width: int = 5,
                       source_col: str = 'BUILT',
                       bin_col: str = 'BUILT_BIN') -> None:
    """
    Add a coarsened build-year column to each frame, in place.

    Exact matching requires a discrete key, so the continuous build year is
    floored to the start of its window. Ships then match only within the same
    window, which imposes a hard boundary: a 1999 hull cannot match a 2000
    hull however close they are in every other respect. That discontinuity is
    the cost of exact matching, and comparing this variant against the
    continuous one is the point of the exercise.

    :param frames: Frames to modify in place.
    :param width: Window width in years.
    :param source_col: Continuous build year column.
    :param bin_col: Name of the coarsened column to create.
    """
    for frame in frames:
        frame[bin_col] = (frame[source_col] // width) * width


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def caliper_from_rms(rms_sd: float,
                     matching_features: list[str],
                     feature_weights: dict[str, float] | None = None) -> float:
    """
    Convert a per-feature RMS discrepancy into a caliper on the joint distance.

    The distance is ``sqrt(sum_i w_i z_i^2)`` with ``z_i`` the standardised
    difference in feature ``i``. If every feature is displaced by the same
    ``z``, the distance reduces to ``z * sqrt(sum_i w_i)``. Specifying the
    caliper through ``rms_sd`` therefore makes it interpretable as an average
    per-feature tolerance in standard deviations, and keeps its meaning stable
    when features or weights are added.

    :param rms_sd: Tolerated per-feature discrepancy, in standard deviations.
    :param matching_features: Features entering the distance.
    :param feature_weights: Per-feature weights, or None for unit weights.
    :returns: Caliper on the joint standardised Euclidean distance.
    :rtype: float
    """
    if feature_weights is None:
        total_weight = float(len(matching_features))
    else:
        total_weight = sum(float(feature_weights[f]) for f in matching_features)

    return rms_sd * sqrt(total_weight)


def run_match(alternative: pd.DataFrame,
              conventional: pd.DataFrame,
              matching_features: list[str],
              feature_weights: dict[str, float] | None = None,
              binning_features: list[str] | None = None,
              caliper: float | None = None,
              n_matches_per_ship: int = 10) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """
    Build one matched conventional cohort.

    Matching is with replacement: every alternative ship searches the full
    conventional pool independently, and the selected ships are pooled and
    deduplicated only at the end. The match records are returned alongside the
    deduplicated frame because the number of times each conventional ship was
    selected is lost in the deduplication, and that multiplicity is what the
    reuse diagnostics are computed from.

    :returns: Matched conventional frame, the per-pair match records, and the
        number of alternative ships that secured at least one partner.
    :rtype: tuple[pd.DataFrame, list[dict[str, Any]], int]
    """

    print(binning_features)
    match_details, matched = MatchedDataset.create_matched_dataset(
        alternative_fleet=alternative,
        conventional_fleet=conventional,
        logger=logger,
        matching_features=matching_features,
        feature_weights=feature_weights,
        binning_features=binning_features,
        caliper=caliper,
        n_matches_per_ship=n_matches_per_ship,
        skip_missing_ships=True,
    )

    n_alt_matched = len({d['alt_ship_idx'] for d in match_details})

    return matched, match_details, n_alt_matched

def summarise_reuse(match_details: list[dict[str, Any]],
                    matched: pd.DataFrame,
                    label: str,
                    n_alternative: int,
                    n_alt_matched: int) -> dict[str, Any]:
    """
    Quantify how heavily conventional ships are reused within one variant.

    Matching with replacement lets several alternative ships select the same
    conventional partner, so the matched cohort is a reweighted view of the
    conventional fleet rather than a subset of it. Deduplication reduces it to
    distinct hulls and discards the multiplicity, and the size of that discard
    is what this reports.

    The effective sample size, ``(sum w)^2 / sum(w^2)`` over the selection
    counts, is the number of equally weighted ships that would carry the same
    information. It falls below the distinct-ship count whenever selections are
    concentrated, and it is the figure to quote when a variant appears to
    retain a large cohort that is in fact a few hulls counted repeatedly.

    :param match_details: Per-pair records from :func:`run_match`.
    :param matched: Deduplicated matched frame, used only for its row count.
    :param label: Variant name, carried into the output row.
    :param n_alternative: Size of the alternative cohort before matching.
    :param n_alt_matched: Alternative ships that secured at least one partner.
    :returns: One row of the reuse table.
    :rtype: dict[str, Any]
    """
    counts = pd.Series(
        [detail['conv_ship_idx'] for detail in match_details]
    ).value_counts()

    weights = counts.to_numpy(dtype=float)
    ess = weights.sum() ** 2 / (weights ** 2).sum() if weights.size else 0.0

    return {
        'variant': label,
        'n_alt_total': n_alternative,
        'n_alt_matched': n_alt_matched,
        'n_alt_unmatched': n_alternative - n_alt_matched,
        'n_pairs': len(match_details),
        'n_conv_unique': len(matched),
        'n_duplicates_dropped': len(match_details) - len(matched),
        'max_reuse': int(counts.max()) if counts.size else 0,
        'mean_reuse': float(counts.mean()) if counts.size else 0.0,
        'effective_sample_size': ess,
    }

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def plot_matched_comparison(alternative: pd.DataFrame,
                            matched: pd.DataFrame,
                            targets: list[str],
                            title: str,
                            save_path: Path | None = None,
                            n_cols: int = 3) -> plt.Figure:
    """
    Overlay the matched conventional and alternative cohorts across all targets.

    Matched ships are drawn first so the smaller alternative cohort sits on
    top and remains legible where the two overlap.

    :param targets: Ordinate variables, one panel each.
    :param title: Figure suptitle, used to identify the matching variant.
    :param save_path: Destination file, or None to skip saving.
    :returns: The figure, left open for interactive inspection.
    :rtype: plt.Figure
    """
    n_rows = ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, target in zip(flat_axes, targets):
        if target not in matched.columns or target not in alternative.columns:
            ax.set_visible(False)
            continue

        ax.scatter(matched[X_VARIABLE], matched[target],
                   s=16, alpha=0.45, color=CONV_COLOUR,
                   linewidths=0, label='Matched conventional')
        ax.scatter(alternative[X_VARIABLE], alternative[target],
                   s=26, alpha=0.85, color=ALT_COLOUR,
                   edgecolors='white', linewidths=0.4, label='Alternative fuel')

        ax.set_xlabel(X_VARIABLE)
        ax.set_ylabel(target)
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in flat_axes[len(targets):]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f'Saved {save_path}')

    return fig


def plot_matched_overlay(alternative: pd.DataFrame,
                         matched_sets: dict[str, pd.DataFrame],
                         targets: list[str],
                         title: str,
                         save_path: Path | None = None,
                         n_cols: int = 3) -> plt.Figure:
    """
    Overlay several matched cohorts on shared axes against the alternative fleet.

    Each matched variant is drawn with open markers in a distinct shape, so
    that cohorts occupying the same design region remain separable rather than
    the last-drawn one masking the others. The alternative fleet is drawn last,
    filled, larger, and at elevated z-order, so it reads as the reference layer
    the variants are being judged against.

    Insertion order of ``matched_sets`` sets the draw order. For nested cohorts
    such as a caliper sweep, pass them widest first so that progressively
    tighter subsets are drawn on top and the nesting is visible.

    :param alternative: Alternative fuel cohort, the reference layer.
    :param matched_sets: Mapping of variant label to matched conventional frame.
    :param targets: Ordinate variables, one panel each.
    :param title: Figure suptitle.
    :param save_path: Destination file, or None to skip saving.
    :returns: The figure, left open for interactive inspection.
    :rtype: plt.Figure
    """
    n_rows = ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, target in zip(flat_axes, targets):
        if target not in alternative.columns:
            ax.set_visible(False)
            continue

        for position, (label, frame) in enumerate(matched_sets.items()):
            if target not in frame.columns:
                continue
            ax.scatter(
                frame[X_VARIABLE], frame[target],
                s=34,
                facecolors='none',
                edgecolors=MATCHED_PALETTE[position % len(MATCHED_PALETTE)],
                marker=MATCHED_MARKERS[position % len(MATCHED_MARKERS)],
                linewidths=0.7,
                alpha=0.6,
                label=label,
            )

        ax.scatter(
            alternative[X_VARIABLE], alternative[target],
            s=52,
            color=ALT_COLOUR,
            alpha=1.0,
            edgecolors='white',
            linewidths=0.6,
            zorder=5,
            label='Alternative fuel',
        )

        ax.set_xlabel(X_VARIABLE)
        ax.set_ylabel(target)
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in flat_axes[len(targets):]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f'Saved {save_path}')

    return fig

def compare_identifiers(matched_sets: dict[str, pd.DataFrame],
                        id_col: str = 'MMSI') -> pd.DataFrame:
    """
    Compare the ship membership of every pair of matched cohorts.

    Membership is compared on an identifier column rather than on the frame
    index, so the result does not depend on how each cohort was assembled. The
    symmetric difference answers how far the variants disagree; the directional
    counts say which variant reached ships the other did not, which is the more
    informative quantity when one variant is a constrained version of the other.

    Jaccard similarity, the shared count over the union, normalises the
    disagreement by cohort size so that variants of unequal size stay
    comparable.

    :param matched_sets: Mapping of variant label to matched conventional frame.
    :param id_col: Identifier column present in every frame.
    :returns: One row per unordered pair of variants.
    :rtype: pd.DataFrame
    :raises KeyError: If ``id_col`` is absent from any frame.
    """
    identifiers: dict[str, set] = {}

    for label, frame in matched_sets.items():
        if id_col not in frame.columns:
            raise KeyError(
                f"Identifier column '{id_col}' not found in matched cohort "
                f"'{label}'. Available columns include {list(frame.columns)[:10]}."
            )

        column = frame[id_col]
        n_missing = column.isna().sum()
        if n_missing:
            logger.warning(
                f"{n_missing}/{len(frame)} ships in '{label}' have no {id_col} "
                f"and are excluded from the membership comparison"
            )

        values = column.dropna()
        n_duplicated = values.duplicated().sum()
        if n_duplicated:
            logger.warning(
                f"{n_duplicated} repeated {id_col} values in '{label}'. Distinct "
                f"hulls can share an {id_col} after reflagging, so the overlap "
                f"counts below are counts of identifiers, not of ships."
            )

        identifiers[label] = set(values)

    rows = []
    for (label_a, set_a), (label_b, set_b) in combinations(identifiers.items(), 2):
        shared = set_a & set_b
        union = set_a | set_b
        rows.append({
            'variant_a': label_a,
            'variant_b': label_b,
            f'n_{id_col.lower()}_a': len(set_a),
            f'n_{id_col.lower()}_b': len(set_b),
            'n_shared': len(shared),
            'n_only_a': len(set_a - set_b),
            'n_only_b': len(set_b - set_a),
            'n_different': len(set_a ^ set_b),
            'jaccard': len(shared) / len(union) if union else float('nan'),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def build_year_experiment(alternative: pd.DataFrame,
                          conventional: pd.DataFrame,
                          targets: list[str],
                          save_dir: Path,
                          shiptype: str,
                          matching_features: list[str],
                          feature_weights: dict[str, float],
                          rms_widths: tuple[float, ...] = (0.50, 0.25, 0.10),
                          bin_width: int = 5,
                          id_col: str = 'MMSI',
                          n_matches_per_ship:int=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare build-year binning against no binning, across caliper widths.

    Both variants rank conventional candidates by the same standardised
    distance over ``matching_features``; the binned variant additionally
    restricts each alternative ship to conventional ships built in the same
    window. Crossing that with the caliper separates two distinct constraints:
    the caliper bounds how dissimilar a partner may be in capacity, the bin
    bounds how far apart in vintage.

    Combining the two is only meaningful because the matching class now
    estimates the distance metric once from the full conventional fleet. Under
    per-bin standardisation a single caliper would mean a different physical
    tolerance in each window.

    :param rms_widths: Per-feature RMS tolerances, in standard deviations.
    :param bin_width: Build-year window width, in years.
    :param id_col: Identifier used for the membership comparison.
    :returns: The reuse table and the membership table, both carrying a
        ``caliper_rms`` column.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    add_build_year_bin([alternative, conventional], width=bin_width)

    variants = [
        ('no_build_year', None,
         f'Build year excluded, matching on {", ".join(matching_features)}'),
        ('build_year_binned', ['BUILT_BIN'],
         f'Build year as exact-matching bin ({bin_width}-year windows)'),
    ]

    reuse_rows: list[dict[str, Any]] = []
    membership_frames: list[pd.DataFrame] = []

    for rms in sorted(rms_widths, reverse=True):
        caliper = caliper_from_rms(rms, matching_features, feature_weights)

        # Reset per caliper: cohorts from different widths are not alternatives
        # to each other and must not share an overlay or a membership table.
        collected: dict[str, pd.DataFrame] = {}
        caliper_dir = save_dir / 'matching_diagnostics' / f'caliper_{rms:.2f}'

        for slug, bins, description in variants:
            matched, match_details, n_alt_matched = run_match(
                alternative=alternative,
                conventional=conventional,
                matching_features=matching_features,
                feature_weights=feature_weights,
                binning_features=bins,
                caliper=caliper,
                n_matches_per_ship=n_matches_per_ship
            )

            fig = plot_matched_comparison(
                alternative=alternative,
                matched=matched,
                targets=targets,
                title=f'{shiptype}\n'
                      f'{description}, caliper {rms:.2f} SD\n'
                      f'{len(matched)} matched ships, '
                      f'{n_alt_matched}/{len(alternative)} alternative ships matched',
                save_path=caliper_dir / f'{slug}_bin{bin_width}.png',
            )
            plt.close(fig)

            label = f'{description.split(",")[0]} (n={len(matched)})'
            collected[label] = matched

            reuse_rows.append({'caliper_rms': rms} | summarise_reuse(
                match_details=match_details,
                matched=matched,
                label=label,
                n_alternative=len(alternative),
                n_alt_matched=n_alt_matched,
            ))

        fig = plot_matched_overlay(
            alternative=alternative,
            matched_sets=collected,
            targets=targets,
            title=f'{shiptype}\n'
                  f'Build year treatment at caliper {rms:.2f} SD '
                  f'({bin_width}-year bins)',
            save_path=caliper_dir / f'build_year_overlay_bin{bin_width}.png',
        )
        plt.close(fig)

        membership = compare_identifiers(collected, id_col=id_col)
        membership.insert(0, 'caliper_rms', rms)
        membership_frames.append(membership)

    reuse_table = pd.DataFrame(reuse_rows)
    membership_table = pd.concat(membership_frames, ignore_index=True)

    reuse_table.insert(0, 'ship_type', shiptype)
    membership_table.insert(0, 'ship_type', shiptype)

    write_diagnostics_workbook(
        tables={'reuse': reuse_table, 'membership': membership_table},
        save_path=save_dir / 'matching_diagnostics'
                  / f'build_year_diagnostics_bin{bin_width}_n{n_matches_per_ship}.xlsx',
    )

    return reuse_table, membership_table

def caliper_experiment(alternative: pd.DataFrame,
                       conventional: pd.DataFrame,
                       targets: list[str],
                       save_dir: Path,
                       shiptype:str,
                       matching_features: list[str],
                       feature_weights: dict[str, float],
                       rms_widths: tuple[float, ...] = (0.50, 0.25, 0.10)) -> None:
    """
    Compare matched cohorts under progressively tighter calipers.

    Each width is specified as a per-feature RMS tolerance and converted to a
    joint distance by :func:`caliper_from_rms`. Tightening the caliper trades
    comparability against sample size, and the count of alternative ships left
    unmatched is the quantity that should decide the final width: an
    alternative ship with no partner inside the caliper is one with no
    conventional analogue, which is a finding rather than a failure.
    """
    collected: dict[str, pd.DataFrame] = {}
    calipers = []

    # Widest first, so that the nested tighter subsets are drawn on top.
    for rms in sorted(rms_widths, reverse=True):
        caliper = caliper_from_rms(rms, matching_features, feature_weights)
        calipers.append(caliper)


    for caliper in calipers:

        matched, n_alt_matched = run_match(
            alternative=alternative,
            conventional=conventional,
            matching_features=matching_features,
            feature_weights=feature_weights,
            caliper=caliper,
        )

        n_dropped = len(alternative) - n_alt_matched
        plot_matched_comparison(
            alternative=alternative,
            matched=matched,
            targets=targets,
            title=f'{shiptype}\n'
                  f'Caliper {rms:.2f} SD per feature (joint distance {caliper:.3f})\n'
                  f'{len(matched)} matched ships, {n_dropped} alternative ships unmatched',
            save_path=save_dir / 'matching_diagnostics' / f'caliper_{rms:.2f}.png',
        )
        collected[f'Caliper {rms:.2f} SD (n={len(matched)})'] = matched



    plot_matched_overlay(
        alternative=alternative,
        matched_sets=collected,
        targets=targets,
        title=f'{shiptype}\n'
              'Caliper width: matched cohorts overlaid (nested, widest drawn first)',
        save_path=save_dir / 'matching_diagnostics' / 'caliper_overlay.png',
    )

def write_diagnostics_workbook(tables: dict[str, pd.DataFrame],
                               save_path: Path) -> None:
    """
    Write one sheet per diagnostic table to a single workbook.

    :param tables: Mapping of sheet name to frame. Sheet names are truncated to
        the 31-character limit Excel imposes.
    :param save_path: Destination workbook.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    logger.info(f'Saved {save_path}')


if __name__ == '__main__':

    types_list = [ShipType.BULK_CARRIER, ShipType.CONTAINER_SHIP, ShipType.TANKER, ShipType.RO_PAX]

    fleets = {}

    all_reuse: list[pd.DataFrame] = []
    all_membership: list[pd.DataFrame] = []

    bin_width=5
    n_matches_per_ship = 10

    for shiptype in types_list:
        bins, variables, _, filepath, save_dir = get_variables(shiptype)

        max_build_year = 1990

        fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
        conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

        alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
        bio_fuels = ['Biofuel']
        alt_fuels = {'Alt_fuel': alt_fuel,
                     'Biofuel': bio_fuels}

        fleet= getting_alt_fuel_ships(filepath=filepath,
                                                         shiptype=shiptype.string,
                                                         fuel_type_columns=fuel_type_columns,
                                                         alt_fuels=alt_fuels,
                                                         conventional_fuels=conventional_fuels,
                                                         max_build_year=max_build_year,
                                      outliers=False)

        save_dir = Path(
            r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\output")

        variable_labels = {'DWT': "DWT",
                           'LOA (M)': "$L_{OA}$ (m)", 'LBP (M)': "$L_{BP}$ (m)",
                           'BEAM (M)': "Beam (m)", 'DRAFT (M)': "Draught (m)",
                           'DEPTH (M)': "Depth (m)", 'MCR (KW)': "P$_{MCR}$ (kW)", 'TEU': "TEU",
                           'Vs': "$V_{s}$", 'Nv': "$N_{v}$", 'Np': "N_{p}", 'LM': "LM"
                           }


        # Speed is included so that it is always taken into account, per the
        # review comment. Check the coverage figures logged by build_cohorts
        # before relying on it: ships missing Vs are dropped from matching.
        BASE_FEATURES = ['DWT']
        BASE_WEIGHTS = {'DWT': 1}

        alternative = fleet.processed_data['dual_fuel_Alt_fuel_alt_main']
        conventional = fleet.processed_data['pure_conventional']
        targets = variables
        save_dir = save_dir / shiptype.string

        reuse, membership = build_year_experiment(
            alternative=alternative,
            conventional=conventional,
            targets=targets,
            save_dir=save_dir,
            shiptype=shiptype.string,
            bin_width=bin_width,
            matching_features=BASE_FEATURES,
            feature_weights=BASE_WEIGHTS,
            n_matches_per_ship=n_matches_per_ship
        )
        all_reuse.append(reuse)
        all_membership.append(membership)

        # caliper_experiment(
        #     alternative=alternative,
        #     conventional=conventional,
        #     targets=targets,
        #     save_dir=save_dir,
        #     matching_features=BASE_FEATURES,
        #     feature_weights=BASE_WEIGHTS,
        #     shiptype=shiptype.string
        # )
        #
        # plt.show()

    save_dir = Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\output")

    write_diagnostics_workbook(
        tables={'reuse': pd.concat(all_reuse, ignore_index=True),
                'membership': pd.concat(all_membership, ignore_index=True)},
        save_path=save_dir / f'build_year_bin_{bin_width}_n_{n_matches_per_ship}_diagnostics_all_types.xlsx',
    )