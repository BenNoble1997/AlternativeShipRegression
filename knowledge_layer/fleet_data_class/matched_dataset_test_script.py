from typing import Tuple, List, Dict, Any
import numpy as np
import logging
from pathlib import Path

from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset

from knowledge_layer.dataset_storage_and_testing.significance_table_generator import SignificanceTableGenerator
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships, get_variables
from parameters.shipshape.ShipShapeBasic_class import ShipType
import pandas as pd

from config.logger_config import LoggerMixin

min_year = 1990
match_number = 10

fuel_type_columns = ['MAIN_FUEL', 'AUX_FUEL', "BUNKER_FUEL"]
alt_fuels = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
bio_fuels = ['biofuel']

alt_fuels = {'Alt_fuel': alt_fuels,
             'Biofuel': bio_fuels}

conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

datasets = {"Dual fuel":"dual_fuel_Alt_fuel",
            "Biofuel":"pure_Biofuel"}


def compare_matched_cohorts(cohorts: Dict[str, tuple],
                            alternative_fleet: pd.DataFrame,
                            conventional_fleet: pd.DataFrame,
                            report_features: List[str],
                            logger: logging.Logger | None = None,
                            verbose: bool = False,
                            debug: bool = False) -> pd.DataFrame:
    """
    Compare matched cohorts built under different matching configurations.

    Each cohort is summarised on three axes that trade against one
    another, so that a configuration cannot be judged on balance alone.

    Balance is reported twice, because the two readings answer different
    questions. The pair-level statistic averages over matches and measures
    how close each treated ship's own controls are, which is the quantity
    the distance was minimising. The cohort-level statistic compares the
    treated mean against the deduplicated control mean, which is the
    quantity that actually enters a model fitted on the cohort. They
    diverge when controls are reused unevenly, since a control matched to
    five treated ships contributes five times to the pair statistic but
    once to the cohort statistic.

    Concentration is reported because adding a matching feature tightens
    competition among candidates: the same well-placed controls are
    selected repeatedly, so the unique count falls even though the match
    count is fixed. The Kish effective size, ``(sum w)^2 / sum(w^2)`` over
    selection counts, expresses that as an equivalent number of
    independent controls and is always at most the unique count.

    The trade-off this exposes is that a feature added to improve balance
    on itself necessarily consumes distance budget that was previously
    spent on the other features. Reporting every feature, matched or not,
    makes that cost visible rather than assumed.

    :param cohorts: Mapping from a label to the
        ``(match_details, matched_conventional)`` pair returned by
        :meth:`create_matched_dataset`. Labels become the output index.
    :type cohorts: Dict[str, tuple]
    :param alternative_fleet: Treated fleet, indexed as at match time.
    :type alternative_fleet: pd.DataFrame
    :param conventional_fleet: Control fleet, indexed as at match time.
    :type conventional_fleet: pd.DataFrame
    :param report_features: Features to report. Include features that were
        not matched on; their drift is the cost of matching on the others.
    :type report_features: List[str]
    :param logger: Optional logger receiving the same content.
    :type logger: logging.Logger | None
    :param verbose: Print a block per cohort as it is computed.
    :type verbose: bool
    :param debug: Print intermediate counts and alignment shapes.
    :type debug: bool
    :returns: One row per cohort with columns ``n_matches``,
        ``n_unique_controls``, ``mean_reuse``, ``effective_n``,
        ``coverage`` and, per feature, ``<f>_pair_median_absdiff``,
        ``<f>_pair_mean_pctdiff`` and ``<f>_cohort_gap`` (treated mean
        minus deduplicated control mean, signed).
    :rtype: pd.DataFrame
    :raises ValueError: If ``cohorts`` is empty or any entry has no matches.
    :raises KeyError: If a reported feature is absent from either fleet.
    """

    def _report(message: str) -> None:
        if verbose:
            print(f'[cohort_comparison] {message}')
        if logger is not None:
            logger.info(message)

    def _debug(message: str) -> None:
        if debug:
            print(f'[cohort_comparison:debug] {message}')

    if not cohorts:
        raise ValueError('At least one cohort must be supplied.')

    for feature in report_features:
        for name, fleet in (('alternative', alternative_fleet),
                            ('conventional', conventional_fleet)):
            if feature not in fleet.columns:
                raise KeyError(f"Feature '{feature}' not found in the {name} fleet.")

    n_alt = len(alternative_fleet)
    rows: List[Dict[str, Any]] = []

    for label, payload in cohorts.items():
        match_details, matched_conventional = payload

        if not match_details:
            raise ValueError(f"Cohort '{label}' contains no matches.")

        frame = pd.DataFrame(match_details)
        alt_idx = frame['alt_ship_idx'].to_numpy()
        conv_idx = frame['conv_ship_idx'].to_numpy()
        _debug(f'{label}: {len(frame)} matches, '
               f'{len(matched_conventional)} unique controls')

        # --- Stage 1: concentration ------------------------------------
        selection_counts = frame['conv_ship_idx'].value_counts().to_numpy(dtype=float)
        total = selection_counts.sum()
        effective_n = float(total ** 2 / (selection_counts ** 2).sum())

        row: Dict[str, Any] = {
            'n_matches': len(frame),
            'n_unique_controls': len(matched_conventional),
            'mean_reuse': float(len(frame) / len(matched_conventional)),
            'effective_n': effective_n,
            'coverage': float(frame['alt_ship_idx'].nunique() / n_alt) if n_alt else 0.0,
        }

        # --- Stage 2: per-feature balance, both readings ----------------
        for feature in report_features:
            alt_paired = alternative_fleet.loc[alt_idx, feature].to_numpy(dtype=float)
            conv_paired = conventional_fleet.loc[conv_idx, feature].to_numpy(dtype=float)
            difference = np.abs(alt_paired - conv_paired)

            with np.errstate(divide='ignore', invalid='ignore'):
                relative = 100.0 * difference / np.abs(alt_paired)
            relative = relative[np.isfinite(relative)]

            row[f'{feature}_pair_median_absdiff'] = float(np.nanmedian(difference))
            row[f'{feature}_pair_mean_pctdiff'] = (
                float(np.mean(relative)) if relative.size else np.nan
            )
            row[f'{feature}_cohort_gap'] = float(
                alternative_fleet[feature].mean() - matched_conventional[feature].mean()
            )

        rows.append(row)

        _report(
            f"{label}: {row['n_unique_controls']} unique controls "
            f"(reuse {row['mean_reuse']:.2f}, effective n {row['effective_n']:.0f}), "
            f"coverage {row['coverage']:.0%}"
        )
        for feature in report_features:
            _report(
                f"  {feature}: pair median |diff| "
                f"{row[f'{feature}_pair_median_absdiff']:.2f}, "
                f"cohort gap {row[f'{feature}_cohort_gap']:+.2f}"
            )

    table = pd.DataFrame(rows, index=pd.Index(cohorts.keys(), name='cohort'))

    if len(table) > 1:
        _report(
            'Compare rows on all three axes together: a configuration that '
            'improves one feature gap while worsening another has moved the '
            'confound rather than removed it, and a fall in effective_n '
            'costs precision regardless of balance.'
        )

    return table


mixin = LoggerMixin()

logger = mixin.logger

types_list = [ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.CONTAINER_SHIP,ShipType.RO_PAX]

save_base = Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\fleet_data_class\cleaned_data")

for shiptype in types_list:

    bins, variables, target_predictor_constraints, filepath, save_dir = get_variables(shiptype=shiptype)


    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   shiptype=shiptype.string,
                                   fuel_type_columns=fuel_type_columns,
                                   max_build_year=min_year,
                                   alt_fuels=alt_fuels,
                                   outliers=False,
                                   conventional_fuels=conventional_fuels)



    cleaned_dataset = fleet.processed_data["base_with_split_fuels"]

    save_path = save_base/f"{shiptype.string}_cleaned_3.xlsx"

    print(save_path)

    cleaned_dataset.to_excel(save_path)


    alternative_fleet = fleet.processed_data["dual_fuel_Alt_fuel_alt_main"]
    conventional_fleet = fleet.processed_data["pure_conventional"]

    datasets = {"conventional":conventional_fleet,
                "dual_fuel":alternative_fleet}

    for name,df in datasets.items():

        df["AGE"] = (df["BUILT"].max()-df["BUILT"])+1

        datasets[name] = df

        print(f"{name} age description \n  {df["AGE"].describe()}")

    eligibility_5 = TemporalEligibility.prior_only('BUILT', max_lag=5)



    match_details_eligible, matched_conventional_eligible = MatchedDataset.create_matched_dataset(
        alternative_fleet=alternative_fleet,
        conventional_fleet=conventional_fleet,
        logger=logger,
        matching_features=['DWT'],
        n_matches_per_ship=10,
        binning_features=None,          # build-year bins are now superseded
        std_reference='conventional',
        std_scope='global',
        eligibility=eligibility_5,
        debug=True,
    )

    balance = MatchedDataset.get_temporal_balance_summary(
        match_details_eligible, alternative_fleet, eligibility_5, logger, verbose=True
    )

    print(f"=================================\n"
          f"using eligibility - AGE description \n {matched_conventional_eligible["AGE"].describe()}"
          f"\n DWT description {matched_conventional_eligible['DWT'].describe()}"
          f"\n=================================")

    match_details_age, matched_conventional_age = MatchedDataset.create_matched_dataset(
        alternative_fleet=alternative_fleet,
        conventional_fleet=conventional_fleet,
        logger=logger,
        matching_features=['DWT', "AGE"],
        n_matches_per_ship=10,
        binning_features=None,          # build-year bins are now superseded
        std_reference='conventional',
        std_scope='global',
        debug=True,
    )

    print(f"=================================\n"
          f"matching with AGE - AGE description \n {matched_conventional_age["AGE"].describe()}"
          f"\n DWT description \n {matched_conventional_age["DWT"].describe()}"
          f"\n=================================")

    comparison = compare_matched_cohorts(
        cohorts={
            'window_5yr_DWT': (match_details_eligible, matched_conventional_eligible),
            'DWT_plus_AGE':   (match_details_age, matched_conventional_age),
        },
        alternative_fleet=alternative_fleet,
        conventional_fleet=conventional_fleet,
        report_features=['DWT', 'AGE', 'MCR (KW)', 'LOA (M)'],
        logger=logger,
        verbose=True,
    )
    print(comparison.T)