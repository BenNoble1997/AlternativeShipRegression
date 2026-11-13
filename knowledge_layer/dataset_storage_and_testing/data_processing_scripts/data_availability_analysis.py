
import logging

import numpy as np
import pandas as pd
from pathlib import Path

from knowledge_layer.dataset_storage_and_testing.data_availability_table_generator import (
    build_availability_counts,
    generate_availability_table,
    extract_fleet_datasets)

from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships,get_variables
from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.dataset_storage_and_testing.availability_subsample_editor import AvailabilitySubsampleAuditor

types_list = [ShipType.BULK_CARRIER,ShipType.CONTAINER_SHIP,ShipType.TANKER, ShipType.RO_PAX]

fleets = {}

def check_training_support(train_present: pd.DataFrame,
                           test_cohort: pd.DataFrame,
                           variables: list[str],
                           lower_pct: float = 2.5,
                           upper_pct: float = 97.5,
                           debug: bool = False) -> pd.DataFrame:
    """
    Check whether a training subsample spans the predictor range of the ships
    it will be used to predict.

    Support, not balance, is the criterion that governs whether missingness in
    the training data disqualifies a variable. A regression fitted on an
    unbalanced but fully-covering subsample interpolates; one fitted on a
    subsample that does not reach the test ships extrapolates, and any
    functional-form error is then unbounded. This function quantifies the
    second case.

    Coverage is assessed against trimmed training percentiles rather than the
    raw min/max, so that a single extreme training ship cannot certify support
    for a region containing no real data. The trim matches the 2.5-97.5
    convention used elsewhere in the pipeline.

    :param train_present: Training rows retained after listwise deletion on the
        candidate variable - i.e. the subsample the model is actually fitted on.
    :type train_present: pd.DataFrame
    :param test_cohort: Ships the model will be evaluated on.
    :type test_cohort: pd.DataFrame
    :param variables: Predictors to assess, including the candidate itself.
    :type variables: list[str]
    :param lower_pct: Lower training percentile defining the supported region.
    :type lower_pct: float
    :param upper_pct: Upper training percentile defining the supported region.
    :type upper_pct: float
    :param debug: Print per-variable progress.
    :type debug: bool
    :returns: One row per variable with the supported interval, the fraction of
        test ships falling inside it, and how far the worst test ship lies
        outside expressed in training standard deviations.
    :rtype: pd.DataFrame
    """
    records = []

    for variable in variables:
        if variable not in train_present.columns or variable not in test_cohort.columns:
            if debug:
                print(f"[support] '{variable}' missing from a frame - skipped")
            continue

        train_values = pd.to_numeric(train_present[variable], errors='coerce')
        test_values = pd.to_numeric(test_cohort[variable], errors='coerce')
        train_values = train_values[np.isfinite(train_values)].to_numpy()
        test_values = test_values[np.isfinite(test_values)].to_numpy()

        if len(train_values) < 2 or len(test_values) < 1:
            if debug:
                print(f"[support] '{variable}': too few finite values "
                      f"(train={len(train_values)}, test={len(test_values)})")
            continue

        low, high = np.percentile(train_values, [lower_pct, upper_pct])
        inside = (test_values >= low) & (test_values <= high)
        train_sd = train_values.std(ddof=1)

        # Signed excursion of the worst test ship, in training SDs.
        below = (low - test_values.min()) / train_sd if train_sd > 0 else np.nan
        above = (test_values.max() - high) / train_sd if train_sd > 0 else np.nan
        worst_excursion = max(below, above, 0.0)

        records.append({
            'variable': variable,
            'n_train': int(len(train_values)),
            'n_test': int(len(test_values)),
            'support_low': float(low),
            'support_high': float(high),
            'fraction_test_covered': float(inside.mean()),
            'n_test_outside': int((~inside).sum()),
            'worst_excursion_sd': float(worst_excursion),
        })

        if debug:
            print(f"[support] {variable}: supported [{low:.3g}, {high:.3g}], "
                  f"{inside.mean():.1%} of test ships covered, worst excursion "
                  f"{worst_excursion:.2f} SD")

    return pd.DataFrame(records)

def support_matrix(pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                   candidates: list[str],
                   base_variables: list[str],
                   debug: bool = False) -> pd.DataFrame:
    """
    Check training support for several candidates across several ship types.

    For each candidate the training partition is filtered to the ships that
    report it - reproducing the listwise deletion a regression on that
    candidate would perform - and coverage is then assessed on the base
    predictors plus the candidate itself.

    :param pairs: Mapping of ship-type label -> ``(conventional_train,
        test_cohort)``.
    :type pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    :param candidates: Variables whose inclusion is in question.
    :type candidates: list[str]
    :param base_variables: Predictors present for essentially every ship.
    :type base_variables: list[str]
    :param debug: Print per-combination progress.
    :type debug: bool
    :returns: Long-form frame with ``pair`` and ``candidate`` identifying
        columns prepended to the per-variable support rows.
    :rtype: pd.DataFrame
    """
    frames = []

    for pair_name, (train_cohort, test_cohort) in pairs.items():
        for candidate in candidates:
            if candidate not in train_cohort.columns:
                if debug:
                    print(f"[support_matrix] {pair_name}: '{candidate}' absent "
                          f"from training cohort - skipped")
                continue

            train_present = train_cohort[train_cohort[candidate].notna()]
            if train_present.empty:
                if debug:
                    print(f"[support_matrix] {pair_name}: no training ships "
                          f"report '{candidate}' - skipped")
                continue

            if debug:
                print(f"\n[support_matrix] {pair_name} | candidate='{candidate}' "
                      f"({len(train_present)} training ships report it)")

            frame = check_training_support(
                train_present=train_present,
                test_cohort=test_cohort,
                variables=[*base_variables, candidate],
                debug=debug,
            )
            if frame.empty:
                continue

            frame.insert(0, 'candidate', candidate)
            frame.insert(0, 'pair', pair_name)
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def alt_fuel_availability():
    for shiptype in types_list:
        bins, variables, _, filepath, save_dir,_ = get_variables(shiptype)

        max_build_year = 1990

        fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
        conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

        alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
        bio_fuels = ['Biofuel']
        alt_fuels = {'Alt_fuel': alt_fuel,
                     'Biofuel': bio_fuels}

        fleets[shiptype.string] = getting_alt_fuel_ships(filepath=filepath,
                                           shiptype=shiptype.string,
                                           fuel_type_columns=fuel_type_columns,
                                           alt_fuels=alt_fuels,
                                           conventional_fuels=conventional_fuels,
                                           max_build_year=max_build_year,)

    auditor = AvailabilitySubsampleAuditor(verbose=False)

    dual_fuel_container_df = fleets['Container ship'].processed_data["dual_fuel_Alt_fuel_alt_main"]

    common = ['DWT', 'LOA (M)', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'MCR (KW)']



    cohorts = {'Container / Dual fuel':dual_fuel_container_df,
               'Container / Biofuel':fleets['Container ship'].processed_data["pure_Biofuel"],
                'Container / Conventional':fleets['Container ship'].processed_data["pure_conventional"],
               'RoPax / Dual fuel':fleets['RoPax'].processed_data["dual_fuel_Alt_fuel_alt_main"],
               'RoPax / Biofuel':fleets['RoPax'].processed_data["pure_Biofuel"],
                'RoPax / Conventional':fleets['RoPax'].processed_data["pure_conventional"]
               }

    matrix = auditor.audit_matrix(
        cohorts=cohorts,
        candidates=['Vs', 'LSW', 'MCR (KW)','Nv', 'Np'],
        common_variables=common,
    )

    result = auditor.audit_variable(
        cohort=dual_fuel_container_df,   # all 73 ships, Vs column mostly NaN
        candidate='Vs',
        common_variables=common,
        cohort_label='Container / Dual fuel',
        debug=True,
    )

    df = pd.DataFrame(matrix)

    df.to_excel(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\output\sister_maker\avaialability_sig.xlsx")

def run_training_support():
    for shiptype in types_list:
        bins, variables, _, filepath, save_dir,datasets = get_variables(shiptype)

        max_build_year = 1990

        fuel_type_columns = ['MAIN_FUEL', "BUNKER_FUEL"]
        conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

        alt_fuel = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
        bio_fuels = ['Biofuel']
        alt_fuels = {'Alt_fuel': alt_fuel,
                     'Biofuel': bio_fuels}

        fleets[shiptype.string] = getting_alt_fuel_ships(filepath=filepath,
                                           shiptype=shiptype.string,
                                           fuel_type_columns=fuel_type_columns,
                                           alt_fuels=alt_fuels,
                                           conventional_fuels=conventional_fuels,
                                           max_build_year=max_build_year,)

    auditor = AvailabilitySubsampleAuditor(verbose=False)

    dual_fuel_container_df = fleets['Container ship'].processed_data["dual_fuel_Alt_fuel_alt_main"]

    common = ['DWT', 'LOA (M)', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'MCR (KW)']



    cohorts = {'Container / Dual fuel':dual_fuel_container_df,
               'Container / Biofuel':fleets['Container ship'].processed_data["pure_Biofuel"],
                'Container / Conventional':fleets['Container ship'].processed_data["pure_conventional"],
               'RoPax / Dual fuel':fleets['RoPax'].processed_data["dual_fuel_Alt_fuel_alt_main"],
               'RoPax / Biofuel':fleets['RoPax'].processed_data["pure_Biofuel"],
                'RoPax / Conventional':fleets['RoPax'].processed_data["pure_conventional"]
               }



    support = support_matrix(
        pairs={
            'Container': (fleets['Container ship'].processed_data["pure_conventional"],
                          fleets['Container ship'].processed_data["dual_fuel_Alt_fuel_alt_main"]),
            'RoPax': (fleets['RoPax'].processed_data["pure_conventional"],
                      fleets['RoPax'].processed_data["dual_fuel_Alt_fuel_alt_main"]),
        },
        candidates=['Vs', 'Nv', 'Np'],
        base_variables=['DWT', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'MCR (KW)'],
        debug=True,
    )

    # The decision-relevant summary: worst coverage per (ship type, candidate).
    verdict = (support
               .groupby(['pair', 'candidate'])
               .agg(min_coverage=('fraction_test_covered', 'min'),
                    max_excursion=('worst_excursion_sd', 'max'),
                    n_train=('n_train', 'min'))
               .reset_index())


if __name__ == "__main__":
    alt_fuel_availability()
    run_training_support()