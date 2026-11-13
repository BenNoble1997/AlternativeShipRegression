"""
This class contains a set of static functions for creating matched datasets using
standardized Euclidean distance for improved scale-invariance and interpretability.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Literal, Mapping

from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility


class MatchedDataset:

    @staticmethod
    def overlap_dataset(alternative_fleet: pd.DataFrame,
                                  conventional_fleet: pd.DataFrame,
                                  logger: logging.Logger,
                                predictor:str):
        """
        Method for creating a dataset which includes only conventional ships within the same size as the alternative dataset
        """

        conventional_fleet = conventional_fleet.copy()
        alternative_fleet = alternative_fleet.copy()

        logger.info('')

        lower = alternative_fleet[predictor].min()
        upper = alternative_fleet[predictor].max()

        mask = (conventional_fleet[predictor] >= lower) & (conventional_fleet[predictor] <= upper)

        overlap = conventional_fleet[mask]

        logger.info(f'Minimum value of {predictor}: {lower}')
        logger.info(f'Maximum value of {predictor}: {upper}')
        logger.info(f'Length of overlap dataset is: {len(overlap)}')

        return overlap


    @staticmethod
    def multiple_matched_datasets(alternative_fleet: pd.DataFrame,
                                  conventional_fleet: pd.DataFrame,
                                  logger: logging.Logger,
                                  predictor_groups: List[tuple],
                                  n_matches_per_ship: int = 10,
                                  binning_features: List[str] | None = None,
                                  std_reference: Literal['conventional', 'alternative'] = 'conventional',
                                    std_scope: Literal['global','per_bin']='global',
                                  distance_metric: Literal[
                                      'standardized_euclidean', 'mahalanobis'] = 'standardized_euclidean',
                                  feature_weights: Dict[str, float] | None = None,
                                  skip_missing_ships:bool=True,
                                    caliper: float | None = None,
                                  eligibility: TemporalEligibility | None = None,
                                  debug: bool = False):
        """
        Create one matched dataset per predictor group.

        The same eligibility rule is applied to every group, so the admissible
        set is held fixed while the distance definition varies. Differences
        between the resulting cohorts are therefore attributable to the
        matching features rather than to the temporal restriction.

        :param predictor_groups: Tuples of feature names, one per dataset.
        :type predictor_groups: List[tuple]
        :param eligibility: Optional directional rule applied to all groups.
        :type eligibility: TemporalEligibility | None
        :param debug: Forwarded to :meth:`create_matched_dataset`.
        :type debug: bool
        :returns: Mapping from predictor tuple to matched control frame.
        :rtype: Dict[tuple, pd.DataFrame]
        """
        predictor_groups = list(set(predictor_groups))
        matched_datasets = {}

        for feature_group in predictor_groups:
            matching_info, matched_dataset = MatchedDataset.create_matched_dataset(
                alternative_fleet=alternative_fleet,
                conventional_fleet=conventional_fleet,
                logger=logger,
                matching_features=list(feature_group),
                n_matches_per_ship=n_matches_per_ship,
                binning_features=binning_features,
                std_reference=std_reference,
                std_scope=std_scope,
                distance_metric=distance_metric,
                feature_weights=feature_weights,
                skip_missing_ships=skip_missing_ships,
                caliper=caliper, eligibility=eligibility,debug=debug
            )
            matched_datasets[feature_group] = matched_dataset

        return matched_datasets

    @staticmethod
    def create_matched_dataset(alternative_fleet: pd.DataFrame,
                               conventional_fleet: pd.DataFrame,
                               logger: logging.Logger,
                               matching_features: List[str],
                               n_matches_per_ship: int = 10,
                               binning_features: list[str] | dict[str, list[float]] | None = None,
                               std_reference: Literal['conventional', 'alternative'] = 'conventional',
                               std_scope: Literal['global', 'per_bin'] = 'global',
                               feature_weights: Dict[str, float] | None = None,
                               skip_missing_ships: bool = True,
                               caliper: float | None = None,
                               distance_metric: Literal[
                                   'standardized_euclidean', 'mahalanobis'] = 'standardized_euclidean',
                               eligibility: TemporalEligibility | None = None,
                               debug: bool = False
                               ):
        """
        Create a matched dataset using a quadratic-form distance.

        Conventional ships closest to each alternative ship are selected under
        the chosen metric. Two independent restrictions may be layered on top
        of the distance ranking, and they operate differently:

        - ``binning_features`` imposes exact matching on a shared partition.
          Both fleets are grouped by the same key and matching happens within
          a group. Admissibility is symmetric and identical for every ship in
          the group.
        - ``eligibility`` imposes a directional constraint anchored on each
          alternative ship individually. The admissible sets overlap rather
          than partition, so this cannot be expressed as a bin.

        Neither restriction alters how distance is measured. The metric is
        still estimated from the reference fleet as specified by
        ``std_reference`` and ``std_scope``, so a caliper retains one meaning
        throughout.

        :param matching_features: Features entering the distance.
        :type matching_features: List[str]
        :param n_matches_per_ship: Maximum controls retained per alternative
            ship. Under a temporal rule the realised count varies by ship,
            because the admissible set shrinks for the earliest alternative
            ships; see :meth:`get_temporal_balance_summary`.
        :type n_matches_per_ship: int
        :param binning_features: Categorical features imposing exact matching,
            or ``None``. A mapping from a continuous column to its cut points
            may be given instead, in which case the label columns are derived
            here by :meth:`add_binned_features`. Derived columns are used for
            matching only and do not appear in the returned cohort.
        :type binning_features: list[str] | dict[str, list[float]] | None
        :param std_reference: Fleet defining the scale and covariance.
        :type std_reference: Literal['conventional', 'alternative']
        :param std_scope: Scope over which the metric is estimated when
            binning is used. See the class notes.
        :type std_scope: Literal['global', 'per_bin']
        :param feature_weights: Per-feature weights applied in squared space.
        :type feature_weights: Dict[str, float] | None
        :param skip_missing_ships: Whether alternative ships with incomplete
            matching features are skipped rather than raising.
        :type skip_missing_ships: bool
        :param caliper: Maximum admissible distance, in standardised units.
        :type caliper: float | None
        :param distance_metric: Which quadratic form to build.
        :type distance_metric: Literal['standardized_euclidean', 'mahalanobis']
        :param eligibility: Optional directional rule on an ordered anchor,
            typically build year. When supplied, an alternative ship is
            matched only to controls inside its own admissible interval.
        :type eligibility: TemporalEligibility | None
        :param debug: Emit stage-by-stage print output alongside logging.
        :type debug: bool
        :returns: ``(match_details, matched_conventional)`` where the first is
            a flat list of per-match dictionaries and the second is the
            deduplicated frame of matched control ships.
        :rtype: tuple[List[Dict[str, Any]], pd.DataFrame]
        """

        if debug:
            logger.setLevel(logging.DEBUG)

        # The columns a bin is derived from, which are the bin columns
        # themselves unless cut points were supplied. Retained because a
        # restriction on a derived label is still a restriction on its source.
        binning_sources = list(binning_features) if binning_features is not None else []

        alt_binned, conv_binned, binning_features = MatchedDataset._resolve_binning_features(
            alternative_fleet=alternative_fleet,
            conventional_fleet=conventional_fleet,
            binning_features=binning_features,
            logger=logger
        )

        # Validate inputs
        MatchedDataset._validate_matching_features(
            alt_binned, conv_binned, matching_features, binning_features, eligibility
        )
        MatchedDataset._validate_feature_weights(matching_features, feature_weights)

        logger.info(f"Creating matched dataset using matching features {matching_features}")

        deduplicated = list(dict.fromkeys(matching_features))
        if len(deduplicated) != len(matching_features):
            logger.info(
                f'Matching features {matching_features} contain repeats; '
                f'matching on the distinct set {deduplicated}. A repeated '
                f'predictor affects the formula, not the similarity search.'
            )
            matching_features = deduplicated

        # --- Stage 1: report and sanity-check the temporal rule -------------
        if eligibility is not None:
            eligibility.validate_columns(alternative_fleet, conventional_fleet, logger)
            logger.info(f'Temporal eligibility active: {eligibility.describe()}')
            logger.debug(f'eligibility rule: {eligibility.describe()}')

            if eligibility.anchor_column in binning_sources:
                logger.warning(
                    f"'{eligibility.anchor_column}' is both a binning feature "
                    f"and the eligibility anchor. The two restrictions then "
                    f"intersect: a bin forbids cross-window matches while the "
                    f"rule forbids later matches, so alternative ships near "
                    f"the start of a window can be left with an empty "
                    f"admissible set. Bin on something else, or drop the bin."
                )

            pool = eligibility.pool_sizes(alternative_fleet[eligibility.anchor_column].to_numpy(dtype=float),
                                          conventional_fleet[eligibility.anchor_column].to_numpy(dtype=float))

            logger.info(
                f'Admissible control pool per alternative ship: '
                f'min={pool.min()}, median={np.median(pool):.0f}, '
                f'max={pool.max()} of {len(conventional_fleet)} conventional ships')

            n_starved = int((pool < n_matches_per_ship).sum())
            if n_starved:
                logger.warning(
                    f'{n_starved}/{len(pool)} alternative ships have fewer than '
                    f'{n_matches_per_ship} admissible controls before any '
                    f'distance or caliper filtering is applied.'
                )

                logger.debug(f'pool sizes: min={pool.min()}, max={pool.max()}, starved={n_starved}')

        # --- Stage 2: build the distance metric and match -------------------

        # Determine if we're binning or not
        if binning_features is None:
            logger.info("No binning features specified - matching across entire dataset")

            # Calculate standard deviations for the entire dataset
            reference_fleet = conventional_fleet if std_reference == 'conventional' else alternative_fleet
            whitening = MatchedDataset._build_distance_metric(
                fleet=reference_fleet,
                matching_features=matching_features,
                feature_weights=feature_weights,
                distance_metric=distance_metric,
                logger=logger,
            )

            match_details = MatchedDataset._match_single_group(
                alternative_fleet=alternative_fleet,
                conventional_fleet=conventional_fleet,
                matching_features=matching_features,
                whitening=whitening,
                n_matches_per_ship=n_matches_per_ship,
                bin_description="all ships",
                logger=logger,
                skip_missing_ships=skip_missing_ships,
                eligibility=eligibility,
                caliper=caliper,
                debug=debug
            )
        else:
            logger.info(f"Binning by: {', '.join(binning_features)}")

            alt_fleet_complete = alt_binned.dropna(subset=binning_features)
            conv_fleet_complete = conv_binned.dropna(subset=binning_features)

            n_alt_excluded = len(alt_binned) - len(alt_fleet_complete)
            n_conv_excluded = len(conv_binned) - len(conv_fleet_complete)

            if n_alt_excluded > 0:
                logger.warning(
                    f"Excluded {n_alt_excluded} alternative ships with missing binning features"
                )
            if n_conv_excluded > 0:
                logger.info(
                    f"Excluded {n_conv_excluded} conventional ships with missing binning features"
                )

            match_details = MatchedDataset._match_with_binning(
                alternative_fleet=alt_fleet_complete,
                conventional_fleet=conv_fleet_complete,
                matching_features=matching_features,
                feature_weights=feature_weights,
                n_matches_per_ship=n_matches_per_ship,
                binning_features=binning_features,
                std_reference=std_reference,
                std_scope=std_scope,
                logger=logger,
                skip_missing_ships=skip_missing_ships,
                caliper=caliper,
                distance_metric=distance_metric,
                eligibility=eligibility,
                debug=debug
            )

        # Create matched dataset and store results
        # --- Stage 3: collapse to the deduplicated control cohort -----------

        all_matched_indices = [detail['conv_ship_idx'] for detail in match_details]
        matched_indices_unique = pd.Index(all_matched_indices).unique()
        matched_conventional = conventional_fleet.loc[matched_indices_unique].copy()

        logger.info(
            f"Matching complete: {len(match_details)} total matches, "
            f"{len(matched_indices_unique)} unique conventional ships matched"
        )

        logger.debug(f'{len(match_details)} matches, {len(matched_indices_unique)} unique controls')

        return match_details, matched_conventional

    @staticmethod
    def _validate_matching_features(alternative_fleet: pd.DataFrame,
                                    conventional_fleet: pd.DataFrame,
                                    matching_features: List[str],
                                    binning_features: List[str] | None,
                                    eligibility:TemporalEligibility|None=None):
        """
        Validate that all referenced columns exist and are unambiguous.

        Duplicate column labels are checked first because selecting a
        duplicated label returns a frame rather than a series, which silently
        widens the distance calculation.

        :param eligibility: Optional rule whose anchor column is also checked.
        :type eligibility: TemporalEligibility | None
        :raises ValueError: If a matching feature label is duplicated.
        :raises KeyError: If any referenced column is absent.
        """

        for name, fleet in (('alternative', alternative_fleet),
                            ('conventional', conventional_fleet)):
            duplicated = fleet.columns[fleet.columns.duplicated()].tolist()
            offending = sorted(set(duplicated) & set(matching_features))
            if offending:
                raise ValueError(
                    f"The {name} dataset has duplicate column labels for "
                    f"{offending}. Selecting these returns more columns than "
                    f"features, which silently corrupts the distance calculation. "
                    f"Deduplicate the frame before matching."
                )

        for col in matching_features:
            if col not in alternative_fleet.columns:
                raise KeyError(f"Matching feature '{col}' not found in alternative dataset"
                               f"matching features passed: {matching_features}")
            if col not in conventional_fleet.columns:
                raise KeyError(f"Matching feature '{col}' not found in conventional dataset")

        if binning_features is not None:
            for col in binning_features:
                if col not in alternative_fleet.columns:
                    raise KeyError(f"Binning feature '{col}' not found in alternative dataset")
                if col not in conventional_fleet.columns:
                    raise KeyError(f"Binning feature '{col}' not found in conventional dataset")

        if eligibility is not None:
            eligibility.validate_columns(alternative_fleet,conventional_fleet)

    @staticmethod
    def _resolve_binning_features(alternative_fleet: pd.DataFrame,
                                  conventional_fleet: pd.DataFrame,
                                  binning_features: list[str] | dict[str, list[float]] | None,
                                  logger: logging.Logger
                                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
        """
        Normalise ``binning_features`` to a list of label columns.

        Two forms are accepted. A list names label columns already present in
        both fleets. A mapping from a continuous column to its cut points is
        coarsened here by :meth:`add_binned_features`, which derives the
        partition from one set of edges and applies it to both fleets, so the
        keys mean the same thing on each side.

        The returned frames are copies only when a partition was derived; the
        list form is passed through untouched, so no copy is made in the
        common case.

        :param binning_features: List of label columns, mapping of cut points,
            or ``None``.
        :type binning_features: list[str] | dict[str, list[float]] | None
        :returns: ``(alternative_fleet, conventional_fleet, binning_features)``
            where the frames carry any derived label columns and the third
            element is a list of column names or ``None``.
        :rtype: tuple[pd.DataFrame, pd.DataFrame, list[str] | None]
        :raises TypeError: If a bare string is passed. A string is iterable,
            so it would otherwise be read as one bin per character.
        """
        if binning_features is None:
            return alternative_fleet, conventional_fleet, None

        if isinstance(binning_features, str):
            raise TypeError(
                f"binning_features must be a list of column names, got the "
                f"string '{binning_features}'. Pass ['{binning_features}']."
            )

        if isinstance(binning_features, Mapping):
            logger.info(
                f'Deriving bin labels from cut points for '
                f'{list(binning_features)}'
            )
            return MatchedDataset.add_binned_features(
                alternative_fleet=alternative_fleet,
                conventional_fleet=conventional_fleet,
                bin_edges=binning_features,
                logger=logger
            )

        return alternative_fleet, conventional_fleet, list(binning_features)

    @staticmethod
    def _validate_feature_weights(matching_features: List[str],
                                  feature_weights: Dict[str, float] | None):
        """Validate that feature weights are provided for all matching features."""
        if feature_weights is not None:
            for feature in matching_features:
                if feature not in feature_weights:
                    raise KeyError(
                        f"Feature '{feature}' in matching_features but not in feature_weights. "
                        f"All matching features must have weights specified."
                    )
            for feature in feature_weights:
                if feature not in matching_features:
                    raise KeyError(
                        f"Feature '{feature}' in feature_weights but not in matching_features. "
                        f"All weights must correspond to matching features."
                    )

    @staticmethod
    def _calculate_feature_stds(fleet: pd.DataFrame,
                                matching_features: list[str],
                                logger: logging.Logger) -> dict[str, float]:
        """
        Calculate standard deviations for matching features.

        :param fleet : pd.DataFrame
            Reference fleet for calculating standard deviations
        :param matching_features : List[str]
            Features to calculate standard deviations for
        :param logger : logging.Logger
            Logger for warnings

        :return: Dictionary mapping feature names to their standard deviations
        :rtype: dict[str, float]

        """
        feature_stds = {}

        for feature in matching_features:

            column = fleet[feature]
            if isinstance(column, pd.DataFrame):
                raise ValueError(
                    f"Column '{feature}' appears {column.shape[1]} times in the "
                    f"reference fleet. Deduplicate before computing distances."
                )
            clean_data = column.dropna()

            clean_data = fleet[feature].dropna()
            std = clean_data.std()

            if std == 0 or pd.isna(std):
                logger.warning(
                    f"Feature '{feature}' has zero or undefined standard deviation "
                    f"across {len(clean_data)} reference ships. Falling back to "
                    f"std=1.0, which leaves this feature in its raw units and "
                    f"makes any caliper meaningless for it."
                )
                std = 1.0

            feature_stds[feature] = std
            logger.info(f"Feature '{feature}': mean={clean_data.mean():.2f}, std={std:.2f}")

        return feature_stds

    @staticmethod
    def _match_with_binning(alternative_fleet: pd.DataFrame,
                            conventional_fleet: pd.DataFrame,
                            matching_features: List[str],
                            logger: logging.Logger,
                            feature_weights: Dict[str, float] | None,
                            n_matches_per_ship: int,
                            binning_features: List[str],
                            std_reference: Literal['conventional', 'alternative'],
                            std_scope: Literal['global','per_bin']='global',
                            skip_missing_ships: bool = True,
                            caliper: float | None = None,
                            distance_metric: Literal['standardized_euclidean', 'mahalanobis'] = 'standardized_euclidean',
                            eligibility: TemporalEligibility|None=None,
                            debug:bool=False) -> list[dict[str, Any]]:
        """
        Perform matching within bins defined by categorical features.

        The bin restricts which conventional ships are admissible; it does not
        redefine the population the features are standardised against. Under
        ``std_scope='global'`` the metric is estimated once from the whole
        reference fleet and reused in every bin, so distances carry the same
        meaning throughout and a single caliper is a single tolerance. Under
        ``'per_bin'`` the metric is re-estimated within each bin, which
        rescales distances by the spread of whichever ships fall in that
        window and makes scores incomparable across bins.

        When ``eligibility`` is supplied it composes with the bin by
        intersection: a control must sit in the same bin *and* inside the
        alternative ship's admissible interval. If the bin is itself defined
        on the anchor, that intersection is close to the tighter of the two
        and will strand ships near a window's lower edge.

        :param eligibility: Optional directional rule, applied identically in
            every bin.
        :type eligibility: TemporalEligibility | None
        :param debug: Forwarded to :meth:`_match_single_group`.
        :type debug: bool
        :returns: Flat list of match detail dictionaries across all bins.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If ``std_scope`` is invalid, or if no matches are
            found in any bin.
        """

        if debug:
            logger.setLevel(logging.DEBUG)

        if std_scope not in ('global', 'per_bin'):
            raise ValueError(
                f"std_scope must be 'global' or 'per_bin', got '{std_scope}'"
            )

        if std_scope == 'per_bin' and caliper is not None:
            logger.warning(
                'A caliper is being applied with per-bin standardisation. The '
                'caliper is compared against distances whose scale differs '
                'between bins, so it does not represent a single tolerance. '
                "Use std_scope='global' if the caliper is to be interpretable."
            )

        global_whitening: np.ndarray | None = None

        if std_scope == 'global':
            reference_fleet = (conventional_fleet if std_reference == 'conventional'
                               else alternative_fleet)
            global_whitening = MatchedDataset._build_distance_metric(
                fleet=reference_fleet,
                matching_features=matching_features,
                feature_weights=feature_weights,
                distance_metric=distance_metric,
                logger=logger,
            )
            logger.info(
                f'Distance metric estimated once from the full {std_reference} '
                f'fleet ({len(reference_fleet)} ships) and shared across bins'
            )

        all_match_details = []

        alt_groups = alternative_fleet.groupby(binning_features, dropna=False)
        conv_groups = conventional_fleet.groupby(binning_features, dropna=False)

        total_bins = len(alt_groups)
        logger.info(f"Created {total_bins} bins based on {binning_features}")
        logger.debug(f'{total_bins} bins from {binning_features}')

        bins_processed = 0
        bins_with_warnings = 0
        bins_with_no_matches = 0

        for bin_key, alt_bin in alt_groups:
            bins_processed += 1
            n_alt_ships = len(alt_bin)

            # Normalise the key: pandas yields tuples for list-valued `by`, but
            # get_group rejects length-one tuples from 2.2 onward.
            lookup_key = bin_key[0] if isinstance(bin_key, tuple) and len(bin_key) == 1 else bin_key

            if isinstance(bin_key, tuple):
                bin_desc = ", ".join(f"{feat}={val}" for feat, val in zip(binning_features, bin_key))
            else:
                bin_desc = f"{binning_features[0]}={bin_key}"

            if n_alt_ships < 5:
                logger.warning(
                    f"Bin [{bin_desc}] has only {n_alt_ships} alternative ships - "
                    f"statistical comparison may be unreliable"
                )
                bins_with_warnings += 1

            # Narrow try: only the group lookup may legitimately raise KeyError.
            try:
                conv_bin = conv_groups.get_group(lookup_key)
            except KeyError:
                logger.warning(
                    f"Bin [{bin_desc}] has {n_alt_ships} alternative ships but "
                    f"no conventional ships - skipping"
                )
                bins_with_no_matches += 1
                continue

            n_conv_ships = len(conv_bin)
            logger.info(
                f"Processing bin {bins_processed}/{total_bins}: [{bin_desc}] - "
                f"{n_alt_ships} alt ships, {n_conv_ships} conv ships"
            )

            logger.debug(f'bin [{bin_desc}]: {n_alt_ships} alt, {n_conv_ships} conv')

            if n_conv_ships < n_matches_per_ship:
                logger.warning(
                    f"Bin [{bin_desc}] has {n_conv_ships} conventional ships but "
                    f"{n_matches_per_ship} matches requested per alternative ship - "
                    f"the full bin will be returned for every alternative ship"
                )

            if global_whitening is not None:
                whitening = global_whitening

            else:

                reference_bin = conv_bin if std_reference == 'conventional' else alt_bin
                whitening = MatchedDataset._build_distance_metric(
                    fleet=reference_bin,
                    matching_features=matching_features,
                    feature_weights=feature_weights,
                    distance_metric=distance_metric,
                    logger=logger,
                )

            bin_matches = MatchedDataset._match_single_group(
                alternative_fleet=alt_bin,
                conventional_fleet=conv_bin,
                matching_features=matching_features,
                whitening=whitening,
                n_matches_per_ship=n_matches_per_ship,
                bin_description=bin_desc,
                logger=logger,
                skip_missing_ships=skip_missing_ships,
                caliper=caliper,
                eligibility=eligibility,
                debug=debug
            )

            all_match_details.extend(bin_matches)

        logger.info(
            f"Binning summary: {bins_processed} bins processed, "
            f"{bins_with_warnings} bins with < 5 alt ships, "
            f"{bins_with_no_matches} bins with no matches"
        )

        if not all_match_details:
            raise ValueError(
                "No matches found across any bins. Check binning features or "
                "consider matching without binning."
            )

        return all_match_details

    @staticmethod
    def _match_single_group(alternative_fleet: pd.DataFrame,
                            conventional_fleet: pd.DataFrame,
                            matching_features: List[str],
                            whitening:np.ndarray,
                            n_matches_per_ship: int,
                            bin_description: str,
                            logger: logging.Logger,
                            skip_missing_ships: bool = True,
                            caliper: float | None = None,
                            eligibility: TemporalEligibility|None=None,
                            debug:bool=False) -> List[Dict[str, Any]]:
        """
        Match one group of alternative ships against one group of controls.

        Selection proceeds in three separable stages per alternative ship.
        Distances to the whole candidate block are computed first, since the
        metric does not depend on which candidates are admissible. Admissible
        positions are then retained and the rest set to infinity, so a single
        stable argsort orders the admissible candidates by distance while
        pushing the inadmissible ones to the tail. Finally the ordering is
        walked until it reaches an inadmissible entry, exceeds the caliper, or
        fills the requested count.

        Two failure modes are counted separately because they diagnose
        different problems. An *empty risk set* means the eligibility rule
        admitted nothing, which indicates the rule is too tight or the
        alternative ship predates the control record. *Outside caliper* means
        admissible controls existed but none was close enough, which indicates
        the alternative ship lies off the control support.

        :param whitening: Matrix from :meth:`_build_distance_metric`, encoding
            scale, correlation if Mahalanobis was requested, and weights.
        :type whitening: np.ndarray
        :param bin_description: Group label used in logging and stored on each
            match.
        :type bin_description: str
        :param eligibility: Optional directional rule. Controls with a missing
            anchor are dropped from the candidate block up front, since they
            can never be admissible.
        :type eligibility: TemporalEligibility | None
        :param debug: Emit per-stage print output.
        :type debug: bool
        :returns: List of match detail dictionaries. Each carries
            ``anchor_lag`` when an eligibility rule is active, defined as the
            alternative anchor minus the control anchor, so a positive value
            means the control is older.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If ``skip_missing_ships`` is ``False`` and any
            alternative ship is incomplete, or if an alternative ship has a
            missing anchor while ``on_missing_anchor='error'``.
        """
        # Clean data - only keep rows with all matching features
        # Use .copy() to ensure we have proper DataFrames

        if debug:
            logger.setLevel(logging.DEBUG)

        # --- Stage 1: candidate block, complete on the matching features ----

        conv_clean = conventional_fleet[matching_features].dropna().copy()
        logger.debug(f'{len(conv_clean)}/{len(conventional_fleet)} controls complete on features')


        if len(conv_clean) == 0:
            logger.warning(
                f"No conventional ships with valid data in {bin_description}"
            )
            return []

        # --- Stage 2: strip controls whose anchor cannot be ordered ---------

        conv_anchor_values: np.ndarray | None = None

        if eligibility is not None:
            conv_anchor = conventional_fleet.loc[conv_clean.index, eligibility.anchor_column]
            has_anchor = conv_anchor.notna()
            n_dropped = int((-has_anchor).sum())

            if n_dropped > 0:
                logger.info(
                    f'Dropped {n_dropped} conventional ships in {bin_description} '
                    f"with a missing '{eligibility.anchor_column}' - their position "
                    f'in the ordering cannot be established, so admissibility is '
                    f'undefined.'
                )
                conv_clean = conv_clean.loc[has_anchor[has_anchor].index]
                conv_anchor = conv_anchor.loc[conv_clean.index]

            if len(conv_clean) == 0:
                logger.warning(
                    f'No conventional ships in {bin_description} have both '
                    f'complete matching features and a valid anchor'
                )
                return []

            conv_anchor_values = conv_anchor.to_numpy(dtype=float)
            logger.debug(f'{len(conv_clean)} controls retained after anchor screening')

        # --- Stage 3: alternative ships, complete on the matching features --



        # ✅ Handle alternative ships based on skip_missing_ships parameter
        if skip_missing_ships:
            # NEW BEHAVIOR: Process each ship individually
            alt_with_data = alternative_fleet[matching_features].dropna()
            n_skipped = len(alternative_fleet) - len(alt_with_data)

            if n_skipped > 0:
                logger.info(
                    f"Skipping {n_skipped}/{len(alternative_fleet)} alternative ships "
                    f"with missing data in {bin_description}"
                )

                # ✅ Identify which ships were skipped (for transparency)
                skipped_indices = alternative_fleet.index.difference(alt_with_data.index)
                if 'NAME' in alternative_fleet.columns:
                    skipped_names = alternative_fleet.loc[skipped_indices, 'NAME'].tolist()
                    logger.debug(f"Skipped ships: {skipped_names[:5]}" +
                                 (f" ... and {len(skipped_names) - 5} more" if len(skipped_names) > 5 else ""))

            alt_clean = alt_with_data.copy()

        else:
            # ORIGINAL BEHAVIOR: Fail if any ship has missing data
            n_missing = alternative_fleet[matching_features].isna().any(axis=1).sum()
            if n_missing > 0:
                raise ValueError(
                    f"{n_missing}/{len(alternative_fleet)} alternative ships have "
                    f"missing matching features. Set skip_missing_ships=True to "
                    f"continue with valid ships only."
                )
            alt_clean = alternative_fleet[matching_features].copy()

        if len(alt_clean) == 0:
            logger.warning(
                f"No alternative ships with complete data in {bin_description} "
                f"after filtering"
            )
            return []

        logger.info(
            f"Matching in {bin_description}: {len(alt_clean)}/{len(alternative_fleet)} "
            f"alt ships vs {len(conv_clean)} conv ships"
        )

        # --- Stage 4: hoist the control matrix out of the per-ship loop -----
        conv_matrix = conv_clean[matching_features].to_numpy(dtype=float)
        conv_index = conv_clean.index.to_numpy()

        alt_anchor: pd.Series | None = None
        if eligibility is not None:
            alt_anchor = alternative_fleet.loc[alt_clean.index, eligibility.anchor_column]

        match_details: List[Dict[str, Any]] = []
        n_empty_risk_set = 0
        n_outside_caliper = 0
        n_missing_anchor = 0

        # --- Stage 5: rank, restrict, truncate ------------------------------

        for alt_idx, alt_ship in alt_clean.iterrows():
            anchor: float | None = None
            if eligibility is not None:
                raw_anchor = alt_anchor.loc[alt_idx]
                if pd.isna(raw_anchor):
                    if eligibility.on_missing_anchor == 'error':
                        raise ValueError(
                            f"Alternative ship {alt_idx} has a missing "
                            f"'{eligibility.anchor_column}'. Set "
                            f"on_missing_anchor='skip' to exclude such ships."
                        )
                    n_missing_anchor += 1
                    continue
                anchor = float(raw_anchor)
                admissible = eligibility.mask(anchor, conv_anchor_values)
                if not admissible.any():
                    n_empty_risk_set += 1
                    logger.debug(f'alt {alt_idx} (anchor={anchor:g}): empty risk set')
                    continue
            else:
                admissible = np.ones(len(conv_index), dtype=bool)

            distances = MatchedDataset._distances_from_matrix(
                conv_matrix=conv_matrix,
                alt_vector=alt_ship[matching_features].to_numpy(dtype=float),
                whitening=whitening,
            )

            scored = np.where(admissible & np.isfinite(distances), distances, np.inf)
            order = np.argsort(scored, kind='stable')

            selected_positions: List[int] = []
            for position in order:
                score = scored[position]
                if not np.isfinite(score):
                    break
                if caliper is not None and score > caliper:
                    break
                selected_positions.append(int(position))
                if len(selected_positions) == n_matches_per_ship:
                    break

            if not selected_positions:
                n_outside_caliper += 1
                continue

            for position in selected_positions:
                conv_idx = conv_index[position]
                detail: Dict[str, Any] = {
                    'alt_ship_idx': alt_idx,
                    'conv_ship_idx': conv_idx,
                    'similarity_score': float(scored[position]),
                    'alt_ship_name': alternative_fleet.loc[alt_idx, 'NAME']
                    if 'NAME' in alternative_fleet.columns else f'Alt_{alt_idx}',
                    'conv_ship_name': conventional_fleet.loc[conv_idx, 'NAME']
                    if 'NAME' in conventional_fleet.columns else f'Conv_{conv_idx}',
                    'bin': bin_description,
                }
                if eligibility is not None:
                    detail['anchor_column'] = eligibility.anchor_column
                    detail['alt_anchor'] = anchor
                    detail['conv_anchor'] = float(conv_anchor_values[position])
                    detail['anchor_lag'] = anchor - float(conv_anchor_values[position])
                match_details.append(detail)

            # --- Stage 6: report the two failure modes separately ---------------
        if n_missing_anchor > 0:
            logger.warning(
                f"{n_missing_anchor}/{len(alt_clean)} alternative ships in "
                f"{bin_description} were skipped for a missing "
                f"'{eligibility.anchor_column}'"
            )
        if n_empty_risk_set > 0:
            logger.warning(
                f'{n_empty_risk_set}/{len(alt_clean)} alternative ships in '
                f'{bin_description} had an empty admissible set under '
                f'{eligibility.describe()}. These ships contribute no controls, '
                f'so the matched cohort under-represents that region of the '
                f'anchor axis.'
            )
        if n_outside_caliper > 0:
            message = (
                f'{n_outside_caliper}/{len(alt_clean)} alternative ships in '
                f'{bin_description} had admissible controls but none within '
            )
            logger.warning(
                message + (f'caliper {caliper:.3f}' if caliper is not None
                           else 'finite distance')
            )

        logger.debug(
            f'{len(match_details)} matches; empty risk set={n_empty_risk_set}, '
            f'outside caliper={n_outside_caliper}, missing anchor={n_missing_anchor}'
        )

        return match_details

    @staticmethod
    def _distances_from_matrix(conv_matrix: np.ndarray,
                               alt_vector: np.ndarray,
                               whitening: np.ndarray) -> np.ndarray:
        """
        Distance from one alternative ship to a preassembled control matrix.

        Kept separate from the DataFrame-facing helpers so the control matrix
        can be materialised once per group rather than once per alternative
        ship. The differences are transformed by the whitening matrix and the
        Euclidean norm taken, which yields standardised Euclidean or
        Mahalanobis distance depending on which matrix was built.

        :param conv_matrix: Control features, shape ``(n, p)``, column order
            matching the whitening matrix.
        :type conv_matrix: np.ndarray
        :param alt_vector: Single alternative ship, shape ``(p,)``.
        :type alt_vector: np.ndarray
        :param whitening: Matrix from :meth:`_build_distance_metric`.
        :type whitening: np.ndarray
        :returns: Distances, shape ``(n,)``.
        :rtype: np.ndarray
        :raises ValueError: On a dimension mismatch, the usual cause being
            duplicate column labels in the source frame.
        """
        if conv_matrix.shape[1] != whitening.shape[-1]:
            raise ValueError(
                f"Control matrix has {conv_matrix.shape[1]} columns but the "
                f"whitening matrix has {whitening.shape[-1]}. Duplicate column "
                f"labels are the usual cause."
            )

        transformed = (conv_matrix - alt_vector) @ whitening.T
        return np.sqrt((transformed ** 2).sum(axis=1))

    @staticmethod
    def _calculate_similarity_scores(alt_ship: pd.Series,
                                     conv_clean: pd.DataFrame,
                                     matching_features: List[str],
                                     whitening: np.ndarray) -> List[tuple]:
        """
        Distance from one alternative ship to every conventional ship.

        Retained as a DataFrame-facing wrapper over
        :meth:`_distances_from_matrix` for callers outside the main loop.

        :returns: List of ``(conventional index, distance)`` tuples.
        :rtype: List[tuple]
        """
        distances = MatchedDataset._distances_from_matrix(
            conv_matrix=conv_clean[matching_features].to_numpy(dtype=float),
            alt_vector=alt_ship[matching_features].to_numpy(dtype=float),
            whitening=whitening,
        )
        return list(zip(conv_clean.index, distances))

    @staticmethod
    def get_temporal_balance_summary(match_details: List[Dict[str, Any]],
                                     alternative_fleet: pd.DataFrame,
                                     eligibility: TemporalEligibility,
                                     logger: logging.Logger | None = None,
                                     verbose: bool = False) -> Dict[str, Any]:
        """
        Quantify the anchor imbalance the matching actually produced.

        A directional rule guarantees that no control postdates its treated
        ship, but it says nothing about how far back the selected controls
        reached. That realised lag is the residual confound: if the matched
        cohort averages several years older than the alternative cohort, any
        target that trends with calendar time will differ between them for
        reasons unrelated to fuel. Reporting the lag distribution converts
        that from an unmeasured confound into a stated one.

        The per-ship match count is reported alongside, because a thin
        admissible set near the start of the record means some alternative
        ships contribute fewer controls than others. The matched cohort is
        then implicitly weighted towards the ships with full match sets.

        :param match_details: Output of :meth:`create_matched_dataset`.
        :type match_details: List[Dict[str, Any]]
        :param alternative_fleet: Treated fleet, used to compute coverage.
        :type alternative_fleet: pd.DataFrame
        :param eligibility: Rule the matching was run under.
        :type eligibility: TemporalEligibility
        :param logger: Optional logger for the same content.
        :type logger: logging.Logger | None
        :param verbose: Print the summary as it is assembled.
        :type verbose: bool
        :returns: Summary with keys ``rule``, ``n_matches``,
            ``n_alternative_total``, ``n_alternative_matched``,
            ``coverage``, ``lag_mean``, ``lag_median``, ``lag_max``,
            ``lag_p90``, ``share_contemporaneous``, ``matches_per_ship_min``,
            ``matches_per_ship_median``, ``mean_alt_anchor`` and
            ``mean_conv_anchor``.
        :rtype: Dict[str, Any]
        :raises ValueError: If the match details carry no anchor information,
            which means matching ran without an eligibility rule.
        """
        def _report(message: str) -> None:
            if verbose:
                print(f'[temporal_balance] {message}')
            if logger is not None:
                logger.info(message)

        if not match_details:
            raise ValueError('No match details supplied.')

        if 'anchor_lag' not in match_details[0]:
            raise ValueError(
                'Match details carry no anchor information, so matching was '
                'run without an eligibility rule. There is no temporal '
                'restriction to summarise.'
            )

        frame = pd.DataFrame(match_details)
        lags = frame['anchor_lag'].to_numpy(dtype=float)
        per_ship = frame.groupby('alt_ship_idx').size()

        summary: Dict[str, Any] = {
            'rule': eligibility.describe(),
            'n_matches': len(frame),
            'n_alternative_total': len(alternative_fleet),
            'n_alternative_matched': int(frame['alt_ship_idx'].nunique()),
            'lag_mean': float(np.mean(lags)),
            'lag_median': float(np.median(lags)),
            'lag_p90': float(np.percentile(lags, 90)),
            'lag_max': float(np.max(lags)),
            'share_contemporaneous': float(np.mean(lags == 0)),
            'matches_per_ship_min': int(per_ship.min()),
            'matches_per_ship_median': float(per_ship.median()),
            'mean_alt_anchor': float(frame['alt_anchor'].mean()),
            'mean_conv_anchor': float(frame['conv_anchor'].mean()),
        }
        summary['coverage'] = (
            summary['n_alternative_matched'] / summary['n_alternative_total']
            if summary['n_alternative_total'] > 0 else 0.0
        )

        _report(f"Rule: {summary['rule']}")
        _report(
            f"Coverage: {summary['n_alternative_matched']}/"
            f"{summary['n_alternative_total']} alternative ships matched "
            f"({summary['coverage']:.1%})"
        )
        _report(
            f"Realised lag: mean={summary['lag_mean']:.2f}, "
            f"median={summary['lag_median']:.1f}, "
            f"p90={summary['lag_p90']:.1f}, max={summary['lag_max']:.1f}"
        )
        _report(
            f"Mean anchor: alternative={summary['mean_alt_anchor']:.1f} vs "
            f"matched conventional={summary['mean_conv_anchor']:.1f} "
            f"(gap {summary['mean_alt_anchor'] - summary['mean_conv_anchor']:.2f})"
        )
        _report(
            f"Matches per alternative ship: min={summary['matches_per_ship_min']}, "
            f"median={summary['matches_per_ship_median']:.0f}"
        )

        if summary['matches_per_ship_min'] < 3:
            _report(
                'Some alternative ships contribute fewer than three controls. '
                'The matched cohort is weighted towards ships with full match '
                'sets, which should be stated or handled by weighting.'
            )

        return summary

    @staticmethod
    def get_matching_summary(match_details: List[Dict[str, Any]],
                             alternative_fleet: pd.DataFrame,
                             matching_features: List[str]) -> Dict[str, Any]:
        """
        Generate summary statistics about the matching process.

        Returns
        -------
        dict
            Summary with keys:
            - 'n_alternative_total': Total alternative ships in dataset
            - 'n_alternative_matched': Alternative ships that were matched
            - 'n_alternative_skipped': Alternative ships skipped (missing data)
            - 'n_conventional_matched': Unique conventional ships in matched dataset
            - 'matching_features': Features used for matching
            - 'skipped_ship_names': Names of skipped ships (if available)
        """
        matched_alt_indices = set(detail['alt_ship_idx'] for detail in match_details)
        all_alt_indices = set(alternative_fleet.index)
        skipped_indices = all_alt_indices - matched_alt_indices

        summary = {
            'n_alternative_total': len(alternative_fleet),
            'n_alternative_matched': len(matched_alt_indices),
            'n_alternative_skipped': len(skipped_indices),
            'n_conventional_matched': len(set(d['conv_ship_idx'] for d in match_details)),
            'matching_features': matching_features,
            'match_ratio': len(matched_alt_indices) / len(alternative_fleet) if len(alternative_fleet) > 0 else 0
        }

        if 'NAME' in alternative_fleet.columns and len(skipped_indices) > 0:
            summary['skipped_ship_names'] = alternative_fleet.loc[list(skipped_indices), 'NAME'].tolist()

        return summary

    @staticmethod
    def _build_distance_metric(fleet: pd.DataFrame,
                               matching_features: List[str],
                               feature_weights: Dict[str, float] | None,
                               distance_metric: Literal['standardized_euclidean', 'mahalanobis'],
                               logger: logging.Logger) -> np.ndarray:
        """
        Build the whitening matrix that defines the distance for one group.

        Both supported metrics are quadratic forms ``d^2 = delta.T @ A @ delta``.
        For standardised Euclidean, ``A`` is diagonal with entries ``w_i /
        sigma_i^2``, so each feature is rescaled independently and the
        equidistant contour is a sphere. For Mahalanobis, ``A = D^0.5 S^-1
        D^0.5`` with ``S`` the reference covariance, which additionally
        removes the correlation between features and makes the contour an
        ellipsoid aligned with the data cloud. The two coincide when ``S`` is
        diagonal, so Mahalanobis is a strict generalisation.

        The returned matrix ``W`` satisfies ``A = W.T @ W``, so the distance is
        the ordinary Euclidean norm of ``W @ delta``. It is obtained from the
        Cholesky factor of ``S`` rather than by inverting ``S``, which is more
        stable and raises rather than silently degrading when the covariance is
        near-singular.

        :param fleet: Reference fleet defining the scale and correlation.
        :param matching_features: Features entering the distance. Column order
            of the returned matrix follows this list.
        :param feature_weights: Per-feature weights applied in squared space,
            or None for unit weights.
        :param distance_metric: Which metric to build.
        :param logger: Logger for diagnostics and fallback warnings.
        :returns: Whitening matrix of shape (p, p).
        :rtype: np.ndarray
        :raises ValueError: If ``distance_metric`` is not recognised.
        """
        if distance_metric not in ('standardized_euclidean', 'mahalanobis'):
            raise ValueError(
                f"distance_metric must be 'standardized_euclidean' or "
                f"'mahalanobis', got '{distance_metric}'"
            )

        weights = np.array(
            [float(feature_weights[f]) if feature_weights else 1.0
             for f in matching_features],
            dtype=float,
        )
        sqrt_weights = np.sqrt(weights)

        feature_stds = MatchedDataset._calculate_feature_stds(
            fleet, matching_features, logger
        )
        std_vector = np.array([feature_stds[f] for f in matching_features], dtype=float)

        if distance_metric == 'standardized_euclidean':
            return np.diag(sqrt_weights / std_vector)

        clean = fleet[matching_features].dropna()
        n_obs, n_features = clean.shape

        if n_obs <= n_features:
            logger.warning(
                f'Only {n_obs} complete reference ships for {n_features} features - '
                f'covariance is rank-deficient. Falling back to standardised Euclidean.'
            )
            return np.diag(sqrt_weights / std_vector)

        covariance = clean.cov().to_numpy(dtype=float)

        # Correlation is the quantity Mahalanobis exists to handle, so it is
        # worth seeing: if it is near zero the two metrics will agree closely.
        MatchedDataset._log_feature_correlation(clean, matching_features, logger)

        try:
            cholesky = np.linalg.cholesky(covariance)
            whitening = np.diag(sqrt_weights) @ np.linalg.inv(cholesky)
        except np.linalg.LinAlgError:
            logger.warning(
                'Covariance matrix is not positive definite (features may be '
                'collinear). Falling back to standardised Euclidean.'
            )
            return np.diag(sqrt_weights / std_vector)

        condition = np.linalg.cond(covariance)
        if condition > 1e10:
            logger.warning(
                f'Covariance condition number is {condition:.2e} - features are '
                f'nearly collinear and Mahalanobis distances will be unstable.'
            )

        return whitening

    @staticmethod
    def _log_feature_correlation(clean: pd.DataFrame,
                                 matching_features: List[str],
                                 logger: logging.Logger) -> None:
        """
        Log the pairwise correlations among the matching features.

        Mahalanobis and standardised Euclidean differ only to the extent that
        the features are correlated, so these values indicate whether the
        choice of metric will materially change the matches.

        :param clean: Reference fleet restricted to complete cases.
        """
        correlation = clean.corr()
        for i, first in enumerate(matching_features):
            for second in matching_features[i + 1:]:
                logger.info(
                    f'Correlation {first} vs {second}: '
                    f'{correlation.loc[first, second]:.3f}'
                )

    @staticmethod
    def add_build_year_window(alternative_fleet: pd.DataFrame,
                              conventional_fleet: pd.DataFrame,
                              year_column: str = 'BUILD YEAR',
                              edges: list[int] | None = None,
                              window_column: str = 'BUILD_YEAR_WINDOW'):
        """
        Add a shared coarsened build-year label to both fleets.

        Edges are fixed and identical across fleets so that bin keys are
        comparable. The outermost edges are unbounded, so no ship is dropped
        for falling outside the nominal range.
        """
        if edges is None:
            edges = [1990, 2000, 2005, 2010, 2015, 2020, 2026]

        edges = [-np.inf, *edges, np.inf]

        for fleet in (alternative_fleet, conventional_fleet):
            fleet[window_column] = pd.cut(
                fleet[year_column], bins=edges, right=False
            ).astype(str)

        return alternative_fleet, conventional_fleet

    @staticmethod
    def add_binned_features(alternative_fleet: pd.DataFrame,
                            conventional_fleet: pd.DataFrame,
                            bin_edges: dict[str, list[float]],
                            logger: logging.Logger,
                            suffix: str = '_BIN',
                            right: bool = False
                            ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Coarsen continuous features into shared bin labels on both fleets.

        A bin is an exact-matching constraint, so the label must mean the same
        thing in both fleets: the partition is derived once from a single set
        of edges and applied to each fleet unchanged. Deriving edges per fleet,
        for instance by quantile, would produce keys that share a name while
        denoting different intervals, and the resulting matches would be
        strata of different width.

        The outermost intervals are left unbounded so that no ship falls
        outside the partition. Ships with a missing value in a binned feature
        retain a missing label rather than being collected into a shared
        ``'nan'`` stratum, which keeps them subject to the existing exclusion
        in :meth:`create_matched_dataset`.

        :param alternative_fleet: Treated fleet. Not modified in place.
        :type alternative_fleet: pd.DataFrame
        :param conventional_fleet: Control fleet. Not modified in place.
        :type conventional_fleet: pd.DataFrame
        :param bin_edges: Mapping from column name to interior cut points.
            Infinite outer edges may be supplied but are not required; they are
            added when absent. Edges must be strictly increasing.
        :type bin_edges: dict[str, list[float]]
        :param suffix: Appended to the source column to name the label column.
        :type suffix: str
        :param right: Whether intervals are closed on the right. The default
            gives left-closed intervals, matching
            :meth:`add_build_year_window`.
        :type right: bool
        :returns: Copies of both fleets carrying the label columns, and the
            list of label column names to pass as ``binning_features``.
        :rtype: tuple[pd.DataFrame, pd.DataFrame, list[str]]
        :raises KeyError: If a keyed column is absent from either fleet.
        :raises ValueError: If a column is non-numeric, or its edges are not
            strictly increasing, or they define no interior boundary.
        """
        if not bin_edges:
            raise ValueError('bin_edges is empty; pass None as binning_features instead')

        alternative_fleet = alternative_fleet.copy()
        conventional_fleet = conventional_fleet.copy()
        binning_features: list[str] = []

        for column, edges in bin_edges.items():
            for name, fleet in (('alternative', alternative_fleet),
                                ('conventional', conventional_fleet)):
                if column not in fleet.columns:
                    raise KeyError(f"Binning column '{column}' not found in {name} dataset")
                if not pd.api.types.is_numeric_dtype(fleet[column]):
                    raise ValueError(
                        f"Binning column '{column}' is non-numeric in the {name} "
                        f'dataset; pass it directly as a binning feature instead '
                        f'of supplying edges'
                    )

            full_edges = MatchedDataset._normalise_bin_edges(column=column, edges=edges)
            labels = MatchedDataset._format_bin_labels(edges=full_edges, right=right)

            label_column = f'{column}{suffix}'
            binning_features.append(label_column)

            for fleet in (alternative_fleet, conventional_fleet):
                fleet[label_column] = pd.cut(
                    fleet[column], bins=full_edges, right=right, labels=labels
                ).astype(object)

            MatchedDataset._log_bin_occupancy(
                alternative_fleet=alternative_fleet,
                conventional_fleet=conventional_fleet,
                label_column=label_column,
                labels=labels,
                logger=logger
            )

        return alternative_fleet, conventional_fleet, binning_features

    @staticmethod
    def _normalise_bin_edges(column: str, edges: list[float]) -> list[float]:
        """
        Return the supplied cut points with unbounded outer edges.

        :param column: Source column, used only in error messages.
        :type column: str
        :param edges: Interior cut points, optionally already unbounded.
        :type edges: list[float]
        :returns: Strictly increasing edges spanning the real line.
        :rtype: list[float]
        """
        interior = [float(edge) for edge in edges if np.isfinite(edge)]

        if not interior:
            raise ValueError(
                f"Bin edges for '{column}' contain no finite boundary, so the "
                f'partition would hold a single bin'
            )
        if any(later <= earlier for earlier, later in zip(interior, interior[1:])):
            raise ValueError(
                f"Bin edges for '{column}' must be strictly increasing, got {edges}"
            )

        return [-np.inf, *interior, np.inf]

    @staticmethod
    def _format_bin_labels(edges: list[float], right: bool) -> list[str]:
        """
        Build compact, unique labels for a set of bin edges.

        Labels rather than interval objects are used because the key is
        carried into group descriptions, warnings and downstream tables, and
        because an object-dtype key avoids the categorical group index that
        arises when grouping on the output of :func:`pandas.cut` directly.

        :param edges: Unbounded, strictly increasing edges.
        :type edges: list[float]
        :param right: Whether intervals are closed on the right.
        :type right: bool
        :returns: One label per interval.
        :rtype: list[str]
        """

        def render(value: float) -> str:
            return f'{value:g}'

        interior = edges[1:-1]
        lower_operator, upper_operator = ('<=', '>') if right else ('<', '>=')

        labels = [f'{lower_operator}{render(interior[0])}']
        labels += [
            f'{render(lower)}-{render(upper)}'
            for lower, upper in zip(interior, interior[1:])
        ]
        labels.append(f'{upper_operator}{render(interior[-1])}')

        return labels

    @staticmethod
    def _log_bin_occupancy(alternative_fleet: pd.DataFrame,
                           conventional_fleet: pd.DataFrame,
                           label_column: str,
                           labels: list[str],
                           logger: logging.Logger) -> None:
        """
        Report how both fleets populate a partition.

        Matching happens within a bin, so a stratum holding alternative ships
        but few controls is where the requested number of matches per ship
        will not be met. Reporting occupancy before matching separates that
        diagnosis from the per-bin warnings raised during matching.

        :param label_column: Column holding the bin labels.
        :type label_column: str
        :param labels: Labels in interval order.
        :type labels: list[str]
        """
        alt_counts = alternative_fleet[label_column].value_counts()
        conv_counts = conventional_fleet[label_column].value_counts()

        logger.info(f'Bin occupancy for {label_column} (alternative / conventional):')
        for label in labels:
            n_alt = int(alt_counts.get(label, 0))
            n_conv = int(conv_counts.get(label, 0))
            logger.info(f'  {label}: {n_alt} / {n_conv}')
            if n_alt > 0 and n_conv == 0:
                logger.warning(
                    f'Bin [{label_column}={label}] holds {n_alt} alternative '
                    f'ships and no conventional ships; these ships cannot be '
                    f'matched under this partition'
                )

        n_alt_missing = int(alternative_fleet[label_column].isna().sum())
        n_conv_missing = int(conventional_fleet[label_column].isna().sum())
        if n_alt_missing or n_conv_missing:
            logger.warning(
                f'{n_alt_missing} alternative and {n_conv_missing} conventional '
                f'ships have a missing value for {label_column} and will be '
                f'excluded from matching'
            )