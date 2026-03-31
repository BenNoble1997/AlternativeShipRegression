"""
This class contains a set of static functions for creating matched datasets using
standardized Euclidean distance for improved scale-invariance and interpretability.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Literal


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
                                  feature_weights: Dict[str, float] | None = None,
                                  skip_missing_ships:bool=True):
        """
        Create multiple matched datasets using different predictor groups.

        Parameters
        ----------
        alternative_fleet : pd.DataFrame
            Dataset containing alternative fuel ships
        conventional_fleet : pd.DataFrame
            Dataset containing conventional fuel ships
        logger : logging.Logger
            Logger for tracking progress and warnings
        predictor_groups : List[tuple]
            List of tuples, each containing feature names to use for matching
        n_matches_per_ship : int, default=10
            Number of conventional ships to match per alternative ship
        binning_features : List[str] | None, default=None
            Categorical features to bin by before matching
        std_reference : {'conventional', 'alternative'}, default='conventional'
            Which fleet to use for calculating feature standard deviations
        feature_weights : Dict[str, float] | None, default=None
            Optional weights for each feature. If None, all features weighted equally.

        Returns
        -------
        Dict[tuple, pd.DataFrame]
            Dictionary mapping predictor tuples to matched datasets
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
                feature_weights=feature_weights,
                skip_missing_ships=skip_missing_ships
            )
            matched_datasets[feature_group] = matched_dataset

        return matched_datasets

    @staticmethod
    def create_matched_dataset(alternative_fleet: pd.DataFrame,
                               conventional_fleet: pd.DataFrame,
                               logger: logging.Logger,
                               matching_features: List[str] = ['DWT', 'SPEED (KN)'],
                               n_matches_per_ship: int = 10,
                               binning_features: List[str] | None = None,
                               std_reference: Literal['conventional', 'alternative'] = 'conventional',
                               feature_weights: Dict[str, float] | None = None,
                               skip_missing_ships: bool = True
                               ):
        """
        Create a matched dataset using standardized Euclidean distance.

        This method finds conventional ships that most closely match alternative fuel ships
        based on standardized distances in the specified features. Standardization ensures
        features with different scales are compared fairly.

        The standardized Euclidean distance is calculated as:
            distance = sqrt(Σ weight_i * ((x_conv_i - x_alt_i) / σ_i)²)

        where σ_i is the standard deviation of feature i in the reference fleet.

        Parameters
        ----------
        alternative_fleet : pd.DataFrame
            Dataset containing alternative fuel ships
        conventional_fleet : pd.DataFrame
            Dataset containing conventional fuel ships to match against
        logger : logging.Logger
            Logger for tracking progress and warnings
        matching_features : List[str], default=['DWT', 'SPEED (KN)']
            Features to use for calculating similarity
        n_matches_per_ship : int, default=10
            Maximum number of similar conventional ships to find per alternative ship
        binning_features : List[str] | None, default=None
            Optional categorical features to bin by before matching. Ships only match
            within the same bin. Example: ['NUMBER OF SCREWS', 'POWER TYPE']
        std_reference : {'conventional', 'alternative'}, default='conventional'
            Which fleet to use for calculating feature standard deviations for
            standardization. 'conventional' treats conventional fleet as reference
            distribution (recommended for most cases).
        feature_weights : Dict[str, float] | None, default=None
            Optional weights for each matching feature. Higher weights increase the
            importance of that feature. If None, all features weighted equally (1.0).
            Example: {'DWT': 2.0, 'SPEED (KN)': 1.0} makes DWT twice as important.
        skip_missing_ships : bool, default=True

        Returns
        -------
        match_details : List[Dict[str, Any]]
            List of dictionaries containing matching information for each match
        matched_dataset : pd.DataFrame
            Deduplicated dataset of matched conventional ships

        Notes
        -----
        - Standardized Euclidean distance is symmetric and scale-invariant
        - Standard deviations are computed per-bin when binning is used
        - Ships with missing values in matching features are excluded
        - The same conventional ship can match multiple alternative ships
        """
        # Validate inputs
        MatchedDataset._validate_matching_features(
            alternative_fleet, conventional_fleet, matching_features, binning_features
        )
        MatchedDataset._validate_feature_weights(matching_features, feature_weights)

        # Determine if we're binning or not
        if binning_features is None:
            logger.info("No binning features specified - matching across entire dataset")

            # Calculate standard deviations for the entire dataset
            reference_fleet = conventional_fleet if std_reference == 'conventional' else alternative_fleet
            feature_stds = MatchedDataset._calculate_feature_stds(
                reference_fleet, matching_features, logger
            )

            match_details = MatchedDataset._match_single_group(
                alternative_fleet=alternative_fleet,
                conventional_fleet=conventional_fleet,
                matching_features=matching_features,
                feature_stds=feature_stds,
                feature_weights=feature_weights,
                n_matches_per_ship=n_matches_per_ship,
                bin_description="all ships",
                logger=logger,
                skip_missing_ships=skip_missing_ships
            )
        else:
            logger.info(f"Binning by: {', '.join(binning_features)}")

            alt_fleet_complete = alternative_fleet.dropna(subset=binning_features)
            conv_fleet_complete = conventional_fleet.dropna(subset=binning_features)

            n_alt_excluded = len(alternative_fleet) - len(alt_fleet_complete)
            n_conv_excluded = len(conventional_fleet) - len(conv_fleet_complete)

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
                logger=logger,
                skip_missing_ships=skip_missing_ships
            )

        # Create matched dataset and store results
        all_matched_indices = [detail['conv_ship_idx'] for detail in match_details]
        matched_indices_unique = list(set(all_matched_indices))
        matched_conventional = conventional_fleet.loc[matched_indices_unique].copy()

        logger.info(
            f"Matching complete: {len(match_details)} total matches, "
            f"{len(matched_indices_unique)} unique conventional ships matched"
        )

        return match_details, matched_conventional

    @staticmethod
    def _validate_matching_features(alternative_fleet: pd.DataFrame,
                                    conventional_fleet: pd.DataFrame,
                                    matching_features: List[str],
                                    binning_features: List[str] | None):
        """Validate that all specified features exist in both datasets."""
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
                                matching_features: List[str],
                                logger: logging.Logger) -> Dict[str, float]:
        """
        Calculate standard deviations for matching features.

        Parameters
        ----------
        fleet : pd.DataFrame
            Reference fleet for calculating standard deviations
        matching_features : List[str]
            Features to calculate standard deviations for
        logger : logging.Logger
            Logger for warnings

        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to their standard deviations
        """
        feature_stds = {}

        for feature in matching_features:
            clean_data = fleet[feature].dropna()
            std = clean_data.std()

            if std == 0 or pd.isna(std):
                logger.warning(
                    f"Feature '{feature}' has zero or undefined standard deviation. "
                    f"Using std=1.0 as fallback (no standardization for this feature)."
                )
                std = 1.0

            feature_stds[feature] = std
            logger.info(f"Feature '{feature}': mean={clean_data.mean():.2f}, std={std:.2f}")

        return feature_stds

    @staticmethod
    def _match_with_binning(alternative_fleet: pd.DataFrame,
                            conventional_fleet: pd.DataFrame,
                            matching_features: List[str],
                            feature_weights: Dict[str, float] | None,
                            n_matches_per_ship: int,
                            binning_features: List[str],
                            std_reference: Literal['conventional', 'alternative'],
                            logger: logging.Logger) -> List[Dict[str, Any]]:
        """Perform matching with binning by categorical features."""
        all_match_details = []

        # Create bins using groupby
        alt_groups = alternative_fleet.groupby(binning_features, dropna=False)
        conv_groups = conventional_fleet.groupby(binning_features, dropna=False)

        total_bins = len(alt_groups)
        logger.info(f"Created {total_bins} bins based on {binning_features}")

        bins_processed = 0
        bins_with_warnings = 0
        bins_with_no_matches = 0

        for bin_key, alt_bin in alt_groups:
            bins_processed += 1
            n_alt_ships = len(alt_bin)

            # Format bin description
            if isinstance(bin_key, tuple):
                bin_desc = ", ".join([f"{feat}={val}" for feat, val in zip(binning_features, bin_key)])
            else:
                bin_desc = f"{binning_features[0]}={bin_key}"

            # Warning for small bins
            if n_alt_ships < 5:
                logger.warning(
                    f"Bin [{bin_desc}] has only {n_alt_ships} alternative ships - "
                    f"statistical comparison may be unreliable"
                )
                bins_with_warnings += 1

            # Check if corresponding conventional bin exists
            try:
                conv_bin = conv_groups.get_group(bin_key)
                n_conv_ships = len(conv_bin)

                logger.info(
                    f"Processing bin {bins_processed}/{total_bins}: [{bin_desc}] - "
                    f"{n_alt_ships} alt ships, {n_conv_ships} conv ships"
                )

                if n_conv_ships == 0:
                    logger.warning(f"Bin [{bin_desc}] has no conventional ships - skipping")
                    bins_with_no_matches += 1
                    continue

                # Calculate per-bin standard deviations
                reference_bin = conv_bin if std_reference == 'conventional' else alt_bin
                feature_stds = MatchedDataset._calculate_feature_stds(
                    reference_bin, matching_features, logger
                )

                # Perform matching within this bin
                bin_matches = MatchedDataset._match_single_group(
                    alternative_fleet=alt_bin,
                    conventional_fleet=conv_bin,
                    matching_features=matching_features,
                    feature_stds=feature_stds,
                    feature_weights=feature_weights,
                    n_matches_per_ship=n_matches_per_ship,
                    bin_description=bin_desc,
                    logger=logger
                )

                all_match_details.extend(bin_matches)

            except KeyError:
                logger.warning(
                    f"Bin [{bin_desc}] has {n_alt_ships} alternative ships but "
                    f"no conventional ships - skipping"
                )
                bins_with_no_matches += 1

        # Summary statistics
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
                            feature_stds: Dict[str, float],
                            feature_weights: Dict[str, float] | None,
                            n_matches_per_ship: int,
                            bin_description: str,
                            logger: logging.Logger,
                            skip_missing_ships:bool=True) -> List[Dict[str, Any]]:
        """
        Perform matching for a single group using standardized Euclidean distance.

        Parameters
        ----------
        alternative_fleet : pd.DataFrame
            Alternative fuel ships to match
        conventional_fleet : pd.DataFrame
            Conventional ships to match against
        matching_features : List[str]
            Features to use for matching
        feature_stds : Dict[str, float]
            Standard deviations for each feature (for standardization)
        feature_weights : Dict[str, float] | None
            Optional weights for each feature
        n_matches_per_ship : int
            Number of matches per alternative ship
        bin_description : str
            Description of this group for logging
        logger : logging.Logger
            Logger instance

        Returns
        -------
        List[Dict[str, Any]]
            List of match detail dictionaries
        """
        # Clean data - only keep rows with all matching features
        # Use .copy() to ensure we have proper DataFrames
        conv_clean = conventional_fleet[matching_features].dropna().copy()

        if len(conv_clean) == 0:
            logger.warning(
                f"No conventional ships with valid data in {bin_description}"
            )
            return []

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

        # ✅ Rest of matching logic unchanged
        match_details = []

        for alt_idx, alt_ship in alt_clean.iterrows():
            similarity_scores = MatchedDataset._calculate_similarity_scores(
                alt_ship=alt_ship,
                conv_clean=conv_clean,
                matching_features=matching_features,
                feature_stds=feature_stds,
                feature_weights=feature_weights
            )

            if not similarity_scores:
                continue

            similarity_scores.sort(key=lambda x: x[1])
            top_matches = similarity_scores[:n_matches_per_ship]

            for conv_idx, score in top_matches:
                match_details.append({
                    'alt_ship_idx': alt_idx,
                    'conv_ship_idx': conv_idx,
                    'similarity_score': score,
                    'alt_ship_name': alternative_fleet.loc[alt_idx, 'NAME']
                    if 'NAME' in alternative_fleet.columns else f'Alt_{alt_idx}',
                    'conv_ship_name': conventional_fleet.loc[conv_idx, 'NAME']
                    if 'NAME' in conventional_fleet.columns else f'Conv_{conv_idx}',
                    'bin': bin_description
                })

        return match_details

    @staticmethod
    def _calculate_similarity_scores(alt_ship: pd.Series,
                                     conv_clean: pd.DataFrame,
                                     matching_features: List[str],
                                     feature_stds: Dict[str, float],
                                     feature_weights: Dict[str, float] | None) -> List[tuple]:
        """
        Calculate standardized Euclidean distance between one alternative ship and all conventional ships.

        The distance is calculated as:
            distance = sqrt(Σ weight_i * ((x_conv_i - x_alt_i) / σ_i)²)

        where:
        - x_conv_i, x_alt_i are the values of feature i
        - σ_i is the standard deviation of feature i in the reference fleet
        - weight_i is the user-specified weight (default 1.0)

        Parameters
        ----------
        alt_ship : pd.Series
            Single alternative ship
        conv_clean : pd.DataFrame
            Cleaned conventional ships dataset
        matching_features : List[str]
            Features to use for matching
        feature_stds : Dict[str, float]
            Standard deviations for standardization
        feature_weights : Dict[str, float] | None
            Optional feature weights

        Returns
        -------
        List[tuple]
            List of (index, score) tuples
        """
        similarity_scores = []

        for conv_idx, conv_ship in conv_clean.iterrows():
            score = 0.0

            for feature in matching_features:
                # Extract scalar values, handling potential Series returns
                alt_val_raw = alt_ship[feature]
                conv_val_raw = conv_ship[feature]

                # Convert to scalar if needed
                if isinstance(alt_val_raw, pd.Series):
                    alt_val = float(alt_val_raw.iloc[0])
                else:
                    alt_val = float(alt_val_raw)

                if isinstance(conv_val_raw, pd.Series):
                    conv_val = float(conv_val_raw.iloc[0])
                else:
                    conv_val = float(conv_val_raw)

                std = float(feature_stds[feature])
                weight = float(feature_weights[feature]) if feature_weights else 1.0

                # Standardized difference
                standardized_diff = (conv_val - alt_val) / std
                score += weight * (standardized_diff ** 2)

            final_score = float(np.sqrt(score))
            similarity_scores.append((conv_idx, final_score))

        return similarity_scores

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