# File: random_forest_analysis_foundation.py

"""
Random Forest Analysis Foundation
=================================
Provides common setup and utilities for Random Forest analyses
via composition rather than inheritance.

This class handles:
- Selector results extraction and validation
- Matched dataset creation
- Data preparation and cleaning
- Common RF-specific utilities

Used by RF analyzer classes via composition.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor

from config.loggermixin.logger_config import LoggerMixin
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.fleet_data_class.temporal_eligibility import TemporalEligibility

class RandomForestAnalysisFoundation(LoggerMixin):
    """
    Foundation class providing common functionality for Random Forest analyses.

    Designed to be used via composition, not inheritance.

    Responsibilities:
    ----------------
    - Extract tested combinations from selector results
    - Create and manage matched conventional datasets
    - Prepare and clean data for RF model fitting
    - Validate inputs and configurations
    - Provide utility methods for RF-specific operations
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 variable_dimensions: List[str],
                 target_dimensions: Optional[List[str]] = None,
                 selector_results: Optional[Dict] = None,
                 rf_params: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = True,
                 selection_metric:str = 'rrmse',
                 target_predictor_constraints:dict[str,list[str]]=None,
                defer_matched_sets: bool = False,
                 n_matches_per_ship:int=10,
                 caliper:float|None=None,
                 build_year_col:str|None=None,
                 year_edges:list[int]=None,
                 matching_bins:dict[str,list[str]]=None,
                 eligibility:TemporalEligibility|None=None,
                 ):
        """
        Initialize RF analysis foundation.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_data : pd.DataFrame
            Alternative fuel ships dataset
        variable_dimensions : List[str]
            List of potential predictor dimensions
        target_dimensions : List[str], optional
            List of target dimensions. If None, uses variable_dimensions.
        selector_results : dict, optional
            Dictionary from RandomForestSelector.results
            Format: {target: {combo_name: {'predictors': tuple, 'metrics': dict, ...}}}
            **REQUIRED for proper operation**
        rf_params : dict, optional
            RandomForestRegressor parameters
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Whether to print progress and warnings
        """
        self.random_state = random_state
        self.verbose = verbose
        self.selection_metric = selection_metric

        # ✅ Store constraints before any combination logic runs
        self.target_predictor_constraints = target_predictor_constraints or {}
        if self.target_predictor_constraints and verbose:
            self._print_constraint_summary()

        # Setup dimensions
        if target_dimensions is None:
            target_dimensions = variable_dimensions

        self._setup_dimensions(
            conventional_data=conventional_data,
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions
        )

        self.build_year_col = build_year_col
        self.year_edges = year_edges
        self.build_eligibility = eligibility
        self.matching_bins = matching_bins

        # Store datasets
        self.conventional_full = self._resolve_data_variables(name='conventional_full', data=conventional_data.copy())
        self.alternative = self._resolve_data_variables(name='alternative', data=alternative_data.copy())

        # Validate and store selector results
        self._validate_selector_results(selector_results)
        self.selector_results = selector_results

        # Default RF parameters
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': random_state,
            'n_jobs': -1,
            'oob_score': True
        }

        self.n_matches_per_ship = n_matches_per_ship
        self.caliper = caliper

        # Validate data quality
        self._validate_data_quality()

        # ✅ Print diagnostics BEFORE creating matched datasets
        if verbose:
            self.print_variable_diagnostics()

        # Matched sets: build now (legacy) or defer until the analyzer has
        # split the conventional data and can supply the test partition.
        if defer_matched_sets:
            self.matched_data_sets = {}
            if verbose:
                print("\n  Matched dataset creation deferred — call "
                      "build_matched_sets(candidate_pool=...) after the "
                      "train/test split.")
        else:
            self.matched_data_sets = self._create_matched_sets()

        self.selector = None  # Will be set by analyzer if available




    # ========================================================================
    # SETUP AND VALIDATION
    # ========================================================================

    def _resolve_data_variables(self, name, data):
        """
        resolve which variables should be extracted and used in the datasets.


        :return:
        """

        retained = list(self.available_dimensions)

        if self.build_year_col is not None:

            if self.build_year_col not in data.columns:
                raise KeyError(
                    f"build_year_col '{self.build_year_col}' is not a column of the "
                    f"{name} dataset, so build-year blocking cannot be applied."
                )
            retained.append(self.build_year_col)

        if self.build_eligibility is not None:

            if self.build_eligibility.anchor_column not in data.columns:
                raise KeyError(
                    f"build_year_col '{self.build_eligibility.anchor_column}' is not a column of the "
                    f"{name} dataset, so build-year blocking cannot be applied."
                )
            retained.append(self.build_eligibility.anchor_column)

        if self.matching_bins is not None:

            if isinstance(self.matching_bins, list):
                columns = self.matching_bins
            else:
                columns = self.matching_bins.keys()

            for column in columns:
                if column not in data.columns:
                    raise KeyError(
                            f"build_year_col '{column}' is not a column of the "
                            f"{name} dataset, so build-year blocking cannot be applied."
                        )
                retained.append(column)

        result = data[list(set(retained))].copy()


        return result

    def _print_constraint_summary(self):
        """Print a summary of active target predictor constraints."""
        print("\n" + "=" * 60)
        print("TARGET PREDICTOR CONSTRAINTS (Random Forest)")
        print("=" * 60)
        for target, allowed in self.target_predictor_constraints.items():
            print(f"  {target}:")
            print(f"    Allowed predictors: {allowed}")
        print("=" * 60 + "\n")

    def _combination_satisfies_constraints(self,
                                           predictors: Tuple[str, ...],
                                           target: str) -> bool:
        """
        Check whether a predictor combination satisfies target constraints.

        If no constraint is defined for this target, always returns True.
        All predictors must be members of the allowed list — any predictor
        outside it causes rejection.

        Parameters
        ----------
        predictors : tuple of str
        target : str

        Returns
        -------
        bool
        """
        if target not in self.target_predictor_constraints:
            return True
        allowed = set(self.target_predictor_constraints[target])
        return all(pred in allowed for pred in predictors)

    def _setup_dimensions(self,
                          conventional_data: pd.DataFrame,
                          variable_dimensions: List[str],
                          target_dimensions: List[str]):
        """Setup and validate dimension lists."""
        all_dimensions = list(set(variable_dimensions + target_dimensions))
        available_dimensions = [d for d in all_dimensions
                                if d in conventional_data.columns]
        missing_dimensions = [d for d in all_dimensions
                              if d not in conventional_data.columns]

        if missing_dimensions and self.verbose:
            print(f"\n  ⚠️  Warning: Some dimensions not found in training_data:")
            for dim in missing_dimensions:
                print(f"    - '{dim}'")
            print(f"  Using {len(available_dimensions)} available dimensions\n")

        if 'DWT' not in available_dimensions:
            raise ValueError("DWT must be in the training data!")

        self.available_dimensions = available_dimensions
        self.variable_dimensions = [d for d in variable_dimensions
                                    if d in available_dimensions]
        self.target_dimensions = [d for d in target_dimensions
                                  if d in available_dimensions]

    def _validate_selector_results(self, selector_results: Optional[Dict]):
        """Validate selector results input."""
        if selector_results is None or len(selector_results) == 0:
            raise ValueError(
                "selector_results is required and cannot be empty!\n"
                "Please run RandomForestSelector.analyze_all_combinations() first,\n"
                "then pass selector.results to this analyzer."
            )

    def _validate_data_quality(self):
        """Check data quality and warn about issues."""
        if not self.verbose:
            return

        for dataset_name, dataset in [
            ('conventional', self.conventional_full),
            ('alternative', self.alternative)
        ]:
            # Check for NaN
            nan_counts = dataset.isna().sum()
            if nan_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has missing values:")
                for col, count in nan_counts[nan_counts > 0].items():
                    pct = 100 * count / len(dataset)
                    print(f"    {col}: {count} ({pct:.1f}%)")

            # Check for inf
            inf_mask = np.isinf(dataset.select_dtypes(include=[np.number]))
            inf_counts = inf_mask.sum()
            if inf_counts.sum() > 0:
                print(f"\n  Warning: {dataset_name} dataset has infinite values:")
                for col, count in inf_counts[inf_counts > 0].items():
                    print(f"    {col}: {count}")

            # Check minimum size
            if len(dataset) < 10:
                print(f"\n  ⚠️  WARNING: {dataset_name} dataset is very small "
                      f"({len(dataset)} ships)")

    def _create_matched_sets(self,
                             candidate_pool: pd.DataFrame | None = None
                             ) -> dict[tuple[str, ...], pd.DataFrame]:
        """
        Create matched conventional datasets for each predictor combination.

        Matching is one dataset per unique predictor tuple, matched on exactly
        those predictors via standardized Euclidean distance.

        :param candidate_pool: Conventional ships eligible to be selected as
            matches. When ``None``, the full conventional fleet is used
            (legacy behaviour, which leaks training ships into matched-set
            evaluation). When the analyzer supplies its held-out test
            partition, matched-set evaluation is strictly out-of-sample.
        :type candidate_pool: pandas.DataFrame or None
        :returns: Mapping ``predictor_tuple → matched_dataset``.
        :rtype: dict[tuple[str, ...], pandas.DataFrame]
        :raises ValueError: If no valid predictor combinations remain after
            availability filtering.
        :raises KeyError: If matching fails due to a predictor missing from
            the candidate pool's columns.
        """
        pool = candidate_pool if candidate_pool is not None else self.conventional_full

        # Collect all unique predictor combinations
        combo_list = []
        for target in self.target_dimensions:
            combos = self.extract_combinations_for_target(target)
            combo_list.extend(combos)

        all_unique_combos = sorted(set(combo_list))

        valid_combos = []
        skipped_combos = []

        for combo in all_unique_combos:
            if all(pred in self.available_dimensions for pred in combo):
                valid_combos.append(combo)
            else:
                missing = [pred for pred in combo if pred not in self.available_dimensions]
                skipped_combos.append((combo, missing))

        if skipped_combos and self.verbose:
            print(f"\n⚠️  Skipped {len(skipped_combos)} combinations due to missing variables:")
            for combo, missing in skipped_combos[:5]:
                print(f"  • {combo}: missing {missing}")
            if len(skipped_combos) > 5:
                print(f"  • ... and {len(skipped_combos) - 5} more")
            print()

        if not valid_combos:
            raise ValueError(
                "No valid predictor combinations found after filtering!\n"
                f"All {len(all_unique_combos)} combinations from selector contain variables "
                f"not present in the current datasets.\n"
                f"Available dimensions: {self.available_dimensions}\n"
                f"Consider passing the same variable list to analyzer as was used in selector."
            )

        if self.verbose:
            pool_desc = ("held-out test partition" if candidate_pool is not None
                         else "FULL conventional fleet (legacy — includes train ships)")
            print(f"\nCreating matched datasets for {len(valid_combos)} valid predictor "
                  f"combinations from {pool_desc} ({len(pool)} candidate ships)...")

        try:

            if self.build_eligibility is not None:
                matched_sets = MatchedDataset.multiple_matched_datasets(
                    alternative_fleet=self.alternative,
                    conventional_fleet=pool,
                    logger=self.logger,
                    predictor_groups=valid_combos,
                    std_reference='conventional',
                    n_matches_per_ship=self.n_matches_per_ship,
                    eligibility=self.build_eligibility,
                    caliper=self.caliper,
                    binning_features=self.matching_bins
                )

            elif self.build_year_col is not None:
                alt_fleet,conv_fleet = MatchedDataset.add_build_year_window(alternative_fleet=self.alternative,
                                                     conventional_fleet=pool,
                                                     year_column=self.build_year_col, edges=self.year_edges)

                matched_sets = MatchedDataset.multiple_matched_datasets(
                    alternative_fleet=alt_fleet,
                    conventional_fleet=conv_fleet,
                    logger=self.logger,
                    predictor_groups=valid_combos,
                    std_reference='conventional',
                    n_matches_per_ship=self.n_matches_per_ship,
                    binning_features=['BUILD_YEAR_WINDOW'],
                    caliper=self.caliper
                )

            else:

                matched_sets = MatchedDataset.multiple_matched_datasets(
                    alternative_fleet=self.alternative,
                    conventional_fleet=pool,
                    logger=self.logger,
                    predictor_groups=valid_combos,
                    std_reference='conventional',
                    n_matches_per_ship=self.n_matches_per_ship,
                    caliper=self.caliper,
                    binning_features=self.matching_bins
                )
        except KeyError as e:
            raise KeyError(
                f"Matched dataset creation failed: {str(e)}\n"
                f"This suggests a predictor variable exists in available_dimensions "
                f"but not in the candidate pool's columns.\n"
                f"Available dimensions: {self.available_dimensions}\n"
                f"Pool columns: {list(pool.columns)}"
            ) from e

        if self.verbose:
            print(f"✅ Created {len(matched_sets)} matched datasets")

        return matched_sets

    def build_matched_sets(self,
                           candidate_pool: pd.DataFrame | None = None,
                           debug: bool = False) -> None:
        """
        Build (or rebuild) the matched conventional datasets.

        Intended to be called by the owning analyzer after it has performed
        its train/test split, passing the held-out test partition as the
        candidate pool. This guarantees that matched-set evaluation never
        scores the model on ships it was trained on, which is essential for
        fair parametric-vs-Random-Forest gap comparisons (flexible models
        memorise training points and would otherwise show deflated matched
        error).

        :param candidate_pool: Conventional ships eligible as match candidates
            (typically the analyzer's ``conventional_test``). ``None`` falls
            back to the full conventional fleet.
        :type candidate_pool: pandas.DataFrame or None
        :param debug: Print candidate pool size and per-combination matched
            sizes after construction.
        :type debug: bool
        """
        self.matched_data_sets = self._create_matched_sets(candidate_pool=candidate_pool)

        if debug:
            pool_n = len(candidate_pool) if candidate_pool is not None else len(self.conventional_full)
            print(f"[build_matched_sets] candidate pool: {pool_n} ships")
            for combo, df in self.matched_data_sets.items():
                print(f"[build_matched_sets]   {' + '.join(combo):35s} → matched n = {len(df)}")

    # ========================================================================
    # COMBINATION EXTRACTION FROM SELECTOR RESULTS
    # ========================================================================

    def extract_combinations_for_target(self,
                                        target: str,
                                        max_predictors: Optional[int] = None,
                                        include_feature_engineered: bool = False,
                                        filter_unavailable: bool = True
                                        ) -> List[Tuple[str, ...]]:
        """
        Extract predictor combinations for a target from selector results.

        Applies three sequential filters:
        1. Feature engineering flag
        2. max_predictors size limit
        3. Availability — all predictors must exist in self.available_dimensions
        4. Constraints — all predictors must be in target_predictor_constraints
                         (only applied when a constraint exists for this target)

        Returns combinations sorted by performance (best first).
        """
        metric = self.selection_metric

        if target not in self.selector_results:
            return []

        combo_data = []
        seen_predictors = set()

        for combo_name, data in self.selector_results[target].items():
            predictors = data['predictors']
            is_engineered = data.get('feature_engineered', False)

            if is_engineered and not include_feature_engineered:
                continue

            if max_predictors is not None and len(predictors) > max_predictors:
                continue

            if filter_unavailable:
                if not all(pred in self.available_dimensions for pred in predictors):
                    continue

            # ✅ Constraint filter
            if not self._combination_satisfies_constraints(predictors, target):
                continue

            predictor_key = tuple(predictors)
            if predictor_key in seen_predictors:
                continue

            seen_predictors.add(predictor_key)

            metric_score = data['metrics'].get(metric, -999)
            combo_data.append((predictor_key, metric_score))

        if metric in ['adj_r2', 'r2', 'oob_r2']:
            combo_data.sort(key=lambda x: x[1], reverse=True)
        else:
            combo_data.sort(key=lambda x: x[1])

        return [combo for combo, score in combo_data]

    def get_combinations_summary(self) -> pd.DataFrame:
        """
        Get summary of available combinations from selector results.

        Returns
        -------
        pd.DataFrame
            Summary with columns: target, n_combinations, n_raw, n_engineered, predictor_sizes
        """
        summary_data = []

        for target in self.target_dimensions:
            if target not in self.selector_results:
                continue

            # Get valid combinations (with filtering)
            valid_combos_raw = self.extract_combinations_for_target(
                target,
                include_feature_engineered=False,
                filter_unavailable=True
            )

            valid_combos_eng = self.extract_combinations_for_target(
                target,
                include_feature_engineered=True,
                filter_unavailable=True
            )

            # Get all combinations (without filtering) to count skipped
            all_combos = self.extract_combinations_for_target(
                target,
                filter_unavailable=False
            )

            n_skipped = len(all_combos) - len(set(valid_combos_raw + valid_combos_eng))

            unique_combos = set(valid_combos_raw + valid_combos_eng)
            if unique_combos:
                sizes = sorted(list(set(len(c) for c in unique_combos)))
                summary_data.append({
                    'target': target,
                    'n_combinations': len(unique_combos),
                    'n_raw': len(valid_combos_raw),
                    'n_engineered': len(valid_combos_eng),
                    'n_skipped': n_skipped,
                    'predictor_sizes': sizes
                })

        return pd.DataFrame(summary_data)

    # ========================================================================
    # MATCHED DATASET CREATION
    # ========================================================================
    #
    # def _create_matched_sets(self) -> Dict[Tuple[str, ...], pd.DataFrame]:
    #     """
    #     Create matched conventional datasets for each predictor combination.
    #
    #     Returns
    #     -------
    #     dict
    #         Maps predictor_tuple → matched_dataset_df
    #     """
    #     # Collect all unique predictor combinations
    #     combo_list = []
    #     for target in self.target_dimensions:
    #         combos = self.extract_combinations_for_target(target)
    #         combo_list.extend(combos)
    #
    #     # Get unique combinations
    #     all_unique_combos = sorted(set(combo_list))
    #
    #     # ✅ FILTER: Only keep combinations where ALL predictors are available
    #     valid_combos = []
    #     skipped_combos = []
    #
    #     for combo in all_unique_combos:
    #         if all(pred in self.available_dimensions for pred in combo):
    #             valid_combos.append(combo)
    #         else:
    #             missing = [pred for pred in combo if pred not in self.available_dimensions]
    #             skipped_combos.append((combo, missing))
    #
    #     # Report skipped combinations
    #     if skipped_combos and self.verbose:
    #         print(f"\n⚠️  Skipped {len(skipped_combos)} combinations due to missing variables:")
    #         for combo, missing in skipped_combos[:5]:
    #             print(f"  • {combo}: missing {missing}")
    #         if len(skipped_combos) > 5:
    #             print(f"  • ... and {len(skipped_combos) - 5} more")
    #         print()
    #
    #     if not valid_combos:
    #         raise ValueError(
    #             "No valid predictor combinations found after filtering!\n"
    #             f"All {len(all_unique_combos)} combinations from selector contain variables "
    #             f"not present in the current datasets."
    #         )
    #
    #     if self.verbose:
    #         print(f"\nCreating matched datasets for {len(valid_combos)} valid predictor combinations "
    #               f"(filtered from {len(all_unique_combos)} total)...")
    #
    #     # Create matched datasets
    #     matched_sets = MatchedDataset.multiple_matched_datasets(
    #         alternative_fleet=self.alternative,
    #         conventional_fleet=self.conventional_full,
    #         logger=self.logger,
    #         predictor_groups=valid_combos,
    #         std_reference='conventional'
    #     )
    #
    #     if self.verbose:
    #         print(f"✅ Created {len(matched_sets)} matched datasets")
    #
    #     return matched_sets

    def get_matched_dataset(self, predictors: Tuple[str, ...]) -> Optional[pd.DataFrame]:
        """Get matched dataset for a specific predictor combination."""
        return self.matched_data_sets.get(tuple(predictors))

    # ========================================================================
    # RF PARAMETERS
    # ========================================================================

    def get_rf_params(self) -> Dict:
        """Get Random Forest parameters."""
        return self.rf_params.copy()

    # ========================================================================
    # DATA PREPARATION UTILITIES
    # ========================================================================

    def prepare_dataset(self,
                        dataset: pd.DataFrame,
                        predictors: List[str],
                        target: str,
                        min_samples: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare and clean a single dataset for RF fitting.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset to prepare
        predictors : list of str
            Predictor variable names
        target : str
            Target variable name
        min_samples : int, default=5
            Minimum required samples after cleaning

        Returns
        -------
        tuple of (X, y) or None
            Cleaned arrays, or None if insufficient data
        """
        X = dataset[predictors].values
        y = dataset[target].values

        # Clean data
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                 np.isinf(X).any(axis=1) | np.isinf(y))

        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < min_samples:
            if self.verbose:
                print(f"  Warning: Insufficient clean data ({len(X_clean)} < {min_samples})")
            return None

        return X_clean, y_clean

    # ========================================================================
    # RF MODEL UTILITIES
    # ========================================================================

    def fit_and_predict(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        rf_params: Optional[Dict] = None) -> Tuple[np.ndarray, RandomForestRegressor]:
        """
        Fit Random Forest and make predictions.

        Parameters
        ----------
        X_train : np.ndarray
            Training predictors
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test predictors
        rf_params : dict, optional
            Override RF parameters. If None, uses self.rf_params

        Returns
        -------
        tuple of (predictions, model)
            Predictions and fitted RF model
        """
        params = rf_params or self.rf_params

        try:
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            return y_pred, rf

        except Exception as e:
            if self.verbose:
                print(f"  RF fitting failed: {str(e)[:80]}")
            return None, None

    # ========================================================================
    # INFORMATION METHODS
    # ========================================================================

    def diagnose_missing_variables(self) -> pd.DataFrame:
        """
        Diagnose which variables from selector results are missing.

        Returns
        -------
        pd.DataFrame
            Summary with variable availability information
        """
        # Get all variables used in selector results
        all_vars_in_selector = set()
        var_usage_count = {}

        for target, combos in self.selector_results.items():
            for combo_name, data in combos.items():
                predictors = data['predictors']
                for pred in predictors:
                    all_vars_in_selector.add(pred)
                    var_usage_count[pred] = var_usage_count.get(pred, 0) + 1

        # Check availability
        diagnostics = []

        for var in sorted(all_vars_in_selector):
            diagnostics.append({
                'variable': var,
                'in_selector_results': True,
                'in_available_dimensions': var in self.available_dimensions,
                'n_combinations_using': var_usage_count[var],
                'status': '✅ Available' if var in self.available_dimensions else '❌ Missing'
            })

        # Also check for variables in available_dimensions not used in selector
        for var in sorted(self.available_dimensions):
            if var not in all_vars_in_selector:
                diagnostics.append({
                    'variable': var,
                    'in_selector_results': False,
                    'in_available_dimensions': True,
                    'n_combinations_using': 0,
                    'status': 'ℹ️  Available but unused'
                })

        df = pd.DataFrame(diagnostics)

        # Sort: missing first, then by usage count
        df['sort_key'] = df.apply(
            lambda row: (0 if not row['in_available_dimensions'] else 1,
                         -row['n_combinations_using']),
            axis=1
        )
        df = df.sort_values('sort_key').drop('sort_key', axis=1)

        return df

    def print_variable_diagnostics(self):
        """Print diagnostic information about variable availability."""
        if not self.verbose:
            return

        print("\n" + "=" * 80)
        print("VARIABLE AVAILABILITY DIAGNOSTICS (Random Forest)")
        print("=" * 80)

        diag_df = self.diagnose_missing_variables()

        missing = diag_df[~diag_df['in_available_dimensions'] & diag_df['in_selector_results']]
        available = diag_df[diag_df['in_available_dimensions'] & diag_df['in_selector_results']]
        unused = diag_df[~diag_df['in_selector_results']]

        if len(missing) > 0:
            print(f"\n❌ MISSING VARIABLES ({len(missing)}):")
            print("These variables are used in selector results but not available in datasets:")
            for _, row in missing.iterrows():
                print(f"  • {row['variable']:<20s} (used in {row['n_combinations_using']} combinations)")

        print(f"\n✅ AVAILABLE VARIABLES ({len(available)}):")
        print("These variables are used in selector results and available:")
        for _, row in available.head(10).iterrows():
            print(f"  • {row['variable']:<20s} (used in {row['n_combinations_using']} combinations)")
        if len(available) > 10:
            print(f"  • ... and {len(available) - 10} more")

        if len(unused) > 0:
            print(f"\nℹ️  UNUSED VARIABLES ({len(unused)}):")
            print("These variables are available but not used in any selector results:")
            for _, row in unused.head(5).iterrows():
                print(f"  • {row['variable']}")
            if len(unused) > 5:
                print(f"  • ... and {len(unused) - 5} more")

        print("=" * 80 + "\n")

    def get_best_combination_for_target(
            self,
            target: str,
            max_predictors: Optional[int] = 1,
            selection_metric: str|None = None,
            filter_by_data_availability: bool = True,  # ← Keep for backward compatibility
            require_alternative_data: bool = False,  # ← CHANGED DEFAULT: False
            min_alternative_samples: int = 5,
            selection_based_on_conventional: bool = True  # ← NEW PARAMETER
    ) -> Optional[Tuple[str, ...]]:
        """
        Get best predictor combination for target.

        Parameters
        ----------
        target : str
            Target dimension
        max_predictors : int, optional
            Maximum number of predictors
        selection_metric : str, default='adj_r2'
            Metric for selection: 'adj_r2', 'rrmse', 'mape', 'oob_r2'
        filter_by_data_availability : bool, default=True
            **DEPRECATED**: Use require_alternative_data instead
        require_alternative_data : bool, default=False
            If True, only return combinations with sufficient alternative data
        min_alternative_samples : int, default=5
            Minimum samples required in alternative dataset
        selection_based_on_conventional : bool, default=True
            **NEW**: If True, selects best by conventional performance, THEN checks
            alternative data availability. If False, uses old behavior (filters first).

        Returns
        -------
        tuple or None
            Best predictor tuple, or None if no valid combination
        """
        if selection_metric is None:

            selection_metric = self.selection_metric


        if target not in self.selector_results:
            if self.verbose:
                print(f"  ⚠️  No selector results for target: {target}")
            return None

        target_results = self.selector_results[target]

        if not target_results:
            return None

        # Handle deprecated parameter
        if filter_by_data_availability and not require_alternative_data:
            require_alternative_data = True

        # ========================================================================
        # Build candidate list
        # ========================================================================

        candidates = []

        for combo_name, combo_data in target_results.items():
            predictors = combo_data['predictors']

            if max_predictors is not None and len(predictors) > max_predictors:
                continue

            if not all(pred in self.available_dimensions for pred in predictors):
                if self.verbose:
                    missing = [p for p in predictors if p not in self.available_dimensions]
                    print(f"  ⚠️  Skipping {predictors}: missing variables {missing}")
                continue

            if not self._combination_satisfies_constraints(predictors, target):
                continue

            metrics = combo_data['metrics']

            # Basic quality filter
            adj_r2 = metrics.get('adj_r2', -999)
            if adj_r2 < 0.5:
                continue

            metric_value = metrics.get(selection_metric, -999)

            candidates.append({
                'predictors': predictors,
                'combo_name': combo_name,
                'metrics': metrics,
                'metric_value': metric_value,
                'n_predictors': len(predictors)
            })

        if not candidates:
            if self.verbose:
                print(f"  ⚠️  No valid candidates for {target}")
            return None

        # ========================================================================
        # BRANCHING LOGIC: New vs Old Behavior
        # ========================================================================

        if selection_based_on_conventional:
            # ✅ NEW BEHAVIOR: Select best, THEN check testability
            return self._select_best_then_check_testability_rf(
                candidates=candidates,
                target=target,
                selection_metric=selection_metric,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

        else:
            # 🔄 OLD BEHAVIOR: Filter by testability, THEN select best
            return self._filter_then_select_best_rf(
                candidates=candidates,
                target=target,
                selection_metric=selection_metric,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

    def _select_best_then_check_testability_rf(
            self,
            candidates: List[Dict],
            target: str,
            selection_metric: str,
            require_alternative_data: bool,
            min_alternative_samples: int
    ) -> Optional[Tuple[str, ...]]:
        """NEW LOGIC: Select best by conventional, then check testability."""

        # Sort by metric
        if selection_metric in ['adj_r2', 'r2', 'oob_r2']:
            candidates.sort(key=lambda x: x['metric_value'], reverse=True)
        else:
            candidates.sort(key=lambda x: x['metric_value'])

        # Try each in order (best first)
        for candidate in candidates:
            predictors = candidate['predictors']

            if not require_alternative_data:
                # Return first (best) without checking
                if self.verbose:
                    metrics = candidate['metrics']
                    print(f"  ✓ Selected: {predictors}")
                    print(f"    Conventional: R²={metrics['adj_r2']:.3f}, "
                          f"{selection_metric}={candidate['metric_value']:.3f}")
                return predictors

            # Check if testable
            if self._check_predictors_data_availability(
                    predictors, target, min_alternative_samples
            ):
                if self.verbose:
                    metrics = candidate['metrics']
                    print(f"  ✓ Selected: {predictors}")
                    print(f"    Conventional: R²={metrics['adj_r2']:.3f}")
                return predictors
            else:
                if self.verbose and candidate == candidates[0]:
                    # Only warn for best combination
                    print(f"  ℹ️  Best combination {predictors} not testable, trying alternatives...")

        # No testable combinations
        if self.verbose:
            print(f"  ⚠️  No testable combinations found for {target}")
        return None

    def _filter_then_select_best_rf(
            self,
            candidates: List[Dict],
            target: str,
            selection_metric: str,
            require_alternative_data: bool,
            min_alternative_samples: int
    ) -> Optional[Tuple[str, ...]]:
        """OLD LOGIC: Filter by testability first, then select best."""

        # Filter by alternative data availability
        if require_alternative_data:
            valid_candidates = []

            for candidate in candidates:
                predictors = candidate['predictors']

                if self._check_predictors_data_availability(
                        predictors, target, min_alternative_samples
                ):
                    valid_candidates.append(candidate)

            if not valid_candidates:
                if self.verbose:
                    print(f"  ⚠️  No combinations with sufficient alternative data")
                return None

            candidates = valid_candidates

        # Select best from (possibly filtered) candidates
        if selection_metric in ['adj_r2', 'r2', 'oob_r2']:
            best_candidate = max(candidates, key=lambda x: x['metric_value'])
        else:
            best_candidate = min(candidates, key=lambda x: x['metric_value'])

        return best_candidate['predictors']

    def _check_target_data_availability(self,
                                        target: str,
                                        min_samples: int) -> bool:
        """
        Check if target has sufficient data in alternative dataset.

        ✅ MATCHES: parametric_analysis_foundation._check_target_data_availability

        Returns
        -------
        bool
            True if sufficient data, False otherwise
        """
        target_valid = self.alternative[target].notna().sum()

        if target_valid < min_samples:
            if self.verbose:
                missing_count = len(self.alternative) - target_valid
                missing_pct = 100 * missing_count / len(self.alternative)
                print(f"  ⚠️  Target '{target}' has insufficient alternative data: "
                      f"{target_valid}/{len(self.alternative)} valid "
                      f"({missing_pct:.0f}% missing)")
            return False

        return True

    def _check_predictors_data_availability(self,
                                            predictors: Tuple[str, ...],
                                            target: str,
                                            min_samples: int) -> bool:
        """
        Check if predictor combination has sufficient data with target.

        ✅ MATCHES: parametric_analysis_foundation._check_predictors_data_availability

        Only prints warnings for predictor-specific issues.

        Returns
        -------
        bool
            True if sufficient data, False otherwise
        """
        # Try to prepare dataset
        result = self.prepare_dataset(
            self.alternative,
            list(predictors),
            target,
            min_samples=min_samples
        )

        if result is None:
            # Check which predictors are missing
            missing_preds = [p for p in predictors
                             if self.alternative[p].isna().all()]

            # Only warn about predictor-specific issues
            if missing_preds and self.verbose:
                print(f"  ⚠️  Skipping {predictors}: "
                      f"predictors completely missing: {missing_preds}")

            return False

        return True