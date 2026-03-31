# File: parametric_analysis_foundation_class.py

"""
Parametric Analysis Foundation
==============================
Provides common setup and utilities for parametric regression analyses
via composition rather than inheritance.

This class handles:
- Model recommendation extraction and validation
- Matched dataset creation
- Data preparation and cleaning
- Common validation logic

Used by analyzer classes via composition.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from config import LoggerMixin
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory
from knowledge_layer.model_selectors.best_model_selector import BestModelSelector
from knowledge_layer.data_split_handling_class import DataSplitHandler
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


class ParametricAnalysisFoundation(LoggerMixin):
    """
    Foundation class providing common functionality for parametric analyses.

    Designed to be used via composition, not inheritance.

    Responsibilities:
    ----------------
    - Extract tested combinations from model_recommendations
    - Create and manage matched conventional datasets
    - Prepare and clean data for model fitting
    - Validate inputs and configurations
    - Provide utility methods for common operations
    """

    def __init__(self,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 independent_variables: List[str],
                 dependent_variables: Optional[List[str]] = None,
                 model_recommendations: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = True,
                 detailed_results: Optional[Dict] = None,
                 selection_metric:str = 'rrmse',
                 selection_strategy: str = 'simple',
                target_predictor_constraints: Optional[Dict[str, List[str]]] = None):
        """
        Initialize analysis foundation.

        Parameters
        ----------
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_data : pd.DataFrame
            Alternative fuel ships dataset
        independent_variables : List[str]
            List of potential predictor dimensions
        dependent_variables : List[str], optional
            List of target dimensions. If None, uses independent_variables.
        model_recommendations : dict, optional
            Dictionary from ParametricModelSelector.model_recommendations
            Format: {(predictor_tuple, target): model_type_string}
            **REQUIRED for proper operation**

        target_predictor_constraints : dict, optional
            Restricts which predictor variables may be used for each target.
            Format: {target_name: [allowed_predictor_1, allowed_predictor_2, ...]}
            Example: {'LBP (M)': ['DWT', 'TEU']}
            Any combination for a constrained target that contains predictors
            outside the allowed list is excluded, regardless of model performance.
            Targets not mentioned in the dict are unconstrained.

        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Whether to print progress and warnings
        """
        self.random_state = random_state
        self.verbose = verbose

        # ✅ Store BOTH for parametric selector
        self.model_recommendations = model_recommendations or {}
        self.detailed_results = detailed_results or {}

        # Store and validate constraints
        self.target_predictor_constraints = target_predictor_constraints or {}
        if self.target_predictor_constraints and verbose:
            self._print_constraint_summary()

        # ✅ Initialize BestModelSelector
        self.model_selector = BestModelSelector(
            selection_metric=selection_metric,
            selection_strategy=selection_strategy,
            verbose=verbose
        )

        self.detailed_results = detailed_results or {}

        # Setup dimensions
        if dependent_variables is None:
            dependent_variables = independent_variables

        self._setup_dimensions(
            conventional_data=conventional_data,
            independent_variables=independent_variables,
            dependent_variables=dependent_variables
        )

        # Store datasets
        self.conventional_full = conventional_data[self.available_dimensions].copy()
        self.alternative = alternative_data[self.available_dimensions].copy()

        # Validate and store model recommendations
        self._validate_model_recommendations(model_recommendations)
        self.model_recommendations = model_recommendations

        # Validate data quality
        self._validate_data_quality()

        # ✅ NEW: Print diagnostics BEFORE creating matched datasets
        if verbose:
            self.print_variable_diagnostics()

        # Create matched datasets (now with filtering)
        self.matched_data_sets = self._create_matched_sets()
        self.selector = None  # Will be set by analyzer if available

    # ========================================================================
    # SETUP AND VALIDATION
    # ========================================================================

    def _print_constraint_summary(self):
        """Print a summary of active target predictor constraints."""
        print("\n" + "=" * 60)
        print("TARGET PREDICTOR CONSTRAINTS")
        print("=" * 60)
        for target, allowed in self.target_predictor_constraints.items():
            print(f"  {target}:")
            print(f"    Allowed predictors: {allowed}")
        print("=" * 60 + "\n")

    def _setup_dimensions(self,
                          conventional_data: pd.DataFrame,
                          independent_variables: List[str],
                          dependent_variables: List[str]):
        """
        Setup and validate dimension lists.

        Sets:
        - self.available_dimensions
        - self.variable_dimensions
        - self.target_dimensions
        """
        all_dimensions = list(set(independent_variables + dependent_variables))
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
        self.variable_dimensions = [d for d in independent_variables
                                    if d in available_dimensions]
        self.target_dimensions = [d for d in dependent_variables
                                  if d in available_dimensions]

    def _validate_model_recommendations(self, model_recommendations: Optional[Dict]):
        """Validate model recommendations input."""
        if model_recommendations is None or len(model_recommendations) == 0:
            raise ValueError(
                "model_recommendations is required and cannot be empty!\n"
                "Please run ParametricModelSelector.analyze_all_combinations() first,\n"
                "then pass selector.model_recommendations to this analyzer."
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

    def _combination_satisfies_constraints(self,
                                           predictors: Tuple[str, ...],
                                           target: str) -> bool:
        """
        Check whether a predictor combination satisfies the target constraints.

        If no constraint is defined for this target, always returns True.
        If a constraint exists, all predictors in the combination must be
        members of the allowed list — any predictor outside the list
        causes the combination to be rejected.

        Parameters
        ----------
        predictors : tuple of str
            The predictor combination to check.
        target : str
            The target variable being predicted.

        Returns
        -------
        bool
            True if the combination is allowed, False otherwise.
        """
        if target not in self.target_predictor_constraints:
            return True  # No constraint for this target

        allowed = set(self.target_predictor_constraints[target])
        return all(pred in allowed for pred in predictors)



    # ========================================================================
    # COMBINATION EXTRACTION
    # ========================================================================

    def extract_combinations_for_target(self,
                                        target: str,
                                        max_predictors: Optional[int] = None,
                                        filter_unavailable: bool = True
                                        ) -> List[Tuple[str, ...]]:
        """
        Extract predictor combinations for a target from model recommendations.

        Parameters
        ----------
        target : str
            Target dimension
        max_predictors : int, optional
            Filter to combinations with at most this many predictors
        filter_unavailable : bool, default=True
            If True, filters out combinations containing unavailable variables

        Returns
        -------
        list of tuples
            Predictor combinations with recommendations, sorted by size

        Notes
        -----
        When filter_unavailable=True, combinations are filtered to only include
        those where ALL predictors exist in self.available_dimensions. This is
        crucial when the analyzer is initialized with a different variable list
        than the selector was trained with.
        """
        combos = []

        for (pred_combo, tgt) in self.model_recommendations.keys():
            if tgt != target:
                continue

            if max_predictors is not None and len(pred_combo) > max_predictors:
                continue

            if filter_unavailable:
                if not all(pred in self.available_dimensions for pred in pred_combo):
                    continue

            # ✅ Constraint filter: reject if any predictor is outside the allowed list
            if not self._combination_satisfies_constraints(pred_combo, target):
                continue

            combos.append(pred_combo)

        combos.sort(key=lambda x: (len(x), x))
        return combos

    def get_combinations_summary(self) -> pd.DataFrame:
        """
        Get summary of available combinations from recommendations.

        Only includes combinations where all predictors are available in the
        current datasets.

        Returns
        -------
        pd.DataFrame
            Summary with columns:
            - target: Target dimension
            - n_combinations: Number of valid combinations
            - n_skipped: Number of combinations skipped (missing variables)
            - predictor_sizes: List of predictor counts in valid combinations
        """
        summary_data = []

        for target in self.target_dimensions:
            # Get valid combinations (with filtering)
            valid_combos = self.extract_combinations_for_target(
                target,
                filter_unavailable=True
            )

            # Get all combinations (without filtering) to count skipped
            all_combos = self.extract_combinations_for_target(
                target,
                filter_unavailable=False
            )

            n_skipped = len(all_combos) - len(valid_combos)

            if valid_combos:
                sizes = sorted(list(set(len(c) for c in valid_combos)))
                summary_data.append({
                    'target': target,
                    'n_combinations': len(valid_combos),
                    'n_skipped': n_skipped,
                    'predictor_sizes': sizes
                })
            elif all_combos:
                # Had combinations but all were filtered
                summary_data.append({
                    'target': target,
                    'n_combinations': 0,
                    'n_skipped': n_skipped,
                    'predictor_sizes': []
                })

        return pd.DataFrame(summary_data)

    # ========================================================================
    # MATCHED DATASET CREATION
    # ========================================================================

    def _create_matched_sets(self) -> Dict[Tuple[str, ...], pd.DataFrame]:
        """
        Create matched conventional datasets for each predictor combination.

        **FILTERS** combinations to only include those with all variables available
        in the current datasets.

        Returns
        -------
        dict
            Maps predictor_tuple → matched_dataset_df
        """
        # Collect all unique predictor combinations
        combo_list = []
        for target in self.target_dimensions:
            combos = self.extract_combinations_for_target(target)
            combo_list.extend(combos)

        # Get unique combinations
        all_unique_combos = sorted(set(combo_list))

        # ✅ FILTER: Only keep combinations where ALL predictors are available
        valid_combos = []
        skipped_combos = []

        for combo in all_unique_combos:
            # Check if all predictors in this combination are available
            if all(pred in self.available_dimensions for pred in combo):
                valid_combos.append(combo)
            else:
                missing = [pred for pred in combo if pred not in self.available_dimensions]
                skipped_combos.append((combo, missing))

        # Report skipped combinations
        if skipped_combos and self.verbose:
            print(f"\n⚠️  Skipped {len(skipped_combos)} combinations due to missing variables:")
            for combo, missing in skipped_combos[:5]:  # Show first 5
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
            print(f"\nCreating matched datasets for {len(valid_combos)} valid predictor combinations "
                  f"(filtered from {len(all_unique_combos)} total)...")

        # Create matched datasets only for valid combinations
        try:
            matched_sets = MatchedDataset.multiple_matched_datasets(
                alternative_fleet=self.alternative,
                conventional_fleet=self.conventional_full,
                logger=self.logger,
                predictor_groups=valid_combos,
                std_reference='conventional'
            )
        except KeyError as e:
            # If matching fails due to missing variables, provide helpful error
            raise KeyError(
                f"Matched dataset creation failed: {str(e)}\n"
                f"This suggests a predictor variable exists in available_dimensions "
                f"but not in the actual dataset columns.\n"
                f"Available dimensions: {self.available_dimensions}\n"
                f"Dataset columns: {list(self.conventional_full.columns)}"
            ) from e

        if self.verbose:
            print(f"✅ Created {len(matched_sets)} matched datasets")

        return matched_sets

    def get_matched_dataset(self, predictors: Tuple[str, ...]) -> Optional[pd.DataFrame]:
        """
        Get matched dataset for a specific predictor combination.

        Parameters
        ----------
        predictors : tuple of str
            Predictor combination

        Returns
        -------
        pd.DataFrame or None
            Matched conventional dataset, or None if not available
        """
        return self.matched_data_sets.get(tuple(predictors))

    # ========================================================================
    # MODEL TYPE RETRIEVAL
    # ========================================================================

    def get_model_type(self, predictors: Tuple[str, ...], target: str) -> str:
        """
        Get recommended model type from selector recommendations.

        Parameters
        ----------
        predictors : tuple of str
            Predictor combination
        target : str
            Target dimension

        Returns
        -------
        str
            Model type string

        Raises
        ------
        ValueError
            If no recommendation found
        """
        key = (tuple(predictors), target)

        if key in self.model_recommendations:
            return self.model_recommendations[key]
        else:
            raise ValueError(
                f"No recommendation found for {predictors} → {target}.\n"
                f"This should not happen when using properly extracted combinations.\n"
                f"Please report this as a bug."
            )

    # ========================================================================
    # DATA PREPARATION UTILITIES
    # ========================================================================

    def prepare_dataset(self,
                        dataset: pd.DataFrame,
                        predictors: List[str],
                        target: str,
                        min_samples: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare and clean a single dataset for model fitting.

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
                if len(predictors) > 1:
                    print(f"for combination: {predictors}")
                else:
                    print(f"for predictor: {predictors}")
            return None

        return X_clean, y_clean

    # ========================================================================
    # MODEL TRAINING UTILITIES
    # ========================================================================

    def fit_and_predict(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        model_type: str) -> Optional[np.ndarray]:
        """
        Fit model and make predictions.

        Parameters
        ----------
        X_train : np.ndarray
            Training predictors
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test predictors
        model_type : str
            Model type to fit

        Returns
        -------
        np.ndarray or None
            Predictions, or None if fitting failed
        """
        try:
            model, preprocessor = ModelFactory.create_and_fit(
                model_type, X_train, y_train
            )

            y_pred = ModelFactory.predict(model, preprocessor, X_test, model_type)

            return y_pred

        except Exception as e:
            if self.verbose:
                print(f"  Model fitting failed: {str(e)[:80]}")
            return None

    def evaluate_with_cv(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         model_type: str,
                         n_folds: int = 10,
                         cv_method: str = 'standard') -> Optional[Dict]:
        """
        Evaluate model via cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Predictor data
        y : np.ndarray
            Target data
        model_type : str
            Model type to evaluate
        n_folds : int, default=10
            Number of CV folds
        cv_method : str, default='standard'
            CV method ('standard', 'stratified', 'grouped')

        Returns
        -------
        dict or None
            Averaged metrics with std dev, or None if CV failed
        """
        # Setup CV
        cv_config = DataSplitHandler.setup_cv_config(
            X=X,
            y=y,
            cv_method=cv_method,
            n_folds=n_folds,
            groups=None,
            random_state=self.random_state
        )

        # Perform CV
        cv_metrics = []

        splitter = cv_config['splitter']
        split_args = cv_config['split_args']
        split_kwargs = cv_config['split_kwargs']

        for train_idx, test_idx in splitter.split(*split_args, **split_kwargs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # Fit and predict
                y_pred = self.fit_and_predict(X_train, y_train, X_test, model_type)

                if y_pred is None:
                    continue

                # Calculate metrics
                n_features = ModelFactory.get_actual_n_features(model_type, X_train)
                metric_calc = RegressionMetrics(y_test, y_pred)
                metrics = metric_calc.calculate_all_metrics(n_features)

                cv_metrics.append(metrics)

            except Exception as e:
                if self.verbose:
                    print(f"    CV fold failed: {str(e)[:60]}")
                continue

        if not cv_metrics:
            return None

        # Average metrics across folds
        return self.average_cv_metrics(cv_metrics)

    @staticmethod
    def average_cv_metrics(cv_metrics: List[Dict]) -> Dict:
        """
        Average metrics across CV folds.

        Parameters
        ----------
        cv_metrics : list of dict
            Metrics from each fold

        Returns
        -------
        dict
            Averaged metrics with std dev
        """
        avg_metrics = {}

        # Metrics to average
        metric_names = ['r2', 'adj_r2', 'rrmse', 'mape', 'mae', 'rmse']

        for metric in metric_names:
            values = [m[metric] for m in cv_metrics
                      if not np.isnan(m.get(metric, np.nan))]

            if values:
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            else:
                avg_metrics[metric] = np.nan
                avg_metrics[f'{metric}_std'] = np.nan

        return avg_metrics

    # ========================================================================
    # INFORMATION METHODS
    # ========================================================================

    def print_initialization_summary(self):
        """Print initialization summary."""
        if not self.verbose:
            return

        print(f"\nParametric Analysis Foundation initialized:")
        print(f"  Conventional full: {len(self.conventional_full)} ships")
        print(f"  Alternative: {len(self.alternative)} ships")
        print(f"  Model recommendations: {len(self.model_recommendations)} combinations")
        print(f"  Matched datasets: {len(self.matched_data_sets)} unique predictor sets")

        # Show breakdown by target
        targets_in_recommendations = set(tgt for (_, tgt) in self.model_recommendations.keys())
        print(f"  Targets with recommendations: {len(targets_in_recommendations)}")
        for target in sorted(targets_in_recommendations):
            n_combos = sum(1 for (_, tgt) in self.model_recommendations.keys()
                           if tgt == target)
            print(f"    • {target}: {n_combos} combinations")

    def diagnose_missing_variables(self) -> pd.DataFrame:
        """
        Diagnose which variables from selector recommendations are missing.

        Useful for understanding why combinations are being filtered.

        Returns
        -------
        pd.DataFrame
            Summary with columns:
            - variable: Variable name
            - in_recommendations: Whether used in recommendations
            - in_available_dimensions: Whether available in datasets
            - n_combinations_using: Number of combinations using this variable
        """
        # Get all variables used in recommendations
        all_vars_in_recommendations = set()
        var_usage_count = {}

        for (pred_combo, target) in self.model_recommendations.keys():
            for pred in pred_combo:
                all_vars_in_recommendations.add(pred)
                var_usage_count[pred] = var_usage_count.get(pred, 0) + 1

        # Check availability
        diagnostics = []

        for var in sorted(all_vars_in_recommendations):
            diagnostics.append({
                'variable': var,
                'in_recommendations': True,
                'in_available_dimensions': var in self.available_dimensions,
                'n_combinations_using': var_usage_count[var],
                'status': '✅ Available' if var in self.available_dimensions else '❌ Missing'
            })

        # Also check for variables in available_dimensions not used in recommendations
        for var in sorted(self.available_dimensions):
            if var not in all_vars_in_recommendations:
                diagnostics.append({
                    'variable': var,
                    'in_recommendations': False,
                    'in_available_dimensions': True,
                    'n_combinations_using': 0,
                    'status': 'ℹ️  Available but unused'
                })

        df = pd.DataFrame(diagnostics)

        # Sort: missing first, then by usage count
        df['sort_key'] = df.apply(
            lambda row: (0 if not row['in_available_dimensions'] else 1, -row['n_combinations_using']),
            axis=1
        )
        df = df.sort_values('sort_key').drop('sort_key', axis=1)

        return df

    def print_variable_diagnostics(self):
        """Print diagnostic information about variable availability."""
        if not self.verbose:
            return

        print("\n" + "=" * 80)
        print("VARIABLE AVAILABILITY DIAGNOSTICS")
        print("=" * 80)

        diag_df = self.diagnose_missing_variables()

        missing = diag_df[~diag_df['in_available_dimensions'] & diag_df['in_recommendations']]
        available = diag_df[diag_df['in_available_dimensions'] & diag_df['in_recommendations']]
        unused = diag_df[~diag_df['in_recommendations']]

        if len(missing) > 0:
            print(f"\n❌ MISSING VARIABLES ({len(missing)}):")
            print("These variables are used in selector recommendations but not available in datasets:")
            for _, row in missing.iterrows():
                print(f"  • {row['variable']:<20s} (used in {row['n_combinations_using']} combinations)")

        print(f"\n✅ AVAILABLE VARIABLES ({len(available)}):")
        print("These variables are used in recommendations and available:")
        for _, row in available.head(10).iterrows():
            print(f"  • {row['variable']:<20s} (used in {row['n_combinations_using']} combinations)")
        if len(available) > 10:
            print(f"  • ... and {len(available) - 10} more")

        if len(unused) > 0:
            print(f"\nℹ️  UNUSED VARIABLES ({len(unused)}):")
            print("These variables are available but not used in any recommendations:")
            for _, row in unused.head(5).iterrows():
                print(f"  • {row['variable']}")
            if len(unused) > 5:
                print(f"  • ... and {len(unused) - 5} more")

        print("=" * 80 + "\n")


    # ========================================================================
    # ✅ NEW: NORMALIZATION SUPPORT
    # ========================================================================

    def prepare_dataset_normalized(self,
                                   dataset: pd.DataFrame,
                                   predictors: List[str],
                                   target: str,
                                   min_samples: int = 5,
                                   normalization_params: Optional[Dict] = None,
                                   fit_params: bool = False,
                                   normalization_method: str = 'zscore'
                                   ) -> Optional[Tuple]:
        """
        Prepare dataset with optional normalization.

        **Use Cases:**
        - fit_params=True: Calculate normalization on training data
        - fit_params=False + params: Apply existing normalization to test data
        - params=None: No normalization (standard preparation)

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
        normalization_params : dict, optional
            Pre-calculated normalization parameters to apply
            Format: {'X': {'mean': array, 'std': array}, 'y': {'mean': float, 'std': float}}
        fit_params : bool, default=False
            If True, calculate and return normalization parameters
        normalization_method : str, default='zscore'
            Normalization method: 'zscore' or 'minmax'

        Returns
        -------
        tuple or None
            - If fit_params=True: (X_norm, y_norm, normalization_params)
            - If fit_params=False: (X_norm, y_norm)
            - None if insufficient data

        Examples
        --------
        # Fit normalization on training data
        >>> X_train, y_train, params = foundation.prepare_dataset_normalized(
        ...     train_data, ['DWT', 'GT'], 'LOA', fit_params=True
        ... )

        # Apply same normalization to test data
        >>> X_test, y_test = foundation.prepare_dataset_normalized(
        ...     test_data, ['DWT', 'GT'], 'LOA',
        ...     normalization_params=params
        ... )

        # No normalization
        >>> X, y = foundation.prepare_dataset_normalized(
        ...     data, ['DWT'], 'LOA'
        ... )
        """
        # Standard preparation
        result = self.prepare_dataset(dataset, predictors, target, min_samples)
        if result is None:
            return None

        X, y = result

        # No normalization requested
        if normalization_params is None and not fit_params:
            return X, y

        # Apply or calculate normalization
        if fit_params:
            X_norm, y_norm, params = self._normalize_data(
                X, y,
                method=normalization_method,
                fit_params=True,
                params=None
            )
            return X_norm, y_norm, params
        else:
            X_norm, y_norm, _ = self._normalize_data(
                X, y,
                method=normalization_method,
                fit_params=False,
                params=normalization_params
            )
            return X_norm, y_norm

    def _normalize_data(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       method: str = 'zscore',
                       fit_params: bool = False,
                       params: Optional[Dict] = None
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """
        Normalize predictor and target data.

        **Normalization Methods:**

        Z-score (standardization):
        - X_norm = (X - mean) / std
        - Centers data at 0, scales to unit variance
        - Preserves outliers
        - Recommended for coefficient comparison

        Min-Max (normalization):
        - X_norm = (X - min) / (max - min)
        - Scales data to [0, 1] range
        - Sensitive to outliers
        - Use when bounded range is important

        Parameters
        ----------
        X : np.ndarray
            Predictor data (n_samples, n_features)
        y : np.ndarray
            Target data (n_samples,)
        method : str, default='zscore'
            Normalization method: 'zscore' or 'minmax'
        fit_params : bool, default=False
            If True, calculate normalization parameters from data
        params : dict, optional
            Pre-calculated normalization parameters to apply

        Returns
        -------
        X_norm : np.ndarray
            Normalized predictors
        y_norm : np.ndarray
            Normalized target
        params : dict or None
            Normalization parameters (if fit_params=True)

        Raises
        ------
        ValueError
            If fit_params=False and params=None
            If unknown normalization method
        """
        if fit_params:
            # Calculate normalization parameters from data
            params = {'X': {}, 'y': {}}

            if method == 'zscore':
                # Z-score normalization: (x - mean) / std
                params['X']['mean'] = np.mean(X, axis=0)
                params['X']['std'] = np.std(X, axis=0)

                params['y']['mean'] = np.mean(y)
                params['y']['std'] = np.std(y)

            elif method == 'minmax':
                # Min-max normalization: (x - min) / (max - min)
                params['X']['min'] = np.min(X, axis=0)
                params['X']['max'] = np.max(X, axis=0)

                params['y']['min'] = np.min(y)
                params['y']['max'] = np.max(y)

            else:
                raise ValueError(
                    f"Unknown normalization method: {method}. "
                    f"Must be 'zscore' or 'minmax'"
                )

        elif params is None:
            raise ValueError(
                "Must provide normalization params when fit_params=False"
            )

        # Apply normalization
        if method == 'zscore':
            X_norm = (X - params['X']['mean']) / (params['X']['std'] + 1e-10)
            y_norm = (y - params['y']['mean']) / (params['y']['std'] + 1e-10)

        elif method == 'minmax':
            X_range = params['X']['max'] - params['X']['min']
            y_range = params['y']['max'] - params['y']['min']

            X_norm = (X - params['X']['min']) / (X_range + 1e-10)
            y_norm = (y - params['y']['min']) / (y_range + 1e-10)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return X_norm, y_norm, params if fit_params else None

    def denormalize_predictions(self,
                               y_pred_norm: np.ndarray,
                               normalization_params: Dict,
                               normalization_method: str = 'zscore'
                               ) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.

        Parameters
        ----------
        y_pred_norm : np.ndarray
            Normalized predictions
        normalization_params : dict
            Normalization parameters from _normalize_data
        normalization_method : str, default='zscore'
            Method used for normalization

        Returns
        -------
        np.ndarray
            Predictions in original scale
        """
        y_params = normalization_params['y']

        if normalization_method == 'zscore':
            y_pred = y_pred_norm * y_params['std'] + y_params['mean']

        elif normalization_method == 'minmax':
            y_range = y_params['max'] - y_params['min']
            y_pred = y_pred_norm * y_range + y_params['min']

        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

        return y_pred

    # In parametric_analysis_foundation.py

    def get_best_combination_for_target(
            self,
            target: str,
            max_predictors: Optional[int] = None,
            filter_by_vif: bool = True,
            vif_threshold: float = 10.0,
            require_alternative_data: bool = False,  # ← CHANGED DEFAULT: False instead of True
            min_alternative_samples: int = 5,
            selection_based_on_conventional: bool = True,
            debug:bool=False# ← NEW PARAMETER
    ) -> Optional[Tuple[Tuple[str, ...], str]]:
        """
        Get best predictor combination for target.

        Parameters
        ----------
        target : str
            Target dimension
        max_predictors : int, optional
            Maximum number of predictors
        filter_by_vif : bool, default=True
            Filter combinations by VIF threshold
        vif_threshold : float, default=10.0
            Maximum acceptable VIF
        require_alternative_data : bool, default=False
            If True, only return combinations with sufficient alternative data.
            **INTERACTION with selection_based_on_conventional**:
            - If selection_based_on_conventional=True: checks testability AFTER selection
            - If selection_based_on_conventional=False: filters candidates BEFORE selection
        min_alternative_samples : int, default=5
            Minimum samples required in alternative dataset
        selection_based_on_conventional : bool, default=True
            **NEW**: If True, selects best by conventional performance, THEN checks
            alternative data availability. If False, uses old behavior (filters first).

        Returns
        -------
        tuple of (predictors, model_type) or None
            Best combination, or None if no valid combination

        Notes
        -----
        **New behavior (selection_based_on_conventional=True)**:
        1. Select best by conventional performance
        2. Check if testable on alternative data
        3. If not, try next-best that IS testable

        **Old behavior (selection_based_on_conventional=False)**:
        1. Filter candidates by alternative data availability
        2. Select best from filtered candidates
        """

        if debug:
            print(f"\n  {'·' * 60}")
            print(f"  DEBUG: get_best_combination_for_target('{target}')")
            print(f"  {'·' * 60}")
            print(f"    Parameters:")
            print(f"      max_predictors: {max_predictors}")
            print(f"      filter_by_vif: {filter_by_vif} (threshold: {vif_threshold})")
            print(f"      require_alternative_data: {require_alternative_data}")
            print(f"      selection_based_on_conventional: {selection_based_on_conventional}")

        # Get all candidate combinations
        candidates = []
        filter_stats = {
            'total_in_recommendations': 0,
            'filtered_wrong_target': 0,
            'filtered_too_many_predictors': 0,
            'filtered_missing_variables': 0,
            'filtered_no_detailed_results': 0,
            'filtered_model_type_missing': 0,
            'filtered_vif': 0,
            'passed_all_filters': 0
        }

        for (predictors_tuple, tgt), model_type in self.model_recommendations.items():
            filter_stats['total_in_recommendations'] += 1

            if tgt != target:
                filter_stats['filtered_wrong_target'] += 1
                continue

            if max_predictors is not None and len(predictors_tuple) > max_predictors:
                filter_stats['filtered_too_many_predictors'] += 1
                if debug:
                    print(f"    ❌ FILTERED (too many predictors): {predictors_tuple} "
                          f"(n={len(predictors_tuple)} > {max_predictors})")
                continue

            if not all(pred in self.available_dimensions for pred in predictors_tuple):
                filter_stats['filtered_missing_variables'] += 1
                missing = [p for p in predictors_tuple if p not in self.available_dimensions]
                if debug:
                    print(f"    ❌ FILTERED (missing variables): {predictors_tuple} "
                          f"(missing: {missing})")
                continue

            if not self._combination_satisfies_constraints(predictors_tuple, target):
                if debug:
                    allowed = self.target_predictor_constraints.get(target, [])
                    rejected = [p for p in predictors_tuple if p not in allowed]
                    print(f"    ❌ FILTERED (constraint): {predictors_tuple} "
                          f"(disallowed predictors: {rejected})")
                continue


            # Get metrics from detailed_results
            key = (predictors_tuple, tgt)

            if key not in self.detailed_results:
                filter_stats['filtered_no_detailed_results'] += 1
                if debug:
                    print(f"    ❌ FILTERED (no detailed_results): {predictors_tuple} "
                          f"(key {key} not in detailed_results)")
                continue

            model_results = self.detailed_results[key]

            # Filter 5: Model type in results
            if model_type not in model_results:
                filter_stats['filtered_model_type_missing'] += 1
                available_models = list(model_results.keys())
                if debug:
                    print(f"    ❌ FILTERED (model type missing): {predictors_tuple} "
                          f"('{model_type}' not in results)")
                    print(f"       Available models: {available_models}")
                continue

            metrics = model_results[model_type]

            # Filter 6: VIF
            if filter_by_vif:
                max_vif = metrics.get('max_vif', 1.0)
                if max_vif > vif_threshold:
                    filter_stats['filtered_vif'] += 1
                    if debug:
                        print(f"    ❌ FILTERED (VIF too high): {predictors_tuple} "
                              f"(VIF={max_vif:.2f} > {vif_threshold})")
                    continue

            filter_stats['passed_all_filters'] += 1

            candidate = {
                'predictors': predictors_tuple,
                'model_type': model_type,
                'metrics': metrics,
                'n_predictors': len(predictors_tuple)
            }

            candidates.append(candidate)

        if debug:
            adj_r2 = metrics.get('adj_r2', np.nan)
            bic = metrics.get('bic', np.nan)
            vif = metrics.get('max_vif', 1.0)
            print(f"    ✅ PASSED: {predictors_tuple} ({model_type})")
            print(f"       Metrics: adj_r2={adj_r2:.4f}, BIC={bic:.1f}, VIF={vif:.2f}")

        # ✅ DEBUG: Show how many candidates survived filtering

        print(f"    Found {len(candidates)} candidates after filtering")
        if len(candidates) == 0:
            # Check why no candidates
            n_for_target = sum(1 for (_, tgt) in self.model_recommendations.keys() if tgt == target)
            print(f"    Total combinations for {target} in recommendations: {n_for_target}")

            # Check if any passed max_predictors filter
            if max_predictors is not None:
                n_within_max = sum(1 for (preds, tgt) in self.model_recommendations.keys()
                                   if tgt == target and len(preds) <= max_predictors)
                print(f"    Within max_predictors={max_predictors}: {n_within_max}")

            # Check if any have detailed_results
            n_with_details = sum(1 for (preds, tgt), model in self.model_recommendations.items()
                                 if tgt == target and (preds, tgt) in self.detailed_results)
            print(f"    With detailed_results: {n_with_details}")

        if debug:
            print(f"\n    Filter Summary:")
            print(f"      Total combinations in recommendations: {filter_stats['total_in_recommendations']}")
            print(f"      └─ Wrong target: {filter_stats['filtered_wrong_target']}")
            print(
                f"      └─ For this target: {filter_stats['total_in_recommendations'] - filter_stats['filtered_wrong_target']}")
            print(f"         ├─ Too many predictors: {filter_stats['filtered_too_many_predictors']}")
            print(f"         ├─ Missing variables: {filter_stats['filtered_missing_variables']}")
            print(f"         ├─ No detailed_results: {filter_stats['filtered_no_detailed_results']}")
            print(f"         ├─ Model type missing: {filter_stats['filtered_model_type_missing']}")
            print(f"         ├─ VIF too high: {filter_stats['filtered_vif']}")
            print(f"         └─ ✅ PASSED: {filter_stats['passed_all_filters']}")

        if not candidates:
            if debug:
                print(f"\n    ⚠️  NO CANDIDATES after filtering!")
            return None

        # ========================================================================
        # BRANCHING LOGIC: New vs Old Behavior
        # ========================================================================
        if debug:
            print(f"\n    Selecting best from {len(candidates)} candidates...")
            print(
                f"    Selection strategy: {'conventional-first' if selection_based_on_conventional else 'filter-first'}")

        if selection_based_on_conventional:
            # ✅ NEW BEHAVIOR: Select best, THEN check testability
            return self._select_best_then_check_testability(
                candidates=candidates,
                target=target,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples,
                debug=debug
            )

        else:
            # 🔄 OLD BEHAVIOR: Filter by testability, THEN select best
            return self._filter_then_select_best(
                candidates=candidates,
                target=target,
                require_alternative_data=require_alternative_data,
                min_alternative_samples=min_alternative_samples
            )

    def _select_best_then_check_testability(
            self,
            candidates: List[Dict],
            target: str,
            require_alternative_data: bool,
            min_alternative_samples: int,
            debug:bool
    ) -> Optional[Tuple[Tuple[str, ...], str]]:
        """
        NEW LOGIC: Select best by conventional, then check testability.

        Strategy:
        1. Sort candidates by conventional performance
        2. Iterate through sorted list
        3. Return first that is testable (or first if not required)
        """

        if debug:
            print(f"\n    Using conventional-first strategy...")

        # Use BestModelSelector to pick best by conventional metrics
        best_candidate = self.model_selector.select_best_combination(
            candidates=candidates,
            selection_metric=self.model_selector.selection_metric,
            max_vif=10.0,
            prefer_simpler=True,
            penalize_heteroscedasticity=True
        )



        if best_candidate is None:
            return None

        best_predictors = best_candidate['predictors']
        best_model = best_candidate['model_type']
        best_metrics = best_candidate['metrics']

        if debug:
            metric = self.model_selector.selection_metric
            metric_val = best_metrics.get(metric, np.nan)
            print(f"    Best by conventional: {best_predictors} ({best_model})")
            print(f"      {metric} = {metric_val:.4f}")

        if not require_alternative_data:
            if debug:
                print(f"    ✅ Returning (require_alternative_data=False)")
            return best_predictors, best_model

        # Check if best is testable
        if debug:
            print(f"    Checking alternative data availability...")

        # If we don't require alternative data, return immediately
        if not require_alternative_data:
            if self.verbose:
                metrics = best_candidate['metrics']
                print(f"  ✓ Selected: {best_predictors} ({best_model})")
                print(f"    Conventional: R²={metrics['adj_r2']:.3f}")
            return best_predictors, best_model

        # Check if best is testable
        if debug:
            print(f"    Checking alternative data availability...")

        is_testable = self._check_predictors_data_availability(
            best_predictors, target, min_alternative_samples
        )

        if is_testable:
            if debug:
                print(f"    ✅ Best combination is testable on alternative data")
            return best_predictors, best_model

        # Best isn't testable - try alternatives
        if debug:
            print(f"    ⚠️  Best combination NOT testable on alternative data")
            print(f"    Trying alternatives...")

        # Sort all candidates by performance
        selection_metric = self.model_selector.selection_metric

        if selection_metric in ['aic', 'bic', 'rrmse', 'mape']:
            candidates.sort(key=lambda x: x['metrics'].get(selection_metric, float('inf')))
        else:
            candidates.sort(key=lambda x: x['metrics'].get(selection_metric, -999), reverse=True)

        # Try each in order
        for i, candidate in enumerate(candidates):
            preds = candidate['predictors']
            model = candidate['model_type']

            if preds == best_predictors:
                continue  # Already tried this

            if debug:
                metric_val = candidate['metrics'].get(selection_metric, np.nan)
                print(f"    Trying alternative #{i + 1}: {preds} ({model})")
                print(f"      {selection_metric} = {metric_val:.4f}")

            if self._check_predictors_data_availability(preds, target, min_alternative_samples):
                if debug:
                    print(f"    ✅ This alternative IS testable - using it")
                return preds, model
            elif debug:
                print(f"    ❌ This alternative NOT testable - continuing...")

        # No testable alternatives
        if debug:
            print(f"    ⚠️  No testable alternatives found!")
        return None

    def _filter_then_select_best(
            self,
            candidates: List[Dict],
            target: str,
            require_alternative_data: bool,
            min_alternative_samples: int
    ) -> Optional[Tuple[Tuple[str, ...], str]]:
        """
        OLD LOGIC: Filter by testability first, then select best.

        This is the original behavior - maintained for backward compatibility.
        """

        # Filter candidates by alternative data availability
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

        # Now select best from filtered candidates
        best_candidate = self.model_selector.select_best_combination(
            candidates=candidates,
            selection_metric=self.model_selector.selection_metric,
            max_vif=10.0,
            prefer_simpler=True,
            penalize_heteroscedasticity=True
        )

        if best_candidate is None:
            return None

        return best_candidate['predictors'], best_candidate['model_type']

    def _check_target_data_availability(self,
                                        target: str,
                                        min_samples: int) -> bool:
        """
        Check if target has sufficient data in alternative dataset.

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
            # (target issues already handled in _check_target_data_availability)
            if missing_preds and self.verbose:
                print(f"  ⚠️  Skipping {predictors}: "
                      f"predictors completely missing: {missing_preds}")

            return False

        return True