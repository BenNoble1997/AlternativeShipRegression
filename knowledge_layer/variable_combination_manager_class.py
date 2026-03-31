"""
Variable Combination Manager
============================

Handles variable validation, filtering, and combination generation for
regression model selectors. This class encapsulates the common logic
for managing dimensions, relationship contexts, and forbidden combinations.

Key responsibilities:
- Validate dimensions against available data
- Filter combinations based on relationship rules
- Generate valid predictor combinations
- Manage dimension groupings and classifications
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Set
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VariableCombinationManager:
    """
    Manages variable validation and combination generation for regression selectors.

    This class extracts common functionality for:
    - Validating dimensions against training data
    - Managing relationship exclusion rules
    - Generating valid predictor combinations
    - Classifying combinations by dimension groups
    """

    def __init__(self,
                 training_data: pd.DataFrame,
                 variable_dimensions: list[str],
                 target_dimensions: list[str]|None = None,
                 fixed_dimensions: list[str]|None = None,
                 relationship_context: dict[str, list[str]]|None = None,
                 forbidden_combinations: list[tuple[str, ...]]|None = None,
                 dimension_groups: dict[str, list[str]]|None = None,
                 verbose: bool = True):
        """
        Initialize the Variable Combination Manager.

        Parameters
        ----------
        training_data : pd.DataFrame
            Training dataset
        variable_dimensions : list of str
            Available predictor dimensions
        target_dimensions : list of str, optional
            Target dimensions (defaults to variable_dimensions)
        fixed_dimensions : list of str, optional
            Dimensions that must be in every combination (e.g., ['DWT'])
        relationship_context : dict, optional
            Exclusion rules: {dimension: [excluded_dimensions]}
        forbidden_combinations : list of tuples, optional
            Specific combinations that should never be generated
        dimension_groups : dict, optional
            Logical groupings of dimensions for classification
            Example: {'hull': ['LOA (M)', 'BEAM (M)'], 'tonnage': ['GT', 'DWT']}
        verbose : bool
            Print informational messages
        """
        self.verbose = verbose
        self.training_data = training_data

        # Apply defaults
        parameters = self._apply_defaults(
            variable_dimensions=variable_dimensions,
            target_dimensions=target_dimensions,
            fixed_dimensions=fixed_dimensions,
            relationship_context=relationship_context,
            forbidden_combinations=forbidden_combinations,
            dimension_groups=dimension_groups
        )

        # Validate fixed dimensions before filtering
        self._validate_fixed_dimensions(
            parameters['fixed_dimensions'],
            parameters['variable_dimensions']
        )

        # Validate and filter dimensions against training data
        self._setup_dimensions(
            variable_dimensions=parameters['variable_dimensions'],
            target_dimensions=parameters['target_dimensions'],
            fixed_dimensions=parameters['fixed_dimensions']
        )

        # Setup validation rules
        self._setup_validation_rules(
            relationship_context=parameters['relationship_context'],
            forbidden_combinations=parameters['forbidden_combinations']
        )

        # Setup dimension grouping
        self._setup_dimension_groups(parameters['dimension_groups'])

    def _apply_defaults(self,
                        variable_dimensions: List[str],
                        target_dimensions: Optional[List[str]],
                        fixed_dimensions: Optional[List[str]],
                        relationship_context: Optional[Dict[str, List[str]]],
                        forbidden_combinations: Optional[List[Tuple[str, ...]]],
                        dimension_groups: Optional[Dict[str, List[str]]]) -> Dict:
        """
        Apply default values to optional parameters.

        Returns
        -------
        dict
            Dictionary with all parameters filled in with defaults where needed
        """
        return {
            'variable_dimensions': variable_dimensions,
            'target_dimensions': target_dimensions if target_dimensions is not None else variable_dimensions,
            'fixed_dimensions': fixed_dimensions if fixed_dimensions is not None else [],
            'relationship_context': relationship_context if relationship_context is not None
            else self._get_default_relationship_context(),
            'forbidden_combinations': forbidden_combinations if forbidden_combinations is not None
            else self._get_default_forbidden_combinations(),
            'dimension_groups': dimension_groups if dimension_groups is not None
            else self._get_default_target_groups()
        }

    def _validate_fixed_dimensions(self,
                                   fixed_dimensions: List[str],
                                   variable_dimensions: List[str]) -> None:
        """
        Validate that fixed dimensions exist in variable dimensions.

        Parameters
        ----------
        fixed_dimensions : list of str
            Dimensions that must be in every combination
        variable_dimensions : list of str
            Available predictor dimensions

        Raises
        ------
        ValueError
            If any fixed dimension is not in variable_dimensions
        """
        invalid = [dim for dim in fixed_dimensions if dim not in variable_dimensions]
        if invalid:
            raise ValueError(
                f"Fixed dimensions not in variable_dimensions: {invalid}"
            )

    def _setup_dimensions(self,
                          variable_dimensions: List[str],
                          target_dimensions: List[str],
                          fixed_dimensions: List[str]) -> None:
        """
        Validate and filter dimensions against training data.

        Sets the following instance attributes:
        - available_dimensions
        - variable_dimensions
        - target_dimensions
        - fixed_dimensions

        Parameters
        ----------
        variable_dimensions : list of str
            Available predictor dimensions
        target_dimensions : list of str
            Target dimensions
        fixed_dimensions : list of str
            Dimensions that must be in every combination
        """
        # Identify all unique dimensions
        all_dimensions = list(set(variable_dimensions + target_dimensions))

        # Filter to only those present in training data
        self.available_dimensions = [
            dim for dim in all_dimensions
            if dim in self.training_data.columns
        ]

        # Report any missing dimensions
        missing_dimensions = [
            dim for dim in all_dimensions
            if dim not in self.training_data.columns
        ]

        if missing_dimensions:
            self._report_missing_dimensions(missing_dimensions)

        # Store filtered dimension lists
        self.variable_dimensions = [
            dim for dim in variable_dimensions
            if dim in self.available_dimensions
        ]
        self.target_dimensions = [
            dim for dim in target_dimensions
            if dim in self.available_dimensions
        ]
        self.fixed_dimensions = [
            dim for dim in fixed_dimensions
            if dim in self.available_dimensions
        ]

    def _report_missing_dimensions(self, missing_dimensions: List[str]) -> None:
        """
        Print warning about dimensions not found in training data.

        Parameters
        ----------
        missing_dimensions : list of str
            Dimensions not present in training data
        """
        if not self.verbose:
            return

        print(f"\n  ⚠️  Warning: Some dimensions not found in training_data:")
        for dim in missing_dimensions:
            print(f"    - '{dim}'")
        print(f"  Using {len(self.available_dimensions)} available dimensions\n")

    def _setup_validation_rules(self,
                                relationship_context: Dict[str, List[str]],
                                forbidden_combinations: List[Tuple[str, ...]]) -> None:
        """
        Setup validation rules and filter to available dimensions.

        Sets the following instance attributes:
        - relationship_context
        - forbidden_combinations

        Parameters
        ----------
        relationship_context : dict
            Exclusion rules: {dimension: [excluded_dimensions]}
        forbidden_combinations : list of tuples
            Specific combinations that should never be generated
        """
        self.relationship_context = relationship_context

        # Filter forbidden combinations to only include available dimensions
        self.forbidden_combinations = [
            tuple(dim for dim in combo if dim in self.available_dimensions)
            for combo in forbidden_combinations
        ]

        # Validate relationship context
        if self.relationship_context:
            self._validate_relationship_context()

    def _setup_dimension_groups(self, dimension_groups: Dict[str, List[str]]) -> None:
        """
        Setup dimension grouping structures.

        Sets the following instance attributes:
        - dimension_groups
        - dim_to_group (via _build_dimension_group_mapping)

        Parameters
        ----------
        dimension_groups : dict
            Logical groupings of dimensions for classification
        """
        self.dimension_groups = dimension_groups
        self._build_dimension_group_mapping()

    def _get_default_relationship_context(self) -> dict[str, List[str]]:
        """Get default relationship exclusion rules."""
        return {
            'LOA (M)': ['LOA (M) / DWT', 'Length/Beam', 'C_B', 'LBP (M)','LBD','LBT'],
            'LBP (M)': ['LOA (M) / DWT', 'Length/Beam', 'C_B', 'LOA (M)','LBD','LBT'],
            'BEAM (M)': ['Length/Beam', 'Beam/Draft', 'C_B','LBD','LBT'],
            'DRAFT (M)': ['Beam/Draft', 'C_B','LBD','LBT'],
            'C_B': ['Length/Beam', 'Beam/Draft', 'LOA (M) / DWT',
                    'SPEED (KN)', 'DRAFT (M)', 'BEAM (M)', 'LOA (M)', 'DWT'],
            'Length/Beam': ['LOA (M) / DWT', 'C_B', 'LBP (M)', 'LOA (M)'],
            'Beam/Draft': ['BEAM (M)', 'DRAFT (M)'],
        'LBD':['LBP (M)','LOA (M)','BEAM (M)','DEPTH (M)'],
            'LBT': ['LBP (M)', 'LOA (M)', 'BEAM (M)', 'DRAFT (M)'],
        }

    def _get_default_forbidden_combinations(self) -> List[Tuple[str, ...]]:
        """Get default forbidden combinations."""
        return [('LOA (M)', 'LBP (M)'), ]

    def _get_default_target_groups(self) -> dict[str, List[str]]:
        """
        Helper method to group target dimensions for classification.

        Returns
        -------
        dict
            Dictionary mapping group names to lists of dimensions
        """
        return {
            'hull': ['LOA (M)', 'LBP (M)', 'DRAFT (M)', 'BEAM (M)', 'DEPTH (M)'],
            'tonnage': ['GT', 'DWT'],
            'propulsion': ['SPEED (KN)', 'TOTAL MAIN POWER (KW)',
                           'TOTAL AUX POWER (KW)', 'MAIN BUNKER CAPACITY (M3)']
        }

    def _build_dimension_group_mapping(self):
        """Build reverse mapping from dimension to group name."""
        self.dim_to_group = {}

        for group_name, dims in self.dimension_groups.items():
            for dim in dims:
                self.dim_to_group[dim] = group_name

    def _validate_relationship_context(self):
        """Validate that relationship context dimensions exist."""
        invalid_keys = []
        invalid_values = []

        for key, values in self.relationship_context.items():
            if key not in self.variable_dimensions:
                invalid_keys.append(key)

            for val in values:
                if val not in self.variable_dimensions and val not in invalid_values:
                    invalid_values.append(val)

        if (invalid_keys or invalid_values) and self.verbose:
            print("\n⚠️  Relationship context validation:")
            if invalid_keys:
                print(f"  Keys not in variable_dimensions: {invalid_keys}")
            if invalid_values:
                print(f"  Values not in variable_dimensions: {invalid_values}")
            print()

    def get_available_dimensions(self) -> List[str]:
        """Get list of validated available dimensions."""
        return self.available_dimensions.copy()

    def get_variable_dimensions(self) -> List[str]:
        """Get list of validated variable dimensions."""
        return self.variable_dimensions.copy()

    def get_target_dimensions(self) -> List[str]:
        """Get list of validated target dimensions."""
        return self.target_dimensions.copy()

    def get_dimension_group(self, dimension: str) -> Optional[str]:
        """
        Get the group name for a dimension.

        Parameters
        ----------
        dimension : str
            Dimension name

        Returns
        -------
        str or None
            Group name if dimension belongs to a group, None otherwise
        """
        return self.dim_to_group.get(dimension)

    def classify_combination_type(self, predictors: Tuple[str, ...]) -> str:
        """
        Classify predictor combination by dimension category.

        Parameters
        ----------
        predictors : tuple of str
            Predictor dimension names

        Returns
        -------
        str
            Classification: group name if all from same group, 'mixed' if from
            multiple groups, 'other' if no group classification
        """
        if not self.dimension_groups:
            return 'other'

        # Get groups for each predictor
        groups = set()
        for pred in predictors:
            group = self.dim_to_group.get(pred)
            if group:
                groups.add(group)

        if len(groups) == 0:
            return 'other'
        elif len(groups) == 1:
            return groups.pop()
        else:
            return 'mixed'

    def is_forbidden_combination(self, predictors: Tuple[str, ...]) -> bool:
        """
        Check if a combination is explicitly forbidden.

        Parameters
        ----------
        predictors : tuple of str
            Predictor combination to check

        Returns
        -------
        bool
            True if combination is forbidden
        """
        pred_set = set(predictors)

        for forbidden in self.forbidden_combinations:
            if set(forbidden).issubset(pred_set):
                return True

        return False


    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIF for each predictor in the design matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix with predictor columns

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'feature', 'vif'
        """
        if X.shape[1] < 2:
            # VIF requires at least 2 features
            return pd.DataFrame({'feature': X.columns, 'vif': [1.0] * X.shape[1]})

        # Handle missing values
        X_clean = X.dropna()

        if len(X_clean) < X.shape[1] + 1:
            # Not enough samples for VIF calculation
            return pd.DataFrame({'feature': X.columns, 'vif': [np.nan] * X.shape[1]})

        # Calculate VIF for each feature
        vif_data = []
        for i, col in enumerate(X_clean.columns):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_data.append({'feature': col, 'vif': vif})
            except Exception as e:
                vif_data.append({'feature': col, 'vif': np.nan})

        return pd.DataFrame(vif_data)

    def _check_vif_combination(self,
                               predictors: Tuple[str, ...],
                               vif_threshold: float = 10.0) -> Tuple[bool, Dict]:
        """
        Check if a predictor combination has acceptable VIF values.

        Parameters
        ----------
        predictors : tuple of str
            Predictor variable names
        vif_threshold : float, default=10.0
            Maximum acceptable VIF value

        Returns
        -------
        is_valid : bool
            True if all VIF values are below threshold
        vif_info : dict
            Dictionary containing VIF values and diagnostic info
        """
        if len(predictors) < 2:
            # Single predictor has VIF = 1
            return True, {'max_vif': 1.0, 'predictors': predictors}

        # Prepare design matrix
        X = self.training_data[list(predictors)].copy()

        # Calculate VIF
        vif_df = self.calculate_vif(X)

        # Check for problematic VIF values
        max_vif = vif_df['vif'].max()
        problematic_features = vif_df[vif_df['vif'] > vif_threshold]

        is_valid = max_vif <= vif_threshold

        vif_info = {
            'max_vif': max_vif,
            'vif_df': vif_df,
            'problematic_features': problematic_features.to_dict('records') if not problematic_features.empty else [],
            'predictors': predictors
        }

        return is_valid, vif_info

    def is_valid_combination(self,
                             target: str,
                             predictors: Tuple[str, ...]) -> bool:
        """
        Check if a predictor combination is valid for a target.

        Checks:
        1. Target not in predictors
        2. No forbidden combinations
        3. No relationship context violations

        Parameters
        ----------
        target : str
            Target dimension
        predictors : tuple of str
            Predictor dimensions

        Returns
        -------
        bool
            True if combination is valid
        """
        # Check basic validity
        if target in predictors:
            return False

        # Check forbidden combinations
        if self.is_forbidden_combination(predictors):
            return False


        return True

    def generate_combinations(self,
                              target: str,
                              min_predictors: int = 1,
                              max_predictors: int = 3,
                              exclude_dimensions: List[str]|None = None,
                              filter_by_vif:bool=True,
                              vif_threshold:float = 10.0) -> List[Tuple[str, ...]]:
        """
        Generate valid predictor combinations for a target.

        Parameters
        ----------
        target : str
            Target dimension
        min_predictors : int
            Minimum number of predictors (including fixed)
        max_predictors : int
            Maximum number of predictors (including fixed)
        exclude_dimensions : list of str, optional
            Additional dimensions to exclude as predictors

        Returns
        -------
        list of tuples
            Valid predictor combinations
        """
        if exclude_dimensions is None:
            exclude_dimensions = []

        # Build pool of variable dimensions
        exclude_set = set([target] + exclude_dimensions)

        variable_pool = [d for d in self.variable_dimensions if d not in exclude_set]

        # Remove fixed dimensions from variable pool (they'll be added separately)
        fixed_dimensions = self.fixed_dimensions

        #remove relationship context dimensions from variable pool

        if target in self.relationship_context.keys():
            relationship_context = self.relationship_context[target]
        else:
            relationship_context = []

        discount_dimensions = set(fixed_dimensions + relationship_context)

        variable_pool = [d for d in variable_pool if d not in discount_dimensions]

        valid_combinations = []

        # Generate combinations of varying sizes
        for size in range(min_predictors, max_predictors + 1):
            # Calculate how many variable dimensions needed (excluding fixed)
            n_variable = size - len(self.fixed_dimensions)

            if n_variable < 0:
                # More fixed dimensions than allowed predictors
                continue
            elif n_variable == 0:
                # Only fixed dimensions
                combo = tuple(sorted(self.fixed_dimensions))
                if self.is_valid_combination(target, combo):
                    valid_combinations.append(combo)
            else:
                # Generate combinations of variable dimensions
                for var_combo in combinations(variable_pool, n_variable):
                    # Combine with fixed dimensions
                    full_combo = tuple(sorted(list(self.fixed_dimensions) + list(var_combo)))

                    if self.is_valid_combination(target, full_combo):
                        valid_combinations.append(full_combo)

        if filter_by_vif:
            filtered_combinations = []
            for combo in valid_combinations:
                is_valid,vif_info = self._check_vif_combination(predictors=combo,vif_threshold= vif_threshold)
                if is_valid:
                    filtered_combinations.append(combo)
            return filtered_combinations

        return valid_combinations


    def get_combination_summary(self) -> Dict[str, int]:
        """
        Get summary statistics about combination generation.

        Returns
        -------
        dict
            Summary with keys: 'targets', 'variables', 'fixed', 'forbidden_rules',
            'relationship_rules', 'dimension_groups'
        """
        return {
            'targets': len(self.target_dimensions),
            'variables': len(self.variable_dimensions),
            'fixed': len(self.fixed_dimensions),
            'forbidden_rules': len(self.forbidden_combinations),
            'relationship_rules': len(self.relationship_context),
            'dimension_groups': len(self.dimension_groups)
        }

    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "=" * 80)
        print("VARIABLE COMBINATION MANAGER - CONFIGURATION")
        print("=" * 80)

        summary = self.get_combination_summary()

        print(f"\nDimensions:")
        print(f"  • Total available: {len(self.available_dimensions)}")
        print(f"  • Variables: {summary['variables']}")
        print(f"  • Targets: {summary['targets']}")
        print(f"  • Fixed (always included): {summary['fixed']}")
        if self.fixed_dimensions:
            print(f"    → {', '.join(self.fixed_dimensions)}")

        print(f"\nValidation Rules:")
        print(f"  • Forbidden combinations: {summary['forbidden_rules']}")
        print(f"  • Relationship exclusions: {summary['relationship_rules']}")

        if self.dimension_groups:
            print(f"\nDimension Groups:")
            print(f"  • Total groups: {summary['dimension_groups']}")
            for group_name, dims in self.dimension_groups.items():
                available = [d for d in dims if d in self.available_dimensions]
                print(f"    → {group_name}: {len(available)} dimensions")

        print("=" * 80 + "\n")