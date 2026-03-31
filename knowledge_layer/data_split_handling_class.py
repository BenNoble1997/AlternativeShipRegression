"""
Data Split and Cross-Validation Handler
========================================

Centralizes data splitting and CV configuration logic for regression model selectors.

Key Design Principles:
- Stateless utility methods (CV config depends on specific data being fitted)
- Reusable across different selector classes
- Consistent CV behavior across all models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from typing import Tuple, Dict, Optional


class DataSplitHandler:
    """
    Handles data splitting and cross-validation configuration for regression models.

    This class provides utility methods for:
    - Train/test splitting
    - Cross-validation setup (standard, stratified, grouped)
    - Data preparation and cleaning
    - Input validation

    Design Note:
    -----------
    This is a utility class with static/class methods. CV configuration is created
    on-demand during fitting, not at initialization, because each predictor
    combination requires its own CV setup based on its specific data.
    """

    @staticmethod
    def create_train_test_split(
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split from dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            Full dataset to split
        test_size : float
            Proportion for test set (default: 0.2)
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        tuple
            (train_data, test_data) as DataFrames

        Example
        -------
        >>> train_df, test_df = DataSplitHandler.create_train_test_split(
        ...     dataset, test_size=0.2
        ... )
        """
        train_data, test_data = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state
        )
        return train_data, test_data

    @staticmethod
    def validate_cv_inputs(cv_method: str, groups: np.ndarray|None) -> None:
        """
        Validate cross-validation inputs.

        Parameters
        ----------
        cv_method : str
            CV method: 'standard', 'stratified', 'grouped', or None
        groups : array-like, optional
            Group labels (required for grouped CV)

        Raises
        ------
        ValueError
            If cv_method='grouped' but groups is None
            If cv_method is not recognized

        Example
        -------
        >>> DataSplitHandler.validate_cv_inputs('grouped', groups=None)
        ValueError: cv_method='grouped' requires 'groups' parameter
        """
        valid_methods = ['standard', 'stratified', 'grouped', None]

        if cv_method not in valid_methods:
            raise ValueError(
                f"Invalid cv_method: {cv_method}. "
                f"Must be one of {valid_methods}"
            )

        if cv_method == 'grouped' and groups is None:
            raise ValueError(
                "cv_method='grouped' requires 'groups' parameter. "
                "Use cv_method='stratified' if you don't have natural groups."
            )

    @staticmethod
    def prepare_combination_data(
        data: pd.DataFrame,
        pred_combo: Tuple[str, ...],
        target: str,
        cv_method: str,
        groups: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare and clean data for a specific predictor-target combination.

        This method:
        1. Extracts X (predictors) and y (target) from data
        2. Reshapes if needed (ensure X is 2D)
        3. Removes rows with NaN values
        4. Masks groups array if using grouped CV

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        pred_combo : tuple of str
            Predictor column names
        target : str
            Target column name
        cv_method : str
            CV method (needed to determine if groups should be masked)
        groups : array-like, optional
            Group labels for grouped CV

        Returns
        -------
        tuple
            (X_clean, y_clean, groups_clean)
            - X_clean: 2D array of predictors with NaNs removed
            - y_clean: 1D array of target with NaNs removed
            - groups_clean: Masked groups array (None if not using grouped CV)

        Raises
        ------
        ValueError
            If no valid data remains after cleaning
            If predictor combination is empty

        Example
        -------
        >>> X, y, grps = DataSplitHandler.prepare_combination_data(
        ...     data=train_df,
        ...     pred_combo=('DWT', 'LOA'),
        ...     target='Speed',
        ...     cv_method='stratified'
        ... )
        """
        # Extract data
        X = data[list(pred_combo)].values
        y = data[target].values

        # Validate predictor combo
        if X.shape[1] == 0:
            raise ValueError("Predictor combination is empty")

        # Handle y dimensions
        if len(y.shape) > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError(f"Target has {y.shape[1]} columns, expected 1")

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Create mask for valid (non-NaN) rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))

        # Apply mask
        X_clean = X[mask]
        y_clean = y[mask]

        # Validate cleaned data
        if len(X_clean) == 0 or X_clean.shape[1] == 0:
            raise ValueError("No valid data after removing NaN values")

        if len(y_clean.shape) > 1 and y_clean.shape[1] > 1:
            raise ValueError("Target is multidimensional after cleaning")

        # Mask groups if using grouped CV
        groups_clean = None
        if cv_method == 'grouped' and groups is not None:
            groups_clean = groups[mask]

        return X_clean, y_clean, groups_clean

    @staticmethod
    def setup_cv_config(
        X: np.ndarray,
        y: np.ndarray,
        cv_method: str|None,
        n_folds: int = 10,
        groups: np.ndarray|None = None,
        random_state: int = 42
    ) -> Dict:
        """
        Setup cross-validation configuration for a specific dataset.

        This method creates a configuration dictionary containing:
        - The CV splitter object (KFold, StratifiedKFold, GroupKFold)
        - Arguments for the split() method
        - Keyword arguments for the split() method

        Why this is method-based rather than stored:
        ---------------------------------------------
        Each predictor combination needs its own CV config because:
        1. Stratified CV uses X[:, 0] to create bins (different for each combo)
        2. After cleaning NaNs, the groups array has different indices
        3. Different X matrices may have different valid sample counts

        Parameters
        ----------
        X : np.ndarray
            Predictor matrix (already cleaned)
        y : np.ndarray
            Target array (already cleaned)
        cv_method : str or None
            CV method: 'standard', 'stratified', 'grouped', or None
        n_folds : int
            Number of CV folds
        groups : array-like, optional
            Group labels (required for grouped CV)
        random_state : int
            Random seed

        Returns
        -------
        dict
            Configuration with keys:
            - 'method': CV method name
            - 'splitter': scikit-learn splitter object (or None)
            - 'split_args': tuple of arguments for splitter.split()
            - 'split_kwargs': dict of keyword arguments for splitter.split()

        Example
        -------
        >>> config = DataSplitHandler.setup_cv_config(
        ...     X, y, cv_method='stratified', n_folds=10
        ... )
        >>>
        >>> # Use in fitting loop:
        >>> for train_idx, test_idx in config['splitter'].split(
        ...     *config['split_args'], **config['split_kwargs']
        ... ):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     # ... fit model ...
        """
        config = {
            'method': cv_method,
            'splitter': None,
            'split_args': None,
            'split_kwargs': {}
        }

        # No cross-validation
        if cv_method is None:
            return config

        # Standard K-Fold CV
        elif cv_method == 'standard':
            config['splitter'] = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state
            )
            config['split_args'] = (X, y)

        # Stratified K-Fold CV
        elif cv_method == 'stratified':
            # Use first predictor column for stratification
            stratify_values = X[:, 0]

            try:
                # Create bins for stratification
                # Use min(5, n_samples // 20) bins to ensure enough samples per bin
                y_bins = pd.qcut(
                    stratify_values,
                    q=min(5, len(y) // 20),
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If binning fails (e.g., too few unique values), fall back to standard CV
                if hasattr(X, '__name__'):  # Check if we can identify the feature
                    print(f"Warning: Stratified CV failed, falling back to standard CV")
                return DataSplitHandler.setup_cv_config(
                    X, y, 'standard', n_folds, groups, random_state
                )

            config['splitter'] = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state
            )
            config['split_args'] = (X, y_bins)

        # Grouped K-Fold CV
        elif cv_method == 'grouped':
            if groups is None:
                raise ValueError("Grouped CV requires 'groups' parameter")

            if len(groups) != len(X):
                raise ValueError(
                    f"Length of groups ({len(groups)}) must match data ({len(X)})"
                )

            # Check number of unique groups
            unique_groups = np.unique(groups)
            n_unique_groups = len(unique_groups)

            # Adjust n_folds if we have fewer groups than folds
            if n_unique_groups < n_folds:
                adjusted_folds = n_unique_groups
                if hasattr(X, '__name__'):
                    print(f"Warning: Only {n_unique_groups} unique groups, "
                          f"using {adjusted_folds} folds instead of {n_folds}")
                n_folds = adjusted_folds

            config['splitter'] = GroupKFold(n_splits=n_folds)
            config['split_args'] = (X, y)
            config['split_kwargs'] = {'groups': groups}

        else:
            raise ValueError(
                f"Invalid cv_method: {cv_method}. "
                f"Must be 'standard', 'stratified', 'grouped', or None"
            )

        return config

    @staticmethod
    def create_stratified_train_test_split(
            X: np.ndarray,
            y: np.ndarray,
            stratify_col_index: int = 0,
            test_size: float = 0.2,
            random_state: int = 42,
            n_bins: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/test split based on a specific column.

        This is useful for domain analyzers that need to split conventional data
        while maintaining similar distributions of a key variable (e.g., DWT).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (already cleaned)
        y : np.ndarray
            Target array (already cleaned)
        stratify_col_index : int
            Column index in X to use for stratification (default: 0)
        test_size : float
            Fraction for test split
        random_state : int
            Random seed
        n_bins : int
            Number of bins for stratification (default: 5)

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)

        Example
        -------
        >>> # Split conventional data stratified by DWT (first column)
        >>> X_train, X_test, y_train, y_test = DataSplitHandler.create_stratified_train_test_split(
        ...     X_clean, y_clean, stratify_col_index=0, test_size=0.2
        ... )
        """
        from sklearn.model_selection import train_test_split

        # Get stratification values
        stratify_values = X[:, stratify_col_index]

        try:
            # Create bins for stratification
            stratify_bins = pd.qcut(
                stratify_values,
                q=min(n_bins, len(stratify_values) // 20),
                labels=False,
                duplicates='drop'
            )
        except (ValueError, TypeError) as e:
            # Fallback: use random split if binning fails
            stratify_bins = None

        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_bins,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_cv_description(cv_method: Optional[str], n_folds: int = 10) -> str:
        """
        Generate human-readable description of CV method.

        Parameters
        ----------
        cv_method : str or None
            CV method name
        n_folds : int
            Number of folds (for display purposes)

        Returns
        -------
        str
            Description text

        Example
        -------
        >>> desc = DataSplitHandler.get_cv_description('stratified', 10)
        >>> print(desc)
        '(10-fold Stratified K-Fold CV)'
        """
        if cv_method is None:
            return "(No CV - Training Set Only)"
        elif cv_method == 'standard':
            return f"({n_folds}-fold K-Fold CV)"
        elif cv_method == 'stratified':
            return f"({n_folds}-fold Stratified K-Fold CV)"
        elif cv_method == 'grouped':
            return f"({n_folds}-fold Grouped K-Fold CV)"
        else:
            return f"({cv_method})"