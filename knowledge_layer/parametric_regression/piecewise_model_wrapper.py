"""
Piecewise Model Wrapper
=======================
Internal class for handling piecewise regression models.

A piecewise model fits separate models to different ranges (bins) of a predictor
variable, then combines predictions. Common in naval architecture where
relationships change at different ship sizes.

Example:
    LOA prediction from DWT might use:
    - Linear model for DWT < 5000
    - Power model for 5000 ≤ DWT < 20000
    - Different power model for DWT ≥ 20000
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings


class _PiecewiseModelWrapper:
    """
    Internal wrapper class for piecewise regression models.

    This class handles:
    - Binning data by a predictor variable
    - Fitting separate models to each bin
    - Combining predictions across bins
    - Calculating diagnostic metrics

    **Not meant to be used directly** - use via ModelFactory.create_and_fit()
    with model_type ending in '-piecewise'.

    Attributes
    ----------
    base_model_type : str
        Type of model to fit in each bin (e.g., 'power', 'linear')
    bins : dict
        Binning specification: {variable_name: [bin_edges]}
    binning_variable : str
        Name of variable to bin on
    bin_models : dict
        Maps bin_index → (model, preprocessor) for each bin
    bin_edges : np.ndarray
        Array of bin edge values
    bin_labels : list
        Human-readable labels for each bin
    min_samples_per_bin : int
        Minimum samples required per bin

    Concepts Explained
    ------------------

    **Binning:**
    Data is partitioned into ranges. For bins [0, 1000, 5000, inf]:
    - Bin 0: 0 ≤ x < 1000
    - Bin 1: 1000 ≤ x < 5000
    - Bin 2: 5000 ≤ x < inf

    **Separate Fitting:**
    Each bin gets its own model with its own parameters.
    This allows the relationship to change character at different scales.

    **Combined Prediction:**
    When predicting, determine which bin each sample belongs to,
    then use that bin's model. Result is a single y_pred array.

    **Discontinuity:**
    Predictions may "jump" at bin boundaries since adjacent bins
    use different models. This is reported in diagnostics.
    """

    def __init__(self,
                 base_model_type: str,
                 bins: Dict[str, List[float]],
                 binning_variable: str,
                 min_samples_per_bin: int = 10,
                 verbose: bool = False):
        """
        Initialize piecewise model wrapper.

        Parameters
        ----------
        base_model_type : str
            Model type to fit in each bin (e.g., 'power', 'linear', 'exponential')
        bins : dict
            Dictionary mapping variable name to list of bin edges
            Example: {'DWT': [0, 1000, 5000, 10000, np.inf]}
        binning_variable : str
            Name of the variable to bin on (must be key in bins dict)
        min_samples_per_bin : int, default=10
            Minimum samples required per bin (bins with fewer samples skipped)
        verbose : bool, default=False
            Whether to print fitting progress and warnings
        """
        self.base_model_type = base_model_type
        self.bins = bins
        self.binning_variable = binning_variable
        self.min_samples_per_bin = min_samples_per_bin
        self.verbose = verbose

        # Storage for fitted models
        self.bin_models = {}  # {bin_index: (model, preprocessor)}
        self.bin_edges = None
        self.bin_labels = []
        self.bin_sample_counts = {}

        # Metadata
        self.n_bins_total = 0
        self.n_bins_fitted = 0
        self.binning_variable_index = None
        self.skipped_bins = []

        # Validation
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate initialization parameters."""
        if self.binning_variable not in self.bins:
            raise ValueError(
                f"binning_variable '{self.binning_variable}' not found in bins dict. "
                f"Available: {list(self.bins.keys())}"
            )

        edges = self.bins[self.binning_variable]
        if len(edges) < 2:
            raise ValueError(
                f"bins must have at least 2 edges (got {len(edges)}). "
                f"Example: [0, 1000, 5000, np.inf]"
            )

        # Check monotonically increasing
        if not all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(
                f"bin edges must be monotonically increasing. Got: {edges}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit piecewise model by fitting base model to each bin.

        Process:
        1. Determine which bin each sample belongs to
        2. For each bin with sufficient samples:
           - Extract samples in that bin
           - Fit base_model_type to those samples
           - Store the fitted model
        3. Record diagnostics (bin sizes, skipped bins, etc.)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Predictor variables
        y : np.ndarray, shape (n_samples,)
            Target variable

        Raises
        ------
        ValueError
            If binning_variable not found in X, or if no bins have sufficient data
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Determine which column to bin on
        self.binning_variable_index = self._get_binning_variable_index(n_features)

        # Get binning variable values
        binning_values = X[:, self.binning_variable_index]

        # Create bins using pandas cut (left-inclusive, right-exclusive)
        self.bin_edges = np.array(self.bins[self.binning_variable])
        self.n_bins_total = len(self.bin_edges) - 1

        # Assign each sample to a bin
        bin_indices = pd.cut(
            binning_values,
            bins=self.bin_edges,
            labels=False,
            include_lowest=True,
            right=False  # Left-inclusive: [a, b)
        )

        # Create bin labels for reporting
        self._create_bin_labels()

        # Fit model for each bin
        for bin_idx in range(self.n_bins_total):
            mask = (bin_indices == bin_idx)
            n_in_bin = np.sum(mask)

            self.bin_sample_counts[bin_idx] = n_in_bin

            if n_in_bin < self.min_samples_per_bin:
                if self.verbose:
                    print(f"  ⚠️  Bin {bin_idx} ({self.bin_labels[bin_idx]}): "
                          f"Only {n_in_bin} samples (min: {self.min_samples_per_bin}), skipping")
                self.skipped_bins.append(bin_idx)
                continue

            # Extract data for this bin
            X_bin = X[mask]
            y_bin = y[mask]

            # Fit model using ModelFactory
            try:
                model, preprocessor = ModelFactory.create_and_fit(
                    self.base_model_type,
                    X_bin,
                    y_bin
                )

                self.bin_models[bin_idx] = (model, preprocessor)
                self.n_bins_fitted += 1

                if self.verbose:
                    print(f"  ✓ Bin {bin_idx} ({self.bin_labels[bin_idx]}): "
                          f"Fitted {self.base_model_type} on {n_in_bin} samples")

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Bin {bin_idx} ({self.bin_labels[bin_idx]}): "
                          f"Fitting failed - {str(e)[:60]}")
                self.skipped_bins.append(bin_idx)

        # Validation: At least one bin must be fitted
        if self.n_bins_fitted == 0:
            raise ValueError(
                f"No bins could be fitted! Check:\n"
                f"  - Bin sample counts: {self.bin_sample_counts}\n"
                f"  - Minimum required: {self.min_samples_per_bin}\n"
                f"  - Consider adjusting bin edges or min_samples_per_bin"
            )

        if self.verbose:
            print(f"\n  Summary: Fitted {self.n_bins_fitted}/{self.n_bins_total} bins")
            if self.skipped_bins:
                print(f"  Skipped bins: {self.skipped_bins}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using piecewise model.

        For each sample:
        1. Determine which bin it belongs to
        2. Use that bin's model for prediction
        3. If bin was skipped, use nearest fitted bin

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Predictor variables

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predictions

        Raises
        ------
        ValueError
            If model not fitted yet (call fit() first)
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        if not self.bin_models:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)

        # Get binning variable values
        binning_values = X[:, self.binning_variable_index]

        # Assign samples to bins
        bin_indices = pd.cut(
            binning_values,
            bins=self.bin_edges,
            labels=False,
            include_lowest=True,
            right=False
        )

        # Initialize predictions array
        y_pred = np.zeros(n_samples)

        # Predict for each bin
        for bin_idx in range(self.n_bins_total):
            mask = (bin_indices == bin_idx)

            if not np.any(mask):
                continue  # No samples in this bin

            # Get model for this bin (or nearest fitted bin)
            model, preprocessor = self._get_model_for_bin(bin_idx)

            if model is None:
                # Should not happen if fit() succeeded, but handle gracefully
                warnings.warn(
                    f"No model available for bin {bin_idx}, using NaN predictions"
                )
                y_pred[mask] = np.nan
                continue

            # Make predictions
            X_bin = X[mask]
            y_pred[mask] = ModelFactory.predict(
                model, preprocessor, X_bin, self.base_model_type
            )

        return y_pred

    def _get_model_for_bin(self, bin_idx: int) -> Tuple[Optional[object], Optional[object]]:
        """
        Get model for a bin, falling back to nearest fitted bin if necessary.

        Parameters
        ----------
        bin_idx : int
            Target bin index

        Returns
        -------
        model : object or None
            Fitted model
        preprocessor : object or None
            Preprocessor (e.g., PolynomialFeatures)
        """
        # Check if this bin has a fitted model
        if bin_idx in self.bin_models:
            return self.bin_models[bin_idx]

        # Bin was skipped - find nearest fitted bin
        fitted_bins = sorted(self.bin_models.keys())

        if not fitted_bins:
            return None, None

        # Find nearest bin
        distances = [abs(bin_idx - fb) for fb in fitted_bins]
        nearest_bin = fitted_bins[np.argmin(distances)]

        if self.verbose:
            warnings.warn(
                f"Bin {bin_idx} was skipped during fitting. "
                f"Using model from bin {nearest_bin} instead."
            )

        return self.bin_models[nearest_bin]

    def _get_binning_variable_index(self, n_features: int) -> int:
        """
        Determine which column index corresponds to binning variable.

        For single predictor: index = 0 (obvious)
        For multiple predictors: currently raises error (not yet supported)

        Parameters
        ----------
        n_features : int
            Number of features in X

        Returns
        -------
        int
            Column index of binning variable
        """
        if n_features == 1:
            return 0
        else:
            # Multi-predictor piecewise: need to know which predictor is binning var
            # For now, raise error - can be extended later
            raise ValueError(
                f"Piecewise models currently only support single predictor. "
                f"Got {n_features} predictors. "
                f"Consider using max_piecewise_predictors=1 in selector."
            )

    def _create_bin_labels(self):
        """Create human-readable labels for bins."""
        self.bin_labels = []

        for i in range(self.n_bins_total):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            # Format bounds
            lower_str = f"{lower:.0f}" if not np.isinf(lower) else "-∞"
            upper_str = f"{upper:.0f}" if not np.isinf(upper) else "∞"

            label = f"[{lower_str}, {upper_str})"
            self.bin_labels.append(label)

    def get_n_parameters(self) -> int:
        """
        Calculate total number of parameters in piecewise model.

        This is important for AIC/BIC calculation. Each bin's model
        contributes its own parameters.

        Returns
        -------
        int
            Total number of parameters across all bins

        Examples
        --------
        >>> # Power model: 2 parameters (a, b) per bin
        >>> # 3 bins fitted → 2 * 3 = 6 total parameters
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        if not self.bin_models:
            return 0

        # Get parameter count for base model type
        # Use a dummy X to determine feature count
        dummy_X = np.array([[1.0]])  # Single predictor assumption
        base_n_params = ModelFactory.get_actual_n_features(
            self.base_model_type, dummy_X
        )

        # Total parameters = parameters_per_bin * number_of_fitted_bins
        total_params = base_n_params * self.n_bins_fitted

        return total_params

    def get_bin_diagnostics(self) -> Dict:
        """
        Get diagnostic information about bins.

        Returns
        -------
        dict
            Diagnostic information including:
            - bin_edges: Array of bin boundaries
            - bin_labels: Human-readable bin labels
            - bin_sample_counts: Samples per bin
            - n_bins_total: Total number of bins
            - n_bins_fitted: Number of successfully fitted bins
            - skipped_bins: Indices of bins that were skipped
            - discontinuities: Prediction jumps at boundaries (if calculated)
        """
        diagnostics = {
            'bin_edges': self.bin_edges,
            'bin_labels': self.bin_labels,
            'bin_sample_counts': self.bin_sample_counts,
            'n_bins_total': self.n_bins_total,
            'n_bins_fitted': self.n_bins_fitted,
            'skipped_bins': self.skipped_bins,
            'binning_variable': self.binning_variable,
            'base_model_type': self.base_model_type,
            'min_samples_per_bin': self.min_samples_per_bin
        }

        return diagnostics

    def calculate_boundary_discontinuities(self, X: np.ndarray = None) -> Dict[int, float]:
        """
        Calculate discontinuities (jumps) at bin boundaries.

        At each boundary between two fitted bins, predict using both
        adjacent models at the boundary point. The difference is the
        discontinuity.

        Parameters
        ----------
        X : np.ndarray, optional
            If provided, uses actual data near boundaries.
            If None, uses boundary values directly.

        Returns
        -------
        dict
            Maps boundary_index → discontinuity_value
            boundary_index refers to bin_edges[i]

        Notes
        -----
        Large discontinuities may indicate:
        - Very different relationships in adjacent bins (legitimate)
        - Insufficient data in one or both bins
        - Inappropriate binning scheme
        """
        from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory

        if not self.bin_models:
            raise ValueError("Model not fitted yet. Call fit() first.")

        discontinuities = {}

        # Check each interior boundary
        for i in range(1, len(self.bin_edges) - 1):
            bin_left = i - 1
            bin_right = i

            # Skip if either bin wasn't fitted
            if bin_left not in self.bin_models or bin_right not in self.bin_models:
                continue

            boundary_value = self.bin_edges[i]

            # Create test point at boundary
            # Assumes single predictor (multi-predictor would need full X)
            X_boundary = np.array([[boundary_value]])

            # Predict with left bin's model
            model_left, prep_left = self.bin_models[bin_left]
            pred_left = ModelFactory.predict(
                model_left, prep_left, X_boundary, self.base_model_type
            )[0]

            # Predict with right bin's model
            model_right, prep_right = self.bin_models[bin_right]
            pred_right = ModelFactory.predict(
                model_right, prep_right, X_boundary, self.base_model_type
            )[0]

            # Calculate discontinuity (absolute and relative)
            discontinuity = abs(pred_right - pred_left)
            mean_prediction = (pred_left + pred_right) / 2
            relative_discontinuity = (discontinuity / mean_prediction * 100
                                      if mean_prediction != 0 else np.inf)

            discontinuities[i] = {
                'boundary_value': boundary_value,
                'pred_left': pred_left,
                'pred_right': pred_right,
                'absolute_jump': discontinuity,
                'relative_jump_pct': relative_discontinuity,
                'bin_left': bin_left,
                'bin_right': bin_right
            }

        return discontinuities

    def __repr__(self):
        """String representation."""
        if not self.bin_models:
            status = "not fitted"
        else:
            status = f"{self.n_bins_fitted}/{self.n_bins_total} bins fitted"

        return (f"PiecewiseModel(base={self.base_model_type}, "
                f"bins={self.n_bins_total}, {status})")