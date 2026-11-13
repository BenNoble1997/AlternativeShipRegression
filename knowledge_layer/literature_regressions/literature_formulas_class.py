import numpy as np
from typing import List, Callable, Tuple
import pandas as pd
import ast
import re

# ============================================================================
# LITERATURE FORMULA DEFINITIONS
# ============================================================================
# ============================================================================
# LITERATURE FORMULA DEFINITIONS (UPDATED WITH PIECEWISE SUPPORT)
# ============================================================================

class LiteratureFormula:
    """
    Container for a literature-based regression formula.

    Attributes
    ----------
    name : str
        Formula identifier (e.g., "DWT_to_LBP_power")
    predictors : List[str]
        Required predictor variables
    target : str
        Target variable to predict
    formula_type : str
        Type: 'power', 'polynomial', 'logarithmic', 'piecewise_power', etc.
    parameters : dict
        Formula parameters (a, b, c, d, etc.)
    reference : str
        Literature citation/source
    bounds : float, optional
        Percentile bounds (e.g., 5 means formula valid for 5th-95th percentile)
    bins : List[float], optional
        For piecewise formulae: bin edges [b0, b1, b2, ..., bn]
    bin_parameters : List[dict], optional
        For piecewise formulae: parameters for each bin segment
    """

    def __init__(self,
                 name: str,
                 predictors: List[str],
                 target: str,
                 formula_type: str,
                 parameters: dict,
                 reference: str = "",
                 bounds: float|None = None,
                 bins: List[float]|None = None,
                 bin_parameters: List[dict] = None,
                 domain_lower: float|dict|None = None,
                 domain_upper: float|dict|None = None):

        self.name = name
        self.predictors = predictors
        self.target = target
        self.formula_type = formula_type
        self.parameters = parameters
        self.reference = reference
        self.bounds = bounds

        # Piecewise formula support
        self.bins = bins
        self.bin_parameters = bin_parameters

        # Explicit domain boundaries
        self.domain_lower = domain_lower
        self.domain_upper = domain_upper

        # If piecewise, automatically set domain from bins
        if self.bins is not None and len(self.bins) > 0:
            self.domain_lower = self.bins[0] if domain_lower is None else domain_lower
            if self.domain_lower is None:
                self.domain_lower = 0.0


            self.domain_upper = self.bins[-1] if domain_upper is None else domain_upper

        # Bounds - for multi-predictor, stores dict of limits per predictor
        self.lower_limits = {}
        self.upper_limits = {}
        self.bounds_calculated = False

        self.predict_func = self._build_prediction_function()

    @property
    def formula_id(self) -> str:
        """
        Unique key for joins and lookups.

        `name` is a display label and is not unique: two sources can publish the
        same functional form for the same predictor-target pair. Anything that
        merges, groups or looks up per-formula results must use this instead.
        """
        preds = '_'.join(p.replace(' ', '_') for p in self.predictors)
        ref = str(self.reference).replace(' ', '_') if self.reference else 'na'
        return f"{preds}_to_{self.target.replace(' ', '_')}_{self.formula_type}_ref{ref}"

    def _normalize_domain_limits(self):
        """
        Normalize domain_lower and domain_upper to dict format internally.

        Converts:
        - None → remains None
        - Single float → {predictor[0]: value} for single predictor
        - Dict → validates and keeps as-is

        This ensures consistent internal representation.
        """
        # Normalize domain_lower
        if self.domain_lower is not None:
            if isinstance(self.domain_lower, (int, float)):
                # Single value - convert to dict
                if len(self.predictors) == 1:
                    self.domain_lower = {self.predictors[0]: float(self.domain_lower)}
                else:
                    # Multi-predictor with single value - shouldn't happen, but handle it
                    print(f"  Warning: Single domain_lower value for multi-predictor formula {self.name}")
                    self.domain_lower = {pred: float(self.domain_lower) for pred in self.predictors}
            elif isinstance(self.domain_lower, dict):
                # Already a dict - validate keys
                for key in self.domain_lower.keys():
                    if key not in self.predictors:
                        print(f"  Warning: domain_lower key '{key}' not in predictors for {self.name}")
            else:
                print(f"  Warning: Unexpected domain_lower type {type(self.domain_lower)} for {self.name}")
                self.domain_lower = None

        # Normalize domain_upper (same logic)
        if self.domain_upper is not None:
            if isinstance(self.domain_upper, (int, float)):
                if len(self.predictors) == 1:
                    self.domain_upper = {self.predictors[0]: float(self.domain_upper)}
                else:
                    print(f"  Warning: Single domain_upper value for multi-predictor formula {self.name}")
                    self.domain_upper = {pred: float(self.domain_upper) for pred in self.predictors}
            elif isinstance(self.domain_upper, dict):
                for key in self.domain_upper.keys():
                    if key not in self.predictors:
                        print(f"  Warning: domain_upper key '{key}' not in predictors for {self.name}")
            else:
                print(f"  Warning: Unexpected domain_upper type {type(self.domain_upper)} for {self.name}")
                self.domain_upper = None

    def check_domain(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Check which predictions are within domain limits.

        Domain limits are the theoretical valid range of the formula
        (e.g., DWT ∈ [5000, 15000], Vs ∈ [20, 30]).

        Different from bounds (percentile-based from data).

        Parameters
        ----------
        X : np.ndarray
            Predictor values, shape (n_samples, n_predictors)

        Returns
        -------
        within_domain : np.ndarray
            Boolean array indicating which samples are within domain
        n_out_of_domain : int
            Count of samples outside domain
        """
        if self.domain_lower is None and self.domain_upper is None:
            # No domain limits specified
            return np.ones(len(X), dtype=bool), 0

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        within_domain = np.ones(len(X), dtype=bool)

        # Check each predictor
        for i, predictor in enumerate(self.predictors):
            x_col = X[:, i]

            # Check lower limit
            if self.domain_lower is not None and predictor in self.domain_lower:
                within_domain &= (x_col >= self.domain_lower[predictor])

            # Check upper limit
            if self.domain_upper is not None and predictor in self.domain_upper:
                within_domain &= (x_col <= self.domain_upper[predictor])

        n_out_of_domain = np.sum(~within_domain)
        return within_domain, n_out_of_domain

    def get_domain_info(self) -> str:
        """
        Get human-readable string describing the formula's domain.

        Returns
        -------
        str
            Domain description

        Examples
        --------
        Single predictor:
        >>> formula.get_domain_info()
        'DWT ∈ [5000, 15000]'

        Multi predictor:
        >>> formula.get_domain_info()
        'DWT ∈ [5000, 15000], Vs ∈ [20, 30]'

        No domain:
        >>> formula.get_domain_info()
        'No domain limits specified'
        """
        if self.domain_lower is None and self.domain_upper is None:
            return "No domain limits specified"

        parts = []
        for predictor in self.predictors:
            lower = self.domain_lower.get(predictor) if self.domain_lower else None
            upper = self.domain_upper.get(predictor) if self.domain_upper else None

            if lower is not None and upper is not None:
                parts.append(f"{predictor} ∈ [{lower:.0f}, {upper:.0f}]")
            elif lower is not None:
                parts.append(f"{predictor} ≥ {lower:.0f}")
            elif upper is not None:
                parts.append(f"{predictor} ≤ {upper:.0f}")

        return ", ".join(parts) if parts else "No domain limits specified"

    def filter_data_by_domain(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to only include ships within formula's domain.

        Parameters
        ----------
        data : pd.DataFrame
            Ship dataset

        Returns
        -------
        pd.DataFrame
            Filtered dataset
        """
        if self.domain_lower is None and self.domain_upper is None:
            return data.copy()

        filtered = data.copy()

        for predictor in self.predictors:
            if predictor not in filtered.columns:
                print(f"  Warning: Predictor '{predictor}' not in data")
                continue

            # Apply lower limit
            if self.domain_lower is not None and predictor in self.domain_lower:
                lower = self.domain_lower[predictor]
                before = len(filtered)
                filtered = filtered[filtered[predictor] >= lower]
                removed = before - len(filtered)
                if removed > 0:
                    print(f"  Removed {removed} ships with {predictor} < {lower}")

            # Apply upper limit
            if self.domain_upper is not None and predictor in self.domain_upper:
                upper = self.domain_upper[predictor]
                before = len(filtered)
                filtered = filtered[filtered[predictor] <= upper]
                removed = before - len(filtered)
                if removed > 0:
                    print(f"  Removed {removed} ships with {predictor} > {upper}")

        return filtered

    def calculate_bounds_from_data(self, data: pd.DataFrame):
        """Calculate bounds for all predictors."""
        if self.bounds is None:
            self.bounds_calculated = True
            return

        for predictor in self.predictors:
            if predictor not in data.columns:
                print(f"  Warning: Predictor '{predictor}' not found in data")
                continue

            predictor_data = data[predictor].dropna()

            if len(predictor_data) == 0:
                continue

            lower_percentile = self.bounds / 100
            upper_percentile = 1 - (self.bounds / 100)

            self.lower_limits[predictor] = predictor_data.quantile(lower_percentile)
            self.upper_limits[predictor] = predictor_data.quantile(upper_percentile)

        self.bounds_calculated = True

        if self.lower_limits:
            print(f"  Bounds for {self.name}:")
            for pred in self.predictors:
                if pred in self.lower_limits:
                    print(f"    {pred}: [{self.lower_limits[pred]:.2f}, {self.upper_limits[pred]:.2f}]")

    def check_bounds(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """Check which predictions are within bounds (all predictors)."""
        if not self.lower_limits:
            return np.ones(len(X), dtype=bool), 0

        within_bounds = np.ones(len(X), dtype=bool)

        for i, predictor in enumerate(self.predictors):
            if predictor in self.lower_limits:
                x_col = X[:, i] if len(X.shape) > 1 else X
                within_bounds &= (x_col >= self.lower_limits[predictor]) & \
                                 (x_col <= self.upper_limits[predictor])

        n_out_of_bounds = np.sum(~within_bounds)
        return within_bounds, n_out_of_bounds

    def _evaluate_term(self, term_str: str, X: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial term string using predictor data.

        Parameters
        ----------
        term_str : str
            Term expression like 'X1', 'X1^2', 'X1*X2^2', etc.
            Variables are referenced by predictor names or indices
        X : np.ndarray
            Predictor data (n_samples, n_predictors)

        Returns
        -------
        np.ndarray
            Evaluated term values (n_samples,)

        Examples
        --------
        'Vs' -> X[:, predictor_index['Vs']]
        'Vs^2' -> X[:, predictor_index['Vs']]**2
        'DWT*Vs' -> X[:, predictor_index['DWT']] * X[:, predictor_index['Vs']]
        'DWT*Vs^2' -> X[:, predictor_index['DWT']] * X[:, predictor_index['Vs']]**2
        """

        # Create mapping of predictor names to column indices
        pred_map = {pred: i for i, pred in enumerate(self.predictors)}

        # Start with ones
        result = np.ones(len(X))

        # Split by multiplication
        factors = term_str.split('*')

        for factor in factors:
            factor = factor.strip()

            # Check for power notation: 'Vs^2'
            if '^' in factor:
                var_name, power_str = factor.split('^')
                var_name = var_name.strip()
                power = float(power_str.strip())

                if var_name not in pred_map:
                    raise ValueError(f"Unknown predictor '{var_name}' in term '{term_str}'")

                var_data = X[:, pred_map[var_name]]
                result *= (var_data ** power)

            else:
                # Simple variable reference
                if factor not in pred_map:
                    raise ValueError(f"Unknown predictor '{factor}' in term '{term_str}'")

                var_data = X[:, pred_map[factor]]
                result *= var_data

        return result

    def _build_prediction_function(self) -> Callable:
        """Build prediction function based on formula type."""

        # NEW: Check for piecewise formula types first
        if self.formula_type in ['piecewise_power', 'piecewise_linear', 'piecewise_mixed']:
            return self._build_piecewise_function()

        # ===== SINGLE PREDICTOR FORMULAS (existing) =====
        if len(self.predictors) == 1:
            return self._build_single_predictor_function()

        # ===== MULTI PREDICTOR FORMULAS (existing) =====
        else:
            return self._build_multi_predictor_function()

    def _build_piecewise_function(self) -> Callable:
        """
        Build prediction function for piecewise formulae.

        Supports different functional forms in each segment:
        - Power law: a + b*x^c + d*x^e
        - Linear: a + c*x
        - Constant: a
        - With constraints: min, max, offset
        """

        if self.bins is None or self.bin_parameters is None:
            raise ValueError(f"Piecewise formula {self.name} missing bins or bin_parameters")

        if len(self.bins) != len(self.bin_parameters) + 1:
            raise ValueError(
                f"Piecewise formula {self.name}: bins must have 1 more element than bin_parameters. "
                f"Got {len(self.bins)} bins and {len(self.bin_parameters)} parameter sets."
            )

        def predict(X):
            """
            Predict using piecewise formula.

            For each data point, determines which bin it falls into,
            then applies the appropriate formula for that bin.
            """
            x = X[:, 0] if len(X.shape) > 1 else X
            y = np.zeros_like(x, dtype=float)

            # Track which points have been assigned
            assigned = np.zeros_like(x, dtype=bool)

            # Process each bin segment
            for i in range(len(self.bin_parameters)):
                lower = self.bins[i]
                upper = self.bins[i + 1]
                params = self.bin_parameters[i]

                # Determine mask for this bin
                if i == len(self.bin_parameters) - 1:
                    # Last bin: include upper boundary
                    mask = (x >= lower) & (x <= upper)
                else:
                    # Other bins: exclude upper boundary
                    mask = (x >= lower) & (x < upper)

                # Skip if no points in this bin
                if not np.any(mask):
                    continue

                # Apply appropriate formula based on available parameters
                # Use .get() with defaults to avoid KeyErrors

                # Handle special modifiers first
                if 'min' in params:
                    # Formula with min constraint: min(a + b*x^c + d*x^e, min_value)
                    a = params.get('a', 0.0)
                    b = params.get('b', 0.0)
                    c = params.get('c', 1.0)
                    d = params.get('d', 0.0)
                    e = params.get('e', 1.0)
                    min_val = params['min']

                    result = a + b * (x[mask] ** c)
                    if d != 0:
                        result += d * (x[mask] ** e)
                    y[mask] = np.minimum(result, min_val)

                elif 'max' in params:
                    # Formula with max constraint: max(a + b*x^c + d*x^e, max_value)
                    a = params.get('a', 0.0)
                    b = params.get('b', 0.0)
                    c = params.get('c', 1.0)
                    d = params.get('d', 0.0)
                    e = params.get('e', 1.0)
                    max_val = params['max']

                    result = a + b * (x[mask] ** c)
                    if d != 0:
                        result += d * (x[mask] ** e)
                    y[mask] = np.maximum(result, max_val)

                elif 'offset' in params:
                    # Formula with offset: a + b*(x + offset)^c
                    offset = params['offset']
                    a = params.get('a', 0.0)
                    b = params.get('b', 1.0)
                    c = params.get('c', 1.0)

                    y[mask] = a + b * ((x[mask] + offset) ** c)

                else:
                    # Standard formula: a + b*x^c + d*x^e
                    a = params.get('a', 0.0)
                    b = params.get('b', 0.0)
                    c = params.get('c', 1.0)
                    d = params.get('d', 0.0)
                    e = params.get('e', 1.0)

                    result = a + b * (x[mask] ** c)
                    if d != 0:
                        result += d * (x[mask] ** e)
                    y[mask] = result

                assigned |= mask

            # Warn if any points were not assigned (outside all bins)
            if not np.all(assigned):
                n_unassigned = np.sum(~assigned)
                unassigned_values = x[~assigned]
                print(f"  Warning: {n_unassigned} points outside bin range "
                      f"[{self.bins[0]}, {self.bins[-1]}] for {self.name}")
                if n_unassigned <= 5:
                    print(f"    Unassigned values: {unassigned_values}")
                else:
                    print(f"    Unassigned values: {unassigned_values[:5]}... (showing first 5)")

            return y

        return predict

    def _build_single_predictor_function(self) -> Callable:
        """Build single-predictor prediction functions (existing logic)."""

        if self.formula_type == 'power':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                return self.parameters['a'] + self.parameters['b'] * (x ** self.parameters['c'])

            return predict

        elif self.formula_type == 'power_min':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                return min((self.parameters['a'] + self.parameters['b'] * (x ** self.parameters['c'])),
                           self.parameters['d'])

            return predict

        elif self.formula_type == 'power_log':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                return self.parameters['a'] + self.parameters['b'] * np.log(x ** self.parameters['c'])

            return predict

        elif self.formula_type == 'log_power':
            # Pattern: y = a + b * ln^c(x)
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                # Add small epsilon to avoid log(0)
                x_safe = np.maximum(x, 1e-10)
                return self.parameters['a'] + self.parameters['b'] * (np.log(x_safe) ** self.parameters['c'])

            return predict

        elif self.formula_type == 'quadrilateral_polynomial':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                a, b, c = self.parameters['a'], self.parameters['b'], \
                    self.parameters['c']
                return a + b*(x) + c*(x**2)

            return predict

        elif self.formula_type == 'cubic_polynomial':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                a, b, c, d = self.parameters['a'], self.parameters['b'], \
                    self.parameters['c'], self.parameters['d']
                return a * (x ** 3) + b * (x ** 2) + c * x + d

            return predict

        elif self.formula_type == 'exponential':
            def predict(X):
                x = X[:, 0] if len(X.shape) > 1 else X
                return self.parameters['b'] + self.parameters['a'] * np.exp(self.parameters['c'] * x)

            return predict

        else:
            raise ValueError(f"Unknown single-predictor formula type: {self.formula_type}")

    def _build_multi_predictor_function(self) -> Callable:
        """Build multi-predictor prediction functions (existing)."""

        if self.formula_type == 'additive_power':
            # f(x) = a1*x1^b1 + a2*x2^b2 + ... + constant

            def predict(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)

                result = np.zeros(len(X))

                # Add each predictor term
                for i in range(len(self.predictors)):
                    if i in self.parameters:
                        a = self.parameters[i]['a']
                        b = self.parameters[i]['b']
                        x_i = X[:, i]
                        result += a * (x_i ** b)

                # Add constant if present
                if 'constant' in self.parameters:
                    result += self.parameters['constant']

                return result

            return predict

        elif self.formula_type == 'multiplicative_power':
            # f(x) = a * (x1^b1) * (x2^b2) * ... + constant

            def predict(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)

                # Start with coefficient 'a'
                result = np.full(len(X), self.parameters.get('a', 1.0))

                # Multiply each predictor term
                for i in range(len(self.predictors)):
                    if i in self.parameters:
                        b = self.parameters[i]['b']
                        x_i = X[:, i]
                        result *= (x_i ** b)

                # Add constant if present
                if 'constant' in self.parameters:
                    result += self.parameters['constant']

                return result

            return predict

        elif self.formula_type == 'multi_linear':


            def predict(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)

                result = np.zeros(len(X))

                # Add constant term (intercept)
                if 'constant' in self.parameters:
                    result += self.parameters['constant']

                # Get term definitions
                if 'terms' not in self.parameters:
                    raise ValueError(f"multi_linear_regression requires 'terms' parameter")

                terms = self.parameters['terms']

                # Process each term
                for i, term_str in enumerate(terms):
                    # Get coefficient for this term (a1, a2, etc.)
                    if i not in self.parameters:
                        continue  # Skip if coefficient not provided

                    coef = self.parameters[i].get('a', 0.0)

                    if coef == 0:
                        continue  # Skip zero coefficients

                    # Evaluate the term
                    term_value = self._evaluate_term(term_str, X)
                    result += coef * term_value

                return result

            return predict

        elif self.formula_type == 'power_product':

            def predict(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)

                result = np.zeros(len(X))
                n_predictors = len(self.predictors)

                # Part 1: Individual power terms
                for i in range(n_predictors):
                    if i in self.parameters:
                        a = self.parameters[i].get('a', 0.0)
                        b_value = self.parameters[i].get('b', 1.0)
                        x_i = X[:, i]

                        func, power = self._get_ln_power(b_value)
                        if func is not None and power is not None:
                            x_transformed = self._apply_function(x_i, func)
                            result += a * (x_transformed ** power)
                        elif func is None and power is not None:
                            result += a * (x_i ** power)
                        else:
                            result += a * (x_i ** b_value)

                # Part 2: Interaction term
                interaction_start = n_predictors
                interaction_end = interaction_start + n_predictors
                has_interaction = False
                product = np.ones(len(X))

                for i in range(n_predictors):
                    param_idx = interaction_start + i
                    if param_idx in self.parameters and 'b' in self.parameters[param_idx]:
                        has_interaction = True
                        b_value = self.parameters[param_idx]['b']
                        x_i = X[:, i]

                        func, power = self._get_ln_power(b_value)
                        if func is not None and power is not None:
                            x_transformed = self._apply_function(x_i, func)
                            product *= (x_transformed ** power)
                        elif func is None and power is not None:
                            product *= (x_i ** power)
                        else:
                            product *= (x_i ** b_value)

                if has_interaction and interaction_end in self.parameters:
                    a_final = self.parameters[interaction_end].get('a', 1.0)
                    b_final_value = self.parameters[interaction_end].get('b', 1.0)

                    func, power = self._get_ln_power(b_final_value)
                    if func is not None and power is not None:
                        product_transformed = self._apply_function(product, func)
                        result += a_final * (product_transformed ** power)
                    elif func is None and power is not None:
                        result += a_final * (product ** power)
                    else:
                        result += a_final * (product ** b_final_value)

                if 'constant' in self.parameters:
                    result += self.parameters['constant']

                return result

            return predict

        else:
            raise ValueError(f"Unknown multi-predictor formula type: {self.formula_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the formula."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Validate number of predictors
        if X.shape[1] != len(self.predictors):
            raise ValueError(f"Expected {len(self.predictors)} predictors, got {X.shape[1]}")

        return self.predict_func(X)

    def get_formula_text(self, precision: int = 4) -> str:
        """Generate text representation of the formula."""

        # NEW: Piecewise formulas
        if self.formula_type in ['piecewise_power', 'piecewise_linear', 'piecewise_mixed']:
            return self._get_piecewise_text(precision)

        # Single predictor formulas (existing)
        if len(self.predictors) == 1:
            return self._get_single_predictor_text(precision)

        # Multi predictor formulas (existing)
        else:
            return self._get_multi_predictor_text(precision)

    def _get_piecewise_text(self, precision: int) -> str:
        """
        Generate text representation for piecewise formulas.

        Returns a multi-line string showing the formula for each bin segment.
        """
        if self.bins is None or self.bin_parameters is None:
            return f"Piecewise formula (bins not defined)"

        lines = []
        predictor_name = self.predictors[0] if self.predictors else 'x'

        for i in range(len(self.bin_parameters)):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            params = self.bin_parameters[i]

            # Format bin range
            if i == len(self.bin_parameters) - 1:
                range_str = f"{lower:,.0f} ≤ {predictor_name} ≤ {upper:,.0f}"
            else:
                range_str = f"{lower:,.0f} ≤ {predictor_name} < {upper:,.0f}"

            # Determine formula form and format
            if 'b' in params and params['b'] != 0 and params['b'] is not None:
                # Power form: a * x^b
                a = params.get('a', 1.0)
                b = params['b']
                formula_str = f"y = {a:.{precision}f} × {predictor_name}^{b:.{precision}f}"

            elif 'c' in params and params['c'] != 0 and params['c'] is not None:
                # Linear form: b + c*x
                b_const = params.get('b', 0.0)
                c = params['c']

                if b_const == 0:
                    formula_str = f"y = {c:.{precision}f} × {predictor_name}"
                else:
                    sign = "+" if b_const >= 0 else "-"
                    formula_str = f"y = {abs(b_const):.{precision}f} {sign} {c:.{precision}f} × {predictor_name}"

            elif 'a' in params:
                # Constant form
                formula_str = f"y = {params['a']:.{precision}f}"

            else:
                formula_str = "y = 0"

            lines.append(f"  {range_str}: {formula_str}")

        return "\n".join(lines)

    def _get_single_predictor_text(self, precision: int) -> str:
        """Text representation for single-predictor formulas (existing logic)."""
        params = self.parameters

        if self.formula_type == 'power':
            a, b, c = params['a'], params['b'], params['c']
            if b == 0:
                return f"y = {a:.{precision}f} × x^{c:.{precision}f}"
            else:
                sign = "+" if a >= 0 else ""
                return f"y = {b:.{precision}f} {sign} {a:.{precision}f} × x^{c:.{precision}f}"

        elif self.formula_type == 'power_log':
            a, b, c = params['a'], params['b'], params['c']
            if b == 0:
                return f"y = {a:.{precision}f} × log(x^{c:.{precision}f})"
            else:
                sign = "+" if a >= 0 else ""
                return f"y = {b:.{precision}f} {sign} {a:.{precision}f} × log(x^{c:.{precision}f})"

        elif self.formula_type == 'log_power':
            a, b, c = params['a'], params['b'], params['c']
            if b == 0:
                return f"y = {a:.{precision}f} × (log(x))^{c:.{precision}f}"
            else:
                sign = "+" if a >= 0 else ""
                return f"y = {b:.{precision}f} {sign} {a:.{precision}f} × (log(x))^{c:.{precision}f}"

        elif self.formula_type == 'cubic_polynomial':
            a, b, c, d = params['a'], params['b'], params['c'], params['d']
            terms = []
            if a != 0:
                terms.append(f"{a:.{precision}f}x³")
            if b != 0:
                sign = "+" if b > 0 else "-"
                terms.append(f"{sign} {abs(b):.{precision}f}x²")
            if c != 0:
                sign = "+" if c > 0 else "-"
                terms.append(f"{sign} {abs(c):.{precision}f}x")
            if d != 0:
                sign = "+" if d > 0 else "-"
                terms.append(f"{sign} {abs(d):.{precision}f}")
            return f"y = {' '.join(terms)}" if terms else "y = 0"

        elif self.formula_type == 'exponential':
            a, b, c = params['a'], params['b'], params['c']
            if b == 0:
                return f"y = {a:.{precision}f} × e^({c:.{precision}f}x)"
            else:
                sign = "+" if a >= 0 else ""
                return f"y = {b:.{precision}f} {sign} {a:.{precision}f} × e^({c:.{precision}f}x)"

        return f"Unknown type: {self.formula_type}"

    def _get_multi_predictor_text(self, precision: int) -> str:
        """Text representation for multi-predictor formulas (existing)."""
        if self.formula_type == 'multi_linear':
            terms = []

            # Add constant first
            if 'constant' in self.parameters:
                c = self.parameters['constant']
                terms.append(f"{c:.{precision}f}")

            # Get term strings
            if 'terms' not in self.parameters:
                return "y = [multi_linear: terms not defined]"

            term_strings = self.parameters['terms']

            # Add each term with its coefficient
            for i, term_str in enumerate(term_strings):
                if i not in self.parameters:
                    continue

                coef = self.parameters[i].get('a', 0.0)
                if coef == 0:
                    continue

                # Format coefficient with sign
                if len(terms) == 0 and 'constant' not in self.parameters:
                    # First term, no constant
                    coef_str = f"{coef:.{precision}f}"
                else:
                    # Subsequent terms need explicit sign
                    sign = "+" if coef >= 0 else "-"
                    coef_str = f"{sign} {abs(coef):.{precision}f}"

                # Clean up term string for display
                term_display = term_str.replace(' (M)', '').replace(' ', '_')

                terms.append(f"{coef_str}·{term_display}")

            return f"y = {' '.join(terms)}" if terms else "y = 0"

        elif self.formula_type == 'additive_power':
            terms = []

            for i, pred in enumerate(self.predictors):
                if i in self.parameters:
                    a = self.parameters[i]['a']
                    b = self.parameters[i]['b']
                    pred_short = pred.replace(' (M)', '').replace(' ', '_')

                    if len(terms) == 0:
                        terms.append(f"{a:.{precision}f}·{pred_short}^{b:.{precision}f}")
                    else:
                        sign = "+" if a >= 0 else "-"
                        terms.append(f"{sign} {abs(a):.{precision}f}·{pred_short}^{b:.{precision}f}")

            if 'constant' in self.parameters:
                c = self.parameters['constant']
                if c != 0:
                    sign = "+" if c >= 0 else "-"
                    terms.append(f"{sign} {abs(c):.{precision}f}")

            return f"y = {' '.join(terms)}" if terms else "y = 0"

        elif self.formula_type == 'multiplicative_power':
            a = self.parameters.get('a', 1.0)
            terms = [f"{a:.{precision}f}"]

            for i, pred in enumerate(self.predictors):
                if i in self.parameters:
                    b = self.parameters[i]['b']
                    pred_short = pred.replace(' (M)', '').replace(' ', '_')
                    terms.append(f"({pred_short}^{b:.{precision}f})")

            result = " × ".join(terms)

            if 'constant' in self.parameters and self.parameters['constant'] != 0:
                c = self.parameters['constant']
                sign = "+" if c >= 0 else "-"
                result += f" {sign} {abs(c):.{precision}f}"

            return f"y = {result}"

        elif self.formula_type == 'power_product':
            terms = []
            n_pred = len(self.predictors)

            for i in range(n_pred):
                if i in self.parameters:
                    a = self.parameters[i].get('a', 0)
                    b_value = self.parameters[i].get('b', 1)
                    pred_short = self.predictors[i].replace(' (M)', '').replace(' ', '_')

                    func, power = self._get_ln_power(b_value)
                    if func is not None and power is not None:
                        term_str = f"{a:.{precision}f}·({func}({pred_short}))^{power:.{precision}f}"
                    else:
                        term_str = f"{a:.{precision}f}·{pred_short}^{b_value:.{precision}f}"

                    if a != 0:
                        if len(terms) == 0:
                            terms.append(term_str)
                        else:
                            sign = "+" if a >= 0 else "-"
                            terms.append(f"{sign} {term_str}")

            interaction_start = n_pred
            interaction_end = interaction_start + n_pred

            if interaction_end in self.parameters:
                a_final = self.parameters[interaction_end].get('a', 1.0)
                b_final_value = self.parameters[interaction_end].get('b', 1.0)

                inner_terms = []
                for i in range(n_pred):
                    param_idx = interaction_start + i
                    if param_idx in self.parameters and 'b' in self.parameters[param_idx]:
                        pred_short = self.predictors[i].replace(' (M)', '').replace(' ', '_')
                        b_value = self.parameters[param_idx]['b']

                        func, power = self._get_ln_power(b_value)
                        if func is not None and power is not None:
                            inner_terms.append(f"({func}({pred_short}))^{power:.{precision}f}")
                        else:
                            inner_terms.append(f"{pred_short}^{b_value:.{precision}f}")

                if inner_terms and a_final != 0:
                    inner_product = " × ".join(inner_terms)
                    func_final, power_final = self._get_ln_power(b_final_value)

                    if func_final is not None and power_final is not None:
                        interaction_str = f"({func_final}({inner_product}))^{power_final:.{precision}f}"
                    else:
                        interaction_str = f"({inner_product})^{b_final_value:.{precision}f}"

                    sign = "+" if a_final >= 0 else "-"
                    terms.append(f"{sign} {abs(a_final):.{precision}f}·{interaction_str}")

            if 'constant' in self.parameters:
                c = self.parameters['constant']
                if c != 0:
                    sign = "+" if c >= 0 else "-"
                    terms.append(f"{sign} {abs(c):.{precision}f}")

            return f"y = {' '.join(terms)}" if terms else "y = 0"

        return f"Unknown type: {self.formula_type}"

    def _get_ln_power(self, value: float | int | str):
        """
        Helper method for finding out whether ln or not

        Returns
        -------
        (func, power) : tuple
            - If string function: ('ln', 3.0) for 'ln3'
            - If numeric: (None, 2.0) for 2.0
            - If invalid: (None, None)
        """
        if isinstance(value, (int, float)):
            return None, value

        elif isinstance(value, str):
            match = re.match(r"([a-zA-Z]+)(\d+(?:\.\d+)?)", value)
            if match:
                func, num = match.groups()
                return func, float(num)

        return None, None

    def _apply_function(self, x: np.ndarray, func: str) -> np.ndarray:
        """
        Apply mathematical function to array.

        Parameters
        ----------
        x : np.ndarray
            Input values
        func : str
            Function name: 'ln', 'log', 'log10', 'exp', 'sqrt', etc.

        Returns
        -------
        np.ndarray
            Transformed values
        """
        func = func.lower()

        # Add small epsilon to avoid log(0) or division by zero
        x_safe = np.maximum(x, 1e-10)

        if func in ['ln', 'log']:
            return np.log(x_safe)
        elif func == 'log10':
            return np.log10(x_safe)
        elif func == 'log2':
            return np.log2(x_safe)
        elif func == 'exp':
            return np.exp(x)
        elif func == 'sqrt':
            return np.sqrt(np.abs(x))
        elif func in ['sin', 'sine']:
            return np.sin(x)
        elif func in ['cos', 'cosine']:
            return np.cos(x)
        elif func in ['tan', 'tangent']:
            return np.tan(x)
        else:
            raise ValueError(f"Unknown function: {func}")

    def __str__(self):
        return f"{self.name}: {self.get_formula_text()}"

    def __repr__(self):
        preds = ', '.join(self.predictors)
        return f"LiteratureFormula('{self.name}', [{preds}] → {self.target}, {self.formula_type})"