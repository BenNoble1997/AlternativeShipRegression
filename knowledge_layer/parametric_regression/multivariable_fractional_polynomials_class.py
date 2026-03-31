
"""
Multivariable Fractional Polynomials (MFP) — Core Implementation
=================================================================

Role in the Architecture
-------------------------
This module contains :class:`MFPRegressor`, the algorithmic core of the
MFP subsystem.  It is an sklearn-style estimator that implements the full
Royston & Sauerbrei (2008) algorithm:

1. **Backward Elimination (BE)** — remove variables whose contribution is
   not statistically significant.
2. **Function Selection Procedure (FSP)** — for each retained variable,
   select the best fractional polynomial transformation from a closed test
   hierarchy: FP2 > FP1 > Linear > Exclude.
3. **Cycle until convergence** — repeat BE → FSP until two consecutive
   cycles produce identical transformations.

:class:`MFPRegressor` is *not* called directly by analysis code.  It is
instantiated and called by
:class:`~knowledge_layer.parametric_regression.mfp_fitter.MFPFitter`,
which satisfies the
:class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
interface required by
:class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`.

The ``fixed_transforms`` Pathway
---------------------------------
The standard algorithm discovers transformations from data.  However,
cross-dataset coefficient comparison — a core requirement of this research
— demands that the *same* transformation structure be applied to multiple
datasets (conventional, dual-fuel, pure-alternative ships).

When ``fixed_transforms`` is supplied to ``__init__``, :meth:`fit` bypasses
BE and FSP entirely and calls :meth:`_fit_with_fixed_transforms`, which
fits ordinary linear regression on the pre-specified transformed features.
This is also used internally by
:meth:`~knowledge_layer.parametric_regression.parametric_model_description_class.ModelDescriptor._refit_mfp_fixed_structure`.

Fractional Polynomial Transformations
---------------------------------------
The power set searched by default is:

    P = {−3, −2.5, −2, −1.5, −1, −0.5, −1/3, 0,
         1/3, 0.5, 1, 1.5, 2, 2.5, 3}

where power 0 denotes ln(x) by MFP convention.

For a single variable x and FP degree 2 (FP2), two terms are generated:

- **Different powers** p₁ ≠ p₂:  x^{p₁}  and  x^{p₂}
- **Repeated power** p₁ = p₂ = p:  x^p  and  x^p · ln(x)

The repeated-power case is the continuous limit as p₂ → p₁ and ensures
the FP2 family is closed in a mathematical sense.

Statistical Testing — Likelihood Ratio Test
--------------------------------------------
All variable significance and transformation selection decisions use the
**likelihood ratio test (LRT)** for nested linear models.

Under the Gaussian likelihood assumption, the LRT statistic is:

.. math::

    \\Lambda = n \\bigl(\\ln(\\text{RSS}_0) - \\ln(\\text{RSS}_1)\\bigr)

where RSS₀ and RSS₁ are the residual sums of squares for the restricted
(simpler) and full (more complex) models respectively, and *n* is the
sample size.  Under H₀, :math:`\\Lambda \\sim \\chi^2(\\Delta k)` where
:math:`\\Delta k = k_1 - k_0` is the difference in the number of
parameters.

References
----------
Royston, P., & Sauerbrei, W. (2008). *Multivariable Model-Building: A
Pragmatic Approach to Regression Analysis Based on Fractional Polynomials
for Modelling Continuous Variables*. Wiley.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings


class MFPRegressor:
    """
    Multivariable Fractional Polynomials (MFP) Regressor.

    Fits a regression model by simultaneously searching for the best
    fractional polynomial transformation of each predictor and performing
    backward variable selection, following the algorithm of Royston &
    Sauerbrei (2008).

    The result is a linear regression model in the transformed feature
    space.  The discovered transformations are stored in
    :attr:`transforms_` and applied automatically during
    :meth:`predict`.

    :param alpha1: Significance level for **backward elimination**.
        A variable is removed when its likelihood ratio test p-value
        exceeds this threshold.  Lower values are more conservative
        (retain more variables).  Defaults to 0.05.
    :type alpha1: float

    :param alpha2: Significance level for the **function selection
        procedure**.  Controls the choice between FP2, FP1, and linear
        within the closed test hierarchy.  Lower values prefer simpler
        transformations.  Defaults to 0.05.
    :type alpha2: float

    :param powers: Power set to search over.  Power 0 is interpreted as
        ln(x) by MFP convention.  When ``None``, :attr:`DEFAULT_POWERS`
        is used.
    :type powers: list[float], optional

    :param max_cycles: Maximum number of BE → FSP cycles before stopping,
        regardless of whether convergence has been detected.  Defaults
        to 20.
    :type max_cycles: int

    :param fixed_transforms: Pre-specified transformation structure.
        When provided, :meth:`fit` bypasses BE and FSP and fits linear
        regression directly on the transformed features.  Used for
        cross-dataset coefficient comparison.

        Format: ``{var_idx: (transform_type, powers)}`` where

        - ``var_idx`` (*int*): zero-indexed column of the predictor
          matrix.
        - ``transform_type`` (*str*): one of ``'linear'``, ``'fp1'``,
          ``'fp2'``, ``'excluded'``.
        - ``powers`` (*list[float]*): transformation powers.  Empty
          list for excluded variables.

    :type fixed_transforms: dict, optional

    :param verbose: If ``True``, prints detailed progress during each
        BE and FSP step.  Defaults to ``False``.
    :type verbose: bool

    :ivar transforms\_: Final transformation assigned to each variable
        after fitting.  Format: ``{var_idx: (transform_type, powers)}``.
        ``None`` before :meth:`fit` is called.
    :vartype transforms\_: dict or None

    :ivar feature_names_in\_: Names of input features in the order they
        appear in the columns of ``X``.  Set during :meth:`fit`.
    :vartype feature_names_in\_: list[str] or None

    :ivar model\_: Fitted :class:`~sklearn.linear_model.LinearRegression`
        on the transformed feature space.  ``None`` before fitting.
    :vartype model\_: LinearRegression or None

    :ivar coef\_: Coefficient array of :attr:`model_`, shape
        ``(n_transformed_features,)``.
    :vartype coef\_: numpy.ndarray or None

    :ivar intercept\_: Intercept of :attr:`model_`.
    :vartype intercept\_: float or None

    :ivar n_cycles\_: Number of BE → FSP cycles completed before
        convergence (or ``max_cycles`` was reached).  Set to 0 when
        ``fixed_transforms`` is used.
    :vartype n_cycles\_: int

    :ivar convergence_history\_: List of transformation dicts captured at
        the end of each cycle.  Useful for diagnosing convergence
        behaviour.
    :vartype convergence_history\_: list[dict]

    :cvar DEFAULT_POWERS: Standard MFP power set from Royston & Sauerbrei.
    :vartype DEFAULT_POWERS: list[float]
    """

    # Standard MFP power set from Royston & Sauerbrei
    DEFAULT_POWERS = [-3,-2.5,-2,-1.5, -1, -0.5,-1/3, 0, 0.5,1/3, 1,1.5,2,2.5,3]

    def __init__(self,
                 alpha1: float = 0.05,
                 alpha2: float = 0.05,
                 powers: list[float]|None = None,
                 max_cycles: int = 20,
                fixed_transforms: dict = None,
                 verbose: bool = False):

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.powers = powers if powers is not None else self.DEFAULT_POWERS
        self.max_cycles = max_cycles
        self.fixed_transforms = fixed_transforms
        self.verbose = verbose

        # Will be set during fit
        self.transforms_ = None
        self.feature_names_in_ = None
        self.model_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_cycles_ = 0
        self.convergence_history_ = []

    # =========================================================================
    # Public API
    # =========================================================================

    def fit(self, X, y, feature_names: list[str] = None):
        """
        Fit the MFP model to training data.

        If :attr:`fixed_transforms` was supplied at construction time,
        :meth:`_fit_with_fixed_transforms` is called and the BE / FSP
        algorithm is skipped entirely.

        Otherwise, the full iterative algorithm runs:

        1. Initialise all variables as linear.
        2. **Backward Elimination** — test each variable for significance
           using the LRT; remove those with p > :attr:`alpha1`.
        3. **Function Selection Procedure** — for each retained variable,
           run the three-step closed test to select the best
           transformation.
        4. Check convergence.  If transformations are unchanged relative
           to the previous cycle, stop.
        5. Rotate the FSP start index for the next cycle to prevent
           order-dependence artefacts.
        6. Fit the final linear model on the fully transformed design
           matrix.

        :param X: Training predictor matrix, shape
            ``(n_samples, n_features)``.  Values ≤ 0 trigger a warning
            because fractional powers and logarithms require positive
            arguments; the data is shifted automatically in such cases.
        :type X: array-like
        :param y: Training target vector, shape ``(n_samples,)``.
        :type y: array-like
        :param feature_names: Names for the predictor columns.  When
            ``None``, names ``'x0'``, ``'x1'``, … are assigned.  Used in
            verbose output, coefficient dictionaries, and
            :meth:`get_selected_transformations`.
        :type feature_names: list[str], optional

        :returns: ``self`` (fitted estimator).
        :rtype: MFPRegressor

        :raises ValueError: If ``n_samples < 5``, ``n_features == 0``, or
            ``len(feature_names) != n_features``.
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Validate data
        if n_samples < 5:
            raise ValueError(f"Need at least 5 samples, got {n_samples}")

        if n_features == 0:
            raise ValueError("X has 0 features")

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(n_features)]
        else:
            feature_names = list(feature_names)
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) doesn't match "
                    f"n_features ({n_features})"
                )

        self.feature_names_in_ = feature_names

        # Check for positive values (required for fractional powers)
        if np.any(X <= 0):
            warnings.warn(
                "X contains non-positive values. MFP will shift data to positive range. "
                "This may affect interpretation of results."
            )

        if self.fixed_transforms is not None:
            return self._fit_with_fixed_transforms(X, y, self.fixed_transforms)

        if self.verbose:
            print("=" * 80)
            print("MULTIVARIABLE FRACTIONAL POLYNOMIALS (MFP)")
            print("=" * 80)
            print(f"Features: {n_features}")
            print(f"Samples: {n_samples}")
            print(f"Alpha1 (BE): {self.alpha1}")
            print(f"Alpha2 (FSP): {self.alpha2}")
            print(f"Powers: {self.powers}")
            print("=" * 80)

        # Initialize all variables as linear
        current_transforms = {i: ('linear', [1]) for i in range(n_features)}
        previous_transforms = None

        cycle = 0
        fsp_start_idx = 0  # Rotates each cycle

        # Main MFP cycle
        while cycle < self.max_cycles:
            cycle += 1
            self.n_cycles_ = cycle

            if self.verbose:
                print(f"\n{'=' * 80}")
                print(f"CYCLE {cycle}")
                print(f"{'=' * 80}")

            # STEP 1: BACKWARD ELIMINATION
            current_transforms, var_order = self._backward_elimination(
                X, y, current_transforms
            )

            if len(var_order) == 0:
                # All variables eliminated!
                if self.verbose:
                    print("\nAll variables eliminated! Fitting intercept-only model.")
                # Fit intercept-only model
                self.model_ = LinearRegression()
                self.model_.fit(np.zeros((n_samples, 1)), y)
                self.transforms_ = {}
                self.coef_ = np.array([])
                self.intercept_ = self.model_.intercept_
                return self

            # STEP 2: FUNCTION SELECTION PROCEDURE
            current_transforms = self._function_selection_procedure(
                X, y, current_transforms, var_order,
                start_idx=fsp_start_idx
            )

            # Store convergence history
            self.convergence_history_.append(current_transforms.copy())

            # Check convergence
            if self._check_convergence(current_transforms, previous_transforms):
                if self.verbose:
                    print(f"\n{'=' * 80}")
                    print(f"✓ CONVERGED after {cycle} cycles!")
                    print(f"{'=' * 80}")
                break

            previous_transforms = current_transforms.copy()

            # Rotate FSP start point for next cycle
            fsp_start_idx = (fsp_start_idx + 1) % len(var_order)

        # Final model fit with selected transformations
        self.transforms_ = current_transforms
        self._fit_final_model(X, y, current_transforms)

        if self.verbose:
            print(self.get_model_summary())

        return self

    def predict(self, X):
        """
        Generate predictions from the fitted MFP model.

        The stored :attr:`transforms_` are applied to ``X`` via
        :meth:`_apply_transforms` to produce the transformed design matrix,
        which is then passed to the fitted
        :class:`~sklearn.linear_model.LinearRegression`.

        For an intercept-only model (all variables excluded during fitting)
        a constant array equal to :attr:`intercept_` is returned.

        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: array-like

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray

        :raises RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_transformed = self._apply_transforms(X, self.transforms_)

        if X_transformed.shape[1] == 0:
            # Intercept-only model
            return np.full(len(X), self.intercept_)

        return self.model_.predict(X_transformed)


    # =========================================================================
    # Algorithm Steps
    # =========================================================================

    def _backward_elimination(self, X, y, current_transforms):
        """
        Perform one backward elimination pass over all current variables.

        For each variable, a likelihood ratio test compares the full model
        (all variables in their current transformations) against a reduced
        model with that variable removed.  Variables whose test p-value
        exceeds :attr:`alpha1` are dropped.

        The retained variables are returned in **ascending p-value order**
        (most significant first).  This ordering governs the sequence in
        which variables are processed during the subsequent Function
        Selection Procedure, and it rotates by one position each cycle
        to reduce order-dependence.

        :param X: Raw predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param current_transforms: Active transformations at the start of
            this pass, ``{var_idx: (transform_type, powers)}``.
        :type current_transforms: dict

        :returns: A two-element tuple:

            - ``updated_transforms`` (*dict*) — transformations after
              removing non-significant variables.
            - ``var_order`` (*list[int]*) — indices of retained variables,
              sorted by ascending p-value.

        :rtype: tuple[dict, list[int]]
        """
        if self.verbose:
            print("\n" + "-" * 80)
            print("STEP 1: BACKWARD ELIMINATION")
            print("-" * 80)

        # Build current model with all variables
        X_transformed = self._apply_transforms(X, current_transforms)

        if X_transformed.shape[1] == 0:
            # No variables left
            return {}, []

        model_full = LinearRegression().fit(X_transformed, y)
        rss_full = self._calculate_rss(model_full, X_transformed, y)
        n_params_full = X_transformed.shape[1] + 1  # +1 for intercept

        # Calculate p-values for each variable using likelihood ratio test
        p_values = {}

        for var_idx in current_transforms.keys():
            # Model without this variable
            transforms_without = {k: v for k, v in current_transforms.items()
                                  if k != var_idx}

            if len(transforms_without) == 0:
                # This is the only variable - test against null model
                X_without = np.zeros((len(y), 0))
                model_without = LinearRegression()
                # Fit intercept-only
                model_without.fit(np.ones((len(y), 1)), y)
                rss_without = np.sum((y - y.mean()) ** 2)
                n_params_without = 1
            else:
                X_without = self._apply_transforms(X, transforms_without)
                model_without = LinearRegression().fit(X_without, y)
                rss_without = self._calculate_rss(model_without, X_without, y)
                n_params_without = X_without.shape[1] + 1

            # Likelihood ratio test
            p_value = self._likelihood_ratio_test(
                rss_without, rss_full,
                n_params_without, n_params_full,
                len(y)
            )
            p_values[var_idx] = p_value

        # Order by p-values (increasing = most significant first)
        var_order = sorted(p_values.keys(), key=lambda i: p_values[i])

        # Remove non-significant variables (p > alpha1)
        updated_transforms = {}

        if self.verbose:
            print(f"\nVariable significance (alpha1={self.alpha1}):")

        for var_idx in var_order:
            p_val = p_values[var_idx]
            var_name = self.feature_names_in_[var_idx]

            if p_val <= self.alpha1:
                updated_transforms[var_idx] = current_transforms[var_idx]
                if self.verbose:
                    transform_type, powers = current_transforms[var_idx]
                    print(f"  {var_name:20s} p={p_val:.4f}  →  KEEP ({transform_type})")
            else:
                if self.verbose:
                    print(f"  {var_name:20s} p={p_val:.4f}  →  REMOVE")

        # Update var_order to only include kept variables
        var_order = [i for i in var_order if i in updated_transforms]

        return updated_transforms, var_order

    def _function_selection_procedure(self, X, y, transforms, var_order, start_idx):
        """
        Perform one Function Selection Procedure (FSP) pass.

        Each retained variable is visited in order (starting from
        ``start_idx`` and wrapping circularly).  For each variable,
        :meth:`_closed_test_procedure` is called to select the best
        transformation, and the result immediately replaces the variable's
        current entry in ``updated_transforms`` before the next variable
        is processed.  This *sequential* update ensures that each
        variable's transformation is optimised with respect to the
        *current best* transformations of all other variables.

        :param X: Raw predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param transforms: Current transformations entering FSP.
        :type transforms: dict
        :param var_order: Variable indices in significance order
            (ascending p-value from BE).
        :type var_order: list[int]
        :param start_idx: Index within ``var_order`` at which to start
            processing.  Rotates each cycle.
        :type start_idx: int

        :returns: Updated transformations after FSP.
        :rtype: dict
        """
        if self.verbose:
            print("\n" + "-" * 80)
            print("STEP 2: FUNCTION SELECTION PROCEDURE")
            print("-" * 80)
            print(f"Starting from variable index {start_idx} in ordered list")

        n_vars = len(var_order)
        updated_transforms = transforms.copy()

        # Process variables starting from start_idx (circular)
        for offset in range(n_vars):
            idx = (start_idx + offset) % n_vars
            var_idx = var_order[idx]
            var_name = self.feature_names_in_[var_idx]

            if self.verbose:
                print(f"\n  Processing: {var_name}")

            # Apply closed test for this variable
            best_transform = self._closed_test_procedure(
                X, y, var_idx, updated_transforms
            )

            updated_transforms[var_idx] = best_transform

            if self.verbose:
                transform_type, powers = best_transform
                if transform_type == 'linear':
                    print(f"    → Selected: LINEAR")
                elif transform_type == 'fp1':
                    power_str = self._format_power(powers[0])
                    print(f"    → Selected: FP1 ({power_str})")
                elif transform_type == 'fp2':
                    if len(powers) == 1:
                        power_str = self._format_power(powers[0]) + " (repeated)"
                    else:
                        power_str = f"{self._format_power(powers[0])}, {self._format_power(powers[1])}"
                    print(f"    → Selected: FP2 ({power_str})")

        return updated_transforms

    def _closed_test_procedure(self, X, y, var_idx, current_transforms):
        """
        Apply the three-step closed test hierarchy for a single variable.

        The closed test ensures the family-wise error rate is controlled
        when making multiple simultaneous comparisons.  The hierarchy is:

        **Test 1 — Null vs best FP2** (at :attr:`alpha1`):
            If the variable adds no significant improvement over a model
            without it, it is excluded.

        **Test 2 — Linear vs best FP2** (at :attr:`alpha2`):
            If a linear term is not significantly worse than the best FP2,
            use linear (prefer parsimony).

        **Test 3 — Best FP1 vs best FP2** (at :attr:`alpha2`):
            If FP1 is not significantly worse than FP2, use the simpler
            FP1.  Otherwise use FP2.

        At each test, the "other variables" are held fixed in their
        *current* transformations (from ``current_transforms``), and
        models are compared using the LRT.

        :param X: Raw predictor matrix.
        :type X: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param var_idx: Column index of the variable being tested.
        :type var_idx: int
        :param current_transforms: Current transformations for all
            variables (including the variable being tested, whose entry
            will be replaced by the result of this method).
        :type current_transforms: dict

        :returns: ``(transform_type, powers)`` — the selected
            transformation for ``var_idx``.
        :rtype: tuple[str, list]
        """
        x_var = X[:, var_idx]

        # Build model with OTHER variables in their current form
        other_transforms = {k: v for k, v in current_transforms.items()
                            if k != var_idx}
        X_others = self._apply_transforms(X, other_transforms)

        n_samples = len(y)

        # Find best FP2 model
        best_fp2_powers, best_fp2_rss = self._find_best_fp2(x_var, y, X_others)
        n_params_fp2 = X_others.shape[1] + len(best_fp2_powers) + 1  # +1 for intercept

        # TEST 1: NULL vs best FP2 (at alpha1)
        # Model without this variable
        if X_others.shape[1] > 0:
            model_null = LinearRegression().fit(X_others, y)
            rss_null = self._calculate_rss(model_null, X_others, y)
        else:
            # Intercept-only model
            rss_null = np.sum((y - y.mean()) ** 2)

        n_params_null = X_others.shape[1] + 1

        p_value_null_vs_fp2 = self._likelihood_ratio_test(
            rss_null, best_fp2_rss,
            n_params_null, n_params_fp2,
            n_samples
        )

        if p_value_null_vs_fp2 > self.alpha1:
            # Variable not significant → exclude
            if self.verbose:
                print(f"      Test 1: NULL vs FP2: p={p_value_null_vs_fp2:.4f} > {self.alpha1} → EXCLUDE")
            return ('excluded', [])

        if self.verbose:
            print(f"      Test 1: NULL vs FP2: p={p_value_null_vs_fp2:.4f} ≤ {self.alpha1} → Keep variable")

        # TEST 2: best FP2 vs Linear (at alpha2)
        linear_rss = self._calculate_fp_rss(x_var, y, X_others, [1])
        n_params_linear = X_others.shape[1] + 1 + 1  # +1 for linear term, +1 for intercept

        p_value_linear_vs_fp2 = self._likelihood_ratio_test(
            linear_rss, best_fp2_rss,
            n_params_linear, n_params_fp2,
            n_samples
        )

        if p_value_linear_vs_fp2 > self.alpha2:
            # Linear is sufficient
            if self.verbose:
                print(f"      Test 2: Linear vs FP2: p={p_value_linear_vs_fp2:.4f} > {self.alpha2} → Use LINEAR")
            return ('linear', [1])

        if self.verbose:
            print(f"      Test 2: Linear vs FP2: p={p_value_linear_vs_fp2:.4f} ≤ {self.alpha2} → Need non-linear")

        # TEST 3: best FP2 vs best FP1 (at alpha2)
        best_fp1_power, best_fp1_rss = self._find_best_fp1(x_var, y, X_others)
        n_params_fp1 = X_others.shape[1] + 1 + 1  # +1 for FP1 term, +1 for intercept

        p_value_fp1_vs_fp2 = self._likelihood_ratio_test(
            best_fp1_rss, best_fp2_rss,
            n_params_fp1, n_params_fp2,
            n_samples
        )

        if p_value_fp1_vs_fp2 <= self.alpha2:
            # FP1 is significantly worse than FP2 → use FP2
            if self.verbose:
                print(f"      Test 3: FP1 vs FP2: p={p_value_fp1_vs_fp2:.4f} ≤ {self.alpha2} → Use FP2")
            return ('fp2', best_fp2_powers)
        else:
            # FP1 is not significantly worse → use simpler FP1
            if self.verbose:
                print(f"      Test 3: FP1 vs FP2: p={p_value_fp1_vs_fp2:.4f} > {self.alpha2} → Use FP1")
            return ('fp1', [best_fp1_power])

    # =========================================================================
    # Power Search Helpers
    # =========================================================================

    def _find_best_fp1(self, x, y, X_others):
        """
        Find the single power in :attr:`powers` that minimises RSS.

        Every power in :attr:`powers` is tried independently.  The
        transformed feature is combined with ``X_others`` and a linear
        regression is fitted; the power yielding the lowest RSS is
        returned.

        :param x: Single-variable vector to transform, shape
            ``(n_samples,)``.
        :type x: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param X_others: Already-transformed matrix of all other
            variables, shape ``(n_samples, n_other)``.
        :type X_others: numpy.ndarray

        :returns: ``(best_power, best_rss)``
        :rtype: tuple[float, float]
        """
        best_power = None
        best_rss = np.inf

        for power in self.powers:
            rss = self._calculate_fp_rss(x, y, X_others, [power])

            if rss < best_rss:
                best_rss = rss
                best_power = power

        return best_power, best_rss

    def _find_best_fp2(self, x, y, X_others):
        """
        Find the pair of powers in :attr:`powers` that minimises RSS.

        All ordered pairs ``(p1, p2)`` from :attr:`powers` are tested.
        When ``p1 == p2``, the repeated-power FP2 form is used
        (generating :math:`x^p` and :math:`x^p \\ln x`), stored as a
        single-element list ``[p]``.  When ``p1 != p2``, two independent
        transformations are generated, stored as ``[p1, p2]`` in sorted
        order.

        :param x: Single-variable vector to transform, shape
            ``(n_samples,)``.
        :type x: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param X_others: Already-transformed matrix of all other
            variables.
        :type X_others: numpy.ndarray

        :returns: ``(best_powers, best_rss)`` where ``best_powers`` is a
            one- or two-element list.
        :rtype: tuple[list, float]
        """
        best_powers = None
        best_rss = np.inf

        # Test all pairs of powers
        for p1 in self.powers:
            for p2 in self.powers:
                if p1 == p2:
                    # Repeated power: store as [p1]
                    powers = [p1]
                else:
                    # Different powers: store as [p1, p2] in sorted order
                    powers = sorted([p1, p2])

                rss = self._calculate_fp_rss(x, y, X_others, powers)

                if rss < best_rss:
                    best_rss = rss
                    best_powers = powers

        return best_powers, best_rss

    def _calculate_fp_rss(self, x, y, X_others, powers):
        """
        Compute the residual sum of squares for a candidate FP transformation.

        The transformed features for ``x`` under ``powers`` are
        concatenated with ``X_others``, a linear regression is fitted, and
        the RSS is returned.  Any numerical failure (overflow, log of a
        non-positive value after shifting) returns ``np.inf`` so that the
        calling search loop safely skips invalid transformations.

        :param x: Single-variable vector, shape ``(n_samples,)``.
        :type x: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param X_others: Transformed matrix of other variables.
        :type X_others: numpy.ndarray
        :param powers: Power list passed to :meth:`_apply_fp_transform`.
        :type powers: list

        :returns: RSS of the fitted model, or ``np.inf`` on failure.
        :rtype: float
        """
        try:
            # Apply FP transformation
            X_fp = self._apply_fp_transform(x, powers)

            # Combine with other variables
            if X_others.shape[1] > 0:
                X_combined = np.hstack([X_others, X_fp])
            else:
                X_combined = X_fp

            # Fit model
            model = LinearRegression().fit(X_combined, y)

            # Calculate RSS
            rss = self._calculate_rss(model, X_combined, y)

            return rss

        except (ValueError, RuntimeWarning, FloatingPointError):
            # Invalid transformation (e.g., log of negative, overflow)
            return np.inf

    def _apply_fp_transform(self, x, powers):
        """
        Apply a fractional polynomial transformation to a single variable.

        The variable is first shifted to a strictly positive range if
        necessary (via :meth:`_shift_to_positive`), then the following
        column generation rules are applied:

        - ``len(powers) == 1`` — FP1: returns ``[x^p]`` (one column).
        - ``len(powers) == 2``, ``p1 == p2`` — repeated FP2: returns
          ``[x^p, x^p · ln(x)]`` (two columns).
        - ``len(powers) == 2``, ``p1 != p2`` — distinct FP2: returns
          ``[x^{p1}, x^{p2}]`` (two columns).

        :param x: Variable vector, shape ``(n_samples,)``.
        :type x: numpy.ndarray
        :param powers: One- or two-element list of powers.
        :type powers: list

        :returns: Transformed feature matrix, shape
            ``(n_samples, n_terms)`` where ``n_terms`` is 1 (FP1) or 2
            (FP2).
        :rtype: numpy.ndarray

        :raises ValueError: If ``len(powers)`` is not 1 or 2.
        """
        x_shifted = self._shift_to_positive(x)

        if len(powers) == 1:
            power = powers[0]

            # Check if this is a repeated FP2 (will have 2 columns)
            # For now, treat single power as FP1
            term1 = self._apply_single_power(x_shifted, power)
            return term1.reshape(-1, 1)

        elif len(powers) == 2:
            p1, p2 = powers

            if p1 == p2:
                # Repeated FP2: x^p and x^p·ln(x)
                term1 = self._apply_single_power(x_shifted, p1)
                term2 = term1 * np.log(x_shifted)
                return np.column_stack([term1, term2])
            else:
                # Different powers: x^p1 and x^p2
                term1 = self._apply_single_power(x_shifted, p1)
                term2 = self._apply_single_power(x_shifted, p2)
                return np.column_stack([term1, term2])

        else:
            raise ValueError(f"Invalid powers length: {len(powers)}")

    def _apply_single_power(self, x, power):
        """
        Apply a single fractional power to a strictly-positive variable.

        Power 0 is interpreted as ln(x) by MFP convention (Royston &
        Sauerbrei, 2008, §2.3).

        :param x: Strictly-positive variable vector.
        :type x: numpy.ndarray
        :param power: Transformation power.  0 → ln(x).
        :type power: float

        :returns: Transformed values.
        :rtype: numpy.ndarray
        """
        if power == 0:
            return np.log(x)
        else:
            return np.power(x, power)

    def _shift_to_positive(self, x):
        """
        Shift ``x`` to a strictly positive range if required.

        If all values are already positive, ``x`` is returned unchanged.
        Otherwise, ``x + |min(x)| + 0.1`` is returned.  This guarantees
        that logarithms and fractional powers are well-defined, at the
        cost of a constant shift in the predictor's scale.

        :param x: Variable vector.
        :type x: numpy.ndarray

        :returns: ``x`` if already positive; ``x + shift`` otherwise.
        :rtype: numpy.ndarray
        """
        if np.all(x > 0):
            return x

        # Shift by minimum + small constant
        shift = abs(x.min()) + 0.1
        return x + shift

    def _apply_transforms(self, X, transforms):
        """
        Construct the full transformed design matrix from raw predictors.

        Variables are processed in ascending index order.  Excluded
        variables contribute no columns.  The outputs of
        :meth:`_apply_fp_transform` for each retained variable are
        concatenated horizontally.

        :param X: Raw predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param transforms: Transformation dict
            ``{var_idx: (transform_type, powers)}``.  ``transform_type``
            may be ``'linear'``, ``'fp1'``, ``'fp2'``, or ``'excluded'``.
        :type transforms: dict

        :returns: Transformed design matrix, shape
            ``(n_samples, n_transformed_features)``.  Returns a zero-column
            matrix if all variables are excluded or ``transforms`` is empty.
        :rtype: numpy.ndarray
        """
        if len(transforms) == 0:
            return np.zeros((X.shape[0], 0))

        feature_list = []

        for var_idx in sorted(transforms.keys()):
            transform_type, powers = transforms[var_idx]

            if transform_type == 'excluded':
                continue

            x = X[:, var_idx]
            X_fp = self._apply_fp_transform(x, powers)
            feature_list.append(X_fp)

        if len(feature_list) == 0:
            return np.zeros((X.shape[0], 0))

        return np.hstack(feature_list)

    def _likelihood_ratio_test(self, rss0, rss1, df0, df1, n):
        """
        Likelihood ratio test for nested Gaussian linear models.

        Under the assumption of i.i.d. Gaussian errors, the log-likelihood
        simplifies and the LRT statistic becomes:

        .. math::

            \\Lambda = n \\bigl(\\ln(\\text{RSS}_0) - \\ln(\\text{RSS}_1)\\bigr)

        Under H₀ (the simpler model is correct),
        :math:`\\Lambda \\sim \\chi^2(k_1 - k_0)`.

        :param rss0: Residual sum of squares for the **simpler** model
            (fewer parameters).
        :type rss0: float
        :param rss1: Residual sum of squares for the **more complex**
            model.
        :type rss1: float
        :param df0: Number of parameters (including intercept) in the
            simpler model.
        :type df0: int
        :param df1: Number of parameters in the more complex model.
        :type df1: int
        :param n: Number of observations.
        :type n: int

        :returns: p-value.  Returns 1.0 if ``df1 <= df0`` (models have
            equal complexity).
        :rtype: float
        """
        # Avoid division by zero or log of zero
        if rss1 <= 0:
            rss1 = 1e-10
        if rss0 <= 0:
            rss0 = 1e-10

        # Likelihood ratio statistic
        lr_stat = n * (np.log(rss0) - np.log(rss1))

        # Degrees of freedom for chi-square test
        df_diff = df1 - df0

        if df_diff <= 0:
            return 1.0  # Models have same complexity

        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

        return p_value

    def _calculate_rss(self, model, X, y):
        """
        Compute the residual sum of squares for a fitted model.

        :param model: A fitted sklearn-compatible regressor with a
            ``predict`` method.
        :param X: Feature matrix used for prediction.
        :type X: numpy.ndarray
        :param y: True target values.
        :type y: numpy.ndarray

        :returns: :math:`\\sum_i (y_i - \\hat{y}_i)^2`
        :rtype: float
        """
        y_pred = model.predict(X)
        residuals = y - y_pred
        return np.sum(residuals ** 2)

    def _check_convergence(self, current_transforms, previous_transforms):
        """
        Determine whether the algorithm has converged.

        Convergence is declared when the transformation dict from the
        current cycle is identical to that from the previous cycle —
        i.e. the same variables are retained with the same transformation
        types and the same power lists.

        :param current_transforms: Transformations at the end of the
            current cycle.
        :type current_transforms: dict
        :param previous_transforms: Transformations at the end of the
            preceding cycle.  ``None`` on the first cycle.
        :type previous_transforms: dict or None

        :returns: ``True`` if transformations are unchanged; ``False``
            otherwise.
        :rtype: bool
        """
        if previous_transforms is None:
            return False

        # Check if same variables present
        if set(current_transforms.keys()) != set(previous_transforms.keys()):
            return False

        # Check if same transformations
        for var_idx in current_transforms.keys():
            curr_type, curr_powers = current_transforms[var_idx]
            prev_type, prev_powers = previous_transforms[var_idx]

            if curr_type != prev_type:
                return False

            if curr_powers != prev_powers:
                return False

        return True

    # =========================================================================
    # Final Model Fitting
    # =========================================================================


    def _fit_final_model(self, X, y, transforms):
        """
        Fit the final linear regression on the fully transformed design matrix.

        Sets :attr:`model_`, :attr:`coef_`, and :attr:`intercept_`.  If
        the transformed matrix has no columns (all variables excluded), an
        intercept-only model is stored.

        :param X: Raw predictor matrix.
        :type X: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param transforms: Final transformation dict from the last cycle.
        :type transforms: dict
        """
        X_transformed = self._apply_transforms(X, transforms)

        if X_transformed.shape[1] == 0:
            # No variables - intercept only
            self.model_ = LinearRegression()
            self.model_.fit(np.ones((len(y), 1)), y)
            self.coef_ = np.array([])
            self.intercept_ = y.mean()
        else:
            self.model_ = LinearRegression()
            self.model_.fit(X_transformed, y)
            self.coef_ = self.model_.coef_
            self.intercept_ = self.model_.intercept_


    def _fit_with_fixed_transforms(self, X, y, fixed_transforms):
        """
        Fit a linear regression using pre-specified transformation structure.

        This pathway is activated when :attr:`fixed_transforms` is provided
        at construction time.  BE and FSP are skipped entirely; the
        transformation structure is taken as given and only the
        coefficients are estimated from ``(X, y)``.

        This is essential for **cross-dataset coefficient comparison**: the
        transformation structure is discovered once on the conventional
        dataset and then re-applied to alternative-fuel datasets, so that
        coefficient differences reflect true population differences rather
        than differences in selected transformations.

        :param X: Training predictor matrix.
        :type X: numpy.ndarray
        :param y: Training target vector.
        :type y: numpy.ndarray
        :param fixed_transforms: Pre-specified transformations in MFP
            format: ``{var_idx: (transform_type, powers)}``.
        :type fixed_transforms: dict

        :returns: ``self`` (fitted estimator).
        :rtype: MFPRegressor
        """
        n_samples = X.shape[0]

        if self.verbose:
            print("=" * 80)
            print("MFP WITH FIXED TRANSFORMATIONS")
            print("=" * 80)
            print(f"Samples: {n_samples}")
            print(f"Using pre-specified transformations (no discovery)")
            print("=" * 80)

        # Store the fixed transforms as final transforms
        self.transforms_ = fixed_transforms.copy()

        # Skip BE and FSP - just fit with provided structure
        self._fit_final_model(X, y, fixed_transforms)

        # Set cycles to 0 since we didn't run the algorithm
        self.n_cycles_ = 0
        self.convergence_history_ = []

        if self.verbose:
            print("\nFitted Model with Fixed Transformations:")
            print("-" * 80)
            transformations = self.get_selected_transformations()
            for var_name, transform_str in transformations.items():
                print(f"  {var_name:20s}: {transform_str}")
            print("=" * 80)

        return self

    # =========================================================================
    # Reporting and Display
    # =========================================================================

    def get_selected_transformations(self):
        """
        Return a human-readable description of each variable's transformation.

        :returns: ``{feature_name: description}`` for every variable that
            was not excluded.  For excluded variables the value is
            ``'EXCLUDED'``.
        :rtype: dict[str, str]

        :raises RuntimeError: If the model has not been fitted.
        """
        if self.transforms_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        transformations = {}

        for var_idx, (transform_type, powers) in self.transforms_.items():
            var_name = self.feature_names_in_[var_idx]

            if transform_type == 'excluded':
                transformations[var_name] = "EXCLUDED"
            elif transform_type == 'linear':
                transformations[var_name] = var_name
            elif transform_type == 'fp1':
                power = powers[0]
                power_str = self._format_power(power).replace('x', var_name)
                transformations[var_name] = power_str
            elif transform_type == 'fp2':
                if len(powers) == 1:
                    # Repeated
                    power = powers[0]
                    power_str1 = self._format_power(power).replace('x', var_name)
                    if power == 0:
                        power_str2 = f"ln({var_name})²"
                    else:
                        power_str2 = f"{power_str1}·ln({var_name})"
                    transformations[var_name] = f"{power_str1} + {power_str2}"
                else:
                    # Different powers
                    p1, p2 = powers
                    power_str1 = self._format_power(p1).replace('x', var_name)
                    power_str2 = self._format_power(p2).replace('x', var_name)
                    transformations[var_name] = f"{power_str1} + {power_str2}"

        return transformations

    def get_model_summary(self):
        """
        Return a formatted text summary of the fitted model.

        Includes convergence information, the intercept, and for each
        retained variable: the transformation description and the fitted
        coefficient(s).

        :returns: Multi-line summary string.
        :rtype: str
        """
        if self.model_ is None:
            return "Model not fitted yet."

        lines = []
        lines.append("=" * 80)
        lines.append("MULTIVARIABLE FRACTIONAL POLYNOMIALS - MODEL SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nConverged after {self.n_cycles_} cycles")
        lines.append(f"Intercept: {self.intercept_:.4f}")

        lines.append("\nSelected Transformations:")
        lines.append("-" * 80)

        transformations = self.get_selected_transformations()

        if len(transformations) == 0:
            lines.append("  (Intercept-only model)")
        else:
            coef_idx = 0
            for var_idx in sorted(self.transforms_.keys()):
                var_name = self.feature_names_in_[var_idx]
                transform_type, powers = self.transforms_[var_idx]

                if transform_type == 'excluded':
                    lines.append(f"  {var_name:20s}: EXCLUDED")
                    continue

                transform_str = transformations[var_name]

                # Get coefficients for this variable
                n_terms = 1 if transform_type in ['linear', 'fp1'] else 2

                if coef_idx < len(self.coef_):
                    if n_terms == 1:
                        coef_str = f"β={self.coef_[coef_idx]:.4f}"
                        coef_idx += 1
                    else:
                        coef_str = f"β₁={self.coef_[coef_idx]:.4f}, β₂={self.coef_[coef_idx + 1]:.4f}"
                        coef_idx += 2

                    lines.append(f"  {var_name:20s}: {transform_str:40s} {coef_str}")
                else:
                    lines.append(f"  {var_name:20s}: {transform_str}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def _format_power(self, power):
        """
        Format a single power value as a human-readable string.

        Special cases use conventional mathematical notation:

        - 0   → ``ln(x)``
        - 0.5 → ``√x``
        - -0.5 → ``1/√x``
        - -1  → ``1/x``
        - 1   → ``x``
        - integer *n* → ``x^n``
        - other → ``x^{power}``

        :param power: Power value to format.
        :type power: float

        :returns: Formatted string with ``'x'`` as the placeholder
            variable name.
        :rtype: str
        """
        if power == 0:
            return "ln(x)"
        elif power == 0.5:
            return "√x"
        elif power == -0.5:
            return "1/√x"
        elif power == -1:
            return "1/x"
        elif power == 1:
            return "x"
        elif power == int(power):
            return f"x^{int(power)}"
        else:
            return f"x^{power}"