"""
Model Descriptor — Structure-Preserving Model Record
=====================================================

Role in the Architecture
-------------------------
:class:`ModelDescriptor` is the **persistent record** of a fitted parametric
model.  It sits between the fitting layer
(:class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`)
and the analysis layer
(:class:`~knowledge_layer.parametric_regression.parametric_cross_dataset_analysis.ParametricCrossDatasetAnalyzer`),
serving three distinct purposes:

1. **Structure storage** — captures *what the model is* (transformations,
   active features, regularisation parameters) separately from *how well
   it fits* (metrics).

2. **Structure-preserving refitting** — re-estimates coefficients on a
   new dataset while keeping the model structure fixed.  This is the
   critical operation for cross-dataset coefficient comparison.

3. **Formula display** — delegates all formula rendering to
   :class:`~knowledge_layer.parametric_regression.parametric_formula_renderer.ParametricFormulaRenderer`,
   keeping this class focused on structure and refitting.

What Changed in This Refactor
------------------------------
All formula rendering logic has been extracted into
:mod:`parametric_formula_renderer`.  Specifically, the following methods
were removed from this class and now live in
:class:`~ParametricFormulaRenderer`:

- ``_get_mfp_formula``
- ``_get_polynomial_formula``
- ``_get_linear_formula``
- ``_get_simple_nonlinear_formula``
- ``_get_log_transform_formula``
- ``_format_power``
- ``_format_power_name``
- ``_format_coefficient``
- ``_simplify_power``
- ``_get_fp2_second_term_name``
- ``_get_transform_name``

The public interface is **unchanged**: :meth:`get_formula_string`,
:meth:`refit`, :meth:`from_fitted_model`, :meth:`to_json`, and
:meth:`from_json` all behave identically to the pre-refactor version.

Internal methods that still need to construct coefficient names
(``_update_from_fitted_mfp``, ``from_fitted_model``) now call the
renderer's public utilities directly, e.g.
``ParametricFormulaRenderer.format_power_name(var, power)``.

Backwards Compatibility
-----------------------
Thin delegating wrappers are provided for the private formatting methods
that were previously on :class:`ModelDescriptor` itself.  This ensures
that any code which called e.g. ``descriptor._format_power(...)`` directly
(outside the public API) continues to work without modification.
These wrappers are marked as deprecated and will be removed in a future
clean-up pass.

Why Structure Preservation Matters
------------------------------------
The central question of this research is whether regression relationships
discovered on conventionally fuelled ships transfer to alternatively
fuelled vessels.  Answering this meaningfully requires that the *same*
model structure is applied to all datasets — if MFP selects ``√DWT`` on
conventional ships but ``DWT²`` on alternative ships, any difference in
coefficients conflates structural change with parameter change.

:meth:`refit` solves this by re-estimating coefficients from a new
``(X, y)`` pair while holding the transformation / feature-selection
structure fixed.

Data Flow
---------
1. :class:`ParametricModelSelector` fits
   ``model, prep = ModelFactory.create_and_fit(...)`` on the conventional
   training set.
2. ``descriptor = ModelDescriptor.from_fitted_model(model, prep, ...)``
   extracts the full model record.
3. :class:`ParametricCrossDatasetAnalyzer` calls
   ``model_alt, prep_alt = descriptor.refit(X_alt, y_alt,
   force_structure=True)``.
4. ``ModelFactory.predict(model_alt, prep_alt, X_alt,
   descriptor.model_type)`` generates predictions using the conventional
   structure with re-estimated coefficients.
"""

from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np

from knowledge_layer.parametric_regression.parametric_formula_render import (
    ParametricFormulaRenderer,
)


@dataclass
class ModelDescriptor:
    """
    Complete, serialisable record of a fitted parametric model.

    Stores both the structural information needed to refit the model on new
    data (preserving comparability) and the display information needed to
    reconstruct human-readable formulas.  Formula rendering is delegated
    entirely to
    :class:`~knowledge_layer.parametric_regression.parametric_formula_renderer.ParametricFormulaRenderer`.

    Instances are normally created via :meth:`from_fitted_model` rather
    than directly, since that class method handles extraction logic for
    every supported model type.

    :param model_type: Model-type identifier string, e.g. ``'mfp'``,
        ``'polynomial-2-lasso'``, ``'power'``.
    :type model_type: str

    :param independent_vars: Names of the predictor variables in the order
        they appear as columns in the training ``X`` matrix.
    :type independent_vars: list[str]

    :param dependent_var: Name of the target variable.
    :type dependent_var: str

    :param transformations: MFP transformation structure.
        Format: ``{var_idx: (transform_type, powers)}``.  ``None`` for
        non-MFP models.
    :type transformations: dict[int, tuple[str, list[float]]], optional

    :param polynomial_degree: Degree for polynomial models.  Inferred
        automatically from ``model_type`` if not supplied explicitly.
    :type polynomial_degree: int, optional

    :param piecewise_bins: Bin-edge specification for piecewise models.
    :type piecewise_bins: dict, optional

    :param piecewise_binning_variable: Name of the variable used for
        binning in piecewise models.
    :type piecewise_binning_variable: str, optional

    :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        Inferred automatically from ``model_type`` if not supplied.
    :type regularization_type: str, optional

    :param regularization_alpha: Regularisation strength used during
        fitting.  Stored so that :meth:`_refit_lasso_fixed_alpha` can
        apply the same penalty on a new dataset.
    :type regularization_alpha: float, optional

    :param selected_alpha: Alpha value chosen by cross-validation for
        ``'-cv'`` model variants.
    :type selected_alpha: float, optional

    :param active_features: Names of features with non-zero Lasso
        coefficients.
    :type active_features: list[str], optional

    :param active_feature_mask: Boolean array marking non-zero Lasso
        coefficients.
    :type active_feature_mask: numpy.ndarray, optional

    :param effective_df: Effective degrees of freedom for AIC / BIC.
        Computed automatically by :meth:`_calculate_effective_df` when
        not supplied.
    :type effective_df: int, optional

    :param coefficients: Fitted coefficient values keyed by descriptive
        feature names, e.g. ``{'√DWT': 0.523, '1/GT': -0.321}``.
    :type coefficients: dict[str, float], optional

    :param intercept: Fitted intercept value.
    :type intercept: float, optional

    :param n_samples: Number of samples used during the original fit.
    :type n_samples: int, optional

    :param metrics: Performance metrics dict.
    :type metrics: dict[str, float]

    :param fitting_date: ISO-format datetime string.
    :type fitting_date: str, optional

    :param notes: Free-text annotation field.
    :type notes: str, optional

    :param metadata: Model-type-specific supplementary information.
    :type metadata: dict[str, Any]
    """

    # =========================================================================
    # Core Identity
    # =========================================================================

    model_type: str
    independent_vars: List[str]
    dependent_var: str

    # =========================================================================
    # Model-Specific Structure
    # =========================================================================

    transformations: Optional[Dict[int, Tuple[str, List[float]]]] = None
    polynomial_degree: Optional[int] = None
    piecewise_bins: Optional[Dict[str, List[float]]] = None
    piecewise_binning_variable: Optional[str] = None

    # =========================================================================
    # Regularisation
    # =========================================================================

    regularization_type: Optional[str] = None
    regularization_alpha: Optional[float] = None
    selected_alpha: Optional[float] = None
    active_features: Optional[List[str]] = None
    active_feature_mask: Optional[np.ndarray] = None
    effective_df: Optional[int] = None

    # =========================================================================
    # Fitted Parameters
    # =========================================================================

    coefficients: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None

    # =========================================================================
    # Metadata
    # =========================================================================

    n_samples: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    fitting_date: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __post_init__(self):
        """
        Validate and infer derived fields immediately after construction.

        Three inference steps:

        1. If ``model_type`` starts with ``'polynomial-'`` and
           ``polynomial_degree`` is ``None``, the degree is parsed from
           the type string.
        2. ``regularization_type`` is inferred from ``model_type`` keywords.
        3. ``effective_df`` is computed via :meth:`_calculate_effective_df`
           if not supplied.

        :raises ValueError: If the polynomial degree cannot be parsed.
        """
        if self.model_type.startswith('polynomial-') and self.polynomial_degree is None:
            try:
                self.polynomial_degree = int(self.model_type.split('-')[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"Cannot determine polynomial degree from '{self.model_type}'"
                )

        if 'lasso' in self.model_type:
            self.regularization_type = 'lasso'
        elif 'ridge' in self.model_type:
            self.regularization_type = 'ridge'

        if self.effective_df is None:
            self.effective_df = self._calculate_effective_df()

    # =========================================================================
    # Parameter Counting
    # =========================================================================

    def _calculate_effective_df(self) -> int:
        """
        Compute the effective number of free parameters.

        Rules applied in priority order:

        - **Lasso**: count of non-zero coefficients + 1 (intercept).
        - **MFP**: 1 (intercept) + 1 per *linear* / *FP1* term + 2 per
          *FP2* term.
        - **Polynomial**: total expansion features via combinatorial
          formula.
        - **Linear / Ridge / other**: ``len(independent_vars) + 1``.

        :returns: Effective number of free parameters.
        :rtype: int
        """
        if self.regularization_type == 'lasso':
            if self.active_features is not None:
                return len(self.active_features)
            if self.active_feature_mask is not None:
                return int(np.sum(self.active_feature_mask))
            if self.coefficients is not None:
                non_zero = sum(1 for v in self.coefficients.values()
                               if abs(v) > 1e-10)
                return non_zero + (1 if self.intercept is not None else 0)

        if self.model_type == 'mfp' and self.transformations:
            n_terms = 1  # intercept
            for _, (t_type, _) in self.transformations.items():
                if t_type == 'excluded':
                    continue
                n_terms += 2 if t_type == 'fp2' else 1
            return n_terms

        if self.polynomial_degree is not None:
            n_input = len(self.independent_vars)
            return sum(
                comb(n_input + d - 1, d)
                for d in range(self.polynomial_degree + 1)
            )

        return len(self.independent_vars) + 1

    def get_effective_parameters(self) -> int:
        """
        Return the effective number of free parameters for AIC / BIC.

        :returns: Effective degrees of freedom.
        :rtype: int
        """
        return self.effective_df if self.effective_df is not None else 0

    # =========================================================================
    # Formula Reconstruction — delegated to ParametricFormulaRenderer
    # =========================================================================

    def get_formula_string(
        self,
        include_coefficients: bool = True,
        notation: str = 'math',
        only_active: bool = True,
        sci_threshold: float = 1e-3,
        sci_max: float = 1e4,
    ) -> str:
        """
        Return a human-readable formula for the fitted model.

        All rendering logic is handled by
        :class:`~ParametricFormulaRenderer`.  The public interface of this
        method is unchanged from the pre-refactor version.

        :param include_coefficients: Embed actual coefficient values when
            ``True``; use generic β symbols when ``False``.
        :type include_coefficients: bool

        :param notation: Output style — ``'math'`` (Unicode), ``'latex'``
            (LaTeX source), or ``'python'`` (valid Python expression).
        :type notation: str

        :param only_active: For Lasso models, suppress zero-coefficient
            terms.
        :type only_active: bool

        :param sci_threshold: Use scientific notation when
            ``|coef| < sci_threshold``.
        :type sci_threshold: float

        :param sci_max: Use scientific notation when ``|coef| > sci_max``.
        :type sci_max: float

        :returns: Formatted formula string.
        :rtype: str
        """
        return ParametricFormulaRenderer.render(
            model_type=self.model_type,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            transformations=self.transformations,
            coefficients=self.coefficients,
            intercept=self.intercept,
            regularization_type=self.regularization_type,
            active_features=self.active_features,
            polynomial_degree=self.polynomial_degree,
            metadata=self.metadata,
            include_coefficients=include_coefficients,
            notation=notation,
            only_active=only_active,
            sci_threshold=sci_threshold,
            sci_max=sci_max,
        )

    # =========================================================================
    # Coefficient Names
    # =========================================================================

    def get_coefficient_names(self, only_active: bool = True) -> List[str]:
        """
        Return descriptive names for each coefficient in the model.

        Names reflect actual transformations, e.g.
        ``['intercept', '√DWT', '1/GT', 'GT²']`` for an MFP model.

        :param only_active: For Lasso, suppress zero-coefficient features.
        :type only_active: bool

        :returns: Ordered list of coefficient names starting with
            ``'intercept'``.
        :rtype: list[str]
        """
        if self.model_type == 'mfp':
            return self._get_mfp_coefficient_names()
        elif self.model_type.startswith('polynomial'):
            return self._get_polynomial_coefficient_names(only_active)
        else:
            names = ['intercept'] if self.intercept is not None else []
            if only_active and self.regularization_type == 'lasso':
                if self.active_features:
                    names.extend(self.active_features)
                else:
                    names.extend([
                        v for v in self.independent_vars
                        if self.coefficients
                        and abs(self.coefficients.get(v, 0)) > 1e-10
                    ])
            else:
                names.extend(self.independent_vars)
            return names

    def _get_mfp_coefficient_names(self) -> List[str]:
        """
        Return coefficient names for an MFP model, ordered by variable
        index.  Uses :class:`ParametricFormulaRenderer` utilities for
        consistent naming with the formula renderer.

        :returns: List starting with ``'intercept'``.
        :rtype: list[str]
        """
        names = ['intercept']
        for var_idx, var_name in enumerate(self.independent_vars):
            if var_idx not in (self.transformations or {}):
                continue
            t_type, powers = self.transformations[var_idx]
            if t_type == 'excluded':
                continue
            elif t_type == 'linear':
                names.append(var_name)
            elif t_type == 'fp1':
                names.append(
                    ParametricFormulaRenderer.format_power_name(var_name, powers[0])
                )
            elif t_type == 'fp2':
                if len(powers) == 1:
                    names.append(
                        ParametricFormulaRenderer.format_power_name(var_name, powers[0])
                    )
                    names.append(
                        ParametricFormulaRenderer.get_fp2_second_term_name(
                            var_name, powers[0]
                        )
                    )
                else:
                    names.extend(
                        ParametricFormulaRenderer.format_power_name(var_name, p)
                        for p in powers
                    )
        return names

    def _get_polynomial_coefficient_names(self, only_active: bool) -> List[str]:
        """
        Return coefficient names for a polynomial model.

        :param only_active: Suppress zero Lasso terms.
        :type only_active: bool

        :returns: List of coefficient names.
        :rtype: list[str]
        """
        if not self.coefficients:
            return ['intercept'] + [
                f"poly_term_{i}" for i in range(self.effective_df - 1)
            ]

        names = ['intercept'] if self.intercept is not None else []
        for name, val in self.coefficients.items():
            if name == 'intercept':
                continue
            if only_active and self.regularization_type == 'lasso':
                if self.active_features and name not in self.active_features:
                    continue
                if abs(val) < 1e-10:
                    continue
            names.append(name)
        return names

    # =========================================================================
    # Refitting
    # =========================================================================

    def refit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        force_structure: bool = True,
    ):
        """
        Re-estimate model coefficients on a new dataset, optionally holding
        the model structure fixed.

        This is the primary method used by
        :class:`~ParametricCrossDatasetAnalyzer` to apply a model
        discovered on conventional ships to an alternative-fuel dataset.

        When ``force_structure=True``:

        - **MFP**: the exact ``{var_idx: (transform_type, powers)}`` mapping
          stored in :attr:`transformations` is preserved.
        - **Lasso**: :attr:`regularization_alpha` is fixed.
        - **Nonlinear**: the same functional form is used (curve_fit
          re-optimises only the parameters).
        - **Linear / Ridge / other**: re-fitted freely via
          :meth:`~ModelFactory.create_and_fit`.

        .. warning::

            :meth:`refit` **mutates** :attr:`intercept` and
            :attr:`coefficients` in-place.  Capture their values before
            calling if you need to compare both datasets.

        :param X: New predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: New target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param force_structure: Preserve model structure when ``True``
            (default).  Passing ``False`` for MFP allows fresh
            transformation discovery (breaks coefficient comparability).
        :type force_structure: bool

        :returns: ``(model, preprocessor)`` ready for use with
            :meth:`~ModelFactory.predict`.
        :rtype: tuple

        :raises ValueError: If :attr:`transformations` is ``None`` when
            MFP structure-forcing is requested.
        """
        if self.model_type == 'mfp':
            return (
                self._refit_mfp_fixed_structure(X, y)
                if force_structure
                else self._refit_mfp_discover(X, y)
            )
        elif self.model_type.endswith('lasso') and force_structure:
            return self._refit_lasso_fixed_alpha(X, y)
        elif self.model_type in [
            'exponential', 'exponential-offset', 'power',
            'power-offset', 'logarithmic', 'logarithmic-power',
        ]:
            return self._refit_nonlinear_model(X, y)
        else:
            from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
            model, preprocessor = ModelFactory.create_and_fit(self.model_type, X, y)
            self._update_from_fitted_model(model, preprocessor)
            return model, preprocessor

    def _refit_mfp_fixed_structure(self, X: np.ndarray, y: np.ndarray):
        """
        Refit an MFP model using the transformation structure stored in
        :attr:`transformations`, bypassing BE / FSP discovery.

        :param X: New predictor matrix.
        :type X: numpy.ndarray
        :param y: New target vector.
        :type y: numpy.ndarray

        :returns: ``(mfp, None)``
        :rtype: tuple[MFPRegressor, None]

        :raises ValueError: If :attr:`transformations` is ``None``.
        """
        from knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class import MFPRegressor

        if self.transformations is None:
            raise ValueError(
                "Cannot force MFP structure: no transformations stored.  "
                "Create the descriptor from a fitted model first."
            )

        alpha1 = (self.metadata or {}).get('alpha1', 0.05)
        alpha2 = (self.metadata or {}).get('alpha2', 0.05)

        mfp = MFPRegressor(
            alpha1=alpha1,
            alpha2=alpha2,
            fixed_transforms=self.transformations,
            verbose=False,
        )
        mfp.fit(X, y, feature_names=self.independent_vars)
        self._update_from_fitted_mfp(mfp)
        return mfp, None

    def _refit_mfp_discover(self, X: np.ndarray, y: np.ndarray):
        """
        Refit an MFP model with fresh transformation discovery.

        .. warning::

            This **breaks coefficient comparability** across datasets.
            Use only when you explicitly want an independent structure for
            this dataset.

        :param X: New predictor matrix.
        :type X: numpy.ndarray
        :param y: New target vector.
        :type y: numpy.ndarray

        :returns: ``(mfp, None)``
        :rtype: tuple[MFPRegressor, None]
        """
        from knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class import MFPRegressor

        alpha1 = (self.metadata or {}).get('alpha1', 0.05)
        alpha2 = (self.metadata or {}).get('alpha2', 0.05)

        mfp = MFPRegressor(alpha1=alpha1, alpha2=alpha2, verbose=False)
        mfp.fit(X, y, feature_names=self.independent_vars)
        self.transformations = mfp.transforms_.copy()
        self._update_from_fitted_mfp(mfp)
        return mfp, None

    def _refit_lasso_fixed_alpha(self, X: np.ndarray, y: np.ndarray):
        """
        Refit a Lasso or polynomial-Lasso model using the stored
        regularisation strength.

        Holding alpha fixed provides a fair cross-dataset comparison.

        :param X: New predictor matrix.
        :type X: numpy.ndarray
        :param y: New target vector.
        :type y: numpy.ndarray

        :returns: ``(model, preprocessor)``
        :rtype: tuple
        """
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import PolynomialFeatures

        alpha = self.regularization_alpha or 0.1

        if self.model_type.startswith('polynomial'):
            poly = PolynomialFeatures(degree=self.polynomial_degree)
            X_in = poly.fit_transform(X)
        else:
            X_in = X
            poly = None

        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_in, y)
        self._update_from_fitted_lasso(model, poly)
        return model, poly

    def _refit_nonlinear_model(self, X: np.ndarray, y: np.ndarray):
        """
        Refit a univariate nonlinear model by re-running the curve-fit
        optimiser on the same functional form.

        :param X: New predictor matrix.
        :type X: numpy.ndarray
        :param y: New target vector.
        :type y: numpy.ndarray

        :returns: ``(model, None)``
        :rtype: tuple[tuple, None]
        """
        from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

        model, preprocessor = ModelFactory.create_and_fit(self.model_type, X, y)
        self._update_from_nonlinear_model(model)
        return model, preprocessor

    # =========================================================================
    # Internal Update Methods
    # =========================================================================

    def _update_from_fitted_model(self, model, preprocessor):
        """
        Update :attr:`intercept` and :attr:`coefficients` from a generic
        sklearn estimator after a plain refit.

        :param model: Fitted sklearn estimator.
        :param preprocessor: Fitted
            :class:`~sklearn.preprocessing.PolynomialFeatures` or ``None``.
        """
        self.intercept = getattr(model, 'intercept_', None)
        if hasattr(model, 'coef_'):
            if preprocessor is not None:
                names = preprocessor.get_feature_names_out(self.independent_vars)
                self.coefficients = dict(zip(names, model.coef_))
            else:
                self.coefficients = dict(zip(self.independent_vars, model.coef_))

    def _update_from_fitted_mfp(self, mfp):
        """
        Synchronise :attr:`intercept`, :attr:`coefficients`, and
        :attr:`metadata` from a freshly fitted
        :class:`~MFPRegressor`.

        Coefficient names are built using
        :meth:`ParametricFormulaRenderer.format_power_name` and
        :meth:`ParametricFormulaRenderer.get_fp2_second_term_name` to
        guarantee they match the keys looked up by the formula renderer.

        :param mfp: Fitted :class:`~MFPRegressor`.
        """
        self.intercept = float(mfp.intercept_)
        self.coefficients = {}

        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}

        self.metadata.update({
            'n_cycles': getattr(mfp, 'n_cycles_', 0),
            'alpha1':   getattr(mfp, 'alpha1', 0.05),
            'alpha2':   getattr(mfp, 'alpha2', 0.05),
        })

        coef_idx = 0
        for var_idx in sorted(self.transformations):
            t_type, powers = self.transformations[var_idx]
            if t_type == 'excluded':
                continue

            var_name = (
                mfp.feature_names_in_[var_idx]
                if (getattr(mfp, 'feature_names_in_', None))
                else (
                    self.independent_vars[var_idx]
                    if var_idx < len(self.independent_vars)
                    else f'x{var_idx}'
                )
            )

            if t_type == 'linear':
                self.coefficients[var_name] = float(mfp.coef_[coef_idx])
                coef_idx += 1

            elif t_type == 'fp1':
                name = ParametricFormulaRenderer.format_power_name(var_name, powers[0])
                self.coefficients[name] = float(mfp.coef_[coef_idx])
                coef_idx += 1

            elif t_type == 'fp2':
                if len(powers) == 1:
                    n1 = ParametricFormulaRenderer.format_power_name(var_name, powers[0])
                    self.coefficients[n1] = float(mfp.coef_[coef_idx])
                    coef_idx += 1
                    n2 = ParametricFormulaRenderer.get_fp2_second_term_name(
                        var_name, powers[0]
                    )
                    self.coefficients[n2] = float(mfp.coef_[coef_idx])
                    coef_idx += 1
                else:
                    for p in powers:
                        n = ParametricFormulaRenderer.format_power_name(var_name, p)
                        self.coefficients[n] = float(mfp.coef_[coef_idx])
                        coef_idx += 1

    def _update_from_fitted_lasso(self, model, preprocessor):
        """
        Update Lasso-specific attributes from a freshly fitted
        :class:`~sklearn.linear_model.Lasso`.

        Re-computes :attr:`active_features`, :attr:`active_feature_mask`,
        and :attr:`effective_df` from the new coefficient array.

        :param model: Fitted :class:`~sklearn.linear_model.Lasso`.
        :param preprocessor: Fitted
            :class:`~sklearn.preprocessing.PolynomialFeatures` or ``None``.
        """
        self.intercept = model.intercept_
        feature_names = (
            preprocessor.get_feature_names_out(self.independent_vars)
            if preprocessor is not None
            else self.independent_vars
        )
        self.coefficients = {}
        active, mask = [], []
        for name, coef in zip(feature_names, model.coef_):
            self.coefficients[name] = coef
            is_active = abs(coef) > 1e-10
            mask.append(is_active)
            if is_active:
                active.append(name)

        self.active_features      = active
        self.active_feature_mask  = np.array(mask)
        self.effective_df         = len(active) + (1 if self.intercept is not None else 0)

    def _update_from_nonlinear_model(self, model):
        """
        Update :attr:`coefficients` and :attr:`metadata` from a
        curve-fit model tuple ``(func_name, popt)``.

        :param model: ``(func_name, popt)`` as returned by
            :class:`~UnivariateNonlinearFitter`.
        :type model: tuple

        :raises ValueError: If ``model`` is not a two-element tuple.
        """
        if not isinstance(model, tuple) or len(model) != 2:
            raise ValueError(
                f"Expected (func_name, popt) tuple, got {type(model)}"
            )
        _, popt = model
        self.coefficients = {}

        if self.model_type in [
            'exponential', 'logarithmic', 'power', 'logarithmic-power',
        ] and len(popt) >= 2:
            self.coefficients = {'a': float(popt[0]), 'b': float(popt[1])}
        elif self.model_type in ['exponential-offset', 'power-offset'] and len(popt) >= 3:
            self.coefficients = {
                'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2]),
            }

        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        self.metadata['popt'] = (
            popt.tolist() if hasattr(popt, 'tolist') else list(popt)
        )
        self.intercept = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_fitted_model(
        cls,
        model,
        preprocessor,
        model_type: str,
        independent_vars: List[str],
        dependent_var: str,
        metrics: Optional[Dict] = None,
    ) -> "ModelDescriptor":
        """
        Construct a :class:`ModelDescriptor` from a fitted model object.

        Inspects the model object and preprocessor and extracts all
        structural information.  Coefficient names are built using
        :class:`ParametricFormulaRenderer` utilities to guarantee
        consistency with formula rendering.

        :param model: Fitted model — sklearn estimator,
            :class:`~MFPRegressor`, or ``(func_name, popt)`` tuple.
        :param preprocessor: Fitted preprocessor or ``None``.
        :param model_type: Model-type identifier string.
        :type model_type: str
        :param independent_vars: Predictor variable names.
        :type independent_vars: list[str]
        :param dependent_var: Target variable name.
        :type dependent_var: str
        :param metrics: Performance metrics to embed.
        :type metrics: dict, optional

        :returns: Fully populated :class:`ModelDescriptor`.
        :rtype: ModelDescriptor

        :raises ValueError: If an :class:`~MFPRegressor` is passed for a
            non-``'mfp'`` model type.
        """
        from knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class import MFPRegressor
        from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV

        # ------------------------------------------------------------------
        # Simple nonlinear — stored as (func_name, popt)
        # ------------------------------------------------------------------
        simple_nonlinear = [
            'exponential', 'exponential-offset',
            'power', 'power-offset',
            'logarithmic', 'logarithmic-power',
        ]
        if model_type in simple_nonlinear:
            if isinstance(model, tuple) and len(model) == 2:
                _, popt = model
                coefficients: Dict[str, float] = {}
                if model_type in [
                    'exponential', 'logarithmic', 'power', 'logarithmic-power',
                ] and len(popt) >= 2:
                    coefficients = {'a': float(popt[0]), 'b': float(popt[1])}
                elif model_type in ['exponential-offset', 'power-offset'] and len(popt) >= 3:
                    coefficients = {
                        'a': float(popt[0]),
                        'b': float(popt[1]),
                        'c': float(popt[2]),
                    }
                return cls(
                    model_type=model_type,
                    independent_vars=independent_vars,
                    dependent_var=dependent_var,
                    coefficients=coefficients,
                    intercept=None,
                    metadata={
                        'popt': popt.tolist()
                        if hasattr(popt, 'tolist') else list(popt)
                    },
                    metrics=metrics or {},
                )

        # ------------------------------------------------------------------
        # MFP
        # ------------------------------------------------------------------
        if model_type == 'mfp':
            if not isinstance(model, MFPRegressor):
                raise ValueError(
                    f"Expected MFPRegressor for model_type='mfp', got {type(model)}"
                )
            if not hasattr(model, 'transforms_'):
                raise ValueError("MFP model not fitted (missing transforms_)")

            transformations = model.transforms_.copy()
            metadata: Dict[str, Any] = {
                'n_cycles':            getattr(model, 'n_cycles_', 0),
                'alpha1':              getattr(model, 'alpha1', 0.05),
                'alpha2':              getattr(model, 'alpha2', 0.05),
                'convergence_history': getattr(model, 'convergence_history_', []),
            }

            if getattr(model, 'feature_names_in_', None):
                metadata['named_transforms'] = {
                    model.feature_names_in_[idx]: {'type': t, 'powers': p}
                    for idx, (t, p) in transformations.items()
                }

            # Map coefficient array → descriptive names via renderer utilities
            coefficients_dict: Dict[str, float] = {}
            intercept = float(model.intercept_) if hasattr(model, 'intercept_') else None

            if hasattr(model, 'coef_') and len(model.coef_) > 0:
                coef_idx = 0
                for var_idx in sorted(transformations):
                    t_type, powers = transformations[var_idx]
                    if t_type == 'excluded':
                        continue

                    var_name = (
                        model.feature_names_in_[var_idx]
                        if getattr(model, 'feature_names_in_', None)
                        else f'x{var_idx}'
                    )

                    if t_type == 'linear':
                        coefficients_dict[var_name] = float(model.coef_[coef_idx])
                        coef_idx += 1

                    elif t_type == 'fp1':
                        name = ParametricFormulaRenderer.format_power_name(
                            var_name, powers[0]
                        )
                        coefficients_dict[name] = float(model.coef_[coef_idx])
                        coef_idx += 1

                    elif t_type == 'fp2':
                        if len(powers) == 1:
                            n1 = ParametricFormulaRenderer.format_power_name(
                                var_name, powers[0]
                            )
                            coefficients_dict[n1] = float(model.coef_[coef_idx])
                            coef_idx += 1
                            n2 = ParametricFormulaRenderer.get_fp2_second_term_name(
                                var_name, powers[0]
                            )
                            coefficients_dict[n2] = float(model.coef_[coef_idx])
                            coef_idx += 1
                        else:
                            for p in powers:
                                n = ParametricFormulaRenderer.format_power_name(
                                    var_name, p
                                )
                                coefficients_dict[n] = float(model.coef_[coef_idx])
                                coef_idx += 1

            return cls(
                model_type='mfp',
                independent_vars=independent_vars,
                dependent_var=dependent_var,
                transformations=transformations,
                coefficients=coefficients_dict,
                intercept=intercept,
                metadata=metadata,
                metrics=metrics or {},
            )

        # ------------------------------------------------------------------
        # Log-transform multivariate
        # ------------------------------------------------------------------
        if model_type in ['power-log-multivariate', 'exponential-log-multivariate']:
            a = float(np.exp(model.intercept_))
            coefficients_dict = {
                f'b_{i}_{v}': float(model.coef_[i])
                for i, v in enumerate(independent_vars)
            }
            return cls(
                model_type=model_type,
                independent_vars=independent_vars,
                dependent_var=dependent_var,
                coefficients=coefficients_dict,
                intercept=a,
                metadata={
                    'log_intercept':  float(model.intercept_),
                    'transform_type': getattr(model, 'transform_type_', None),
                },
                metrics=metrics or {},
            )

        # ------------------------------------------------------------------
        # All remaining sklearn model types
        # ------------------------------------------------------------------
        descriptor = cls(
            model_type=model_type,
            independent_vars=independent_vars,
            dependent_var=dependent_var,
            metrics=metrics or {},
        )

        if isinstance(model, MFPRegressor):
            raise ValueError(
                "MFPRegressor passed for a non-'mfp' model_type.  "
                "Use model_type='mfp'."
            )

        if model_type.startswith('polynomial'):
            descriptor.intercept = model.intercept_
            if preprocessor is not None:
                feature_names = preprocessor.get_feature_names_out(independent_vars)
                descriptor.coefficients = dict(zip(feature_names, model.coef_))

                if isinstance(model, (Lasso, LassoCV)) or 'lasso' in model_type:
                    descriptor.regularization_alpha = getattr(model, 'alpha', None)
                    descriptor.selected_alpha       = getattr(model, 'selected_alpha_', None)
                    active_mask = np.abs(model.coef_) > 1e-10
                    descriptor.active_feature_mask  = active_mask
                    descriptor.active_features      = [
                        n for n, a in zip(feature_names, active_mask) if a
                    ]
                    descriptor.effective_df = len(descriptor.active_features) + 1

                elif isinstance(model, (Ridge, RidgeCV)) or 'ridge' in model_type:
                    descriptor.regularization_alpha = getattr(model, 'alpha', None)
                    descriptor.selected_alpha       = getattr(model, 'selected_alpha_', None)
        else:
            descriptor.intercept = getattr(model, 'intercept_', None)
            if hasattr(model, 'coef_'):
                descriptor.coefficients = dict(zip(independent_vars, model.coef_))

                if isinstance(model, (Lasso, LassoCV)) or 'lasso' in model_type:
                    descriptor.regularization_alpha = getattr(model, 'alpha', None)
                    descriptor.selected_alpha       = getattr(model, 'selected_alpha_', None)
                    active_mask = np.abs(model.coef_) > 1e-10
                    descriptor.active_feature_mask  = active_mask
                    descriptor.active_features      = [
                        v for v, a in zip(independent_vars, active_mask) if a
                    ]
                    descriptor.effective_df = len(descriptor.active_features) + 1

                elif isinstance(model, (Ridge, RidgeCV)) or 'ridge' in model_type:
                    descriptor.regularization_alpha = getattr(model, 'alpha', None)
                    descriptor.selected_alpha       = getattr(model, 'selected_alpha_', None)

        return descriptor

    # =========================================================================
    # Serialisation
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the descriptor to a plain Python dictionary.

        :class:`numpy.ndarray` fields are converted to lists for JSON
        compatibility.

        :returns: Dictionary representation.
        :rtype: dict[str, Any]
        """
        return {
            'model_type':               self.model_type,
            'independent_vars':         self.independent_vars,
            'dependent_var':            self.dependent_var,
            'transformations':          self.transformations,
            'polynomial_degree':        self.polynomial_degree,
            'piecewise_bins':           self.piecewise_bins,
            'piecewise_binning_variable': self.piecewise_binning_variable,
            'regularization_type':      self.regularization_type,
            'regularization_alpha':     self.regularization_alpha,
            'selected_alpha':           self.selected_alpha,
            'active_features':          self.active_features,
            'active_feature_mask':      (
                self.active_feature_mask.tolist()
                if self.active_feature_mask is not None else None
            ),
            'coefficients':             self.coefficients,
            'intercept':                self.intercept,
            'effective_df':             self.effective_df,
            'n_samples':                self.n_samples,
            'metrics':                  self.metrics,
            'fitting_date':             self.fitting_date,
            'notes':                    self.notes,
            'metadata':                 self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDescriptor":
        """
        Deserialise a :class:`ModelDescriptor` from a plain dictionary.

        Handles two JSON-serialisation artefacts:

        - ``active_feature_mask``: restored from list to
          :class:`numpy.ndarray`.
        - ``transformations``: JSON encodes integer keys as strings;
          they are converted back to ``int`` here so that all internal
          lookups work correctly without any caller-side patching.

        :param data: Dictionary as produced by :meth:`to_dict`.
        :type data: dict

        :returns: Reconstructed descriptor.
        :rtype: ModelDescriptor
        """
        data = dict(data)  # shallow copy — avoid mutating caller's dict

        if data.get('transformations') is not None:
            data['transformations'] = {
                int(k): v for k, v in data['transformations'].items()
            }

        if data.get('active_feature_mask') is not None:
            data['active_feature_mask'] = np.array(data['active_feature_mask'])

        return cls(**data)

    def to_json(self, filepath: str):
        """
        Serialise the descriptor to a JSON file.

        :param filepath: Destination file path.
        :type filepath: str
        """
        with open(filepath, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(f"✅ ModelDescriptor saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "ModelDescriptor":
        """
        Load a descriptor from a JSON file written by :meth:`to_json`.

        :param filepath: Path to the JSON file.
        :type filepath: str

        :returns: Reconstructed descriptor.
        :rtype: ModelDescriptor
        """
        with open(filepath) as fh:
            data = json.load(fh)
        descriptor = cls.from_dict(data)
        print(f"✅ ModelDescriptor loaded from {filepath}")
        return descriptor

    # =========================================================================
    # Display
    # =========================================================================

    def __str__(self) -> str:
        lines = [
            "ModelDescriptor:",
            f"  Type      : {self.model_type}",
            f"  Formula   : {self.get_formula_string()}",
            f"  Eff. DF   : {self.effective_df}",
        ]
        if self.regularization_type == 'lasso':
            n_total  = len(self.coefficients) if self.coefficients else 0
            n_active = len(self.active_features) if self.active_features else 0
            lines.append(
                f"  Lasso     : {n_active}/{n_total} features active "
                f"(α={self.regularization_alpha:.4f})"
            )
        if self.metrics:
            r2 = self.metrics.get('r2')
            if r2 is not None:
                lines.append(f"  R²        : {r2:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ModelDescriptor(type='{self.model_type}', "
            f"vars={self.independent_vars}, "
            f"target='{self.dependent_var}', "
            f"df={self.effective_df})"
        )

    # =========================================================================
    # Deprecated delegating wrappers
    # =========================================================================
    # These preserve backwards compatibility for any code that called the
    # private formatting methods directly on the descriptor.
    # They will be removed in a future clean-up pass.

    def _format_power(self, var: str, power: float, notation: str) -> str:
        """Deprecated — use ParametricFormulaRenderer.format_power() directly."""
        return ParametricFormulaRenderer.format_power(var, power, notation)

    def _format_power_name(self, var_name: str, power: float) -> str:
        """Deprecated — use ParametricFormulaRenderer.format_power_name() directly."""
        return ParametricFormulaRenderer.format_power_name(var_name, power)

    def _format_coefficient(self, coef_val: float, notation: str = 'math',
                            sci_threshold: float = 1e-3,
                            sci_max: float = 1e4) -> str:
        """Deprecated — use ParametricFormulaRenderer.format_coefficient() directly."""
        return ParametricFormulaRenderer.format_coefficient(
            coef_val, notation, sci_threshold, sci_max
        )

    def _simplify_power(self, power: float, tolerance: float = 1e-9) -> str:
        """Deprecated — use ParametricFormulaRenderer.simplify_power() directly."""
        return ParametricFormulaRenderer.simplify_power(power, tolerance)

    def _get_fp2_second_term_name(self, var_name: str, power: float) -> str:
        """Deprecated — use ParametricFormulaRenderer.get_fp2_second_term_name() directly."""
        return ParametricFormulaRenderer.get_fp2_second_term_name(var_name, power)