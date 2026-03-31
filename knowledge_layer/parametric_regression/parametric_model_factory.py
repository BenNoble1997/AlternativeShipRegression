"""
Parametric Model Factory — Unified Façade for Parametric Regression
=====================================================================

Role in the Architecture
-------------------------
:class:`ModelFactory` is the single public entry point for all parametric
regression operations.  Callers never instantiate individual fitters
directly; instead they call static / class methods on :class:`ModelFactory`
and the factory routes each call to the appropriate concrete fitter.

Data Flow Through the System
-----------------------------
Below is a step-by-step trace of a typical ``create_and_fit → predict``
cycle to show how the outputs of each class feed into the next:

1. **Caller** invokes ``ModelFactory.create_and_fit('polynomial-2', X, y)``.

2. :meth:`_clean_data` removes rows containing NaN or Inf from both ``X``
   and ``y``, returning ``(X_clean, y_clean)``.

3. :meth:`_get_fitter` looks up ``'polynomial-2'`` in
   :attr:`_model_to_fitter_map` and returns the
   :class:`~knowledge_layer.parametric_regression.polynomial_parametric_models.PolynomialFitter`
   singleton stored in :attr:`_fitters`.

4. :meth:`PolynomialFitter.create_and_fit` is called with
   ``(X_clean, y_clean)``.  Internally it:

   a. Constructs a :class:`~sklearn.preprocessing.PolynomialFeatures`
      instance and calls ``fit_transform(X_clean)`` to produce
      ``X_poly``.
   b. Fits a :class:`~sklearn.linear_model.LinearRegression` on
      ``(X_poly, y_clean)``.
   c. Returns ``(sklearn_model, poly_preprocessor)``.

5. The ``(model, preprocessor)`` tuple is returned to the caller.

6. **Later**, the caller invokes
   ``ModelFactory.predict(model, preprocessor, X_new, 'polynomial-2')``.

7. :meth:`_get_fitter` routes to :class:`PolynomialFitter` again.

8. :meth:`PolynomialFitter.predict` calls
   ``preprocessor.transform(X_new)`` (not ``fit_transform``) and then
   ``model.predict(X_poly_new)``.

9. The prediction array is returned to the caller.

For **MFP** the flow differs slightly: no preprocessor is returned
(``None``), and :class:`MFPRegressor` stores and applies its own
transformations internally via :attr:`~MFPRegressor.transforms_`.

Architecture — Strategy Pattern via Composition
-----------------------------------------------
:class:`ModelFactory` implements the **Strategy** design pattern:

- :class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
  defines the interface (*Strategy*).
- Concrete fitters are the implementations (*ConcreteStrategy*).
- :class:`ModelFactory` is the stateless context that delegates to
  strategies — it is **composed** of fitters, not derived from them.

Fitters are instantiated **once** at class load time as singletons in
:attr:`_fitters`.  The :attr:`_model_to_fitter_map` is built at the
same time by iterating over each fitter's
:meth:`~BaseModelFitter.get_available_models`.  This means adding a new
model family requires only:

1. Writing a new :class:`BaseModelFitter` subclass.
2. Adding an instance to :attr:`_fitters`.

No other changes are needed anywhere else in the system.

All public methods are **class methods** or **static methods** — the
factory is never instantiated.
"""

import numpy as np

from knowledge_layer.parametric_regression.base_model_fitter_class import BaseModelFitter
from knowledge_layer.parametric_regression.multivariable_parametric_models import MultivariateLinearFitter
from knowledge_layer.parametric_regression.single_variable_parametric_models import UnivariateNonlinearFitter
from knowledge_layer.parametric_regression.polynomial_parametric_models import PolynomialFitter
from knowledge_layer.parametric_regression.mfp_fitter import MFPFitter


class ModelFactory:
    """
    Unified, stateless façade for all parametric regression models.

    All interaction with the parametric regression subsystem goes through
    this class.  It maintains a registry of concrete
    :class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
    instances and routes every call to the appropriate fitter based on the
    model-type string.

    **Usage pattern** — no instantiation required::

        model, prep = ModelFactory.create_and_fit('mfp', X_train, y_train)
        y_pred = ModelFactory.predict(model, prep, X_test, 'mfp')
        n_params = ModelFactory.get_n_parameters('mfp', X_train, fitted_model=model)

    :cvar _fitters: Registry mapping fitter-family keys to singleton fitter
        instances.  Populated once at class load time.
    :vartype _fitters: dict[str, BaseModelFitter]

    :cvar _model_to_fitter_map: Derived mapping from every individual
        model-type string to the fitter-family key that handles it.
        Built automatically from :meth:`~BaseModelFitter.get_available_models`
        at class load time.
    :vartype _model_to_fitter_map: dict[str, str]
    """

    # =========================================================================
    # Class-Level Registry (built once at import time)
    # =========================================================================

    _fitters: dict[str, BaseModelFitter] = {
        'univariate_nonlinear': UnivariateNonlinearFitter(),
        'multivariate_linear':  MultivariateLinearFitter(),
        'polynomial':           PolynomialFitter(),
        'mfp':                  MFPFitter(),
    }

    # Reverse-lookup: model_type → fitter_family_key
    _model_to_fitter_map: dict[str, str] = {}
    for _fitter_key, _fitter_inst in _fitters.items():
        for _model_type in _fitter_inst.get_available_models():
            _model_to_fitter_map[_model_type] = _fitter_key

    # =========================================================================
    # Internal Routing
    # =========================================================================

    @classmethod
    def _get_fitter(cls, model_type: str) -> BaseModelFitter:
        """
        Retrieve the fitter instance responsible for ``model_type``.

        :param model_type: Model-type identifier (e.g. ``'mfp'``,
            ``'polynomial-2-lasso'``).
        :type model_type: str

        :returns: The singleton :class:`~BaseModelFitter` that handles
            this model type.
        :rtype: BaseModelFitter

        :raises ValueError: If ``model_type`` is not registered in any
            fitter.
        """
        if model_type not in cls._model_to_fitter_map:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available types: {sorted(cls._model_to_fitter_map.keys())}"
            )
        return cls._fitters[cls._model_to_fitter_map[model_type]]

    # =========================================================================
    # Model Discovery
    # =========================================================================

    @staticmethod
    def get_available_models() -> list[str]:
        """
        Return all model-type strings registered across all fitters.

        The list is assembled by calling
        :meth:`~BaseModelFitter.get_available_models` on each registered
        fitter.

        :returns: All available model-type identifiers.
        :rtype: list[str]
        """
        result = []
        for fitter in ModelFactory._fitters.values():
            result.extend(fitter.get_available_models())
        return result

    @classmethod
    def get_model_description(cls, model_type: str) -> str:
        """
        Return a concise human-readable description of the model type.

        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Description string.
        :rtype: str

        :raises ValueError: If ``model_type`` is unknown.
        """
        return cls._get_fitter(model_type).get_model_description(model_type)

    @classmethod
    def get_model_descriptions(cls) -> dict[str, str]:
        """
        Return descriptions for all registered model types.

        :returns: ``{model_type: description}`` for every registered
            model type.
        :rtype: dict[str, str]
        """
        return {m: cls.get_model_description(m) for m in cls.get_available_models()}

    # =========================================================================
    # Applicability Checking
    # =========================================================================

    @classmethod
    def is_model_applicable(
        cls,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        clean_first: bool = True,
        min_samples: int = 5,
    ) -> tuple[bool, str]:
        """
        Check whether a model type can be applied to the given data.

        Each fitter encodes its own structural and mathematical constraints
        (sample-size requirements, positivity constraints, single-predictor
        requirements, etc.).  This method dispatches to the appropriate
        fitter after optionally cleaning the data.

        :param model_type: Model-type identifier to check.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param clean_first: If ``True``, NaN / Inf rows are removed before
            the applicability check, mirroring the cleaning that
            :meth:`create_and_fit` performs.  Defaults to ``True``.
        :type clean_first: bool
        :param min_samples: Minimum number of (clean) samples required.
        :type min_samples: int

        :returns: ``(True, '')`` if applicable; ``(False, reason)``
            otherwise.
        :rtype: tuple[bool, str]
        """
        if clean_first:
            X, y = cls._clean_data(X, y)

        X = X.reshape(-1, 1) if X.ndim == 1 else X

        try:
            fitter = cls._get_fitter(model_type)
            return fitter.is_applicable(model_type, X, y, min_samples)
        except ValueError as exc:
            return False, str(exc)

    @classmethod
    def get_applicable_models(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        exclude_models: list[str] = None,
        clean_first: bool = True,
        min_samples: int = 5,
    ) -> list[str]:
        """
        Return all model types that are applicable to the given data.

        Iterates over every registered model type and calls
        :meth:`is_model_applicable`.  Models listed in ``exclude_models``
        are filtered before the applicability check.

        This is the primary mechanism used by
        :class:`~knowledge_layer.parametric_regression.parametric_regresssion_selector_class.ParametricModelSelector`
        to determine which models to test for each predictor-target pair.

        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param exclude_models: Model-type strings to skip.  Useful for
            suppressing regularised variants or computationally expensive
            models.
        :type exclude_models: list[str], optional
        :param clean_first: If ``True``, data is cleaned before checking.
        :type clean_first: bool
        :param min_samples: Minimum number of (clean) samples required.
        :type min_samples: int

        :returns: Model-type identifiers that passed all applicability
            constraints.
        :rtype: list[str]
        """
        candidates = cls.get_available_models()
        if exclude_models:
            candidates = [m for m in candidates if m not in exclude_models]

        return [
            m for m in candidates
            if cls.is_model_applicable(m, X, y, clean_first, min_samples)[0]
        ]

    @classmethod
    def get_inapplicable_models_with_reasons(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        exclude_models: list[str]|None = None,
        clean_first: bool = True,
        min_samples: int = 5,
    ) -> dict[str, str]:
        """
        Return a mapping of inapplicable model types to their reasons.

        Useful for debugging: shows exactly which constraint each model
        fails on for a particular dataset.

        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param exclude_models: Model-type strings to skip.
        :type exclude_models: list[str], optional
        :param clean_first: If ``True``, data is cleaned before checking.
        :type clean_first: bool
        :param min_samples: Minimum number of (clean) samples required.
        :type min_samples: int

        :returns: ``{model_type: reason}`` for every inapplicable model.
        :rtype: dict[str, str]
        """
        candidates = cls.get_available_models()
        if exclude_models:
            candidates = [m for m in candidates if m not in exclude_models]

        result = {}
        for m in candidates:
            ok, reason = cls.is_model_applicable(m, X, y, clean_first, min_samples)
            if not ok:
                result[m] = reason
        return result

    # =========================================================================
    # Model Creation and Fitting
    # =========================================================================

    @classmethod
    def create_and_fit(
        cls,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Clean the data, route to the appropriate fitter, and return the
        fitted model and preprocessor.

        This is the primary entry point for model creation.  The
        ``**kwargs`` are forwarded unchanged to the concrete fitter, so
        model-specific hyper-parameters (e.g. ``alpha``, ``fixed_transforms``,
        ``verbose`` for MFP) can be passed here without any routing logic.

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Training predictor matrix, shape
            ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Training target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param kwargs: Model-specific keyword arguments forwarded to the
            concrete fitter (e.g. ``alpha=10.0`` for Ridge,
            ``fixed_transforms={…}`` for MFP).

        :returns: ``(model, preprocessor)`` as returned by the concrete
            fitter.  ``preprocessor`` is ``None`` for most model types and
            a fitted :class:`~sklearn.preprocessing.PolynomialFeatures` for
            polynomial models.
        :rtype: tuple[object, object | None]

        :raises ValueError: If fewer than 3 clean samples remain after
            removing NaN / Inf rows, or if the model fails to fit.
        """
        X, y = cls._clean_data(X, y)
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        if len(X) < 3:
            raise ValueError(
                f"Insufficient clean data: {len(X)} samples remain after "
                f"removing NaN / Inf rows."
            )

        fitter = cls._get_fitter(model_type)
        return fitter.create_and_fit(model_type, X, y, **kwargs)

    # =========================================================================
    # Prediction
    # =========================================================================

    @classmethod
    def predict(
        cls,
        model,
        preprocessor,
        X: np.ndarray,
        model_type: str,
    ) -> np.ndarray:
        """
        Route prediction to the appropriate fitter.

        The fitter's :meth:`~BaseModelFitter.predict` is responsible for
        any inverse transformations required to return predictions in the
        original target space (e.g. ``exp()`` for log-linearised models).

        :param model: Fitted model object as returned by
            :meth:`create_and_fit`.
        :param preprocessor: Preprocessor returned by
            :meth:`create_and_fit`, or ``None``.
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return cls._get_fitter(model_type).predict(model, preprocessor, X, model_type)

    # =========================================================================
    # Parameter Counting
    # =========================================================================

    @classmethod
    def get_n_parameters(
        cls,
        model_type: str,
        X: np.ndarray,
        fitted_model=None,
    ) -> int:
        """
        Return the effective number of free parameters for AIC / BIC.

        Delegates to the appropriate fitter.  For sparse models (Lasso,
        MFP), passing ``fitted_model`` is necessary to obtain the actual
        parameter count — without it, the maximum possible count is
        returned.

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
            Used to determine input feature count.
        :type X: numpy.ndarray
        :param fitted_model: Fitted model object.  Required for accurate
            counts with Lasso (counts non-zero coefficients) and MFP
            (counts active transformations).  Defaults to ``None``.

        :returns: Effective number of free parameters.
        :rtype: int
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return cls._get_fitter(model_type).get_n_parameters(
            model_type, X, fitted_model=fitted_model
        )

    # =========================================================================
    # Data Cleaning Utility
    # =========================================================================

    @staticmethod
    def _clean_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove rows containing NaN or Inf values from both ``X`` and ``y``.

        A row is removed if *any* element in the corresponding row of
        ``X`` is NaN / Inf, or if the corresponding element of ``y`` is
        NaN / Inf.  The same boolean mask is applied to both arrays so
        that alignment is preserved.

        This cleaning mirrors what :meth:`create_and_fit` would perform
        implicitly, so applicability checks run on the same data that will
        actually be used for fitting.

        :param X: Predictor matrix, shape ``(n_samples, n_features)`` or
            ``(n_samples,)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray

        :returns: ``(X_clean, y_clean)`` with invalid rows removed.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        mask = ~(
            np.isnan(X).any(axis=1) | np.isnan(y) |
            np.isinf(X).any(axis=1) | np.isinf(y)
        )
        return X[mask], y[mask]