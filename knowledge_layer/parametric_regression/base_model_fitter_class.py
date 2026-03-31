"""
Base Model Fitter — Abstract Interface for Parametric Regression Strategies
============================================================================

Role in the Architecture
-------------------------
This module defines the contract that every concrete fitter must fulfil.
It sits at the base of the **Strategy Pattern** used throughout the
parametric regression subsystem:

- :class:`BaseModelFitter` is the *Strategy* interface.
- :class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
  is the *Context* — it holds a registry of concrete strategies and
  delegates all work to them.
- Concrete fitters
  (:class:`~knowledge_layer.parametric_regression.single_variable_parametric_models.UnivariateNonlinearFitter`,
  :class:`~knowledge_layer.parametric_regression.multivariable_parametric_models.MultivariateLinearFitter`,
  :class:`~knowledge_layer.parametric_regression.polynomial_parametric_models.PolynomialFitter`,
  :class:`~knowledge_layer.parametric_regression.mfp_fitter.MFPFitter`)
  are the *ConcreteStrategy* implementations.

Why an Abstract Base?
---------------------
Each model family has different fitting logic, prediction logic, and
applicability constraints.  Defining a shared interface here means that
:class:`ModelFactory` can dispatch to any fitter without knowing its
internals — adding a new model family requires only a new subclass of
:class:`BaseModelFitter`, not changes to the factory.

Design Notes
------------
- **Composition over inheritance**: Concrete fitters are *composed into*
  :class:`ModelFactory` rather than inheriting from it.  No user-facing
  class inherits from :class:`BaseModelFitter` directly.
- **Stateless fitters**: Fitters store no per-fit state; all results are
  returned from methods.  This makes the fitters safe to share as
  class-level singletons inside :class:`ModelFactory`.
- **Preprocessor convention**: Methods that create models return a
  ``(model, preprocessor)`` tuple.  ``preprocessor`` is ``None`` when not
  needed (e.g. linear regression) and an sklearn transformer (e.g.
  :class:`~sklearn.preprocessing.PolynomialFeatures`) when the input must
  be transformed before prediction.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class BaseModelFitter(ABC):
    """
    Abstract interface for parametric regression fitting strategies.

    Every concrete fitter handles one *family* of models — for example,
    all univariate non-linear models, or all polynomial models.  The family
    is responsible for:

    - advertising which model type strings it handles
      (:meth:`get_available_models`),
    - deciding whether a model is applicable to a given dataset
      (:meth:`is_applicable`),
    - creating and fitting a model (:meth:`create_and_fit`),
    - making predictions from a fitted model (:meth:`predict`),
    - counting parameters for information-criterion calculations
      (:meth:`get_n_parameters`), and
    - providing human-readable descriptions (:meth:`get_model_description`).

    :class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
    builds a registry of :class:`BaseModelFitter` instances at class
    initialisation time and routes every public call through
    :meth:`~ModelFactory._get_fitter`.

    Notes
    -----
    All methods are intentionally stateless — pass all required state as
    arguments and receive results as return values.  This allows a single
    fitter instance to be reused safely across many fits.
    """

    # =========================================================================
    # Model Discovery
    # =========================================================================

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return the list of model-type strings handled by this fitter.

        The strings returned here are the keys used throughout the system —
        they are passed to :meth:`create_and_fit`, :meth:`predict`, etc.
        Every string must be unique across *all* registered fitters.

        :returns: Model-type identifiers, e.g. ``['linear', 'ridge', 'lasso']``.
        :rtype: list[str]
        """
        pass

    # =========================================================================
    # Applicability Checking
    # =========================================================================

    @abstractmethod
    def is_applicable(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 5,
    ) -> Tuple[bool, str]:
        """
        Determine whether a model can be applied to the supplied data.

        Each fitter encodes its own structural and mathematical constraints
        here — for example, that a logarithmic model requires all ``X > 0``,
        or that a linear model requires ``n_samples > n_features``.  This
        method is called by
        :meth:`~ModelFactory.get_applicable_models` to filter the candidate
        set before any fitting is attempted.

        The data passed in should already be clean (NaN / Inf removed) so
        that sample-count checks reflect what will actually be used for
        fitting.

        :param model_type: Model-type identifier to check.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param min_samples: Minimum number of samples required.  Defaults to 5.
        :type min_samples: int

        :returns: A two-element tuple:

            - ``is_applicable`` (*bool*) — ``True`` if the model can be
              used on this data.
            - ``reason`` (*str*) — Human-readable explanation when
              ``is_applicable`` is ``False``; empty string when ``True``.

        :rtype: tuple[bool, str]
        """
        pass

    # =========================================================================
    # Model Creation and Fitting
    # =========================================================================

    @abstractmethod
    def create_and_fit(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Instantiate and fit a model of the specified type.

        This is the primary fitting entry point.  Implementations should
        be as pure as possible — receive data, return a fitted object —
        and avoid storing any per-fit state on ``self``.

        The ``**kwargs`` mechanism allows callers to pass model-specific
        hyper-parameters (e.g. ``alpha`` for Ridge, ``fixed_transforms``
        for MFP) without the base class needing to know about them.

        :param model_type: Model-type identifier (must be in
            :meth:`get_available_models`).
        :type model_type: str
        :param X: Training predictor matrix, shape
            ``(n_samples, n_features)``.  Must be clean (no NaN / Inf).
        :type X: numpy.ndarray
        :param y: Training target vector, shape ``(n_samples,)``.  Must be
            clean (no NaN / Inf).
        :type y: numpy.ndarray
        :param kwargs: Model-specific keyword arguments forwarded by
            :class:`ModelFactory`.

        :returns: A two-element tuple:

            - ``model`` — The fitted model object.  The concrete type
              depends on the fitter: an sklearn estimator for linear /
              polynomial models, a tuple ``(func_name, popt)`` for
              curve-fit models, an
              :class:`~knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class.MFPRegressor`
              for MFP.
            - ``preprocessor`` — A fitted transformer (e.g.
              :class:`~sklearn.preprocessing.PolynomialFeatures`) or
              ``None`` if no input transformation is required.

        :rtype: tuple[object, object | None]
        """
        pass

    # =========================================================================
    # Prediction
    # =========================================================================

    @abstractmethod
    def predict(
        self,
        model,
        preprocessor,
        X: np.ndarray,
        model_type: str,
    ) -> np.ndarray:
        """
        Generate predictions from a previously fitted model.

        The implementation must mirror whatever transformation was applied
        during :meth:`create_and_fit`.  For example:

        - Polynomial fitters must call ``preprocessor.transform(X)`` before
          ``model.predict()``.
        - Log-transform multivariate fitters must back-transform via
          ``exp()`` after predicting in log-space.
        - MFP fitters delegate directly to
          :meth:`~MFPRegressor.predict`, which applies stored
          transformations internally.

        :param model: Fitted model object returned by
            :meth:`create_and_fit`.
        :param preprocessor: Preprocessor returned by
            :meth:`create_and_fit`, or ``None``.
        :param X: Predictor matrix for prediction, shape
            ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param model_type: Model-type identifier (used to route prediction
            logic in fitters that handle heterogeneous model families).
        :type model_type: str

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        pass

    # =========================================================================
    # Parameter Counting
    # =========================================================================

    @abstractmethod
    def get_n_parameters(
        self,
        model_type: str,
        X: np.ndarray,
        fitted_model=None,
    ) -> int:
        """
        Return the effective number of free parameters in the model.

        Parameter count is used to compute AIC and BIC — information
        criteria that penalise model complexity.  For most model types the
        count is determined entirely by the data shape and model
        configuration; for *sparse* models (Lasso, MFP) it depends on the
        fitted model itself.

        Counting rules by model family:

        - **Linear / Ridge**: ``n_features + 1`` (intercept always present).
        - **Lasso**: count of non-zero coefficients + 1 (intercept).
          Requires ``fitted_model`` for accuracy; falls back to
          ``n_features + 1`` otherwise.
        - **Polynomial degree** *d*, *p* inputs:
          :math:`\\binom{p+d}{d}` (all interaction terms).
        - **MFP**: 1 (intercept) + 1 per *linear* / *FP1* variable +
          2 per *FP2* variable; excluded variables contribute 0.
          Requires ``fitted_model`` for accuracy.
        - **Univariate nonlinear**: fixed counts (2 for two-parameter
          forms, 3 for offset variants).

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Predictor matrix used to determine feature count, shape
            ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param fitted_model: The fitted model object.  When provided,
            implementations should use it to return the *actual* parameter
            count (important for sparse models such as Lasso and MFP where
            the count cannot be determined from data shape alone).
            Defaults to ``None``.

        :returns: Effective number of free parameters (including intercept
            where applicable).
        :rtype: int
        """
        pass

    # =========================================================================
    # Human-Readable Description
    # =========================================================================

    @abstractmethod
    def get_model_description(self, model_type: str) -> str:
        """
        Return a concise human-readable description of the model type.

        Used by display utilities and reporting code.  Should include the
        functional form where practical, e.g.
        ``"Power: y = a·x^b [single predictor]"``.

        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Short description of the model.
        :rtype: str
        """
        pass