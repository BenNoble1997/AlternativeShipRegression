"""
Multivariate Linear and Log-Transform Parametric Fitter
=========================================================

Role in the Architecture
-------------------------
This module contains :class:`MultivariateLinearFitter`, a concrete
implementation of
:class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
that handles all **multi-predictor linear and linearisable** model families.

It is registered in
:class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
under the key ``'multivariate_linear'``.

Model Families
--------------
Three conceptually distinct groups are handled here:

1. **Plain linear and regularised linear models**
   (:math:`y = \\beta_0 + \\boldsymbol{\\beta}^\\top \\mathbf{x}`)

   Fitted by the sklearn estimators :class:`~sklearn.linear_model.LinearRegression`,
   :class:`~sklearn.linear_model.Ridge`, :class:`~sklearn.linear_model.Lasso`,
   :class:`~sklearn.linear_model.RidgeCV`, and
   :class:`~sklearn.linear_model.LassoCV`.

2. **Multiplicative power model**
   (:math:`y = a \\cdot x_1^{b_1} \\cdot x_2^{b_2} \\cdots`)

   Taking the logarithm of both sides gives:

   .. math::

       \\ln y = \\ln a + b_1 \\ln x_1 + b_2 \\ln x_2 + \\cdots

   which is linear in the transformed variables.  A
   :class:`~sklearn.linear_model.LinearRegression` is fitted in log-space
   and predictions are back-transformed via :func:`numpy.exp`.

3. **Additive exponential model**
   (:math:`y = a \\cdot \\exp(b_1 x_1 + b_2 x_2 + \\cdots)`)

   Taking the logarithm:

   .. math::

       \\ln y = \\ln a + b_1 x_1 + b_2 x_2 + \\cdots

   Again linear — only ``y`` is log-transformed this time.  Predictions
   are back-transformed via :func:`numpy.exp`.

Regularisation
--------------
Ridge (L2) and Lasso (L1) penalties prevent overfitting by shrinking
coefficients:

- **Ridge** minimises
  :math:`\\text{RSS} + \\alpha \\sum_j \\beta_j^2`.
  All coefficients are retained but driven toward zero.

- **Lasso** minimises
  :math:`\\text{RSS} + \\alpha \\sum_j |\\beta_j|`.
  Some coefficients are driven *exactly* to zero, providing automatic
  feature selection.

The ``-cv`` variants (:class:`~sklearn.linear_model.RidgeCV`,
:class:`~sklearn.linear_model.LassoCV`) select :math:`\\alpha`
automatically via cross-validation.

Parameter Counting for Sparse Models
--------------------------------------
For Lasso, the effective number of parameters is the count of non-zero
coefficients plus one (intercept), not the total number of input
features.  :meth:`~MultivariateLinearFitter.get_n_parameters` returns
this sparse count when a fitted model is provided, ensuring AIC / BIC
penalties are computed correctly.
"""

from knowledge_layer.parametric_regression.base_model_fitter_class import BaseModelFitter

from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV



class MultivariateLinearFitter(BaseModelFitter):
    """
    Concrete fitter for multivariate linear and linearisable models.

    Handles plain linear regression, L1 / L2 regularised variants, and
    two multi-predictor non-linear forms whose fitting is reduced to
    linear regression via a log transformation.

    :cvar MODELS: Registry mapping each model-type string to a metadata
        dict with keys:

        - ``regularized`` (*bool*): whether a penalty term is applied.
        - ``cv_alpha`` (*bool*): whether alpha is selected via CV.
        - ``default_alpha`` (*float*, optional): fallback alpha when not
          supplied by the caller.
        - ``log_transform`` (*str*, optional): ``'power'`` or
          ``'exponential'`` for log-linearisable models.
    :vartype MODELS: dict[str, dict]
    """

    MODELS = {
        'linear': {'regularized': False, 'cv_alpha': False},
        'ridge': {'regularized': True, 'cv_alpha': False, 'default_alpha': 1.0},
        'lasso': {'regularized': True, 'cv_alpha': False, 'default_alpha': 0.1},
        'ridge-cv': {'regularized': True, 'cv_alpha': True},
        'lasso-cv': {'regularized': True, 'cv_alpha': True},
        'power-log-multivariate': {'log_transform': 'power'},
        'exponential-log-multivariate': {'log_transform': 'exponential'},
    }

    def get_available_models(self) -> list[str]:

        """
        Return all model-type strings handled by this fitter.

        :returns: List of model-type identifiers.
        :rtype: list[str]
        """
        return list(self.MODELS.keys())

    def is_applicable(self, model_type: str, X: np.ndarray, y: np.ndarray,
                      min_samples: int = 5) -> tuple[bool, str]:
        """
        Check whether a multivariate linear model can be applied to the data.

        Checks are applied in this order:

        1. The model-type string must belong to this fitter.
        2. ``n_samples`` must meet *min_samples*.
        3. Linear / regularised models require ``n_samples > n_features``.
        4. The multiplicative power model requires all ``X > 0`` and
           ``y > 0`` (log transform is applied to both).
        5. The additive exponential model requires all ``y > 0`` (only
           ``y`` is log-transformed).

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param min_samples: Minimum number of required samples.
        :type min_samples: int

        :returns: ``(True, '')`` if applicable; ``(False, reason)``
            otherwise.
        :rtype: tuple[bool, str]
        """

        if model_type not in self.MODELS:
            return False, f"model type '{model_type}' not handled by MultivariateLinearFitter"

        n_samples, n_features = X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)

        # Check sample size
        if n_samples < min_samples:
            return False, f"insufficient samples ({n_samples} < {min_samples})"

        # Linear models need more samples than features
        if model_type in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']:
            if n_samples <= n_features:
                return False, f"insufficient samples ({n_samples} ≤ {n_features} predictors)"

        # Check value constraints for log-transform models
        if model_type == 'power-log-multivariate':
            if not (np.all(X > 0) and np.all(y > 0)):
                return False, "requires all X > 0 and y > 0"

        if model_type == 'exponential-log-multivariate':
            if not np.all(y > 0):
                return False, "requires all y > 0"

        return True, ""

    def create_and_fit(self, model_type: str, X: np.ndarray, y: np.ndarray,
                       alpha: Optional[float] = None, cv_folds: int = 5, **kwargs):
        """
        Fit a multivariate linear or log-linearised model.

        For log-linearised models the input / output transformations are
        applied internally; the returned model object is a standard sklearn
        :class:`~sklearn.linear_model.LinearRegression` fitted in the
        transformed space.  :meth:`predict` handles the back-transformation.

        :param model_type: Model-type identifier.  Must be in
            :attr:`MODELS`.
        :type model_type: str
        :param X: Training predictor matrix, shape
            ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Training target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param alpha: Regularisation strength for ``'ridge'`` or
            ``'lasso'``.  When ``None`` the ``default_alpha`` recorded in
            :attr:`MODELS` is used.  Ignored for ``'-cv'`` variants.
        :type alpha: float, optional
        :param cv_folds: Number of cross-validation folds for ``'ridge-cv'``
            and ``'lasso-cv'``.  Defaults to 5.
        :type cv_folds: int
        :param kwargs: Additional keyword arguments (ignored; accepted for
            interface compatibility).

        :returns: ``(model, None)`` for all model types.  No preprocessor
            is required because feature-level transformations are applied
            inside this method and inverse-applied inside :meth:`predict`.
        :rtype: tuple[object, None]

        :raises ValueError: If ``model_type`` is unknown.
        """

        model_info = self.MODELS[model_type]

        # Handle log-transform models
        if 'log_transform' in model_info:
            return self._fit_log_transform_model(model_type, X, y, model_info['log_transform'])

        # Handle regular linear models
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            return model, None

        elif model_type == 'ridge':
            if alpha is None:
                alpha = model_info['default_alpha']
            model = Ridge(alpha=alpha)
            model.fit(X, y)
            return model, None

        elif model_type == 'lasso':
            if alpha is None:
                alpha = model_info['default_alpha']
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X, y)
            return model, None

        elif model_type == 'ridge-cv':
            alphas = np.logspace(-3, 3, 50)
            model = RidgeCV(alphas=alphas, cv=cv_folds)
            model.fit(X, y)
            model.selected_alpha_ = model.alpha_
            return model, None

        elif model_type == 'lasso-cv':
            alphas = np.logspace(-4, 1, 50)
            model = LassoCV(alphas=alphas, cv=cv_folds, max_iter=5000)
            model.fit(X, y)
            model.selected_alpha_ = model.alpha_
            return model, None

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _fit_log_transform_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                                 transform_type: str):
        """
        Fit a log-linearised model by transforming inputs / outputs to
        log-space, fitting ordinary linear regression, and tagging the
        model object for back-transformation at prediction time.

        For the **power** transform:

        .. math::

            \\ln y = \\ln a + b_1 \\ln x_1 + b_2 \\ln x_2 + \\cdots

        For the **exponential** transform:

        .. math::

            \\ln y = \\ln a + b_1 x_1 + b_2 x_2 + \\cdots

        The two additional attributes ``transform_type_`` and
        ``model_type_`` are stamped onto the sklearn
        :class:`~sklearn.linear_model.LinearRegression` object so that
        :meth:`predict` can recover the correct back-transformation.

        :param model_type: Full model-type string (e.g.
            ``'power-log-multivariate'``).
        :type model_type: str
        :param X: Training predictor matrix.  Must be all positive for the
            power transform.
        :type X: numpy.ndarray
        :param y: Training target vector.  Must be all positive.
        :type y: numpy.ndarray
        :param transform_type: Either ``'power'`` (log both ``X`` and
            ``y``) or ``'exponential'`` (log ``y`` only).
        :type transform_type: str

        :returns: ``(model, None)`` where ``model`` is a fitted
            :class:`~sklearn.linear_model.LinearRegression` in log-space,
            with ``transform_type_`` and ``model_type_`` attributes set.
        :rtype: tuple[LinearRegression, None]

        :raises ValueError: If ``transform_type`` is unrecognised.
        """

        if transform_type == 'power':
            # y = a·x₁^b₁·x₂^b₂ → ln(y) = ln(a) + b₁·ln(x₁) + b₂·ln(x₂)
            X_log = np.log(X)
            y_log = np.log(y)
        elif transform_type == 'exponential':
            # y = a·exp(b₁·x₁ + b₂·x₂) → ln(y) = ln(a) + b₁·x₁ + b₂·x₂
            X_log = X
            y_log = np.log(y)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        model = LinearRegression()
        model.fit(X_log, y_log)

        model.transform_type_ = transform_type
        model.model_type_ = model_type

        return model, None

    def predict(self, model, preprocessor, X: np.ndarray, model_type: str) -> np.ndarray:
        """
        Generate predictions, applying the correct back-transformation for
        log-linearised models.

        For log-linearised models the ``transform_type_`` attribute stamped
        during :meth:`create_and_fit` determines which transformation to
        invert:

        - ``'power'``: apply ``log(X)`` before predicting, then
          ``exp()`` to the output.
        - ``'exponential'``: predict directly on ``X``, then ``exp()`` to
          the output.

        For all other model types sklearn's ``predict`` is called directly.

        :param model: Fitted model object as returned by
            :meth:`create_and_fit`.
        :param preprocessor: Unused (always ``None`` for this fitter).
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """

        # Handle log-transform models
        if hasattr(model, 'transform_type_'):
            transform_type = model.transform_type_

            if transform_type == 'power':
                X_transformed = np.log(X)
            else:  # exponential
                X_transformed = X

            y_log_pred = model.predict(X_transformed)
            return np.exp(y_log_pred)

            # Regular linear models
        return model.predict(X)

    def get_n_parameters(self, model_type: str, X: np.ndarray, fitted_model=None) -> int:
        """
        Return the effective number of free parameters.

        For Lasso models, if ``fitted_model`` is supplied, the count is
        the number of non-zero coefficients plus one for the intercept.
        This sparse count is the correct value for AIC / BIC; using the
        total feature count would over-penalise Lasso's automatic feature
        selection.

        For all other model types the count is ``n_features + 1``.

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Predictor matrix used to determine feature count.
        :type X: numpy.ndarray
        :param fitted_model: Fitted sklearn estimator.  Inspected for
            ``coef_`` when ``model_type`` contains ``'lasso'``.
            Defaults to ``None``.

        :returns: Effective number of parameters (including intercept).
        :rtype: int
        """
        n_features = X.shape[1] if len(X.shape) > 1 else 1

        # Check if this is Lasso with fitted model
        if fitted_model is not None and 'lasso' in model_type:
            # Count non-zero coefficients
            if hasattr(fitted_model, 'coef_'):
                active_count = np.sum(np.abs(fitted_model.coef_) > 1e-10)
                return active_count + 1  # +1 for intercept

        # Default: all features
        if model_type in ['power-log-multivariate', 'exponential-log-multivariate']:
            return n_features + 1

        return n_features + 1  # β₁, β₂, ..., βₙ + β₀

        # Linear/regularized models
        return n_features + 1  # β₁, β₂, ..., βₙ + β₀

    def get_model_description(self, model_type: str) -> str:
        """
        Return a concise description including the functional form.

        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Human-readable description.
        :rtype: str
        """
        descriptions = {
            'linear': 'Linear regression: y = β₀ + β₁·x₁ + β₂·x₂ + ...',
            'ridge': 'Ridge regression (L2): Linear with coefficient shrinkage',
            'lasso': 'Lasso regression (L1): Linear with feature selection',
            'ridge-cv': 'Ridge with auto alpha selection (CV)',
            'lasso-cv': 'Lasso with auto alpha selection (CV)',
            'power-log-multivariate': 'Multiplicative power: y = a·x₁^b₁·x₂^b₂·... [multi-predictor]',
            'exponential-log-multivariate': 'Additive exponential: y = a·exp(b₁·x₁+b₂·x₂+...) [multi-predictor]',
        }
        return descriptions.get(model_type, 'Unknown model type')
