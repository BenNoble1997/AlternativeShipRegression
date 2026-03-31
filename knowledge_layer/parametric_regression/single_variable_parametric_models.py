
"""
Univariate Non-Linear Parametric Fitter
========================================

Role in the Architecture
-------------------------
This module contains :class:`UnivariateNonlinearFitter`, a concrete
implementation of
:class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
responsible for all **single-predictor, non-linear** model families.

It is registered in
:class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
under the key ``'univariate_nonlinear'``.

Fitting Method — Non-Linear Least Squares
------------------------------------------
Unlike linear models (which have a closed-form solution via the normal
equations), the models here have no such guarantee.
:func:`scipy.optimize.curve_fit` is used, which internally runs
**Levenberg–Marquardt** (or trust-region reflective when bounds are
supplied).  This iteratively adjusts the parameters ``(a, b[, c])`` to
minimise the sum of squared residuals:

.. math::

    \\text{RSS}(\\boldsymbol{\\theta}) = \\sum_{i=1}^{n}
    \\bigl(y_i - f(x_i;\\,\\boldsymbol{\\theta})\\bigr)^2

Because optimisation is iterative, convergence is not guaranteed, and
:meth:`~UnivariateNonlinearFitter.create_and_fit` wraps the call in a
``try / except`` and raises a :exc:`ValueError` with a descriptive
message on failure.

Model Return Convention
-----------------------
Fitted non-linear models are returned as a ``(func_name, popt)`` tuple —
not as sklearn estimator objects.  ``func_name`` is the model-type string
(e.g. ``'power'``), and ``popt`` is the NumPy array of optimised
parameters.  :meth:`~UnivariateNonlinearFitter.predict` unpacks this
tuple to reconstruct and evaluate the function.

Supported Models
----------------
+----------------------+--------------------------------+------------------------+
| Model type           | Functional form                | Constraints            |
+======================+================================+========================+
| ``exponential``      | :math:`a \\cdot e^{bx}`        | y > 0                  |
+----------------------+--------------------------------+------------------------+
| ``logarithmic``      | :math:`a \\ln(x) + b`          | x > 0                  |
+----------------------+--------------------------------+------------------------+
| ``power``            | :math:`a \\cdot x^{b}`         | x > 0, y > 0           |
+----------------------+--------------------------------+------------------------+
| ``exponential-offset``| :math:`a \\cdot e^{bx} + c`   | none                   |
+----------------------+--------------------------------+------------------------+
| ``power-offset``     | :math:`a \\cdot x^b + c`       | x > 0, y > 0           |
+----------------------+--------------------------------+------------------------+
| ``logarithmic-power``| :math:`a \\cdot (\\ln x)^b`    | x > 0                  |
+----------------------+--------------------------------+------------------------+

All models require exactly **one predictor**.
"""

import numpy as np

from knowledge_layer.parametric_regression.base_model_fitter_class import BaseModelFitter
from scipy.optimize import curve_fit


class UnivariateNonlinearFitter(BaseModelFitter):
    """
    Concrete fitter for single-predictor non-linear regression models.

    Fits parametric curves of the form :math:`y = f(x; \\boldsymbol{\\theta})`
    using non-linear least squares via :func:`scipy.optimize.curve_fit`.

    Each model family is characterised by its number of parameters and its
    domain constraints (positivity requirements on ``x`` and/or ``y``).
    These constraints are recorded in :attr:`MODELS` and checked in
    :meth:`is_applicable` before any fitting is attempted.

    :cvar MODELS: Registry mapping each model-type string to its metadata:

        - ``params`` (*int*): number of parameters ``(a, b)`` or
          ``(a, b, c)``.
        - ``requires_positive_x`` (*bool*): whether all predictor values
          must be strictly positive (required by logarithms and
          fractional powers).
        - ``requires_positive_y`` (*bool*): whether all target values
          must be strictly positive (required where :math:`y` is the
          argument to a logarithm in the transformation).
    :vartype MODELS: dict[str, dict]
    """

    MODELS = {
        'exponential': {'params': 2, 'requires_positive_x': False, 'requires_positive_y': True},
        'logarithmic': {'params': 2, 'requires_positive_x': True, 'requires_positive_y': False},
        'power': {'params': 2, 'requires_positive_x': True, 'requires_positive_y': True},
        'exponential-offset': {'params': 3, 'requires_positive_x': False, 'requires_positive_y': False},
        'power-offset': {'params': 3, 'requires_positive_x': True, 'requires_positive_y': True},
        'logarithmic-power': {'params': 2, 'requires_positive_x': True, 'requires_positive_y': False},
    }

    def get_available_models(self) -> list[str]:
        return list(self.MODELS.keys())

    def is_applicable(self, model_type: str, X: np.ndarray, y: np.ndarray,
                      min_samples: int = 5) -> tuple[bool, str]:
        """
        Check whether a univariate non-linear model can be applied to the data.

        Checks are applied in this order:

        1. The model-type string must belong to this fitter.
        2. ``n_samples`` must meet *min_samples*.
        3. The data must have exactly **one predictor column**.
        4. Value constraints (positivity) taken from :attr:`MODELS`.

        :param model_type: Model-type identifier to check.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param min_samples: Minimum number of required samples.
        :type min_samples: int

        :returns: ``(True, '')`` if applicable; ``(False, reason)`` otherwise.
        :rtype: tuple[bool, str]
        """

        if model_type not in self.MODELS:
            return False, f"model type '{model_type}' not handled by UnivariateNonlinearFitter"

        # Check sample size
        if len(X) < min_samples:
            return False, f"insufficient samples ({len(X)} < {min_samples})"

        # Must be single predictor
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        if n_features != 1:
            return False, f"requires single predictor (got {n_features})"

        # Check value constraints
        model_info = self.MODELS[model_type]

        if model_info['requires_positive_x']:
            if not np.all(X > 0):
                min_x = np.min(X)
                return False, f"requires all x > 0 (found min x = {min_x:.4f})"

        if model_info['requires_positive_y']:
            if not np.all(y > 0):
                min_y = np.min(y)
                return False, f"requires all y > 0 (found min y = {min_y:.4f})"

        return True, ""

    def create_and_fit(self, model_type: str, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit a non-linear model using :func:`scipy.optimize.curve_fit`.

        The function definition and parameter bounds are determined
        internally from ``model_type``.  Bounds are set conservatively
        to avoid numerical instability (e.g. the exponent ``b`` in
        exponential models is clamped to ``[-10, 10]``).

        On convergence failure :func:`~scipy.optimize.curve_fit` raises
        :exc:`RuntimeError`, which is caught and re-raised as
        :exc:`ValueError` with a more informative message.

        :param model_type: Model-type identifier.  Must be in
            :attr:`MODELS`.
        :type model_type: str
        :param X: Training predictor matrix, shape ``(n_samples, 1)``.
        :type X: numpy.ndarray
        :param y: Training target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param kwargs: Unused; accepted for interface compatibility.

        :returns: ``((model_type, popt), None)`` where ``popt`` is the
            array of optimised parameters.
        :rtype: tuple[tuple[str, numpy.ndarray], None]

        :raises ValueError: If the optimiser fails to converge or the
            model type is unknown.
        """

        x = X.flatten()

        # Define function and bounds based on model type
        if model_type == 'exponential':
            func = lambda x, a, b: a * np.exp(b * x)
            bounds = ([-np.inf, -10], [np.inf, 10])

        elif model_type == 'logarithmic':
            func = lambda x, a, b: a * np.log(x) + b
            bounds = (-np.inf, np.inf)

        elif model_type == 'power':
            func = lambda x, a, b: a * x ** b
            bounds = ([-np.inf, -5], [np.inf, 5])

        elif model_type == 'exponential-offset':
            func = lambda x, a, b, c: a * np.exp(b * x) + c
            bounds = ([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])

        elif model_type == 'power-offset':
            func = lambda x, a, b, c: a * (x ** b) + c
            bounds = ([-np.inf, -5, -np.inf], [np.inf, 5, np.inf])

        elif model_type == 'logarithmic-power':
            func = lambda x, a, b: a * (np.log(x) ** b)
            bounds = ([-np.inf, 0.1], [np.inf, 5])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Fit using non-linear least squares
        try:
            popt, _ = curve_fit(func, x, y, maxfev=5000, bounds=bounds)
            return (model_type, popt), None
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"{model_type} fit failed: {e}")

    def predict(self, model, preprocessor, X: np.ndarray, model_type: str) -> np.ndarray:
        """
        Evaluate the fitted non-linear function on new data.

        The function is reconstructed from the stored ``(func_name, popt)``
        tuple and applied elementwise to ``X.flatten()``.

        :param model: Fitted model tuple ``(func_name, popt)`` as returned
            by :meth:`create_and_fit`.
        :param preprocessor: Unused (always ``None`` for this fitter);
            accepted for interface compatibility.
        :param X: Predictor matrix, shape ``(n_samples, 1)`` or
            ``(n_samples,)``.
        :type X: numpy.ndarray
        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray

        :raises ValueError: If ``func_name`` is unrecognised.
        """

        func_name, popt = model
        x = X.flatten()

        if func_name == 'exponential':
            return popt[0] * np.exp(popt[1] * x)
        elif func_name == 'logarithmic':
            return popt[0] * np.log(x) + popt[1]
        elif func_name == 'power':
            return popt[0] * x ** popt[1]
        elif func_name == 'exponential-offset':
            return popt[0] * np.exp(popt[1] * x) + popt[2]
        elif func_name == 'power-offset':
            return popt[0] * (x ** popt[1]) + popt[2]
        elif func_name == 'logarithmic-power':
            return popt[0] * (np.log(x) ** popt[1])
        else:
            raise ValueError(f"Unknown model: {func_name}")

    def get_n_parameters(self, model_type: str, X: np.ndarray, fitted_model=None) -> int:
        """
        Return the number of parameters for the specified non-linear model.

        Parameter counts are fixed by the functional form and are read
        directly from :attr:`MODELS` — the data shape and any fitted model
        object are irrelevant:

        - Two-parameter forms (``exponential``, ``logarithmic``,
          ``power``, ``logarithmic-power``): returns **2**.
        - Three-parameter offset forms (``exponential-offset``,
          ``power-offset``): returns **3**.

        :param model_type: Model-type identifier.
        :type model_type: str
        :param X: Predictor matrix (used only for interface consistency;
            not inspected here).
        :type X: numpy.ndarray
        :param fitted_model: Unused; accepted for interface compatibility.

        :returns: Number of parameters.
        :rtype: int

        :raises ValueError: If ``model_type`` is not in :attr:`MODELS`.
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.MODELS[model_type]['params']

    def get_model_description(self, model_type: str) -> str:
        """
        Return a concise description including the functional form.

        :param model_type: Model-type identifier.
        :type model_type: str

        :returns: Human-readable description.
        :rtype: str
        """
        descriptions = {
            'exponential': 'Exponential: y = a·exp(b·x) [single predictor]',
            'logarithmic': 'Logarithmic: y = a·ln(x) + b [single predictor]',
            'power': 'Power: y = a·x^b [single predictor]',
            'exponential-offset': 'Exponential with offset: y = a·exp(b·x) + c [single predictor]',
            'power-offset': 'Power with offset: y = a·x^b + c [single predictor]',
            'logarithmic-power': 'Logarithmic power: y = a·(ln(x))^b [single predictor]',
        }
        return descriptions.get(model_type, 'Unknown model type')