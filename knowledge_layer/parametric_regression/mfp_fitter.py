"""
Multivariable Fractional Polynomials (MFP) Fitter
===================================================

Role in the Architecture
-------------------------
This module contains :class:`MFPFitter`, a concrete implementation of
:class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
that delegates MFP fitting to
:class:`~knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class.MFPRegressor`.

It is registered in
:class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
under the key ``'mfp'``.

What MFP Does — a Primer
--------------------------
Standard polynomial regression fixes the transformation applied to each
predictor in advance (e.g. x, x², x³).  MFP instead *discovers* the
best transformation for each variable by searching over a pre-defined
power set:

    P = {−3, −2.5, −2, …, 0, …, 1.5, 2, 2.5, 3}

where power 0 is interpreted as the natural logarithm by convention.

For each predictor, the algorithm chooses the best of:

- **Exclusion**: the variable is dropped (not significant).
- **Linear**: no transformation (power = 1).
- **FP1**: a single-power transformation, e.g. :math:`x^{0.5}` (= √x).
- **FP2**: two terms — either two different powers, e.g.
  :math:`x^{-1} + x^2`, or a *repeated* power which generates
  :math:`x^p` and :math:`x^p \\ln x`.

The transformations and variable selection are determined jointly by an
iterative **Backward Elimination → Function Selection** cycle, which
converges when two consecutive cycles produce identical transformations.

This makes MFP simultaneously a *variable selection* and a
*non-linear transformation discovery* algorithm, more parsimonious than
high-degree polynomials and more interpretable than purely data-driven
smoothers.

For full algorithmic details see
:class:`~knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class.MFPRegressor`
and Royston & Sauerbrei (2008).

Structure-Preserving Refitting
--------------------------------
Because MFP discovers its own structure during fitting, applying a model
to a *different* dataset (e.g. alternative-fuel ships) requires a
decision: re-discover structure, or force the *same* structure?

For cross-dataset coefficient comparison (a core requirement of this
research), the same transformation structure must be preserved.  This is
achieved by passing ``fixed_transforms`` to
:class:`~MFPRegressor` — see
:meth:`~knowledge_layer.parametric_regression.parametric_model_description_class.ModelDescriptor._refit_mfp_fixed_structure`.

References
----------
Royston, P., & Sauerbrei, W. (2008). *Multivariable Model-Building: A
Pragmatic Approach to Regression Analysis Based on Fractional Polynomials
for Modelling Continuous Variables*. Wiley.
"""

from typing import List, Tuple, Optional

import numpy as np

from knowledge_layer.parametric_regression.base_model_fitter_class import BaseModelFitter
from knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class import MFPRegressor


class MFPFitter(BaseModelFitter):
    """
    Concrete fitter that wraps
    :class:`~knowledge_layer.parametric_regression.multivariable_fractional_polynomials_class.MFPRegressor`.

    This class is intentionally thin — all algorithmic logic lives in
    :class:`MFPRegressor`.  :class:`MFPFitter` exists solely to satisfy
    the :class:`~knowledge_layer.parametric_regression.base_model_fitter_class.BaseModelFitter`
    interface so that MFP models can participate in the
    :class:`~knowledge_layer.parametric_regression.parametric_model_factory.ModelFactory`
    dispatch system.
    """

    # =========================================================================
    # BaseModelFitter Interface
    # =========================================================================

    def get_available_models(self) -> List[str]:
        """
        Return the single model-type string handled by this fitter.

        :returns: ``['mfp']``
        :rtype: list[str]
        """
        return ['mfp']

    def is_applicable(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 5,
    ) -> Tuple[bool, str]:
        """
        Check whether the MFP algorithm can be applied to the data.

        MFP requires more samples than predictors because it must fit a
        linear regression to all remaining variables as part of the
        likelihood-ratio test at each step of backward elimination.  With
        ``n_samples ≤ n_features`` the system is underdetermined and the
        test statistics are meaningless.

        A rule of thumb of at least 10 observations per predictor is
        recommended for stable transformation discovery (Royston &
        Sauerbrei, 2008, §4.2).  This threshold is noted internally but
        does not block applicability — the call still proceeds with a
        warning logged at a higher level.

        :param model_type: Must be ``'mfp'``.
        :type model_type: str
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Target vector, shape ``(n_samples,)``.  Unused in
            constraint checking; accepted for interface compatibility.
        :type y: numpy.ndarray
        :param min_samples: Minimum number of required samples.
        :type min_samples: int

        :returns: ``(True, '')`` if applicable; ``(False, reason)``
            otherwise.
        :rtype: tuple[bool, str]
        """
        if model_type != 'mfp':
            return False, f"model type '{model_type}' not handled by MFPFitter"

        n_samples = len(X)
        n_features = X.shape[1] if X.ndim > 1 else 1

        if n_samples < min_samples:
            return False, f"insufficient samples ({n_samples} < {min_samples})"

        if n_samples <= n_features:
            return False, (
                f"insufficient samples for MFP "
                f"({n_samples} samples ≤ {n_features} predictors)"
            )

        return True, ""

    def create_and_fit(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        alpha1: Optional[float] = None,
        alpha2: Optional[float] = None,
        powers: Optional[List[float]] = None,
        max_cycles: int = 20,
        verbose: bool = False,
        fixed_transforms: Optional[dict] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Instantiate and fit an :class:`MFPRegressor`.

        When ``fixed_transforms`` is provided, the algorithm skips backward
        elimination and function selection entirely and fits ordinary linear
        regression using the pre-specified transformation structure.  This
        is used for cross-dataset refitting where structural comparability
        must be preserved.

        For full algorithmic details see
        :meth:`MFPRegressor.fit` and
        :meth:`MFPRegressor._fit_with_fixed_transforms`.

        :param model_type: Must be ``'mfp'``.
        :type model_type: str
        :param X: Training predictor matrix, shape
            ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param y: Training target vector, shape ``(n_samples,)``.
        :type y: numpy.ndarray
        :param alpha1: Significance level for backward elimination
            (variable removal).  Lower values retain more variables.
            Defaults to 0.05.
        :type alpha1: float, optional
        :param alpha2: Significance level for function selection procedure
            (transformation selection).  Lower values prefer simpler
            transformations.  Defaults to 0.05.
        :type alpha2: float, optional
        :param powers: Custom power set to search over.  When ``None``
            the default set defined in
            :attr:`MFPRegressor.DEFAULT_POWERS` is used.
        :type powers: list[float], optional
        :param max_cycles: Maximum number of BE → FSP cycles before
            stopping, regardless of convergence.  Defaults to 20.
        :type max_cycles: int
        :param verbose: If ``True``, :class:`MFPRegressor` prints
            detailed progress during each cycle.  Defaults to ``False``.
        :type verbose: bool
        :param fixed_transforms: Pre-specified transformation structure in
            MFP internal format: ``{var_idx: (transform_type, powers)}``.
            When provided, the discovery algorithm is bypassed.  Typical
            use case: applying the structure discovered on conventional
            ships to an alternative-fuel dataset.
        :type fixed_transforms: dict, optional
        :param feature_names: Names for the predictor columns, forwarded to
            :meth:`MFPRegressor.fit`.  When provided, coefficient keys in
            the fitted model's ``feature_names_in_`` attribute will use
            these names (e.g. ``'√DWT'``) rather than the fallback
            auto-generated names (e.g. ``'√x0'``).  This is required for
            :meth:`~ModelDescriptor.from_fitted_model` to produce correctly
            labelled coefficient dictionaries.
        :type feature_names: list[str], optional

        :returns: ``(mfp, None)`` where ``mfp`` is the fitted
            :class:`MFPRegressor`.  No preprocessor is needed; MFP stores
            and applies its transformations internally.
        :rtype: tuple[MFPRegressor, None]
        """
        mfp = MFPRegressor(
            alpha1=alpha1 if alpha1 is not None else 0.05,
            alpha2=alpha2 if alpha2 is not None else 0.05,
            powers=powers,
            max_cycles=max_cycles,
            fixed_transforms=fixed_transforms,
            verbose=verbose,
        )
        mfp.fit(X, y, feature_names=feature_names)
        return mfp, None

    def predict(
        self,
        model,
        preprocessor,
        X: np.ndarray,
        model_type: str,
    ) -> np.ndarray:
        """
        Delegate prediction to :meth:`MFPRegressor.predict`.

        :class:`MFPRegressor` stores the discovered transformation
        structure in :attr:`~MFPRegressor.transforms_` and applies it
        internally when :meth:`~MFPRegressor.predict` is called — no
        external preprocessor is required.

        :param model: Fitted :class:`MFPRegressor`.
        :param preprocessor: Unused (always ``None`` for MFP); accepted
            for interface compatibility.
        :param X: Predictor matrix, shape ``(n_samples, n_features)``.
        :type X: numpy.ndarray
        :param model_type: Must be ``'mfp'`` (accepted for interface
            compatibility; not inspected here).
        :type model_type: str

        :returns: Predicted values, shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        return model.predict(X)

    def get_n_parameters(
        self,
        model_type: str,
        X: np.ndarray,
        fitted_model: Optional[MFPRegressor] = None,
    ) -> int:
        """
        Return the effective number of free parameters for the MFP model.

        The parameter count depends on which transformations were
        discovered during fitting:

        - 1 for the intercept.
        - +1 for each variable assigned a *linear* or *FP1* transformation.
        - +2 for each variable assigned an *FP2* transformation (two
          regression terms).
        - +0 for each excluded variable.

        When ``fitted_model`` is ``None`` (i.e. before fitting), the
        worst-case maximum is returned: all variables as FP2 gives
        ``2 × n_features + 1``.  This is used for a conservative
        applicability pre-check, not for actual AIC / BIC computation.

        :param model_type: Must be ``'mfp'``.
        :type model_type: str
        :param X: Predictor matrix used to determine feature count.
        :type X: numpy.ndarray
        :param fitted_model: A fitted :class:`MFPRegressor`.  When
            provided, the *actual* parameter count is computed from the
            :attr:`~MFPRegressor.transforms_` attribute.  Required for
            correct AIC / BIC computation after fitting.
            Defaults to ``None``.

        :returns: Effective number of free parameters.
        :rtype: int

        :raises ValueError: If ``fitted_model`` is provided but lacks the
            ``transforms_`` attribute (i.e. has not been fitted).
        """
        if fitted_model is not None:
            if not hasattr(fitted_model, 'transforms_'):
                raise ValueError(
                    "fitted_model does not have a 'transforms_' attribute — "
                    "ensure the MFPRegressor has been fitted before calling "
                    "get_n_parameters with a fitted_model."
                )
            n_params = 1  # intercept
            for _, (transform_type, _) in fitted_model.transforms_.items():
                if transform_type == 'excluded':
                    continue
                elif transform_type in ('linear', 'fp1'):
                    n_params += 1
                elif transform_type == 'fp2':
                    n_params += 2
            return n_params

        # Pre-fit estimate: worst case (all variables as FP2)
        n_features = X.shape[1] if X.ndim > 1 else 1
        return 2 * n_features + 1

    def get_model_description(self, model_type: str) -> str:
        """
        Return a concise description of the MFP model.

        :param model_type: Must be ``'mfp'``.
        :type model_type: str

        :returns: Human-readable description.
        :rtype: str
        """
        return (
            'MFP: Multivariable Fractional Polynomials — '
            'automatically discovers optimal non-linear transformations '
            'and performs backward variable selection.'
        )