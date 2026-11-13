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
    #: Model-type keys handled by this fitter, and the MFPRegressor
    #: configuration each one denotes.
    #:
    #: 'mfp'  is the legacy key, retained so existing configurations and
    #:        stored results remain reproducible.
    #: 'mfp1' permits {excluded, linear, FP1}: one regression term per
    #:        retained variable.
    #: 'mfp2' permits {excluded, linear, FP1, FP2}.
    #:
    #: 'mfp1' and 'mfp2' both use df_accounting='royston'.  This matters:
    #: 'mfp1' *requires* it (linear and FP1 contribute the same number of
    #: terms, so under term-count accounting the linear-vs-FP1 test has
    #: zero degrees of freedom and FP1 can never be selected), and giving
    #: 'mfp2' the same convention keeps the pair differing in exactly one
    #: respect: the permitted family.  A comparison between an FP1 model
    #: on 'royston' df and an FP2 model on 'terms' df would confound the
    #: family with the testing convention.
    MODEL_CONFIG = {
        'mfp': {'max_degree': 2, 'df_accounting': 'terms'},
        'mfp1': {'max_degree': 1, 'df_accounting': 'royston'},
        'mfp2': {'max_degree': 2, 'df_accounting': 'royston'},
    }

    # =========================================================================
    # BaseModelFitter Interface
    # =========================================================================

    def get_available_models(self) -> List[str]:
        return list(self.MODEL_CONFIG)

    def is_applicable(self, model_type, X, y, min_samples: int = 5):
        if model_type not in self.MODEL_CONFIG:
            return False, f"model type '{model_type}' not handled by MFPFitter"

        n_samples = len(X)
        n_features = X.shape[1] if X.ndim > 1 else 1

        if n_samples < min_samples:
            return False, f"insufficient samples ({n_samples} < {min_samples})"

        # Terms in the largest model the closed test may reach.  MFP fits
        # this model at every step of the function selection procedure, so
        # the design must be overdetermined for it, not merely for the
        # model that is eventually selected.
        max_degree = self.MODEL_CONFIG[model_type]['max_degree']
        max_terms = (1 if max_degree <= 1 else 2) * n_features

        if n_samples <= max_terms:
            return False, (
                f"insufficient samples for {model_type} "
                f"({n_samples} samples ≤ {max_terms} terms at "
                f"max_degree={max_degree})"
            )

        return True, ""

    @classmethod
    def _config(cls, model_type: str) -> dict:
        """Return the MFPRegressor configuration for a model-type key."""
        try:
            return cls.MODEL_CONFIG[model_type]
        except KeyError:
            raise ValueError(
                f"model type '{model_type}' not handled by MFPFitter; "
                f"expected one of {sorted(cls.MODEL_CONFIG)}"
            ) from None

    def create_and_fit(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        alpha1: Optional[float] = None,
        alpha2: Optional[float] = None,
        powers: Optional[List[float]] = None,
        max_cycles: int = 20,
            max_degree=None,
            df_accounting=None,
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
        config = self._config(model_type)

        mfp = MFPRegressor(
            alpha1=alpha1 if alpha1 is not None else 0.05,
            alpha2=alpha2 if alpha2 is not None else 0.05,
            powers=powers,
            max_cycles=max_cycles,
            max_degree=(max_degree if max_degree is not None
                        else config['max_degree']),
            df_accounting=(df_accounting if df_accounting is not None
                           else config['df_accounting']),
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

        # Pre-fit estimate: worst case within the permitted family
        n_features = X.shape[1] if X.ndim > 1 else 1
        max_degree = self._config(model_type)['max_degree']
        return (1 if max_degree <= 1 else 2) * n_features + 1

    def get_model_description(self, model_type: str) -> str:
        """
        Return a concise description of the MFP model.

        :param model_type: Must be ``'mfp'``.
        :type model_type: str

        :returns: Human-readable description.
        :rtype: str
        """
        base = ('MFP: Multivariable Fractional Polynomials, discovers '
                'non-linear transformations and performs backward '
                'variable selection')
        detail = {
            'mfp': 'up to FP2 per variable (legacy df accounting)',
            'mfp1': 'at most one term per variable (FP1)',
            'mfp2': 'up to two terms per variable (FP2)',
        }[model_type]
        return f'{base}; {detail}.'