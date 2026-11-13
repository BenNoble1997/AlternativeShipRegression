"""
Parametric Formula Renderer
============================

Role in the Architecture
-------------------------
:class:`ParametricFormulaRenderer` is a stateless utility class that
converts the structural information stored in a
:class:`~knowledge_layer.parametric_regression.parametric_model_description_class.ModelDescriptor`
into human-readable formula strings.

It was extracted from :class:`ModelDescriptor` during a refactor to
separate *rendering* (a display concern) from *structure and refitting*
(a modelling concern).  The dependency is strictly one-directional:

    ModelDescriptor  →  ParametricFormulaRenderer

:class:`ModelDescriptor` calls :class:`ParametricFormulaRenderer` for all
formula rendering.  Nothing in the fitting layer calls this class directly.

What Moved Here
---------------
The following methods were moved from :class:`ModelDescriptor` verbatim,
with ``self`` replaced by explicit parameters:

- ``get_formula_string``          → :meth:`render`
- ``_get_mfp_formula``            → :meth:`_render_mfp`
- ``_get_polynomial_formula``     → :meth:`_render_polynomial`
- ``_get_linear_formula``         → :meth:`_render_linear`
- ``_get_simple_nonlinear_formula``→ :meth:`_render_simple_nonlinear`
- ``_get_log_transform_formula``  → :meth:`_render_log_transform`
- ``_format_power``               → :meth:`format_power`  (public utility)
- ``_format_power_name``          → :meth:`format_power_name`  (public utility)
- ``_format_coefficient``         → :meth:`format_coefficient`  (public utility)
- ``_simplify_power``             → :meth:`simplify_power`  (public utility)
- ``_get_transform_name``         → :meth:`_get_transform_name`
- ``_get_fp2_second_term_name``   → :meth:`get_fp2_second_term_name`  (public utility)

Public vs Private
-----------------
The four core formatting utilities — :meth:`format_power`,
:meth:`format_power_name`, :meth:`format_coefficient`, and
:meth:`simplify_power` — are made **public** here because they are
genuinely useful to external code (e.g. the LaTeX table generator) and
have no side effects.  All rendering methods that assemble full formula
strings remain private.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np


class ParametricFormulaRenderer:
    """
    Stateless utility class for rendering parametric model formulas.

    All methods are class methods or static methods — this class is never
    instantiated.  It accepts the structural fields of a
    :class:`~ModelDescriptor` as plain arguments, keeping it decoupled
    from the descriptor's class hierarchy.

    Typical usage (inside :class:`ModelDescriptor`)::

        formula = ParametricFormulaRenderer.render(
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
    """

    # =========================================================================
    # Notation constants
    # =========================================================================

    #: Translation table mapping ASCII digits to Unicode subscripts (U+2080–U+2089).
    #: These render correctly in matplotlib titles and tick labels with no LaTeX backend.
    _SUBSCRIPT_DIGITS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

    #: Coefficient keys that never receive a β index of their own.
    #: ``'1'`` is the PolynomialFeatures bias column, which duplicates the intercept.
    BIAS_KEYS = frozenset({'intercept', '1'})

    @classmethod
    def beta_symbol(cls, index: int, notation: str = 'math') -> str:
        """
        Return the canonical symbol for the ``index``-th model coefficient.

        Centralises β typography.  Previously each renderer built the symbol
        inline, producing ``'β₀'`` for the intercept but ``'β1'``, ``'β2'`` for
        every subsequent term — two typographies inside a single formula — and
        emitting ``'beta_1'`` rather than ``'\\beta_{1}'`` under LaTeX notation.

        :param index: Zero-based coefficient index.  Index 0 is conventionally
            the intercept for additive models, or the leading multiplicative
            constant for power / exponential forms.
        :type index: int
        :param notation: ``'math'`` (Unicode subscripts), ``'latex'``, or
            ``'python'``.
        :type notation: str

        :returns: e.g. ``'β₁₀'``, ``'\\beta_{10}'``, ``'beta_10'``.
        :rtype: str
        """
        if notation == 'latex':
            return rf"\beta_{{{index}}}"
        if notation == 'python':
            return f"beta_{index}"
        return 'β' + str(index).translate(cls._SUBSCRIPT_DIGITS)

    @classmethod
    def build_beta_index_map(
            cls,
            coefficients: Optional[Dict[str, float]],
            intercept: Optional[float],
    ) -> Dict[str, int]:
        """
        Return the canonical β index assigned to each stored coefficient.

        **This is the single source of truth for coefficient numbering.**  Every
        renderer here consumes it, :meth:`ModelDescriptor.get_beta_index_map`
        delegates to it, and the coefficient-change plotter reads it through the
        descriptor — so the β₂ in a printed formula labels the same quantity as
        the bar annotated β₂ beneath it, by construction rather than by
        coincidence.

        Ordering convention:

        1. ``'intercept'`` takes index 0 whenever ``intercept`` is not ``None``.
        2. Remaining coefficients follow in dict insertion order, which mirrors
           the order the fitting code wrote them.
        3. :attr:`BIAS_KEYS` are excluded.

        Models with no additive intercept (all ``power`` / ``exponential``
        forms, which store their leading constant as coefficient ``'a'``)
        therefore start numbering at β₀ on that constant, instead of the
        previous behaviour of starting at β₁ with no β₀ present.

        :param coefficients: ``{feature_name: value}`` dict, or ``None``.
        :type coefficients: dict, optional
        :param intercept: Fitted intercept, or ``None``.
        :type intercept: float, optional

        :returns: Mapping ``{coefficient_name: beta_index}``.
        :rtype: dict[str, int]
        """
        ordering: List[str] = []

        if intercept is not None:
            ordering.append('intercept')

        if coefficients:
            ordering.extend(n for n in coefficients if n not in cls.BIAS_KEYS)

        return {name: idx for idx, name in enumerate(ordering)}

    @classmethod
    def suppressed_coefficients(
            cls,
            coefficients: Optional[Dict[str, float]],
            regularization_type: Optional[str] = None,
            active_features: Optional[List[str]] = None,
            only_active: bool = True,
            zero_tolerance: float = 1e-10,
    ) -> set:
        """
        Return the coefficient names that the renderers omit from a formula.

        Companion to :meth:`build_beta_index_map`.  The two are deliberately
        separate: the index map stays **complete** so that suppressing a term
        leaves a gap (β₀, β₁, β₃) rather than renumbering the survivors, which
        would silently relabel the cubic term as β₂.  Suppression is therefore a
        second, independent decision — and this method is its single definition,
        so the formula and the coefficient-change plot omit exactly the same
        terms.

        Only Lasso families suppress anything.  For those, a coefficient is
        suppressed when it is absent from ``active_features`` (when that list is
        available) or when its magnitude is within ``zero_tolerance`` of zero.

        :param coefficients: ``{feature_name: value}`` dict, or ``None``.
        :type coefficients: dict, optional
        :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        :type regularization_type: str, optional
        :param active_features: Names retained by the Lasso fit.
        :type active_features: list[str], optional
        :param only_active: When ``False``, nothing is suppressed.
        :type only_active: bool
        :param zero_tolerance: Magnitude at or below which a coefficient counts
            as zeroed.
        :type zero_tolerance: float

        :returns: Set of suppressed coefficient names.  Never includes
            ``'intercept'``, which Lasso does not penalise.
        :rtype: set[str]
        """
        if not only_active or regularization_type != 'lasso' or not coefficients:
            return set()

        active = set(active_features) if active_features else None

        return {
            name for name, value in coefficients.items()
            if name not in cls.BIAS_KEYS
               and (
                       (active is not None and name not in active)
                       or abs(value) <= zero_tolerance
               )
        }

    @classmethod
    def resolve_log_exponents(
            cls,
            coefficients: Optional[Dict[str, float]],
            independent_vars: List[str],
    ) -> List[Tuple[str, Optional[str], Optional[float]]]:
        """
        Resolve the per-predictor exponents of a log-transform model.

        Coefficient keys for these models have been written under two
        conventions in this codebase: ``'b_{i}_{var}'`` by
        :meth:`~ModelDescriptor.from_fitted_model`, and the bare variable name
        by the pre-fix refit path.  A rigid lookup against one convention hit
        the ``.get()`` default of ``0.0``, which rendered as a
        plausible-but-fabricated exponent of ``0.000``.

        This method tries every known convention and returns ``None`` — never a
        fabricated zero — when a coefficient genuinely cannot be located,
        following the *invalidate rather than silently substitute* principle.
        The caller then falls back to a β symbol, which is visibly a symbol
        rather than a value.

        :param coefficients: ``{feature_name: value}`` dict, or ``None``.
        :type coefficients: dict, optional
        :param independent_vars: Predictor names, in model column order.
        :type independent_vars: list[str]

        :returns: One tuple per predictor:
            ``(variable_name, resolved_key_or_None, value_or_None)``.
        :rtype: list[tuple[str, str | None, float | None]]
        """
        coefs = coefficients or {}
        positional = [(k, v) for k, v in coefs.items() if k not in cls.BIAS_KEYS]

        resolved: List[Tuple[str, Optional[str], Optional[float]]] = []

        for i, var in enumerate(independent_vars):
            candidates = [f'b_{i}_{var}', f'b_{i}', f'b{i}', var]
            key = next((c for c in candidates if c in coefs), None)

            if key is not None:
                resolved.append((var, key, float(coefs[key])))
            elif len(positional) == len(independent_vars):
                # Positional fallback, only when the arity matches exactly
                k, v = positional[i]
                resolved.append((var, k, float(v)))
            else:
                resolved.append((var, None, None))

        return resolved

    @classmethod
    def _term_prefix(
            cls,
            value: Optional[float],
            index: int,
            include_coeffs: bool,
            notation: str,
            sci_threshold: float,
            sci_max: float,
            leading: bool = False,
    ) -> str:
        """
        Return the signed coefficient string, or the β symbol, for one term.

        Shared by all renderers so that the value-vs-symbol decision and the β
        typography are made in exactly one place.

        :param value: Coefficient value, or ``None`` if unresolved.
        :type value: float, optional
        :param index: Canonical β index for this term.
        :type index: int
        :param include_coeffs: Render the numeric value when available.
        :type include_coeffs: bool
        :param notation: ``'math'``, ``'latex'``, or ``'python'``.
        :type notation: str
        :param sci_threshold: Lower magnitude bound for regular notation.
        :type sci_threshold: float
        :param sci_max: Upper magnitude bound for regular notation.
        :type sci_max: float
        :param leading: When ``True`` the ``'+'`` prefix is suppressed — used
            for the first term of an expression.
        :type leading: bool

        :returns: e.g. ``'+0.523'``, ``'-0.089'``, ``'+β₂'``, ``'β₀'``.
        :rtype: str
        """
        if include_coeffs and value is not None:
            text = cls.format_coefficient(value, notation, sci_threshold, sci_max)
            return text.lstrip('+') if leading else text

        symbol = cls.beta_symbol(index, notation)
        return symbol if leading else f"+{symbol}"

    # =========================================================================
    # Primary Entry Point
    # =========================================================================

    @classmethod
    def render(
        cls,
        model_type: str,
        dependent_var: str,
        independent_vars: List[str],
        transformations: Optional[Dict],
        coefficients: Optional[Dict[str, float]],
        intercept: Optional[float],
        regularization_type: Optional[str],
        active_features: Optional[List[str]],
        polynomial_degree: Optional[int],
        metadata: Optional[Dict],
        include_coefficients: bool = True,
        notation: str = 'math',
        only_active: bool = True,
        sci_threshold: float = 1e-3,
        sci_max: float = 1e4,
    ) -> str:
        """
        Render a formula string for a parametric model.

        Dispatches to the appropriate private renderer based on
        ``model_type``.

        :param model_type: Model-type identifier string.
        :type model_type: str
        :param dependent_var: Name of the target variable.
        :type dependent_var: str
        :param independent_vars: Names of predictor variables.
        :type independent_vars: list[str]
        :param transformations: MFP transformation dict
            ``{var_idx: (transform_type, powers)}``, or ``None``.
        :type transformations: dict, optional
        :param coefficients: ``{feature_name: value}`` dict, or ``None``.
        :type coefficients: dict, optional
        :param intercept: Fitted intercept, or ``None``.
        :type intercept: float, optional
        :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        :type regularization_type: str, optional
        :param active_features: Non-zero Lasso feature names, or ``None``.
        :type active_features: list[str], optional
        :param polynomial_degree: Polynomial degree, or ``None``.
        :type polynomial_degree: int, optional
        :param metadata: Supplementary metadata dict (used for nonlinear
            models to recover ``popt``).
        :type metadata: dict, optional
        :param include_coefficients: Embed actual coefficient values
            (``True``) or generic β symbols (``False``).
        :type include_coefficients: bool
        :param notation: ``'math'``, ``'latex'``, or ``'python'``.
        :type notation: str
        :param only_active: For Lasso models, suppress zero-coefficient
            terms from the output.
        :type only_active: bool
        :param sci_threshold: Use scientific notation when
            ``|coef| < sci_threshold``.
        :type sci_threshold: float
        :param sci_max: Use scientific notation when ``|coef| > sci_max``.
        :type sci_max: float

        :returns: Formatted formula string.
        :rtype: str
        """
        kwargs = dict(
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            coefficients=coefficients,
            intercept=intercept,
            include_coeffs=include_coefficients,
            notation=notation,
            sci_threshold=sci_threshold,
            sci_max=sci_max,
        )

        if model_type == 'mfp':
            return cls._render_mfp(transformations=transformations, **kwargs)

        if model_type.startswith('polynomial'):
            return cls._render_polynomial(
                polynomial_degree=polynomial_degree,
                regularization_type=regularization_type,
                active_features=active_features,
                only_active=only_active,
                **kwargs,
            )

        if model_type in ['linear', 'ridge', 'lasso', 'ridge-cv', 'lasso-cv']:
            return cls._render_linear(
                regularization_type=regularization_type,
                only_active=only_active,
                **kwargs,
            )

        if model_type in [
            'exponential', 'exponential-offset',
            'power', 'power-offset',
            'logarithmic', 'logarithmic-power',
        ]:
            return cls._render_simple_nonlinear(
                model_type=model_type,
                metadata=metadata,
                **kwargs,
            )

        if model_type in ['power-log-multivariate', 'exponential-log-multivariate']:
            return cls._render_log_transform(
                model_type=model_type,
                **kwargs,
            )

        # Generic fallback
        return f"{dependent_var} ~ {model_type}({', '.join(independent_vars)})"

    # =========================================================================
    # Model-Type Renderers (private)
    # =========================================================================

    @classmethod
    def _render_mfp(
        cls,
        dependent_var: str,
        independent_vars: List[str],
        transformations: Optional[Dict],
        coefficients: Optional[Dict[str, float]],
        intercept: Optional[float],
        include_coeffs: bool,
        notation: str,
        sci_threshold: float,
        sci_max: float,
    ) -> str:
        """
        Render an MFP formula from stored transformations and coefficients.

        Iterates over :attr:`transformations` in variable-index order.
        FP2 variables generate two terms and are handled with ``continue``
        to skip the generic ``terms.append`` at the end of the loop.

        :param transformations: ``{var_idx: (transform_type, powers)}``.
        :type transformations: dict, optional
        """
        def _mul():
            return ' \\cdot ' if notation == 'latex' else ' * ' if notation == 'python' else '·'

        def _coef_str(val, idx):
            return cls._term_prefix(
                val, idx, include_coeffs, notation, sci_threshold, sci_max,
            )

        terms = []
        if include_coeffs and intercept is not None:
            terms.append(
                cls.format_coefficient(intercept, notation, sci_threshold, sci_max)
                .lstrip('+')
            )
        else:
            terms.append(cls.beta_symbol(0, notation))

        if not transformations:
            return f"{dependent_var} = {terms[0]}"

        coef_idx = 1 if intercept is not None else 0
        for var_idx, var_name in enumerate(independent_vars):
            if var_idx not in transformations:
                continue

            t_type, powers = transformations[var_idx]
            if t_type == 'excluded':
                continue

            coefs = coefficients or {}

            if t_type == 'linear':
                c = _coef_str(coefs.get(var_name), coef_idx)
                terms.append(f"{c}{_mul()}{var_name}")
                coef_idx += 1

            elif t_type == 'fp1':
                t_str = cls.format_power(var_name, powers[0], notation)
                name  = cls.format_power_name(var_name, powers[0])
                c     = _coef_str(coefs.get(name), coef_idx)
                terms.append(f"{c}{_mul()}{t_str}")
                coef_idx += 1

            elif t_type == 'fp2':
                if len(powers) == 1:
                    p = powers[0]
                    t_str1 = cls.format_power(var_name, p, notation)
                    n1     = cls.format_power_name(var_name, p)
                    c1     = _coef_str(coefs.get(n1), coef_idx)
                    terms.append(f"{c1}{_mul()}{t_str1}")
                    coef_idx += 1

                    n2 = cls.get_fp2_second_term_name(var_name, p)
                    if notation == 'latex':
                        t_str2 = (
                            f"(\\ln({var_name}))^2" if p == 0
                            else f"{t_str1} \\cdot \\ln({var_name})"
                        )
                    else:
                        t_str2 = (
                            f"(ln({var_name}))²" if p == 0
                            else f"{t_str1}·ln({var_name})"
                        )
                    c2 = _coef_str(coefs.get(n2), coef_idx)
                    terms.append(f"{c2}{_mul()}{t_str2}")
                    coef_idx += 1

                else:
                    for p in powers:
                        t_str = cls.format_power(var_name, p, notation)
                        n     = cls.format_power_name(var_name, p)
                        c     = _coef_str(coefs.get(n), coef_idx)
                        terms.append(f"{c}{_mul()}{t_str}")
                        coef_idx += 1

        formula = f"{dependent_var} = {' '.join(terms)}"
        return formula.replace("+ -", "- ").replace("+-", "-")

    @classmethod
    def _render_polynomial(
            cls,
            dependent_var: str,
            independent_vars: List[str],
            coefficients: Optional[Dict[str, float]],
            intercept: Optional[float],
            polynomial_degree: Optional[int],
            regularization_type: Optional[str],
            active_features: Optional[List[str]],
            only_active: bool,
            include_coeffs: bool,
            notation: str,
            sci_threshold: float,
            sci_max: float,
    ) -> str:
        """
        Render a polynomial (or polynomial-regularised) formula.

        For Lasso, terms absent from ``active_features`` or with effectively
        zero coefficient are suppressed when ``only_active=True``.  Suppressed
        terms leave gaps in the β sequence — the index identifies the
        parameter, so the gap is informative.

        β indices now come from :meth:`build_beta_index_map`, replacing the
        local counter that renumbered terms after each suppression and so
        disagreed with the coefficient-change bar labels.

        :param polynomial_degree: Polynomial degree.
        :type polynomial_degree: int, optional
        :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        :type regularization_type: str, optional
        :param active_features: Non-zero Lasso feature names.
        :type active_features: list[str], optional
        :param only_active: Suppress zero Lasso terms.
        :type only_active: bool
        """
        if not coefficients:
            return f"{dependent_var} = polynomial(degree={polynomial_degree})"

        beta_map = cls.build_beta_index_map(coefficients, intercept)

        terms = []
        if intercept is not None:
            terms.append(cls._term_prefix(
                intercept, beta_map['intercept'],
                include_coeffs, notation, sci_threshold, sci_max, leading=True,
            ))

        for feature_name, coef_val in coefficients.items():
            if feature_name in cls.BIAS_KEYS:
                continue

            if only_active and regularization_type == 'lasso':
                if active_features and feature_name not in active_features:
                    continue
                if abs(coef_val) < 1e-10:
                    continue

            c_str = cls._term_prefix(
                coef_val, beta_map.get(feature_name, len(beta_map)),
                include_coeffs, notation, sci_threshold, sci_max,
            )

            if notation == 'latex':
                terms.append(f"{c_str} \\cdot {feature_name.replace('*', ' ')}")
            elif notation == 'python':
                terms.append(f"{c_str} * {feature_name}")
            else:
                terms.append(f"{c_str}·{feature_name.replace('*', '·').replace(' ', '')}")

        formula = f"{dependent_var} = {' '.join(terms)}"
        return formula.replace("+ -", "- ").replace("+-", "-")

    @classmethod
    def _render_linear(
            cls,
            dependent_var: str,
            independent_vars: List[str],
            coefficients: Optional[Dict[str, float]],
            intercept: Optional[float],
            regularization_type: Optional[str],
            only_active: bool,
            include_coeffs: bool,
            notation: str,
            sci_threshold: float,
            sci_max: float,
    ) -> str:
        """
        Render a linear / Ridge / Lasso formula.

        β indices come from :meth:`build_beta_index_map` rather than a local
        counter.  The local counter started at 1 unconditionally, so a model
        with no intercept opened at β₁ with no β₀ anywhere in the expression —
        and disagreed by one with the bar labels in the coefficient-change plot.

        Note that suppressed Lasso terms leave gaps in the β sequence (β₀, β₂,
        β₅).  This is deliberate: the index identifies the parameter, so a gap
        signals a zeroed coefficient rather than being renumbered away.

        :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        :type regularization_type: str, optional
        :param only_active: Suppress zero Lasso terms.
        :type only_active: bool
        """
        if not coefficients:
            return f"{dependent_var} = linear({', '.join(independent_vars)})"

        beta_map = cls.build_beta_index_map(coefficients, intercept)
        mul = ' \\cdot ' if notation == 'latex' else ' * ' if notation == 'python' else '·'

        terms = []
        if intercept is not None:
            terms.append(cls._term_prefix(
                intercept, beta_map['intercept'],
                include_coeffs, notation, sci_threshold, sci_max, leading=True,
            ))

        for var_name in independent_vars:
            coef_val = coefficients.get(var_name)
            if coef_val is None:
                continue
            if only_active and regularization_type == 'lasso' and abs(coef_val) < 1e-10:
                continue

            c_str = cls._term_prefix(
                coef_val, beta_map.get(var_name, len(beta_map)),
                include_coeffs, notation, sci_threshold, sci_max,
            )
            terms.append(f"{c_str}{mul}{var_name}")

        formula = f"{dependent_var} = {' '.join(terms)}"
        return formula.replace("+ -", "- ").replace("+-", "-")

    @classmethod
    def _render_simple_nonlinear(
            cls,
            model_type: str,
            dependent_var: str,
            independent_vars: List[str],
            coefficients: Optional[Dict[str, float]],
            intercept: Optional[float],
            metadata: Optional[Dict],
            include_coeffs: bool,
            notation: str,
            sci_threshold: float,
            sci_max: float,
            debug: bool = False,
    ) -> str:
        """
        Render a univariate nonlinear formula.

        Coefficients are stored under keys ``'a'``, ``'b'``, ``'c'``.
        When absent, reconstruction is attempted from ``metadata['popt']``.

        These families carry **no additive intercept**: the leading constant is
        coefficient ``'a'``, so ``intercept`` is ``None`` by contract (see
        :meth:`ModelDescriptor.from_fitted_model`).  β numbering therefore runs
        ``a → β₀``, ``b → β₁``, ``c → β₂``, which is what
        :meth:`build_beta_index_map` returns for an ``intercept=None``
        descriptor — the printed formula and the plotted bar labels agree by
        construction rather than by coincidence.

        .. note::
           The single predictor name is bound to ``var`` here.  Previously the
           templates below referenced ``var`` without ever binding it, so every
           call raised ``NameError`` and callers silently fell back to a generic
           additive-linear string built from the raw parameter names — e.g.
           ``BEAM (M) = β₀ + β₁·a + β₂·b + β₃·c`` instead of
           ``BEAM (M) = β₀·DWT^β₁ + β₂``.

        :param model_type: Specific nonlinear model type.
        :type model_type: str
        :param dependent_var: Target variable name.
        :type dependent_var: str
        :param independent_vars: Predictor names; must have length 1.
        :type independent_vars: list[str]
        :param coefficients: ``{'a': …, 'b': …, 'c': …}``, or ``None``.
        :type coefficients: dict, optional
        :param intercept: Unused for this family; present for signature
            uniformity across renderers.
        :type intercept: float, optional
        :param metadata: May contain ``'popt'`` array for coefficient
            reconstruction.
        :type metadata: dict, optional
        :param include_coeffs: Render numeric values rather than β symbols.
        :type include_coeffs: bool
        :param notation: ``'math'``, ``'latex'``, or ``'python'``.
        :type notation: str
        :param sci_threshold: Lower magnitude bound for regular notation.
        :type sci_threshold: float
        :param sci_max: Upper magnitude bound for regular notation.
        :type sci_max: float
        :param debug: Emit stage-by-stage diagnostics to stdout.
        :type debug: bool

        :returns: Rendered formula, or a structural fallback of the form
            ``'y ~ power-offset(DWT)'`` when the parameters cannot be resolved.
        :rtype: str
        """
        fallback = (
            f"{dependent_var} ~ "
            f"{model_type}({', '.join(independent_vars)})"
        )

        if debug:
            print(f"[_render_simple_nonlinear] type={model_type} "
                  f"vars={independent_vars} keys={list((coefficients or {}).keys())} "
                  f"include_coeffs={include_coeffs} notation={notation}")

        # ------------------------------------------------------------------
        # Stage 1 — reconstruct coefficients from popt if absent
        # ------------------------------------------------------------------
        coefficients = dict(coefficients or {})
        if not coefficients:
            popt = (metadata or {}).get('popt')
            if popt is None:
                if debug:
                    print("  ✗ no coefficients and no metadata['popt'] → fallback")
                return fallback
            if model_type in [
                'exponential', 'logarithmic', 'power', 'logarithmic-power',
            ] and len(popt) >= 2:
                coefficients = {'a': float(popt[0]), 'b': float(popt[1])}
            elif model_type in ['exponential-offset', 'power-offset'] and len(popt) >= 3:
                coefficients = {
                    'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2]),
                }
            else:
                if debug:
                    print(f"  ✗ popt arity {len(popt)} incompatible with "
                          f"{model_type} → fallback")
                return fallback
            if debug:
                print(f"  reconstructed from popt: {coefficients}")

        # ------------------------------------------------------------------
        # Stage 2 — arity guard and predictor binding
        # ------------------------------------------------------------------
        if len(independent_vars) != 1:
            if debug:
                print(f"  ✗ expected 1 predictor, got {len(independent_vars)} "
                      f"→ fallback")
            return fallback

        var = independent_vars[0]

        # ------------------------------------------------------------------
        # Stage 3 — resolve a / b / c, positionally if the keys are unusual
        # ------------------------------------------------------------------
        vals = [v for k, v in coefficients.items() if k not in cls.BIAS_KEYS]

        def _param(name: str, position: int):
            if name in coefficients:
                return coefficients[name]
            return vals[position] if len(vals) > position else None

        a = _param('a', 0)
        b = _param('b', 1)
        c = coefficients.get('c', coefficients.get('offset'))
        if c is None and len(vals) > 2:
            c = vals[2]

        if a is None or b is None:
            if debug:
                print(f"  ✗ could not resolve a/b (a={a}, b={b}) → fallback")
            return fallback

        if debug:
            print(f"  resolved: var={var!r} a={a} b={b} c={c}")

        # ------------------------------------------------------------------
        # Stage 4a — β-symbol mode
        # ------------------------------------------------------------------
        if not include_coeffs:
            templates = {
                'exponential': {
                    'math': f"{dependent_var} = β₀·exp(β₁·{var})",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot \\exp(\\beta_1 \\cdot {var})",
                    'python': f"{dependent_var} = beta_0 * exp(beta_1 * {var})",
                },
                'exponential-offset': {
                    'math': f"{dependent_var} = β₀·exp(β₁·{var}) + β₂",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot \\exp(\\beta_1 \\cdot {var}) + \\beta_2",
                    'python': f"{dependent_var} = beta_0 * exp(beta_1 * {var}) + beta_2",
                },
                'power': {
                    'math': f"{dependent_var} = β₀·{var}^β₁",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot {var}^{{\\beta_1}}",
                    'python': f"{dependent_var} = beta_0 * {var}**beta_1",
                },
                'power-offset': {
                    'math': f"{dependent_var} = β₀·{var}^β₁ + β₂",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot {var}^{{\\beta_1}} + \\beta_2",
                    'python': f"{dependent_var} = beta_0 * {var}**beta_1 + beta_2",
                },
                'logarithmic': {
                    'math': f"{dependent_var} = β₀·ln({var}) + β₁",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot \\ln({var}) + \\beta_1",
                    'python': f"{dependent_var} = beta_0 * log({var}) + beta_1",
                },
                'logarithmic-power': {
                    'math': f"{dependent_var} = β₀·(ln({var}))^β₁",
                    'latex': f"{dependent_var} = \\beta_0 \\cdot (\\ln({var}))^{{\\beta_1}}",
                    'python': f"{dependent_var} = beta_0 * (log({var}))**beta_1",
                },
            }
            rendered = templates.get(model_type, {}).get(notation, fallback)
            if debug:
                print(f"  → β-symbol: {rendered}")
            return rendered

        # ------------------------------------------------------------------
        # Stage 4b — numeric mode
        # ------------------------------------------------------------------
        a_str = cls.format_coefficient(a, notation, sci_threshold, sci_max).lstrip('+')
        b_str = cls.format_coefficient(b, notation, sci_threshold, sci_max).lstrip('+')
        c_str = (
            cls.format_coefficient(c, notation, sci_threshold, sci_max)
            if c is not None else None
        )

        if model_type == 'exponential':
            formula = (
                f"{dependent_var} = {a_str} \\cdot \\exp({b_str} \\cdot {var})"
                if notation == 'latex'
                else f"{dependent_var} = {a_str}·exp({b_str}·{var})"
            )
        elif model_type == 'exponential-offset':
            base = (
                f"{a_str} \\cdot \\exp({b_str} \\cdot {var})"
                if notation == 'latex'
                else f"{a_str}·exp({b_str}·{var})"
            )
            formula = (
                f"{dependent_var} = {base} {c_str}"
                if c_str else f"{dependent_var} = {base}"
            )
        elif model_type == 'power':
            formula = (
                f"{dependent_var} = {a_str} \\cdot {var}^{{{b_str}}}"
                if notation == 'latex'
                else f"{dependent_var} = {a_str}·{var}^{b_str}"
            )
        elif model_type == 'power-offset':
            base = (
                f"{a_str} \\cdot {var}^{{{b_str}}}"
                if notation == 'latex'
                else f"{a_str}·{var}^{b_str}"
            )
            formula = (
                f"{dependent_var} = {base} {c_str}"
                if c_str else f"{dependent_var} = {base}"
            )
        elif model_type == 'logarithmic':
            formula = (
                f"{dependent_var} = {a_str} \\cdot \\ln({var}) {b_str}"
                if notation == 'latex'
                else f"{dependent_var} = {a_str}·ln({var}) {b_str}"
            )
        elif model_type == 'logarithmic-power':
            formula = (
                f"{dependent_var} = {a_str} \\cdot (\\ln({var}))^{{{b_str}}}"
                if notation == 'latex'
                else f"{dependent_var} = {a_str}·(ln({var}))^{b_str}"
            )
        else:
            formula = fallback

        formula = formula.replace("+ -", "- ").replace("+-", "-")

        if debug:
            print(f"  → numeric: {formula}")

        return formula

    @classmethod
    def _render_log_transform(
            cls,
            model_type: str,
            dependent_var: str,
            independent_vars: List[str],
            coefficients: Optional[Dict[str, float]],
            intercept: Optional[float],
            include_coeffs: bool,
            notation: str,
            sci_threshold: float,
            sci_max: float,
    ) -> str:
        """
        Render a log-linearised multivariate formula.

        Functional forms:

        - ``power-log-multivariate``:
          :math:`y = a\\,x_1^{b_1} x_2^{b_2}`, fitted as
          :math:`\\ln y = \\ln a + b_1 \\ln x_1 + b_2 \\ln x_2`
        - ``exponential-log-multivariate``:
          :math:`y = a\\,e^{b_1 x_1 + b_2 x_2}`, fitted as
          :math:`\\ln y = \\ln a + b_1 x_1 + b_2 x_2`

        ``intercept`` is expected to hold the **back-transformed** constant
        :math:`a`, matching the storage contract established by
        :meth:`~ModelDescriptor.from_fitted_model` and
        :meth:`~ModelDescriptor._update_from_log_transform_model`.

        Fixes applied
        -------------
        - ``include_coeffs`` is now honoured.  Previously it was accepted and
          discarded, so these models always printed numbers while every other
          family printed β symbols when asked for symbolic output.
        - Operators are notation-appropriate.  ``'math'`` previously fell
          through to the Python branch, producing ``0.189*DWT**0.000`` instead
          of ``1.208·DWT^0.310``.
        - Exponents resolve through :meth:`resolve_log_exponents`, which
          tolerates both key conventions and never substitutes a fabricated
          ``0.0``.  An unresolvable exponent renders as its β symbol.
        - Values route through :meth:`format_coefficient`, so ``sci_threshold``
          and ``sci_max`` now apply here as they do elsewhere.
        - The no-coefficients branch respects ``notation``; it previously
          emitted LaTeX unconditionally.

        :param model_type: ``'power-log-multivariate'`` or
            ``'exponential-log-multivariate'``.
        :type model_type: str
        """
        beta_map = cls.build_beta_index_map(coefficients, intercept)
        exponents = cls.resolve_log_exponents(coefficients, independent_vars)

        # Notation-dependent operators
        if notation == 'latex':
            mul, pow_open, pow_close = r" \cdot ", "^{", "}"
            exp_open, exp_close = r"\exp\left(", r"\right)"
        elif notation == 'python':
            mul, pow_open, pow_close = " * ", "**", ""
            exp_open, exp_close = "exp(", ")"
        else:  # math
            mul, pow_open, pow_close = "·", "^", ""
            exp_open, exp_close = "exp(", ")"

        # Leading multiplicative constant a
        a_str = cls._term_prefix(
            intercept, beta_map.get('intercept', 0),
            include_coeffs, notation, sci_threshold, sci_max,
            leading=True,
        )

        # β index offset: exponents follow the intercept when one is present
        offset = 1 if 'intercept' in beta_map else 0

        def _exponent(position: int, key: Optional[str],
                      value: Optional[float]) -> str:
            """Unsigned exponent string — numeric when resolved, else β."""
            idx = beta_map.get(key, position + offset) if key else position + offset
            return cls._term_prefix(
                value, idx, include_coeffs, notation, sci_threshold, sci_max,
                leading=True,
            )

        # ── Power form: y = a · x₁^b₁ · x₂^b₂ ─────────────────────────────
        if model_type == 'power-log-multivariate':
            parts = [a_str]
            for i, (var, key, value) in enumerate(exponents):
                parts.append(f"{var}{pow_open}{_exponent(i, key, value)}{pow_close}")
            return f"{dependent_var} = " + mul.join(parts)

        # ── Exponential form: y = a · exp(b₁·x₁ + b₂·x₂) ──────────────────
        exp_terms: List[str] = []
        for i, (var, key, value) in enumerate(exponents):
            idx = beta_map.get(key, i + offset) if key else i + offset
            prefix = cls._term_prefix(
                value, idx, include_coeffs, notation, sci_threshold, sci_max,
            )
            exp_terms.append(f"{prefix}{mul}{var}")

        content = " ".join(exp_terms).lstrip('+').strip()

        return (f"{dependent_var} = {a_str}{mul}"
                f"{exp_open}{content}{exp_close}")

    # =========================================================================
    # Public Formatting Utilities
    # =========================================================================

    @classmethod
    def format_power(cls, var: str, power: float, notation: str) -> str:
        """
        Format a fractional power transformation for display.

        Applies special-case rules before falling back to
        :meth:`simplify_power` for general fraction detection.

        Special cases (applied in order):

        - 0   → ``ln(var)``
        - 0.5 → ``√var`` / ``\\sqrt{var}``
        - −0.5 → ``1/√var``
        - −1  → ``1/var``
        - 1   → ``var``
        - integer *n* → ``var^n``
        - fraction (e.g. 1/3) → ``var^(1/3)``
        - decimal → ``var^0.789``

        :param var: Variable name placeholder.
        :type var: str
        :param power: Transformation power.
        :type power: float
        :param notation: ``'math'``, ``'latex'``, or ``'python'``.
        :type notation: str

        :returns: Formatted string with ``var`` substituted.
        :rtype: str
        """
        if power == 0:
            return (f"\\ln({var})" if notation == 'latex'
                    else f"log({var})" if notation == 'python'
                    else f"ln({var})")
        if abs(power - 0.5) < 1e-9:
            return (f"\\sqrt{{{var}}}" if notation == 'latex'
                    else f"sqrt({var})" if notation == 'python'
                    else f"√{var}")
        if abs(power + 0.5) < 1e-9:
            return (f"\\frac{{1}}{{\\sqrt{{{var}}}}}" if notation == 'latex'
                    else f"(1/sqrt({var}))" if notation == 'python'
                    else f"1/√{var}")
        if abs(power + 1) < 1e-9:
            return (f"\\frac{{1}}{{{var}}}" if notation == 'latex'
                    else f"(1/{var})" if notation == 'python'
                    else f"1/{var}")
        if abs(power - 1) < 1e-9:
            return var

        power_str = cls.simplify_power(power)

        if power == int(power):
            n = int(power)
            return (f"{var}^{{{n}}}" if notation == 'latex'
                    else f"({var}**{n})" if notation == 'python'
                    else f"{var}^{n}")

        if '/' in power_str:
            num, den = power_str.split('/')
            if notation == 'latex':
                if num.startswith('-'):
                    return f"{var}^{{-\\frac{{{num[1:]}}}{{{den}}}}}"
                return f"{var}^{{\\frac{{{num}}}{{{den}}}}}"
            elif notation == 'python':
                return f"({var}**({power_str}))"
            else:
                return f"{var}^({power_str})"

        return (f"{var}^{{{power_str}}}" if notation == 'latex'
                else f"({var}**{power_str})" if notation == 'python'
                else f"{var}^{power_str}")

    @classmethod
    def format_power_name(cls, var_name: str, power: float) -> str:
        """
        Return the coefficient-dict key for a power transformation.

        This key must match exactly what is stored in
        :attr:`~ModelDescriptor.coefficients` during
        :meth:`~ModelDescriptor._update_from_fitted_mfp` and
        :meth:`~ModelDescriptor.from_fitted_model`.

        :param var_name: Variable name, e.g. ``'DWT'``.
        :type var_name: str
        :param power: Transformation power.
        :type power: float

        :returns: Coefficient key, e.g. ``'√DWT'``, ``'1/GT'``,
            ``'DWT^(1/3)'``.
        :rtype: str
        """
        if abs(power) < 1e-9:           return f'ln({var_name})'
        if abs(power - 0.5) < 1e-9:     return f'√{var_name}'
        if abs(power + 0.5) < 1e-9:     return f'1/√{var_name}'
        if abs(power + 1) < 1e-9:       return f'1/{var_name}'
        if abs(power - 1) < 1e-9:       return var_name
        if abs(power - int(power)) < 1e-9:
            return f'{var_name}^{int(power)}'
        p_str = cls.simplify_power(power)
        return f'{var_name}^({p_str})' if '/' in p_str else f'{var_name}^{p_str}'

    @classmethod
    def format_coefficient(
        cls,
        coef_val: float,
        notation: str = 'math',
        sci_threshold: float = 1e-3,
        sci_max: float = 1e4,
    ) -> str:
        """
        Format a coefficient value with sign prefix and optional scientific
        notation.

        The returned string always begins with ``'+'`` or ``'-'`` so that
        it can be concatenated directly into a formula string.  Callers
        strip the leading ``'+'`` for the first term (intercept) via
        ``.lstrip('+')``.

        :param coef_val: Coefficient value.
        :type coef_val: float
        :param notation: ``'math'``, ``'latex'``, or ``'python'``.
        :type notation: str
        :param sci_threshold: Lower magnitude bound for regular notation.
        :type sci_threshold: float
        :param sci_max: Upper magnitude bound for regular notation.
        :type sci_max: float

        :returns: Formatted string, e.g. ``'+1.23 \\times 10^{-4}'``,
            ``'-0.523'``.
        :rtype: str
        """
        abs_val = abs(coef_val)
        sign = '+' if coef_val >= 0 else '-'

        if abs_val == 0:
            return '+0.000'

        if abs_val < sci_threshold or abs_val > sci_max:
            exp      = int(np.floor(np.log10(abs_val)))
            mantissa = abs_val / 10 ** exp
            m_str    = (
                f'{mantissa:.1f}'
                if abs(mantissa - round(mantissa, 1)) < 1e-10
                else f'{mantissa:.2f}'
            )
            if notation == 'latex':
                return f'{sign}{m_str} \\times 10^{{{exp}}}'
            elif notation == 'python':
                return f'{sign}{m_str}e{exp}'
            else:
                return f'{sign}{m_str}×10^{exp}'

        return f'{sign}{abs_val:.3f}'

    @classmethod
    def simplify_power(cls, power: float, tolerance: float = 1e-9) -> str:
        """
        Express a floating-point power as a simplified fraction when
        possible.

        Searches denominators 2–12 for a match within ``tolerance``.
        This prevents formulas like ``x^0.33333333`` when ``x^(1/3)``
        is cleaner.

        :param power: Power value to simplify.
        :type power: float
        :param tolerance: Matching tolerance for fraction detection.
        :type tolerance: float

        :returns: Fraction string (e.g. ``'1/3'``) or decimal string
            (e.g. ``'0.789'``).
        :rtype: str
        """
        for denom in range(2, 13):
            for numer in range(-12 * denom, 12 * denom + 1):
                if numer == 0 and denom != 1:
                    continue
                if abs(power - numer / denom) < tolerance:
                    frac = Fraction(numer, denom)
                    if frac in (Fraction(1, 2), Fraction(-1, 2)):
                        return f'{frac.numerator}/{frac.denominator}'
                    return (
                        str(frac.numerator)
                        if frac.denominator == 1
                        else f'{frac.numerator}/{frac.denominator}'
                    )

        if abs(power) < 0.01:  return f'{power:.4f}'
        elif abs(power) < 1:   return f'{power:.3f}'
        else:                  return f'{power:.2f}'

    @classmethod
    def get_fp2_second_term_name(cls, var_name: str, power: float) -> str:
        """
        Return the coefficient-dict key for the second term of a
        repeated-power FP2 transformation.

        The second term is :math:`x^p \\ln x` for :math:`p \\neq 0`, or
        :math:`(\\ln x)^2` for :math:`p = 0`.

        :param var_name: Variable name.
        :type var_name: str
        :param power: Repeated power value.
        :type power: float

        :returns: Key matching :attr:`~ModelDescriptor.coefficients`.
        :rtype: str
        """
        if power == 0:
            return f"(ln({var_name}))²"
        return f"{cls.format_power_name(var_name, power)}·ln({var_name})"

    @classmethod
    def _get_transform_name(
        cls, var_name: str, transform_type: str, powers: List[float]
    ) -> str:
        """
        Return the coefficient-dict key for the first term of a
        transformation, used as a lookup fallback in MFP rendering.

        :param var_name: Variable name.
        :type var_name: str
        :param transform_type: ``'linear'``, ``'fp1'``, or ``'fp2'``.
        :type transform_type: str
        :param powers: Power list.
        :type powers: list[float]

        :returns: Key string matching :attr:`~ModelDescriptor.coefficients`.
        :rtype: str
        """
        if transform_type == 'linear':
            return var_name
        return cls.format_power_name(var_name, powers[0])