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
            if include_coeffs and val is not None:
                return cls.format_coefficient(val, notation, sci_threshold, sci_max)
            sym = 'beta' if notation == 'python' else 'β'
            sep = '_' if notation == 'python' else ''
            return f"+{sym}{sep}{idx}"

        terms = []
        if include_coeffs and intercept is not None:
            terms.append(
                cls.format_coefficient(intercept, notation, sci_threshold, sci_max)
                .lstrip('+')
            )
        else:
            terms.append('beta_0' if notation == 'python' else 'β₀')

        if not transformations:
            return f"{dependent_var} = {terms[0]}"

        coef_idx = 1
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

        For Lasso, terms absent from ``active_features`` or with
        effectively zero coefficient are suppressed when
        ``only_active=True``.

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

        terms = []
        if intercept is not None:
            if include_coeffs:
                terms.append(
                    cls.format_coefficient(intercept, notation, sci_threshold, sci_max)
                    .lstrip('+')
                )
            else:
                terms.append('beta_0' if notation == 'python' else 'β₀')

        coef_idx = 1
        for feature_name, coef_val in coefficients.items():
            if feature_name in ('intercept', '1'):
                continue

            if only_active and regularization_type == 'lasso':
                if active_features and feature_name not in active_features:
                    continue
                if abs(coef_val) < 1e-10:
                    continue

            if include_coeffs:
                c_str = cls.format_coefficient(coef_val, notation, sci_threshold, sci_max)
            else:
                sym = 'beta' if notation == 'python' else 'β'
                sep = '_' if notation == 'python' else ''
                c_str = f"+{sym}{sep}{coef_idx}"

            if notation == 'latex':
                f_str = feature_name.replace('*', ' ')
                terms.append(f"{c_str} \\cdot {f_str}")
            elif notation == 'python':
                terms.append(f"{c_str} * {feature_name}")
            else:
                f_str = feature_name.replace('*', '·').replace(' ', '')
                terms.append(f"{c_str}·{f_str}")

            coef_idx += 1

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

        :param regularization_type: ``'lasso'``, ``'ridge'``, or ``None``.
        :type regularization_type: str, optional
        :param only_active: Suppress zero Lasso terms.
        :type only_active: bool
        """
        if not coefficients:
            return f"{dependent_var} = linear({', '.join(independent_vars)})"

        terms = []
        if intercept is not None:
            if include_coeffs:
                terms.append(
                    cls.format_coefficient(intercept, notation, sci_threshold, sci_max)
                    .lstrip('+')
                )
            else:
                terms.append('beta_0' if notation == 'python' else 'β₀')

        coef_idx = 1
        for var_name in independent_vars:
            coef_val = coefficients.get(var_name)
            if coef_val is None:
                continue
            if only_active and regularization_type == 'lasso':
                if abs(coef_val) < 1e-10:
                    continue

            if include_coeffs:
                c_str = cls.format_coefficient(coef_val, notation, sci_threshold, sci_max)
            else:
                sym = 'beta' if notation == 'python' else 'β'
                sep = '_' if notation == 'python' else ''
                c_str = f"+{sym}{sep}{coef_idx}"

            mul = ' * ' if notation == 'python' else '·'
            terms.append(f"{c_str}{mul}{var_name}")
            coef_idx += 1

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
    ) -> str:
        """
        Render a univariate nonlinear formula.

        Coefficients are stored under keys ``'a'``, ``'b'``, ``'c'``.
        When absent, reconstruction is attempted from ``metadata['popt']``.

        :param model_type: Specific nonlinear model type.
        :type model_type: str
        :param metadata: May contain ``'popt'`` array for coefficient
            reconstruction.
        :type metadata: dict, optional
        """
        fallback = (
            f"{dependent_var} ~ "
            f"{model_type}({', '.join(independent_vars)})"
        )

        # Reconstruct coefficients from popt if absent
        coefficients = dict(coefficients or {})
        if not coefficients:
            popt = (metadata or {}).get('popt')
            if popt is None:
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
                return fallback

        if len(independent_vars) != 1:
            return fallback

        var = independent_vars[0]
        vals = list(coefficients.values())
        a = coefficients.get('a') or (vals[0] if len(vals) > 0 else None)
        b = coefficients.get('b') or (vals[1] if len(vals) > 1 else None)
        c = (
            coefficients.get('c')
            or coefficients.get('offset')
            or (vals[2] if len(vals) > 2 else None)
        )

        # β-symbol mode
        if not include_coeffs:
            templates = {
                'exponential': {
                    'math':   f"{dependent_var} = β₀·exp(β₁·{var})",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot \\exp(\\beta_1 \\cdot {var})",
                    'python': f"{dependent_var} = beta_0 * exp(beta_1 * {var})",
                },
                'exponential-offset': {
                    'math':   f"{dependent_var} = β₀·exp(β₁·{var}) + β₂",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot \\exp(\\beta_1 \\cdot {var}) + \\beta_2",
                    'python': f"{dependent_var} = beta_0 * exp(beta_1 * {var}) + beta_2",
                },
                'power': {
                    'math':   f"{dependent_var} = β₀·{var}^β₁",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot {var}^{{\\beta_1}}",
                    'python': f"{dependent_var} = beta_0 * {var}**beta_1",
                },
                'power-offset': {
                    'math':   f"{dependent_var} = β₀·{var}^β₁ + β₂",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot {var}^{{\\beta_1}} + \\beta_2",
                    'python': f"{dependent_var} = beta_0 * {var}**beta_1 + beta_2",
                },
                'logarithmic': {
                    'math':   f"{dependent_var} = β₀·ln({var}) + β₁",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot \\ln({var}) + \\beta_1",
                    'python': f"{dependent_var} = beta_0 * log({var}) + beta_1",
                },
                'logarithmic-power': {
                    'math':   f"{dependent_var} = β₀·(ln({var}))^β₁",
                    'latex':  f"{dependent_var} = \\beta_0 \\cdot (\\ln({var}))^{{\\beta_1}}",
                    'python': f"{dependent_var} = beta_0 * (log({var}))**beta_1",
                },
            }
            return templates.get(model_type, {}).get(notation, fallback)

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

        return formula.replace("+ -", "- ").replace("+-", "-")

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

        The intercept is stored back-transformed (``a = exp(log_intercept)``).

        :param model_type: ``'power-log-multivariate'`` or
            ``'exponential-log-multivariate'``.
        :type model_type: str
        """
        if not coefficients:
            if model_type == 'power-log-multivariate':
                terms = [f"{v}^{{b_{i}}}" for i, v in enumerate(independent_vars)]
                return f"{dependent_var} = a \\cdot " + " \\cdot ".join(terms)
            else:
                terms = [f"b_{i} \\cdot {v}" for i, v in enumerate(independent_vars)]
                return (f"{dependent_var} = a \\cdot \\exp\\left("
                        + " + ".join(terms) + "\\right)")

        a = intercept
        if model_type == 'power-log-multivariate':
            parts = [f"{a:.3f}"]
            for i, var in enumerate(independent_vars):
                b = coefficients.get(f'b_{i}_{var}', 0.0)
                parts.append(
                    f"{var}^{{{b:.3f}}}" if notation == 'latex'
                    else f"{var}**{b:.3f}"
                )
            mul = r" \cdot " if notation == 'latex' else " * "
            return f"{dependent_var} = " + mul.join(parts)
        else:
            exp_terms = []
            for i, var in enumerate(independent_vars):
                b = coefficients.get(f'b_{i}_{var}', 0.0)
                sign = "+" if b >= 0 else "-"
                exp_terms.append(f"{sign} {abs(b):.3f} \\cdot {var}")
            content = " ".join(exp_terms).lstrip("+ ")
            if notation == 'latex':
                return (f"{dependent_var} = {a:.3f} "
                        f"\\cdot \\exp\\left({content}\\right)")
            return f"{dependent_var} = {a:.3f} * exp({content})"

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