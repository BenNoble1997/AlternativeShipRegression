# literature_formula_latex_generator.py
"""
LaTeX Table Generator for Literature Formulae
===============================================
Creates publication-ready tables showing the mathematical form of each
literature formula with proper notation.

Table structure:
    Target | Predictor(s) | Formula | Type | Reference

The variable_name_map controls how raw column names (e.g. 'LOA (M)')
are rendered both in column headers and inside equations.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from knowledge_layer.display_functions.LaTeX_table_utils import LaTeXFormatter, LaTeXPreambleGenerator
from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula


# ============================================================================
# DEFAULT VARIABLE MAPS
# ============================================================================

# Used in column header cells: "Target" / "Predictor(s)" columns
DEFAULT_DISPLAY_MAP: Dict[str, str] = {
    'LOA (M)':      r'$L_{OA}$ (m)',
    'LBP (M)':      r'$L_{BP}$ (m)',
    'BEAM (M)':     r'Beam (m)',
    'BEAM MLD (M)': r'Beam (m)',
    'DRAFT (M)':    r'Draught (m)',
    'DEPTH (M)':    r'$D$ (m)',
    'DWT':          r'DWT (t)',
    'GT':           r'$GT$',
    'TEU':          r'TEU',
    'Np':           r'$N_p$',
    'Nv':           r'$N_v$',
    'MCR (KW)':     r'$P_{MCR}$ (kW)',
    'Vs':           r'$v_s$ (kn)',
    'Main_Power':   r'$P_{main}$ (kW)',
    'Disp':         r'$\nabla$ (m$^3$)',
}

# Used inside equation bodies: symbols without units
DEFAULT_NOTATION_MAP: Dict[str, str] = {
    'LOA (M)':      r'L_{OA}',
    'LBP (M)':      r'L_{BP}',
    'BEAM (M)':     r'B',
    'BEAM MLD (M)': r'B',
    'DRAFT (M)':    r'T',
    'DEPTH (M)':    r'D',
    'DWT':          r'\mathrm{DWT}',
    'GT':           r'\mathrm{GT}',
    'TEU':          r'\mathrm{TEU}',
    'Np':           r'N_p',
    'Nv':           r'N_v',
    'MCR (KW)':     r'P_{MCR}',
    'Vs':           r'v_s',
    'Main_Power':   r'P_{main}',
    'Disp':         r'\nabla',
}


# ============================================================================
# FORMULA LATEX RENDERER
# ============================================================================

class LiteratureFormulaRenderer:
    """
    Converts a LiteratureFormula object into a LaTeX math string.

    Separated from the table generator so the rendering logic can be
    tested independently and extended for new formula types.

    Concept
    -------
    Each formula type has a known algebraic structure. We reconstruct
    that structure using the stored parameters and then substitute
    variable names with LaTeX symbols via the notation_map.

    For example, a 'power' formula with parameters a=7.6, b=0.0, c=0.30
    and predictor 'DWT' targeting 'LBP (M)' becomes:

        L_{BP} = 7.6 \cdot \mathrm{DWT}^{0.30}

    Piecewise formulas use the LaTeX `cases` environment.
    """

    def __init__(self, notation_map: Dict[str, str]):
        self.notation_map = notation_map

    def sym(self, var: str) -> str:
        """Translate a raw variable name to its LaTeX symbol."""
        return self.notation_map.get(var, r'\text{' + var.replace('_', r'\_') + '}')

    @staticmethod
    def _fmt(value: float, precision: int = 4) -> str:
        """
        Format a numeric coefficient for LaTeX.

        Uses scientific notation for very large or very small values
        to keep equations readable.
        """
        if value == 0:
            return '0'
        abs_v = abs(value)
        if abs_v < 1e-3 or abs_v >= 1e5:
            exp = int(np.floor(np.log10(abs_v)))
            mantissa = value / 10 ** exp
            sign = '-' if value < 0 else ''
            return rf'{sign}{mantissa:.{precision-1}f} \times 10^{{{exp}}}'
        return f'{value:.{precision}f}'.rstrip('0').rstrip('.')

    def render(self, formula: LiteratureFormula, debug: bool = False) -> str:
        """
        Render formula as a LaTeX math string (without surrounding $...$).

        Parameters
        ----------
        formula : LiteratureFormula
        debug : bool
            Print intermediate rendering steps

        Returns
        -------
        str
            LaTeX math string, e.g. r'L_{BP} = 7.60 \cdot \mathrm{DWT}^{0.30}'
        """
        lhs = self.sym(formula.target)

        if debug:
            print(f"  [render] formula={formula.name}, type={formula.formula_type}")
            print(f"  [render] target_sym={lhs}, predictors={formula.predictors}")

        ft = formula.formula_type

        if ft.startswith('piecewise'):
            rhs = self._render_piecewise(formula, debug)
        elif len(formula.predictors) == 1:
            rhs = self._render_single(formula, debug)
        else:
            rhs = self._render_multi(formula, debug)

        if rhs is None:
            rhs = r'\text{(unknown type: ' + ft + ')}'

        result = f'{lhs} = {rhs}'
        if debug:
            print(f"  [render] result={result}")
        return result

    # ── Single-predictor renderers ─────────────────────────────────────────

    def _render_single(self, formula: LiteratureFormula, debug: bool) -> Optional[str]:
        """Render single-predictor formula types."""
        p = formula.parameters
        x = self.sym(formula.predictors[0])
        ft = formula.formula_type

        if debug:
            print(f"  [_render_single] type={ft}, params={p}, x={x}")

        if ft == 'power':
            # y = a + b * x^c  (commonly b=0 so just a*x^c)
            a, b, c = p.get('a', 0), p.get('b', 0), p.get('c', 1)
            core = rf'{self._fmt(b)} \cdot {x}^{{{self._fmt(c)}}}'
            if a != 0:
                sign = '+' if a > 0 else '-'
                core = rf'{self._fmt(abs(a))} {sign} {core}'
            return core

        elif ft == 'power_min':
            a, b, c, d = p.get('a', 0), p.get('b', 0), p.get('c', 1), p.get('d', 0)
            inner = rf'{self._fmt(b)} \cdot {x}^{{{self._fmt(c)}}}'
            if a != 0:
                inner = rf'{self._fmt(a)} + {inner}'
            return rf'\min\left({inner},\; {self._fmt(d)}\right)'

        elif ft == 'power_log':
            a, b, c = p.get('a', 0), p.get('b', 0), p.get('c', 1)
            core = rf'{self._fmt(b)} \cdot \ln\!\left({x}^{{{self._fmt(c)}}}\right)'
            if a != 0:
                sign = '+' if a > 0 else '-'
                core = rf'{self._fmt(abs(a))} {sign} {core}'
            return core

        elif ft == 'log_power':
            a, b, c = p.get('a', 0), p.get('b', 0), p.get('c', 1)
            core = rf'{self._fmt(b)} \cdot \left(\ln {x}\right)^{{{self._fmt(c)}}}'
            if a != 0:
                sign = '+' if a > 0 else '-'
                core = rf'{self._fmt(abs(a))} {sign} {core}'
            return core

        elif ft == 'quadrilateral_polynomial':
            a, b, c = p.get('a', 0), p.get('b', 0), p.get('c', 0)
            terms = []
            if a != 0:
                terms.append(self._fmt(a))
            if b != 0:
                sign = '+' if b > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(b))} {x}')
            if c != 0:
                sign = '+' if c > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(c))} {x}^{{2}}')
            return ' '.join(terms) if terms else '0'

        elif ft == 'cubic_polynomial':
            a, b, c, d = p.get('a', 0), p.get('b', 0), p.get('c', 0), p.get('d', 0)
            terms = []
            if a != 0:
                terms.append(rf'{self._fmt(a)} {x}^{{3}}')
            if b != 0:
                sign = '+' if b > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(b))} {x}^{{2}}')
            if c != 0:
                sign = '+' if c > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(c))} {x}')
            if d != 0:
                sign = '+' if d > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(d))}')
            return ' '.join(terms) if terms else '0'

        elif ft == 'exponential':
            a, b, c = p.get('a', 0), p.get('b', 0), p.get('c', 0)
            core = rf'{self._fmt(b)} e^{{{self._fmt(c)} {x}}}'
            if a != 0:
                sign = '+' if a > 0 else '-'
                core = rf'{self._fmt(abs(a))} {sign} {core}'
            return core

        return None

    # ── Multi-predictor renderers ──────────────────────────────────────────

    def _render_multi(self, formula: LiteratureFormula, debug: bool) -> Optional[str]:
        """Render multi-predictor formula types."""
        p = formula.parameters
        ft = formula.formula_type
        preds = formula.predictors

        if debug:
            print(f"  [_render_multi] type={ft}, params={p}, preds={preds}")

        if ft == 'additive_power':
            terms = []
            for i, pred in enumerate(preds):
                if i not in p:
                    continue
                a, b = p[i].get('a', 0), p[i].get('b', 1)
                x = self.sym(pred)
                term = rf'{self._fmt(a)} \cdot {x}^{{{self._fmt(b)}}}'
                if terms:
                    sign = '+' if a >= 0 else '-'
                    term = rf'{sign} {self._fmt(abs(a))} \cdot {x}^{{{self._fmt(b)}}}'
                terms.append(term)
            if 'constant' in p and p['constant'] != 0:
                c = p['constant']
                sign = '+' if c > 0 else '-'
                terms.append(rf'{sign} {self._fmt(abs(c))}')
            return ' '.join(terms) if terms else '0'

        elif ft == 'multiplicative_power':
            a = p.get('a', 1.0)
            factors = [self._fmt(a)]
            for i, pred in enumerate(preds):
                if i not in p:
                    continue
                b = p[i].get('b', 1)
                x = self.sym(pred)
                factors.append(rf'{x}^{{{self._fmt(b)}}}')
            result = r' \cdot '.join(factors)
            if 'constant' in p and p['constant'] != 0:
                c = p['constant']
                sign = '+' if c > 0 else '-'
                result += rf' {sign} {self._fmt(abs(c))}'
            return result

        elif ft == 'multi_linear':
            terms = []
            if 'constant' in p:
                terms.append(self._fmt(p['constant']))
            if 'terms' in p:
                for i, term_str in enumerate(p['terms']):
                    if i not in p:
                        continue
                    coef = p[i].get('a', 0)
                    if coef == 0:
                        continue
                    sign = '+' if coef > 0 else '-'
                    rendered_term = self._render_term_string(term_str, preds)
                    terms.append(rf'{sign} {self._fmt(abs(coef))} \cdot {rendered_term}')
            return ' '.join(terms) if terms else '0'

        elif ft == 'power_product':
            return self._render_power_product(formula, p, preds, debug)

        return None

    def _render_term_string(self, term_str: str, predictors: List[str]) -> str:
        """
        Convert a term string like 'DWT*Vs^2' to LaTeX like
        r'\mathrm{DWT} \cdot v_s^{2}'.
        """
        factors = term_str.split('*')
        rendered = []
        for factor in factors:
            factor = factor.strip()
            if '^' in factor:
                var, power = factor.split('^')
                rendered.append(rf'{self.sym(var.strip())}^{{{power.strip()}}}')
            else:
                rendered.append(self.sym(factor))
        return r' \cdot '.join(rendered)

    def _render_power_product(self, formula: LiteratureFormula, p: dict,
                               preds: List[str], debug: bool) -> str:
        """
        Render the `power_product` formula type.

        Parameter index layout (mirrors `_build_multi_predictor_function`):
        - Index 0 … n-1          : individual predictor terms
                                    a = coefficient, b = power (may be 'ln<N>' string)
        - Index n … 2n-1         : interaction product factors
                                    b = power for each predictor
        - Index 2n               : interaction outer term
                                    a = outer coefficient, b = outer power
        - 'constant'             : additive constant

        Rendered form:
            y = a0·f(x0)^p0 + a1·f(x1)^p1 + ... + a_int·(f(x0)^q0 · f(x1)^q1 ...)^q_int + C
        """
        n = len(preds)
        terms = []

        # ── Part 1: individual predictor terms ────────────────────────────
        for i in range(n):
            if i not in p:
                continue
            a = p[i].get('a', 0.0)
            if a == 0:
                continue
            b_val = p[i].get('b', 1.0)
            x_sym = self.sym(preds[i])

            inner = self._render_power_factor(x_sym, b_val)
            term = rf'{self._fmt(a)} \cdot {inner}'

            if terms:
                sign = '+' if a >= 0 else '-'
                term = rf'{sign} {self._fmt(abs(a))} \cdot {inner}'
            terms.append(term)

            if debug:
                print(f"  [_render_power_product] term {i}: a={a}, b={b_val} → {term}")

        # ── Part 2: interaction product term ─────────────────────────────
        interaction_start = n
        interaction_end = interaction_start + n

        inner_factors = []
        for i in range(n):
            idx = interaction_start + i
            if idx not in p or 'b' not in p[idx]:
                continue
            b_val = p[idx]['b']
            x_sym = self.sym(preds[i])
            inner_factors.append(self._render_power_factor(x_sym, b_val))

        if inner_factors and interaction_end in p:
            a_out = p[interaction_end].get('a', 1.0)
            b_out = p[interaction_end].get('b', 1.0)
            product_str = r' \cdot '.join(inner_factors)
            outer = self._render_power_factor(rf'\left({product_str}\right)', b_out)
            sign = '+' if a_out >= 0 else '-'
            prefix = f'{sign} ' if terms else ''
            terms.append(rf'{prefix}{self._fmt(abs(a_out))} \cdot {outer}')

            if debug:
                print(f"  [_render_power_product] interaction: a={a_out}, b={b_out}")
                print(f"    inner_factors={inner_factors}")

        # ── Constant ──────────────────────────────────────────────────────
        if 'constant' in p and p['constant'] != 0:
            c = p['constant']
            sign = '+' if c > 0 else '-'
            prefix = f'{sign} ' if terms else ''
            terms.append(rf'{prefix}{self._fmt(abs(c))}')

        return ' '.join(terms) if terms else '0'

    def _render_power_factor(self, sym: str, b_val) -> str:
        """
        Render a single factor `f(sym)^power` for use in power_product terms.

        Handles both numeric powers and string notations like 'ln3'
        (meaning ln(sym)^3).
        """
        if isinstance(b_val, str):
            import re
            match = re.match(r'([a-zA-Z]+)(\d+(?:\.\d+)?)', b_val)
            if match:
                func, power_str = match.groups()
                power = float(power_str)
                func_latex = {'ln': r'\ln', 'log': r'\ln', 'log10': r'\log_{10}'}.get(
                    func.lower(), rf'\{func.lower()}'
                )
                if power == 1:
                    return rf'{func_latex}\!\left({sym}\right)'
                return rf'\left({func_latex}\!\left({sym}\right)\right)^{{{self._fmt(power)}}}'
            try:
                b_val = float(b_val)
            except ValueError:
                return rf'{sym}^{{\text{{{b_val}}}}}'

        power = float(b_val)
        if power == 1:
            return sym
        return rf'{sym}^{{{self._fmt(power)}}}'

    # ── Piecewise renderer ─────────────────────────────────────────────────

    def _render_piecewise(self, formula: LiteratureFormula, debug: bool) -> str:
        """
        Render a piecewise formula using the LaTeX `cases` environment.

        Output structure:
            \begin{cases}
                <formula_1> & <lower_1> \leq x < <upper_1> \\
                <formula_2> & <lower_2> \leq x \leq <upper_2>
            \end{cases}
        """
        if formula.bins is None or formula.bin_parameters is None:
            return r'\text{(piecewise: bins not defined)}'

        x_sym = self.sym(formula.predictors[0])
        cases = []

        for i, params in enumerate(formula.bin_parameters):
            lo = formula.bins[i]
            hi = formula.bins[i + 1]

            a = params.get('a', 0)
            b = params.get('b', 0)
            c = params.get('c', 1)
            d = params.get('d', 0)
            e = params.get('e', 1)

            if debug:
                print(f"  [_render_piecewise] segment {i}: lo={lo}, hi={hi}, params={params}")

            # Reconstruct: a + b*x^c + d*x^e
            seg_terms = []
            if a != 0:
                seg_terms.append(self._fmt(a))
            if b != 0:
                sign = '+' if b > 0 and seg_terms else ''
                prefix = f'{sign} ' if seg_terms else ''
                if c == 1:
                    seg_terms.append(rf'{prefix}{self._fmt(b)} \cdot {x_sym}')
                else:
                    seg_terms.append(rf'{prefix}{self._fmt(b)} \cdot {x_sym}^{{{self._fmt(c)}}}')
            if d != 0:
                sign = '+' if d > 0 else '-'
                prefix = f'{sign} ' if seg_terms else ''
                if e == 1:
                    seg_terms.append(rf'{prefix}{self._fmt(abs(d))} \cdot {x_sym}')
                else:
                    seg_terms.append(rf'{prefix}{self._fmt(abs(d))} \cdot {x_sym}^{{{self._fmt(e)}}}')

            seg_str = ' '.join(seg_terms) if seg_terms else '0'

            lo_fmt = f'{lo:,.0f}'.replace(',', r'\,')
            hi_fmt = f'{hi:,.0f}'.replace(',', r'\,')
            is_last = (i == len(formula.bin_parameters) - 1)
            leq_hi = r'\leq' if is_last else '<'
            cond = rf'{lo_fmt} \leq {x_sym} {leq_hi} {hi_fmt}'

            cases.append(rf'    {seg_str} & {cond}')

        inner = r' \\' + '\n'
        return r'\begin{cases}' + '\n' + inner.join(cases) + '\n' + r'\end{cases}'


# ============================================================================
# MAIN TABLE GENERATOR
# ============================================================================

class LiteratureFormulaLatexGenerator:
    """
    Generate LaTeX tables listing the mathematical form of each literature formula.

    **Distinct from** `MultiDatasetLiteratureLatexGenerator` (which shows
    performance metrics like R² and RRMSE). This generator shows *what the
    formula is*, not how well it performs.

    Table structure (one row per formula):
        Target | Predictor(s) | Formula | [Type] | [R²] | [RRMSE] | [Reference]

    Data flow
    ---------
    1. MultiDatasetLiteratureEvaluator
           └── .evaluators[fuel_type].formulae          (list[LiteratureFormula])
    2. LiteratureFormulaLatexGenerator.__init__
           └── receives evaluator, extracts .formulae
    3. generate_formula_table()
           └── iterates formulae
           └── LiteratureFormulaRenderer.render(formula) → LaTeX math string
           └── _build_table_structure()                 → complete .tex string
    4. save_formula_table()
           └── writes .tex + preamble to disk
    """

    def __init__(self,
                 evaluator,
                 display_map: Optional[Dict[str, str]] = None,
                 notation_map: Optional[Dict[str, str]] = None,
                 verbose: bool = True,
                 r2_min: float | None = 0.0,
                 r2_max: float | None = 1.0,
                 rrmse_max: float | None = 100.0):


        # Store instance-level thresholds
        self._r2_min    = r2_min
        self._r2_max    = r2_max
        self._rrmse_max = rrmse_max

        self.verbose = verbose

        # ── Extract formulae ───────────────────────────────────────────────
        if hasattr(evaluator, 'evaluators') and evaluator.evaluators:
            first_key = next(iter(evaluator.evaluators))
            self.formulae: List[LiteratureFormula] = evaluator.evaluators[first_key].formulae
            self._comparison_results = getattr(evaluator, 'comparison_results', None)
        elif hasattr(evaluator, 'formulae'):
            self.formulae = evaluator.formulae
            self._comparison_results = getattr(evaluator, 'results', None)
        else:
            raise ValueError(
                "evaluator must have a '.formulae' attribute or a '.evaluators' dict."
            )

        self.shiptype = getattr(evaluator, 'shiptype', None)

        # ── Build merged maps ──────────────────────────────────────────────
        merged_display = dict(DEFAULT_NOTATION_MAP)
        if display_map:
            merged_display.update(display_map)

        merged_notation = dict(DEFAULT_NOTATION_MAP)
        if notation_map:
            merged_notation.update(notation_map)

        self.display_map = merged_display
        self.notation_map = merged_notation

        # ── Composition: formatter and renderer ───────────────────────────
        self.formatter = LaTeXFormatter()
        self.renderer = LiteratureFormulaRenderer(notation_map=self.notation_map)

        # ── NEW: metric validation ─────────────────────────────────────────────

    def _metrics_are_valid(self,
                           r2: Optional[float],
                           rrmse: Optional[float]) -> bool:
        """
        Return True if the conventional-dataset metrics are within acceptable
        bounds, or if no metrics are available (None → pass through).

        The None case is important: if evaluate_all_datasets() has not been
        run yet, there are no metrics to check and we should still render the
        formula definitions.

        Parameters
        ----------
        r2 : float or None
            Conventional R² for this formula.
        rrmse : float or None
            Conventional RRMSE (%) for this formula.

        Returns
        -------
        bool
        """
        if r2 is not None and not (self._r2_min <= r2 <= self._r2_max):
            return False
        if rrmse is not None and rrmse > self._rrmse_max:
            return False
        return True

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_formula_table(self,
                               targets: Optional[List[str]] = None,
                               references: Optional[List[str]] = None,
                               caption: Optional[str] = None,
                               label: Optional[str] = None,
                               landscape: bool = False,
                               show_reference: bool = True,
                               show_formula_type: bool = True,
                               show_metrics: bool = True,
                               debug: bool = False) -> str:
        """
        Generate complete LaTeX table string.

        Parameters
        ----------
        targets : list[str], optional
            Filter to only these target variables. If None, all targets included.
        references : list[str], optional
            Filter to only these references/sources.
        caption : str, optional
            Custom table caption.
        label : str, optional
            Custom \\label{} identifier.
        landscape : bool
            Wrap in `sidewaystable` environment (for wide tables).
        show_reference : bool
            Include Reference column.
        show_formula_type : bool
            Include Formula Type column.
        show_metrics : bool
            Include conventional R² and RRMSE columns (requires evaluation results).
        debug : bool
            Print rendering details for each formula.

        Returns
        -------
        str
            Complete LaTeX table as a string.
        """
        rows = self._prepare_rows(targets, references, debug)

        if not rows:
            if self.verbose:
                print("⚠️  No formulae matched the filters — empty table.")
            return ''

        if caption is None:
            caption = self._default_caption(len(rows), targets)
        if label is None:
            ship_str = self.shiptype.name if self.shiptype else 'unknown'
            label = f'tab:{ship_str}_literature_formulae'

        return self._build_table_structure(
            rows=rows,
            caption=caption,
            label=label,
            landscape=landscape,
            show_reference=show_reference,
            show_formula_type=show_formula_type,
            show_metrics=show_metrics,
        )

    def save_formula_table(self,
                           output_dir: str = '.',
                           prefix: Optional[str] = None,
                           targets: Optional[List[str]] = None,
                           references: Optional[List[str]] = None,
                           landscape: bool = False,
                           show_reference: bool = True,
                           show_formula_type: bool = True,
                           show_metrics: bool = True,
                           debug: bool = False) -> Dict[str, Path]:
        """
        Generate and save the formula table to disk.

        Saves two files:
        - ``<prefix>_formulae.tex``     — the table itself
        - ``<prefix>_preamble.tex``     — required LaTeX package imports

        Returns
        -------
        dict
            ``{'table': Path, 'preamble': Path}``
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            ship_str = self.shiptype.name if self.shiptype else 'literature'
            prefix = f'{ship_str}_literature'

        latex = self.generate_formula_table(
            targets=targets,
            references=references,
            landscape=landscape,
            show_reference=show_reference,
            show_formula_type=show_formula_type,
            show_metrics=show_metrics,
            debug=debug,
        )

        if not latex:
            if self.verbose:
                print("⚠️  No table generated.")
            return {}

        table_path = out / f'{prefix}_formulae.tex'
        preamble_path = out / f'{prefix}_formulae_preamble.tex'

        table_path.write_text(latex, encoding='utf-8')
        LaTeXPreambleGenerator.save_preamble(str(preamble_path))

        if self.verbose:
            print(f"✅ Literature formula table saved:")
            print(f"   Table:    {table_path}")
            print(f"   Preamble: {preamble_path}")

        return {'table': table_path, 'preamble': preamble_path}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _metrics_are_valid(self,
                           r2: Optional[float],
                           rrmse: Optional[float]) -> bool:
        """
        Return True if the conventional-dataset metrics are within acceptable
        bounds, or if no metrics are available (None → pass through).

        The None case is important: if evaluate_all_datasets() has not been
        run yet, there are no metrics to check and we should still render the
        formula definitions.

        Parameters
        ----------
        r2 : float or None
            Conventional R² for this formula.
        rrmse : float or None
            Conventional RRMSE (%) for this formula.

        Returns
        -------
        bool
        """
        if r2 is not None and not (self._r2_min <= r2 <= self._r2_max):
            return False
        if rrmse is not None and rrmse > self._rrmse_max:
            return False
        return True

    def _prepare_rows(self,
                      targets:    Optional[List[str]],
                      references: Optional[List[str]],
                      debug:      bool) -> List[Dict]:
        """
        Build a list of row dicts from self.formulae.

        Steps
        -----
        1. Build metric lookup dict from comparison_results for O(1) access.
        2. Iterate formulae, apply target/reference filters.
        3. Call LiteratureFormulaRenderer.render() for each formula.
        4. Look up conventional R² and RRMSE (averaged across fuel types if
           multiple fuel types are present in comparison_results).
        5. Filter rows whose metrics fall outside the configured thresholds.
        6. Return list sorted by target then number of predictors.
        """
        # ── Build metric lookup ───────────────────────────────────────────
        metric_lookup: Dict[str, Dict[str, Optional[float]]] = {}

        if self._comparison_results is not None and len(self._comparison_results) > 0:
            cr = self._comparison_results
            if 'formula_name' in cr.columns:
                r2_col    = 'conventional_r2'    if 'conventional_r2'    in cr.columns else None
                rrmse_col = 'conventional_rrmse' if 'conventional_rrmse' in cr.columns else None
                for fname, grp in cr.groupby('formula_name'):
                    metric_lookup[fname] = {
                        'r2':    float(grp[r2_col].mean())    if r2_col    else None,
                        'rrmse': float(grp[rrmse_col].mean()) if rrmse_col else None,
                    }

        if debug and metric_lookup:
            print(f"  [_prepare_rows] Metric lookup built for {len(metric_lookup)} formulae")

        # ── Build rows ────────────────────────────────────────────────────
        rows     = []
        n_skipped = 0

        for formula in self.formulae:
            if targets is not None and formula.target not in targets:
                continue
            if references is not None and formula.reference not in references:
                continue

            if debug:
                print(f"\n[_prepare_rows] Processing: {formula.name}")

            try:
                formula_latex = self.renderer.render(formula, debug=debug)
            except Exception as exc:
                if self.verbose:
                    print(f"  ⚠️  Render error for {formula.name}: {exc}")
                formula_latex = r'\text{(render error)}'

            target_display = self.display_map.get(formula.target, formula.target)
            preds_display  = ', '.join(
                self.display_map.get(p, p) for p in formula.predictors
            )
            metrics = metric_lookup.get(formula.name, {'r2': None, 'rrmse': None})

            # ── Quality gate ──────────────────────────────────────────────
            if not self._metrics_are_valid(metrics['r2'], metrics['rrmse']):
                n_skipped += 1
                if self.verbose:
                    print(
                        f"  ⚠️  Skipping '{formula.name}': "
                        f"R²={metrics['r2']}, RRMSE={metrics['rrmse']}% "
                        f"— outside thresholds "
                        f"(R² ∈ [{self._r2_min}, {self._r2_max}], "
                        f"RRMSE ≤ {self._rrmse_max}%)"
                    )
                continue

            rows.append({
                'target':             formula.target,
                'target_display':     target_display,
                'predictors':         formula.predictors,
                'predictors_display': preds_display,
                'formula_latex':      formula_latex,
                'formula_type':       formula.formula_type,
                'reference':          formula.reference or '---',
                'n_predictors':       len(formula.predictors),
                'conventional_r2':    metrics['r2'],
                'conventional_rrmse': metrics['rrmse'],
            })

            if debug:
                print(f"  conventional_r2={metrics['r2']}, conventional_rrmse={metrics['rrmse']}")

        rows.sort(key=lambda r: (r['target'], r['n_predictors']))

        if self.verbose:
            msg = f"  Prepared {len(rows)} formula rows for table"
            if n_skipped:
                msg += f" ({n_skipped} skipped — outside quality thresholds)"
            print(msg)

        return rows

    def _build_table_structure(self,
                               rows: List[Dict],
                               caption: str,
                               label: str,
                               landscape: bool,
                               show_reference: bool,
                               show_formula_type: bool,
                               show_metrics: bool) -> str:
        """
        Assemble the complete LaTeX table string.

        Column layout (left to right):
            Target | Predictor(s) | Formula | [Type] | [R²] | [RRMSE] | [Reference]
        """
        lines = []

        env = 'sidewaystable' if landscape else 'table*'
        lines.append(rf'\begin{{{env}}}[ht]')
        lines.append(r'\centering')
        lines.append(r'\footnotesize')
        lines.append(rf'\caption{{{self.formatter.escape_latex(caption)}}}')
        lines.append(rf'\label{{{label}}}')

        # ── Column spec ────────────────────────────────────────────────────
        col_fracs = {
            'target':    '0.07',
            'predictor': '0.12',
            'formula':   '0.40',
            'type':      '0.10',
            'r2':        '0.075',
            'rrmse':     '0.075',
            'reference': '0.20',
        }

        def _pcol(frac: str) -> str:
            return rf'p{{{frac}\linewidth}}'

        def _rcol(frac: str) -> str:
            """Raggedright p-column — prevents hyphenation stretch in narrow cells."""
            return rf'>{{\\raggedright\\arraybackslash}}p{{{frac}\linewidth}}'

        colspec = (
            _pcol(col_fracs['target'])
            + _pcol(col_fracs['predictor'])
            + _pcol(col_fracs['formula'])
        )
        if show_formula_type:
            colspec += _pcol(col_fracs['type'])
        if show_metrics:
            colspec += _rcol(col_fracs['r2']) + _rcol(col_fracs['rrmse'])
        if show_reference:
            colspec += _pcol(col_fracs['reference'])

        lines.append(rf'\begin{{tabular}}{{{colspec}}}')
        lines.append(r'\toprule')

        # ── Header row ─────────────────────────────────────────────────────
        hdr = [r'\textbf{Target}', r'\textbf{Predictor(s)}', r'\textbf{Formula}']
        if show_formula_type:
            hdr.append(r'\textbf{Type}')
        if show_metrics:
            hdr.append(r'$R_{\mathrm{adj}}^{2}$')
            hdr.append(r'RRMSE (\%)')
        if show_reference:
            hdr.append(r'\textbf{Reference}')
        lines.append(' & '.join(hdr) + r' \\')
        lines.append(r'\midrule')

        # ── Data rows ──────────────────────────────────────────────────────
        prev_target = None
        for row in rows:
            if prev_target is not None and row['target'] != prev_target:
                lines.append(r'\midrule')
            prev_target = row['target']

            cell_target  = self.formatter.escape_latex(row['target_display'])
            cell_preds   = self.formatter.escape_latex(row['predictors_display'])

            # Piecewise formulas need \displaystyle for readable cases environment
            if r'\begin{cases}' in row['formula_latex']:
                cell_formula = r'$\displaystyle ' + row['formula_latex'] + r'$'
            else:
                cell_formula = r'$' + row['formula_latex'] + r'$'

            parts = [cell_target, cell_preds, cell_formula]

            if show_formula_type:
                parts.append(row['formula_type'].replace('_', r'\_'))

            if show_metrics:
                r2_val    = row.get('conventional_r2')
                rrmse_val = row.get('conventional_rrmse')
                parts.append(f'{r2_val:.3f}'    if r2_val    is not None else '---')
                parts.append(f'{rrmse_val:.1f}' if rrmse_val is not None else '---')

            if show_reference:
                parts.append(self.formatter.escape_latex(row['reference']))

            lines.append(' & '.join(parts) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(rf'\end{{{env}}}')

        return '\n'.join(lines) + '\n'

    def _default_caption(self, n: int, targets: Optional[List[str]]) -> str:
        """Build a sensible default caption."""
        ship_str   = self.shiptype.string if self.shiptype else 'Ship'
        target_str = (
            f"for {', '.join(targets)}" if targets else 'across all target dimensions'
        )
        return (
            f"Literature formulae {target_str} for {ship_str} ($n={n}$ formulae). "
            r"Coefficients as published in the cited sources."
        )