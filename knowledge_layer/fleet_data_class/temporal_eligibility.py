"""
Directional admissibility constraints for matched-cohort construction.

Provides a predicate object that restricts which control ships an individual
treated ship may be matched to, based on an ordered anchor variable such as
build year. This implements risk-set matching: the admissible set is defined
relative to each treated unit's own anchor value rather than by a shared
partition of the anchor axis.
"""


import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalEligibility:
    """
    Per-unit admissibility rule on an ordered anchor variable.

    A candidate control with anchor value ``c`` is admissible for a treated
    unit with anchor value ``t`` when ``t - max_lag <= c <= t + max_lead``.
    Setting ``max_lead`` to zero yields backward-only (risk-set) matching, in
    which a treated unit may only be matched to controls that already existed
    at its own anchor value.

    This differs fundamentally from binning on the anchor. Binning imposes a
    single partition shared by both fleets, so admissibility is symmetric and
    the resulting relation is transitive; it is also discontinuous at the bin
    edges, so two units one unit apart may be inadmissible while two units
    nine apart are admissible. The rule here is anchored on each treated unit
    individually, so the admissible sets overlap rather than partition, no
    edge artefact exists, and the relation can be made asymmetric.

    Because the admissible sets are not disjoint, this rule cannot be
    expressed through the binning mechanism and must be evaluated inside the
    matching loop.

    :param anchor_column: Column holding the ordered variable, present in both
        fleets. Typically ``'BUILD YEAR'``.
    :type anchor_column: str
    :param max_lag: Maximum permitted amount by which a control may precede
        the treated unit, in the units of the anchor. ``None`` places no lower
        bound, which admits the entire prior record and maximises pool size at
        the cost of unbounded anchor imbalance.
    :type max_lag: float | None
    :param max_lead: Maximum permitted amount by which a control may follow
        the treated unit. Zero enforces the no-lookahead constraint. ``None``
        places no upper bound, which reduces the rule to a one-sided lookback
        window.
    :type max_lead: float | None
    :param allow_equal: Whether controls with an anchor value exactly equal to
        the treated unit's are admissible. Only meaningful when ``max_lead``
        is zero, where ``False`` gives strictly-prior matching. For build year
        the default of ``True`` is normally correct, since a ship built in the
        same year reflects contemporaneous rather than subsequent practice.
    :type allow_equal: bool
    :param on_missing_anchor: Behaviour when a treated unit has a missing
        anchor value. ``'skip'`` excludes it and counts it; ``'error'`` raises.
        Controls with a missing anchor are always inadmissible, since their
        position in the ordering cannot be established.
    :type on_missing_anchor: Literal['skip', 'error']
    :raises ValueError: If a bound is negative, or if ``allow_equal`` is
        ``False`` while ``max_lead`` is non-zero, which would punch a hole at
        the anchor value itself rather than enforcing strict priority.
    """

    anchor_column: str
    max_lag: float | None = None
    max_lead: float | None = 0.0
    allow_equal: bool = True
    on_missing_anchor: Literal['skip', 'error'] = 'skip'

    def __post_init__(self) -> None:
        """Validate the interval specification at construction time."""
        if self.max_lag is not None and self.max_lag < 0:
            raise ValueError(
                f'max_lag must be non-negative or None, got {self.max_lag}. '
                f'A negative lag would exclude the treated unit itself from '
                f'its own admissible interval.'
            )
        if self.max_lead is not None and self.max_lead < 0:
            raise ValueError(
                f'max_lead must be non-negative or None, got {self.max_lead}.'
            )
        if not self.allow_equal and self.max_lead != 0:
            raise ValueError(
                f'allow_equal=False is only defined for max_lead=0, where it '
                f'means strictly-prior matching. With max_lead={self.max_lead} '
                f'it would exclude exact ties while admitting later controls, '
                f'which is not a coherent rule.'
            )
        if self.on_missing_anchor not in ('skip', 'error'):
            raise ValueError(
                f"on_missing_anchor must be 'skip' or 'error', got "
                f"'{self.on_missing_anchor}'"
            )

    @classmethod
    def prior_only(cls,
                   anchor_column: str,
                   max_lag: float | None = None,
                   allow_equal: bool = True,
                   on_missing_anchor: Literal['skip', 'error'] = 'skip'
                   ) -> 'TemporalEligibility':
        """
        Construct a backward-only rule.

        A bounded ``max_lag`` is normally preferable to ``None``. Leaving the
        lookback unbounded lets the geometric distance reach arbitrarily far
        into the past whenever an old ship happens to be dimensionally close,
        which systematically ages the matched cohort relative to the treated
        cohort. Bounding it caps the anchor imbalance the rule can introduce
        and makes the realised lag a reportable quantity rather than an
        uncontrolled one.

        :param max_lag: Lookback window in anchor units, or ``None`` for the
            full prior record.
        :returns: Rule admitting only controls at or before the treated value.
        :rtype: TemporalEligibility
        """
        return cls(anchor_column=anchor_column,
                   max_lag=max_lag,
                   max_lead=0.0,
                   allow_equal=allow_equal,
                   on_missing_anchor=on_missing_anchor)

    @classmethod
    def symmetric_window(cls,
                         anchor_column: str,
                         half_width: float,
                         on_missing_anchor: Literal['skip', 'error'] = 'skip'
                         ) -> 'TemporalEligibility':
        """
        Construct a symmetric proximity rule.

        This is the closest continuous analogue of the fixed-window binning it
        replaces: it enforces anchor proximity without imposing direction, and
        removes the bin-edge discontinuity. Use it when the objective is
        balance on the anchor rather than no-lookahead.

        :param half_width: Permitted deviation in either direction.
        :returns: Rule admitting controls within ``half_width`` of the treated
            value.
        :rtype: TemporalEligibility
        """
        return cls(anchor_column=anchor_column,
                   max_lag=half_width,
                   max_lead=half_width,
                   on_missing_anchor=on_missing_anchor)

    def bounds(self, anchor: float) -> tuple[float, float]:
        """
        Return the admissible interval for one treated unit.

        :param anchor: Anchor value of the treated unit.
        :type anchor: float
        :returns: Inclusive ``(lower, upper)`` bounds, either of which may be
            infinite.
        :rtype: tuple[float, float]
        :raises ValueError: If the anchor is not finite.
        """
        if not np.isfinite(anchor):
            raise ValueError(
                f'Cannot form an admissible interval around a non-finite '
                f'anchor value ({anchor}).'
            )
        lower = -np.inf if self.max_lag is None else anchor - self.max_lag
        upper = np.inf if self.max_lead is None else anchor + self.max_lead
        return lower, upper

    def mask(self, anchor: float, candidate_values: np.ndarray) -> np.ndarray:
        """
        Evaluate admissibility of every candidate for one treated unit.

        Candidates with a non-finite anchor are inadmissible: their position
        in the ordering cannot be established, so neither can their
        eligibility. This is a deliberate exclusion rather than a silent pass.

        :param anchor: Anchor value of the treated unit.
        :type anchor: float
        :param candidate_values: Anchor values of the candidate controls,
            aligned positionally with the candidate block.
        :type candidate_values: np.ndarray
        :returns: Boolean mask over the candidates.
        :rtype: np.ndarray
        """
        lower, upper = self.bounds(anchor)
        values = np.asarray(candidate_values, dtype=float)

        finite = np.isfinite(values)
        with np.errstate(invalid='ignore'):
            admissible = finite & (values >= lower) & (values <= upper)

        if not self.allow_equal:
            admissible &= values != anchor

        return admissible

    def pool_sizes(self,
                   treated_values: np.ndarray,
                   candidate_values: np.ndarray) -> np.ndarray:
        """
        Count admissible controls for each treated unit, without matching.

        Intended as a pre-flight diagnostic. Running this before matching
        shows immediately whether the rule is close to non-binding, which is
        often the case when the treated cohort sits at the recent end of the
        record, or whether it starves the earliest treated units.

        :param treated_values: Anchor values of the treated units.
        :type treated_values: np.ndarray
        :param candidate_values: Anchor values of the candidate controls.
        :type candidate_values: np.ndarray
        :returns: Admissible count per treated unit, aligned with
            ``treated_values``. Units with a non-finite anchor return zero.
        :rtype: np.ndarray
        """
        treated = np.asarray(treated_values, dtype=float)
        sizes = np.zeros(len(treated), dtype=int)

        for position, anchor in enumerate(treated):
            if not np.isfinite(anchor):
                continue
            sizes[position] = int(self.mask(float(anchor), candidate_values).sum())

        return sizes

    def describe(self) -> str:
        """
        Render a one-line human-readable statement of the rule.

        :returns: Description suitable for logging or a methods section.
        :rtype: str
        """
        lag = 'unbounded' if self.max_lag is None else f'{self.max_lag:g}'
        lead = 'unbounded' if self.max_lead is None else f'{self.max_lead:g}'
        ties = 'admissible' if self.allow_equal else 'inadmissible'
        return (f"'{self.anchor_column}' in [treated - {lag}, treated + {lead}]; "
                f"exact ties {ties}")

    def validate_columns(self,
                         alternative_fleet: pd.DataFrame,
                         conventional_fleet: pd.DataFrame,
                         logger: logging.Logger | None = None) -> None:
        """
        Check the anchor column exists and is numeric in both fleets.

        :param alternative_fleet: Treated fleet.
        :param conventional_fleet: Control fleet.
        :param logger: Optional logger for coverage reporting.
        :raises KeyError: If the anchor column is absent from either fleet.
        :raises TypeError: If the anchor column is not numeric, since the rule
            relies on an ordering.
        """
        for name, fleet in (('alternative', alternative_fleet),
                            ('conventional', conventional_fleet)):
            if self.anchor_column not in fleet.columns:
                raise KeyError(
                    f"Anchor column '{self.anchor_column}' not found in the "
                    f"{name} fleet. Temporal eligibility requires the anchor "
                    f"in both fleets."
                )
            column = fleet[self.anchor_column]
            if isinstance(column, pd.DataFrame):
                raise KeyError(
                    f"Anchor column '{self.anchor_column}' appears "
                    f"{column.shape[1]} times in the {name} fleet."
                )
            if not pd.api.types.is_numeric_dtype(column):
                raise TypeError(
                    f"Anchor column '{self.anchor_column}' has dtype "
                    f"{column.dtype} in the {name} fleet. An ordered numeric "
                    f"type is required."
                )
            if logger is not None:
                n_missing = int(column.isna().sum())
                logger.info(
                    f'{name.capitalize()} fleet anchor coverage: '
                    f'{len(column) - n_missing}/{len(column)} ships with a '
                    f'valid {self.anchor_column}'
                )