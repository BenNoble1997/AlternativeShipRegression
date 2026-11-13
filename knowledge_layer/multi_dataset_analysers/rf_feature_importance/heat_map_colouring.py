from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd

IMPORTANCE_COLUMNS: Mapping[str, str] = {
    'conventional_full': 'importance_conventional_full',
    'conventional_matched': 'importance_conventional_matched',
    'alternative': 'importance_alternative',
}

@dataclass
class HeatmapPanelResult:
    """
    Outcome of drawing one heatmap panel.

    Returned by :meth:`_plot_heatmap_on_axis` so callers can build titles from
    the features actually drawn and attach a figure-level colorbar to the
    mappable without re-deriving either.

    :param features: Feature names drawn, in display (top-to-bottom) order.
    :type features: list[str]
    :param mappable: The ``AxesImage`` produced by ``imshow``, suitable for
        passing to ``Figure.colorbar``.
    :type mappable: matplotlib.image.AxesImage
    :param vmax: Upper bound of the colour scale used for this panel.
    :type vmax: float
    """

    features: List[str]
    mappable: object
    vmax: float


@dataclass
class ImportanceColorScale:
    """
    Resolve the colour-scale bounds for feature-importance heatmaps.

    Concept
    -------
    Per-panel ``vmax`` makes colour encode *rank within panel* rather than
    importance magnitude: identical hues in two panels can represent very
    different values, so cross-panel visual comparison is invalid. Freezing a
    single scale across the figure restores magnitude as the encoded quantity,
    at the cost of dynamic range in panels whose values are all small.

    The strategy controls what the top of the ramp anchors to:

    - ``'panel'`` — legacy behaviour, recomputed per target. No clipping,
      no cross-panel comparability.
    - ``'figure'`` — the largest importance among all drawn cells. Nothing
      clips; colour is directly interpretable.
    - ``'percentile'`` — the ``percentile``-th percentile of all drawn cells.
      Dominant cells saturate, freeing the ramp for the crowded low end.
    - ``'spread'`` — the largest cross-dataset range (max − min for a single
      feature) among all drawn cells. Devotes the ramp to the magnitude at
      which cohort differences occur, making small shifts legible.
    - ``'fixed'`` — a caller-supplied ``fixed_vmax``, for matching scales
      across separately generated figures.

    Clipping is comparatively cheap when cell annotations are enabled, since
    the exact value remains printed even where colour saturates. The
    :attr:`extend` property reports whether the colorbar should be drawn with
    an overflow arrow.

    :param strategy: One of ``'panel'``, ``'figure'``, ``'percentile'``,
        ``'spread'``, ``'fixed'``.
    :type strategy: str
    :param fixed_vmax: Explicit upper bound, required when
        ``strategy='fixed'``.
    :type fixed_vmax: float, optional
    :param percentile: Percentile in ``(0, 100]`` used when
        ``strategy='percentile'``.
    :type percentile: float
    :param round_to: Round the resolved bound up to the nearest multiple of
        this value, matching the convention in
        :meth:`_calculate_optimal_vmax`. ``None`` disables rounding and
        reproduces the legacy raw maximum under ``strategy='panel'``.
    :type round_to: float, optional
    :param debug: Emit diagnostics while freezing.
    :type debug: bool
    :raises ValueError: If ``strategy`` is unrecognised, if
        ``strategy='fixed'`` without a positive ``fixed_vmax``, or if
        ``percentile`` is outside ``(0, 100]``.
    """

    strategy: str = 'figure'
    fixed_vmax: Optional[float] = None
    percentile: float = 95.0
    round_to: Optional[float] = 0.05
    debug: bool = False

    _vmax: Optional[float] = field(default=None, init=False, repr=False)
    _observed_max: float = field(default=0.0, init=False, repr=False)
    _is_frozen: bool = field(default=False, init=False, repr=False)

    VALID_STRATEGIES = ('panel', 'figure', 'percentile', 'spread', 'fixed')


    def __post_init__(self) -> None:
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown color_scale strategy {self.strategy!r}. "
                f"Valid options: {', '.join(self.VALID_STRATEGIES)}"
            )
        if self.strategy == 'fixed' and not (self.fixed_vmax and self.fixed_vmax > 0):
            raise ValueError(
                "strategy='fixed' requires a positive fixed_vmax."
            )
        if not 0 < self.percentile <= 100:
            raise ValueError(
                f"percentile must lie in (0, 100], got {self.percentile}."
            )

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def is_shared(self) -> bool:
        """
        Whether a single scale applies to every panel.

        :returns: ``True`` for all strategies except ``'panel'``.
        :rtype: bool
        """
        return self.strategy != 'panel'

    @property
    def vmax(self) -> Optional[float]:
        """
        The frozen upper bound.

        :returns: Resolved ``vmax``, or ``None`` if not frozen / per-panel.
        :rtype: float or None
        """
        return self._vmax

    @property
    def clips(self) -> bool:
        """
        Whether any drawn cell exceeds the resolved bound.

        :returns: ``True`` if saturation occurs somewhere in the figure.
        :rtype: bool
        """
        return bool(self._vmax is not None and self._observed_max > self._vmax + 1e-12)

    @property
    def extend(self) -> str:
        """
        Colorbar ``extend`` mode implied by the resolved bound.

        :returns: ``'max'`` when clipping occurs, otherwise ``'neither'``.
        :rtype: str
        """
        return 'max' if self.clips else 'neither'

    # ── internals ───────────────────────────────────────────────────────────

    def _round_up(self, value: float) -> float:
        """
        Round a bound up to a clean multiple, with a non-zero floor.

        :param value: Raw bound.
        :type value: float
        :returns: Rounded bound, never zero.
        :rtype: float
        """
        if self.round_to:
            value = float(np.ceil(value / self.round_to) * self.round_to)
            return max(value, self.round_to)
        return max(float(value), 1e-6)

    def _collect(self,
                 rows: Iterable[pd.Series],
                 order_resolver: Optional['FeatureOrderResolver'],
                 top_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather every cell value and per-feature spread that will be drawn.

        The scale is derived from the cells that actually appear, so a
        ``top_n`` cut excludes the discarded tail from the calculation.

        :param rows: Retained comparison-result rows.
        :type rows: collections.abc.Iterable[pandas.Series]
        :param order_resolver: Shared ordering policy, used to determine which
            features each panel will show.
        :type order_resolver: FeatureOrderResolver, optional
        :param top_n: Row cap per panel.
        :type top_n: int, optional
        :returns: ``(cell_values, feature_spreads)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        values: List[float] = []
        spreads: List[float] = []

        # Suppress per-panel ordering chatter during the scale pass
        prior_debug = getattr(order_resolver, 'debug', False)
        if order_resolver is not None:
            order_resolver.debug = False

        try:
            for row in rows:
                dicts = [row[column] or {} for column in IMPORTANCE_COLUMNS.values()]

                if order_resolver is not None:
                    features = order_resolver.order_for(row, top_n=top_n)
                else:
                    ranked = sorted(dicts[0], key=lambda f: dicts[0][f], reverse=True)
                    features = ranked[:top_n] if top_n is not None else ranked

                for feature in features:
                    cells = [float(d.get(feature, 0.0)) for d in dicts]
                    values.extend(cells)
                    spreads.append(max(cells) - min(cells))
        finally:
            if order_resolver is not None:
                order_resolver.debug = prior_debug

        return np.asarray(values, dtype=float), np.asarray(spreads, dtype=float)

    # ── freezing ────────────────────────────────────────────────────────────

    def freeze(self,
               rows: Iterable[pd.Series],
               order_resolver: Optional['FeatureOrderResolver'] = None,
               top_n: Optional[int] = None) -> Optional[float]:
        """
        Compute and store the shared colour bound for the whole figure.

        Must be called after feature ordering has been frozen, since the set
        of drawn cells depends on the ordering and ``top_n`` cut. No-op when
        ``strategy='panel'``.

        :param rows: Retained comparison-result rows.
        :type rows: collections.abc.Iterable[pandas.Series]
        :param order_resolver: Shared ordering policy.
        :type order_resolver: FeatureOrderResolver, optional
        :param top_n: Row cap per panel.
        :type top_n: int, optional
        :returns: The resolved ``vmax``, or ``None`` for per-panel scaling.
        :rtype: float or None
        """
        if self.strategy == 'panel':
            if self.debug:
                print("   [scale] strategy='panel' — vmax recomputed per subplot")
            return None

        values, spreads = self._collect(rows, order_resolver, top_n)

        if values.size == 0:
            self._vmax = self._round_up(0.0)
            self._is_frozen = True
            return self._vmax

        self._observed_max = float(np.nanmax(values))

        if self.strategy == 'fixed':
            raw = float(self.fixed_vmax)
        elif self.strategy == 'figure':
            raw = self._observed_max
        elif self.strategy == 'percentile':
            raw = float(np.nanpercentile(values, self.percentile))
        else:  # 'spread'
            raw = float(np.nanmax(spreads)) if spreads.size else self._observed_max

        self._vmax = float(self.fixed_vmax) if self.strategy == 'fixed' else self._round_up(raw)
        self._is_frozen = True

        if self.debug:
            n_clipped = int(np.sum(values > self._vmax + 1e-12))
            print(f"   [scale] strategy={self.strategy!r} → vmax={self._vmax:.4f}")
            print(f"      cells considered : {values.size}")
            print(f"      observed max     : {self._observed_max:.4f}")
            print(f"      largest spread   : "
                  f"{(np.nanmax(spreads) if spreads.size else 0.0):.4f}")
            print(f"      cells clipped    : {n_clipped} "
                  f"({100 * n_clipped / values.size:.1f}%)")
            if n_clipped:
                print("      note: clipped cells retain their printed value "
                      "when annotate=True")

        return self._vmax

    def vmax_for(self, importance_matrix: np.ndarray) -> float:
        """
        Return the upper bound to apply to one panel.

        :param importance_matrix: The panel's ``(n_features, 3)`` matrix.
        :type importance_matrix: numpy.ndarray
        :returns: Frozen shared bound, or the panel maximum under
            ``strategy='panel'``.
        :rtype: float
        """
        if self._is_frozen and self._vmax is not None:
            return self._vmax

        panel_max = float(np.nanmax(importance_matrix)) if importance_matrix.size else 0.0
        return self._round_up(panel_max) if self.round_to else max(panel_max, 1e-6)