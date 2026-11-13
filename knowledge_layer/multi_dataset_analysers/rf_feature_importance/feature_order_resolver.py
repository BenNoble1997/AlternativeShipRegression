from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


#: Maps a short strategy/dataset key to the results-DataFrame column holding
#: the corresponding ``{feature: importance}`` mapping.
IMPORTANCE_COLUMNS: Mapping[str, str] = {
    'conventional_full': 'importance_conventional_full',
    'conventional_matched': 'importance_conventional_matched',
    'alternative': 'importance_alternative',
}


@dataclass
class FeatureOrderResolver:
    """
    Resolve a stable row ordering for feature-importance heatmaps.

    Concept
    -------
    Heatmap row order and heatmap row *membership* are separate decisions.
    Membership is driven by magnitude (``top_n``); order is a presentation
    choice. Deriving both from a single dataset's importance ranking — as a
    naive ``sorted(conv_full.items())`` does — silently privileges that
    dataset as the reference frame and makes panels mutually incomparable,
    because each target re-derives its own ordering.

    This resolver separates the two and additionally distinguishes *scope*:

    - ``scope='panel'`` reproduces the legacy behaviour: order (and any
      ``top_n`` selection) is recomputed independently for each target.
    - ``scope='global'`` computes one canonical ordering across all retained
      targets via :meth:`freeze`, then applies it to every panel. Features
      absent from a given target's predictor set are dropped from that panel
      without disturbing the relative order of the rest.

    :param strategy: How the ranking score is derived. One of
        ``'conventional_full'``, ``'conventional_matched'``,
        ``'alternative'`` (rank by that dataset's importance), ``'mean'``
        (cross-dataset mean, reference-frame neutral), ``'max'``
        (cross-dataset maximum, favours features important *anywhere*),
        ``'alphabetical'``, or ``'explicit'`` (caller-supplied order).
    :type strategy: str
    :param scope: ``'global'`` or ``'panel'``. See above.
    :type scope: str
    :param explicit_order: Canonical feature sequence, required when
        ``strategy='explicit'``. Features present in the data but missing
        from this sequence are appended, ranked by cross-dataset mean.
    :type explicit_order: collections.abc.Sequence[str], optional
    :param debug: Emit stage-by-stage diagnostics during freezing and
        per-panel resolution.
    :type debug: bool
    :raises ValueError: If ``strategy`` or ``scope`` is unrecognised, or if
        ``strategy='explicit'`` without an ``explicit_order``.
    """

    strategy: str = 'mean'
    scope: str = 'global'
    explicit_order: Optional[Sequence[str]] = None
    debug: bool = False

    _frozen_order: List[str] = field(default_factory=list, init=False, repr=False)
    _is_frozen: bool = field(default=False, init=False, repr=False)

    VALID_STRATEGIES = (
        'conventional_full', 'conventional_matched', 'alternative',
        'mean', 'max', 'alphabetical', 'explicit',
    )
    VALID_SCOPES = ('global', 'panel')

    def __post_init__(self) -> None:
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown feature_order strategy {self.strategy!r}. "
                f"Valid options: {', '.join(self.VALID_STRATEGIES)}"
            )
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(
                f"Unknown feature_order scope {self.scope!r}. "
                f"Valid options: {', '.join(self.VALID_SCOPES)}"
            )
        if self.strategy == 'explicit' and not self.explicit_order:
            raise ValueError(
                "strategy='explicit' requires a non-empty explicit_order "
                "sequence of feature names."
            )

    # ── scoring ─────────────────────────────────────────────────────────────

    def _panel_scores(self, row: pd.Series) -> Dict[str, float]:
        """
        Compute the per-feature ranking score for one target.

        Missing features are scored as ``0.0`` rather than raising, so that
        targets with heterogeneous predictor sets can be combined.

        :param row: One row of the comparison results DataFrame.
        :type row: pandas.Series
        :returns: Mapping from feature name to ranking score.
        :rtype: dict[str, float]
        """
        per_dataset = {
            key: (row[column] or {})
            for key, column in IMPORTANCE_COLUMNS.items()
        }
        features = set().union(*(d.keys() for d in per_dataset.values()))

        if self.strategy in IMPORTANCE_COLUMNS:
            source = per_dataset[self.strategy]
            return {f: float(source.get(f, 0.0)) for f in features}

        if self.strategy == 'max':
            return {
                f: max(float(d.get(f, 0.0)) for d in per_dataset.values())
                for f in features
            }

        # 'mean', plus the magnitude fallback used by 'alphabetical'/'explicit'
        n_datasets = len(per_dataset)
        return {
            f: sum(float(d.get(f, 0.0)) for d in per_dataset.values()) / n_datasets
            for f in features
        }

    def _rank(self, scores: Mapping[str, float]) -> List[str]:
        """
        Convert a score mapping into an ordered feature list.

        :param scores: Mapping from feature name to ranking score.
        :type scores: collections.abc.Mapping[str, float]
        :returns: Feature names in display order.
        :rtype: list[str]
        """
        if self.strategy == 'explicit':
            declared = [f for f in self.explicit_order if f in scores]
            declared_set = set(self.explicit_order)
            remainder = sorted(
                (f for f in scores if f not in declared_set),
                key=lambda f: (-scores[f], f),
            )
            return declared + remainder

        if self.strategy == 'alphabetical':
            return sorted(scores)

        return sorted(scores, key=lambda f: (-scores[f], f))

    # ── global freezing ─────────────────────────────────────────────────────

    def freeze(self, rows: Iterable[pd.Series]) -> List[str]:
        """
        Compute and store the canonical ordering across all targets.

        Scores are averaged across targets in which each feature appears, so
        a feature available for only two of five targets is not penalised for
        the three absences. No-op when ``scope='panel'``.

        :param rows: Rows of the retained comparison results.
        :type rows: collections.abc.Iterable[pandas.Series]
        :returns: The frozen canonical order (empty when scope is ``'panel'``).
        :rtype: list[str]
        """
        if self.scope != 'global':
            if self.debug:
                print("   [order] scope='panel' — no global freeze performed")
            return []

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for row in rows:
            for feature, score in self._panel_scores(row).items():
                totals[feature] = totals.get(feature, 0.0) + score
                counts[feature] = counts.get(feature, 0) + 1

        mean_scores = {f: totals[f] / counts[f] for f in totals}
        self._frozen_order = self._rank(mean_scores)
        self._is_frozen = True

        if self.debug:
            print(f"   [order] frozen canonical order "
                  f"(strategy={self.strategy!r}, {len(self._frozen_order)} features):")
            for rank, feature in enumerate(self._frozen_order, start=1):
                print(f"      {rank:>2}. {feature:<20} "
                      f"score={mean_scores[feature]:.4f} "
                      f"(present in {counts[feature]} target(s))")

        return list(self._frozen_order)

    @property
    def frozen_order(self) -> List[str]:
        """
        The canonical order established by :meth:`freeze`.

        :returns: Copy of the frozen order; empty if not frozen.
        :rtype: list[str]
        """
        return list(self._frozen_order)

    # ── per-panel resolution ────────────────────────────────────────────────

    def order_for(self,
                  row: pd.Series,
                  top_n: Optional[int] = None) -> List[str]:
        """
        Resolve the ordered feature list for one heatmap panel.

        Selection and ordering are applied in that sequence. Under
        ``scope='global'`` the ``top_n`` cut is taken from the frozen order,
        so every panel shows the same features (minus any unavailable for
        that target). Under ``scope='panel'`` the cut is taken from that
        panel's own ranking.

        :param row: One row of the comparison results DataFrame.
        :type row: pandas.Series
        :param top_n: Maximum number of features to display. ``None`` shows
            all available features.
        :type top_n: int, optional
        :returns: Feature names in final display order (top row first).
        :rtype: list[str]
        """
        scores = self._panel_scores(row)
        available = set(scores)

        if self._is_frozen:
            canonical = [f for f in self._frozen_order if f in available]
            selected = canonical[:top_n] if top_n is not None else canonical
        else:
            canonical = self._rank(scores)
            if top_n is not None:
                keep = set(sorted(available,
                                  key=lambda f: (-scores[f], f))[:top_n])
                selected = [f for f in canonical if f in keep]
            else:
                selected = canonical

        if self.debug:
            dropped = sorted(f for f in self._frozen_order if f not in available)
            print(f"   [order] target={row['target']!r}: "
                  f"{len(selected)}/{len(available)} feature(s) shown"
                  + (f", unavailable: {dropped}" if dropped else ""))

        return selected