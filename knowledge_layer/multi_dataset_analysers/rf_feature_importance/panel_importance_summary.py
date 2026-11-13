from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

IMPORTANCE_COLUMNS: Mapping[str, str] = {
    'conventional_full': 'importance_conventional_full',
    'conventional_matched': 'importance_conventional_matched',
    'alternative': 'importance_alternative',
}


@dataclass(frozen=True)
class PanelImportanceSummary:
    """
    Summary statistics for one (target, dataset) importance dictionary.

    Concept
    -------
    Collapses a ``{feature: importance}`` mapping into the quantities needed
    for quick cross-dataset comparison: which feature dominates, by how much,
    and how concentrated the importance distribution is. The ``spread``
    (max − min) is diagnostic in its own right: under range restriction in
    matched subsets, RF impurity importance diffuses across collinear
    predictors, so a shrinking spread flags panels where top-feature identity
    is not meaningful.

    :param top_feature: Feature with the largest importance, or ``None`` for
        an empty panel.
    :type top_feature: str, optional
    :param top_value: Importance of ``top_feature``.
    :type top_value: float
    :param min_value: Smallest importance in the panel.
    :type min_value: float
    :param max_value: Largest importance in the panel.
    :type max_value: float
    :param spread: ``max_value − min_value``.
    :type spread: float
    :param n_features: Number of predictors in the panel.
    :type n_features: int
    """

    top_feature: Optional[str]
    top_value: float
    min_value: float
    max_value: float
    spread: float
    n_features: int

    @classmethod
    def from_importances(cls,
                         importances: Optional[Mapping[str, float]]
                         ) -> 'PanelImportanceSummary':
        """
        Build a summary from one importance mapping.

        Handles ``None`` and empty mappings gracefully (returns a summary of
        NaNs), since a cohort can be missing for a given target.

        :param importances: Mapping from feature name to importance.
        :type importances: collections.abc.Mapping[str, float], optional
        :returns: The computed summary.
        :rtype: PanelImportanceSummary
        """
        if not importances:
            nan = float('nan')
            return cls(top_feature=None, top_value=nan, min_value=nan,
                       max_value=nan, spread=nan, n_features=0)

        top_feature = max(importances, key=lambda f: importances[f])
        values = [float(v) for v in importances.values()]
        return cls(
            top_feature=top_feature,
            top_value=float(importances[top_feature]),
            min_value=min(values),
            max_value=max(values),
            spread=max(values) - min(values),
            n_features=len(values),
        )


class ImportanceSummaryTableBuilder:
    """
    Build and export a per-target, per-dataset importance summary table.

    Concept
    -------
    For each target in the comparison results and each of the three cohorts
    (Conventional Full, Conventional Matched, Alternative), reports the most
    important predictor and the min / max / spread of the importance
    distribution. Reading the two together is the point: top-feature identity
    is only trustworthy where the spread is large, so the table pairs every
    "who wins" with "by how much the field is separated".

    The builder is composed from three stages, each a separate method:
    per-panel summarisation (:class:`PanelImportanceSummary`), assembly into
    a tidy long DataFrame (:meth:`build_long`), and reshaping into a wide
    MultiIndex layout (:meth:`build_wide`). Export
    (:meth:`to_excel`) consumes either layout.

    :param comparison_results: The analyzer's results DataFrame; must contain
        the columns in :data:`IMPORTANCE_COLUMNS` plus ``'target'``.
    :type comparison_results: pandas.DataFrame
    :param dataset_labels: Display names for the three cohorts, keyed by the
        short keys of :data:`IMPORTANCE_COLUMNS`. Defaults to
        ``'Conv. (Full)'``, ``'Conv. (Matched)'``, ``'Alternative'``.
    :type dataset_labels: dict[str, str], optional
    :param variable_name_map: Optional mapping from internal variable names to
        publication-friendly labels, applied to targets and top features.
    :type variable_name_map: dict[str, str], optional
    :param float_precision: Decimal places used when rounding numeric columns
        for export. ``None`` disables rounding.
    :type float_precision: int, optional
    """

    DEFAULT_LABELS = {
        'conventional_full': 'Conv. (Full)',
        'conventional_matched': 'Conv. (Matched)',
        'alternative': 'Alternative',
    }

    #: Metric column order within each dataset block.
    METRICS = ('Top feature', 'Top imp.', 'Min', 'Max', 'Spread')

    def __init__(self,
                 comparison_results: pd.DataFrame,
                 dataset_labels: Optional[Dict[str, str]] = None,
                 variable_name_map: Optional[Dict[str, str]] = None,
                 float_precision: Optional[int] = 3):
        self.comparison_results = comparison_results
        self.dataset_labels = dataset_labels or dict(self.DEFAULT_LABELS)
        self.variable_name_map = variable_name_map or {}
        self.float_precision = float_precision

    # ── stage 1: per-row summarisation ──────────────────────────────────────

    def _summarise_row(self, row: pd.Series
                       ) -> Dict[str, PanelImportanceSummary]:
        """
        Summarise all three cohort panels for one target.

        :param row: One row of the comparison results DataFrame.
        :type row: pandas.Series
        :returns: Mapping from cohort key to its panel summary.
        :rtype: dict[str, PanelImportanceSummary]
        """
        return {
            key: PanelImportanceSummary.from_importances(row.get(column))
            for key, column in IMPORTANCE_COLUMNS.items()
        }

    def _display_name(self, name: Optional[str]) -> Optional[str]:
        """
        Map an internal variable name to its display label.

        :param name: Internal variable name, or ``None``.
        :type name: str, optional
        :returns: Mapped label, or the input unchanged when unmapped.
        :rtype: str, optional
        """
        if name is None:
            return None
        return self.variable_name_map.get(name, name)

    # ── stage 2: long assembly ──────────────────────────────────────────────

    def build_long(self, debug: bool = False) -> pd.DataFrame:
        """
        Assemble the tidy long-format table: one row per (target, dataset).

        :param debug: Print each target's per-cohort summary as it is built.
        :type debug: bool
        :returns: Long-format summary with columns ``Target``, ``Dataset``,
            and the metric columns.
        :rtype: pandas.DataFrame
        """
        records: List[Dict] = []

        for _, row in self.comparison_results.iterrows():
            target_display = self._display_name(row['target'])
            summaries = self._summarise_row(row)

            if debug:
                print(f"   [summary] target={row['target']!r}")

            for key, summary in summaries.items():
                if debug:
                    print(f"      {self.dataset_labels[key]:<16} "
                          f"top={summary.top_feature} "
                          f"({summary.top_value:.3f}), "
                          f"range=[{summary.min_value:.3f}, "
                          f"{summary.max_value:.3f}], "
                          f"spread={summary.spread:.3f}")

                records.append({
                    'Target': target_display,
                    'Dataset': self.dataset_labels[key],
                    'Top feature': self._display_name(summary.top_feature),
                    'Top imp.': summary.top_value,
                    'Min': summary.min_value,
                    'Max': summary.max_value,
                    'Spread': summary.spread,
                    'N features': summary.n_features,
                })

        return pd.DataFrame.from_records(records)

    # ── stage 3: wide reshaping ─────────────────────────────────────────────

    def build_wide(self, debug: bool = False) -> pd.DataFrame:
        """
        Reshape into the wide layout: one row per target, two-level columns
        ``(dataset, metric)``.

        Dataset blocks follow the order of ``dataset_labels``; metrics follow
        :attr:`METRICS`. Excel renders the top level as merged group headers.

        :param debug: Forwarded to :meth:`build_long`.
        :type debug: bool
        :returns: Wide-format summary indexed by ``Target``.
        :rtype: pandas.DataFrame
        """
        long_df = self.build_long(debug=debug)
        if long_df.empty:
            return long_df

        wide = long_df.pivot(index='Target', columns='Dataset',
                             values=list(self.METRICS))
        # pivot yields (metric, dataset); flip to (dataset, metric) and order
        wide = wide.swaplevel(axis=1)
        ordered = [(label, metric)
                   for label in self.dataset_labels.values()
                   for metric in self.METRICS]
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered))

        # Preserve the original target order rather than alphabetical
        target_order = [self._display_name(t)
                        for t in self.comparison_results['target']]
        wide = wide.reindex(target_order)

        return wide

    # ── export ──────────────────────────────────────────────────────────────

    def to_excel(self,
                 filepath: str | Path,
                 layout: str = 'wide',
                 sheet_name: str = 'Importance summary',
                 debug: bool = False) -> Path:
        """
        Build the table and write it to an Excel file.

        Numeric columns are rounded to :attr:`float_precision` and column
        widths are set from content so the file is readable without manual
        adjustment.

        :param filepath: Destination ``.xlsx`` path; parent directories are
            created as needed.
        :type filepath: str or pathlib.Path
        :param layout: ``'wide'`` (one row per target, grouped dataset
            columns) or ``'long'`` (one row per target-dataset pair).
        :type layout: str
        :param sheet_name: Worksheet name.
        :type sheet_name: str
        :param debug: Print per-stage diagnostics during building and export.
        :type debug: bool
        :returns: The resolved output path.
        :rtype: pathlib.Path
        :raises ValueError: If ``layout`` is not ``'wide'`` or ``'long'``.
        """
        if layout not in ('wide', 'long'):
            raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")

        table = (self.build_wide(debug=debug) if layout == 'wide'
                 else self.build_long(debug=debug))

        if self.float_precision is not None:
            table = table.round(self.float_precision)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if debug:
            print(f"   [summary] layout={layout!r}, shape={table.shape}")
            print(f"   [summary] writing to {filepath}")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            table.to_excel(writer, sheet_name=sheet_name,
                           index=True, merge_cells=True)
            self._autofit_columns(writer.sheets[sheet_name], table, debug)

        if debug:
            print("   [summary] export complete")

        return filepath

    @staticmethod
    def _autofit_columns(worksheet, table: pd.DataFrame,
                         debug: bool = False) -> None:
        """
        Set column widths from the longest cell value per column.

        Uses a simple character-count heuristic (clamped to 8–30 characters),
        which is sufficient for numeric and short-string content.

        :param worksheet: The openpyxl worksheet to adjust.
        :param table: The DataFrame that was written, used to size columns.
        :type table: pandas.DataFrame
        :param debug: Print the widths applied.
        :type debug: bool
        """
        from openpyxl.utils import get_column_letter

        n_index_cols = table.index.nlevels
        frame = table.reset_index()

        for pos, column in enumerate(frame.columns, start=1):
            header_len = max(len(str(part)) for part in
                             (column if isinstance(column, tuple) else (column,)))
            content_len = int(
                frame.iloc[:, pos - 1].astype(str).str.len().max()
            )
            width = min(max(header_len, content_len, 8) + 2, 30)
            worksheet.column_dimensions[get_column_letter(pos)].width = width

        if debug:
            print(f"   [summary] autofit applied to "
                  f"{len(frame.columns)} columns "
                  f"({n_index_cols} index level(s))")