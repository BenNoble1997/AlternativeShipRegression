# cross_performance_results.py
"""
Result-set container and summariser for multi-dataset cross-performance analysis.
=================================================================================

This module defines the persistence boundary between analysis and reporting.

**Concept:**
A cross-performance analysis produces a results table, but the table alone is not
interpretable. Column ordering in the reporting tables follows the declared fuel
cohort order, captions must state the evaluation protocol, and relative error
metrics are only meaningful alongside the reference scale used as their divisor.
Persisting the frame without this context yields a file that can be read but not
faithfully typeset.

``CrossPerformanceResultSet`` bundles the numeric results with exactly the context
required to regenerate a publication table. It is the single object exchanged
between the analyser, the workbook writer/reader, and the LaTeX generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

#: Columns holding Python sequences that require JSON round-tripping through Excel.
LIST_COLUMNS: tuple[str, ...] = ("predictors",)

#: Schema version written into the workbook metadata sheet.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class CrossPerformanceRunConfig:
    """
    Immutable description of the evaluation protocol used for one analysis run.

    **Concept:**
    When comparing validation strategies, the strategy name alone is an incomplete
    identifier: five-fold cross-validation repeated ten times and repeated twenty
    times are distinct protocols producing distinct variance estimates. This class
    carries the method together with its governing parameters, and exposes a
    single filesystem-safe tag that distinguishes runs from one another.

    :param evaluation_method: One of ``'standard-cv'``, ``'repeated-cv'``,
        ``'bootstrap'``.
    :param n_folds: Number of cross-validation folds, where applicable.
    :param n_repeats: Number of repetitions for ``'repeated-cv'``.
    :param n_bootstraps: Number of bootstrap replicates for ``'bootstrap'``.
    :param max_predictors: Cap on predictor count applied during candidate
        selection, recorded because it changes which models enter the table.
    :param selection_metric: Metric used to rank candidate combinations.
    :param random_state: Seed governing resampling, recorded for reproducibility.
    """

    evaluation_method: str
    n_folds: int | None = None
    n_repeats: int | None = None
    n_bootstraps: int | None = None
    max_predictors: int | None = None
    selection_metric: str = "rrmse"
    random_state: int = 42

    @property
    def validation_tag(self) -> str:
        """
        Compact, filesystem-safe identifier of the evaluation protocol.

        Encodes the governing parameters so that two runs differing only in fold
        or replicate count produce different filenames.

        :returns: Tag such as ``'bootstrap632-nb200'`` or ``'repeated-cv-k5x10'``.
        """
        method = self.evaluation_method
        if method == "bootstrap":
            return f"bootstrap632-nb{self.n_bootstraps or 0}"
        if method == "repeated-cv":
            return f"repeated-cv-k{self.n_folds or 0}x{self.n_repeats or 0}"
        if method == "standard-cv":
            return f"standard-cv-k{self.n_folds or 0}"
        return method.replace("_", "-")

    @property
    def caption_phrase(self) -> str:
        """
        Human-readable description of the protocol for use in table captions.

        :returns: Phrase such as ``'.632+ bootstrap (200 replicates)'``.
        """
        method = self.evaluation_method
        if method == "bootstrap":
            return f".632+ bootstrap ({self.n_bootstraps} replicates)"
        if method == "repeated-cv":
            return f"{self.n_folds}-fold cross-validation, {self.n_repeats} repeats"
        if method == "standard-cv":
            return f"{self.n_folds}-fold cross-validation"
        return method

    def to_mapping(self) -> dict[str, Any]:
        """
        Flatten to a plain mapping for metadata serialisation.

        :returns: Mapping of field name to value.
        """
        return {
            "evaluation_method": self.evaluation_method,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "n_bootstraps": self.n_bootstraps,
            "max_predictors": self.max_predictors,
            "selection_metric": self.selection_metric,
            "random_state": self.random_state,
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "CrossPerformanceRunConfig":
        """
        Rebuild from a metadata mapping, tolerating absent optional fields.

        :param mapping: Mapping as produced by :meth:`to_mapping`.
        :raises KeyError: If ``evaluation_method`` is absent.
        :returns: Reconstructed configuration.
        """
        if "evaluation_method" not in mapping:
            raise KeyError(
                "Metadata lacks 'evaluation_method'; the workbook was not written "
                "by CrossPerformanceWorkbookWriter or is from an older schema."
            )

        def _as_int(key: str) -> int | None:
            value = mapping.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return int(value)

        return cls(
            evaluation_method=str(mapping["evaluation_method"]),
            n_folds=_as_int("n_folds"),
            n_repeats=_as_int("n_repeats"),
            n_bootstraps=_as_int("n_bootstraps"),
            max_predictors=_as_int("max_predictors"),
            selection_metric=str(mapping.get("selection_metric", "rrmse")),
            random_state=int(mapping.get("random_state", 42) or 42),
        )


@dataclass
class CrossPerformanceResultSet:
    """
    Self-describing record of one multi-dataset cross-performance analysis.

    **Concept:**
    This is the interchange object between computation and reporting. It holds the
    aggregated results frame together with every piece of context a downstream
    consumer needs: the declared fuel cohort ordering that fixes table column
    order, the evaluation protocol quoted in captions, the reference scale
    underlying the relative error metrics, and cohort sizes. Because it is
    complete, a table generated from a workbook on disk is byte-identical to one
    generated from the live analyser.

    :param comparison_results: Long-format results frame, one row per
        (fuel type, target, predictor combination).
    :param fuel_types: Fuel cohort names in declared order. Controls table column
        order and must not be re-derived from the frame.
    :param analyzer_type: ``'parametric'`` or ``'random_forest'``.
    :param run_config: Evaluation protocol used.
    :param dataset_sizes: Mapping of cohort name to ship count, including the
        conventional cohort.
    :param scale_reference_means: Fixed per-target divisors used for relative
        error metrics.
    :param scale_reference_source: Provenance label of the reference scale.
    :param ship_type_name: Name of the ``ShipType`` enum member, if known.
    :param conventional_label: Token identifying the conventional cohort in
        filenames and summaries.
    :param created_at: ISO timestamp of construction.
    :param source_path: Workbook the set was read from, if any.
    """

    comparison_results: pd.DataFrame
    fuel_types: list[str]
    analyzer_type: str
    run_config: CrossPerformanceRunConfig
    dataset_sizes: dict[str, int] = field(default_factory=dict)
    scale_reference_means: dict[str, float] = field(default_factory=dict)
    scale_reference_source: str = ""
    ship_type_name: str | None = None
    conventional_label: str = "conv"
    created_at: str = ""
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.comparison_results is None:
            raise ValueError("comparison_results cannot be None.")
        if not isinstance(self.comparison_results, pd.DataFrame):
            raise TypeError(
                f"comparison_results must be a DataFrame, got "
                f"{type(self.comparison_results).__name__}."
            )
        if not self.fuel_types:
            raise ValueError(
                "fuel_types is empty. Column ordering in the reporting tables "
                "depends on this being the declared cohort order."
            )
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def dataset_tokens(self) -> list[str]:
        """
        Cohort tokens in the order they appear in reporting tables.

        :returns: Conventional label followed by the fuel cohort names.
        """
        return [self.conventional_label, *self.fuel_types]

    @property
    def targets(self) -> list[str]:
        """
        Target dimensions present in the results, in sorted order.

        :returns: Sorted unique target names.
        """
        if "target" not in self.comparison_results.columns:
            return []
        return sorted(self.comparison_results["target"].dropna().unique().tolist())

    @property
    def is_empty(self) -> bool:
        """
        Whether the set carries any rows.

        :returns: ``True`` if the results frame has no rows.
        """
        return len(self.comparison_results) == 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_analyzer(cls,
                      analyzer: Any,
                      run_config: CrossPerformanceRunConfig | None = None,
                      ship_type: Any | None = None,
                      conventional_label: str = "conv",
                      debug: bool = False) -> "CrossPerformanceResultSet":
        """
        Package a live ``MultiDatasetCrossPerformanceAnalyzer`` into a result set.

        Reads the analyser by attribute rather than by type, so the same routine
        serves both the parametric and Random Forest configurations.

        :param analyzer: Analyser holding ``comparison_results``.
        :param run_config: Protocol description. Falls back to
            ``analyzer.last_run_config`` when omitted.
        :param ship_type: ``ShipType`` member recorded for filename and caption.
        :param conventional_label: Token for the conventional cohort.
        :param debug: Emit stage-by-stage diagnostics.
        :raises ValueError: If the analyser holds no results or no run config
            can be determined.
        :returns: Populated result set.
        """
        results = getattr(analyzer, "comparison_results", None)
        if results is None or len(results) == 0:
            raise ValueError(
                "Analyser holds no comparison_results. Run analyze_all_datasets() "
                "before exporting."
            )

        if run_config is None:
            run_config = getattr(analyzer, "last_run_config", None)
        if run_config is None:
            raise ValueError(
                "No run configuration available. Either pass run_config explicitly "
                "or run analyze_all_datasets(), which records it as "
                "analyzer.last_run_config."
            )

        if debug:
            print("[ResultSet] stage 1: results frame acquired "
                  f"({len(results)} rows, {results.shape[1]} columns)")

        alternative = getattr(analyzer, "alternative_datasets", {}) or {}
        fuel_types = list(alternative.keys())
        if not fuel_types:
            fuel_types = list(results["fuel_type"].unique())

        sizes: dict[str, int] = {
            conventional_label: int(len(getattr(analyzer, "conventional_data", [])))
        }
        for name, frame in alternative.items():
            sizes[name] = int(len(frame))

        if debug:
            print(f"[ResultSet] stage 2: cohorts {fuel_types}, sizes {sizes}")

        scale = getattr(analyzer, "scale_reference", None)
        means = dict(getattr(scale, "means", {}) or {}) if scale is not None else {}
        source = str(getattr(scale, "source", "")) if scale is not None else ""

        ship_type_name = None
        if ship_type is not None:
            ship_type_name = getattr(ship_type, "name", str(ship_type))

        if debug:
            print(f"[ResultSet] stage 3: scale reference '{source}' "
                  f"over {len(means)} targets; ship type {ship_type_name}")

        return cls(
            comparison_results=results.copy(),
            fuel_types=fuel_types,
            analyzer_type=str(getattr(analyzer, "analyzer_type", "unknown")),
            run_config=run_config,
            dataset_sizes=sizes,
            scale_reference_means=means,
            scale_reference_source=source,
            ship_type_name=ship_type_name,
            conventional_label=conventional_label,
        )

    # ------------------------------------------------------------------
    # Metadata serialisation
    # ------------------------------------------------------------------

    def to_metadata(self) -> dict[str, Any]:
        """
        Flatten all non-tabular context into a single mapping.

        :returns: Mapping written to the workbook metadata sheet.
        """
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "analyzer_type": self.analyzer_type,
            "fuel_types": list(self.fuel_types),
            "conventional_label": self.conventional_label,
            "dataset_sizes": dict(self.dataset_sizes),
            "scale_reference_source": self.scale_reference_source,
            "ship_type_name": self.ship_type_name,
            "created_at": self.created_at,
            "n_rows": int(len(self.comparison_results)),
        }
        metadata.update(self.run_config.to_mapping())
        return metadata

    @classmethod
    def from_parts(cls,
                   comparison_results: pd.DataFrame,
                   metadata: dict[str, Any],
                   scale_reference_means: dict[str, float],
                   source_path: Path | None = None) -> "CrossPerformanceResultSet":
        """
        Rebuild from the tabular and metadata components read out of a workbook.

        :param comparison_results: Results frame with list columns restored.
        :param metadata: Mapping as produced by :meth:`to_metadata`.
        :param scale_reference_means: Per-target reference divisors.
        :param source_path: Workbook the components came from.
        :raises KeyError: If required metadata keys are absent.
        :returns: Reconstructed result set.
        """
        fuel_types = metadata.get("fuel_types")
        if not fuel_types:
            raise KeyError(
                "Metadata lacks 'fuel_types'. Cohort ordering cannot be recovered, "
                "so table column order would be arbitrary."
            )

        return cls(
            comparison_results=comparison_results,
            fuel_types=list(fuel_types),
            analyzer_type=str(metadata.get("analyzer_type", "unknown")),
            run_config=CrossPerformanceRunConfig.from_mapping(metadata),
            dataset_sizes={k: int(v) for k, v in
                           (metadata.get("dataset_sizes") or {}).items()},
            scale_reference_means=scale_reference_means,
            scale_reference_source=str(metadata.get("scale_reference_source", "")),
            ship_type_name=metadata.get("ship_type_name"),
            conventional_label=str(metadata.get("conventional_label", "conv")),
            created_at=str(metadata.get("created_at", "")),
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # List-column handling
    # ------------------------------------------------------------------

    @staticmethod
    def encode_list_columns(frame: pd.DataFrame,
                            columns: Iterable[str] = LIST_COLUMNS) -> pd.DataFrame:
        """
        Serialise sequence-valued columns to JSON for spreadsheet storage.

        **Concept:**
        A spreadsheet cell holds a scalar. Writing a Python list produces its
        ``repr``, which reads back as an opaque string and silently breaks any
        consumer expecting a sequence. JSON encoding makes the round trip
        explicit and lossless.

        :param frame: Frame to encode. Not modified in place.
        :param columns: Candidate column names to encode.
        :returns: Copy with the named columns JSON-encoded.
        """
        encoded = frame.copy()
        for column in columns:
            if column not in encoded.columns:
                continue
            encoded[column] = encoded[column].map(
                lambda value: json.dumps(list(value))
                if isinstance(value, (list, tuple, np.ndarray))
                else ("" if value is None else json.dumps([str(value)]))
            )
        return encoded

    @staticmethod
    def decode_list_columns(frame: pd.DataFrame,
                            columns: Iterable[str] = LIST_COLUMNS) -> pd.DataFrame:
        """
        Restore JSON-encoded sequence columns to Python lists.

        Falls back to splitting ``predictor_str`` when a cell is unparseable,
        so an externally edited workbook still yields usable predictor lists.

        :param frame: Frame read from a workbook. Not modified in place.
        :param columns: Candidate column names to decode.
        :returns: Copy with the named columns restored to lists.
        """
        decoded = frame.copy()
        for column in columns:
            if column not in decoded.columns:
                continue

            def _parse(value: Any, row_index: int) -> list[str]:
                if isinstance(value, (list, tuple)):
                    return list(value)
                if isinstance(value, str) and value.strip():
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return [str(item) for item in parsed]
                    except json.JSONDecodeError:
                        pass
                if "predictor_str" in decoded.columns:
                    fallback = decoded.at[row_index, "predictor_str"]
                    if isinstance(fallback, str) and fallback.strip():
                        return [part.strip() for part in fallback.split("+")]
                return []

            decoded[column] = [
                _parse(decoded.at[idx, column], idx) for idx in decoded.index
            ]
        return decoded


class CrossPerformanceSummariser:
    """
    Derive per-cohort and per-target summary tables from a result set.

    **Concept:**
    Summaries are aggregations over the results frame plus cohort sizes. Both
    inputs live on the result set, so summarisation depends on no live analyser
    state. This is what allows the summary sheets to be regenerated from a
    workbook read back off disk.

    Composed, not inherited: the summariser holds a result set and reads it.

    :param result_set: Result set to summarise.
    :param debug: Emit stage-by-stage diagnostics.
    """

    def __init__(self, result_set: CrossPerformanceResultSet, debug: bool = False):
        self.result_set = result_set
        self.debug = debug

    def _mean_if_present(self, frame: pd.DataFrame, column: str) -> float:
        """
        Mean of a column, returning NaN when the column is absent.

        :param frame: Frame to reduce.
        :param column: Column name.
        :returns: Column mean or NaN.
        """
        if column not in frame.columns:
            return float("nan")
        return float(pd.to_numeric(frame[column], errors="coerce").mean())

    def by_fuel_type(self) -> pd.DataFrame:
        """
        Summarise mean performance for each fuel cohort.

        :returns: One row per fuel cohort with mean R2 and relative RMSE across
            all targets analysed.
        """
        results = self.result_set.comparison_results
        rows: list[dict[str, Any]] = []

        if "fuel_type" not in results.columns:
            if self.debug:
                print("[Summariser] stage 1: no 'fuel_type' column; empty summary")
            return pd.DataFrame()

        for fuel_type in self.result_set.fuel_types:
            subset = results[results["fuel_type"] == fuel_type]
            if len(subset) == 0:
                if self.debug:
                    print(f"[Summariser] stage 1: no rows for {fuel_type}; skipped")
                continue

            conv_r2 = self._mean_if_present(subset, "conv_r2")
            alt_r2 = self._mean_if_present(subset, "alt_r2")

            rows.append({
                "fuel_type": fuel_type,
                "n_ships": self.result_set.dataset_sizes.get(fuel_type, np.nan),
                "n_targets_analyzed": int(subset["target"].nunique())
                if "target" in subset.columns else len(subset),
                "mean_conv_r2": conv_r2,
                "mean_matched_r2": self._mean_if_present(subset, "matched_r2"),
                "mean_alt_r2": alt_r2,
                "mean_conv_rrmse": self._mean_if_present(subset, "conv_rrmse"),
                "mean_matched_rrmse": self._mean_if_present(subset, "matched_rrmse"),
                "mean_alt_rrmse": self._mean_if_present(subset, "alt_rrmse"),
                "mean_r2_drop_conv_to_alt": conv_r2 - alt_r2,
            })

        if self.debug:
            print(f"[Summariser] stage 2: fuel-type summary built ({len(rows)} rows)")

        return pd.DataFrame(rows)

    def by_target(self) -> pd.DataFrame:
        """
        Summarise dispersion of alternative-cohort performance for each target.

        :returns: One row per target with spread of alternative-trained R2 across
            fuel cohorts, and the best- and worst-performing cohort.
        """
        results = self.result_set.comparison_results
        rows: list[dict[str, Any]] = []

        if "target" not in results.columns or "alt_r2" not in results.columns:
            if self.debug:
                print("[Summariser] stage 1: required columns absent; empty summary")
            return pd.DataFrame()

        for target in self.result_set.targets:
            subset = results[results["target"] == target]
            values = pd.to_numeric(subset["alt_r2"], errors="coerce")
            if values.notna().sum() == 0:
                if self.debug:
                    print(f"[Summariser] stage 1: {target} has no finite alt_r2")
                continue

            rows.append({
                "target": target,
                "n_fuel_types": int(subset["fuel_type"].nunique())
                if "fuel_type" in subset.columns else len(subset),
                "mean_alt_r2": float(values.mean()),
                "std_alt_r2": float(values.std()),
                "min_alt_r2": float(values.min()),
                "max_alt_r2": float(values.max()),
                "best_fuel_type": subset.loc[values.idxmax(), "fuel_type"]
                if "fuel_type" in subset.columns else None,
                "worst_fuel_type": subset.loc[values.idxmin(), "fuel_type"]
                if "fuel_type" in subset.columns else None,
            })

        if self.debug:
            print(f"[Summariser] stage 2: target summary built ({len(rows)} rows)")

        return pd.DataFrame(rows)

    def scale_reference_frame(self) -> pd.DataFrame:
        """
        Tabulate the fixed reference divisors for reporting alongside the results.

        :returns: One row per target with its reference mean and provenance.
        """
        rows = [
            {"target": target,
             "reference_mean": float(mean),
             "source": self.result_set.scale_reference_source}
            for target, mean in sorted(self.result_set.scale_reference_means.items())
        ]
        return pd.DataFrame(rows)