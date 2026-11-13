# cross_performance_workbook.py
"""
Excel persistence for multi-dataset cross-performance result sets.
==================================================================

**Concept:**
Reporting is decoupled from computation by routing both through a file. The
workbook is the durable artefact: it holds the numbers, the cohort ordering, the
evaluation protocol and the reference scale, so a table can be regenerated months
later without re-running the analysis. The filename encodes the discriminating
attributes of the run, which is what permits several validation strategies to
coexist in one directory without overwriting one another.

The writer and reader are separate classes because they have separate failure
modes and separate reasons to change: writing concerns layout and formatting,
reading concerns schema tolerance.
"""



import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from knowledge_layer.multi_dataset_analysers.cross_dataset.cross_performance_config_class import (
    CrossPerformanceResultSet,
    CrossPerformanceSummariser,
)

#: Sheet names, held centrally so writer and reader cannot drift apart.
SHEET_METADATA = "Metadata"
SHEET_RESULTS = "All_Results"
SHEET_FUEL_SUMMARY = "Fuel_Type_Summary"
SHEET_TARGET_SUMMARY = "Target_Summary"
SHEET_SCALE = "Scale_Reference"

#: Separator between filename fields. Chosen so it cannot occur inside a
#: sanitised token, which makes parsing unambiguous.
FIELD_SEPARATOR = "__"


class CrossPerformanceFilenameBuilder:
    """
    Construct and parse filenames that identify a cross-performance run.

    **Concept:**
    When several validation protocols are compared, the filename is the primary
    index over the result files. It must therefore be injective with respect to
    every attribute that distinguishes one run from another: ship type, model
    family, cohorts included, and the full evaluation protocol including its
    parameters. A name reading only ``bootstrap`` collides across replicate
    counts; ``bootstrap632-nb200`` does not.

    Field layout::

        {prefix}__{ship}__{analyzer}__{datasets}__{validation}__{suffix}.xlsx

    :param prefix: Leading token identifying the artefact family.
    :param separator: Field separator. Must not appear in sanitised tokens.
    :param dataset_separator: Separator between cohort names within the dataset
        field.
    :param max_datasets: Cohorts named in full before the field is abbreviated,
        guarding against unusable path lengths.
    """

    def __init__(self,
                 prefix: str = "crossperf",
                 separator: str = FIELD_SEPARATOR,
                 dataset_separator: str = "-",
                 max_datasets: int = 6):
        self.prefix = prefix
        self.separator = separator
        self.dataset_separator = dataset_separator
        self.max_datasets = max_datasets

    @staticmethod
    def sanitise(token: str) -> str:
        """
        Reduce a token to a filesystem-safe, separator-free form.

        Collapses whitespace and punctuation to single hyphens and strips
        underscores, so that the double-underscore field separator remains
        unambiguous.

        :param token: Raw token, for example a fuel cohort name.
        :returns: Sanitised token.
        """
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(token)).strip("-")
        cleaned = cleaned.replace("_", "-")
        return cleaned or "na"

    def build(self,
              result_set: CrossPerformanceResultSet,
              suffix: str | None = None,
              extension: str = ".xlsx",
              debug: bool = False) -> str:
        """
        Build a filename describing a result set.

        :param result_set: Result set to be named.
        :param suffix: Free-text discriminator, for example a revision label.
        :param extension: File extension including the leading dot.
        :param debug: Emit the assembled fields.
        :returns: Filename, without directory.
        """
        ship = self.sanitise(result_set.ship_type_name or "allships")
        analyzer = self.sanitise(result_set.analyzer_type)

        tokens = [self.sanitise(token) for token in result_set.dataset_tokens]
        if len(tokens) > self.max_datasets:
            kept = tokens[: self.max_datasets]
            tokens = kept + [f"plus{len(tokens) - self.max_datasets}"]
        datasets = self.dataset_separator.join(tokens)

        validation = self.sanitise(result_set.run_config.validation_tag)

        fields = [self.prefix, ship, analyzer, datasets, validation]
        if suffix:
            fields.append(self.sanitise(suffix))

        if debug:
            print(f"[Filename] fields: {fields}")

        return self.separator.join(fields) + extension

    def parse(self, filename: str | Path) -> dict[str, Any]:
        """
        Recover the field values encoded in a filename.

        Provided for indexing directories of results; the workbook metadata
        sheet remains the authoritative source and should be preferred where
        the file can be opened.

        :param filename: Filename or path to parse.
        :returns: Mapping of field name to value, with ``'parsed'`` indicating
            whether the layout was recognised.
        """
        stem = Path(filename).stem
        fields = stem.split(self.separator)

        if len(fields) < 5 or fields[0] != self.prefix:
            return {"parsed": False, "stem": stem}

        return {
            "parsed": True,
            "prefix": fields[0],
            "ship_type": fields[1],
            "analyzer_type": fields[2],
            "datasets": fields[3].split(self.dataset_separator),
            "validation_tag": fields[4],
            "suffix": fields[5] if len(fields) > 5 else None,
        }


class CrossPerformanceWorkbookWriter:
    """
    Write a result set to a multi-sheet Excel workbook.

    **Concept:**
    The workbook separates the record of *what was run* from the record of *what
    came out*. The metadata sheet is a key/value table carrying both a
    human-readable rendering and a JSON rendering of each value; the former makes
    the file inspectable by eye, the latter makes the round trip lossless for
    nested values such as the cohort list.

    Composed, not inherited: the writer holds a filename builder and a summariser
    rather than extending either.

    :param filename_builder: Naming policy. A default builder is created when
        omitted.
    :param verbose: Print the path written.
    :param debug: Emit stage-by-stage diagnostics.
    """

    def __init__(self,
                 filename_builder: CrossPerformanceFilenameBuilder | None = None,
                 verbose: bool = True,
                 debug: bool = False):
        self.filename_builder = filename_builder or CrossPerformanceFilenameBuilder()
        self.verbose = verbose
        self.debug = debug

    def write(self,
              result_set: CrossPerformanceResultSet,
              output_dir: str | Path = ".",
              suffix: str | None = None,
              filename: str | None = None,
              overwrite: bool = True) -> Path:
        """
        Serialise a result set to disk.

        :param result_set: Result set to write.
        :param output_dir: Directory to write into. Created when absent.
        :param suffix: Free-text discriminator carried into the filename.
        :param filename: Explicit filename, bypassing the naming policy.
        :param overwrite: When ``False``, an incrementing index is appended
            rather than replacing an existing file.
        :raises ValueError: If the result set holds no rows.
        :returns: Path written.
        """
        if result_set.is_empty:
            raise ValueError(
                "Result set holds no rows; refusing to write an empty workbook."
            )

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = self.filename_builder.build(
                result_set, suffix=suffix, debug=self.debug
            )
        path = directory / filename

        if not overwrite:
            path = self._next_free_path(path)

        if self.debug:
            print(f"[Writer] stage 1: target path {path}")

        metadata_frame = self._build_metadata_frame(result_set)
        results_frame = CrossPerformanceResultSet.encode_list_columns(
            result_set.comparison_results
        )

        if self.debug:
            print(f"[Writer] stage 2: metadata rows {len(metadata_frame)}, "
                  f"results rows {len(results_frame)}")

        summariser = CrossPerformanceSummariser(result_set, debug=self.debug)
        fuel_summary = summariser.by_fuel_type()
        target_summary = summariser.by_target()
        scale_frame = summariser.scale_reference_frame()

        if self.debug:
            print(f"[Writer] stage 3: summaries built "
                  f"(fuel={len(fuel_summary)}, target={len(target_summary)}, "
                  f"scale={len(scale_frame)})")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            metadata_frame.to_excel(writer, sheet_name=SHEET_METADATA, index=False)
            results_frame.to_excel(writer, sheet_name=SHEET_RESULTS, index=False)
            if len(fuel_summary):
                fuel_summary.to_excel(
                    writer, sheet_name=SHEET_FUEL_SUMMARY, index=False
                )
            if len(target_summary):
                target_summary.to_excel(
                    writer, sheet_name=SHEET_TARGET_SUMMARY, index=False
                )
            if len(scale_frame):
                scale_frame.to_excel(writer, sheet_name=SHEET_SCALE, index=False)

            self._apply_column_widths(writer)

        if self.debug:
            print(f"[Writer] stage 4: workbook closed")

        if self.verbose:
            size_kb = path.stat().st_size / 1024
            print(f"Cross-performance workbook written: {path} ({size_kb:.0f} kB)")

        return path

    @staticmethod
    def _next_free_path(path: Path) -> Path:
        """
        Find an unused path by appending an incrementing index.

        :param path: Desired path.
        :returns: Original path when free, otherwise an indexed variant.
        """
        if not path.exists():
            return path
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def _build_metadata_frame(result_set: CrossPerformanceResultSet) -> pd.DataFrame:
        """
        Render the metadata mapping as a three-column key/value table.

        The ``value`` column is for reading; ``value_json`` is authoritative and
        is what the reader consumes.

        :param result_set: Result set supplying the metadata.
        :returns: Metadata frame.
        """
        rows = []
        for key, value in result_set.to_metadata().items():
            if isinstance(value, (list, dict)):
                display = ", ".join(map(str, value)) if isinstance(value, list) \
                    else "; ".join(f"{k}={v}" for k, v in value.items())
            else:
                display = "" if value is None else str(value)
            rows.append({
                "key": key,
                "value": display,
                "value_json": json.dumps(value),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _apply_column_widths(writer: pd.ExcelWriter,
                             minimum: int = 10,
                             maximum: int = 45) -> None:
        """
        Set approximate column widths so the workbook is readable when opened.

        Purely cosmetic; failures are swallowed because layout must never
        prevent the data from being written.

        :param writer: Open Excel writer.
        :param minimum: Lower bound on column width.
        :param maximum: Upper bound on column width.
        """
        try:
            from openpyxl.utils import get_column_letter

            for worksheet in writer.book.worksheets:
                for index, column_cells in enumerate(worksheet.columns, start=1):
                    lengths = [
                        len(str(cell.value)) for cell in column_cells
                        if cell.value is not None
                    ]
                    width = max(lengths) + 2 if lengths else minimum
                    worksheet.column_dimensions[get_column_letter(index)].width = \
                        float(min(max(width, minimum), maximum))
        except Exception:
            pass


class CrossPerformanceWorkbookReader:
    """
    Reconstruct a result set from a workbook written by the corresponding writer.

    **Concept:**
    Reading is deliberately tolerant of extra sheets and extra columns, because
    the workbook is also a human artefact that may accumulate annotations, but
    strict about the fields that determine table semantics. A missing cohort
    ordering is an error rather than a fallback, since silently re-deriving it
    would reorder table columns without warning.

    :param verbose: Print a short description of what was read.
    :param debug: Emit stage-by-stage diagnostics.
    """

    def __init__(self, verbose: bool = True, debug: bool = False):
        self.verbose = verbose
        self.debug = debug

    def read(self, path: str | Path) -> CrossPerformanceResultSet:
        """
        Read a workbook into a result set.

        :param path: Path to the workbook.
        :raises FileNotFoundError: If the path does not exist.
        :raises ValueError: If the required sheets are absent.
        :returns: Reconstructed result set with list columns restored.
        """
        workbook_path = Path(path)
        if not workbook_path.exists():
            raise FileNotFoundError(f"No cross-performance workbook at {workbook_path}")

        sheets = pd.read_excel(workbook_path, sheet_name=None)

        if self.debug:
            print(f"[Reader] stage 1: sheets found {list(sheets)}")

        missing = [name for name in (SHEET_METADATA, SHEET_RESULTS)
                   if name not in sheets]
        if missing:
            raise ValueError(
                f"Workbook {workbook_path.name} lacks required sheet(s) {missing}. "
                f"Present: {list(sheets)}. Was it written by "
                f"CrossPerformanceWorkbookWriter?"
            )

        metadata = self._parse_metadata(sheets[SHEET_METADATA])

        if self.debug:
            print(f"[Reader] stage 2: metadata keys {sorted(metadata)}")

        results = CrossPerformanceResultSet.decode_list_columns(sheets[SHEET_RESULTS])

        if self.debug:
            print(f"[Reader] stage 3: results decoded "
                  f"({len(results)} rows, {results.shape[1]} columns)")

        means: dict[str, float] = {}
        if SHEET_SCALE in sheets:
            scale_frame = sheets[SHEET_SCALE]
            if {"target", "reference_mean"}.issubset(scale_frame.columns):
                means = {
                    str(row["target"]): float(row["reference_mean"])
                    for _, row in scale_frame.iterrows()
                    if np.isfinite(row["reference_mean"])
                }

        result_set = CrossPerformanceResultSet.from_parts(
            comparison_results=results,
            metadata=metadata,
            scale_reference_means=means,
            source_path=workbook_path,
        )

        if self.debug:
            print(f"[Reader] stage 4: result set assembled")

        if self.verbose:
            print(f"Loaded {len(results)} results from {workbook_path.name}")
            print(f"   Model family:  {result_set.analyzer_type}")
            print(f"   Cohorts:       {', '.join(result_set.fuel_types)}")
            print(f"   Validation:    {result_set.run_config.validation_tag}")

        return result_set

    @staticmethod
    def _parse_metadata(frame: pd.DataFrame) -> dict[str, Any]:
        """
        Convert the key/value metadata sheet back into a mapping.

        Prefers the JSON column so that nested values and numeric types survive;
        falls back to the display column when JSON is absent or malformed.

        :param frame: Metadata sheet.
        :raises ValueError: If the sheet lacks a ``key`` column.
        :returns: Metadata mapping.
        """
        if "key" not in frame.columns:
            raise ValueError(
                "Metadata sheet lacks a 'key' column; cannot reconstruct context."
            )

        metadata: dict[str, Any] = {}
        for _, row in frame.iterrows():
            key = str(row["key"])
            raw = row.get("value_json")
            if isinstance(raw, str) and raw.strip():
                try:
                    metadata[key] = json.loads(raw)
                    continue
                except json.JSONDecodeError:
                    pass
            value = row.get("value")
            metadata[key] = None if (isinstance(value, float) and np.isnan(value)) \
                else value
        return metadata