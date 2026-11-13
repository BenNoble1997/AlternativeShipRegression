"""
sister_ship_trace.py

Reconstruct which vessels were removed at each sister-ship filtering step,
and which retained vessel each removed vessel was treated as a sister of,
from snapshots of the dataset before and after each step.
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

INDEX_NAME = "SHIP_ID"


@dataclass(frozen=True)
class FilterStep:
    """Declarative description of one sister-ship filtering step.

    Two vessels are treated as sisters by this step when all
    :attr:`key_columns` match, after optional rounding.

    :param name: Short identifier used in output tables.
    :param key_columns: Columns that must match for two vessels to be sisters.
    :param round_spec: Optional mapping of column name to number of decimals
        applied before matching. Negative values round to tens, hundreds, etc.
    """

    name: str
    key_columns: tuple[str, ...]
    round_spec: Mapping[str, int] = field(default_factory=dict)


class ShipIdNormaliser:
    """Normalise ship identifiers and re-index frames by them.

    Identifiers are converted to strings with whitespace and a trailing
    ``.0`` removed. This makes MMSI values comparable when one snapshot
    stores them as integers, another as floats (e.g. after a CSV round trip
    with missing values) and another as strings.

    :param id_column: Name of the identifier column, e.g. ``"MMSI"``.
    """

    def __init__(self, id_column: str = "MMSI") -> None:
        self.id_column = id_column

    def normalise(self, ids: pd.Series) -> pd.Series:
        """Return normalised string identifiers; empty values become ``NA``.

        :param ids: Raw identifier values.
        :return: Normalised identifiers.
        """
        out = ids.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
        return out.mask(out == "")

    def index_frame(self, df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict[str, object]]:
        """Index a frame by normalised identifier.

        Rows without an identifier cannot be traced and are dropped; their
        number is reported.

        :param df: Snapshot to index.
        :param name: Snapshot name, used in messages and the summary.
        :return: Tuple of the re-indexed frame and a summary record.
        :raises KeyError: If the identifier column is absent.
        :raises ValueError: If any identifier occurs more than once.
        """
        if self.id_column not in df.columns:
            raise KeyError(f"{name}: column '{self.id_column}' not found.")
        ids = self.normalise(df[self.id_column])
        present = ids.notna()
        duplicated = ids[present & ids.duplicated(keep=False)]
        if not duplicated.empty:
            examples = list(duplicated.unique()[:5])
            raise ValueError(f"{name}: {duplicated.nunique()} duplicated {self.id_column} values, "
                             f"e.g. {examples}. Tracing requires unique identifiers.")
        out = df.loc[present.to_numpy()].copy()
        out.index = pd.Index(ids[present].to_numpy(), name=INDEX_NAME)
        summary = {"snapshot": name, "rows": len(df), "rows_without_id": int((~present).sum())}
        return out, summary


class RemovalMatcher:
    """Match vessels removed by one filter step to the retained vessels sharing their key.

    Key values for both removed and retained vessels are taken from the
    *before* snapshot, so the comparison does not depend on how the *after*
    snapshot was stored. Columns in ``protect_columns`` that exist in the
    data are added to the key, mirroring the rule that vessels with a
    different fuel configuration are never sisters. Missing values in
    protect columns are treated as their own category; missing values in
    the step's key columns prevent matching.

    :param step: Filter step to reconstruct.
    :param protect_columns: Columns that must also match, e.g. fuel cohort,
        main engine fuel and bunker fuel.
    """

    def __init__(self, step: FilterStep, protect_columns: Sequence[str] = ("FUEL_COHORT",)) -> None:
        self.step = step
        self.protect_columns = tuple(protect_columns)

    def _keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the matching-key frame.

        :param df: Snapshot indexed by ship identifier.
        :return: Key frame aligned to ``df``.
        :raises KeyError: If a key column is absent.
        """
        missing = [c for c in self.step.key_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Step '{self.step.name}': key columns not found: {missing}")
        keys = df[list(self.step.key_columns)].copy()
        for col, decimals in self.step.round_spec.items():
            keys[col] = pd.to_numeric(keys[col], errors="coerce").round(decimals)
        for col in self.protect_columns:
            if col in df.columns:
                keys[f"{col}__protect"] = df[col].astype("object").where(df[col].notna(), "<missing>")
        return keys

    def match(self, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        """Identify removed vessels and their retained sister(s).

        :param before: Snapshot before the step, indexed by ship identifier.
        :param after: Snapshot after the step, indexed by ship identifier.
        :return: One row per removed vessel with columns ``ship_id``,
            ``step``, ``kept_id`` (set only for a unique match),
            ``n_candidates``, ``candidate_ids`` (semicolon-separated) and
            ``match_status`` (``unique``, ``ambiguous``, ``unmatched`` or
            ``missing_key``).
        """
        removed_ids = before.index.difference(after.index)
        kept_ids = after.index.intersection(before.index)
        keys = self._keys(before)
        complete = keys[list(self.step.key_columns)].notna().all(axis=1)

        removed_keys = keys.loc[removed_ids][complete.loc[removed_ids]]
        kept_keys = keys.loc[kept_ids][complete.loc[kept_ids]]
        pairs = (removed_keys.rename_axis("removed_id").reset_index()
                 .merge(kept_keys.rename_axis("kept_id").reset_index(), on=list(keys.columns)))

        candidates: dict[str, list[str]] = {rid: [] for rid in removed_ids}
        for rid, kid in pairs[["removed_id", "kept_id"]].itertuples(index=False):
            candidates[rid].append(kid)

        rows = []
        for rid in removed_ids:
            found = sorted(candidates[rid])
            if not complete.loc[rid]:
                status = "missing_key"
            elif len(found) == 1:
                status = "unique"
            elif not found:
                status = "unmatched"
            else:
                status = "ambiguous"
            rows.append({"ship_id": rid, "step": self.step.name,
                         "kept_id": found[0] if status == "unique" else pd.NA,
                         "n_candidates": len(found), "candidate_ids": ";".join(found),
                         "match_status": status})
        columns = ["ship_id", "step", "kept_id", "n_candidates", "candidate_ids", "match_status"]
        return pd.DataFrame(rows, columns=columns)


@dataclass
class SisterTraceResult:
    """Output of :meth:`SisterShipTracer.trace`.

    :param removals: One row per removed vessel and step, with match details
        and ``final_kept_id``, the vessel representing it in the final
        snapshot.
    :param groups: One row per final representative that absorbed at least
        one sister, with removal counts per step and member identifiers.
    :param step_summary: Counts per step: before, after, removed, removals
        per match status and retained vessels absent from the before
        snapshot.
    :param id_summary: Rows and rows without an identifier per snapshot.
    :param frames: Snapshots indexed by normalised identifier, keyed by
        snapshot name.
    :param step_names: Step names in application order.
    """

    removals: pd.DataFrame
    groups: pd.DataFrame
    step_summary: pd.DataFrame
    id_summary: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    step_names: tuple[str, ...]

    def membership(self, step: str | None = None) -> pd.DataFrame:
        """Long table of group membership.

        With ``step=None`` groups are the final groups, keyed by
        ``final_kept_id``, and include sisters removed at every step. With a
        step name, groups are those formed by that step alone, keyed by
        ``kept_id``.

        :param step: Step name, or ``None`` for final groups.
        :return: Columns ``group_id``, ``member_id`` and ``role``
            (``kept`` or ``removed_<step>``).
        :raises ValueError: If ``step`` is not a known step name.
        """
        if step is not None and step not in self.step_names:
            raise ValueError(f"Unknown step '{step}'. Known: {self.step_names}")
        group_col = "final_kept_id" if step is None else "kept_id"
        rem = self.removals if step is None else self.removals[self.removals["step"] == step]
        rem = rem[rem[group_col].notna()]
        removed = pd.DataFrame({"group_id": rem[group_col], "member_id": rem["ship_id"],
                                "role": "removed_" + rem["step"]})
        kept_ids = rem[group_col].unique()
        kept = pd.DataFrame({"group_id": kept_ids, "member_id": kept_ids, "role": "kept"})
        return pd.concat([kept, removed], ignore_index=True).sort_values(["group_id", "role"], ignore_index=True)

    def to_excel(self, path: str | Path) -> Path:
        """Write all tables to one workbook.

        :param path: Output ``.xlsx`` path.
        :return: The path written.
        """
        path = Path(path)
        with pd.ExcelWriter(path) as writer:
            self.step_summary.to_excel(writer, sheet_name="step_summary", index=False)
            self.id_summary.to_excel(writer, sheet_name="id_summary", index=False)
            self.removals.to_excel(writer, sheet_name="removals", index=False)
            self.groups.to_excel(writer, sheet_name="groups", index=False)
        return path


class SisterShipTracer:
    """Reconstruct sister-ship removals from successive dataset snapshots.

    Composes a :class:`ShipIdNormaliser` and one :class:`RemovalMatcher` per
    step. Given ``n`` steps it expects ``n + 1`` snapshots: the dataset
    before any sister-ship removal followed by the dataset after each step.

    :param steps: Filter steps in application order.
    :param normaliser: Identifier normaliser.
    :param protect_columns: Columns that must match for any sister link.
    :raises ValueError: If ``steps`` is empty.
    """

    def __init__(
        self,
        steps: Sequence[FilterStep],
        normaliser: ShipIdNormaliser,
        protect_columns: Sequence[str] = ("FUEL_COHORT",),
    ) -> None:
        if not steps:
            raise ValueError("At least one FilterStep is required.")
        self.steps = tuple(steps)
        self.normaliser = normaliser
        self.matchers = tuple(RemovalMatcher(s, protect_columns) for s in self.steps)

    def _resolve_final(self, removals: pd.DataFrame) -> pd.Series:
        """Follow kept-by links forward to the vessel in the final snapshot.

        After passing through step ``j``'s mapping, an identifier is in the
        snapshot after step ``j`` and cannot be removed by an earlier step,
        so one pass over the steps in order is sufficient.

        :param removals: Concatenated matcher outputs.
        :return: Final representative per row; ``NA`` if any link in the
            chain is not unique.
        """
        final = removals["kept_id"].copy()
        for step in self.steps:
            mapping = removals.loc[removals["step"] == step.name].set_index("ship_id")["kept_id"]
            hit = final.isin(mapping.index)
            final.loc[hit] = final.loc[hit].map(mapping)
        return final

    def _groups(self, removals: pd.DataFrame) -> pd.DataFrame:
        """Summarise final groups.

        :param removals: Removals with ``final_kept_id``.
        :return: One row per final representative.
        """
        resolved = removals[removals["final_kept_id"].notna()]
        if resolved.empty:
            return pd.DataFrame(columns=["final_kept_id", "group_size", "removed_ids"])
        counts = (resolved.pivot_table(index="final_kept_id", columns="step", values="ship_id",
                                       aggfunc="size", fill_value=0)
                  .reindex(columns=[s.name for s in self.steps], fill_value=0)
                  .add_prefix("n_removed_"))
        counts["group_size"] = counts.sum(axis=1) + 1
        counts["removed_ids"] = resolved.groupby("final_kept_id")["ship_id"].agg(lambda s: ";".join(sorted(s)))
        return counts.sort_values("group_size", ascending=False).reset_index()

    def trace(self, snapshots: Sequence[pd.DataFrame], debug: bool = False) -> SisterTraceResult:
        """Trace removals through all steps.

        :param snapshots: ``len(steps) + 1`` frames, before any removal and
            after each step, each containing the identifier column.
        :param debug: Print progress and counts at each stage.
        :return: Trace result.
        :raises ValueError: If the number of snapshots is wrong.
        """
        if len(snapshots) != len(self.steps) + 1:
            raise ValueError(f"Expected {len(self.steps) + 1} snapshots, got {len(snapshots)}.")

        # Stage 1: normalise identifiers
        names = ["before_sister_removal"] + [f"after_{s.name}" for s in self.steps]
        indexed, id_rows = [], []
        for name, df in zip(names, snapshots):
            frame, summary = self.normaliser.index_frame(df, name)
            indexed.append(frame)
            id_rows.append(summary)
            if debug:
                print(f"[trace] stage 1: {name}: {len(df)} rows, {summary['rows_without_id']} without id")

        # Stage 2: match removals per step
        parts, step_rows = [], []
        for matcher, before, after in zip(self.matchers, indexed[:-1], indexed[1:]):
            matched = matcher.match(before, after)
            parts.append(matched)
            status = matched["match_status"].value_counts()
            not_in_before = after.index.difference(before.index).size
            step_rows.append({"step": matcher.step.name, "n_before": len(before), "n_after": len(after),
                              "n_removed": len(matched), "pct_removed": len(matched) / len(before),
                              **{f"n_{s}": int(status.get(s, 0))
                                 for s in ("unique", "ambiguous", "unmatched", "missing_key")},
                              "n_after_not_in_before": not_in_before})
            if debug:
                print(f"[trace] stage 2: {matcher.step.name}: removed {len(matched)} "
                      f"of {len(before)}; statuses {status.to_dict()}")
                if not_in_before:
                    print(f"[trace]   warning: {not_in_before} retained ships absent from the before snapshot")

        # Stage 3: resolve final representatives
        removals = pd.concat(parts, ignore_index=True)
        removals["final_kept_id"] = self._resolve_final(removals)
        removals["kept_removed_later"] = removals["kept_id"].notna() & (removals["kept_id"] != removals["final_kept_id"])
        if debug:
            n_unresolved = int(removals["final_kept_id"].isna().sum())
            n_relinked = int(removals["kept_removed_later"].sum())
            print(f"[trace] stage 3: {n_relinked} removals re-linked through a later step, "
                  f"{n_unresolved} without a final representative")

        # Stage 4: group summary
        groups = self._groups(removals)
        if debug:
            largest = int(groups["group_size"].max()) if not groups.empty else 0
            print(f"[trace] stage 4: {len(groups)} final groups, largest has {largest} ships")

        return SisterTraceResult(removals=removals, groups=groups,
                                 step_summary=pd.DataFrame(step_rows), id_summary=pd.DataFrame(id_rows),
                                 frames=dict(zip(names, indexed)), step_names=tuple(s.name for s in self.steps))


class SisterGroupSpread:
    """Within-group relative spread of dimensions for traced sister groups.

    The relative spread of a dimension is ``(max - min) / median`` over the
    group's members. ``max_spread`` is the largest relative spread across
    dimensions, i.e. the tolerance within which the group agrees on every
    dimension.

    :param dimension_columns: Dimensions to assess.
    """

    def __init__(self, dimension_columns: Sequence[str] = ("LOA (M)", "LBP (M)", "BEAM (M)", "DRAFT (M)", "DEPTH (M)")) -> None:
        self.dimension_columns = tuple(dimension_columns)

    def compute(self, frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
        """Compute spreads per group.

        :param frame: Snapshot indexed by ship identifier containing all
            group members, e.g. ``result.frames["before_sister_removal"]``.
        :param membership: Output of :meth:`SisterTraceResult.membership`.
        :return: One row per group with spreads per dimension,
            ``group_size`` and ``max_spread``.
        """
        cols = [c for c in self.dimension_columns if c in frame.columns]
        members = membership[membership["member_id"].isin(frame.index)]
        values = frame.loc[members["member_id"], cols].apply(pd.to_numeric, errors="coerce")
        values["group_id"] = members["group_id"].to_numpy()
        grouped = values.groupby("group_id")
        spread = (grouped[cols].max() - grouped[cols].min()) / grouped[cols].median()
        spread["group_size"] = grouped.size()
        spread["max_spread"] = spread[cols].max(axis=1)
        return spread.reset_index()


# ---------------------------------------------------------------------------
# Script layer
# ---------------------------------------------------------------------------

ID_COLUMN = "MMSI"
PROTECT_COLUMNS = ("MAIN_FUEL_1",)
STEPS = (
    FilterStep("hull_dimensions", ("LOA (M)", "LBP (M)", "BEAM (M)", "DRAFT (M)", "DEPTH (M)")),
    FilterStep("company_builder_mcr", ("MCR (KW)", "COMPANY", "BUILDER")),
)


def trace_ship_type(
    raw: pd.DataFrame,
    after_hull: pd.DataFrame,
    after_cbm: pd.DataFrame,
    debug: bool = False,
) -> SisterTraceResult:
    """Trace sister-ship removal for one ship type.

    :param raw: Dataset after missing-data removal, before sister removal.
    :param after_hull: Dataset after the hull-dimension filter.
    :param after_cbm: Dataset after the company, builder and MCR filter.
    :param debug: Print progress at each stage.
    :return: Trace result.
    """
    tracer = SisterShipTracer(STEPS, ShipIdNormaliser(ID_COLUMN), PROTECT_COLUMNS)
    return tracer.trace([raw, after_hull, after_cbm], debug=debug)


if __name__ == "__main__":

    from pathlib import Path
    source_dir =Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\processing_checks")

    save_dir = Path(r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\data_processing_scripts\sister_ships")
    save_dir.mkdir(parents=True, exist_ok=True)

    sister_ship_analysis = {}

    ship_types = ["Bulk carrier", "Tanker", "Container ship", "RoPax"]

    for ship_type in ship_types:

        print("==================================================")
        print(ship_type)
        print("==================================================")

        target_dir = source_dir / ship_type / "preprocessing.xlsx"

        dfs = pd.read_excel(target_dir, sheet_name=None)

        print(dfs.keys())

        raw = dfs["missing_data"]
        after_hull = dfs["sister_ship_1"]
        after_cbm = dfs["sister_ship_2"]

        removed_cbm = after_hull[after_hull["MMSI"].isin(after_cbm["MMSI"])]

        duplicate_cols = ['MCR (KW)', 'COMPANY', 'BUILDER', 'MAIN_FUEL_1']

        # Find the duplicate groups in the original dataset
        duplicate_groups = (
            after_hull.groupby(duplicate_cols, dropna=False)
            .filter(lambda x: len(x) > 1)
        )

        # Get the ships that were retained in df_b
        kept_ids = set(after_cbm["MMSI"])

        # Add whether each ship was the one retained
        duplicate_groups = duplicate_groups.copy()

        duplicate_groups["kept"] = duplicate_groups["MMSI"].isin(kept_ids)

    # raw = pd.read_excel("bulk_raw.xlsx")              # replace with your loaders
    # after_hull = pd.read_excel("bulk_after_hull.xlsx")
    # after_cbm = pd.read_excel("bulk_after_cbm.xlsx")

        result = trace_ship_type(raw, after_hull, after_cbm, debug=True)

        sister_ship_analysis[ship_type] = {
            "step_summary": result.step_summary,
            "groups": result.groups,
            "duplicate_groups": duplicate_groups,
            "after_hull_dimensions": result.frames["after_hull_dimensions"],

        }

    with pd.ExcelWriter(
            save_dir / "sister_ship_analysis_depth.xlsx",
            engine="openpyxl"
    ) as writer:

        for ship_type, dataframes in sister_ship_analysis.items():

            for name, df in dataframes.items():
                sheet_name = f"{ship_type}_{name}"[:31]

                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )

