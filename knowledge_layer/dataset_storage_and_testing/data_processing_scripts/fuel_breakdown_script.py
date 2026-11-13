




"""
Fuel usage summary across ship types.

Counts, for every ship type, how many vessels use each fuel after the standard
pre-processing chain has been applied.

Fuel identification is delegated entirely to :meth:`FuelType.from_identifier`,
so the rows of the output table are exactly the members of :class:`FuelType`
and nothing else. Any raw string the enumeration does not recognise is reported
separately rather than becoming a new category.

Counting is done at the level of *ships* rather than *cells*: a vessel with a
dual-fuel main engine and an LNG bunker entry appears once under METHANE, not
three times.

Outputs a multi-sheet workbook:

* ``matrix_any``            fuel x ship type, counts over all fuel roles combined
* ``matrix_any_pct``        the same matrix as a percentage of each fleet
* ``long``                  tidy long-format table (ship type, fuel, role, count)
* ``by_role_<shiptype>``    fuel x role breakdown for one ship type
* ``raw_values``            every raw fuel string encountered, with its classification
* ``unclassified``          raw strings FuelType.from_identifier returned None for
"""


import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
import re

from equipment.fuel_classes.Fuel_class import FuelType
from parameters.shipshape.ShipShapeBasic_class import ShipType

# Adjust this import to wherever load_fleet_data / pre_processing / get_variables live.
from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import get_variables, pre_processing

logger = logging.getLogger(__name__)

#: Prefixes of the split fuel columns produced by ``processing_fleet_data``.
#: ``MAIN_FUEL`` is split into ``MAIN_FUEL_1``, ``MAIN_FUEL_2`` and so on, so the
#: columns are discovered by prefix rather than hard-coded.
DEFAULT_ROLE_PREFIXES: dict[str, str] = {
    'MAIN': 'MAIN_FUEL',
    'AUX': 'AUX_FUEL',
    'BUNKER': 'BUNKER_FUEL',
}

#: Row order of the output tables. Taken directly from the enumeration, so the
#: table can never contain a fuel the modelling code does not know about.
FUEL_ORDER: list[str] = [fuel_type.name for fuel_type in FuelType]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=4096)
def classify_value(value: str) -> str | None:
    """
    Resolve one raw fuel string to a :class:`FuelType` member name.

    The call is cached because the number of distinct fuel strings in a fleet
    database is small (tens) while the number of cells is large (tens of
    thousands), and ``from_identifier`` performs regex extraction and a linear
    scan over its alias mapping on every call.

    :param value: Raw fuel string, already stripped of surrounding whitespace.
    :type value: str
    :returns: Name of the matching :class:`FuelType` member, or ``None``.
    :rtype: str or None
    """
    fuel_type = FuelType.from_identifier(name=value)
    return fuel_type.name if fuel_type is not None else None


def normalise_series(series: pd.Series) -> pd.Series:
    """
    Strip a fuel column and convert blank placeholders to missing values.

    Case is deliberately left alone: ``from_identifier`` lower-cases internally,
    and its formula extraction is case-sensitive by design.

    :param series: Raw fuel column.
    :type series: pandas.Series
    :returns: Cleaned string series.
    :rtype: pandas.Series
    """
    cleaned = series.astype('string').str.strip()
    return cleaned.replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA,
                            'None': pd.NA, 'N/A': pd.NA, 'n/a': pd.NA})


def classify_column(series: pd.Series) -> pd.Series:
    """
    Map a raw fuel column to :class:`FuelType` member names.

    Classification is applied to the unique values only and then mapped back,
    which keeps the cost proportional to the vocabulary rather than to the
    number of rows.

    :param series: Raw fuel column.
    :type series: pandas.Series
    :returns: Series of member names, with ``pandas.NA`` where unrecognised.
    :rtype: pandas.Series
    """
    cleaned = normalise_series(series)
    lookup = {value: classify_value(value) for value in cleaned.dropna().unique()}
    return cleaned.map(lookup).astype('string')


# --------------------------------------------------------------------------- #
# Column discovery
# --------------------------------------------------------------------------- #

def discover_fuel_columns(dataframe: pd.DataFrame,
                          role_prefixes: dict[str, str]) -> dict[str, list[str]]:
    """
    Map each fuel role to the split value columns actually present.

    The split step creates a variable number of columns depending on how many
    fuels the most complex vessel declares, so the column set is resolved at
    runtime instead of being assumed. It also creates metadata columns beside
    them, notably ``<PREFIX>_count``, which hold integers rather than fuel
    names. Matching is therefore anchored: a column qualifies only if it is the
    bare prefix or the prefix followed by an underscore and an integer index.

    :param dataframe: Processed fleet dataframe with split fuel columns.
    :type dataframe: pandas.DataFrame
    :param role_prefixes: Mapping of role label to column-name prefix.
    :type role_prefixes: dict[str, str]
    :returns: Mapping of role label to the list of matching column names,
        ordered by split index.
    :rtype: dict[str, list[str]]
    """
    discovered: dict[str, list[str]] = {}

    for role, prefix in role_prefixes.items():
        pattern = re.compile(rf'^{re.escape(prefix.upper())}(_\d+)?$')
        columns = [column for column in dataframe.columns
                   if isinstance(column, str) and pattern.match(column.upper())]

        if columns:
            discovered[role] = sorted(columns)
        else:
            logger.warning("No columns found for fuel role '%s' (prefix '%s')", role, prefix)

        skipped = [column for column in dataframe.columns
                   if isinstance(column, str)
                   and column.upper().startswith(prefix.upper())
                   and column not in columns]
        if skipped:
            logger.debug("Role '%s': ignoring non-value columns %s", role, skipped)

    return discovered


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def count_fuels_for_fleet(dataframe: pd.DataFrame,
                          fuel_columns: dict[str, list[str]]) -> pd.DataFrame:
    """
    Count ships per :class:`FuelType`, split by fuel role and combined over roles.

    For each fuel and role a boolean mask over vessels is built and combined
    with a logical OR before summing, so a vessel declaring the same fuel in
    several columns contributes at most one to that fuel. Row totals therefore
    exceed the fleet size whenever dual-fuel vessels are present, which is
    correct: the categories overlap rather than partition the fleet.

    :param dataframe: Processed fleet dataframe for one ship type.
    :type dataframe: pandas.DataFrame
    :param fuel_columns: Role label to the columns belonging to that role.
    :type fuel_columns: dict[str, list[str]]
    :returns: Frame indexed by :class:`FuelType` name, with one column per role
        plus ``ANY``, ``FLEET_SIZE`` and ``PCT_OF_FLEET``.
    :rtype: pandas.DataFrame
    """
    fleet_size = len(dataframe)

    classified: dict[str, pd.DataFrame] = {
        role: pd.DataFrame({column: classify_column(dataframe[column])
                            for column in columns})
        for role, columns in fuel_columns.items()
    }

    records: dict[str, dict[str, int]] = {}

    for fuel_name in FUEL_ORDER:
        row: dict[str, int] = {}
        any_mask = pd.Series(False, index=dataframe.index)

        for role, frame in classified.items():
            role_mask = (frame == fuel_name).any(axis=1)
            row[role] = int(role_mask.sum())
            any_mask |= role_mask

        row['ANY'] = int(any_mask.sum())
        records[fuel_name] = row

    counts = pd.DataFrame.from_dict(records, orient='index')
    counts.index.name = 'FUEL'
    counts['FLEET_SIZE'] = fleet_size
    counts['PCT_OF_FLEET'] = (counts['ANY'] / fleet_size * 100).round(2) if fleet_size else 0.0

    return counts


def count_raw_values(dataframe: pd.DataFrame,
                     fuel_columns: dict[str, list[str]]) -> pd.DataFrame:
    """
    Count the frequency of every distinct raw fuel string and its classification.

    This is the audit trail for the counts. A string appearing here with an
    empty ``FUEL_TYPE`` is one the enumeration does not recognise, and is
    therefore invisible in the summary tables.

    :param dataframe: Processed fleet dataframe.
    :type dataframe: pandas.DataFrame
    :param fuel_columns: Role label to the columns belonging to that role.
    :type fuel_columns: dict[str, list[str]]
    :returns: Frame with columns ``ROLE``, ``COLUMN``, ``VALUE``, ``FUEL_TYPE``,
        ``COUNT``.
    :rtype: pandas.DataFrame
    """
    rows: list[dict[str, object]] = []

    for role, columns in fuel_columns.items():
        for column in columns:
            values = normalise_series(dataframe[column]).dropna()
            for value, count in Counter(values).items():
                rows.append({'ROLE': role,
                             'COLUMN': column,
                             'VALUE': value,
                             'FUEL_TYPE': classify_value(value),
                             'COUNT': int(count)})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(['ROLE', 'COLUMN', 'COUNT'], ascending=[True, True, False])

    return frame.reset_index(drop=True)


def summarise_unclassified(raw_values: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unrecognised raw strings across columns.

    The per-column detail in ``raw_values`` is the wrong granularity for a
    console warning, since the same unrecognised string typically appears in
    every column of a role. Collapsing to one row per distinct string, with the
    columns it occurred in, gives a list short enough to read.

    :param raw_values: Output of :func:`count_raw_values`.
    :type raw_values: pandas.DataFrame
    :returns: Frame with columns ``VALUE``, ``COUNT`` and ``COLUMNS``, sorted by
        descending count. Empty if everything classified.
    :rtype: pandas.DataFrame
    """
    if raw_values.empty:
        return raw_values

    unclassified = raw_values[raw_values['FUEL_TYPE'].isna()]
    if unclassified.empty:
        return unclassified

    grouped = (unclassified
               .groupby('VALUE')
               .agg(COUNT=('COUNT', 'sum'),
                    COLUMNS=('COLUMN', lambda values: ', '.join(sorted(set(values)))))
               .reset_index()
               .sort_values('COUNT', ascending=False))

    return grouped.reset_index(drop=True)


def report_unclassified(raw_values: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Log every distinct raw fuel string that :class:`FuelType` did not recognise.

    :param raw_values: Output of :func:`count_raw_values`.
    :type raw_values: pandas.DataFrame
    :param label: Ship type label used in the log messages.
    :type label: str
    :returns: The aggregated frame, for onward use by the caller.
    :rtype: pandas.DataFrame
    """
    grouped = summarise_unclassified(raw_values)

    if grouped.empty:
        logger.info("%s: all raw fuel strings classified", label)
        return grouped

    logger.warning("%s: %d unrecognised fuel strings (%d cells)",
                   label, len(grouped), int(grouped['COUNT'].sum()))

    for record in grouped.itertuples(index=False):
        logger.warning("%s:   %r  (%d cells, in %s)",
                       label, record.VALUE, record.COUNT, record.COLUMNS)

    return grouped


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def summarise_shiptype(shiptype: ShipType,
                       max_build_year: int = 1990,
                       dataset_name: str = 'base_with_split_fuels',
                       role_prefixes: dict[str, str] | None = None,
                       sister_ships: bool = True,
                       incomplete_data: bool = True,
                       outliers: bool = True,
                       outlier_method: str = 'percentile',
                       percentile: float = 0.95,
                       iqr_window: float = 1.5) -> dict[str, pd.DataFrame]:
    """
    Run pre-processing for one ship type and count fuel usage in the result.

    :param shiptype: Ship type to process.
    :type shiptype: ShipType
    :param max_build_year: Minimum build year retained by the date filter.
    :type max_build_year: int
    :param dataset_name: Named dataset within :class:`FleetData` to count. The
        default is the fully pre-processed base set with fuels split out.
    :type dataset_name: str
    :param role_prefixes: Role label to split-column prefix.
    :type role_prefixes: dict[str, str] or None
    :param sister_ships: Whether to apply sister-ship deduplication.
    :type sister_ships: bool
    :param incomplete_data: Whether to drop records with missing key dimensions.
    :type incomplete_data: bool
    :param outliers: Whether to apply outlier removal.
    :type outliers: bool
    :param outlier_method: ``'percentile'`` or ``'iqr'``.
    :type outlier_method: str
    :param percentile: Central fraction retained when using the percentile method.
    :type percentile: float
    :param iqr_window: Tukey fence multiplier when using the IQR method.
    :type iqr_window: float
    :returns: Mapping with keys ``counts`` and ``raw_values``.
    :rtype: dict[str, pandas.DataFrame]
    """
    role_prefixes = role_prefixes or DEFAULT_ROLE_PREFIXES

    _, _, _, filepath, _ = get_variables(shiptype)

    fleet = pre_processing(filepath=filepath,
                           shiptype=shiptype.string,
                           sister_ships=sister_ships,
                           incomplete_data=incomplete_data,
                           outliers=outliers,
                           max_build_year=max_build_year,
                           outlier_method=outlier_method,
                           iqr_window=iqr_window,
                           percentile=percentile)

    dataframe = fleet.processed_data[dataset_name]
    logger.info("%s: %d ships in '%s'", shiptype.string, len(dataframe), dataset_name)

    fuel_columns = discover_fuel_columns(dataframe, role_prefixes)
    logger.info("%s: fuel columns %s", shiptype.string, fuel_columns)

    counts = count_fuels_for_fleet(dataframe, fuel_columns)
    raw_values = count_raw_values(dataframe, fuel_columns)

    report_unclassified(raw_values, shiptype.string)

    return {'counts': counts, 'raw_values': raw_values}


def build_matrix(summaries: dict[str, dict[str, pd.DataFrame]],
                 column: str = 'ANY') -> pd.DataFrame:
    """
    Assemble a fuel-by-shiptype matrix from the per-ship-type summaries.

    :param summaries: Ship type label to the output of :func:`summarise_shiptype`.
    :type summaries: dict[str, dict[str, pandas.DataFrame]]
    :param column: Which count column to pivot on.
    :type column: str
    :returns: Frame indexed by fuel with one column per ship type.
    :rtype: pandas.DataFrame
    """
    matrix = pd.DataFrame({label: summary['counts'][column]
                           for label, summary in summaries.items()})
    matrix.index.name = 'FUEL'

    fleet_sizes = {label: summary['counts']['FLEET_SIZE'].iloc[0]
                   for label, summary in summaries.items()}
    matrix.loc['FLEET SIZE'] = pd.Series(fleet_sizes)

    return matrix


def build_long_table(summaries: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Flatten the per-ship-type counts into a tidy long-format table.

    Long format is the convenient form for pivot tables and for plotting, since
    every observation is one row rather than one cell in a wide grid.

    :param summaries: Ship type label to the output of :func:`summarise_shiptype`.
    :type summaries: dict[str, dict[str, pandas.DataFrame]]
    :returns: Frame with columns ``SHIP_TYPE``, ``FUEL``, ``ROLE``, ``N_SHIPS``,
        ``FLEET_SIZE`` and ``PCT_OF_FLEET``.
    :rtype: pandas.DataFrame
    """
    rows: list[dict[str, object]] = []

    for label, summary in summaries.items():
        counts = summary['counts']
        fleet_size = int(counts['FLEET_SIZE'].iloc[0])
        role_columns = [column for column in counts.columns
                        if column not in ('FLEET_SIZE', 'PCT_OF_FLEET')]

        for fuel, row in counts.iterrows():
            for role in role_columns:
                n_ships = int(row[role])
                rows.append({
                    'SHIP_TYPE': label,
                    'FUEL': fuel,
                    'ROLE': role,
                    'N_SHIPS': n_ships,
                    'FLEET_SIZE': fleet_size,
                    'PCT_OF_FLEET': round(n_ships / fleet_size * 100, 2) if fleet_size else 0.0,
                })

    return pd.DataFrame(rows)


def export_summary(summaries: dict[str, dict[str, pd.DataFrame]],
                   output_path: str | Path) -> Path:
    """
    Write all summary tables to a single multi-sheet workbook.

    :param summaries: Ship type label to the output of :func:`summarise_shiptype`.
    :type summaries: dict[str, dict[str, pandas.DataFrame]]
    :param output_path: Destination ``.xlsx`` path.
    :type output_path: str or Path
    :returns: The path written.
    :rtype: Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix(summaries, column='ANY')
    fleet_sizes = matrix.loc['FLEET SIZE']
    matrix_pct = (matrix.drop(index='FLEET SIZE') / fleet_sizes * 100).round(2)

    raw = pd.concat([summary['raw_values'].assign(SHIP_TYPE=label)
                     for label, summary in summaries.items()
                     if not summary['raw_values'].empty],
                    ignore_index=True) if summaries else pd.DataFrame()

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        matrix.to_excel(writer, sheet_name='matrix_any')
        matrix_pct.to_excel(writer, sheet_name='matrix_any_pct')
        build_long_table(summaries).to_excel(writer, sheet_name='long', index=False)

        for label, summary in summaries.items():
            summary['counts'].to_excel(writer, sheet_name=f'by_role_{label}'[:31])

        if not raw.empty:
            raw.to_excel(writer, sheet_name='raw_values', index=False)

            unclassified = raw[raw['FUEL_TYPE'].isna()]
            if not unclassified.empty:
                unclassified.to_excel(writer, sheet_name='unclassified', index=False)

    logger.info("Wrote fuel summary to %s", output_path)
    return output_path


def main() -> None:
    """
    Build the fuel usage summary for every configured ship type.

    :returns: None
    :rtype: None
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s')

    types_list = [ShipType.BULK_CARRIER,
                  ShipType.CONTAINER_SHIP,
                  ShipType.TANKER,
                  ShipType.RO_PAX]

    output_path = Path(
        r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\results\new'
        r'\fuel_usage_summary.xlsx')

    summaries: dict[str, dict[str, pd.DataFrame]] = {}

    for shiptype in types_list:
        summaries[shiptype.string] = summarise_shiptype(shiptype=shiptype,
                                                        max_build_year=1990)

    export_summary(summaries, output_path)

    print(build_matrix(summaries, column='ANY').to_string())


if __name__ == '__main__':
    main()
