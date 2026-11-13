from pathlib import Path
import numpy as np
import pandas as pd

from data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.fleet_data_class import FleetData
from parameters.shipshape.ShipShapeBasic_class import ShipType
from itertools import combinations
from matplotlib import pyplot as plt


def load_fleet_data(filepath: str|Path) -> FleetData:

    fleet=FleetData(verbose=True)


    rename_dict = {'MMSI (MARITIME MOBILE SERVICE IDENTITY)':'MMSI',
                   'NUMBER OF SCREWS':'SCREWS',
                   'BEAM MLD (M)':'BEAM (M)',
                   'DEPTH MOULDED (M)':'DEPTH (M)',
                   'ENGINE DERIVED TOTAL MECHANICAL PROPULSION (KW)':'TOTAL MAIN POWER (KW)',
                   'ENGINE DERIVED TOTAL MAIN ENGINE MECHANICAL KW': 'MCR (KW)',
                    'TOTAL MAIN POWER (KW)':'MCR (KW)',
                   'AUXILIARY DERIVED TOTAL MECHANICAL GENERATED (KW)':'TOTAL AUX POWER (KW)',
                   'MAIN ENGINE FUEL TYPE':"MAIN_FUEL",
                   'AUXILIARY ENGINE FUEL TYPE':"AUX_FUEL",
                   "MAIN BUNKER FUEL TYPE":"BUNKER_FUEL",
                   'SPEED (KNOTS)':'Vs',
                   'SPEED (KN)':'Vs',
                   'VS':'Vs',
                   'LANE LENGTH (M)':'LM',
                   'PASSENGERS':'Np',
                   'VEHICLE CAPACITY':'Nv',
                   'NP':'Np',
                   'NV':'Nv',
                   'LDT':'LSW',
                   'MAIN BUNKER CAPACITY (M3)':'BUNKER (M3)'}

    fleet.import_data(file_path=filepath,
                      column_renames=rename_dict, )

    fleet.list_datasets()



    # export_data = fleet.raw_data['raw1']
    #
    # filtered_df = export_data[~export_data['TYPE'].str.contains('maran', case=False, na=False)]
    #
    # filtered_df.to_excel(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX\ROPAX.xlsx')


    return fleet

def tolerances_from_dataset(
        df: pd.DataFrame,
        dimension_columns: list[str],
        fraction: float = 0.01,
        statistic: str = "median",
) -> dict[str, float]:
    """Absolute tolerance per dimension as a fraction of a dataset-wide statistic.

    statistic is any pandas aggregation name, e.g. "median" or "mean".
    """

    # .agg() applies the statistic column by column, returning one value per dimension
    return (fraction * df[dimension_columns].agg(statistic)).to_dict()

def within_group_pairwise_differences(
        candidates: pd.DataFrame,
        group_columns: list[str],
        dimension_columns: list[str],
        relative: bool = True,
) -> pd.DataFrame:
    """Differences in each dimension for every ship pair within each group.

    With relative=True, each difference is divided by the mean of the two
    values, matching the definition used by the 1 % similarity filter.
    Pairs where either value is missing give NaN for that dimension.
    """

    blocks = []
    for _, group in candidates.groupby(group_columns, dropna=True):
        n = len(group)
        if n < 2:
            continue

        # Row and column positions of the upper triangle of an n x n matrix:
        # every unordered pair (i, j) with i < j, exactly once
        i, j = np.triu_indices(n, k=1)

        block = {}
        for column in dimension_columns:
            values = group[column].to_numpy(dtype=float)
            diff = np.abs(values[i] - values[j])   # NaN if either value is NaN

            if relative:
                mean = (np.abs(values[i]) + np.abs(values[j])) / 2
                # Divide only where the mean is positive; elsewhere leave NaN
                diff = np.divide(diff, mean, out=np.full_like(diff, np.nan), where=mean > 0)

            block[column] = diff

        blocks.append(pd.DataFrame(block))

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=dimension_columns)


def plot_pairwise_difference_histograms(
        pair_differences: pd.DataFrame,
        threshold: float | dict[str, float],
        relative: bool = True,
        n_bins: int = 40,
):
    """Log-scale histogram of pairwise differences per dimension.

    threshold is a fraction (e.g. 0.01) when relative=True, or a dict of
    absolute tolerances per column when relative=False. Pairs with zero
    difference cannot be shown on a log axis and are counted in the title.
    """

    columns = list(pair_differences.columns)
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.5), squeeze=False)

    for ax, column in zip(axes[0], columns):
        values = pair_differences[column].dropna()
        if relative:
            values = values * 100   # show as percent

        n_zero = (values == 0).sum()
        positive = values[values > 0]

        if positive.nunique() > 1:
            # Bin edges evenly spaced in powers of ten
            bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), n_bins)
            ax.hist(positive, bins=bins)
            ax.set_xscale("log")

        line = threshold * 100 if relative else threshold[column]
        ax.axvline(line, color="red", linestyle="--", label="threshold")

        ax.set_title(f"{column}\n({n_zero} of {len(values)} pairs identical)")
        ax.set_xlabel("Difference (%)" if relative else "Absolute difference")

    axes[0][0].set_ylabel("Number of pairs")
    axes[0][0].legend()
    fig.tight_layout()
    return fig


def build_group_lookup(
        reference_df: pd.DataFrame,
        name_column: str,
        group_column: str,
) -> pd.Series:
    """Create a name -> group lookup from a reference dataset.

    The reference dataset usually has one row per ship, so each builder or
    company appears many times. This reduces it to one group per name.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Dataset containing both the name and its group.
    name_column : str
        Column with the builder or company name, e.g. "BUILDER".
    group_column : str
        Column with the corresponding group, e.g. "BUILDER GROUP".

    Returns
    -------
    pd.Series
        Indexed by name, with the group as values. Can be passed directly
        to ``Series.map()``.

    Notes
    -----
    If a name appears with more than one group (e.g. a yard that changed
    owner, so older and newer ships list different groups), the most
    frequent group is used and a warning lists the affected names.
    Names whose group is missing are left out of the lookup, so they keep
    their own name when the lookup is applied.
    """

    pairs = reference_df[[name_column, group_column]].dropna().copy()

    # Stray whitespace is a common cause of failed matches between datasets
    pairs[name_column] = pairs[name_column].astype(str).str.strip()
    pairs[group_column] = pairs[group_column].astype(str).str.strip()

    # # Number of rows (ships) for each name-group combination
    # counts = pairs.groupby([name_column, group_column]).size()
    counts = reference_df.groupby([name_column, group_column])["N_SHIPS"].sum()

    # Names linked to more than one group
    groups_per_name = counts.groupby(level=name_column).size()
    conflicts = groups_per_name[groups_per_name > 1].index.tolist()
    if conflicts:
        print(
            f"{len(conflicts)} names in '{name_column}' map to more than one "
            f"group; the most frequent group is used: {conflicts}"
        )

    # Sort by frequency, then keep the first (most frequent) group per name
    lookup = (
        counts.sort_values(ascending=False)
        .reset_index()
        .drop_duplicates(subset=name_column, keep="first")
        .set_index(name_column)[group_column]
    )

    return lookup


def add_group_column(
        df: pd.DataFrame,
        lookup: pd.Series,
        name_column: str,
        new_column: str,
        fill_with_name: bool = True,
) -> pd.DataFrame:
    """Add a group column to a ship dataset using a lookup.

    Parameters
    ----------
    df : pd.DataFrame
        Ship dataset containing ``name_column``.
    lookup : pd.Series
        Name -> group lookup from ``build_group_lookup``.
    name_column : str
        Column in ``df`` to look up, e.g. "BUILDER".
    new_column : str
        Name of the group column to add, e.g. "BUILDER_GROUP".
    fill_with_name : bool, default True
        If True, names not found in the lookup keep their own name as their
        group, so an independent yard forms a group of its own. If False,
        they are left as NaN, which makes unmatched names easy to find.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the new column.
    """

    df = df.copy()
    names = df[name_column].astype("string").str.strip()

    # .map() looks up each name in the lookup's index; unmatched names give NaN
    df[new_column] = names.map(lookup)

    unmatched = names[df[new_column].isna() & names.notna()].unique()
    if len(unmatched):
        print(f"{new_column}: {len(unmatched)} names in '{name_column}' not found in lookup")

    if fill_with_name:
        df[new_column] = df[new_column].fillna(names)

    return df


def add_builder_and_company_groups(
        ship_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        builder_column: str = "BUILDER",
        builder_group_column: str = "BUILDER GROUP",
        company_column: str = "COMPANY",
        company_group_column: str = "COMPANY GROUP",
) -> pd.DataFrame:
    """Add BUILDER_GROUP and COMPANY_GROUP columns to a ship dataset.

    Column names refer to the reference dataset; the ship dataset must use
    the same names for the builder and company columns.
    """

    builder_lookup = build_group_lookup(reference_df, builder_column, builder_group_column)
    company_lookup = build_group_lookup(reference_df, company_column, company_group_column)

    ship_df = add_group_column(ship_df, builder_lookup, builder_column, "BUILDER_GROUP")
    ship_df = add_group_column(ship_df, company_lookup, company_column, "COMPANY_GROUP")

    return ship_df


def pre_processing(filepath: str | Path,
                   shiptype: str,
                   sister_ships:bool=True,
                   incomplete_data:bool=True,
                   outliers:bool=True,
                   outlier_columns:list[str]=['DWT', 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)'],
                   fuel_type_columns: list[str] = ['MAIN_FUEL',"AUX_FUEL","BUNKER_FUEL"],
                   max_nan_columns: list[str] = ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)', 'DWT','MAIN_FUEL',"BUNKER_FUEL"],
                   zero_as_missing: list[str] = ['LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)', 'DWT'],
                   duplicate_hull_columns:list[str] =['LOA (M)', 'LBP (M)', 'BEAM (M)','DEPTH (M)',"DRAFT (M)",'BUNKER_FUEL'],
                   duplicate_maker_column:list[str] = ["BUILDER_GROUP",'BUNKER_FUEL'],
                   max_build_year: int = 1990,
                   outlier_method: str = 'percentile',
                   iqr_window: float = 1.5,
                   percentile: float = 0.95):
    """
    Pre-process fleet data from raw import through to a cleaned base dataset.

    :param filepath: Path to the data directory or file.
    :type filepath: str or Path
    :param shiptype: Ship type identifier string (unused directly here, passed for context).
    :type shiptype: str
    :param fuel_type_columns: Column names containing fuel type information to split.
    :type fuel_type_columns: list[str]
    :param max_build_year: Minimum build year for ships to be retained (inclusive).
    :type max_build_year: int
    :param outlier_method: Outlier removal strategy. Either ``'percentile'`` (removes the
        outermost tail fraction of each column's distribution symmetrically) or ``'iqr'``
        (removes values outside ``Q1 - window*IQR`` and ``Q3 + window*IQR``). For skewed
        distributions such as DWT and LOA, the IQR method tends to be more conservative
        because it anchors on the interquartile range rather than absolute tail quantiles.
    :type outlier_method: str
    :param iqr_window: Multiplier for the IQR bounds. Only used when
        ``outlier_method='iqr'``. A value of 1.5 is the standard Tukey fence;
        3.0 targets only extreme outliers.
    :type iqr_window: float
    :param percentile: Central fraction of each distribution to retain. Only used when
        ``outlier_method='percentile'``. A value of 0.95 keeps the middle 95%
        (excludes 2.5% on each tail).
    :type percentile: float
    :returns: Processed :class:`FleetData` object with named intermediate datasets.
    :rtype: FleetData
    :raises ValueError: If ``outlier_method`` is not ``'percentile'`` or ``'iqr'``.
    """
    if outlier_method not in ('percentile', 'iqr'):
        raise ValueError(
            f"outlier_method must be 'percentile' or 'iqr', got '{outlier_method}'"
        )

    fleet_data = load_fleet_data(filepath)

    date_filter = fleet_data.create_filter(
        column='BUILT',
        value=max_build_year,
        operator=ComparisonOperator.GREATER_EQUAL,
        require_all=False
    )



    # Resolve outlier kwargs based on chosen method
    if outlier_method == 'percentile':
        outlier_kwargs = dict(
            percentile_filter_cols=outlier_columns,
            percentile=percentile,
            IQR_filter=None,
        )
    else:
        outlier_kwargs = dict(
            IQR_filter=outlier_columns,
            percentile_filter_cols=None,
            percentile=None,
        )

    data_name = 'raw1'



    if incomplete_data:

        fleet_data.processing_fleet_data(
            datafile=data_name,
            name='processed_missing_data',
            filter_by=date_filter,
            duplicates=None,
            max_nan_columns = max_nan_columns,
            max_nan_no = 0,
            zero_as_missing = zero_as_missing,
            fuel_columns=fuel_type_columns
        )

        data_name = 'processed_missing_data'

    fleet_data.processing_fleet_data(
        datafile=data_name,
        name='base_with_split_fuels',
        fuel_columns=fuel_type_columns,
    )

    data_name = 'base_with_split_fuels'

    if sister_ships:


        un_filtered_df = fleet_data.processed_data[data_name]

        # group_reference_df = pd.read_excel(r"C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\newer\full_data_11_10\Full_dataset_buildersandcompanies.xlsx")
        #
        # group_reference_df = (
        #     group_reference_df
        #     .groupby(["BUILDER", "BUILDER GROUP", "COMPANY", "COMPANY GROUP"], dropna=False)
        #     .size()
        #     .reset_index(name="N_SHIPS")
        # )
        #
        #
        #
        # group_reference_df.columns = group_reference_df.columns.str.upper()

        group_reference_df = pd.read_pickle(r"C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\newer\full_data_11_10\Full_dataset_buildersandcompanies.pkl")

        un_filtered_df = add_builder_and_company_groups(
            ship_df=un_filtered_df,
            reference_df=group_reference_df,
            builder_column = "BUILDER",
            company_column = "COMPANY",
            builder_group_column="BUILDER GROUP",  # as named in the reference data
            company_group_column="GROUP COMPANY",
        )

        sister_ship_columns_1 = duplicate_hull_columns

        completeness_columns = ['Vs','COMPANY','BUILDER',"MCR (KW)"]

        filtered_sister_1 = (
            un_filtered_df.assign(
                completeness=un_filtered_df[completeness_columns].notna().sum(axis=1),
            )
            .sort_values(by="completeness", ascending=False
                         )
            .drop_duplicates(subset=sister_ship_columns_1, keep='first')
            .drop(columns='completeness')
        )

        #
        fleet_data.add_processed_data(df=filtered_sister_1,
                                      name = 'processed_sister_ships_hull')



        data_name = 'processed_sister_ships_hull'



        sister_ship_candidates_maker = (
        filtered_sister_1.groupby(duplicate_maker_column, dropna=False)
          .filter(lambda x: len(x) > 1)
    )




        # 'LBP (M)', 'BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LOA (M)'



        def ships_are_similar(
                ship_a: pd.Series,
                ship_b: pd.Series,
                dimension_columns: list[str],
                tolerances: dict[str, float] | None = None,
                relative_tolerance: float | None = None,
                required_columns: tuple[str, ...] = (),
                min_matching: int = 4,
        ) -> bool:
            """Decide whether two ships are dimensionally similar.

            Exactly one of ``tolerances`` (absolute, in metres per dimension) or
            ``relative_tolerance`` (a fraction, e.g. 0.01 for 1 %) must be given.
            With a relative tolerance, the allowed difference for each dimension is
            that fraction of the mean of the two ships' values, so the test is
            symmetric: comparing A with B gives the same result as B with A.
            """

            if (tolerances is None) == (relative_tolerance is None):
                raise ValueError("Specify either 'tolerances' or 'relative_tolerance', not both or neither.")

            n_matching = 0
            for column in dimension_columns:
                value_a, value_b = ship_a[column], ship_b[column]

                if pd.isna(value_a) or pd.isna(value_b):
                    within_tolerance = False
                else:
                    if relative_tolerance is not None:
                        allowed = relative_tolerance * (abs(value_a) + abs(value_b)) / 2
                    else:
                        allowed = tolerances[column]
                    within_tolerance = abs(value_a - value_b) <= allowed

                if column in required_columns and not within_tolerance:
                    return False

                n_matching += within_tolerance

            return n_matching >= min_matching

        def find_similar_ship_pairs(
                candidates: pd.DataFrame,
                group_columns: list[str],
                dimension_columns: list[str],
                ship_id_column: str = "MMSI",
                **similarity_kwargs,
        ) -> pd.DataFrame:
            """List every pair of ships within a group that are dimensionally similar.

            This function is for inspection only. It reports all similar pairs, so a
            cluster of three sisters A, B, C appears as (A, B), (A, C) and (B, C).
            It does not decide which ship to keep; ``assign_sister_references`` does
            that. Use it to check which ships are matched and by how much their
            dimensions differ.

            Parameters
            ----------
            candidates : pd.DataFrame
                Ships to compare, one row per ship. Usually pre-filtered to groups
                with more than one ship.
            group_columns : list[str]
                Columns that sisters must share exactly, e.g.
                ``["BUILDER", "COMPANY", "MCR (KW)"]``. Ships are only compared with
                other ships in the same group. Must be a list, even for one column,
                so that ``groupby`` returns tuple keys.
            dimension_columns : list[str]
                Dimension columns passed to ``ships_are_similar``.
            tolerances : dict[str, float]
                Per-dimension tolerances passed to ``ships_are_similar``.
            ship_id_column : str, default "MMSI"
                Column that uniquely identifies each ship.

            Returns
            -------
            pd.DataFrame
                One row per similar pair, with columns:

                - ``MMSI_A``, ``MMSI_B``: identifiers of the two ships.
                - one column per entry in ``group_columns``: the shared group values.
                - ``<dimension>_difference``: absolute difference in each dimension.

                Empty if no similar pairs are found.
            """

            pairs = []

            # dropna=False keeps ships whose group values are missing, so they are
            # grouped together rather than silently dropped.
            for group_key, group in candidates.groupby(group_columns, dropna=False):

                # A single ship cannot form a pair.
                if len(group) < 2:
                    continue

                # combinations() yields every unordered pair exactly once:
                # n ships give n * (n - 1) / 2 comparisons.
                for idx_a, idx_b in combinations(group.index, 2):

                    ship_a = group.loc[idx_a]
                    ship_b = group.loc[idx_b]

                    if ships_are_similar(ship_a, ship_b, dimension_columns, **similarity_kwargs):
                        pair = {
                            "MMSI_A": ship_a[ship_id_column],
                            "MMSI_B": ship_b[ship_id_column],
                        }

                        # group_key is a tuple with one value per grouping column,
                        # so zipping the two gives {column name: shared value}.
                        pair.update(dict(zip(group_columns, group_key)))

                        # Record how close the pair is on every dimension, which is
                        # useful for judging whether the tolerances are reasonable.
                        for column in dimension_columns:
                            pair[f"{column}_difference"] = abs(ship_a[column] - ship_b[column])

                        pairs.append(pair)

            return pd.DataFrame(pairs)

        def assign_sister_references(
                candidates: pd.DataFrame,
                group_columns: list[str],
                dimension_columns: list[str],
                ship_id_column: str = "MMSI",
                order_by: str | list[str] | None = None,
        **similarity_kwargs,
        ) -> pd.DataFrame:
            """Cluster sister ships and choose one reference ship to keep per cluster.

            Within each group, ships are processed in order. The first unassigned
            ship becomes a reference (kept). Every remaining unassigned ship that is
            similar to that reference is assigned to it as a sister (to be removed).
            The next unassigned ship then becomes a new reference, and so on until
            every ship in the group is assigned.

            Comparing each ship against a single reference, rather than linking all
            similar pairs together, prevents chaining: if A is similar to B and B to
            C, but A is not similar to C, C will not be grouped with A. A ship is
            only removed if it is similar to the ship that is kept.

            Parameters
            ----------
            candidates : pd.DataFrame
                Ships to cluster, one row per ship (duplicate identifiers will
                produce incorrect results).
            group_columns : list[str]
                Columns that sisters must share exactly. Clustering is done
                independently within each group.
            dimension_columns : list[str]
                Dimension columns passed to ``ships_are_similar``.
            tolerances : dict[str, float]
                Per-dimension tolerances passed to ``ships_are_similar``.
            ship_id_column : str, default "MMSI"
                Column that uniquely identifies each ship.
            order_by : str, list[str] or None, default None
                Column(s) used to sort ships within each group before clustering.
                The first ship in this order becomes the reference, so this decides
                which sister is kept. ``["BUILT", "MMSI"]`` keeps the earliest-built
                ship (the lead ship), with MMSI as a deterministic tie-breaker for
                ships built in the same year. Missing values sort last. If None,
                the existing row order of ``candidates`` is used.

            Returns
            -------
            pd.DataFrame
                One row per candidate ship, with columns:

                - ``<ship_id_column>``: the ship's identifier.
                - ``SISTER_REFERENCE``: identifier of the reference ship of its
                  cluster. For a reference ship, this is its own identifier.
                - ``IS_REFERENCE``: True for the ship to keep, False for sisters to
                  remove.
                - ``CLUSTER_SIZE``: number of ships in the cluster. 1 means no
                  sister was found for that ship.

            Notes
            -----
            The result depends on ``order_by``, because each reference is chosen
            before its sisters are searched for. The similarity rule itself
            (``required_columns`` and ``min_matching``) is set by the defaults of
            ``ships_are_similar``.
            """

            records = []

            for _, group in candidates.groupby(group_columns, dropna=False):

                # Sorting decides which ship is "first" and therefore kept.
                # kind="stable" preserves the original order of tied rows.
                if order_by is not None:
                    group = group.sort_values(order_by, kind="stable")

                # Row labels of the ships not yet assigned to any cluster.
                unassigned = list(group.index)

                while unassigned:
                    # The first unassigned ship starts a new cluster as its reference.
                    reference_idx = unassigned.pop(0)
                    reference_ship = group.loc[reference_idx]
                    reference_id = reference_ship[ship_id_column]

                    # Every unassigned ship similar to the reference is its sister.
                    sister_indices = [
                        idx for idx in unassigned
                        if ships_are_similar(
                            reference_ship,
                            group.loc[idx],
                            dimension_columns,
                            **similarity_kwargs,
                        )
                    ]

                    # Assigned sisters leave the pool, so they cannot become
                    # references themselves or be assigned to a second cluster.
                    unassigned = [idx for idx in unassigned if idx not in sister_indices]

                    records.append({
                        ship_id_column: reference_id,
                        "SISTER_REFERENCE": reference_id,
                        "IS_REFERENCE": True,
                    })

                    for idx in sister_indices:
                        records.append({
                            ship_id_column: group.loc[idx, ship_id_column],
                            "SISTER_REFERENCE": reference_id,
                            "IS_REFERENCE": False,
                        })

            assignments = pd.DataFrame(records)

            # transform("size") returns the size of each ship's cluster on every row
            # of that cluster, so the column aligns with the original rows.
            assignments["CLUSTER_SIZE"] = (
                assignments.groupby("SISTER_REFERENCE")[ship_id_column].transform("size")
            )

            return assignments

        def find_cross_builder_matches(
                candidates: pd.DataFrame,
                dimension_columns: list[str],
                builder_column: str = "BUILDER",
                mcr_column: str = "MCR (KW)",
                ship_id_column: str = "MMSI",
                **similarity_kwargs
        ) -> pd.DataFrame:
            """Summarise builder combinations that produce dimensionally similar ships
            with identical MCR, to identify which builders should share a group."""

            records = []
            for _, group in candidates.groupby(mcr_column):

                # Only groups with more than one builder can produce cross-builder pairs
                if group[builder_column].nunique() < 2:
                    continue

                for idx_a, idx_b in combinations(group.index, 2):
                    ship_a, ship_b = group.loc[idx_a], group.loc[idx_b]

                    if ship_a[builder_column] == ship_b[builder_column]:
                        continue

                    if ships_are_similar(ship_a, ship_b, dimension_columns, **similarity_kwargs):
                        # Sorting makes (A, B) and (B, A) the same combination
                        builder_1, builder_2 = sorted([ship_a[builder_column], ship_b[builder_column]])
                        records.append({
                            "BUILDER_1": builder_1,
                            "BUILDER_2": builder_2,
                            "MMSI_A": ship_a[ship_id_column],
                            "MMSI_B": ship_b[ship_id_column],
                        })

            pairs = pd.DataFrame(records)
            if pairs.empty:
                return pairs

            return (
                pairs.groupby(["BUILDER_1", "BUILDER_2"])
                .agg(
                    n_pairs=("MMSI_A", "size"),
                    example_mmsi_a=("MMSI_A", "first"),
                    example_mmsi_b=("MMSI_B", "first"),
                )
                .reset_index()
                .sort_values("n_pairs", ascending=False)
            )

        DIMENSION_TOLERANCES = {
            "LOA (M)": 1.0,
            "LBP (M)": 1.0,
            "BEAM (M)": 0.5,
            "DEPTH (M)": 0.5,
            "DRAFT (M)": 0.5,
            "MCR (KW)": 10,
        }

        dimension_columns = ['LOA (M)', 'LBP (M)', 'BEAM (M)','DEPTH (M)', 'DRAFT (M)', "MCR (KW)"]
        # dimension_columns = ['LOA (M)', 'LBP (M)', 'BEAM (M)','DEPTH (M)', 'DRAFT (M)']

        comparison = pd.DataFrame({
            "manual": pd.Series(DIMENSION_TOLERANCES),
            "1% of median": pd.Series(
                tolerances_from_dataset(filtered_sister_1, dimension_columns, statistic="median")),
            "1% of mean": pd.Series(tolerances_from_dataset(filtered_sister_1, dimension_columns, statistic="mean")),
        })
        print(comparison.round(3))

        similarity_settings = dict(
            tolerances=tolerances_from_dataset(filtered_sister_1, dimension_columns, statistic="median"),

            min_matching=len(dimension_columns),
        )

        similarity_settings = dict(
            tolerances=tolerances_from_dataset(filtered_sister_1, dimension_columns, statistic="median"),
            required_columns=("MCR (KW)",),
            min_matching=len(dimension_columns) - 1,
        )


        builder_review = find_cross_builder_matches(filtered_sister_1, dimension_columns=dimension_columns, tolerances=DIMENSION_TOLERANCES)


        filtered_1_sister_df = un_filtered_df



        fleet_data.add_processed_data(df=filtered_1_sister_df,
                                      name='processed_sister_ships')

        sister_pairs = find_similar_ship_pairs(candidates=sister_ship_candidates_maker,
                                               group_columns=duplicate_maker_column,
                                               dimension_columns=dimension_columns,
                                               **similarity_settings)

        similarity_settings = dict(
            tolerances=DIMENSION_TOLERANCES,
            required_columns=("MCR (KW)",),
            min_matching=len(dimension_columns) - 1,  # MCR + 4 of the 5 hull dimensions
        )

        builder_review = find_cross_builder_matches(
            filtered_sister_1, dimension_columns=dimension_columns, **similarity_settings)

        sister_pairs = find_similar_ship_pairs(
            candidates=sister_ship_candidates_maker,
            group_columns=duplicate_maker_column,
            dimension_columns=dimension_columns,
            **similarity_settings)

        sister_assignments = assign_sister_references(
            candidates=sister_ship_candidates_maker,
            group_columns=duplicate_maker_column,
            dimension_columns=dimension_columns,
            order_by=["BUILT", "MMSI"],
            **similarity_settings)


        mmsis_to_remove = sister_assignments.loc[
            ~sister_assignments["IS_REFERENCE"], "MMSI"
        ]

        labelled_candidates = sister_ship_candidates_maker.merge(
            sister_assignments,
            on="MMSI",
            how="left",
            validate="one_to_one",
        )

        labelled_candidates["SISTER_STATUS"] = np.select(
            condlist=[
                labelled_candidates["CLUSTER_SIZE"] == 1,
                labelled_candidates["IS_REFERENCE"],
            ],
            choicelist=[
                "no sister found",
                "kept (lead ship)",
            ],
            default="removed (sister)",
        )

        # Order rows so each cluster appears together, lead ship first
        labelled_candidates = labelled_candidates.sort_values(
            duplicate_maker_column + ["SISTER_REFERENCE", "BUILT", "MMSI"]
        )

        inspection_columns = (
                duplicate_maker_column
                + ["SISTER_REFERENCE", "SISTER_STATUS", "MMSI",'NAME', "BUILT",
                   "LOA (M)", "LBP (M)", "BEAM (M)", "DEPTH (M)", "DRAFT (M)", "MCR (KW)"]
        )

        inspected_candidates = labelled_candidates[inspection_columns]

        filtered_sister_2 = filtered_sister_1[
            ~filtered_sister_1["MMSI"].isin(mmsis_to_remove)
        ].copy()

        fleet_data.add_processed_data(df=builder_review,
                                      name='builder_review_analysis')

        fleet_data.add_processed_data(df=sister_assignments,
                                       name='analysis_sister_assignments')

        fleet_data.add_processed_data(df=inspected_candidates,
                                      name='inspected_sister_ship_candidates')

        fleet_data.add_processed_data(df=sister_ship_candidates_maker,
                                      name='sister_ship_candidates')

        fleet_data.add_processed_data(df=filtered_sister_2,
                                      name='sister_ship_maker_new')


        data_name ='sister_ship_maker_new'

    if outliers:

        fleet_data.processing_fleet_data(
            datafile=data_name,
            name='processed_outliers',
            duplicates=None,
            **outlier_kwargs
        )

        data_name = 'processed_outliers'

    fleet_data.add_processed_data(df=fleet_data.processed_data[data_name], name="pre_processed_data")


    print(f"\n=== Added ratio columns and applied age filter to datasets "
          f"(outlier method: {outlier_method}) ===")
    fleet_data.list_datasets()

    return fleet_data


def getting_alt_fuel_ships(filepath: str | Path,
                           shiptype: str|ShipType,
                           fuel_type_columns: list[str],
                           max_build_year:int,
                           alt_fuels: list[str]|dict[str,list],
                           conventional_fuels: list[str],
                           sister_ships:bool=True,
                           incomplete_data:bool=True,
                           outliers:bool=True,
                           outlier_method: str = 'percentile',
                           iqr_window: float = 1.5,
                           percentile: float = 0.95,
                           ):
    """
    Load, pre-process, and categorise ships by fuel type.

    :param filepath: Path to the raw data directory or file.
    :type filepath: str or Path
    :param shiptype: Ship type label, used for display and routing in :func:`get_variables`.
    :type shiptype: str
    :param fuel_type_columns: Raw column names containing fuel type strings.
    :type fuel_type_columns: list[str]
    :param alt_fuels: Fuel type substrings that classify a ship as alternative-fuelled.
    :type alt_fuels: list[str]
    :param conventional_fuels: Fuel type substrings that classify a ship as conventionally fuelled.
    :type conventional_fuels: list[str]
    :param outlier_method: Outlier removal strategy passed to :func:`pre_processing`.
        ``'percentile'`` removes symmetric tails; ``'iqr'`` uses the Tukey fence.
    :type outlier_method: str
    :param iqr_window: IQR fence multiplier. Only used when ``outlier_method='iqr'``.
    :type iqr_window: float
    :param percentile: Tail fraction to retain. Only used when ``outlier_method='percentile'``.
    :type percentile: float
    :param create_plots: Whether to produce diagnostic scatter plots.
    :type create_plots: bool
    :param plot_target: Y-axis variable for the matching importance plot.
    :type plot_target: str
    :param plot_predictor: X-axis variable for the matching importance plot.
    :type plot_predictor: str
    :param

    :returns: :class:`FleetData` instance with all fuel-type subsets stored as named datasets.
    :rtype: FleetData
    """

    if not isinstance(shiptype, str):
        shiptype = shiptype.string


    fleet_data = pre_processing(
        filepath=filepath,
        shiptype=shiptype,
        outliers=outliers,
        sister_ships=sister_ships,
        incomplete_data=incomplete_data,
        outlier_method=outlier_method,
        iqr_window=iqr_window,
        percentile=percentile,
        max_build_year=max_build_year,
    )
    # fleet_data = load_fleet_data(filepath)
    #

    if isinstance(alt_fuels, list):
        alt_fuels = {"alternative_fuels": alt_fuels}

    for string, alt_fuel in alt_fuels.items():


        # Filter 1: Ships with ANY alternative fuel (includes dual-fuel)
        all_alt_fuel_filter = fleet_data.create_filter(
            column=fuel_type_columns,
            value=alt_fuel,
            operator=ComparisonOperator.CONTAINS,
            require_all=False
        )

        fleet_data.processing_fleet_data(
            datafile="pre_processed_data",
            name=f'any_{string}_all',
            filter_by=all_alt_fuel_filter,
        )

        # Filter 2: Ships with ONLY alternative fuels (no conventional fuels)
        # This requires checking that NO fuel columns contain conventional fuels
        conventional_fuel_filter = fleet_data.create_filter(
            column=['MAIN_FUEL_1', 'MAIN_FUEL_2',
                    "AUX_FUEL_1", "AUX_FUEL_2",
                    "BUNKER_FUEL"],
            value=conventional_fuels,
            operator=ComparisonOperator.CONTAINS,
            require_all=False,
            negate=True  # Keep rows that DON'T contain conventional fuels
        )

        fleet_data.processing_fleet_data(
            datafile=f'any_{string}_all',
            name=f'pure_{string}',
            filter_by=conventional_fuel_filter,
        )

        # Filter 3: Dual-fuel ships (have both alternative AND conventional)
        # Start with ships that have alt fuels, then keep only those WITH conventional
        conventional_fuel_filter_positive = fleet_data.create_filter(
            column=['MAIN_FUEL_1', 'MAIN_FUEL_2',
                    "AUX_FUEL_1", "AUX_FUEL_2",
                    "BUNKER_FUEL"],
            value=conventional_fuels,
            operator=ComparisonOperator.CONTAINS,
            require_all=False,
            negate=False  # Keep rows that DO contain conventional fuels
        )

        fleet_data.processing_fleet_data(
            datafile=f'any_{string}_all',
            name=f'dual_fuel_{string}',
            filter_by=conventional_fuel_filter_positive,
        )

        if isinstance(alt_fuel, list):
            for spec_fuel in alt_fuel:

                alt_spec_filter = fleet_data.create_filter(
                    column=fuel_type_columns,
                    value=spec_fuel,
                    operator=ComparisonOperator.CONTAINS,
                    require_all=False
                )

                fleet_data.processing_fleet_data(
                    datafile="pre_processed_data",
                    name=f'any_{spec_fuel}',
                    filter_by=alt_spec_filter,
                )

                fleet_data.processing_fleet_data(
                    datafile=f'any_{spec_fuel}',
                    name=f'pure_{spec_fuel}',
                    filter_by=conventional_fuel_filter,
                )

                fleet_data.processing_fleet_data(
                    datafile=f'any_{spec_fuel}',
                    name=f'dual_fuel_{spec_fuel}',
                    filter_by=conventional_fuel_filter_positive,
                )


        #Filter 4 - dual fuelled ships where the main bunker fuel is alternative

        alt_bunker_filter = fleet_data.create_filter(column = "BUNKER_FUEL_1",
                                                     value = alt_fuel,
                                                     operator=ComparisonOperator.CONTAINS,
                                                     negate=False)

        fleet_data.processing_fleet_data(
            datafile=f'dual_fuel_{string}',
            name=f'dual_fuel_{string}_alt_main',
            filter_by=alt_bunker_filter,
        )

        fleet_data.filter_by_dataset(
            datafile=f'dual_fuel_{string}',
            column_name='MMSI',
            reference_values=f'dual_fuel_{string}_alt_main',
            new_dataframe=True,
            name=f'dual_fuel_{string}_conv_main'
        )

        print("\n======================")
        print(f"\n{shiptype.upper()}")
        print("\n======================")

        fleet_data.list_datasets()



    # Filter 5: Pure conventional ships (no alternative fuels at all)

    all_alt_fuels = []
    for key, items in alt_fuels.items():
        all_alt_fuels+=items

    all_alt_fuels = list(all_alt_fuels)

    print(all_alt_fuels)
    no_alt_fuel_filter = fleet_data.create_filter(
        column=fuel_type_columns,
        value=all_alt_fuels,
        operator=ComparisonOperator.CONTAINS,
        require_all=False,
        negate=True  # Keep rows that DON'T contain alt fuels
    )

    fleet_data.processing_fleet_data(
        datafile="pre_processed_data",
        name='pure_conventional',
        filter_by=no_alt_fuel_filter,
    )

    for key in alt_fuels.keys():

        # Now remove dual-fuel ships from pure_conventional using reference filter
        fleet_data.filter_by_dataset(
            datafile='pure_conventional',
            column_name='MMSI',
            reference_values=f'dual_fuel_{string}',
        )

    # fleet_data.plot_matching_importance(conventional_df='pure_conventional',
    #                                     alternative_df=f'pure_{string}',
    #                                     target_variable=plot_target,
    #                                     predictor_variable=plot_predictor,
    #                                     name=f'{shiptype} matching dataset comparison')


    # Print summary
    print(f"\n=== FLEET CATEGORIZATION SUMMARY {shiptype.upper()} ===")
    print(f"Total ships (after deduplication): {len(fleet_data.processed_data['base_with_split_fuels'])}")
    print(f"Pure conventional fuel only: {len(fleet_data.processed_data['pure_conventional'])}")

    for string, alt_fuel in alt_fuels.items():

        print(f"\n=== Datasets found for {string} ===")

        print(f"{shiptype.upper()} with ANY {string}: {len(fleet_data.processed_data[f'any_{string}_all'])}")
        print(f"  - Pure {string} only: {len(fleet_data.processed_data[f'pure_{string}'])}")
        print(f"  - Dual-fuel ({string} + conventional): {len(fleet_data.processed_data[f'dual_fuel_{string}'])}")
        print(f"  - Dual-fuel ({string} main): {len(fleet_data.processed_data[f'dual_fuel_{string}_alt_main'])}")
        print(f"  - Dual-fuel (conventional main): {len(fleet_data.processed_data[f'dual_fuel_{string}_conv_main'])}")



    fleet_data.list_datasets()



    print(fleet_data.processed_data["pre_processed_data"].columns)

    return fleet_data

def get_variables(shiptype:'ShipType'):

    print(f'USING DATA FOR {shiptype} SHIP')

    constant_vars = ['BEAM (M)','DEPTH (M)','DRAFT (M)','LBP (M)','LOA (M)','DWT']

    default_pred_constraints = {'BEAM (M)': ['DWT'],
                                'DEPTH (M)': ['DWT'],
                                'DRAFT (M)': ['DWT'],
                                'LBP (M)': ['DWT'],
                                'LOA (M)': ['DWT'],
                                'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)']}

    if shiptype == ShipType.TANKER:

        print('Tanker filepath used')
        filepath = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\Tanker')
        save_dir = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\results\new\Tanker')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 170000, 250000, 330000, np.inf]}
        variables = constant_vars + ['MCR (KW)']
        pred_constraints = default_pred_constraints
        datasets = datasets = {"dual fuel LNG": "dual_fuel_LNG", "Dual fuel Methanol": "dual_fuel_Methanol"}


    elif shiptype == ShipType.BULK_CARRIER:
        print('bulk carrier filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\Bulk_carrier')
        save_dir = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\results\new\Bulk_carrier')
        bins = {'DWT': [10000, 25000, 55000, 75000, 120000, 250000, 330000, np.inf]}
        variables = constant_vars+['MCR (KW)']
        # variables = ['DWT', 'LBP (M)', 'DRAFT (M)']
        pred_constraints = default_pred_constraints

        # datasets = {"dual fuel LNG": "dual_fuel_LNG", "Bio fuel": "pure_Biofuel"}
        datasets = {"dual fuel LNG": "dual_fuel_LNG"}

    elif shiptype == ShipType.CONTAINER_SHIP:
        print('container ship filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\Containership')
        save_dir = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\results\new\Containership')
        # bins = None
        bins = {'TEU': [2999,5999,7999,11999,16999, np.inf]}
        # variables = constant_vars + ['TEU','Vs','MCR (KW)']
        # pred_constraints = {'BEAM (M)': ['DWT', 'TEU', 'Vs'],
        #                     'DEPTH (M)': ['DWT', 'TEU', 'Vs'],
        #                     'DRAFT (M)': ['DWT', 'TEU', 'Vs'],
        #                     'LBP (M)': ['DWT', 'TEU', 'Vs'],
        #                     'LOA (M)': ['DWT', 'TEU', 'Vs'],
        #                     'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'TEU'],
        #                     'TEU': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT']}
        variables = constant_vars + ['TEU','MCR (KW)']
        pred_constraints = {'BEAM (M)': ['DWT', 'TEU'],
                            'DEPTH (M)': ['DWT', 'TEU'],
                            'DRAFT (M)': ['DWT', 'TEU'],
                            'LBP (M)': ['DWT', 'TEU'],
                            'LOA (M)': ['DWT', 'TEU'],
                            'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'TEU'],
                            'TEU': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT']}

        datasets = {"dual fuel LNG": "dual_fuel_LNG", "Dual fuel Methanol": "dual_fuel_Methanol"}

    else:
        print('ropax filepath used')
        filepath=Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\static_data\RO_PAX\roro_only')
        save_dir = Path(r'C:\Users\bnoble\GitHub\AlternativeShipRegression\knowledge_layer\results\new\RO_Pax')
        bins = ['TYPE']
        variables = constant_vars + [ 'Np', 'Nv','Vs']
        pred_constraints = {'BEAM (M)': ['DWT', 'Np', 'Nv', 'Vs', 'LBP (M)', 'LOA (M)'],
                            'DEPTH (M)': ['DWT', 'Np', 'Nv', 'Vs', 'LBP (M)', 'LOA (M)'],
                            'DRAFT (M)': ['DWT', 'Np', 'Nv', 'Vs', 'LBP (M)', 'LOA (M)'],
                            'LBP (M)': ['DWT', 'Np', 'Nv', 'Vs'],
                            'LOA (M)': ['DWT', 'Np', 'Nv', 'Vs'],
                            'DWT': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'Vs'],
                            'Np': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT', 'Nv', 'Vs'],
                            'Nv': ['BEAM (M)', 'DEPTH (M)', 'DRAFT (M)', 'LBP (M)', 'LOA (M)', 'DWT', 'Np', 'Vs']}

        # datasets = {"dual fuel LNG": "dual_fuel_LNG", "Bio fuel": "pure_Biofuel"}
        datasets = {"dual fuel LNG": "dual_fuel_LNG"}


    return bins,variables,pred_constraints, filepath, save_dir, datasets

if __name__ == '__main__':

    types_list = [ShipType.BULK_CARRIER,ShipType.TANKER,ShipType.CONTAINER_SHIP, ShipType.RO_PAX]

    types_list = [ShipType.CONTAINER_SHIP]
    # types_list = [ShipType.TANKER, ShipType.RO_PAX]

    base_dir = Path(
        r"C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\dataset_storage_and_testing\processing_checks")

    results = []
    pre_proc_results = []

    version = "refinement7_mcr_in_nearfilter"

    for shiptype in types_list:
        bins, variables, target_predictor_constraints, filepath, save_dir, datasets = get_variables(shiptype)

        max_build_year = 1990

        fuel_type_columns = ['MAIN_FUEL',"BUNKER_FUEL"]
        conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

        alt_fuel = ['LNG', 'Methanol','Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
        bio_fuels = ['Biofuel']
        alt_fuels = {'Alt_fuel': alt_fuel,
                     "biofuel": bio_fuels}


        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=shiptype.string,
                                       fuel_type_columns=fuel_type_columns,
                                       alt_fuels=alt_fuels,
                                       conventional_fuels=conventional_fuels,
                                       max_build_year=max_build_year,
                                       outliers=False,)


        path_dir = base_dir / shiptype.string

        path_dir.mkdir(parents=True, exist_ok=True)

        pre_processing_sheets = {"raw":fleet.raw_data["raw1"],
                          "missing_data":fleet.processed_data["processed_missing_data"],
                          "sister_ship_1":fleet.processed_data["processed_sister_ships_hull"],
                          "sister_ship_2":fleet.processed_data["pre_processed_data"],
                            "sister_ship_candidates":fleet.processed_data["sister_ship_candidates"],
                            "inspected_candidates":fleet.processed_data['inspected_sister_ship_candidates'],
                            "'builder_review_analysis'":fleet.processed_data['builder_review_analysis'],
                                 }

        fuel_data_sheets = {"dual fuel LNG":fleet.processed_data["dual_fuel_LNG"],
                "pure LNG":fleet.processed_data["pure_LNG"],
                "dual fuel Methanol":fleet.processed_data["dual_fuel_Methanol"],
                "pure Methanol":fleet.processed_data["pure_Methanol"],
                "any_ethanol":fleet.processed_data["any_Ethanol"],
                "any_ammonia":fleet.processed_data["any_Ammonia"],
                "any_hydrogen":fleet.processed_data["any_Hydrogen"],
                "any_cng":fleet.processed_data["any_cng"],
                "any_ethane":fleet.processed_data["any_ethane"],
                "any_Biofuel":fleet.processed_data["any_Biofuel"],
                "conventional":fleet.processed_data["pure_conventional"],}

        with pd.ExcelWriter(path_dir/f"{shiptype.string}_preprocessing_{version}.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in pre_processing_sheets.items():
                df.to_excel(writer, sheet_name = sheet_name, index = False)

        with pd.ExcelWriter(path_dir/f"{shiptype.string}_fuel_data_{version}.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in fuel_data_sheets.items():
                df.to_excel(writer, sheet_name = sheet_name, index = False)

        pre_processing_res = {"shiptype":shiptype.string,
                            "raw": len(fleet.raw_data["raw1"]),
                            "missing_data": len(fleet.processed_data["processed_missing_data"]),
                            "sister_ship_1": len(fleet.processed_data["processed_sister_ships_hull"]),
                             "sister_ship_2": len(fleet.processed_data["sister_ship_maker_new"])
                               }

        fuel_data = {"shiptype":shiptype.string,
                "dual fuel LNG":len(fleet.processed_data["dual_fuel_LNG"]),
                "pure LNG":len(fleet.processed_data["pure_LNG"]),
                "dual fuel Methanol":len(fleet.processed_data["dual_fuel_Methanol"]),
                "pure Methanol":len(fleet.processed_data["pure_Methanol"]),
                "any_ethanol":len(fleet.processed_data["any_Ethanol"]),
                "any_ammonia":len(fleet.processed_data["any_Ammonia"]),
                "any_hydrogen":len(fleet.processed_data["any_Hydrogen"]),
                "any_cng":len(fleet.processed_data["any_cng"]),
                "any_ethane":len(fleet.processed_data["any_ethane"]),
                "any_Biofuel":len(fleet.processed_data["any_Biofuel"]),
                "conventional":len(fleet.processed_data["pure_conventional"]),}

        results.append(fuel_data)
        pre_proc_results.append(pre_processing_res)

    ship_fuel_data = pd.DataFrame(results)
    preproc_result_data = pd.DataFrame(pre_proc_results)

    analysis_path = base_dir / version
    analysis_path.mkdir(parents=True, exist_ok=True)

    ship_fuel_data.to_excel(analysis_path/f"fuel_data_final_{version}.xlsx")
    preproc_result_data.to_excel(analysis_path/f"preprocessing_results_final_{version}.xlsx")




