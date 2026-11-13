
"""
Collinearity diagnostics for ship dimension datasets.

Reports, for a given dataset and feature set:
  - Pearson correlation matrix
  - Variance inflation factors (VIF)
  - Eigenvalue spectrum of the correlation matrix, with PC1 variance share
  - Condition number

Intended use is comparing feature sets across vessel types, e.g. whether
the principal dimensions of container ships are less collinear than those
of tankers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# core computation
# ---------------------------------------------------------------------------

def collinearity_report(
    data: pd.DataFrame,
    features: list[str],
    name: str = "dataset",
    dropna: bool = True,
) -> dict:
    """
    Compute collinearity diagnostics for `features` in `data`.

    Returns a dict with the correlation matrix, VIFs, eigenvalues,
    PC1 variance share, condition number and the sample size used.
    """
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise KeyError(f"{name}: features not in dataframe: {missing}")

    X = data[features].apply(pd.to_numeric, errors="coerce")
    n_before = len(X)
    if dropna:
        X = X.dropna()
    n = len(X)

    if n <= len(features):
        raise ValueError(
            f"{name}: only {n} complete rows for {len(features)} features; "
            "diagnostics need n > p and are unstable unless n >> p."
        )

    # Standardise. Correlation-matrix eigenvalues are scale free, which
    # matters because DWT (tonnes) and beam (metres) differ by orders of
    # magnitude and a covariance-based PCA would be dominated by DWT.
    Z = (X - X.mean()) / X.std(ddof=1)

    corr = X.corr()

    # VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing feature j
    # on all the others. It answers "how well is this feature predicted by
    # the rest?", which pairwise correlations cannot capture when the
    # dependence involves three or more features at once.
    vifs = {}
    for j, f in enumerate(features):
        y = Z[f].values
        others = Z.drop(columns=[f]).values
        A = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vifs[f] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)

    # Eigenvalues of the correlation matrix. With p standardised features
    # they sum to p, so each eigenvalue divided by p is the share of total
    # variance carried by that principal component.
    eigvals = np.linalg.eigvalsh(corr.values)[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    p = len(features)
    pc_share = eigvals / p

    # Condition number: sqrt(largest / smallest eigenvalue). Above roughly
    # 30 is conventionally treated as serious collinearity.
    smallest = eigvals[-1]
    cond = np.inf if smallest <= 1e-12 else float(np.sqrt(eigvals[0] / smallest))

    # Effective dimensionality: how many components to reach 95% of variance.
    n_components_95 = int(np.searchsorted(np.cumsum(pc_share), 0.95) + 1)

    return {
        "name": name,
        "features": list(features),
        "n": n,
        "n_dropped": n_before - n,
        "corr": corr,
        "vif": pd.Series(vifs),
        "eigenvalues": eigvals,
        "pc_share": pc_share,
        "pc1_share": float(pc_share[0]),
        "condition_number": cond,
        "n_components_95": n_components_95,
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_report(rep: dict, decimals: int = 3) -> None:
    """Print a single collinearity report."""
    w = 78
    print("=" * w)
    print(f"COLLINEARITY: {rep['name']}")
    print("=" * w)
    print(f"  n = {rep['n']}  (rows dropped for missingness: {rep['n_dropped']})")
    print(f"  p = {len(rep['features'])}")
    print()

    print("Correlation matrix")
    print("-" * w)
    print(rep["corr"].round(decimals).to_string())
    print()

    print("Variance inflation factors")
    print("-" * w)
    vif = rep["vif"].sort_values(ascending=False)
    for f, v in vif.items():
        flag = ""
        if v > 100:
            flag = "  <-- severe"
        elif v > 10:
            flag = "  <-- high"
        print(f"  {f:<24s} {v:>12.1f}{flag}")
    print()

    print("Eigenvalue spectrum of the correlation matrix")
    print("-" * w)
    cum = np.cumsum(rep["pc_share"])
    for i, (ev, sh, cs) in enumerate(zip(rep["eigenvalues"], rep["pc_share"], cum), 1):
        print(f"  PC{i}   eigenvalue {ev:>8.4f}   share {sh:>7.3%}   cumulative {cs:>7.3%}")
    print()

    print("Summary")
    print("-" * w)
    print(f"  PC1 variance share          {rep['pc1_share']:.3%}")
    print(f"  Components for 95% variance {rep['n_components_95']} of {len(rep['features'])}")
    cond = rep["condition_number"]
    cond_str = "inf" if not np.isfinite(cond) else f"{cond:.1f}"
    verdict = (
        "severe" if not np.isfinite(cond) or cond > 100
        else "serious" if cond > 30
        else "moderate" if cond > 10
        else "low"
    )
    print(f"  Condition number            {cond_str}  ({verdict})")
    print()


def compare(reports: list[dict], decimals: int = 3) -> pd.DataFrame:
    """
    Build and print a side-by-side summary of several reports.

    Use this for the cross-vessel-type comparison: one report per dataset,
    same or comparable feature sets.
    """
    rows = []
    for rep in reports:
        cond = rep["condition_number"]
        rows.append({
            "dataset": rep["name"],
            "n": rep["n"],
            "p": len(rep["features"]),
            "PC1 share": rep["pc1_share"],
            "PC1+PC2 share": float(rep["pc_share"][:2].sum()),
            "n comp. 95%": rep["n_components_95"],
            "cond. number": cond,
            "max VIF": rep["vif"].max(),
            "mean VIF": rep["vif"].mean(),
        })
    df = pd.DataFrame(rows).set_index("dataset")

    w = 78
    print("=" * w)
    print("COMPARISON")
    print("=" * w)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(df.to_string())
    print()
    print("  Higher PC1 share, higher condition number and higher VIFs all")
    print("  indicate stronger collinearity, i.e. fewer effective dimensions")
    print("  of variation among the features.")
    print()
    return df


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------

def _normalise_entry(name, entry):
    """
    Accept several shapes for a datasets{} value and return (df, features).

    Supported:
        df                      -> all numeric columns used as features
        (df, features)          -> as documented
        (df, features, ...)     -> extra elements ignored, with a warning
    """
    if isinstance(entry, pd.DataFrame):
        feats = entry.select_dtypes(include=[np.number]).columns.tolist()
        if not feats:
            raise ValueError(f"{name}: no numeric columns to use as features.")
        print(f"  [note] {name}: no feature list given, using all "
              f"{len(feats)} numeric columns: {feats}")
        return entry, feats

    if isinstance(entry, (tuple, list)):
        if len(entry) < 2:
            raise ValueError(
                f"{name}: expected (dataframe, features), got {len(entry)} item(s)."
            )
        df, feats = entry[0], entry[1]
        if len(entry) > 2:
            print(f"  [note] {name}: ignoring {len(entry) - 2} extra item(s) "
                  "after (dataframe, features).")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{name}: first item should be a DataFrame, got {type(df).__name__}."
            )
        if isinstance(feats, str):
            feats = [feats]
        return df, list(feats)

    raise TypeError(
        f"{name}: value should be a DataFrame or a (DataFrame, features) tuple, "
        f"got {type(entry).__name__}."
    )


def analyse(
    datasets: dict,
    verbose: bool = True,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Run diagnostics on several datasets and print everything.

    `datasets` maps a display name to (dataframe, feature list), e.g.

        analyse({
            "Tanker (hull only)":    (tanker_conv,    HULL),
            "Container (hull only)": (container_conv, HULL),
            "Container (with TEU)":  (container_conv, HULL + ["TEU"]),
        })
    """
    reports = []
    for name, entry in datasets.items():
        df, feats = _normalise_entry(name, entry)
        rep = collinearity_report(df, feats, name=name)
        reports.append(rep)
        if verbose:
            print_report(rep, decimals=decimals)
    return compare(reports, decimals=decimals)

if __name__ == "__main__":

    from knowledge_layer.dataset_storage_and_testing.alt_fuel_datasets import getting_alt_fuel_ships, get_variables
    from parameters.shipshape.ShipShapeBasic_class import ShipType

    min_year = 1990
    match_number = 5
    caliper = None

    fuel_type_columns = ['MAIN_FUEL', 'AUX_FUEL', "BUNKER_FUEL"]
    alt_fuels = ['LNG', 'Methanol', 'Ethanol', 'Ammonia', 'Hydrogen', 'cng', 'ethane']
    bio_fuels = ['biofuel']

    alt_fuels = {'Alt_fuel': alt_fuels,
                 'Biofuel': bio_fuels}

    conventional_fuels = ['DIESEL', 'HFO', 'MDO', 'MGO', 'IFO']

    datasets = {"Dual fuel": "dual_fuel_Alt_fuel",
                "Biofuel": "pure_Biofuel"}

    selector_suffix = '_with_outliers_basic'

    types_list = [ShipType.BULK_CARRIER,ShipType.CONTAINER_SHIP, ShipType.RO_PAX,ShipType.TANKER]
    #
    #
    #
    alt_datasets = ["dual_fuel_LNG", "dual_fuel_Methanol"]
    #
    # "dual fuel LNG": "dual_fuel_LNG", "Bio fuel": "pure_Biofuel"

    year = 1990

    # main(fuel_type_columns=fuel_type_columns,
    #      make_plots=False,
    #      block1=False,
    #      matching_sensitivity=False,
    #      progressive_comparison=False,
    #      multi_ship_comparison=True,
    #      run_diagnostics=False,
    #      alt_datasets=alt_datasets,
    #      match_type='full',
    #      min_date=year)

    #
    ship_data = {}

    for shiptype in types_list:
        bins, variables, target_predictor_constraints, filepath, save_dir, datasets = get_variables(shiptype=shiptype)


        # if shiptype == ShipType.RO_PAX:
        #     variables.remove('Np')
        #     variables.remove('Nv')

        fleet = getting_alt_fuel_ships(filepath=filepath,
                                       shiptype=shiptype.string,
                                       fuel_type_columns=fuel_type_columns,
                                       max_build_year=min_year,
                                       alt_fuels=alt_fuels,
                                       outliers=False,
                                       conventional_fuels=conventional_fuels)

        variables.remove("LOA (M)")

        ship_data[shiptype.name] = (fleet.processed_data["pure_conventional"], variables)

    analyse(datasets=ship_data, verbose=True, decimals=3)