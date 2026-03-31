"""
Source Dataset Manager - Filters datasets per literature source

Key Concept:
    Literature studies have inclusion criteria that apply to ALL formulas
    from that study. Instead of filtering per-formula, we filter per-source
    once and reuse the filtered datasets.

Example:
    Watson (1998) studied bulk carriers with DWT 10,000-100,000
    → All Watson formulas should use the same filtered dataset
    → Filter once, report once, evaluate many formulas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# DATA CLASSES FOR STRUCTURED INFORMATION
# ============================================================================

@dataclass
class DomainConstraints:
    """
    Domain constraints for a literature source.

    Attributes
    ----------
    source : str
        Literature reference/source name
    lower_bounds : Dict[str, float]
        Lower bounds per dimension, e.g. {'DWT': 5000, 'LOA (M)': 50}
    upper_bounds : Dict[str, float]
        Upper bounds per dimension, e.g. {'DWT': 50000, 'LOA (M)': 300}
    """
    source: str
    lower_bounds: Dict[str, float] = field(default_factory=dict)
    upper_bounds: Dict[str, float] = field(default_factory=dict)

    def get_all_constrained_dimensions(self) -> set:
        """Get all dimensions that have constraints."""
        return set(self.lower_bounds.keys()) | set(self.upper_bounds.keys())

    def __repr__(self):
        dims = self.get_all_constrained_dimensions()
        ranges = []
        for dim in sorted(dims):
            lower = self.lower_bounds.get(dim, '−∞')
            upper = self.upper_bounds.get(dim, '+∞')
            ranges.append(f"{dim}: [{lower}, {upper}]")
        return f"DomainConstraints({self.source}): {', '.join(ranges)}"


@dataclass
class FilteredDatasetInfo:
    """
    Information about a filtered dataset.

    Tracks how many data points were excluded and why.
    """
    source: str
    dataset_name: str  # 'conventional', 'matched', 'alternative'
    original_size: int
    filtered_size: int
    n_excluded: int
    pct_excluded: float
    exclusions_by_dimension: Dict[str, int] = field(default_factory=dict)

    def __repr__(self):
        return (f"FilteredDatasetInfo({self.source} | {self.dataset_name}): "
                f"{self.filtered_size}/{self.original_size} kept "
                f"({self.pct_excluded:.1f}% excluded)")


@dataclass
class SourceDatasets:
    """
    Filtered datasets for a specific literature source.

    Attributes
    ----------
    source : str
        Literature reference/source name
    domain_constraints : DomainConstraints
        The constraints applied to create these datasets
    datasets : Dict[str, pd.DataFrame]
        Filtered datasets: {'conventional': df, 'matched': df, 'alternative': df}
    filter_info : Dict[str, FilteredDatasetInfo]
        Information about filtering per dataset
    """
    source: str
    domain_constraints: DomainConstraints
    datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    filter_info: Dict[str, FilteredDatasetInfo] = field(default_factory=dict)

    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get a specific filtered dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found for source '{self.source}'")
        return self.datasets[dataset_name]

    def get_filter_info(self, dataset_name: str) -> FilteredDatasetInfo:
        """Get filtering information for a dataset."""
        if dataset_name not in self.filter_info:
            raise ValueError(f"Filter info for '{dataset_name}' not found")
        return self.filter_info[dataset_name]


# ============================================================================
# SOURCE DATASET MANAGER
# ============================================================================

class SourceDatasetManager:
    """
    Manages dataset filtering per literature source.

    Key responsibilities:
    1. Extract domain constraints from formula file (per source)
    2. Filter datasets once per source
    3. Store filtered datasets with metadata
    4. Provide reporting on data quality per source

    Usage
    -----
    >>> # Extract domain info from formula library
    >>> domain_map = SourceDatasetManager.extract_domains_from_library(
    ...     formulae_list
    ... )
    >>>
    >>> # Create manager and filter datasets
    >>> manager = SourceDatasetManager(
    ...     conventional=conv_df,
    ...     matched=matched_df,
    ...     alternative=alt_df
    ... )
    >>> manager.filter_datasets_by_source(domain_map)
    >>>
    >>> # Use in evaluator
    >>> evaluator = LiteratureFormulaEvaluator(
    ...     dataset_manager=manager,
    ...     formulae=formulae_list
    ... )
    """

    def __init__(self,
                 conventional: pd.DataFrame,
                 matched: pd.DataFrame,
                 alternative: pd.DataFrame,
                 verbose: bool = True):
        """
        Initialize manager with raw datasets.

        Parameters
        ----------
        conventional : pd.DataFrame
            Full conventional ship dataset
        matched : pd.DataFrame
            Matched conventional ship dataset
        alternative : pd.DataFrame
            Alternative fuel ship dataset
        verbose : bool
            Print filtering details
        """
        self.raw_datasets = {
            'conventional': conventional.copy(),
            'matched': matched.copy(),
            'alternative': alternative.copy()
        }
        self.verbose = verbose

        # Storage for filtered datasets per source
        self.source_datasets: Dict[str, SourceDatasets] = {}

        # Default: unfiltered datasets for formulas without domain constraints
        self._create_default_source()

    def _create_default_source(self):
        """Create 'default' source with no filtering."""
        default_constraints = DomainConstraints(
            source='default',
            lower_bounds={},
            upper_bounds={}
        )

        default_datasets = SourceDatasets(
            source='default',
            domain_constraints=default_constraints
        )

        # Copy raw datasets
        for name, df in self.raw_datasets.items():
            default_datasets.datasets[name] = df.copy()
            default_datasets.filter_info[name] = FilteredDatasetInfo(
                source='default',
                dataset_name=name,
                original_size=len(df),
                filtered_size=len(df),
                n_excluded=0,
                pct_excluded=0.0,
                exclusions_by_dimension={}
            )

        self.source_datasets['default'] = default_datasets

    @staticmethod
    def extract_domains_from_library(formulae: List['LiteratureFormula']) -> Dict[str, DomainConstraints]:
        """
        Extract domain constraints from a list of formulas, grouped by source.

        Aggregates domain constraints across all formulas from the same source.
        If formulas from same source have different domains, takes the intersection
        (most restrictive).

        Parameters
        ----------
        formulae : List[LiteratureFormula]
            List of formulas with domain information

        Returns
        -------
        Dict[str, DomainConstraints]
            Mapping from source name to domain constraints
        """
        source_domains: Dict[str, DomainConstraints] = {}

        for formula in formulae:
            source = formula.reference if formula.reference else 'default'

            # Initialize if first formula from this source
            if source not in source_domains:
                source_domains[source] = DomainConstraints(
                    source=source,
                    lower_bounds={},
                    upper_bounds={}
                )

            constraints = source_domains[source]

            # Extract domain info from formula
            domain_lower = formula.domain_lower
            domain_upper = formula.domain_upper

            # Convert to dict if needed
            if isinstance(domain_lower, (int, float)):
                # Single predictor - use predictor name
                if len(formula.predictors) == 1:
                    domain_lower = {formula.predictors[0]: float(domain_lower)}
                else:
                    domain_lower = {}

            if isinstance(domain_upper, (int, float)):
                if len(formula.predictors) == 1:
                    domain_upper = {formula.predictors[0]: float(domain_upper)}
                else:
                    domain_upper = {}

            # Merge with existing constraints (take most restrictive)
            if isinstance(domain_lower, dict):
                for dim, value in domain_lower.items():
                    if dim in constraints.lower_bounds:
                        # Take maximum of lower bounds (most restrictive)
                        constraints.lower_bounds[dim] = max(
                            constraints.lower_bounds[dim],
                            value
                        )
                    else:
                        constraints.lower_bounds[dim] = value

            if isinstance(domain_upper, dict):
                for dim, value in domain_upper.items():
                    if dim in constraints.upper_bounds:
                        # Take minimum of upper bounds (most restrictive)
                        constraints.upper_bounds[dim] = min(
                            constraints.upper_bounds[dim],
                            value
                        )
                    else:
                        constraints.upper_bounds[dim] = value

        return source_domains

    def filter_datasets_by_source(self,
                                  domain_map: Dict[str, DomainConstraints]) -> None:
        """
        Filter all datasets according to domain constraints per source.

        Parameters
        ----------
        domain_map : Dict[str, DomainConstraints]
            Mapping from source name to domain constraints
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("FILTERING DATASETS BY SOURCE")
            print("=" * 80)

        for source, constraints in domain_map.items():
            if self.verbose:
                print(f"\nSource: {source}")
                print(f"  Constraints: {constraints}")

            # Create filtered datasets for this source
            source_data = SourceDatasets(
                source=source,
                domain_constraints=constraints
            )

            # Filter each dataset type
            for dataset_name, raw_df in self.raw_datasets.items():
                filtered_df, filter_info = self._filter_single_dataset(
                    dataset=raw_df,
                    dataset_name=dataset_name,
                    constraints=constraints,
                    source=source
                )

                source_data.datasets[dataset_name] = filtered_df
                source_data.filter_info[dataset_name] = filter_info

                if self.verbose:
                    print(f"  {dataset_name}: {filter_info.filtered_size}/{filter_info.original_size} "
                          f"({filter_info.pct_excluded:.1f}% excluded)")
                    if filter_info.exclusions_by_dimension:
                        for dim, count in filter_info.exclusions_by_dimension.items():
                            print(f"    {dim}: {count} excluded")

            # Store filtered datasets
            self.source_datasets[source] = source_data

        if self.verbose:
            print("\n" + "=" * 80)
            self._print_summary()

    def _filter_single_dataset(self,
                               dataset: pd.DataFrame,
                               dataset_name: str,
                               constraints: DomainConstraints,
                               source: str) -> Tuple[pd.DataFrame, FilteredDatasetInfo]:
        """
        Filter a single dataset according to domain constraints.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset to filter
        dataset_name : str
            Name of dataset ('conventional', 'matched', 'alternative')
        constraints : DomainConstraints
            Domain constraints to apply
        source : str
            Source name (for tracking)

        Returns
        -------
        filtered_df : pd.DataFrame
            Filtered dataset
        info : FilteredDatasetInfo
            Information about filtering
        """
        original_size = len(dataset)

        # Start with all rows included
        mask = np.ones(len(dataset), dtype=bool)
        exclusions_by_dim = {}

        # Get all constrained dimensions
        constrained_dims = constraints.get_all_constrained_dimensions()

        # Check which dimensions are missing
        missing_dims = constrained_dims - set(dataset.columns)
        if missing_dims and self.verbose:
            print(f"    Warning: Dimensions {missing_dims} not in {dataset_name} dataset")

        available_dims = constrained_dims & set(dataset.columns)

        # Apply constraints per dimension
        for dim in available_dims:
            dim_mask = np.ones(len(dataset), dtype=bool)
            dim_values = dataset[dim].values

            # Apply lower bound
            if dim in constraints.lower_bounds:
                lower = constraints.lower_bounds[dim]
                dim_mask &= (dim_values >= lower)

            # Apply upper bound
            if dim in constraints.upper_bounds:
                upper = constraints.upper_bounds[dim]
                dim_mask &= (dim_values <= upper)

            # Track exclusions for this dimension
            n_excluded_this_dim = np.sum(~dim_mask)
            if n_excluded_this_dim > 0:
                exclusions_by_dim[dim] = n_excluded_this_dim

            # Update overall mask
            mask &= dim_mask

        # Create filtered dataset
        filtered_df = dataset[mask].copy()

        # Create info object
        n_excluded = np.sum(~mask)
        pct_excluded = 100 * n_excluded / original_size if original_size > 0 else 0

        info = FilteredDatasetInfo(
            source=source,
            dataset_name=dataset_name,
            original_size=original_size,
            filtered_size=len(filtered_df),
            n_excluded=n_excluded,
            pct_excluded=pct_excluded,
            exclusions_by_dimension=exclusions_by_dim
        )

        return filtered_df, info

    def get_datasets_for_source(self, source: str) -> SourceDatasets:
        """
        Get filtered datasets for a specific source.

        Parameters
        ----------
        source : str
            Source/reference name

        Returns
        -------
        SourceDatasets
            Filtered datasets with metadata
        """
        if source not in self.source_datasets:
            # Fall back to default (no filtering)
            if self.verbose:
                print(f"  Warning: No filtered datasets for source '{source}', using default (unfiltered)")
            return self.source_datasets['default']

        return self.source_datasets[source]

    def _print_summary(self):
        """Print summary of filtering across all sources."""
        print("\nFILTERING SUMMARY BY SOURCE")
        print("-" * 80)

        for source, source_data in self.source_datasets.items():
            if source == 'default':
                continue

            print(f"\n{source}:")

            for dataset_name in ['conventional', 'matched', 'alternative']:
                info = source_data.filter_info[dataset_name]
                print(f"  {dataset_name:12s}: {info.filtered_size:5d}/{info.original_size:5d} "
                      f"({info.pct_excluded:5.1f}% excluded)")

    def export_filtering_report(self, output_path: str) -> None:
        """
        Export detailed filtering report to CSV.

        Parameters
        ----------
        output_path : str
            Path for output CSV file
        """
        records = []

        for source, source_data in self.source_datasets.items():
            for dataset_name, info in source_data.filter_info.items():
                record = {
                    'source': source,
                    'dataset': dataset_name,
                    'original_size': info.original_size,
                    'filtered_size': info.filtered_size,
                    'n_excluded': info.n_excluded,
                    'pct_excluded': info.pct_excluded
                }

                # Add dimension-specific exclusions
                for dim, count in info.exclusions_by_dimension.items():
                    record[f'excluded_by_{dim}'] = count

                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"\nFiltering report exported to: {output_path}")