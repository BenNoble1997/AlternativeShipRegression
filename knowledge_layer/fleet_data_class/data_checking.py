import pandas as pd
import numpy as np

from knowledge_layer.display_functions.LaTeX_table_utils import LaTeXFormatter, LaTeXPreambleGenerator

class DataChecking:

    def __init__(self,
                 datasets:dict[str,pd.DataFrame],
                 variables:list[str],):

        self.datasets = datasets
        self.variables = variables
        self.checked_data = DataChecking.data_checking(variables=self.variables,
                                                       data=self.datasets)

    @staticmethod
    def available_data(data: pd.DataFrame,
                       variables: list[str],
                       warn_missing: bool = False):
        """
        Calculate availability statistics for each variable.

        Args:
            data: DataFrame to check
            variables: List of variable names
            warn_missing: If True, print warning for missing columns

        Returns:
            dict mapping variable to (count_available, percent_available)

        Concept: Gracefully handles schema differences while optionally
                 alerting user to unexpected missing columns
        """
        data_available = {}
        data_len = len(data)

        # Optionally identify missing columns
        missing_columns = [v for v in variables if v not in data.columns]
        if warn_missing and missing_columns:
            print(f"Warning: Columns not found in dataset: {missing_columns}")

        for variable in variables:
            if variable not in data.columns:
                data_available[variable] = (0, 0.0)
            else:
                no_datapoints = data[variable].notna().sum()
                percent = (no_datapoints / data_len) * 100 if data_len > 0 else 0.0
                data_available[variable] = (no_datapoints, percent)

        return data_available

    @staticmethod
    def data_checking(data: dict[str, pd.DataFrame],
                      variables: list[str],
                      warn_missing: bool = False):
        """
        Check data availability across multiple datasets.

        Args:
            data: Dict of dataset_name -> DataFrame
            variables: Variables to check
            warn_missing: If True, warn about missing columns

        Concept: Applies availability checking consistently across all datasets,
                 handling schema variations gracefully
        """
        checked_data = {}

        for name, df in data.items():
            data_available = DataChecking.available_data(df, variables, warn_missing)
            checked_data[name] = data_available

        return checked_data

    def create_availability_dataframe(self,
                                      metric: str = 'percent') -> pd.DataFrame:
        """
        Convert nested availability dict to DataFrame for easy analysis.

        Args:
            metric: 'count', 'percent', or 'both'

        Returns:
            DataFrame with variables as rows, datasets as columns

        Concept: Transforms nested dict into tabular format where:
            - Each row = one variable
            - Each column = one dataset
            - Values = availability metric
        """
        checked_data = self.checked_data

        # Extract all unique variables across datasets
        all_variables = set()
        for availability in checked_data.values():
            all_variables.update(availability.keys())
        all_variables = sorted(all_variables)

        if metric == 'both':
            # Create MultiIndex columns for count and percent
            data = {}
            for dataset_name, availability in checked_data.items():
                data[f"{dataset_name}_count"] = [
                    availability.get(var, (None, None))[0] for var in all_variables
                ]
                data[f"{dataset_name}_percent"] = [
                    availability.get(var, (None, None))[1] for var in all_variables
                ]

            df = pd.DataFrame(data, index=all_variables)
            df.index.name = 'Variable'

        else:
            # Single metric per dataset
            idx = 0 if metric == 'count' else 1
            data = {}
            for dataset_name, availability in checked_data.items():
                data[dataset_name] = [
                    availability.get(var, (None, None))[idx] for var in all_variables
                ]

            df = pd.DataFrame(data, index=all_variables)
            df.index.name = 'Variable'

        return df

    def display_availability(self,
                             metric: str = 'percent',
                             sort_by: str = None,
                             ascending: bool = False) -> pd.DataFrame:
        """
        Create and display formatted availability DataFrame.

        Args:
            metric: 'count', 'percent', or 'both'
            sort_by: Column name to sort by, or None for variable name
            ascending: Sort order

        Returns:
            Formatted DataFrame for display and further analysis

        Concept: Combines conversion with pandas display formatting for
                 immediate readable output
        """
        df = self.create_availability_dataframe(metric=metric)

        # Sort if requested
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        elif sort_by is None:
            df = df.sort_index()

        return df


    def create_summary_stats(self) -> pd.DataFrame:
        """
        Create summary statistics DataFrame for each dataset.

        Returns:
            DataFrame with datasets as rows and summary metrics as columns

        Concept: Aggregates variable-level data into dataset-level metrics
                 by computing statistics across all variables in each dataset
        """
        summary_data = []

        for dataset_name, availability in self.checked_data.items():
            if not availability:
                continue

            counts = [count for count, _ in availability.values()]
            percentages = [percent for _, percent in availability.values()]

            summary_data.append({
                'Dataset': dataset_name,
                'Variables': len(availability),
                'Avg %': np.mean(percentages),
                'Min %': np.min(percentages),
                'Max %': np.max(percentages),
                'Std %': np.std(percentages),
                'Total Records': counts[0] if counts else None  # Assuming same for all vars
            })

        df = pd.DataFrame(summary_data).set_index('Dataset')
        return df


class LatexAvailabilityTable:
    """
    Generate LaTeX tables showing data availability across ship types and fuel categories.

    Concept: This class acts as a presentation layer that takes processed data
             from DataChecking instances and formats it into publication-ready
             LaTeX tables with multi-level headers.
    """

    def __init__(self,
                 ship_type_data: dict[str, dict[str, pd.DataFrame]],
                 variables: list[str],
                 metrics: list[str] = ['count', 'percent'],
                 caption: str = "Data availability across ship types and fuel categories",
                 label: str = "tab:data_availability"):
        """
        Initialize LaTeX table generator.

        Args:
            ship_type_data: Nested dict structure
            variables: List of variable names to check
            metrics: List of metrics to include. Options: 'count', 'percent'
                     Default: ['count', 'percent']
                     Use ['count'] for count-only table
            caption: LaTeX table caption
            label: LaTeX table label for referencing
        """
        self.ship_type_data = ship_type_data
        self.variables = variables
        self.metrics = metrics
        self.caption = caption
        self.label = label

        # Validate metrics
        valid_metrics = {'count', 'percent'}
        if not set(metrics).issubset(valid_metrics):
            raise ValueError(f"metrics must be subset of {valid_metrics}")

        # Composition: Create formatter for consistent styling
        self.formatter = LaTeXFormatter()

        # Create DataChecking instances
        self.checkers = self._create_checkers()

        # Calculate columns per fuel category based on metrics
        self.cols_per_fuel_category = len(self.metrics)

    def _create_checkers(self) -> dict[str, dict[str, DataChecking]]:
        """
        Create DataChecking instances for each ship type and fuel category.

        Returns:
            Nested dict of DataChecking instances matching ship_type_data structure

        Concept: This separates data processing (DataChecking) from presentation
                 (this class), following single responsibility principle
        """
        checkers = {}

        for ship_type, fuel_categories in self.ship_type_data.items():
            checkers[ship_type] = {}

            for fuel_category, df in fuel_categories.items():
                # Create DataChecking instance for this combination
                dataset_dict = {f"{ship_type}_{fuel_category}": df}
                checker = DataChecking(
                    datasets=dataset_dict,
                    variables=self.variables
                )
                checkers[ship_type][fuel_category] = checker

        return checkers

    def _count_columns(self) -> int:
        """
        Calculate total number of columns needed.

        Returns:
            Total columns (1 for variables + 2 per fuel category per ship type)

        Concept: Each fuel category gets 2 columns (No and %), so:
                 total = 1 + (num_ship_types × num_fuel_categories × 2)
        """
        num_ship_types = len(self.ship_type_data)
        # Assume all ship types have same fuel categories
        num_fuel_categories = len(next(iter(self.ship_type_data.values())))

        return 1 + (num_ship_types * num_fuel_categories * 2)

    def _generate_column_spec(self) -> str:
        """
        Generate LaTeX column specification string.

        Returns:
            String like "p{0.1\textwidth}p{0.025\textwidth}..."

        Concept: First column (variables) is wider, data columns use 0.025\textwidth
                    to match the compact style
        """
        # Count total columns needed
        num_ship_types = len(self.ship_type_data)
        num_fuel_categories = len(next(iter(self.ship_type_data.values())))
        cols_per_fuel = self.cols_per_fuel_category

        total_data_cols = num_ship_types * num_fuel_categories * cols_per_fuel

        # First column for variable names (0.1 width)
        spec = "p{0.1\\textwidth}"

        # Remaining columns for data (0.025 width each)
        spec += "p{0.025\\textwidth}" * total_data_cols

        return spec

    def _generate_ship_type_header(self) -> str:
        """
        Generate first header row with ship type names and cmidrules.

        Returns:
            Multi-line string with header and cmidrules
        """
        header_parts = [""]  # Empty first cell

        cmidrule_parts = []
        col_start = 2  # Start after variable column

        for ship_type, fuel_categories in self.ship_type_data.items():
            num_cols = len(fuel_categories) * self.cols_per_fuel_category
            header_parts.append(f"  \\multicolumn{{{num_cols}}}{{c}}{{{ship_type}}}")

            # Add cmidrule for this ship type
            col_end = col_start + num_cols - 1
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
            col_start = col_end + 1

        header_line = "& ".join(header_parts) + "\\\\"
        cmidrule_line = "".join(cmidrule_parts)

        return f"    {header_line}\n        {cmidrule_line}"
    def _generate_fuel_category_header(self) -> str:
        """
        Generate second header row with fuel category names and cmidrules.

        Returns:
            Multi-line string with header and cmidrules
        """
        header_parts = [""]  # Empty first cell

        cmidrule_parts = []
        col_start = 2

        for ship_type, fuel_categories in self.ship_type_data.items():
            for fuel_category in fuel_categories.keys():
                num_cols = self.cols_per_fuel_category
                header_parts.append(f"  \\multicolumn{{{num_cols}}}{{c}}{{{fuel_category}}}")

                # Add cmidrule for this fuel category
                col_end = col_start + num_cols - 1
                cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
                col_start = col_end + 1

        header_line = "& ".join(header_parts) + "\\\\"
        cmidrule_line = "".join(cmidrule_parts)

        return f"    {header_line}\n        {cmidrule_line}"

    def _generate_column_header(self) -> str:
        """
        Generate third header row with column names based on selected metrics.

        Returns:
            LaTeX string like "Variables& No& \%& No& \%& ..."
        """
        header_parts = ["  Variables"]

        for ship_type, fuel_categories in self.ship_type_data.items():
            for _ in fuel_categories.keys():
                for metric in self.metrics:
                    if metric == 'count':
                        header_parts.append("  No")
                    elif metric == 'percent':
                        header_parts.append("  \\%")

        return "& ".join(header_parts) + "\\\\"

    def _get_availability_data(self,
                               ship_type: str,
                               fuel_category: str,
                               variable: str) -> tuple[int, float]:
        """
        Extract availability data for specific combination.

        Returns:
            Tuple of (count, percentage)

        Concept: Navigates nested structure to get data from appropriate DataChecking instance
        """
        checker = self.checkers[ship_type][fuel_category]
        dataset_name = f"{ship_type}_{fuel_category}"

        # Get the availability data for this variable
        availability = checker.checked_data[dataset_name].get(variable, (None, None))

        return availability

    def _generate_data_rows(self) -> str:
        """
        Generate all data rows for each variable (no separators between rows).

        Returns:
            LaTeX string with one row per variable
        """
        rows = []

        for variable in self.variables:
            row_parts = []

            # Format variable name using LaTeX formatter
            formatted_var = self.formatter.format_variable_name(variable)
            row_parts.append(f" {formatted_var}")

            for ship_type, fuel_categories in self.ship_type_data.items():
                for fuel_category in fuel_categories.keys():
                    count, percent = self._get_availability_data(
                        ship_type, fuel_category, variable
                    )

                    # Add data based on selected metrics
                    for metric in self.metrics:
                        if metric == 'count':
                            if count is not None and count > 0:
                                row_parts.append(f"  {count}")
                            else:
                                row_parts.append("  ---")
                        elif metric == 'percent':
                            if count is not None and count > 0:
                                row_parts.append(f"  {percent:.1f}")
                            else:
                                row_parts.append("  ---")

            rows.append("& ".join(row_parts) + "\\\\")

        return "\n".join(rows)

    def generate_table(self) -> str:
        """
        Generate complete LaTeX table matching the exact format.

        Returns:
            Complete LaTeX table as string
        """
        column_spec = self._generate_column_spec()
        ship_type_header = self._generate_ship_type_header()
        fuel_category_header = self._generate_fuel_category_header()
        column_header = self._generate_column_header()
        data_rows = self._generate_data_rows()

        # Assemble table
        table = f"""\\begin{{table}}[ht]
            \\centering
            \\caption{{{self.caption}}}
            \\footnotesize
            \\begin{{tabular}}{{{column_spec}}}
            \\toprule
    {ship_type_header}
    {fuel_category_header}
    {column_header}
            \\midrule
    {data_rows}
            \\bottomrule
            \\end{{tabular}}
            \\label{{{self.label}}}
        \\end{{table}}"""

        return table

    def save_table(self, filepath: str) -> None:
        """
        Save generated LaTeX table to file.

        Args:
            filepath: Path to save .tex file

        Concept: Convenience method for direct file output
        """
        table = self.generate_table()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table)

        print(f"LaTeX table saved to {filepath}")

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame showing availability across all combinations.

        Returns:
            MultiIndex DataFrame with ship types and fuel categories

        Concept: Provides a pandas view of the same data for quick analysis
                 before generating LaTeX
        """
        data = []

        for variable in self.variables:
            row_data = {'Variable': variable}

            for ship_type, fuel_categories in self.ship_type_data.items():
                for fuel_category in fuel_categories.keys():
                    count, percent = self._get_availability_data(
                        ship_type, fuel_category, variable
                    )

                    col_name = f"{ship_type}_{fuel_category}"
                    row_data[f"{col_name}_count"] = count
                    row_data[f"{col_name}_percent"] = percent

            data.append(row_data)

        df = pd.DataFrame(data)
        df = df.set_index('Variable')

        return df