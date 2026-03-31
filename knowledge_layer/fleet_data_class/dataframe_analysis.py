"""
Class containing functions for analysng the content of dataframes
"""


from typing import List
import pandas as pd
from config import LoggerMixin

class DataframeAnalysis(LoggerMixin):

    def __init__(self,
                 debug:bool=False,
                 verbose:bool=True):
        self.debug = debug
        self.verbose = verbose

    def get_dimension_summary(self,
                              columns: List[str],
                              datasets: str | List[str] | None = None,
                              include_processed: bool = True,
                              include_raw: bool = True) -> pd.DataFrame:
        """
        Create a summary DataFrame showing min/max ranges for specified columns across datasets

        :param columns: List of column names to analyze (e.g., ['GT', 'DWT', 'LOA'])
        :param datasets: Specific dataset names to analyze. If None, analyzes all available datasets
        :param include_processed: Whether to include processed datasets in analysis
        :param include_raw: Whether to include raw datasets in analysis

        :return: DataFrame with columns: ['dataset', 'min_{col}', 'max_{col}', ...] for each specified column

        :raises ValueError: If no datasets match the criteria or columns don't exist
        """

        # Collect datasets to analyze
        datasets_to_analyze = {}

        if include_raw:
            datasets_to_analyze.update({f"{name}": df for name, df in self.raw_data.items()})

        if include_processed:
            datasets_to_analyze.update({f"{name}": df for name, df in self.processed_data.items()})

        # Filter to specific datasets if requested
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]

            filtered_datasets = {}
            for dataset_name in datasets:
                if dataset_name in datasets_to_analyze:
                    filtered_datasets[dataset_name] = datasets_to_analyze[dataset_name]
                else:
                    available = list(datasets_to_analyze.keys())
                    raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

            datasets_to_analyze = filtered_datasets

        if not datasets_to_analyze:
            raise ValueError("No datasets available for analysis")

        # Prepare result data
        summary_data = []

        for dataset_name, df in datasets_to_analyze.items():
            row_data = {'dataset': dataset_name}

            # Check which columns exist in this dataset
            missing_columns = []
            for col in columns:
                if col not in df.columns:
                    missing_columns.append(col)
                    continue

                # Calculate min and max for numeric columns only
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Handle NaN values by excluding them
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        row_data[f'min_{col}'] = valid_data.min()
                        row_data[f'max_{col}'] = valid_data.max()
                    else:
                        row_data[f'min_{col}'] = None
                        row_data[f'max_{col}'] = None
                else:
                    self.logger.warning(f"Column '{col}' in dataset '{dataset_name}' is not numeric, skipping")
                    row_data[f'min_{col}'] = None
                    row_data[f'max_{col}'] = None

            if missing_columns:
                self.logger.warning(f"Dataset '{dataset_name}' missing columns: {missing_columns}")

            summary_data.append(row_data)

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Ensure consistent column order: dataset first, then min/max pairs for each dimension
        ordered_columns = ['dataset']
        for col in columns:
            ordered_columns.extend([f'min_{col}', f'max_{col}'])

        # Only include columns that exist in the summary_df
        existing_columns = [col for col in ordered_columns if col in summary_df.columns]
        summary_df = summary_df[existing_columns]

        self.logger.info(f"Generated dimension summary for {len(summary_data)} datasets and {len(columns)} dimensions")

        return summary_df


    def get_available_numeric_columns(self,
                                      datasets: str | List[str] | None = None,
                                      include_processed: bool = True,
                                      include_raw: bool = True) -> dict[str, List[str]]:
        """
        Helper function to identify available numeric columns across datasets

        :param datasets: Specific dataset names to check. If None, checks all available datasets
        :param include_processed: Whether to include processed datasets
        :param include_raw: Whether to include raw datasets

        :return: Dictionary mapping dataset names to lists of their numeric columns
        """

        # Collect datasets to analyze (same logic as get_dimension_summary)
        datasets_to_analyze = {}

        if include_raw:
            datasets_to_analyze.update({f"{name}": df for name, df in self.raw_data.items()})

        if include_processed:
            datasets_to_analyze.update({f"{name}": df for name, df in self.processed_data.items()})

        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]

            filtered_datasets = {}
            for dataset_name in datasets:
                if dataset_name in datasets_to_analyze:
                    filtered_datasets[dataset_name] = datasets_to_analyze[dataset_name]

            datasets_to_analyze = filtered_datasets

        # Find numeric columns for each dataset
        numeric_columns_by_dataset = {}

        for dataset_name, df in datasets_to_analyze.items():
            numeric_cols = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)

            numeric_columns_by_dataset[dataset_name] = numeric_cols

        return numeric_columns_by_dataset