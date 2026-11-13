import re
import numpy as np
from enum import Enum
import pandas as pd
import os
import sys
import logging
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional, Union, Callable, Any
from tqdm import tqdm
from pathlib import Path
from operator import lt, le, gt, ge, eq, ne
from sklearn.preprocessing import Normalizer, MinMaxScaler,StandardScaler, RobustScaler
os.chdir(r"/")


sys.path.append(r"/")

from logger_config import LoggerMixin


class ColumnOperation(Enum):
    """Enum defining available column operations."""
    DIFFERENCE = 'difference'
    SUM = 'sum'
    PRODUCT = 'product'

    @staticmethod
    def get_operation_function(operation):
        """Returns the corresponding function for the given operation."""
        operations = {
            ColumnOperation.DIFFERENCE: lambda x: x[0] - x[1],
            ColumnOperation.SUM: lambda x: np.sum(x, axis=0),
            ColumnOperation.PRODUCT: lambda x: np.prod(x, axis=0)
        }
        return operations.get(operation, None)

class ComparisonOperator(Enum):
    """
    Enumeration of supported comparison operators for filtering operations.

    Each operator is paired with its corresponding Python operator function
    and a human-readable description for logging and error messages.
    """
    LESS_THAN = ('<', lt, 'less than')
    LESS_EQUAL = ('<=', le, 'less than or equal to')
    GREATER_THAN = ('>', gt, 'greater than')
    GREATER_EQUAL = ('>=', ge, 'greater than or equal to')
    EQUAL = ('==', eq, 'equal to')
    NOT_EQUAL = ('!=', ne, 'not equal to')

    def __init__(self, symbol: str, func: Callable, description: str):
        """
        Initialize a comparison operator with its symbol, function, and description.

        Args:
            symbol: The mathematical symbol representing the operator
            func: The Python function implementing the comparison
            description: A human-readable description of the operation
        """
        self.symbol = symbol
        self.func = func
        self.description = description

    @classmethod
    def from_symbol(cls, symbol: str) -> 'ComparisonOperator':
        """
        Get the ComparisonOperator enum member corresponding to a symbol.

        This allows backward compatibility with code that might still use
        string operators.

        Args:
            symbol: The operator symbol (e.g., '<', '>=', '==')

        Returns:
            The corresponding ComparisonOperator enum member

        Raises:
            ValueError: If the symbol doesn't match any operator
        """
        for operator in cls:
            if operator.symbol == symbol:
                return operator
        raise ValueError(
            f"Invalid operator symbol: '{symbol}'. "
            f"Valid operators are: {[op.symbol for op in cls]}"
        )
class AggregationFunction(Enum):
    """
    Enumeration of supported aggregation functions with their implementation details.

    Each enum member contains:
    - name: The string name used in pandas aggregation
    - description: Human-readable description for display/logging
    - function: The actual function to use (if custom implementation needed)
    """


    def weighted_average(series, weights):
        if np.sum(weights) == 0:
            return 0  # or any other default value you prefer
        else:
            return np.average(series, weights=weights)

    SUM = ('sum', 'Sum of values', None)
    MEAN = ('mean', 'Arithmetic mean', None)
    WEIGHTED_MEAN = ('weighted_mean', 'Weighted average', weighted_average)
    COUNT = ('count', 'Count of values', None)
    MIN = ('min', 'Minimum value', None)
    MAX = ('max', 'Maximum value', None)
    STD = ('std', 'Standard deviation', None)
    VAR = ('var', 'Variance', None)
    MEDIAN = ('median', 'Median value', None)
    FIRST = ('first', 'First value', None)
    LAST = ('last', 'Last value', None)

    def __init__(self, string: str,
                 description: str,
                 function: Optional[Callable] = None):
        self.string = string
        self.description = description
        self.function = function

    @staticmethod
    def create_weighted_mean(weight_column: str):
        """Creates a weighted mean function that will be called during groupby aggregation"""

        def weighted_mean(group, parent_df):  # pandas groupby passes the group directly
            # series = group
            # df = group.groupby_parent_df  # we'll store the parent df during aggregation
            weights = parent_df.loc[group.index, weight_column]
            return AggregationFunction.weighted_average(group, weights)

        return weighted_mean


class TransformFunction(Enum):
    """
    Enumeration of supported transform functions with their implementation details.

    Each enum member contains:
    - name: The string identifier
    - description: A human-readable description
    - function: The function to apply
    """

    @staticmethod
    def transform_samplingtime(time_value):
        """Converts a datetime or timedelta object to the total number of minutes represented."""
        # If time_value is a datetime object, calculate minutes
        if isinstance(time_value, datetime):
            return time_value.hour * 60 + time_value.minute

            # If time_value is a time object, calculate minutes

        elif isinstance(time_value, time):

            return time_value.hour * 60 + time_value.minute + time_value.second / 60

        # If time_value is a timedelta object, extract total minutes
        elif isinstance(time_value, pd.Timedelta):
            return time_value.total_seconds() / 60  # Convert total seconds to minutes

        else:
            raise TypeError("Expected a datetime or timedelta object")

    TIME_TO_MINUTES = ('time_to_minutes', 'Time to minutes', transform_samplingtime)  # Directly reference function

    def apply(self, series):
        """Applies the transformation function to a pandas Series."""
        return series.apply(self.value[2])  # Access the function reference

    #Insert more functions here

# class Operators(Enum):
#
#     EQUAL_TO = ('==',eq, 'equal to')
#     NOT_EQUAL_TO = ('!=', ne, 'not equal to')
#     GREATER_EQUAL = ('>=', ge, 'greater than or equal to')
#     GREATER_THAN = ('>', gt, 'greater than')
#     LESS_EQUAL = ('<=', le, 'less than or equal to')
#     LESS_THAN = ('<', lt, 'less than ')
#
#
#     def __init__(self, symbol: str, func: Callable, description: str):
#         """
#         Initialize a comparison operator with its symbol, function, and description.
#
#         Args:
#             symbol: The mathematical symbol representing the operator
#             func: The Python function implementing the comparison
#             description: A human-readable description of the operation
#         """
#         self.symbol = symbol
#         self.func = func
#         self.description = description
#
#
#     @classmethod
#     def from_symbol(cls, symbol: str) -> 'ComparisonOperator':
#         """
#         Get the ComparisonOperator enum member corresponding to a symbol.
#
#         This allows backward compatibility with code that might still use
#         string operators.
#
#         Args:
#             symbol: The operator symbol (e.g., '<', '>=', '==')
#
#         Returns:
#             The corresponding ComparisonOperator enum member
#
#         Raises:
#             ValueError: If the symbol doesn't match any operator
#         """
#         for operator in cls:
#             if operator.symbol == symbol:
#                 return operator
#         raise ValueError(
#             f"Invalid operator symbol: '{symbol}'. "
#             f"Valid operators are: {[op.symbol for op in cls]}"
#         )

class NormalizationMethod(Enum):
    """
    Enumeration of supported normalization methods with their corresponding scalers.

    Each enum member contains:
    - name: String identifier for the method
    - scaler_class: The scikit-learn scaler class to use
    - description: Human-readable description of what the method does
    """
    MINMAX = ('minmax', MinMaxScaler, 'Scales data to a fixed range [0, 1]')
    STANDARD = ('standard', StandardScaler, 'Standardizes data to zero mean and unit variance')
    ROBUST = ('robust', RobustScaler, 'Scales data using statistics robust to outliers')

    def __init__(self, string: str, scaler_class: type, description: str):
        self.string = string
        self.scaler_class = scaler_class
        self.description = description


class DataFileManager(LoggerMixin):
    """
    Handles file discovery and validation operations.

    This class manages all file-related operations including finding files
    in directories and validating file types.
    """

    def __init__(self):
        self.supported_extensions = {
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.csv': pd.read_csv,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet
        }

    def list_files(self, folder_path: str, file_types: List[str] = None) -> List[str]:
        """Find all files of specified types in a directory."""
        if not os.path.exists(folder_path):
            self.logger.error(f'{folder_path} is not a valid path')
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

        if file_types is None:
            file_types = ['xls', 'xlsx']
            self.logger.info(f"No file types specified. Defaulting to: {file_types}")

        files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(tuple(file_types))
        ]

        if len(files) == 0:
            self.logger.warning(f"No files found in '{folder_path}' with extensions: {file_types}")

        self.logger.info(f"Found {len(files)} files in '{folder_path}'.")
        return files

    def get_import_function(self, file_path: str) -> Callable:
        """Determine the appropriate import function based on file extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        import_function = self.supported_extensions.get(file_extension)

        if import_function is None:
            self.logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")

        return import_function


class DataProcessor(LoggerMixin):
    """
    Handles data processing operations including cleaning, column selection,
    and transformations.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        """
        Initialises the Data Processor class

        """
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def clean_data(self,
                   datafile: pd.DataFrame,
                   strip_whitespace: bool = True,
                   capitalize: bool = True,
                   replace_spaces: Union[bool, str] = False,
                   drop_first_row: bool = True,
                   preserve_parentheses_case: bool = True) -> pd.DataFrame:
        """
        Apply configurable cleaning operations to a DataFrame.

        Args:
            datafile: DataFrame to clean
            strip_whitespace: Whether to remove leading/trailing whitespace
            capitalize: Whether to capitalize text
            replace_spaces: If True, replaces spaces with underscores
                           If a string, replaces spaces with that string
            drop_first_row: Whether to drop the first row of data
            preserve_parentheses_case: Whether to preserve case of text inside parentheses

        Returns:
            Cleaned DataFrame

        Example:
            >>> cleaned_df = manipulator.clean_data(
            ...     df,
            ...     capitalize=True,
            ...     replace_spaces='_',
            ...     preserve_parentheses_case=True
            ... )
        """
        try:
            self.logger.info("Starting data cleaning process")
            cleaned_df = datafile.copy()

            # Process column names
            if strip_whitespace or capitalize or replace_spaces:
                # Get original column names
                columns = cleaned_df.columns.tolist()

                # Process each column name
                for i, col in enumerate(columns):
                    if not isinstance(col, str):
                        continue

                    # Handle stripping
                    if strip_whitespace:
                        col = col.strip()

                    # Handle capitalization with parentheses preservation
                    if capitalize:
                        if preserve_parentheses_case:
                            # Find all text within parentheses
                            parentheses_texts = re.findall(r'\([^)]*\)', col)
                            # Temporarily replace them with placeholders
                            for j, ptext in enumerate(parentheses_texts):
                                placeholder = f"__PAREN_{j}__"
                                col = col.replace(ptext, placeholder)

                            # Capitalize the text
                            col = col.upper()

                            # Restore original parentheses text
                            for j, ptext in enumerate(parentheses_texts):
                                placeholder = f"__PAREN_{j}__"
                                col = col.replace(placeholder, ptext)
                        else:
                            col = col.upper()

                    # Handle space replacement
                    if replace_spaces:
                        replacement = '_' if replace_spaces is True else replace_spaces
                        col = col.replace(' ', replacement)

                    columns[i] = col

                # Apply the processed column names
                cleaned_df.columns = columns
                self.logger.info(f"Processed {len(columns)} column names")

            # Drop the first row if requested
            if drop_first_row and not cleaned_df.empty:
                cleaned_df = cleaned_df.iloc[1:]
                self.logger.info("Dropped first row of data")

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def handle_columns_to_keep(self,
                               datafile: pd.DataFrame,
                               columns_to_keep: Optional[Union[List, str, slice]] = None,
                               ) -> pd.DataFrame:
        """
        Handle column selection with support for various input types.
        """
        all_columns = list(datafile.columns)

        if columns_to_keep is not None:
            # Handle programmatic column selection
            try:
                if isinstance(columns_to_keep, (list, np.ndarray)):
                    # Handle list of column names
                    selected_columns = list(columns_to_keep)

                elif isinstance(columns_to_keep, str):
                    # Handle comma-separated string
                    selected_columns = [x.strip() for x in columns_to_keep.split(',')]

                elif isinstance(columns_to_keep, tuple) and len(columns_to_keep) == 2:
                    # Handle column range selection
                    start_col, end_col = columns_to_keep

                    if start_col not in all_columns or end_col not in all_columns:
                        raise ValueError(f"Column names not found: {start_col} or {end_col}")

                    start_idx = all_columns.index(start_col)
                    end_idx = all_columns.index(end_col)

                    # Handle reverse order if needed
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx

                    selected_columns = all_columns[start_idx:end_idx + 1]

                else:
                    raise ValueError(
                        "columns_to_keep must be None, a list of column names, "
                        "a comma-separated string, or a tuple of (start_column, end_column)"
                    )

            except Exception as e:
                logger.error(f"Error processing columns_to_keep: {str(e)}")
                raise

        else:
            self.logger.info("No columns specified, starting interactive column selection")
            print("\nAvailable columns:")
            for i, col in enumerate(all_columns, 1):
                print(f"{i}. {col}")

            print("\nSelect columns by:")
            print("1. Column indices")
            print("2. Column names")
            print("3. Column range")

            choice = input("Enter your choice (1/2/3): ").strip()

            try:
                if choice == '1':
                    # Handle selection by indices
                    user_input = input("Enter column indices (e.g., 1,2,3): ").strip()
                    selected_columns = [all_columns[int(x) - 1] for x in user_input.split(',')]

                elif choice == '2':
                    # Handle selection by names
                    user_input = input("Enter column names (comma-separated): ").strip()
                    selected_columns = [x.strip() for x in user_input.split(',')]

                elif choice == '3':
                    # Handle range selection
                    start_col = input("Enter start column name: ").strip()
                    end_col = input("Enter end column name: ").strip()

                    if start_col not in all_columns or end_col not in all_columns:
                        raise ValueError(f"Column names not found: {start_col} or {end_col}")

                    start_idx = all_columns.index(start_col)
                    end_idx = all_columns.index(end_col)

                    # Handle reverse order if needed
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx

                    selected_columns = all_columns[start_idx:end_idx + 1]
                else:
                    raise ValueError("Invalid selection method")

            except Exception as e:
                self.logger.error(f"Error in column selection: {str(e)}")
                raise

        invalid_columns = [col for col in selected_columns if col not in all_columns]
        if invalid_columns:
            self.logger.error(f"Columns not found: {invalid_columns}")
            raise ValueError(f"Columns not found: {invalid_columns}")

        # Select the columns and return
        result = datafile.loc[:, selected_columns]

        self.logger.debug(f"Selected {len(selected_columns)} columns: {selected_columns}")
        self.logger.info(f"Selected {len(selected_columns)}")

        return result

    def handle_column_renames(self,
                              datafile: pd.DataFrame,
                              column_renames: Optional[Dict[str, str]] = None,
                              verbose: bool = True) -> pd.DataFrame:

        """Handle column renaming with support for interactive input."""
        if column_renames is not None:
            self.logger.info(f"Renamed columns: {column_renames}")
            return datafile.rename(columns=column_renames)

        else:
            self.logger.info("Starting interactive column selection")
            print("Current column names:")
            for i, col in enumerate(datafile.columns, 1):
                print(f"{i}. {col}")

            # Prompt the user to specify columns to rename
            selected_indices = input("Enter column indices to rename (e.g., 1,3,5): ").strip()
            try:
                selected_indices = [int(x) - 1 for x in selected_indices.split(",")]
                selected_columns = [datafile.columns[i] for i in selected_indices]
            except Exception as e:
                raise ValueError(f"Invalid input for column indices: {e}")

            # Prompt for new names for the selected columns
            rename_map = {}
            for col in selected_columns:
                new_name = input(f"Enter new name for '{col}' (or press Enter to skip): ").strip()
                if new_name:
                    rename_map[col] = new_name

            datafile = datafile.rename(columns=rename_map)
            if verbose and rename_map:
                logging.info(f"Renamed columns: {rename_map}")

        return datafile


class DataImporter(LoggerMixin):
    """
    Main class that orchestrates the data import workflow.
    """

    def __init__(self, verbose: bool = True,
                 debug: bool = False):
        self.verbose = verbose
        self.debug = debug
        self.file_manager = DataFileManager()
        self.processor = DataProcessor()

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def import_file(self, file_path: str, import_function: Optional[Callable] = None) -> pd.DataFrame:
        """Import a single data file."""
        if import_function is None:
            import_function = self.file_manager.get_import_function(file_path)

        try:
            self.logger.info(f"Attempting to import file: {file_path}")
            with tqdm(total=100, desc=f"Importing {file_path}", disable=not self.verbose) as pbar:
                datafile = import_function(file_path)
                pbar.update(100)

            if self.verbose:
                self.logger.info(f"Imported {len(datafile)} data points from {file_path}")

            return datafile

        except Exception as e:
            self.logger.error(f"Error importing {file_path}: {e}")
            raise

    def merge_dataframes(self, datafiles: List[pd.DataFrame], merge_on: str) -> pd.DataFrame:
        """Merge multiple DataFrames."""
        if not datafiles:
            raise ValueError("No DataFrames provided for merging.")

        merged_df = datafiles[0]
        for df in datafiles[1:]:
            merged_df = merged_df.merge(df, on=merge_on)

        if self.verbose:
            self.logger.info("DataFrames merged successfully")

        return merged_df


class DataFilter(LoggerMixin):
    """
    Handles data filtering operations.

    requires:

    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)

        elif self.debug:
            self.logger.setLevel(logging.DEBUG)

        else:
            self.logger.setLevel(logging.WARNING)

    def filter_by_value(self,
                        datafile: pd.DataFrame,
                        column_name: Union[str, List[str]],
                        value: float,
                        operator: Union[ComparisonOperator, str] = ComparisonOperator.EQUAL,
                        require_all: bool = False,
                        negate: bool = False) -> pd.DataFrame:
        """
        Filter DataFrame rows based on comparison with a specified value.

        This method supports complex filtering operations with multiple options:
        - Single or multiple column filtering
        - Custom comparison operators
        - AND/OR logic for multiple columns
        - Optional negation of the final condition

        Args:
            datafile: The DataFrame to filter
            column_name: Column(s) to apply the filter to
            value: The value to compare against
            operator: The comparison operator to use (ComparisonOperator enum or string)
            require_all: If True, all columns must meet the condition (AND logic)
                        If False, any column meeting the condition is sufficient (OR logic)
            negate: If True, reverses the final condition (keeps rows that don't match)

        Returns:
            Filtered DataFrame

        Example:
            # Keep rows where either SFOC is positive (equivalent to removing rows where both are negative)
            df = filter_by_value(df, ['ME_SFOC', 'DG_SFOC'], 0,
                               ComparisonOperator.LESS_EQUAL,
                               require_all=True,
                               negate=True)
        """
        # Convert string operator to enum if necessary (for backward compatibility)
        if isinstance(operator, str):
            try:
                operator = ComparisonOperator.from_symbol(operator)
            except ValueError as e:
                self.logger.error(f"Invalid operator provided: {operator}")
                raise

        filtered_data = datafile.copy()
        initial_count = len(filtered_data)

        try:
            # Convert single column to list for consistent processing
            columns = [column_name] if isinstance(column_name, str) else column_name

            # Validate all columns exist before processing
            for col in columns:
                if col not in filtered_data.columns:
                    self.logger.error(f"Column not found: {col}")
                    raise KeyError(f"Column '{col}' not found in DataFrame")

            if len(columns) == 1:
                # Single column processing - simpler logic
                condition = operator.func(filtered_data[columns[0]], value)
                if negate:
                    condition = ~condition
                filtered_data = filtered_data[condition]

                # Log the filtering operation
                rows_removed = initial_count - len(filtered_data)
                self.logger.info(
                    f"Filter: {columns[0]} {'NOT ' if negate else ''}{operator.description} {value} "
                    f"(removed {rows_removed:,} rows, "
                    f"{rows_removed / initial_count:.1%} of data)"
                )

            else:
                # Multiple column processing
                if require_all:
                    # AND logic: combine all conditions
                    condition = pd.Series(True, index=filtered_data.index)
                    for col in columns:
                        condition = condition & operator.func(filtered_data[col], value)

                    # Apply negation if requested
                    if negate:
                        condition = ~condition

                    filtered_data = filtered_data[condition]

                    # Log the combined operation
                    rows_removed = initial_count - len(filtered_data)
                    self.logger.info(
                        f"Filter: {'NOT ' if negate else ''}ALL columns {operator.description} {value} "
                        f"(removed {rows_removed:,} rows, "
                        f"{rows_removed / initial_count:.1%} of data)"
                    )

                else:
                    # OR logic: process columns sequentially
                    for col in columns:
                        condition = operator.func(filtered_data[col], value)
                        if negate:
                            condition = ~condition
                        filtered_data = filtered_data[condition]

                        # Log each column's operation
                        rows_removed = initial_count - len(filtered_data)
                        self.logger.info(
                            f"Filter: {col} {'NOT ' if negate else ''}{operator.description} {value} "
                            f"(removed {rows_removed:,} rows, "
                            f"{rows_removed / initial_count:.1%} of data)"
                        )

            self.logger.info(f"Filter applied. {len(filtered_data)} points remaining")
            return filtered_data

        except Exception as e:
            self.logger.error(f"Filter operation failed: {str(e)}")
            raise

    def filter_out_negative(self,
                            datafile: pd.DataFrame,
                            column_name: Union[str, List[str]],
                            require_all: bool = False) -> pd.DataFrame:
        """
        Filter out negative values from specified columns.

        This function keeps rows where values are greater than or equal to 0.
        When multiple columns are specified, you can control whether to:
        - Remove rows where ANY column is negative (require_all=False)
        - Remove rows where ALL columns are negative (require_all=True)

        Args:
            datafile: The DataFrame to filter
            column_name: Column(s) to check for negative values
            require_all: Controls the logic for multiple columns:
                        - False (default): Remove if ANY column is negative (OR logic)
                        - True: Remove only if ALL columns are negative (AND logic)

        Returns:
            DataFrame with negative values filtered according to specified logic

        Example:
            # Remove rows where either SFOC is negative
            df = filter_out_negative(df, ['ME_SFOC', 'DG_SFOC'], require_all=False)

            # Remove rows where both SFOC values are negative
            df = filter_out_negative(df, ['ME_SFOC', 'DG_SFOC'], require_all=True)
        """
        columns = column_name if isinstance(column_name, list) else [column_name]
        self.logger.info(
            f"Filtering {', '.join(columns)} "
            f"to remove {'all negative' if require_all else 'any negative'} values"
        )

        return self.filter_by_value(
            datafile=datafile,
            column_name=column_name,
            value=0,
            operator=ComparisonOperator.GREATER_EQUAL,
            require_all=require_all,
            negate=False  # We don't need negation here
        )

    def filter_out_null(self,
                        datafile: pd.DataFrame,
                        column_name: Union[str, List[str]]) -> pd.DataFrame:
        """
        Filter out zero values from specified columns.

        This function removes rows where the specified column(s) have zero values.
        For multiple columns, it removes a row if ANY column has a zero value.

        Args:
            datafile: The DataFrame to filter
            column_name: Column(s) to check for zero values

        Returns:
            DataFrame with zero values removed

        Example:
            # Remove rows where shaft power is zero
            df = filter_out_null(df, 'SHAFT_PWR')

            # Remove rows where either power value is zero
            df = filter_out_null(df, ['ME_PWR', 'DG_PWR'])
        """
        columns = column_name if isinstance(column_name, list) else [column_name]
        self.logger.info(
            f"Filtering {', '.join(columns)} to remove zero values"
        )

        return self.filter_by_value(
            datafile=datafile,
            column_name=column_name,
            value=0,
            operator=ComparisonOperator.EQUAL,
            require_all=False,  # Remove if ANY column equals zero
            negate=True  # Keep values that are NOT equal to zero
        )

    def apply_multiple_filters(self,
                               datafile: pd.DataFrame,
                               filters: Union[List[Dict[str, str]], Dict[str, str]]) -> pd.DataFrame:
        """
        Apply multiple filters sequentially to a DataFrame.

        This method processes one or more filters in sequence. Each filter is defined
        by a dictionary specifying the column(s) to filter, the comparison operator,
        and the value to compare against.

        Args:
            datafile: The DataFrame to filter
            filters: Either a single filter dictionary or a list of filter dictionaries.
                    Each dictionary should have:
                    - 'column': str or List[str] - Column(s) to filter
                    - 'operator': ComparisonOperator or str - The comparison operator
                    - 'value': numeric - The value to compare against

        Returns:
            DataFrame with all filters applied in sequence

        Example:
            filters = [
                {
                    'column': 'speed',
                    'operator': ComparisonOperator.GREATER_THAN,
                    'value': 20
                },
                {
                    'column': ['power', 'thrust'],
                    'operator': ComparisonOperator.LESS_EQUAL,
                    'value': 1000
                }
            ]
        """
        filtered_df = datafile.copy()

        # Convert single filter to list for consistent processing
        if isinstance(filters, dict):
            filters = [filters]

        self.logger.info(f"Beginning application of {len(filters)} filters")

        for i, filter_dict in enumerate(filters, 1):
            try:
                column_name = filter_dict['column']
                operator = filter_dict['operator']
                value = filter_dict['value']

                # Convert string operator to enum if necessary
                if isinstance(operator, str):
                    try:
                        operator = ComparisonOperator.from_symbol(operator)
                    except ValueError as e:
                        self.logger.error(f"Invalid operator in filter {i}: {operator}")
                        raise

                # Log the filter being applied using the operator's description
                self.logger.info(
                    f"Applying filter {i}: {column_name} {operator.description} {value}"
                )

                filtered_df = self.filter_by_value(
                    filtered_df,
                    column_name,
                    value,
                    operator
                )

            except KeyError as e:
                self.logger.error(f"Missing required key in filter {i}: {str(e)}")
                raise KeyError(
                    f"Filter dictionary must contain 'column', 'operator', and 'value'. "
                    f"Missing: {str(e)}"
                )

        self.logger.info(
            f"Filter sequence completed. Retained {len(filtered_df)} of {len(datafile)} "
            f"original rows ({len(filtered_df) / len(datafile):.1%})"
        )
        return filtered_df

    def filter_by_differential(self,
                               datafile: pd.DataFrame,
                               column_name: Union[str, List[str]],
                               max_diff: Optional[float] = None,
                               min_diff: Optional[float] = None) -> pd.DataFrame:
        """
        Filter rows based on the differential between consecutive values in a column.
        """
        filtered_df = datafile.copy()

        # Handle both string and list inputs
        col = column_name[0] if isinstance(column_name, list) else column_name

        # Calculate the differential
        diff = filtered_df[col].diff().abs()

        # Keep track of which rows to keep
        mask = pd.Series(True, index=filtered_df.index)

        if max_diff is not None:
            # First row will have NaN diff, we want to keep it
            mask = mask & (diff.isna() | (diff <= max_diff))

        if min_diff is not None:
            mask = mask & (diff.isna() | (diff >= min_diff))

        # Apply the mask to keep the rows we want
        filtered_df = filtered_df[mask]

        self.logger.info(
            f'Filtered by {"max_diff: " + str(max_diff) if max_diff is not None else ""}'
            f'{"min_diff: " + str(min_diff) if min_diff is not None else ""}. '
            f'{len(filtered_df)} data points left.'
        )

        return filtered_df

    def filter_out_blanks(self,
                          datafile: pd.DataFrame,
                          column_name: Union[str, List[str]]) -> pd.DataFrame:
        """
        Filter out rows where specified column(s) contain blank values.

        Blank values include empty strings, None, and NaN.

        :param datafile: The DataFrame to filter.
        :type datafile: pd.DataFrame
        :param column_name: Column(s) to check for blank values.
        :type column_name: str or list of str
        :returns: DataFrame with rows containing blanks removed.
        :rtype: pd.DataFrame

        :Example:

            >>> df = pd.DataFrame({'A': ['', 'data', None, 'more'], 'B': [1, 2, 3, 4]})
            >>> filter_obj.filter_out_blanks(df, 'A')
            # Returns DataFrame with only rows where A contains 'data' and 'more'
        """
        filtered_data = datafile.copy()
        initial_count = len(filtered_data)

        # Convert single column to list for consistent processing
        columns = [column_name] if isinstance(column_name, str) else column_name

        try:
            for col in columns:
                if col not in filtered_data.columns:
                    self.logger.error(f"Column not found: {col}")
                    raise KeyError(f"Column '{col}' not found in DataFrame")

                # Filter out empty strings, None, and NaN values
                condition = filtered_data[col].notna() & (filtered_data[col] != '')
                filtered_data = filtered_data[condition]

                rows_removed = initial_count - len(filtered_data)
                self.logger.info(
                    f"Filtered blank values from column: {col} "
                    f"(removed {rows_removed:,} rows, "
                    f"{rows_removed / initial_count:.1%} of data)"
                )

            return filtered_data

        except Exception as e:
            self.logger.error(f"Filter operation failed: {str(e)}")
            raise



class DataTransformer(LoggerMixin):
    """
    A class to handle various data transformation operations.

    This class provides methods for calculating new columns and applying
    transformations to existing columns.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):

        self.operations = {
            'difference': lambda x: x[0] - x[1],
            'sum': sum,
            'product': lambda x: pd.np.prod(x)
        }

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def calculate_column_operation(self,
                                   datafile: pd.DataFrame,
                                   columns: List[str],
                                   new_column_name: str,
                                   operation: ColumnOperation,
                                   handle_negatives: str = 'absolute') -> pd.DataFrame:
        """
        Performs calculations between columns and stores the result in a new column.

        :param df: DataFrame containing the data.
        :param columns: List of column names to be used in the operation.
        :param new_column_name: Name of the new column to store the result.
        :param operation: The operation to perform (DIFFERENCE, SUM, PRODUCT).
        :param handle_negatives: How to handle negative values ('keep', 'absolute', 'zero', or float threshold)
                           - 'keep': Keep negative values as they are
                           - 'absolute': Convert all values to absolute values
                           - 'zero': Convert negative values to 0
                           - float value: Convert negative values above this threshold to 0
        :return: DataFrame with the new column added.
        """
        try:
            if operation == ColumnOperation.DIFFERENCE and len(columns) != 2:
                raise ValueError("Difference operation requires exactly two columns")

            self.logger.info(f"Calculating {operation.value} of {columns} into {new_column_name}")

            #Convert columns to numeric, forcing no-numeric values to NaN

            numeric_df = datafile[columns].apply(pd.to_numeric, errors='coerce')

            operation_func = ColumnOperation.get_operation_function(operation)
            if operation_func is None:
                raise ValueError(f"Unsupported operation: {operation}")

            result = operation_func(numeric_df.values.T)

            # Count negative values before handling
            negative_mask = result < 0
            initial_negatives = negative_mask.sum()

            # Handle negative values according to the specified option
            if handle_negatives != 'keep':
                if handle_negatives == 'absolute':
                    self.logger.info(f"Converting {initial_negatives} negative values to absolute values")
                    result = np.abs(result)

                elif handle_negatives == 'zero':
                    self.logger.info(f"Converting {initial_negatives} negative values to zero")
                    result = np.where(result < 0, 0, result)

                elif isinstance(handle_negatives, (int, float)):
                    threshold_mask = (result < 0) & (result > -handle_negatives)
                    converted_count = threshold_mask.sum()
                    self.logger.info(
                        f"Converting {converted_count} negative values above threshold -{handle_negatives} to zero "
                        f"({initial_negatives - converted_count} values below threshold kept)"
                    )
                    result = np.where(threshold_mask, 0, result)

                # Log the effect of the transformation
                remaining_negatives = (result < 0).sum()
                if remaining_negatives > 0:
                    self.logger.info(f"Remaining negative values after handling: {remaining_negatives}")
                else:
                    self.logger.info("All negative values have been handled")

            else:
                if initial_negatives > 0:
                    self.logger.info(f"Keeping {initial_negatives} negative values as per 'keep' option")

            #Ensure result is numeric
            result=pd.to_numeric(result, errors='coerce')

            insert_pos = datafile.columns.get_loc(columns[-1]) + 1

            datafile.insert(insert_pos, new_column_name, result)

            non_numeric_mask = pd.isna(result)

            if non_numeric_mask.any():
                self.logger.warning(f"Found {non_numeric_mask.sum()} non-numeric results in {new_column_name}")

            return datafile

        except Exception as e:
            self.logger.error(f"Column operation failed: {str(e)}")
            raise

    def transform_data(self,
                       datafile: pd.DataFrame,
                       transform_columns: Optional[Dict[str, Union[Callable, TransformFunction]]] = None
                       ) -> pd.DataFrame:
        """
        Apply transformations to specified columns.

        Supports both direct callable functions and predefined TransformFunction enums.
        """

        for col, transform in transform_columns.items():
            if col not in datafile.columns:
                self.logger.warning(f"Column {col} not found in DataFrame.")
                continue

            try:
                if isinstance(transform, TransformFunction):
                    # Use the Enum's apply method
                    datafile[col] = transform.apply(datafile[col])
                elif callable(transform):
                    # Handle direct function call
                    datafile[col] = transform(datafile[col])
            except Exception as e:
                self.logger.error(f"Error transforming column {col}: {str(e)}")
                raise
        self.logger.info(f"Transform applied to {transform_columns.keys()} columns.")
        return datafile
    def validate_dataframe(self,
                           columns: List[str],
                           df: pd.DataFrame) -> None:
        """
        Validates that the specified column is suitable for transform/column operations.
        :param columns: The column being validated
        :param df: the dataframe that contains the column
        :return: None
        """
        if columns not in df.columns:
            raise ValueError("Required column 'SPEED' not found in DataFrame")

        required_column = df[columns]
        non_numeric_mask = ~pd.to_numeric(required_column, errors='coerce').notna()

        if non_numeric_mask.any():
            problem_values = required_column[non_numeric_mask].unique()
            problem_indices = non_numeric_mask[non_numeric_mask].index.tolist()
            raise ValueError(
                f"Column {columns} contains non-numerical values at indices {problem_indices}. "
                f"Problematic values: {problem_values}. "
                "Please ensure all values are numbers."
            )

    def create_constant_column(self,
                               df: pd.DataFrame,
                               new_columns: Union[str, List[str]],
                               values: Union[Any, List[Any]],
                               position: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Creates one or more new columns with constant values across all rows.

        This function is particularly useful for adding vessel characteristics
        (like draft, length, or beam) that remain constant throughout the dataset.

        Args:
            df: The input DataFrame to modify
            new_columns: Either a single column name or a list of column names to create
            values: Either a single value or a list of values corresponding to new_columns.
                    Each value will be repeated for every row in the DataFrame.
            position: Where to insert the new column(s). Can be:
                     - None: append at the end (default)
                     - int: specific position index
                     - str: name of the column to insert after

        Returns:
            DataFrame with the new constant-value column(s) added

        Examples:
            # Single constant column
            df = transformer.create_constant_column(
                df,
                new_columns='VESSEL_DRAFT',
                values=8.5
            )

            # Multiple constant columns
            df = transformer.create_constant_column(
                df,
                new_columns=['VESSEL_DRAFT', 'VESSEL_LENGTH'],
                values=[8.5, 185.0]
            )
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            result = df.copy()

            # Convert inputs to lists if they're single values
            columns = [new_columns] if isinstance(new_columns, str) else new_columns
            constant_values = [values] if not isinstance(values, list) else values

            # Validate inputs
            if len(columns) != len(constant_values):
                raise ValueError(
                    f"Number of columns ({len(columns)}) must match number of values ({len(constant_values)})"
                )

            # Check for duplicate column names
            existing_columns = set(result.columns)
            duplicates = [col for col in columns if col in existing_columns]
            if duplicates:
                raise ValueError(f"Column(s) already exist in DataFrame: {duplicates}")

            # Determine insertion position
            if position is None:
                # Append at the end
                insert_positions = [len(result.columns)] * len(columns)
            elif isinstance(position, int):
                # Insert all columns at the specified position
                insert_positions = list(range(position, position + len(columns)))
            elif isinstance(position, str):
                # Insert after the specified column
                base_position = result.columns.get_loc(position) + 1
                insert_positions = list(range(base_position, base_position + len(columns)))
            else:
                raise ValueError("Position must be None, int, or string (column name)")

            # Create and insert the new columns
            for col, value, pos in zip(columns, constant_values, insert_positions):
                # Create the constant value series
                constant_series = pd.Series([value] * len(result), index=result.index)

                # Insert the new column
                result.insert(pos, col, constant_series)

                # Log the operation
                self.logger.info(
                    f"Created constant column '{col}' with value {value} "
                    f"at position {pos}"
                )

                # Log the data type for debugging purposes
                self.logger.debug(
                    f"Column '{col}' created with dtype: {result[col].dtype}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to create constant column(s): {str(e)}")
            raise

    # def transform_data(self,
    #                    datafile: pd.DataFrame,
    #                    transform_columns: Optional[Dict[str, Union[Callable, Dict]]] = None
    #                    ) -> pd.DataFrame:
    #     """
    #     Apply transformations to specified columns.
    #
    #     Supports both direct callable functions and complex transformations
    #     specified through dictionaries.
    #     """
    #     if not transform_columns:
    #         return datafile
    #
    #     result = datafile.copy()
    #
    #     for col, transform in transform_columns.items():
    #         if col not in result.columns:
    #             self.logger.warning(f"Column {col} not found in DataFrame.")
    #             continue
    #
    #         try:
    #             if isinstance(transform, dict):
    #                 # Handle dictionary specification with function and parameters
    #                 func = transform['function']
    #                 args = transform.get('args', [])
    #                 kwargs = transform.get('kwargs', {})
    #                 result[col] = func(result[col], *args, **kwargs)
    #             elif callable(transform):
    #                 # Handle direct callable functions
    #                 result[col] = transform(result[col])
    #         except Exception as e:
    #             self.logger.error(f"Error transforming column {col}: {str(e)}")
    #             raise
    #
    #     return result



class DataAggregator(LoggerMixin):
    """
    Handles data aggregation operations including grouping, weighted averages,
    and custom aggregation functions.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        super().__init__()
        self.available_functions = {func: func.description for func in AggregationFunction}

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def create_aggregation_dict(self,
                                columns: List[str],
                                agg_values: Optional[List[str]] = None,
                                interactive: bool = True) -> Dict[str, str]:
        """
        Creates an aggregation dictionary mapping columns to their aggregation functions.

        Args:
            columns: List of columns to create aggregation mapping for
            agg_values: Optional preset aggregation values
            interactive: Whether to prompt user for input

        Returns:
            Dictionary mapping columns to their aggregation functions
        """
        if agg_values is None and interactive:
            self.logger.info("Available aggregation functions:")
            for i, (func, desc) in enumerate(self.available_functions.items(), 1):
                self.logger.info(f"{i}. {func.name}: {desc}")

            agg_values = []
            for col in columns:
                while True:
                    try:
                        self.logger.info(f"\nColumn: {col}")
                        idx = int(input("Enter function number: "))
                        func = list(self.available_functions.keys())[idx - 1]
                        agg_values.append(func.name)
                        break
                    except (ValueError, IndexError):
                        self.logger.error("Invalid selection. Please try again.")

        return dict(zip(columns, agg_values))

    def group_and_aggregate(self,
                            df: pd.DataFrame,
                            group_key: str,
                            aggregation_dict: Optional[Dict[str, Union[str, Callable]]] = None,
                            freq: str = 'h') -> pd.DataFrame:
        """
        Groups DataFrame by time key and applies specified aggregations.

        Args:
            df: Input DataFrame
            group_key: Column name containing datetime values
            aggregation_dict: Mapping of columns to aggregation functions
            freq: Frequency for grouping ('H' for hourly, 'D' for daily, etc.)
            context_df: Optional DataFrame to use for calculations (like weights)
        """


        try:
            if aggregation_dict is None:
                aggregation_dict = self.create_aggregation_dict(list(df.columns))

            self.logger.info(f"Starting aggregation process")
            self.logger.info(f"Input DataFrame shape: {df.shape}")
            self.logger.info(f"Grouping by {group_key} with frequency {freq}")
            self.logger.debug(f"Aggregation functions: {aggregation_dict}")

            # # Handle custom aggregation functions
            # self.logger.debug("Creating groupby object")
            # grouped = df.groupby(pd.Grouper(key=group_key, freq=freq))
            #
            # grouped.groupby_parent_df = df
            # self.logger.debug(f"Created {len(grouped)} groups")


            custom_aggs = {}
            std_aggs = {}

            self.logger.debug("Separating standard and custom aggregations")
            for col, func in aggregation_dict.items():
                if isinstance(func, AggregationFunction) and func.function is not None:
                    self.logger.debug(f"Adding {col} to custom aggregations")
                    custom_aggs[col] = func.function
                elif callable(func):
                    self.logger.debug(f"Adding {col} to custom aggregations (callable)")

                    def wrapped_func(group, original_func=func, parent_df=df):
                        return original_func(group, parent_df)
                    custom_aggs[col] = wrapped_func
                else:
                    self.logger.debug(f"Adding {col} to standard aggregations")
                    std_aggs[col] = func.name if isinstance(func, AggregationFunction) else func

            self.logger.info(f"Standard aggregations: {list(std_aggs.keys())}")
            self.logger.info(f"Custom aggregations: {list(custom_aggs.keys())}")

            # Perform aggregation
            self.logger.info("Performing standard aggregations")
            result = df.groupby(pd.Grouper(key=group_key, freq=freq)).agg(std_aggs)
            # grouped = df.groupby(pd.Grouper(key=group_key, freq=freq))
            #
            # result = grouped.agg(aggregation_dict)
            # self.logger.info("Performing standard aggregations")
            # result = grouped.agg(std_aggs)
            # self.logger.debug(f"Standard aggregation result shape: {result.shape}")

            # Apply custom aggregations
            self.logger.info("Applying custom aggregations")
            for col, func in custom_aggs.items():
                result[col] = df.groupby(pd.Grouper(key=group_key, freq=freq))[col].apply(func)

            self.logger.info(f"Aggregation complete. Result has {len(result)} rows")
            self.logger.debug(f"Final result columns: {result.columns.tolist()}")
            return result

        except Exception as e:
            self.logger.error(f"Aggregation failed: {str(e)}")
            raise

    def create_weighted_aggregation(self,
                                    column: str,
                                    weight_column: str) -> Callable:
        """
        Creates a weighted aggregation function for a column using specified weights.

        Args:
            column: Column to aggregate
            weight_column: Column to use as weights
        """
        return AggregationFunction.create_weighted_mean(weight_column)

    def create_multiple_weighted_aggregations(self,
                                     columns: List[str],
                                     weight_column: str) -> Dict[str, Callable]:
        """
        Creates weighted aggregations for multiple columns using the same weight.

        Args:
            columns: List of columns to create weighted aggregations for
            weight_column: Column to use as weight for all aggregations

        Returns:
            Dictionary of column names to weighted aggregation functions

            # Usage example:
            aggregator = DataAggregator()

            # Create weighted aggregations for multiple temperature measurements
            temp_columns = ['INLET_TEMP', 'OUTLET_TEMP', 'AMBIENT_TEMP']
            temp_aggs = aggregator.create_weighted_aggregations(temp_columns, 'FLOW_RATE')

            # Combine with other aggregations
            agg_dict = {
                **temp_aggs,
                'FLOW_RATE': AggregationFunction.MEAN,
                'PRESSURE': AggregationFunction.MAX
}
        """
        return {
            col: AggregationFunction.create_weighted_mean(weight_column)
            for col in columns
        }


class DataManipulator(LoggerMixin):
    """
    Handles various data manipulation operations including column calculations
    and duplicate handling.
    """

    def __init__(self, verbose: bool = True, debug: bool = False):
        super().__init__()
        self.verbose = verbose
        self.debug = debug

        if debug:
            self.logger.setLevel(logging.DEBUG)
        elif verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def calculate_column_operation(self,
                                   df: pd.DataFrame,
                                   columns: List[str],
                                   new_column_name: str,
                                   operation: ColumnOperation) -> pd.DataFrame:
        """
        Performs calculations between columns and stores the result in a new column.

        :param df: DataFrame containing the data.
        :param columns: List of column names to be used in the operation.
        :param new_column_name: Name of the new column to store the result.
        :param operation: The operation to perform (DIFFERENCE, SUM, PRODUCT).
        :return: DataFrame with the new column added.
        """
        try:
            if operation == ColumnOperation.DIFFERENCE and len(columns) != 2:
                raise ValueError("Difference operation requires exactly two columns")

            self.logger.info(f"Calculating {operation.value} of {columns} into {new_column_name}")

            operation_func = ColumnOperation.get_operation_function(operation)
            if operation_func is None:
                raise ValueError(f"Unsupported operation: {operation}")

            result = operation_func(df[columns].values.T)
            insert_pos = df.columns.get_loc(columns[-1]) + 1
            df.insert(insert_pos, new_column_name, result)

            return df

        except Exception as e:
            self.logger.error(f"Column operation failed: {str(e)}")
            raise

    def drop_duplicates(self,
                        df: pd.DataFrame,
                        subset: Union[str, List[str]],
                        keep: str = 'first') -> pd.DataFrame:
        """
        Removes duplicate rows and resets the index.

        :param df: DataFrame containing the data.
        :param subset: Column(s) to consider for duplicate detection.
        :param keep: Which duplicate to keep ('first', 'last', or False to remove all).
        :return: DataFrame with duplicates removed.
        """
        try:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

            removed = initial_rows - len(df)
            percentage_removed = (removed / initial_rows * 100) if initial_rows else 0
            self.logger.info(f"Removed {removed} duplicate rows ({percentage_removed:.1f}% of data)")

            return df

        except Exception as e:
            self.logger.error(f"Duplicate removal failed: {str(e)}")
            raise

    def remove_duplicates_with_na(self,
                                  df: pd.DataFrame,
                                  subset: Union[str, List[str]],
                                  keep: str = 'first',
                                  remove_na: bool = False) -> pd.DataFrame:
        """
        Removes duplicate rows with option to handle NaN values.

        Args:
            df: DataFrame containing the data
            subset: Column(s) to consider for duplicate detection
            keep: Which duplicate to keep ('first', 'last', or False to remove all)
            remove_na: Whether to remove rows with NaN values in subset columns before
                      removing duplicates

        Returns:
            DataFrame with duplicates (and optionally NaN values) removed

        Example:
            >>> manipulator = DataManipulator()
            >>> clean_df = manipulator.remove_duplicates_with_na(
            ...     df,
            ...     subset=['col1', 'col2'],
            ...     remove_na=True
            ... )
        """
        try:
            initial_rows = len(df)
            working_df = df.copy()

            # Handle NaN values if requested
            if remove_na:
                pre_na_rows = len(working_df)
                working_df = working_df.dropna(subset=subset)
                na_removed = pre_na_rows - len(working_df)

                if na_removed > 0:
                    self.logger.info(
                        f"Removed {na_removed:,} rows with NaN values "
                        f"({na_removed / pre_na_rows:.1%} of data)"
                    )

            # Remove duplicates
            working_df = working_df.drop_duplicates(
                subset=subset,
                keep=keep
            ).reset_index(drop=True)

            # Log results
            total_removed = initial_rows - len(working_df)
            self.logger.info(
                f"Total rows removed: {total_removed:,} "
                f"({total_removed / initial_rows:.1%} of data)"
            )

            if self.debug:
                self.logger.debug(f"Remaining rows: {len(working_df):,}")
                self.logger.debug(f"Columns used for deduplication: {subset}")

            return working_df

        except Exception as e:
            self.logger.error(f"Duplicate and NaN removal failed: {str(e)}")
            raise

    def clean_data(self,
                   datafile: pd.DataFrame,
                   strip_whitespace: bool = True,
                   capitalize: bool = True,
                   replace_spaces: Union[bool, str] = False,
                   drop_first_row: bool = True,
                   preserve_parentheses_case: bool = True) -> pd.DataFrame:
        """
        Apply configurable cleaning operations to a DataFrame.

        Args:
            datafile: DataFrame to clean
            strip_whitespace: Whether to remove leading/trailing whitespace
            capitalize: Whether to capitalize text
            replace_spaces: If True, replaces spaces with underscores
                           If a string, replaces spaces with that string
            drop_first_row: Whether to drop the first row of data
            preserve_parentheses_case: Whether to preserve case of text inside parentheses

        Returns:
            Cleaned DataFrame

        Example:
            >>> cleaned_df = manipulator.clean_data(
            ...     df,
            ...     capitalize=True,
            ...     replace_spaces='_',
            ...     preserve_parentheses_case=True
            ... )
        """
        try:
            self.logger.info("Starting data cleaning process")
            cleaned_df = datafile.copy()

            # Process column names
            if strip_whitespace or capitalize or replace_spaces:
                # Get original column names
                columns = cleaned_df.columns.tolist()

                # Process each column name
                for i, col in enumerate(columns):
                    if not isinstance(col, str):
                        continue

                    # Handle stripping
                    if strip_whitespace:
                        col = col.strip()

                    # Handle capitalization with parentheses preservation
                    if capitalize:
                        if preserve_parentheses_case:
                            # Find all text within parentheses
                            parentheses_texts = re.findall(r'\([^)]*\)', col)
                            # Temporarily replace them with placeholders
                            for j, ptext in enumerate(parentheses_texts):
                                placeholder = f"__PAREN_{j}__"
                                col = col.replace(ptext, placeholder)

                            # Capitalize the text
                            col = col.upper()

                            # Restore original parentheses text
                            for j, ptext in enumerate(parentheses_texts):
                                placeholder = f"__PAREN_{j}__"
                                col = col.replace(placeholder, ptext)
                        else:
                            col = col.upper()

                    # Handle space replacement
                    if replace_spaces:
                        replacement = '_' if replace_spaces is True else replace_spaces
                        col = col.replace(' ', replacement)

                    columns[i] = col

                # Apply the processed column names
                cleaned_df.columns = columns
                self.logger.info(f"Processed {len(columns)} column names")

            # Drop the first row if requested
            if drop_first_row and not cleaned_df.empty:
                cleaned_df = cleaned_df.iloc[1:]
                self.logger.info("Dropped first row of data")

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def create_type_keys(self,
                         df: pd.DataFrame,
                         column_name: str,
                         custom_keys: Optional[Dict[str, int]] = None,
                         key_column_name: Optional[str] = None) -> pd.DataFrame:
        """
        Creates a new column with numeric keys for categorical values in a specified column.

        This function generates a mapping between categorical values and numeric keys, with
        the option to specify custom keys for certain values. Any values not covered by
        custom keys will be assigned sequential numbers continuing from the highest custom key.

        Args:
            df: DataFrame containing the data
            column_name: Name of the column to create keys for
            custom_keys: Optional dictionary mapping specific values to desired key numbers
            key_column_name: Optional name for the new key column. If not provided,
                            defaults to {column_name}_key

        Returns:
            DataFrame with the new key column added

        Example:
            >>> custom_keys = {'Bulk Carrier': 1, 'Tanker': 2, 'Cruise Ship': 3}
            >>> df = manipulator.create_type_keys(
            ...     df,
            ...     'vessel_type',
            ...     custom_keys=custom_keys
            ... )
        """
        try:
            # Input validation
            if column_name not in df.columns:
                raise KeyError(f"Column '{column_name}' not found in DataFrame")

            working_df = df.copy()

            # Get unique values from the specified column
            unique_values = working_df[column_name].unique()

            # Initialize the mapping dictionary
            value_mapping = {}

            if custom_keys:
                # Validate custom keys
                invalid_keys = [k for k in custom_keys.keys() if k not in unique_values]
                if invalid_keys:
                    self.logger.warning(
                        f"Custom keys provided for values not in the data: {invalid_keys}"
                    )

                # Start with custom keys
                value_mapping.update(custom_keys)
                max_key = max(custom_keys.values())

                # Log custom key assignments
                self.logger.info(f"Applied {len(custom_keys)} custom keys")
                if self.debug:
                    for value, key in custom_keys.items():
                        self.logger.debug(f"Custom key: {value} → {key}")
            else:
                max_key = 0

            # Add remaining values with sequential keys
            remaining_values = [v for v in unique_values if v not in value_mapping]
            for value in remaining_values:
                max_key += 1
                value_mapping[value] = max_key

            # Create the key column name
            if key_column_name is None:
                key_column_name = f"{column_name}_key"

            # Add the key column
            insert_pos = working_df.columns.get_loc(column_name) + 1
            working_df.insert(
                insert_pos,
                key_column_name,
                working_df[column_name].map(value_mapping)
            )

            # Log the mapping creation
            self.logger.info(
                f"Created {key_column_name} with {len(value_mapping)} unique keys"
            )
            if self.debug:
                # Create a mapping DataFrame for debugging
                mapping_df = pd.DataFrame(
                    list(value_mapping.items()),
                    columns=[column_name, key_column_name]
                ).sort_values(key_column_name)
                self.logger.debug(f"\nFull mapping table:\n{mapping_df}")

            return working_df

        except Exception as e:
            self.logger.error(f"Type key creation failed: {str(e)}")
            raise

class DataNormaliser(LoggerMixin):
    """
    Handles data normalization operations using various scaling methods.

    This class provides functionality to normalize data using different scikit-learn
    scalers or custom normalization functions. It maintains the normalization
    parameters for potential later use in denormalization.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        # Store normalization parameters for each dataset
        self.normalization_params = {}
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies numeric columns in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of column names with numeric data types
        """
        numeric_cols = df.select_dtypes(
            include=['float64', 'float32', 'int64', 'int32']
        ).columns.tolist()

        self.logger.INFO(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
        return numeric_cols

    def _validate_columns(self, df: pd.DataFrame, columns: List[str]):
        """
        Ensures all specified columns exist in the DataFrame.

        Args:
            df: Input DataFrame
            columns: List of columns to validate

        Raises:
            ValueError: If any specified columns are missing
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            self.logger.INFO(f"Columns not found in DataFrame: {missing_cols}")
            raise ValueError(f"Columns not found: {missing_cols}")

    def _create_scaler(self, method: Union[str, Callable, NormalizationMethod]):
        """
        Creates the appropriate scaler based on the specified method.

        Args:
            method: Normalization method to use (string, enum, or callable)

        Returns:
            Initialized scaler object
        """
        if isinstance(method, NormalizationMethod):
            return method.scaler_class()
        elif isinstance(method, str):
            try:
                method = NormalizationMethod[method.upper()]
                return method.scaler_class()
            except KeyError:
                self.logger.error(f"Unknown normalization method: {method}")
                raise ValueError(
                    f"Unsupported method: {method}. "
                    f"Use one of: {[m.name.lower() for m in NormalizationMethod]}"
                )
        elif callable(method):
            return method()
        else:
            raise TypeError("method must be a string, NormalizationMethod, or callable")

    def _extract_scaler_params(self,
                               scaler,
                               columns: List[str]) -> Dict:
        """
        Extracts normalization parameters from the fitted scaler.

        Args:
            scaler: Fitted scikit-learn scaler
            columns: List of column names that were normalized

        Returns:
            Dictionary of normalization parameters
        """
        params = {
            'columns': columns,
            'method': type(scaler).__name__
        }

        # Extract relevant parameters based on scaler type
        if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
            params.update({
                'min_': scaler.min_,
                'scale_': scaler.scale_
            })
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            params.update({
                'mean_': scaler.mean_,
                'scale_': scaler.scale_
            })

        return params

    def normalise(self,
                  datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                  columns_to_normalise: Optional[List[str]] = None,
                  method: Union[str, Callable, NormalizationMethod] = NormalizationMethod.MINMAX
                  ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Normalizes specified columns using the chosen scaling method.

        This method handles both single DataFrames and dictionaries of DataFrames.
        It automatically detects numeric columns if none are specified and stores
        normalization parameters for potential denormalization.

        Args:
            datafile: DataFrame or dictionary of DataFrames to normalize
            columns_to_normalize: Specific columns to normalize (optional)
            method: Normalization method to use

        Returns:
            Normalized DataFrame(s)

        Example:
            normalizer = DataNormalizer()

            # Normalize specific columns using standard scaling
            normalized_df = normalizer.normalize(
                df,
                columns_to_normalize=['speed', 'power'],
                method=NormalizationMethod.STANDARD
            )

            # Normalize all numeric columns using min-max scaling
            normalized_df = normalizer.normalize(df)
        """
        try:
            if isinstance(datafile, dict):
                return self._normalize_dict(datafile, columns_to_normalise, method)
            elif isinstance(datafile, pd.DataFrame):
                return self._normalize_dataframe(datafile, columns_to_normalise, method)
            else:
                raise TypeError("Input must be a DataFrame or dictionary of DataFrames")

        except Exception as e:
            self.logger.info(f"Normalization failed: {str(e)}")
            raise

    def _normalize_dataframe(self,
                             df: pd.DataFrame,
                             columns_to_normalize: Optional[List[str]],
                             method: Union[str, Callable, NormalizationMethod]
                             ) -> pd.DataFrame:
        """
        Normalizes a single DataFrame.
        """
        df = df.copy()

        # Determine columns to normalize
        cols_to_norm = (columns_to_normalize if columns_to_normalize
                        else self._get_numeric_columns(df))

        self._validate_columns(df, cols_to_norm)

        # Create and fit scaler
        scaler = self._create_scaler(method)
        original_data = df[cols_to_norm].copy()
        df[cols_to_norm] = scaler.fit_transform(original_data)

        # Store normalization parameters
        self.normalization_params = self._extract_scaler_params(scaler, cols_to_norm)

        self.logger.info(
            f"Normalized {len(cols_to_norm)} columns using "
            f"{self.normalization_params['method']}"
        )

        return df

    def _normalize_dict(self,
                        datafiles: Dict[str, pd.DataFrame],
                        columns_to_normalize: Optional[List[str]],
                        method: Union[str, Callable, NormalizationMethod]
                        ) -> Dict[str, pd.DataFrame]:
        """
        Normalizes a dictionary of DataFrames.
        """
        normalized_datafiles = {}

        for name, df in datafiles.items():
            self.logger.INFO(f"Normalizing dataset: {name}")
            normalized_datafiles[name] = self._normalize_dataframe(
                df, columns_to_normalize, method
            )
            # Store parameters with dataset name
            self.normalization_params[name] = self.normalization_params.copy()

        return normalized_datafiles


class ExportFormat(Enum):
    """
    Enumeration of supported export file formats.

    Each format includes:
    - extension: The file extension to use
    - description: A human-readable description of the format
    - limitations: Any specific limitations or requirements
    """
    CSV = ('csv', 'Comma-separated values', 'Single DataFrame only')
    EXCEL = ('xlsx', 'Microsoft Excel workbook', 'Supports multiple sheets')

    def __init__(self, extension: str, description: str, limitations: str):
        self.extension = extension
        self.description = description
        self.limitations = limitations


class DataExporter(LoggerMixin):
    """
    Handles exporting DataFrames to various file formats with comprehensive error checking
    and logging. This class supports both single DataFrame exports and multiple DataFrame
    exports to Excel workbooks.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        self.last_export_path = None
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _ensure_directory_exists(self, directory: Union[str, Path]) -> Path:
        """
        Ensures the export directory exists, creating it if necessary.

        Args:
            directory: Directory path as string or Path object

        Returns:
            Path object representing the directory

        Raises:
            OSError: If directory creation fails
        """
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")
            return directory

        except Exception as e:
            self.logger.info(f"Failed to create directory {directory}: {str(e)}")
            raise OSError(f"Could not create directory {directory}: {str(e)}")

    def _validate_dataframe(self,
                            df: pd.DataFrame,
                            name: str = "DataFrame"):
        """
        Validates that a DataFrame is not empty and has the expected structure.

        Args:
            df: DataFrame to validate
            name: Name of the DataFrame for error messages

        Raises:
            ValueError: If DataFrame validation fails
        """
        if df.empty:
            self.logger.error(f"{name} is empty")
            raise ValueError(f"{name} cannot be empty")

        if df.columns.duplicated().any():
            self.logger.info(f"{name} contains duplicate column names")
            raise ValueError(f"{name} contains duplicate column names")

    def export(self,
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               directory: Union[str, Path],
               filename: str,
               format: Union[str, ExportFormat] = ExportFormat.CSV,
               sheet_names: Optional[List[str]] = None) -> Path:
        """
        Exports DataFrame(s) to the specified format and location.

        This method handles both single DataFrames and collections of DataFrames,
        automatically selecting the appropriate export method based on the input
        and desired format.

        Args:
            data: DataFrame or dictionary of DataFrames to export
            directory: Directory where the file should be saved
            filename: Name of the file (without extension)
            format: Export format (CSV or Excel)
            sheet_names: Optional sheet names for Excel exports

        Returns:
            Path object pointing to the exported file

        Example:
            exporter = DataExporter()

            # Export a single DataFrame to CSV
            exporter.export(df, 'exports', 'data_export')

            # Export multiple DataFrames to Excel
            dfs = {'Sheet1': df1, 'Sheet2': df2}
            exporter.export(
                dfs,
                'exports',
                'multi_sheet_export',
                format=ExportFormat.EXCEL
            )
        """
        try:
            # Convert string format to enum if necessary
            if isinstance(format, str):
                format = ExportFormat[format.upper()]

            # Ensure valid directory
            directory = self._ensure_directory_exists(directory)

            # Create full file path
            file_path = directory / f"{filename}.{format.extension}"

            self.logger.info(f"Beginning export to {file_path}")

            if format == ExportFormat.CSV:
                self._export_csv(data, file_path)
            elif format == ExportFormat.EXCEL:
                self._export_excel(data, file_path, sheet_names)

            self.last_export_path = file_path
            self.logger.info(f"Successfully exported to {file_path}")

            return file_path

        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise

    def _export_csv(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    file_path: Path):
        """
        Handles CSV export for a single DataFrame.
        """
        if isinstance(data, dict):
            self.logger.error("CSV format only supports single DataFrame export")
            raise ValueError(
                "CSV export requires a single DataFrame. "
                "For multiple DataFrames, use Excel format."
            )

        self._validate_dataframe(data)

        try:
            data.to_csv(file_path, index=False)
            self.logger.info(f"Exported {len(data)} rows to CSV")
        except Exception as e:
            self.logger.info(f"CSV export failed: {str(e)}")
            raise

    def _export_excel(self,
                      data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                      file_path: Path,
                      sheet_names: Optional[List[str]] = None):
        """
        Handles Excel export for single or multiple DataFrames.
        """
        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                if isinstance(data, pd.DataFrame):
                    # Single DataFrame export
                    self._validate_dataframe(data)
                    data.to_excel(writer, sheet_name="Sheet1", index=False)
                    self.logger.info(
                        f"Exported single DataFrame with {len(data)} rows"
                    )
                else:
                    # Multiple DataFrame export
                    for i, (name, df) in enumerate(data.items()):
                        self._validate_dataframe(df, name)
                        sheet_name = sheet_names[i] if sheet_names else name
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        self.logger.info(
                            f"Exported sheet '{sheet_name}' with {len(df)} rows"
                        )

        except Exception as e:
            self.logger.error(f"Excel export failed: {str(e)}")
            raise


class DataProcessingWorkflow(LoggerMixin):
    """
    Main class for orchestrating the complete data processing workflow
    """
    def __init__(self,
                 debug: bool = False,
                 verbose: bool = True):
        self.filter = DataFilter(debug=debug,
                                 verbose=verbose)
        self.aggregator = DataAggregator(debug=debug,
                                 verbose=verbose)
        self.transformer = DataTransformer(debug=debug,
                                           verbose=verbose)
        self.manipulator = DataManipulator(debug=debug,
                                 verbose=verbose)
        self.normalizer = DataNormaliser(debug=debug,
                                 verbose=verbose)

        self.exporter = DataExporter(debug=debug,
                                     verbose=verbose)





