"""
This module contains the Data Transformer class

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Callable, Any
from data_processing.transform.Distance_transformation_class import DistanceTransformation
from data_processing.transform.time_transformation_class import TimeTransform
from data_processing.transform.Speed_transformation_class import SpeedTransformation


from config.loggermixin.logger_config import LoggerMixin

from data_processing.data_processing_enum_classes import TransformFunction, ColumnOperation

class DataTransformer(LoggerMixin):
    """
    A class to handle various data transformation operations.

    This class provides methods for calculating new columns and applying
    transformations to existing columns.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):

        # self.operations = {
        #     'difference': lambda x: x[0] - x[1],
        #     'sum': sum,
        #     'product': lambda x: np.prod(x)
        # }

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        self.distance_transformation = DistanceTransformation()
        self.time_transformation = TimeTransform()
        self.speed_transformation = SpeedTransformation()

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

    def _calculate_column_operation_single(self,
                                   datafile: pd.DataFrame,
                                   columns: List[str],
                                   new_column_name: str,
                                   operation: ColumnOperation,
                                   handle_negatives: str = 'absolute') -> pd.DataFrame:
        """
        Performs calculations between columns and stores the result in a new column.

        :param datafile: DataFrame containing the data.
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
            if getattr(operation, 'value', str(operation)) == 'difference' and len(columns) != 2:
                raise ValueError("Difference operation requires exactly two columns")

            op_value = getattr(operation, 'value', str(operation))
            self.logger.info(f"Calculating {op_value} of {columns} into {new_column_name}")

            # Convert columns to numeric, forcing no-numeric values to NaN
            numeric_df = datafile[columns].apply(pd.to_numeric, errors='coerce')

            # Use operation value for mapping
            if op_value == 'difference':
                operation_func = lambda x: x[0] - x[1]
            elif op_value == 'sum':
                operation_func = lambda x: np.sum(x, axis=0)
            elif op_value == 'product':
                operation_func = lambda x: np.prod(x, axis=0)
            else:
                raise ValueError(f"Unsupported operation: {op_value}")

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

    def calculate_column_operation(self,
                                   datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                   columns: List[str],
                                   new_column_name: str,
                                   operation: ColumnOperation,
                                   handle_negatives: str = 'absolute') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Performs calculations between columns and stores the result in a new column.

        :param datafile: DataFrame containing the data. or a dictionary of dataframes
        :param columns: List of column names to be used in the operation.
        :param new_column_name: Name of the new column to store the result.
        :param operation: The operation to perform (DIFFERENCE, SUM, PRODUCT).
        :param handle_negatives: How to handle negative values ('keep', 'absolute', 'zero', or float threshold)
                           - 'keep': Keep negative values as they are
                           - 'absolute': Convert all values to absolute values
                           - 'zero': Convert negative values to 0
                           - float value: Convert negative values above this threshold to 0
        :return: DataFrame with the new column added or dictionary of DataFrames with the column added.
        """

        if isinstance(datafile, dict):
            result = {}
            for key, df in datafile.items():
                result[key] = self._calculate_column_operation_single(df,
                                                                      columns,
                                                                      new_column_name,
                                                                      # Use the provided column name, not the key
                                                                      operation,
                                                                      handle_negatives)
            return result
        else:
            return self._calculate_column_operation_single(datafile,
                                                           columns,
                                                           new_column_name,  # Fix parameter order
                                                           operation,
                                                           handle_negatives)

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

    def _create_type_key_column_single(self,
                               df: pd.DataFrame,
                               column_name: str,
                               new_column_name: Optional[str] = None,
                               custom_keys: Optional[Dict[Any, int]] = None,
                               position: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Creates a type key column based on the unique values in a source column.

        This function assigns numeric keys to categorical values, allowing for custom mappings
        while automatically handling any remaining values not specified in custom keys.

        Args:
            df: The input DataFrame to modify
            column_name: The column containing categorical values to be keyed
            new_column_name: Name of the new column to store keys (defaults to source_column + '_Key')
            custom_keys: Optional dictionary mapping specific values to custom key numbers
                         Any values not in this dictionary will be assigned sequential keys
                         after the custom keys
            position: Where to insert the new column. Can be:
                     - None: insert after the source column (default)
                     - int: specific position index
                     - str: name of the column to insert after

        Returns:
            DataFrame with the new type key column added

        Examples:
            # Basic usage with default sequential keys
            df = transformer.create_type_key_column(
                df,
                source_column='Type'
            )

            # With custom keys for specific values
            df = transformer.create_type_key_column(
                df,
                source_column='Type',
                custom_keys={'Bulk Carrier': 1, 'Tanker': 2, 'Cruise Ship': 3}
            )
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            result = df.copy()

            # Validate source column exists
            if column_name not in result.columns:
                raise ValueError(f"Source column '{column_name}' not found in DataFrame")

            # Set default new column name if not provided
            if new_column_name is None:
                new_column_name = f"{column_name}_KEY"

            # Check if new column name already exists
            if new_column_name in result.columns:
                self.logger.warning(f"Column '{new_column_name}' already exists and will be overwritten")

            # Get unique values from source column
            unique_values = result[column_name].unique()
            self.logger.info(f"Found {len(unique_values)} unique values in '{column_name}'")

            # Initialize custom_keys dictionary if not provided
            if custom_keys is None:
                custom_keys = {}

            # Create ordered list with custom keys first
            priority_values = list(custom_keys.keys())
            remaining_values = [val for val in unique_values if val not in custom_keys]
            ordered_values = priority_values + remaining_values

            # Create mapping dictionary
            type_mapping = {}
            next_key = 1  # Start with 1 as default

            # First, add all custom keys to the mapping
            for value, key in custom_keys.items():
                type_mapping[value] = key
                next_key = max(next_key, key + 1)  # Ensure next key is higher than any custom key

            # Then add remaining values with sequential keys
            for value in remaining_values:
                type_mapping[value] = next_key
                next_key += 1

            # Apply mapping to create new column
            result[new_column_name] = result[column_name].map(type_mapping)

            # Create a summary DataFrame for logging
            mapping_df = pd.DataFrame(list(type_mapping.items()),
                                      columns=[column_name, new_column_name])

            # Log the mapping for reference
            mapping_table = mapping_df.to_string(index=False)
            self.logger.info(f"Created type key mapping:\n{mapping_table}")

            # If necessary, reposition the column
            if position is not None or position != new_column_name:
                # First remove the newly added column
                column_data = result[new_column_name]
                result = result.drop(columns=[new_column_name])

                # Determine insertion position
                if position is None:
                    # Insert after source column
                    insert_pos = result.columns.get_loc(column_name) + 1
                elif isinstance(position, int):
                    insert_pos = position
                elif isinstance(position, str):
                    insert_pos = result.columns.get_loc(position) + 1
                else:
                    raise ValueError("Position must be None, int, or string (column name)")

                # Reinsert the column at the desired position
                result.insert(insert_pos, new_column_name, column_data)

            # Check for NaN values that might indicate mapping failures
            nan_count = result[new_column_name].isna().sum()
            if nan_count > 0:
                self.logger.warning(
                    f"Found {nan_count} NaN values in the new key column. "
                    f"These correspond to values in '{column_name}' that weren't in the mapping."
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to create type key column: {str(e)}")
            raise

    def create_type_key_column(self,
                               df: Union[pd.DataFrame, Dict[str,pd.DataFrame]],
                               column_name: str,
                               custom_keys: Optional[Dict[Any, int]] = None,
                               position: Optional[Union[str, int]] = None) -> pd.DataFrame:
        if isinstance(df, dict):
            result = {}
            for key,df in df.items():
                result[key] = self._create_type_key_column_single(df = df, column_name = column_name, custom_keys = custom_keys, position = position)
            return result

        else:
            return self._create_type_key_column_single(df = df, column_name = column_name, custom_keys = custom_keys, position = position)

    def combine_dataframes_by_key(self,
                                  datafiles: dict[str, pd.DataFrame],
                                  key_column: str,
                                  combine_method: str = 'first',
                                  join_method: str = 'outer') -> pd.DataFrame:
        """
        General function to combine multiple DataFrames from a dictionary into a single DataFrame aligned by a key column.

        :param datafiles: Dictionary mapping identifiers to DataFrames
        :param key_column: Name of the column to use as the key for combining
        :param combine_method: How to combine values for duplicate keys ('first', 'last', 'mean', 'sum')
        :param join_method: How to join dataframes ('outer', 'inner', 'left', 'right')

        :return: A single combined DataFrame
        """


        self.logger.info(f"Combining {len(datafiles)} datafiles based on '{key_column}' column")

        # Process each dataframe to ensure key column is index
        processed_dfs = []
        for key, df in datafiles.items():
            temp_df = df.copy()

            # Set key column as index if it's not already
            if key_column in temp_df.columns:
                temp_df = temp_df.set_index(key_column)
            elif temp_df.index.name != key_column:
                # Check if index is unnamed but could be the key
                if temp_df.index.name is None:
                    temp_df.index.name = key_column
                else:
                    self.logger.error(f"Key column '{key_column}' not found in dataframe")
                    raise KeyError(f"Key column '{key_column}' not found in dataframe")

            processed_dfs.append(temp_df)

        # Combine all dataframes using pandas concat
        if join_method in ['outer', 'inner']:
            combined_df = pd.concat(processed_dfs, axis=0, join=join_method)
        else:
            # For left/right joins, we need to use a different approach
            combined_df = processed_dfs[0]
            for df in processed_dfs[1:]:
                combined_df = combined_df.join(df, how=join_method, rsuffix='_dup')

            # Remove duplicate columns created by join
            duplicate_cols = [col for col in combined_df.columns if col.endswith('_dup')]
            if duplicate_cols:
                combined_df = combined_df.drop(columns=duplicate_cols)

        # Handle duplicate keys based on combine_method
        if combined_df.index.duplicated().sum() > 0:
            if combine_method == 'first':
                combined_df = combined_df.groupby(level=0).first()
            elif combine_method == 'last':
                combined_df = combined_df.groupby(level=0).last()
            elif combine_method == 'mean':
                combined_df = combined_df.groupby(level=0).mean(numeric_only=True)
            elif combine_method == 'sum':
                combined_df = combined_df.groupby(level=0).sum(numeric_only=True)
            else:
                self.logger.warning(f"Unknown combine_method '{combine_method}', using 'first'")
                combined_df = combined_df.groupby(level=0).first()

        # Reset index to make key column a regular column again
        combined_df = combined_df.reset_index()

        self.logger.info(
            f"Final combined dataframe has {len(combined_df)} rows and {len(combined_df.columns)} columns"
        )

        return combined_df

    def combine_dataframes(self,
                           datafiles: dict[str, pd.DataFrame],
                           timestamp_column: str,
                           resample_rule: str|None = None,
                           combine_method: str = 'first') -> pd.DataFrame:
        """
        Combine multiple DataFrames from a dictionary into a single DataFrame aligned by timestamp.
        This is now a wrapper around the general combine function with timeseries-specific preprocessing.

        :param datafiles: Dictionary mapping identifiers to DataFrames
        :param timestamp_column: Name of the column containing timestamps
        :param resample_rule: Optional pandas resample rule (e.g., 'H' for hourly)
        :param combine_method: How to combine values for the same column ('first', 'last', 'mean')

        :return: A single combined DataFrame
        """
        if not datafiles:
            self.logger.error("No datafiles provided for combining")
            raise ValueError("Empty dictionary of datafiles provided")

        # Preprocess dataframes for timeseries-specific requirements
        processed_datafiles = {}

        for key, df in datafiles.items():
            temp_df = df.copy()

            # Ensure timestamp column exists and is datetime
            if timestamp_column in temp_df.columns:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(temp_df[timestamp_column]):
                    temp_df[timestamp_column] = pd.to_datetime(temp_df[timestamp_column])
            elif temp_df.index.name == timestamp_column or (
                    temp_df.index.name is None and isinstance(temp_df.index, pd.DatetimeIndex)
            ):
                # Index is already the timestamp, ensure it's named correctly
                if temp_df.index.name is None:
                    temp_df.index.name = timestamp_column
                # Reset index to make it a column for general function
                temp_df = temp_df.reset_index()
            else:
                self.logger.error(f"Timestamp column '{timestamp_column}' not found in dataframe")
                raise KeyError(f"Timestamp column '{timestamp_column}' not found in dataframe")

            # Apply resampling if requested (this is timeseries-specific)
            if resample_rule:
                # Set timestamp as index for resampling
                temp_df = temp_df.set_index(timestamp_column)
                temp_df = temp_df.resample(resample_rule).asfreq()
                # Reset index back to column for general function
                temp_df = temp_df.reset_index()

            processed_datafiles[key] = temp_df

        # Use the general combine function
        return self.combine_dataframes_by_key(
            datafiles=processed_datafiles,
            key_column=timestamp_column,
            combine_method=combine_method,
            join_method='outer'  # Default for timeseries is usually outer join
        )