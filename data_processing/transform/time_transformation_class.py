"""
Contains a class for creating time columns

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Any, Dict, List, Tuple, Optional

from config.loggermixin.logger_config import LoggerMixin
from data_processing.data_processing_enum_classes import  TimeUnit, convert_time


class TimeTransform(LoggerMixin):

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
            self.logger.setLevel(logging.ERROR)

    def _convert_to_datetime_single(self,
                                    datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                    columns: Union[List[str], str],
                                    timestamp_precision: Optional[str] = 'S'):
        """
        Converts specified column or columns to datetime format.
        :param datafile: Datafile containing the data
        :param columns: List of column names to be used in the operation
        :param timestamp_precision: Optional precision to round timestamps to (e.g., 'S' for seconds, '10S' for 10 seconds, 'T' for minutes)
        :return:
        """

        result = datafile.copy()

        def _apply_precision(series, precision):
            """Apply timestamp precision rounding to a datetime series"""
            if precision is None:
                return series

            try:
                if precision.upper() == 'SECOND' or precision.upper() == 'S':
                    return series.dt.round('s')
                elif precision.upper() == 'MINUTE' or precision.upper() == 'T':
                    return series.dt.round('T')
                elif precision.upper() == 'HOUR' or precision.upper() == 'H':
                    return series.dt.round('H')
                else:
                    # Handle cases like '10S', '30S', '5T', etc.
                    return series.dt.round(precision)
            except Exception as e:
                self.logger.warning(f"Could not apply timestamp precision '{precision}': {e}")
                return series

        if isinstance(columns, str):
            self.logger.info(f"Converting column {columns} to datetime format")
            if timestamp_precision:
                self.logger.info(f"Will round timestamps to precision: {timestamp_precision}")

            self.logger.debug(f"Column {columns} original dtype: {result[columns].dtype}")

            result[columns] = pd.to_datetime(result[columns], errors='coerce')

            # Apply timestamp precision if specified
            if timestamp_precision:
                result[columns] = _apply_precision(result[columns], timestamp_precision)

            null_count = result[columns].isnull().sum()
            if null_count > 0:
                self.logger.debug(f"Warning: {null_count} values couldn't be converted to datetime in column {columns}")

            self.logger.debug(f"Column {columns} converted to dtype: {result[columns].dtype}")
            return result

        else:
            self.logger.info(f"Converting columns {columns} to datetime format")
            if timestamp_precision:
                self.logger.info(f"Will round timestamps to precision: {timestamp_precision}")

            for column in columns:
                self.logger.debug(f"Processing column {column} - original dtype: {result[column].dtype}")

                result[column] = pd.to_datetime(result[column], errors='coerce')

                # Apply timestamp precision if specified
                if timestamp_precision:
                    result[column] = _apply_precision(result[column], timestamp_precision)

                null_count = result[column].isnull().sum()
                if null_count > 0:
                    self.logger.debug(
                        f"Warning: {null_count} values couldn't be converted to datetime in column {column}")

                self.logger.debug(f"Column {column} converted to dtype: {result[column].dtype}")
            return result

    def convert_to_datetime(self,
                            datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            columns: Union[List[str], str],
                            timestamp_precision: Optional[str] ='S'):
        """
        A function for converting specified column or columns to datetime format.
        usable with a specific datafile or a dictionary of datafiles
        :param datafile: the datafile or dictionary of datafiles being modified
        :param columns: the list of column names to be used in the operation
        :param timestamp_precision: Optional precision to round timestamps to (e.g., 'S' for seconds, '10S' for 10 seconds, 'T' for minutes)
        :return: modified datafile or dictionary of datafiles
        """

        if isinstance(datafile, dict):
            self.logger.info(f"Converting datetime for {len(datafile)} dataframes")
            if timestamp_precision:
                self.logger.info(f"Applying timestamp precision: {timestamp_precision}")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._convert_to_datetime_single(df, columns, timestamp_precision)

            self.logger.info("Completed datetime conversion for all dataframes")
            return result
        else:
            self.logger.info(f"Converting datetime for single dataframe with shape {datafile.shape}")
            return self._convert_to_datetime_single(datafile, columns, timestamp_precision)

    def _create_sample_time_single(self,
                                   datafile: pd.DataFrame,
                                   timestamp_column: str = None,
                                   existing_sampletime_column: str = None,
                                   keep_timedelta: bool = False,
                                   unit: Union[TimeUnit, str] = TimeUnit.MINUTES) -> pd.DataFrame:
        """
        Creates a sample time column (if it doesn't already exist)

        :param datafile: the datafile containing data being transformed
        :param timestamp_column: timestamp column used to calculate the sample time column
        :param existing_sampletime_column: the column name containing the existing sample time
        :param keep_timedelta: keep the time delta column
        :param unit: target time unit for the sample time column
        :return: A dataframe with sample time column in specified units
        """

        # Convert unit to enum if it's a string
        if isinstance(unit, str):
            self.logger.debug(f"Converting unit string '{unit}' to TimeUnit enum")
            unit = TimeUnit.from_string(unit)

        self.logger.info(f"Creating sample time column in {unit.name} units")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")
        self.logger.debug(
            f"Parameters - timestamp_column: {timestamp_column}, existing_sampletime_column: {existing_sampletime_column}")

        result = datafile.copy()

        # Case 1: Using existing sample time column
        if existing_sampletime_column and existing_sampletime_column in result.columns:
            self.logger.info(f"Using existing sample time column: {existing_sampletime_column}")
            self.logger.debug(f"Existing column dtype: {result[existing_sampletime_column].dtype}")

            result['SAMPLE_TIME_DELTA'] = pd.to_timedelta(result[existing_sampletime_column])

            # Convert to specified unit using the enum
            column_name = f'SAMPLE_TIME_{unit.name}'
            self.logger.debug(f"Creating column: {column_name}")

            result[column_name] = result['SAMPLE_TIME_DELTA'].apply(
                lambda td: unit.from_timedelta(td) if pd.notna(td) else 0
            )

            # Keep original column name for backwards compatibility if it was MINUTES
            if unit == TimeUnit.MINUTES and existing_sampletime_column != column_name:
                self.logger.debug(f"Maintaining backwards compatibility - copying to {existing_sampletime_column}")
                result[existing_sampletime_column] = result[column_name]

            self.logger.info(f"Successfully created sample time column from existing data")
            return result

        # Case 2: Calculate from timestamp column
        if timestamp_column is None:
            self.logger.error("Neither timestamp_column nor existing_sampletime_column provided")
            raise ValueError("Either timestamp_column or existing_sampletime_column must be provided")

        if timestamp_column not in result.columns:
            self.logger.error(f"Timestamp column '{timestamp_column}' not found in dataframe")
            self.logger.debug(f"Available columns: {list(result.columns)}")
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in dataframe")

        self.logger.info(f"Calculating sample time from timestamp column: {timestamp_column}")
        self.logger.debug(f"Timestamp column dtype: {result[timestamp_column].dtype}")

        # Ensure the column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_column]):
            self.logger.debug(f"Converting {timestamp_column} to datetime format")
            result[timestamp_column] = pd.to_datetime(result[timestamp_column])
            self.logger.debug(f"Column converted to dtype: {result[timestamp_column].dtype}")
        else:
            self.logger.debug(f"Column {timestamp_column} is already datetime format")

        # Calculate time differences
        self.logger.debug("Calculating time differences between consecutive rows")
        time_diffs = result[timestamp_column].diff().fillna(pd.Timedelta(0))

        # Log some statistics about time differences
        self.logger.debug(f"Time differences - mean: {time_diffs.mean()}, median: {time_diffs.median()}")
        self.logger.debug(f"Time differences - min: {time_diffs.min()}, max: {time_diffs.max()}")

        # Convert time differences to specified unit
        column_name = f'SAMPLE_TIME_{unit.name}'
        self.logger.debug(f"Converting time differences to {unit.name} and storing in column: {column_name}")

        result[column_name] = time_diffs.apply(
            lambda td: unit.from_timedelta(td) if pd.notna(td) else 0
        )

        # Log statistics about the final sample time values
        sample_time_stats = result[column_name].describe()
        self.logger.debug(
            f"Sample time statistics: mean={sample_time_stats['mean']:.2f}, std={sample_time_stats['std']:.2f}")

        # Keep backwards compatibility for MINUTES
        if unit == TimeUnit.MINUTES:
            self.logger.debug("Creating SAMPLE_TIME_MINUTES column for backwards compatibility")
            result['SAMPLE_TIME_MINUTES'] = result[column_name]

        # Keep timedelta if requested
        if keep_timedelta:
            self.logger.debug("Keeping timedelta column as requested")
            result['SAMPLE_TIME_DELTA'] = time_diffs

        self.logger.info(f"Successfully created sample time column with {len(result)} rows")
        return result

    def create_sample_time(self,
                           datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                           timestamp_column: str,
                           existing_sampletime_column: str = None,
                           keep_timedelta: bool = False,
                           unit: Union[TimeUnit, str] = TimeUnit.MINUTES) -> Union[
        pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Creates a sample time column (if it doesn't already exist)
        Works with either a single dataframe or a dictionary of dataframes

        :param datafile: the datafile or dictionary of datafiles being transformed
        :param timestamp_column: Column used to calculate sample time
        :param existing_sampletime_column: the column name containing the existing sample time
        :param keep_timedelta: keep the time delta column
        :param unit: target time unit for the sample time column
        :return: Modified datafile or dictionary of datafiles with sample time column(s)
        """

        if isinstance(datafile, dict):
            self.logger.info(f"Creating sample time for {len(datafile)} dataframes")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._create_sample_time_single(
                    df, timestamp_column, existing_sampletime_column, keep_timedelta, unit
                )

            self.logger.info("Completed sample time creation for all dataframes")
            return result
        else:
            self.logger.info(f"Creating sample time for single dataframe with shape {datafile.shape}")
            return self._create_sample_time_single(
                datafile, timestamp_column, existing_sampletime_column, keep_timedelta, unit
            )

    def convert_sample_time_units(self,
                                  datafile: pd.DataFrame,
                                  sample_time_column: str,
                                  from_unit: Union[TimeUnit, str],
                                  to_unit: Union[TimeUnit, str],
                                  new_column_name: str = None) -> pd.DataFrame:
        """
        Convert existing sample time column to different units.

        :param datafile: dataframe containing sample time data
        :param sample_time_column: name of existing sample time column
        :param from_unit: current unit of the sample time column
        :param to_unit: target unit for conversion
        :param new_column_name: name for new column (defaults to SAMPLE_TIME_{UNIT_NAME})
        :return: dataframe with converted sample time column
        """

        if isinstance(from_unit, str):
            self.logger.debug(f"Converting from_unit string '{from_unit}' to TimeUnit enum")
            from_unit = TimeUnit.from_string(from_unit)
        if isinstance(to_unit, str):
            self.logger.debug(f"Converting to_unit string '{to_unit}' to TimeUnit enum")
            to_unit = TimeUnit.from_string(to_unit)

        self.logger.info(f"Converting sample time units from {from_unit.name} to {to_unit.name}")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")

        if sample_time_column not in datafile.columns:
            self.logger.error(f"Sample time column '{sample_time_column}' not found in dataframe")
            self.logger.debug(f"Available columns: {list(datafile.columns)}")
            raise ValueError(f"Sample time column '{sample_time_column}' not found")

        if new_column_name is None:
            new_column_name = f'SAMPLE_TIME_{to_unit.name}'

        self.logger.debug(f"Creating new column: {new_column_name}")
        self.logger.debug(f"Source column '{sample_time_column}' dtype: {datafile[sample_time_column].dtype}")

        result = datafile.copy()
        result[new_column_name] = result[sample_time_column].apply(
            lambda value: convert_time(value, from_unit, to_unit) if pd.notna(value) else value
        )

        # Log conversion statistics
        converted_stats = result[new_column_name].describe()
        self.logger.debug(
            f"Converted values statistics: mean={converted_stats['mean']:.2f}, std={converted_stats['std']:.2f}")

        self.logger.info(f"Successfully converted sample time units to column: {new_column_name}")
        return result

    def _sort_by_datetime_single(self,
                                 datafile: pd.DataFrame,
                                 time_col: str) -> pd.DataFrame:
        """
        Sort dataframe by datetime column if it exists

        :param datafile: DataFrame to sort
        :param time_col: Name of the datetime column to sort by
        :return: Sorted DataFrame
        """

        if time_col not in datafile.columns:
            self.logger.debug(f"Time column '{time_col}' not found - proceeding without sorting")
            self.logger.debug(f"Available columns: {list(datafile.columns)}")
            return datafile.copy()

        self.logger.info(f"Sorting dataframe by datetime column: {time_col}")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")

        result = datafile.copy()

        # Check if column is already datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
            self.logger.debug(f"Converting {time_col} to datetime before sorting")
            result[time_col] = pd.to_datetime(result[time_col])

        # Store original order for comparison
        original_order = result.index.tolist()

        # Sort by datetime
        result = result.sort_values(by=time_col)

        # Check if order changed
        new_order = result.index.tolist()
        if original_order != new_order:
            self.logger.debug("Dataframe was reordered based on timestamp")

            # Log time range for validation
            time_range = result[time_col].agg(['min', 'max'])
            self.logger.debug(f"Time range after sorting: {time_range['min']} to {time_range['max']}")
        else:
            self.logger.debug("Dataframe was already in correct temporal order")

        self.logger.info(f"Successfully sorted {len(result)} rows by datetime")
        return result

    def sort_by_datetime(self,
                         datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         time_col: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Sort dataframe(s) by datetime column
        Works with either a single dataframe or a dictionary of dataframes

        :param datafile: DataFrame or dictionary of DataFrames to sort
        :param time_col: Name of the datetime column to sort by
        :return: Sorted DataFrame(s)
        """

        if isinstance(datafile, dict):
            self.logger.info(f"Sorting {len(datafile)} dataframes by datetime")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._sort_by_datetime_single(df, time_col)

            self.logger.info("Completed datetime sorting for all dataframes")
            return result
        else:
            self.logger.info(f"Sorting single dataframe with shape {datafile.shape}")
            return self._sort_by_datetime_single(datafile, time_col)