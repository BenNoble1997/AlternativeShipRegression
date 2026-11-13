"""
A class for storing distance transformations that can be applied to a pandas dataframe

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""
import logging
import pandas as pd
import numpy as np
from typing import Union, Dict

#for type hinting
from pandas import DataFrame

from config.loggermixin.logger_config import LoggerMixin
from data_processing.data_processing_enum_classes import DistanceUnit


class DistanceTransformation(LoggerMixin):
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
                         datafile: DataFrame|dict[str,DataFrame],
                         time_col: str) -> DataFrame|dict[str,DataFrame]:
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

    def _convert_to_decimal_coordinates_single(self,
                                               datafile: pd.DataFrame,
                                               lat_degrees_col: str = 'LAT_DEGREES',
                                               lat_minutes_col: str = 'LAT_MINUTES',
                                               lat_letter_col: str = 'LAT_LETTER',
                                               long_degrees_col: str = 'LONG_DEGREES',
                                               long_minutes_col: str = 'LONG_MINUTES',
                                               long_letter_col: str = 'LONG_LETTER',
                                               lat_output_col: str = 'LATITUDE',
                                               long_output_col: str = 'LONGITUDE'):

        self.logger.info(f"Converting coordinates to decimal format")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")
        self.logger.debug(f"Input columns - lat: {lat_degrees_col}, {lat_minutes_col}, {lat_letter_col}")
        self.logger.debug(f"Input columns - long: {long_degrees_col}, {long_minutes_col}, {long_letter_col}")
        self.logger.debug(f"Output columns - lat: {lat_output_col}, long: {long_output_col}")

        # Validate required columns exist
        required_cols = [lat_degrees_col, lat_minutes_col, lat_letter_col,
                         long_degrees_col, long_minutes_col, long_letter_col]
        missing_cols = [col for col in required_cols if col not in datafile.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            self.logger.debug(f"Available columns: {list(datafile.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = datafile.copy()

        # Convert latitude
        self.logger.debug("Converting latitude coordinates")
        df[lat_output_col] = df.apply(
            lambda row: row[lat_degrees_col] + row[lat_minutes_col] / 60
            if row[lat_letter_col] in ['N', 'E']
            else -1 * (row[lat_degrees_col] + row[lat_minutes_col] / 60),
            axis=1
        )

        # Convert longitude
        self.logger.debug("Converting longitude coordinates")
        df[long_output_col] = df.apply(
            lambda row: row[long_degrees_col] + row[long_minutes_col] / 60
            if row[long_letter_col] in ['E', 'N']
            else -1 * (row[long_degrees_col] + row[long_minutes_col] / 60),
            axis=1
        )

        # Log coordinate statistics
        lat_stats = df[lat_output_col].describe()
        long_stats = df[long_output_col].describe()
        self.logger.debug(f"Latitude range: {lat_stats['min']:.6f} to {lat_stats['max']:.6f}")
        self.logger.debug(f"Longitude range: {long_stats['min']:.6f} to {long_stats['max']:.6f}")

        self.logger.info(f"Successfully converted {len(df)} coordinate pairs to decimal format")
        return df

    def convert_to_decimal_coordinates(self,
                                       datafile: pd.DataFrame|dict[str, pd.DataFrame],
                                       lat_degrees_col: str = 'LAT_DEGREES',
                                       lat_minutes_col: str = 'LAT_MINUTES',
                                       lat_letter_col: str = 'LAT_LETTER',
                                       long_degrees_col: str = 'LONG_DEGREES',
                                       long_minutes_col: str = 'LONG_MINUTES',
                                       long_letter_col: str = 'LONG_LETTER',
                                       lat_output_col: str = 'LATITUDE',
                                       long_output_col: str = 'LONGITUDE'):

        if isinstance(datafile, dict):
            self.logger.info(f"Converting decimal coordinates for {len(datafile)} dataframes")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._convert_to_decimal_coordinates_single(datafile=df,
                                                                          lat_degrees_col=lat_degrees_col,
                                                                          lat_minutes_col=lat_minutes_col,
                                                                          lat_letter_col=lat_letter_col,
                                                                          long_degrees_col=long_degrees_col,
                                                                          long_minutes_col=long_minutes_col,
                                                                          long_letter_col=long_letter_col,
                                                                          lat_output_col=lat_output_col,
                                                                          long_output_col=long_output_col)

            self.logger.info("Completed decimal coordinate conversion for all dataframes")
            return result
        else:
            self.logger.info(f"Converting decimal coordinates for single dataframe with shape {datafile.shape}")
            return self._convert_to_decimal_coordinates_single(datafile=datafile,
                                                               lat_degrees_col=lat_degrees_col,
                                                               lat_minutes_col=lat_minutes_col,
                                                               lat_letter_col=lat_letter_col,
                                                               long_degrees_col=long_degrees_col,
                                                               long_minutes_col=long_minutes_col,
                                                               long_letter_col=long_letter_col,
                                                               lat_output_col=lat_output_col,
                                                               long_output_col=long_output_col)


    def _add_distance_column_single(self,
                                    datafile: pd.DataFrame,
                                    lat_col: str = 'LATITUDE',
                                    lon_col: str = 'LONGITUDE',
                                    time_col: str = 'TIMESTAMP',
                                    distance_unit: str|DistanceUnit = DistanceUnit.KILOMETERS,
                                    result_col: str = None,
                                    cumulative_col: str = None,
                                    include_cumulative: bool = False) -> pd.DataFrame:
        """
        Add distance columns to the dataframe
        - Distance between consecutive points
        - Cumulative distance travelled
        """

        # Convert unit to enum if it's a string
        if isinstance(distance_unit, str):
            self.logger.debug(f"Converting distance unit string '{distance_unit}' to DistanceUnit enum")
            distance_unit = DistanceUnit.from_string(distance_unit)

        self.logger.info(f"Adding distance column in {distance_unit.name} units")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")
        self.logger.debug(f"Using columns - lat: {lat_col}, lon: {lon_col}, time: {time_col}")
        self.logger.debug(f"Include cumulative distance: {include_cumulative}")

        # Validate required columns
        required_cols = [lat_col, lon_col]
        missing_cols = [col for col in required_cols if col not in datafile.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns for distance calculation: {missing_cols}")
            self.logger.debug(f"Available columns: {list(datafile.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Auto-generate column names based on unit
        if result_col is None:
            result_col = f'DISTANCE_{distance_unit.symbol.upper()}'
        if cumulative_col is None:
            cumulative_col = f'CUMULATIVE_DISTANCE_{distance_unit.name}'

        self.logger.debug(f"Distance column name: {result_col}")
        if include_cumulative:
            self.logger.debug(f"Cumulative distance column name: {cumulative_col}")

        # Sort the dataframe by timestamp using the dedicated function
        result = self._sort_by_datetime_single(datafile, time_col)

        # Initialize the distance column with zeros
        result[result_col] = 0.0
        self.logger.debug("Initialized distance column with zeros")

        # Calculate distances between consecutive points
        if len(result) > 1:
            self.logger.debug(f"Calculating Haversine distances for {len(result)} points")

            # Get coordinates for current points and previous points
            lat1 = result[lat_col].shift(1)
            lon1 = result[lon_col].shift(1)
            lat2 = result[lat_col]
            lon2 = result[lon_col]

            # Log coordinate range for validation
            self.logger.debug(f"Latitude range: {lat2.min():.6f} to {lat2.max():.6f}")
            self.logger.debug(f"Longitude range: {lon2.min():.6f} to {lon2.max():.6f}")

            # Convert to radians
            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)

            # Haversine formula components
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # Earth radius in meters (more precise base unit)
            R_METERS = 6371000  # Earth radius in meters
            self.logger.debug(f"Using Earth radius: {R_METERS} meters")

            # Calculate distance in meters first
            distance_meters = R_METERS * c

            # Convert to desired unit using the enum
            distance_in_unit = distance_unit.from_meters(distance_meters)

            # DEBUG: Check what we calculated
            print(f"First 10 calculated distances (raw): {distance_in_unit[:10].tolist()}")
            print(f"Result column before assignment: {result[result_col][:10].tolist()}")
            print(f"Distance unit conversion factor: {distance_unit.symbol}")

            # FIXED: Use explicit indexing to ensure proper assignment
            # Set the first row to 0 (no previous point to calculate distance from)
            result.iloc[0, result.columns.get_loc(result_col)] = 0.0

            # Assign distances for rows 1 onwards
            for i in range(1, len(result)):
                if not np.isnan(distance_in_unit.iloc[i]) and not np.isinf(distance_in_unit.iloc[i]):
                    result.iloc[i, result.columns.get_loc(result_col)] = distance_in_unit.iloc[i]
                else:
                    result.iloc[i, result.columns.get_loc(result_col)] = 0.0

            # DEBUG: Check what we assigned
            print(f"Result column after assignment: {result[result_col][:10].tolist()}")
            print(f"Non-zero distances count: {(result[result_col] > 0).sum()}")

            # Log distance statistics for debugging
            non_zero_distances = result[result_col][result[result_col] > 0]
            if len(non_zero_distances) > 0:
                self.logger.debug(f"Distance statistics ({distance_unit.symbol}): "
                                  f"mean={non_zero_distances.mean():.6f}, "
                                  f"median={non_zero_distances.median():.6f}, "
                                  f"max={non_zero_distances.max():.6f}, "
                                  f"non-zero count={len(non_zero_distances)}")
            else:
                self.logger.warning("No non-zero distances calculated - this might indicate an issue!")

            # Additional debugging: show sample of calculated distances
            self.logger.debug(f"Sample distances: {result[result_col].head(10).tolist()}")

        else:
            self.logger.debug("Only one or no data points - no distances to calculate")

        # Calculate cumulative distance
        if include_cumulative:
            self.logger.debug("Calculating cumulative distance")
            result[cumulative_col] = result[result_col].cumsum()
            total_distance = result[cumulative_col].iloc[-1] if len(result) > 0 else 0
            self.logger.debug(f"Total cumulative distance: {total_distance:.3f} {distance_unit.symbol}")

        # Final verification
        self.logger.debug(f"Final result shape: {result.shape}")
        self.logger.debug(f"Final result columns: {list(result.columns)}")
        self.logger.debug(f"Distance column type: {result[result_col].dtype}")
        self.logger.debug(
            f"Distance column stats: min={result[result_col].min():.6f}, max={result[result_col].max():.6f}")

        self.logger.info(f"Successfully added distance column with {len(result)} rows")
        return result

    def add_distance_column(self,
                            datafile: pd.DataFrame|dict[str, pd.DataFrame],
                            lat_col: str = 'LATITUDE',
                            lon_col: str = 'LONGITUDE',
                            time_col: str = 'TIMESTAMP',
                            distance_unit: str|DistanceUnit = DistanceUnit.KILOMETERS,
                            result_col: str = None,
                            cumulative_col: str = None,
                            include_cumulative: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Add distance columns to the dataframe or dictionary of dataframes
        - Distance between consecutive points
        - Cumulative distance travelled
        Works with either a single dataframe or a dictionary of dataframes

        :param datafile: file or dictionary of files containing the data being modified
        :param lat_col: Latitude column name
        :param lon_col: Longitude column name
        :param time_col: Time column name
        :param distance_unit: Unit for distance calculations (enum or string)
        :param result_col: returns the distance column (auto-generated if None)
        :param cumulative_col: returns the cumulative distance column (auto-generated if None)
        :param include_cumulative: whether to include the cumulative distance column
        :return: Modified datafile or dictionary of datafiles with distance column(s)
        """

        if isinstance(datafile, dict):
            self.logger.info(f"Adding distance columns for {len(datafile)} dataframes")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._add_distance_column_single(
                    df, lat_col, lon_col, time_col, distance_unit,
                    result_col, cumulative_col, include_cumulative
                )

            self.logger.info("Completed distance column addition for all dataframes")
            return result
        else:
            self.logger.info(f"Adding distance column for single dataframe with shape {datafile.shape}")
            return self._add_distance_column_single(
                datafile, lat_col, lon_col, time_col, distance_unit,
                result_col, cumulative_col, include_cumulative
            )

    def convert_distance_units(self,
                               datafile: pd.DataFrame,
                               distance_columns: str|list,
                               from_unit: str|DistanceUnit,
                               to_unit: str|DistanceUnit,
                               suffix: str = None,
                               replace_original: bool = False) -> pd.DataFrame:
        """
        Convert existing distance columns to different units.

        :param datafile: DataFrame containing distance data
        :param distance_columns: Column name(s) to convert
        :param from_unit: Current unit of the distance columns
        :param to_unit: Target unit for conversion
        :param suffix: Suffix for new column names (defaults to unit name)
        :param replace_original: If True, replaces original columns; if False, creates new ones
        :return: DataFrame with converted distance columns
        """

        if isinstance(from_unit, str):
            self.logger.debug(f"Converting from_unit string '{from_unit}' to DistanceUnit enum")
            from_unit = DistanceUnit.from_string(from_unit)
        if isinstance(to_unit, str):
            self.logger.debug(f"Converting to_unit string '{to_unit}' to DistanceUnit enum")
            to_unit = DistanceUnit.from_string(to_unit)

        self.logger.info(f"Converting distance units from {from_unit.name} to {to_unit.name}")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")

        if isinstance(distance_columns, str):
            distance_columns = [distance_columns]

        self.logger.debug(f"Converting columns: {distance_columns}")
        self.logger.debug(f"Replace original columns: {replace_original}")

        # Validate columns exist
        missing_cols = [col for col in distance_columns if col not in datafile.columns]
        if missing_cols:
            self.logger.error(f"Distance columns not found: {missing_cols}")
            self.logger.debug(f"Available columns: {list(datafile.columns)}")
            raise ValueError(f"Distance columns not found: {missing_cols}")

        if suffix is None:
            suffix = f"_{to_unit.name}"

        result = datafile.copy()

        for col in distance_columns:
            self.logger.debug(f"Processing column: {col}")

            # Determine new column name
            if replace_original:
                new_col_name = col  # Keep same name, just convert values
                self.logger.debug(f"Replacing original column: {col}")
            else:
                # Create new column with updated name
                new_col_name = f"{col.replace(from_unit.name, to_unit.name)}"
                if new_col_name == col:  # If no unit in original name
                    new_col_name = f"{col}{suffix}"
                self.logger.debug(f"Creating new column: {new_col_name}")

            # Log original data statistics
            original_stats = result[col].describe()
            self.logger.debug(f"Original {col} statistics ({from_unit.symbol}): "
                              f"mean={original_stats['mean']:.3f}, max={original_stats['max']:.3f}")

            # Convert using enum methods
            converted_values = result[col].apply(
                lambda value: to_unit.from_meters(from_unit.to_meters(value))
                if pd.notna(value) else value
            )

            # Assign to new or existing column
            result[new_col_name] = converted_values

            # Log converted data statistics
            converted_stats = converted_values.describe()
            self.logger.debug(f"Converted {new_col_name} statistics ({to_unit.symbol}): "
                              f"mean={converted_stats['mean']:.3f}, max={converted_stats['max']:.3f}")

            # If replacing and column name changed, drop the original
            if replace_original and new_col_name != col:
                self.logger.debug(f"Dropping original column: {col}")
                result = result.drop(columns=[col])

        self.logger.info(f"Successfully converted {len(distance_columns)} distance column(s)")
        return result