"""
Contains class for creating speed columns

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""
import logging
import pandas as pd
import numpy as np
from typing import Union, Dict

from config.loggermixin.logger_config import LoggerMixin
from data_processing.data_processing_enum_classes import TimeUnit, SpeedUnit, DistanceUnit
from data_processing.transform.time_transformation_class import TimeTransform
from data_processing.transform.Distance_transformation_class import DistanceTransformation


class SpeedTransformation(LoggerMixin):

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

        self.distance_transformation = DistanceTransformation(verbose=verbose, debug=debug)
        self.time_transformation = TimeTransform(verbose=verbose, debug=debug)

    def _calculated_speed_column_single(self,
                                        datafile: pd.DataFrame,
                                        distance_col: str = 'DISTANCE_KM',
                                        time_col: str = 'TIMESTAMP',
                                        sample_time_col: str = 'SAMPLE_TIME',
                                        lat_col: str = 'LATITUDE',
                                        lon_col: str = 'LONGITUDE',
                                        speed_unit: Union[str, SpeedUnit] = SpeedUnit.KNOTS,
                                        distance_unit: Union[str, DistanceUnit] = None,
                                        time_unit: Union[str, TimeUnit] = None,
                                        result_col: str = None) -> pd.DataFrame:
        """
        A method for creating a calculated speed column
        :param datafile: pandas dataframe containing the required data
        :param distance_col: the distance column being used
        :param time_col: the time column being used
        :param sample_time_col: the sample time column being used
        :param lat_col: The latitude column being used
        :param lon_col: The longitude column being used
        :param speed_unit: the target speed unit (enum or string)
        :param distance_unit: unit of the distance column (auto-detected if None)
        :param time_unit: unit of the sample time column (auto-detected if None)
        :param result_col: name for the speed column (auto-generated if None)
        :return: A column storing the speed of the ship during this operation
        """

        # Convert speed unit to enum if string
        if isinstance(speed_unit, str):
            self.logger.debug(f"Converting speed unit string '{speed_unit}' to SpeedUnit enum")
            speed_unit = SpeedUnit.from_string(speed_unit)

        self.logger.info(f"Calculating speed column in {speed_unit.name} units")
        self.logger.debug(f"Input dataframe shape: {datafile.shape}")
        self.logger.debug(f"Parameters - distance_col: {distance_col}, time_col: {time_col}")
        self.logger.debug(f"Parameters - sample_time_col: {sample_time_col}, lat_col: {lat_col}, lon_col: {lon_col}")
        self.logger.debug(f"Parameters - distance_unit: {distance_unit}, time_unit: {time_unit}")

        result = datafile.copy()

        # Check if distance column exists, calculate if missing
        if distance_col not in result.columns:
            self.logger.warning(f'Distance column {distance_col} not found in datafile')
            self.logger.debug(f"Available columns: {list(result.columns)}")
            self.logger.info('Calculating distance column using Haversine method')

            # Validate required columns for distance calculation
            required_cols = [lat_col, lon_col, time_col]
            missing_cols = [col for col in required_cols if col not in result.columns]
            if missing_cols:
                self.logger.error(f"Cannot calculate distance - missing columns: {missing_cols}")
                raise ValueError(f"Missing required columns for distance calculation: {missing_cols}")

            result = self.distance_transformation._add_distance_column_single(
                datafile=result,
                lat_col=lat_col,
                lon_col=lon_col,
                time_col=time_col,
                result_col=distance_col
            )

            if result is None:
                self.logger.error("Distance transformation returned None!")
                raise ValueError("Failed to calculate distance column")

            self.logger.debug(f"Distance column created. New dataframe shape: {result.shape}")
        else:
            self.logger.debug(f"Distance column {distance_col} found in dataframe")

        # Check if sample time column exists, calculate if missing
        if sample_time_col and sample_time_col not in result.columns:
            self.logger.warning(f'Sample time column {sample_time_col} not found in datafile')
            self.logger.debug(f"Available columns: {list(result.columns)}")
            self.logger.info('Calculating sample time column using timestamp column')

            if time_col not in result.columns:
                self.logger.error(f"Cannot calculate sample time - timestamp column {time_col} not found")
                raise ValueError(f"Timestamp column {time_col} required for sample time calculation")

            result = self.time_transformation._create_sample_time_single(
                datafile=result,
                timestamp_column=time_col
            )

            if result is None:
                self.logger.error("Time transformation returned None!")
                raise ValueError("Failed to calculate sample time column")

            self.logger.debug(f"Sample time column created. New dataframe shape: {result.shape}")
        elif sample_time_col:
            self.logger.debug(f"Sample time column {sample_time_col} found in dataframe")

        # Auto-detect or convert distance unit
        if distance_unit is None:
            self.logger.debug(f"Auto-detecting distance unit from column name: {distance_col}")
            distance_unit = DistanceUnit.detect_from_column_name(distance_col)
            self.logger.debug(f"Detected distance unit: {distance_unit.name}")
        elif isinstance(distance_unit, str):
            self.logger.debug(f"Converting distance unit string '{distance_unit}' to DistanceUnit enum")
            distance_unit = DistanceUnit.from_string(distance_unit)

        # Auto-detect or convert time unit
        if time_unit is None:
            self.logger.debug(f"Auto-detecting time unit from column name: {sample_time_col}")
            time_unit = TimeUnit.detect_from_column_name(sample_time_col)
            self.logger.debug(f"Detected time unit: {time_unit.name}")
        elif isinstance(time_unit, str):
            self.logger.debug(f"Converting time unit string '{time_unit}' to TimeUnit enum")
            time_unit = TimeUnit.from_string(time_unit)

        # Auto-generate result column name if not provided
        if result_col is None:
            result_col = f'CALCULATED_SPEED_{speed_unit.name}'
            self.logger.debug(f"Auto-generated result column name: {result_col}")

        self.logger.info(f"Calculating speed: {distance_unit.symbol}/{time_unit.symbol} -> {speed_unit.symbol}")
        self.logger.debug(f"Using distance column: {distance_col}, sample time column: {sample_time_col}")

        # Validate required columns are present
        required_cols = [distance_col, sample_time_col]
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns for speed calculation: {missing_cols}")
            self.logger.debug(f"Available columns: {list(result.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Log data statistics before calculation
        distance_stats = result[distance_col].describe()
        time_stats = result[sample_time_col].describe()
        self.logger.debug(
            f"Distance statistics ({distance_unit.symbol}): mean={distance_stats['mean']:.3f}, max={distance_stats['max']:.3f}")
        self.logger.debug(
            f"Time statistics ({time_unit.symbol}): mean={time_stats['mean']:.3f}, max={time_stats['max']:.3f}")

        # Calculate speed with proper unit conversions
        self.logger.debug("Step 1: Converting distance to meters")
        distance_meters = result[distance_col] * distance_unit.to_meters_factor
        self.logger.debug(f"Distance conversion factor: {distance_unit.to_meters_factor}")

        self.logger.debug("Step 2: Converting time to seconds")
        time_seconds = result[sample_time_col] * time_unit.to_seconds_factor
        self.logger.debug(f"Time conversion factor: {time_unit.to_seconds_factor}")

        # Check for zero time values before division
        zero_time_count = (time_seconds == 0).sum()
        if zero_time_count > 0:
            self.logger.warning(f"Found {zero_time_count} zero time values - these will result in infinite speed")

        self.logger.debug("Step 3: Calculating speed in m/s")
        speed_mps = distance_meters / time_seconds

        # Count infinite values before handling
        inf_count = np.isinf(speed_mps).sum()
        if inf_count > 0:
            self.logger.debug(f"Found {inf_count} infinite speed values (division by zero)")

        self.logger.debug("Step 4: Handling division by zero and infinite values")
        speed_mps = speed_mps.replace([np.inf, -np.inf], np.nan)

        # Count NaN values after handling
        nan_count = speed_mps.isna().sum()
        self.logger.debug(f"Total NaN values in speed calculation: {nan_count}")

        self.logger.debug(f"Step 5: Converting to target speed unit ({speed_unit.symbol})")
        result[result_col] = speed_unit.from_mps(speed_mps)

        # Log final speed statistics
        final_speed_stats = result[result_col].describe()
        self.logger.debug(f"Final speed statistics ({speed_unit.symbol}): "
                          f"mean={final_speed_stats['mean']:.3f}, "
                          f"median={final_speed_stats['50%']:.3f}, "
                          f"max={final_speed_stats['max']:.3f}")

        # Check for unrealistic speeds (e.g., over 100 knots for ships)
        if speed_unit == SpeedUnit.KNOTS:
            high_speed_count = (result[result_col] > 100).sum()
            if high_speed_count > 0:
                self.logger.warning(f"Found {high_speed_count} unrealistically high speeds (>100 knots)")

        self.logger.info(f"Successfully calculated speed column '{result_col}' with {len(result)} rows")
        self.logger.debug(f"Output dataframe shape: {result.shape}")

        return result

    def calculate_speed_column(self,
                               datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                               distance_col: str = 'DISTANCE_KM',
                               time_col: str = 'TIMESTAMP',
                               sample_time_col: str = 'SAMPLE_TIME',
                               lat_col: str = 'LATITUDE',
                               lon_col: str = 'LONGITUDE',
                               speed_unit: Union[str, SpeedUnit] = SpeedUnit.KNOTS,
                               distance_unit: Union[str, DistanceUnit] = None,
                               time_unit: Union[str, TimeUnit] = None,
                               result_col: str = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate speed column for single DataFrame or dictionary of DataFrames.

        :param datafile: DataFrame or dictionary of DataFrames
        :param distance_col: the distance column being used
        :param time_col: the time column being used
        :param sample_time_col: the sample time column being used
        :param lat_col: The latitude column being used
        :param lon_col: The longitude column being used
        :param speed_unit: the target speed unit (enum or string)
        :param distance_unit: unit of the distance column (auto-detected if None)
        :param time_unit: unit of the sample time column (auto-detected if None)
        :param result_col: name for the speed column (auto-generated if None)
        :return: DataFrame(s) with calculated speed column
        """

        if isinstance(datafile, dict):
            self.logger.info(f"Calculating speed columns for {len(datafile)} dataframes")
            self.logger.debug(f"Dataframe keys: {list(datafile.keys())}")

            result = {}
            for key, df in datafile.items():
                self.logger.debug(f"Processing dataframe '{key}' with shape {df.shape}")
                result[key] = self._calculated_speed_column_single(
                    df, distance_col, time_col, sample_time_col, lat_col, lon_col,
                    speed_unit, distance_unit, time_unit, result_col
                )

                if result[key] is None:
                    self.logger.error(f"Speed calculation returned None for dataframe '{key}'")
                    raise ValueError(f"Failed to calculate speed for dataframe '{key}'")

            self.logger.info("Completed speed calculation for all dataframes")
            return result
        else:
            self.logger.info(f"Calculating speed column for single dataframe with shape {datafile.shape}")
            result = self._calculated_speed_column_single(
                datafile, distance_col, time_col, sample_time_col, lat_col, lon_col,
                speed_unit, distance_unit, time_unit, result_col
            )

            if result is None:
                self.logger.error("Speed calculation returned None!")
                raise ValueError("Failed to calculate speed column")

            return result