import pandas as pd
import numpy as np
import logging
from config.loggermixin.logger_config import LoggerMixin
from typing import List, Dict, Union, Any

class DifferentialFilters(LoggerMixin):

    def __init__(self,
                 verbose:bool = True,
                 debug:bool = False,):

        self.verbose = verbose
        self.debug = debug

    def build_differential_filter(self,
                                  column: Union[str, List[str]],
                                  max_diff:float|None = None,
                                  min_diff: float|None = None) -> dict[str, Any]:
        """
        Build a differential filter dictionary for use with differential filtering.

        This helper method makes it easy to create properly formatted differential filter
        dictionaries for controlling the rate of change in time series data.

        :param column:The column to calculate differentials on. If a list is provided,
                   only the first column will be used for differential calculation.
        :param max_diff:Maximum allowed difference between consecutive values (inclusive).
                     Rows with higher differences will be removed.
        :param min_diff:Minimum allowed difference between consecutive values (inclusive).
                     Rows with lower differences will be removed.

        :param return: A properly formatted differential filter dictionary

        Examples:
            # Create a filter to keep rows where speed changes by at most 5 units
            speed_diff_filter = filter_builder.build_differential_filter('speed', max_diff=5)

            # Create a filter to keep only significant changes in temperature
            temp_diff_filter = filter_builder.build_differential_filter('temperature', min_diff=2)

            # Create a filter for moderate changes in pressure (between 1 and 10 units)
            pressure_diff_filter = filter_builder.build_differential_filter('pressure', min_diff=1, max_diff=10)

            # Apply the differential filter
            filtered_df = filter_builder.filter_by_differential(
                df,
                column_name=pressure_diff_filter['column'],
                max_diff=pressure_diff_filter['max_diff'],
                min_diff=pressure_diff_filter['min_diff']
            )
        """
        # Validate inputs
        if max_diff is None and min_diff is None:
            self.logger.warning("Both max_diff and min_diff are None. No filtering will be applied.")

        if max_diff is not None and max_diff < 0:
            self.logger.warning(f"Negative max_diff ({max_diff}) specified. Using absolute value.")
            max_diff = abs(max_diff)

        if min_diff is not None and min_diff < 0:
            self.logger.warning(f"Negative min_diff ({min_diff}) specified. Using absolute value.")
            min_diff = abs(min_diff)

        if max_diff is not None and min_diff is not None and max_diff < min_diff:
            self.logger.error(f"Invalid differential range: max_diff ({max_diff}) is less than min_diff ({min_diff})")
            raise ValueError(f"max_diff must be greater than or equal to min_diff")

        # Create and return the differential filter dictionary
        diff_filter = {
            'column': column,
            'max_diff': max_diff,
            'min_diff': min_diff
        }

        return diff_filter

    def _filter_differential_single_df(self,
                                       datafile: pd.DataFrame,
                                       column_name: str|List[str],
                                       max_diff: float|None = None,
                                       min_diff: float|None = None) -> pd.DataFrame:
        """
        Helper method to filter a single dataframe by differential.
        Internal use only - called by filter_by_differential.
        """
        filtered_df = datafile.copy()
        initial_count = len(filtered_df)

        # Handle both string and list inputs
        col = column_name[0] if isinstance(column_name, list) else column_name

        # Ensure the column exists
        if col not in filtered_df.columns:
            self.logger.error(f"Column not found: {col}")
            raise KeyError(f"Column '{col}' not found in DataFrame")

        # Calculate the differential
        diff = filtered_df[col].diff().abs()

        # Keep track of which rows to keep
        mask = pd.Series(True, index=filtered_df.index)

        filter_conditions = []

        if max_diff is not None:
            # First row will have NaN diff, we want to keep it
            mask = mask & (diff.isna() | (diff <= max_diff))
            filter_conditions.append(f"max_diff: {max_diff}")

        if min_diff is not None:
            mask = mask & (diff.isna() | (diff >= min_diff))
            filter_conditions.append(f"min_diff: {min_diff}")

        # Apply the mask to keep the rows we want
        filtered_df = filtered_df[mask]

        # Log the results
        total_removed = initial_count - len(filtered_df)
        filter_desc = " and ".join(filter_conditions)

        self.logger.info(
            f'Filtered by differential on column {col} with {filter_desc}. '
            f'Removed {total_removed:,} rows ({total_removed / initial_count:.1%}). '
            f'{len(filtered_df):,} data points remaining.'
        )

        return filtered_df

    def filter_by_differential(self,
                               datafile: pd.DataFrame|dict[str, pd.DataFrame],
                               column_name: str|List[str],
                               max_diff: float|None= None,
                               min_diff: float|None = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Filter rows based on the differential between consecutive values in a column.

        This function removes rows where the absolute difference between consecutive values
        in the specified column is outside the given bounds. Works with either a single
        DataFrame or a dictionary of DataFrames.

        :param datafile: A DataFrame or dictionary of DataFrames to filter
        :param column_name: The column to calculate differentials on. If a list is provided,
                        only the first column will be used for differential calculation.
        :param max_diff: Maximum allowed difference between consecutive values (inclusive).
                     Rows with higher differences will be removed.
        :param min_diff: Minimum allowed difference between consecutive values (inclusive).
                     Rows with lower differences will be removed.

        Returns:
            Filtered DataFrame or dictionary of filtered DataFrames

        Examples:
            # Filter a single DataFrame to keep only rows where the speed changes
            # by at most 5 units between consecutive readings
            filtered_df = filter_builder.filter_by_differential(df, 'speed', max_diff=5)

            # Filter multiple DataFrames in a dictionary
            filtered_ships = filter_builder.filter_by_differential(
                ship_data, 'speed', max_diff=5, min_diff=0.1
            )
        """
        # Check if neither max_diff nor min_diff is provided
        if max_diff is None and min_diff is None:
            self.logger.warning("No filter criteria provided. Returning original data.")
            return datafile

        # Check if we're dealing with a dictionary of dataframes
        if isinstance(datafile, Dict):
            self.logger.info(f"Filtering differentials for multiple datafiles ({len(datafile)} files)")
            result_dict = {}

            # Process each dataframe in the dictionary
            for key, df in datafile.items():
                self.logger.info(f"Filtering differentials for dataframe '{key}'")
                initial_count = len(df)

                # Apply differential filtering to this specific dataframe
                filtered_df = self._filter_differential_single_df(
                    df, column_name, max_diff, min_diff
                )

                # Log results for this specific dataframe
                total_removed = initial_count - len(filtered_df)
                self.logger.info(
                    f"Dataframe '{key}': {len(filtered_df):,} rows remaining from {initial_count:,} "
                    f"({total_removed:,} removed, {total_removed / initial_count:.1%} of original data)"
                )

                result_dict[key] = filtered_df

            return result_dict
        else:
            # Single dataframe case
            return self._filter_differential_single_df(datafile, column_name, max_diff, min_diff)

    def apply_multiple_differential_filters(self,
                                            datafile: pd.DataFrame|dict[str, pd.DataFrame],
                                            filters: dict[str, Any]|List[dict[str, Any]]) ->pd.DataFrame|dict[str, pd.DataFrame]:
        """
        Apply multiple differential filters sequentially to a DataFrame or dictionary of DataFrames.

        This method processes one or more differential filters in sequence. Each filter is defined
        by a dictionary specifying the column to calculate differentials on, and the max/min
        allowed differences.

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :param filters: A list of differential filter dictionaries created with build_differential_filter.
                    Each dictionary should have:
                    - 'column': str or List[str] - Column(s) to calculate differentials for
                    - 'max_diff': float (optional) - Maximum allowed difference
                    - 'min_diff': float (optional) - Minimum allowed difference

        Returns:
            Filtered DataFrame or dictionary of filtered DataFrames

        Examples:
            # Create multiple differential filters
            filters = [
                filter_builder.build_differential_filter('speed', max_diff=5),
                filter_builder.build_differential_filter('temperature', min_diff=1, max_diff=10)
            ]

            # Apply all filters
            filtered_df = filter_builder.apply_multiple_differential_filters(df, filters)

            # Apply to multiple DataFrames
            filtered_ships = filter_builder.apply_multiple_differential_filters(ship_data, filters)
        """


        if not filters:
            self.logger.warning("No filters provided. Returning original data.")
            return datafile

        if isinstance(filters,dict):
            filters = [filters]

        self.logger.info(f"Beginning application of {len(filters)} differential filters")

        # Make a copy to avoid modifying the original
        result = datafile.copy() if isinstance(datafile, pd.DataFrame) else {
            key: df.copy() for key, df in datafile.items()
        }

        # Apply each filter in sequence
        for i, filter_dict in enumerate(filters, 1):
            try:
                column = filter_dict['column']
                max_diff = filter_dict.get('max_diff')
                min_diff = filter_dict.get('min_diff')

                # Validate that at least one of max_diff or min_diff is provided
                if max_diff is None and min_diff is None:
                    self.logger.warning(f"Filter {i}: Both max_diff and min_diff are None. Skipping this filter.")
                    continue

                column_desc = column if isinstance(column, str) else column[0]
                filter_desc = []
                if max_diff is not None:
                    filter_desc.append(f"max_diff={max_diff}")
                if min_diff is not None:
                    filter_desc.append(f"min_diff={min_diff}")

                self.logger.info(f"Applying differential filter {i}: {column_desc} with {' and '.join(filter_desc)}")

                # Apply the filter
                result = self.filter_by_differential(
                    result,
                    column_name=column,
                    max_diff=max_diff,
                    min_diff=min_diff
                )

            except KeyError as e:
                self.logger.error(f"Missing required key in filter {i}: {str(e)}")
                raise KeyError(
                    f"Differential filter dictionary must contain 'column'. Missing: {str(e)}"
                )

        self.logger.info(f"Differential filter sequence completed")

        return result