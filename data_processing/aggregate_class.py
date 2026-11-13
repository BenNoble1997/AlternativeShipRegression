"""
Module stores the DataAggregator class
"""

import pandas as pd
import os
import sys
import logging

from typing import List, Dict, Optional, Union, Callable

os.chdir(r"/")


sys.path.append(r"/")

from config.loggermixin.logger_config import LoggerMixin
from data_processing.data_processing_enum_classes import AggregationFunction


class DataAggregator(LoggerMixin):
    """
    Handles data aggregation operations including grouping, weighted averages,
    and custom aggregation functions.

    This class supports both single DataFrames and dictionaries of DataFrames,
    providing flexible aggregation capabilities for time-series data analysis
    and statistical computations.

    :param verbose: Enable verbose logging output. Defaults to True.
    :type verbose: bool
    :param debug: Enable debug-level logging. Defaults to False.
    :type debug: bool

    .. note::
       When both verbose and debug are True, debug takes precedence.

    Example:
        Basic usage::

            aggregator = DataAggregator(verbose=True)
            agg_dict = {'temperature': 'mean', 'power': 'sum'}
            result = aggregator.group_and_aggregate(df, 'timestamp', agg_dict, freq='H')
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
                                agg_values: List[str] = None,
                                interactive: bool = True) -> dict[str, str]:
        """
        Create an aggregation dictionary mapping columns to their aggregation functions.

        This method helps build the column-to-function mapping required for data aggregation.
        Can work interactively by prompting user input or use predefined values.

        :param columns: List of column names to create aggregation mapping for
        :type columns: List[str]
        :param agg_values: Optional preset aggregation function names. If None and interactive
                          is True, will prompt for user input. Must match length of columns.
        :type agg_values: Optional[List[str]]
        :param interactive: Whether to prompt user for function selection when agg_values is None.
                           If False and agg_values is None, raises an error.
        :type interactive: bool
        :returns: Dictionary mapping column names to aggregation function names
        :rtype: Dict[str, str]
        :raises ValueError: If agg_values length doesn't match columns length
        :raises RuntimeError: If interactive is False but agg_values is None

        Example:
            Interactive mode::

                agg_dict = aggregator.create_aggregation_dict(['temp', 'pressure'])
                # Prompts user to select functions for each column

            Predefined values::

                agg_dict = aggregator.create_aggregation_dict(
                    ['temp', 'pressure'],
                    ['mean', 'max'],
                    interactive=False
                )
                # Returns: {'temp': 'mean', 'pressure': 'max'}

        .. note::
           Available aggregation functions are displayed when in interactive mode.
           The function names correspond to the AggregationFunction enum values.
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

    def _group_and_aggregate_single(self,
                                    df: pd.DataFrame,
                                    group_key: str,
                                    aggregation_dict: dict[str, str | Callable] | None = None,
                                    freq: str = 'h') -> pd.DataFrame:
        """
        Group a single DataFrame by time key and apply specified aggregations.

        Internal helper method that performs the core aggregation logic on a single DataFrame.
        Separates standard pandas aggregations from custom callable functions for optimal performance.

        :param df: Input DataFrame to aggregate
        :type df: pd.DataFrame
        :param group_key: Column name containing datetime values for grouping
        :type group_key: str
        :param aggregation_dict: Mapping of column names to aggregation functions.
                               Can be strings (for standard functions) or callable objects.
                               If None, will create interactively.
        :type aggregation_dict: Optional[Dict[str, Union[str, Callable]]]
        :param freq: Pandas frequency string for time-based grouping
                    (e.g., 'H' for hourly, 'D' for daily, '15min' for 15-minute intervals)
        :type freq: str
        :returns: Aggregated DataFrame with grouped results
        :rtype: pd.DataFrame
        :raises KeyError: If group_key column is not found in the DataFrame
        :raises Exception: Re-raises any aggregation-related exceptions with context

        .. note::
           This is an internal method. Use :meth:`group_and_aggregate` for the public interface.

        .. seealso::
           :meth:`group_and_aggregate` for the main aggregation interface
        """
        try:
            if aggregation_dict is None:
                aggregation_dict = self.create_aggregation_dict(list(df.columns))

            self.logger.info(f"Starting aggregation process")
            self.logger.info(f"Input DataFrame shape: {df.shape}")
            self.logger.info(f"Grouping by {group_key} with frequency {freq}")
            self.logger.debug(f"Aggregation functions: {aggregation_dict}")

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

    def group_and_aggregate(self,
                            datafile: pd.DataFrame| dict[str, pd.DataFrame],
                            group_key: str,
                            aggregation_dict: dict[str, str | Callable] | None = None,
                            freq: str = 'h') -> pd.DataFrame| dict[str, pd.DataFrame]:
        """
        Group DataFrame(s) by time key and apply specified aggregations.

        Main public interface for time-based data aggregation. Works with either a single
        DataFrame or a dictionary of DataFrames, applying the same aggregation logic to each.

        :param datafile: Input DataFrame or dictionary of DataFrames to aggregate
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param group_key: Column name containing datetime values for time-based grouping
        :type group_key: str
        :param aggregation_dict: Mapping of column names to aggregation functions.
                               Functions can be strings (e.g., 'mean', 'sum') or callable objects.
                               If None, will prompt for interactive selection.
        :type aggregation_dict: Optional[Dict[str, Union[str, Callable]]]
        :param freq: Pandas frequency string for time-based grouping.
                    Common values: 'H' (hourly), 'D' (daily), '15min', 'M' (monthly)
        :type freq: str
        :returns: Aggregated DataFrame or dictionary of aggregated DataFrames (same structure as input)
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :raises KeyError: If group_key column is not found in any DataFrame
        :raises Exception: Re-raises any aggregation-related exceptions

        Example:
            Single DataFrame aggregation::

                agg_dict = {'temperature': 'mean', 'power': 'sum', 'efficiency': 'max'}
                hourly_data = aggregator.group_and_aggregate(
                    df,
                    group_key='timestamp',
                    aggregation_dict=agg_dict,
                    freq='H'
                )

            Multiple DataFrames::

                daily_summaries = aggregator.group_and_aggregate(
                    ship_data_dict,  # Dict of DataFrames
                    group_key='datetime',
                    aggregation_dict=agg_dict,
                    freq='D'
                )

            Custom aggregation functions::

                custom_aggs = {
                    'fuel_rate': lambda x: x.quantile(0.95),  # 95th percentile
                    'speed': 'mean'
                }
                result = aggregator.group_and_aggregate(df, 'time', custom_aggs)

        .. seealso::
           :meth:`create_aggregation_dict` for building aggregation mappings
        """
        if isinstance(datafile, dict):
            self.logger.info(f"Processing dictionary with {len(datafile)} DataFrames")
            result = {}
            for key, df in datafile.items():
                self.logger.info(f"Processing DataFrame '{key}'")
                result[key] = self._group_and_aggregate_single(df, group_key, aggregation_dict, freq)
            return result
        else:
            return self._group_and_aggregate_single(datafile, group_key, aggregation_dict, freq)

    def create_weighted_aggregation(self,
                                    column: str,
                                    weight_column: str) -> Callable:
        """
        Create a weighted aggregation function for a column using specified weights.

        Creates a callable function that computes weighted averages, useful for cases where
        data points have different importance or representativeness.

        :param column: Column name to aggregate (not directly used in function creation,
                      but useful for documentation/clarity)
        :type column: str
        :param weight_column: Column name to use as weights for the aggregation
        :type weight_column: str
        :returns: Callable function that computes weighted mean for the specified weight column
        :rtype: Callable
        :raises AttributeError: If AggregationFunction.create_weighted_mean is not available

        Example:
            Create weighted fuel consumption average by engine power::

                weighted_func = aggregator.create_weighted_aggregation(
                    'fuel_consumption',
                    'engine_power'
                )

                # Use in aggregation
                agg_dict = {
                    'fuel_consumption': weighted_func,
                    'speed': 'mean'
                }

        .. note::
           The returned function calculates: Σ(value_i × weight_i) / Σ(weight_i)

        .. seealso::
           :meth:`create_multiple_weighted_aggregations` for bulk creation of weighted functions
        """
        return AggregationFunction.create_weighted_mean(weight_column)

    def create_multiple_weighted_aggregations(self,
                                              columns: List[str],
                                              weight_column: str) -> Dict[str, Callable]:
        """
        Create weighted aggregations for multiple columns using the same weight.

        Convenience method for creating multiple weighted aggregation functions that all
        use the same weighting column. Useful for fleet analysis where multiple metrics
        need to be weighted by the same factor (e.g., engine size, vessel capacity).

        :param columns: List of column names to create weighted aggregations for
        :type columns: List[str]
        :param weight_column: Column name to use as weight for all aggregations
        :type weight_column: str
        :returns: Dictionary mapping each column to its weighted aggregation function
        :rtype: Dict[str, Callable]

        Example:
            Create weighted aggregations for multiple fuel metrics::

                weighted_aggs = aggregator.create_multiple_weighted_aggregations(
                    ['fuel_consumption', 'fuel_efficiency', 'emissions'], 
                    'engine_power'
                )
                
                # Result: {
                #     'fuel_consumption': weighted_func,
                #     'fuel_efficiency': weighted_func, 
                #     'emissions': weighted_func
                # }
                
                # Use in group aggregation
                result = aggregator.group_and_aggregate(df, 'timestamp', weighted_aggs)

            .. seealso::
           :meth:`create_weighted_aggregation` for single weighted function creation
            """
        return {col: AggregationFunction.create_weighted_mean(weight_column) for col in columns}

    def _apply_weighted_mean_single(self,
                                    df: pd.DataFrame,
                                    value_column: str,
                                    weight_column: str,
                                    group_key: str = None,
                                    freq: str = 'h') -> Union[float, pd.Series]:
        """
        Apply weighted mean calculation to a single DataFrame.

        Internal helper method that performs weighted mean calculations on a single DataFrame,
        with optional time-based grouping.

        :param df: Input DataFrame containing the data
        :type df: pd.DataFrame
        :param value_column: Column name containing values to average
        :type value_column: str
        :param weight_column: Column name containing weights for the calculation
        :type weight_column: str
        :param group_key: Optional column name for time-based grouping. If None,
                         calculates single weighted mean for entire DataFrame.
        :type group_key: Optional[str]
        :param freq: Pandas frequency string for grouping when group_key is provided
        :type freq: str
        :returns: Single weighted mean value (if no grouping) or Series of weighted means (if grouped)
        :rtype: Union[float, pd.Series]
        :raises KeyError: If value_column or weight_column not found in DataFrame
        :raises ZeroDivisionError: If sum of weights equals zero

        .. note::
           This is an internal method. Use :meth:`apply_weighted_mean` for the public interface.

        .. warning::
           Returns NaN if the sum of weights is zero to avoid division by zero errors.
        """
        self.logger.info(f"Calculating weighted mean for {value_column} using {weight_column} as weights")

        # If no grouping, calculate single weighted mean
        if group_key is None:
            weighted_sum = (df[value_column] * df[weight_column]).sum()
            weight_sum = df[weight_column].sum()

            if weight_sum == 0:
                self.logger.warning(f"Sum of weights is zero, returning NaN")
                return float('nan')

            return weighted_sum / weight_sum

        # With grouping, calculate weighted mean for each group
        else:
            weighted_func = AggregationFunction.create_weighted_mean(weight_column)
            return df.groupby(pd.Grouper(key=group_key, freq=freq))[value_column].apply(weighted_func)

    def apply_weighted_mean(self,
                            datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            value_column: str,
                            weight_column: str,
                            group_key: str = None,
                            freq: str = 'h') -> Union[float, pd.Series, Dict[str, Union[float, pd.Series]]]:
        """
        Apply weighted mean calculations to DataFrame(s).

        Main public interface for weighted mean calculations. Works with either a single
        DataFrame or a dictionary of DataFrames. Supports both overall weighted means
        and time-grouped weighted means.

        :param datafile: Input DataFrame or dictionary of DataFrames
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param value_column: Column name containing values to average
        :type value_column: str
        :param weight_column: Column name containing weights for the calculation
        :type weight_column: str
        :param group_key: Optional column name for time-based grouping. If None,
                         calculates overall weighted mean for each DataFrame.
        :type group_key: Optional[str]
        :param freq: Pandas frequency string for grouping when group_key is provided.
                    Examples: 'H' (hourly), 'D' (daily), '15min'
        :type freq: str
        :returns: Weighted mean result(s). Type depends on inputs:

                 * Single DataFrame, no grouping: float
                 * Single DataFrame, with grouping: pd.Series
                 * Dict of DataFrames, no grouping: Dict[str, float]
                 * Dict of DataFrames, with grouping: Dict[str, pd.Series]
        :rtype: Union[float, pd.Series, Dict[str, Union[float, pd.Series]]]
        :raises KeyError: If value_column or weight_column not found in any DataFrame

        Example:
            Single overall weighted mean::

                fleet_avg_consumption = aggregator.apply_weighted_mean(
                    df,
                    value_column='fuel_consumption',
                    weight_column='engine_power'
                )
                # Returns: 15.6 (single float value)

            Time-grouped weighted means::

                daily_weighted_avg = aggregator.apply_weighted_mean(
                    df,
                    value_column='fuel_consumption',
                    weight_column='engine_power',
                    group_key='timestamp',
                    freq='D'
                )
                # Returns: Series with daily weighted averages

            Multiple DataFrames::

                ship_averages = aggregator.apply_weighted_mean(
                    ship_data_dict,  # Dictionary of ship DataFrames
                    value_column='speed',
                    weight_column='displacement',
                    group_key='datetime',
                    freq='H'
                )
                # Returns: Dict with each ship's hourly weighted speed series

        .. note::
           Weighted mean formula: Σ(value_i × weight_i) / Σ(weight_i)

           This is particularly useful for:

           * Fleet averages weighted by vessel size/power
           * Portfolio metrics weighted by investment amounts
           * Performance metrics weighted by operational hours

        .. seealso::
           :meth:`create_weighted_aggregation` for creating reusable weighted functions
        """
        if isinstance(datafile, dict):
            self.logger.info(f"Processing dictionary with {len(datafile)} DataFrames")
            result = {}
            for key, df in datafile.items():
                self.logger.info(f"Processing DataFrame '{key}'")
                result[key] = self._apply_weighted_mean_single(df, value_column, weight_column, group_key, freq)
            return result
        else:
            return self._apply_weighted_mean_single(datafile, value_column, weight_column, group_key, freq)