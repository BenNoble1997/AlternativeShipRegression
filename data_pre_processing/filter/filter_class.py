"""
This module stores the DataFilter Class

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0

"""

import pandas as pd
import os
import sys
import logging
from data_pre_processing.filter.common_filter_function import common_filter_function
from data_pre_processing.filter.differential_filters import DifferentialFilters
from data_pre_processing.filter.special_filters import SpecialFilter

from typing import List, Dict, Optional, Union, Any

os.chdir(r"/")


sys.path.append(r"/")

from config.logger_config import LoggerMixin
from data_pre_processing.data_processing_enum_classes import ComparisonOperator


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

        self.differential = DifferentialFilters(verbose=self.verbose,
                                                debug=self.debug)
        self.special = SpecialFilter(verbose=self.verbose,
                                     debug=self.debug)

    def common_filter_function(self,
                               datafile: pd.DataFrame,
                               column_name: str|List[str],
                               value: float,
                               operator: ComparisonOperator|str = ComparisonOperator.EQUAL,
                               require_all: bool = False,
                               negate: bool = False,) -> pd.DataFrame:
        """
        Filter DataFrame rows based on comparison with a specified value.

        This method supports complex filtering operations with multiple options:

        * Single or multiple column filtering
        * Custom comparison operators
        * AND/OR logic for multiple columns
        * Optional negation of the final condition

        :param datafile: The DataFrame to filter
        :type datafile: pd.DataFrame
        :param column_name: Column(s) to apply the filter to
        :type column_name: Union[str, List[str]]
        :param value: The value to compare against
        :type value: float
        :param operator: The comparison operator to use (ComparisonOperator enum or string)
        :type operator: Union[ComparisonOperator, str]
        :param require_all: If True, all columns must meet the condition (AND logic).
                           If False, any column meeting the condition is sufficient (OR logic).
                           Defaults to False.
        :type require_all: bool
        :param negate: If True, reverses the final condition (keeps rows that don't match).
                      Defaults to False.
        :type negate: bool
        :returns: Filtered DataFrame
        :rtype: pd.DataFrame
        :raises KeyError: If specified column(s) are not found in the DataFrame
        :raises ValueError: If an invalid operator string is provided

        Example:
            Keep rows where at least one SFOC is positive::

                df = self.common_filter_function(df, ['ME_SFOC', 'DG_SFOC'], 0,
                                               ComparisonOperator.GREATER_THAN,
                                               require_all=False)

            Keep rows where both SFOCs are positive::

                df = self.common_filter_function(df, ['ME_SFOC', 'DG_SFOC'], 0,
                                               ComparisonOperator.GREATER_THAN,
                                               require_all=True)

            Remove rows where both SFOCs are negative or zero::

                df = self.common_filter_function(df, ['ME_SFOC', 'DG_SFOC'], 0,
                                               ComparisonOperator.LESS_EQUAL,
                                               require_all=True,
                                               negate=True)
        """

        return common_filter_function(datafile=datafile,
                                      column_name=column_name,
                                      value=value,
                                      operator=operator,
                                      require_all=require_all,
                                      negate=negate,
                                      logger=self.logger
                                      )


    def filter_by_value(self,
                        datafile: pd.DataFrame|dict[str, pd.DataFrame],
                        value: Any,
                        column_name: str|List[str],
                        operator: ComparisonOperator = ComparisonOperator.NOT_EQUAL,
                        require_all: bool = False,
                        negate: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Filter DataFrame(s) by removing rows based on column value comparisons.

        This function acts as a wrapper around :meth:`common_filter_function` and can handle
        both single DataFrames and dictionaries of DataFrames. It applies the same filtering
        logic to each DataFrame in the collection.

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param value: The value to compare against
        :type value: Any
        :param column_name: Column(s) to apply the filter to
        :type column_name: Union[str, List[str]]
        :param operator: The comparison operator to use. Defaults to NOT_EQUAL.
        :type operator: ComparisonOperator
        :param require_all: If True, all columns must meet the condition (AND logic).
                           If False, any column meeting the condition is sufficient (OR logic).
                           Defaults to False.
        :type require_all: bool
        :param negate: If True, reverses the final condition (keeps rows that don't match).
                      Defaults to False.
        :type negate: bool
        :returns: Filtered DataFrame or dictionary of filtered DataFrames (same type as input)
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :raises Exception: Re-raises any exceptions from the underlying filter operation with context

        Example:
            Filter single DataFrame to remove rows where value equals 0::

                filtered_df = self.filter_by_value(df, 0, 'column_name',
                                                 ComparisonOperator.EQUAL)

            Filter multiple DataFrames to remove negative values from any specified column::

                filtered_dfs = self.filter_by_value(df_dict, 0, ['col1', 'col2'],
                                                  ComparisonOperator.LESS_THAN)

        .. seealso::
           :meth:`common_filter_function` for detailed filtering logic and additional examples.
        """
        columns = column_name if isinstance(column_name, list) else [column_name]

        # Create logging message
        logic_desc = "all" if require_all else "any"
        op_desc = operator.description if hasattr(operator, 'description') else str(operator)
        negate_desc = "NOT " if negate else ""

        self.logger.info(
            f"Filtering {', '.join(columns)} where {negate_desc}{logic_desc} columns "
            f"{op_desc} {value}"
        )

        if isinstance(datafile, dict):  # Fixed: was Dict instead of dict
            self.logger.info(f'Filtering {len(datafile)} datafiles')
            result = {}
            for key, df in datafile.items():
                try:
                    self.logger.info(f"Applying filter to '{key}'")
                    result[key] = self.common_filter_function(
                        datafile=df,
                        column_name=column_name,
                        value=value,
                        operator=operator,
                        require_all=require_all,
                        negate=negate
                    )
                except Exception as e:
                    self.logger.error(f"Error filtering datafile '{key}': {str(e)}")
                    raise
            return result
        else:
            return self.common_filter_function(
                datafile=datafile,
                column_name=column_name,
                value=value,
                operator=operator,
                require_all=require_all,
                negate=negate
            )

    def filter_out_negative(self,
                            datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            column_name: Union[str, List[str]],
                            require_all: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Filter out negative values from specified columns.

        This function keeps rows where values are greater than or equal to 0.
        When multiple columns are specified, you can control the filtering logic:

        * ``require_all=False`` (default): Remove rows where ANY column is negative
        * ``require_all=True``: Remove rows only where ALL columns are negative

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param column_name: Column(s) to check for negative values
        :type column_name: Union[str, List[str]]
        :param require_all: Controls the logic for multiple columns.
                           If False (default), remove rows where ANY column is negative.
                           If True, remove rows only where ALL columns are negative.
        :type require_all: bool
        :returns: DataFrame or dictionary of DataFrames with negative values filtered
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :raises KeyError: If specified column(s) are not found in the DataFrame
        :raises Exception: Re-raises any exceptions from the underlying filter operation

        Example:
            Remove rows where either SFOC is negative::

                df = self.filter_out_negative(df, ['ME_SFOC', 'DG_SFOC'], require_all=False)

            Remove rows only where both SFOC values are negative::

                df = self.filter_out_negative(df, ['ME_SFOC', 'DG_SFOC'], require_all=True)

            Filter multiple DataFrames::

                filtered_dfs = self.filter_out_negative(df_dict, 'column_name')

        .. note::
           This is a convenience wrapper around :meth:`common_filter_function` with
           ``operator=ComparisonOperator.GREATER_EQUAL`` and ``value=0``.

        .. seealso::
           :meth:`common_filter_function` for more flexible filtering options.
        """
        columns = column_name if isinstance(column_name, list) else [column_name]
        self.logger.info(
            f"Filtering {', '.join(columns)} "
            f"to remove {'rows where all columns are negative' if require_all else 'rows where any column is negative'}"
        )

        if isinstance(datafile, dict):
            self.logger.info(f'Filtering {len(datafile)} datafiles')
            result = {}
            for key, df in datafile.items():
                try:
                    self.logger.info(f"Applying negative filter to '{key}'")
                    result[key] = self.common_filter_function(
                        datafile=df,
                        column_name=column_name,
                        value=0,
                        operator=ComparisonOperator.GREATER_EQUAL,
                        require_all=require_all,
                        negate=False
                    )
                except Exception as e:
                    self.logger.error(f"Error filtering negative values in '{key}': {str(e)}")
                    raise
            return result
        else:
            return self.common_filter_function(
                datafile=datafile,
                column_name=column_name,
                value=0,
                operator=ComparisonOperator.GREATER_EQUAL,
                require_all=require_all,
                negate=False
            )

    def filter_out_zero(self,
                        datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                        column_name: Union[str, List[str]],
                        filter_zeros: bool = True,
                        filter_nulls: bool = False,
                        require_all: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Filter out zero values and/or null values from specified columns.

        This function provides flexible filtering of zero values (0) and/or null values
        (NaN, None) across one or more DataFrame columns. The filtering logic can be
        controlled to determine whether rows are removed based on ANY column meeting
        the criteria or only when ALL columns meet the criteria. Can work on a single
        DataFrame or a dictionary of DataFrames.


        :param datafile:A DataFrame or dictionary of DataFrames to filter
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param column_name:Column(s) to check for zero/null values
        :type column_name: Union[str, List[str]]
        :param filter_zeros:If True, filter out zero values (0)
        :type filter_zeros: bool
        :param filter_nulls:If True, filter out null values (NaN, None)
        :type filter_nulls: bool
        :param require_all:Controls the filtering logic for multiple columns:
                        - False (default): Remove rows where ANY specified column has a
                          zero/null value (OR logic)
                        - True: Remove rows where ALL specified columns have zero/null
                          values (AND logic). This keeps more data.
        :type require_all: bool

        :returns: Filtered DataFrame or dictionary of filtered DataFrames
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

        Examples:
            # Remove rows where shaft power is zero
            df = filter_out_null_and_zero(df, 'SHAFT_PWR')

            # Filter multiple dataframes in a dictionary
            filtered_dict = filter_out_null_and_zero(
                {'ship1': df1, 'ship2': df2},
                'POWER',
                filter_zeros=True,
                filter_nulls=True
            )
        """
        # Convert single column to list for consistent processing
        columns = [column_name] if isinstance(column_name, str) else column_name

        # Log the filtering operation being performed
        filter_types = []
        if filter_zeros:
            filter_types.append("zero values")
        if filter_nulls:
            filter_types.append("null values")

        if not filter_types:
            self.logger.info("No filtering requested (both filter_zeros and filter_nulls are False)")
            return datafile  # Return original data unchanged

        filter_logic = "ALL" if require_all else "ANY"
        column_str = ", ".join(columns)

        self.logger.info(
            f"Filtering rows where {filter_logic} of these columns have {' or '.join(filter_types)}: {column_str}"
        )

        # Check if we're dealing with a dictionary of dataframes
        if isinstance(datafile, Dict):
            self.logger.info(f"Filtering multiple datafiles ({len(datafile)} files)")
            result_dict = {}

            # Process each dataframe in the dictionary
            for key, df in datafile.items():
                initial_count = len(df)
                filtered_df = self._filter_single_dataframe(
                    df, columns, filter_zeros, filter_nulls, require_all
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
            return self._filter_single_dataframe(
                datafile, columns, filter_zeros, filter_nulls, require_all
            )

    def _filter_single_dataframe(self,
                                 df: pd.DataFrame,
                                 columns: List[str],
                                 filter_zeros: bool,
                                 filter_nulls: bool,
                                 require_all: bool) -> pd.DataFrame:
        """
        Helper method to filter a single dataframe.
        Internal use only - called by filter_out_zero.
        """
        filtered_df = df.copy()
        initial_count = len(filtered_df)

        try:
            # Step 1: Filter zeros if requested
            if filter_zeros:
                filtered_df = self.common_filter_function(
                    datafile=filtered_df,
                    column_name=columns,
                    value=0,
                    operator=ComparisonOperator.EQUAL,
                    require_all=require_all,
                    negate=True  # Keep values that are NOT equal to zero
                )

            # Step 2: Filter nulls if requested
            if filter_nulls:
                # For each column, create a temporary column with True/False for null check
                temp_columns = []

                for i, col in enumerate(columns):
                    temp_col = f"__temp_notnull_{i}"
                    filtered_df[temp_col] = filtered_df[col].notna()
                    temp_columns.append(temp_col)

                # Use common_filter_function to filter based on these temporary columns
                filtered_df = self.common_filter_function(
                    datafile=filtered_df,
                    column_name=temp_columns,
                    value=True,
                    operator=ComparisonOperator.EQUAL,
                    require_all=require_all,
                    negate=False
                )

                # Remove temporary columns
                filtered_df = filtered_df.drop(columns=temp_columns)

            # Final log entry with overall results
            total_removed = initial_count - len(filtered_df)
            self.logger.info(
                f"Total filtering result: {len(filtered_df):,} rows remaining from {initial_count:,} "
                f"({total_removed:,} removed, {total_removed / initial_count:.1%} of original data)"
            )

            return filtered_df

        except Exception as e:
            self.logger.error(f"Filter operation failed: {str(e)}")
            raise

    def build_filter(self,
                     column: str|List[str],
                     value: Any,
                     operator: str|ComparisonOperator = ComparisonOperator.EQUAL,
                     require_all: bool = False,
                     negate: bool = False) -> Dict[str, Any]:
        """
        Build a filter dictionary for use with apply_multiple_filters.

        This helper method makes it easy to create properly formatted filter
        dictionaries without having to remember the exact structure required.

        :param column: The column(s) to filter. Can be a single column name or a list of columns.
        :param value:The value to compare against.
        :param operator:The comparison operator to use. Can be a ComparisonOperator enum
                     or a string symbol like '>', '<=', '==', etc.
        :param require_all:If True and multiple columns are specified, all columns must
                         meet the condition (AND logic). If False, any column meeting
                         the condition is sufficient (OR logic).
        :param negate:If True, reverses the condition (keeps rows that don't match).

        :return:A properly formatted filter dictionary that can be used with apply_multiple_filters.

        Examples:
            # Create a filter to keep rows where 'speed' > 20
            speed_filter = filter_builder.build_filter('speed', 20, '>')

            # Create a filter that requires both power and thrust to be <= 1000
            power_filter = filter_builder.build_filter(
                ['power', 'thrust'], 1000, ComparisonOperator.LESS_EQUAL, require_all=True
            )

            # Apply multiple filters
            filtered_df = filter_builder.apply_multiple_filters(
                df, [speed_filter, power_filter]
            )

            # Create a filter that excludes rows where ALL columns are null
            not_all_null_filter = filter_builder.build_filter(
                ['col1', 'col2', 'col3'], None, ComparisonOperator.EQUAL,
                require_all=True, negate=True
            )
        """
        # Convert string operator to enum if necessary
        if isinstance(operator, str):
            try:
                operator = ComparisonOperator.from_symbol(operator)
            except ValueError as e:
                self.logger.error(f"Invalid operator provided: {operator}")
                raise

        # Create and return the filter dictionary
        filter_dict = {
            'column': column,
            'operator': operator,
            'value': value,
            'require_all': require_all,
            'negate': negate
        }

        return filter_dict

    def apply_multiple_filters(self,
                               datafile:pd.DataFrame|dict[str, pd.DataFrame],
                               filters: List[dict[str, Any]]|dict[str, Any]) ->pd.DataFrame|dict[str, pd.DataFrame]:
        """
        Apply multiple filters sequentially to a DataFrame or dictionary of DataFrames.

        This method processes one or more filters in sequence. Each filter is defined
        by a dictionary specifying the column(s) to filter, the comparison operator,
        the value to compare against, and optional parameters for complex filtering.

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :param filters: Either a single filter dictionary or a list of filter dictionaries.
                    Each dictionary should have:
                    - 'column': str or List[str] - Column(s) to filter
                    - 'operator': ComparisonOperator or str - The comparison operator
                    - 'value': Any - The value to compare against
                    - 'require_all': bool (optional) - Whether all columns must meet the condition
                    - 'negate': bool (optional) - Whether to reverse the condition
        :return: Filtered DataFrame or dictionary of filtered DataFrames

        Examples:
            # Apply to a single DataFrame
            filtered_df = filter_builder.apply_multiple_filters(df, filters)

            # Apply to multiple DataFrames in a dictionary
            ship_data = {'ship1': df1, 'ship2': df2}
            filtered_ships = filter_builder.apply_multiple_filters(ship_data, filters)
        """
        # Convert single filter to list for consistent processing
        if isinstance(filters, dict) and 'column' in filters:
            filters = [filters]

        self.logger.info(f"Beginning application of {len(filters)} filters")

        # Check if we're dealing with a dictionary of dataframes
        if isinstance(datafile, Dict):
            self.logger.info(f"Filtering multiple datafiles ({len(datafile)} files)")
            result_dict = {}

            # Process each dataframe in the dictionary
            for key, df in datafile.items():
                self.logger.info(f"Applying filters to dataframe '{key}'")
                initial_count = len(df)

                # Apply filters to this specific dataframe
                filtered_df = self._apply_filters_to_single_df(df, filters)

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
            return self._apply_filters_to_single_df(datafile, filters)

    def _apply_filters_to_single_df(self,
                                    datafile: pd.DataFrame,
                                    filters: List[dict[str, Any]]) -> pd.DataFrame:
        """
        Helper method to apply multiple filters to a single DataFrame.
        Internal use only - called by apply_multiple_filters.
        """
        filtered_df = datafile.copy()
        initial_count = len(filtered_df)

        for i, filter_dict in enumerate(filters, 1):
            try:
                column_name = filter_dict['column']
                operator = filter_dict['operator']
                value = filter_dict['value']

                # Get optional parameters with defaults
                require_all = filter_dict.get('require_all', False)
                negate = filter_dict.get('negate', False)

                # Convert string operator to enum if necessary
                if isinstance(operator, str):
                    try:
                        operator = ComparisonOperator.from_symbol(operator)
                    except ValueError as e:
                        self.logger.error(f"Invalid operator in filter {i}: {operator}")
                        raise

                # Describe the filter being applied
                column_desc = column_name if isinstance(column_name, str) else ', '.join(column_name)
                logic_desc = "ALL" if require_all and not isinstance(column_name, str) else "ANY"
                negation_desc = "NOT " if negate else ""

                self.logger.info(
                    f"Applying filter {i}: {negation_desc}{logic_desc} of [{column_desc}] "
                    f"{operator.description} {value}"
                )

                filtered_df = self.common_filter_function(
                    filtered_df,
                    column_name,
                    value,
                    operator,
                    require_all=require_all,
                    negate=negate
                )

            except KeyError as e:
                self.logger.error(f"Missing required key in filter {i}: {str(e)}")
                raise KeyError(
                    f"Filter dictionary must contain 'column', 'operator', and 'value'. "
                    f"Missing: {str(e)}"
                )

        # Final log entry with overall results
        total_removed = initial_count - len(filtered_df)
        self.logger.info(
            f"Filter sequence completed. Retained {len(filtered_df)} of {len(datafile)} "
            f"original rows ({total_removed:,} removed, {total_removed / initial_count:.1%})"
        )
        return filtered_df




    def _filter_out_blanks_single(self,
                          datafile: pd.DataFrame,
                          column_name: str|List[str]) -> pd.DataFrame:
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

    def filter_out_blanks(self,
                          datafile: pd.DataFrame,
                          column_name: str|List[str]) -> pd.DataFrame:
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

        if isinstance(filtered_data, dict):
            for key,df in filtered_data.items():
                df = self._filter_out_blanks_single(datafile = df,
                                                    column_name = column_name)
                filtered_data[key] = df
        else:

            filtered_data = self._filter_out_blanks_single(datafile = filtered_data,
                                                           column_name = column_name)
        return filtered_data

    def filter_by_reference_data(self,
                                 datafile:pd.DataFrame| dict[str, pd.DataFrame],
                                 column_name: str,
                                 reference_values: List|pd.Series|pd.DataFrame,
                                 reference_column: str|None = None,
                                 keep_matching: bool = False) -> pd.DataFrame|dict[str, pd.DataFrame]:
        """
        Filter DataFrame by removing (or keeping) rows based on values from a reference dataset.

        This method is more efficient than applying multiple NOT_CONTAINS filters sequentially,
        especially when dealing with large exclusion lists.

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :type datafile: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param column_name: Column in the datafile to check
        :type column_name: str
        :param reference_values: Reference data containing values to filter by.
                                Can be a list, pandas Series, or DataFrame.
        :type reference_values: Union[List, pd.Series, pd.DataFrame]
        :param reference_column: If reference_values is a DataFrame, specify which column to use.
                                Required when reference_values is a DataFrame.
        :type reference_column: Optional[str]
        :param keep_matching: If False (default), remove rows where values match the reference.
                             If True, keep only rows where values match the reference.
        :type keep_matching: bool
        :returns: Filtered DataFrame or dictionary of filtered DataFrames
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

        Example:
            # Remove rows where fuel type is in the exclusion list
            excluded_fuels = ['HFO', 'LSFO', 'MDO']
            filtered_df = self.filter_by_reference_data(df, 'fuel_type', excluded_fuels)

            # Keep only rows where engine model exists in approved list
            approved_df = pd.DataFrame({'model': ['Model_A', 'Model_B']})
            filtered_df = self.filter_by_reference_data(
                df, 'engine_model', approved_df,
                reference_column='model',
                keep_matching=True
            )

            # Filter multiple dataframes
            filtered_dict = self.filter_by_reference_data(
                {'ship1': df1, 'ship2': df2},
                'fuel_type',
                excluded_fuels
            )
        """
        # Convert reference data to a set for efficient lookup
        if isinstance(reference_values, pd.DataFrame):
            if reference_column is None:
                raise ValueError("reference_column must be specified when reference_values is a DataFrame")
            reference_set = set(reference_values[reference_column].dropna().unique())
        elif isinstance(reference_values, pd.Series):
            reference_set = set(reference_values.dropna().unique())
        elif isinstance(reference_values, list):
            reference_set = set(reference_values)
        else:
            raise TypeError("reference_values must be a list, pandas Series, or pandas DataFrame")

        action = "keeping" if keep_matching else "removing"
        self.logger.info(
            f"Filtering column '{column_name}' by {action} rows with values in reference set "
            f"({len(reference_set)} unique values)"
        )

        if isinstance(datafile, dict):
            self.logger.info(f'Filtering {len(datafile)} datafiles')
            result = {}
            for key, df in datafile.items():
                try:
                    self.logger.info(f"Applying reference filter to '{key}'")
                    result[key] = self._filter_single_by_reference(
                        df, column_name, reference_set, keep_matching
                    )
                except Exception as e:
                    self.logger.error(f"Error filtering datafile '{key}': {str(e)}")
                    raise
            return result
        else:
            return self._filter_single_by_reference(
                datafile, column_name, reference_set, keep_matching
            )

    def _filter_single_by_reference(self,
                                    df: pd.DataFrame,
                                    column_name: str,
                                    reference_set: set,
                                    keep_matching: bool) -> pd.DataFrame:
        """
        Helper method to filter a single dataframe by reference values.
        Internal use only - called by filter_by_reference_data.
        """
        initial_count = len(df)

        if column_name not in df.columns:
            self.logger.error(f"Column not found: {column_name}")
            raise KeyError(f"Column '{column_name}' not found in DataFrame")

        # Create boolean mask based on whether values are in reference set
        if keep_matching:
            mask = df[column_name].isin(reference_set)
        else:
            mask = ~df[column_name].isin(reference_set)

        filtered_df = df[mask].copy()

        rows_removed = initial_count - len(filtered_df)
        action = "kept matching" if keep_matching else "removed matching"

        self.logger.info(
            f"Filter: {action} values in '{column_name}' "
            f"(removed {rows_removed:,} rows, "
            f"{rows_removed / initial_count:.1%} of data)"
        )
        self.logger.info(f"Filter applied. {len(filtered_df)} points remaining")

        return filtered_df

    def build_differential_filter(self,
                                  column: str|list[str],
                                  max_diff: float | None = None,
                                  min_diff: float | None = None) -> dict[str, Any]:

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

        return self.differential.build_differential_filter(column=column, max_diff=max_diff, min_diff=min_diff)

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

        return self.differential.filter_by_differential(datafile=datafile, column_name=column_name, max_diff=max_diff, min_diff=min_diff)


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

        return self.differential.apply_multiple_differential_filters(datafile=datafile, filters=filters)

    def IQR_filter(self,
                   data: pd.DataFrame,
                   column_name: str,
                   window:float=1.5):

        """
        Specialised filter for applying interquartile range law.

        Data points are considered if they are greater or below 1,5*IQR+Q3 or 1,5IQR-Q1
        """

        return self.special.IQR_filter(data, column_name, window)

    def multiple_IQR_filter(self,
                            data: pd.DataFrame,
                            columns: list[str],
                            window:float=1.5):
        """
        Method for applying multiple interquartile range law filters to the same dataset
        """

        return self.special.multiple_IQR_filter(data, columns, window)

    def maximum_nan_filter(self,
                              data:pd.DataFrame,
                              columns:list[str],
                              no_blanks:int):
        """
        A filter for specifying a number of columns that are allowed to contain blank values

        :param columns: list of columns for filter to be applied to
        :param no_blanks: number of blank columns to be allowed
        """

        return self.special.maximum_nan_filter(data, columns, no_blanks)

    def percentile_filter(self,
                          data: pd.DataFrame,
                          columns: list[str]|str,
                          percentile:float=0.95):

        """
        Apply percentile filter to the dataset

        :param data: datafile to apply filter to
        :param columns: list of columns for filter to be applied to
        :param percentile: percentile for filter to be applied

        """

        return self.special.multiple_percentile_filter(data=data, columns=columns, percentile=percentile)