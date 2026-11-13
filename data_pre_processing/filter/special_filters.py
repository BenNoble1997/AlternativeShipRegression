"""
Purpose of this class is to store functions for building specific filters.
"""
import pandas as pd
import numpy as np
import logging
from config import LoggerMixin
from typing import Tuple


class SpecialFilter(LoggerMixin):

    def __init__(self,
                 verbose:bool = True,
                 debug:bool = False,
                 ):
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)

        elif self.debug:
            self.logger.setLevel(logging.DEBUG)

        else:
            self.logger.setLevel(logging.WARNING)

    def IQR(self,
            data: pd.DataFrame,
            column: str):
        """
        Function for getting the interquartile range of a given dataframe and column

        :param data: Dataframe containing column for which interquartile range is found.
        :param column: specific column for which the interquartile range should be found.
        """
        data = data.copy()

        # Select the column FIRST, then filter
        column_data = data[column]
        valid_data = column_data[(column_data > 0) & (column_data.notna())]

        if len(valid_data) == 0:
            self.logger.warning(f'No valid (non-zero) data found for column {column}')
            return None

        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1

        return iqr

    def IQR_bounds(self,
                   data: pd.DataFrame,
                   column: str,
                   window: float = 1.5):
        """
        :param data: pandas dataframe containing data being filtered
        :param column: column to find the IQR bounds (to be used for filtering)
        :param window: window size for filtering. smaller window = stricter filter
        """
        data = data.copy()

        # Select the column FIRST, then filter
        column_data = data[column]
        valid_data = column_data[(column_data > 0) & (column_data.notna())]

        if len(valid_data) == 0:
            self.logger.warning(f'No valid (non-zero) data found for column {column}')
            return (None, None)

        iqr = self.IQR(data=data, column=column)

        if iqr is None:
            return (None, None)

        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)

        higher_iqr = (window * iqr) + q3
        lower_iqr = q1 - (window * iqr)

        return (lower_iqr, higher_iqr)

    # def _apply_filters(self, data, column, bounds):
    #     """Helper method to apply filters from column_bounds dict"""
    #     lower,upper = bounds
    #     if column not in data.columns:
    #         self.logger.warning(f'Column {column} not found in dataset')
    #         return data
    #
    #
    #     data = data[(data[column] >= lower) & (data[column] <= upper)]
    #     return data

    def IQR_filter(self,
                   data: pd.DataFrame,
                   column: str,
                   window: float = 1.5):

        """
        Specialised filter for applying interquartile range law.

        Data points are considered if they are greater or below 1,5*IQR+Q3 or 1,5IQR-Q1.

        :param data: pandas dataframe containing data being filtered
        :param column: column name being filtered
        :param window: window size for filtering. smaller window = stricter filter
        """


        df = data.copy()

        original_len = len(df)

        lower_iqr, higher_iqr = self.IQR_bounds(data=df,
                                                column=column,
                                                window=window, )
        self.logger.info(f'Applying IQR law filter to {column}')
        self.logger.info(f'lower bound = {lower_iqr}, higher bound = {higher_iqr}')

        filtered_df = self._apply_filters(data=df,column=column,bounds=(lower_iqr, higher_iqr))

        self.logger.info(f'IQR law filter applied to {column}. {len(filtered_df) - original_len} data points removed ({len(filtered_df)/original_len:.1%} remaining)')

        return filtered_df

    def multiple_IQR_filter(self,
                            data: pd.DataFrame,
                            columns: list[str],
                            window: float = 1.5):
        """
        Method for applying multiple interquartile range law filters to the same dataset

        :param data: pandas dataframe containing data being filtered
        :param columns: list of columns for filter to be applied to
        :param window: window size for filtering. smaller window = stricter filter
        """

        data=data.copy()
        original_len = len(data)



        column_bounds = {}
        for column in columns:
            column_bounds[column]=self.IQR_bounds(data=data,
                                                  column=column,
                                                  window=window)

        # Little piece of code for getting filter sizes (could be useful)

        if self.verbose:
            filter_strength ={}
            for column in columns:
                lower, upper = column_bounds[column]  # You already have these!
                mask = (data[column] >= lower) & (data[column] <= upper)
                n_removed = len(data) - mask.sum()
                filter_strength[column] = -n_removed  # Negative for sorting

            filter_strength = dict(sorted(filter_strength.items(), key=lambda x: x[1]))
        else:
            filter_strength = None

        if filter_strength is not None:
            self.logger.info(f'Multiple IQR filters being applied dataset, in order of filter strength')
            for column in filter_strength.keys():

                len_before = len(data)
                bounds = column_bounds[column]
                lower,upper = bounds
                data = self._apply_filters(data=data,column=column,bounds=bounds)
                len_after = len(data)

                self.logger.info(f'IQR filter applied to {column}. {len_before - len_after} data points removed'
                                 f' ({len_after/original_len:.1%}  remaining)'
                                 f'minimum = {lower}, maximum = {upper}')

        else:
            self.logger.info(f'Multiple IQR filters being applied to dataset in no particular order')
            for column,bounds in column_bounds.items():

                data = self._apply_filters(data=data,column=column,bounds=bounds)


        self.logger.info(f'multiple IQR filters applied. {original_len - len(data)} data points removed ({len(data)/original_len:.1%} remaining)')
        return data

    def maximum_nan_filter(self,
                              data:pd.DataFrame,
                              columns:list[str],
                              no_blanks:int):
        """
        A filter for specifying a number of columns that are allowed to contain blank values

        :param columns: list of columns for filter to be applied to
        :param no_blanks: number of blank columns to be allowed
        """

        data=data.copy()
        original_len = len(data)

        df = data.replace('',pd.NA)

        blank_counts = df[columns].isna().sum(axis=1)

        filtered_df = df[blank_counts < no_blanks]

        new_length = len(filtered_df)

        points_removed = original_len - new_length
        percentage_removed = (points_removed / original_len * 100) if original_len > 0 else 0

        self.logger.info(
            f'maximum_nan_filter applied: '
            f'original length = {original_len}, new length = {new_length}, '
            f'{percentage_removed:.2f}% of points removed ({points_removed})'
        )

        return filtered_df

    def _apply_filters(self, data: pd.DataFrame, column: str, bounds:tuple[float, float]) -> pd.DataFrame:
        """
        Helper method to apply filters from column_bounds dict.

        This filters the data to keep only rows where the column value
        falls within the specified bounds. Rows with NaN values in the
        column are automatically excluded.

        Parameters
        ----------
        data : pd.DataFrame
            The data to filter
        column : str
            Column name to apply filter to
        bounds : Tuple[float, float]
            (lower_bound, upper_bound) for filtering

        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        lower, upper = bounds

        if column not in data.columns:
            self.logger.warning(f'Column {column} not found in dataset')
            return data

        # Create boolean mask (this automatically excludes NaN values)
        mask = (data[column] >= lower) & (data[column] <= upper)

        # Count NaN values that will be excluded
        n_nan = data[column].isna().sum()
        if n_nan > 0:
            self.logger.debug(f'{n_nan} NaN values in {column} will be excluded')

        filtered_data = data[mask].copy()

        return filtered_data

    def get_percentile_bounds(self,
                              data: pd.DataFrame,
                              column: str,
                              percentile: float = 0.95) -> tuple[float, float]|None:
        """
        Calculate symmetric percentile bounds for a column.

        This calculates the bounds such that the central 'percentile' fraction
        of the data is retained. For example, with percentile=0.95, this returns
        the 2.5th and 97.5th percentiles, keeping the middle 95% of data.

        The calculation only considers positive, non-NaN values.

        Parameters
        ----------
        data : pd.DataFrame
            The data containing the column
        column : str
            Column name to calculate bounds for
        percentile : float, default=0.95
            Fraction of data to retain (between 0 and 1)

        Returns
        -------
        Optional[Tuple[float, float]]
            (lower_bound, upper_bound) or None if no valid data

        Examples
        --------
        With percentile=0.95:
            - Lower bound: 2.5th percentile (excludes bottom 2.5%)
            - Upper bound: 97.5th percentile (excludes top 2.5%)
        """
        if column not in data.columns:
            self.logger.warning(f'Column {column} not found in dataset')
            return None

        column_data = data[column]

        # Filter to positive, non-NaN values
        valid_data = column_data[(column_data > 0) & (column_data.notna())]

        if len(valid_data) == 0:
            self.logger.warning(f'No valid data in {column} - cannot calculate bounds')
            return None

        # Calculate symmetric bounds
        # For 95% percentile: keep middle 95%, exclude 2.5% on each tail
        tail_percentage = (1 - percentile) / 2
        lower_percentile = tail_percentage
        upper_percentile = 1 - tail_percentage

        lower_bound = valid_data.quantile(lower_percentile)
        upper_bound = valid_data.quantile(upper_percentile)

        self.logger.info(
            f'{column}: {percentile:.1%} bounds = '
            f'[{lower_bound:.3f}, {upper_bound:.3f}] '
            f'({lower_percentile:.1%} to {upper_percentile:.1%} percentiles)'
        )

        return lower_bound, upper_bound

    def percentile_filter(self,
                          data: pd.DataFrame,
                          column: str,
                          percentile: float = 0.95) -> pd.DataFrame:
        """
        Filter a DataFrame based on percentile bounds for a single column.

        This retains rows where the column value falls within the central
        'percentile' fraction of the distribution. Values outside this range,
        as well as NaN values, are excluded.

        Parameters
        ----------
        data : pd.DataFrame
            The data to filter
        column : str
            Column to apply percentile filter to
        percentile : float, default=0.95
            Fraction of data to retain (between 0 and 1)

        Returns
        -------
        pd.DataFrame
            Filtered data

        Examples
        --------
        >>> filtered_df = obj.percentile_filter(df, 'LBP (M)', percentile=0.90)
        # Keeps rows where LBP is in the middle 90% (excludes extreme 5% on each end)
        """
        initial_len = len(data)

        # Get bounds
        bounds = self.get_percentile_bounds(
            data=data,
            column=column,
            percentile=percentile
        )

        # If no valid bounds, return original data
        if bounds is None:
            self.logger.warning(f'Cannot filter {column} - returning original data')
            return data.copy()

        # Apply filter
        result = self._apply_filters(data=data, column=column, bounds=bounds)

        final_len = len(result)
        n_removed = initial_len - final_len
        pct_removed = (n_removed / initial_len * 100) if initial_len > 0 else 0
        pct_remaining = (final_len / initial_len * 100) if initial_len > 0 else 0

        self.logger.info(
            f'Percentile filter applied to {column}: '
            f'{n_removed} rows removed ({pct_removed:.1f}%), '
            f'{final_len} remaining ({pct_remaining:.1f}%)'
        )

        return result

    def multiple_percentile_filter(self,
                                   data:pd.DataFrame,
                                   columns:list[str]|str,
                                   percentile:float = 0.95):

        """
        a function for filtering a number of columns by a specified percentile

        :param columns: list of columns for filter to be applied to
        :param percentile: percentile for filtering
        """

        if isinstance(columns, str):
            columns = [columns]

        result = data.copy()
        initial_len = len(data)


        self.logger.info(f'Calculating percentile bounds ({percentile:.1%}) for {len(columns)} columns')

        # STEP 1: Calculate all bounds based on the ORIGINAL data
        column_bounds = {}

        for column in columns:
            bounds = self.get_percentile_bounds(data=data,  # Use original data
                                                column=column,
                                                percentile=percentile)
            if bounds is not None:
                column_bounds[column] = bounds
            else:
                self.logger.warning(f'Skipping {column} - no valid bounds calculated')

        # STEP 2: Apply all filters sequentially
        self.logger.info(f'Applying filters to {len(column_bounds)} columns')

        for column, bounds in column_bounds.items():
            len_before = len(result)

            result = self._apply_filters(data=result, column=column, bounds=bounds)

            len_after = len(result)
            rows_removed_this_step = len_before - len_after
            percentage_removed_this_step = (rows_removed_this_step / len_before * 100) if len_before > 0 else 0

            self.logger.info(f'Percentile filter applied to {column}.')
            self.logger.info(
                f'{percentage_removed_this_step:.2f}% of rows removed by this filter '
                f'({rows_removed_this_step} rows), {len_after} remaining'
            )

        # Final summary
        final_len = len(result)
        total_removed = initial_len - final_len
        total_percentage_removed = (total_removed / initial_len * 100) if initial_len > 0 else 0

        self.logger.info(f'Multiple percentile filter complete:')
        self.logger.info(
            f'Total: {total_removed} rows removed ({total_percentage_removed:.2f}%), '
            f'{final_len} rows remaining ({100 - total_percentage_removed:.2f}%)'
        )

        return result

