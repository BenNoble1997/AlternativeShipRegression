"""
Module for storing data manipulation classes

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0

"""
import pandas as pd
import os
import sys
import logging
import numpy as np

from typing import List, Tuple
from itertools import combinations as iter_combinations
from config.fuel_classes.Fuel_class import FuelType





from config.loggermixin.logger_config import LoggerMixin
from data_processing.data_processing_enum_classes import ColumnOperation

SEAWATER_DENSITY = 1.025

class DataManipulator(LoggerMixin):
    """
    Handles various data manipulation operations including column calculations
    and duplicate handling.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        super().__init__()
        self.verbose = verbose
        self.debug = debug

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        elif self.verbose:
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

    def _drop_duplicates_single(self,
                        df: pd.DataFrame,
                        subset: str|List[str],
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

    def drop_duplicates(self,
                        df: dict[str,pd.DataFrame]|str,
                        subset: str| List[str],
                        keep: str = 'first') -> pd.DataFrame:
        """
        Removes duplicate rows and resets the index.

        :param df: DataFrame containing the data.
        :param subset: Column(s) to consider for duplicate detection.
        :param keep: Which duplicate to keep ('first', 'last', or False to remove all).
        :return: DataFrame with duplicates removed.
        """

        dataset = df.copy()

        if isinstance(dataset, dict):
            for key,df in dataset.items():
                dataset[key] = self._drop_duplicates_single(df, subset=subset, keep=keep)
        else:
            dataset = self._drop_duplicates_single(df, subset=subset, keep=keep)

        return dataset

    def merge_data(self,
                   data: dict[str,pd.DataFrame]|List[pd.DataFrame],
                   id_column: str = 'ID',
                   how_to_merge: str = 'outer')-> pd.DataFrame:

        """
        Function for merging dataframes around a specific column.

        :param data: dictionary or list containing the set of dataframes to be merged
        :param id_column: Name of the ID column
        :param how_to_merge: Type of merge ('left', 'right', 'outer', 'inner')

        """

        if isinstance(data, list):
            self.logger.debug(f"Converting list of dataframes to a dictionary")
            idx=1
            dataset = {}
            for df in data:
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Input data must be a dataframe, not {type(df)}")
                dataset[f'df_{idx}'] = df
                idx+=1

        else:
            dataset = data

        if len(dataset) < 2:
            raise ValueError("Need at least 2 DataFrames to merge")

        else:
            self.logger.info(f"Merging {len(dataset)} dataframes")

        # Validate ID column exists in all DataFrames
        for name, df in dataset.items():
            if id_column not in df.columns:
                raise ValueError(f"Column '{id_column}' not found in {name}")

        # Compute intersection of all columns
        common_columns = set(next(iter(dataset.values())).columns)
        for df in dataset.values():
            common_columns &= set(df.columns)
        common_columns = list(common_columns)

        # Merge all dataframes successively
        merged_data = dataset[next(iter(dataset))]  # start with first df
        for df in list(dataset.values())[1:]:  # merge rest
            merged_data = merged_data.merge(df, on=common_columns, how=how_to_merge)

        self.logger.info(f'merged data into dataframe with {len(merged_data)} rows and {len(merged_data.columns)} columns')

        return merged_data

    def calculate_cb_column(self,
            df: pd.DataFrame,
            col_L: str = 'LOA',
            col_B: str = 'B',
            col_T: str = 'DRAFT',
            col_DWT: str = 'DWT',
            col_LDT: str = 'LDT',
            col_speed: str = 'service_speed',
            shiptype: str = 'BULK_CARRIER',
            rho_SW: float = SEAWATER_DENSITY,
    ) -> Tuple[pd.DataFrame, dict[str, any]]:
        """Calculate block coefficient (C_B) for ship database using two methods.

        This function calculates the block coefficient using two approaches and combines
        them into a single column. Method 1 (preferred) uses displacement-based calculation
        from DWT/LDT, while Method 2 uses the Katsoulis regression formula based on service
        speed and ship dimensions.

        **Calculation Methods:**

        **Method 1 (Preferred - DWT/LDT based):**
            Uses volumetric displacement calculation:

            .. math::
                C_B = \\frac{(DWT + LDT) / \\rho_{SW}}{L \\times B \\times T}

            where:
                - DWT: Deadweight tonnage (tonnes)
                - LDT: Lightweight tonnage (tonnes)
                - ρ_SW: Seawater density (t/m³)
                - L, B, T: Length, breadth, draught (m)

        **Method 2 (Fallback - Katsoulis formula):**
            Uses regression formula based on service speed:

            .. math::
                C_B = \\left(\\frac{V_s}{a_1 \\times L^{a_3} \\times B^{a_4} \\times T^{a_5}}\\right)^{1/a_2}

            where coefficients vary by ship type:
                - Tanker: [1.633, -0.3280, 0.7256, -0.5122, -0.0214]
                - Bulk Carrier: [3.7682, -0.3236, 0.3566, -0.0455, -0.1866]
                - Container Ship: [0.9804, -0.9809, 0.7416, -0.3742, -0.025]
                - Default: [0.7261, -1.63, 0.6846, -0.5007, 0.2805]

        :param df: Ship database DataFrame containing ship dimensions and characteristics
        :type df: pd.DataFrame
        :param col_L: Column name for ship length overall (LOA) in meters
        :type col_L: str, optional
        :param col_B: Column name for ship breadth (beam) in meters
        :type col_B: str, optional
        :param col_T: Column name for ship draught (draft) in meters
        :type col_T: str, optional
        :param col_DWT: Column name for deadweight tonnage in tonnes
        :type col_DWT: str, optional
        :param col_LDT: Column name for lightweight tonnage in tonnes
        :type col_LDT: str, optional
        :param col_speed: Column name for service speed in kn
        :type col_speed: str, optional
        :param col_shiptype: Column name for ship type classification
        :type col_shiptype: str, optional
        :param rho_SW: Seawater density in tonnes per cubic meter
        :type rho_SW: float, optional
        :param logger: Logger instance for output messages. If None, creates default logger
        :type logger: logging.Logger, optional

        :return: Tuple containing:
            - **df** (*pd.DataFrame*) -- Modified DataFrame with three new columns:
                - 'C_B_method1': Block coefficient from DWT/LDT method
                - 'C_B_method2': Block coefficient from Katsoulis formula
                - 'C_B': Final block coefficient (defaults to Method 1 when both available)
            - **stats** (*dict*) -- Dictionary with calculation statistics containing:
                - 'n_total': Total number of rows
                - 'n_calculated': Rows with C_B successfully calculated
                - 'n_missing': Rows without C_B value
                - 'n_method1_only': Rows using only Method 1
                - 'n_method2_only': Rows using only Method 2
                - 'n_both_methods': Rows where both methods were calculated
                - 'avg_difference': Average absolute difference between methods
                - 'median_difference': Median absolute difference between methods
                - 'max_difference': Maximum absolute difference between methods
        :rtype: tuple[pd.DataFrame, dict]

        .. note::
            - Method 1 is always preferred when both methods produce valid results
            - Missing LDT values are treated as 0 for Method 1 calculation
            - Method 2 includes sanity checks (0.3 < C_B < 1.0) to filter unrealistic values
            - The function creates a copy of the input DataFrame

        .. warning::
            Rows with insufficient data for both methods will have NaN in the final C_B column

        :raises: None - Function handles missing data gracefully with NaN values

        **Example:**

        .. code-block:: python

            import pandas as pd
            import logging

            # Prepare your ship database
            df = pd.read_csv('ship_database.csv')

            # Calculate block coefficients with custom column names
            df_result, stats = calculate_cb_column(
                df,
                col_L='Length_Overall',
                col_B='Beam',
                col_T='Draft',
                col_DWT='DeadweightTonnage',
                col_LDT='LightweightTonnage',
                col_speed='ServiceSpeed',
                col_shiptype='VesselType',
                rho_SW=1.025
            )

            # Access results
            print(f"Calculated C_B for {stats['n_calculated']} ships")
            print(f"Average C_B: {df_result['C_B'].mean():.3f}")

            # View rows where both methods were used
            both_methods = df_result['C_B_method1'].notna() & df_result['C_B_method2'].notna()
            comparison = df_result.loc[both_methods, ['C_B_method1', 'C_B_method2', 'C_B']]

        **References:**

        Katsoulis, G. (2025). "Ship Design Speed Prediction using Regression Analysis"
        DOI: 10.6911/WSRJ.202507_11(7).0007

        .. seealso::
            For more information on block coefficient and ship design parameters, refer to
            standard naval architecture references such as "Principles of Naval Architecture"
        """

        df = df.copy()

        # Initialize result columns
        df['C_B_method1'] = np.nan  # DWT/LDT method
        df['C_B_method2'] = np.nan  # Katsoulis speed method
        df['C_B'] = np.nan

        # Method 1: DWT/LDT calculation
        self.logger.info("Calculating C_B using Method 1 (DWT/LDT)...")
        mask_method1 = (
                df[col_L].notna() & (df[col_L] > 0) &
                df[col_B].notna() & (df[col_B] > 0) &
                df[col_T].notna() & (df[col_T] > 0) &
                df[col_DWT].notna() & (df[col_DWT] > 0)
        )

        # Calculate Method 1
        ldt_values = df.loc[mask_method1, col_LDT].fillna(0)
        dwt_values = df.loc[mask_method1, col_DWT]

        DWT_v = (dwt_values + ldt_values) / rho_SW
        df.loc[mask_method1, 'C_B_method1'] = DWT_v / (
                df.loc[mask_method1, col_L] *
                df.loc[mask_method1, col_B] *
                df.loc[mask_method1, col_T]
        )

        # Method 2: Katsoulis formula with service speed
        self.logger.info("Calculating C_B using Method 2 (Katsoulis formula)...")

        # Define coefficients for different ship types
        coeff_dict = {
            'TANKER': [1.633, -0.3280, 0.7256, -0.5122, -0.0214],
            'BULK_CARRIER': [3.7682, -0.3236, 0.3566, -0.0455, -0.1866],
            'CONTAINER_SHIP': [0.9804, -0.9809, 0.7416, -0.3742, -0.025],
            'DEFAULT': [0.7261, -1.63, 0.6846, -0.5007, 0.2805]
        }

        mask_method2 = (
                df[col_L].notna() & (df[col_L] > 0) &
                df[col_B].notna() & (df[col_B] > 0) &
                df[col_T].notna() & (df[col_T] > 0) &
                df[col_speed].notna() & (df[col_speed] > 0)
        )

        # Calculate Method 2 for each row
        for idx in df[mask_method2].index:
            L = df.loc[idx, col_L]
            B = df.loc[idx, col_B]
            T = df.loc[idx, col_T]
            vs = df.loc[idx, col_speed] * 0.514444  # Convert kn to m/s


            if shiptype and str(shiptype).upper() in coeff_dict:
                coeff = coeff_dict[str(shiptype).upper()]
            else:
                coeff = coeff_dict['DEFAULT']

            a1, a2, a3, a4, a5 = coeff

            # Calculate C_B: C_B = (Vs / (a1 * L^a3 * B^a4 * T^a5))^(1/a2)
            try:
                C_B = (vs / (a1 * (L ** a3) * (B ** a4) * (T ** a5))) ** (1 / a2)
                # Sanity check for reasonable C_B values (typically 0.5-0.9)
                if 0.3 < C_B < 1.0:
                    df.loc[idx, 'C_B_method2'] = C_B
            except:
                pass

        # Create final C_B column (prefer Method 1)
        df['C_B'] = df['C_B_method1'].fillna(df['C_B_method2'])

        # Calculate statistics
        n_total = len(df)
        n_method1_only = (df['C_B_method1'].notna() & df['C_B_method2'].isna()).sum()
        n_method2_only = (df['C_B_method1'].isna() & df['C_B_method2'].notna()).sum()
        n_both = (df['C_B_method1'].notna() & df['C_B_method2'].notna()).sum()
        n_neither = df['C_B'].isna().sum()
        n_calculated = df['C_B'].notna().sum()

        # Calculate average difference for rows with both methods
        both_mask = df['C_B_method1'].notna() & df['C_B_method2'].notna()
        if both_mask.sum() > 0:
            differences = abs(df.loc[both_mask, 'C_B_method1'] - df.loc[both_mask, 'C_B_method2'])
            avg_difference = differences.mean()
            max_difference = differences.max()
            median_difference = differences.median()
        else:
            avg_difference = None
            max_difference = None
            median_difference = None

        # Log results
        self.logger.info("=" * 60)
        self.logger.info("BLOCK COEFFICIENT CALCULATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total rows in database: {n_total}")
        self.logger.info(f"Rows with C_B calculated: {n_calculated} ({n_calculated / n_total * 100:.1f}%)")
        self.logger.info(f"Rows without C_B: {n_neither} ({n_neither / n_total * 100:.1f}%)")
        self.logger.info("-" * 60)
        self.logger.info(f"Method 1 only (DWT/LDT): {n_method1_only} rows")
        self.logger.info(f"Method 2 only (Katsoulis): {n_method2_only} rows")
        self.logger.info(f"Both methods available: {n_both} rows (Method 1 used)")
        self.logger.info("-" * 60)

        if avg_difference is not None:
            self.logger.info(f"For rows with both methods calculated:")
            self.logger.info(f"  Average absolute difference: {avg_difference:.4f}")
            self.logger.info(f"  Median absolute difference: {median_difference:.4f}")
            self.logger.info(f"  Maximum absolute difference: {max_difference:.4f}")

        self.logger.info("=" * 60)

        stats = {
            'n_total': n_total,
            'n_calculated': n_calculated,
            'n_missing': n_neither,
            'n_method1_only': n_method1_only,
            'n_method2_only': n_method2_only,
            'n_both_methods': n_both,
            'avg_difference': avg_difference,
            'median_difference': median_difference,
            'max_difference': max_difference
        }

        return df, stats

    def calculate_fb_column(self,
                            dataframe:pd.DataFrame,
                            v_column:str,
                            l_column:str):
        """
        Function for adding a length column to the dataframe
        :param v_column: column for ship speed in kn
        :param l_column: column for ship length in metres

        """

        result = dataframe.copy()

        g=9.81

        v_column = result[v_column]*0.514444

        result['F_d']=v_column/np.sqrt(g*result[l_column])

        return dataframe

    def split_fuel_columns(self,
                           df: pd.DataFrame,
                           fuel_column_name: List[str]|str,
                           separator: str = ',',
                           deduplicate: bool = True,
                           keep_original_names: bool = False) -> pd.DataFrame:
        """
        Split a column containing lists of fuels into multiple columns.
        Optionally deduplicates fuels that map to the same fuel type (e.g., VLS IFO and IFO 380).

        :param df: pd.DataFrame
            The dataframe containing the fuel data
        :param fuel_column_name: Union[List[str], str]
            Name(s) of the column(s) containing the fuel lists
        :param separator: str, default=','
            Character(s) used to separate fuels in the list
        :param deduplicate: bool, default=True
            If True, removes duplicate fuel types (e.g., treats VLS IFO and IFO 380 as same type)
        :param keep_original_names: bool, default=True
            If True, keeps original fuel names. If False, uses standardized FuelType names

        Returns:
        --------
        pd.DataFrame
            Original dataframe with added columns for fuel count and individual fuels
        """

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Ensure fuel_column_name is a list
        if isinstance(fuel_column_name, str):
            fuel_column_name = [fuel_column_name]

        for fuel_column in fuel_column_name:

            if fuel_column not in result_df.columns:
                self.logger.error(f"available columns: {result_df.columns}")
                raise KeyError(f"available columns: {result_df.columns}")

            # Split the fuel column and strip whitespace
            fuel_lists = result_df[fuel_column].fillna('').astype(str).apply(
                lambda x: [fuel.strip() for fuel in x.split(separator) if fuel.strip()]
            )

            # Deduplicate if requested
            if deduplicate:
                fuel_lists = fuel_lists.apply(
                    lambda fuels: self._deduplicate_fuels(fuels, keep_original_names)
                )

            # Count the number of fuels for each ship (after deduplication)
            result_df[f'{fuel_column}_count'] = fuel_lists.apply(len)

            # Find the maximum number of fuels (longest list after deduplication)
            max_fuels = fuel_lists.apply(len).max()

            # Handle case where all fuel lists are empty
            if max_fuels == 0:
                continue

            # Create individual fuel columns
            for i in range(max_fuels):
                result_df[f'{fuel_column}_{i + 1}'] = fuel_lists.apply(
                    lambda x: x[i] if i < len(x) else None
                )

        return result_df

    def _deduplicate_fuels(self,
                           fuel_list: List[str],
                           keep_original_names: bool = True) -> List[str]:
        """
        Remove duplicate fuel types from a list.
        For example, ['VLS IFO', 'IFO 380', 'LNG'] becomes ['VLS IFO', 'LNG']
        because both VLS IFO and IFO 380 map to the same FuelType.HFO.

        Parameters:
        -----------
        :param fuel_list: List[str], List of fuel names
        :param keep_original_names: bool
            If True, keeps the first occurrence's original name
            If False, uses the standardized FuelType name

        Returns:
        --------
        List[str]
            Deduplicated list of fuels
        """
        seen_types = set()
        unique_fuels = []

        for fuel_name in fuel_list:
            # Try to match to a FuelType
            fuel_type = FuelType.from_identifier(fuel_name)

            if fuel_type is None:
                # Unknown fuel - keep it anyway but warn
                unique_fuels.append(fuel_name)
                continue

            # Check if we've already seen this fuel type
            if fuel_type not in seen_types:
                seen_types.add(fuel_type)

                # Decide what name to use
                if keep_original_names:
                    unique_fuels.append(fuel_name)
                else:
                    # Use standardized name from enum
                    unique_fuels.append(fuel_type.name)

        return unique_fuels

    def create_product_column(self,
                              df: pd.DataFrame,
                              columns: List[str],
                              name:str|None=None,
                              all_combinations:bool=False):
        """
        Method for creating a column that is a product

        :param columns: List of columns to create a product from
        :param name: Name of the product column
        all_combinations: bool, default=False. if true, columns are created for each possible combinations from the list of columns
        """

        if len(columns) == 1:
            self.logger.warning('incorrect input given to product column function')
            return df

        result=df.copy()

        if all_combinations:
            # Generate all combinations of length 2 to len(columns)
            combo_list = []
            for r in range(2, len(columns) + 1):
                combo_list.extend(iter_combinations(columns, r))

            for combo in combo_list:
                result = self._product_column_func(df=result,
                                                   columns=combo,
                                                   name=None)
        else:
            result = self._product_column_func(df=result,
                                               columns=columns,
                                               name=name)
        return result



    def _product_column_func(self,
                              df: pd.DataFrame,
                              columns: Tuple[str]|List[str],
                              name:str|None=None):

        """
        Helper function for product column creation
        """
        first_column = columns[0]

        if name is None:
            full_name = '*'.join(columns)

        else:
            full_name = name

        self.logger.info(f'New column name: {full_name}')

        df[full_name] = df[list(columns)].prod(axis=1)

        return df

    def create_power_column(self,
                            df: pd.DataFrame,
                            columns: List[str],
                            power: int = 2,
                            all_powers: bool = False):
        """
        Method for creating column(s) raised to a power

        :param df: DataFrame to add power column(s) to
        :param columns: List of columns to raise to power(s)
        :param power: Power to raise columns to (default=2). If all_powers=True,
                      this is the maximum power
        :param all_powers: bool, default=False. If True, creates columns for all
                          powers from 2 to the specified power
        :return: DataFrame with new power column(s)
        """

        result = df.copy()

        if all_powers:
            # Create all powers from 2 to the specified power
            for col in columns:
                for p in range(2, power + 1):
                    result = self._power_column_func(df=result,
                                                     column=col,
                                                     power=p)
        else:
            # Create single power for each column
            for col in columns:
                result = self._power_column_func(df=result,
                                                 column=col,
                                                 power=power)

        return result

    def _power_column_func(self,
                           df: pd.DataFrame,
                           column: str,
                           power: int):
        """
        Helper function for power column creation

        :param df: DataFrame to add power column to
        :param column: Column to raise to power
        :param power: Power to raise column to
        :return: DataFrame with new power column
        """

        # Create name like "DWT^3"
        full_name = f"{column}^{power}"

        self.logger.info(f'Creating power column: {full_name}')

        # Calculate power
        df[full_name] = df[column] ** power

        return df












