"""
This module contains a class for loading and processing fleet data, primarily for Clarksons Database

Author: Ben Noble
Date: 31-05-2025
Version: 1.0.0
"""

import logging
import os

from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

from data_processing.data_processing_enum_classes import ComparisonOperator
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.fleet_data_class.Plotting_fleet_class import FleetPlottingClass

from config.loggermixin.logger_config import LoggerMixin
from data_processing.workflow_class import DataProcessingWorkflow
from data_processing.file_import_class import DataImporter






# df_clarksons = importer.import_directory(directory_path=file_path)


class FleetData(LoggerMixin):
    """
    A class for storing fleet database to be used with the digital twin driven design method
    """
    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        self.verbose = verbose
        self.debug = debug

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        elif self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)


        self.importer = DataImporter(verbose = self.verbose, debug=self.debug)
        self.workflow = DataProcessingWorkflow(verbose = self.verbose, debug=self.debug)
        self.plotting_tools = FleetPlottingClass(verbose = self.verbose, debug=self.debug)
        self.raw_data = {}
        self.processed_data = {}

    def _return_data_from_name(self,
                               data: str|List[str])-> pd.DataFrame|list[pd.DataFrame]:

        if isinstance(data, str):
            self.logger.debug('Searching for datafile within stored datafiles of AIS data class')
            if data in self.raw_data.keys():
                self.logger.info(f'{data} found in stored raw datafiles')
                data = self.raw_data[data]
            elif data in self.processed_data.keys():
                self.logger.info(f'{data} found in stored processed datafiles')
                data = self.processed_data[data]
            else:
                raise FileNotFoundError(f"No data file found with name {data}")

            return data
        elif isinstance(data, list):
            datafiles = {}
            for name in data:
                if name in self.raw_data.keys():
                    self.logger.info(f'{name} found in stored raw datafiles')
                    datafiles[name]=self.raw_data[name]
                elif name in self.processed_data.keys():
                    self.logger.info(f'{name} found in stored processed datafiles')
                    datafiles[name]=self.processed_data[name]
                else:
                    raise FileNotFoundError(f"No data file found with name {name}")
            return datafiles

        else:
            raise TypeError(f"Type {type(data)} not recognized")

    def _store_dataframe(self,
                         df: pd.DataFrame,
                         source_name: str | None,
                         new_dataframe: bool,
                         name: str | None,
                         operation_suffix: str) -> 'FleetData':
        """
        Helper function to store a dataframe in either raw or processed data dictionaries.

        :param df: The dataframe to store
        :param source_name: Name of the source dataset (if updating existing)
        :param new_dataframe: If True, create new dataset; if False, modify existing
        :param name: Custom name for the dataset
        :param operation_suffix: Suffix to describe the operation (e.g., 'with_ratios', 'filtered')
        :return: Self for method chaining
        """

        if new_dataframe or isinstance(source_name, pd.DataFrame) or source_name is None:
            # Create new dataset
            if name is None:
                name = f'processed{len(self.processed_data) + 1}_{operation_suffix}'

            self.processed_data[name] = df
            self.logger.info(f"Created new dataset '{name}'")

        else:
            # Update existing dataset
            if source_name in self.processed_data:
                if name is None:
                    # Update in place
                    self.processed_data[source_name] = df
                    self.logger.info(f"Updated processed dataset '{source_name}'")
                else:
                    # Create new processed dataset with custom name
                    self.processed_data[name] = df
                    self.logger.info(f"Created new processed dataset '{name}'")

            elif source_name in self.raw_data:
                # Move from raw to processed
                if name is None:
                    name = f'{source_name}_{operation_suffix}'
                self.processed_data[name] = df
                self.logger.info(f"Created processed dataset '{name}' from raw dataset '{source_name}'")

        return self

    def get_dataset(self,
                    name: str,
                    processed: bool = False) -> pd.DataFrame:
        """
        Retrieve a specific dataset by name

        :param name: Name of the dataset
        :param processed: Whether to get from processed (True) or raw (False) datasets

        :return: The requested DataFrame

        :raises KeyError: If dataset name not found
        """
        source = self.processed_data if processed else self.raw_data

        if name not in source:
            available = list(source.keys())
            dataset_type = "processed" if processed else "raw"
            raise KeyError(f"Dataset '{name}' not found in {dataset_type} datasets. "
                           f"Available: {available}")

        return source[name]



    def list_datasets(self) -> None:
        """
        Display information about loaded datasets
        """
        print("Raw datasets:")
        for name, df in self.raw_data.items():
            print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")

        if self.processed_data:
            print("\nProcessed datasets:")
            for name, df in self.processed_data.items():
                print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")

        if not self.raw_data and not self.processed_data:
            print("No datasets loaded yet.")



    def import_data(self,
                    file_path: str|Path,
                    file_type: str|None = None,
                    combine: bool = True,
                    combine_col:str|None = 'MMSI',
                    column_renames: dict[str, str] = None,
                    name: str = None):
        """
        Function imports datafiles from a given filepath. can import a specific datafile or multiple datafimes from a directory.

        :param file_path: Path to the datafile
        :param combine: Whether to combine the datafiles or not. defaults to True
        :param column_renames: Dictionary of column names to rename
        :param name: Name of the datafile

        """

        if isinstance(file_path, str):
            file_path = Path(file_path)

        if name is None:
            idx = len(self.raw_data)+1
            name = f"raw{idx}"

        # Check if path exists
        if not file_path.exists():
            raise FileNotFoundError(f"Path does not exist: {file_path}")

        if file_path.is_dir():
            datafile = self.importer.import_directory(directory_path=file_path,
                                                      file_types=file_type,
                                                      columns_to_rename=column_renames,
                                                      uppercase=True,
                                                      strip_whitespace=True,
                                                      replace_spaces=False)

            if combine:
                datafile = self.workflow.transformer.combine_dataframes_by_key(datafiles=datafile,
                                                                               key_column=combine_col)

                raw_data = {name: datafile}
                self.raw_data.update(raw_data)

            else:
                self.raw_data.update(datafile)

        elif file_path.is_file():
            datafile = self.importer.import_file(file_path=file_path,
                                                 columns_to_rename=column_renames,
                                                 uppercase=True,
                                                 strip_whitespace=True,
                                                 replace_spaces=False)

            raw_data = {name: datafile}
            self.raw_data.update(raw_data)


        else:
            raise ValueError(f"Path is neither a file nor a directory: {file_path}")



        return self

    def create_filter(self,
                     column: str|List[str],
                     value: Any,
                     operator: str|ComparisonOperator = ComparisonOperator.EQUAL,
                     require_all: bool = False,
                     negate: bool = False) -> Dict[str, Any]:

        """
        Calls the filter builder function from the filter library, creating filters to be passed to processing fleet data

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
        return self.workflow.filter.build_filter(column=column,
                                                 value=value,
                                                 operator=operator,
                                                 require_all=require_all,
                                                 negate=negate)

    def add_processed_data(
            self,
            df: pd.DataFrame,
            name: str,
    ) -> 'FleetData':
        """Add an existing DataFrame directly to the processed data."""
        self.processed_data[name] = df
        self.logger.info(f"Added processed dataset '{name}'")
        return self

    def processing_fleet_data(self,
                              datafile: str|pd.DataFrame|None=None,
                              name:str|None = None,
                              custom_keys: dict[str,int] = {'Bulk Carrier': 1, 'Tanker': 2, 'Cruise Ship': 3},
                              duplicates: List[str]|None = None,
                              filter_by: List[dict[str, Any]]|dict[str, Any]|None = None,
                              new_dataframe: bool = True,
                              fuel_columns:List[str]|str|None=None,
                              products_columns:List[str]|str|None=None,
                              powers_columns:List[str]|str|None=None,
                              power_value:int=2,
                              column_renames: dict[str,str]|None=None,
                              IQR_filter:str|list[str]|None=None,
                              percentile_filter_cols:list[str]|None=None,
                              percentile:float|None=0.95,
                              max_nan_columns:list[str]|str|None=None,
                              zero_as_missing:list[str]|None=None,
                              max_nan_no:int = 3
                              ):
        """
        Method for processing the fleet database
        :param duplicates: Columns for removing sister ships from the datafile
        :param custom_keys: custom keys to be specified
        :param filter by: Custom filter function created by the create filter function
        :param new_dataframe: If True a new dataframe will be created
        :param fuel_columns: List of columns which contain fuels that should be split
        :param products_columns: List of columns for which to create products
        :param powers_columns: List of columns for which to create powers
        :param column_renames: dictionary of columns to rename
        :param IQR_filter: Applies IQR filter to specified columns
        :return:
        """

        # Get the source dataframe
        if datafile is None:
            if len(self.raw_data) == 1:
                result = list(self.raw_data.values())[0].copy()
                source_name = list(self.raw_data.keys())[0]
            else:
                raise ValueError("Multiple datasets available - please specify datafile name")
        else:
            source_df = self._return_data_from_name(data=datafile)
            result = source_df.copy()
            source_name = datafile if isinstance(datafile, str) else None

        #
        # result = workflow.manipulator.merge_data(data = result,
        #                                          id_column=id_column,
        #                                          how_to_merge=how_to_merge)

        result = self.workflow.processor.rename_columns(datafile=result,
                                                   column_renames=column_renames,)

        if filter_by is not None:
            result = self.workflow.filter.apply_multiple_filters(datafile=result,
                                                            filters=filter_by)

        result = self.workflow.transformer.create_type_key_column(df=result,
                                                            column_name='TYPE',
                                                            custom_keys=custom_keys)

        if IQR_filter is not None:
            result=self.workflow.filter.multiple_IQR_filter(data=result,
                                                       columns=IQR_filter)

        if percentile_filter_cols is not None:
            result = self.workflow.filter.percentile_filter(data=result,
                                                            columns=percentile_filter_cols,
                                                            percentile=percentile)

        if max_nan_columns is not None:
            result = self.workflow.filter.maximum_nan_filter(data=result, columns=max_nan_columns,
                                                             no_blanks=max_nan_no, zero_as_missing=zero_as_missing)

        if duplicates is not None:
            duplicate_columns = duplicates
            result = self.workflow.manipulator.drop_duplicates(df=result,subset=duplicate_columns)



        if fuel_columns is not None:
            result = self.workflow.manipulator.split_fuel_columns(df=result,
                                                              fuel_column_name=fuel_columns)

        if products_columns is not None:
            result = self.workflow.manipulator.create_product_column(df=result,
                                                                columns=products_columns,
                                                                all_combinations=True)

        if powers_columns is not None:
            result = self.workflow.manipulator.create_power_column(df=result,
                                                               columns=powers_columns,
                                                               power=power_value,
                                                              all_powers=True)


        if isinstance(result, dict):
            processed_data = {}
            for key in result.keys():

                if name is not None:
                    new_key = name
                else:
                    new_key = f"{key}_{'processed'}"
                processed_data[new_key] = result[key]


        else:
            if name is not None:
                new_key = name
            else:
                idx = len(self.processed_data) + 1
                new_key = f'processed{idx}'

            processed_data = {new_key: result}

        if new_dataframe:
            self.processed_data.update(processed_data)

        else:
            if processed_data.keys() in self.processed_data.keys():
                self.processed_data[processed_data.keys()]=processed_data
            else:
                self.processed_data.update(processed_data)

        return self

    def filter_by_dataset(self,
                          datafile: str,
                          column_name:str,
                          reference_values:str,
                          reference_column:str|None=None,
                          keep_matched:bool=False,
                          new_dataframe:bool=False,
                          name:str|None=None,
                          ):
        """
        Enables filtering of one dataset by another.

        :param datafile: The DataFrame or dictionary of DataFrames to filter
        :type datafile: str
        :param column_name: Column in the datafile to check
        :type column_name: str
        :param reference_values: the dataframe to remove from the existing dataframe
        :param reference_column: column to remove same values from
        :param keep_matched: whether to remove the matches or remove the none matches
        """

        # Get the source dataframe
        if datafile is None:
            if len(self.raw_data) == 1:
                result = list(self.raw_data.values())[0].copy()
                source_name = list(self.raw_data.keys())[0]
            else:
                raise ValueError("Multiple datasets available - please specify datafile name")
        else:
            source_df = self._return_data_from_name(data=datafile)
            result = source_df.copy()
            source_name = datafile if isinstance(datafile, str) else None

        reference_values = self._return_data_from_name(data=reference_values)

        result= self.workflow.filter.filter_by_reference_data(datafile=result,
                                                              column_name=column_name,
                                                              reference_values=reference_values,
                                                              reference_column=column_name,
                                                              keep_matching=keep_matched)

        return self._store_dataframe(
            df=result,
            source_name=source_name,
            new_dataframe=new_dataframe,
            name=name,
            operation_suffix='with_ratios'
        )


    def create_ratio_columns(self,
                             datafile: pd.DataFrame | str,
                             columns: List[tuple[str, str]] | dict[str, tuple[str, str]] | tuple[str, str],
                             new_dataframe: bool = False,
                             name: str = None,
                             handle_zero_division: str = 'nan'  # 'nan', 'inf', or 'skip'
                             ) -> 'FleetData':
        """
        Add ratio columns to fleet data

        :param datafile: DataFrame or name of stored dataset to modify
        :param columns: Column pairs for ratios. Can be:
                       - Single tuple: ('numerator', 'denominator')
                       - List of tuples: [('num1', 'den1'), ('num2', 'den2')]
                       - Dict: {'ratio_name': ('numerator', 'denominator')}
        :param new_dataframe: If True, create new dataset; if False, modify existing
        :param name: Name for the new/modified dataset
        :param handle_zero_division: How to handle division by zero ('nan', 'inf', 'skip')
        :return: Self for method chaining

        Examples:
            # Single ratio
            fleet.create_ratio_columns('raw1', ('LOA (M)', 'BEAM MLD (M)'))

            # Multiple ratios with custom names
            fleet.create_ratio_columns('raw1', {
                'L_B_ratio': ('LOA (M)', 'BEAM MLD (M)'),
                'DWT_GT_ratio': ('DWT', 'GT')
            })
        """

        # Get the source dataframe
        if datafile is None:
            if len(self.raw_data) == 1:
                result = list(self.raw_data.values())[0].copy()
                source_name = list(self.raw_data.keys())[0]
            else:
                raise ValueError("Multiple datasets available - please specify datafile name")
        else:
            source_df = self._return_data_from_name(data=datafile)
            result = source_df.copy()
            source_name = datafile if isinstance(datafile, str) else None

        # Standardize input to dictionary format
        ratio_definitions = {}

        if isinstance(columns, tuple):
            # Single tuple case
            col1, col2 = columns
            ratio_definitions[f'{col1}/{col2}'] = columns

        elif isinstance(columns, list):
            # List of tuples case
            for i, col_tuple in enumerate(columns):
                if not isinstance(col_tuple, tuple) or len(col_tuple) != 2:
                    raise ValueError(f"List item {i} must be a tuple of length 2")
                col1, col2 = col_tuple
                ratio_definitions[f'{col1}/{col2}'] = col_tuple

        elif isinstance(columns, dict):
            # Dictionary case - validate values are tuples
            for ratio_name, col_tuple in columns.items():
                if not isinstance(col_tuple, tuple) or len(col_tuple) != 2:
                    raise ValueError(f"Dictionary value for '{ratio_name}' must be a tuple of length 2")
                ratio_definitions[ratio_name] = col_tuple
        else:
            raise TypeError(f"columns must be tuple, list, or dict, got {type(columns)}")

        # Validate all columns exist
        missing_cols = []
        for ratio_name, (col1, col2) in ratio_definitions.items():
            if col1 not in result.columns:
                missing_cols.append(col1)
            if col2 not in result.columns:
                missing_cols.append(col2)

        if missing_cols:
            available_cols = list(result.columns)
            raise KeyError(f"Missing columns: {missing_cols}. Available: {available_cols}")

        # Create ratio columns
        created_ratios = []
        for ratio_name, (numerator_col, denominator_col) in ratio_definitions.items():
            try:
                # Perform the division
                if handle_zero_division == 'skip':
                    # Only calculate where denominator != 0
                    mask = result[denominator_col] != 0
                    result.loc[mask, ratio_name] = result.loc[mask, numerator_col] / result.loc[mask, denominator_col]
                else:
                    result[ratio_name] = result[numerator_col] / result[denominator_col]

                    if handle_zero_division == 'nan':
                        result[ratio_name] = result[ratio_name].replace([float('inf'), float('-inf')], float('nan'))
                    # If 'inf', leave as is (default pandas behavior)

                created_ratios.append(ratio_name)
                self.logger.info(f"Created ratio column '{ratio_name}' = {numerator_col}/{denominator_col}")

            except Exception as e:
                self.logger.error(f"Failed to create ratio '{ratio_name}': {e}")
                continue

        if not created_ratios:
            self.logger.warning("No ratio columns were successfully created")
            return self

        # Determine output name and storage
        return self._store_dataframe(
            df=result,
            source_name=source_name,
            new_dataframe=new_dataframe,
            name=name,
            operation_suffix='with_ratios'
        )

    def create_cb_column(self,
                        datafile: pd.DataFrame | str,
                         name: str = None,
                         length_column: str = 'LOA (M)',
                         beam_column: str = 'BEAM MLD (M)',
                        draft_column: str = 'DRAFT (M)',
                         dwt_column: str = 'DWT',
                         ldt_column: str = 'LDT',
                         speed_column: str = 'SPEED (KN)',
                         shiptype: str = 'BULK_CARRIER',
                         new_dataframe: bool = False,
                         ):

        """
        Method for adding a cb column to the dataset.

        :param datafile: Datafile having C_B column added to it
        :param name: Name for the new dataset with C_B column

        """

        # Get the source dataframe
        if datafile is None:
            if len(self.raw_data) == 1:
                result = list(self.raw_data.values())[0].copy()
                source_name = list(self.raw_data.keys())[0]
            else:
                raise ValueError("Multiple datasets available - please specify datafile name")
        else:
            source_df = self._return_data_from_name(data=datafile)
            result = source_df.copy()
            source_name = datafile if isinstance(datafile, str) else None

        result,stats = self.workflow.manipulator.calculate_cb_column(df=result,
                                                                     col_L=length_column,
                                                                     col_B=beam_column,
                                                                     col_T=draft_column,
                                                                     col_DWT=dwt_column,
                                                                     col_LDT=ldt_column,
                                                                     col_speed=speed_column,
                                                                     shiptype=shiptype)

        # Determine output name and storage
        return self._store_dataframe(
            df=result,
            source_name=source_name,
            new_dataframe=new_dataframe,
            name=name,
            operation_suffix='with_ratios'
        )

    def add_overlap_dataset(self,
                            alternative_df: str,
                            conventional_df: str,
                            name: str,
                            parameter:str,
                            save_dataset:bool=True):
        """
        Creates an overlap dataset and adds it to the dataset.
        """

        conventional_fleet = self.processed_data[conventional_df]
        alternative_fleet = self.processed_data[alternative_df]

        overlap_dataset = MatchedDataset.overlap_dataset(alternative_fleet=alternative_fleet,
                                                         conventional_fleet=conventional_fleet,
                                                         predictor=parameter,
                                                         logger=self.logger)

        if save_dataset:

            if name is None:
                name = f'processed{len(self.processed_data) + 1}'

            self.processed_data[name] = overlap_dataset

        return overlap_dataset

    def add_matched_dataset(self,
                            alternative_df: str,
                            conventional_df: str,
                            name: str,
                            matching_features: List[str] | dict[str, float] = ['DWT', 'SPEED (KN)'],
                            feature_weights:dict[str,float]|None=None,
                            n_matches_per_ship: int = 10,
                            binning_features: List[str] | None = None,
                            store_dataset:bool=True):
        """
        Create a matched dataset of conventional ships that are similar to alternative fuel ships.

        This method finds conventional ships that most closely match the characteristics of
        alternative fuel ships based on specified features. Optionally, ships can be binned
        by categorical features (e.g., number of screws) to ensure only truly comparable
        ships are matched.

        The matching process:

        1. Optionally bin both datasets by categorical features
        2. For each bin (or whole dataset if no binning):
           a. For each alternative fuel ship, calculate similarity scores against all conventional ships
           b. Select the N most similar conventional ships (lowest scores)
        3. Return deduplicated set of matched conventional ships

        :param alternative_df: Name of the processed dataset containing alternative fuel ships.
                              This dataset must exist in ``self.processed_data``.
        :type alternative_df: str

        :param conventional_df: Name of the processed dataset containing conventional fuel ships
                               to match against. This dataset must exist in ``self.processed_data``.
        :type conventional_df: str

        :param name: Name for the new matched dataset that will be created in ``self.processed_data``.
        :type name: str

        :param matching_features: Features to use for similarity matching. Can be:

                                 - **List of strings**: Features with equal weighting (coefficient = 1.0)
                                   e.g., ``['DWT', 'SPEED (KN)']``

        :type matching_features: Union[List[str], Dict[str, float]]

        :param feature_weights: specify a custom set of weights that should be applied to the matching formluae

        :param n_matches_per_ship: Maximum number of similar conventional ships to find for each
                                  alternative fuel ship. Defaults to 10.
        :type n_matches_per_ship: int

        :param binning_features: Optional list of categorical feature column names to use for
                                binning ships before matching. Ships will only be matched within
                                the same bin. Common examples: ``['NUMBER OF SCREWS', 'POWER TYPE']``.
                                If None, no binning is performed and all ships are matched together.
                                **Warning**: Bins with fewer than 5 alternative ships will generate
                                a warning as statistical comparison becomes unreliable.
        :type binning_features: Optional[List[str]]

        :returns: List of dictionaries containing detailed matching information.
        :rtype: List[Dict[str, Any]]

        :raises KeyError: If datasets or features not found
        :raises ValueError: If binning results in no valid matches

        :Example:

            Match without binning::

                match_details = fleet.create_matched_dataset(
                    alternative_df='alt_fuels',
                    conventional_df='conventional',
                    name='matched_basic',
                    matching_features=['DWT', 'SPEED (KN)'],
                    n_matches_per_ship=10
                )

            Match with binning by propulsion configuration::

                match_details = fleet.create_matched_dataset(
                    alternative_df='alt_fuels',
                    conventional_df='conventional',
                    name='matched_binned',
                    matching_features={'DWT': 3.5, 'SPEED (KN)': 1.0},
                    n_matches_per_ship=10,
                    binning_features=['NUMBER OF SCREWS', 'POWER TYPE']
                )

        .. note::

           - Ships with zero or NaN values for matching features are excluded
           - Bins with < 5 alternative ships generate warnings
           - The same conventional ship can match multiple alternative ships
           - Results are automatically deduplicated
        """

        conventional_fleet = self.processed_data[conventional_df]
        alternative_fleet = self.processed_data[alternative_df]

        match_details,matched_conventional = MatchedDataset.create_matched_dataset(alternative_fleet=alternative_fleet,
                                         conventional_fleet=conventional_fleet,
                                         logger=self.logger,
                                         matching_features=matching_features,
                                         feature_weights=feature_weights,
                                         n_matches_per_ship=n_matches_per_ship,
                                         binning_features=binning_features,)

        if store_dataset:

            if name is None:
                name = f'processed{len(self.processed_data) + 1}'

            self.processed_data[name] = matched_conventional

        return matched_conventional

    def get_dimension_summary(self,
                              columns: List[str],
                              datasets: str|List[str]|None = None,
                              include_processed: bool = True,
                              include_raw: bool = True) -> pd.DataFrame:
        """
        Create a summary DataFrame showing min/max ranges for specified columns across datasets

        :param columns: List of column names to analyze (e.g., ['GT', 'DWT', 'LOA'])
        :param datasets: Specific dataset names to analyze. If None, analyzes all available datasets
        :param include_processed: Whether to include processed datasets in analysis
        :param include_raw: Whether to include raw datasets in analysis

        :return: DataFrame with columns: ['dataset', 'min_{col}', 'max_{col}', ...] for each specified column

        :raises ValueError: If no datasets match the criteria or columns don't exist
        """

        # Collect datasets to analyze
        datasets_to_analyze = {}

        if include_raw:
            datasets_to_analyze.update({f"{name}": df for name, df in self.raw_data.items()})

        if include_processed:
            datasets_to_analyze.update({f"{name}": df for name, df in self.processed_data.items()})

        # Filter to specific datasets if requested
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]

            filtered_datasets = {}
            for dataset_name in datasets:
                if dataset_name in datasets_to_analyze:
                    filtered_datasets[dataset_name] = datasets_to_analyze[dataset_name]
                else:
                    available = list(datasets_to_analyze.keys())
                    raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

            datasets_to_analyze = filtered_datasets

        if not datasets_to_analyze:
            raise ValueError("No datasets available for analysis")

        # Prepare result data
        summary_data = []

        for dataset_name, df in datasets_to_analyze.items():
            row_data = {'dataset': dataset_name}

            # Check which columns exist in this dataset
            missing_columns = []
            for col in columns:
                if col not in df.columns:
                    missing_columns.append(col)
                    continue

                # Calculate min and max for numeric columns only
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Handle NaN values by excluding them
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        row_data[f'min_{col}'] = valid_data.min()
                        row_data[f'max_{col}'] = valid_data.max()
                    else:
                        row_data[f'min_{col}'] = None
                        row_data[f'max_{col}'] = None
                else:
                    self.logger.warning(f"Column '{col}' in dataset '{dataset_name}' is not numeric, skipping")
                    row_data[f'min_{col}'] = None
                    row_data[f'max_{col}'] = None

            if missing_columns:
                self.logger.warning(f"Dataset '{dataset_name}' missing columns: {missing_columns}")

            summary_data.append(row_data)

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Ensure consistent column order: dataset first, then min/max pairs for each dimension
        ordered_columns = ['dataset']
        for col in columns:
            ordered_columns.extend([f'min_{col}', f'max_{col}'])

        # Only include columns that exist in the summary_df
        existing_columns = [col for col in ordered_columns if col in summary_df.columns]
        summary_df = summary_df[existing_columns]

        self.logger.info(f"Generated dimension summary for {len(summary_data)} datasets and {len(columns)} dimensions")

        return summary_df

    def get_available_numeric_columns(self,
                                      datasets: str|List[str]|None = None,
                                      include_processed: bool = True,
                                      include_raw: bool = True) -> dict[str, List[str]]:
        """
        Helper function to identify available numeric columns across datasets

        :param datasets: Specific dataset names to check. If None, checks all available datasets
        :param include_processed: Whether to include processed datasets
        :param include_raw: Whether to include raw datasets

        :return: Dictionary mapping dataset names to lists of their numeric columns
        """

        # Collect datasets to analyze (same logic as get_dimension_summary)
        datasets_to_analyze = {}

        if include_raw:
            datasets_to_analyze.update({f"{name}": df for name, df in self.raw_data.items()})

        if include_processed:
            datasets_to_analyze.update({f"{name}": df for name, df in self.processed_data.items()})

        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]

            filtered_datasets = {}
            for dataset_name in datasets:
                if dataset_name in datasets_to_analyze:
                    filtered_datasets[dataset_name] = datasets_to_analyze[dataset_name]

            datasets_to_analyze = filtered_datasets

        # Find numeric columns for each dataset
        numeric_columns_by_dataset = {}

        for dataset_name, df in datasets_to_analyze.items():
            numeric_cols = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)

            numeric_columns_by_dataset[dataset_name] = numeric_cols

        return numeric_columns_by_dataset

    def q_q_plots(self,
                      parameters: List[str],
                        conventional_df: str,
                        alternative_df: str,
                        name: str = 'Q-Q Plot',
                        print_stats: bool = False,
                        colors: List[str] = None,
                        use_different_colors: bool = True,
                        comparison_type: str = 'theoretical',
                        show_reference_line: bool = True,
                        show_confidence_bands: bool = True,
                  use_log_transform:bool=False):

        """
        Create Q-Q (Quantile-Quantile) plots for distribution comparison.

        Q-Q plots compare quantiles of your data against:
        - 'theoretical': Normal distribution (tests normality assumption)
        - 'datasets': Each other (compares if both follow same distribution)

        Points along the diagonal line indicate distributions match.
        Deviations show where and how distributions differ.

        Args:
            parameters: List of parameter names to plot
            conventional_df: DataFrame containing conventional fuel ships
            alternative_df: DataFrame containing alternative fuel ships
            name: Name for the overall plot title
            print_stats: Whether to print statistical summaries
            colors: List of colors for [conventional, alternative]
            use_different_colors: Whether to use different colors for each dataset
            comparison_type: 'theoretical' (vs normal dist) or 'datasets' (vs each other)
            show_reference_line: Whether to show the theoretical diagonal line
            show_confidence_bands: Whether to show confidence bands (theoretical only)
        """

        conventional_df = self._return_data_from_name(data=conventional_df)
        conventional_df = conventional_df.copy()

        alternative_df = self._return_data_from_name(data=alternative_df)
        alternative_df = alternative_df.copy()

        self.plotting_tools.plotting_qq_plots(parameters=parameters,conventional_df=conventional_df,alternative_df=alternative_df,
                            name=name,print_stats=print_stats,colors=colors,use_different_colors=use_different_colors,
                            comparison_type=comparison_type,show_reference_line=show_reference_line,
                            show_confidence_bands=show_confidence_bands, use_log_transform=use_log_transform)

    def histogram_plots(self,
                            parameters: List[str],
                            conventional_df: str,
                            alternative_df: str,
                            name: str = 'histogram',
                            bins: int = 30,
                            print_stats: bool = False,
                            alphas: List[float] = None,
                            colors: List[str] = None,
                            use_different_colors: bool = True,
                            density: bool = False,
                            show_kde: bool = True,
                            show_mean: bool = True,
                            show_median: bool = True):

        """
        Create histogram plots comparing conventional and alternative fuel ship datasets.

        The histogram shows the frequency distribution of values, helping identify:
        - Distribution shape (normal, skewed, bimodal)
        - Central tendency and spread
        - Outliers and gaps in data

        Args:
            parameters: List of parameter names to plot
            conventional_df: DataFrame containing conventional fuel ships
            alternative_df: DataFrame containing alternative fuel ships
            name: Name for the overall plot title
            bins: Number of bins for histogram (default 30)
            print_stats: Whether to print statistical summaries
            alphas: List of transparency values for [conventional, alternative]
            colors: List of colors for [conventional, alternative]
            use_different_colors: Whether to use different colors for each dataset
            density: If True, plot probability density instead of counts
            show_kde: Whether to overlay kernel density estimation curve
            show_mean: Whether to show vertical line at mean
            show_median: Whether to show vertical line at median
        """

        conventional_df = self._return_data_from_name(data=conventional_df)
        conventional_df = conventional_df.copy()

        alternative_df = self._return_data_from_name(data=alternative_df)
        alternative_df = alternative_df.copy()

        self.plotting_tools.plotting_histograms(parameters=parameters,conventional_df=conventional_df,alternative_df=alternative_df,
                                                name=name,print_stats=print_stats,colors=colors,use_different_colors=use_different_colors,
                                                density=density,show_kde=show_kde,show_mean=show_mean,show_median=show_median)

    def plotting_box_plots(self,
                           parameters: List[str],
                           conventional_df: pd.DataFrame,
                           alternative_df: pd.DataFrame,
                           name: str = 'box_plot',
                           show_outliers: bool = True,
                           print_stats: bool = False,
                           alphas: List[float] = None,
                           use_different_colors: bool = True,
                           show_median_lines: bool = True,
                           show_mean_lines: bool = False,
                           shade_between: bool = True):

        conventional_df = self._return_data_from_name(data=conventional_df)
        conventional_df = conventional_df.copy()

        alternative_df = self._return_data_from_name(data=alternative_df)
        alternative_df = alternative_df.copy()

        self.plotting_box_plots(parameters=parameters,conventional_df=conventional_df, alternative_df=alternative_df,
                                name=name, show_outliers=show_outliers,print_stats=print_stats,alphas=alphas,
                                use_different_colors=use_different_colors,show_median_lines=show_median_lines,
                                show_mean_lines=show_mean_lines,shade_between=shade_between)

    def plot_matching_importance(self,
                                 conventional_df: str,
                                 alternative_df: str,
                                 target_variable: str,
                                 predictor_variable: str = 'DWT',
                                 matching_features: List[str] | None = None,
                                 feature_weights: Dict[str, float] | None = None,
                                 n_matches_per_ship: int = 10,
                                 binning_features: List[str] | None = None,
                                 variable_name_map: Dict[str, str] | None = None,
                                 separate_figures: bool = False,
                                 colors: List[str] | None = None,
                                 alphas: List[float] | None = None,
                                 show_statistics: bool = False,
                                 show_histograms: bool = False,
                                 subplot_width_ratios: List[float] | None = None,
                                 name: str = 'Matching Dataset Importance',
                                 print_debug: bool = False):
        """
        Create scatter plots demonstrating the importance of matched datasets.

        CONCEPT: This visualization shows why matched datasets are critical for fair comparison.
        Three plots are created:

        1. CONVENTIONAL FULL: All conventional ships - may span much wider range than alternatives
        2. MATCHED CONVENTIONAL: Subset of conventional ships similar to alternatives - fair comparison
        3. ALTERNATIVE: The alternative fuel ships we want to study

        By comparing these three, you can see:
        - Whether distributions naturally overlap (if yes, matching less critical)
        - How matching narrows the conventional fleet to comparable ships
        - Whether alternative ships occupy a different region of the design space

        WORKFLOW:
        1. Load conventional and alternative datasets by name
        2. Create matched dataset using specified matching features
        3. Plot all three datasets with consistent axes for comparison
        4. Optionally show marginal histograms to highlight distribution differences

        Args:
            conventional_df: Name of conventional dataset in processed_data
            alternative_df: Name of alternative dataset in processed_data
            target_variable: Y-axis variable to plot (e.g., 'BEAM (M)', 'MCR (KW)')
            predictor_variable: X-axis variable (default 'DWT'), also used for matching if matching_features=None
            matching_features: Features for similarity matching. If None, uses [predictor_variable]
            feature_weights: Optional weights for matching features
            n_matches_per_ship: Number of conventional ships to match per alternative ship
            binning_features: Optional categorical features to bin by before matching
            variable_name_map: Dictionary mapping internal names to display names
            separate_figures: If True, create 3 separate figures; if False, 1 figure with 3 subplots
            colors: Colors for [conventional, matched, alternative] datasets
            alphas: Transparency values for [conventional, matched, alternative]
            show_statistics: Whether to display summary statistics on plots
            show_histograms: Whether to show marginal histograms
            name: Base name for plot titles
            print_debug: Whether to print detailed debugging information

        Example:
            # Basic usage - show why matching by DWT is important for BEAM comparison
            fleet.plot_matching_importance(
                conventional_df='pure_conventional',
                alternative_df='pure_alt_fuel',
                target_variable='BEAM (M)',
                predictor_variable='DWT'
            )

            # Advanced - match by DWT and speed with custom weights
            fleet.plot_matching_importance(
                conventional_df='pure_conventional',
                alternative_df='pure_alt_fuel',
                target_variable='MCR (KW)',
                predictor_variable='DWT',
                matching_features=['DWT', 'Vs'],
                feature_weights={'DWT': 3.5, 'Vs': 1.0},
                show_histograms=True
            )
        """

        if print_debug:
            print("\n" + "=" * 60)
            print("STEP 1: Loading datasets")
            print("=" * 60)

        # Step 1: Load datasets
        conventional_fleet = self._return_data_from_name(data=conventional_df).copy()
        alternative_fleet = self._return_data_from_name(data=alternative_df).copy()

        if print_debug:
            print(f"Conventional fleet: {len(conventional_fleet)} ships")
            print(f"Alternative fleet: {len(alternative_fleet)} ships")
            print(f"\nTarget variable: {target_variable}")
            print(f"Predictor variable: {predictor_variable}")

        # Step 2: Determine matching features
        if matching_features is None:
            matching_features = [predictor_variable]
            if print_debug:
                print(f"\nNo matching features specified, using predictor: {matching_features}")
        else:
            if print_debug:
                print(f"\nMatching features: {matching_features}")
                if feature_weights:
                    print(f"Feature weights: {feature_weights}")

        if print_debug:
            print("\n" + "=" * 60)
            print("STEP 2: Creating matched dataset")
            print("=" * 60)

        # Step 3: Create matched dataset
        from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset

        match_details, matched_conventional = MatchedDataset.create_matched_dataset(
            alternative_fleet=alternative_fleet,
            conventional_fleet=conventional_fleet,
            logger=self.logger,
            matching_features=matching_features,
            feature_weights=feature_weights,
            n_matches_per_ship=n_matches_per_ship,
            binning_features=binning_features
        )

        if print_debug:
            print(f"\nMatched dataset created: {len(matched_conventional)} ships")
            print(f"Matching ratio: {len(matched_conventional)}/{len(conventional_fleet)} "
                  f"({len(matched_conventional) / len(conventional_fleet) * 100:.1f}%)")

            # Show matching summary
            summary = MatchedDataset.get_matching_summary(
                match_details, alternative_fleet, matching_features
            )
            print(f"\nMatching summary:")
            print(f"  Alternative ships matched: {summary['n_alternative_matched']}/{summary['n_alternative_total']}")
            if summary['n_alternative_skipped'] > 0:
                print(f"  Alternative ships skipped: {summary['n_alternative_skipped']}")
            print(f"  Conventional ships selected: {summary['n_conventional_matched']}")

        # Step 4: Setup variable names
        if variable_name_map is None:
            variable_name_map = {}

        def get_display_name(var_name: str) -> str:
            """Helper to get display name from variable name."""
            return variable_name_map.get(var_name, var_name)

        x_label = get_display_name(predictor_variable)
        y_label = get_display_name(target_variable)

        if print_debug:
            print("\n" + "=" * 60)
            print("STEP 3: Setting up plot parameters")
            print("=" * 60)
            print(f"X-axis: {predictor_variable} → '{x_label}'")
            print(f"Y-axis: {target_variable} → '{y_label}'")

        # Step 5: Setup colors and alphas
        if colors is None:
            colors = ['lightblue', 'orange', 'red']  # conventional, matched, alternative

        if alphas is None:
            alphas = [0.3, 0.5, 0.6]  # conventional faint, matched medium, alternative prominent

        # Step 6: Prepare data for plotting
        datasets = {
            'Conventional (Full)': conventional_fleet,
            'Conventional (Matched)': matched_conventional,
            'Alternative': alternative_fleet
        }

        if print_debug:
            print("\n" + "=" * 60)
            print("STEP 4: Preparing plot data")
            print("=" * 60)

        # Clean data and calculate statistics
        plot_data = {}
        for dataset_name, df in datasets.items():
            clean_df = df[[predictor_variable, target_variable]].dropna()

            if len(clean_df) > 0:
                plot_data[dataset_name] = {
                    'x': clean_df[predictor_variable],
                    'y': clean_df[target_variable],
                    'n': len(clean_df),
                    'x_mean': clean_df[predictor_variable].mean(),
                    'y_mean': clean_df[target_variable].mean(),
                    'x_std': clean_df[predictor_variable].std(),
                    'y_std': clean_df[target_variable].std()
                }

                if print_debug:
                    print(f"\n{dataset_name}:")
                    print(f"  N = {len(clean_df)}")
                    print(
                        f"  {x_label}: μ={plot_data[dataset_name]['x_mean']:.1f}, σ={plot_data[dataset_name]['x_std']:.1f}")
                    print(
                        f"  {y_label}: μ={plot_data[dataset_name]['y_mean']:.1f}, σ={plot_data[dataset_name]['y_std']:.1f}")
            else:
                self.logger.warning(f"No valid data for {dataset_name}")

        if not plot_data:
            self.logger.error("No valid data for any dataset")
            return

        if print_debug:
            print("\n" + "=" * 60)
            print("STEP 5: Creating plots")
            print("=" * 60)

        # Step 7: Create plots
        if separate_figures:
            self._plot_matching_importance_separate(
                plot_data=plot_data,
                x_label=x_label,
                y_label=y_label,
                colors=colors,
                alphas=alphas,
                show_statistics=show_statistics,
                show_histograms=show_histograms,
                name=name,
                print_debug=print_debug
            )
        else:
            self._plot_matching_importance_combined(
                plot_data=plot_data,
                x_label=x_label,
                y_label=y_label,
                colors=colors,
                alphas=alphas,
                show_statistics=show_statistics,
                show_histograms=show_histograms,
                name=name,
                print_debug=print_debug,
                subplot_width_ratios=subplot_width_ratios
            )

        if print_debug:
            print("\n" + "=" * 60)
            print("Plotting complete!")
            print("=" * 60)

    def _plot_matching_importance_combined(self,
                                           plot_data: Dict,
                                           x_label: str,
                                           y_label: str,
                                           colors: List[str],
                                           alphas: List[float],
                                           show_statistics: bool,
                                           show_histograms: bool,
                                           name: str,
                                           print_debug: bool,
                                           subplot_width_ratios: List[float] | None = None):
        """
        Create combined figure with 3 subplots.

        This internal method creates a single figure with three scatter plots side-by-side,
        allowing direct visual comparison of conventional full, matched, and alternative datasets.

        The plots share y-axes to enable easy comparison of target variable distributions.

        Args:
            subplot_width_ratios: Relative widths of the three subplots [conv_full, matched, alt].
                                 Example: [2, 1, 1] makes first plot twice as wide as others.
                                 Default: [1, 1, 1] (equal widths)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        if print_debug:
            print("Creating combined figure with 3 subplots...")

        # Default to equal widths if not specified
        if subplot_width_ratios is None:
            subplot_width_ratios = [1, 1, 1]

        if len(subplot_width_ratios) != 3:
            self.logger.warning(
                f"subplot_width_ratios must have 3 values, got {len(subplot_width_ratios)}. Using defaults.")
            subplot_width_ratios = [1, 1, 1]

        if print_debug:
            print(f"Subplot width ratios: {subplot_width_ratios}")

        if show_histograms:
            # Create figure with gridspec for main plots + histograms
            fig = plt.figure(figsize=(18, 8))
            gs = GridSpec(2, 3,
                          height_ratios=[3, 1],
                          width_ratios=subplot_width_ratios,
                          hspace=0.05,
                          wspace=0.3)

            axes_main = [fig.add_subplot(gs[0, i]) for i in range(3)]
            axes_hist = [fig.add_subplot(gs[1, i]) for i in range(3)]
        else:
            # Create figure with gridspec for consistent width control
            fig = plt.figure(figsize=(18, 5))
            gs = GridSpec(1, 3,
                          width_ratios=subplot_width_ratios,
                          wspace=0.3)

            axes_main = [fig.add_subplot(gs[0, i]) for i in range(3)]
            axes_hist = None

        dataset_names = ['Conventional (Full)', 'Conventional (Matched)', 'Alternative']

        # Determine global x and y limits for consistent axes
        all_x = np.concatenate([plot_data[name]['x'] for name in dataset_names if name in plot_data])
        all_y = np.concatenate([plot_data[name]['y'] for name in dataset_names if name in plot_data])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        if print_debug:
            print(f"Global X range: [{x_min:.1f}, {x_max:.1f}]")
            print(f"Global Y range: [{y_min:.1f}, {y_max:.1f}]")

        # Plot each dataset
        for idx, dataset_name in enumerate(dataset_names):
            if dataset_name not in plot_data:
                continue

            ax = axes_main[idx]
            data = plot_data[dataset_name]

            if print_debug:
                print(f"\nPlotting {dataset_name} on subplot {idx + 1}")

            # Scatter plot
            ax.scatter(data['x'], data['y'],
                       alpha=alphas[idx],
                       color=colors[idx],
                       s=30,
                       edgecolors='black',
                       linewidth=0.3)

            # Set limits
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Labels
            ax.set_xlabel(x_label, fontsize=11)
            if idx == 0:
                ax.set_ylabel(y_label, fontsize=11)

            # Title with sample size - keep black
            ax.set_title(f"{dataset_name}\n(n={data['n']})",
                         fontsize=12, fontweight='bold',
                         color='black')

            # Add statistics box
            if show_statistics:
                stats_text = (f"μ_x = {data['x_mean']:.1f}\n"
                              f"σ_x = {data['x_std']:.1f}\n"
                              f"μ_y = {data['y_mean']:.1f}\n"
                              f"σ_y = {data['y_std']:.1f}")

                ax.text(0.02, 0.98, stats_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=8)

            # Add mean crosshairs
            ax.axvline(data['x_mean'], color=colors[idx], linestyle='--',
                       linewidth=1.5, alpha=0.5)
            ax.axhline(data['y_mean'], color=colors[idx], linestyle='--',
                       linewidth=1.5, alpha=0.5)

            ax.grid(True, alpha=0.3)

            # Add marginal histogram if requested
            if show_histograms and axes_hist is not None:
                ax_hist = axes_hist[idx]

                ax_hist.hist(data['x'], bins=30, color=colors[idx],
                             alpha=0.7, edgecolor='black', linewidth=0.5)

                ax_hist.axvline(data['x_mean'], color='red',
                                linestyle='--', linewidth=1.5)

                ax_hist.set_xlim(x_min - x_margin, x_max + x_margin)
                ax_hist.set_xlabel(x_label, fontsize=9)
                ax_hist.set_ylabel('Count', fontsize=9)
                ax_hist.grid(True, alpha=0.3, axis='y')

        # Overall title - raised to avoid overlap
        fig.suptitle(f'{name}\n{y_label} vs {x_label}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.show()

        if print_debug:
            print("Combined figure complete")

    def _plot_matching_importance_separate(self,
                                           plot_data: Dict,
                                           x_label: str,
                                           y_label: str,
                                           colors: List[str],
                                           alphas: List[float],
                                           show_statistics: bool,
                                           show_histograms: bool,
                                           name: str,
                                           print_debug: bool):
        """
        Create three separate figures for each dataset.

        This internal method creates individual figures for each dataset, allowing
        focused examination of each distribution. Useful when you need separate
        exports or want to emphasize differences.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if print_debug:
            print("Creating 3 separate figures...")

        dataset_names = ['Conventional (Full)', 'Conventional (Matched)', 'Alternative']

        # Determine global limits for consistency
        all_x = np.concatenate([plot_data[name]['x'] for name in dataset_names if name in plot_data])
        all_y = np.concatenate([plot_data[name]['y'] for name in dataset_names if name in plot_data])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        # Create separate figure for each dataset
        for idx, dataset_name in enumerate(dataset_names):
            if dataset_name not in plot_data:
                continue

            if print_debug:
                print(f"\nCreating figure for {dataset_name}")

            data = plot_data[dataset_name]

            if show_histograms:
                from matplotlib.gridspec import GridSpec
                fig = plt.figure(figsize=(8, 8))
                gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                              hspace=0.05, wspace=0.05)

                ax_main = fig.add_subplot(gs[1, 0])
                ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
                ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

                # Hide tick labels on histogram axes
                ax_top.tick_params(labelbottom=False)
                ax_right.tick_params(labelleft=False)

            else:
                fig, ax_main = plt.subplots(1, 1, figsize=(8, 6))
                ax_top = None
                ax_right = None

            # Main scatter plot
            ax_main.scatter(data['x'], data['y'],
                            alpha=alphas[idx],
                            color=colors[idx],
                            s=40,
                            edgecolors='black',
                            linewidth=0.3)

            ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
            ax_main.set_ylim(y_min - y_margin, y_max + y_margin)

            ax_main.set_xlabel(x_label, fontsize=12)
            ax_main.set_ylabel(y_label, fontsize=12)

            # Add statistics
            if show_statistics:
                stats_text = (f"n = {data['n']}\n"
                              f"μ_x = {data['x_mean']:.1f} ± {data['x_std']:.1f}\n"
                              f"μ_y = {data['y_mean']:.1f} ± {data['y_std']:.1f}")

                ax_main.text(0.02, 0.98, stats_text,
                             transform=ax_main.transAxes,
                             verticalalignment='top',
                             horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                             fontsize=10)

            # Mean lines
            ax_main.axvline(data['x_mean'], color=colors[idx],
                            linestyle='--', linewidth=2, alpha=0.6,
                            label=f'Mean {x_label}')
            ax_main.axhline(data['y_mean'], color=colors[idx],
                            linestyle=':', linewidth=2, alpha=0.6,
                            label=f'Mean {y_label}')

            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc='best', fontsize=9)

            # Marginal histograms
            if show_histograms and ax_top is not None:
                # Top histogram (X variable)
                ax_top.hist(data['x'], bins=30, color=colors[idx],
                            alpha=0.7, edgecolor='black', linewidth=0.5)
                ax_top.axvline(data['x_mean'], color='red',
                               linestyle='--', linewidth=2)
                ax_top.set_ylabel('Count', fontsize=9)
                ax_top.grid(True, alpha=0.3, axis='y')

                # Right histogram (Y variable)
                ax_right.hist(data['y'], bins=30, color=colors[idx],
                              alpha=0.7, edgecolor='black', linewidth=0.5,
                              orientation='horizontal')
                ax_right.axhline(data['y_mean'], color='red',
                                 linestyle='--', linewidth=2)
                ax_right.set_xlabel('Count', fontsize=9)
                ax_right.grid(True, alpha=0.3, axis='x')

            # Title - raised to avoid overlap with subplot content
            fig.suptitle(f'{name}\n{dataset_name} - {y_label} vs {x_label}',
                         fontsize=14, fontweight='bold', y=0.98)

            plt.tight_layout()
            plt.show()

        if print_debug:
            print("All separate figures complete")

# fleet_data = FleetData()
#
# file_path = r'C:\Users\bnoble\PycharmProjects\T1.6_deliverable\parameters\Databases\clarksons_data\CLARKSONS_27_05.xlsx'
#
# df_clarksons = fleet_data.import_data(file_path=file_path
#                                       )
# data_path = r'/parameters/Databases/clarksons_data/CLARKSONS_27_05.xlsx'
#
# df_clarksons.to_excel(data_path)


