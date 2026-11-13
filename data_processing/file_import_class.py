"""
This module stores classes that are used for importing data
Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""
import re
import numpy as np
import pandas as pd
import os
import sys
import logging
from pathlib import Path
import hashlib
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Callable, Tuple


os.chdir(r"/")


sys.path.append(r"/")

from config.loggermixin.logger_config import LoggerMixin


class DataFileManager(LoggerMixin):
    """
    Handles file discovery and validation operations.

    This class manages all file-related operations including finding files
    in directories and validating file types.
    """

    def __init__(self):
        self.supported_extensions = {
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.csv': pd.read_csv,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet
        }

    def list_files(self,
                   folder_path: str|Path,
                   file_types: List[str] = None,
                   full_path: bool = True) -> List[str]:
        """
        Find all files of specified types in a directory.

        :param folder_path: Path to the folder containing files to scan.
        :param file_types: List of file extensions to find
        :param full_path: If True, return full paths; if False, return just filenames

        :return: List of files (either full paths or just filenames
        """

        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            self.logger.error(f"'{folder_path}' is not a valid directory path")
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist or is not a directory.")

        if isinstance(file_types, str):
            file_types = [file_types]

        if file_types is None:
            file_types = ['.xls', '.xlsx']
            self.logger.info(f"No file types specified. Defaulting to: {file_types}")

        # Add dots to file types if not present
        file_types = [ft if ft.startswith('.') else f'.{ft}' for ft in file_types]

        # List matching files using Path.glob
        matching_files = [f for f in folder_path.iterdir() if f.suffix.lower() in file_types and f.is_file()]


        if not matching_files:
            self.logger.warning(f"No files found in '{folder_path}' with extensions: {file_types}")
            return []


        if full_path:
            return matching_files

        else:
            return [f.name for f in matching_files]


    def get_import_function(self, file_path: str|Path) -> Callable:
        """Determine the appropriate import function based on file extension."""

        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        # file_extension = os.path.splitext(file_path)[1].lower()
        import_function = self.supported_extensions.get(file_extension)

        if import_function is None:
            self.logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")

        return import_function


class DataProcessor(LoggerMixin):
    """
    Handles data processing operations including cleaning, column selection,
    and transformations.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        """
        Initialises the Data Processor class

        """
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def clean_data(self,
                   datafile: pd.DataFrame,
                   strip_whitespace: bool = True,
                   uppercase: bool = True,
                   replace_spaces: Union[bool, str] = False,
                   drop_first_row: bool = True) -> pd.DataFrame:
        """
        Apply configurable cleaning operations to a DataFrame.
        """
        try:
            self.logger.info("Starting data cleaning process")
            cleaned_df = datafile.copy()

            # Process column names
            if strip_whitespace or uppercase or replace_spaces:
                # Get original column names
                columns = cleaned_df.columns.tolist()

                # Process each column name
                for i, col in enumerate(columns):
                    if not isinstance(col, str):
                        continue

                    # Handle stripping
                    if strip_whitespace:
                        col = col.strip()

                    # Handle capitalization
                    if uppercase:
                        col = col.upper()

                    # Handle space replacement
                    if replace_spaces:
                        replacement = '_' if replace_spaces is True else replace_spaces
                        col = col.replace(' ', replacement)

                    columns[i] = col

                # Apply the processed column names
                cleaned_df.columns = columns
                self.logger.info(f"Processed {len(columns)} column names")

            # Drop the first row if requested
            if drop_first_row and not cleaned_df.empty:
                cleaned_df = cleaned_df.iloc[1:]
                self.logger.info("Dropped first row of data")

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def select_columns(self,
                       datafile: pd.DataFrame,
                       columns_to_keep: Optional[Union[List[str], str, Tuple[str, str], slice]] = None) -> pd.DataFrame:
        """
        Select columns from the DataFrame.

        Args:
            datafile: DataFrame to process
            columns_to_keep: Columns to keep. Can be:
                - List of column names
                - Slice with start and end column names
                - String with column names separated by commas
                - Tuple with start and end column names

        Returns:
            DataFrame with selected columns
        """
        all_columns = list(datafile.columns)

        if columns_to_keep is None:
            return datafile

        # Handle string representation of column range (e.g., ":'SPEED_OVER_GROUND_LOG'")
        if isinstance(columns_to_keep, str) and columns_to_keep.startswith(':'):
            end_col = columns_to_keep[1:].strip("'\"")
            if end_col in all_columns:
                end_idx = all_columns.index(end_col)
                selected_columns = all_columns[:end_idx + 1]
                result = datafile.loc[:, selected_columns]
                self.logger.info(f"Selected columns from start to '{end_col}'")
                return result

        # Handle list of column names
        if isinstance(columns_to_keep, list):
            selected_columns = columns_to_keep

        # Handle comma-separated string
        elif isinstance(columns_to_keep, str) and ',' in columns_to_keep:
            selected_columns = [x.strip() for x in columns_to_keep.split(',')]

        # Handle column range as tuple
        elif isinstance(columns_to_keep, tuple) and len(columns_to_keep) == 2:
            start_col, end_col = columns_to_keep

            if start_col not in all_columns or end_col not in all_columns:
                raise ValueError(f"Column names not found: {start_col} or {end_col}")

            start_idx = all_columns.index(start_col)
            end_idx = all_columns.index(end_col)

            # Handle reverse order if needed
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            selected_columns = all_columns[start_idx:end_idx + 1]

        # Handle slice object
        elif isinstance(columns_to_keep, slice):
            start = columns_to_keep.start
            stop = columns_to_keep.stop

            if start is None:
                start_idx = 0
            elif isinstance(start, str) and start in all_columns:
                start_idx = all_columns.index(start)
            else:
                raise ValueError(f"Start column '{start}' not found")

            if stop is None:
                end_idx = len(all_columns) - 1
            elif isinstance(stop, str) and stop in all_columns:
                end_idx = all_columns.index(stop)
            else:
                raise ValueError(f"End column '{stop}' not found")

            selected_columns = all_columns[start_idx:end_idx + 1]
        else:
            raise ValueError(
                "columns_to_keep must be None, a list of column names, "
                "a comma-separated string, a tuple of (start_column, end_column), "
                "or a slice"
            )

        # Validate selected columns
        invalid_columns = [col for col in selected_columns if col not in all_columns]
        if invalid_columns:
            self.logger.warning(f"Columns not found and will be skipped: {invalid_columns}")
            selected_columns = [col for col in selected_columns if col in all_columns]

        # Select the columns and return
        result = datafile.loc[:, selected_columns]
        self.logger.info(f"Selected {len(selected_columns)} columns")
        return result

    def rename_columns(self,
                       datafile: pd.DataFrame,
                       column_renames: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Rename columns in the DataFrame.

        Args:
            datafile: DataFrame to process
            column_renames: Dictionary mapping old column names to new column names

        Returns:
            DataFrame with renamed columns
        """
        if column_renames is None:
            return datafile

        result = datafile.rename(columns=column_renames)
        self.logger.info(f"Renamed {len(column_renames)} columns")
        return result


class FileReader(LoggerMixin):
    """
    Class for reading data files using appropriate import functions.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False,
                 cache_dir = None):
        self.verbose = verbose
        self.debug = debug
        self.file_manager = DataFileManager()

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        #Directory for storing pickled dataframes


        self.cache_dir = None

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created cache directory: {self.cache_dir}")



    def _get_pickle_path(self,
                         file_path:str|Path,
                         sheet_name: int|str|None = 0) -> Path:
        """
        Generate a path for the pickle file corresponding to the original file.

        :param file_path: Path to the original file
        :param sheet_name: Sheet name or index for Excel files
        :return: Path to the corresponding pickle file
        """
        # Create a unique filename based on the original file path and sheet if applicable
        file_path = Path(file_path)
        file_name = file_path.stem  # equivalent to basename without extension

        # Include sheet name in the pickle filename for Excel files
        if file_path.suffix.lower() in ('.xlsx', '.xls'):
            sheet_str = str(sheet_name).replace('/', '_').replace('\\', '_')
            pickle_filename = f"{file_name}_sheet_{sheet_str}.pkl"
        else:
            pickle_filename = f"{file_name}.pkl"

        if self.cache_dir:
            return self.cache_dir / pickle_filename
        else:
            return file_path.parent / pickle_filename

    def _is_pickle_valid(self, file_path: Union[str, Path], pickle_path: Union[str, Path]) -> bool:
        """
        Check if a pickle file is valid and up-to-date.

        :param file_path: Path to the original file
        :param pickle_path: Path to the pickle file
        :return: True if the pickle file is valid and newer than the original file
        """

        file_path = Path(file_path)
        pickle_path = Path(pickle_path)

        # Check if pickle file exists
        if not pickle_path.exists():
            return False


        return pickle_path.stat().st_mtime > file_path.stat().st_mtime

    def save_to_pickle(self, df: pd.DataFrame,
                       file_path: str|Path,
                       sheet_name: int|str= 0) -> str:
        """
        Save a DataFrame to a pickle file.

        :param df: DataFrame to save
        :param file_path: Path to the original file (used to generate pickle name)
        :param sheet_name: Sheet name or index for Excel files
        :return: Path to the created pickle file
        """
        pickle_path = self._get_pickle_path(file_path, sheet_name)

        try:
            self.logger.info(f"Saving DataFrame to pickle: {pickle_path}")
            df.to_pickle(pickle_path)
            self.logger.info(f"Successfully saved DataFrame to {pickle_path}")
            return str(pickle_path)
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to pickle {pickle_path}: {e}")
            raise

    def read_file(self,
                  file_path: str|Path,
                  sheet_name: int| str= 0,
                  import_function: Callable|None = None,
                  show_progress: bool = True,
                  use_cache: bool = True,
                  save_cache: bool = True,
                  nrows: Optional[int] = None,
                  usecols: List[int]|List[str]| Callable|None=None,
                  encodings: List[str] = ['utf-8', 'cp1252', 'latin1']) -> pd.DataFrame:
        """
        Read a data file using the appropriate import function, with optional caching.

        :param file_path: Path to the file to read
        :param sheet_name: Sheet name or index for Excel files
        :param import_function: Custom import function to use instead of auto-detection
        :param show_progress: Whether to show a progress bar
        :param use_cache: Whether to use cached pickle file if available
        :param save_cache: Whether to save the loaded data to a pickle file
        :param nrows: Number of rows to read (None means all rows)
        :param usecols: Columns to use. Can be:
            - List of column indices (0-indexed)
            - List of column names
            - Callable function that takes a column name and returns a bool
        :return: DataFrame with loaded data"""

        # Create a pickle path that includes the selection parameters

        file_path = Path(file_path)

        custom_pickle_path = None
        if nrows is not None or usecols is not None:
            cache_key_parts = [file_path.stem]
            if nrows is not None:
                cache_key_parts.append(f"nrows_{nrows}")
            if usecols is not None:
                cache_key_parts.append(f"usecols_{len(usecols)}")

            cache_key = '_'.join(cache_key_parts)

            if len(cache_key) > 200:
                hash_obj = hashlib.md5(cache_key.encode())
                cache_key = f"{file_path.stem}_{hash_obj.hexdigest()}"
            pickle_filename = f"{cache_key}.pkl"

            custom_pickle_path = (self.cache_dir or file_path.parent) / pickle_filename


        pickle_path = custom_pickle_path or self._get_pickle_path(file_path, sheet_name)

        if use_cache and self._is_pickle_valid(file_path, pickle_path):
        # if custom_pickle_path or self._is_pickle_valid(file_path, pickle_path):
            try:
                self.logger.info(f"Loading from cached pickle: {pickle_path}")

                with tqdm(total=100, desc=f"Loading cached {os.path.basename(file_path)}",
                          disable=not (self.verbose and show_progress)) as pbar:
                    pbar.update(10)
                    datafile = pd.read_pickle(pickle_path)
                    pbar.update(90)

                self.logger.info(f"Successfully loaded {len(datafile)} rows from cache")
                return datafile
            except Exception as e:
                self.logger.warning(f"Error loading from pickle {pickle_path}: {e}. Falling back to original file.")
                # Continue to load from original file if pickle load fails

        if import_function is None:
            import_function = self.file_manager.get_import_function(file_path)
            self.logger.info(f"Import function: {import_function.__name__}")

        try:
            self.logger.info(f"Reading file: {file_path}")
            if nrows is not None:
                self.logger.info(f"Reading {nrows} rows from file: {file_path}")
            if usecols is not None:
                self.logger.info(f"Reading {len(usecols)} columns from file: {file_path}")

            # Create a progress bar
            filename = file_path.name
            with tqdm(total=100, desc=f"Importing {filename}",
                      disable=not (self.verbose and show_progress)) as pbar:
                # Update progress to 10% to indicate we're starting
                pbar.update(10)

                kwargs={}

                if nrows is not None:
                    kwargs['nrows'] = nrows
                if usecols is not None:
                    kwargs['usecols'] = usecols

                # Special case for Excel files to handle sheet_name and column/row selection
                if file_path.suffix.lower() in {'.xlsx', '.xls'}:

                    kwargs['sheet_name'] = sheet_name
                    datafile = import_function(file_path, **kwargs)

                elif file_path.suffix.lower() in {'.csv'}:
                    for encoding in encodings:
                        try:
                            kwargs['encoding'] = encoding
                            datafile = import_function(file_path, **kwargs)
                            break
                        except UnicodeDecodeError:
                            continue

                else:
                    datafile = import_function(file_path, **kwargs)

                pbar.update(90)

            if isinstance(datafile, pd.DataFrame):
                row_count = len(datafile)
                col_count = len(datafile.columns)
            else:  # Handle case where result is a dict of DataFrames (e.g., multiple Excel sheets)
                row_count = sum(len(df) for df in datafile.values())
                col_count = sum(len(df.columns) for df in datafile.values())

            self.logger.info(f"Successfully read {row_count} rows and {col_count} columns from {filename}")

            if save_cache:
                self.save_to_pickle(df=datafile,
                                    file_path=file_path,
                                    sheet_name=sheet_name)
                self.logger.info(f"Successfully cached data to {pickle_path}")
            return datafile

        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            raise

class DataImporter(LoggerMixin):
    """
    Main class that orchestrates the data import workflow.

    This class combines the functionality of FileReader and DataProcessor
    to provide a complete data import solution.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False,
                 cache_dir: Optional[str] = None):

        self.verbose = verbose
        self.debug = debug
        self.file_reader = FileReader(verbose=verbose, debug=debug)
        self.processor = DataProcessor(verbose=verbose, debug=debug)
        self.file_manager = self.file_reader.file_manager
        self.cache_dir = cache_dir

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def import_file(self,
                    file_path: str|Path,
                    sheet_name: int| str = 0,
                    columns_to_keep: List[str]|str|Tuple[str, str]|slice|None = None,
                    columns_to_rename: dict[str, str]|None = None,
                    strip_whitespace: bool = True,
                    uppercase: bool = True,
                    replace_spaces: bool|str = False,
                    drop_first_row: bool = True,
                    import_function: Callable|None = None,
                    show_file_progress: bool = True,
                    use_cache: bool = True,
                    save_cache: bool = True,
                    cache_processed_data: bool = False) -> pd.DataFrame:
        """
        Import and process a single data file with configurable options.


        :param file_path: Path to the file to import
        :param sheet_name: Sheet name or index for Excel files
        :param columns_to_keep: Columns to keep (see select_columns for options)
        :param columns_to_rename: Dictionary mapping old column names to new column names
        :param strip_whitespace: Whether to strip whitespace from column names
        :param uppercase: Whether to convert column names to uppercase
        :param replace_spaces: If True, replace spaces with underscores; if string, use that as replacement
        :param drop_first_row: Whether to drop the first row of data
        :param import_function: Custom import function to use instead of auto-detection
        :param show_file_progress: Whether to show a progress bar for file reading
        :param use_cache: Whether to use cached pickle file if available
        :param save_cache: Whether to save the loaded data to a pickle file
        :param cache_processed_data: Whether to cache the fully processed data rather than just the raw import

        Returns:
            Processed DataFrame
        """
        try:
            file_path = Path(file_path)
            self.logger.info(f"Importing file: {file_path}")

            # Setup a pickle path for post-processed data if requested
            processed_pickle_path = None
            if cache_processed_data and self.cache_dir:
                cache_dir_path = Path(self.cache_dir)
                cache_dir_path.mkdir(parents=True, exist_ok=True)

                # Construct the processed pickle file name
                stem = file_path.stem
                if file_path.suffix.lower() in {'.xlsx', '.xls'}:
                    sheet_str = str(sheet_name).replace('/', '_').replace('\\', '_')
                    processed_pickle_filename = f"{stem}_sheet_{sheet_str}_processed.pkl"
                else:
                    processed_pickle_filename = f"{stem}_processed.pkl"

                processed_pickle_path = cache_dir_path / processed_pickle_filename

                # Try to load fully processed data from cache
                if use_cache and processed_pickle_path.exists():
                    orig_mtime = file_path.stat().st_mtime
                    pickle_mtime = processed_pickle_path.stat().st_mtime

                    if pickle_mtime > orig_mtime:
                        try:
                            self.logger.info(f"Loading processed data from cache: {processed_pickle_path}")
                            datafile = pd.read_pickle(processed_pickle_path)
                            datafile.name = stem
                            self.logger.info(f"Successfully loaded processed data from cache: {len(datafile)} rows")
                            return datafile
                        except Exception as e:
                            self.logger.warning(f"Failed to load processed cache: {e}. Re-importing from source.")

            # Read the file with caching options
            datafile = self.file_reader.read_file(
                file_path=file_path,
                sheet_name=sheet_name,
                import_function=import_function,
                show_progress=show_file_progress,
                use_cache=use_cache and not cache_processed_data,  # If we cache processed data, no need to cache raw
                save_cache=save_cache and not cache_processed_data  # Only cache raw data if not caching processed
            )

            # Setup a progress bar for the processing steps
            filename = file_path.name
            with tqdm(total=3, desc=f"Processing {filename}",
                      disable=not (self.verbose and show_file_progress)) as pbar:

                # Clean the data
                datafile = self.processor.clean_data(
                    datafile,
                    strip_whitespace=strip_whitespace,
                    uppercase=uppercase,
                    replace_spaces=replace_spaces,
                    drop_first_row=drop_first_row
                )
                pbar.update(1)

                # Select columns
                datafile = self.processor.select_columns(datafile, columns_to_keep)
                pbar.update(1)

                # Rename columns
                datafile = self.processor.rename_columns(datafile, columns_to_rename)
                pbar.update(1)

            # Add file name as an attribute
            # file_name = os.path.splitext(os.path.basename(file_path))[0]
            datafile.name = file_path.stem

            # Save processed data to cache if requested
            if cache_processed_data and processed_pickle_path and save_cache:
                try:
                    self.logger.info(f"Saving processed data to cache: {processed_pickle_path}")
                    datafile.to_pickle(processed_pickle_path)
                    self.logger.info(f"Successfully saved processed data cache")
                except Exception as e:
                    self.logger.warning(f"Error saving processed data cache: {e}")

            if self.verbose:
                self.logger.info(f"Imported and processed {len(datafile)} data points from {file_path}")

            return datafile

        except Exception as e:
            self.logger.error(f"Error importing {file_path}: {e}")
            raise

    def import_directory(self,
                         directory_path: str|Path,
                         file_types: List[str] = None,
                         sheet_name: int|str = 0,
                         columns_to_keep: List[str]|str|Tuple[str, str]|slice = None,
                         columns_to_rename: dict[str, str]|None = None,
                         strip_whitespace: bool = True,
                         uppercase: bool = True,
                         replace_spaces: bool| str = False,
                         drop_first_row: bool = True,
                         show_progress: bool = True,
                         use_cache: bool = True,
                         save_cache: bool = True,
                         cache_processed_data: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Import all files of specified types from a directory.

        :param directory_path: Path to the directory containing files to import
        :param file_types: List of file extensions to import
        :param sheet_name: Sheet name or index for Excel files
        :param columns_to_keep: Columns to keep (see select_columns for options)
        :param  columns_to_rename: Dictionary mapping old column names to new column names
        :param strip_whitespace: Whether to strip whitespace from column names
        :param uppercase: Whether to convert column names to uppercase
        :param replace_spaces: If True, replace spaces with underscores; if string, use that as replacement
        :param drop_first_row: Whether to drop the first row of data
        :param show_progress: Whether to show a progress bar
        :param use_cache: Whether to use cached pickle files if available
        :param save_cache: Whether to save the loaded data to pickle files
        :param cache_processed_data: Whether to cache the fully processed data rather than just the raw import

        Returns:
            Dictionary mapping file names to DataFrames
        """
        # Always get full paths for actual file reading
        file_paths = self.file_manager.list_files(directory_path,
                                                  file_types,
                                                  full_path=True)

        if not file_paths:
            self.logger.warning(f"No files found in '{directory_path}' with the specified extensions.")
            return {}

        imported_data = {}

        # Create a progress bar for the directory import
        with tqdm(total=len(file_paths), desc=f"Importing files from {directory_path}",
                  disable=not (self.verbose and show_progress)) as dir_pbar:

            for file_path in file_paths:
                try:
                    # Don't show individual file progress bars if showing directory progress
                    df = self.import_file(
                        file_path,
                        sheet_name=sheet_name,
                        columns_to_keep=columns_to_keep,
                        columns_to_rename=columns_to_rename,
                        strip_whitespace=strip_whitespace,
                        uppercase=uppercase,
                        replace_spaces=replace_spaces,
                        drop_first_row=drop_first_row,
                        show_file_progress=False,
                        use_cache=use_cache,
                        save_cache=save_cache,
                        cache_processed_data=cache_processed_data
                    )
                    imported_data[df.name] = df
                    dir_pbar.update(1)
                except Exception as e:
                    self.logger.error(f"Error importing {file_path}: {e}")
                    dir_pbar.update(1)

        self.logger.info(f"Successfully imported {len(imported_data)} files from {directory_path}")
        return imported_data

    def clear_cache(self,
                    file_path: str|None = None,
                    directory_path: str|None = None) -> None:
        """
        Clear cached pickle files.

        Args:
            file_path: If provided, clear cache for this specific file
            directory_path: If provided, clear all caches in this directory

        Note: If both file_path and directory_path are None, clears all caches in self.cache_dir
        """

        # If specific file is provided, clear its cache
        if file_path:
            file_path = Path(file_path)
            pickle_path = self.file_reader._get_pickle_path(file_path)
            if pickle_path.exists():
                try:
                    pickle_path.unlink()
                    self.logger.info(f"Removed cache file: {pickle_path}")
                except Exception as e:
                    self.logger.error(f"Failed to remove cache file {pickle_path}: {e}")

            # Also check for processed data cache
            processed_path = self.file_reader._get_pickle_path(file_path)
            processed_path = processed_path.with_name(pickle_path.stem + '_processed.pkl')
            if processed_path.exists():
                try:
                    processed_path.unlink()
                    self.logger.info(f"Removed processed cache file: {processed_path}")
                except Exception as e:
                    self.logger.error(f"Failed to remove processed cache file {processed_path}: {e}")

        # If directory is provided, clear all caches in it
        elif directory_path:
            directory_path = Path(directory_path)
            pickle_files = directory_path.glob('*.pkl')
            for pkl_file in pickle_files:
                try:
                    pkl_file.unlink()
                    self.logger.info(f"Removed cache file: {pkl_file}")
                except Exception as e:
                    self.logger.error(f"Failed to remove cache file {pkl_file}: {e}")

        # If no path specified, clear all caches in the cache_dir
        elif self.cache_dir:
            cache_dir = Path(self.cache_dir)
            pickle_files = cache_dir.glob('*.pkl')
            for pkl_file in pickle_files:
                try:
                    pkl_file.unlink()
                    self.logger.info(f"Removed cache file: {pkl_file}")
                except Exception as e:
                    self.logger.error(f"Failed to remove cache file {pkl_file}: {e}")
        else:
            self.logger.warning("No cache directory set and no specific path provided. Nothing to clear.")
