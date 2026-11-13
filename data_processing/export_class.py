"""
Stores export class and Enum
"""

import pandas as pd
import os
import sys
import logging
from enum import Enum

from typing import List, Dict, Optional, Union

from pathlib import Path

os.chdir(r"/")


sys.path.append(r"/")

from config.loggermixin.logger_config import LoggerMixin

class ExportFormat(Enum):
    """
    Enumeration of supported export file formats.

    Each format includes:
    - extension: The file extension to use
    - description: A human-readable description of the format
    - limitations: Any specific limitations or requirements
    """
    CSV = ('csv', 'Comma-separated values', 'Single DataFrame only')
    EXCEL = ('xlsx', 'Microsoft Excel workbook', 'Supports multiple sheets')

    def __init__(self, extension: str, description: str, limitations: str):
        self.extension = extension
        self.description = description
        self.limitations = limitations
class DataExporter(LoggerMixin):
    """
    Handles exporting DataFrames to various file formats with comprehensive error checking
    and logging. This class supports both single DataFrame exports and multiple DataFrame
    exports to Excel workbooks.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        self.last_export_path = None
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _ensure_directory_exists(self, directory: Union[str, Path]) -> Path:
        """
        Ensures the export directory exists, creating it if necessary.

        :param directory: Directory path as string or Path object

        :return: Path object representing the directory

        Raises:
            OSError: If directory creation fails
        """
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")
            return directory

        except Exception as e:
            self.logger.info(f"Failed to create directory {directory}: {str(e)}")
            raise OSError(f"Could not create directory {directory}: {str(e)}")

    def _validate_dataframe(self,
                            df: pd.DataFrame,
                            name: str = "DataFrame"):
        """
        Validates that a DataFrame is not empty and has the expected structure.

        :param df: DataFrame to validate
        :type df: pd.DataFrame
        :param name: Name of the DataFrame for error messages
        :type name: str

        Raises:
            ValueError: If DataFrame validation fails
        """
        if df.empty:
            self.logger.error(f"{name} is empty")
            raise ValueError(f"{name} cannot be empty")

        if df.columns.duplicated().any():
            self.logger.info(f"{name} contains duplicate column names")
            raise ValueError(f"{name} contains duplicate column names")

    def export(self,
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               directory: Union[str, Path],
               filename: str,
               format: Union[str, ExportFormat] = ExportFormat.CSV,
               sheet_names: Optional[List[str]] = None) -> Path:
        """
        Exports DataFrame(s) to the specified format and location.

        This method handles both single DataFrames and collections of DataFrames,
        automatically selecting the appropriate export method based on the input
        and desired format.

        :param data: DataFrame or dictionary of DataFrames to export
        :param directory: Directory where the file should be saved
        :param filename: Name of the file (without extension)
        :param format: Export format (CSV or Excel)
        :param sheet_names: Optional sheet names for Excel exports

        Returns:
            Path object pointing to the exported file

        Example:
            exporter = DataExporter()

            # Export a single DataFrame to CSV
            exporter.export(df, 'exports', 'data_export')

            # Export multiple DataFrames to Excel
            dfs = {'Sheet1': df1, 'Sheet2': df2}
            exporter.export(
                dfs,
                'exports',
                'multi_sheet_export',
                format=ExportFormat.EXCEL
            )
        """
        try:
            # Convert string format to enum if necessary
            if isinstance(format, str):
                format = ExportFormat[format.upper()]

            # Ensure valid directory
            directory = self._ensure_directory_exists(directory)

            # Create full file path
            file_path = directory / f"{filename}.{format.extension}"

            self.logger.info(f"Beginning export to {file_path}")

            if format == ExportFormat.CSV:
                self.logger.info(f"Exporting {filename} to {file_path} as .csv file")
                self._export_csv(data, file_path)
            elif format == ExportFormat.EXCEL:
                self.logger.info(f"Exporting {filename} to {file_path} as .xlsx file")
                self._export_excel(data, file_path, sheet_names)

            self.last_export_path = file_path
            self.logger.info(f"Successfully exported to {file_path}")

            return file_path

        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise

    def _export_csv(self,
                    data: Union[pd.DataFrame,Dict[str, pd.DataFrame]],
                    file_path: Union[Path,str]):
        """
        Handles CSV export for a single DataFrame.
        """

        # Convert path to string for compatability
        file_path_str = str(file_path)

        if isinstance(data, dict):
            self.logger.error("CSV format only supports single DataFrame export")
            raise ValueError(
                "CSV export requires a single DataFrame. "
                "For multiple DataFrames, use Excel format."
            )

        self._validate_dataframe(data)

        try:
            data.to_csv(file_path_str, index=False)
            self.logger.info(f"Exported {len(data)} rows to CSV")
        except Exception as e:
            self.logger.info(f"CSV export failed: {str(e)}")
            raise

    def _export_excel(self,
                      data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                      file_path: Union[Path, str],
                      sheet_names: Optional[List[str]] = None):
        """
        Handles Excel export for single or multiple DataFrames.
        """
        try:

            #Convert path to string for compatability
            file_path_str = str(file_path)

            with pd.ExcelWriter(file_path_str, engine="openpyxl") as writer:
                if isinstance(data, pd.DataFrame):
                    # Single DataFrame export
                    self._validate_dataframe(data)
                    data.to_excel(writer, sheet_name="Sheet1", index=False)
                    self.logger.info(
                        f"Exported single DataFrame with {len(data)} rows"
                    )
                else:
                    # Multiple DataFrame export
                    for i, (name, df) in enumerate(data.items()):
                        self._validate_dataframe(df, name)
                        sheet_name = sheet_names[i] if sheet_names else name
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        self.logger.info(
                            f"Exported sheet '{sheet_name}' with {len(df)} rows"
                        )
                if os.path.exists(file_path_str):
                    self.logger.info(f"Successfully verified file exists: {file_path_str}")
                else:
                    self.logger.error(f"Export appeared to succeed but file does not exist: {file_path_str}")
                    raise IOError(f"File was not created: {file_path_str}")


        except Exception as e:
            self.logger.error(f"Excel export failed: {str(e)}")
            raise

