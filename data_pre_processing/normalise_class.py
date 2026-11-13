"""
Class for storing data normalisation classes

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""
import pandas as pd
import os
import sys
import logging

from typing import List, Dict, Callable


from config.logger_config import LoggerMixin
from data_pre_processing.data_processing_enum_classes import NormalizationMethod

class DataNormaliser(LoggerMixin):
    """
    Handles data normalization operations using various scaling methods.

    This class provides functionality to normalize data using different scikit-learn
    scalers or custom normalization functions. It maintains the normalization
    parameters for potential later use in denormalization.
    """

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        # Store normalization parameters for each dataset
        self.normalization_params = {}
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies numeric columns in the DataFrame.

        :param df: Input DataFrame

        :return:List of column names with numeric data types
        """
        numeric_cols = df.select_dtypes(
            include=['float64', 'float32', 'int64', 'int32']
        ).columns.tolist()

        self.logger.INFO(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
        return numeric_cols

    def _validate_columns(self, df: pd.DataFrame, columns: List[str]):
        """
        Ensures all specified columns exist in the DataFrame.

        :param df: Input DataFrame
        :param columns: List of columns to validate

        :raise: ValueError If any specified columns are missing
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            self.logger.INFO(f"Columns not found in DataFrame: {missing_cols}")
            raise ValueError(f"Columns not found: {missing_cols}")

    def _create_scaler(self, method: str|Callable|NormalizationMethod):
        """
        Creates the appropriate scaler based on the specified method.

        :param method: Normalization method to use (string, enum, or callable)

        Returns:
            Initialized scaler object
        """
        if isinstance(method, NormalizationMethod):
            return method.scaler_class()
        elif isinstance(method, str):
            try:
                method = NormalizationMethod[method.upper()]
                return method.scaler_class()
            except KeyError:
                self.logger.error(f"Unknown normalization method: {method}")
                raise ValueError(
                    f"Unsupported method: {method}. "
                    f"Use one of: {[m.name.lower() for m in NormalizationMethod]}"
                )
        elif callable(method):
            return method()
        else:
            raise TypeError("method must be a string, NormalizationMethod, or callable")

    def _extract_scaler_params(self,
                               scaler,
                               columns: List[str]) -> Dict:
        """
        Extracts normalization parameters from the fitted scaler.

        :param scaler: Fitted scikit-learn scaler
        :param columns: List of column names that were normalized

        :return: Dictionary of normalization parameters
        """
        params = {
            'columns': columns,
            'method': type(scaler).__name__
        }

        # Extract relevant parameters based on scaler type
        if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
            params.update({
                'min_': scaler.min_,
                'scale_': scaler.scale_
            })
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            params.update({
                'mean_': scaler.mean_,
                'scale_': scaler.scale_
            })

        return params

    def normalise(self,
                  datafile: pd.DataFrame|dict[str, pd.DataFrame],
                  columns_to_normalise: List[str] = None,
                  method: str|Callable|NormalizationMethod = NormalizationMethod.MINMAX
                  ) -> pd.DataFrame|dict[str,pd.DataFrame]:
        """
        Normalizes specified columns using the chosen scaling method.

        This method handles both single DataFrames and dictionaries of DataFrames.
        It automatically detects numeric columns if none are specified and stores
        normalization parameters for potential denormalization.

        :param datafile: DataFrame or dictionary of DataFrames to normalize
        :param columns_to_normalize: Specific columns to normalize (optional)
        :param method: Normalization method to use

        Returns:
            Normalized DataFrame(s)

        Example:
            normalizer = DataNormalizer()

            # Normalize specific columns using standard scaling
            normalized_df = normalizer.normalize(
                df,
                columns_to_normalize=['speed', 'power'],
                method=NormalizationMethod.STANDARD
            )

            # Normalize all numeric columns using min-max scaling
            normalized_df = normalizer.normalize(df)
        """
        try:
            if isinstance(datafile, dict):
                return self._normalize_dict(datafile, columns_to_normalise, method)
            elif isinstance(datafile, pd.DataFrame):
                return self._normalize_dataframe(datafile, columns_to_normalise, method)
            else:
                raise TypeError("Input must be a DataFrame or dictionary of DataFrames")

        except Exception as e:
            self.logger.info(f"Normalization failed: {str(e)}")
            raise

    def _normalize_dataframe(self,
                             df: pd.DataFrame,
                             columns_to_normalize: List[str],
                             method: str| Callable| NormalizationMethod
                             ) -> pd.DataFrame:
        """
        Normalizes a single DataFrame.
        """
        df = df.copy()

        # Determine columns to normalize
        cols_to_norm = (columns_to_normalize if columns_to_normalize
                        else self._get_numeric_columns(df))

        self._validate_columns(df, cols_to_norm)

        # Create and fit scaler
        scaler = self._create_scaler(method)
        original_data = df[cols_to_norm].copy()
        df[cols_to_norm] = scaler.fit_transform(original_data)

        # Store normalization parameters
        self.normalization_params = self._extract_scaler_params(scaler, cols_to_norm)

        self.logger.info(
            f"Normalized {len(cols_to_norm)} columns using "
            f"{self.normalization_params['method']}"
        )

        return df

    def _normalize_dict(self,
                        datafiles: dict[str, pd.DataFrame],
                        columns_to_normalize: List[str],
                        method: str|Callable|NormalizationMethod
                        ) -> dict[str, pd.DataFrame]:
        """
        Normalizes a dictionary of DataFrames.
        """
        normalized_datafiles = {}

        for name, df in datafiles.items():
            self.logger.INFO(f"Normalizing dataset: {name}")
            normalized_datafiles[name] = self._normalize_dataframe(
                df, columns_to_normalize, method
            )
            # Store parameters with dataset name
            self.normalization_params[name] = self.normalization_params.copy()

        return normalized_datafiles
