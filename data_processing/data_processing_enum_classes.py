"""
This module stores Enum's that are used by DT4GS data processing classes

Author: Ben Noble
Date: 31-05-2025
Version 1.0.0
"""

import numpy as np
from enum import Enum
import pandas as pd
import os
import sys
from datetime import datetime, time
from typing import Optional, Callable, Union

from operator import lt, le, gt, ge, eq, ne
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
os.chdir(r"/")


sys.path.append(r"/")

# from config.logger_config import LoggerMixin


class ColumnOperation(Enum):
    """Enum defining available column operations."""
    DIFFERENCE = 'difference'
    SUM = 'sum'
    PRODUCT = 'product'

    @staticmethod
    def get_operation_function(operation):
        """Returns the corresponding function for the given operation."""
        operations = {
            ColumnOperation.DIFFERENCE: lambda x: x[0] - x[1],
            ColumnOperation.SUM: lambda x: np.sum(x, axis=0),
            ColumnOperation.PRODUCT: lambda x: np.prod(x, axis=0)
        }

        if isinstance(operation, ColumnOperation):
            return operations.get(operation, None)
        else:
            for enum_op, func in operations.items():
                if enum_op.value == operation:
                    return func
        return None


class ComparisonOperator(Enum):
    """
    Enumeration of supported comparison operators for filtering operations.

    Each operator is paired with its corresponding Python operator function
    and a human-readable description for logging and error messages.
    """
    LESS_THAN = ('<', lt, 'less than')
    LESS_EQUAL = ('<=', le, 'less than or equal to')
    GREATER_THAN = ('>', gt, 'greater than')
    GREATER_EQUAL = ('>=', ge, 'greater than or equal to')
    EQUAL = ('==', eq, 'equal to')
    NOT_EQUAL = ('!=', ne, 'not equal to')

    # CONTAINS = ('contains', lambda a, b: str(a).find(str(b)) != -1, 'contains')
    CONTAINS = ('contains',
                lambda a, b: (
                    # If b is a list, check if any item in the list is contained
                    a.astype(str).str.contains('|'.join(map(str, b)), case=False, na=False, regex=True)
                    if isinstance(b, list) and hasattr(a, 'str')
                    # Single value with pandas Series
                    else a.astype(str).str.contains(str(b), case=False, na=False)
                    if hasattr(a, 'str')
                    # Fallback for non-Series
                    else str(b) in str(a)
                ),
                'contains')
    NOT_CONTAINS = ('not_contains', lambda a, b: str(a).find(str(b)) == -1, 'does not contain')
    STARTS_WITH = ('startswith', lambda a, b: str(a).startswith(str(b)), 'starts with')
    ENDS_WITH = ('endswith', lambda a, b: str(a).endswith(str(b)), 'ends with')

    def __init__(self,
                 symbol: str, func: Callable,
                 description: str):
        """
        Initialize a comparison operator with its symbol, function, and description.

        Args:
            symbol: The mathematical symbol representing the operator
            func: The Python function implementing the comparison
            description: A human-readable description of the operation
        """
        self.symbol = symbol
        self.func = func
        self.description = description

    @classmethod
    def from_symbol(cls, symbol: str) -> 'ComparisonOperator':
        """
        Get the ComparisonOperator enum member corresponding to a symbol.

        This allows backward compatibility with code that might still use
        string operators.

        :param symbol: The operator symbol (e.g., '<', '>=', '==')

        Returns:
            The corresponding ComparisonOperator enum member

        Raises:
            ValueError: If the symbol doesn't match any operator
        """
        for operator in cls:
            if operator.symbol == symbol:
                return operator
        raise ValueError(
            f"Invalid operator symbol: '{symbol}'. "
            f"Valid operators are: {[op.symbol for op in cls]}"
        )


class AggregationFunction(Enum):
    """
    Enumeration of supported aggregation functions with their implementation details.

    Each enum member contains:
    - name: The string name used in pandas aggregation
    - description: Human-readable description for display/logging
    - function: The actual function to use (if custom implementation needed)
    """
    def weighted_average(series, weights):
        if np.sum(weights) == 0:
            return 0  # or any other default value you prefer
        else:
            return np.average(series, weights=weights)

    SUM = ('sum', 'Sum of values', None)
    MEAN = ('mean', 'Arithmetic mean', None)
    WEIGHTED_MEAN = ('weighted_mean', 'Weighted average', weighted_average)
    COUNT = ('count', 'Count of values', None)
    MIN = ('min', 'Minimum value', None)
    MAX = ('max', 'Maximum value', None)
    STD = ('std', 'Standard deviation', None)
    VAR = ('var', 'Variance', None)
    MEDIAN = ('median', 'Median value', None)
    FIRST = ('first', 'First value', None)
    LAST = ('last', 'Last value', None)

    def __init__(self, string: str,
                 description: str,
                 function: Optional[Callable] = None):
        self.string = string
        self.description = description
        self.function = function

    @staticmethod
    def create_weighted_mean(weight_column: str):
        """Creates a weighted mean function that will be called during groupby aggregation"""

        def weighted_mean(group, parent_df):  # pandas groupby passes the group directly
            # series = group
            # df = group.groupby_parent_df  # we'll store the parent df during aggregation
            weights = parent_df.loc[group.index, weight_column]
            return AggregationFunction.weighted_average(group, weights)

        return weighted_mean


class TransformFunction(Enum):
    """
    Enumeration of supported transform functions with their implementation details.

    Each enum member contains:
    - name: The string identifier
    - description: A human-readable description
    - function: The function to apply
    """

    @staticmethod
    def transform_samplingtime(time_value):
        """Converts a datetime or timedelta object to the total number of minutes represented."""
        # If time_value is a datetime object, calculate minutes
        if isinstance(time_value, datetime):
            return time_value.hour * 60 + time_value.minute

            # If time_value is a time object, calculate minutes

        elif isinstance(time_value, time):

            return time_value.hour * 60 + time_value.minute + time_value.second / 60

        # If time_value is a timedelta object, extract total minutes
        elif isinstance(time_value, pd.Timedelta):
            return time_value.total_seconds() / 60  # Convert total seconds to minutes

        else:
            raise TypeError("Expected a datetime or timedelta object")

    TIME_TO_MINUTES = ('time_to_minutes', 'Time to minutes', transform_samplingtime)  # Directly reference function

    def apply(self, series):
        """Applies the transformation function to a pandas Series."""
        return series.apply(self.value[2])  # Access the function reference


class NormalizationMethod(Enum):
    """
    Enumeration of supported normalization methods with their corresponding scalers.

    Each enum member contains:
    - name: String identifier for the method
    - scaler_class: The scikit-learn scaler class to use
    - description: Human-readable description of what the method does
    """
    MINMAX = ('minmax', MinMaxScaler, 'Scales data to a fixed range [0, 1]')
    STANDARD = ('standard', StandardScaler, 'Standardizes data to zero mean and unit variance')
    ROBUST = ('robust', RobustScaler, 'Scales data using statistics robust to outliers')

    def __init__(self, string: str, scaler_class: type, description: str):
        self.string = string
        self.scaler_class = scaler_class
        self.description = description

class SpeedUnit(Enum):
    # Each enum value is a tuple: (symbol, conversion_factor_to_mps)
    METERS_PER_SECOND = ("m/s", 1.0)
    KILOMETERS_PER_HOUR = ("km/h", 1 / 3.6)  # 1 km/h = 1/3.6 m/s
    MILES_PER_HOUR = ("mph", 0.44704)  # 1 mph = 0.44704 m/s
    FEET_PER_SECOND = ("ft/s", 0.3048)  # 1 ft/s = 0.3048 m/s
    KNOTS = ("kn", 0.514444)  # 1 knot = 0.514444 m/s

    def __init__(self, symbol: str, to_mps_factor: float):
        self.symbol = symbol
        self.to_mps_factor = to_mps_factor

    @classmethod
    def from_string(cls, unit_str: str):
        """Convert string to SpeedUnit enum, with flexible matching."""
        unit_str = unit_str.lower().strip()

        # Direct symbol matching
        for unit in cls:
            if unit.symbol.lower() == unit_str:
                return unit

        # Common aliases
        aliases = {
            "mps": cls.METERS_PER_SECOND,
            "m/sec": cls.METERS_PER_SECOND,
            "meters/second": cls.METERS_PER_SECOND,
            "kmh": cls.KILOMETERS_PER_HOUR,
            "kph": cls.KILOMETERS_PER_HOUR,
            "km/hour": cls.KILOMETERS_PER_HOUR,
            "kilometers/hour": cls.KILOMETERS_PER_HOUR,
            "mph": cls.MILES_PER_HOUR,
            "miles/hour": cls.MILES_PER_HOUR,
            "fps": cls.FEET_PER_SECOND,
            "ft/sec": cls.FEET_PER_SECOND,
            "feet/second": cls.FEET_PER_SECOND,
            "knots": cls.KNOTS,
            "nautical": cls.KNOTS,
        }

        if unit_str in aliases:
            return aliases[unit_str]

        valid_units = [unit.symbol for unit in cls]
        raise ValueError(f"Unknown speed unit: '{unit_str}'. "
                         f"Valid units: {valid_units}")

    def to_mps(self, value: float) -> float:
        """Convert a value in this unit to meters per second."""
        return value * self.to_mps_factor

    def from_mps(self, mps_value: float) -> float:
        """Convert a value from meters per second to this unit."""
        return mps_value / self.to_mps_factor


class DistanceUnit(Enum):
    # Each enum value is a tuple: (symbol, conversion_factor_to_meters)
    METERS = ("m", 1.0)
    KILOMETERS = ("km", 1000.0)
    CENTIMETERS = ("cm", 0.01)
    MILLIMETERS = ("mm", 0.001)
    INCHES = ("in", 0.0254)
    FEET = ("ft", 0.3048)
    YARDS = ("yd", 0.9144)
    MILES = ("mi", 1609.344)
    NAUTICAL_MILES = ("nmi", 1852.0)

    def __init__(self, symbol: str, to_meters_factor: float):
        self.symbol = symbol
        self.to_meters_factor = to_meters_factor

    def __str__(self) -> str:
        return self.symbol

    @classmethod
    def from_string(cls, unit_str: str):
        """Convert string to DistanceUnit enum, with flexible matching."""
        unit_str = unit_str.lower().strip()

        # Direct symbol matching
        for unit in cls:
            if unit.symbol.lower() == unit_str:
                return unit

        # Common aliases
        aliases = {
            "meter": cls.METERS,
            "metres": cls.METERS,
            "kilometer": cls.KILOMETERS,
            "kilometres": cls.KILOMETERS,
            "km": cls.KILOMETERS,
            "centimeter": cls.CENTIMETERS,
            "centimetre": cls.CENTIMETERS,
            "cm": cls.CENTIMETERS,
            "millimeter": cls.MILLIMETERS,
            "millimetre": cls.MILLIMETERS,
            "mm": cls.MILLIMETERS,
            "inch": cls.INCHES,
            "inches": cls.INCHES,
            '"': cls.INCHES,
            "foot": cls.FEET,
            "feet": cls.FEET,
            "'": cls.FEET,
            "yard": cls.YARDS,
            "yards": cls.YARDS,
            "mile": cls.MILES,
            "miles": cls.MILES,
            "nautical": cls.NAUTICAL_MILES,
            "nm": cls.NAUTICAL_MILES,
        }

        if unit_str in aliases:
            return aliases[unit_str]

        valid_units = [unit.symbol for unit in cls]
        raise ValueError(f"Unknown distance unit: '{unit_str}'. "
                         f"Valid units: {valid_units}")

    def to_meters(self, value: float) -> float:
        """Convert a value in this unit to meters."""
        return value * self.to_meters_factor

    def from_meters(self, meters_value: float) -> float:
        """Convert a value from meters to this unit."""
        return meters_value / self.to_meters_factor

    @classmethod
    def detect_from_column_name(cls, column_name: str, default=None):
        """
        Try to detect distance unit from column name.

        :param column_name: Column name to analyze
        :param default: Default unit if detection fails (defaults to KILOMETERS)
        :return: Detected DistanceUnit
        """
        if default is None:
            default = cls.KILOMETERS

        column_lower = column_name.lower()

        # Detection patterns (order matters - more specific first)
        detection_patterns = [
            (['nmi', 'nautical'], cls.NAUTICAL_MILES),
            (['km', 'kilometer', 'kilometre'], cls.KILOMETERS),
            (['mile', '_mi', 'miles'], cls.MILES),
            (['ft', 'feet', 'foot'], cls.FEET),
            (['yard', 'yd'], cls.YARDS),
            (['inch', 'in', '"'], cls.INCHES),
            (['cm', 'centimeter', 'centimetre'], cls.CENTIMETERS),
            (['mm', 'millimeter', 'millimetre'], cls.MILLIMETERS),
            (['meter', 'metre', '_m'], cls.METERS),  # More general, so check last
        ]

        for patterns, unit in detection_patterns:
            if any(pattern in column_lower for pattern in patterns):
                return unit

        return default



class PowerUnit(Enum):
    # Each enum value is a tuple: (symbol, conversion_factor_to_watts)
    WATTS = ("W", 1.0)
    KILOWATTS = ("kW", 1000.0)
    MEGAWATTS = ("MW", 1_000_000.0)
    HORSEPOWER = ("hp", 745.7)  # Mechanical horsepower
    METRIC_HORSEPOWER = ("PS", 735.5)  # Metric horsepower (Pferdestärke)
    BTU_PER_HOUR = ("BTU/h", 0.293071)
    FOOT_POUNDS_PER_SECOND = ("ft⋅lbf/s", 1.35582)

    def __init__(self, symbol: str, to_watts_factor: float):
        self.symbol = symbol
        self.to_watts_factor = to_watts_factor

    def __str__(self) -> str:
        return self.symbol

    @classmethod
    def from_string(cls, unit_str: str):
        """Convert string to PowerUnit enum, with flexible matching."""
        unit_str = unit_str.lower().strip()

        # Direct symbol matching
        for unit in cls:
            if unit.symbol.lower() == unit_str:
                return unit

        # Common aliases
        aliases = {
            "watt": cls.WATTS,
            "watts": cls.WATTS,
            "kilowatt": cls.KILOWATTS,
            "kilowatts": cls.KILOWATTS,
            "kw": cls.KILOWATTS,
            "megawatt": cls.MEGAWATTS,
            "megawatts": cls.MEGAWATTS,
            "mw": cls.MEGAWATTS,
            "horsepower": cls.HORSEPOWER,
            "horse power": cls.HORSEPOWER,
            "bhp": cls.HORSEPOWER,  # Brake horsepower
            "metric horsepower": cls.METRIC_HORSEPOWER,
            "pferdestärke": cls.METRIC_HORSEPOWER,
            "btu/hour": cls.BTU_PER_HOUR,
            "btu/hr": cls.BTU_PER_HOUR,
            "btuh": cls.BTU_PER_HOUR,
            "ft-lbf/s": cls.FOOT_POUNDS_PER_SECOND,
            "ft*lbf/s": cls.FOOT_POUNDS_PER_SECOND,
        }

        if unit_str in aliases:
            return aliases[unit_str]

        valid_units = [unit.symbol for unit in cls]
        raise ValueError(f"Unknown power unit: '{unit_str}'. "
                         f"Valid units: {valid_units}")

    def to_watts(self, value: float) -> float:
        """Convert a value in this unit to watts."""
        return value * self.to_watts_factor

    def from_watts(self, watts_value: float) -> float:
        """Convert a value from watts to this unit."""
        return watts_value / self.to_watts_factor


class TimeUnit(Enum):
    # Each enum value is a tuple: (symbol, conversion_factor_to_seconds)
    NANOSECONDS = ("ns", 1e-9)
    MICROSECONDS = ("μs", 1e-6)
    MILLISECONDS = ("ms", 1e-3)
    SECONDS = ("s", 1.0)
    MINUTES = ("min", 60.0)
    HOURS = ("h", 3600.0)
    DAYS = ("d", 86400.0)
    WEEKS = ("w", 604800.0)

    def __init__(self, symbol: str, to_seconds_factor: float):
        self.symbol = symbol
        self.to_seconds_factor = to_seconds_factor

    def __str__(self) -> str:
        return self.symbol

    @classmethod
    def from_string(cls, unit_str: str):
        """Convert string to TimeUnit enum, with flexible matching."""
        unit_str = unit_str.lower().strip()

        # Direct symbol matching
        for unit in cls:
            if unit.symbol.lower() == unit_str:
                return unit

        # Common aliases
        aliases = {
            "nanosecond": cls.NANOSECONDS,
            "nanoseconds": cls.NANOSECONDS,
            "nano": cls.NANOSECONDS,
            "microsecond": cls.MICROSECONDS,
            "microseconds": cls.MICROSECONDS,
            "micro": cls.MICROSECONDS,
            "us": cls.MICROSECONDS,
            "millisecond": cls.MILLISECONDS,
            "milliseconds": cls.MILLISECONDS,
            "milli": cls.MILLISECONDS,
            "second": cls.SECONDS,
            "seconds": cls.SECONDS,
            "sec": cls.SECONDS,
            "minute": cls.MINUTES,
            "minutes": cls.MINUTES,
            "mins": cls.MINUTES,
            "hour": cls.HOURS,
            "hours": cls.HOURS,
            "hrs": cls.HOURS,
            "hr": cls.HOURS,
            "day": cls.DAYS,
            "days": cls.DAYS,
            "week": cls.WEEKS,
            "weeks": cls.WEEKS,
            "wks": cls.WEEKS,
        }

        if unit_str in aliases:
            return aliases[unit_str]

        valid_units = [unit.symbol for unit in cls]
        raise ValueError(f"Unknown time unit: '{unit_str}'. "
                         f"Valid units: {valid_units}")

    def to_seconds(self, value: float) -> float:
        """Convert a value in this unit to seconds."""
        return value * self.to_seconds_factor

    def from_seconds(self, seconds_value: float) -> float:
        """Convert a value from seconds to this unit."""
        return seconds_value / self.to_seconds_factor

    def from_timedelta(self, td: pd.Timedelta) -> float:
        """Convert a pandas Timedelta to this unit."""
        return self.from_seconds(td.total_seconds())

    @classmethod
    def detect_from_column_name(cls, column_name: str, default=None):
        """
        Try to detect time unit from column name.

        :param column_name: Column name to analyze
        :param default: Default unit if detection fails (defaults to MINUTES)
        :return: Detected TimeUnit
        """
        if default is None:
            default = cls.MINUTES

        column_lower = column_name.lower()

        # Detection patterns (order matters - more specific first)
        detection_patterns = [
            (['nanosecond', 'nanoseconds', 'nano', 'ns'], cls.NANOSECONDS),
            (['microsecond', 'microseconds', 'micro', 'μs', 'us'], cls.MICROSECONDS),
            (['millisecond', 'milliseconds', 'milli', 'ms'], cls.MILLISECONDS),
            (['minute', 'minutes', 'mins', 'min'], cls.MINUTES),
            (['hour', 'hours', 'hrs', 'hr', '_h'], cls.HOURS),
            (['second', 'seconds', 'sec', '_s'], cls.SECONDS),  # Check after hours to avoid 'hrs'
            (['day', 'days', '_d'], cls.DAYS),
            (['week', 'weeks', 'wks', '_w'], cls.WEEKS),
        ]

        for patterns, unit in detection_patterns:
            if any(pattern in column_lower for pattern in patterns):
                return unit

        return default


def convert_time(value: float, from_unit: Union[TimeUnit, str],
                 to_unit: Union[TimeUnit, str]) -> float:
    """Convert time from one unit to another."""
    if isinstance(from_unit, str):
        from_unit = TimeUnit.from_string(from_unit)
    if isinstance(to_unit, str):
        to_unit = TimeUnit.from_string(to_unit)

    seconds_value = from_unit.to_seconds(value)
    return to_unit.from_seconds(seconds_value)

def _detect_distance_unit_from_string(self, column_name: str) -> DistanceUnit:
    """Try to detect distance unit from column name."""
    column_lower = column_name.lower()

    if 'km' in column_lower or 'kilometer' in column_lower:
        return DistanceUnit.KILOMETERS
    elif '_m' in column_lower or 'meter' in column_lower:
        return DistanceUnit.METERS
    elif 'mile' in column_lower or '_mi' in column_lower:
        return DistanceUnit.MILES
    elif 'nmi' in column_lower or 'nautical' in column_lower:
        return DistanceUnit.NAUTICAL_MILES
    elif 'ft' in column_lower or 'feet' in column_lower:
        return DistanceUnit.FEET
    else:
        # Default assumption
        self.logger.warning(f"Could not detect unit from column name '{column_name}', assuming kilometers")
        return DistanceUnit.KILOMETERS

def _detect_time_unit_from_string(self, column_name: str) -> TimeUnit:
    """Try to detect time unit from column name."""
    column_lower = column_name.lower()

    if 'minute' in column_lower or 'min' in column_lower:
        return TimeUnit.MINUTES
    elif 'second' in column_lower or 'sec' in column_lower:
        return TimeUnit.SECONDS
    elif 'hour' in column_lower or 'hr' in column_lower:
        return TimeUnit.HOURS
    else:
        # Default assumption based on your original code
        self.logger.warning(f"Could not detect unit from column name '{column_name}', assuming minutes")
        return TimeUnit.MINUTES
