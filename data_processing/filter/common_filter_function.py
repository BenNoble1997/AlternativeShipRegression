import pandas as pd
import logging
from typing import List
from data_processing.data_processing_enum_classes import ComparisonOperator

def common_filter_function(datafile: pd.DataFrame,
                           column_name: str | List[str],
                           value: float,
                           logger: logging.Logger,
                           operator: ComparisonOperator | str = ComparisonOperator.EQUAL,
                           require_all: bool = False,
                           negate: bool = False, ) -> pd.DataFrame:
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
    # Convert string operator to enum if necessary (for backward compatibility)
    if isinstance(operator, str):
        try:
            operator = ComparisonOperator.from_symbol(operator)
        except ValueError as e:
            logger.error(f"Invalid operator provided: {operator}")
            raise

    filtered_data = datafile.copy()
    initial_count = len(filtered_data)

    if initial_count == 0:
        logger.warning(f"No rows to filter: {datafile}")
        return datafile

    try:
        # Convert single column to list for consistent processing
        columns = [column_name] if isinstance(column_name, str) else column_name

        # Validate all columns exist before processing
        for col in columns:
            if col not in filtered_data.columns:
                logger.error(f"Column not found: {col}")
                raise KeyError(f"Column '{col}' not found in DataFrame")

        if len(columns) == 1:
            # Single column processing - simpler logic
            condition = operator.func(filtered_data[columns[0]], value)
            if negate:
                condition = ~condition
            filtered_data = filtered_data[condition]

            # Log the filtering operation
            rows_removed = initial_count - len(filtered_data)
            logger.info(
                f"Filter: {columns[0]} {'NOT ' if negate else ''}{operator.description} {value} "
                f"(removed {rows_removed:,} rows, "
                f"{rows_removed / initial_count:.1%} of data)"
            )

        else:
            # Multiple column processing - fixed logic for proper AND/OR behavior
            if require_all:
                # AND logic: all conditions must be true
                condition = pd.Series(True, index=filtered_data.index)
                for col in columns:
                    condition = condition & operator.func(filtered_data[col], value)
            else:
                # OR logic: any condition can be true
                condition = pd.Series(False, index=filtered_data.index)
                for col in columns:
                    condition = condition | operator.func(filtered_data[col], value)

            # Apply negation if requested
            if negate:
                condition = ~condition

            filtered_data = filtered_data[condition]

            # Log the combined operation
            rows_removed = initial_count - len(filtered_data)
            logic_type = "ALL" if require_all else "ANY"
            logger.info(
                f"Filter: {'NOT ' if negate else ''}{logic_type} columns {operator.description} {value} "
                f"(columns: {', '.join(columns)}) "
                f"(removed {rows_removed:,} rows, "
                f"{rows_removed / initial_count:.1%} of data)"
            )

        logger.info(f"Filter applied. {len(filtered_data)} points remaining")
        return filtered_data

    except Exception as e:
        logger.error(f"Filter operation failed: {str(e)}")
        raise