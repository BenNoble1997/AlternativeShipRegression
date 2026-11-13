# Data Processing Package

A comprehensive Python package for processing and analyzing operational data, particularly designed for maritime and fleet operations. This package provides a complete workflow for importing, cleaning, filtering, transforming, aggregating, and exporting operational data.

## Overview

This package consists of several interconnected classes that handle different aspects of data processing:

- **Data Import: Reading from various file formats with caching capabilities
- **Data Filtering**: Advanced filtering operations with multiple criteria
- **Data Transformation**: Time, distance, and speed calculations with unit conversions
- **Data Aggregation**: Time-based aggregations with custom functions
- **Data Manipulation**: Column operations and duplicate handling
- **Data Normalization**: Statistical normalization using various scaling methods
- **Workflow Management**: Orchestrated processing pipelines

## Installation

```python
# Import the main workflow class
from Operational_profile.data_processing.workflow_class import DataProcessingWorkflow

# Or import individual classes as needed
from Operational_profile.data_processing.file_import_class import DataImporter
from Operational_profile.data_processing.filter.filter_class import DataFilter
# ... etc
```

## Quick Start

```python
# Create a workflow instance
workflow = DataProcessingWorkflow(verbose=True, debug=False)

# Import data from a directory
data_dict = workflow.importer.import_directory(
    directory_path='data/vessel_logs',
    file_types=['.xlsx', '.csv']
)

# Filter out negative values
filtered_data = workflow.filter.filter_out_negative(
    data_dict, 
    column_name=['FUEL_CONSUMPTION', 'SPEED']
)

# Aggregate to hourly data
hourly_data = workflow.aggregator.group_and_aggregate(
    filtered_data,
    group_key='TIMESTAMP',
    aggregation_dict={'FUEL_CONSUMPTION': 'mean', 'SPEED': 'mean'},
    freq='H'
)
```

## Core Classes

### 1. DataImporter - File Import and Processing

Handles reading data from various file formats with intelligent caching and preprocessing capabilities.

**Key Features:**
- Supports Excel (.xlsx, .xls), CSV, JSON, and Parquet files
- Automatic caching with pickle files for faster subsequent loads
- Column selection, renaming, and cleaning during import
- Batch processing of directories

**Example Usage:**
```python
from Operational_profile.data_processing.file_import_class import DataImporter

# Create importer instance
importer = DataImporter(verbose=True, cache_dir='cache/')

# Import a single file with preprocessing
df = importer.import_file(
    file_path='data/vessel_log.xlsx',
    sheet_name=0,
    columns_to_keep=['TIMESTAMP', 'SPEED', 'FUEL_CONSUMPTION'],
    columns_to_rename={'OLD_NAME': 'NEW_NAME'},
    uppercase=True,  # Convert column names to uppercase
    replace_spaces='_',  # Replace spaces with underscores
    use_cache=True
)

# Import entire directory
data_dict = importer.import_directory(
    directory_path='data/logs/',
    file_types=['.xlsx', '.csv'],
    columns_to_keep=['TIMESTAMP', 'SPEED', 'FUEL_CONSUMPTION', 'POSITION'],
    use_cache=True
)
```

### 2. DataFilter - Advanced Data Filtering

Provides sophisticated filtering capabilities with support for multiple conditions and logical operations.

**Key Features:**
- Filter by value with various comparison operators
- Remove negative, zero, or null values
- Differential filtering based on rate of change
- Multiple filter application with AND/OR logic
- Works with single DataFrames or dictionaries of DataFrames

**Example Usage:**

```python
from Operational_profile.data_processing.filter.filter_class import DataFilter
from Operational_profile.data_processing.data_processing_enum_classes import ComparisonOperator

filter_engine = DataFilter(verbose=True)

# Remove negative fuel consumption values
clean_data = filter_engine.filter_out_negative(
    data_dict,
    column_name='FUEL_CONSUMPTION'
)

# Filter by speed range
speed_filtered = filter_engine.filter_by_value(
    clean_data,
    value=5.0,
    column_name='SPEED',
    operator=ComparisonOperator.GREATER_THAN
)

# Create and apply multiple filters
filters = [
    filter_engine.build_filter('SPEED', 20, '>', require_all=False),
    filter_engine.build_filter(['FUEL_CONSUMPTION', 'ENGINE_POWER'], 0, '>=', require_all=True)
]

final_data = filter_engine.apply_multiple_filters(speed_filtered, filters)

# Differential filtering to remove unrealistic speed changes
smooth_data = filter_engine.filter_by_differential(
    final_data,
    column_name='SPEED',
    max_diff=5.0  # Maximum 5 unit change between consecutive readings
)
```

### 3. DataTransformer - Data Transformation and Feature Engineering

Handles complex data transformations including time, distance, and speed calculations with automatic unit conversions.

**Key Features:**
- Time transformations: datetime conversion, sample time calculation
- Distance calculations using Haversine formula
- Speed calculations from distance and time
- Coordinate conversions (degrees/minutes to decimal)
- Custom column operations and transformations
- Unit conversions between different measurement systems

**Example Usage:**
```python
from Operational_profile.data_processing.transform.transform_class import DataTransformer
from Operational_profile.data_processing.data_processing_enum_classes import SpeedUnit, DistanceUnit

transformer = DataTransformer(verbose=True)

# Convert timestamps to datetime
data_with_time = transformer.time_transformation.convert_to_datetime(
    data_dict,
    columns='TIMESTAMP',
    timestamp_precision='S'  # Round to seconds
)

# Convert position coordinates to decimal format
decimal_coords = transformer.distance_transformation.convert_to_decimal_coordinates(
    data_with_time,
    lat_degrees_col='LAT_DEG',
    lat_minutes_col='LAT_MIN',
    lat_letter_col='LAT_DIR',
    long_degrees_col='LONG_DEG',
    long_minutes_col='LONG_MIN',
    long_letter_col='LONG_DIR'
)

# Calculate distances between consecutive positions
with_distance = transformer.distance_transformation.add_distance_column(
    decimal_coords,
    lat_col='LATITUDE',
    lon_col='LONGITUDE',
    time_col='TIMESTAMP',
    distance_unit=DistanceUnit.NAUTICAL_MILES,
    include_cumulative=True
)

# Calculate speed from distance and time
with_speed = transformer.speed_transformation.calculate_speed_column(
    with_distance,
    distance_col='DISTANCE_NMI',
    time_col='TIMESTAMP',
    speed_unit=SpeedUnit.KNOTS
)

# Create custom calculated columns
from Operational_profile.data_processing.data_processing_enum_classes import ColumnOperation

fuel_efficiency = transformer.calculate_column_operation(
    with_speed,
    columns=['DISTANCE_NMI', 'FUEL_CONSUMPTION'],
    new_column_name='FUEL_EFFICIENCY',
    operation=ColumnOperation.DIFFERENCE
)
```

### 4. DataAggregator - Time-based Data Aggregation

Performs time-based aggregations with support for custom aggregation functions and weighted averages.

**Key Features:**
- Time-based grouping (hourly, daily, monthly, custom frequencies)
- Standard aggregations (mean, sum, min, max, etc.)
- Custom aggregation functions
- Weighted averages
- Handles single DataFrames or dictionaries

**Example Usage:**
```python
from Operational_profile.data_processing.aggregate_class import DataAggregator

aggregator = DataAggregator(verbose=True)

# Simple hourly aggregation
hourly_summary = aggregator.group_and_aggregate(
    datafile=fuel_efficiency,
    group_key='TIMESTAMP',
    aggregation_dict={
        'SPEED': 'mean',
        'FUEL_CONSUMPTION': 'sum',
        'ENGINE_POWER': 'mean',
        'DISTANCE_NMI': 'sum'
    },
    freq='H'  # Hourly frequency
)

# Daily aggregation with weighted averages
weighted_daily = aggregator.group_and_aggregate(
    datafile=fuel_efficiency,
    group_key='TIMESTAMP',
    aggregation_dict={
        'FUEL_CONSUMPTION': aggregator.create_weighted_aggregation(
            'FUEL_CONSUMPTION', 'ENGINE_POWER'
        ),
        'SPEED': 'mean',
        'DISTANCE_NMI': 'sum'
    },
    freq='D'  # Daily frequency
)

# Calculate fleet-wide weighted averages
fleet_averages = aggregator.apply_weighted_mean(
    datafile=fuel_efficiency,
    value_column='FUEL_CONSUMPTION',
    weight_column='ENGINE_POWER',
    group_key='TIMESTAMP',
    freq='D'
)
```

### 5. DataManipulator - Data Manipulation Operations

Handles column operations and data manipulation tasks.

**Key Features:**
- Column calculations (difference, sum, product)
- Duplicate removal with flexible options
- Data reshaping operations

**Example Usage:**
```python
from Operational_profile.data_processing.manipulator_class import DataManipulator
from Operational_profile.data_processing.data_processing_enum_classes import ColumnOperation

manipulator = DataManipulator(verbose=True)

# Calculate power difference between main engine and auxiliary
power_diff = manipulator.calculate_column_operation(
    df=hourly_summary,
    columns=['MAIN_ENGINE_POWER', 'AUX_ENGINE_POWER'],
    new_column_name='POWER_DIFFERENCE',
    operation=ColumnOperation.DIFFERENCE
)

# Remove duplicates based on timestamp
clean_data = manipulator.drop_duplicates(
    df=power_diff,
    subset='TIMESTAMP',
    keep='first'
)
```

### 6. DataNormaliser - Statistical Normalization

Provides statistical normalization using various scaling methods.

**Key Features:**
- Multiple normalization methods (Min-Max, Standard, Robust)
- Automatic numeric column detection
- Preservation of normalization parameters
- Support for denormalization

**Example Usage:**
```python
from Operational_profile.data_processing.normalise_class import DataNormaliser
from Operational_profile.data_processing.data_processing_enum_classes import NormalizationMethod

normalizer = DataNormaliser(verbose=True)

# Normalize specific columns using Min-Max scaling
normalized_data = normalizer.normalise(
    datafile=clean_data,
    columns_to_normalise=['SPEED', 'FUEL_CONSUMPTION', 'ENGINE_POWER'],
    method=NormalizationMethod.MINMAX
)

# Normalize all numeric columns using Standard scaling
standardized_data = normalizer.normalise(
    datafile=clean_data,
    method=NormalizationMethod.STANDARD
)
```

## Advanced Usage Examples

### Complete Processing Workflow

```python
# Initialize the workflow
workflow = DataProcessingWorkflow(verbose=True, debug=False)

# 1. Import data from multiple Excel files
raw_data = workflow.importer.import_directory(
    directory_path='vessel_data/',
    file_types=['.xlsx'],
    columns_to_keep=['TIMESTAMP', 'LAT_DEG', 'LAT_MIN', 'LAT_DIR', 
                    'LONG_DEG', 'LONG_MIN', 'LONG_DIR', 'SPEED_LOG', 
                    'FUEL_RATE', 'ENGINE_POWER'],
    uppercase=True,
    replace_spaces='_'
)

# 2. Data cleaning and filtering
# Remove rows with missing critical data
clean_data = workflow.filter.filter_out_zero(
    raw_data, 
    ['SPEED_LOG', 'FUEL_RATE'], 
    filter_zeros=True, 
    filter_nulls=True
)

# Remove unrealistic values
realistic_data = workflow.filter.filter_by_value(
    clean_data,
    value=50,  # Remove speeds > 50 knots (unrealistic for most vessels)
    column_name='SPEED_LOG',
    operator='<='
)

# 3. Coordinate and distance calculations
# Convert coordinates to decimal
with_coords = workflow.transformer.distance_transformation.convert_to_decimal_coordinates(
    realistic_data
)

# Calculate distances and speeds
with_distance = workflow.transformer.distance_transformation.add_distance_column(
    with_coords,
    distance_unit='nautical_miles',
    include_cumulative=True
)

# 4. Time-based aggregations
# Create hourly operational summaries
hourly_ops = workflow.aggregator.group_and_aggregate(
    with_distance,
    group_key='TIMESTAMP',
    aggregation_dict={
        'SPEED_LOG': 'mean',
        'FUEL_RATE': 'mean',
        'ENGINE_POWER': 'mean',
        'DISTANCE_NMI': 'sum',
        'CUMULATIVE_DISTANCE_NAUTICAL_MILES': 'last'
    },
    freq='H'
)

# 5. Feature engineering
# Calculate fuel efficiency
fuel_efficient = workflow.transformer.calculate_column_operation(
    hourly_ops,
    columns=['DISTANCE_NMI', 'FUEL_RATE'],
    new_column_name='FUEL_EFFICIENCY_NMI_PER_UNIT',
    operation='difference'  # Distance per unit fuel
)

# 6. Export results
workflow.exporter.export_to_excel(
    fuel_efficient,
    'processed_vessel_data.xlsx'
)
```

### Batch Processing Multiple Vessels

```python
# Process multiple vessel datasets
vessel_data = {}
vessel_names = ['vessel_001', 'vessel_002', 'vessel_003']

for vessel in vessel_names:
    # Import vessel-specific data
    vessel_data[vessel] = workflow.importer.import_directory(
        directory_path=f'data/{vessel}/',
        file_types=['.xlsx'],
        cache_processed_data=True
    )

# Apply consistent filtering to all vessels
filtered_fleet = workflow.filter.filter_out_negative(
    vessel_data,
    column_name=['SPEED', 'FUEL_CONSUMPTION']
)

# Create fleet-wide aggregations
fleet_summary = workflow.aggregator.group_and_aggregate(
    filtered_fleet,
    group_key='TIMESTAMP',
    aggregation_dict={
        'SPEED': 'mean',
        'FUEL_CONSUMPTION': 'sum',
        'DISTANCE': 'sum'
    },
    freq='D'  # Daily summaries
)

# Export individual vessel data and fleet summary
for vessel_name, data in fleet_summary.items():
    workflow.exporter.export_to_excel(
        data,
        f'fleet_analysis_{vessel_name}.xlsx'
    )
```

## Configuration and Customization

### Logging Configuration
All classes support verbose and debug logging modes:
```python
# Enable detailed logging
processor = DataProcessor(verbose=True, debug=True)

# Quiet mode
processor = DataProcessor(verbose=False, debug=False)
```

### Caching Configuration
```python
# Configure caching directory
importer = DataImporter(cache_dir='custom_cache_directory/')

# Disable caching
importer = DataImporter()
df = importer.import_file('data.xlsx', use_cache=False, save_cache=False)

# Cache processed data instead of raw data
df = importer.import_file('data.xlsx', cache_processed_data=True)
```

### Custom Aggregation Functions
```python
# Create custom aggregation function
def percentile_95(series):
    return series.quantile(0.95)

# Use in aggregation
custom_agg = aggregator.group_and_aggregate(
    data,
    group_key='TIMESTAMP',
    aggregation_dict={
        'SPEED': percentile_95,
        'FUEL_CONSUMPTION': 'mean'
    },
    freq='H'
)
```

## Error Handling and Validation

The package includes comprehensive error handling and validation:

- **Missing columns**: Clear error messages when required columns are not found
- **Data type validation**: Automatic conversion and validation of numeric data
- **File format support**: Graceful handling of unsupported file formats
- **Memory management**: Efficient processing of large datasets with caching
- **Progress tracking**: Progress bars for long-running operations


## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- tqdm (for progress bars)
- openpyxl (for Excel file support)

## License

help yourself