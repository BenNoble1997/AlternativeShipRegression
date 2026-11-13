# Data Visualization Classes

A comprehensive set of visualization classes designed for operational data analysis, with specialized support for maritime vessel tracking, route visualization, and interactive data exploration.

## Overview

This visualization package provides two main classes:

- **GenericDataVisualizer**: A flexible plotting class for general data analysis with comparison plots, time series visualization, and statistical analysis
- **ShipRoutePlotter**: Specialized for maritime operations with interactive route mapping, port visualization, and geographic data plotting

## Core Classes


### 1. ShipRoutePlotter - Maritime Route Visualization

Specialized class for visualizing vessel routes, port calls, and maritime operational data with interactive maps and geographic analysis.

**Key Features:**
- Interactive route plotting with Plotly
- Port visualization with duration-based sizing
- Folium integration for detailed maps
- Time-based filtering and animation support
- Multi-vessel comparison capabilities

**Example Usage:**
```python
from Operational_profile.data_processing.plotting_functions.plot_ship_data_class import ShipRoutePlotter

# Create route plotter
plotter = ShipRoutePlotter(verbose=True)

# Basic route plot
plotter.plot_route(
    title='Vessel Route Analysis',
    datafile=vessel_df,
    output_folder='maps/',
    name='MV_EXAMPLE',
    timestamp_column='TIMESTAMP',
    lat_column='LATITUDE',
    lon_column='LONGITUDE',
    colour_column='SPEED',
    colour_column_name='Speed (knots)',
    start_time='2024-01-01',
    end_time='2024-01-31'
)

# Route with ports overlay
plotter.plot_route_with_ports(
    title='Vessel Route with Port Calls',
    datafile=vessel_df,
    ports_dataframe=ports_df,
    output_folder='maps/',
    name='ROUTE_WITH_PORTS',
    timestamp_column='TIMESTAMP',
    lat_column='LATITUDE',
    lon_column='LONGITUDE',
    colour_column='FUEL_RATE',
    colour_column_name='Fuel Rate (MT/h)',
    port_id_column='PORT_NAME',
    port_lat_column='PORT_LAT',
    port_lon_column='PORT_LON',
    port_duration_column='PORT_DURATION_HOURS',
    port_type_column='IS_DATABASE_PORT'  # Colors ports by type
)
```

#### Route Visualization

Create interactive maps showing vessel movements with customizable data overlays.

```python
# Advanced route plotting with multiple data layers
plotter.plot_route(
    title='Fleet Operations Q1 2024',
    datafile=fleet_data_dict,  # Dictionary of vessel DataFrames
    output_folder='fleet_analysis/',
    timestamp_column='TIMESTAMP',
    lat_column='LATITUDE',
    lon_column='LONGITUDE',
    colour_column='WEATHER_ROUTING_EFFICIENCY',
    colour_column_name='Routing Efficiency (%)',
    start_time='2024-01-01',
    end_time='2024-03-31'
)

# Creates individual maps for each vessel in the dictionary
# Maps are saved as interactive HTML files for sharing
```

#### Port Analysis

Visualize port calls with duration-based sizing and type-based coloring.

```python
# Port-only visualization for logistics analysis
plotter.plot_ports_only(
    title='Global Port Network Analysis',
    ports_dataframe=global_ports_df,
    output_folder='port_analysis/',
    name='GLOBAL_PORTS',
    port_id_column='PORT_CODE',
    port_lat_column='LATITUDE',
    port_lon_column='LONGITUDE',
    port_duration_column='AVERAGE_STAY_HOURS',
    port_type_column='IS_MAJOR_PORT'
)

# Ports are sized by duration and colored by type
# Interactive hover shows detailed port information
```

#### Multi-Vessel Time Series

Create comprehensive time series visualizations across multiple vessels.

```python
# Time segment analysis across fleet
fig, axes = plotter.plot_time_segment(
    datafiles=fleet_data_dict,
    timestamp_column='TIMESTAMP',
    y_columns=['FUEL_EFFICIENCY', 'SPEED_EFFICIENCY', 'WEATHER_IMPACT'],
    figsize=(20, 15),
    cols=3,
    start_time=datetime(2024, 6, 1),
    end_time=datetime(2024, 8, 31),
    fill_area={'FUEL_EFFICIENCY': True},
    fill_alpha=0.2
)

# Creates subplots for each vessel and metric combination
# Only includes vessels with data in the specified time range
# Provides comprehensive fleet performance overview
```
### 2. GenericDataVisualizer - General Purpose Data Visualization

A highly customizable plotting class that handles various types of data visualization needs with support for multiple axes, custom styling, and advanced statistical plots.

**Key Features:**
- Comparison plots with reference lines and percentage bands
- Time series visualization with dual axes
- Error analysis with statistical metrics
- Bar charts with grouping and stacking options
- Extensive customization options

**Example Usage:**
```python
from Operational_profile.data_processing.plotting_functions.Data_visualiser_class import GenericDataVisualizer

# Create visualizer instance
visualizer = GenericDataVisualizer(
    data=df, 
    color_palette='viridis',
    verbose=True
)

# Create comparison plot with reference lines
fig, ax = visualizer.comparison_plot(
    x_column='ACTUAL_SPEED',
    y_columns='PREDICTED_SPEED',
    plot_type='scatter',
    title='Speed Prediction Validation',
    add_reference_line=True,
    reference_line_style='1:1',
    add_percent_bands=True,
    percent_band_values=[10, 20]
)

# Time series plot with secondary axis
fig, (ax1, ax2) = visualizer.time_series_plot(
    time_column='TIMESTAMP',
    y_columns=['FUEL_CONSUMPTION', 'ENGINE_POWER'],
    secondary_y_columns='SPEED',
    title='Operational Parameters Over Time',
    fill_area={'FUEL_CONSUMPTION': True},
    fill_alpha=0.3
)

# Error analysis for model validation
fig, axes = visualizer.error_analysis_plot(
    predicted_column='PREDICTED_FUEL_RATE',
    actual_column='ACTUAL_FUEL_RATE',
    x_column='SPEED',
    title='Fuel Rate Prediction Analysis',
    error_thresholds_pct=[5, 10, 15]
)
```

#### Comparison Plots

Create sophisticated comparison plots with reference lines, trend analysis, and percentage bands.

```python
# Advanced comparison with multiple features
fig, ax = visualizer.comparison_plot(
    x_column='MEASURED_POWER',
    y_columns=['CALCULATED_POWER', 'ESTIMATED_POWER'],
    reference_column='DESIGN_POWER',
    plot_type='both',  # Both scatter and line
    add_trendline=True,
    add_reference_line=True,
    reference_line_style='best-fit',
    add_percent_bands=True,
    percent_band_values=[5, 10, 20],
    band_colors=['green', 'yellow', 'red'],
    title='Power Analysis: Measured vs Calculated vs Estimated'
)
```

#### Time Series Visualization

Handle complex time series data with multiple variables and dual axes.

```python
# Comprehensive time series analysis
fig, (ax1, ax2) = visualizer.time_series_plot(
    time_column='TIMESTAMP',
    y_columns=['MAIN_ENGINE_FUEL', 'AUX_ENGINE_FUEL'],
    secondary_y_columns=['VESSEL_SPEED', 'SEA_STATE'],
    component_columns=['FUEL_ADDITIVE'],
    start_time='2024-01-01',
    end_time='2024-12-31',
    title='Annual Fuel Consumption Analysis',
    fill_area={
        'MAIN_ENGINE_FUEL': True,
        'AUX_ENGINE_FUEL': True
    },
    fill_alpha=0.2,
    date_format='%m-%d'
)
```

#### Error Analysis

Comprehensive error analysis with multiple statistical visualizations.

```python
# Model validation with detailed error metrics
fig, axes = visualizer.error_analysis_plot(
    predicted_column='MODEL_FUEL_CONSUMPTION',
    actual_column='MEASURED_FUEL_CONSUMPTION',
    x_column='VESSEL_SPEED',
    title='Fuel Consumption Model Validation',
    error_thresholds_pct=[5, 10, 15, 25],
    threshold_colors=['green', 'blue', 'orange', 'red'],
    figsize=(18, 12)
)

# The plot automatically displays:
# - Scatter plot with correlation coefficient
# - Error distribution histogram
# - Error vs speed relationship
# - Percentage error analysis
# - Statistical summary with RMSE and accuracy metrics
```

#### Bar Charts

Flexible bar charts with grouping, stacking, and dual axes support.

```python
# Grouped bar chart with error bars
fig, (ax1, ax2) = visualizer.bar_chart(
    x_column='VESSEL_TYPE',
    y_columns=['AVG_FUEL_CONSUMPTION', 'MAX_FUEL_CONSUMPTION'],
    secondary_y_columns='OPERATIONAL_HOURS',
    group_by='ROUTE_TYPE',
    orientation='vertical',
    stacked=False,
    show_values=True,
    value_format='{:.1f}',
    add_error_bars=True,
    error_columns=['FUEL_STD', 'FUEL_MAX_STD'],
    sort_bars='descending',
    title='Fleet Fuel Consumption by Vessel and Route Type'
)
```


## Advanced Usage Examples

### Comprehensive Route Analysis

```python
# Complete voyage analysis workflow
plotter = ShipRoutePlotter(verbose=True)
visualizer = GenericDataVisualizer(voyage_data, color_palette='plasma')

# 1. Interactive route map with efficiency data
plotter.plot_route_with_ports(
    title='Voyage Efficiency Analysis',
    datafile=voyage_data,
    ports_dataframe=port_calls,
    output_folder='voyage_analysis/',
    name='VOYAGE_001',
    colour_column='FUEL_EFFICIENCY',
    colour_column_name='Fuel Efficiency (nm/MT)',
    port_duration_column='BERTH_TIME_HOURS'
)

# 2. Time series analysis of key parameters
fig, (ax1, ax2) = visualizer.time_series_plot(
    time_column='TIMESTAMP',
    y_columns=['SPEED', 'FUEL_RATE'],
    secondary_y_columns='SEA_STATE',
    title='Operational Parameters - Voyage 001',
    fill_area={'FUEL_RATE': True}
)

# 3. Speed vs fuel consumption analysis
fig, ax = visualizer.comparison_plot(
    x_column='SPEED',
    y_columns='FUEL_RATE',
    title='Speed-Fuel Relationship',
    add_trendline=True,
    plot_type='scatter'
)

# 4. Port turnaround efficiency analysis
fig, ax = visualizer.bar_chart(
    x_column='PORT_NAME',
    y_columns='BERTH_TIME_HOURS',
    secondary_y_columns='CARGO_HANDLED_MT',
    title='Port Efficiency Analysis',
    sort_bars='ascending',
    show_values=True
)
```

### Fleet Performance Dashboard

```python
# Multi-vessel fleet analysis
fleet_plotter = ShipRoutePlotter(verbose=True)

# 1. Fleet route overview
fleet_plotter.plot_route(
    title='Fleet Operations Overview',
    datafile=fleet_data,  # Dict with vessel data
    output_folder='fleet_dashboard/',
    colour_column='OPERATIONAL_STATUS',
    colour_column_name='Operational Status'
)

# 2. Comparative time series analysis
fig, axes = fleet_plotter.plot_time_segment(
    datafiles=fleet_data,
    timestamp_column='TIMESTAMP',
    y_columns=['FUEL_EFFICIENCY', 'SCHEDULE_ADHERENCE'],
    figsize=(16, 12),
    cols=2,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

# 3. Fleet-wide performance metrics
for vessel_name, vessel_data in fleet_data.items():
    visualizer = GenericDataVisualizer(vessel_data)
    
    # Performance analysis per vessel
    fig, axes = visualizer.error_analysis_plot(
        predicted_column='PLANNED_FUEL',
        actual_column='ACTUAL_FUEL',
        x_column='VOYAGE_DISTANCE',
        title=f'{vessel_name} - Fuel Planning Accuracy'
    )
```

### Model Validation Workflow

```python
# Comprehensive model validation using visualization tools
validation_visualizer = GenericDataVisualizer(
    validation_data, 
    color_palette='RdYlBu_r'
)

# 1. Model prediction accuracy
fig, axes = validation_visualizer.error_analysis_plot(
    predicted_column='MODEL_PREDICTION',
    actual_column='MEASURED_VALUE',
    x_column='VESSEL_SPEED',
    title='Model Validation Results',
    error_thresholds_pct=[5, 10, 15, 25],
    figsize=(16, 12)
)

# 2. Prediction vs actual comparison
fig, ax = validation_visualizer.comparison_plot(
    x_column='MEASURED_VALUE',
    y_columns='MODEL_PREDICTION',
    plot_type='scatter',
    add_reference_line=True,
    reference_line_style='1:1',
    add_percent_bands=True,
    percent_band_values=[10, 20],
    title='Prediction Accuracy Assessment'
)

# 3. Time-based model performance
fig, ax = validation_visualizer.time_series_plot(
    time_column='TIMESTAMP',
    y_columns=['MEASURED_VALUE', 'MODEL_PREDICTION'],
    title='Model Performance Over Time',
    style_kwargs={'linewidth': 1.5}
)

# 4. Categorical performance analysis
fig, ax = validation_visualizer.bar_chart(
    x_column='VESSEL_TYPE',
    y_columns='PREDICTION_ACCURACY',
    group_by='MODEL_VERSION',
    title='Model Accuracy by Vessel Type and Version',
    show_values=True,
    value_format='{:.2f}%'
)
```

## Customization Options

### Styling and Appearance

```python
# Custom color palettes and styling
visualizer = GenericDataVisualizer(
    data=df,
    color_palette='Set3'  # Or any matplotlib/seaborn palette
)

# Advanced styling parameters
style_kwargs = {
    'alpha': 0.7,
    'linewidth': 2.5,
    'markersize': 8,
    'marker': 's'  # Square markers
}

fig, ax = visualizer.time_series_plot(
    time_column='TIMESTAMP',
    y_columns='DATA_COLUMN',
    style_kwargs=style_kwargs,
    figsize=(16, 10)
)
```

### Export and Sharing

```python
# Route plotter automatically saves HTML files
plotter.plot_route(
    title='Shareable Route Analysis',
    datafile=route_data,
    output_folder='exports/',
    name='VESSEL_ROUTE_2024'
)
# Creates: exports/VESSEL_ROUTE_2024_sailing_route_VESSEL_ROUTE_2024.html

# Save matplotlib figures for reports
fig, ax = visualizer.comparison_plot(
    x_column='X_DATA',
    y_columns='Y_DATA',
    title='Report Figure'
)

# Save in multiple formats
fig.savefig('report_figure.png', dpi=300, bbox_inches='tight')
fig.savefig('report_figure.pdf', bbox_inches='tight')
fig.savefig('report_figure.svg', bbox_inches='tight')
```

## Integration with Data Processing Pipeline

The visualization classes integrate seamlessly with the data processing pipeline:

```python
from Operational_profile.data_processing.workflow_class import DataProcessingWorkflow
from Operational_profile.data_processing.plotting_functions.Data_visualiser_class import GenericDataVisualizer
from Operational_profile.data_processing.plotting_functions.plot_ship_data_class import ShipRoutePlotter

# Complete analysis workflow
workflow = DataProcessingWorkflow(verbose=True)
plotter = ShipRoutePlotter(verbose=True)

# 1. Data processing
processed_data = workflow.importer.import_directory('raw_data/')
clean_data = workflow.filter.filter_out_negative(processed_data, ['SPEED', 'FUEL_RATE'])
hourly_data = workflow.aggregator.group_and_aggregate(
    clean_data, 
    'TIMESTAMP', 
    {'SPEED': 'mean', 'FUEL_RATE': 'mean'}, 
    freq='H'
)

# 2. Visualization
for vessel_name, vessel_data in hourly_data.items():
    # Route visualization
    plotter.plot_route(
        title=f'{vessel_name} Processed Route',
        datafile=vessel_data,
        output_folder='analysis_results/',
        name=vessel_name,
        colour_column='FUEL_RATE'
    )
    
    # Statistical analysis
    visualizer = GenericDataVisualizer(vessel_data)
    fig, ax = visualizer.comparison_plot(
        x_column='SPEED',
        y_columns='FUEL_RATE',
        title=f'{vessel_name} Speed-Fuel Analysis'
    )
```

## Performance Considerations

- **Interactive Maps**: HTML files can become large with extensive route data. Consider data decimation for long voyages
- **Memory Usage**: Large datasets may require chunking for visualization. Consider aggregating data appropriately
- **Rendering Speed**: Complex plots with thousands of points may render slowly. Use appropriate point limits or sampling

## Dependencies

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- folium >= 0.12.0
- pandas >= 1.3.0
- numpy >= 1.20.0

## Output Formats

**GenericDataVisualizer:**
- Returns matplotlib Figure and Axes objects
- Supports all matplotlib export formats (PNG, PDF, SVG, etc.)
- Interactive features in Jupyter notebooks

**ShipRoutePlotter:**
- Saves interactive HTML files automatically
- Plotly-based maps with zoom, pan, and hover capabilities
- Folium maps for detailed geographic analysis
- Files ready for web sharing and embedding

## Error Handling

Both classes include comprehensive error handling:

- **Column Validation**: Clear error messages for missing columns
- **Data Type Checking**: Automatic datetime conversion where needed
- **Empty Data Handling**: Graceful handling of filtered datasets with no data
- **File I/O**: Robust file saving with path validation and creation

## License

Help yourself