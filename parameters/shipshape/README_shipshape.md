# Ship Parameter Interpolation Module

A Python module for predicting missing ship dimensions and parameters using machine learning-based interpolation on historical ship databases.

## Overview

This module provides a comprehensive system for ship design parameter prediction, allowing naval architects and ship designers to estimate missing ship dimensions from minimal input data. The system uses regression analysis on historical ship databases to predict parameters like breadth, draft, and tonnage values from basic ship characteristics.

## Key Features

- **Minimal Input Requirements**: Predict ship dimensions from just ship type + one tonnage parameter + one dimension parameter
- **Machine Learning Interpolation**: Uses multiple regression models to find optimal prediction methods
- **Range Validation**: Warns when inputs or predictions fall outside historical data ranges
- **Database-Driven**: Fast parameter access through pandas DataFrame storage
- **Iterative Prediction**: Improves accuracy by using previous predictions to enhance subsequent ones
- **Comprehensive Logging**: Detailed feedback on prediction confidence and methods used

## Core Components

### 1. [ShipShapeBasic](./parameters/shipshape/README_ShipShapeBasic.md)
The main class for ship parameter management with automatic interpolation.

**Purpose**: Stores ship parameters in a database-like structure and automatically fills missing values through interpolation.

**Key Methods**:
- Automatic parameter interpolation during initialization
- Property-based parameter access with O(1) lookup
- Dynamic parameter modification with automatic recalculation

### 2. ShipParameterInterpolator
The machine learning engine that performs parameter prediction.

**Purpose**: Tests multiple regression approaches and selects the best method based on statistical performance.

**Key Methods**:
- Single and multiple parameter regression
- Iterative prediction for improved accuracy
- R² validation and confidence assessment
- Range validation against dataset bounds

### 3. FleetData
Database management class for loading and processing ship databases.

**Purpose**: Handles import and preprocessing of ship databases from various file formats.

**Key Methods**:
- Excel/CSV file import
- Data cleaning and validation
- Database combining and filtering

## System Architecture

```
User Input (L, DWT, ship_type) 
         ↓
   ShipShapeBasic
         ↓
ShipParameterInterpolator ← FleetData (ship database)
         ↓
   Regression Analysis
         ↓
   Parameter Prediction
         ↓
   Range Validation
         ↓
   Complete Ship Parameters
```

## Installation and Dependencies

### Required Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
pathlib
logging
typing
```

### Installation
```bash
# Clone or download the module files
# Ensure all dependencies are installed
pip install pandas numpy scikit-learn
```

## Quick Start

### Basic Usage
```python
from parameters.shipshape.ShipShapeBasic_class import ShipShapeBasic, ShipType

# Create ship with minimal parameters
ship = ShipShapeBasic(
    shiptype=ShipType.BULK_CARRIER,
    L=205.0,                    # Length Overall (m)
    DWT=37500,                  # Deadweight Tonnage (tonnes)
    C_M=0.98,                   # Midship coefficient
    shipdata='ship_database.xlsx'  # Ship database file
)

# Access interpolated parameters
print(f"Breadth: {ship.B:.1f}m")           # Automatically predicted
print(f"Draft: {ship.T:.1f}m")             # Automatically predicted
print(f"L_PP: {ship.L_PP:.1f}m")           # Automatically predicted
print(f"GT: {ship.gt:.0f}")                # Automatically predicted
```

### Advanced Usage with Custom Interpolation
```python
from parameters.shipshape.ship_parameter_interpolation_class import ShipParameterInterpolator

# Create interpolator directly
interpolator = ShipParameterInterpolator(
    shiptype=ShipType.CONTAINER_SHIP,
    shipdata=ship_database,
    verbose=True
)

# Set known parameters
interpolator.set_parameters(L=180.0, DWT=50000)

# Run prediction
interpolator.predict_missing_parameters()

# Get results with statistics
params = interpolator.get_parameters()
stats = interpolator.get_prediction_stats()

print(f"Predicted B = {params['B']:.2f}m (R² = {stats['B']['r_squared']:.3f})")
```

## Database Requirements

### Required Columns
Your ship database must contain these columns:

| Column Name | Description | Units |
|-------------|-------------|-------|
| TYPE_KEY | Ship type identifier (1-6) | Integer |
| LOA (M) | Length Overall | Meters |
| LBP (M) | Length Between Perpendiculars | Meters |
| BEAM MLD (M) | Moulded Breadth | Meters |
| DRAFT (M) | Design Draft | Meters |
| DWT | Deadweight Tonnage | Tonnes |
| GT | Gross Tonnage | Tonnes |

### Ship Type Keys
- 1: Bulk Carrier
- 2: Tanker  
- 3: Cruise Ship
- 4: RO-RO
- 5: Cargo
- 6: Container Ship

### Data Quality Guidelines
- **Minimum 50 ships per ship type** for reliable predictions
- **Remove outliers** and erroneous data points
- **Consistent units** across all measurements
- **Representative size range** for your target applications

## Input Requirements

### Minimum Required Parameters
You must provide **at least**:

1. **One tonnage parameter**: DWT OR GT
2. **One dimension parameter**: L, L_PP, B, OR T
3. **Ship type**: One of the supported ship types
4. **Midship coefficient**: C_M

### Common Input Combinations
```python
# Scenario 1: Basic design
ShipShapeBasic(L=180, DWT=50000, shiptype=ShipType.CONTAINER_SHIP, C_M=0.98)

# Scenario 2: With known breadth
ShipShapeBasic(L=180, _B=32, DWT=50000, shiptype=ShipType.CONTAINER_SHIP, C_M=0.98)

# Scenario 3: Multiple dimensions known
ShipShapeBasic(L=180, _B=32, _T=12, DWT=50000, shiptype=ShipType.CONTAINER_SHIP, C_M=0.98)
```

## How the Interpolation Works

### 1. Data Preparation
- Filters ship database to matching ship type
- Validates required columns exist
- Calculates statistical ranges for validation

### 2. Regression Analysis
For each missing parameter, the system:
- Tests single-parameter regression (e.g., B from L)
- Tests multiple-parameter regression (e.g., B from L+DWT)
- Selects method with highest R² correlation

### 3. Iterative Prediction
- **Iteration 1**: Predict parameters possible from initial inputs
- **Iteration 2**: Use newly predicted values to improve remaining predictions
- **Continue**: Until all parameters predicted or no further progress

### 4. Validation and Confidence
- **Range checking**: Warn if predictions fall outside dataset bounds
- **Confidence assessment**: Combine R² scores with range validation
- **Ratio validation**: Check L/B, B/T, DWT/GT ratios against dataset norms

## Understanding the Output

### Prediction Confidence Levels
- **High**: R² > 0.9 and within normal data range
- **Medium**: R² 0.7-0.9 or slightly outside typical range  
- **Low**: R² < 0.7 or outside 90% of data range
- **Very Low**: Poor statistical fit and outside dataset bounds

### Warning Levels
- **Normal**: Parameter within typical range (80% of dataset)
- **Minor**: Outside typical range but within dataset bounds
- **Major**: In extreme 10% of dataset
- **Critical**: Outside entire dataset range - very unreliable

### Example Output
```
Predicted Parameters:
  B: 32.40 (R² = 0.891, Confidence: medium-high, Iteration: 1)
  T: 12.10 (R² = 0.823, Confidence: medium, Iteration: 1) [MINOR WARNING]
  L_PP: 172.00 (R² = 0.954, Confidence: high, Iteration: 1)
  GT: 28500 (R² = 0.887, Confidence: medium, Iteration: 1)
```

## Limitations and Considerations

### Prediction Accuracy
- **Accuracy depends entirely on database quality and representativeness**
- **Linear regression may miss non-linear ship relationships**
- **Extrapolation beyond database range is unreliable**
- **R² scores indicate training fit, not necessarily prediction accuracy**

### When to Be Cautious
- Ships significantly larger/smaller than database ships
- Novel ship designs not represented in historical data
- Ship types with limited database examples
- Parameter combinations far from database norms

### Validation Recommendations
- **Cross-check results** against naval architecture principles
- **Verify feasibility** of predicted ship proportions
- **Use multiple databases** if possible for comparison
- **Validate critical designs** with detailed calculations

## Troubleshooting

### Common Errors

**"Missing required columns in shipdata"**
- Check your database file contains all required columns
- Verify column names match exactly (case-sensitive)

**"No data available for ship type"**
- Check TYPE_KEY values in your database
- Ensure ship type exists in your dataset

**"Failed to predict parameters"**
- Database may lack sufficient examples of your ship type
- Try providing more input parameters
- Check for data quality issues in database

**"Parameter X outside dataset range"**
- Your ship may be unusual compared to database
- Consider using different ship type or database
- Verify input parameters are reasonable

### Getting Better Results

1. **Use larger, higher-quality databases**
2. **Provide more input parameters when possible**
3. **Choose ship type that best matches your design**
4. **Validate results against known similar ships**
5. **Use range summary to understand database coverage**

## Example Applications

### 1. Conceptual Design
```python
# Quick estimation for early design phases
ship = ShipShapeBasic(L=200, DWT=75000, shiptype=ShipType.BULK_CARRIER, C_M=0.98)
print(f"Estimated breadth: {ship.B:.1f}m")
```

### 2. Parametric Studies
```python
# Study effect of length on other parameters
for length in [180, 200, 220, 240]:
    ship = ShipShapeBasic(L=length, DWT=50000, shiptype=ShipType.CONTAINER_SHIP, C_M=0.98)
    print(f"L={length}m → B={ship.B:.1f}m, T={ship.T:.1f}m")
```

### 3. Design Validation
```python
# Check if design parameters are reasonable
ship = ShipShapeBasic(L=180, _B=32, _T=12, DWT=50000, shiptype=ShipType.CONTAINER_SHIP, C_M=0.98)
warnings = ship.interpolator.validate_ship_feasibility()
if warnings:
    print("Design warnings:", warnings)
```

## API Reference

### ShipShapeBasic Class

#### Constructor
```python
ShipShapeBasic(shiptype, L, C_M=None, C_B=None, DWT=None, shipdata=None, 
               _L_PP=None, _B=None, _T=None, gt=None, 
               verbose=True, debug=False)
```

#### Key Properties
- `L`, `B`, `T`, `L_PP`: Ship dimensions (read/write)
- `DWT`, `gt`: Tonnage parameters (read/write)  
- `C_M`, `C_B()`: Coefficients

#### Key Methods
- `re_interpolate_parameters()`: Re-run interpolation after changes

### ShipParameterInterpolator Class

#### Constructor
```python
ShipParameterInterpolator(shiptype, shipdata, verbose=True, debug=False)
```

#### Key Methods
- `set_parameters(**kwargs)`: Set known parameters
- `predict_missing_parameters()`: Run interpolation
- `get_parameters()`: Get all parameters
- `get_prediction_stats()`: Get R² and confidence data
- `print_summary()`: Display detailed results
- `validate_ship_feasibility()`: Check parameter ratios

### FleetData Class

#### Constructor
```python
FleetData(verbose=True, debug=False)
```

#### Key Methods
- `import_data(file_path)`: Load ship database
- `get_dimension_summary(columns)`: Analyze database ranges
- `list_datasets()`: Show loaded datasets

## Contributing

When contributing to this module:

1. **Maintain backward compatibility** with existing API
2. **Add comprehensive tests** for new functionality  
3. **Update documentation** for any API changes
4. **Follow existing code style** and logging patterns
5. **Validate against multiple ship databases** before releasing

## License and Citation

This module is part of a ship design research project. When using this code in academic work, please cite appropriately and reference the underlying regression methodologies.

## Support

For technical issues:
1. Check the troubleshooting section above
2. Verify your database format matches requirements
3. Test with provided example scripts
4. Review log output for detailed error information

## Authors

- **Ben Noble** - TU Delft
- Version: 1.0
- Date: 02-06-25
- Project: Digital Twin for Green Shipping