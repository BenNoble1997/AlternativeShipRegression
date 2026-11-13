# Ship Shape Basic Parameter System

A Python module for ship parameter management using database-storage approach with automatic interpolation of missing ship dimensions and coefficient calculations.

## Overview

This module provides a comprehensive system for ship design parameter prediction and management, allowing naval architects and ship designers to estimate missing ship dimensions from minimal input data. The system uses regression analysis on historical ship databases to predict parameters and automatically calculates ship coefficients like block coefficient (C_B) and midship coefficient (C_M).

## Key Features

- **Database Storage**: All parameters stored in pandas DataFrame for fast O(1) access
- **Automatic Interpolation**: Missing parameters predicted using regression analysis on historical ship data
- **Coefficient Calculation**: Automatic calculation of C_B using Katsoulis formula and C_M using multiple methods
- **Minimal Input Requirements**: Predict ship dimensions from just ship type + one tonnage parameter + one dimension parameter
- **Parameter Validation**: Comprehensive input validation and requirement checking
- **Property System**: Clean getter/setter interface with automatic recalculation
- **Service Speed Integration**: Calculate block coefficient from service speed using Katsoulis formulation
- **Comprehensive Logging**: Detailed feedback on prediction confidence and methods used

## Core Components

### 1. ShipShapeBasic
The main class for ship parameter management with database storage and automatic interpolation.

**Purpose**: Stores ship parameters in a database-like structure (pandas DataFrame) and automatically fills missing values through interpolation. Provides fast parameter access and automatic coefficient calculations.

**Key Features**:
- Database storage system for fast parameter access
- Automatic parameter interpolation during initialization
- Property-based parameter access with automatic recalculation
- Coefficient calculations (C_B, C_M) with multiple methods
- Service speed integration for C_B calculation

### 2. ShipParameterInterpolator
The machine learning engine that performs parameter prediction.

**Purpose**: Tests multiple regression approaches and selects the best method based on statistical performance.

### 3. FleetData
Database management class for loading and processing ship databases.

**Purpose**: Handles import and preprocessing of ship databases from various file formats.

## Supported Ship Types

The system supports the following ship types via ShipType enum:

- **Container Ships** (key=6)
- **Bulk Carriers** (key=1)
- **Tankers** (key=2)
- **Cruise Ships** (key=3)
- **RO-RO** (key=4)
- **General Cargo** (key=5)


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
from shipshape import ShipShapeBasic, ShipType

# Create ship with minimal parameters
ship = ShipShapeBasic(
    shiptype=ShipType.CONTAINER_SHIP,
    L=180.0,                    # Length Overall (m)
    DWT=50000,                  # Deadweight Tonnage (tonnes)
    C_M=0.98,                   # Midship coefficient
    shipdata='fleet_data.xlsx'  # Ship database file
)

# Access all parameters (interpolated automatically)
print(f"Ship dimensions: L={ship.L:.1f}m, B={ship.B:.1f}m, T={ship.T:.1f}m")
print(f"Tonnage: DWT={ship.DWT:.0f}t, GT={ship.gt:.0f}t")
print(f"Coefficients: C_B={ship.C_B():.3f}, C_M={ship.C_M:.3f}")

# Modify parameters (triggers automatic recalculation)
ship.L = 200.0
ship.DWT = 60000
```

### Usage with Service Speed
```python
# Calculate C_B from service speed using Katsoulis formula
ship = ShipShapeBasic(
    shiptype=ShipType.BULK_CARRIER,
    L=205.0,
    DWT=37500,
    service_speed=14.5,         # Service speed in knots
    shipdata='ship_database.xlsx'
)

# C_B automatically calculated from service speed
print(f"Block coefficient from service speed: {ship.C_B():.3f}")
```

### Advanced Usage with Custom Database
```python
import pandas as pd

# Use custom ship database
custom_data = pd.read_excel('my_ship_data.xlsx')

ship = ShipShapeBasic(
    shiptype=ShipType.TANKER,
    L=250.0,
    gt=75000,                   # Using GT instead of DWT
    C_M=0.95,
    shipdata=custom_data,
    verbose=True,               # Enable detailed logging
    debug=False                 # Disable debug logging
)

# Manual re-interpolation after parameter changes
ship.B = 32.0
ship.T = 12.0
ship.re_interpolate_parameters()  # Update other parameters based on changes
```

## Input Requirements

### Minimum Required Parameters
You must provide **at least**:

1. **Ship type**: One of the supported ship types
2. **One tonnage parameter**: DWT OR GT
3. **One dimension parameter**: L, L_PP, B, OR T

### Optional Parameters
- **C_M**: Midship coefficient (calculated automatically if not provided)
- **C_B**: Block coefficient (calculated automatically if not provided)
- **service_speed**: Service speed in knots (enables C_B calculation via Katsoulis formula)
- **shipdata**: Ship database for interpolation

### Common Input Combinations
```python
# Scenario 1: Basic design with service speed
ShipShapeBasic(L=180, DWT=50000, service_speed=22, shiptype=ShipType.CONTAINER_SHIP)

# Scenario 2: With known coefficients
ShipShapeBasic(L=180, DWT=50000, C_M=0.98, C_B=0.65, shiptype=ShipType.CONTAINER_SHIP)

# Scenario 3: Multiple dimensions known
ShipShapeBasic(L=180, _B=32, _T=12, DWT=50000, shiptype=ShipType.CONTAINER_SHIP)
```

## Coefficient Calculations

### Block Coefficient (C_B)

The system calculates C_B using two methods:

#### 1. Katsoulis Formula (when service_speed provided)
```
Vs = a1 * C_B^a2 * L^a3 * B^a4 * T^a5
```

Coefficients by ship type:
- **Bulk Carrier**: [3.7682, -0.3236, 0.3566, -0.0455, -0.1866]
- **Container Ship**: [0.9804, -0.9809, 0.7416, -0.3742, -0.025]
- **Oil Tanker**: [1.633, -0.3280, 0.7256, -0.5122, -0.0214]
- **Other**: [0.7261, -1.63, 0.6846, -0.5007, 0.2805]

#### 2. Volume Method (when dimensions and DWT available)
```
C_B = DWT_volume / (L × B × T)
```

### Midship Coefficient (C_M)

C_M is calculated using the average of three methods with bounds checking (0.96 ≤ C_M ≤ 1.0):

1. **Linear method**: `C_M1 = 0.9 + 0.1 × C_B`
2. **Exponential method**: `C_M2 = 1 / (1 + (1 - C_B)^3.5)`
3. **Power method**: `C_M3 = 1.006 - (0.0056 × C_B^(-3.56))`

Only methods producing realistic results (within bounds) contribute to the final average.

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

## Performance Features

- **O(1) Parameter Access**: All parameters stored in DataFrame for instant access
- **Single Interpolation Run**: Interpolation performed once during initialization
- **Selective Recalculation**: Only affected parameters recalculated when values change
- **Large Database Support**: Efficiently handles databases with >10,000 ships

## API Reference

### ShipShapeBasic Class

#### Constructor
```python
ShipShapeBasic(shiptype, L, C_M=None, C_B=None, service_speed=None,
               DWT=None, gt=None, _L_PP=None, _B=None, _T=None, 
               shipdata=None, verbose=True, debug=False)
```

**Parameters:**
- `shiptype`: ShipType enum value
- `L`: Length Overall (meters)
- `C_M`: Midship coefficient (optional, calculated if not provided)
- `C_B`: Block coefficient (optional, calculated if not provided)
- `service_speed`: Service speed in knots (optional, enables Katsoulis C_B calculation)
- `DWT`: Deadweight tonnage (tonnes)
- `gt`: Gross tonnage (tonnes)
- `_L_PP`, `_B`, `_T`: Length between perpendiculars, breadth, draft (optional)
- `shipdata`: Ship database file path or DataFrame
- `verbose`: Enable INFO level logging
- `debug`: Enable DEBUG level logging

#### Key Properties (Read/Write)
- `L`, `B`, `T`, `L_PP`: Ship dimensions
- `DWT`, `gt`: Tonnage parameters  
- `C_M`: Midship coefficient

#### Key Methods
- `C_B()`: Get block coefficient (calculated property)
- `re_interpolate_parameters()`: Re-run interpolation after parameter changes

### ShipType Enum

#### Methods
- `ShipType.from_string(name)`: Convert string to ShipType
- `ShipType.from_key(key)`: Convert numeric key to ShipType

#### Properties
- `.name`: Formatted name (e.g., "Container_Ship")
- `.key`: Numeric identifier (1-6)

## Error Handling

The class provides comprehensive error handling for:

- **Invalid ship type specifications**
- **Insufficient parameter combinations**
- **Missing or malformed ship data**
- **Interpolation failures** (graceful fallback to provided parameters)
- **Parameter validation failures**
- **Coefficient calculation errors**

## Troubleshooting

### Common Issues

**"Must specify at least one tonnage parameter"**
- Provide either DWT or gt parameter

**"Must specify at least one dimension parameter"**
- Provide at least one of: L, _L_PP, _B, or _T

**"Missing required columns in shipdata"**
- Ensure database contains all required columns with exact names

**Coefficient calculation warnings**
- Check that ship dimensions are realistic
- Verify service speed is appropriate for ship type

## Example Applications

### 1. Conceptual Design
```python
# Quick estimation for early design phases
ship = ShipShapeBasic(
    L=200, 
    DWT=75000, 
    shiptype=ShipType.BULK_CARRIER,
    service_speed=14.0
)
print(f"Estimated dimensions: B={ship.B:.1f}m, T={ship.T:.1f}m")
print(f"Block coefficient: {ship.C_B():.3f}")
```

### 2. Parametric Studies
```python
# Study effect of service speed on block coefficient
for speed in [12, 14, 16, 18]:
    ship = ShipShapeBasic(
        L=200, DWT=50000, service_speed=speed, 
        shiptype=ShipType.BULK_CARRIER
    )
    print(f"Speed={speed}kn → C_B={ship.C_B():.3f}")
```

### 3. Design Optimization
```python
# Optimize ship parameters
ship = ShipShapeBasic(L=180, DWT=50000, shiptype=ShipType.CONTAINER_SHIP)

# Test different lengths
for length in [170, 180, 190, 200]:
    ship.L = length  # Triggers automatic recalculation
    print(f"L={length}m → B={ship.B:.1f}m, C_B={ship.C_B():.3f}")
```

## Version History

- **Version 1.0.0** (31-05-2025): 
  - Database-based parameter storage
  - Automatic coefficient calculations
  - Service speed integration
  - Comprehensive input validation

## Authors

- **Ben Noble** - TU Delft
- Date: 31-05-2025
- Project: T1.6 Deliverable - Digital Twin for Green Shipping

## Dependencies

- **pandas**: DataFrame storage and manipulation
- **numpy**: Numerical calculations
- **scikit-learn**: Regression analysis (via ShipParameterInterpolator)
- **Custom modules**: ShipType, LoggerMixin, FleetData

## Notes

This class is optimized for applications requiring frequent parameter access and modification. The database storage approach provides significant performance benefits over on-demand calculations, making it suitable for interactive design applications and parametric studies.

Parameter interpolation accuracy depends on the quality and completeness of the provided ship database. Ensure your dataset contains sufficient examples of the target ship type for reliable predictions.