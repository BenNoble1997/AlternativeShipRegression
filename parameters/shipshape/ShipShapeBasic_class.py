"""
Ship Shape Basic Parameter System
==================================

This module provides the core ship parameter management system using a database-storage
approach for fast parameter access and automatic interpolation of missing ship dimensions.

Overview
--------
The ShipShapeBasic class is designed to handle ship parameter storage, validation, and
calculation with high performance through DataFrame-based storage. It automatically
interpolates missing ship parameters using regression analysis on historical ship data.

Author: Ben Noble
Date: 31-05-2025
version: 1.0.0

Key Features
------------
* **Database Storage**: All parameters stored in pandas DataFrame for fast access
* **Automatic Interpolation**: Missing parameters predicted using regression analysis
* **Parameter Validation**: Comprehensive input validation and requirement checking
* **Derived Calculations**: Automatic calculation of coefficients (C_B, C_M)
* **Property System**: Clean getter/setter interface with automatic recalculation
* **Comprehensive Logging**: Detailed logging for debugging and monitoring

Supported Ship Types
--------------------
The system supports the following ship types (via ShipType enum):

* Container Ships (key=0)
* Bulk Carriers (key=1)
* Tankers (key=2)
* General Cargo (key=3)
* Passenger Ships (key=4)
* Other Commercial (key=5)

Required Parameters
-------------------
Users must provide at least:

* **One tonnage parameter**: DWT (Deadweight Tonnage) OR GT (Gross Tonnage)
* **One dimension parameter**: L (Length Overall), L_PP (Length Between Perpendiculars),
  B (Breadth), OR T (Draft)

Parameter Interpolation
-----------------------
The system uses the ShipParameterInterpolator to predict missing parameters through:

1. **Regression Analysis**: Multiple regression models tested automatically
2. **Best Method Selection**: Highest R² correlation chosen for each parameter
3. **Iterative Prediction**: Previously predicted values used to improve subsequent predictions
4. **Statistical Validation**: Predictions validated against dataset ranges

Basic Usage
-----------

.. code-block:: python

    from shipshape import ShipShapeBasic, ShipType

    # Create ship with minimum required parameters
    ship = ShipShapeBasic(
        shiptype=ShipType.CONTAINER,
        L=180.0,           # Length overall (m)
        C_M=0.98,          # Midship coefficient
        DWT=50000,         # Deadweight tonnage (tonnes)
        shipdata='fleet_data.xlsx'  # Optional: ship database for interpolation
    )

    # Access all parameters (interpolated automatically)
    print(f"Ship dimensions: L={ship.L:.1f}m, B={ship.B:.1f}m, T={ship.T:.1f}m")
    print(f"Tonnage: DWT={ship.DWT:.0f}t, GT={ship.gt:.0f}t")
    print(f"Coefficients: C_B={ship.C_B():.3f}, C_M={ship.C_M:.3f}")

    # Modify parameters (triggers automatic recalculation)
    ship.L = 200.0
    ship.DWT = 60000

Advanced Usage
--------------

.. code-block:: python

    # Use custom ship database
    import pandas as pd
    custom_data = pd.read_excel('my_ship_data.xlsx')

    ship = ShipShapeBasic(
        shiptype=ShipType.BULK_CARRIER,
        L=250.0,
        C_M=0.95,
        gt=75000,          # Using GT instead of DWT
        shipdata=custom_data,
        verbose=True,      # Enable detailed logging
        debug=False        # Disable debug logging
    )

    # Manual re-interpolation after parameter changes
    ship.B = 32.0
    ship.T = 12.0
    ship.re_interpolate_parameters()  # Update other parameters based on changes

Ship Database Format
--------------------
If providing ship data for interpolation, the DataFrame/file must contain:

.. list-table:: Required Columns
   :widths: 25 25 50
   :header-rows: 1

   * - Column Name
     - Description
     - Units
   * - TYPE_KEY
     - Ship type identifier (0-5)
     - Integer
   * - LOA (M)
     - Length Overall
     - Meters
   * - LBP (M)
     - Length Between Perpendiculars
     - Meters
   * - BEAM MLD (M)
     - Moulded Breadth
     - Meters
   * - DRAFT (M)
     - Design Draft
     - Meters
   * - DWT
     - Deadweight Tonnage
     - Tonnes
   * - GT
     - Gross Tonnage
     - Tonnes

Performance Notes
-----------------
* Parameter access is O(1) via DataFrame indexing
* Interpolation runs once during initialization
* Property setters trigger selective recalculation only
* Large ship databases (>10,000 ships) supported efficiently

Error Handling
--------------
The class provides comprehensive error handling for:

* Invalid ship type specifications
* Insufficient parameter combinations
* Missing or malformed ship data
* Interpolation failures (graceful fallback to provided parameters)
* Parameter validation failures

Dependencies
------------
* pandas: DataFrame storage and manipulation
* numpy: Numerical calculations
* scikit-learn: Regression analysis (via ShipParameterInterpolator)
* Custom modules: ShipType, LoggerMixin, FleetData

See Also
--------
* :class:`ShipParameterInterpolator`: Core interpolation engine
* :class:`ShipType`: Ship type enumeration
* :class:`FleetData`: Ship database import utilities

.. note::
   This class is optimized for applications requiring frequent parameter access
   and modification. For one-time calculations, consider using the interpolator directly.

.. warning::
   Parameter interpolation accuracy depends on the quality and completeness of
   the provided ship database. Ensure your dataset contains sufficient examples
   of the target ship type.
"""

# Modules imports.
import sys
import os

import pandas as pd

import logging
from pathlib import Path


os.chdir(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable")


sys.path.append(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable\config")

from config.config_constants import SEAWATER_DENSITY
from config.loggermixin.logger_config import LoggerMixin

from enum import Enum


from knowledge_layer.fleet_data_class.fleet_data_class import FleetData

class ShipType(Enum):
    """
    Creates Enum object for ship type.

    format:
    name =
    - Capitalized first word
    - _ instead of space


    """
    BULK_CARRIER = ("Bulk_Carrier",1,"Bulk carrier")
    TANKER = ("Tanker", 2, "Tanker")
    CRUISE_SHIP = ("Cruise_Ship",3,"Cruise ship")
    RO_RO = ("RO_RO",4,"Ro-Ro")
    RO_PAX = ("RO_PAX",5,"RoPax")
    CARGO = ("Cargo",6,"Cargo")
    CONTAINER_SHIP = ('Container_Ship',7,"Container ship")

    @property
    def name(self):
        return self.value[0]

    @property
    def key(self):
        return self.value[1]

    @property
    def string(self):
        return self.value[2]

    @classmethod
    def from_string(cls, name):
        """Convert a string to corresponding ShipType enum.
        Accepts both formatted names (e.g., 'Bulk_Carrier') and regular names (e.g., 'Bulk Carrier')
        """
        # First try exact match
        for ship_type in cls:
            if ship_type.name == name:
                return ship_type

        # Try with spaces replaced by underscores
        formatted_name = name.replace(' ', '_')
        for ship_type in cls:
            if ship_type.name == formatted_name:
                return ship_type

        # Try case-insensitive
        name_lower = name.lower()
        for ship_type in cls:
            if ship_type.name.lower() == name_lower or ship_type.name.lower() == formatted_name.lower():
                return ship_type

        raise ValueError(f"No ShipType found matching '{name}'")

    @classmethod
    def from_key(cls, key):
        """Convert a numeric key to corresponding ShipType enum"""
        for ship_type in cls:
            if ship_type.key == key:
                return ship_type
        raise ValueError(f"No ShipType found with key {key}")


class ShipShapeBasic(LoggerMixin):
    """
    Ship shape class using database storage as the primary system.

    All parameters are stored in a DataFrame and calculated once during initialization.
    Property access is fast DataFrame lookups rather than on-demand calculations.
    """

    def __init__(self,
                 shiptype: ShipType,
                 L: float|None,
                 C_M: float|None=None,
                 C_B: float|None=None,
                 service_speed: float|None=None,
                 DWT: float|None= None,
                 gt: float|None= None,
                 ldt: float|None= None,
                 _L_PP: float|None= None,
                 _B: float|None= None,
                 _T: float|None= None,
                 interpolate_params:bool=True,
                 shipdata: str|Path|pd.DataFrame = None,
                 verbose: bool = True,
                 debug: bool = False):

        """
        Initialise the ShipShapeBasic class.

        :param shiptype: Ship type enumeration
        :param L: Length of the ship
        :param C_M: midship coefficient of the ship
        :param service_speed: service speed of the ship in knots
        :param DWT: Dead weight tonnage of the ship (in tonnes)
        :param gt: Gross tonnage of the ship (in tonnes)
        :param _L_PP: Length between perpendiculars of the ship (optional, otherwise determined through interpolation)
        :param _B: Breadth of the ship in metres (optional, otherwise determined through interpolation)
        :param _T: Draft of the ship (optional, otherwise determined through interpolation)
        :param shipdata: Ship database used for interpolating values
        :param verbose: set logging level to INFO
        :param debug: set logging level to DEBUG

        """

        # ===== BASIC SETUP =====
        self.logger.info(f"Initializing ShipShapeBasic with shiptype={shiptype.name}, L={L}, C_M={C_M}")

        self.verbose = verbose
        self.debug = debug


        self.interpolate_params = interpolate_params

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        # Store input parameters
        self.shiptype = shiptype
        self.shipdata = shipdata
        self.data_importer = FleetData(verbose=self.verbose, debug=self.debug)
        self.rho_SW = SEAWATER_DENSITY
        self.service_speed = service_speed

        # Store initial user inputs (for reference and validation)
        self._input_L = L
        self._input_C_M = C_M
        self._input_C_B = C_B
        self._input_DWT = DWT
        self._input_gt = gt
        self._input_ldt = ldt
        self._input_L_PP = _L_PP
        self._input_B = _B
        self._input_T = _T

        # Legacy attributes (for compatibility with existing code that might access them)
        self.parameter = []
        self.Type = []
        self.value = []

        # ===== VALIDATION AND PROCESSING =====
        self.logger.info("Validating inputs...")
        self._validate_parameter_requirements()
        self.logger.debug(f"Input validation passed. Ship type: {self.shiptype.name}, key: {self.shiptype.key}")

        self.logger.info("Processing ship data...")
        self._process_shipdata()

        # ===== DATABASE STORAGE SYSTEM =====
        self.logger.info("Setting up database storage...")
        self._setup_database_storage()

        self.logger.info(f"Successfully initialized ShipShapeBasic for {shiptype.name}")


    def _validate_parameter_requirements(self):
        """Validate that the user has provided sufficient parameters for prediction.

        User must specify at least one tonnage value from [DWT,GT]
        and one dimenion from ['L', 'L_PP', 'B', 'T']

        If not, raises an error detailing which parameters are missing
        """
        # Check tonnage parameters (at least one required)
        provided_tonnage = []
        if self._input_DWT is not None:
            provided_tonnage.append('DWT')
        if self._input_gt is not None:
            provided_tonnage.append('GT')
        if self._input_ldt is not None:
            provided_tonnage.append('LDT')

        # Check dimension parameters (at least one required)
        provided_dimensions = []
        if self._input_L is not None:
            provided_dimensions.append('L')
        if self._input_L_PP is not None:
            provided_dimensions.append('L_PP')
        if self._input_B is not None:
            provided_dimensions.append('B')
        if self._input_T is not None:
            provided_dimensions.append('T')

        # Log what was provided
        self.logger.debug(f"Provided tonnage parameters: {provided_tonnage}")
        self.logger.debug(f"Provided dimension parameters: {provided_dimensions}")

        # Validate requirements
        if not provided_tonnage:
            self.logger.error("No tonnage parameters provided")
            raise ValueError("Must specify at least one tonnage parameter: DWT or gt")

        if not provided_dimensions:
            self.logger.error("No dimension parameters provided")
            raise ValueError("Must specify at least one dimension parameter: L, _L_PP, _B, or _T")

        # Success - log what will be predicted
        missing_tonnage = [p for p in ['DWT', 'GT'] if p not in provided_tonnage]
        missing_dimensions = [p for p in ['L', 'L_PP', 'B', 'T'] if p not in provided_dimensions]

        if missing_tonnage or missing_dimensions:
            all_missing = missing_tonnage + missing_dimensions
            self.logger.info(f"Parameters that will be predicted: {all_missing}")
            self.logger.info(f"Parameters available for prediction: {provided_tonnage + provided_dimensions}")
        else:
            self.logger.info("All parameters provided - no prediction needed")

        self.logger.debug("Parameter requirements satisfied")

    def _process_shipdata(self):
        """Process the ship data from file or DataFrame."""
        self._load_shipdata_if_needed()
        self._validate_required_columns()
        self._log_shiptype_statistics()

    def _load_shipdata_if_needed(self):
        """Load ship data from file if not already a DataFrame."""
        if isinstance(self.shipdata, pd.DataFrame):
            return

        if not self.interpolate_params or self.shipdata is None:
            return

        if isinstance(self.shipdata, str):
            self.shipdata = Path(self.shipdata)

        self.logger.info(f"Loading ship data from file: {self.shipdata}")

        try:
            imported_data = self.data_importer.import_data(
                file_path=self.shipdata,
                combine=True
            )

            imported_data.processing_fleet_data(name='processed1')

            imported_data = imported_data.processed_data['processed1']

            self.logger.debug(f"Import_data returned type: {type(imported_data)}")

            self.shipdata = self._extract_dataframe_from_import(imported_data)
            self.logger.debug(f"Successfully loaded ship data with {len(self.shipdata)} rows")

        except Exception as e:
            self.logger.error(f"Failed to load ship data: {str(e)}")
            raise

    def _extract_dataframe_from_import(self, imported_data):
        """Extract DataFrame from the imported data dictionary."""
        if not isinstance(imported_data, dict):
            return imported_data

        if len(imported_data) == 0:
            raise ValueError("No data returned from import_data")

        if len(imported_data) == 1:
            dataset_name, dataframe = list(imported_data.items())[0]
            self.logger.debug(f"Extracted DataFrame '{dataset_name}' from import result")
            return dataframe

        # Multiple DataFrames - concatenate them
        self.logger.info(f"Multiple datasets returned: {list(imported_data.keys())}")
        dataframes = list(imported_data.values())
        self.logger.debug(f"Concatenated {len(dataframes)} DataFrames")
        return pd.concat(dataframes, ignore_index=True)

    def _validate_required_columns(self):
        """Validate that all required columns are present in the ship data."""
        if self.shipdata is None:
            return

        required_columns = [
            'TYPE_KEY', 'LOA (M)', 'LBP (M)', 'BEAM MLD (M)',
            'DRAFT (M)', 'DWT', 'GT', 'LDT'
        ]

        self.logger.debug("Checking required columns in ship data...")
        missing_columns = [
            col for col in required_columns
            if col not in self.shipdata.columns
        ]

        if missing_columns:
            self.logger.error(f"Missing columns in ship data: {missing_columns}")
            self.logger.debug(f"Available columns: {self.shipdata.columns.tolist()}")
            raise ValueError(
                f"Missing required columns in shipdata: {missing_columns}"
            )

        self.logger.info("Ship data contains all required columns")

    def _log_shiptype_statistics(self):
        """Log statistics for ships matching the current ship type."""
        if self.shipdata is None:
            return

        type_filter = self.shipdata['TYPE_KEY'] == self.shiptype.key
        type_count = type_filter.sum()
        self.logger.info(
            f"Found {type_count} ships of type {self.shiptype.name} in data"
        )

        if type_count == 0:
            self.logger.warning(
                f"No ships of type {self.shiptype.name} found in the data"
            )
            return

        self._log_column_statistics(self.shipdata[type_filter])

    def _log_column_statistics(self, type_data):
        """Log basic statistics for numeric columns."""
        numeric_columns = [
            'LOA (M)', 'LBP (M)', 'BEAM MLD (M)',
            'DRAFT (M)', 'DWT', 'GT', 'LDT'
        ]

        self.logger.debug(f"Statistics for {self.shiptype.name} ships:")
        for col in numeric_columns:
            if col in type_data.columns:
                stats = type_data[col].describe()
                self.logger.debug(
                    f"{col}: min={stats['min']:.2f}, "
                    f"mean={stats['mean']:.2f}, "
                    f"max={stats['max']:.2f}"
                )

    def _setup_database_storage(self):
        """Set up the database storage system."""
        self.logger.debug("Setting up database storage system...")

        # Step 1: Run interpolation to get all parameters
        self._run_parameter_interpolation()

        # Step 2: Create ship record with all parameters
        self._ship_record = self._create_ship_record()

        self.logger.info("Database storage system initialized successfully")

    def _run_parameter_interpolation(self):
        """Run interpolation to fill missing parameters."""
        if self.shipdata is None or self.interpolate_params:
            self.logger.debug("No shipdata available for interpolation")
            # Use provided values directly
            self._final_params = {
                'L': self._input_L,
                'L_PP': self._input_L_PP,
                'B': self._input_B,
                'T': self._input_T,
                'DWT': self._input_DWT,
                'GT': self._input_gt,
                'LDT': self._input_ldt,
            }
            return

        try:
            # Import the interpolation system
            from parameters.shipshape.ship_parameter_interpolation_class import ShipParameterInterpolator

            # Create interpolator
            interpolator = ShipParameterInterpolator(
                shiptype=self.shiptype,
                shipdata=self.shipdata,
                verbose = self.verbose,
                debug=self.debug
            )

            # Collect parameters that user provided
            known_params = {}
            if self._input_L is not None:
                known_params['L'] = self._input_L
            if self._input_L_PP is not None:
                known_params['L_PP'] = self._input_L_PP
            if self._input_B is not None:
                known_params['B'] = self._input_B
            if self._input_T is not None:
                known_params['T'] = self._input_T
            if self._input_DWT is not None:
                known_params['DWT'] = self._input_DWT
            if self._input_gt is not None:
                known_params['GT'] = self._input_gt
            if self._input_ldt is not None:
                known_params['LDT'] = self._input_ldt

            self.logger.debug(f"Known parameters for interpolation: {list(known_params.keys())}")

            # Run interpolation
            if known_params:
                interpolator.set_parameters(**known_params)
                interpolator.predict_missing_parameters()

                # Store final parameters
                self._final_params = interpolator.get_parameters()

                self.logger.info("Parameter interpolation completed successfully")
                self.logger.debug(f"Final parameters: L={self._final_params['L']:.2f}, "
                                  f"B={self._final_params['B']:.2f}, T={self._final_params['T']:.2f}, "
                                  f"L_PP={self._final_params['L_PP']:.2f}, DWT={self._final_params['DWT']:.0f}, "
                                  f"GT={self._final_params['GT']:.0f}")
            else:
                self.logger.warning("No parameters provided for interpolation")
                self._final_params = known_params

        except ImportError:
            self.logger.warning("ShipParameterInterpolator not available, using provided parameters only")
            self._final_params = {
                'L': self._input_L,
                'L_PP': self._input_L_PP,
                'B': self._input_B,
                'T': self._input_T,
                'DWT': self._input_DWT,
                'GT': self._input_gt,
                'LDT': self._input_ldt,
            }
        except Exception as e:
            self.logger.warning(f"Parameter interpolation failed: {e}, using provided parameters only")
            self._final_params = {
                'L': self._input_L,
                'L_PP': self._input_L_PP,
                'B': self._input_B,
                'T': self._input_T,
                'DWT': self._input_DWT,
                'GT': self._input_gt,
                'LDT': self._input_ldt,
            }

    def _create_ship_record(self) -> pd.DataFrame:
        """Create ship record containing all parameters."""

        # Primary parameters from interpolation/input
        ship_data = {
            'L': self._final_params['L'],
            'L_PP': self._final_params['L_PP'],
            'L_WL': self._calculate_L_WL(L_PP=self._final_params['L_PP']),
            'B': self._final_params['B'],
            'T': self._final_params['T'],
            'DWT': self._final_params['DWT'],
            'GT': self._final_params['GT'],
            'LDT': self._final_params['LDT'],
            'C_M': self._input_C_M,
            'C_B': self._input_C_B,
        }

        # Calculate derived parameters
        if self._input_C_M is None:
            ship_data['C_M'] = self._calculate_C_M(ship_data)

        if self._input_C_B is None:
            ship_data['C_B'] = self._calculate_C_B(ship_data)



        # Add metadata
        ship_data.update({
            'ship_type': self.shiptype.name,
            'TYPE_KEY': self.shiptype.key,
            'created_timestamp': pd.Timestamp.now()
        })

        self.logger.debug(f"Created ship record with parameters: {list(ship_data.keys())}")

        return ship_data

    def _calculate_C_B(self, params: dict[str, float] = None) -> float:
        """Calculate block coefficient using provided or current parameters.

        :TODO: Katsoulis formula DOI:10.6911/WSRJ.202507_11(7).0007

        Vs = a1*C_B**a2*LOA**a3*B**a4*T**a5

        where vs is service speed

        for Bulk carrier = [3.7682,-0.3236,0.3566,-0.0455,-0.1866]
        for Container Ship = [0.9804,-0.9809,0.7416,-0.3742,-0.025]
        for Oil tanker = [1.633,-0.3280,0.7256,-0.5122,-0.0214]

        else [0.7261,-1.63,0.6846,-0.5007,0.2805]
        """
        if params is None:
            params = self._ship_record

        try:
            L = params['L']
            B = params['B']
            T = params['T']
            DWT = params['DWT']
            LDT = params['LDT']

            if self.service_speed is not None:
                self.logger.debug('Calculating C_B using service speed and Katsoulis formulation')

                if self.shiptype == ShipType.TANKER:

                    coeff = [1.633,-0.3280,0.7256,-0.5122,-0.0214]

                elif self.shiptype == ShipType.BULK_CARRIER:
                    coeff = [3.7682,-0.3236,0.3566,-0.0455,-0.1866]

                elif self.shiptype == ShipType.CONTAINER_SHIP:
                    coeff = [0.9804,-0.9809,0.7416,-0.3742,-0.025]

                else:
                    coeff = [0.7261,-1.63,0.6846,-0.5007,0.2805]

                a1,a2,a3,a4,a5 = coeff

                C_B = (self.service_speed / (a1 * (L ** a3) * (B ** a4) * (T ** a5))) ** (1/a2)

                self.logger.info(f'Calculating C_B using service speed and Katsoulis formulation. C_B = {C_B:.2f}')
                return C_B


            elif all([L, B, T, DWT]) and all(x > 0 for x in [L, B, T, DWT]):

                if LDT is None:
                    LDT = 0.0
                    self.logger.warning('LDT is not defined, using 0 value, DWT_v will not be correct (affects C_B estimation)')
                DWT_v = (DWT +LDT)/ self.rho_SW
                C_B = DWT_v / (L * B * T)
                return C_B
            else:
                return 0.0
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_C_M(self, params: dict[str, float] = None) -> float:

        """
        :TODO: Currently only unsing Jensen

        Theres a few from 'Ship Design for Efficiency and Economy' H. Schneekluth and V. Bertram


        :return:
        """
        if self._input_C_B is None:
            self.logger.debug('Using calculated reccomendation of C_B for C_M estimation')
            C_B = self._calculate_C_B(params)
        else:
            C_B = self._input_C_B

        C_M_jensen = 0.4759*(C_B**3)-1.6567*(C_B**2)+(1.7973*C_B)+0.3757

        if self.debug:

            C_M = {}

            C_M['Jensen'] = C_M_jensen

            C_M1 = 0.9 + 0.1 * C_B
            # if 0.92 <= C_M1 <= 1.0:
            C_M['estimate 1']= C_M1

            C_M2 = 1 / (1 + (1 - C_B) ** 3.5)
            # if 0.92 <= C_M2 <= 1.0:
            C_M['estimate 2']= C_M2

            C_M3 = 1.006 - (0.0056 * C_B ** (-3.56))
            # if 0.92 <= C_M3 <= 1.0:
            C_M['estimate 3']= C_M2

            self.logger.debug(f'Calculated C_M estimates:'
                              f'{pd.DataFrame(C_M)}')

        return C_M_jensen

    def _calculate_L_WL(self, L_PP:float) -> float:
        """
        Calculates the wetted length of the ship, according to formula given by Kristensen and Lutzen (2012)

        :param L_PP: Length between perpendiculars (m)
        """

        if self.shiptype in [ShipType.CONTAINER_SHIP, ShipType.RO_RO]:
            L_wl = 1.01*L_PP
        else:
            L_wl = 1.02*L_PP

        self.logger.debug(f'Calculating L_W as {L_wl}')

        return L_wl

    def re_interpolate_parameters(self):
        """Re-run interpolation with current parameter values."""
        if hasattr(self, '_interpolator') and self._interpolator is not None:
            # Get current values
            current_params = {
                'L': self.L,
                'DWT': self.DWT,
                'GT': self.gt,
                'L_PP': self.L_PP,
                'B': self.B,
                'T': self.T
            }

            # Re-run interpolation
            self._interpolator.set_parameters(**current_params)
            self._interpolator.predict_missing_parameters()

            # Update ship record
            updated_params = self._interpolator.get_parameters()
            for param, value in updated_params.items():
                self._ship_record[param] = value

            self._recalculate_derived_parameters()

    def _recalculate_derived_parameters(self):
        """Recalculate derived parameters after primary parameter changes."""
        self._ship_record['C_B'] = self._calculate_C_B()
        self._ship_record['C_M'] = self._calculate_C_M()
        self._ship_record['L_WL'] = self._calculate_L_WL()

        self.logger.debug("Derived parameters recalculated")

    # ===== PROPERTIES (DATABASE STORAGE ONLY) =====

    @property
    def L(self) -> float:
        """Get the ship length."""
        return self._ship_record['L']

    @L.setter
    def L(self, value: float):
        """Set the ship length."""
        self.logger.info(f"Setting L to {value:.4f}")
        self._ship_record['L'] = value
        self._recalculate_derived_parameters()

    @property
    def B(self) -> float:
        """Get the ship breadth."""
        return self._ship_record['B']

    @B.setter
    def B(self, value: float):
        """Set the ship breadth."""
        self.logger.info(f"Setting B to {value:.4f}")
        self._ship_record['B'] = value
        self._recalculate_derived_parameters()

    @property
    def T(self) -> float:
        """Get the ship draught."""
        return self._ship_record['T']

    @T.setter
    def T(self, value: float):
        """Set the ship draught."""
        self.logger.info(f"Setting T to {value:.4f}")
        self._ship_record['T'] = value
        self._recalculate_derived_parameters()

    @property
    def L_PP(self) -> float:
        """Get the length between perpendiculars."""
        return self._ship_record['L_PP']

    @L_PP.setter
    def L_PP(self, value: float):
        """Set the length between perpendiculars."""
        self.logger.info(f"Setting L_PP to {value:.4f}")
        self._ship_record['L_PP'] = value
        self._recalculate_derived_parameters()

    @property
    def DWT(self) -> float:
        """Get the deadweight tonnage."""
        return self._ship_record['DWT']

    @DWT.setter
    def DWT(self, value: float):
        """Set the deadweight tonnage."""
        self.logger.info(f"Setting DWT to {value:.4f}")
        self._ship_record['DWT'] = value
        self._recalculate_derived_parameters()

    @property
    def gt(self) -> float:
        """Get the gross tonnage."""
        return self._ship_record['GT']

    @gt.setter
    def gt(self, value: float):
        """Set the gross tonnage."""
        self.logger.info(f"Setting GT to {value:.4f}")
        self._ship_record['GT'] = value

    @property
    def ldt(self) -> float:
        """Get the gross tonnage."""
        return self._ship_record['GT']

    @gt.setter
    def ldt(self, value: float):
        """Set the gross tonnage."""
        self.logger.info(f"Setting LDT to {value:.4f}")
        self._ship_record['GT'] = value

    @property
    def C_M(self) -> float:
        """Get the midship coefficient."""
        return self._ship_record['C_M']

    @C_M.setter
    def C_M(self, value: float):
        """Set the midship coefficient."""
        self.logger.info(f"Setting C_M to {value:.4f}")
        self._ship_record['C_M'] = value
        self._recalculate_derived_parameters()

    # ===== CALCULATED PROPERTIES =====

    def C_B(self) -> float:
        """Get the block coefficient.

        """
        return self._ship_record['C_B']

    def L_WL(self) -> float:
        """
        Get the wetted length
        """

        return self._ship_record['L_WL']










