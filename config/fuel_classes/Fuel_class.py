r"""
Ship Fuel Modeling System
==========================

This module provides a fuel object for modelling and analysing ship fuels,
including energy calculations, emission factors, storage requirements, and cost analysis.

.. module author:: Ben Noble, TU Delft
.. version:: v1.2
.. date:: 31-05-2025

Classes
-------

.. class:: FuelType

   Enumeration of standard marine fuel types with their thermodynamic and emission properties.
   Includes hydrogen, methane, methanol, ammonia, diesel, and heavy fuel oil.

.. class:: Fuel

   Main class for representing fuel instances with detailed properties and calculation methods.
   Supports fuel loading from databases, energy calculations, emission analysis, and cost modeling.

Key Features
------------

* **Fuel Type Recognition**: Intelligent identification of fuels from various naming conventions
  and chemical formulas (e.g., "LNG", "CH4", "natural gas" all map to methane)

* **Energy Calculations**: Calculate fuel mass and volume requirements for given energy needs

* **Storage Analysis**: Determine tank volumes accounting for geometry and filling factors

* **Environmental Impact**: Calculate CO2 emissions (tank-to-wake and well-to-wake)

* **Economic Analysis**: Fuel cost calculations and comparative analysis

* **Database Integration**: Load fuel properties from CSV, Excel, or JSON databases

* **Fuel Registry**: Centralized registry system for managing fuel instances

Typical Usage
-------------

.. code-block:: python

   # Load fuels from database
   fuels = Fuel.from_database('fuel_database.csv')

   # Get specific fuel
   diesel = Fuel.get_fuel('MDO')

   # Calculate requirements for voyage
   energy_needed = 1500  # kWh
   fuel_mass = diesel.mass_of_fuel(energy_needed)  # kg
   tank_volume = diesel.required_tank_volume(energy_needed)  # m³
   co2_emissions = diesel.mass_of_co2(energy_needed)  # kg CO2
   fuel_cost = diesel.cost_of_fuel(energy_needed)  # euros

   # Compare fuel options
   hydrogen = Fuel.get_fuel('LH2')
   print(f"Diesel: {diesel.EF_co2} kg CO2/kg fuel")
   print(f"Hydrogen: {hydrogen.EF_co2} kg CO2/kg fuel")

Supported Fuel Types
--------------------

:Hydrogen (H2): 33.33 kWh/kg, zero emissions
:Methane (CH4): 13.89 kWh/kg, includes LNG, CNG, natural gas
:Methanol (CH3OH): 5.52 kWh/kg, renewable fuel option
:Ammonia (NH3): 5.17 kWh/kg, zero carbon emissions
:Marine Diesel: 11.9 kWh/kg, conventional marine fuel
:Heavy Fuel Oil (HFO): 11.5 kWh/kg, traditional bunker fuel

Database Format
---------------

The module supports loading fuel data from CSV/Excel files with columns such as:

* ``name``, ``fuel_name``, ``fuel``: Fuel identifier
* ``type``, ``formula``: Fuel type specification
* ``lhv``, ``LHV``: Lower heating value (kWh/kg)
* ``rho``, ``density``: Fuel density (kg/m³)
* ``EF_co2``, ``cf_co2``: CO2 emission factor (kg CO2/kg fuel)
* ``price``, ``p_fuel``: Fuel price (euros/kWh)

Units
-----

.. list-table::
   :header-rows: 1
   :widths: 20 20

   * - Quantity
     - Unit
   * - Energy
     - kWh
   * - Mass
     - kg
   * - Volume
     - m³
   * - Density
     - kg/m³
   * - Emission factors
     - kg pollutant/kg fuel
   * - Price
     - euros/kWh

Notes
-----

.. important::
   * Emission factors are based on IMO Fourth GHG Study (2020)
   * LHV values taken from project deliverable D1.3
   * Tank filling factors default to 95% for safety margins
   * Energy density validation prevents unphysical fuel properties

.. warning::
   All fuel instances are automatically registered in the class registry.
   Creating fuels with duplicate names will overwrite previous instances.

See Also
--------

* :mod:`pandas`: For database loading functionality
* :mod:`pathlib`: For file path handling
* :mod:`enum`: For FuelType enumeration

Examples
--------

Creating a custom fuel:

.. code-block:: python

   custom_fuel = Fuel(
       name="Custom Biodiesel",
       fuel_type=FuelType.DIESEL,
       lhv=11.5,
       rho=880,
       EF_co2=2.8
   )

Loading and analyzing multiple fuels:

.. code-block:: python

   # Load all fuels
   fuels = Fuel.from_database('fuel_data.csv')

   # Compare emissions for same energy requirement
   energy = 1000  # kWh
   for fuel_name in Fuel.list_available_fuels():
       fuel = Fuel.get_fuel(fuel_name)
       emissions = fuel.mass_of_co2(energy)
       print(f"{fuel_name}: {emissions:.2f} kg CO2")
"""

import os
import sys
import pandas as pd
from enum import Enum
import re
from pathlib import Path
from typing import Union

# Setting direction so python can find the damn folders  that are right f***ing next to it....
os.chdir(r"/")


sys.path.append(r"/config")
sys.path.append(r"/equipment")

#
from config.loggermixin.logger_config import LoggerMixin


class FuelType(Enum):
    """
    Enumeration of fuel types with their properties.
    Each fuel type stores:
    - Chemical formula
    - Lower Heating Value (LHV) in kWh/kg
    - Emission Factor (EF) in kg CO2/kWh
    - Density in kg/liter (for volume-to-mass conversions)
    """
    # Format: (formula, lhv_kwh_per_kg, ef_kg_co2_per_kwh, density_kg_per_liter)
    HYDROGEN = ("h2", 33.33, 0, 0.071)  # Liquid hydrogen at -253°C
    METHANE = ("ch4", 13.89, 2.75, 0.42)  # LNG at -162°C
    METHANOL = ("ch3oh", 5.52, 1.38, 0.79)  # Methanol
    AMMONIA = ("nh3", 5.17, 0, 0.68)  # Ammonia
    DIESEL = ("diesel", 11.9, 3.15, 0.85)  # Marine diesel oil (MDO/MGO)
    HFO = ("hfo", 11.5, 3.2, 0.99)  # Heavy/Intermediate fuel oil
    BIOFUEL = ("biofuel", 11.0, 0.5, 0.88)  # Bio-diesel (approximate values)

    def __init__(self, formula: str, lhv: float, ef: float, density: float):
        """
        Initialize fuel type with its properties.

        :param formula: Chemical formula
        :param lhv: Lower Heating Value in kWh/kg
        :param ef: Emission factor in kg CO2/kWh
        :param density: Density in kg/liter
        """
        self.formula = formula
        self.lhv = lhv  # Specific energy (kWh/kg)
        self.ef = ef  # Emission factor
        self.density = density  # kg/liter

    @property
    def volumetric_energy_density(self) -> float:
        """
        Calculate volumetric energy density in kWh/liter.

        :return: Energy density in kWh/liter
        """
        return self.lhv * self.density

    @property
    def display_name(self) -> str:
        """
        Get a human-readable name for the fuel type.

        :return: Display name
        """
        name_mapping = {
            'HYDROGEN': 'Hydrogen',
            'METHANE': 'Natural Gas (LNG/CNG)',
            'METHANOL': 'Methanol',
            'AMMONIA': 'Ammonia',
            'DIESEL': 'Marine Diesel (MDO/MGO)',
            'HFO': 'Heavy/Intermediate Fuel Oil',
            'BIOFUEL': 'Biofuel'
        }
        return name_mapping.get(self.name, self.name)

    @staticmethod
    def _normalize_string(s: str|None) -> str:
        """
        Normalize a string for comparison.

        :param s: Input string
        :return: Normalized string (lowercase, trimmed, single spaces)
        """
        if not isinstance(s, str):
            return ""
        # Convert to lowercase, remove extra spaces, replace underscores with spaces
        return re.sub(r'\s+', ' ', s.lower().strip()).replace('_', ' ')

    @staticmethod
    def _extract_formula(text: str) -> str|None:
        """
        Extract chemical formula from text using regex.

        :param text: Text potentially containing a chemical formula
        :return: Extracted formula or None
        """
        if not isinstance(text, str):
            return None
        # Match common chemical formulas (H2, CH4, C2H5OH, NH3, etc.)
        formula_pattern = r'(?<![A-Za-z])([A-Z][a-z]?\d*)+(?![A-Za-z])'
        match = re.search(formula_pattern, text)
        return match.group(0).lower() if match else None

    @classmethod
    def from_identifier(cls, name: str|None = None,
                        formula: str|None = None) -> 'FuelType|None':
        """
        Determine the FuelType based on the fuel name or chemical formula.
        Handles various naming conventions including sulfur content variants (VLS, ULS).

        :param name: Name of the fuel (e.g., 'LNG', 'VLS MDO', 'IFO 380')
        :param formula: Chemical formula of the fuel (e.g., 'CH4', 'H2')
        :return: Matching FuelType or None if no match found
        """
        # Normalize inputs
        name_normalized = cls._normalize_string(name)
        formula_normalized = cls._normalize_string(formula)

        # If formula not provided, try to extract from name
        if not formula_normalized and name:
            extracted = cls._extract_formula(name)
            if extracted:
                formula_normalized = extracted

        # Check by formula first (most precise)
        if formula_normalized:
            for fuel_type in cls:
                if fuel_type.formula == formula_normalized:
                    return fuel_type

        # Comprehensive mapping of fuel names and aliases
        fuel_name_mapping = {
            # Hydrogen and variants
            'h2': cls.HYDROGEN,
            'hydrogen': cls.HYDROGEN,
            'lh2': cls.HYDROGEN,
            'liquid hydrogen': cls.HYDROGEN,

            # Methane/Natural Gas and variants
            'ch4': cls.METHANE,
            'methane': cls.METHANE,
            'lng': cls.METHANE,
            'cng': cls.METHANE,
            'natural gas': cls.METHANE,
            'liquefied natural gas': cls.METHANE,
            'compressed natural gas': cls.METHANE,

            # Methanol and variants
            'ch3oh': cls.METHANOL,
            'methanol': cls.METHANOL,
            'meoh': cls.METHANOL,

            # Ammonia and variants
            'nh3': cls.AMMONIA,
            'ammonia': cls.AMMONIA,

            # Diesel family (MDO/MGO) - includes all sulfur variants
            'diesel': cls.DIESEL,
            'mdo': cls.DIESEL,
            'mgo': cls.DIESEL,
            'marine diesel oil': cls.DIESEL,
            'marine gas oil': cls.DIESEL,
            # Very Low Sulfur variants
            'vls mdo': cls.DIESEL,
            'vls mgo': cls.DIESEL,
            # Ultra Low Sulfur variants
            'uls mdo': cls.DIESEL,
            'uls mgo': cls.DIESEL,

            # HFO family (IFO/HFO) - includes all sulfur variants and viscosity grades
            'hfo': cls.HFO,
            'heavy fuel oil': cls.HFO,
            'bunker': cls.HFO,
            'bunker fuel': cls.HFO,
            'ifo': cls.HFO,
            'intermediate fuel oil': cls.HFO,
            # Very Low Sulfur variants
            'vls ifo': cls.HFO,
            'vls hfo': cls.HFO,
            # Ultra Low Sulfur variants
            'uls ifo': cls.HFO,
            'uls hfo': cls.HFO,
            # Viscosity grades (with and without spaces)
            'ifo 180': cls.HFO,
            'ifo180': cls.HFO,
            'ifo 380': cls.HFO,
            'ifo380': cls.HFO,

            # Biofuel variants
            'biofuel': cls.BIOFUEL,
            'bio-fuel': cls.BIOFUEL,
            'biodiesel': cls.BIOFUEL,
            'bio-diesel': cls.BIOFUEL,
            'bio diesel': cls.BIOFUEL,
        }

        # Try exact match
        if name_normalized in fuel_name_mapping:
            return fuel_name_mapping[name_normalized]

        # Try partial match (check if any key is contained in the name)
        for key, value in fuel_name_mapping.items():
            if key in name_normalized:
                return value

        return None

    @classmethod
    def from_identifier_with_fallback(cls, name: str|None = None,
                                      formula: str|None = None,
                                      default: 'FuelType|None' = None,
                                      log_unknown: bool = False) -> 'FuelType|None':
        """
        Like from_identifier but with a fallback option for unmatched fuels.
        Useful for handling unknown fuels gracefully and logging them.

        :param name: Name of the fuel
        :param formula: Chemical formula of the fuel
        :param default: Default FuelType to return if no match found
        :param log_unknown: If True, print a warning for unknown fuels
        :return: Matching FuelType, default, or None
        """
        result = cls.from_identifier(name, formula)

        if result is None:
            if log_unknown:
                print(f"Warning: Unknown fuel '{name}' (formula: '{formula}'). Using default.")
            return default

        return result


class Fuel(LoggerMixin):
    """
    A class representing fuel properties used in ship energy systems.

    """

    fuel_registry = {}

    def __init__(self,
                 name: str,
                 fuel_type: FuelType = None,
                 lhv: float = None,
                 rho: float = None,
                 EF_co2: float = None,
                 filling_factor: float = 0.95,
                 phase: str = 'liquid',
                 engine_cycle: str = 'otto',
                 EF_SO2: float = None,
                 EF_NOx: float = None,
                 EF_CH4: float = None,
                 EF_N2O: float = None,
                 wtt_EF_co2: float = None,
                 p_fuel: float = 0,
                 verbose: bool = True,
                 debug: bool = False
                 ):
        """
        Initialize a Fuel object with specified attributes.
        Handles immediate type checking and simple range validation

        :param name: Name of the fuel.
        :type name: str

        :param lhv: Lower calorific value (LHV) of the fuel in kilowatt hour per kilogram (kWh/kg).
        :type lhv: float

        :param rho: Density of the fuel in kilograms per cubic meter (kg/m**3).
        :type rho: float

        :param wtt_EF_co2: Well to Wake Emission factor of the fuel in tons of CO2 equivalent per ton of fuel.
        :type wtt_EF_co2: float

        :param EF_co2: tank to wake Emission factor of the fuel in tons CO2 equivalent per ton of fuel, based on IMO
        fourth GHG
        study, 2020.
        :type EF_co2: float

        :param EF_SO2: tank to wake Emission factor of the fuel in tons of SOx equivalent per ton of fuel
        :type EF_SO2: float

        :param EF_NOx: tank to wake Emission factor of the fuel in tons of NOX equivalent per ton of fuel.
        :type EF_NOx: float

        :param EF_CH4: tank to wake emission factor of the fuel in tons of CH4 per ton of fuel

        :param EF_N2O: tank to wake Emission factor of the fuel in tons of NOX equivalent per ton of fuel.
        :type EF_N2O: float

        :param p_fuel: Price of the fuel in euros per kWh ($/kWh), as of 2020 and 2030 based on Baldi et al.
        :type p_fuel: float

        :param engine_cycle: The engine cycle that the fuel is used in. the emission factors are affected by this.
        :type engine_cycle: str

        """

        # Basic type validation
        if not isinstance(name, str):
            raise TypeError('Name must be a string.')
        if fuel_type is not None and not isinstance(fuel_type, FuelType):
            raise TypeError('Fuel type must be a FuelType object or None.')

        # Set basic attributes
        self.name = name
        self.fuel_type = fuel_type

        # Set LHV with fallback to fuel_type default
        self.LHV = lhv if lhv is not None else (fuel_type.lhv if fuel_type else None)

        # Set EF_co2 with fallback to fuel_type default
        self.EF_co2 = EF_co2 if EF_co2 is not None else (fuel_type.ef if fuel_type else None)

        # Set other attributes
        self.rho = rho
        self.p_fuel = p_fuel
        self.filling_factor = filling_factor
        self.phase = phase
        self.engine_cycle = engine_cycle
        self.EF_SO2 = EF_SO2
        self.EF_NOx = EF_NOx
        self.EF_CH4 = EF_CH4
        self.EF_N2O = EF_N2O
        self.wtt_EF_co2 = wtt_EF_co2

        # Convert any integers to floats
        try:
            if self.LHV is not None:
                self.LHV = float(self.LHV)
            if self.rho is not None:
                self.rho = float(self.rho)
            if self.p_fuel is not None:
                self.p_fuel = float(self.p_fuel)
            if self.EF_co2 is not None:
                self.EF_co2 = float(self.EF_co2)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All numerical values must be convertible to float: {str(e)}")

        # Validate required parameters
        if self.rho is not None and self.rho <= 0:
            raise ValueError('rho must be greater than zero.')

        if self.p_fuel is not None and self.p_fuel < 0:
            raise ValueError("Prices cannot be negative")

        if self.EF_co2 is not None and self.EF_co2 < 0:
            raise ValueError("Emission factor cannot be negative")

        # Cross-validate energy density if we have both LHV and rho
        if self.LHV is not None and self.rho is not None:
            energy_density = self.LHV * self.rho
            if energy_density > 120_000_000:  # Maximum theoretical energy density in kJ/m³
                raise ValueError(
                    f"Calculated energy density {energy_density} kJ/m³ exceeds "
                    "physical limits for chemical fuels")

        # Register this fuel instance in the class registry ONLY after all validation passes
        Fuel.fuel_registry[self.name] = self

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel("INFO")
        elif self.debug:
            self.logger.setLevel("DEBUG")
        else:
            self.logger.setLevel("WARNING")
    @classmethod
    def from_database(cls,
                      database_path: Union[str,Path],
                      filter_criteria=None):
        """
        Create fuel instances from a database file and return them as a dictionary.

        :param database_path: Path to the database file (CSV, JSON, etc.)
        :type database_path: str
        :param filter_criteria: Optional criteria to filter which fuels to load
        :type filter_criteria: dict or None

        :return: Dictionary of fuel instances with fuel names as keys
        :rtype: dict
        """
        database_path = Path(database_path)

        file_extension = database_path.suffix

        if file_extension == '.csv':
            # Special handling for your CSV format with semicolon delimiter
            fuel_data = pd.read_csv(database_path, sep=';')
        elif file_extension == '.json':
            fuel_data = pd.read_json(database_path)
        elif file_extension in ['.xlsx', '.xls']:
            fuel_data = pd.read_excel(database_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Apply filters if provided
        if filter_criteria:
            for column, value in filter_criteria.items():
                if column in fuel_data.columns:
                    if isinstance(value, (list, tuple)):
                        fuel_data = fuel_data[fuel_data[column].isin(value)]
                    else:
                        fuel_data = fuel_data[fuel_data[column] == value]

        # Create fuel instances
        fuel_instances = {}

        # Identify columns that might contain chemical formulas
        formula_cols = [col for col in fuel_data.columns if any(
            c in col.lower() for c in ['formula', 'chem', 'molecule', 'compound', 'type'])]

        # Helper to find any chemical formula in a text string using regex
        def extract_formula(text):
            if not isinstance(text, str):
                return None
            # Match common chemical formulas (like H2, CH4, C2H5OH, NH3, etc.)
            formula_pattern = r'(?<![A-Za-z])([A-Z][a-z]?\d*)+(?![A-Za-z])'
            match = re.search(formula_pattern, text)
            return match.group(0) if match else None

        for _, row in fuel_data.iterrows():
            try:
                # Convert row to dictionary
                fuel_dict = row.to_dict()

                # Extract required parameters with multiple potential column names
                name = fuel_dict.get('name') or fuel_dict.get('fuel_name') or fuel_dict.get('fuel')
                if not name:
                    print(f"Skipping row - no name found: {fuel_dict}")
                    continue  # Skip entries without a name


                formula = None
                for col in formula_cols:
                    if col in fuel_dict and fuel_dict[col]:
                        formula = fuel_dict[col]
                        break

                # If no formula found in dedicated columns, try to extract from name or description
                if not formula:
                    description = fuel_dict.get('description', "")
                    formula = extract_formula(name) or extract_formula(description)


                fuel_type_str = fuel_dict.get('type')
                if fuel_type_str:
                    fuel_type = FuelType.from_identifier(fuel_type_str)
                else:
                    # If no type column, try name/formula
                    fuel_type = FuelType.from_identifier(name, formula)

                # Special case for methanol - check for 'MeOH' explicitly
                if not fuel_type and (name == 'MeOH' or fuel_type_str == 'methanol'):
                    fuel_type = FuelType.METHANOL

                # Extract parameters with fallbacks and multiple potential column names
                lhv = fuel_dict.get('lhv') or fuel_dict.get('LHV') or fuel_dict.get('lower_heating_value')
                if lhv is None and fuel_type:
                    lhv = fuel_type.lhv

                rho = fuel_dict.get('rho') or fuel_dict.get('density')

                # For EF_co2, check different possible column names based on your CSV
                ef_co2 = (fuel_dict.get('EF_co2') or
                          fuel_dict.get('emission_factor_co2') or
                          fuel_dict.get('co2_factor') or
                          fuel_dict.get('cf_co2'))

                if ef_co2 is None and fuel_type:
                    ef_co2 = fuel_type.ef

                # Extract optional parameters with defaults
                filling_factor = fuel_dict.get('filling_factor', 0.95)
                phase = fuel_dict.get('phase', 'liquid')


                engine_cycle = fuel_dict.get('engine_cycle', 'otto')

                # For other emission factors, check different possible column names
                ef_so2 = fuel_dict.get('EF_SO2') or fuel_dict.get('emission_factor_so2')
                ef_nox = fuel_dict.get('EF_NOx') or fuel_dict.get('emission_factor_nox')


                wtt_ef_co2 = fuel_dict.get('wtt_EF_co2') or fuel_dict.get('wtt_emission_factor')
                p_fuel = fuel_dict.get('p_fuel') or fuel_dict.get('price') or 0



                # Debug logging


                # Check if we have the minimum required parameters
                if name and ((lhv is not None and rho is not None and ef_co2 is not None) or fuel_type):
                    try:
                        # Convert string values to appropriate types
                        if isinstance(lhv, str):
                            lhv = float(lhv)
                        if isinstance(rho, str):
                            rho = float(rho)
                        if isinstance(ef_co2, str):
                            ef_co2 = float(ef_co2)

                        fuel = cls(
                            name=name,
                            fuel_type=fuel_type,
                            lhv=lhv,
                            rho=rho,
                            EF_co2=ef_co2,
                            filling_factor=filling_factor,
                            phase=phase,
                            engine_cycle=engine_cycle,
                            EF_SO2=ef_so2,
                            EF_NOx=ef_nox,
                            wtt_EF_co2=wtt_ef_co2,
                            p_fuel=p_fuel
                        )
                        fuel_instances[name] = fuel
                        print(f"Created fuel instance: {name}")
                    except Exception as e:
                        print(f"Error creating fuel instance for {name}: {str(e)}")
                else:
                    pass
            except Exception as e:
                print(f"Error creating fuel instance for {name}: {str(e)}")

        return fuel_instances

    @classmethod
    def get_fuel(cls, name):
        """
        Get a fuel instance by name from the registry.

        :param name: Name of the fuel to retrieve
        :type name: str

        :return: Fuel instance
        :rtype: Fuel
        """
        if name in cls.fuel_registry:
            return cls.fuel_registry[name]
        else:
            raise KeyError(f"Fuel '{name}' not found in the registry")

    @classmethod
    def list_available_fuels(cls):
        """
        Return a list of all available fuel names in the registry.

        :return: List of fuel names
        :rtype: list
        """
        return list(cls.fuel_registry.keys())

    @property
    def cEDV(self) -> float:
        """
        Calculate and return the contained volumetric energy density of the fuel.

        :return: Contained volumetric energy density of the fuel in gigajoules per cubic meter (kWh/m**3).
        :rtype: float
        """
        return self.LHV * self.rho

    def mass_of_fuel(self,
                     energy_required: float) -> float:
        """
        Calculate and return the mass of fuel required for a given energy requirement.
        :param energy_required: given energy requirement in kWh
        :type energy_required: float

        :return: mass of fuel required in kilograms (kg)
        :rtype: float
        """
        self.logger.info("returning mass of fuel required in kilograms (kg), based on given energy requirement"
                         "which should be in kWh ")

        mass_required =  energy_required/self.LHV

        return mass_required

    def volume_of_fuel(self, energy_required: float,
                       tank_correction: float = 1.0) -> float:
        """
        Calculate and return the volume of fuel required for a given energy requirement,
        with an optional correction factor for tank geometry.

        :param energy_required: Given energy requirement in kWh
        :type energy_required: float

        :return: Volume of fuel required in cubic meters (m³)
        :rtype: float
        """
        # First calculate the mass of fuel needed
        mass_required = self.mass_of_fuel(energy_required)

        # Calculate the theoretical volume
        theoretical_volume = mass_required / self.rho

        return theoretical_volume

    def required_tank_volume(self,
                             energy_required: float,
                             tank_shape: str = "rectangular",
                             gas_correction: float = None) -> float:
        """
        Calculate the required tank volume considering both geometry and practical factors.

        :param energy_required: Given energy requirement in kWh
        :type energy_required: float
        :param tank_shape: Shape of tank, either "rectangular" or "cylindrical"
        :type tank_shape: str
        :param gas_correction: Additional correction for gaseous fuels (compressibility, etc.)
        :type gas_correction: float or None

        :return: Required tank volume in cubic meters (m³)
        :rtype: float
        """
        # First calculate the theoretical volume
        theoretical_volume = self.mass_of_fuel(energy_required) / self.rho

        # Determine gaseous fuel correction if not explicitly provided
        if gas_correction is None:
            if self.phase.lower() == 'gas':
                self.logger.warning('Using values that may not be right')
                gas_correction = 1.1 #Completely working value to check whether system works
            else:
                gas_correction = 1.0

        # Apply geometric correction based on tank shape
        geometric_factor = 0.785 if tank_shape.lower() == "cylindrical" else 1.0

        # Calculate final volume with all factors
        required_volume = theoretical_volume / (geometric_factor * self.filling_factor) * gas_correction

        return required_volume

    def mass_of_co2(self,
                    energy_required: float,
                    wtt: bool = False) -> float:
        """
        Calculate and return the co2 of fuel required for a given energy requirement.

        :param energy_required: given energy requirement in kWh
        :type energy_required: float
        :param wtt: User specifies whether to include the well to tank CO2 equivalent emission
        :type wtt: bool

        :return: the mass of co2 that is generated
        :rtype: float
        """

        if wtt:
            mass_co2_wtt = self.mass_of_fuel(energy_required) * self.wtt_EF_co2
            mass_co2_ttw = self.mass_of_co2(energy_required) * self.EF_co2
            return mass_co2_wtt + mass_co2_ttw
        else:
            mass_co2_ttw = self.mass_of_co2(energy_required) * self.EF_co2
            return mass_co2_ttw

    def cost_of_fuel(self,
                     energy_required: float) -> float:
        """
        Calculate and return the cost of fuel required for a given energy requirement.

        :param energy_required: given energy requirement in kWh
        :type energy_required: float
        :return: cost of fuel in euros
        :rtype: float
        """

        cost = ((self.mass_of_fuel(energy_required))/self.LHV) * self.p_fuel
        return cost

