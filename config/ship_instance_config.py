"""
This module contains configuration classes which are used for storing configurations that may be passed
to the ShipInstance builder class
"""
from enum import Enum

#This is the carbon factor of the grid
GRID_CF = 0.249
class ShipType(Enum):
    """
    Creates Enum object for ship type.

    format:
    name =
    - Capitalized first word
    - _ instead of space


    """
    BULK_CARRIER = ("Bulk_Carrier",1)
    TANKER = ("Tanker", 2)
    CRUISE_SHIP = ("Cruise_Ship",3)
    RO_RO = ("RO_RO",4)
    CARGO = ("Cargo",5)
    CONTAINER_SHIP = ('Container_Ship',6)
    OIL_TANKER = ('Oil_Tanker',7)
    CHEMICAL_TANKER = ('Chemical_Tanker',8)
    LNG_TANKER = ('LNG_Tanker',9)
    PASSENGER_SHIP = ('Passenger_Ship',10)
    GENERAL_SHIP = ('General_Ship',11)
    CAR_CARRIER = ('Carrier',12)
    OFFSHORE_SUPPLY = ('Offshore_Supply',13)
    DRILLING_SHIP = ('Drilling_Ship',14)
    RESEARCH_VESSEL = ('Research_Vessel_Ship',15)

    @property
    def name(self):
        return self.value[0]

    @property
    def key(self):
        return self.value[1]

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



