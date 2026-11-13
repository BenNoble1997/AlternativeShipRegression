

# Modules imports.
import sys
import os

import pandas as pd

import logging
from pathlib import Path


from config.config_constants import SEAWATER_DENSITY
from config.logger_config import LoggerMixin

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





