"""
Stores the ShipShapeExplorer class
"""

# Modules imports.
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable, Dict, Any

#
# os.chdir(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable")
#
#
# sys.path.append(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable\config")
# sys.path.append(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable\parameters")
# sys.path.append(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable\parameters\shipshape")

from config.config_constants import GRAVITY, SEAWATER_DENSITY, NU
from config.ship_instance_config import ShipType
from config.logger_config import LoggerMixin
from config.config_general_old import cached_property, cached_corrected_property

from parameters.shipshape.ShipShapeBasic_class import ShipShapeBasic


@dataclass
class CorrectionFactor(LoggerMixin):
    """
    Represents a correction to be applied to a ship parameter calculation.

    Attributes:
        parameter: The parameter being corrected
        correction_type: The type of correction ('offset', 'factor', or 'function')
        value: The correction value or function
        conditions: Optional conditions for when to apply the correction
    """
    parameter: str
    correction_type: str  # 'offset', 'factor', or 'function'
    value: Union[float, Callable]
    conditions: Optional[Dict[str, Any]] = None

    def apply(self,
              base_value: float,
              ship_params: Dict[str, Any]) -> float:
        """Apply the correction if conditions are met."""
        if not self._check_conditions(ship_params):
            return base_value

        if self.correction_type == 'offset':
            self.logger.info(f"offset {base_value} by {self.value}")
            return base_value + self.value

        elif self.correction_type == 'factor':
            self.logger.info(f"factor {base_value} by {self.value}")
            return base_value * self.value

        elif self.correction_type == 'function':
            self.logger.info(f" function {base_value} by {self.value}")

            return self.value(base_value, ship_params)

    def _check_conditions(self, ship_params: Dict[str, Any]) -> bool:
        """Check if conditions for applying the correction are met."""
        if not self.conditions:
            return True

        return all(
            ship_params.get(param) == value
            for param, value in self.conditions.items()
        )

class BowThrusters(LoggerMixin):
    """
    A class for storing the parameters of bow thrusters
    """
    def __init__(self,
                 diameter: float,
                 resistance_coefficient: float = 0.008):
        """
        initialising the bow thrusters class
        :param diameter: diameter of the bow thrusters in metres
        :param resistance_coefficient: resistance coefficient of the bow thrusters, defaults to 0.008
        """
        self._diameter = diameter
        self._resistance_coefficient = resistance_coefficient


class ShipShapeExplorer(LoggerMixin):
    """
    A class to represent the shape parameters of a ship. Takes a select number of initial parameters and calculates
    the rest based on best-fit line equations from data on existing ships, as well as equations given by sources such
    as Papanikolaou.
    This class provides options to explore different stern shapes, appendages, and other features.


    **Appendage Types:**

    - 1: Rudder behind skeg
    - 2: Rudder behind stern
    - 3: Twin-screw balance rudders
    - 4: Shaft brackets
    - 5: Skeg
    - 6: Strut bossings
    - 7: Hull bossings
    - 8: Shafts
    - 9: Stabilizer fins
    - 10: Dome
    - 11: Bilge keels

    **Stern Shape Choices:**

    - 1: Pram with gondola
    - 2: V-shaped section
    - 3: Normal section
    - 4: U-shaped section with Hogner stern

    **Ice Class Choices:**

    - IA_Super
    - IA
    - IB
    - IC
    """

    def __init__(
            self,
            v_knots: float,
            correction_library: Optional[Dict[str, CorrectionFactor]] = None,
            shipshape: ShipShapeBasic = None,
            shiptype: ShipType = None,
            C_sternchoice: int = 3,
            wake_fraction: float = None,
            thrust_deduction: float = None,
            app_types: Union[List[int], int, None] = None,
            bulbchoice: Optional[bool] = None,
            bow_thrusters: Union[BowThrusters, List[BowThrusters]] = None,
            BT_n: Optional[int] = None,
            BT_d: Optional[float] = None,
            C_BT: Optional[float] = None,
            L: Optional[float] = None,
            L_PP: Optional[float] = None,
            B: Optional[float] = None,
            T: Optional[float] = None,
            DWT: Optional[float] = None,
            C_M: Optional[float] = None,
            trim: Optional[float] = None,
            _DWT_v: Optional[float] = None,
            _T_F: Optional[float] = None,
            _T_A: Optional[float] = None,
            _A_BT: Optional[float] = None,
            _h_B: Optional[float] = None,
            _C_WP: Optional[float] = None,
            _A_T: Optional[float] = None,
            _A_m: Optional[float] = None,
            _A_W: Optional[float] = None,
            _lcb: Optional[float] = None,
            _S: Optional[float] = None,
            _S_APP: Optional[float] = None,
            ice_class: Optional[str] = 'None',
            verbose: bool = False,
            debug: bool = False
    ):
        """
        Initialisation of the ship shape explorer
        :param v_knots: Speed in knots
        :param correction_library: A dictionary of corrections to class formulas. Enables the user to specify if
        values are
        calculated in a different way than the class
        :param shipshape: An instance of ShipShapeBasic. Used for initial parameters.
        :param shiptype: The type of Ship. Must be in instance of ShipType
        :param C_sternchoice: The type of stern shape. Must be one of:
        **Stern Shape Choices:**

        - 1: Pram with gondola
        - 2: V-shaped section
        - 3: Normal section
        - 4: U-shaped section with Hogner stern

        :param app_types: An optional ist of appendage types. Must come from this list:
        **Appendage Types:**

        - 1: Rudder behind skeg
        - 2: Rudder behind stern
        - 3: Twin-screw balance rudders
        - 4: Shaft brackets
        - 5: Skeg
        - 6: Strut bossings
        - 7: Hull bossings
        - 8: Shafts
        - 9: Stabilizer fins
        - 10: Dome
        - 11: Bilge keels

        :param bulbchoice: Specify whether the ship includes a bulbous bow
        :param BT_n: Specifies number of bow thrusters
        :param BT_d: Specifies diameter of bow thrusters
        :param C_BT: Resistance coefficient for bow thrusters.
        :param L: Length of the ship, in metres
        :param L_PP: Length between perpendiculars, in metres
        :param B: Moulded breadth of the ship, in metres
        :param T: Moulded draft of the ship, in metres
        :param DWT: dead weght tonnage of the ship, in tonnes
        :param C_M: Midship coefficient of the ship. dimensionless. should be provided if ShipBasic class is not
        :param trim: Trim of the ship, in metres
        :param _DWT_v: Displacement volume of the ship, in m^3
        :param _T_F:Moulded forward draft of the ship, fore in metres
        :param _T_A:moulded aft draft of the ship, in metres
        :param _A_BT: transverse area in m^2 of the bulb
        :param _h_B: Position of the cebtre if te trabsverse area A_BT above the keel line
        :param _C_WP: Waterplane coefficient,
        :param _A_T:Transom area in m^2
        :param _A_m: Midship area in m^2
        :param _A_W:
        :param _lcb: logitudinal position of the centrre of buoyancy
        :param _S: The wetted area of the hull
        :param _S_APP: The wetted area of appendages
        :param ice_class: The ice class of the ship.
        :param verbose: boolean. if true, logger level is set to INFO.
        :param debug: boolean. If true, logger level is set to DEBUG.
        """
        self.verbose = verbose
        self.debug = debug
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        if shipshape is not None:
            self.logger.info("loading basic parameters from ship basic class")
            self.shiptype = shipshape.shiptype
            self.L = shipshape.L
            self.L_PP = shipshape.L_PP
            self.B = shipshape.B
            self.T = shipshape.T
            self.DWT = shipshape.DWT
            self.C_M = shipshape.C_M

        else:
            self.logger.warning("It is preferred that ship basic is used for initial design parameters")
            if shiptype is None:
                raise ValueError("shiptype undefined")
            else:
                self.shiptype = shiptype

            # Ensure that all required parameters are provided
            required_params = {
                "L": L, "L_PP": L_PP, "B": B, "T": T, "C_M": C_M
            }
            missing_params = [param for param, value in required_params.items() if value is None]

            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)} "
                                 "when ShipShape is not provided.")
            self.L = L
            self.L_PP = L_PP
            self.B = B
            self.T = T
            self.DWT = DWT
            self.C_M = C_M

        super().__init__()
        self.correction_library = correction_library or {}

        # Log information about applied corrections
        if correction_library:
            self.logger.info(f"Initialized with {len(correction_library)} corrections")
            for param, correction in correction_library.items():
                self.logger.debug(f"Correction registered for {param}: {correction}")

        self.shipshape = shipshape
        self.v_knots = v_knots
        self.C_sternchoice = C_sternchoice
        self.wake_fraction = wake_fraction
        self.thrust_deduction = thrust_deduction

        if isinstance(app_types, int):
            self.app_types = [app_types]  # Convert single integer to list
        elif app_types is None:
            self.app_types = [1]  # Default value
        else:
            self.app_types = app_types  # Use provided list

        self.bulbchoice = bulbchoice
        self.trim = trim
        self.bow_thrusters = bow_thrusters
        self.BT_n = BT_n
        self.BT_d = BT_d
        self.C_BT = C_BT
        self._DWT_v = _DWT_v
        self._T_F = _T_F
        self._T_A = _T_A
        self._A_m = _A_m
        self._A_W = _A_W
        self._A_BT = _A_BT
        self._h_B = _h_B
        self._C_WP = _C_WP
        self._A_T = _A_T
        self._lcb = _lcb
        self._S = _S
        self._S_APP = _S_APP
        self.ice_class = ice_class

        # Constants
        self.g = GRAVITY
        self.rho_SW = SEAWATER_DENSITY
        self.nu = NU

    def __post_init__(self):
        """
        Post-initialization processing for the `ShipShapeExplorer` class.

        This method performs additional validations and default assignments after the
        initialization of the `ShipShapeExplorer` object. It ensures consistency and
        validity of the parameters based on the provided and default values.

        **Validations and Assignments:**

        - Ensures that if either `BT_d` or `C_BT` is defined, both must be defined together.
        - Ensures that `BT_n` is defined if either `BT_d` or `C_BT` is provided.
        - Validates that the value of `C_BT` is within the expected range (0.003 - 0.012). A warning is logged if it
         is out of this range.
        - Ensures that `C_sternchoice` is one of the allowed values: [1, 2, 3, 4]. Raises a `ValueError` if the value
        is invalid.
        - Assigns default values from `shipshape` attributes if the corresponding parameters are not provided during
        initialization. This includes `L`, `L_PP`, `T`, `B`, `DWT`, `DWT_v`, and `C_M`.

        **Raises:**

        - `ValueError`: If `BT_d` and `C_BT` are inconsistently defined.
        - `ValueError`: If `BT_n` is not defined when either `BT_d` or `C_BT` is provided.
        - `ValueError`: If `C_sternchoice` is not one of the allowed values.

        **Logging:**

        - Logs a warning if `C_BT` is out of the expected range.

        **Note:**

        - This method is automatically called after the object is initialized due to the use of the `@dataclass`
         decorator.
        """

        # Ensure that BT_d and C_BT must be defined if either is defined
        if (self.BT_d is None) != (self.C_BT is None):
            raise ValueError("Both BT_d and C_BT must be either defined or undefined together."
                             )
        # Ensure that BT_n must be defined if either BT_d or C_BT is defined
        if (self.BT_d is not None or self.C_BT is not None) and self.BT_n is None:
            raise ValueError("BT_n must be defined if BT_d or C_BT is defined.")

        # Validate C_BT range or print a warning if it's not within the expected range
        if self.C_BT is not None:
            if not self.C_BT < 0.003 or self.C_BT > 0.012:
                self.logger.warning("Warning: C_BT is out of the expected range (0.003 - 0.012).")

        if self.C_sternchoice not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid C_sternchoice: {self.C_sternchoice}. Must be one of [1,2,3,4].")

    # Might consider using hasattr to reduce computation. That is for another time though
    def _apply_correction(self, parameter: str, base_value: float) -> float:
        """
        Applies registered corrections to a calculated parameter with caching.
        """
        # Create a unique cache key for this parameter and base value
        cache_key = f"_correction_cache_{parameter}_{base_value}"

        # Check if this specific correction is already cached
        if hasattr(self, cache_key):
            self.logger.debug(f"Using cached correction for {parameter}({base_value})")
            return getattr(self, cache_key)

        # No correction needed
        if parameter not in self.correction_library:
            setattr(self, cache_key, base_value)
            return base_value

        # Apply correction
        correction = self.correction_library[parameter]
        if not isinstance(correction, CorrectionFactor):
            self.logger.warning(f"Invalid correction type for {parameter}. Expected CorrectionFactor.")
            setattr(self, cache_key, base_value)
            return base_value

        current_params = self._get_current_params()

        try:
            self.logger.info(f"Applying correction to {base_value}")
            corrected_value = correction.apply(base_value, current_params)
            # Cache the result
            setattr(self, cache_key, corrected_value)
            return corrected_value
        except Exception as e:
            self.logger.warning(
                f"Failed to apply correction to {parameter}: {str(e)}. Using base value."
            )
            setattr(self, cache_key, base_value)
            return base_value

    # def _apply_correction(self, parameter: str, base_value: float) -> float:
    #     """
    #     Applies registered corrections to a calculated parameter.
    #     """
    #     if parameter not in self.correction_library:
    #         return base_value
    #
    #     correction = self.correction_library[parameter]
    #     if not isinstance(correction, CorrectionFactor):
    #         self.logger.warning(f"Invalid correction type for {parameter}. Expected CorrectionFactor.")
    #         return base_value
    #
    #     current_params = self._get_current_params()
    #
    #     try:
    #         self.logger.info(f"Applying correction to {base_value}")
    #         return correction.apply(base_value, current_params)
    #     except Exception as e:
    #         self.logger.warning(
    #             f"Failed to apply correction to {parameter}: {str(e)}. Using base value."
    #         )
    #         return base_value

    def _get_current_params(self) -> Dict[str, Any]:
        """
        Creates a dictionary of current parameter values needed for corrections.
        This method centralizes parameter access for correction calculations.
        """
        return {
            'V': self.v_knots,
            'T': self.T,
            'L': self.L,
            'B': self.B,
            'shiptype': self.shiptype,
            'C_M': self.C_M,
            'DWT': self.DWT,
            'C_sternchoice': self.C_sternchoice
            # Add other parameters as needed
        }

    def _invalidate_property_caches(self, *properties):
        """Invalidate caches for specific properties or all if none specified."""
        all_attrs = dir(self)

        if not properties:
            # Invalidate all property caches
            for attr in all_attrs:
                if (attr.startswith('_base_cached_') or
                        attr.startswith('_corrected_cached_') or
                        attr.startswith('_correction_key_') or
                        attr.startswith('_correction_cache_')):
                    delattr(self, attr)
            self.logger.debug("Invalidated all property caches")
        else:
            # Invalidate specific property caches
            for prop in properties:
                for prefix in ['_base_cached_', '_corrected_cached_', '_correction_key_']:
                    cache_attr = prefix + prop
                    if cache_attr in all_attrs:
                        delattr(self, cache_attr)

                # Also remove any correction cache entries for this property
                for attr in all_attrs:
                    if attr.startswith(f'_correction_cache_{prop}_'):
                        delattr(self, attr)

                self.logger.debug(f"Invalidated cache for property: {prop}")

    def clear_direct_value(self, property_name):
        """Clear a directly set property value, allowing calculation to take over."""
        direct_attr = f"_direct_{property_name}"
        if hasattr(self, direct_attr):
            delattr(self, direct_attr)
            self.logger.info(f"Cleared direct value for {property_name}")

            # Also clear cached values to force recalculation
            for prefix in ['_base_cache_', '_corrected_cache_', '_correction_state_']:
                attr = f"{prefix}{property_name}"
                if hasattr(self, attr):
                    delattr(self, attr)

    @property
    def v_ms(self) -> float:
        """Convert ship speed in knots to m/s.

        :Return: v
        :rtype: float
        """
        return self.v_knots * 0.5144

    # @cached_property
    # def DWT_v(self) -> float:
    #     """Get the moulded displacement in m³, calculating if needed."""
    #     if self._DWT_v is None:
    #         self.logger.info("DWT_v not provided, calculating from DWT...")
    #         self._DWT_v = self.DWT * 1000 / self.rho_SW
    #         self.logger.info(
    #             f"DWT_v calculated: {self._DWT_v:.4f} m³ (using DWT={self.DWT} tonnes, rho_SW={self.rho_SW} kg/m³)")
    #     else:
    #         self.logger.debug(f"Using provided DWT_v: {self._DWT_v:.4f} m³")
    #     return self._DWT_v
    #
    # @DWT_v.setter
    # def DWT_v(self, value: float):
    #     """Set the moulded displacement in m³."""
    #     self.logger.info(f"Setting DWT_v to {value:.4f} m³")
    #     self._DWT_v = value

    @cached_property
    def DWT_v(self):
        """Get the moulded displacement in m³, calculating if needed."""
        if hasattr(self, '_property_DWT_v'):
            return self._property_DWT_v

        if self._DWT_v is None:
            self.logger.info("DWT_v not provided, calculating from DWT...")
            return self.DWT * 1000 / self.rho_SW
        else:
            self.logger.debug(f"Using provided DWT_v: {self._DWT_v:.4f} m³")
            return self._DWT_v

    @DWT_v.setter
    def DWT_v(self, value):
        """Set the moulded displacement in m³."""
        self.logger.info(f"Setting DWT_v to {value:.4f} m³")
        self._property_DWT_v = value
        # Invalidate dependent properties
        self._invalidate_property_caches('C_B', 'k1', 'lcb')

    @property
    def Re(self) -> float:
        """Calculate the Reynolds number.

        :return: Reynolds number
        :rtype: float
        """
        return self.v_ms * self.L / self.nu

    @property
    def F_n(self) -> float:
        """Calculate the Froude number.

        :return: F_n
        :rtype: float
        """
        return self.v_ms / (np.sqrt(self.g * self.L))

    @property
    def C_stern(self) -> float:
        """fA
        Calculate the stern coefficient based on the choice.
        return: C_stern
        :rtype: float
        """
        stern_choices = {
            1: -25,  # pram with gondola
            2: -10,  # v-shaped section
            3: 0,  # normal section
            4: 10  # U-shaped section with Hogner stern
        }

        if not hasattr(self, '_C_stern'):
            _C_stern = stern_choices[self.C_sternchoice]
            self.logger.info(f"Stern choice: {self.C_sternchoice}, C_stern coefficient: {_C_stern}")
        return _C_stern

    @property
    def c14(self) -> float:
        """
        Calculate the c14 coefficient based on the stern choice.
        :return: c14
        :rtype: float
        """
        return 1 + 0.01 * self.C_stern

    @property
    def k1(self) -> float:
        """
        Calculate the k1 form factor
        k1 is a form factor describing the viscous resistance of the hull form in relation to R_F (See Holtrop and
         Mennen method
        :return: (k1) factor.
        :rtype: float
        """

        L = self.L #in metres
        B = self.B #in metres
        T = self.T # in metres
        C_P = self.C_P # in metres
        DWT_v = self.DWT_v # in metres
        c14 = self.c14 # in metres
        L_R = self.L_R # in metres

        k1 = (0.93 + 0.487188 * c14 * ((B / L) ** 1.06806)
              * ((T / L) ** 0.46106)
              * ((L / L_R) ** 0.121563)
              * ((L ** 3 / DWT_v) ** 0.36486)
              * ((1 - C_P) ** -0.604247)) - 1

        return k1

    @property
    def k2(self) -> List[float]:
        """Calculate the k2 resistance factors for appendages.

        :return: k2:
        :rtype: List[float]
        """
        if not hasattr(self, '_k2_values'):
            k2_map = {
                1: 1.5,  # rudder behind skeg
                2: 1.3,  # rudder behind stern
                3: 2.8,  # twin-screw balance rudders
                4: 3,  # shaft brackets
                5: 1.5,  # skeg
                6: 3,  # strut bossing
                7: 2,  # hull bossing
                8: 3,  # shafts
                9: 2.8,  # stabilizer fins
                10: 2.7,  # dome
                11: 1.4  # bilge keels
            }
            _k2_values = []
            for value in self.app_types:
                if value not in k2_map:
                    self.logger.error(f"Invalid appendage type: {value}. Must be one of {list(k2_map.keys())}.")
                    raise ValueError(f"Invalid appendage type: {value}. Must be one of {list(k2_map.keys())}.")
                k2_value = k2_map[value]
                _k2_values.append(k2_value)
                self.logger.info(f"Appendage type: {value}, k2 value: {k2_value}")

        return _k2_values

    @property
    def T_A(self) -> float:
        """
        Calculate or retrieve the aft draught of the ship in meters

        :return: T_A
        :rtype: float
        """

        if self._T_A is not None:
            return self._T_A

        elif self.trim is not None:
            base_value = self.T-(self.trim/2)
        else:
            base_value = self.T
        return self._apply_correction('T_A', base_value)

    @T_A.setter
    def T_A(self, value: float):
        self._T_A = value
    @property
    def T_F(self) -> float:
        """
        Calculate or retrieve the forward draught of the ship in meters

        :return: T_F
        :rtype: float
        """
        if self._T_F is not None:
            return self._T_F
        elif self.trim is not None:
            return self.T_A + self.trim
        base_value = self.T
        return self._apply_correction('T_F', base_value)

    @T_F.setter
    def T_F(self, value: float):
        if value >= 0.01 * self.L:
            logging.warning(f'T_F value {value} is greater than 0.01 * L ({0.01 * self.L})')
        self._T_F = value



    # @property
    # def A_m(self) -> float:
    #     """
    #     Calculate or retrieve the Midship area in square meters.
    #     :return: A_m
    #     :rtype: float
    #     """
    #
    #     if self._A_m is not None:
    #         return self._A_m
    #
    #     base_value = self.B * self.T * self.C_M
    #     return self._apply_correction('A_m', base_value)
    #
    # @A_m.setter
    # def A_m(self, value: float):
    #     self._A_m = value

    @cached_corrected_property
    def A_m(self):
        """
        Calculate the midship area in square meters.
        This is the base calculation before any corrections are applied.
        """
        self.logger.debug("Calculating base midship area A_m")
        return self.B * self.T * self.C_M


    @property
    def A_BT(self) -> float:
        """
        Calculate or retrieve the transverse sectional area of the bulb at the position where the still-water surface
         intersects the stern. in m^2

        :return: A_BT
        :rtype: float
        """
        if self._A_BT is not None:
            return self._A_BT
        base_value = 0.09 * self.A_m
        self.logger.info(f"A_BT calculated: {base_value}")
        return self._apply_correction('A_BT', base_value)

    @A_BT.setter
    def A_BT(self, value: float):
        self._A_BT = value

    @property
    def A_T(self) -> float:
        """
        Calculate or retrieve the immersed part of the transverse area of the transom at zero speed in m^2

        :return: A_T
        :rtype: float
        """
        if self._A_T is not None:
            return self._A_T

        base_value = 0.8 * self.A_BT
        self.logger.info(f"A_T calculated: {self._A_T}")
        return self._apply_correction('A_T', base_value)

    @A_T.setter
    def A_T(self, value: float):
        self._A_T = value

    @property
    def h_B(self) -> float:
        """
        Calculates or retrieves of the centre of the transverse area A_BT above the keel line.

        :return: h_B
        :rtype: float
        """
        if self._h_B is not None:
            return self._h_B
        base_value = 0.45 * self.T
        self.logger.info(f"h_B calculated: {base_value}")
        return self._apply_correction('h_B', base_value)

    @h_B.setter
    def h_B(self, value: float):
        self._h_B = value

    @property
    def C_F(self) -> float:
        """
        Calculate the coefficient of frictional resistance of the ship according to the ITTC-1957 formula.
        :return: C_F
        :rtype: float
        """
        return 0.075 / ((np.log10(self.Re)) - 2) ** 2

    @property
    def C_B(self) -> float:
        """
        Calculate the block coefficient based on the ship volume, length between perpendiculars, breadth and draught.
        :return: C_B
        :rtype: float
        """
        return self.DWT_v / (self.L_PP * self.B * self.T)

    @property
    def C_P(self) -> float:
        """
        Calculate the Prismatic coefficient based on the waterline length.
        :return: C_P
        :rtype: float
        """
        return self.C_B / self.C_M

    @property
    def C_WP(self) -> float:
        """
        Calculate or retrieve the Waterplane area coefficient.
        :return: C_WP
        :rtype: float
        """
        if self._C_WP is not None:
            return self._C_WP

        if self.C_sternchoice == 2:
            base_value = (0.743 * self.C_B) + 0.297

        elif self.C_sternchoice == 3:
            base_value = (1 + (2 * self.C_B)) / 3

        else:
            base_value = 0.95 * self.C_P + 0.17 * (1 - self.C_P) ** (1 / 3)

        self.logger.info(f"C_WP calculated: {self._C_WP}")
        return self._apply_correction('C_WP', base_value)

    @C_WP.setter
    def C_WP(self, value: float):
        self._C_WP = value

    @property
    def A_W(self) -> float:
        """
        Calculate or retrieve the waterplane area in square meters.
        :return: A_W
        :rtype: float
        """
        if self._A_W is not None:
            return self._A_W

        base_value = self.L * self.B * self.C_WP
        return self._apply_correction('A_W', base_value)

    @A_W.setter
    def A_W(self, value: float):
        self._A_W = value

    @property
    def lcb(self) -> float:
        """
        Longitudinal position of the centre of buoyancy forward of 0.5L as a percentage of L.

        :return: lcb
        :rtype: float
        """

        if self._lcb is not None:
            return self._lcb

        base_value = ((50 * self.F_n) - 10.5) / 100

        corrected_value = self._apply_correction('lcb', base_value)

        if corrected_value is None:
            raise ValueError("lcb calculation failed - returned None")

        return corrected_value

    @lcb.setter
    def lcb(self, value: float):
        self._lcb = value

    @property
    def S(self) -> float:
        """
        Calculate the Approximate wetted area of the hull in m^2.

        :return: S
        :rtype: float
        """
        if self._S is not None:
            return self._S
        else:
            T = self.T
            B = self.B
            C_B = self.C_B
            C_WP = self.C_WP
            A_BT = self.A_BT
            return self.L * (2 * T + B) * (self.C_M ** 0.5) * (
                    0.453 + 0.4425 * C_B - 0.2862 * self.C_M - 0.003467 * B / T + 0.3696 * C_WP) + 2.38 * A_BT / C_B

    @S.setter
    def S(self, value: float):
        self._S = value

    @property
    def L_R(self) -> float:
        """
        used in the form factor equation for Holtrop and Mennen. L_R in this equation is a parameter reflecting
         the length of the run. in meters
        :return: L_R
        :rtype: float
        """

        return self.L * (1 - self.C_P + 0.06 * self.C_P * self.lcb / (4 * self.C_P - 1))

    @property
    def S_APP(self) -> float:
        """
        Calculate or retreive the wetted areas of the appendages.

        :return: S_APP
        :rtype: float
        """
        if self._S_APP is None:
            C_B = self.C_B
            B = self.B
            L_PP = self.L_PP
            return 0.001 * (1 + 50 * (C_B ** 2) * ((B / L_PP) ** 2))
        self.logger.info(f"S_APP calculated: {self._S_APP}")
        return self._S_APP

    @S_APP.setter
    def S_APP(self, value: float):
        self._S_APP = value

    @property
    def I_E(self) -> float:
        """
        The half angle of entrance iE is the angle of the waterline
        at the bow in degrees with reference to the centre plane
        but neglecting the local shape at the stem

        :return: I_E
        :rtype: float
        """
        return 1 + (89 * np.exp(
            -((self.L / self.B) ** 0.80856) * ((1 - self.C_WP) ** 0.30484) * (
                    (1 - self.C_P - 0.0225 * self.lcb) ** 0.6367) * (
                    (self.L_R / self.B) ** 0.34574) * (100 * self.DWT_v / self.L ** 3) ** 0.16302))

