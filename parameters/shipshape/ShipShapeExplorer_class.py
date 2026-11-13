# Modules imports.
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Any
import pandas as pd
import math

from config.config_constants import GRAVITY, SEAWATER_DENSITY, NU
from config.ship_instance_config import ShipType
from config.loggermixin.logger_config import LoggerMixin


from parameters.shipshape.ShipShapeBasic_class import ShipShapeBasic


@dataclass
class CorrectionFactor(LoggerMixin):
    """
    Represents a correction to be applied to a ship parameter calculation.

    :ivar parameter: The parameter being corrected.
    :vartype parameter: str
    :ivar correction_type: The type of correction ('offset', 'factor', or 'function').
    :vartype correction_type: str
    :ivar value: The correction value or function.
    :vartype value: float, int, or Callable
    :ivar conditions: Optional conditions for when to apply the correction.
    :vartype conditions: dict or None
    """
    parameter: str
    correction_type: str  # 'offset', 'factor', or 'function'
    value: float|Callable
    conditions: dict[str, Any]|None = None

    def apply(self,
              base_value: float,
              ship_params: dict[str, Any]) -> float:
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

    def _check_conditions(self, ship_params: dict[str, Any]) -> bool:
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


"""
    def __init__(
            self,
            v_knots: float,
            correction_library: dict[str, CorrectionFactor]|None = None,
            shipshape: ShipShapeBasic|None = None,
            shiptype: ShipType|None = None,
            C_sternchoice: int=3,
            wake_fraction: float|None = None,
            thrust_deduction: float|None = None,
            app_types: List[int]|int|None = None,
            bulbchoice: bool|None = None,
            bow_thrusters: BowThrusters|List[BowThrusters]|None=None,
            rudders:int|None=None,
            brackets:int|None=None,
            bosses:int|None=None,
            BT_n: int|None = None,
            BT_d: float|None = None,
            C_BT: float|None = None,
            L: float|None = None,
            L_PP: float|None = None,
            L_WL: float|None = None,
            B: float|None = None,
            T: float|None = None,
            DWT: float|None = None,
            gt: float|None = None,
            ldt: float|None = None,
            C_M: float|None = None,
            C_B: float|None = None,
            trim: float|None = None,
            laden: bool = False,
            ballast: bool = False,
            teu: int|None = None,
            _DWT_v: float|None=None,
            _T_F: float|None= None,
            _T_A: float|None= None,
            _A_BT: float|None= None,
            _h_B: float|None= None,
            _C_WP: float|None= None,
            _A_T: float|None= None,
            _A_m: float|None= None,
            _A_W: float|None= None,
            _lcb: float|None= None,
            _S: float|None= None,
            _S_APP: float|None = None,
            ice_class: str='None',
            verbose: bool =False,
            debug: bool =False
    ):
        """
        :param v_knots: Ship speed in knots
        :param correction_library: Correction factor to apply to all dimensions
        :param shipshape: Ship shape basic class storing initial dimensions
        :param shiptype: Specific ship type being designed
        :param C_sternchoice: Stern type of the ship
        :param wake_fraction: Specific wake fraction of the ship (usually taken from sea trial data)
        :param thrust_deduction: Thrust deduction of the ship (usually taken from seatrial data)
        :param app_types: List of app types that the ships hull has
        :param bulbchoice: Bulb choice of the ship
        :param bow_thrusters: List of bow thruster classes, depending on if the ship has them
        :param rudders: Number of rudders that the ship has
        :param brackets: Number of brackets that the ship has
        :param bosses: Number of bosses that the ship has
        :param BT_n: Number of bow thrusters that the ship has
        :param BT_d: Diameter of the ships bow thrusters in metres
        :param C_BT: Roughness coefficient of the ships bow thrusters
        :param L: Specific Length of the ship in metres
        :param L_PP: Specific Length between perpendiculars of the ship in metres
        :param L_WL: Specific waterline length of the ship in metres
        :param B: Specific breadth of the ship in metres
        :param T: Specific draft of the ship in metres
        :param DWT: Specific dead weight tonnage of the ship in tonnes
        :param gt: Specific gross tonnage of the ship in tonnes
        :param ldt: Specific light weight tonnage of the ship in tonnes
        :param C_M: Specific midship coefficient of the ship
        :param C_B: Specific block coefficient of the ship
        :param trim: Trom of the ship
        :param laden: True if the ship has ladem
        :param ballast: True if the ship has ballast
        :param teu: container capacity
        :param _DWT_v: DWT coefficient of the ship
        :param _T_F: Draft at the fore of the ship
        :param _T_A: Draft at the aft of the ship
        :param _A_BT: Bulb tranverse area in metres squared
        :param _h_B: Height of bulb centre above the keel
        :param _C_WP: Waterplane coefficient
        :param _A_T: Transom area in metres squared
        :param _A_m: Midship area in metres squared
        :param _A_W: Waterplane area in metres squared
        :param _lcb: centre of buoyancy as percentage of ship length
        :param _S: Specific wetted area of the ship
        :param _S_APP: Specific wetted area of appendages
        :param ice_class: ice class of the ship
        :param verbose: True if the ship has verbose
        :param debug: True if the ship has debug

        """

        # Initialize logging first
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)


        self._get_shipshapebasic_parameters(shipshape=shipshape,
                                            shiptype=shiptype,
                                            L=L,
                                            L_PP=L_PP,
                                            L_WL=L_WL,
                                            B=B,
                                            T=T,
                                            DWT=DWT,
                                            ldt=ldt,
                                            gt=gt)

        self._get_specific_operating_point(v_knots=v_knots,
                                           wake_fraction=wake_fraction,
                                           thrust_deduction=thrust_deduction,
                                           trim=trim,
                                           laden=laden,
                                           ballast=ballast)

        self._get_hullshape_parameters(C_sternchoice=C_sternchoice,
                                       teu=teu,
                                       app_types=app_types,
                                       bulbchoice=bulbchoice,
                                       bow_thrusters=bow_thrusters,
                                       BT_n=BT_n,
                                       BT_d=BT_d,
                                       C_BT=C_BT,
                                       ice_class=ice_class,
                                       rudders=rudders,
                                       brackets=brackets,
                                       bosses=bosses)

        self._get_parameter_overrides(_DWT_v = _DWT_v ,
                                      _T_F = _T_F,
                                      _T_A = _T_A,
                                      _A_BT = _A_BT,
                                      _h_B = _h_B,
                                      _C_WP = _C_WP,
                                      _A_T = _A_T,
                                      _A_m =  _A_m,
                                      _A_W = _A_W,
                                      _lcb =_lcb,
                                      _S = _S,
                                      _S_APP = _S_APP)

        # Constants
        self.g = GRAVITY
        self.rho_SW = SEAWATER_DENSITY
        self.nu = NU
        self.correction_library = correction_library

        # Log information about applied corrections
        if self.correction_library:
            self.logger.info(f"Initialized with {len(self.correction_library)} corrections")
            for param, correction in self.correction_library.items():
                self.logger.debug(f"Correction registered for {param}: {correction}")

        # Run post-initialization validation
        self._validate_parameters()

        # Lazy calculation - compute all parameters on first access
        self._ship_record = None

        super().__init__()

    def _get_shipshapebasic_parameters(self,
                                       **kwargs):
        """
        loads parameters which are stored by the shipshape_basic class.

        Creates an instance of the shipshapebasic class in order to utilise the methods.
        """
        if kwargs['shipshape'] is not None:
            self.shipshape = kwargs['shipshape']
            self.logger.info("Loading basic parameters from ship basic class")
            self.shiptype = self.shipshape.shiptype
            self.L = self.shipshape.L
            self.L_PP = self.shipshape.L_PP
            self.L_WL = self.shipshape.L_WL()
            self.B = self.shipshape.B
            self.T = self.shipshape.T
            self.DWT = self.shipshape.DWT
            self.ldt = self.shipshape.ldt
            self.C_M = self.shipshape.C_M
            self.C_B = self.shipshape.C_B()
        else:
            self.logger.warning("It is preferred that ship basic is used for initial design parameters")
            if kwargs['shiptype'] is None:
                raise ValueError("shiptype undefined")
            else:
                self.shiptype = kwargs['shiptype']

            # Ensure that all required parameters are provided
            required_params = {
                "L": kwargs['L'], "L_PP": kwargs['L_PP'], "B": kwargs['B'],
                "T": kwargs['T']
            }
            missing_params = [param for param, value in required_params.items() if value is None]

            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)} "
                                 "when ShipShape is not provided.")

            self.logger.info('Creating ship basic object')

            shipshape=ShipShapeBasic(shiptype =kwargs['shiptype'],L=kwargs['L'], DWT=kwargs['DWT'],_L_PP=kwargs['L_PP'],_B=kwargs['B'],_T=kwargs['T'], verbose=self.verbose, debug=self.debug)
            self.shipshape = shipshape

            self.L = kwargs['L']
            self.L_PP = kwargs['L_PP']
            self.L_WL= self.shipshape.L_WL()
            self.B = kwargs['B']
            self.T = kwargs['T']
            self.DWT = kwargs['DWT']
            self.ldt = kwargs['LDT']
            self.C_M = self.shipshape.C_M
            self.C_B = self.shipshape.C_B()

    def _get_hullshape_parameters(self,
                                  **kwargs):
        """
        Gets specific hull shape parameters which influence resistance calculations.

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

        Including:
        Stern shape
        Hull apendages
        Bow thrusters
        Ice class
        """

        self.C_sternchoice = kwargs['C_sternchoice']

        # Handle appendage types
        if isinstance(kwargs['app_types'], int):
            self.app_types = [kwargs['app_types']]
        elif kwargs['app_types'] is None:
            self.app_types = [1]  # Default value
        else:
            self.app_types = kwargs['app_types']

        # Store optional parameters
        self.bulbchoice = kwargs['bulbchoice']

        self.bow_thrusters = kwargs['bow_thrusters']

        # :TODO: get bow thruster parameters from bow thruster object

        self.BT_n = kwargs['BT_n']
        self.BT_d = kwargs['BT_d']
        self.C_BT = kwargs['C_BT']
        self.ice_class = kwargs['ice_class']
        self.teu = kwargs['teu']
        self.rudder = kwargs['rudders']
        self.bracket = kwargs['brackets']
        self.bosses = kwargs['bosses']

    def _get_specific_operating_point(self,
                                      **kwargs):
        """
        Get the specific operating point of the ship shape, if these details are provided.
        """
        self.v_knots = kwargs['v_knots']
        self.wake_fraction = kwargs['wake_fraction']
        self.thrust_deduction = kwargs['thrust_deduction']

        self.trim = kwargs['trim']
        self.laden = kwargs['laden']
        self.ballast = kwargs['ballast']

    def _get_parameter_overrides(self,
                                 **kwargs):
        """
        Gets parameters that user has specified with the specific shipshape explorer class instance
        """
        # Store direct value overrides (prefixed with _)
        self._DWT_v = kwargs['_DWT_v']
        self._T_F = kwargs['_T_F']
        self._T_A = kwargs['_T_A']
        self._A_BT = kwargs['_A_BT']
        self._h_B = kwargs['_h_B']
        self._C_WP = kwargs['_C_WP']
        self._A_T = kwargs['_A_T']
        self._A_m = kwargs['_A_m']
        self._A_W = kwargs['_A_W']
        self._lcb = kwargs['_lcb']
        self._S = kwargs['_S']
        self._S_APP = kwargs['_S_APP']

    def _validate_parameters(self):
        """Validate parameters after initialization."""
        # Ensure that BT_d and C_BT must be defined if either is defined
        if (self.BT_d is None) != (self.C_BT is None):
            raise ValueError("Both BT_d and C_BT must be either defined or undefined together.")

        # Ensure that BT_n must be defined if either BT_d or C_BT is defined
        if (self.BT_d is not None or self.C_BT is not None) and self.BT_n is None:
            raise ValueError("BT_n must be defined if BT_d or C_BT is defined.")

        # Validate C_BT range
        if self.C_BT is not None:
            if not (0.003 <= self.C_BT <= 0.012):
                self.logger.warning("Warning: C_BT is out of the expected range (0.003 - 0.012).")

        if self.C_sternchoice not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid C_sternchoice: {self.C_sternchoice}. Must be one of [1,2,3,4].")

    def _ensure_calculated(self):
        """Calculate all parameters if not already done."""
        if self._ship_record is None:
            self.logger.info("Calculating all ship parameters...")
            all_params = self._calculate_all_parameters()
            self._ship_record = all_params
            self.logger.info("All ship parameters calculated and stored")

    def _calculate_all_parameters(self) -> dict[str, float]:
        """Calculate all ship parameters in correct dependency order."""
        params = {}

        # Layer 1: Basic input parameters
        params.update(self._get_basic_parameters())

        # Layer 2: Simple speed-related parameters
        params['v_ms'] = self._calc_v_ms(params)
        params['Re'] = self._calc_reynolds(params)
        params['F_n'] = self._calc_froude(params)

        # Layer 3: Volume and coefficients
        params['DWT_v'] = self._calc_dwt_v(params)
        # params['C_B'] = self._calc_c_b(params)
        params['C_P'] = self._calc_c_p(params)

        # Layer 4: Draft parameters (depend on trim)
        params['T_A'] = self._calc_t_a(params)
        params['T_F'] = self._calc_t_f(params)
        params['T_C'] = self._calc_t_c(params)

        # Layer 5: Areas (depend on coefficients and dimensions)
        params['A_m'] = self._calc_a_m(params)
        params['A_BT'] = self._calc_a_bt(params)
        params['A_T'] = self._calc_a_t(params)
        params['h_B'] = self._calc_h_b(params)

        # Layer 6: Waterplane coefficient (depends on stern choice)
        params['C_WP'] = self._calc_c_wp(params)
        params['A_W'] = self._calc_a_w(params)

        # Layer 7: Speed-dependent position parameters
        params['lcb'] = self._calc_lcb(params)

        # Layer 8: Stern and appendage parameters (must come before k1 calculation)
        params['C_stern'] = self._calc_c_stern(params)
        # params['c14'] = self._calc_c14(params)
        # params['k2'] = self._calc_k2(params)  # Returns list

        # Layer 9: Parameters that depend on lcb
        params['L_R'] = self._calc_l_r(params)
        params['I_E'] = self._calc_i_e(params)

        # Later 10: Parameters dependent on i_e
        params['L_E'] = self._calc_l_e(params)

        # Layer 11: Wetted surface areas
        params['S'] = self._calc_s(params)
        params['S_APP'] = self._calc_s_app(params)

        # Layer 12: Resistance coefficients
        params['C_F'] = self._calc_cf(params)

        return params

    def _get_basic_parameters(self) -> dict[str, float]:
        """Get basic input parameters."""
        return {
            'L': self.L,
            'L_PP': self.L_PP,
            'L_WL': self.L_WL,
            'B': self.B,
            'T': self.T,
            'DWT': self.DWT,
            'LDT':self.ldt,
            'C_M': self.C_M,
            'C_B': self.C_B,
            'v_knots': self.v_knots,
            'C_sternchoice': self.C_sternchoice,
            'trim': self.trim
        }

    # ===== CALCULATION METHODS =====
    def _calc_v_ms(self, params: dict[str, float]) -> float:
        """Convert speed from knots to m/s."""
        return params['v_knots'] * 0.5144

    def _calc_reynolds(self, params: dict[str, float]) -> float:
        """Calculate Reynolds number."""
        return params['v_ms'] * params['L'] / self.nu

    def _calc_froude(self, params: dict[str, float]) -> float:
        """Calculate Froude number."""
        return params['v_ms'] / (np.sqrt(self.g * params['L']))

    def _calc_dwt_v(self, params: dict[str, float]) -> float:
        """Calculate displacement volume."""
        if self._DWT_v is not None:
            self.logger.debug(f"Using provided DWT_v: {self._DWT_v:.4f} m³")
            return self._DWT_v
        else:

            if params['LDT'] is None:
                self.logger.warning("No superstructure tonnage provided so this value may be quite far off.")
                params['LDT'] = 0

            dwt_v = (params['DWT']+params['LDT']) * 1000 / self.rho_SW
            self.logger.info(f"DWT_v calculated: {dwt_v:.4f} m³")
            return dwt_v

    # def _calc_c_b(self, params: Dict[str, float]) -> float:
    #     """Calculate block coefficient."""
    #     return params['DWT_v'] / (params['L_PP'] * params['B'] * params['T'])

    def _calc_c_p(self, params: dict[str, float]) -> float:
        """Calculate prismatic coefficient."""
        return params['C_B'] / params['C_M']

    def _calc_t_a(self, params: dict[str, float]) -> float:
        """Calculate aft draft."""
        if self._T_A is not None:
            return self._T_A
        elif params['trim'] is not None:
            base_value = params['T'] - (params['trim'] / 2)
        else:
            base_value = params['T']
        return self._apply_correction('T_A', base_value)

    def _calc_t_f(self, params: dict[str, float]) -> float:
        """Calculate forward draft."""
        if self._T_F is not None:
            return self._T_F
        elif params['trim'] is not None:
            return params['T_A'] + params['trim']
        else:
            base_value = params['T']
            return self._apply_correction('T_F', base_value)

    def _calc_t_c(self, params: dict[str, float]) -> float:
        """
        Calculates the current draught (T_C) using the ratio between current draight and design draught (T_D)
        provided by Olmer et al.,2017
        """

        data = {
            "Ship type": [
                ShipType.BULK_CARRIER,
                ShipType.CHEMICAL_TANKER,
                ShipType.GENERAL_SHIP,
                ShipType.LNG_TANKER,
                ShipType.OIL_TANKER,
                ShipType.CONTAINER_SHIP,
                ShipType.PASSENGER_SHIP,
                ShipType.RO_RO,
            ],
            "Ballast": [0.57, 0.65, 0.65, 0.67, 0.60, np.nan, np.nan, np.nan],
            "Laden": [0.91, 0.89, 0.89, 0.88, 0.89, np.nan, np.nan, np.nan],
            "Average": [0.80, 0.85, 0.84, 0.87, 0.82, 0.87, 0.93, 0.91],
        }

        ratio_df = pd.DataFrame(data)

        laden = self.laden
        ballast = self.ballast
        shiptype = self.shiptype
        t_d = self.T  # design draught

        # Select row for current ship type
        row = ratio_df.loc[ratio_df["Ship type"] == shiptype]

        if row.empty:
            logging.warning(f"Unknown ship type {shiptype}. Falling back to default ratio = 1.0")
            ratio = 1.0
        elif laden and ballast:
            ratio = row["Average"].values[0]
            logging.warning("Both 'laden' and 'ballast' are True. Using Average ratio.")
        elif laden:
            ratio = row["Laden"].values[0]
            if np.isnan(ratio):  # fallback if missing
                logging.warning(f"No Laden ratio for {shiptype}. Falling back to Average.")
                ratio = row["Average"].values[0]
        elif ballast:
            ratio = row["Ballast"].values[0]
            if np.isnan(ratio):  # fallback if missing
                logging.warning(f"No Ballast ratio for {shiptype}. Falling back to Average.")
                ratio = row["Average"].values[0]
        else:
            ratio = row["Average"].values[0]

        return ratio * t_d

    def _calc_a_m(self, params: dict[str, float]) -> float:
        """Calculate midship area."""
        if self._A_m is not None:
            return self._A_m
        base_value = params['B'] * params['T'] * params['C_M']
        return self._apply_correction('A_m', base_value)

    def _calc_a_bt(self, params: dict[str, float]) -> float:
        """Calculate bulb transverse area."""
        if self._A_BT is not None:
            return self._A_BT
        base_value = 0.09 * params['A_m']
        self.logger.info(f"A_BT calculated: {base_value}")
        return self._apply_correction('A_BT', base_value)

    def _calc_a_t(self, params: dict[str, float]) -> float:
        """Calculate transom area."""
        if self._A_T is not None:
            return self._A_T
        base_value = 0.8 * params['A_BT']
        self.logger.info(f"A_T calculated: {base_value}")
        return self._apply_correction('A_T', base_value)

    def _calc_h_b(self, params: dict[str, float]) -> float:
        """Calculate height of bulb center."""
        if self._h_B is not None:
            return self._h_B
        base_value = 0.45 * params['T']
        self.logger.info(f"h_B calculated: {base_value}")
        return self._apply_correction('h_B', base_value)

    def _calc_c_wp(self, params: dict[str, float]) -> float:
        """Calculate waterplane coefficient."""
        if self._C_WP is not None:
            return self._C_WP

        #Getting Holtrop value

        if params['C_sternchoice'] == 2:
            hb_value = (0.743 * params['C_B']) + 0.297
        elif params['C_sternchoice'] == 3:
            hb_value = (1 + (2 * params['C_B'])) / 3
        else:
            hb_value = 0.95 * params['C_P'] + 0.17 * (1 - params['C_P']) ** (1 / 3)

        self.logger.debug(f"C_WP calculated: {hb_value}")

        #Using estimations from Kim et al 2023 https://doi.org/10.1016/j.oceaneng.2023.114758

        if self.shiptype in [ShipType.TANKER,ShipType.CARGO,ShipType.BULK_CARRIER]: #Bertram and Wobig 1999
            self.logger.debug(f"C_WP calculated for ship type cargo, tanker or bulker")
            C_WP = 0.763* (params['C_P']+0.34)

        elif self.shiptype == ShipType.CONTAINER_SHIP: #Betram and Wobig
            self.logger.debug(f"C_WP calculated for ship type container ship")
            C_WP = 3.226*(params['C_P']-0.36)

        else: # Papanikolaou 2014
            self.logger.debug(f"C_WP calculated using generic formula")
            C_WP = (1+2*params['C_B'])/3

        base_value = (C_WP * hb_value)/2
        self.logger.debug(f"C_WP calculated was {base_value}")

        corrected = self._apply_correction('C_WP', base_value)
        self.logger.debug(f"C_WP corrected was {corrected}")

        return corrected

    def _calc_a_w(self, params: dict[str, float]) -> float:
        """Calculate waterplane area."""
        if self._A_W is not None:
            return self._A_W
        base_value = params['L'] * params['B'] * params['C_WP']
        return self._apply_correction('A_W', base_value)

    def _calc_lcb(self, params: dict[str, float]) -> float:
        """Calculate longitudinal center of buoyancy."""
        if self._lcb is not None:
            return self._lcb
        base_value = ((50 * params['F_n']) - 10.5) / 100
        corrected_value = self._apply_correction('lcb', base_value)
        if corrected_value is None:
            raise ValueError("lcb calculation failed - returned None")
        return corrected_value

    def _calc_l_r(self, params: dict[str, float]) -> float:
        """Calculate length of run."""
        return params['L_WL'] * (1 - params['C_P'] + 0.06 * params['C_P'] * params['lcb'] / (4 * params['C_P'] - 1))

    def _calc_i_e(self, params: dict[str, float]) -> float:
        """Calculate half angle of entrance - using formula provided by holtrop and mennen."""
        return 1 + (89 * np.exp(
            -((params['L'] / params['B']) ** 0.80856) * ((1 - params['C_WP']) ** 0.30484) * (
                    (1 - params['C_P'] - 0.0225 * params['lcb']) ** 0.6367) * (
                    (params['L_R'] / params['B']) ** 0.34574) *
            (100 * params['DWT_v'] / params['L'] ** 3) ** 0.16302))

    def _calc_l_e(self, params: dict[str, float]) -> float:
        """Calculate the length of waterline entrance. as defined in figure for of Liu et al (2016)
        found using the length at which point water is at 99% of B

        """
        i_e = params['I_E']
        B = params['B']

        B99 = (B*0.99)/2

        return B99/(math.tan(i_e))

    def _calc_s(self, params: dict[str, float]) -> float:
        """Calculate wetted surface area."""
        if self._S is not None:
            return self._S

        L = params['L']
        T_d = params['T']
        B = params['B']
        C_B = params['C_B']
        C_M = params['C_M']
        C_WP = params['C_WP']
        A_BT = params['A_BT']

        # Calculation for S_hm taken from holtrop and mennen
        S_hm = L * (2 * T_d + B) * (C_M ** 0.5) * (
                0.453 + 0.4425 * C_B - 0.2862 * C_M - 0.003467 * B / T_d + 0.3696 * C_WP) + 2.38 * A_BT / C_B

        T_c = params['T_C']
        L_w = params['L_WL']
        DWT_v = params['DWT_v']
        shiptype = self.shiptype

        #Methods for calculating S from Kristensen and Lutzen (2012)

        if shiptype == ShipType.CONTAINER_SHIP:
            S_td = 0.995*((DWT_v/T_d) +(1.9*L_w*T_d))
            S_tc = S_td-(2.4*(T_d-T_c))
            self.logger.debug(f"S_td corrected was {S_tc} using Kristensen and Lutzen")
            return S_tc

        elif shiptype == ShipType.BULK_CARRIER or ShipType.OIL_TANKER or ShipType.LNG_TANKER or ShipType.CHEMICAL_TANKER or ShipType.TANKER:
            S_td = 0.99 * ((DWT_v / T_d) + (1.9 * L_w * T_d))
            S_tc = S_td - (2 * (T_d - T_c))
            self.logger.debug(f"S_td corrected was {S_tc} using Kristensen and Lutzen")
            return S_tc

        elif shiptype == ShipType.RO_RO:
            S_td = 0.87 * ((DWT_v / T_d) + (2.7 * L_w * T_d))
            S_tc = S_td - (3 * (T_d - T_c))
            self.logger.debug(f"S_td corrected was {S_tc} using Kristensen and Lutzen")
            return S_tc


        elif shiptype == ShipType.PASSENGER_SHIP:
            S_td = 1.11 * ((DWT_v / T_d) + (1.7 * L_w * T_d))
            S_tc = S_td - (2.4 * (T_d - T_c))
            self.logger.debug(f"S_td corrected was {S_tc} using Kristensen and Lutzen")
            return S_tc

        else:
            self.logger.debug(f"S_td corrected was {S_hm} using Holtrop and Mennen formula")
            return S_hm

    def _calc_s_app(self, params: dict[str, float]) -> float:
        """Calculate appendage wetted surface area."""
        if self._S_APP is not None:
            self.logger.info(f"S_APP provided: {self._S_APP}")
            return self._S_APP

        C_B = params['C_B']
        B = params['B']
        L_PP = params['L_PP']
        return 0.001 * (1 + 50 * (C_B ** 2) * ((B / L_PP) ** 2))

    def _calc_cf(self, params: dict[str, float]) -> float:
        """Calculate frictional resistance coefficient."""
        return 0.075 / ((np.log10(params['Re'])) - 2) ** 2

    def _calc_c_stern(self, params: dict[str, float]) -> float:
        """Calculate stern coefficient."""
        stern_choices = {
            1: -25,  # pram with gondola
            2: -10,  # v-shaped section
            3: 0,  # normal section
            4: 10  # U-shaped section with Hogner stern
        }
        c_stern = stern_choices[params['C_sternchoice']]
        self.logger.info(f"Stern choice: {params['C_sternchoice']}, C_stern coefficient: {c_stern}")
        return c_stern


    def _apply_correction(self, parameter: str, base_value: float) -> float:
        """Apply registered corrections to a calculated parameter."""
        if self.correction_library is None:
            return base_value

        if parameter not in self.correction_library:
            return base_value

        correction = self.correction_library[parameter]
        if not isinstance(correction, CorrectionFactor):
            self.logger.warning(f"Invalid correction type for {parameter}. Expected CorrectionFactor.")
            return base_value

        current_params = self._get_current_params()

        try:
            self.logger.info(f"Applying correction to {parameter}: {base_value}")
            return correction.apply(base_value, current_params)
        except Exception as e:
            self.logger.warning(
                f"Failed to apply correction to {parameter}: {str(e)}. Using base value."
            )
            return base_value

    def _get_current_params(self) -> dict[str, Any]:
        """Get current parameter values for corrections."""
        return {
            'V': self.v_knots,
            'T': self.T,
            'L': self.L,
            'B': self.B,
            'shiptype': self.shiptype,
            'C_M': self.C_M,
            'DWT': self.DWT,
            'C_sternchoice': self.C_sternchoice
        }

    # ===== PROPERTIES (All use lazy calculation) =====

    @property
    def v_ms(self) -> float:
        """Ship speed in m/s."""
        self._ensure_calculated()
        return self._ship_record['v_ms']

    @property
    def DWT_v(self) -> float:
        """Displacement volume in m³."""
        self._ensure_calculated()
        return self._ship_record['DWT_v']

    @DWT_v.setter
    def DWT_v(self, value: float):
        """Set displacement volume and invalidate calculations."""
        self.logger.info(f"Setting DWT_v to {value:.4f} m³")
        self._DWT_v = value
        self._ship_record = None  # Force recalculation

    @property
    def Re(self) -> float:
        """Reynolds number."""
        self._ensure_calculated()
        return self._ship_record['Re']

    @property
    def F_n(self) -> float:
        """Froude number."""
        self._ensure_calculated()
        return self._ship_record['F_n']

    @property
    def C_stern(self) -> float:
        """Stern coefficient."""
        self._ensure_calculated()
        return self._ship_record['C_stern']

    @property
    def T_A(self) -> float:
        """Aft draft."""
        self._ensure_calculated()
        return self._ship_record['T_A']

    @T_A.setter
    def T_A(self, value: float):
        """Set aft draft."""
        self._T_A = value
        self._ship_record = None  # Force recalculation

    @property
    def T_F(self) -> float:
        """Forward draft."""
        self._ensure_calculated()
        return self._ship_record['T_F']

    @T_F.setter
    def T_F(self, value: float):
        """Set forward draft."""
        if value >= 0.01 * self.L:
            self.logger.warning(f'T_F value {value} is greater than 0.01 * L ({0.01 * self.L})')
        self._T_F = value
        self._ship_record = None  # Force recalculation

    @property
    def A_m(self) -> float:
        """Midship area in m²."""
        self._ensure_calculated()
        return self._ship_record['A_m']

    @A_m.setter
    def A_m(self, value: float):
        """Set midship area."""
        self._A_m = value
        self._ship_record = None  # Force recalculation

    @property
    def A_BT(self) -> float:
        """Bulb transverse area in m²."""
        self._ensure_calculated()
        return self._ship_record['A_BT']

    @A_BT.setter
    def A_BT(self, value: float):
        """Set bulb transverse area."""
        self._A_BT = value
        self._ship_record = None  # Force recalculation

    @property
    def A_T(self) -> float:
        """Transom area in m²."""
        self._ensure_calculated()
        return self._ship_record['A_T']

    @A_T.setter
    def A_T(self, value: float):
        """Set transom area."""
        self._A_T = value
        self._ship_record = None  # Force recalculation

    @property
    def h_B(self) -> float:
        """Height of bulb center above keel."""
        self._ensure_calculated()
        return self._ship_record['h_B']

    @h_B.setter
    def h_B(self, value: float):
        """Set height of bulb center."""
        self._h_B = value
        self._ship_record = None  # Force recalculation

    @property
    def C_F(self) -> float:
        """Frictional resistance coefficient."""
        self._ensure_calculated()
        return self._ship_record['C_F']

    @property
    def C_P(self) -> float:
        """Prismatic coefficient."""
        self._ensure_calculated()
        return self._ship_record['C_P']

    @property
    def C_WP(self) -> float:
        """Waterplane coefficient."""
        self._ensure_calculated()
        return self._ship_record['C_WP']

    @C_WP.setter
    def C_WP(self, value: float):
        """Set waterplane coefficient."""
        self._C_WP = value
        self._ship_record = None  # Force recalculation

    @property
    def A_W(self) -> float:
        """Waterplane area in m²."""
        self._ensure_calculated()
        return self._ship_record['A_W']

    @A_W.setter
    def A_W(self, value: float):
        """Set waterplane area."""
        self._A_W = value
        self._ship_record = None  # Force recalculation

    @property
    def lcb(self) -> float:
        """Longitudinal center of buoyancy."""
        self._ensure_calculated()
        return self._ship_record['lcb']

    @lcb.setter
    def lcb(self, value: float):
        """Set longitudinal center of buoyancy."""
        self._lcb = value
        self._ship_record = None  # Force recalculation

    @property
    def S(self) -> float:
        """Wetted surface area in m²."""
        self._ensure_calculated()
        return self._ship_record['S']

    @S.setter
    def S(self, value: float):
        """Set wetted surface area."""
        self._S = value
        self._ship_record = None  # Force recalculation

    @property
    def L_R(self) -> float:
        """Length of run."""
        self._ensure_calculated()
        return self._ship_record['L_R']

    @property
    def S_APP(self) -> float:
        """Appendage wetted surface area in m²."""
        self._ensure_calculated()
        return self._ship_record['S_APP']

    @S_APP.setter
    def S_APP(self, value: float):
        """Set appendage wetted surface area."""
        self._S_APP = value
        self._ship_record = None  # Force recalculation

    @property
    def I_E(self) -> float:
        """Half angle of entrance in degrees."""
        self._ensure_calculated()
        return self._ship_record['I_E']

    @I_E.setter
    def I_E(self, value: float):
        """
        set half angle of entrance in degrees.
        """
        self._I_E = value
        self._ship_record = None

    @property
    def L_E(self) -> float:
        """Length of waterline entrance."""
        self._ensure_calculated()
        return self._ship_record['L_E']

    @L_E.setter
    def L_E(self, value: float):
        """Length of waterline entrance."""
        self._L_E = value
        self._ship_record = None

    # ===== COMPATIBILITY AND UTILITY METHODS =====

    def clear_direct_value(self, property_name: str):
        """Clear a directly set property value, allowing calculation to take over."""
        direct_attr = f"_{property_name}"
        if hasattr(self, direct_attr):
            setattr(self, direct_attr, None)
            self.logger.info(f"Cleared direct value for {property_name}")
            self._ship_record = None  # Force recalculation

    def get_all_parameters(self) -> dict[str, Any]:
        """Get all calculated parameters as a dictionary."""
        self._ensure_calculated()
        return self._ship_record

    def get_calculation_summary(self) -> str:
        """Get a summary of calculated parameters."""
        self._ensure_calculated()
        params = self._ship_record

        summary = []
        summary.append("=== Ship Shape Explorer Parameter Summary ===")
        summary.append(f"Ship Type: {self.shiptype.name if hasattr(self.shiptype, 'name') else self.shiptype}")
        summary.append(f"Speed: {self.v_knots:.1f} knots ({params['v_ms']:.2f} m/s)")
        summary.append("")

        summary.append("Basic Dimensions:")
        summary.append(f"  L = {params['L']:.2f} m")
        summary.append(f"  L_PP = {params['L_PP']:.2f} m")
        summary.append(f"  B = {params['B']:.2f} m")
        summary.append(f"  T = {params['T']:.2f} m")
        summary.append(f"  DWT = {params['DWT']:.0f} tonnes")
        summary.append("")

        summary.append("Coefficients:")
        summary.append(f"  C_B = {params['C_B']:.4f}")
        summary.append(f"  C_P = {params['C_P']:.4f}")
        summary.append(f"  C_M = {params['C_M']:.4f}")
        summary.append(f"  C_WP = {params['C_WP']:.4f}")
        summary.append("")

        summary.append("Areas:")
        summary.append(f"  A_m = {params['A_m']:.2f} m²")
        summary.append(f"  A_W = {params['A_W']:.2f} m²")
        summary.append(f"  S = {params['S']:.2f} m²")
        summary.append("")

        summary.append("Dimensionless Numbers:")
        summary.append(f"  Re = {params['Re']:.2e}")
        summary.append(f"  F_n = {params['F_n']:.4f}")
        summary.append(f"  C_F = {params['C_F']:.6f}")
        summary.append("")

        summary.append("Form Factors:")
        summary.append(f"  lcb = {params['lcb']:.4f}")
        summary.append("=" * 45)

        return "\n".join(summary)

    def __setattr__(self, name: str, value: Any):
        """Handle dynamic attribute setting from external classes."""
        # Allow setting during initialization
        if not hasattr(self, '_ship_record'):
            super().__setattr__(name, value)
            return

        # Handle special parameters that should trigger recalculation
        if hasattr(self, '_ship_record') and self._ship_record is not None:
            # Parameters that would affect calculations
            calc_affecting_params = {
                'L', 'B', 'T', 'DWT', 'L_PP', 'C_M', 'v_knots',
                'trim', 'C_sternchoice'
            }

            if name in calc_affecting_params:
                super().__setattr__(name, value)
                self._ship_record = None  # Force recalculation
                self.logger.debug(f"Parameter {name} changed to {value}, will recalculate on next access")
                return

            # Handle direct value overrides (prefixed with _)
            if name.startswith('_') and name[1:] in ['DWT_v', 'T_F', 'T_A', 'A_BT', 'h_B', 'C_WP', 'A_T', 'A_m', 'A_W',
                                                     'lcb', 'S', 'S_APP']:
                super().__setattr__(name, value)
                self._ship_record = None  # Force recalculation
                self.logger.debug(f"Direct override {name} set to {value}, will recalculate on next access")
                return

        # Default behavior
        super().__setattr__(name, value)
