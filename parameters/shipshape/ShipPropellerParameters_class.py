"""
This module contains classes which store the physical parameters and performance parameters of propellers
"""

# Modules imports.
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from pathlib import Path

os.chdir(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable")


sys.path.append(r"C:\Users\bnoble\PycharmProjects\T1.6_deliverable\config")

from config.logger_config import LoggerMixin
from parameters.propeller_classes.propeller_config import WAGENINGEN_KQ_FILE,WAGENINGEN_KT_FILE

class ShipPropellerParameters(LoggerMixin):
    """
    This class is used to store and calculate propeller parameters.
    Stores physical parameters and performance characteristics of the propeler.
    """
    def __init__(self,
                 D: float,
                 Z: float,
                 h: float,
                 AEA0: Optional[float]= None,
                 k_p: float = 0.00003,
                 P: Optional[float] = None,
                 P_D: Optional[float] = None,
                 screws: Optional[int] = None,
                 K: Optional[float] = None,
                 J: Optional[np.ndarray] = None,
                 Kt: Optional[np.ndarray] = None,
                 Kq: Optional[np.ndarray] = None,
                 wageningen: bool = True,
                 cpp: bool = False,
                 verbose: bool = True,
                 debug: bool = False
                 ):
        """
        Initialize the propeller parameters.

        :param D: Diameter of propeller in meters
        :type D: float

        :param Z: Number of propeller blades
        :type Z: float

        :param h: Water height of propeller blades in meters
        :type h: float

        :param P: Pitch of the propeller in meters, optional
        :type P: Optional[float]

        :param AEA0: Expanded blade area ratio, optional
        :type AEA0: Optional[float]

        :param k_p: Propeller blade surface roughness in meters, defaults to 0.00003
        :type k_p: float

        :param P_D: Pitch-to-diameter ratio, optional
        :type P_D: Optional[float]

        :param screws: Number of screws (propellers), optional
        :type screws: Optional[int]

        :param K: Coefficient for twin-screw ships, optional
        :type K: Optional[float]

        :param J: Advance coefficient values array, optional
        :type J: Optional[np.ndarray]

        :param Kt: Thrust coefficient data array, optional
        :type Kt: Optional[np.ndarray]

        :param Kq: Torque coefficient data array, optional
        :type Kq: Optional[np.ndarray]

        :param wageningen: If True, use Wageningen B-series calculations, defaults to False
        :type wageningen: bool

        :param cpp: If True, efficiency calculations will be maximised using Wangeningen data.
        :type cpp: bool
        """

        # Physical parameters
        self.D = D
        self.Z = Z
        self.h = h
        self.P = P
        self.AEA0 = AEA0
        self.k_p = k_p
        self._P_D = P_D

        # Propulsion system parameters
        self.screws = screws
        self.K = K

        # Performance data
        self.J = J
        self.Kt = Kt
        self.Kq = Kq

        # Calculation method parameters
        self.wageningen = wageningen
        self.cpp

        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Cache for computed values
        self._kt_cache = {}
        self._kq_cache = {}
        self._eff_O_cache = {}

        # Validate inputs and load data
        self._initialize()

    def _initialize(self):
        """
        Initialize the propeller object by validating parameters and loading data.

        :raises ValueError: If validation fails
        """
        self.logger.info(f"Initializing propeller: D={self.D}m, Z={self.Z} blades")

        # Validate screw/K configuration
        self._validate_screws_and_K()

        # Select data source
        if self.J is not None and self.Kt is not None and self.Kq is not None:
            self.logger.info("Using user-supplied propeller performance data")
            self._validate_performance_data()
        elif self.wageningen:
            self.logger.info("Using Wageningen B-series calculation for propeller performance")
            self._validate_wageningen_parameters()
            self._load_wageningen_data()
        else:
            self.logger.warning("No performance data provided and Wageningen not enabled. Many methods will not work.")

    def _validate_screws_and_K(self):
        """
        Validate and configure screws and K parameters.

        :raises ValueError: If validation fails
        """
        if self.K is not None and self.screws is not None:
            raise ValueError("Specify either 'screws' or 'K', not both.")

        if self.K is None:
            if self.screws is None:
                raise ValueError("Either 'screws' or 'K' must be specified.")
            elif self.screws == 1:
                self.K = 0.2  # Default for a single screw
                self.logger.info(f"Setting K={self.K} for single screw configuration")
            elif self.screws == 2:
                self.K = 0.1  # Default for twin screws
                self.logger.info(f"Setting K={self.K} for twin screw configuration")
            else:
                raise ValueError("Invalid number of screws. Must be 1 or 2.")
        else:
            if 0 <= self.K <= 0.1:
                self.screws = 2
                self.logger.info(f"Derived twin screw configuration from K={self.K}")
            elif 0.1 < self.K <= 0.2:
                self.screws = 1
                self.logger.info(f"Derived single screw configuration from K={self.K}")
            else:
                raise ValueError("Invalid value of K. Must be between 0-0.1 or between 0.1-0.2.")

    def _validate_performance_data(self):
        """
        Validate performance curve data if provided.

        :raises ValueError: If validation fails
        """
        # Check if arrays are provided and have the same length
        if self.J is not None or self.Kt is not None or self.Kq is not None:
            if self.J is None or self.Kt is None or self.Kq is None:
                raise ValueError("All three arrays (J, Kt, Kq) must be provided together.")

            if len(self.J) != len(self.Kt) or len(self.J) != len(self.Kq):
                raise ValueError("J, Kt, and Kq arrays must be of the same length.")

            # Convert to numpy arrays if they aren't already
            self.J = np.array(self.J) if not isinstance(self.J, np.ndarray) else self.J
            self.Kt = np.array(self.Kt) if not isinstance(self.Kt, np.ndarray) else self.Kt
            self.Kq = np.array(self.Kq) if not isinstance(self.Kq, np.ndarray) else self.Kq

            self.logger.info(
                f"Validated performance data arrays: {len(self.J)} points from J={min(self.J):.4f} to J={max(self.J):.4f}")

    def _validate_wageningen_parameters(self):
        """
        Validate parameters specific to Wageningen method.

        :raises ValueError: If validation fails
        """
        # Validate the number of propeller blades
        if not (2 <= self.Z <= 7):
            raise ValueError('Blade number is outside of limits for Wageningen (2-7)')

        # Validate the expanded blade area ratio
        if self.AEA0 is None:
            raise ValueError('AEA0 (expanded blade area ratio) must be specified for Wageningen calculations')

        if not (0.3 <= self.AEA0 <= 1.05):
            raise ValueError('Ae/A0 is outside of limits for Wageningen (0.3-1.05)')

        # Validate the Pitch-to-Diameter ratio (P_D)
        if not self.cpp and if self.P_D is None:
            raise ValueError('P_D (pitch-to-diameter ratio) must be specified for Wageningen calculations with fixed pitch')

        if self.P_D is not None:
            if isinstance(self.P_D, np.ndarray):
                if not np.all((0.5 <= self.P_D) & (self.P_D <= 1.40)):
                    raise ValueError('P/D is outside of limits for Wageningen (0.5-1.40)')
            elif not (0.5 <= self.P_D <= 1.40):
                raise ValueError('P/D is outside of limits for Wageningen (0.5-1.40)')

        self.logger.info(f"Validated Wageningen parameters: Z={self.Z}, AEA0={self.AEA0:.4f}, P_D={self.P_D:.4f}")

    def _load_wageningen_data(self):
        """
        Load and calculate Wageningen B-series data using the Wageningen calculator.
        """
        try:
            # Validate parameters needed for Wageningen
            if self.AEA0 is None:
                raise ValueError('AEA0 (expanded blade area ratio) must be specified for Wageningen calculations')

            # Generate J values
            jc = np.arange(0, 1.01, 0.01)  # 0 to 1.0 in steps of 0.01
            self.logger.info(f"Generated {len(jc)} J values from 0 to 1.0")

            # Create P_D as array if it's a single value
            if self.cpp and self.P_D is None:
                # For CPP with no initial P_D, we'll set a default of 0.9 (middle range)
                self.logger.info("Setting initial P_D=0.9 for CPP calculations")
                p_d_array = np.array([0.9])
            else:
                p_d_array = np.array([self.P_D]) if not isinstance(self.P_D, np.ndarray) else self.P_D

            # Create and use the Wageningen calculator with its default file paths
            self.logger.info("Creating Wageningen calculator with default coefficient files")
            self.wageningen_calculator = Wageningen(
                AEAO=self.AEA0,
                zp=self.Z,
                Pd=p_d_array,
                Jc=jc
            )

            # Get the calculated values
            self.J, kt, kq, etaO = self.wageningen_calculator.compute_all()

            # Extract 1D arrays if only one P_D value
            if len(p_d_array) == 1:
                self.Kt = kt[0]
                self.Kq = kq[0]
                self.etaO = etaO[0]
            else:
                self.Kt = kt
                self.Kq = kq
                self.etaO = etaO

            self.logger.info("Successfully calculated Wageningen propeller characteristics")
            self.logger.debug(f"Kt range: {np.min(self.Kt):.4f} to {np.max(self.Kt):.4f}")
            self.logger.debug(f"Kq range: {np.min(self.Kq):.4f} to {np.max(self.Kq):.4f}")

        except Exception as e:
            self.logger.error(f"Failed to load Wageningen data: {e}")
            raise

    @property
    def P_D(self):
        """
        Get the pitch-to-diameter ratio.

        :return: Pitch-to-diameter ratio
        :rtype: float or None
        """
        if self._P_D is None:
            if self.P is not None and self.D is not None:
                result = self.P / self.D
                self.logger.debug(f"Calculated P/D = {result:.4f} from P={self.P} and D={self.D}")
                return result
            else:
                return None
        return self._P_D

    @P_D.setter
    def P_D(self, value):
        """
        Set the pitch-to-diameter ratio.

        :param value: New pitch-to-diameter ratio
        :type value: float
        """
        self._P_D = value
        self.logger.info(f"Set P/D ratio to {value:.4f}")

    @property
    def KtJ2(self):
        """
        Calculate Kt/J^2 values for matching procedures.

        :return: Array of Kt/J^2 values
        :rtype: np.ndarray or None
        """
        if self.J is None or self.Kt is None:
            return None

        # Avoid division by zero by starting from index 1
        if len(self.J) > 1:
            result = self.Kt[1:] / (self.J[1:] ** 2)
            self.logger.debug(f"Calculated KtJ2 array with {len(result)} values")
            return result
        return np.array([])

    def get_Kt(self,
               j_value: float):
        """
        Get thrust coefficient for a given advance coefficient.

        :param j_value: Advance coefficient value
        :type j_value: float

        :return: Thrust coefficient
        :rtype: float or None
        """
        if self.J is None or self.Kt is None:
            self.logger.warning("Cannot get Kt: J or Kt data not available")
            return None

        # Use cached value if available
        if j_value in self._kt_cache:
            return self._kt_cache[j_value]

        # Interpolate value
        kt = np.interp(j_value, self.J, self.Kt)
        self._kt_cache[j_value] = kt

        self.logger.debug(f"Interpolated Kt={kt:.6f} for J={j_value:.4f}")
        return kt

    def get_Kq(self,
               j_value: float):
        """
        Get torque coefficient for a given advance coefficient.

        :param j_value: Advance coefficient value
        :type j_value: float

        :return: Torque coefficient
        :rtype: float or None
        """
        if self.J is None or self.Kq is None:
            self.logger.warning("Cannot get Kq: J or Kq data not available")
            return None

        # Use cached value if available
        if j_value in self._kq_cache:
            return self._kq_cache[j_value]

        # Interpolate value
        kq = np.interp(j_value, self.J, self.Kq)
        self._kq_cache[j_value] = kq

        self.logger.debug(f"Interpolated Kq={kq:.6f} for J={j_value:.4f}")
        return kq

    def get_efficiency(self,
                       j_value: float):
        """
        Calculate open water efficiency for a given advance coefficient.

        :param j_value: Advance coefficient value
        :type j_value: float

        :return: Open water efficiency
        :rtype: float
        """
        # Use cached value if available
        if j_value in self._eff_O_cache:
            return self._eff_O_cache[j_value]

        kt = self.get_Kt(j_value)
        kq = self.get_Kq(j_value)

        if kt is None or kq is None or kq == 0:
            self.logger.warning(f"Cannot calculate efficiency for J={j_value}: invalid Kt or Kq")
            return 0

        efficiency = (j_value * kt) / (2 * np.pi * kq)
        self._eff_O_cache[j_value] = efficiency

        self.logger.info(f"Calculated efficiency={efficiency:.4f} for J={j_value:.4f} using kq={kq:.2f} and kt={kt:.2f}")
        return efficiency

    def calculate_J(self,
                    Va: float,
                    n: float):
        """
        Calculate advance coefficient J.

        :param Va: Advance velocity in m/s
        :type Va: float

        :param n: Rotational speed in revolutions per second
        :type n: float

        :return: Advance coefficient J
        :rtype: float
        """
        if n == 0:
            self.logger.warning("Cannot calculate J: rotational speed is zero")
            return 0

        j = Va / (n * self.D)
        self.logger.debug(f"Calculated J={j:.4f} from Va={Va:.2f}m/s, n={n:.2f}rps, D={self.D}m")
        return j

    def find_J_for_KtJ2(self, ktj2_value: float):
        """
        Find J value for a given Kt/J^2 value - used in propeller matching.
        """
        ktj2_array = self.KtJ2
        j_array = self.J[1:]  # Skip the first element to match KtJ2 length

        if ktj2_array is None or len(ktj2_array) == 0:
            self.logger.warning("Cannot find J for KtJ2: KtJ2 array is empty")
            return None

        # Sort to handle non-monotonic KtJ2 values
        idx = np.argsort(ktj2_array)
        sorted_ktj2 = ktj2_array[idx]
        sorted_j = j_array[idx]

        # Check if value is within range and handle edge cases
        min_ktj2 = min(sorted_ktj2)
        max_ktj2 = max(sorted_ktj2)

        if ktj2_value < min_ktj2:
            self.logger.warning(f"KtJ2 value {ktj2_value:.6f} is below minimum available {min_ktj2:.6f}")
            # For values below min, use a reasonable value rather than extrapolating
            # (maybe use the J corresponding to min_ktj2)
            return sorted_j[np.argmin(sorted_ktj2)]

        elif ktj2_value > max_ktj2:
            self.logger.warning(f"KtJ2 value {ktj2_value:.6f} is above maximum available {max_ktj2:.6f}")
            # For values above max, use a reasonable value rather than extrapolating
            # (maybe use the J corresponding to max_ktj2)
            return sorted_j[np.argmax(sorted_ktj2)]

        # Interpolate to find J value
        j_value = np.interp(ktj2_value, sorted_ktj2, sorted_j)
        self.logger.info(f"Found J={j_value:.4f} for KtJ2={ktj2_value:.6f}")
        return j_value

    def find_rpm_from_J(self,
                        J: float,
                        Va: float):
        """
        A method for finding J
        :param J: advance coefficient
        :return: The rpm required by the given advance coefficient.
        """
        return Va/self.D*J

    def find_optimal_pitch(self, j_value: float, target_kt: float):
        """
        Find the optimal pitch-to-diameter ratio (P_D) for a given advance coefficient (J)
        and target thrust coefficient (Kt) that maximizes efficiency.

        :param j_value: Advance coefficient value
        :type j_value: float

        :param target_kt: Target thrust coefficient (Kt)
        :type target_kt: float

        :return: Tuple of (optimal P_D, maximum efficiency, corresponding Kq)
        :rtype: tuple(float, float, float)
        """
        self.logger.info(f"Finding optimal P_D for J={j_value:.4f} and target Kt={target_kt:.6f}")

        # Define range of P_D values to search (within Wageningen limits)
        p_d_values = np.linspace(0.5, 1.4, 50)  # 50 points between 0.5 and 1.4

        # Store results for each P_D
        efficiencies = []
        kq_values = []
        kt_values = []

        # For each P_D, calculate a new Wageningen propeller and find efficiency at the specified J
        for p_d in p_d_values:
            # Create a temporary Wageningen calculator with this P_D
            temp_calculator = Wageningen(
                AEAO=self.AEA0,
                zp=self.Z,
                Pd=np.array([p_d]),
                Jc=np.array([j_value])
            )

            # Calculate performance for this specific J and P_D
            _, kt, kq, eta = temp_calculator.compute_all()

            # Store results (extracting single values from arrays)
            kt_value = kt[0][0]
            kq_value = kq[0][0]
            eta_value = eta[0][0]

            kt_values.append(kt_value)
            kq_values.append(kq_value)
            efficiencies.append(eta_value)

            self.logger.debug(f"P_D={p_d:.2f}: Kt={kt_value:.6f}, Kq={kq_value:.6f}, η={eta_value:.4f}")

        # Convert to numpy arrays
        p_d_values = np.array(p_d_values)
        kt_values = np.array(kt_values)
        kq_values = np.array(kq_values)
        efficiencies = np.array(efficiencies)

        # Find P_D values that provide the target thrust (within a tolerance)
        tolerance = 0.01  # 1% tolerance on Kt
        valid_indices = np.where(np.abs(kt_values - target_kt) / target_kt < tolerance)[0]

        if len(valid_indices) == 0:
            # If no exact match, find closest P_D to target Kt
            closest_idx = np.argmin(np.abs(kt_values - target_kt))
            self.logger.warning(
                f"No P_D value gives exactly Kt={target_kt:.6f} within tolerance. "
                f"Using closest match: P_D={p_d_values[closest_idx]:.4f} with Kt={kt_values[closest_idx]:.6f}"
            )
            valid_indices = [closest_idx]

        # Among valid P_D values, find the one with maximum efficiency
        valid_efficiencies = efficiencies[valid_indices]
        max_eff_idx = valid_indices[np.argmax(valid_efficiencies)]

        optimal_p_d = p_d_values[max_eff_idx]
        max_efficiency = efficiencies[max_eff_idx]
        corresponding_kq = kq_values[max_eff_idx]

        self.logger.info(
            f"Optimal P_D={optimal_p_d:.4f} gives efficiency η={max_efficiency:.4f} "
            f"with Kt={kt_values[max_eff_idx]:.6f}, Kq={corresponding_kq:.6f}"
        )

        # Optionally plot the results
        if self.debug:
            plt.figure(figsize=(12, 8))

            # Plot efficiency vs P_D
            plt.subplot(2, 2, 1)
            plt.plot(p_d_values, efficiencies)
            plt.scatter(optimal_p_d, max_efficiency, color='red', s=80, zorder=5)
            plt.xlabel('P/D Ratio')
            plt.ylabel('Efficiency (η)')
            plt.title('Efficiency vs P/D')
            plt.grid(True)

            # Plot Kt vs P_D
            plt.subplot(2, 2, 2)
            plt.plot(p_d_values, kt_values)
            plt.axhline(y=target_kt, color='r', linestyle='--', label=f'Target Kt={target_kt:.6f}')
            plt.scatter(optimal_p_d, kt_values[max_eff_idx], color='red', s=80, zorder=5)
            plt.xlabel('P/D Ratio')
            plt.ylabel('Thrust Coefficient (Kt)')
            plt.title('Kt vs P/D')
            plt.grid(True)
            plt.legend()

            # Plot Kq vs P/D
            plt.subplot(2, 2, 3)
            plt.plot(p_d_values, kq_values)
            plt.scatter(optimal_p_d, corresponding_kq, color='red', s=80, zorder=5)
            plt.xlabel('P/D Ratio')
            plt.ylabel('Torque Coefficient (Kq)')
            plt.title('Kq vs P/D')
            plt.grid(True)

            # Highlight valid range
            plt.subplot(2, 2, 4)
            plt.plot(p_d_values, np.abs(kt_values - target_kt) / target_kt)
            plt.axhline(y=tolerance, color='r', linestyle='--', label=f'Tolerance={tolerance:.2f}')
            plt.xlabel('P/D Ratio')
            plt.ylabel('Relative Kt Error')
            plt.title('Kt Error vs P/D')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

        return optimal_p_d, max_efficiency, corresponding_kq

    def get_cpp_performance(self, j_value: float, required_thrust: float, water_density: float = 1025, n: float = None,
                            d: float = None):
        """
        Calculate the performance of a controllable pitch propeller (CPP) for a given operating condition.
        Finds and applies the optimal pitch setting for the given conditions.

        :param j_value: Advance coefficient
        :type j_value: float

        :param required_thrust: Required thrust in Newtons
        :type required_thrust: float

        :param water_density: Water density in kg/m³, defaults to 1025 (seawater)
        :type water_density: float

        :param n: Propeller rotational speed in revolutions per second, optional
        :type n: float

        :param d: Propeller diameter to use (defaults to self.D if not provided)
        :type d: float

        :return: Dictionary with performance data: {'P_D', 'Kt', 'Kq', 'efficiency', 'thrust', 'torque', 'power'}
        :rtype: dict
        """
        if not self.cpp:
            self.logger.warning("This method is intended for CPP propellers. Set cpp=True when initializing.")

        # Use class diameter if not specified
        d = self.D if d is None else d

        # Calculate target Kt if actual thrust is provided
        if n is not None:
            # Convert thrust to Kt: Kt = T / (ρ * n² * D⁴)
            target_kt = required_thrust / (water_density * n ** 2 * d ** 4)
        else:
            # If n not provided, assume Kt is directly provided
            target_kt = required_thrust

        # Find optimal pitch for this operating point
        optimal_p_d, efficiency, kq = self.find_optimal_pitch(j_value, target_kt)

        # Apply the optimal pitch to this propeller
        self.P_D = optimal_p_d

        # Calculate physical values if n is provided
        if n is not None:
            thrust = target_kt * water_density * n ** 2 * d ** 4
            torque = kq * water_density * n ** 2 * d ** 5
            power = 2 * np.pi * n * torque  # Power in watts
        else:
            thrust = None
            torque = None
            power = None

        # Return performance data
        return {
            'P_D': optimal_p_d,
            'Kt': target_kt,
            'Kq': kq,
            'efficiency': efficiency,
            'thrust': thrust,
            'torque': torque,
            'power': power
        }


    def plot_curves(self, save_path=None):
        """
        Plot propeller performance curves.

        :param save_path: Path to save the plot image (optional)
        :type save_path: str or None

        :return: None
        """
        if self.J is None or self.Kt is None or self.Kq is None:
            self.logger.error("Cannot plot: J, Kt, or Kq data not available")
            return

        self.logger.info("Generating propeller performance curves plot")

        plt.figure(figsize=(10, 6))
        plt.plot(self.J, self.Kt, 'b-', label='Kt')
        plt.plot(self.J, 10 * self.Kq, 'r-', label='10×Kq')

        # Calculate and plot efficiency
        eff = np.zeros_like(self.J)
        for i in range(len(self.J)):
            j_val = self.J[i]
            kq_val = self.Kq[i]
            if j_val > 0 and kq_val > 0:
                eff[i] = (j_val * self.Kt[i]) / (2 * np.pi * kq_val)

        plt.plot(self.J, eff, 'g-', label='η₀')

        plt.grid(True)
        plt.xlabel('Advance Coefficient (J)')
        plt.ylabel('Coefficient Value')
        plt.title(f'Propeller Performance Curves (D={self.D}m, Z={self.Z}, P/D={self.P_D:.2f})')
        plt.legend()
        plt.xlim(0, max(self.J))
        plt.ylim(0, max(1.0, max(self.Kt) * 1.1))

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved propeller performance plot to {save_path}")

        plt.show()

    def __str__(self):
        """
        Return a string representation of the propeller.

        :return: String description of the propeller
        :rtype: str
        """
        desc = [
            f"Propeller: D={self.D}m, Z={self.Z} blades",
            f"P/D ratio: {self.P_D:.4f}" if self.P_D is not None else "P/D ratio: Not specified",
            f"Configuration: {self.screws} screw(s), K={self.K:.4f}",
        ]

        if self.wageningen:
            desc.append(f"Type: Wageningen B-series (AEA0={self.AEA0:.4f})")
        else:
            desc.append("Type: Custom propeller characteristics")

        if self.J is not None:
            desc.append(f"Performance data: {len(self.J)} points from J={min(self.J):.4f} to J={max(self.J):.4f}")

        return "\n".join(desc)


class Wageningen:
    """
    This class calculates propeller parameters, using the Wageningen 'B-series' method

    """
    def __init__(self,
                 AEAO: float,
                 zp: float,
                 Pd: np.ndarray,
                 Jc: np.ndarray,
                 elikt=WAGENINGEN_KT_FILE,
                 elikq= WAGENINGEN_KQ_FILE):
        """
        Initialising the Wageningen class

        :param AEAO: Expanded blade area ratio
        :type AEAO: float

        :param zp: Blade number
        :type zp: float

        :param Pd: Pitch over diameter numbers
        :type Pd: np.ndarray

        :param Jc: Advance coefficient
        :type Jc: np.ndarray

        :param elikt: Thrust coefficient
        :type elikt: np.ndarray

        :param elikq: Torque coefficient
        :type elikq: np.ndarray
        """

        self.AEAO = AEAO
        self.zp  = zp
        self.Pd = Pd
        self.Jc = Jc

        self.ni = 1.19e-6  # + [m2/s] kinematic viscosity

        # Handle file loading or direct data assignment
        if isinstance(elikt, (str, Path)):
            self.elikt = np.loadtxt(elikt)
        else:
            self.elikt = elikt

        if isinstance(elikq, (str, Path)):
            self.elikq = np.loadtxt(elikq)
        else:
            self.elikq = elikq

        self.ni = 1.19e-6

        self._validate_inputs()


    def _validate_inputs(self):
        """
        validate input parameters for the Wageningen class

        :raises ValueError: if the input parameters are invalid.
        """

        if not (0.3 <= self.AEAO <= 1.05):
            raise ValueError('Ae/Ao is outside of limits (0.3-1.05)')
        if not (2 <= self.zp <= 7):
            raise ValueError('Blade number is outside of limits (2-7)')
        if isinstance(self.Pd, np.ndarray):
            if not np.all((0.5 <= self.Pd) & (self.Pd <= 1.40)):
                raise ValueError('P/D is outside of limits (0.5-1.40)')
        elif not (0.5 <= self.Pd <= 1.40):
            raise ValueError('P/D is outside of limits (0.5-1.40)')

    def compute_kt(self):
        """
        Compute the thrust coefficient (kt).


        :return: Thrust coefficient (kt).
        :rtype: np.ndarray
        """

        nt = self.elikt.shape[0]
        jc, pd = np.meshgrid(self.Jc, self.Pd)
        self._kt = np.zeros_like(jc)
        for i in range(nt):
            self._kt += (self.elikt[i, 0] * jc ** self.elikt[i, 1] * pd ** self.elikt[i, 2] *
                             self.AEAO ** self.elikt[i, 3] * self.zp ** self.elikt[i, 4])
        return self._kt

    def compute_kq(self):
        """
        Compute the torque coefficient (kq).

        :return: Torque coefficient (kq).
        :rtype: np.ndarray
        """
        nq = self.elikq.shape[0]
        jc, pd = np.meshgrid(self.Jc, self.Pd)
        self._kq = np.zeros_like(jc)
        for i in range(nq):
            self._kq += (self.elikq[i, 0] * jc ** self.elikq[i, 1] * pd ** self.elikq[i, 2] *
                             self.AEAO ** self.elikq[i, 3] * self.zp ** self.elikq[i, 4])
        return self._kq

    def compute_etaO(self):
        """
        Compute the open water efficiency (eta0).

        :return: Open water efficiency (etaO).
        :rtype: float
        """

        kt = self.compute_kt()
        kq = self.compute_kq()
        self._etaO = (self.Jc * kt) / (2 * np.pi * kq)
        return self._etaO


    def compute_all(self):
        """
        Compute all propeller parameters.

        :return:  A tuple containing advance coefficient, thrust coefficient, torque coefficient, and open water efficiency.
        :rtype: tuple of (np.ndarray, np.ndarray, np.ndarray, float).
        """
        kt = self.compute_kt()
        kq = self.compute_kq()
        etaO = self.compute_etaO()

        return self.Jc, kt, kq, etaO


    def plot_results(self, kt, kq, etaO):
        """
        Plot the computed results.

        :param kt: Thrust coefficient.
        :type kt: np.ndarray

        :param kq: Torque coefficient.
        :type kq: np.ndarray

        :param etaO: Open water efficiency.
        :type etaO: float
        """
        plt.figure()
        plt.plot(self.Jc, kt.T, label='Kt')
        plt.plot(self.Jc, 10 * kq.T, label='10*Kq')
        plt.plot(self.Jc, etaO.T, label='etaO')
        plt.xlim([0, 1.1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_results_sns(self, kt, kq, etaO):
        """
        Plot the computed results with descriptive labels and Seaborn style.

        :param kt: Thrust coefficient (Kt).
        :type kt: np.ndarray

        :param kq: Torque coefficient (Kq).
        :type kq: np.ndarray

        :param etaO: Open water efficiency (etaO).
        :type etaO: np.ndarray
        """
        sns.set(style="whitegrid")

        # Create a new figure for the plot
        plt.figure(figsize=(10, 6))

        # Plot Kt (thrust coefficient)
        plt.plot(self.Jc, kt.T, label='Thrust Coefficient (Kt)', color='b', linewidth=2)

        # Plot 10*Kq (scaled torque coefficient)
        plt.plot(self.Jc, 10 * kq.T, label='Torque Coefficient (10 * Kq)', color='r', linestyle='--', linewidth=2)

        # Plot etaO (open water efficiency)
        plt.plot(self.Jc, etaO.T, label='Open Water Efficiency (etaO)', color='g', linestyle='-.', linewidth=2)

        # Add titles and labels
        plt.title('Propeller Performance Characteristics', fontsize=16)
        plt.xlabel('Advance Coefficient (J)', fontsize=14)
        plt.ylabel('Coefficients and Efficiency', fontsize=14)

        # Set limits for better visualization
        plt.xlim([0, 1.1])
        plt.ylim([0, 1])

        # Enable grid and legend
        plt.grid(True)
        plt.legend(loc='best', fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()

