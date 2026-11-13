class ShipPropulsionFactors(LoggerMixin):
    """
    A class representing ship propulsion factors based on shape and propeller parameters.

    This class supports multiple calculation modes:
    1. 'resistance': Calculate propulsion factors and power from ship resistance
    2. 'rpm': Calculate resistance and power from propeller RPM

    The class handles the relationships between resistance, propeller performance,
    efficiency factors, and power requirements.
    """

    # Valid calculation modes
    RESISTANCE_MODE = 'resistance'
    RPM_MODE = 'rpm'
    VALID_MODES = [RESISTANCE_MODE, RPM_MODE]

    def __init__(self,
                 ship_shape: ShipShapeExplorer,
                 ship_prop: ShipPropeller,
                 ship_resistance: ResistanceCalculator,
                 calculation_mode: str = 'resistance',
                 shaft_rotation: Optional[float] = None,
                 _eff_0: Optional[float] = None,
                 _eff_R: Optional[float] = None,
                 _eff_S: float = 0.99,
                 _eff_gb: float = 0.99,
                 holtrop_correction: bool = True,

                 enable_cpp_optimization: bool = True,
                 rpm_range: Tuple[float, float] = (60, 300),
                 rpm_optimization_tolerance: float = 0.02,

                 verbose: bool = True,
                 debug: bool = False
                 ):
        """
        Initialize the ShipPropulsionFactors class.

        :param ship_shape: Shape parameters of the ship
        :param ship_prop: Propeller parameters of the ship
        :param ship_resistance: Resistance components of the ship
        :param calculation_mode: Mode of calculation ('resistance' or 'rpm')
        :param shaft_rotation: Propeller shaft rotation (required if mode='rpm')
        :param _eff_0: Optional override for open water efficiency
        :param _eff_R: Optional override for relative rotative efficiency
        :param _eff_S: Shaft efficiency (default 0.99)
        :param _eff_gb: Gear box efficiency (default 0.99)
        :param enable_cpp_optimization: Enable automatic RPM/P-D optimization for CPP

        :param rpm_range: Range of RPM values to search for optimization (min, max)
        :param rpm_optimization_tolerance: Thrust matching tolerance for optimization
        :param holtrop_correction: Holtrop correction (default True) - a correction method for kt and kq

        Raises:
            ValueError: If calculation_mode is invalid or required parameters are missing
        """
        # Validate calculation mode
        if calculation_mode not in self.VALID_MODES:
            raise ValueError(f"calculation_mode must be one of {self.VALID_MODES}")

        # Validate required parameters based on mode
        if calculation_mode == self.RPM_MODE and shaft_rotation is None:
            raise ValueError(f"shaft_rotation must be provided when calculation_mode='{self.RPM_MODE}'")

        if verbose:
            self.logger.setLevel(logging.INFO)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        # Initialize basic parameters
        self.shipshape = ship_shape
        self.ship_propeller = ship_prop
        self.resistance_calc = ship_resistance
        self.calculation_mode = calculation_mode
        self.shaft_rotation = shaft_rotation

        # Initialize efficiency parameters
        self._eff_0 = _eff_0
        self._eff_R = _eff_R
        self._eff_S = _eff_S
        self._eff_gb = _eff_gb

        # CPP Optimization parameters
        self.enable_cpp_optimization = enable_cpp_optimization
        self.rpm_range = rpm_range
        self.rpm_optimization_tolerance = rpm_optimization_tolerance

        # Physical constants
        self.g = GRAVITY
        self.rho_SW = SEAWATER_DENSITY
        self.nu = NU
        self.p0pv = P0PV

        self.logger.info(f"Initialized in {calculation_mode}-driven mode")

        # Cached values for expensive calculations
        self.is_cpp = self.ship_propeller.parameters.type is PropellerType.CPP
        self._cached_values = {}
        if self.is_cpp:
            self._initialize_cpp_performance()

        self.holtrop_correction = holtrop_correction

        self.logger.info(f"Initialized in {calculation_mode}-driven mode with "
                         f"{'controllable' if self.is_cpp else 'fixed'} pitch propeller")

    def _check_parameters(self):
        """
        Checks parameters of different systems and raises warnings
        :return:
        """
        # Checking T_A value
        if self.shipshape.T_A >= (self.ship_propeller.parameters.D + 0.2 + 0.4):
            self.logger.warning(
                f'T_A value {self.shipshape.T_A} is greater than D +0.2+0.4. potential problem '
                f'according to Papanikolaou page 137)')

    def find_optimal_cpp_operating_point(self,
                                         required_thrust: float,
                                         speed_of_advance: float,
                                         rpm_range: Optional[Tuple[float, float]] = None,
                                         rpm_step_coarse: float = 10.0,
                                         rpm_step_fine: float = 2.0,
                                         thrust_tolerance: float = None) -> Dict:
        """
        Find the optimal RPM and P/D combination for a CPP propeller that maximizes efficiency
        while meeting the required thrust.

        This method implements the global optimization algorithm discussed earlier.

        :param required_thrust: Required thrust in Newtons
        :param speed_of_advance: Ship speed through water (m/s)
        :param rpm_range: RPM search range (min, max), uses instance default if None
        :param rpm_step_coarse: Coarse search step size in RPM
        :param rpm_step_fine: Fine search step size in RPM
        :param thrust_tolerance: Acceptable thrust matching tolerance, uses instance default if None
        :return: Dictionary with optimal operating point and all search results
        """

        if not self.is_cpp:
            raise ValueError("CPP optimization is only available for controllable pitch propellers")

        if not hasattr(self.ship_propeller, 'cpp_calculator') or self.ship_propeller.cpp_calculator is None:
            raise ValueError("CPP calculator not available in propeller instance")

        # Use instance defaults if not provided
        if rpm_range is None:
            rpm_range = self.rpm_range
        if thrust_tolerance is None:
            thrust_tolerance = self.rpm_optimization_tolerance

        self.logger.info(f"🎯 Starting CPP optimization for thrust={required_thrust / 1000:.1f} kN, "
                         f"Va={speed_of_advance:.2f} m/s")

        # STAGE 1: Coarse search
        coarse_results = self._cpp_optimization_search(
            required_thrust=required_thrust,
            speed_of_advance=speed_of_advance,
            rpm_range=rpm_range,
            rpm_step=rpm_step_coarse,
            thrust_tolerance=thrust_tolerance,
            search_type="coarse"
        )

        if not coarse_results['valid_results']:
            raise ValueError("No valid operating points found in coarse search")

        # STAGE 2: Fine search around optimum
        coarse_optimum = coarse_results['optimal']
        fine_rpm_range = (
            max(rpm_range[0], coarse_optimum['rpm'] - 20),
            min(rpm_range[1], coarse_optimum['rpm'] + 20)
        )

        self.logger.info(f"🔍 Refining search around {coarse_optimum['rpm']:.1f} RPM "
                         f"(range: {fine_rpm_range[0]:.1f}-{fine_rpm_range[1]:.1f})")

        fine_results = self._cpp_optimization_search(
            required_thrust=required_thrust,
            speed_of_advance=speed_of_advance,
            rpm_range=fine_rpm_range,
            rpm_step=rpm_step_fine,
            thrust_tolerance=thrust_tolerance * 0.5,  # Tighter tolerance for fine search
            search_type="fine"
        )

        # Return the refined results with comparison
        optimal_result = fine_results['optimal']

        self.logger.info(f"✅ CPP OPTIMIZATION COMPLETE")
        self.logger.info(f"   Optimal RPM: {optimal_result['rpm']:.1f}")
        self.logger.info(f"   Optimal P/D: {optimal_result['P_D']:.3f}")
        self.logger.info(f"   Max Efficiency: {optimal_result['efficiency']:.3f}")
        self.logger.info(f"   Required Power: {optimal_result['power_kw']:.1f} kW")

        return {
            'optimal': optimal_result,
            'coarse_search': coarse_results,
            'fine_search': fine_results,
            'improvement': {
                'efficiency_gain': fine_results['optimal']['efficiency'] - coarse_results['optimal']['efficiency'],
                'power_reduction': coarse_results['optimal']['power_kw'] - fine_results['optimal']['power_kw']
            }
        }

    def _cpp_optimization_search(self,
                                 required_thrust: float,
                                 speed_of_advance: float,
                                 rpm_range: Tuple[float, float],
                                 rpm_step: float,
                                 thrust_tolerance: float,
                                 search_type: str = "search") -> Dict:
        """
        Perform CPP optimization search over specified RPM range.

        :param required_thrust: Required thrust in Newtons
        :param speed_of_advance: Speed of advance in m/s
        :param rpm_range: RPM search range (min, max)
        :param rpm_step: RPM step size
        :param thrust_tolerance: Thrust matching tolerance
        :param search_type: Type of search for logging purposes
        :return: Dictionary with search results
        """

        # Convert RPM range to revolutions per second
        n_min, n_max = rpm_range[0] / 60, rpm_range[1] / 60
        n_values = np.arange(n_min, n_max + rpm_step / 60, rpm_step / 60)

        results = []
        propeller_diameter = self.ship_propeller.parameters.D

        self.logger.debug(f"🔄 {search_type.title()} search: {len(n_values)} RPM values "
                          f"from {rpm_range[0]:.1f} to {rpm_range[1]:.1f} RPM")

        for n in n_values:
            try:
                # Calculate advance coefficient: J = Va/(n*D)
                J = speed_of_advance / (n * propeller_diameter)

                # Skip if J is outside calculator range
                cpp_calc = self.ship_propeller.cpp_calculator
                if J < min(cpp_calc.j_values) or J > max(cpp_calc.j_values):
                    continue

                # Calculate target thrust coefficient
                target_kt = required_thrust / (self.rho_SW * n ** 2 * propeller_diameter ** 4)

                # Skip if target Kt is unrealistic
                if target_kt <= 0 or target_kt > 1.0:
                    continue

                # Find optimal pitch for this operating point using your existing method
                optimal_pd, efficiency, kq = cpp_calc.find_optimal_pitch(
                    J, target_kt, tolerance=thrust_tolerance
                )

                # Calculate actual performance values
                actual_thrust = target_kt * self.rho_SW * n ** 2 * propeller_diameter ** 4
                torque = kq * self.rho_SW * n ** 2 * propeller_diameter ** 5
                power = 2 * np.pi * n * torque  # Watts

                # Store comprehensive results
                results.append({
                    'rpm': n * 60,
                    'n_rps': n,
                    'J': J,
                    'P_D': optimal_pd,
                    'Kt': target_kt,
                    'Kq': kq,
                    'efficiency': efficiency,
                    'thrust': actual_thrust,
                    'torque': torque,
                    'power': power,
                    'power_kw': power / 1000,
                    'power_hp': power / 745.7,
                    'thrust_error': abs(actual_thrust - required_thrust) / required_thrust
                })

            except Exception as e:
                self.logger.debug(f"⚠️  Failed to optimize at {n * 60:.1f} RPM: {e}")
                continue

        if not results:
            return {'valid_results': [], 'optimal': None}

        # Find the operating point with maximum efficiency
        efficiencies = [r['efficiency'] for r in results]
        best_idx = np.argmax(efficiencies)
        optimal_result = results[best_idx]

        return {
            'valid_results': results,
            'optimal': optimal_result,
            'search_stats': {
                'total_points': len(n_values),
                'valid_points': len(results),
                'success_rate': len(results) / len(n_values),
                'efficiency_range': (min(efficiencies), max(efficiencies))
            }
        }

    def _initialize_cpp_performance(self):
        """
        Enhanced CPP performance initialization with automatic optimization.

        This method now supports automatic RPM/P-D optimization when shaft_rotation
        is not provided in resistance mode.
        """
        if not self.is_cpp:
            return

        # Skip if already calculated
        if '_cpp_performance' in self._cached_values:
            return

        try:
            if self.calculation_mode == self.RPM_MODE:
                # RPM mode: use provided shaft rotation
                if self.shaft_rotation is None:
                    raise ValueError("shaft_rotation required for RPM mode")

                n = self.shaft_rotation / 60
                j_value = self.V_A / (n * self.ship_propeller.parameters.D)

                # Calculate required thrust from resistance
                resistance = self.resistance_calc.total_resistance()
                thrust_n = resistance * 1000 / (1 - self.t_factor)

                # Get CPP performance at specified RPM
                cpp_performance = self.ship_propeller.get_cpp_performance(
                    j_value=j_value,
                    required_thrust=thrust_n,
                    n=n
                )

                # Store results
                self._store_cpp_performance_results(cpp_performance, optimization_used=False)

            else:
                # RESISTANCE MODE: Two approaches based on settings
                resistance = self.resistance_calc.total_resistance()
                thrust_n = resistance * 1000 / (1 - self.t_factor)

                if self.enable_cpp_optimization and self.shaft_rotation is None:
                    # NEW: Use global optimization to find best RPM/P-D combination
                    self.logger.info("🚀 Using CPP optimization to find optimal operating point")

                    optimization_results = self.find_optimal_cpp_operating_point(
                        required_thrust=thrust_n,
                        speed_of_advance=self.V_A
                    )

                    optimal_point = optimization_results['optimal']

                    # Store optimized results
                    cpp_performance = {
                        'P_D': optimal_point['P_D'],
                        'Kt': optimal_point['Kt'],
                        'Kq': optimal_point['Kq'],
                        'efficiency': optimal_point['efficiency'],
                        'thrust': optimal_point['thrust'],
                        'torque': optimal_point['torque'],
                        'power': optimal_point['power']
                    }

                    # Also store the optimal RPM for later use
                    self._cached_values['_optimized_rpm'] = optimal_point['rpm']
                    self._cached_values['_optimization_results'] = optimization_results

                    self._store_cpp_performance_results(cpp_performance, optimization_used=True)

                elif self.shaft_rotation is not None:
                    # Traditional approach: use provided RPM
                    n = self.shaft_rotation / 60
                    j_value = self.V_A / (n * self.ship_propeller.parameters.D)

                    cpp_performance = self.ship_propeller.get_cpp_performance(
                        j_value=j_value,
                        required_thrust=thrust_n,
                        n=n
                    )

                    self._store_cpp_performance_results(cpp_performance, optimization_used=False)

                else:
                    # Fallback: use traditional resistance-based approach
                    self.logger.info("Using traditional resistance-based CPP approach")
                    self._use_traditional_cpp_approach(thrust_n)

        except Exception as e:
            self.logger.error(f"Error initializing CPP performance: {str(e)}")
            self._provide_cpp_fallback_values()

    def _store_cpp_performance_results(self, cpp_performance: Dict, optimization_used: bool):
        """Helper method to store CPP performance results in cache."""
        self._cached_values['_cpp_performance'] = cpp_performance
        self._cached_values['_cpp_optimal_pitch'] = cpp_performance['P_D']
        self._cached_values['_cpp_kt'] = cpp_performance['Kt']
        self._cached_values['_cpp_kq'] = cpp_performance['Kq']
        self._cached_values['_cpp_efficiency'] = cpp_performance['efficiency']
        self._cached_values['_optimization_used'] = optimization_used

        optimization_msg = "with global optimization" if optimization_used else "at specified operating point"
        self.logger.info(f"CPP performance initialized {optimization_msg}: "
                         f"P/D={cpp_performance['P_D']:.4f}, η={cpp_performance['efficiency']:.4f}")

    def _use_traditional_cpp_approach(self, thrust_n: float):
        """Use traditional resistance-based approach for CPP."""
        # This implements your original approach
        ktj2 = self.get_ktj2_factor(self.resistance_calc.total_resistance())
        J_from_Kt2 = self.ship_propeller.find_J_for_KtJ2(ktj2)

        # Find optimal pitch for this J and thrust
        optimal_pd, efficiency, kq = self.ship_propeller.cpp_calculator.find_optimal_pitch(
            J_from_Kt2,
            thrust_n / (self.rho_SW * ((self.V_A / (J_from_Kt2 * self.ship_propeller.parameters.D)) ** 2) * (
                        self.ship_propeller.parameters.D ** 4))
        )

        cpp_performance = {
            'P_D': optimal_pd,
            'Kt': thrust_n / (self.rho_SW * ((self.V_A / (J_from_Kt2 * self.ship_propeller.parameters.D)) ** 2) * (
                        self.ship_propeller.parameters.D ** 4)),
            'Kq': kq,
            'efficiency': efficiency,
            'thrust': thrust_n,
            'torque': kq * self.rho_SW * ((self.V_A / (J_from_Kt2 * self.ship_propeller.parameters.D)) ** 2) * (
                        self.ship_propeller.parameters.D ** 5),
            'power': 2 * np.pi * (self.V_A / (J_from_Kt2 * self.ship_propeller.parameters.D)) * kq * self.rho_SW * (
                        (self.V_A / (J_from_Kt2 * self.ship_propeller.parameters.D)) ** 2) * (
                                 self.ship_propeller.parameters.D ** 5)
        }

        self._store_cpp_performance_results(cpp_performance, optimization_used=False)

    def _provide_cpp_fallback_values(self):
        """Provide fallback values if CPP initialization fails."""
        self.logger.warning("Providing fallback CPP values to prevent cascading errors")
        fallback_values = {'P_D': 0.8, 'Kt': 0.1, 'Kq': 0.02, 'efficiency': 0.6}
        self._store_cpp_performance_results(fallback_values, optimization_used=False)

    # Enhanced property methods that utilize optimization results

    @property
    def actual_rpm(self) -> float:
        """
        Get actual or calculated RPM based on calculation mode.
        Now supports optimized RPM for CPP propellers.
        """
        if self.calculation_mode == self.RPM_MODE:
            return self.shaft_rotation
        else:
            # Check if we have optimized RPM from CPP optimization
            if '_optimized_rpm' in self._cached_values:
                return self._cached_values['_optimized_rpm']

            # Fall back to traditional calculation
            if '_cached_rpm' not in self._cached_values:
                rpm = self.required_rpm()
                self._cached_values['_cached_rpm'] = rpm
            return self._cached_values['_cached_rpm']

    def get_optimization_summary(self) -> Optional[Dict]:
        """
        Get a summary of the CPP optimization results if optimization was used.

        :return: Dictionary with optimization summary or None if not optimized
        """
        if not self._cached_values.get('_optimization_used', False):
            return None

        if '_optimization_results' not in self._cached_values:
            return None

        results = self._cached_values['_optimization_results']
        optimal = results['optimal']

        return {
            'optimization_used': True,
            'optimal_rpm': optimal['rpm'],
            'optimal_pd': optimal['P_D'],
            'max_efficiency': optimal['efficiency'],
            'power_required': optimal['power_kw'],
            'advance_coefficient': optimal['J'],
            'search_statistics': results['fine_search']['search_stats'],
            'efficiency_improvement': results.get('improvement', {}).get('efficiency_gain', 0),
            'power_reduction': results.get('improvement', {}).get('power_reduction', 0)
        }

    def plot_cpp_optimization_results(self) -> Optional[plt.Figure]:
        """
        Plot the CPP optimization results if available.

        :return: matplotlib Figure or None if no optimization results available
        """
        if not self._cached_values.get('_optimization_used', False):
            self.logger.warning("No CPP optimization results available to plot")
            return None

        if '_optimization_results' not in self._cached_values:
            return None

        results = self._cached_values['_optimization_results']
        fine_results = results['fine_search']['valid_results']
        optimal = results['optimal']

        if not fine_results:
            self.logger.warning("No valid optimization results to plot")
            return None

        # Extract data for plotting
        rpms = [r['rpm'] for r in fine_results]
        efficiencies = [r['efficiency'] for r in fine_results]
        powers = [r['power_kw'] for r in fine_results]
        pds = [r['P_D'] for r in fine_results]

        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Efficiency vs RPM
        ax1.plot(rpms, efficiencies, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axvline(optimal['rpm'], color='r', linestyle='--', linewidth=2, label='Optimal')
        ax1.scatter(optimal['rpm'], optimal['efficiency'], color='red', s=150, zorder=5, marker='*')
        ax1.set_xlabel('RPM')
        ax1.set_ylabel('Propeller Efficiency')
        ax1.set_title('CPP Efficiency Optimization')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Power vs RPM
        ax2.plot(rpms, powers, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.axvline(optimal['rpm'], color='r', linestyle='--', linewidth=2, label='Optimal')
        ax2.scatter(optimal['rpm'], optimal['power_kw'], color='red', s=150, zorder=5, marker='*')
        ax2.set_xlabel('RPM')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('Power Requirement vs RPM')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # P/D vs RPM
        ax3.plot(rpms, pds, 'm-', linewidth=2, marker='^', markersize=4)
        ax3.axvline(optimal['rpm'], color='r', linestyle='--', linewidth=2, label='Optimal')
        ax3.scatter(optimal['rpm'], optimal['P_D'], color='red', s=150, zorder=5, marker='*')
        ax3.set_xlabel('RPM')
        ax3.set_ylabel('Optimal P/D Ratio')
        ax3.set_title('Optimal Pitch vs RPM')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Efficiency vs Power trade-off
        scatter = ax4.scatter(powers, efficiencies, c=rpms, cmap='viridis', s=60, alpha=0.7)
        ax4.scatter(optimal['power_kw'], optimal['efficiency'], color='red', s=200, zorder=5, marker='*')
        ax4.set_xlabel('Power (kW)')
        ax4.set_ylabel('Efficiency')
        ax4.set_title('Efficiency vs Power Trade-off')
        ax4.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('RPM')

        plt.suptitle(f'CPP Optimization Results\n'
                     f'Optimal: {optimal["rpm"]:.1f} RPM, P/D={optimal["P_D"]:.3f}, '
                     f'η={optimal["efficiency"]:.3f}', fontsize=14)
        plt.tight_layout()

        return fig