# """
# This explorer enables user to
# """
#
# import pandas as pd
# import numpy as np
# import logging
# from typing import List, Dict, Optional, Callable, Any, Union, Type, Tuple
# from pathlib import Path
#
# from parameters.resistance_calculator import ResistanceCalculator, ResistanceModelFactory, AndersenWindResistance
# from config import LoggerMixin, ResistanceModelConfig, ResistanceModelType, ResistanceModelName


class ShipParameterExplorer(LoggerMixin):
    """
    A specialized transformer for calculating ship resistance while varying ship parameters
    rather than speed. This allows for parametric studies of how ship dimensions affect resistance
    at a fixed speed.
    """

    def __init__(self,
                 base_shipbasic: ShipShapeBasic,
                 fixed_speed: float,
                 explorer_class: Optional[Type[ShipShapeExplorer]] = None,
                 resistance_configs: Optional[List[ResistanceModelConfig]] = None,
                 ship_propeller: Optional[ShipPropeller] = None,
                 parameter_derivations: Optional[Dict[str, Callable]] = None,
                 correction_library: Optional[Dict[str, CorrectionFactor]] = None,
                 verbose: bool = False,
                 debug: bool = False):
        """
        Initialize the transformer with a base ship configuration and fixed speed.

        Args:
            base_shipbasic: Base ship configuration that will be modified
            fixed_speed: Fixed speed in knots at which resistance will be calculated
            explorer_class: Custom ShipShapeExplorer class to use (defaults to base ShipShapeExplorer)
            resistance_configs: List of resistance model configurations
            ship_propeller: Optional propeller parameters
            parameter_derivations: Custom functions to derive parameters
            correction_library: Corrections to apply to calculated parameters
        """
        self.base_shipbasic = base_shipbasic
        self.fixed_speed = fixed_speed
        self.explorer_class = explorer_class or ShipShapeExplorer
        self.ship_propeller = ship_propeller
        self.parameter_derivations = parameter_derivations or {}
        self.correction_library = correction_library or {}
        self.verbose = verbose
        self.debug = debug

        # Set up logger
        if verbose:
            self.logger.setLevel(logging.INFO)
        elif debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        # Set up default resistance configuration if none provided
        if resistance_configs is None:
            self.resistance_configs = [
                ResistanceModelConfig(
                    model_type=ResistanceModelType.CALM_WATER,
                    model_name=ResistanceModelName.HOLTROP
                )
            ]
        else:
            self.resistance_configs = resistance_configs

        # Create the resistance model factory
        self.factory = ResistanceModelFactory()

    def create_modified_shipbasic(self,
                                  fix_derived_params: bool = True,
                                  **param_changes) -> ShipShapeBasic:
        """
        Create a new ShipShapeBasic with modified parameters.

        Args:
            fix_derived_params: If True, keep original values for derived parameters like L_PP, B, T
                                If False, allow them to be derived from regression if not explicitly provided
            **param_changes: Keyword arguments with parameter changes to apply to the base ship
                             (e.g., L=120, B=16, T=8, etc.)

        Returns:
            A new ShipShapeBasic instance with the modified parameters
        """
        # Start with the base ship's core attributes
        params = {
            'shiptype': self.base_shipbasic.shiptype,
            'shipdata': self.base_shipbasic.shipdata,
            'verbose': self.verbose,
            'debug': self.debug
        }

        # Copy the main dimensional parameters
        params.update({
            'L': self.base_shipbasic.L,
            'C_M': self.base_shipbasic.C_M,
            'DWT': self.base_shipbasic.DWT,
            'gt': getattr(self.base_shipbasic, 'gt', None),
        })

        # If we're fixing derived parameters, include them explicitly
        if fix_derived_params:
            params.update({
                '_L_PP': getattr(self.base_shipbasic, '_L_PP', None) or self.base_shipbasic.L_PP,
                '_B': getattr(self.base_shipbasic, '_B', None) or self.base_shipbasic.B,
                '_T': getattr(self.base_shipbasic, '_T', None) or self.base_shipbasic.T,
            })

        # Override params with the provided changes
        for param, value in param_changes.items():
            if param == 'L_PP':
                params['_L_PP'] = value
            elif param == 'B':
                params['_B'] = value
            elif param == 'T':
                params['_T'] = value
            else:
                params[param] = value

        self.logger.info(
            f"Creating modified ShipShapeBasic with changes: {param_changes}, "
            f"fix_derived_params={fix_derived_params}"
        )

        # Create new ShipShapeBasic with modified parameters
        return ShipShapeBasic(**params)

    def create_explorer(self,
                        modified_shipbasic: ShipShapeBasic,
                        extra_params: Optional[Dict[str, Any]] = None) -> ShipShapeExplorer:
        """
        Create a ShipShapeExplorer instance using the modified ShipShapeBasic.

        Args:
            modified_shipbasic: The modified ship basic parameters
            extra_params: Additional parameters to pass to the explorer

        Returns:
            Configured ShipShapeExplorer
        """
        # Set up parameters for the explorer
        params = {
            'shipshape': modified_shipbasic,
            'v_knots': self.fixed_speed,
            'correction_library': self.correction_library,
            'verbose': self.verbose,
            'debug': self.debug
        }

        # Add any additional parameters
        if extra_params:
            params.update(extra_params)

        # Create explorer
        explorer = self.explorer_class(**params)

        # Apply any custom parameter derivations
        for param_name, derive_func in self.parameter_derivations.items():
            if not hasattr(explorer, param_name):
                self.logger.warning(f"Unknown parameter {param_name}")
                continue

            try:
                derived_value = derive_func(explorer)
                setattr(explorer, param_name, derived_value)
                self.logger.debug(f"Applied derived value for {param_name}: {derived_value}")
            except Exception as e:
                self.logger.error(f"Failed to derive {param_name}: {str(e)}")

        return explorer

    def create_resistance_calculator(self,
                                     explorer: ShipShapeExplorer,
                                     wind_speed: Optional[float] = None,
                                     wind_direction: Optional[float] = None) -> ResistanceCalculator:
        """
        Create and configure a ResistanceCalculator for the given explorer.

        Args:
            explorer: ShipShapeExplorer instance
            wind_speed: Optional wind speed for wind resistance calculations
            wind_direction: Optional wind direction for wind resistance calculations

        Returns:
            Configured ResistanceCalculator
        """
        # Initialize the basic calculator
        calculator = ResistanceCalculator(shipshape=explorer)

        # Add each resistance model with appropriate parameters
        for config in self.resistance_configs:
            if config.model_type == ResistanceModelType.CALM_WATER and config.model_name == ResistanceModelName.HOLTROP:
                # For Holtrop-Mennen, we create the model with both ship shape and propeller
                model = self.factory.create_model(
                    config.model_type.value,
                    config.model_name.value,
                    explorer,
                    self.ship_propeller
                )

            elif config.model_type == ResistanceModelType.WIND and config.model_name == ResistanceModelName.ANDERSEN:
                # For wind resistance, first create the basic model
                model = self.factory.create_model(
                    config.model_type.value,
                    config.model_name.value,
                    explorer
                )

                # Then configure wind-specific parameters if provided
                if isinstance(model, AndersenWindResistance):
                    if wind_speed is not None:
                        model.wind_speed = wind_speed
                    if wind_direction is not None:
                        model.wind_direction = wind_direction

            else:
                factory_params = config.get_factory_params()
                model = self.factory.create_model(
                    factory_params['model_type'],
                    factory_params['model_name'],
                    explorer
                )

            calculator.add_model(model)

        return calculator

    def create_parameter_variations(self,
                                    parameter_name: str,
                                    values: List[float],
                                    fix_derived_params: bool = True,
                                    extra_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create a DataFrame with resistance calculations for variations of a single parameter.

        Args:
            parameter_name: The name of the parameter to vary (e.g. 'L', 'B', 'T', 'C_M')
            values: List of values to use for the parameter
            fix_derived_params: If True, keep original values for other derived parameters
                                If False, allow parameters to be re-derived from regression models
            extra_params: Additional parameters to set on the explorer

        Returns:
            DataFrame with parameter values and corresponding resistance calculations
        """
        results = []

        for value in values:
            # Create the parameter change dictionary
            param_changes = {parameter_name: value}

            try:
                # Create modified ship and calculate resistance
                modified_shipbasic = self.create_modified_shipbasic(
                    fix_derived_params=fix_derived_params,
                    **param_changes
                )
                explorer = self.create_explorer(modified_shipbasic, extra_params)
                calculator = self.create_resistance_calculator(explorer)

                # Calculate total resistance and breakdown
                total_resistance = calculator.total_resistance()
                resistance_breakdown = calculator.get_resistance_breakdown()

                # Create result row
                result = {
                    parameter_name: value,
                    'total_resistance': total_resistance
                }

                # Add breakdown components
                for component, component_value in resistance_breakdown.items():
                    result[f"resistance_{component}"] = component_value

                # Add calculated ship parameters that might be useful for analysis
                result.update({
                    'L': explorer.L,
                    'L_PP': explorer.L_PP,
                    'B': explorer.B,
                    'T': explorer.T,
                    'C_B': explorer.C_B,
                    'C_P': explorer.C_P,
                    'C_M': explorer.C_M,
                    'S': explorer.S,  # Wetted surface
                    'DWT_v': explorer.DWT_v,  # Displacement volume
                    'speed': self.fixed_speed
                })

                # Add information about whether parameters were fixed or derived
                result['derived_params_fixed'] = fix_derived_params

                results.append(result)
                self.logger.info(f"Calculated resistance for {parameter_name}={value}: {total_resistance:.2f} kN")

            except Exception as e:
                self.logger.error(f"Error calculating resistance for {parameter_name}={value}: {str(e)}")
                # Still add the parameter value to results, but with NaN for calculations
                results.append({parameter_name: value, 'total_resistance': np.nan})

        # Convert results to DataFrame and sort by parameter value
        df = pd.DataFrame(results)
        return df.sort_values(by=parameter_name)

    def create_multi_parameter_variations(self,
                                          parameter_variations: Dict[str, List[float]],
                                          extra_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create a DataFrame with resistance calculations for combinations of multiple parameters.

        Args:
            parameter_variations: Dictionary mapping parameter names to lists of values
            extra_params: Additional parameters to set on the explorer

        Returns:
            DataFrame with parameter value combinations and corresponding resistance calculations
        """
        from itertools import product

        # Create all combinations of parameter values
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            # Create the parameter change dictionary for this combination
            param_changes = {param_names[i]: combo[i] for i in range(len(param_names))}

            try:
                # Create modified ship and calculate resistance
                modified_shipbasic = self.create_modified_shipbasic(**param_changes)
                explorer = self.create_explorer(modified_shipbasic, extra_params)
                calculator = self.create_resistance_calculator(explorer)

                # Calculate total resistance and breakdown
                total_resistance = calculator.total_resistance()
                resistance_breakdown = calculator.get_resistance_breakdown()

                # Create result row with all parameter values
                result = {**param_changes, 'total_resistance': total_resistance}

                # Add breakdown components
                for component, component_value in resistance_breakdown.items():
                    result[f"resistance_{component}"] = component_value

                # Add calculated ship parameters
                result.update({
                    'L': explorer.L,
                    'L_PP': explorer.L_PP,
                    'B': explorer.B,
                    'T': explorer.T,
                    'C_B': explorer.C_B,
                    'C_P': explorer.C_P,
                    'C_M': explorer.C_M,
                    'S': explorer.S,
                    'DWT_v': explorer.DWT_v,
                    'speed': self.fixed_speed
                })

                results.append(result)
                self.logger.info(f"Calculated resistance for parameters {param_changes}: {total_resistance:.2f} kN")

            except Exception as e:
                self.logger.error(f"Error calculating resistance for parameters {param_changes}: {str(e)}")
                # Add the parameter values to results with NaN for calculations
                results.append({**param_changes, 'total_resistance': np.nan})

        # Convert results to DataFrame
        return pd.DataFrame(results)

