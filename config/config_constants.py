"""

This module stores constants and environmental conditions that are common to all classes
in the DT4GS digital twin driven design tool.

This package contains tools for:
- Storing constants (Constant class)
- Storing environmental conditions (Environment class)`

"""

GRAVITY: float = 9.81  # Acceleration due to gravity in m/s^2
SEAWATER_DENSITY: float = 1025 #Density of seawater in kg/m^3
RHO_AIR: float = 1.23 # density air [kg/m**3]
NU:float = 1.13902e-6 #kinematic viscosity of water at 20 degrees C
P0PV: float = 99047 #p_o-p_v, for sea water of 15°C in N/m^2

class SeaConditions:
    """
    A class that represents sea conditions relevant to a ship's sailing environment.

    :param Va: Relative wind speed in meters per second (m/s). Defaults to 0.
    :type Va: float
    :param Va_dir_rel: Relative wind direction in degrees. Defaults to 0.
    :type Va_dir_rel: float
    :param t_sw: Sea water temperature in degrees Celsius, ranging between 1°C and 30°C.
                 Defaults to None, which skips the computation of dynamic viscosity.
    :type t_sw: float, optional

    :raises ValueError: If the sea water temperature is not between 1°C and 30°C.

    :ivar Va: Relative wind speed in meters per second (m/s).
    :ivar Va_dir_rel: Relative wind direction in degrees.
    :ivar nu: Dynamic viscosity of sea water, computed based on the sea temperature.
    """
    def __init__(self,
                 Va: float =0, #Relative wind speed in m/s
                 Va_dir_rel: float = 0, #The relative wind direction in degrees
                 t_sw: float = None, #The temperature of the sea water
                 ):

        self.Va = Va
        self.Va_dir_rel = Va_dir_rel

        if t_sw is not None:
            temps = list(range(1, 31))
            nus_sw = [
                1.7926E-06, 1.7341E-06, 1.6787E-06, 1.6262E-06, 1.5762E-06,
                1.5288E-06, 1.4836E-06, 1.4406E-06, 1.3995E-06, 1.3604E-06,
                1.3230E-06, 1.2873E-06, 1.2532E-06, 1.2205E-06, 1.1892E-06,
                1.1592E-06, 1.1304E-06, 1.1028E-06, 1.0763E-06, 1.0508E-06,
                1.0263E-06, 1.0027E-06, 9.8002E-07, 9.5818E-07, 9.3713E-07,
                9.1683E-07, 8.9726E-07, 8.7837E-07, 8.6014E-07, 8.4253E-07
            ]  # https://www.ittc.info/media/7989/75-02-01-03.pdf

            if not (1 <= t_sw <= 31):
                raise ValueError(f"Seawater temperature must be between 1°C and 30°C, got {t_sw}°C.")

            self.nu = np.interp(t_sw, temps, nus_sw)
        self.nu = NU