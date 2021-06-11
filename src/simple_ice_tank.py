from math import pi, exp
from typing import Optional, Union

from CoolProp.CoolProp import PropsSI


def smoothing_function(x: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """
    Uses a sigmoid function to smooth based on the argument bounds

    https://en.wikipedia.org/wiki/Sigmoid_function

    :param x: independent variable
    :param x_min: minimum value of x
    :param x_max: maximum value of x
    :param y_min: minimum value of y
    :param y_max: maximum value of y
    :return: smoothed float between y_min and y_max
    """

    if x < x_min:
        return y_min
    elif x > x_max:
        return y_max
    else:
        x_normalized = (x - x_min) / (x_max - x_min)

    # scales x = [0, 1] to [-6, 6]
    x_sig = 16.0 * x_normalized + -8.0

    # compute sigmoid
    y_sig = 1 / (1 + exp(-x_sig))

    return (y_max - y_min) * y_sig + y_min


def c_to_k(temp):
    """
    Convert degrees Celsius to Kelvin

    :param temp: degrees Celsius
    :return: Kelvin
    """

    return temp + 273.15


def density(fluid_str: str, temp: float):
    """
    Gets fluid density using CoolProp

    Assumes a pressure of 140 kPa

    :param fluid_str: fluid input string, as defined by CoolProp
    :param temp: temperature, C
    :return: density, kg/m3
    """

    return PropsSI("D", "T", c_to_k(temp), "P", 140000, fluid_str)


def specific_heat(fluid_str: str, temp: float):
    """
    Gets fluid specific heat using CoolProp

    Assumes a pressure of 140 kPa

    :param fluid_str: fluid input string, as defined by CoolProp
    :param temp: temperature, C
    :return: specific heat, J/kg-K
    """

    return PropsSI("C", "T", c_to_k(temp), "P", 140000, fluid_str)


class IceTank(object):

    def __init__(self, data: dict):
        # fluid strings
        self.fluid_str = "WATER"  # Water
        self.brine_str = "INCOMP::MPG[0.3]"  # Propylene Glycol - 30% by mass

        # geometry
        self.diameter = float(data["tank_diameter"])  # m
        self.height = float(data["tank_height"])  # m
        self.fluid_volume = float(data["fluid_volume"])  # m3
        self.area_lid = pi / 4 * (self.diameter ** 2)  # m2
        self.area_base = self.area_lid  # m2
        self.area_wall = pi * self.diameter * self.height  # m2
        self.area_total = self.area_lid + self.area_base + self.area_wall  # m2

        # thermodynamics and heat transfer
        self.total_fluid_mass = self.fluid_volume * density(self.fluid_str, 20)  # kg
        self.ice_mass = None
        self.ice_mass_prev = None
        self.tank_temp = None
        self.tank_temp_prev = None
        self.outlet_fluid_temp = None
        self.r_value_lid = float(data["r_value_lid"])  # m2-K/W
        self.r_value_base = float(data["r_value_base"])  # m2-K/W
        self.r_value_wall = float(data["r_value_wall"])  # m2-K/W

        # TODO: convection coefficients should be different based on surface orientation
        conv_coeff_inside_tank_wall = 1000  # W/m2-K
        conv_coeff_outside_tank_wall = 100  # W/m2-K
        self.resist_inside_tank_wall = 1 / (conv_coeff_inside_tank_wall * self.area_total)  # K/W
        self.resist_outside_tank_wall = 1 / (conv_coeff_outside_tank_wall * self.area_total)  # K/W
        resist_lid = self.r_value_lid / self.area_lid  # K/W
        resist_base = self.r_value_base / self.area_base  # K/W
        resist_wall = self.r_value_wall / self.area_wall  # K/W

        # TODO: wall R-value should be considered as a radial resistance
        self.resist_conduction_tank_wall = 1 / ((1 / resist_lid) + (1 / resist_base) + (1 / resist_wall))  # K/W
        self.tank_ua_env = 1 / self.resist_conduction_tank_wall  # W/K

        # time tracking
        self.time = 0

        # set initial conditions if passed to the ctor
        if "initial_temperature" in data:
            self.init_state(tank_init_temp=float(data["initial_temperature"]))
        elif "latent_state_of_charge" in data:
            self.init_state(latent_state_of_charge=float(data["latent_state_of_charge"]))

        # other
        self.tank_is_charging = None

    def init_state(self,
                   latent_state_of_charge: Optional[Union[int, float]] = None,
                   tank_init_temp: Optional[Union[int, float]] = None):
        """
        Initialize the state of the tank

        :param latent_state_of_charge: latent state of charge for tank. Non-dimensional. Any real value 0-1.
        Cannot be used in conjunction with 'tank_init_temp'.

        :param tank_init_temp: bulk tank temperature. Degrees Celsius. Used to set the tank temperature.
        Temperatures >= 0 deg C result in a state of charge (SOC) = 0. Temperatures < 0 deg C result in a SOC = 1.
        Cannot be used in conjunction with 'latent_state_of_charge'.
        """

        # validate inputs
        if latent_state_of_charge is not None and tank_init_temp is not None:
            msg = "Can not set both 'latent_state_of_charge' and 'tank_init_temp'. Only one may be used."
            raise IOError(msg)
        elif latent_state_of_charge is None and tank_init_temp is None:
            msg = "Must set either 'latent_state_of_charge' or 'tank_init_temp'."
            raise IOError(msg)

        # reset times
        self.time = 0

        # set state based on latent state of charge
        if latent_state_of_charge is not None:
            # set tank temperature
            self.tank_temp = 0
            self.tank_temp_prev = self.tank_temp

            # init outlet fluid temp
            self.outlet_fluid_temp = 0

            # bound latent charge state
            latent_state_of_charge = float(max(0, min(1, latent_state_of_charge)))

            # set state of charge
            self.ice_mass = latent_state_of_charge * self.total_fluid_mass
            self.ice_mass_prev = self.ice_mass

            # we're done
            return

        # set state if based on tank temperature
        if tank_init_temp is not None:

            # set tank temp
            self.tank_temp = tank_init_temp
            self.tank_temp_prev = self.tank_temp

            # init outlet fluid temp
            self.outlet_fluid_temp = tank_init_temp

            # set state of charge based on temperature
            if tank_init_temp >= 0:
                self.ice_mass = 0
                self.ice_mass_prev = self.ice_mass
            else:
                self.ice_mass = self.total_fluid_mass
                self.ice_mass_prev = self.ice_mass

            # we're done
            return

        # we should never make it here
        assert False  # pragma: no cover

    @property
    def liquid_mass(self):
        """
        Convenient property to associate total mass and ice mass

        :return: liquid mass, kg
        """
        return self.total_fluid_mass - self.ice_mass

    @property
    def state_of_charge(self):
        """
        Computes the state of charge, base on the ratio of ice mass to total mass
        """
        return min(max(1 - self.liquid_mass / self.total_fluid_mass, 0), 1)

    def effectiveness_soc_degradation_discharging(self):
        """
        Effectiveness modifier due to state of charge for discharging mode
        """

        soc = self.state_of_charge

        # piecewise degradation function for state of charge
        piecewise_cutoff = 0.25

        # linear portion
        max_degradation_linear = 1.0
        min_degradation_linear = 0.75
        degradation_slope = (max_degradation_linear - min_degradation_linear) / (1 - piecewise_cutoff)
        degradation_intercept = max_degradation_linear - degradation_slope

        # sigmoid portion
        max_degradation_sig = min_degradation_linear
        min_degradation_sig = 0.2

        if soc > piecewise_cutoff:
            return degradation_slope * soc + degradation_intercept
        else:
            soc_degradation = smoothing_function(soc,
                                                 0,
                                                 piecewise_cutoff,
                                                 min_degradation_sig,
                                                 max_degradation_sig)
            return min(max(soc_degradation, min_degradation_sig), max_degradation_sig)

    def effectiveness_soc_degradation_charging(self):
        """
        Effectiveness modifier due to state of charge for discharging mode
        """

        soc = self.state_of_charge

        # linear data
        max_degradation_linear = 0.9
        min_degradation_linear = 1.2
        degradation_slope = (max_degradation_linear - min_degradation_linear)  # / 1
        degradation_intercept = max_degradation_linear - degradation_slope

        return degradation_slope * soc + degradation_intercept

    def effectiveness(self, temperature: float, mass_flow_rate: float):
        """
        Simple correlation for mass flow rate to effectiveness

        :param temperature: temperature, C
        :param mass_flow_rate: mass flow rate, kg/s
        :return: effectiveness, non-dimensional
        """

        # check for flow before proceeding
        if mass_flow_rate <= 0.0:
            return 1

        # set effectiveness due to mass flow effects
        tank_ua_hx = 20000
        num_transfer_units = tank_ua_hx / (mass_flow_rate * specific_heat(self.brine_str, temperature))
        effectiveness = 1 - exp(-num_transfer_units)

        # set effectiveness due to state of charge effects
        # no change in effectiveness due to state of charge when charging
        if self.tank_is_charging:
            return effectiveness * self.effectiveness_soc_degradation_charging()
        else:
            return effectiveness * self.effectiveness_soc_degradation_discharging()

    def q_brine_max(self, inlet_temp: float, mass_flow_rate: float, timestep: float):
        """
        Maximum possible brine heat transfer exchange
        Sign convention - positive for heat transfer into tank

        :param inlet_temp: inlet brine temperature, C
        :param mass_flow_rate: brine mass flow rate, kg/s
        :param timestep: simulation timestep, sec
        :return: max possible heat transfer exchanged with tank, Joules
        """

        ave_temp = (self.tank_temp + inlet_temp) / 2.0
        cp = specific_heat(self.brine_str, ave_temp)
        return mass_flow_rate * cp * (inlet_temp - self.tank_temp) * timestep

    def q_brine(self, inlet_temp: float, mass_flow_rate: float, timestep: float):
        """
        Brine heat transfer exchange with tank
        Sign convention - positive heat transfer to tank

        Assumes a fixed effectiveness

        :param inlet_temp: inlet brine temperature, C
        :param mass_flow_rate: brine mass flow rate, kg/s
        :param timestep: simulation timestep, sec
        :return: heat transfer exchanged with tank, Joules
        """

        # check for flow before proceeding
        if mass_flow_rate <= 0.0:
            return 0.0

        q_max = self.q_brine_max(inlet_temp, mass_flow_rate, timestep)
        return self.effectiveness(inlet_temp, mass_flow_rate) * q_max

    def q_env(self, env_temp: float, timestep: float):
        """
        Heat transfer exchange between environment
        Sign convention - positive for heat transfer into tank

        :param env_temp: environment temperature, C
        :param timestep: simulation timestep, sec
        :return: heat transfer exchanged with tank, Joules
        """

        return self.tank_ua_env * (env_temp - self.tank_temp) * timestep

    def compute_state(self, dq: float):
        """
        Computes the charge state of the tank
        Sign convention - positive heat transfer into tank

        :param dq: heat transfer change, Joules
        :return: None
        """

        if dq < 0:
            self.compute_charging(dq)
            self.tank_is_charging = True
        else:
            self.compute_discharging(dq)
            self.tank_is_charging = False

    def compute_charging(self, dq: float):
        """
        Computes the charging mode state of the tank
        Sign convention - positive heat transfer into tank

        :param dq: heat transfer change, Joules
        :return: None
        """

        # for convenience, dq converted to a positive number
        dq = abs(float(dq))

        # piece-wise computation of charging state

        # sensible fluid charging
        if self.tank_temp > 0:
            # compute liquid sensible capacity available
            cp_sens_liq = specific_heat(self.fluid_str, self.tank_temp)
            q_sens_avail = self.liquid_mass * cp_sens_liq * self.tank_temp

            # can the load be fully met with sensible-only charging?
            # yes, sensible-only charging can meet remaining load
            if q_sens_avail > dq:
                new_tank_temp = -dq / (self.liquid_mass * cp_sens_liq) + self.tank_temp

                # if we've made it this far, we should be OK to return
                self.tank_temp = new_tank_temp
                return

            # no, we have sensible for a portion, and then some ice charging
            else:
                # need to decrement the dq so we know how much remains in the next section
                # don't return early, we need to fallthrough to compute latent charging
                dq -= q_sens_avail
                self.tank_temp = 0

        # latent ice charging
        if dq > 0 and self.ice_mass < self.total_fluid_mass:
            # latent heat of fusion, water
            # TODO: support something besides water in the tank
            h_if = 334000  # J/kg

            # compute latent charging capacity available
            q_lat_avail = h_if * self.liquid_mass

            # can we meet the remaining load with latent-only charging?
            # yes, latent-only charging can meet remaining load
            if q_lat_avail > dq:
                delta_ice_mass = dq / h_if
                self.ice_mass += delta_ice_mass

                # if we've made it this far, we should be OK to return
                self.tank_temp = 0
                return

            # no, we have a latent portion then have to meet the load with some sensible charging, i.e. ice temp < 0
            else:
                self.ice_mass = self.total_fluid_mass
                dq -= q_lat_avail

        # sensible subcooled ice charging
        if dq > 0:
            cp_ice = 2030  # J/kg-K
            self.tank_temp += -dq / (self.total_fluid_mass * cp_ice)

    def compute_discharging(self, dq: float):
        """
        Computes the discharging mode state of the tank
        Sign convention - positive heat transfer into tank

        :param dq: heat transfer change, Joules
        :return: None
        """

        # piece-wise computation of discharging state
        # discharging has to occur in the reverse direction from charging

        # sensible ice discharging
        if self.tank_temp < 0:
            # compute solid ice sensible capacity available
            # TODO: support other fluids
            cp_sens = 2030
            q_sens_avail = abs(self.ice_mass * cp_sens * self.tank_temp)

            # can the load be fully met with sensible-only discharging?
            # yes, sensible-only discharging can meet remaining load
            if q_sens_avail > dq:
                self.tank_temp += dq / (self.ice_mass * cp_sens)

                # if we've made it this far, we should be OK to return
                return

            # no, we have sensible for a portion, and then some ice melting
            else:
                # need to decrement the dq so we know how much remains in the next section
                # don't return early, we need to fallthrough to compute latent discharging
                dq -= q_sens_avail
                self.tank_temp = 0

        # latent ice discharging
        if dq > 0 and self.ice_mass > 0:
            # latent heat of fusion, water
            # TODO: support something besides water in the tank
            h_if = 334000  # J/kg

            # compute latent charging capacity available
            q_lat_avail = h_if * self.ice_mass

            # can we meet the remaining load with latent-only discharging?
            # yes, latent-only discharging can meet remaining load
            if q_lat_avail > dq:
                delta_ice_mass = dq / h_if
                self.ice_mass -= delta_ice_mass

                # if we've made it this far, we should be OK to return
                self.tank_temp = 0
                return

            # no, we have a latent portion then have to meet the load with some sensible discharging
            else:
                self.ice_mass = 0
                dq -= q_lat_avail

        # sensible liquid discharging
        if self.ice_mass == 0:
            cp_sens_liq = specific_heat(self.fluid_str, self.tank_temp)
            self.tank_temp += dq / (self.total_fluid_mass * cp_sens_liq)

    def calculate_outlet_fluid_temp(self, inlet_temp: float, mass_flow_rate: float):
        """
        Computes the outlet fluid temperature based on the current state of the tank

        :param inlet_temp: inlet temperature of the brine fluid, degrees Celsius
        :param mass_flow_rate: mass flow rate of the brine fluid, kg/s
        """

        # check for flow before proceeding
        if mass_flow_rate <= 0.0:
            return inlet_temp

        return inlet_temp - self.effectiveness(inlet_temp, mass_flow_rate) * (inlet_temp - self.tank_temp)

    def calculate(self, inlet_temp: float, mass_flow_rate: float, env_temp: float, sim_time: float, timestep: float):
        """
        Calculates the tank energy balance

        :param inlet_temp: inlet temperature of the brine fluid, degrees Celsius
        :param mass_flow_rate: mass flow rate of the brine fluid, kg/s
        :param env_temp: environment temperature, degrees Celsius
        :param sim_time: total elapsed simulation time, sec
        :param timestep: timestep, sec
        """
        # General governing equation
        # M_fluid du/dt = sum(q_env, q_brine)

        # tank_temp and ice_mass are our state variables
        # if time has advanced, we need to lock down the 'previous' state from the last iteration
        # otherwise, we will set them to to previous state and then update assuming we're in an iteration

        if sim_time > self.time:
            self.time = sim_time
            self.tank_temp_prev = self.tank_temp
            self.ice_mass_prev = self.ice_mass
        else:
            self.tank_temp = self.tank_temp_prev
            self.ice_mass = self.ice_mass_prev

        # right-hand side
        # sum(q_env * dt, q_brine * dt)
        q_brine = self.q_brine(inlet_temp, mass_flow_rate, timestep)
        q_env = self.q_env(env_temp, timestep)
        q_tot = q_brine + q_env

        # left-hand side
        # M_fluid du = M_fluid * (u_fluid - u_fluid_old)
        self.compute_state(q_tot)

        # compute new outlet fluid temp
        self.outlet_fluid_temp = self.calculate_outlet_fluid_temp(inlet_temp, mass_flow_rate)
