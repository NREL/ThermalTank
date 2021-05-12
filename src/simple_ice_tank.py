from math import pi

from CoolProp.CoolProp import PropsSI


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

        # initial conditions
        self.tank_temp = float(data["initial_temperature"])

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
        if self.tank_temp > 0:
            self.ice_mass = 0.0  # kg
        else:
            self.ice_mass = self.total_fluid_mass
        self.r_value_lid = float(data["r_value_lid"])  # m2-K/W
        self.r_value_base = float(data["r_value_base"])  # m2-K/W
        self.r_value_wall = float(data["r_value_wall"])  # m2-K/W
        self.outlet_fluid_temp = self.tank_temp

        # TODO: convection coefficients should be different based on surface orientation
        h_i = 1000  # W/m2-K
        h_o = 100  # W/m2-K
        self.resist_inside = 1 / (h_i * self.area_total)  # K/W
        self.resist_outside = 1 / (h_o * self.area_total)  # K/W
        resist_lid = self.r_value_lid / self.area_lid  # K/W
        resist_base = self.r_value_base / self.area_base  # K/W
        resist_wall = self.r_value_wall / self.area_wall  # K/W

        # TODO: wall R-value should be considered as a radial resistance
        self.resist_conduction = 1 / ((1 / resist_lid) + (1 / resist_base) + (1 / resist_wall))  # K/W
        self.overall_ua = 1 / self.resist_conduction  # W/K

    @property
    def liquid_mass(self):
        """
        Convenient property to associate total mass and ice mass

        :return: liquid mass, kg
        """
        return self.total_fluid_mass - self.ice_mass

    @staticmethod
    def effectiveness(mass_flow_rate):
        """
        Simple correlation for mass flow rate to effectiveness

        :param mass_flow_rate: mass flow rate, kg/s
        :return: effectiveness, non-dimensional
        """

        min_effectiveness = 0.3
        max_effectiveness = 0.95

        return max(min_effectiveness, min(max_effectiveness, -0.125 * mass_flow_rate + 1.225))

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

        if mass_flow_rate > 0:
            q_max = self.q_brine_max(inlet_temp, mass_flow_rate, timestep)
            return self.effectiveness(mass_flow_rate) * q_max
        else:
            return 0.0

    def q_env(self, env_temp: float, timestep: float):
        """
        Heat transfer exchange between environment
        Sign convention - positive for heat transfer into tank

        # TODO: radiative effects are not considered, but are probably negligible

        :param env_temp: environment temperature, C
        :param timestep: simulation timestep, sec
        :return: heat transfer exchanged with tank, Joules
        """

        return self.overall_ua * (env_temp - self.tank_temp) * timestep

    def compute_state(self, dq: float):
        """
        Computes the charge state of the tank
        Sign convention - positive heat transfer into tank

        :param dq: heat transfer change, Joules
        :return: None
        """

        if dq < 0:
            self.compute_charging(dq)
        else:
            self.compute_discharging(dq)

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
            # TODO: support something besides water
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

    def calculate_outlet_fluid_temp(self, inlet_temp, mass_flow_rate):
        # TODO: fix this
        # this is obviously really dumb right now, but if timestep's are small, we maybe can get away with this
        return inlet_temp - self.effectiveness(mass_flow_rate) * (inlet_temp - self.tank_temp)

    def calculate(self, inlet_temp: float, m_dot: float, env_temp: float, timestep: float):
        # General governing equation
        # M_fluid du/dt = sum(q_env, q_brine)

        # right-hand side
        # sum(q_env * dt, q_brine * dt)
        q_brine = self.q_brine(inlet_temp, m_dot, timestep)
        q_env = self.q_env(env_temp, timestep)
        q_tot = q_brine + q_env

        # update outlet temp before updating tank temp
        self.outlet_fluid_temp = self.calculate_outlet_fluid_temp(inlet_temp, m_dot)

        # left-hand side
        # M_fluid du = M_fluid * (u_fluid - u_fluid_old)
        self.compute_state(q_tot)
