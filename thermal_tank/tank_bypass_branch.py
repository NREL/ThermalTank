from enum import IntEnum

import numpy as np
from scipy.optimize import minimize

from thermal_tank.simple_ice_tank import IceTank


class OpMode(IntEnum):
    CHARGING = 1
    DISCHARGING = -1
    FLOAT = 0


class TankBypassBranch(object):

    def __init__(self, num_tanks, tank_data=None):

        if tank_data is None:
            tank_data = {
                "tank_diameter": 89 * 0.0254,
                "tank_height": 101 * 0.0254,
                "fluid_volume": 1655 * 0.00378541,
                "r_value_lid": 24 / 5.67826,
                "r_value_base": 9 / 5.67826,
                "r_value_wall": 9 / 5.67826,
                "initial_temperature": -5,
            }

        self.tank = IceTank(tank_data)
        self.num_tanks = num_tanks
        self.outlet_temp = 0
        self.bypass_fraction = 0
        self.tank_mass_flow = 0

    def simulate(self,
                 inlet_temp: float,
                 mass_flow_rate: float,
                 env_temp: float,
                 branch_set_point: float,
                 op_mode: int,
                 sim_time: float,
                 timestep: float,
                 bypass_frac: float = None):

        if op_mode == 0:
            op_mode = OpMode.FLOAT
        elif op_mode == 1:
            op_mode = OpMode.CHARGING
        elif op_mode == -1:
            op_mode = OpMode.DISCHARGING

        # bypass condition
        if op_mode == OpMode.FLOAT:
            self.outlet_temp = inlet_temp
            self.tank.calculate(inlet_temp, 0, env_temp, sim_time, timestep)
            self.bypass_fraction = 1
            return

        # charging condition
        # all flow through tank
        if op_mode == OpMode.CHARGING:
            m_dot_per_tank_max = mass_flow_rate / self.num_tanks
            self.tank.calculate(inlet_temp, m_dot_per_tank_max, env_temp, sim_time, timestep)
            self.outlet_temp = self.tank.outlet_fluid_temp
            self.bypass_fraction = 0
            return

        # discharging condition
        # target tank/bypass flow rate split to target set point
        if op_mode == OpMode.DISCHARGING:

            # outlet temp at no flow through tank is inlet temp
            t_out_low = inlet_temp

            # there's no need to discharge, inlet temp is already lower than branch setpoint
            if t_out_low < branch_set_point:
                self.tank.calculate(inlet_temp, 0, env_temp, sim_time, timestep)
                self.outlet_temp = inlet_temp
                self.bypass_fraction = 1
                return

            # if we've made it here, we need to discharge
            # find outlet temp at flow rate through tank
            m_dot_per_tank = mass_flow_rate / self.num_tanks
            self.tank.calculate(inlet_temp, m_dot_per_tank, env_temp, sim_time, timestep)
            t_out_high = self.tank.outlet_fluid_temp

            # tank can't meet demand, all flow through tank
            if t_out_high > branch_set_point:
                self.outlet_temp = t_out_high
                self.bypass_fraction = 0
                return

        # finally, if we've made it here, we need to split flow between the tank and the bypass
        init_guess = bypass_frac if bypass_frac else self.bypass_fraction
        x0 = np.array([init_guess])
        args = (inlet_temp, mass_flow_rate, env_temp, branch_set_point, sim_time, timestep)
        bounds = ((0.0, 1.0),)
        tol = 0.01
        ret = minimize(self.obj_f, x0=x0, args=args, bounds=bounds, tol=tol)
        self.bypass_fraction = ret.x[0]
        self.outlet_temp = self.branch_outlet_temp(self.bypass_fraction, inlet_temp, mass_flow_rate, env_temp,
                                                   sim_time,
                                                   timestep)
        self.tank_mass_flow = mass_flow_rate * (1 - self.bypass_fraction)
        return

    def branch_outlet_temp(self, bypass_frac, inlet_temp, mass_flow_rate, env_temp, sim_time, timestep):
        m_dot_bypass = bypass_frac * mass_flow_rate
        m_dot_per_tank = (mass_flow_rate - m_dot_bypass) / self.num_tanks
        self.tank.calculate(inlet_temp, m_dot_per_tank, env_temp, sim_time, timestep)
        t_out_tank = self.tank.outlet_fluid_temp
        return ((m_dot_per_tank * t_out_tank * self.num_tanks) + (m_dot_bypass * inlet_temp)) / mass_flow_rate

    def obj_f(self, bypass_frac, inlet_temp, mass_flow_rate, env_temp, branch_set_point, sim_time, timestep):
        t_out_branch = self.branch_outlet_temp(bypass_frac, inlet_temp, mass_flow_rate, env_temp, sim_time, timestep)
        return abs(t_out_branch - branch_set_point)
