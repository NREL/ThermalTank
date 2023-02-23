import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pyomo.environ import ConcreteModel, Var, Param, Constraint, Objective, SolverFactory, Block, \
    Binary, NonNegativeReals, NonNegativeIntegers, Reals, Expression, exp, value, RangeSet, Piecewise, summation, units
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
from scipy.stats import linregress
from thermal_tank.simple_ice_tank import IceTank, specific_heat
import idaes.logger as idaeslog
import pandas as pd
from pathlib import Path

from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds

thermal_tank_log_dir = Path(__file__).parent.parent.absolute() / "controls" / "schedules_mdoff" / "coned"
num_tanks = 1
test_mass_flow_rate = 4.4963
brine_cp = 3812.428     # at 3.605 C, midway between max_tank_temp and brine_freezing_pt
# brine_cp = 3809.47
water_cp = 4207
# water_cp = 4195
ice_cp = 2030
h_if = 334000
tank_data = {
    "tank_diameter": 89 * 0.0254,
    "tank_height": 101 * 0.0254,
    "fluid_volume": 1655 * 0.00378541,
    "r_value_lid": 24 / 5.67826,
    "r_value_base": 9 / 5.67826,
    "r_value_wall": 9 / 5.67826,
    "initial_temperature": 15,
    "max_mass_flow_rate": test_mass_flow_rate
}
max_chiller_load = 500

def brine_cp_linear_fxn():
    tank = IceTank(tank_data)
    brine_str = tank.brine_str
    temps = np.linspace(brine_freezing_pt, max_tank_temp)
    cps = specific_heat(brine_str, temps - 273.15)
    slope, intercept, r, p, se = linregress(temps, cps)
    return slope, intercept, temps, cps


def fluid_cp_linear_fxn():
    tank = IceTank(tank_data)
    fluid_str = tank.fluid_str
    temps = np.linspace(273.15, max_tank_temp)
    cps = specific_heat(fluid_str, temps - 273.15)
    slope, intercept, r, p, se = linregress(temps, cps)
    plt.plot(temps, cps, label="water")
    plt.plot(temps, slope * temps + intercept, '--', label="pred")
    plt.legend()
    plt.show()
    return slope, intercept, temps, cps


brine_freezing_pt = 260.360896
min_tank_temp = -10 + 273.15
max_tank_temp = 20 + 273.15
phase_change_e = 1e-2
Q_scaling_factor = 1e-6

tank_phase_change_k = [min_tank_temp, 273.15 - phase_change_e, 273.15 + phase_change_e, max_tank_temp]
tank_phase_change_Q = [
    0, 
    ice_cp * (-phase_change_e - (min_tank_temp - 273.15)), 
    ice_cp * (-phase_change_e - (min_tank_temp - 273.15)) + h_if,
    ice_cp * (-phase_change_e - (min_tank_temp - 273.15)) + h_if + water_cp * (max_tank_temp - 273.15 - phase_change_e)
    ]
tank_phase_change_Q

def temp_model(m, pw_repn='incremental'):
    m.epsilon = Param(domain=NonNegativeReals, mutable=False, default=phase_change_e)
    m.avail_Q = Var(m.horiz_t_init, domain=NonNegativeReals)
    if pw_repn == "incremental":
        m.tank_temp_d1 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, -m.epsilon + 10))
        m.tank_temp_d2 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, 2 * m.epsilon))
        m.tank_temp_d3 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, 20 - m.epsilon))
        m.tank_temp_b1 = Var(m.horiz_t_init, domain=Binary)
        m.tank_temp_b2 = Var(m.horiz_t_init, domain=Binary)
        m.tank_temp = Expression(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d1[i] + m.tank_temp_d2[i] + m.tank_temp_d3[i] + m.tank_temp_min)
        m.soc = Expression(m.horiz_t_init, rule=lambda m, i: 1. - m.tank_temp_d2[i] / (2 * m.epsilon))

        m.tank_temp_d1_lb = Constraint(m.horiz_t_init, rule=lambda m, i: (-m.epsilon + 10) * m.tank_temp_b1[i]
                                                                <= m.tank_temp_d1[i])
        m.tank_temp_d1_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d1[i] <= -m.epsilon + 10)
        m.tank_temp_d2_lb = Constraint(m.horiz_t_init, rule=lambda m, i: (2 * m.epsilon) * m.tank_temp_b2[i] <= m.tank_temp_d2[i])
        m.tank_temp_d2_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d2[i] <= 2 * m.epsilon * m.tank_temp_b1[i])
        m.tank_temp_d3_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d3[i] <=
                                                                (20 - m.epsilon) * m.tank_temp_b2[i])
        m.avail_Q_vs_T = Constraint(m.horiz_t_init, rule=lambda m, i: (ice_cp * m.tank_temp_d1[i] + h_if * m.tank_temp_d2[i] / 2 / m.epsilon
                                                                    + water_cp * m.tank_temp_d3[i]) * m.total_fluid_mass * Q_scaling_factor == m.avail_Q[i])
    elif pw_repn == "incremental_constant_T":
        # temperature delta variables
        m.tank_temp_d1 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, 10))
        m.tank_temp_d2 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, 1))
        m.tank_temp_d3 = Var(m.horiz_t_init, domain=NonNegativeReals, bounds=(0, 20))
        # binary variables
        m.tank_temp_b1 = Var(m.horiz_t_init, domain=Binary)
        m.tank_temp_b2 = Var(m.horiz_t_init, domain=Binary)
        # tank temperature and soc
        m.tank_temp = Expression(m.horiz_t_init, rule=lambda m, i: -10 + m.tank_temp_d1[i] + m.tank_temp_d3[i] + 273.15)
        m.soc = Expression(m.horiz_t_init, rule=lambda m, i: 1. - m.tank_temp_d2[i])
        # delta variable bounds
        m.tank_temp_d1_lb = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_b1[i]*10 <= m.tank_temp_d1[i])
        m.tank_temp_d1_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d1[i] <= 10)
        m.tank_temp_d2_lb = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_b2[i] <= m.tank_temp_d2[i])
        m.tank_temp_d2_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d2[i] <= m.tank_temp_b1[i])
        m.tank_temp_d3_ub = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_d3[i] <= 20*m.tank_temp_b2[i])
        # binary constraints
        m.b1_gt_b2 = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_b1[i] >= m.tank_temp_b2[i])
        # energy PWL equation
        m.avail_Q_vs_T = Constraint(m.horiz_t_init, rule=lambda m, i: m.avail_Q[i] == (ice_cp*m.tank_temp_d1[i] + h_if*m.tank_temp_d2[i] + water_cp*m.tank_temp_d3[i]) * m.total_fluid_mass * Q_scaling_factor)
    else:
        m.tank_temp = Var(m.horiz_t_init, bounds=(min_tank_temp, max_tank_temp), units=units.K)
        m.pwl = Piecewise(m.horiz_t_init, m.avail_Q, m.tank_temp, 
                        pw_pts=tank_phase_change_k, 
                        pw_constr_type="EQ", 
                        f_rule=[i * Q_scaling_factor * value(m.total_fluid_mass) for i in tank_phase_change_Q], 
                        pw_repn=pw_repn) #SOS2
        m.soc = Expression(m.horiz_t_init, rule=lambda m, i: m.pwl[i].SOS2_y[0] + m.pwl[i].SOS2_y[1])
        for i in m.horiz_t_init:
            m.pwl[i].SOS2_y.setub(1)


def model(m, tank, n=2, max_mass_flow_rate=test_mass_flow_rate):
    m.num_tanks = Param(domain=NonNegativeIntegers, mutable=False, default=1)

    m.tank_temp_min = Param(domain=Reals, mutable=False, default=min_tank_temp, units=units.K)
    m.tank_temp_max = Param(domain=Reals, mutable=False, default=max_tank_temp, units=units.K)
    m.total_fluid_mass = Param(domain=NonNegativeReals, mutable=False, default=tank.total_fluid_mass)
    # m.tank_temp = Var(m.horiz_init, domain=Reals, bounds=(m.tank_temp_min, m.tank_temp_max))
    m.inlet_temp = Var(m.horiz_t, domain=Reals, bounds=(brine_freezing_pt, max_tank_temp), units=units.K)
    m.outlet_temp = Var(m.horiz_t, domain=Reals, bounds=(brine_freezing_pt, max_tank_temp), units=units.K)
    m.outlet_settemp = Var(m.horiz_t_init, domain=Reals, units=units.K, bounds=(brine_freezing_pt, max_tank_temp))
    # m.ice_mass = Var(m.horiz_t, domain=NonNegativeReals, bounds=(0, m.total_fluid_mass))

    m.env_temp = Param(m.horiz_t, domain=Reals, mutable=True, default=297.15, units=units.K)

    m.tank_ua_env = Param(domain=NonNegativeReals, mutable=False, default=tank.tank_ua_env)
    m.tank_ua_hx = Param(domain=NonNegativeReals, mutable=False, default=tank.tank_ua_hx)

    m.water_specific_heat = Param(domain=NonNegativeReals, mutable=True,
                                  default=water_cp)
    # if not using avg temp to calculate brine specific heat, there can be at least 15% difference in q_tot
    # m.brine_specific_heat = Param(domain=NonNegativeReals, mutable=True,
                                #   default=brine_cp)  # examine how sensitive later
    # m.del_component("brine_specific_heat")
    slope, intercept, temps, cps = brine_cp_linear_fxn()
    m.brine_cp_slope = Param(domain=Reals, mutable=False, default=slope)
    m.brine_cp_intercept = Param(domain=Reals, mutable=False, default=intercept)

    m.max_mass_flow_rate = Param(domain=NonNegativeReals, mutable=False,
                                 default=max_mass_flow_rate)

    # keep non-zero lb else ampl_error in the exp() function of effectiveness at flow rates of 0
    m.tank_flow_fraction = Var(m.horiz_t, domain=NonNegativeReals, bounds=(1e-6, 1), initialize=1, doc="frac of flow to tank, rest is bypassed")

    return m


def operation_model(m, calc_mode):
    if not calc_mode:
        m.is_charging = Param(m.horiz_t, domain=Binary, mutable=True, default=0)  # 1 if True
        m.is_discharging = Param(m.horiz_t, domain=Binary,  mutable=True, default=0)
    else:
        m.is_charging = Var(m.horiz_t, domain=Binary)  # 1 if True
        m.is_discharging = Expression(m.horiz_t, rule=lambda m, i: 1 - m.is_charging[i])
        m.temp_diff_ub = Param(domain=NonNegativeReals, mutable=False, default=max_tank_temp - brine_freezing_pt, units=units.K)
        m.discharge_mode_calc = Constraint(m.horiz_t,
                                        rule=lambda m, i: m.outlet_temp[i] - m.inlet_temp[i] <=
                                                        m.is_charging[i] * m.temp_diff_ub)
        m.charge_mode_calc = Constraint(m.horiz_t,
                                        rule=lambda m, i: m.inlet_temp[i] - m.outlet_temp[i] <=
                                                        m.is_discharging[i] * m.temp_diff_ub)


L, k, x_0, a, b = (0.46399073996160234, 79.99996255766429, 0.12387593954791037, 0.3494863028354558, 0.1931963545684955)
def heat_transfer_model(m, simple=True):
    m.mass_flow_rate = Expression(m.horiz_t, rule=lambda m, i: m.tank_flow_fraction[i] * m.max_mass_flow_rate)
    if simple:
        m.tank_flow_fraction.fix(1)

    m.brine_specific_heat = Expression(m.horiz_t, rule=lambda m, i: m.brine_cp_slope * m.inlet_temp[i] + m.brine_cp_intercept)
    m.brine_specific_heat_avg = Expression(m.horiz_t, rule=lambda m, i: m.brine_cp_slope * (m.inlet_temp[i] + m.tank_temp[i - 1]) / 2 + m.brine_cp_intercept)
    m.effectiveness = Expression(m.horiz_t, rule=lambda m, i: 1.0 - exp(
        -m.tank_ua_hx / ((m.mass_flow_rate[i] + 1e-8) * m.brine_specific_heat[i])))
    m.brine_effectiveness_charging = Expression(m.horiz_t, rule=lambda m, i: -0.3 * m.soc[i - 1] + 1.2)
    m.brine_effectiveness_discharging = Expression(m.horiz_t, rule=lambda m, i: L / (1 + exp(-k * (m.soc[i - 1] - x_0))) + a * m.soc[i - 1] + b)

    m.q_max_brine = Expression(m.horiz_t, rule=lambda m, i: (m.mass_flow_rate[i] * m.brine_specific_heat_avg[i]
                                                      * (m.inlet_temp[i] - m.tank_temp[i - 1]) * m.timestep))
    m.q_brine = Expression(m.horiz_t, rule=lambda m, i: m.q_max_brine[i]
                                                      * m.effectiveness[i] 
                                                      * (m.brine_effectiveness_discharging[i] * m.is_discharging[i] + m.brine_effectiveness_charging[i] * m.is_charging[i]) * Q_scaling_factor)
    m.q_env = Expression(m.horiz_t, rule=lambda m, i: m.tank_ua_env * (m.env_temp[i] - m.tank_temp[i - 1]) * m.timestep * Q_scaling_factor)
    m.q_tot = Expression(m.horiz_t, rule=lambda m, i: m.q_brine[i] + m.q_env[i])


def outlet_temp_model(m):
    m.outlet_effectiveness_charging = Expression(m.horiz_t, rule=lambda m, i: -0.3 * m.soc[i] + 1.2)
    m.outlet_effectiveness_discharging = Expression(m.horiz_t, rule=lambda m, i: L / (1 + exp(-k * (m.soc[i] - x_0))) + a * m.soc[i] + b)

    m.outlet_effectiveness = Expression(m.horiz_t, rule=lambda m, i: m.effectiveness[i] * (
                                                                    m.outlet_effectiveness_discharging[i] * m.is_discharging[i] + m.outlet_effectiveness_charging[i] * m.is_charging[i]))
    m.outlet_temp_calc = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_temp[i] == m.inlet_temp[i] 
                                                                - m.outlet_effectiveness[i] * (m.inlet_temp[i] - m.tank_temp[i]))
    # branch outlet temp, as mix of bypass and tank outlet, needs to meet set point
    m.m_dot_bypass = Expression(m.horiz_t, rule=lambda m, i: m.max_mass_flow_rate - m.mass_flow_rate[i])
    m.meet_outlet_setpoint = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] == 
            (m.mass_flow_rate[i] * m.outlet_temp[i] + m.m_dot_bypass[i] * m.inlet_temp[i]) / m.max_mass_flow_rate)

    m.tank_Q_evolution = Constraint(m.horiz_t, rule=lambda m, i: m.avail_Q[i - 1]  == (m.avail_Q[i] - m.q_tot[i]))
    
    return m


chiller_a = [ 0.35750133, -9.7829402 ] 
chiller_b = -100.14402057970096
def building_loop_model(m):
    # atmospheric dependence
    m.cond_inlet = Param(m.horiz_t, domain=Reals, initialize=273.15, mutable=True, units=units.K)
    # building thermal load's resulting delta T
    m.evap_diff = Param(m.horiz_t, domain=Reals, initialize=0, mutable=True, units=units.K)
    # chiller power depends on 2 variables above: more efficient at lower condenser temp and uses more power for more delta T
    m.chiller_power = Var(m.horiz_t, domain=NonNegativeReals, bounds=(0, max_chiller_load))
    # chilled-water-loop inlet temperature == chilled-water-loop outlet temperature from last time step + building's delta T across time step
    m.loop_inlet_temp = Expression(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i - 1] + m.evap_diff[i])
    # temperature difference across chiller (tank inlet temp - chilled-water-loop inlet )
    m.chiller_delta_T = Expression(m.horiz_t, rule=lambda m, i:  m.inlet_temp[i] -  m.loop_inlet_temp[i])
    # chiller power is fitted model (chiller_load.ipynb)
    m.chiller_power_calc = Constraint(m.horiz_t, rule=lambda m, i: 
        m.chiller_power[i] == chiller_a[0] * m.cond_inlet[i] + chiller_a[1] * m.chiller_delta_T[i] + chiller_b)
    m.chiller_delta_T_negative = Constraint(m.horiz_t, rule=lambda m, i: m.chiller_delta_T[i] <= 0)
    # m.chiller_power_lb = Constraint(m.horiz_t, rule=lambda m, i: m.chiller_power[i] >= 0)

evap_diff_df = None
def init_building(m, starting_timestep):
    n = len(m.horiz_t)
    global evap_diff_df
    if evap_diff_df is None:
        evap_diff_df = pd.read_parquet(thermal_tank_log_dir / "mpc_forecast_states.parquet")
    evap_diff = evap_diff_df[f"evap_diff_{num_tanks}"].values[starting_timestep:starting_timestep+n]
    cond_inlet_temps = (evap_diff_df["cond_inlet"] + 273.15).values[starting_timestep:starting_timestep+n]
    loop_inlet_temps = (evap_diff_df[f"loop_inlet_{num_tanks}"] + 273.15).values[starting_timestep:starting_timestep+n+1]
    loop_outlet_temps = (evap_diff_df[f"loop_outlet_{num_tanks}"] + 273.15).values[starting_timestep:starting_timestep+n]
    
    # set previous outlet temp to be the same as beginning inlet temp, so no chiller power
    m.outlet_settemp[-1].fix(loop_inlet_temps[0])
    m.inlet_temp[n - 1].set_value(loop_inlet_temps[n - 1])

    for i in m.horiz_t:
        m.cond_inlet[i].set_value(cond_inlet_temps[i])
        evap_diff[i] = max(loop_inlet_temps[i + 1] - loop_outlet_temps[i], 0)
        m.evap_diff[i].set_value(evap_diff[i])
        # loop_inlet_temp_i = value(m.outlet_settemp[i - 1] + m.evap_diff[i - 1])
        chiller_delta_i = value(m.inlet_temp[i] - loop_inlet_temps[i])
        if chiller_delta_i > 0:
            # m.evap_diff[i - 1].set_value(value(m.inlet_temp[i] - m.outlet_settemp[i - 1]))
            chiller_delta_i = value(m.inlet_temp[i] - loop_inlet_temps[i])
            # print(f"error at {i}: Chiller Delta was > 0: inlet {loop_inlet_temp_i}, outlet {value(m.inlet_temp[i])}")
        chiller_power_i = value(0.22952657 * m.cond_inlet[i] - 10.74753087 * chiller_delta_i - 66.4183144)
        if chiller_power_i < 0:
            chiller_power_i = 0
        m.chiller_power[i].set_value(chiller_power_i)

def init_temp(m, tank_temp_C, latent_soc, index, fixed=True):
    if hasattr(m, "avail_Q_vs_T"):
        if abs(tank_temp_C) > 1e-4:
            if tank_temp_C < -phase_change_e:
                m.tank_temp_d1[index].fix(tank_temp_C + 10)
                m.tank_temp_b1[index].set_value(0)
                m.tank_temp_d2[index].fix(0)
                m.tank_temp_b2[index].set_value(0)
                m.tank_temp_d3[index].fix(0)
            elif tank_temp_C < phase_change_e:
                m.tank_temp_d1[index].fix(10 - phase_change_e)
                m.tank_temp_b1[index].set_value(1)
                m.tank_temp_d2[index].fix(tank_temp_C + phase_change_e)
                m.tank_temp_b2[index].set_value(0)
                m.tank_temp_d3[index].fix(0)
            else:
                m.tank_temp_d1[index].fix(10 - phase_change_e)
                m.tank_temp_b1[index].set_value(1)
                m.tank_temp_d2[index].fix(2 * phase_change_e)
                m.tank_temp_b2[index].set_value(1)
                m.tank_temp_d3[index].fix(tank_temp_C - phase_change_e)
            assert abs(value(m.tank_temp_d1[index] + m.tank_temp_d2[index] + m.tank_temp_d3[index] - 10) - tank_temp_C) < 1e-5
        else:
            m.tank_temp_d1[index].fix(10 - phase_change_e)
            m.tank_temp_b1[index].set_value(1)
            m.tank_temp_d2[index].fix((1. - latent_soc) * 2 * m.epsilon)
            m.tank_temp_b2[index].set_value(0)
            m.tank_temp_d3[index].fix(0)
        m.avail_Q[index].set_value(value(ice_cp * m.tank_temp_d1[index] + h_if * m.tank_temp_d2[index] / 2 / m.epsilon
                                                            + water_cp * m.tank_temp_d3[index]) * m.total_fluid_mass * Q_scaling_factor)
        if not fixed:
            m.avail_Q[index].unfix()
            m.tank_temp_d1[index].unfix()
            m.tank_temp_d2[index].unfix()
            m.tank_temp_d3[index].unfix()
            m.tank_temp_b1[index].unfix()
            m.tank_temp_b2[index].unfix()

    elif hasattr(m, "pwl"):
        if tank_temp_C != 0:
            m.tank_temp[index].fix(273.15 + tank_temp_C)
        else:
            m.tank_temp[index].fix(273.15 + (1. - latent_soc) * 2 * m.epsilon - m.epsilon)
        if not fixed:
            m.tank_temp[index].unfix()

def init_mpc(m, tank: IceTank, verbose=True, solver_name="bonmin"):
    latent_soc = tank.state_of_charge
    tank_temp = tank.tank_temp
    init_temp(m, tank_temp, latent_soc, -1)
    m.obj = Objective(expr=0)
    solver = SolverFactory(solver_name)
    solver.options['tol'] = 1e-6
    # solver.options['halt_on_ampl_error'] = 'yes'
    res = solver.solve(m, tee=verbose)
    m.del_component("obj")
    solved = res.Solver.status == 'ok'
    if not solved:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
    return solved


def full_model(tank, max_mass_flow_rate=test_mass_flow_rate, simple=False, calc_mode=False, n=2, existing_model=None, pw_repn="incremental", timestep=60):
    if existing_model is None:
        m = ConcreteModel()
    else:
        m = existing_model
    m.horiz_t = RangeSet(0, n)
    n = len(m.horiz_t) - 1
    m.horiz_t_init = RangeSet(-1, n)
    m.timestep = Param(domain=NonNegativeReals, mutable=False,
                       default=timestep)
    model(m, tank, n, max_mass_flow_rate)
    temp_model(m, pw_repn)
    operation_model(m, calc_mode)
    heat_transfer_model(m, simple=simple)
    outlet_temp_model(m)
    return m, tank


def construct_tes(m, timeset, tank, start_ts, mins_per_ts, KW_SCALING,
                  bldg_load, chw_outlet_temp):
    seconds_per_ts = mins_per_ts * 60
    n = int(24 * (60 / mins_per_ts))

    # construct MPC
    def block_rule(b, t, tank, timestep):
        full_model(tank, existing_model=b, simple=True, calc_mode=True, n=0, timestep=timestep)
        building_loop_model(b)
        b.bldg_elec_load = Expression(RangeSet(0, 0), rule=lambda m, i: b.chiller_power[0] * KW_SCALING + bldg_load[(start_ts + t) % len(bldg_load)])

    m.block = Block(timeset, rule=partial(block_rule, tank=tank, timestep=seconds_per_ts))

    m.tank_temp_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp[0] == m.block[i+1].tank_temp[-1])
    m.loop_temp_evol = Constraint(range(n), rule=lambda m, i:m.block[i].outlet_settemp[0] == m.block[i+1].outlet_settemp[-1])
    m.avail_Q_evol = Constraint(range(n), rule=lambda m, i:m.block[i].avail_Q[0] == m.block[i+1].avail_Q[-1])
    m.tank_temp_d1_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp_d1[0] == m.block[i+1].tank_temp_d1[-1])
    m.tank_temp_d2_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp_d2[0] == m.block[i+1].tank_temp_d2[-1])
    m.tank_temp_d3_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp_d3[0] == m.block[i+1].tank_temp_d3[-1])
    m.tank_temp_b1_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp_b1[0] == m.block[i+1].tank_temp_b1[-1])
    m.tank_temp_b2_evol = Constraint(range(n), rule=lambda m, i:m.block[i].tank_temp_b2[0] == m.block[i+1].tank_temp_b2[-1])

    m.tank_temp_periodic = Constraint(expr=m.block[n].tank_temp[0] == m.block[0].tank_temp[-1])
    m.loop_temp_periodic = Constraint(expr=m.block[n].outlet_settemp[0] == m.block[0].outlet_settemp[-1])
    m.avail_Q_periodic = Constraint(expr=m.block[n].avail_Q[0] == m.block[0].avail_Q[-1])
    m.tank_temp_d1_periodic = Constraint(expr=m.block[n].tank_temp_d1[0] == m.block[0].tank_temp_d1[-1])
    m.tank_temp_d2_periodic = Constraint(expr=m.block[n].tank_temp_d2[0] == m.block[0].tank_temp_d2[-1])
    m.tank_temp_d3_periodic = Constraint(expr=m.block[n].tank_temp_d3[0] == m.block[0].tank_temp_d3[-1])
    m.tank_temp_b1_periodic = Constraint(expr=m.block[n].tank_temp_b1[0] == m.block[0].tank_temp_b1[-1])
    m.tank_temp_b2_periodic = Constraint(expr=m.block[n].tank_temp_b2[0] == m.block[0].tank_temp_b2[-1])

    # minimize the max load while making sure the outlet temp doesn't get too warm
    m.max_outlet_temp = Constraint(timeset, rule=lambda m, i: m.block[i].outlet_settemp[0] <= max(chw_outlet_temp) + 273.15)
    
    def deriv_reg_2nd(m, i):
        if i == 0:
            return (m.block[0].outlet_temp[0] - 2 * m.block[n].outlet_temp[0] + m.block[n-1].outlet_temp[0]) ** 2 \
                    + (m.block[0].chiller_power[0] - 2 * m.block[n].chiller_power[0] + m.block[n-1].chiller_power[0]) ** 2
        elif i == 1:
            return (m.block[1].outlet_temp[0] - 2 * m.block[0].outlet_temp[0] + m.block[n].outlet_temp[0]) ** 2 \
                    + (m.block[1].chiller_power[0] - 2 * m.block[0].chiller_power[0] + m.block[n].chiller_power[0]) ** 2
        return (m.block[i].outlet_temp[0] - 2 * m.block[i-1].outlet_temp[0] + m.block[i-2].outlet_temp[0]) ** 2 \
                + (m.block[i].chiller_power[0] - 2 * m.block[i-1].chiller_power[0] + m.block[i-2].chiller_power[0]) ** 2

    def deriv_reg_1st(m, i):
        if i == 0:
            return (m.block[n].outlet_temp[0] - m.block[0].outlet_temp[0]) ** 2 + (m.block[n].chiller_power[0] - m.block[0].chiller_power[0]) ** 2
        return (m.block[i].outlet_temp[0] - m.block[i-1].outlet_temp[0]) ** 2 + (m.block[i].chiller_power[0] - m.block[i-1].chiller_power[0]) ** 2
        
    m.deriv_1st_enforce = Param(domain=NonNegativeReals, default=0, mutable=True)
    m.deriv_2nd_enforce = Param(domain=NonNegativeReals, default=0, mutable=True)
    m.deriv_reg = Expression(timeset, rule=lambda m, i: deriv_reg_1st(m, i) * m.deriv_1st_enforce + deriv_reg_2nd(m, i) * m.deriv_2nd_enforce)

    def binary_close(m, i):
        return sum(-16 * (getattr(m.block[i], var)[0] - 0.5 )**4 + 1 for var in binary_vars)

    m.binary_close = Expression(timeset, rule=binary_close)
    m.binary_enforce = Param(domain=NonNegativeReals, default=0, mutable=True)


def get_var_values(m, verbose=False):
    tank_temp = []
    mass_flow = []
    q_tot = []
    tank_flow_frac = []
    inlet_temp = []
    outlet_temp = []
    outlet_settemp = []
    avail_Q = []

    print("-1")
    print("tank_temp", value(m.tank_temp[-1]))
    if hasattr(m, "tank_temp_d1"):
        print("tank_temp_d1", value(m.tank_temp_d1[-1]))
        print("tank_temp_d2", value(m.tank_temp_d2[-1]))
        print("tank_temp_d3", value(m.tank_temp_d3[-1]))
    print("avail_Q", value(m.avail_Q[-1]))
    print()
    for i in m.horiz_t:
        if verbose:
            print(i)
            print("effectiveness", value(m.effectiveness[i]))
            print("num_transfer_units", value(-m.tank_ua_hx / (m.mass_flow_rate[i] * m.brine_specific_heat[i])))
            print("brine_effectiveness_charging", value(m.brine_effectiveness_charging[i]))
            print("q_brine", value(m.q_brine[i]))
            print("q_env", value(m.q_env[i]))
            print("q_tot", value(m.q_tot[i]))
            print("mass_flow", value(m.mass_flow_rate[i]))
            print("inlet_temp", value(m.inlet_temp[i]))
            if hasattr(m, "tank_temp_d1"):
                print("tank_temp_d1", value(m.tank_temp_d1[i]))
                print("tank_temp_d2", value(m.tank_temp_d2[i]))
                print("tank_temp_d3", value(m.tank_temp_d3[i]))
            print("tank_temp", value(m.tank_temp[i]))
            print("avail_Q", value(m.avail_Q[i]))
            print("outlet_temp", value(m.outlet_temp[i]))
            print("tank_flow_fraction", value(m.tank_flow_fraction[i]))
            print("outlet_settemp", value(m.outlet_settemp[i]))

        tank_temp.append(value(m.tank_temp[i]))
        mass_flow.append(value(m.mass_flow_rate[i]))
        q_tot.append(value(m.q_tot[i]))
        outlet_temp.append(value(m.outlet_temp[i]))
        tank_flow_frac.append(value(m.tank_flow_fraction[i]))
        inlet_temp.append(value(m.inlet_temp[i]))
        outlet_settemp.append(value(m.outlet_settemp[i]))
        avail_Q.append(value(m.avail_Q[i]))
    return tank_temp, mass_flow, q_tot, outlet_temp, tank_flow_frac, inlet_temp, outlet_settemp, avail_Q


def plot_model(m):
    tank_temp, mass_flow, q_tot, outlet_temp, tank_flow_frac, inlet_temp, outlet_settemp = get_var_values(m)
    fig = plt.figure(figsize=(16, 5))
    plt.plot(tank_temp, label="tank temp")
    plt.plot(inlet_temp, label="inlet temp")
    plt.plot(outlet_temp, label="outlet temp")
    plt.xlabel("Min")
    plt.ylabel("C")
    plt.legend()
    plt.grid()
    plt.show()


def phase_model(m):
    m.tank_temp = Var(m.horiz_t_init, domain=Reals)
    m.epsilon = Param(domain=NonNegativeReals, mutable=False, default=1e-7)
    m.tank_temp_ub = Param(domain=NonNegativeReals, mutable=False, default=50)
    
    # sensible fluid vs latent ice vs sensible subcooled ice
    m.is_sensible_fluid_mode = Var(m.horiz_t, domain=Binary)
    m.is_latent_ice_mode = Var(m.horiz_t, domain=Binary)
    m.is_sensible_ice_mode = Var(m.horiz_t, domain=Binary)
    m.one = Param(domain=Binary, mutable=False, default=1)

    m.phase_constr_ub = Constraint(m.horiz_t, rule=lambda m, i: m.is_sensible_ice_mode[i] + m.is_sensible_fluid_mode[i] +
                                                                  m.is_latent_ice_mode[i] == 1)
    m.sensible_fluid_mode_calc = Constraint(m.horiz_t,
                                            rule=lambda m, i: -m.tank_temp[i] <= -m.epsilon + m.tank_temp_ub * (
                                                    m.one - m.is_sensible_fluid_mode[i]))
    m.sensible_ice_mode_calc = Constraint(m.horiz_t,
                                            rule=lambda m, i: m.tank_temp[i] <= -m.epsilon + m.tank_temp_ub * (
                                                    m.one - m.is_sensible_ice_mode[i]))
    m.latent_ice_mode_calc_ub = Constraint(m.horiz_t,
                                            rule=lambda m, i: m.tank_temp[i] <= m.epsilon + m.tank_temp_ub * (
                                                    m.one - m.is_latent_ice_mode[i]))
    m.latent_ice_mode_calc_lb = Constraint(m.horiz_t,
                                            rule=lambda m, i: -m.tank_temp[i] <= m.epsilon + m.tank_temp_ub * (
                                                    m.one - m.is_latent_ice_mode[i]))


binary_vars = ['tank_temp_b1', 'tank_temp_b2', 'is_charging']
def check_binary_variables(m, n):
    diffs = {}
    for i in range(n + 1):
        for var in binary_vars:
            val = value(getattr(m.block[i], var)[0])
            if abs(val - 0) > 1e-3 and abs(val - 1) > 1e-3:
                if val <= 0.5:
                    diffs[str(getattr(m.block[i], var))] = val
                else:
                    diffs[str(getattr(m.block[i], var))] = 1 - val
    return (True if len(diffs) == 0 else False), diffs