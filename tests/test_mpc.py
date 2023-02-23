import pytest
from functools import partial
import copy
import json
import math
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import pyomo.environ as pyo
import idaes.logger as idaeslog
from timeit import default_timer
from thermal_tank.init_from_log import populate_values, plot_modified_operation, save_vars_to_df, save_params_to_df
from thermal_tank.tank_bypass_branch import TankBypassBranch
from thermal_tank.mpc import *
from thermal_tank.compare_op_strategy import day_avg_bldg_load, coned_schedule, day_avg_evap_diff


def test_brine_cp_linear_fxn():
    slope, intercept, temps, cps = brine_cp_linear_fxn()
    plt.scatter(temps, cps, label='coolprop')
    plt.plot(temps, slope * temps + intercept, label='linear model')
    plt.xlabel('T [K]')
    plt.ylabel('Cp [J/kg K]')
    plt.title(f'Cp Range = {max(cps)-min(cps):.1f} || Range/Avg = {(max(cps)-min(cps))/np.mean(cps)*100:.1f} %')
    plt.legend()
    plt.show()


def test_brine_cp_vs_glycol_fraction():
    temps = np.arange(-20, 80)
    for fraction in [0, 0.05, 0.1, 0.2, 0.3]:
        fluid = 'INCOMP::MPG[' + str(fraction) + ']'
        cps = specific_heat(fluid, temps)
        plt.plot(temps, cps, label=f'{fraction:.2f}')
    plt.legend(title='fraction of glycol in water')
    plt.ylabel('Cp [J/kg K]')
    plt.xlabel('T [C]')
    plt.show()


def test_charging_mode():
    tank = IceTank(tank_data)
    m = ConcreteModel()
    n = 2
    m.horiz_t = RangeSet(0, n)
    n = len(m.horiz_t) - 1
    m.horiz_t_init = RangeSet(-1, n)
    m.timestep = Param(domain=NonNegativeReals, mutable=False, default=60)
    m.inlet_temp = Var(m.horiz_t, domain=Reals, bounds=(brine_freezing_pt, max_tank_temp))
    m.outlet_temp = Var(m.horiz_t, domain=Reals, bounds=(brine_freezing_pt, max_tank_temp))
    operation_model(m, True)
    for n in m.horiz_t:
        m.outlet_temp[n].fix(274.15)
        m.inlet_temp[n].fix(value(n + 274.16))
    m.obj = Objective(expr=0)
    solver = SolverFactory("bonmin")
    solver.solve(m, tee=False)
    for n in m.horiz_t:
        if value(m.outlet_temp[n] - m.inlet_temp[n]) > 0:
            assert value(m.is_charging[n])
        elif value(m.outlet_temp[n] - m.inlet_temp[n]) < 0:
            assert value(m.is_discharging[n])
        else:
            assert value(m.is_discharging[n] + m.is_charging[n])


def test_piecewise_temperatures():
    """
    Test 
    """
    phase_change_dt = 0.01
    # test input
    expected_temps = [-10, -phase_change_dt, phase_change_dt, 10]

    # expected output
    d1 = [0, 10 - phase_change_dt, 10 - phase_change_dt, 10 - phase_change_dt]
    d2 = [0, 0, 2 * phase_change_dt, 2 * phase_change_dt]
    d3 = [0, 0, 0, 10 - phase_change_dt]
    expected_avail_Q = [0, 126.824, 2215.570, 2478.401]

    for i in range(4):
        data = copy.deepcopy(tank_data)
        # if T = -10 or T = 10, set the initial temperature to the expected value
        if abs(expected_temps[i]) > 2 * phase_change_dt:
            data['initial_temperature'] = expected_temps[i]
        # if T = -0.01 or T = 0.01, remove initial temperature and set state of charge to 0 or 1
        else:
            data.pop('initial_temperature')
            data['latent_state_of_charge'] = 1 if expected_temps[i] < 0 else 0
        tank = IceTank(data)
        m = ConcreteModel()
        n = 1
        m.horiz_t = RangeSet(0, n)
        m.horiz_t_init = RangeSet(-1, n)
        m.timestep = Param(domain=NonNegativeReals, mutable=False, default=60)
        model(m, tank, n)
        temp_model(m, pw_repn="incremental")
        operation_model(m, calc_mode=False)
        heat_transfer_model(m, simple=False)
        outlet_temp_model(m)
        m.is_discharging[0].set_value(1)
        m.is_charging[0].set_value(0)

        init_mpc(m, tank)

        assert value(m.tank_temp_d1[-1]) == pytest.approx(d1[i], abs=1e-3)
        assert value(m.tank_temp_d2[-1]) == pytest.approx(d2[i], abs=1e-3)
        assert value(m.tank_temp_d3[-1]) == pytest.approx(d3[i], abs=1e-3)
        assert value(m.avail_Q[-1]) == pytest.approx(expected_avail_Q[i], abs=1e3)

        assert abs(value(m.tank_temp[-1]) - (expected_temps[i] + 273.15)) < 1e-2

    for i in range(4):
        data = copy.deepcopy(tank_data)
        if abs(expected_temps[i]) > 2 * phase_change_dt:
            data['initial_temperature'] = expected_temps[i]
        else:
            data.pop('initial_temperature')
            data['latent_state_of_charge'] = 1 if expected_temps[i] < 0 else 0
        tank = IceTank(data)
        m = ConcreteModel()
        n = 1
        m.horiz_t = RangeSet(0, n)
        m.horiz_t_init = RangeSet(-1, n)
        m.timestep = Param(domain=NonNegativeReals, mutable=False,
                        default=60)
        model(m, tank, n)

        temp_model(m, pw_repn="SOS2")
        operation_model(m, calc_mode=False)
        heat_transfer_model(m, simple=False)
        outlet_temp_model(m, )
        m.is_discharging[0].set_value(1)
        m.is_charging[0].set_value(0)

        init_mpc(m, tank, solver_name='bonmin')
        assert value(m.avail_Q[-1]) == pytest.approx(expected_avail_Q[i], abs=1e3)
        assert abs(value(m.tank_temp[-1]) - (expected_temps[i] + 273.15)) < 1e-2


def test_plot_PWL_temp_model():
    # create model 
    m = ConcreteModel()
    n = 1
    m.horiz_t = RangeSet(0, n)
    m.horiz_t_init = RangeSet(-1, n)
    m.timestep = Param(domain=NonNegativeReals, mutable=False, default=60)

    # create temperature model
    data = copy.deepcopy(tank_data)
    # data['initial_temperature'] = -10.
    tank = IceTank(data)
    model(m, tank, n)
    temp_model(m, pw_repn="incremental")

    # fake objective to solve for constraints
    m.obj = Var(domain=NonNegativeReals)
    m.objective = Objective(expr=m.obj)

    # Fix enthalpy and calculate tank temperature
    # Q_range = np.linspace(0, 2478.401, 15)
    Q_range = np.hstack([np.linspace(0, 126.824, 5), np.linspace(126.824, 2215.570, 5), np.linspace(2215.570, 2741.232, 5)])
    T_range = np.zeros_like(Q_range)
    for (i, Q_value) in enumerate(Q_range):
        m.avail_Q.fix(Q_value)
        solver = SolverFactory('cbc')
        results = solver.solve(m, tee=False)
        T_range[i] = m.tank_temp[-1]()    
    plt.plot(T_range-273.15, Q_range, label='fix tank energy and calculate temp')
    
    # Fix temperature and calculate enthalpy
    m.avail_Q.free()
    # need to add a variable to set it to the desired value (model uses an expression for tank temp)
    m.tank_temp_variable = Var(m.horiz_t_init, domain=NonNegativeReals)
    m.tank_temp_constraint = Constraint(m.horiz_t_init, rule=lambda m, i: m.tank_temp_variable[i] == m.tank_temp[i])
    T_range = np.hstack([np.linspace(-10, -0.02, 5), np.linspace(-0.01, 0.01, 5), np.linspace(0.02, 20, 5)]) + 273.15
    Q_range = np.zeros_like(T_range)
    for (i, T_value) in enumerate(T_range):
        # initialize temperature:
        m.tank_temp_variable.fix(T_value)
        solver = SolverFactory('cbc')
        results = solver.solve(m, tee=False)
        Q_range[i] = m.avail_Q[-1].value    
    plt.plot(T_range-273.15, Q_range, '--', label='fix tank temp and calculate energy')
    plt.ylabel('Tank Energy [J]')
    plt.xlabel('Tank Temperature [K]')
    plt.legend()
    plt.show()


def test_plot_PWL_temp_model_robi():
    
    # tank model, need total fluid mass
    tank = IceTank(tank_data)

    # initialize pyomo model
    m = ConcreteModel()
    
    # indexed sets (might not need t)
    n_timesteps = 1
    m.t = RangeSet(n_timesteps)  # 1:n

    # variables
    m.avail_Q = Var(m.t, domain=NonNegativeReals)
    # m.tank_temp = Var(m.t, domain=NonNegativeReals, bounds=(-10, 20))
    m.tank_temp = Var(m.t, domain=Reals)
    # temeprature delta variables
    m.tank_temp_d1 = Var(m.t, domain=NonNegativeReals, bounds=(0, 10))
    m.tank_temp_d2 = Var(m.t, domain=NonNegativeReals, bounds=(0, 1))
    m.tank_temp_d3 = Var(m.t, domain=NonNegativeReals, bounds=(0, 20))
    # binary variables
    m.tank_temp_b1 = Var(m.t, domain=Binary)
    m.tank_temp_b2 = Var(m.t, domain=Binary)
    # delta variable bounds
    m.tank_temp_d1_lb = Constraint(m.t, rule=lambda m, i: m.tank_temp_b1[i]*10 <= m.tank_temp_d1[i])
    m.tank_temp_d1_ub = Constraint(m.t, rule=lambda m, i: m.tank_temp_d1[i] <= 10)
    m.tank_temp_d2_lb = Constraint(m.t, rule=lambda m, i: m.tank_temp_b2[i] <= m.tank_temp_d2[i])
    m.tank_temp_d2_ub = Constraint(m.t, rule=lambda m, i: m.tank_temp_d2[i] <= m.tank_temp_b1[i])
    m.tank_temp_d3_ub = Constraint(m.t, rule=lambda m, i: m.tank_temp_d3[i] <= 20*m.tank_temp_b2[i])
    # binary constraints
    m.b1_gt_b2 = Constraint(m.t, rule=lambda m, i: m.tank_temp_b1[i] >= m.tank_temp_b2[i])
    # energy PWL equation
    m.avail_Q_vs_T = Constraint(m.t, rule=lambda m, i: m.avail_Q[i] == (ice_cp*m.tank_temp_d1[i] + h_if*m.tank_temp_d2[i] + water_cp*m.tank_temp_d3[i]) * tank.total_fluid_mass)
    # tank temperature PWL
    m.tank_temp_cons = Constraint(m.t, rule=lambda m, i:  m.tank_temp[i] == -10 + m.tank_temp_d1[i] + m.tank_temp_d3[i])
    # fake objective
    m.obj = Var(domain=NonNegativeReals)
    m.objective = Objective(expr=m.obj)

    # evaluate model at varying temperatures and energy
    m.avail_Q.free()
    m.tank_temp.free()
    nd = 5
    
    # range of energy and temperature values to evaluate
    Q_range = tank.total_fluid_mass*np.hstack([np.linspace(0, 9, nd)*ice_cp,
                                               10*ice_cp + np.linspace(0, 1, nd)*h_if,
                                               10*ice_cp + h_if + np.linspace(11, 20, nd)*water_cp])
    # Q_range = np.linspace(0, 2478.401, 15)
    T_range = np.zeros_like(Q_range)
    b1_range = np.zeros_like(Q_range)
    b2_range = np.zeros_like(Q_range)
    d1_range = np.zeros_like(Q_range)
    d2_range = np.zeros_like(Q_range)
    d3_range = np.zeros_like(Q_range)
    
    # solver
    solver = SolverFactory('cbc')

    # vary tank energy, calculate tank temperature
    for (i, Q_value) in enumerate(Q_range):
        m.avail_Q.fix(Q_value)
        # solve model
        results = solver.solve(m, tee=False)
        T_range[i] = m.tank_temp[1].value
        b1_range[i] = m.tank_temp_b1[1].value
        b2_range[i] = m.tank_temp_b2[1].value
        d1_range[i] = m.tank_temp_d1[1].value
        d2_range[i] = m.tank_temp_d2[1].value
        d3_range[i] = m.tank_temp_d3[1].value
        
    plt.plot(T_range, Q_range)
    plt.ylabel('Tank Energy [J]')
    plt.xlabel('Tank Temperature [C]')
    plt.show()


def run_tank_model(m, tank_data, verbose=False):
    """
    """
    tank_bypass_branch = TankBypassBranch(num_tanks=1, tank_data=tank_data)

    icemass = [tank_bypass_branch.tank.ice_mass]
    tanktemp = [tank_bypass_branch.tank.tank_temp]
    soc = [tank_bypass_branch.tank.state_of_charge]
    branchtemp = []
    outlettemp = []
    tankflowfrac = []
    if verbose:
        print("inlet_temp, tank_temp, state_of_chrg, tankflowfrac, outlet_temp, outlet_fluid_temp")

    if hasattr(m, 'horiz_t'):
        n = len(m.horiz_t)
    else:
        n = len(m.block)

    for sim_time in range(n):
        if hasattr(m, 'horiz_t'):
            blk = m
            i = sim_time
        else:
            blk = m.block[sim_time]
            i = 0

        # float_mode = value(blk.tank_flow_fraction[i]) <= 2e-5
        # if float_mode:
            # mode = 0
        # else:
        mode = 1 if value(blk.is_charging[i]) else -1
        tank_bypass_branch.simulate(inlet_temp=value(blk.inlet_temp[i] - 273.15),
                                    mass_flow_rate=value(blk.mass_flow_rate[i]),
                                    env_temp=value(blk.env_temp[i] - 273.15),
                                    branch_set_point=value(blk.outlet_settemp[i] - 273.15),
                                    op_mode=mode,
                                    sim_time=sim_time,
                                    timestep=value(blk.timestep),
                                    bypass_frac=1.0-value(blk.tank_flow_fraction[i]))
        icemass.append(tank_bypass_branch.tank.ice_mass / tank_bypass_branch.tank.total_fluid_mass)
        tanktemp.append(tank_bypass_branch.tank.tank_temp)
        outlettemp.append(tank_bypass_branch.tank.outlet_fluid_temp)
        soc.append(tank_bypass_branch.tank.state_of_charge)
        branchtemp.append(tank_bypass_branch.outlet_temp)
        tankflowfrac.append((1-tank_bypass_branch.bypass_fraction))
        if verbose:
            print("\t", value(blk.inlet_temp[i]), "\t", round(tank_bypass_branch.tank.tank_temp, 3), "\t", round(soc[-1], 3), "\t\t", tankflowfrac[-1], "\t\t", round(tank_bypass_branch.outlet_temp, 3), "\t\t", round(tank_bypass_branch.tank.outlet_fluid_temp, 3))
    #     print(b.tank.state_of_charge)

    return tanktemp, outlettemp, branchtemp, tankflowfrac, soc


def test_piecewise_heat_transfer_discharge_1C():
    """
    Starting TES at 1 C and Discharge the TES by increasing the Tank Temp as the Objective

    Inlet Temp at 20 C
    """
    data = copy.deepcopy(tank_data)
    data['initial_temperature'] = 1
    tank = IceTank(data)

    m, tank = full_model(tank, simple=False, calc_mode=False)

    # Fix Values to Setup Test
    m.env_temp[:].set_value(20. + 273.15)
    m.inlet_temp.fix(20 + 273.15)
    m.is_discharging[:].set_value(1)
    m.is_charging[:].set_value(0)

    init_mpc(m, tank, True, solver_name="couenne")

    m.obj = Objective(expr=-summation(m.tank_temp) * 1e5)
    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-5
    res = solver.solve(m, tee=True)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)

    # assert res.Solver.status == 'ok'
    expected_q_tot = [2.631, 2.609, 2.595]
    expected_inlet_temp = [20, 20, 20]
    expected_tank_temp = [1.099, 1.199, 1.297]
    expected_outlet_temp = [17.478, 17.492, 17.505]
    get_var_values(m, True)

    # Check that the MPC control variables are as expected
    for i in m.horiz_t:
        assert value(m.tank_temp[i]) == pytest.approx(expected_tank_temp[i] + 273.15, rel=1e-2)
        assert value(m.inlet_temp[i]) == pytest.approx(expected_inlet_temp[i] + 273.15, rel=1e-2)
        assert value(m.q_tot[i]) == pytest.approx(expected_q_tot[i], rel=1e-2)
        assert value(m.outlet_temp[i]) == pytest.approx(expected_outlet_temp[i] + 273.15, rel=1e-2)

    # Check that the TankBypassBranch simulated state variables are as expected
    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert expected_tank_temp[i] == pytest.approx(sim_tanktemp[i+1], rel=1e-2)
        assert expected_outlet_temp[i] == pytest.approx(sim_outlettemp[i], rel=1e-1)
        assert expected_outlet_temp[i] == pytest.approx(sim_branchtemp[i], rel=1e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], rel=1e-2)


def test_piecewise_heat_transfer_charge_1C():
    """
    Starting TES at 1 C and Charge the TES by decreasing the Tank Temp as the Objective
    """
    data = copy.deepcopy(tank_data)
    data['initial_temperature'] = 1
    tank = IceTank(data)

    m, tank = full_model(tank, simple=False, calc_mode=False)

    m.env_temp[:].set_value(20. + 273.15)
    m.is_discharging[:].set_value(0)
    m.is_charging[:].set_value(1)

    init_mpc(m, tank, True, solver_name="couenne")

    m.obj = Objective(expr=summation(m.tank_temp))
    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-6
    solver.options['bonmin.algorithm'] = 'B-BB'
    res = solver.solve(m, tee=True)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)

    print([value(m.tank_temp[i]) for i in m.horiz_t])
    print([value(m.q_tot[i]) for i in m.horiz_t])
    print([value(m.outlet_temp[i]) for i in m.horiz_t])

    expected_q_tot = [-11.695, -11.315, -10.937]
    expected_inlet_temp = [-12.789 + 273.15] * 3
    expected_tank_temp = [273.706, 273.277, 273.159]
    expected_outlet_temp = [271.458, 271.102, 271.089]

    get_var_values(m, True)
    for i in m.horiz_t:
        assert value(m.is_charging[i])
        assert value(m.tank_temp[i]) == pytest.approx(expected_tank_temp[i], rel=1e-3)
        assert value(m.inlet_temp[i]) == pytest.approx(expected_inlet_temp[i], rel=1e-3)
        assert value(m.q_tot[i]) == pytest.approx(expected_q_tot[i], rel=1e-3)
        assert value(m.outlet_temp[i]) == pytest.approx(expected_outlet_temp[i], rel=1e-3)

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert expected_tank_temp[i] == pytest.approx(sim_tanktemp[i+1] + 273.15, rel=1e-2)
        assert expected_outlet_temp[i] == pytest.approx(sim_outlettemp[i] + 273.15, rel=1e-2)
        assert expected_outlet_temp[i] == pytest.approx(sim_branchtemp[i] + 273.15, rel=1e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], rel=1e-2)


# assign results
def assign_results(m, n_hours_horiz, dt_hours):
    # t = list(m.horiz_t)
    # tt = list(m.horiz_t_init)
    # tt = t + [t[-1] + 1]
    t = np.arange(0, n_hours_horiz + dt_hours, dt_hours)
    tt = np.arange(0, n_hours_horiz + dt_hours*2, dt_hours)
    tank_temp = m.tank_temp[:]()
    outlet_temp = m.outlet_temp[:]()
    inlet_temp = [m.inlet_temp[i].value for i in m.horiz_t]
    avail_Q = [m.avail_Q[i].value for i in m.horiz_t_init]
    q_tot = m.q_tot[:]()
    mass_flow = m.mass_flow_rate[:]()
    tank_flow_fraction = [m.tank_flow_fraction[i].value for i in m.horiz_t]
    is_charging = [m.is_charging[i].value for i in m.horiz_t]
    is_discharging = [m.is_discharging[i].value for i in m.horiz_t]
    soc = m.soc[:]()
    return t, tt, tank_temp, avail_Q, q_tot, mass_flow, tank_flow_fraction, \
           outlet_temp, inlet_temp, is_charging, is_discharging, soc


# step plot with repeated last value
def stepp(ax, x, y, label='', color=None):
    return ax.step(x, y + [y[-1]], where='post', label=label, color=color)


def test_plot_max_discharge():
    '''
    Start tank at -10 C
    Discharge tank and maximize tank temperature
    Inlet temperature fixed at 20 C
    Manipulated input: flow rate (fix at maximum)
    Truncate bonmin after 5 s (not as important without obj)
    '''

    data = copy.deepcopy(tank_data)
    data['initial_temperature'] = -10
    tank = IceTank(data)
    # horizon = 6 h, dt = 15 min
    n_min_per_timestep = 15
    n_timesteps_per_hour = 60/n_min_per_timestep
    n_hours_horiz = 12
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n_timesteps_per_hour*n_hours_horiz, timestep=n_min_per_timestep*60, pw_repn = "incremental_constant_T")

    # Fix Values to Setup Test
    m.env_temp[:].set_value(20. + 273.15)
    m.inlet_temp.fix(20 + 273.15)
    m.is_discharging[:].set_value(1)
    m.is_charging[:].set_value(0)

    # initialize model
    init_mpc(m, tank, True, solver_name="couenne")

    # solve model
    m.obj = Objective(expr=-summation(m.tank_temp))
    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-5
    solver.options['bonmin.time_limit'] = 5
    res = solver.solve(m, tee=True)

    t, tt, tank_temp, avail_Q, q_tot, mass_flow, tank_flow_fraction, \
        outlet_temp, inlet_temp, is_charging, is_discharging, soc = assign_results(m, n_hours_horiz, dt_hours=n_min_per_timestep/60)

    # plot optimization results
    fig, axs = plt.subplots(4, 1, sharex=True)

    # plot tank, inlet and outlet temperatures
    ax = axs[0]
    ax.plot(tt, avail_Q)
    ax.plot(tt, np.ones_like(tt)*2741.232, 'k--', alpha=0.5)
    ax.plot(tt, np.zeros_like(tt), 'k--', alpha=0.5)
    ax.set_ylabel('tank energy [J]')
    ax_twin = ax.twinx()
    p = stepp(ax_twin, tt, q_tot, '', 'tab:green')
    ax_twin.set_ylabel(r'heat transferred' '\n' 'at timestep [J]', color=p[0].get_color())
    # ax.set_title('total heat transfer rate (J/s)')
    ax.set_title('tank energy and heat transferred at timestep [J]')

    # plot tank, inlet and outlet temperatures
    ax = axs[1]
    ax.plot(tt, [T-273.15 for T in tank_temp], label='T_tank')
    stepp(ax, tt, [T-273.15 for T in inlet_temp], label='T_in')
    stepp(ax, tt, [T-273.15 for T in outlet_temp], label='T_out')
    ax.plot(tt, np.ones_like(tt)*20, 'k--', alpha=0.2)
    ax.plot(tt, np.ones_like(tt)*-10, 'k--', alpha=0.2)
    ax.set_title('tank, inlet, outlet temperature [°C]')
    ax.legend(framealpha=0.3)

    # plot operating mode
    ax = axs[2]
    stepp(ax, tt, is_charging, label='is_charging')
    stepp(ax, tt, is_discharging, label='is_discharging')
    ax.set_title('operation mode')
    ax.legend(framealpha=0.3)

    # plot flow rate
    ax = axs[3]
    stepp(ax, tt, mass_flow, label='tank mass flow rate')
    ax.plot(tt, np.zeros_like(tt), '--k', alpha=0.2)
    ax.plot(tt, np.ones_like(tt)*test_mass_flow_rate, '--k', alpha=0.2)
    ax.set_title('tank flow rate [kg/s]')
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax.legend()

    # wrap up plot
    axs[-1].set_xlabel(f'hours (dt = {n_min_per_timestep:n} min)')
    plt.tight_layout()
    plt.show()

    # simulation results
    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, False)

    # plot optimization results vs. simulation results
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle('Comparing optimization vs. simulation results')
    
    ax = axs[0]
    ax.plot(tt, [T-273.15 for T in tank_temp], label='opt')
    ax.plot(tt, sim_tanktemp, label='sim')
    ax.set_title('tank temperature (K)')
    ax.legend()
    
    ax = axs[1]
    stepp(ax, tt, [T-273.15 for T in outlet_temp], label='opt')
    stepp(ax, tt, sim_outlettemp, label='sim (T_out)')
    stepp(ax, tt, sim_branchtemp, label='sim (T_branch)')
    ax.set_title('outlet temperature (K)')
    ax.legend()    
    
    ax = axs[2]
    ax.plot(tt, soc, label='opt')
    ax.plot(tt, sim_soc, label='sim')
    ax.set_title('state of charge')
    ax.legend()    
    
    axs[-1].set_xlabel(f'hours (dt = {n_min_per_timestep:n} min)')
    plt.tight_layout()
    plt.show()


def test_min_discharge_temp_SOC_100():
    """
    Starting TES at 0 C with Full SOC (all Ice) and Discharge the TES by minimizing the Upper Bound of the Tank Outlet Temp

    Inlet Temp at 20 C
    """
    data = copy.deepcopy(tank_data)
    data.pop("initial_temperature")
    data['latent_state_of_charge'] = 1
    tank = IceTank(data)

    n = 5
    m, tank = full_model(tank, simple=False, calc_mode=True, n=n, pw_repn='SOS2')

    for i in m.horiz_t:
        m.env_temp[i].set_value(40. + 273.15)
        m.inlet_temp[i].fix(20 + 273.15)
        m.is_charging[i].fix(0)

    init_mpc(m, tank, solver_name="couenne")

    m.min_outlet_temp = Var(domain=NonNegativeReals)
    m.same_outlet_temp = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] <= m.min_outlet_temp)
    m.obj = Objective(expr=m.min_outlet_temp * 1e6)

    solver = SolverFactory("bonmin")
    res = solver.solve(m, tee=True)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
    
    print([value(m.tank_temp[i]) for i in m.horiz_t])
    print([value(m.q_tot[i]) for i in m.horiz_t])
    print([value(m.outlet_temp[i]) for i in m.horiz_t])

    get_var_values(m, True)

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert value(m.tank_temp[i]) == pytest.approx(sim_tanktemp[i+1] + 273.15, abs=1e-2)
        assert value(m.outlet_settemp[i]) == pytest.approx(sim_branchtemp[i] + 273.15, rel=1e-2)
        if value(m.outlet_settemp[i] - m.inlet_temp[i]) != 0:
            assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], rel=5e-2)


def test_min_discharge_SOC_20():
    """
    Starting TES at 20 C with 20% SOC and Discharge the TES by minimizing the Upper Bound of the Tank Outlet Temp

    Inlet Temp at 20 C
    """
    data = copy.deepcopy(tank_data)
    data.pop("initial_temperature")
    data['latent_state_of_charge'] = 0.20
    tank = IceTank(data)

    n = 5
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n)

    for i in m.horiz_t:
        m.env_temp[i].set_value(40. + 273.15)
        m.inlet_temp[i].fix(20 + 273.15)
        m.is_discharging[i].set_value(1)
        m.is_charging[i].set_value(0)
    
    init_mpc(m, tank, solver_name="couenne")
    
    m.max_outlet_temp = Var(domain=NonNegativeReals)
    m.same_outlet_temp = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] <= m.max_outlet_temp)
    m.obj = Objective(expr=m.max_outlet_temp * 1e6)

    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-6
    # solver.options['bonmin.algorithm'] = 'B-OA'
    solver.solve(m, tee=True)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1)

    print([value(m.tank_temp[i]) for i in m.horiz_t])
    print([value(m.q_tot[i]) for i in m.horiz_t])
    print([value(m.outlet_temp[i]) for i in m.horiz_t])

    get_var_values(m, True)

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert value(m.tank_temp[i]) == pytest.approx(sim_tanktemp[i+1] + 273.15, abs=1e-2)
        assert value(m.outlet_settemp[i]) == pytest.approx(sim_branchtemp[i] + 273.15, rel=1e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], abs=5e-2)


def test_min_discharge_temp_SOC_0():
    
    data = copy.deepcopy(tank_data)
    data.pop("initial_temperature")
    data['latent_state_of_charge'] = 0
    tank = IceTank(data)

    n = 5
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n)

    for i in m.horiz_t:
        m.env_temp[i].set_value(40. + 273.15)
        m.inlet_temp[i].fix(20 + 273.15)
        m.is_discharging[i].set_value(1)
        m.is_charging[i].set_value(0)
    
    init_mpc(m, tank, True)
    
    m.min_outlet_temp = Var(domain=NonNegativeReals)
    m.same_outlet_temp = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] <= m.min_outlet_temp)
    m.obj = Objective(expr=m.min_outlet_temp * 1e6)

    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-6

    res = solver.solve(m, tee=True)
    assert value(m.min_outlet_temp) - 273.15 == pytest.approx(17.438, rel=1e-1)

    for i in range(0, len(m.horiz_t)):
        assert value(m.outlet_settemp[i]) == pytest.approx(value(m.min_outlet_temp), rel=1e-1)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1)
    get_var_values(m, True)

    tanktemp, outlettemp, branchtemp, tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert value(m.tank_temp[i]) == pytest.approx(tanktemp[i] + 273.15, abs=5e-1)
        assert value(m.outlet_settemp[i]) == pytest.approx(branchtemp[i] + 273.15, rel=5e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(tankflowfrac[i], abs=6e-2)


def test_min_charge_temp_SOC_100():
    data = copy.deepcopy(tank_data)
    data.pop("initial_temperature")
    data['latent_state_of_charge'] = 100
    tank = IceTank(data)

    n = 50
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n, pw_repn="incremental")

    for i in m.horiz_t:
        m.env_temp[i].set_value(40. + 273.15)
        m.inlet_temp[i].fix(-10 + 273.15)
        m.is_discharging[i].set_value(0)
        m.is_charging[i].set_value(1)
        m.tank_flow_fraction[i].fix(1)

    start = default_timer()

    init_mpc(m, tank, False)
    
    m.max_outlet_temp = Var(domain=NonNegativeReals)
    m.same_outlet_temp = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] >= m.max_outlet_temp)
    m.obj = Objective(expr=m.max_outlet_temp * 1e1, sense=-1)

    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-9

    res = solver.solve(m, tee=False)
    timed = default_timer() - start
    # assert value(m.max_outlet_temp) - 273.15 >= -5.42

    for i in range(0, len(m.horiz_t)):
        assert value(m.outlet_settemp[i]) == pytest.approx(value(m.max_outlet_temp), rel=1e-1)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1)
    get_var_values(m, True)

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert value(m.tank_temp[i]) == pytest.approx(sim_tanktemp[i+1] + 273.15, abs=25e-2)
        assert value(m.outlet_settemp[i]) == pytest.approx(sim_branchtemp[i] + 273.15, rel=5e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], rel=5e-2)
    
    print(timed)


def test_min_charge_temp_SOC_0():
    data = copy.deepcopy(tank_data)
    data.pop("initial_temperature")
    data['latent_state_of_charge'] = 0
    tank = IceTank(data)

    n = 5
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n, pw_repn="incremental")

    for i in m.horiz_t:
        m.env_temp[i].set_value(40. + 273.15)
        m.inlet_temp[i].fix(-10 + 273.15)
        m.is_discharging[i].set_value(0)
        m.is_charging[i].set_value(1)
        m.tank_flow_fraction[i].fix(1)

    start = default_timer()

    init_mpc(m, tank, False)
    
    m.max_outlet_temp = Var(domain=NonNegativeReals)
    m.same_outlet_temp = Constraint(m.horiz_t, rule=lambda m, i: m.outlet_settemp[i] >= m.max_outlet_temp)
    m.obj = Objective(expr=m.max_outlet_temp * 1e1, sense=-1)

    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-9

    res = solver.solve(m, tee=False)
    timed = default_timer() - start
    assert value(m.max_outlet_temp) - 273.15 >= -1.74

    for i in range(0, len(m.horiz_t)):
        assert value(m.outlet_settemp[i]) == pytest.approx(value(m.max_outlet_temp), rel=1e-1)

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1)
    get_var_values(m, True)

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    for i in range(0, len(m.horiz_t)):
        assert value(m.tank_temp[i]) == pytest.approx(sim_tanktemp[i+1] + 273.15, abs=25e-2)
        assert value(m.outlet_settemp[i]) == pytest.approx(sim_branchtemp[i] + 273.15, rel=5e-2)
        assert value(m.tank_flow_fraction[i]) == pytest.approx(sim_tankflowfrac[i], rel=5e-2)
    
    print(timed)


def test_load_init():
    """
    """
    data = copy.deepcopy(tank_data)
    data['initial_temperature'] = 15
    tank = IceTank(data)
    n = 60
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n-1)

    with open(thermal_tank_log_dir / "1tanks" / f"solution_1440.json", 'r') as f:
        solution = json.load(f)
        solution = solution[str(0)]
    for cuid, val in solution.items():
        if cuid == "diff_sum":
            continue
        try:
            m.find_component(cuid).value = val
        except:
            pass

    init_temp(m, 15, 0, -1)

    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-6
    m.obj = Objective(expr=0)
    res = solver.solve(m, tee=True)
    m.del_component('obj')

    # m.outlet_settemp[-1].fix()
    # m.outlet_settemp[1339].fix()
    tank_temp_0 = [value(m.tank_temp[i]) for i in m.horiz_t]
    soc_0 = [value(m.soc[i]) for i in m.horiz_t]
    outlet_settemp_0 = [value(m.outlet_settemp[i]) for i in m.horiz_t]
    print(np.average(outlet_settemp_0))
    m.obj = Objective(expr=sum([(outlet_settemp_0[i] - m.outlet_settemp[i])**2 for i in m.horiz_t])
                            + sum([(tank_temp_0[i] - m.tank_temp[i])**2 for i in m.horiz_t])
                            + sum((soc_0[i] - m.soc[i])**2 for i in m.horiz_t))

    res = solver.solve(m, tee=True)
    tank_temp_1 = [value(m.tank_temp[i]) for i in m.horiz_t]
    soc_1 = [value(m.soc[i]) for i in m.horiz_t]
    outlet_settemp_1 = [value(m.outlet_settemp[i]) for i in m.horiz_t]

    assert sum(abs(tank_temp_0[i] - tank_temp_1[i]) for i in m.horiz_t) < 5
    assert sum(abs(soc_0[i] - soc_1[i]) for i in m.horiz_t) < 1e-3
    assert sum(abs(outlet_settemp_0[i] - outlet_settemp_1[i]) for i in m.horiz_t) < 1


def test_building_loop_model():
    data = copy.deepcopy(tank_data)
    data['initial_temperature'] = 15
    tank = IceTank(data)
    n = 10
    m, tank = full_model(tank, simple=False, calc_mode=False, n=n-1)
    with open(thermal_tank_log_dir / "1tanks" / f"solution_1440.json", 'r') as f:
        solution = json.load(f)
        solution = solution[str(0)]
    for cuid, val in solution.items():
        if cuid == "diff_sum":
            continue
        try:
            m.find_component(cuid).value = val
        except:
            pass
    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-6
    m.obj = Objective(expr=0)

    init_temp(m, tank_temp_C=15, latent_soc=0, index=-1)
    
    res = solver.solve(m, tee=True)
    
    building_loop_model(m)
    init_building(m, 0)

    res = solver.solve(m, tee=True)
    m.del_component('obj')

    m.tank_temp_b1.fix()
    m.tank_temp_b2.fix()
    tank_temp_0 = [value(m.tank_temp[i]) for i in m.horiz_t]
    soc_0 = [value(m.soc[i]) for i in m.horiz_t]
    outlet_temp_0 = [value(m.outlet_settemp[i]) for i in m.horiz_t]
    evap_diff_0 = [value(m.evap_diff[i]) for i in m.horiz_t]
    m.obj = Objective(expr=sum(m.chiller_power[i] for i in m.horiz_t))

    solver_start = default_timer()
    solver = SolverFactory("ipopt")
    solver.options['tol'] = 1e-6
    # solver.options['bonmin.time_limit'] = 60
    # solver.options['bonmin.iteration_limit'] = 200
    # solver.options['bonmin.solution_limit'] = 1
    # solver.options['max_iter'] = 5000

    res = solver.solve(m, tee=True)
    print([value(m.chiller_power[i]) for i in m.horiz_t])
    print(res)

    tank_temp_1 = [value(m.tank_temp[i]) for i in m.horiz_t]
    outlet_temp_1 = [value(m.outlet_settemp[i]) for i in m.horiz_t]
    soc_1 = [value(m.soc[i]) for i in m.horiz_t]
    evap_diff_1 = [value(m.evap_diff[i]) for i in m.horiz_t]

    fig, ax = plt.subplots(3, 1, figsize=(18,9))
    ax[0].plot(outlet_temp_0, label="prev")
    ax[0].plot(outlet_temp_1, label="after")
    ax[0].set_title("outlet temp")
    ax[0].legend()
    ax[1].plot(soc_0, label="prev")
    ax[1].plot(soc_1, label="after")
    ax[1].set_title("soc")
    ax[1].legend()
    ax[2].plot(evap_diff_0, label="prev")
    ax[2].plot(evap_diff_1, label="after")
    ax[2].set_title("evap diff")
    ax[2].legend()
    plt.show()
    print("time", default_timer() - solver_start)


binary_vars = ['tank_temp_b1', 'tank_temp_b2', 'is_charging']
def check_binary_variables(m, n):
    for i in range(n + 1):
        for var in binary_vars:
            val = value(getattr(m.block[i], var)[0])
            if abs(val - 0) > 1e-3 and abs(val - 1) > 1e-3:
                print(getattr(m.block[i], var), val)
                return False
    return True

def test_hourly_opt_separate():
    # load the initial state variables of the TES and chiller from simulated results w/ TES-schedule-based operation
    mins_per_ts = 15
    seconds_per_ts = mins_per_ts * 60
    df_states = pd.read_parquet(thermal_tank_log_dir / "1tanks" / "chw_states_minute_day.parquet")
    df_states = df_states.set_index(pd.to_datetime(df_states['DateTime'])).resample(f'{mins_per_ts}min').mean()
    pv_output = pd.read_parquet(thermal_tank_log_dir / ".." / "pv.parquet")["pv_kw"].values
    pv_size_kw = 150
    bldg_load = np.copy(day_avg_bldg_load)
    bldg_load -= pv_output * pv_size_kw
    peak_tou_mult = 20
    binary_enforce = 1e1

    n = 96
    start_ts = 0

    # construct the tank simulator
    data = copy.deepcopy(tank_data)
    data.pop('initial_temperature')
    data['latent_state_of_charge'] = 0.5
    tank = IceTank(data)

    # construct MPC
    m = ConcreteModel()
    m.time = RangeSet(0, n)

    def block_rule(b, t, tank, timestep):
        full_model(tank, existing_model=b, simple=True, calc_mode=True, n=0, timestep=timestep)
        building_loop_model(b)
        b.bldg_elec_load = Expression(RangeSet(0, 0), rule=lambda m, i: b.chiller_power[0] + bldg_load[t % len(bldg_load)])

    m.block = pyo.Block(m.time, rule=partial(block_rule, tank=tank, timestep=seconds_per_ts))
    
    # initialize the TES and chiller's temp at time = -1 and load up the initial state variables into MPC
    prev_outlet_temp = -2.5 + 273.15
    prev_temp = 0
    prev_soc = data['latent_state_of_charge']

    solver_name = "bonmin"
    bonmin_solver = SolverFactory(solver_name)
    bonmin_solver.options['tol'] = 1e-5
    bonmin_solver.options['max_iter'] = 1500
    bonmin_solver.options['bonmin.time_limit'] = 60 * 4
    bonmin_solver.options['bonmin.iteration_limit'] = 10000

    ipopt_solver = SolverFactory("ipopt")
    ipopt_solver.options['tol'] = 1e-5
    ipopt_solver.options['max_iter'] = 1500

    ts_soc = [prev_soc]
    ts_chiller_outtemp = [prev_outlet_temp - 273.15]
    ts_tank_outlet_temp = [prev_outlet_temp - 273.15]
    ts_tank_temp = [prev_temp]

    for i, (time, row) in enumerate(df_states.iterrows()):
        if i > n:
            break
        b = m.block[i]

        # carry over previous state
        init_temp(b, tank_temp_C=prev_temp, latent_soc=prev_soc, index=-1)
        b.outlet_settemp[-1].fix(prev_outlet_temp)

        # duplicate previous state for current state as initial guess
        init_temp(b, tank_temp_C=prev_temp, latent_soc=prev_soc, index=0, fixed=False)
        b.outlet_settemp[0] = prev_outlet_temp

        # set setpoints
        b.cond_inlet[0].set_value(row['cond_inlet'] + 273.15)
        b.env_temp[0].set_value(pv_output[i % len(pv_output)] * 20 + 273.15)
        chiller_setpoint, loop_setpoint, mode = coned_schedule(time)
        b.inlet_temp[0].setub(chiller_setpoint + 273.15)
        b.outlet_settemp[0].setub(loop_setpoint + 273.15)

        # init inlet temp
        ind_15min = time.hour * int(60 / mins_per_ts) + int(time.minute / mins_per_ts)
        evap_diff_prev = day_avg_evap_diff[(ind_15min - 1) % len(day_avg_evap_diff)]
        b.evap_diff[0].set_value(evap_diff_prev * 1.5)

        # if mode = 1 = charging, then increase SOC
        # b.obj = Objective(expr=b.soc[0] * 1e2 * -mode + b.chiller_power[0])
        b.obj = Objective(expr=(b.outlet_settemp[0] - (loop_setpoint + 273.15)) ** 2 
                                + (b.inlet_temp[0] - (chiller_setpoint + 273.15)) ** 2
                                + b.chiller_power[0] * 1e-3
                                )
        # b.obj = Objective(expr=(b.inlet_temp[0] - (chiller_setpoint + 273.15)) ** 2 + b.chiller_power[0])
        res = ipopt_solver.solve(b, tee=False)
        res = bonmin_solver.solve(b, tee=False)
        b.del_component('obj')

        solved = res.Solver.status == 'ok'
        if not solved:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
            log_infeasible_constraints(b, logger=solve_log, tol=1e-3, log_expression=True, log_variables=True)
            log_infeasible_bounds(b, logger=solve_log, tol=1e-4)

            assert False

        if not check_binary_variables(m, 0):
            raise RuntimeError

        prev_outlet_temp = value(b.outlet_settemp[0])
        prev_temp = value(b.tank_temp[0] - 273.15)
        prev_soc = value(b.soc[0])

        ts_soc.append(prev_soc)
        ts_tank_outlet_temp.append(prev_outlet_temp - 273.15)
        ts_tank_temp.append(prev_temp)
        ts_chiller_outtemp.append(value(b.inlet_temp[0]) - 273.15)
        stop = 0

        # sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(b, data, True)
        # data['initial_temperature'] = prev_temp
        # data['latent_state_of_charge'] = prev_soc

        # free the variables and add cross-period constraints
        b.inlet_temp[0].unfix()
        b.inlet_temp[0].setub(20 + 273.15)
        b.outlet_settemp[-1].unfix()
        b.outlet_settemp[0].setub(20 + 273.15)
        b.tank_temp_d1[-1].unfix()
        b.tank_temp_d2[-1].unfix()
        b.tank_temp_d3[-1].unfix()

    df_new = save_vars_to_df(m, df_states.index[start_ts:start_ts + n + 1], f"optimized_{bonmin_solver.options['bonmin.iteration_limit']}")
    df_new.to_csv(Path(__file__).parent / "init_res.csv")

    if abs(value(m.block[0].tank_temp[-1]) - 273.15) < phase_change_e:
        if 'initial_temperature' in data.keys():
            data.pop('initial_temperature')
        data['latent_state_of_charge'] = value(m.block[0].soc[-1])
    else:
        data['initial_temperature'] = value(m.block[0].tank_temp[-1]) - 273.15

    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)

    fig, ax = plt.subplots(4, 1, figsize=(12, 5), sharex=True)

    ax[0].plot([value(m.block[i].outlet_temp[0] - m.block[i].inlet_temp[0]) for i in m.time], label="mpc")
    ax[0].set_title("Delta Temp Tank")
    ax[0].legend()

    ax[1].plot([value(m.block[i].soc[0]) for i in m.time], label="mpc")
    ax[1].plot(sim_soc[1:], label="sim", linestyle='--')
    ax[1].set_title("SOC")
    ax[1].legend()

    ax[2].plot([value(m.block[i].outlet_temp[0]) - 273.15 for i in m.time], label="mpc")
    ax[2].plot(sim_outlettemp, label="sim", linestyle='--')
    ax[2].set_title("Tank Outlet Temp")
    ax[2].legend()

    ax[3].plot([value(m.block[i].outlet_settemp[0]) - 273.15 for i in m.time], label="mpc")
    ax[3].plot(sim_branchtemp, label="sim", linestyle='--')
    ax[3].set_title("CHW Outlet Temp")
    ax[3].legend()
    plt.show()

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

    b = m.block[0]
    # b.tank_temp_d1[-1].fix()
    # b.tank_temp_d2[-1].fix()
    # b.tank_temp_d3[-1].fix()
    # b.tank_temp_b1[-1].fix()
    # b.tank_temp_b2[-1].fix()

    fig, ax = plt.subplots(3, 1, figsize=(12,4), sharex=True)
    ax[0].plot(ts_soc, label="old")
    ts_chiller_p = [value(m.block[i].chiller_power[0]) for i in m.time]
    ax[1].plot(ts_chiller_p, label="old chiller power")
    ts_bldg_p = [value(m.block[i].bldg_elec_load[0]) for i in m.time]
    ax[1].plot(ts_bldg_p, label="old bldg elec power")
    ts_chiller_out = [value(m.block[i].inlet_temp[0]) for i in m.time]
    ax[2].plot(ts_chiller_out, label="old")
    for a in ax:
        a.legend()
    plt.savefig(f"test_hourly_opt_separate_{pv_size_kw}_pv_{peak_tou_mult}_peak.png")

    old_max_load = max(value(m.block[i].bldg_elec_load[0]) for i in m.time)
    df_old = save_vars_to_df(m, df_states.index[start_ts:start_ts + n + 1], "schedule")

    # minimize the max load while making sure the outlet temp doesn't get too warm
    m.max_outlet_temp = Constraint(m.time, rule=lambda m, i: m.block[i].outlet_settemp[0] <= 8 + 273.15)
    # m.peak = Var(domain=NonNegativeReals, initialize=max(value(m.block[i].bldg_elec_load[0]) for i in m.time))
    m.peak = Var(domain=NonNegativeReals, initialize=max(value(m.block[i].bldg_elec_load[0]) for i in m.time))
    m.peak_demand = Constraint(m.time, rule=lambda m, i: m.peak >= m.block[i].bldg_elec_load[0])

    m.peak_time = RangeSet(13 * 4, 19 * 4)
    m.peak_tou = Var(domain=NonNegativeReals, initialize=max(value(m.block[i].bldg_elec_load[0]) for i in m.peak_time))
    m.peak_tou_demand = Constraint(m.peak_time, rule=lambda m, i: m.peak_tou >= m.block[i].bldg_elec_load[0])

    def deriv_reg_2nd(m, i):
        if i == 0:
            return (m.block[0].outlet_temp[0] - 2 * m.block[n].outlet_temp[0] + m.block[n-1].outlet_temp[0]) ** 2
        elif i == 1:
            return (m.block[1].outlet_temp[0] - 2 * m.block[0].outlet_temp[0] + m.block[n].outlet_temp[0]) ** 2
        return (m.block[i].outlet_temp[0] - 2 * m.block[i-1].outlet_temp[0] + m.block[i-2].outlet_temp[0]) ** 2

    def deriv_reg_1st(m, i):
        if i == 0:
            return (m.block[n].outlet_temp[0] - m.block[0].outlet_temp[0]) ** 2
        return (m.block[i].outlet_temp[0] - m.block[i-1].outlet_temp[0]) ** 2
        
    m.deriv_reg = Expression(m.time, rule=lambda m, i: deriv_reg_1st(m, i) + deriv_reg_2nd(m, i))

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1e-1, log_expression=True, log_variables=True)

    def binary_close(m, i):
        return sum(-4 * (getattr(m.block[i], var)[0] - 0.5 )**2 + 1 for var in binary_vars)

    m.binary_close = Expression(m.time, rule=binary_close)

    # peak 
    m.obj = Objective(expr=m.peak * 1 + m.peak_tou * peak_tou_mult 
                            # + summation(m.deriv_reg) * 1e-2
                            + summation(m.binary_close) * binary_enforce)
    res = ipopt_solver.solve(m, tee=True)
    m.del_component('obj')
    m.obj = Objective(expr=m.peak * 1 + m.peak_tou * peak_tou_mult 
                            + summation(m.deriv_reg) * 1e-2
                            + summation(m.binary_close) * binary_enforce * 10)
    res = ipopt_solver.solve(m, tee=True)
    
    check_binary_variables(m, n)

    bonmin_solver = SolverFactory(solver_name)
    bonmin_solver.options['tol'] = 1e-5
    bonmin_solver.options['max_iter'] = 5000 * 3
    bonmin_solver.options['bonmin.time_limit'] = 60 * 20
    bonmin_solver.options['bonmin.iteration_limit'] = 10000 * 4

    # m.obj = Objective(expr=m.peak * 1 + m.peak_tou * peak_tou_mult + summation(m.deriv_reg) * binary_enforce)
    start = default_timer()
    res = bonmin_solver.solve(m, tee=True)
    print("total solve time: ", default_timer() - start)

    if res.Solver.Status != 'ok':
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-2, log_expression=True, log_variables=True)

    res = check_binary_variables(m, n)

    if not res:
        print("Binary failure")

    ts_soc = [value(m.block[i].soc[0]) for i in m.time]
    ax[0].plot(ts_soc, label="new")
    ax[0].set_title("soc")
    ts_chiller_p = [value(m.block[i].chiller_power[0]) for i in m.time]
    ax[1].plot(ts_chiller_p, label="new chiller power")
    ax[1].set_title("chiller_power and bldg_elec_load")
    ts_bldg_p = [value(m.block[i].bldg_elec_load[0]) for i in m.time]
    ax[1].plot(ts_bldg_p, label="new bldg elec power")
    ts_chiller_out = [value(m.block[i].inlet_temp[0]) for i in m.time]
    ax[2].plot(ts_chiller_out, label="new")
    ax[2].set_title("tank inlet_temp")

    for a in ax:
        a.legend()
    plt.savefig(f"test_hourly_opt_separate_{pv_size_kw}_pv_{peak_tou_mult}_peak.png")
    plt.show()

    new_max_load = max(value(m.block[i].bldg_elec_load[0]) for i in m.time)
    df_new = save_vars_to_df(m, df_states.index[start_ts:start_ts + n + 1], f"optimized_{bonmin_solver.options['bonmin.iteration_limit']}")

    df_params = save_params_to_df(m, df_new.index)

    df = pd.concat([df_old, df_new])
    df.to_parquet(f"test_hourly_opt_separate_{pv_size_kw}_pv_{peak_tou_mult}_peak.parquet")

    # assert new_max_load < old_max_load
    # sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, True)
    print(value(m.peak), value(m.peak_tou), value(m.peak + m.peak_tou * peak_tou_mult))

    # try running in TankBypassBranch
    # update tank data
    if abs(value(m.block[0].tank_temp[-1]) - 273.15) < phase_change_e:
        if 'initial_temperature' in data.keys():
            data.pop('initial_temperature')
        data['latent_state_of_charge'] = value(m.block[0].soc[-1])
    else:
        data['initial_temperature'] = value(m.block[0].tank_temp[-1]) - 273.15


    sim_tanktemp, sim_outlettemp, sim_branchtemp, sim_tankflowfrac, sim_soc = run_tank_model(m, data, False)

    fig, ax = plt.subplots(4, 1, figsize=(12, 5), sharex=True)

    ax[0].plot([value(m.block[i].outlet_temp[0] - m.block[i].inlet_temp[0]) for i in m.time], label="mpc")
    ax[0].set_title("Delta Temp Tank")
    ax[0].legend()

    ax[1].plot([value(m.block[i].soc[0]) for i in m.time], label="mpc")
    ax[1].plot(sim_soc[1:], label="sim", linestyle='--')
    ax[1].set_title("SOC")
    ax[1].legend()

    ax[2].plot([value(m.block[i].outlet_temp[0]) - 273.15 for i in m.time], label="mpc")
    ax[2].plot(sim_outlettemp, label="sim", linestyle="--")
    ax[2].set_title("Tank Outlet Temp")
    ax[2].legend()

    ax[3].plot([value(m.block[i].outlet_settemp[0]) - 273.15 for i in m.time], label="mpc")
    ax[3].plot(sim_branchtemp, label="sim", linestyle="--")
    ax[3].set_title("CHW Outlet Temp")
    ax[3].legend()
    plt.show()

    df_sim = pd.DataFrame(index=df_new.index)
    df_sim['run'] = 'simulate'
    df_sim['tank_temp'] = np.array(sim_tanktemp[1:]) + 273.15
    df_sim['outlet_temp'] = np.array(sim_outlettemp) + 273.15
    df_sim['loop_outlet_temp'] = np.array(sim_branchtemp) + 273.15
    df_sim['tank_flow_fraction'] = sim_tankflowfrac
    df_sim['soc'] = sim_soc[1:]
    df_sim['inlet_temp'] = [value(m.block[i].inlet_temp[0]) for i in m.time]
    df_sim['tank_delta_T'] = df_sim['outlet_temp'] - df_sim['inlet_temp']

    df = pd.concat([df_old, df_new, df_sim])
    df.to_parquet(f"test_hourly_opt_separate_{pv_size_kw}_pv_{peak_tou_mult}_peak.parquet")




# test_hourly_opt_separate()

# def test_time_averaging():


# def test_compare_simulation_timesteps():
    # solve optimization with modulating inlet flow or temp (charge and discharge the tank)
    # simulate the tank with increasing timesteps: 1 min, 3 min, 5 min, 15 min
    # compare results (tank energy, temperature and soc, stream outlet temperature)



# def test_timeofuse_pricing():

#     '''inputs (average E+ variables for every 15-min)'''

#     '''create and solve optimization model'''

#     '''simulate tank with same manipulated variables (use '''

#     '''plot comparison'''

