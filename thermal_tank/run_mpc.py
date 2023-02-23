import copy
from functools import partial
from pyomo.environ import Block
from pyomo.contrib.fbbt.fbbt import fbbt
from .mpc import *
from .eplus_timeseries import make_chiller_schedule

KW_SCALING = 1e-2

def make_model(df, start_ts, peak_tou_mult, mins_per_ts=15, run_fbbt=True):
    evap_T_diff = df['evap_T_diff'].values
    chw_outlet_temp = df['chw_outlet_temp'].values
    chiller_power = df['chiller_power'].values
    chiller_delta_T = df['chiller_delta_T'].values
    pv_gen = df['pv_gen'].values
    load = df['load'].values
    steps_per_hour = int(60 / mins_per_ts)
    seconds_per_ts = mins_per_ts * 60
    n = int(24 * (60 / mins_per_ts))

    time_slice = slice(start_ts, start_ts + n)
    pv_output = pv_gen[time_slice]
    pv_size_kw = 0
    bldg_load = np.copy(load)[time_slice]
    evap_diff = np.copy(evap_T_diff)[time_slice]
    bldg_load -= pv_output * pv_size_kw

    # construct the tank simulator
    data = copy.deepcopy(tank_data)
    data.pop('initial_temperature')
    data['latent_state_of_charge'] = 0.5
    tank = IceTank(data)

    m = ConcreteModel()
    m.time = RangeSet(0, n)

    construct_tes(m, m.time, tank, start_ts, mins_per_ts, KW_SCALING,
                  bldg_load, chw_outlet_temp)

    # m.peak = Var(domain=NonNegativeReals, initialize=max(value(m.block[i].bldg_elec_load[0]) for i in m.time))
    m.peak = Var(domain=NonNegativeReals, initialize=max(bldg_load))
    m.peak_demand = Constraint(m.time, rule=lambda m, i: m.peak >= m.block[i].bldg_elec_load[0])

    m.peak_time = RangeSet(13 * int(60/mins_per_ts), 19 * int(60/mins_per_ts))
    m.peak_tou = Var(domain=NonNegativeReals, initialize=max(bldg_load[i] for i in m.peak_time))
    m.peak_tou_demand = Constraint(m.peak_time, rule=lambda m, i: m.peak_tou >= m.block[i].bldg_elec_load[0])
    m.peak_tou_mult = Param(domain=NonNegativeReals, default=peak_tou_mult, mutable=True)

    if run_fbbt:
        fbbt(m)
        # fix bounds for tank_temp_d2
        for i in m.time:
            m.block[i].tank_temp_d2[-1].setlb(0)
            m.block[i].tank_temp_d2[0].setlb(0)

    orig_set = {"chiller_p": chiller_power[time_slice],
                "bldg_p": chiller_power[time_slice] + load[time_slice],
                "chiller_out": chiller_delta_T[time_slice],
                "chiller_delta_T": chiller_delta_T[time_slice],
                "chw_outlet_temp": chw_outlet_temp[time_slice] - 273.15}

    return m, orig_set

def calc_peak_obj(m):
    peak = round(max(value(m.block[i].bldg_elec_load[0]) for i in m.time), 2)
    peak_tou = round(max(value(m.block[i].bldg_elec_load[0]) for i in m.peak_time), 2)
    return peak, peak_tou, peak + peak_tou * value(m.peak_tou_mult)

def get_data(m):
    ts_chiller_p = [value(m.block[i].chiller_power[0]) * KW_SCALING for i in m.time]
    ts_bldg_p = [value(m.block[i].bldg_elec_load[0]) for i in m.time]
    ts_chiller_out = [value(m.block[i].inlet_temp[0]) for i in m.time]
    ts_chiller_delta_T = [value(m.block[i].chiller_delta_T[0]) for i in m.time]
    ts_chw_outlet_T = [value(m.block[i].outlet_settemp[0]) - 273.15 for i in m.time]
    ts_soc = [value(m.block[i].soc[0]) for i in m.time]

    data_set = {"chiller_p": ts_chiller_p,
                "bldg_p": ts_bldg_p,
                "chiller_out": ts_chiller_out,
                "chiller_delta_T": ts_chiller_delta_T,
                "chw_outlet_temp": ts_chw_outlet_T,
                "soc": ts_soc}
    return data_set

def init_model(df, m, start_ts, prev_outlet_temp, obj_fx, socs=None):
    chw_outlet_temp = df['chw_outlet_temp']
    cond_inlet_K = df['cond_inlet_K']
    evap_T_diff = df['evap_T_diff']
    chiller_power = df['chiller_power']

    prev_outlet_temp = min(m.block[0].outlet_settemp[-1].ub, prev_outlet_temp)
    prev_outlet_temp = max(m.block[0].outlet_settemp[-1].lb, prev_outlet_temp)
    prev_temp = 0
    prev_soc = 0.5

    ts_soc = [prev_soc]
    ts_chiller_outtemp = [prev_outlet_temp - 273.15]
    ts_tank_outlet_temp = [prev_outlet_temp - 273.15]
    ts_tank_temp = [prev_temp]

    chiller_schedule = make_chiller_schedule(1 / int(len(m.time) / 24))

    for i in m.time:
        b = m.block[i]
        t = start_ts + i

        # carry over previous state
        init_temp(b, tank_temp_C=prev_temp, latent_soc=prev_soc, index=-1)

        b.outlet_settemp[-1].fix(prev_outlet_temp)

        # duplicate previous state for current state as initial guess
        target_soc = prev_soc
        if socs is not None:
            target_soc = socs[i]
        init_temp(b, tank_temp_C=prev_temp, latent_soc=target_soc, index=0, fixed=False)

        b.outlet_settemp[0] = prev_outlet_temp

        # set setpoints
        b.cond_inlet[0].set_value(cond_inlet_K[t])
        b.env_temp[0].set_value(20 + 273.15)
        
        chiller_setpoint = chiller_schedule[t % len(chiller_schedule)]
        b.outlet_settemp[0].setub(chw_outlet_temp[t])

        # init inlet temp
        b.evap_diff[0].set_value(evap_T_diff[t])

        # if mode = 1 = charging, then increase SOC
        if chw_outlet_temp[i] < 7.5 + 273.15:
            mode = -1
        else:
            mode = 1

        b.del_component('obj')

        b.obj = obj_fx(b, outlet_settemp=chw_outlet_temp[t], inlet_temp=chiller_setpoint,
                        soc=target_soc, chiller_power=chiller_power[t])
        # b.obj = Objective(expr=(b.inlet_temp[0] - (chiller_setpoint + 273.15)) ** 2 + b.chiller_power[0])
        res = ipopt_solver.solve(b, tee=False)
        res = bonmin_solver.solve(b, tee=False)
        b.del_component('obj')

        if not check_binary_variables(m, 0):
            raise RuntimeError

        solved = res.Solver.status == 'ok'
        if not solved:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
            log_infeasible_constraints(b, logger=solve_log, tol=1e-5, log_expression=True, log_variables=True)
            log_infeasible_bounds(b, logger=solve_log, tol=1e-4)
            assert False

        prev_outlet_temp = value(b.outlet_settemp[0])
        prev_temp = value(b.tank_temp[0] - 273.15)
        prev_soc = value(b.soc[0])

        ts_soc.append(prev_soc)
        ts_tank_outlet_temp.append(prev_outlet_temp - 273.15)
        ts_tank_temp.append(prev_temp)
        ts_chiller_outtemp.append(value(b.inlet_temp[0]) - 273.15)

        # free the variables and add cross-period constraints
        b.inlet_temp[0].unfix()
        b.inlet_temp[0].setub(20 + 273.15)
        b.del_component("inlet_temp_setub")
        b.inlet_temp_setub = Constraint(b.horiz_t, rule=lambda b, i:b.inlet_temp[i] <= 20 + 273.15)
        b.outlet_settemp[-1].unfix()
        b.outlet_settemp[0].setub(20 + 273.15)
        b.del_component("outlet_settemp_setub")
        b.outlet_settemp_setub = Constraint(b.horiz_t, rule=lambda b, i:b.outlet_settemp[i] <= 20 + 273.15)
        b.tank_temp_d1[-1].unfix()
        b.tank_temp_d2[-1].unfix()
        b.tank_temp_d3[-1].unfix()
        b.tank_flow_fraction[0].unfix()

    data_set = get_data(m)

    return data_set

def init_obj(b, outlet_settemp=0, outlet_settemp_mult=0,
                inlet_temp=0, inlet_temp_mult=0, 
                soc=0, soc_mult=0, 
                loop_inlet_temp=0, loop_inlet_temp_mult=0,
                tank_flow_fraction=0, tank_flow_fraction_mult=0,
                chiller_power=0, chiller_power_mult=0):
    return Objective(expr=0
                            + (b.outlet_settemp[0] - (outlet_settemp)) ** 2 * outlet_settemp_mult
                            + (b.inlet_temp[0] - (inlet_temp + 273.15)) ** 2 * inlet_temp_mult
                            + (b.soc[0] - soc) ** 2 * soc_mult
                            + (b.loop_inlet_temp[0] - (loop_inlet_temp)) ** 2 * loop_inlet_temp_mult
                            + (b.tank_flow_fraction[0] - tank_flow_fraction) ** 2 * tank_flow_fraction_mult
                            + (b.chiller_power[0] - chiller_power / KW_SCALING) ** 2 * chiller_power_mult
                            )

def make_soc(m, start_ts, chiller_power_new, bldg_load_new, mask_start_hr, mask_end_hr):
    steps_per_hour = int(len(m.block) / 24)
    peak_times = [1 if i in m.peak_time else 0 for i in m.time]

    diff_chiller_p_chrg = -(np.array(bldg_load_new) - np.max(bldg_load_new))
    diff_chiller_p_dchrg = -np.array(peak_times)

    mask = np.zeros_like(peak_times)
    timesteps = np.arange(0, len(peak_times), 1)

    mask[(timesteps >= mask_start_hr * steps_per_hour) & (timesteps < mask_end_hr * steps_per_hour)] = 1
    diff_chiller_p_chrg *= mask - 1

    cumsum_tank_chrg = np.cumsum(diff_chiller_p_chrg)
    cumsum_tank_dchrg = np.cumsum(diff_chiller_p_dchrg)

    norm_tank_chrg = cumsum_tank_chrg / cumsum_tank_chrg[-1] * 0.5
    norm_tank_dchrg = -cumsum_tank_dchrg / cumsum_tank_dchrg[-1] * 0.5

    soc = norm_tank_chrg + norm_tank_dchrg
    soc = soc / max(abs(soc)) * 0.3 + 0.5

    plt.figure()
    plt.plot(soc, label="soc")
    plt.legend()
    return soc

def plot_comparison(m, set_a, set_b, set_c=None):
    fig, ax = plt.subplots(5, 1, figsize=(16, 8))
    time = slice(0, 96)
    ax[0].set_title(f"Chiller load max {round(max(set_a['chiller_p']), 2)}")
    ax[0].plot(set_a['chiller_p'], label="Orig chiller load")
    ax[1].plot(set_a['bldg_p'], label="Orig total load")
    ax[1].set_title(f"Total load max {round(max(set_a['bldg_p']), 2)}")
    ax[2].plot(set_a['chiller_delta_T'], label="Orig chiller_delta_T")
    ax[3].plot(set_a['chw_outlet_temp'], label="Orig chw outlet")

    ax[0].plot(set_b['chiller_p'], label="B chiller load")
    ax[1].plot(set_b['bldg_p'], label="B total load")
    ax[2].plot(set_b['chiller_delta_T'], label="B chiller_delta_T")
    ax[3].plot(set_b['chw_outlet_temp'], label="B chw outlet")
    ax[4].plot(set_b['soc'], label='B soc')

    if set_c is not None:
        ax[0].plot(set_b['chiller_p'], label="C chiller load")
        ax[1].plot(set_c['bldg_p'], label="C total load")
        ax[2].plot(set_c['chiller_delta_T'], label="C chiller_delta_T")
        ax[3].plot(set_c['chw_outlet_temp'], label="C chw outlet")
        ax[4].plot(set_c['soc'], label='C soc')

    ax[4].set_ylim(-0.05, 1.05)
    [a.legend() for a in ax]
    plt.suptitle(f"Comparing: {calc_peak_obj(m)}")
    plt.tight_layout()

from pyomo.environ import Suffix

def solve_relaxed_model(m, binary_enforce, deriv_1st_enforce, deriv_2nd_enforce, binary_enforce_2=None, socs=None, socs_mult=0):
    m.binary_enforce.set_value(binary_enforce)
    m.deriv_1st_enforce.set_value(deriv_1st_enforce)
    m.deriv_2nd_enforce.set_value(deriv_2nd_enforce)

    m.del_component('obj')
    m.obj = Objective(expr=m.peak * 1 + m.peak_tou * m.peak_tou_mult 
                            + summation(m.deriv_reg)
                            + summation(m.binary_close) * m.binary_enforce)
    if socs is not None:
        m.obj += sum([(m.block[i].soc[0] - socs[i]) ** 2 for i in m.time]) * socs_mult
    ipopt_solver.options['OF_ma27_liw_init_factor'] = 50.0
    ipopt_solver.options['OF_ma27_la_init_factor'] = 50.0
    ipopt_solver.options['max_iter'] = 15000
    # ipopt_solver.options['bound_push'] = 1e-9
    # ipopt_solver.options.pop('bound_push')
    if not hasattr(m, "dual"):
        m.dual = Suffix(direction=Suffix.IMPORT)

    res = ipopt_solver.solve(m, tee=False)
    if res.Solver.Status != 'ok':
        raise RuntimeError
    print("First IPOPT solve", calc_peak_obj(m), value(m.obj))

    if binary_enforce_2 is not None:
        m.binary_enforce.set_value(binary_enforce_2)
        res = ipopt_solver.solve(m, tee=False)
        if res.Solver.Status != 'ok':
            raise RuntimeError
        print("Second IPOPT solve", calc_peak_obj(m), value(m.obj))

    return get_data(m)

solver_name = "bonmin"
bonmin_solver = SolverFactory(solver_name)
bonmin_solver.options['tol'] = 1e-5
bonmin_solver.options['max_iter'] = 10000
bonmin_solver.options['bonmin.time_limit'] = 60 * 10
bonmin_solver.options['bonmin.iteration_limit'] = 20000

ipopt_solver = SolverFactory("ipopt")
ipopt_solver.options['tol'] = 1e-8
ipopt_solver.options['max_iter'] = 15000