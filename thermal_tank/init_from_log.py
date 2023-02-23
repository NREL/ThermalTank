import multiprocessing as mp
import json
import pandas as pd
import os
import copy
from pathlib import Path
from timeit import default_timer
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.core.base.block import ComponentUID

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from thermal_tank.tank_bypass_branch import TankBypassBranch, OpMode
from thermal_tank.mpc import *

n_tanks = 1
thermal_tank_log_dir = Path(__file__).absolute().parent.parent / "controls"/ "schedules_mdoff" / "coned" / f"{n_tanks}tanks"

tank_states_file = thermal_tank_log_dir / "chw_states_fixed.parquet"
max_mass_flow_rate = 13.4351549033471

if not os.path.exists(tank_states_file):
    tank_file = thermal_tank_log_dir / f"thermal_tank_{n_tanks}_coned.log"
    init = {}
    states = []
    with open(tank_file, "r") as f:
        for line in f:
            state = json.loads(line)
            if 'num_tanks' in state.keys():
                n_tanks = state['num_tanks']
            if 'timestep' not in state.keys():
                continue
            if state['timestep'] == -1:
                init = state
            else:
                states.append(state)
    df = pd.DataFrame.from_records(states)
    df.to_parquet(tank_states_file, index=False)
else:
    df = pd.read_parquet(tank_states_file)


def check_res(res, m):
    solved = res.Solver.status == 'ok'
    if not solved:
        # solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        # log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        # log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
        raise RuntimeError("Solve Failed")
    return solved


def get_diffs(m, start, end):
    model_attrs = []
    for k in columns:
        model_attrs.append(getattr(m, k))
    diffs = {k: [] for k in columns}
    df_load = df[df.index.isin(range(start, end))]
    for index, row in df_load.iterrows():
        index -= start
        for i, k in enumerate(columns):
            val = row[k]
            if 'temp' in k:
                val += 273.15
            diff = (val - value(model_attrs[i][index])) ** 2
            if 'temp' in k:
                diff /=  val
            diffs[k].append(diff)
    return diffs


def plot_diffs(m, start, end):
    inlet_temps = [value(m.inlet_temp[i]) - 273.15 for i in m.horiz]
    outlet_settemps = [value(m.outlet_settemp[i]) - 273.15 for i in m.horiz]
    tank_temps = [value(m.tank_temp[i]) - 273.15 for i in m.horiz]
    tank_flow = [value(m.tank_flow_fraction[i]) for i in m.horiz]

    socs = [value(m.soc[i]) for i in m.horiz]
    df_load = df[df.index.isin(range(start, end))]
    inlet_temps_0 = df_load['inlet_temp'].values
    outlet_settemps_0 = df_load['outlet_settemp'].values
    tank_temps_0 = df_load['tank_temp'].values
    tank_flow_0 = df_load['tank_flow_fraction'].values
    socs_0 = df_load['soc'].values
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    ax[0].plot(inlet_temps_0, 'b-', label="inlet_temps before")
    ax[0].plot(inlet_temps, 'b--', label="inlet_temps after")
    ax[0].plot(outlet_settemps_0, 'g-', label="outlet_settemps before")
    ax[0].plot(outlet_settemps, 'g--', label="outlet_settemps after")
    ax[0].legend()
    ax[1].plot(tank_temps_0, 'b-', label="tank temp before")
    ax[1].plot(tank_temps, 'b--', label="tank temp after")
    ax[1].plot(tank_flow_0, 'g-', label="tank_flow before")
    ax[1].plot(tank_flow, 'g--', label="tank_flow after")
    ax[1].legend()
    ax[2].plot(socs_0, 'b-', label="socs before")
    ax[2].plot(socs, 'b--', label="socs after")
    ax[2].legend()
    img_path = thermal_tank_log_dir / f"images_{int(add_tank_flow_to_obj)}"
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    plt.savefig(img_path / f"{start}_{end}.png")
    # plt.show()

n = 1440

run_parallel = False
add_tank_flow_to_obj = True
solutions_all_ts = {}
def compile_solutions(m, start):
    labels = ComponentUID.generate_cuid_string_map(m)
    solution = {}
    for var in m.component_data_objects(Var):
        solution[labels[var]] = var.value
    for var in m.component_data_objects(Param):
        solution[labels[var]] = var.value

    if not run_parallel:
        solutions_all_ts[start] = solution
        sol_file = thermal_tank_log_dir / f"solution_{n}_start_{start}.json"
        with open(sol_file, "w") as f:
            json.dump(solutions_all_ts, f)

    return start, solution


columns = ['tank_flow_fraction', 'inlet_temp', 'outlet_temp', 'outlet_settemp', 'tank_temp',
       'env_temp', 'is_charging', 'soc', 'cond_inlet', 'evap_diff', 'chiller_power']

def populate_values(m, start, offset=0, df_states=None):
    if df_states is None:
        df_states = df
    start += offset
    if start > 0:
        init_temp(m, df_states['tank_temp'].values[start - 1], df_states['soc'].values[start - 1], index=-1)
    obj_expr = 0
    n = len(m.horiz_t)
    end = start + n - 1
    df_load = df_states[df_states.index.isin(range(start, end + 1))]
    loop_inlet_vals = df_states['loop_inlet'].values + 273.15
    if start == 0:
        m.outlet_settemp[-1].fix(loop_inlet_vals[0])
    else:
        m.outlet_settemp[-1].fix(df_states['outlet_settemp'].values[start-1] + 273.15)
    for index, row in df_load.iterrows():
        index -= start
        tank_flow_rate_zero = False
        if index > n:
            break
        for i, k in enumerate(columns):
            val = row[k]
            # Building Items: cond_inlet, evap_diff, chiller_power
            if k == 'cond_inlet':
                val += 273.15
            if k == 'chiller_power':
                chiller_delta = value(m.inlet_temp[index] - m.loop_inlet_temp[index])
                val = value(chiller_a[0] * m.cond_inlet[index] + chiller_a[1] * chiller_delta + chiller_b)
                if val >= 0:
                    obj_expr += (m.chiller_power[index] - val) ** 2 * 1e3
                else:
                    continue
            if k == "evap_diff":
                man_val = value(loop_inlet_vals[start + index + 1] - m.outlet_settemp[index])
                m.evap_diff[index].set_value(val)
            # Tank Items
            if k == 'soc':
                continue
            if k == 'tank_temp':
                init_temp(m, val, row['soc'], index)
                tank_temp_mpc = value(m.tank_temp[index])
                obj_expr += (m.tank_temp[index] - tank_temp_mpc) ** 2
                soc = value(m.soc[index])
                obj_expr += (m.tank_temp[index] - tank_temp_mpc) ** 2 * 1e3
                continue
            if 'temp' in k:
                val += 273.15
            if k == 'tank_flow_fraction':
                if val == 0:
                    tank_flow_rate_zero = True
                val = max(m.tank_flow_fraction[0].lb * 2, val)
                if add_tank_flow_to_obj:
                    obj_expr += (m.tank_flow_fraction[index] - val) ** 2 * 10
            if k == 'is_charging':
                val = row['op_mode']
                if val == OpMode.FLOAT:
                    if row['tank_temp'] < row['inlet_temp']:
                        val = OpMode.DISCHARGING
                    else:
                        val = OpMode.CHARGING
                if val == OpMode.CHARGING:
                    m.is_discharging[index].set_value(0)
                    m.is_charging[index].set_value(1)
                    if val == OpMode.CHARGING:
                        m.tank_flow_fraction[index].set_value(1)
                else:
                    m.is_charging[index].set_value(0)
                    m.is_discharging[index].set_value(1)
                continue
            getattr(m, k)[index].set_value(val)
            if tank_flow_rate_zero:
                continue
            if k == "outlet_settemp" or k == "outlet_temp":
                obj_expr += (getattr(m, k)[index] - val) ** 2 * 1e-2
            if k == "inlet_temp" and val != 0:
                obj_expr += (getattr(m, k)[index] - val) ** 2 * 1e-2
        if row['op_mode'] == OpMode.FLOAT:
            new_outlet_temp = value(m.inlet_temp[index] 
                                            - m.effectiveness[index] * (
                                                m.outlet_effectiveness_discharging[index] * m.is_discharging[index] + m.outlet_effectiveness_charging[index] * m.is_charging[index]) 
                                            * (m.inlet_temp[index] - m.tank_temp[index]))
            m.outlet_temp[index] = new_outlet_temp
    # m.inlet_temp[n - 1].set_value(loop_inlet_vals[n - 1])
    return obj_expr


def load_data(start, end, temp_0, latent_soc):
    tank_data_copy = copy.deepcopy(tank_data)
    if temp_0 == 0:
        tank_data_copy['latent_state_of_charge'] = latent_soc
    else:
        tank_data_copy['initial_temperature'] = temp_0
    tank_bypass_branch = TankBypassBranch(num_tanks=1, tank_data=tank_data_copy)
    tank_data_copy['max_mass_flow_rate'] = max_mass_flow_rate
    n = end - start - 1
    m, tank = full_model(tank_bypass_branch.tank, max_mass_flow_rate, simple=False, calc_mode=False, n=n)

    model_attrs = []
    for k in columns:
        model_attrs.append(getattr(m, k))

    init_temp(m, temp_0, latent_soc, -1)

    # obj_expr = 2 * (m.tank_temp[-1] - (temp_0 + 273.15)) ** 2 + (m.soc[-1] - latent_soc) ** 2 * 1e3
    obj_expr = populate_values(m, start)
    init_building(m, start)

    return m, obj_expr


def solve_matching_log(start, end):
    if start == 0:
        temp_0 = 15
        latent_soc = 0
    else:
        df_init = df[df.index == start - 1]
        temp_0 = df_init['tank_temp'].values[0]
        latent_soc = df_init['soc'].values[0]

    m, obj_expr = load_data(start, end, temp_0, latent_soc)
    m.obj = Objective(expr=obj_expr)
    m.tank_temp_d1.unfix()
    m.tank_temp_d2.unfix()
    m.tank_temp_d3.unfix()
    m.tank_temp_d1[-1].fix()
    m.tank_temp_d2[-1].fix()
    m.tank_temp_d3[-1].fix()
    m.inlet_temp[0].fix()
    # m.outlet_settemp[0].fix()
    m.tank_temp_d1[n - 1].fix()
    m.tank_temp_d2[n - 1].fix()
    m.tank_temp_d3[n - 1].fix()
    # m.inlet_temp[n - 1].fix()
    m.outlet_settemp[n - 1].fix()

    solver_start = default_timer()
    solver = SolverFactory("bonmin")
    # solver.options['bonmin.iteration_limit'] = 90
    # solver.options['bonmin.iteration_limit'] = 90
    solver.options['max_iter'] = 10000
    # solver.options['bonmin.algorithm'] = 'B-iFP'

    solver.options['tol'] = 1e-5
    res = solver.solve(m, tee=True)
    print(f"{start}: time", default_timer() - solver_start, value(m.obj))

    check_res(res, m)
    return m


def task(start):
    end = start + n
    try:
        m = solve_matching_log(start, end)
        key, solution = compile_solutions(m, start)
        plot_diffs(m, start, end)
        diffs = get_diffs(m, start, end)
        # solution['diff_sum'] = sum(sum(v) for k, v in diffs.items())
    except Exception as e:
        key, solution = start, "Failed"
        print(start, "failed: ", e)
        # raise Exception
    
    return key, solution


def run_initialization():
    if len(sys.argv) > 2:
        node = int(sys.argv[1])
        nodes_tot = int(sys.argv[2])
        num_solves = int(525600 / nodes_tot)
        start_solve = num_solves * node
        end_solve = min(start_solve + num_solves, 525600)
    elif len(sys.argv) > 1:
        start_solve = int(sys.argv[1])
        end_solve = 525600
    else:
        start_solve = 0
        end_solve = 525600

    print(f"Solving TS: {start_solve} to {end_solve}")
    if run_parallel:
        starts = range(start_solve, end_solve, n)
        with mp.Pool(mp.cpu_count() - 1) as pool:
            res = pool.map(task, starts)
            for k, s in res:
                solutions_all_ts[k] = s
                
            sol_file = thermal_tank_log_dir / f"solution_{n}_{start_solve}_{end_solve}.json"
            with open(sol_file, "w") as f:
                json.dump(solutions_all_ts, f)

    else:
        start = start_solve
        end = start + n
    
        while end < 365 * 24 * n:
            key, solution = task(start)

            start = end
            end += n


def load_initialization(start=0, temp_0=15, latent_soc=0):
    tank_data_copy = copy.deepcopy(tank_data)
    if temp_0 == 0:
        tank_data_copy['latent_state_of_charge'] = latent_soc
    else:
        tank_data_copy['initial_temperature'] = temp_0
    tank_bypass_branch = TankBypassBranch(num_tanks=1, tank_data=tank_data_copy)
    m, tank = full_model(tank_bypass_branch.tank, max_mass_flow_rate, simple=False, calc_mode=False, n=1440)
    with open(thermal_tank_log_dir / "solution_1440.json", 'r') as f:
        solution = json.load(f)
        solution = solution[str(start)]
    for cuid, val in solution.items():
        try:
            m.find_component(cuid).value = val
        except:
            print(cuid)
            exit()

    m.obj = Objective(expr=0)

    solver_start = default_timer()
    solver = SolverFactory("bonmin")
    solver.options['tol'] = 1e-5
    res = solver.solve(m, tee=False)
    print(f"{start}: load time", default_timer() - solver_start, value(m.obj))


def plot_modified_operation(m, df_states_orig, start_ts, n_ts):
    new_tank_temp = [value(m.tank_temp[i])  - 273.15 for i in m.horiz_t]
    new_loop_inlet = [value(m.loop_inlet_temp[i])  - 273.15 for i in m.horiz_t]
    new_inlet_temp = [value(m.inlet_temp[i])  - 273.15 for i in m.horiz_t]
    new_outlet_temp = [value(m.outlet_settemp[i])  - 273.15 for i in m.horiz_t]
    new_soc = [value(m.soc[i]) for i in m.horiz_t]
    new_evap_diff = [value(m.evap_diff[i]) for i in m.horiz_t]
    new_chiller_power = [value(m.chiller_power[i]) for i in m.horiz_t]
    new_bldg_elec_load = [value(m.bldg_elec_load[i]) for i in m.horiz_t]
    new_max_load = max(value(m.bldg_elec_load[i]) for i in m.horiz_t)

    start = start_ts
    end = start + n_ts
    old_loop_inlet = df_states_orig['loop_inlet'][start:end].values
    old_inlet_temp = df_states_orig['inlet_temp'][start:end].values
    old_outlet_temp = df_states_orig['outlet_settemp'][start:end].values
    old_soc = df_states_orig['soc'][start:end].values
    old_tank_temp = df_states_orig['tank_temp'][start:end].values
    old_chiller_power = df_states_orig['chiller_power_pred'][start:end].values
    old_bldg_elec_load = df_states_orig['bldg_elec_load'][start:end].values
    old_evap_diff = df_states_orig['evap_diff'][start:end].values
    old_max_load = max(old_bldg_elec_load)

    fig, ax = plt.subplots(4, 1, figsize=(18,12))

    # ax[0].plot(old_loop_inlet, 'b', linestyle='dashed', label="old: chiller inlet temp ")
    # ax[0].plot(old_inlet_temp, 'y', linestyle='dashed', label="old: tank inlet temp")
    # ax[0].plot(old_outlet_temp, 'g', linestyle='dashed', label="old: tank out temp")
    
    # ax[0].plot(new_loop_inlet, 'b', linestyle='solid', label="new: chiller inlet temp")
    # ax[0].plot(new_inlet_temp, 'y', linestyle='solid', label="new: tank inlet temp")
    # ax[0].plot(new_outlet_temp, 'g', linestyle='solid', label="new: tank out temp")

    ax[0].plot(old_evap_diff, 'k',linestyle='dashed',  label="old: Building Delta T / evap_diff")
    ax[0].plot(new_evap_diff, 'k', linestyle='solid', label="old: Building Delta T / evap_diff")

    # ax[1].plot(old_soc, 'k', linestyle='dashed', label="old: tank soc")
    # ax[1].plot(old_tank_temp, 'r', linestyle='dashed', label="old: tank temp")
    
    # ax[1].plot(new_soc, 'k', linestyle='solid', label="new: tank soc")
    # ax[1].plot(new_tank_temp, 'r', linestyle='solid', label="new: tank temp")

    ax[1].plot(np.array(old_outlet_temp) - np.array(old_inlet_temp), 'r', linestyle='dashed', label="old: tank delta t")
    ax[1].plot(np.array(new_outlet_temp) - np.array(new_inlet_temp), 'k', linestyle='solid', label="new: tank delta t")

    ax[2].plot(old_chiller_power, 'b', linestyle='dashed', label="old: chiller power")
    ax[2].plot(old_bldg_elec_load, 'g', linestyle='dashed', label="old: bldg power")

    ax[2].plot(new_chiller_power, 'b', linestyle='solid', label="new: ciller power")
    ax[2].plot(new_bldg_elec_load, 'g', linestyle='solid', label="new: bldg power")

    ax[2].plot([old_max_load] * n, 'r', linestyle='dashed', label="old: max load")
    ax[2].plot([new_max_load] * n, 'r', linestyle='solid', label="new: max load")

    ax[3].plot(np.array(old_inlet_temp) - np.array(old_loop_inlet), 'k',linestyle='dashed',  label="old: chiller delta T")
    ax[3].plot(np.array(new_inlet_temp) - np.array(new_loop_inlet), 'k', linestyle='solid', label="new: chiller delta T")
    
    for a in ax:
        a.legend()
    plt.show()


def save_vars_to_df(m, index, run_name=None):
    df = pd.DataFrame(columns=["loop_inlet_temp", 'loop_outlet_temp', 'bldg_delta_T', 'tank_temp', 'soc', "tank_delta_T",
                               "chiller_outlet_temp", "chiller_delta_T", "chiller_power", "bldg_load"], index=index)
    
    df['loop_inlet_temp'] = [value(m.block[i].loop_inlet_temp[0]) for i in m.time]
    df['loop_outlet_temp'] = [value(m.block[i].outlet_settemp[0]) for i in m.time]
    df['bldg_delta_T'] = [value(m.block[i].evap_diff[0]) for i in m.time]
    df['tank_temp'] = [value(m.block[i].tank_temp[0]) for i in m.time]
    df['soc'] = [value(m.block[i].soc[0]) for i in m.time]
    df['tank_delta_T'] = [value(m.block[i].outlet_temp[0] - m.block[i].inlet_temp[0]) for i in m.time]
    df['chiller_outlet_temp'] = [value(m.block[i].inlet_temp[0]) for i in m.time]
    df['chiller_delta_T'] = [value(m.block[i].inlet_temp[0] - m.block[i].loop_inlet_temp[0]) for i in m.time]
    df['chiller_power'] = [value(m.block[i].chiller_power[0]) for i in m.time]
    df['bldg_load'] = [value(m.block[i].bldg_elec_load[0]) for i in m.time]
    df['outlet_temp'] = [value(m.block[i].outlet_temp[0]) for i in m.time]
    df['outlet_temp'] = [value(m.block[i].env_temp[0]) for i in m.time]
    df['tank_flow_fraction'] = [value(m.block[i].tank_flow_fraction[0]) for i in m.time]
    df['is_charging'] = [value(m.block[i].is_charging[0]) for i in m.time]
    df['env_temp'] = [value(m.block[i].env_temp[0]) for i in m.time]

    # calculating q_brine_max
    df['brine_ave_temp'] = [value((m.block[i].inlet_temp[0] + m.block[i].tank_temp[-1]) / 2) for i in m.time]
    df['brine_specific_heat_avg'] = [value(m.block[i].brine_specific_heat_avg[0]) for i in m.time]
    df['q_brine_max'] = [value(m.block[i].q_max_brine[0]) for i in m.time]
    
    # calculating q_brine
    df['brine_specific_heat'] = [value(m.block[i].brine_specific_heat[0]) for i in m.time]
    df['effectiveness'] = [value(m.block[i].effectiveness[0]) for i in m.time]
    df['brine_effectiveness_charging'] = [value(m.block[i].brine_effectiveness_charging[0]) for i in m.time]
    df['brine_effectiveness_discharging'] = [value(m.block[i].brine_effectiveness_discharging[0]) for i in m.time]
    df['brine_effectiveness_total'] = [value(m.block[i].effectiveness[0] * (m.block[i].brine_effectiveness_discharging[0] * m.block[i].is_discharging[0] 
                                + m.block[i].brine_effectiveness_charging[0] * m.block[i].is_charging[0])) for i in m.time]
    df['q_brine'] = [value(m.block[i].q_brine[0]) for i in m.time]
    df['q_env'] = [value(m.block[i].q_env[0]) for i in m.time]
    df['q_tot'] = [value(m.block[i].q_tot[0]) for i in m.time]
    
    # calculating outlet fluid
    df['outlet_effectiveness_charging'] = [value(m.block[i].outlet_effectiveness_charging[0]) for i in m.time]
    df['outlet_effectiveness_discharging'] = [value(m.block[i].outlet_effectiveness_discharging[0]) for i in m.time]
    df['outlet_effectiveness'] = [value(m.block[i].outlet_effectiveness[0]) for i in m.time]

    if run_name is not None:
        df['run'] = run_name
    return df


def save_params_to_df(m, index):
    df_params = pd.DataFrame(index=index)
    param_vars = []
    for var in m.block[0].__dir__():
        var = getattr(m.block[0], var)
        if 'IndexedParam' not in str(type(var)):
            continue
        name = var.name.split('.')[1]
        if '[' in name:
            name = name.split('[')[0]
        param_vars.append(name)
    param_vars = list(set(param_vars))

    n = len(index)
    df_params['tank_temp_init'] = [value(m.block[i].tank_temp[-1]) - 273.15 for i in range(n)]
    df_params['tank_temp'] = [value(m.block[i].tank_temp[0]) - 273.15 for i in range(n)]
    df_params['avail_Q'] = [value(m.block[i].avail_Q[-1]) for i in range(n)]
    df_params['avail_Q_init'] = [value(m.block[i].avail_Q[0]) for i in range(n)]
    df_params['outlet_settemp_init'] = [value(m.block[i].outlet_settemp[-1]) - 273.15 for i in range(n)]
    df_params['outlet_settemp'] = [value(m.block[i].outlet_settemp[0]) - 273.15 for i in range(n)]
    df_params['soc'] = [value(m.block[i].soc[0]) for i in range(n)]
    df_params['soc_init'] = [value(m.block[i].soc[-1]) for i in range(n)]
    df_params['tank_temp_d1'] = [value(m.block[i].tank_temp_d1[0]) for i in range(n)]
    df_params['tank_temp_d2'] = [value(m.block[i].tank_temp_d2[0]) for i in range(n)]
    df_params['tank_temp_d3'] = [value(m.block[i].tank_temp_d3[0]) for i in range(n)]
    df_params['tank_temp_d1' + "_init"] = [value(m.block[i].tank_temp_d1[-1]) for i in range(n)]
    df_params['tank_temp_d2' + "_init"] = [value(m.block[i].tank_temp_d2[-1]) for i in range(n)]
    df_params['tank_temp_d3' + "_init"] = [value(m.block[i].tank_temp_d3[-1]) for i in range(n)]
    df_params['avail_Q'] = [value(m.block[i].avail_Q[0]) for i in range(n)]
    df_params['avail_Q' + "_init"] = [value(m.block[i].avail_Q[-1]) for i in range(n)]
    return df_params


if __name__ == "__main__":
    run_initialization()