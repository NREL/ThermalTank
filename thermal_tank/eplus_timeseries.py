import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


tes_m_default = {"charge_end": "07:58",
                "charge_start": "22:00",
                "charge_temp": -4.8,
                "chiller_trim_temp": 8.7,
                "discharge_end": "18:00",
                "discharge_start": "07:59"}

def convert_time_to_timesteps(time_clock, dt_hr):
    hour, min = time_clock.split(":")
    minutes = float(hour) * 60 + float(min)
    minutes_per_ts = 60 * dt_hr
    return minutes // minutes_per_ts

def make_chiller_schedule(dt_hr=1/4, tes_m=tes_m_default):
    args = tes_m
    charge_end = convert_time_to_timesteps(args['charge_end'], dt_hr)
    charge_start = convert_time_to_timesteps(args['charge_start'], dt_hr)
    charge_temp = args['charge_temp']
    discharge_end = convert_time_to_timesteps(args['discharge_end'], dt_hr)
    discharge_start = convert_time_to_timesteps(args['discharge_start'], dt_hr)
    chiller_trim_temp = args['chiller_trim_temp']

    chiller_schedule = {}
    for t in range(round(1 / dt_hr) * 24):
        if t >= discharge_start and t < discharge_end:
            chiller_schedule[t] = chiller_trim_temp
        elif t <= charge_end or t > charge_start:
            chiller_schedule[t] = charge_temp
        elif t >= discharge_end and t <= charge_start:
            temp = (charge_temp - chiller_trim_temp) / (charge_start - discharge_end) * (t - discharge_end) + chiller_trim_temp
            chiller_schedule[t] = temp
        else:
            temp = (chiller_trim_temp - charge_temp) / (discharge_start - charge_end) * (t - charge_end) + charge_temp
            chiller_schedule[t] = temp
    return chiller_schedule

# evaporator delta T can have incorrect jumps that need to be smoothed out...
def calc_delta_Ts(df_orig):
    loop_inlet = df_orig['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'].values + 273.15
    chw_outlet_temp = df_orig['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'].values + 273.15
    evap_T_diff = [loop_inlet[i+1] - chw_outlet_temp[i] for i in range(0, len(df_orig) - 1)] + [loop_inlet[0] - chw_outlet_temp[len(df_orig) - 1]]
    chiller_delta_T = chw_outlet_temp - loop_inlet
    return evap_T_diff, chiller_delta_T


def remove_evap_T_outliers(df_orig):
    df = df_orig.copy(deep=True)
    evap_T_diff, chiller_delta_T = calc_delta_Ts(df)
    ts_per_day = len(df_orig) // 365

    for day in range(365):
        id = day * ts_per_day
        day_slice = slice(id, id+ts_per_day)
        rolling_average = np.average(evap_T_diff[day_slice])
        rolling_std = np.std(evap_T_diff[day_slice])
        for i, evap_dT in enumerate(evap_T_diff[day_slice]):
            j = i + id
            if evap_dT > 2.25 * rolling_std + rolling_average or evap_dT < -2.25 * rolling_std + rolling_average:
                evap_T_diff[j] = (evap_T_diff[j-1] + evap_T_diff[(j+1) % len(evap_T_diff)]) / 2
                # evap_T_diff[j] = evap_T_diff[j-1]
    
    chw_outlet_temp = df['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'].values
    loop_inlets = df['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'].values
    new_loop_inlets = []
    new_loop_outlets = []
    for i, evap_dT in enumerate(evap_T_diff):
        if i == 0:
            new_loop_inlet = loop_inlets[0] - 273.15
            new_loop_outlet = chw_outlet_temp[0]
        else:
            new_loop_inlet = evap_T_diff[i - 1] + new_loop_outlets[i - 1]
            new_loop_outlet = new_loop_inlet + chiller_delta_T[i]
        new_loop_inlets.append(new_loop_inlet)
        new_loop_outlets.append(new_loop_outlet)

    df['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] = new_loop_inlets
    df['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] = new_loop_outlets
    evap_T_diff, chiller_delta_T = calc_delta_Ts(df)
    return df, new_loop_inlets, new_loop_outlets, evap_T_diff, chiller_delta_T

    
def smooth_df(df_orig):
    df = df_orig.copy(deep=True)
    evap_T_diff, chiller_delta_T = calc_delta_Ts(df)
    ts_per_hour = len(df_orig) // 365 // 24
    
    zerod_i = []
    for i in range(len(df_orig)):
        evap_dT = evap_T_diff[i]
        len_slice = 5
        avg_slice = slice(max(0, i-len_slice), min(len(evap_T_diff), i+len_slice))
        rolling_average = np.average(evap_T_diff[avg_slice])
        rolling_std = np.std(evap_T_diff[avg_slice])
        if evap_dT < rolling_average - 2.25 * rolling_std and rolling_std > 0.1:
            # case where building suddenly cools down significantly, need to start up chiller slowly beforehand
            chiller_dT_avg = np.delete(np.round(chiller_delta_T[avg_slice], 2), len_slice)
            chiller_dT_avg = np.average(chiller_delta_T) / 2

            n_timesteps_to_change = round(abs(evap_dT / chiller_dT_avg))
            j = i 
            while j > i - n_timesteps_to_change:
                new_loop_outlet = df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)']
                new_loop_inlet = new_loop_outlet - chiller_dT_avg
                j_ind = df.index.values[j]
                df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] = new_loop_inlet
                df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] = new_loop_outlet
                evap_T_diff[j] = df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                evap_T_diff[j+1] = df.iloc[j+2]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']            
                chiller_delta_T[j] = df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)']
                j -= 1
                if abs(j - i) > 100:
                    raise RuntimeError

            evap_dT_avg = np.delete(np.round(evap_T_diff[avg_slice], 2), len_slice)
            i = j

        if (evap_dT > 2.35 * rolling_std + rolling_average and rolling_std > 0.5) or (evap_dT > 2 * rolling_std + rolling_average and rolling_std > 2.5):
            # case where building suddenly heats up significantly, need to ramp down chiller slowly beforehand
            chiller_dT_avg = np.delete(np.round(chiller_delta_T[avg_slice], 2), len_slice)
            chiller_dT_avg = np.average(chiller_delta_T)
            evap_dT_avg = np.delete(np.round(evap_T_diff[avg_slice], 2), len_slice)
            evap_dT_avg = np.average(evap_dT_avg)
            chiller_dT_avg = max(-evap_dT_avg * 0.5, chiller_dT_avg)

            next_hour = evap_T_diff[i:i + ts_per_hour]
            next_min = (next_hour - rolling_average)
            next_min_mask = np.zeros_like(next_min)
            next_min_mask[next_min < -2 * rolling_std] = 1
            if max(next_min_mask):
                # print(i)
                j = i + 1
                new_loop_inlet = df.iloc[j-1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                if  evap_T_diff[j] == 0:
                    new_evap_T = 0
                else:
                    new_evap_T = max(0, (evap_T_diff[j-2] + evap_T_diff[j])/2)
                new_loop_outlet = new_loop_inlet + new_evap_T
                while j <= i + np.argmax(next_min_mask):
                    j_ind = df.index[j]
                    df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] = new_loop_inlet
                    df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] = new_loop_outlet
                    evap_T_diff[j] = df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                    evap_T_diff[j-1] = df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j-1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                    evap_T_diff[j+1] = df.iloc[j+2]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']            
                    chiller_delta_T[j] = df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)']
                    j += 1


            else:
                j = i + 1
                while evap_dT > rolling_std + rolling_average and j < len(df):
                    new_loop_inlet = df.iloc[j-1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] + evap_dT_avg
                    new_loop_outlet = new_loop_inlet + chiller_dT_avg

                    j_ind = df.index.values[j]
                    df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] = new_loop_inlet
                    df.at[j_ind, 'CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] = new_loop_outlet
                    evap_T_diff[j-1] = df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j-1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                    if j + 1 < len(df):
                        evap_T_diff[j] = df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                    if j + 2 < len(df):
                        evap_T_diff[j+1] = df.iloc[j+2]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'] - df.iloc[j+1]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)']
                    evap_dT = evap_T_diff[j]
                    chiller_delta_T[j] = df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'] - df.iloc[j]['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)']
                    avg_slice = slice(max(0, j-len_slice), min(len(evap_T_diff), j+len_slice))
                    rolling_average = np.average(evap_T_diff[avg_slice])
                    rolling_std = np.std(evap_T_diff[avg_slice])
                    j += 1
                    if j >= len(df):
                        break
                # print(i, abs(j-i), df.index[i], df.index[j], "bottom")
                evap_dT_avg = np.delete(np.round(evap_T_diff[avg_slice], 2), len_slice)
                # if np.average(evap_dT_avg) > 4:
                    # print(i, evap_T_diff[avg_slice])

            i = j
            if i >= len(df):
                break

    df['evap_T_diff'] = evap_T_diff
    df['chiller_delta_T'] = chiller_delta_T
    return df, df['CHW LOOP CHILLER:Chiller Evaporator Inlet Temperature [C](TimeStep)'], \
    df['CHW LOOP CHILLER:Chiller Evaporator Outlet Temperature [C](TimeStep)'], evap_T_diff, chiller_delta_T, zerod_i

def get_resampled_df(df, dt_hr):
    if dt_hr == 1/60:
        return df
    df_resample = df.copy(deep=True)
    df_resample = df_resample.resample(f'{int(60 * dt_hr)}T', origin='start').mean()[0:int(8760 / dt_hr)]
    load = df_resample['load'].values
    pv_gen = df_resample['pv_gen'].values
    cond_inlet_K = df_resample['cond_inlet_K'].values
    chiller_power = df_resample['chiller_power'].values
    chw_outlet_temp = df_resample['chw_outlet_temp'].values
    loop_inlet = df_resample['loop_inlet'].values
    evap_T_diff, chiller_delta_T = calc_delta_Ts(df_resample)
    df_resample['chiller_delta_T'] = chiller_delta_T
    df_resample['evap_T_diff'] = evap_T_diff
    return df_resample, loop_inlet, chw_outlet_temp, evap_T_diff, chiller_delta_T

def get_day_data(df, start_ts):
    n = len(df) // 365
    time_slice = slice(start_ts, start_ts + n)
    orig_set = {"chiller_p": df['chiller_power'][time_slice],
            "bldg_p": df['chiller_power'][time_slice] + df['load'][time_slice],
            "chiller_out": df['chiller_delta_T'][time_slice],
            "chiller_delta_T": df['chiller_delta_T'][time_slice],
            "chw_outlet_temp": df['chw_outlet_temp'][time_slice] - 273.15}
    return orig_set