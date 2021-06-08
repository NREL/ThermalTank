from pyenergyplus.plugin import EnergyPlusPlugin
from simple_ice_tank import IceTank


class UsrDefPlntCmpSet(EnergyPlusPlugin):

    def __init__(self):
        super().__init__()

    def on_user_defined_component_model(self, state) -> int:
        return 0


class UsrDefPlntCmpSim(EnergyPlusPlugin):
    loop_exit_temp = 6.7
    loop_delt_temp = 8.33

    def __init__(self):

        super().__init__()
        self.need_to_get_handles = True
        self.glycol = None

        # setpoint schedule handles
        self.t_set_chiller_hndl = None
        self.t_set_icetank_hndl = None

        # loop internal variable handle
        self.vdot_loop_hndl = None

        # component internal variable handles
        self.t_in_hndl = None
        self.mdot_in_hndl = None
        self.cp_in_hndl = None
        self.load_in_hndl = None

        # setup actuator handles
        self.mdot_min_hndl = None
        self.mdot_max_hndl = None
        self.vdot_des_hndl = None
        self.load_min_hndl = None
        self.load_max_hndl = None
        self.load_opt_hndl = None

        # sim actuator handles
        self.t_out_hndl = None
        self.mdot_out_hndl = None

        # global variable handles
        self.bypass_frac_hndl = None
        self.soc_hndl = None

        # tank data
        self.tank_data = {
            "tank_diameter": 89 * 0.0254,
            "tank_height": 101 * 0.0254,
            "fluid_volume": 1655 * 0.00378541,
            "r_value_lid": 24 / 5.67826,
            "r_value_base": 9 / 5.67826,
            "r_value_wall": 9 / 5.67826,
            "initial_temperature": -20,
        }

        # init tank
        self.tank = IceTank(self.tank_data)

    def get_handles(self, state):

        # get setpoint schedule handles
        self.t_set_chiller_hndl = self.api.exchange.get_variable_handle(state, 'Schedule Value', 'Chiller Temp Sch')
        self.t_set_icetank_hndl = self.api.exchange.get_variable_handle(state, 'Schedule Value', 'Ice Tank Temp Sch')

        # get loop internal variable handle
        self.vdot_loop_hndl = self.api.exchange.get_internal_variable_handle(state, 'Plant Design Volume Flow Rate',
                                                                             'CHW Loop')

        # get component internal variable handles
        self.t_in_hndl = self.api.exchange.get_internal_variable_handle(state,
                                                                        'Inlet Temperature for Plant Connection 1',
                                                                        'CHW Loop Ice Tank')
        self.mdot_in_hndl = self.api.exchange.get_internal_variable_handle(state,
                                                                           'Inlet Mass Flow Rate for Plant Connection 1',
                                                                           'CHW Loop Ice Tank')
        self.cp_in_hndl = self.api.exchange.get_internal_variable_handle(state,
                                                                         'Inlet Specific Heat for Plant Connection 1',
                                                                         'CHW Loop Ice Tank')
        self.load_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Load Request for Plant Connection 1',
                                                                           'CHW Loop Ice Tank')

        # get setup actuator handles
        self.mdot_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Minimum Mass Flow Rate', 'CHW Loop Ice Tank')
        self.mdot_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Maximum Mass Flow Rate', 'CHW Loop Ice Tank')
        self.vdot_des_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Design Volume Flow Rate', 'CHW Loop Ice Tank')
        self.load_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Minimum Loading Capacity', 'CHW Loop Ice Tank')
        self.load_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Maximum Loading Capacity', 'CHW Loop Ice Tank')
        self.load_opt_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1',
                                                                   'Optimal Loading Capacity', 'CHW Loop Ice Tank')

        # get sim actuator handles
        self.t_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Outlet Temperature',
                                                                'CHW Loop Ice Tank')
        self.mdot_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Mass Flow Rate',
                                                                   'CHW Loop Ice Tank')

        # get global handles
        self.bypass_frac_hndl = self.api.exchange.get_global_handle(state, "bypass_frac")
        self.soc_hndl = self.api.exchange.get_global_handle(state, "soc")

        self.need_to_get_handles = False

    # reinitialize tank after warmup
    def on_after_new_environment_warmup_is_complete(self, state) -> int:
        print("on_after_new_environment_warmup_is_complete")
        self.tank.init_state(tank_init_temp=15)
        print(self.tank.state_of_charge)
        return 0

    def on_user_defined_component_model(self, state) -> int:

        # wait until API is ready
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        # get handles
        if self.need_to_get_handles:
            self.get_handles(state)
            self.glycol = self.api.functional.glycol(state, u'water')

        # get setpoint schedule values
        t_set_chiller = self.api.exchange.get_variable_value(state, self.t_set_chiller_hndl)
        t_set_icetank = self.api.exchange.get_variable_value(state, self.t_set_icetank_hndl)

        # get loop internal variable values
        vdot_loop = self.api.exchange.get_internal_variable_value(state, self.vdot_loop_hndl)

        # get component internal variable values
        t_in = self.api.exchange.get_internal_variable_value(state, self.t_in_hndl)
        mdot_in = self.api.exchange.get_internal_variable_value(state, self.mdot_in_hndl)
        cp_in = self.api.exchange.get_internal_variable_value(state, self.cp_in_hndl)
        load_in = self.api.exchange.get_internal_variable_value(state, self.load_in_hndl)

        # calcs
        cp_loop = self.glycol.specific_heat(state, self.loop_exit_temp)
        rho_loop = self.glycol.density(state, self.loop_exit_temp)
        mdot_loop = rho_loop * vdot_loop
        cap_loop = mdot_loop * cp_loop * self.loop_delt_temp

        # set flow actuators
        self.api.exchange.set_actuator_value(state, self.mdot_min_hndl, 0)
        self.api.exchange.set_actuator_value(state, self.mdot_max_hndl, mdot_loop)
        self.api.exchange.set_actuator_value(state, self.vdot_des_hndl, vdot_loop)

        # set load actuators
        self.api.exchange.set_actuator_value(state, self.load_min_hndl, 0)
        self.api.exchange.set_actuator_value(state, self.load_max_hndl, cap_loop)
        self.api.exchange.set_actuator_value(state, self.load_opt_hndl, cap_loop)

        # determine flow rate
        mdot_max = mdot_loop
        if mdot_in == 0:
            mdot_act = mdot_max
        else:
            mdot_act = mdot_in

        # average specific heat
        cp_out = self.glycol.specific_heat(state, self.loop_exit_temp)
        cp_avg = (cp_in + cp_out) / 2

        # determine load
        load = mdot_act * cp_avg * (t_in - t_set_icetank)
        load_max = cap_loop
        if load > load_max:
            load_act = load_max
        else:
            load_act = load

        # set current date/time and init bypass fraction
        dt = self.api.exchange.actual_date_time(state)
        bypass_frac = 0

        # charge
        if t_set_chiller < t_set_icetank:
            self.tank.calculate(t_in, mdot_act, 24, dt, 60)
            t_out = self.tank.outlet_fluid_temp

        # pass-through if no load
        elif t_set_chiller == t_set_icetank:
            self.api.exchange.set_actuator_value(state, self.t_out_hndl, t_in)
            self.api.exchange.set_actuator_value(state, self.mdot_out_hndl, mdot_in)
            return 0

        # determine tank outlet temp at full flow rate
        else:
            self.tank.calculate(t_in, mdot_act, 24, dt, 60)
            t_out = self.tank.outlet_fluid_temp

            # find flow rate that results in setpoint if less at full flow
            if t_out < t_set_icetank:
                for i in range(1, 100):
                    mdot_tank = (1 - (i / 100)) * mdot_act
                    mdot_bypass = mdot_act - mdot_tank
                    bypass_frac = mdot_tank / mdot_act
                    self.tank.calculate(t_in, mdot_tank, 24, dt, 60)
                    t_tank = self.tank.outlet_fluid_temp
                    t_out = ((mdot_tank * t_tank) + (mdot_bypass * t_in)) / (mdot_tank + mdot_bypass)
                    if abs(t_out - t_set_icetank) < 0.1:
                        break

        soc = self.tank.state_of_charge
        self.api.exchange.set_global_value(state, self.bypass_frac_hndl, bypass_frac)
        self.api.exchange.set_global_value(state, self.soc_hndl, soc)

        # set outlet actuators
        self.api.exchange.set_actuator_value(state, self.mdot_out_hndl, mdot_act)
        self.api.exchange.set_actuator_value(state, self.t_out_hndl, t_out)

        return 0
