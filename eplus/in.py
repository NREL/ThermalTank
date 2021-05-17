from pyenergyplus.plugin import EnergyPlusPlugin
from CoolProp.CoolProp import PropsSI
from simple_ice_tank import IceTank

class UsrDefPlntCmpSet(EnergyPlusPlugin):

    def __init__(self):
        super().__init__()

    def on_user_defined_component_model(self, state) -> int:
        return 0

class UsrDefPlntCmpSim(EnergyPlusPlugin):

    loop_exit_temp = 6.67
    loop_delt_temp = 8.33
    loop_flow_rate = 0.005

    def __init__(self):

        super().__init__()
        self.need_to_get_handles = True
        self.glycol = None

        # component internal variable handles
        self.t_in_hndl = None
        self.mdot_in_hndl = None
        self.rho_in_hndl = None
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

        self.tank_data = {
            "tank_diameter": 89 * 0.0254,
            "tank_height": 101 * 0.0254,
            "fluid_volume": 1655 * 0.00378541,
            "r_value_lid": 24 / 5.67826,
            "r_value_base": 9 / 5.67826,
            "r_value_wall": 9 / 5.67826,
            "initial_temperature": 20,
        }

        self.tank = IceTank(self.tank_data)

    def get_handles(self, state):

        # get component internal variable handles
        self.t_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Temperature for Plant Connection 1', 'User Def Plant Comp')
        self.mdot_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Mass Flow Rate for Plant Connection 1', 'User Def Plant Comp')
        self.rho_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Density for Plant Connection 1', 'User Def Plant Comp')
        self.cp_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Specific Heat for Plant Connection 1', 'User Def Plant Comp')
        self.load_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Load Request for Plant Connection 1', 'User Def Plant Comp')

        # get setup actuator handles
        self.mdot_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Minimum Mass Flow Rate', 'User Def Plant Comp')
        self.mdot_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Maximum Mass Flow Rate', 'User Def Plant Comp')
        self.vdot_des_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Design Volume Flow Rate', 'User Def Plant Comp')
        self.load_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Minimum Loading Capacity', 'User Def Plant Comp')
        self.load_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Maximum Loading Capacity', 'User Def Plant Comp')
        self.load_opt_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Optimal Loading Capacity', 'User Def Plant Comp')

        # get sim actuator handles
        self.t_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Outlet Temperature', 'User Def Plant Comp')
        self.mdot_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Mass Flow Rate', 'User Def Plant Comp')

        self.need_to_get_handles = False

    def on_user_defined_component_model(self, state) -> int:

        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        if not self.glycol:
            self.glycol = self.api.functional.glycol(state, u'water')

        if self.need_to_get_handles:
            self.get_handles(state)

        # get internal variables
        t_in = self.api.exchange.get_internal_variable_value(state, self.t_in_hndl)
        mdot_in = self.api.exchange.get_internal_variable_value(state, self.mdot_in_hndl)
        rho_in = self.api.exchange.get_internal_variable_value(state, self.rho_in_hndl)
        cp_in = self.api.exchange.get_internal_variable_value(state, self.cp_in_hndl)
        load_in = self.api.exchange.get_internal_variable_value(state, self.load_in_hndl)

        # calcs
        cp_loop = self.glycol.specific_heat(state, self.loop_exit_temp)
        rho_loop = self.glycol.density(state, self.loop_exit_temp)
        mdot_loop = rho_loop * self.loop_flow_rate
        cap_loop = mdot_loop * cp_loop * self.loop_delt_temp

        # set flow actuators
        self.api.exchange.set_actuator_value(state, self.mdot_min_hndl, 0)
        self.api.exchange.set_actuator_value(state, self.mdot_max_hndl, mdot_loop)
        self.api.exchange.set_actuator_value(state, self.vdot_des_hndl, self.loop_flow_rate)

        # set load actuators
        self.api.exchange.set_actuator_value(state, self.load_min_hndl, 0)
        self.api.exchange.set_actuator_value(state, self.load_max_hndl, cap_loop)
        self.api.exchange.set_actuator_value(state, self.load_opt_hndl, cap_loop)

        # pass-through if no load
        if load_in > -100:
            self.api.exchange.set_actuator_value(state, self.t_out_hndl, t_in)
            self.api.exchange.set_actuator_value(state, self.mdot_out_hndl, mdot_in)

            return 0

        # determine flow rate
        mdot_max = mdot_loop
        if mdot_in == 0:
            mdot_act = mdot_max
        else:
            mdot_act = mdot_in

        # determine load
        load_abs = abs(load_in)
        load_max = cap_loop
        if load_abs > load_max:
            load_act = load_max
        else:
            load_act = load_abs

        # average specific heat
        cp_out = self.glycol.specific_heat(state, self.loop_exit_temp)
        cp_avg = (cp_in + cp_out) / 2

        # calc tank
        self.tank.calculate(t_in, mdot_act, 24, 60)
        t_out = self.tank.outlet_fluid_temp
        print(f't_in={t_in}')
        print(f't_out={t_out}')

        # set outlet actuators
        self.api.exchange.set_actuator_value(state, self.mdot_out_hndl, mdot_act)
        self.api.exchange.set_actuator_value(state, self.t_out_hndl, t_out)

        return 0
