from pyenergyplus.plugin import EnergyPlusPlugin


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

    def get_handles(self, state):

        # get loop internal variable handle
        self.vdot_loop_hndl = self.api.exchange.get_internal_variable_handle(state, 'Plant Design Volume Flow Rate', 'CHW Loop')

        # get component internal variable handles
        self.t_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Temperature for Plant Connection 1', 'CHW Loop Ice Tank')
        self.mdot_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Mass Flow Rate for Plant Connection 1', 'CHW Loop Ice Tank')
        self.cp_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Inlet Specific Heat for Plant Connection 1', 'CHW Loop Ice Tank')
        self.load_in_hndl = self.api.exchange.get_internal_variable_handle(state, 'Load Request for Plant Connection 1', 'CHW Loop Ice Tank')

        # get setup actuator handles
        self.mdot_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Minimum Mass Flow Rate', 'CHW Loop Ice Tank')
        self.mdot_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Maximum Mass Flow Rate', 'CHW Loop Ice Tank')
        self.vdot_des_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Design Volume Flow Rate', 'CHW Loop Ice Tank')
        self.load_min_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Minimum Loading Capacity', 'CHW Loop Ice Tank')
        self.load_max_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Maximum Loading Capacity', 'CHW Loop Ice Tank')
        self.load_opt_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Optimal Loading Capacity', 'CHW Loop Ice Tank')

        # get sim actuator handles
        self.t_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Outlet Temperature', 'CHW Loop Ice Tank')
        self.mdot_out_hndl = self.api.exchange.get_actuator_handle(state, 'Plant Connection 1', 'Mass Flow Rate', 'CHW Loop Ice Tank')

        # set get handles to false
        self.need_to_get_handles = False


    def on_user_defined_component_model(self, state) -> int:

        # wait until API is ready
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        # get handles
        if self.need_to_get_handles:
            self.get_handles(state)
            self.glycol = self.api.functional.glycol(state, u'water')

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

        # assign outputs
        mdot_out = mdot_in
        t_out = t_in

        # set outlet actuators
        self.api.exchange.set_actuator_value(state, self.mdot_out_hndl, mdot_out)
        self.api.exchange.set_actuator_value(state, self.t_out_hndl, t_in)

        return 0
