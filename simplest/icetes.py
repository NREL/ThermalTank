from pyenergyplus.plugin import EnergyPlusPlugin

from tank_bypass_branch import TankBypassBranch


class UsrDefPlntCmpSet(EnergyPlusPlugin):

    def __init__(self):

        super().__init__()
        self.need_to_get_handles = True
        self.glycol = None

        # loop parameters
        self.loop_exit_temp = 6.7
        self.loop_delt_temp = 8.33

        # loop internal variable handle
        self.vdot_loop_hndl = None

        # setup actuator handles
        self.mdot_min_hndl = None
        self.mdot_max_hndl = None
        self.vdot_des_hndl = None
        self.load_min_hndl = None
        self.load_max_hndl = None
        self.load_opt_hndl = None

    def get_handles(self, state):

        # get loop internal variable handle
        self.vdot_loop_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Plant Design Volume Flow Rate",
            "CHW Loop"
        )

        # get setup actuator handles
        self.mdot_min_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Minimum Mass Flow Rate",
            "CHW Loop Ice Tank"
        )
        self.mdot_max_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Maximum Mass Flow Rate",
            "CHW Loop Ice Tank"
        )
        self.vdot_des_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Design Volume Flow Rate",
            "CHW Loop Ice Tank"
        )
        self.load_min_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Minimum Loading Capacity",
            "CHW Loop Ice Tank"
        )
        self.load_max_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Maximum Loading Capacity",
            "CHW Loop Ice Tank"
        )
        self.load_opt_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Optimal Loading Capacity",
            "CHW Loop Ice Tank"
        )

        # set get handles to false
        self.need_to_get_handles = False

    def on_user_defined_component_model(self, state) -> int:

        # wait until API is ready
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        # get handles
        if self.need_to_get_handles:
            self.get_handles(state)
            self.glycol = self.api.functional.glycol(state, u"water")

        # get loop internal variable values
        vdot_loop = self.api.exchange.get_internal_variable_value(state, self.vdot_loop_hndl)

        # calcs
        cp_loop = self.glycol.specific_heat(
            state,
            self.loop_exit_temp
        )
        rho_loop = self.glycol.density(
            state,
            self.loop_exit_temp
        )
        mdot_loop = rho_loop * vdot_loop
        cap_loop = mdot_loop * cp_loop * self.loop_delt_temp

        # set flow actuators
        self.api.exchange.set_actuator_value(
            state,
            self.mdot_min_hndl,
            0
        )
        self.api.exchange.set_actuator_value(
            state,
            self.mdot_max_hndl,
            mdot_loop
        )
        self.api.exchange.set_actuator_value(
            state,
            self.vdot_des_hndl,
            vdot_loop
        )

        # set load actuators
        self.api.exchange.set_actuator_value(
            state,
            self.load_min_hndl,
            0
        )
        self.api.exchange.set_actuator_value(
            state,
            self.load_max_hndl,
            cap_loop
        )
        self.api.exchange.set_actuator_value(
            state,
            self.load_opt_hndl,
            cap_loop
        )

        return 0


class UsrDefPlntCmpSim(EnergyPlusPlugin):

    def __init__(self):

        super().__init__()
        self.need_to_get_userdef_handles = True
        self.need_to_get_timestep_handles = True
        self.glycol = None

        # loop parameters
        self.loop_exit_temp = 6.7

        # loop internal variable handle
        self.vdot_loop_hndl = None

        # component internal variable handles
        self.t_in_hndl = None
        self.mdot_in_hndl = None
        self.cp_in_hndl = None
        self.load_in_hndl = None

        # sim actuator handles
        self.t_out_hndl = None
        self.mdot_out_hndl = None

        # schedule handles
        self.chrg_sch_hndl = None
        self.t_set_icetank_hndl = None

        # setpoint schedule actuator handle
        self.t_set_chiller_hndl = None

        # global variable handles
        self.soc_hndl = None
        self.t_branch_in_hndl = None
        self.t_branch_out_hndl = None
        self.t_tank_out_hndl = None
        self.mdot_branch_hndl = None
        self.mdot_tank_hndl = None

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

        # other inits
        self.t_in = None
        self.mdot_in = None

        # init tank
        self.tank_branch = TankBypassBranch(1, self.tank_data)
        self.tank_is_full = False

        # main outputs
        self.mdot_out = None
        self.t_out = None

    def get_user_def_handles(self, state):

        # get loop internal variable handle
        self.vdot_loop_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Plant Design Volume Flow Rate",
            "CHW Loop"
        )

        # get component internal variable handles
        self.t_in_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Inlet Temperature for Plant Connection 1",
            "CHW Loop Ice Tank"
        )
        self.mdot_in_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Inlet Mass Flow Rate for Plant Connection 1",
            "CHW Loop Ice Tank"
        )
        self.cp_in_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Inlet Specific Heat for Plant Connection 1",
            "CHW Loop Ice Tank"
        )
        self.load_in_hndl = self.api.exchange.get_internal_variable_handle(
            state,
            "Load Request for Plant Connection 1",
            "CHW Loop Ice Tank"
        )

        # get sim actuator handles
        self.t_out_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Outlet Temperature",
            "CHW Loop Ice Tank"
        )
        self.mdot_out_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Plant Connection 1",
            "Mass Flow Rate",
            "CHW Loop Ice Tank"
        )

        # set get handles to false
        self.need_to_get_userdef_handles = False

    def get_timestep_handles(self, state):

        # get schedule handles
        self.chrg_sch_hndl = self.api.exchange.get_variable_handle(
            state,
            "Schedule Value",
            "Charge Sch"
        )
        self.t_set_icetank_hndl = self.api.exchange.get_variable_handle(
            state,
            "Schedule Value",
            "Ice Tank Temp Sch"
        )

        # get chiller setpoint schedule actuator handles
        self.t_set_chiller_hndl = self.api.exchange.get_actuator_handle(
            state,
            "Schedule:Compact",
            "Schedule Value",
            "Chiller Temp Sch"
        )

        # get global handles
        self.soc_hndl = self.api.exchange.get_global_handle(
            state,
            "soc"
        )
        self.t_branch_in_hndl = self.api.exchange.get_global_handle(
            state,
            "t_branch_in"
        )
        self.t_branch_out_hndl = self.api.exchange.get_global_handle(
            state,
            "t_branch_out"
        )
        self.t_tank_out_hndl = self.api.exchange.get_global_handle(
            state,
            "t_tank_out"
        )
        self.mdot_branch_hndl = self.api.exchange.get_global_handle(
            state,
            "mdot_branch"
        )
        self.mdot_tank_hndl = self.api.exchange.get_global_handle(
            state,
            "mdot_tank"
        )

        # set get handles to false
        self.need_to_get_timestep_handles = False

    # reinitialize tank after warmup
    def on_after_new_environment_warmup_is_complete(self, state) -> int:

        # get number of tanks
        num_tanks_hndl = self.api.exchange.get_variable_handle(
            state,
            "Schedule Value",
            "Num Tanks"
        )
        num_tanks = self.api.exchange.get_variable_value(
            state,
            num_tanks_hndl
        )

        # reconstruct with num tanks
        self.tank_branch = TankBypassBranch(num_tanks, self.tank_data)

        # init tank state
        self.tank_branch.tank.init_state(tank_init_temp=15)
        self.tank_is_full = False

        return 0

    def on_begin_timestep_before_predictor(self, state) -> int:

        # wait until API is ready
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        # get handles
        if self.need_to_get_timestep_handles:
            self.get_timestep_handles(state)

        # set current date/time
        datetime = self.api.exchange.day_of_year(state) * 24 + self.api.exchange.current_time(state)
        timestep = self.api.exchange.zone_time_step(state) * 60 * 60

        # get schedule values
        chrg_sch = self.api.exchange.get_variable_value(
            state,
            self.chrg_sch_hndl
        )
        t_set_icetank = self.api.exchange.get_variable_value(
            state,
            self.t_set_icetank_hndl
        )

        # set chiller in E+
        if chrg_sch == 1:
            self.api.exchange.set_actuator_value(
                state,
                self.t_set_chiller_hndl,
                -3.8
            )
        elif chrg_sch == -1:
            self.api.exchange.set_actuator_value(
                state,
                self.t_set_chiller_hndl,
                10
            )
        elif chrg_sch == 0:
            self.api.exchange.set_actuator_value(
                state,
                self.t_set_chiller_hndl,
                6.7
            )

        # charge to 100% SOC, considered fully charged until 95%
        if self.tank_branch.tank.state_of_charge == 1:
            self.tank_is_full = True
        else:
            if self.tank_is_full:
                if self.tank_branch.tank.state_of_charge > 0.95:
                    self.tank_is_full = True
                else:
                    self.tank_is_full = False
            else:
                self.tank_is_full = False

        # in float mode or in charge mode but tank is full, bypass tank but compute env losses
        if (chrg_sch == 0) or ((chrg_sch == 1) and self.tank_is_full):
            self.tank_branch.simulate(
                self.t_in,
                self.mdot_in,
                20,
                t_set_icetank,
                0,
                datetime,
                timestep
            )
            self.api.exchange.set_actuator_value(
                state,
                self.t_set_chiller_hndl,
                6.7
            )
            self.t_out = self.t_in

        # else charge or discharge tank
        else:
            self.tank_branch.simulate(
                self.t_in,
                self.mdot_in,
                20,
                t_set_icetank,
                chrg_sch,
                datetime,
                timestep
            )

            # get outlet temperature and SOC from python
            self.t_out = self.tank_branch.outlet_temp
            soc = self.tank_branch.tank.state_of_charge

            # set global variables
            self.api.exchange.set_global_value(
                state,
                self.soc_hndl,
                soc
            )
            self.api.exchange.set_global_value(
                state,
                self.t_branch_in_hndl,
                self.t_in
            )
            self.api.exchange.set_global_value(
                state,
                self.t_branch_out_hndl,
                self.t_out
            )
            self.api.exchange.set_global_value(
                state,
                self.t_tank_out_hndl,
                self.tank_branch.tank.outlet_fluid_temp
            )
            self.api.exchange.set_global_value(
                state,
                self.mdot_branch_hndl,
                self.mdot_in
            )
            self.api.exchange.set_global_value(
                state,
                self.mdot_tank_hndl,
                self.tank_branch.tank_mass_flow
            )

        return 0

    def on_user_defined_component_model(self, state) -> int:

        # wait until API is ready
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        # get handles
        if self.need_to_get_userdef_handles:
            self.get_user_def_handles(state)
            self.glycol = self.api.functional.glycol(state, u"water")

        # get loop internal variable values
        vdot_loop = self.api.exchange.get_internal_variable_value(state, self.vdot_loop_hndl)

        # calcs
        rho_loop = self.glycol.density(state, self.loop_exit_temp)
        mdot_loop = rho_loop * vdot_loop

        # get component internal variable values
        self.t_in = self.api.exchange.get_internal_variable_value(
            state,
            self.t_in_hndl
        )
        self.mdot_in = self.api.exchange.get_internal_variable_value(
            state,
            self.mdot_in_hndl
        )

        # assign outputs
        if self.mdot_out is None:
            self.mdot_out = self.mdot_in

        if self.mdot_out == 0:
            self.mdot_out = mdot_loop

        if self.t_out is None:
            self.t_out = self.t_in

        # set outlet actuators
        self.api.exchange.set_actuator_value(
            state,
            self.mdot_out_hndl,
            self.mdot_out
        )
        self.api.exchange.set_actuator_value(
            state,
            self.t_out_hndl,
            self.t_out
        )

        return 0
