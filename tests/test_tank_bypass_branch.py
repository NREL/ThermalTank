import unittest

from thermaltank.tank_bypass_branch import TankBypassBranch


class TestTankBypassBranch(unittest.TestCase):

    def test_simulate_float(self):
        b = TankBypassBranch(num_tanks=1)
        b.simulate(inlet_temp=10,
                   mass_flow_rate=3.0,
                   env_temp=20,
                   branch_set_point=20,
                   op_mode=0,
                   sim_time=0,
                   timestep=60)

        self.assertEqual(b.outlet_temp, 10)

    def test_simulate_charging(self):
        tank_data = {
            "tank_diameter": 89 * 0.0254,
            "tank_height": 101 * 0.0254,
            "fluid_volume": 1655 * 0.00378541,
            "r_value_lid": 24 / 5.67826,
            "r_value_base": 9 / 5.67826,
            "r_value_wall": 9 / 5.67826,
            "initial_temperature": 10,
            "coeff_c0_ua_charging": 4.950e+04,
            "coeff_c1_ua_charging": -1.262e+05,
            "coeff_c2_ua_charging": 2.243e+05,
            "coeff_c3_ua_charging": -1.455e+05,
            "coeff_c0_ua_discharging": 1.848e+03,
            "coeff_c1_ua_discharging": 7.429e+04,
            "coeff_c2_ua_discharging": -1.419e+05,
            "coeff_c3_ua_discharging": 9.366e+04
        }
        b = TankBypassBranch(num_tanks=1, tank_data=tank_data)

        b.simulate(inlet_temp=-5.0,
                   mass_flow_rate=3.0,
                   env_temp=20,
                   branch_set_point=20,
                   op_mode=1,
                   sim_time=0,
                   timestep=60)

        self.assertAlmostEqual(b.outlet_temp, 9.42, delta=0.01)
        self.assertAlmostEqual(b.tank.tank_temp, 9.61, delta=0.01)
        self.assertAlmostEqual(b.tank.state_of_charge, 0, delta=0.01)

    def test_simulate_discharging(self):
        tank_data = {
            "tank_diameter": 89 * 0.0254,
            "tank_height": 101 * 0.0254,
            "fluid_volume": 1655 * 0.00378541,
            "r_value_lid": 24 / 5.67826,
            "r_value_base": 9 / 5.67826,
            "r_value_wall": 9 / 5.67826,
            "initial_temperature": -10,
            "coeff_c0_ua_charging": 4.950e+04,
            "coeff_c1_ua_charging": -1.262e+05,
            "coeff_c2_ua_charging": 2.243e+05,
            "coeff_c3_ua_charging": -1.455e+05,
            "coeff_c0_ua_discharging": 1.848e+03,
            "coeff_c1_ua_discharging": 7.429e+04,
            "coeff_c2_ua_discharging": -1.419e+05,
            "coeff_c3_ua_discharging": 9.366e+04
        }
        b = TankBypassBranch(num_tanks=1, tank_data=tank_data)

        b.simulate(inlet_temp=15.0,
                   mass_flow_rate=3.0,
                   env_temp=20,
                   branch_set_point=5,
                   op_mode=-1,
                   sim_time=0,
                   timestep=60)

        self.assertAlmostEqual(b.outlet_temp, 5.0, delta=0.01)
