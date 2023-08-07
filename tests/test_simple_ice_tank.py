import unittest

from thermaltank.simple_ice_tank import IceTank, smoothing_function


class TestSimpleIceTank(unittest.TestCase):

    def setUp(self) -> None:
        self.data = {
            "tank_diameter": 89 * 0.0254,  # 89 inches
            "tank_height": 101 * 0.0254,  # 101 inches
            "fluid_volume": 1655 * 0.00378541,  # 1655 gal
            "r_value_lid": 24 / 5.67826,  # R-24
            "r_value_base": 9 / 5.67826,  # R-9
            "r_value_wall": 9 / 5.67826,  # R-9
            "initial_temperature": 20,
            "coeff_c0_ua_charging": 4.950e+04,
            "coeff_c1_ua_charging": -1.262e+05,
            "coeff_c2_ua_charging": 2.243e+05,
            "coeff_c3_ua_charging": -1.455e+05,
            "coeff_c0_ua_discharging": 1.848e+03,
            "coeff_c1_ua_discharging": 7.429e+04,
            "coeff_c2_ua_discharging": -1.419e+05,
            "coeff_c3_ua_discharging": 9.366e+04
        }

    def test_init(self):
        # init from initial temp passed into ctor
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.diameter, 2.26, delta=0.01)
        self.assertAlmostEqual(tank.height, 2.57, delta=0.01)
        self.assertAlmostEqual(tank.fluid_volume, 6.26, delta=0.01)
        self.assertAlmostEqual(tank.total_fluid_mass, 6253.60, delta=0.01)
        self.assertAlmostEqual(tank.area_lid, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_base, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_wall, 18.22, delta=0.01)
        self.assertAlmostEqual(tank.area_total, 26.25, delta=0.01)
        self.assertAlmostEqual(tank.r_value_lid, 4.23, delta=0.01)
        self.assertAlmostEqual(tank.r_value_base, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.r_value_wall, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.resist_inside_tank_wall, 3.81e-05, delta=1e-05)
        self.assertAlmostEqual(tank.resist_outside_tank_wall, 3.81e-04, delta=1e-04)
        self.assertAlmostEqual(tank.resist_conduction_tank_wall, 6.68e-02, delta=1e-02)
        self.assertAlmostEqual(tank.tank_ua_env, 14.98, delta=0.01)

        # init from SOC passed into ctor
        self.data.pop("initial_temperature", None)
        self.data["latent_state_of_charge"] = 0.0
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.diameter, 2.26, delta=0.01)
        self.assertAlmostEqual(tank.height, 2.57, delta=0.01)
        self.assertAlmostEqual(tank.fluid_volume, 6.26, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.01)
        self.assertAlmostEqual(tank.total_fluid_mass, 6253.60, delta=0.01)
        self.assertAlmostEqual(tank.area_lid, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_base, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_wall, 18.22, delta=0.01)
        self.assertAlmostEqual(tank.area_total, 26.25, delta=0.01)
        self.assertAlmostEqual(tank.r_value_lid, 4.23, delta=0.01)
        self.assertAlmostEqual(tank.r_value_base, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.r_value_wall, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.resist_inside_tank_wall, 3.81e-05, delta=1e-05)
        self.assertAlmostEqual(tank.resist_outside_tank_wall, 3.81e-04, delta=1e-04)
        self.assertAlmostEqual(tank.resist_conduction_tank_wall, 6.68e-02, delta=1e-02)
        self.assertAlmostEqual(tank.tank_ua_env, 14.98, delta=0.01)

    def test_q_brine_max(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.q_brine_max(10, 1, 1), -38434, delta=1)

    def test_q(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.q_brine(10, 1, 1), -14711, delta=1)

    def test_q_env(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.q_env(10, 1), -149.7, delta=0.1)

    def test_compute_charging(self):
        tank = IceTank(self.data)

        # sensible-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 12.35, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)

        # sensible-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 4.72, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)

        # sensible and latent charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 227.3, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 826.1, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 1424.9, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2023.8, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2622.6, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3221.4, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3820.2, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 4419.0, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5017.8, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5616.6, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 6215.4, delta=0.1)

        # latent and ice subcooling charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, -14.7, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 6253.7, delta=0.1)

    def test_compute_discharging(self):
        self.data["initial_temperature"] = -10.0
        tank = IceTank(self.data)

        # initial conditions
        self.assertAlmostEqual(tank.tank_temp, -10.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 6253.7, delta=0.1)

        # sensible-only subcooled ice discharging
        tank.compute_discharging(1e8)
        self.assertAlmostEqual(tank.tank_temp, -2.2, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 6253.7, delta=0.1)

        # sensible subcooled ice discharging and latent discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5735.4, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5136.6, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 4537.8, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3939.0, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3340.2, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2741.4, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2142.6, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 1543.8, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 945.0, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 346.3, delta=0.1)

        # latent and sensible discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 3.19, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)

        # sensible-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 10.8, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)

    def test_calculate(self):
        self.data["initial_temperature"] = 5.0
        tank = IceTank(self.data)

        # initial conditions
        self.assertAlmostEqual(tank.tank_temp, 5.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)

        # charging
        tank.calculate(-5.0, 5.0, 20, 0, 300)
        self.assertAlmostEqual(tank.tank_temp, 4.80, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 4.08, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300, 300)
        self.assertAlmostEqual(tank.tank_temp, 2.84, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 2.25, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 600, 300)
        self.assertAlmostEqual(tank.tank_temp, 1.26, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 0.80, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 900, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.01, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -0.35, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 1200, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 77.9, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -0.39, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 1500, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 156.1, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -0.42, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 1800, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 77.8, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 4.32, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 2100, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 66.1, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 4.35, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 2400, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 54.8, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 4.38, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 2700, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.00, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 44.1, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 4.41, delta=0.01)

        # no mass flow
        tank.calculate(5.0, 0.0, 20, 3000, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.00, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 43.8, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 5.0, delta=0.01)
        self.assertAlmostEqual(tank.state_of_charge, 0.0, delta=0.01)

    def test_effectiveness(self):
        # charging mode
        tank = IceTank(self.data)
        tank.tank_is_charging = True
        self.assertAlmostEqual(tank.effectiveness(-2.0, 2), 1.00, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.96, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(-2.0, 6), 0.89, delta=0.01)

        # discharging mode
        tank.tank_is_charging = False
        soc = 1.0
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.84, delta=0.01)

        soc = 0.9
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.76, delta=0.01)

        soc = 0.3
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.59, delta=0.01)

        soc = 0.14
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.47, delta=0.01)

        soc = 0.075
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.36, delta=0.01)

        soc = 0.01
        tank.ice_mass = tank.total_fluid_mass * soc
        self.assertAlmostEqual(tank.effectiveness(-2.0, 4), 0.15, delta=0.01)

    def test_init_state(self):
        # uninitialized tank
        self.data.pop("initial_temperature", None)
        tank = IceTank(self.data)
        self.assertEqual(tank.ice_mass, None)
        self.assertEqual(tank.tank_temp, None)
        self.assertEqual(tank.outlet_fluid_temp, None)

        # can't set both error
        with self.assertRaises(IOError):
            tank.init_state(latent_state_of_charge=0, tank_init_temp=0)

        # can't set none error
        with self.assertRaises(IOError):
            tank.init_state()

        # fully charged tank
        tank.init_state(latent_state_of_charge=1.0)
        self.assertEqual(tank.ice_mass, tank.total_fluid_mass)
        self.assertEqual(tank.tank_temp, 0.0)
        self.assertEqual(tank.outlet_fluid_temp, 0.0)

        # fully discharged tank
        tank.init_state(latent_state_of_charge=0.0)
        self.assertEqual(tank.ice_mass, 0.0)
        self.assertEqual(tank.tank_temp, 0.0)
        self.assertEqual(tank.outlet_fluid_temp, 0.0)

        # warm tank
        tank.init_state(tank_init_temp=20)
        self.assertEqual(tank.ice_mass, 0.0)
        self.assertEqual(tank.tank_temp, 20.0)
        self.assertEqual(tank.outlet_fluid_temp, 20.0)

        # sub-cooled tank
        tank.init_state(tank_init_temp=-20.0)
        self.assertEqual(tank.ice_mass, tank.total_fluid_mass)
        self.assertEqual(tank.tank_temp, -20.0)
        self.assertEqual(tank.outlet_fluid_temp, -20.0)

    def test_smoothing_function(self):
        # basic test, x bounded from [0, 1], y bounded from [0, 1]
        # x = 0
        args = [0, 0, 1, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 1
        args = [1, 0, 1, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 1.0, delta=0.01)

        # x = 0.5
        args = [0.5, 0, 1, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.5, delta=0.01)

        # x = -10
        args = [-10, 0, 1, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 10
        args = [10, 0, 1, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 1.0, delta=0.01)

        # x bounded from [0, 1], y bounded from [0, 10]
        # x = 0
        args = [0, 0, 1, 0, 10]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 1
        args = [1, 0, 1, 0, 10]
        self.assertAlmostEqual(smoothing_function(*args), 10, delta=0.01)

        # x = 0.5
        args = [0.5, 0, 1, 0, 10]
        self.assertAlmostEqual(smoothing_function(*args), 5.0, delta=0.01)

        # x = -10
        args = [-10, 0, 1, 0, 10]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 10
        args = [10, 0, 1, 0, 10]
        self.assertAlmostEqual(smoothing_function(*args), 10.0, delta=0.01)

        # x bounded from [0, 10], y bounded from [0, 1]
        # x = 0
        args = [0, 0, 10, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 10
        args = [10, 0, 10, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 1.0, delta=0.01)

        # x = 5
        args = [5, 0, 10, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.5, delta=0.01)

        # x = -10
        args = [-10, 0, 10, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 0.0, delta=0.01)

        # x = 10
        args = [10, 0, 10, 0, 1]
        self.assertAlmostEqual(smoothing_function(*args), 1.0, delta=0.01)
