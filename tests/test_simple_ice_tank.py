import unittest

from src.simple_ice_tank import IceTank


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
        }

    def test_init(self):
        # init from initial temp passed into ctor
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.diameter, 2.26, delta=0.01)
        self.assertAlmostEqual(tank.height, 2.57, delta=0.01)
        self.assertAlmostEqual(tank.fluid_volume, 6.26, delta=0.01)
        self.assertAlmostEqual(tank.total_fluid_mass, 6253.73, delta=0.01)
        self.assertAlmostEqual(tank.area_lid, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_base, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_wall, 18.22, delta=0.01)
        self.assertAlmostEqual(tank.area_total, 26.25, delta=0.01)
        self.assertAlmostEqual(tank.r_value_lid, 4.23, delta=0.01)
        self.assertAlmostEqual(tank.r_value_base, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.r_value_wall, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.resist_inside, 3.81e-05, delta=1e-05)
        self.assertAlmostEqual(tank.resist_outside, 3.81e-04, delta=1e-04)
        self.assertAlmostEqual(tank.resist_conduction, 6.68e-02, delta=1e-02)
        self.assertAlmostEqual(tank.overall_ua, 14.98, delta=0.01)

        # init from SOC passed into ctor
        self.data.pop("initial_temperature", None)
        self.data["latent_state_of_charge"] = 0.0
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.diameter, 2.26, delta=0.01)
        self.assertAlmostEqual(tank.height, 2.57, delta=0.01)
        self.assertAlmostEqual(tank.fluid_volume, 6.26, delta=0.01)
        self.assertAlmostEqual(tank.total_fluid_mass, 6253.73, delta=0.01)
        self.assertAlmostEqual(tank.area_lid, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_base, 4.01, delta=0.01)
        self.assertAlmostEqual(tank.area_wall, 18.22, delta=0.01)
        self.assertAlmostEqual(tank.area_total, 26.25, delta=0.01)
        self.assertAlmostEqual(tank.r_value_lid, 4.23, delta=0.01)
        self.assertAlmostEqual(tank.r_value_base, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.r_value_wall, 1.58, delta=0.01)
        self.assertAlmostEqual(tank.resist_inside, 3.81e-05, delta=1e-05)
        self.assertAlmostEqual(tank.resist_outside, 3.81e-04, delta=1e-04)
        self.assertAlmostEqual(tank.resist_conduction, 6.68e-02, delta=1e-02)
        self.assertAlmostEqual(tank.overall_ua, 14.98, delta=0.01)

    def test_q_brine_max(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.q_brine_max(10, 1, 1), -38434, delta=1)

    def test_q(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.q_brine(10, 1, 1), -36513, delta=1)

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
        self.assertAlmostEqual(tank.ice_mass, 226.6, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 825.4, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 1424.2, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2023.0, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2621.8, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3220.6, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3819.4, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 4418.2, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5017.0, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5615.8, delta=0.1)

        # latent-only charging
        tank.compute_charging(-2e8)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 6214.6, delta=0.1)

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
        self.assertAlmostEqual(tank.ice_mass, 5735.6, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 5136.8, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 4538.0, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3939.2, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 3340.4, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2741.6, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 2142.8, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 1544.0, delta=0.1)

        # latent-only discharging
        tank.compute_discharging(2e8)
        self.assertAlmostEqual(tank.tank_temp, 0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 945.2, delta=0.1)

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
        self.assertAlmostEqual(tank.tank_temp, 10.7, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)

    def test_calculate(self):
        self.data["initial_temperature"] = 5.0
        tank = IceTank(self.data)

        # initial conditions
        self.assertAlmostEqual(tank.tank_temp, 5.0, delta=0.1)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 3.70, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 1.0, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 2.57, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 0.22, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 1.59, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -0.45, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.74, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -1.04, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -1.55, delta=0.01)

        # charging
        tank.calculate(-5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 50.87, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, -1.99, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.0, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0.0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 1.99, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 0.66, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 2.00, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 1.22, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 2.39, delta=0.01)

        # discharging
        tank.calculate(5.0, 5.0, 20, 300)
        self.assertAlmostEqual(tank.tank_temp, 1.72, delta=0.01)
        self.assertAlmostEqual(tank.ice_mass, 0, delta=0.1)
        self.assertAlmostEqual(tank.outlet_fluid_temp, 2.73, delta=0.01)

    def test_effectiveness(self):
        tank = IceTank(self.data)
        self.assertAlmostEqual(tank.effectiveness(0), 0.95, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(1), 0.95, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(2), 0.95, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(3), 0.85, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(4), 0.72, delta=0.01)
        self.assertAlmostEqual(tank.effectiveness(5), 0.6, delta=0.01)

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
