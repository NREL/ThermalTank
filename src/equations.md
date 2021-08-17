# Simple Tank Model Equations

## Simulate Method

### 1. Compute environmental losses

```text
q_evn = UA_tank * (env_temp - tank_temp) * timestep
```

### 2. Compute brine heat transfer rate

#### Max possible heat transfer rate at this timestep
```text
q_max_brine = mdot * cp_brine * (inlet_temp - tank_temp) * timestep
```

#### HX Effectiveness
```text
tank_ua_hx = 20000
NTU = NTU = tank_ua_hx / (mdot * cp_brine)
effectiveness = 1 - exp(-NTU) * SOC_modifier
```

#### SOC modifier for charging

This is basic "y = mx + b" linear equation with the slope and intercept defined by max and min values
```text
max_degradation_linear = 0.9
min_degradation_linear = 1.2
degradation_slope = (max_degradation_linear - min_degradation_linear)  # / 1
degradation_intercept = max_degradation_linear - degradation_slope
SOC_modifier = degradation_slope * soc + degradation_intercept
```

#### SOC modifier for discharging
This modifier is similar to the charging; however, there's a linear portion for the upper portion above a cutoff. Below the cutoff, there's a smooth transition sigmoid function that transitions from the lower cutoff to a minimum value smoothly. This is likely easier to interpret by looking at the code.

### 3. Total heat transfer to apply to change tank state

```text
q_tot = q_env + q_brine
```

### 4. Update tank state

#### Charging

Piece-wise charging of tank. From sensible liquid, to latent ice charging, then finally to sensible sub-cooled ice charging.

##### Sensible liquid charging

We know we can do this if `tank_temp > 0`.

First, compute the available capacity for this mode.

```text
q_sen_available = tank_mass * cp_f * tank_temp
```

If `q_sen_available > q_tot`, we have enough capacity to meet this timestep's needs. If so, we compute a new tank temperature.
```text
tank_temp_new = -q_tot / (tank_mass * cp_f) + tank_temp
```

We can return early if we've made it here.

If `q_sen_available < q_tot`, we don't have enough capacity for a sensible-only solution. We decrement `q_tot` by `q_sens_available`, set `tank_temp = 0`, and move on to latent ice charging.

##### Latent Ice Charging

We know we can do this if `q_tot > 0` and `ice_mass < tank_mass`.

Compute available latent capacity available.

```text
q_lat_available = h_if * liquid_mass
h_if = 334000
liquid_mass = tank_mass - ice_mass
```

If `q_lat_available > q_tot`, we have enough capacity to meet the load at this timestep for this mode. Compute the addtional ice generated.

```text
delta_ice_mass = q_tot / h_if
ice_mass += delta_ice_mass
```

We can return early if we've made it here.

If `q_lat_available < q_tot` we can't fully meet the load with latent charging only. Decrement `q_tot` by `q_lat_available`, set `ice_mass = tank_mass`. And move on to sensible ice subcooling.

##### Sensible Ice Sub-cooling

Set a new tank temperature.

```text
tank_temp += -q_tot / (tank_mass * cp_ice)
cp_ice = 2030 
```

#### Discharging

Discharging is just charging in reverse. The discharging calculations are essentially the same as the charging, execept we go from sensible sub-cooled ice discharging, then to latent discharging, and finally to sensible liquid discharging. At each mode we check the available capacity, and walk down the modes until we run out of energy.

### 5. Outlet Fluid Temperature

```text
outlet_temp = inlet_temp - effectiveness * (inlet_temp - tank_temp)
```

## Nomenclature
```text
q_evn = environmental gains/losses [J]
UA_tank = tank's overall heat transfer coefficient [W/K]
env_temp = environment temperature [C]
tank_temp = tank bulk fluid temperature [C]
timestep = simulation timestep [s]
mdot = mass flow rate of brine through tank [kg/s]
cp_brine = specific heat of brine flowing through tank [J/kg-K]
inlet_temp = inlet temperature of brine [C]
tank_ua_hx = tank heat exchanger effectiveness [W/K]
NTU = number of heat transfer units [-]
effectiveness = effectiveness of HX. ratio of q/qmax [-]
SOC_modifier = state of charge modifier to calibrate model [-]
q_tot = total heat transfer to apply to change of tank state [J
tank_mass = mass of fluid in tank [kg]
cp_f = specific heat of tank fluid [J/kg-k]
h_if = latent heat of fusion/meltion of water [J/kg]
cp_ice = specific heat of ice [J/kg-K]
```

