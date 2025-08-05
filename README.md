# NanoNets

**NanoNets** is a Python package for simulating single-electron transport in complex nanoparticle networks.  
It provides tools for generating and analyzing nanoparticle device topologies, computing network electrostatics, and running efficient kinetic Monte Carlo (KMC) simulations of single-electron tunneling.  
Designed for both fundamental research and device engineering.

## Features

- **Flexible Topology**: Create regular lattice or random planar nanoparticle networks with customizable geometry and connectivity.
- **Physical Electrostatics**: Automatically computes full capacitance matrices and induced charges using physical NP parameters.
- **Kinetic Monte Carlo**: High-performance, Numba-optimized KMC engine for simulating electron tunneling, network currents, and time-resolved device response.
- **Constant and Floating Electrodes**: Simulate both voltage-biased and floating (open circuit) contacts.
- **Heterogeneous Devices**: Supports multiple nanoparticle types, resistive disorder, and tunable electrode configurations.
- **Extensible & Modular**: Clear class structure enables easy modification and integration with other scientific Python tools.
- **Batch and Time-Dependent Simulation**: Supports stationary (fixed voltage) and dynamic (time-varying voltage) simulation modes.
- **Rich Output**: Exports observables, charge/potential landscapes, network currents, and more, directly to CSV for analysis.

## Class Overview

<details>
<summary><strong>Click to expand full class documentation</strong></summary>

### `NanoparticleTopology`
- Generate, modify, and analyze nanoparticle networks with electrodes.
- Built on `networkx` for flexible topology and visualization.

### `NanoparticleElectrostatic`
- Adds electrostatics: computes NP radii, capacitance, and charge induction.
- Efficiently packs nanoparticles and enforces physical constraints.

### `NanoparticleTunneling`
- Adds single-electron tunneling and resistance network.
- Precomputes tunneling events and manages tunnel junction resistances.

### `Simulation`
- High-level device simulation class: sets up topology, electrostatics, electrodes, and resistances.
- Runs KMC for stationary (DC) or dynamic (pulsed/AC) driving.

### `MonteCarlo` (jitclass)
- Fast KMC simulation core. Computes currents, potentials, and observables using Numba for speed.
- Supports both steady-state and time-resolved simulation.

</details>

## Quickstart Example

```from nanonets import Simulation
import numpy as np

# Define your network topology and parameters
topology_parameter = {
    'Nx': 5, 'Ny': 5,                           # 5x5 lattice
    'e_pos': [[0,0], [4,4]],                    # Electrodes at two corners
    'electrode_type': ['constant', 'constant']  # Both are not floating
}

# Initialize simulation
sim = Simulation(topology_parameter)

# Run a stationary simulation (fixed voltages)
N_volt        = 100                             # Number of voltages
voltages      = np.zeros((N_volt,3))            # 3 Columns (Two E + Gate)
voltages[:,0] = np.linspace(-0.1, 0.1, N_volt)  # Voltage Sweep at E1
sim.run_const_voltages(voltages, target_electrode=1)

# Access results
currents    = sim.get_observable_storage()
potentials  = sim.get_potential_storage()
```

## Citing
If you use NanoNets for published work, please cite:
<pre>@article{mensing2024kinetic,
  title={A kinetic Monte Carlo approach for Boolean logic functionality in gold nanoparticle networks},
  author={Mensing, Jonas and van der Wiel, Wilfred G and Heuer, Andreas},
  journal={Frontiers in Nanotechnology},
  volume={6},
  pages={1364985},
  year={2024},
  publisher={Frontiers Media SA}}</pre>