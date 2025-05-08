# NanoNets: Nanoparticle Network Simulator

NanoNets is a Python-based simulator for modeling electron transport in nanoparticle networks. It combines physical modeling of electrostatic interactions with kinetic Monte Carlo methods to simulate single electron tunneling events between nanoparticles and electrodes.

## Features

- Supports both cubic lattice and random network topologies
- Models single electron tunneling events
- Handles constant and floating electrodes
- Includes electrostatic interactions between nanoparticles
- Supports memristive behavior with variable junction resistances
- Uses Numba optimization for efficient simulation
- Provides detailed tracking of network states and observables

## Project Structure

The project consists of four main Python modules:

- `topology.py`: Defines network topology and electrode connections
- `electrostatic.py`: Handles electrostatic calculations and capacitance matrices
- `tunneling.py`: Manages tunneling events and junction properties
- `nanonets.py`: Implements the main simulation logic and KMC algorithm

## Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/NanoNets.git
cd NanoNets
```

### Dependencies

- NumPy
- SciPy
- NetworkX
- Pandas
- Numba

Install dependencies using:
```bash
pip install numpy scipy networkx pandas numba
```

## Usage

### Basic Example

```python
from nanonets import simulation

# Define network topology
topology = {
    "Nx": 3,               # Number of particles in x-direction
    "Ny": 3,               # Number of particles in y-direction
    "Nz": 1,               # Number of particles in z-direction
    "e_pos": [[0,0,0],     # Electrode positions
              [1,2,0]],
    "electrode_type": ['constant', 'floating']  # Electrode types
}

# Create simulation instance
sim = simulation(topology_parameter=topology)

# Define voltage configurations
voltages = np.array([[0.8, 0.0, 0.0]])  # [V_e1, V_e2, V_G]

# Run simulation
sim.run_const_voltages(
    voltages=voltages,
    target_electrode=1,    # Index of electrode to monitor
    T_val=0.0             # Temperature in Kelvin
)
```

### Advanced Features

- Custom nanoparticle properties:
```python
np_info = {
    "eps_r": 2.6,         # Junction permittivity
    "eps_s": 3.9,         # Environment permittivity
    "mean_radius": 10.0,  # Average particle radius (nm)
    "std_radius": 0.0,    # Radius standard deviation
    "np_distance": 1.0    # Inter-particle spacing (nm)
}
```

## Physical Model

The simulator implements:
- Electrostatic interactions using capacitance matrices
- Single electron tunneling based on orthodox theory
- Kinetic Monte Carlo for time evolution
- Optional memristive effects in tunnel junctions

