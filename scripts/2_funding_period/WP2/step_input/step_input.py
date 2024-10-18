import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")

import nanonets

# Simulation Function
def run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/", seed=seed)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/step_input/1I_1O/"
    stat_size           = 500
    eq_steps            = 1000000
    network_topology    = "cubic"
    seed                = 0
    scale               = np.array([0.75, 0.82, 0.89, 0.96, 1.  , 1.06, 1.11, 1.17, 1.22, 1.28, 1.33, 1.41, 1.5 , 1.62, 1.85, 2.13])

    for i,N in enumerate(range(3,13)):

        # Network Topology
        topology_parameter  = {
            "Nx"    :   N,
            "Ny"    :   N,
            "Nz"    :   1,
            "e_pos" :   [[0,0,0],[N-1,N-1,0]],
        }

        # Time / Voltage Values    
        voltages    = scale[i]*np.loadtxt(folder+'volt.csv')
        time_steps  = np.loadtxt(folder+'time.csv')

        run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed)
