import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/", seed=seed, add_to_path=f'_{seed}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/step_input/1I_1O_disorder/"
    stat_size           = 500
    eq_steps            = 1000000
    network_topology    = "random"
    
    # Network Topology
    topology_parameter  = {
        "Np"    :   49,
        "Nj"    :   4,
        "e_pos" :   [[-1,-1],[1,1]],
    }

    # Time / Voltage Values    
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')
    
    procs = []

    for seed in range(10):

        process = multiprocessing.Process(target=run_simulation, args=(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed))
        process.start()
        procs.append(process)

    for p in procs:
        p.join()