# Libraries
import numpy as np
import pandas as pd
import multiprocessing

# Extend PATH Variable
import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def parallel_code(thread, voltages, rows):

    network_topology    = "random"
    topology_parameter  = {
        "Np"    : 100,
        "Nj"    : 4,
        "e_pos" : [[-1,-1],[0,-1],[1,-1],[-1,0],[-1,1],[0,1],[1,0],[1,1]]
    }

    thread_rows = rows[thread]
    volt_vals   = voltages[thread_rows]
    folder      = f"/mnt/c/Users/jonas/Desktop/phd/NanoNets/scripts/const_voltages/boolean_logic_experiment_comparsion/data/"
    add_to_path = f'_{thread}'

    # Run Simulation
    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder, add_to_path=add_to_path)
    np_network_sim.run_const_voltages(voltages=volt_vals, target_electrode=7, save_th=10)

if __name__ == '__main__':

    # Electrode Voltages
    N_voltages      = 80000
    voltages        = np.zeros(shape=(N_voltages,9))
    i1              = np.tile([0.0,0.1,0.0,0.1], int(N_voltages/4))
    i2              = np.tile([0.0,0.0,0.1,0.1], int(N_voltages/4))
    v_rand          = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=((int(N_voltages/4),5))), 4, axis=0)
    voltages[:,0]   = i1
    voltages[:,1]   = i2
    voltages[:,2:7] = v_rand

    N_processes     = 10
    index           = [i for i in range(N_voltages)]
    rows            = [index[i::N_processes] for i in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,voltages, rows))
        process.start()



