# Libraries
import numpy as np
import pandas as pd
import multiprocessing

# Extend PATH Variable
import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonas/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def parallel_code(thread, N_voltages, N_Networks):

    rs = np.random.RandomState(thread)
    
    for i_net in range(N_Networks):

        # Electrode Voltages
        voltages        = np.zeros(shape=(N_voltages,9))
        i1              = np.tile([0.0,0.01,0.0,0.01], int(N_voltages/4))*1.23
        i2              = np.tile([0.0,0.0,0.01,0.01], int(N_voltages/4))*1.23
        v_rand          = np.repeat(rs.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),5))), 4, axis=0)*1.23
        v_gates         = np.repeat(rs.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
        voltages[:,0]   = i1
        voltages[:,1]   = i2
        voltages[:,2:7] = v_rand
        voltages[:-1]   = v_gates

        network_topology    = "random"
        topology_parameter  = {
            "Np"    : 100,
            "Nj"    : 4,
            "e_pos" : [[-1,-1],[0,-1],[1,-1],[-1,0],[-1,1],[0,1],[1,0],[1,1]],
            "seed"  : None
        }

        # folder      = f"/mnt/c/Users/jonas/Desktop/phd/NanoNets/scripts/const_voltages/boolean_logic_experiment_comparsion/data/"
        folder      = f"/home/jonas/phd/NanoNets/scripts/const_voltages/boolean_logic_experiment_comparsion/data/"
        add_to_path = f'_{thread}_{i_net}'

        # Run Simulation
        np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder, add_to_path=add_to_path)
        np_network_sim.run_const_voltages(voltages=voltages, target_electrode=7, save_th=10)

if __name__ == '__main__':

    N_processes     = 10
    N_voltages      = 1600
    N_Networks      = 5

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N_voltages, N_Networks))
        process.start()



