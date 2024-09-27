# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

import multiprocessing

def parallel_code(thread : int, N : int, V_arr : np.array, path : str):

    topology_parameter = {
        "Np"    :   N,
        "Nj"    :   0,
        "e_pos" :   [[-1,-1],[1,1]],
    }
    np_network_random   = nanonets.simulation(network_topology='random', topology_parameter=topology_parameter, seed=thread)
    
    np_network_random.run_const_voltages(V_arr, 1, save_th=0.1)
    results = np_network_random.return_output_values()

    np.savetxt(path+f"N_{N}_th_{thread}.csv", results)

V_arr       = np.zeros((100,3))
V_min       = 0.5
V_max       = 1.0
V_arr[:,0]  = np.linspace(V_min, V_max, 100, endpoint=True)
N_processes = 10
path        = "scripts/1_funding_period/current_magnitude/data_disorder/"

for N in np.arange(10,311,20):

    procs = []
    for i in range(10,N_processes+10):

        process = multiprocessing.Process(target=parallel_code, args=(i, N, V_arr, path))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()