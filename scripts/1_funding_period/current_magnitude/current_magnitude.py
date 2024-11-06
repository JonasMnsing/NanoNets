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
        "Nx"                : N,
        "Ny"                : N,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[N-1,N-1,0]],
        "electrode_type"    : ['constant','floating']
    }

    np_network_cubic    = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter)
    
    np_network_cubic.run_const_voltages(V_arr, 1, save_th=0.1, output_potential=True)
    results = np_network_cubic.return_output_values()

    np.savetxt(path+f"N_{N}_th_{thread}_low.csv", results)

V_arr       = np.zeros((100,3))
V_min       = 0.0 #0.5
V_max       = 0.1 #1.0
V_arr[:,0]  = np.linspace(V_min, V_max, 100, endpoint=True)
N_processes = 10
path        = "scripts/1_funding_period/current_magnitude/data/"

for N in range(3,19):

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N, V_arr, path))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()