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

def parallel_code(eq_steps : int, N : int, V_arr : np.array, path : str):

    topology_parameter = {
        "Nx"    : N,
        "Ny"    : N,
        "Nz"    : 1,
        "e_pos" : [[0,0,0],[N-1,N-1,0]]
    }

    sim_dic = {
        'error_th'          :   0.05,
        'max_jumps'         :   10000000,
        'eq_steps'          :   eq_steps,
        'jumps_per_batch'   :   1000,
        'kmc_counting'      :   False,
        'min_batches'       :   10
    }

    np_network_cubic    = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter)
    
    np_network_cubic.run_const_voltages(V_arr, 1, save_th=0.1, sim_dic=sim_dic)
    results = np_network_cubic.return_output_values()

    np.savetxt(path+f"N_{N}_th_{eq_steps}.csv", results)

V_arr       = np.zeros((100,3))
V_min       = 0.5
V_max       = 1.0
V_arr[:,0]  = np.linspace(V_min, V_max, 100, endpoint=True)
N_processes = 2
path        = "scripts/1_funding_period/current_magnitude/data_equilibrium/"
# pre_jumps   = [0,10,50,100,500,1000,5000,10000,50000,100000]
pre_jumps   = [500000,1000000]

for N in range(3,19):

    procs = []
    for i in range(N_processes):

        eq_steps    = pre_jumps[i]
        process     = multiprocessing.Process(target=parallel_code, args=(eq_steps, N, V_arr, path))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()