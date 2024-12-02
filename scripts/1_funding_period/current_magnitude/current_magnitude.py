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

def parallel_code(thread : int, N : int, V_arr : np.array, input_at : int, path : str, N_processes: int):

    topology_parameter = {
        "Nx"                : N,
        "Ny"                : N,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[(N-1)//2,0,0],[0,(N-1)//2,0],[N-1,0,0],[0,N-1,0],[N-1,(N-1)//2,0],[(N-1)//2,N-1,0],[N-1,N-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
    }

    sim_dic =   {
        "error_th"        : 0.05,      
        "max_jumps"       : 10000000,
        "eq_steps"        : 100000,
        "jumps_per_batch" : 5000,
        "kmc_counting"    : False,
        "min_batches"     : 10
    }

    jpb                 = sim_dic["jumps_per_batch"]
    voltages            = nanonets_utils.distribute_array_across_processes(process=thread, data=V_arr, N_processes=N_processes)
    np_network_cubic    = nanonets.simulation(topology_parameter=topology_parameter, folder=path, add_to_path=f"_jpb_{jpb}_E{input_at}")
    
    np_network_cubic.run_const_voltages(voltages=voltages, target_electrode=7, save_th=10, sim_dic=sim_dic)

N_data      = 1000
V_min       = 0.0
V_max       = 0.1
N_processes = 10
path        = "scripts/1_funding_period/current_magnitude/data/"

for N in range(3,10):

    for input_at in [0,1,3,5]:

        V_arr               = np.zeros((N_data,9))
        V_arr[:,input_at]   = np.linspace(V_min, V_max, N_data, endpoint=True)

        procs = []
        for i in range(N_processes):

            process = multiprocessing.Process(target=parallel_code, args=(i, N, V_arr, input_at, path, N_processes))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()