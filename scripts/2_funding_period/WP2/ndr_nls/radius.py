"""
Run KMC Code for const set of voltages with fixed parameters defined in "params.csv" @ folder.
Output Electrode @ last position in topology_parameter key "pos"
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, rows, voltages, topology_parameter, folder, np_info, res_info):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    thread_rows         = rows[thread]
        
    sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+"data/",
                                    res_info=res_info, np_info=np_info, seed=thread, add_to_path=f"_th={thread}")
    sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/ndr_nls/radius/"
    voltages    = np.loadtxt(folder+'volt.csv')
    N_voltages  = voltages.shape[0]

    N_processes         = 10
    topology_parameter  = {
        "Nx"    :   7,
        "Ny"    :   7,
        "Nz"    :   1,
        "e_pos" :   [[0,0,0],[3,0,0],[6,0,0],[0,3,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]],
    } 
    np_info = {
        "eps_r"         : 2.6,
        "eps_s"         : 3.9,
        "mean_radius"   : 10.0,
        "std_radius"    : 2.0,
        "np_distance"   : 1.0
    }
    res_info = {
        "mean_R"    : 25.0,
        "std_R"     : 0.0,
        "dynamic"   : False
    }
    
    rows        = [np.arange(j*N_voltages/N_processes,(j+1)*N_voltages/N_processes, dtype=int) for j in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, voltages, topology_parameter, folder,
                                                                      np_info, res_info))
        process.start()