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
def parallel_code(thread, rows, voltages, topology_parameter, sim_dic, folder, np_info, res_info, T_val, save_th):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    thread_rows         = rows[thread]
        
    sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+"data/",
                                    res_info=res_info, np_info=np_info, seed=thread, add_to_path=f"_th={thread}")
    sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, T_val=T_val, sim_dic=sim_dic, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/ndr_nls/resistance/"
    voltages    = np.loadtxt(folder+'volt.csv')
    N_voltages  = voltages.shape[0]

    N_processes, network_topology, topology_parameter, sim_dic, np_info, res_info, T_val, save_th = nanonets_utils.load_params(folder=folder)
    
    rows        = [np.arange(j*N_voltages/N_processes,(j+1)*N_voltages/N_processes, dtype=int) for j in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, voltages, topology_parameter, sim_dic, folder,
                                                                      np_info, res_info, T_val, save_th))
        process.start()