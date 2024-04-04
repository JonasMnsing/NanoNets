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
def parallel_code(thread, voltages, time_steps, topology_parameter, folder, np_info, T_val, save_th, add_to_path):

    target_electrode = len(topology_parameter["e_pos"]) - 1
        
    sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder, np_info=np_info, add_to_path=add_to_path)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, T_val=T_val, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/step_input/"
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')

    N_processes, network_topology, topology_parameter, sim_dic, np_info, T_val, save_th, add_to_path = nanonets_utils.load_params(folder=folder)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, voltages, topology_parameter, sim_dic, folder,
                                                                      np_info, T_val, save_th, add_to_path))
        process.start()