import numpy as np
import sys
import time

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, voltages, time_steps, topology_parameter, res_info, eq_steps, folder, np_info, res_info2, T_val, save_th, stat_size, seed):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    R_val               = res_info2["R"]
    
    for s in range(stat_size):

        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+f"data/R_{R_val}/", res_info=res_info, np_info=np_info, res_info2=res_info2, add_to_path=f'_t{thread}_s{s}', seed=seed)
        sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, T_val=T_val, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/spatial_correlation/resistance_corr/"
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')
    stat_size   = 10
    seed        = 0

    # Set Seed
    # np.random.seed(seed)

    N_processes, network_topology, topology_parameter, eq_steps, np_info, res_info, T_val, save_th = nanonets_utils.load_time_params(folder=folder)

    res_info2 = {
        "R" : 800,
        "np_index" : [16,17,18,23,24,25,30,31,32]
    }

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, eq_steps, folder,
                                                                      np_info, res_info2, T_val, save_th, stat_size, seed))
        process.start()